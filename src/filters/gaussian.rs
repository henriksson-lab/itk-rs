use rayon::prelude::*;

use crate::image::{Image, Index, Region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

/// Separable Gaussian smoothing filter. Analog to itk::DiscreteGaussianImageFilter.
///
/// Applies D successive 1-D convolutions (one per axis), which is equivalent
/// to full N-D Gaussian convolution but far more efficient.
///
/// Requires `P: NumericPixel` (Add + Mul<f64> + From<f64>).
pub struct GaussianFilter<S> {
    pub source: S,
    /// Standard deviation in physical units (same units as `spacing`).
    pub sigma: f64,
}

impl<S> GaussianFilter<S> {
    pub fn new(source: S, sigma: f64) -> Self {
        Self { source, sigma }
    }

    /// Half-width of the 1-D kernel in pixels for a given axis spacing.
    fn kernel_radius_for_spacing(&self, spacing: f64) -> usize {
        let sigma_pixels = self.sigma / spacing;
        (3.0 * sigma_pixels).ceil() as usize
    }

    /// Build a normalised 1-D Gaussian kernel of length `2*radius + 1`.
    fn build_kernel(sigma_pixels: f64, radius: usize) -> Vec<f64> {
        let len = 2 * radius + 1;
        let mut k: Vec<f64> = (0..len)
            .map(|i| {
                let x = i as f64 - radius as f64;
                (-0.5 * (x / sigma_pixels).powi(2)).exp()
            })
            .collect();
        let sum: f64 = k.iter().sum();
        k.iter_mut().for_each(|v| *v /= sum);
        k
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for GaussianFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> {
        self.source.largest_region()
    }
    fn spacing(&self) -> [f64; D] {
        self.source.spacing()
    }
    fn origin(&self) -> [f64; D] {
        self.source.origin()
    }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let spacing = self.source.spacing();
        let radii: [usize; D] = {
            let mut r = [0usize; D];
            for d in 0..D {
                r[d] = self.kernel_radius_for_spacing(spacing[d]);
            }
            r
        };
        output_region
            .padded_per_axis(&radii)
            .clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let spacing = self.spacing();
        let bounds = self.source.largest_region();

        // Compute per-axis radii
        let radii: [usize; D] = {
            let mut r = [0usize; D];
            for d in 0..D {
                let sigma_px = self.sigma / spacing[d];
                r[d] = (3.0 * sigma_px).ceil() as usize;
            }
            r
        };

        // Fetch input padded in ALL axes
        let input_region = requested.padded_per_axis(&radii).clipped_to(&bounds);
        let input = self.source.generate_region(input_region);

        // Apply separable convolution axis by axis.
        // After processing axis `k`, keep padding for axes k+1..D-1 so the
        // next pass has enough neighbours. Only the final pass crops to `requested`.
        let mut current = input;
        for axis in 0..D {
            let sigma_px = self.sigma / spacing[axis];
            let kernel = Self::build_kernel(sigma_px, radii[axis]);

            // Output region for this intermediate: requested, padded in remaining axes
            let out_region = if axis + 1 < D {
                let mut remaining = [0usize; D];
                for d in (axis + 1)..D {
                    remaining[d] = radii[d];
                }
                requested.padded_per_axis(&remaining).clipped_to(&bounds)
            } else {
                requested
            };

            current = convolve_axis(&current, axis, &kernel, out_region);
        }

        current
    }
}

/// Apply a 1-D convolution kernel along `axis` of `src`.
/// The output region is `requested` — pixels outside `src` use zero-flux
/// Neumann boundary (clamp-to-edge).
///
/// Uses rayon to parallelise over the output pixels.
fn convolve_axis<P, const D: usize>(
    src: &Image<P, D>,
    axis: usize,
    kernel: &[f64],
    requested: Region<D>,
) -> Image<P, D>
where
    P: NumericPixel,
{
    let radius = kernel.len() / 2;
    let out_len = requested.linear_len();

    // Collect all output indices so we can parallelise them.
    let indices: Vec<Index<D>> = {
        let mut v = Vec::with_capacity(out_len);
        collect_indices(&requested, &mut v);
        v
    };

    let data: Vec<P> = indices
        .par_iter()
        .map(|&out_idx| {
            let mut acc = P::zero();
            for (k_i, &w) in kernel.iter().enumerate() {
                let offset = k_i as i64 - radius as i64;
                let mut sample_idx = out_idx.0;
                sample_idx[axis] += offset;
                // Clamp to src region (zero-flux Neumann boundary)
                sample_idx[axis] = sample_idx[axis]
                    .max(src.region.index.0[axis])
                    .min(src.region.index.0[axis] + src.region.size.0[axis] as i64 - 1);
                acc = acc + src.get_pixel(Index(sample_idx)).scale(w);
            }
            acc
        })
        .collect();

    Image {
        region: requested,
        spacing: src.spacing,
        origin: src.origin,
        data,
    }
}

/// Fill `out` with every index in `region` in row-major order.
fn collect_indices<const D: usize>(region: &Region<D>, out: &mut Vec<Index<D>>) {
    crate::image::iter_region(region, |idx| out.push(idx));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;

    fn impulse_image_2d(size: usize) -> Image<f32, 2> {
        let mid = (size / 2) as i64;
        let mut img = Image::<f32, 2>::allocate(
            Region::new([0, 0], [size, size]),
            [1.0, 1.0],
            [0.0, 0.0],
            0.0f32,
        );
        img.set_pixel(Index([mid, mid]), 1.0);
        img
    }

    #[test]
    fn test_gaussian_impulse_energy_conserved() {
        let img = impulse_image_2d(21);
        let filter = GaussianFilter::new(img, 1.5);
        let out = filter.generate_region(filter.largest_region());
        let sum: f32 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "energy not conserved: sum={sum}");
    }

    #[test]
    fn test_gaussian_peak_at_centre() {
        let img = impulse_image_2d(21);
        let size = 21;
        let mid = (size / 2) as i64;
        let filter = GaussianFilter::new(img, 1.5);
        let out = filter.generate_region(filter.largest_region());
        let peak = out.get_pixel(Index([mid, mid]));
        // All other pixels should be smaller than the peak
        for &v in &out.data {
            assert!(v <= peak + 1e-6);
        }
    }
}
