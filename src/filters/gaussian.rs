//! Separable Gaussian smoothing. Analog to `itk::DiscreteGaussianImageFilter`.

use crate::image::{Image, Region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

use super::conv::convolve_axis;

/// Separable Gaussian smoothing filter.
///
/// Applies D successive 1-D convolutions (one per axis), equivalent to full
/// N-D Gaussian convolution but O(n·r) instead of O(n·r^D).
pub struct GaussianFilter<S> {
    pub source: S,
    /// Standard deviation in physical units.
    pub sigma: f64,
}

impl<S> GaussianFilter<S> {
    pub fn new(source: S, sigma: f64) -> Self {
        Self { source, sigma }
    }

    fn kernel_radius(&self, spacing: f64) -> usize {
        let sigma_px = self.sigma / spacing;
        (3.0 * sigma_px).ceil() as usize
    }

    pub(super) fn build_kernel(sigma_pixels: f64, radius: usize) -> Vec<f64> {
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
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let spacing = self.source.spacing();
        let mut radii = [0usize; D];
        for d in 0..D { radii[d] = self.kernel_radius(spacing[d]); }
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let spacing = self.spacing();
        let bounds = self.source.largest_region();

        let radii: [usize; D] = {
            let mut r = [0usize; D];
            for d in 0..D {
                r[d] = self.kernel_radius(spacing[d]);
            }
            r
        };

        let input_region = requested.padded_per_axis(&radii).clipped_to(&bounds);
        let input = self.source.generate_region(input_region);

        let mut current = input;
        for axis in 0..D {
            let sigma_px = self.sigma / spacing[axis];
            let kernel = Self::build_kernel(sigma_px, radii[axis]);

            let out_region = if axis + 1 < D {
                let mut rem = [0usize; D];
                for d in (axis + 1)..D { rem[d] = radii[d]; }
                requested.padded_per_axis(&rem).clipped_to(&bounds)
            } else {
                requested
            };

            current = convolve_axis(&current, axis, &kernel, out_region);
        }
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    fn impulse_image_2d(size: usize) -> Image<f32, 2> {
        let mid = (size / 2) as i64;
        let mut img = Image::<f32, 2>::allocate(
            Region::new([0, 0], [size, size]), [1.0, 1.0], [0.0, 0.0], 0.0f32,
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
        for &v in &out.data {
            assert!(v <= peak + 1e-6);
        }
    }
}
