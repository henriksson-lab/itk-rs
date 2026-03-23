//! Mean (box) smoothing filters.
//! Analogs to `itk::MeanImageFilter` and `itk::BoxMeanImageFilter`.

use crate::image::{Image, Region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

use super::conv::convolve_axis;

// ---------------------------------------------------------------------------
// Shared kernel builder
// ---------------------------------------------------------------------------

/// Uniform box kernel of length `2*radius + 1` (each weight = 1/len).
fn box_kernel(radius: usize) -> Vec<f64> {
    let len = 2 * radius + 1;
    vec![1.0 / len as f64; len]
}

// ---------------------------------------------------------------------------
// MeanFilter
// ---------------------------------------------------------------------------

/// Uniform-mean box filter. Analog to `itk::MeanImageFilter`.
///
/// Each output pixel is the mean of all input pixels within a box of
/// `[-radius, +radius]` on each axis (in index space).  Applied separably.
pub struct MeanFilter<S> {
    pub source: S,
    /// Box half-width in voxels (same on all axes).
    pub radius: usize,
}

impl<S> MeanFilter<S> {
    pub fn new(source: S, radius: usize) -> Self {
        Self { source, radius }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for MeanFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let r = self.radius;
        let radii = {
            let mut a = [0usize; D];
            a.iter_mut().for_each(|v| *v = r);
            a
        };
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let bounds = self.source.largest_region();
        let r = self.radius;
        let radii = {
            let mut a = [0usize; D];
            a.iter_mut().for_each(|v| *v = r);
            a
        };

        let input_region = requested.padded_per_axis(&radii).clipped_to(&bounds);
        let mut current = self.source.generate_region(input_region);

        let kernel = box_kernel(r);
        for axis in 0..D {
            let out_region = if axis + 1 < D {
                let mut rem = [0usize; D];
                for d in (axis + 1)..D { rem[d] = r; }
                requested.padded_per_axis(&rem).clipped_to(&bounds)
            } else {
                requested
            };
            current = convolve_axis(&current, axis, &kernel, out_region);
        }
        current
    }
}

// ---------------------------------------------------------------------------
// BoxMeanFilter
// ---------------------------------------------------------------------------

/// Box-mean filter with per-axis radius. Analog to `itk::BoxMeanImageFilter`.
///
/// Functionally identical to [`MeanFilter`] but allows a different radius on
/// each axis.  ITK's `BoxMeanImageFilter` uses a running-sum (O(1)-per-pixel)
/// algorithm; this implementation uses separable convolution.
pub struct BoxMeanFilter<S> {
    pub source: S,
    /// Per-axis box half-widths in voxels.
    pub radii: [usize; 3],
}

/// A version that accepts an array of radii sized at compile time.
pub struct BoxMeanFilterN<S, const D: usize> {
    pub source: S,
    pub radii: [usize; D],
}

impl<S, const D: usize> BoxMeanFilterN<S, D> {
    pub fn new(source: S, radii: [usize; D]) -> Self {
        Self { source, radii }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for BoxMeanFilterN<S, D>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        output_region.padded_per_axis(&self.radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let bounds = self.source.largest_region();
        let input_region = requested.padded_per_axis(&self.radii).clipped_to(&bounds);
        let mut current = self.source.generate_region(input_region);

        for axis in 0..D {
            let kernel = box_kernel(self.radii[axis]);
            let out_region = if axis + 1 < D {
                let mut rem = [0usize; D];
                for d in (axis + 1)..D { rem[d] = self.radii[d]; }
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

    fn ramp_1d(n: usize) -> Image<f32, 1> {
        let mut img = Image::<f32, 1>::allocate(Region::new([0], [n]), [1.0], [0.0], 0.0f32);
        for i in 0..n as i64 {
            img.set_pixel(Index([i]), i as f32);
        }
        img
    }

    #[test]
    fn mean_constant_image() {
        let img = Image::<f32, 1>::allocate(Region::new([0], [10]), [1.0], [0.0], 5.0f32);
        let f = MeanFilter::new(img, 2);
        let out = f.generate_region(f.largest_region());
        for &v in &out.data {
            assert!((v - 5.0).abs() < 1e-5, "expected 5.0, got {v}");
        }
    }

    #[test]
    fn mean_ramp_interior() {
        // Mean of a ramp over a symmetric window = centre value
        let img = ramp_1d(20);
        let f = MeanFilter::new(img, 2);
        let out = f.generate_region(f.largest_region());
        // Interior (skipping boundary where clamp affects result): x ∈ [2, 17]
        for i in 2..18i64 {
            let v = out.get_pixel(Index([i]));
            assert!((v - i as f32).abs() < 1e-4, "at {i}: got {v}");
        }
    }

    #[test]
    fn box_mean_2d_constant() {
        let img = Image::<f32, 2>::allocate(
            Region::new([0, 0], [8, 8]), [1.0; 2], [0.0; 2], 3.0f32,
        );
        let f = BoxMeanFilterN::new(img, [1, 2]);
        let out = f.generate_region(f.largest_region());
        for &v in &out.data {
            assert!((v - 3.0).abs() < 1e-5);
        }
    }
}
