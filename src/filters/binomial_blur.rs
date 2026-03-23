//! Binomial blur filter. Analog to `itk::BinomialBlurImageFilter`.
//!
//! Repeatedly convolves with the kernel `[1, 2, 1] / 4` (Pascal's row 2).
//! After `n` repetitions per axis the effective sigma is `sqrt(n) / 2` pixels,
//! approximating a Gaussian in the limit.
//!
//! Applied separably: each of the `repetitions` passes convolves all D axes
//! with the `[1, 2, 1] / 4` kernel.

use crate::image::{Image, Region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

use super::conv::convolve_axis;

/// Binomial blur filter.
pub struct BinomialBlurFilter<S> {
    pub source: S,
    /// Number of times to apply the `[1, 2, 1] / 4` kernel per axis.
    pub repetitions: usize,
}

impl<S> BinomialBlurFilter<S> {
    pub fn new(source: S, repetitions: usize) -> Self {
        Self { source, repetitions }
    }

    /// Effective Gaussian sigma (in pixels) after `repetitions` passes.
    pub fn effective_sigma_pixels(repetitions: usize) -> f64 {
        (repetitions as f64 / 4.0).sqrt() // variance = n/4, so sigma = sqrt(n)/2
    }
}

/// Single-pass `[1, 2, 1] / 4` kernel.
const BINOMIAL_KERNEL: [f64; 3] = [0.25, 0.5, 0.25];

impl<P, S, const D: usize> ImageSource<P, D> for BinomialBlurFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        // Each pass needs radius=1 per axis; total = repetitions * 1.
        let total_radius = self.repetitions;
        let radii = {
            let mut a = [0usize; D];
            a.iter_mut().for_each(|v| *v = total_radius);
            a
        };
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let bounds = self.source.largest_region();
        let total_r = self.repetitions;

        // Fetch input padded for all passes
        let radii = {
            let mut a = [0usize; D];
            a.iter_mut().for_each(|v| *v = total_r);
            a
        };
        let input_region = requested.padded_per_axis(&radii).clipped_to(&bounds);
        let mut current = self.source.generate_region(input_region);

        for rep in 0..self.repetitions {
            // Remaining padding after this repetition
            let remaining_r = total_r - rep - 1;
            for axis in 0..D {
                // Keep pad on axes not yet processed in this rep and for future reps
                let is_last = rep + 1 == self.repetitions && axis + 1 == D;
                let out_region = if is_last {
                    requested
                } else {
                    let mut pad = [0usize; D];
                    for d in 0..D {
                        pad[d] = if d > axis { remaining_r + 1 } else { remaining_r };
                    }
                    requested.padded_per_axis(&pad).clipped_to(&bounds)
                };
                current = convolve_axis(&current, axis, &BINOMIAL_KERNEL, out_region);
            }
        }
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    #[test]
    fn binomial_constant_image_preserved() {
        let img = Image::<f32, 1>::allocate(Region::new([0], [20]), [1.0], [0.0], 3.0f32);
        let f = BinomialBlurFilter::new(img, 4);
        let out = f.generate_region(f.largest_region());
        // Interior pixels should still be 3.0
        for i in 4..16i64 {
            let v = out.get_pixel(Index([i]));
            assert!((v - 3.0).abs() < 1e-5, "at {i}: {v}");
        }
    }

    #[test]
    fn binomial_impulse_spreads() {
        let mut img = Image::<f32, 1>::allocate(Region::new([0], [21]), [1.0], [0.0], 0.0f32);
        img.set_pixel(Index([10]), 1.0);
        let f = BinomialBlurFilter::new(img, 4);
        let out = f.generate_region(f.largest_region());

        // Peak should be at centre and less than 1
        let peak = out.get_pixel(Index([10]));
        assert!(peak > 0.0 && peak < 1.0, "peak={peak}");

        // Sum should be approximately 1 (energy conserved in interior)
        let sum: f32 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 0.05, "sum={sum}");
    }

    #[test]
    fn effective_sigma() {
        let sigma = BinomialBlurFilter::<()>::effective_sigma_pixels(4);
        assert!((sigma - 1.0).abs() < 1e-10); // sqrt(4/4) = 1.0
    }
}
