//! Gaussian-weighted interpolation.
//!
//! [`GaussianInterpolator`] — analog to `itk::GaussianInterpolateImageFunction`.
//! Computes a Gaussian-weighted average of nearby pixels using the error function
//! (erf) for exact integral weights over each pixel's extent.
//!
//! [`LabelGaussianInterpolator`] — analog to `itk::LabelImageGaussianInterpolateImageFunction`.
//! Returns the label (pixel value) with the highest Gaussian-weighted vote.
//! Suitable for interpolating segmentation masks without blurring labels together.

use std::collections::HashMap;

use crate::image::{Image, iter_region, Region};
use crate::pixel::{NumericPixel, Pixel};

use super::{Interpolate, clamp_index};

// ---------------------------------------------------------------------------
// Gaussian interpolator
// ---------------------------------------------------------------------------

/// Gaussian-weighted interpolation. Parameters:
/// - `sigma`: standard deviation in physical units (applied uniformly to all axes).
/// - `alpha`: cutoff multiplier. Neighbourhood radius = `ceil(alpha × sigma / spacing)`.
///
/// Default: `sigma = 1.0`, `alpha = 1.0`.
pub struct GaussianInterpolator {
    pub sigma: f64,
    pub alpha: f64,
}

impl Default for GaussianInterpolator {
    fn default() -> Self {
        Self { sigma: 1.0, alpha: 1.0 }
    }
}

impl GaussianInterpolator {
    pub fn new(sigma: f64, alpha: f64) -> Self {
        Self { sigma, alpha }
    }

    fn radius(&self, spacing: f64) -> usize {
        (self.alpha * self.sigma / spacing).ceil() as usize
    }

    fn scaling_factor(&self, spacing: f64) -> f64 {
        1.0 / (std::f64::consts::SQRT_2 * self.sigma / spacing)
    }
}

impl<P: NumericPixel, const D: usize> Interpolate<P, D> for GaussianInterpolator {
    fn evaluate(&self, image: &Image<P, D>, index: [f64; D]) -> P {
        // Compute per-axis erf weight arrays and neighbourhood region
        let mut region_start = [0i64; D];
        let mut region_size = [0usize; D];
        let mut erf_weights: Vec<Vec<f64>> = Vec::with_capacity(D);

        for d in 0..D {
            let spacing = image.spacing[d];
            let sf = self.scaling_factor(spacing);
            let radius = self.radius(spacing);
            let lo = image.region.index.0[d];
            let hi = lo + image.region.size.0[d] as i64;
            let start = ((index[d] - radius as f64).floor() as i64).max(lo);
            let end = ((index[d] + radius as f64).ceil() as i64 + 1).min(hi);

            region_start[d] = start;
            region_size[d] = (end - start).max(0) as usize;

            // erf difference for each voxel interval
            let mut weights = Vec::with_capacity(region_size[d]);
            for k in 0..region_size[d] {
                let voxel = start + k as i64;
                let t0 = (voxel as f64 - 0.5 - index[d]) * sf;
                let t1 = (voxel as f64 + 0.5 - index[d]) * sf;
                weights.push((erf(t1) - erf(t0)).abs());
            }
            erf_weights.push(weights);
        }

        let region = Region::new(region_start, region_size);
        let mut sum_me = P::zero();
        let mut sum_m = 0.0f64;

        iter_region(&region, |idx| {
            let mut w = 1.0f64;
            for d in 0..D {
                let local = (idx.0[d] - region_start[d]) as usize;
                w *= erf_weights[d][local];
            }
            let pixel = image.get_pixel(clamp_index(image, idx.0));
            sum_me = sum_me + pixel.scale(w);
            sum_m += w;
        });

        if sum_m > 0.0 {
            sum_me.scale(1.0 / sum_m)
        } else {
            P::zero()
        }
    }
}

// ---------------------------------------------------------------------------
// Label Gaussian interpolator
// ---------------------------------------------------------------------------

/// Returns the label whose Gaussian-weighted vote is highest.
///
/// Treats each distinct pixel value as a separate label and convolves each
/// label's binary mask with the Gaussian kernel, then returns the winning label.
///
/// Pixel type must additionally implement `Eq + std::hash::Hash` so labels can
/// be collected into a map.
pub struct LabelGaussianInterpolator {
    pub sigma: f64,
    pub alpha: f64,
}

impl Default for LabelGaussianInterpolator {
    fn default() -> Self {
        Self { sigma: 1.0, alpha: 1.0 }
    }
}

impl LabelGaussianInterpolator {
    pub fn new(sigma: f64, alpha: f64) -> Self {
        Self { sigma, alpha }
    }
}

// We need P: Pixel + Eq + Hash for the label map.
// We can't easily add Hash to NumericPixel without orphan issues for f32/f64.
// Use a separate bound: only works for integer-like label types.
impl<P, const D: usize> Interpolate<P, D> for LabelGaussianInterpolator
where
    P: Pixel + Eq + std::hash::Hash,
{
    fn evaluate(&self, image: &Image<P, D>, index: [f64; D]) -> P {
        let inner = GaussianInterpolator { sigma: self.sigma, alpha: self.alpha };

        // Compute neighbourhood region and erf weights (same logic as Gaussian)
        let mut region_start = [0i64; D];
        let mut region_size = [0usize; D];
        let mut erf_weights: Vec<Vec<f64>> = Vec::with_capacity(D);

        for d in 0..D {
            let spacing = image.spacing[d];
            let sf = 1.0 / (std::f64::consts::SQRT_2 * inner.sigma / spacing);
            let radius = (inner.alpha * inner.sigma / spacing).ceil() as usize;
            let lo = image.region.index.0[d];
            let hi = lo + image.region.size.0[d] as i64;
            let start = ((index[d] - radius as f64).floor() as i64).max(lo);
            let end = ((index[d] + radius as f64).ceil() as i64 + 1).min(hi);

            region_start[d] = start;
            region_size[d] = (end - start).max(0) as usize;

            let mut weights = Vec::with_capacity(region_size[d]);
            for k in 0..region_size[d] {
                let voxel = start + k as i64;
                let t0 = (voxel as f64 - 0.5 - index[d]) * sf;
                let t1 = (voxel as f64 + 0.5 - index[d]) * sf;
                weights.push((erf(t1) - erf(t0)).abs());
            }
            erf_weights.push(weights);
        }

        let region = Region::new(region_start, region_size);
        let mut votes: HashMap<P, f64> = HashMap::new();

        iter_region(&region, |idx| {
            let mut w = 1.0f64;
            for d in 0..D {
                let local = (idx.0[d] - region_start[d]) as usize;
                w *= erf_weights[d][local];
            }
            let label = image.get_pixel(clamp_index(image, idx.0));
            *votes.entry(label).or_insert(0.0) += w;
        });

        votes
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(label, _)| label)
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Error function (erf) — approximation sufficient for interpolation
// ---------------------------------------------------------------------------

/// Rational approximation of erf with max error < 1.5e-7.
/// Source: Abramowitz & Stegun 7.1.26.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Index, Region};

    fn uniform_2d(val: f32) -> Image<f32, 2> {
        Image::<f32, 2>::allocate(Region::new([0, 0], [5, 5]), [1.0; 2], [0.0; 2], val)
    }

    #[test]
    fn gaussian_uniform_image_returns_same_value() {
        let img = uniform_2d(7.0);
        let interp = GaussianInterpolator::new(1.0, 2.0);
        let v = interp.evaluate(&img, [2.0, 2.0]);
        assert!((v - 7.0).abs() < 0.01, "expected ~7.0, got {v}");
    }

    #[test]
    fn gaussian_large_sigma_returns_mean() {
        // Very wide Gaussian → weighted mean ≈ image mean
        let mut img = Image::<f32, 2>::allocate(
            Region::new([0, 0], [5, 5]), [1.0; 2], [0.0; 2], 0.0,
        );
        // Fill half with 1.0, half with 0.0 → mean = 0.5
        for y in 0..5i64 {
            for x in 0..5i64 {
                if x < 3 { img.set_pixel(Index([x, y]), 1.0); }
            }
        }
        let interp = GaussianInterpolator::new(10.0, 3.0);
        let v = interp.evaluate(&img, [2.0, 2.0]);
        // Should be near 3/5 = 0.6 (weighted toward x<3 pixels)
        assert!((v - 0.6).abs() < 0.15, "expected ~0.6, got {v}");
    }

    #[test]
    fn label_gaussian_returns_dominant_label() {
        let mut img = Image::<u8, 2>::allocate(
            Region::new([0, 0], [5, 5]), [1.0; 2], [0.0; 2], 1u8,
        );
        // Small patch of label 2 far from query
        img.set_pixel(Index([4, 4]), 2);
        let interp = LabelGaussianInterpolator::new(1.0, 2.0);
        // Query near centre — should return dominant label 1
        let v = interp.evaluate(&img, [2.0, 2.0]);
        assert_eq!(v, 1, "expected label 1, got {v}");
    }
}
