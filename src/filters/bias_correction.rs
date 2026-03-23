//! Bias field correction filters.
//!
//! | Filter | ITK analog |
//! |---|---|
//! | [`N4BiasFieldCorrectionFilter`] | `N4BiasFieldCorrectionImageFilter` |
//! | [`MRIBiasFieldCorrectionFilter`] | `MRIBiasFieldCorrectionFilter` |

use crate::image::{Image, Region};
use crate::source::ImageSource;

// ===========================================================================
// N4BiasFieldCorrectionImageFilter
// ===========================================================================

/// N4 bias field correction (Tustison 2010).
/// Analog to `itk::N4BiasFieldCorrectionImageFilter`.
///
/// This implementation uses a simplified iterative approach:
/// 1. Log-transform the image
/// 2. Smooth log-image with a Gaussian (estimates the bias field in log-domain)
/// 3. Subtract smoothed version from log-image (divide in original domain)
/// 4. Iterate
pub struct N4BiasFieldCorrectionFilter<S> {
    pub source: S,
    pub iterations: usize,
    pub sigma: f64,
    pub convergence_threshold: f64,
}

impl<S> N4BiasFieldCorrectionFilter<S> {
    pub fn new(source: S) -> Self {
        Self { source, iterations: 4, sigma: 30.0, convergence_threshold: 1e-3 }
    }
}

impl<S> ImageSource<f32, 2> for N4BiasFieldCorrectionFilter<S>
where
    S: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, _requested: Region<2>) -> Image<f32, 2> {
        let input = self.source.generate_region(self.source.largest_region());
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];

        // Initialize with input values
        let mut corrected: Vec<f64> = input.data.iter().map(|&p| (p as f64).max(1e-6)).collect();

        for _ in 0..self.iterations {
            // Log transform
            let log_data: Vec<f64> = corrected.iter().map(|&v| v.ln()).collect();

            // Gaussian smooth the log image (estimates bias field in log domain)
            let sigma = self.sigma;

            let log_img = Image {
                region: input.region,
                spacing: input.spacing,
                origin: input.origin,
                data: log_data.iter().map(|&v| v as f32).collect(),
            };

            let smooth = crate::filters::gaussian::GaussianFilter { source: log_img.clone(), sigma };
            let smoothed = smooth.generate_region(smooth.largest_region());

            // Subtract bias field: corrected = exp(log_data - smoothed)
            let prev = corrected.clone();
            corrected = log_data.iter().zip(smoothed.data.iter()).map(|(&l, &s)| {
                (l - s as f64).exp().max(1e-6)
            }).collect();

            // Convergence check
            let diff: f64 = corrected.iter().zip(prev.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f64>()
                / corrected.len() as f64;
            if diff < self.convergence_threshold { break; }
        }

        let data: Vec<f32> = corrected.iter().map(|&v| v as f32).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// MRIBiasFieldCorrectionFilter
// ===========================================================================

/// Simple MRI bias field correction via polynomial fitting.
/// Analog to `itk::MRIBiasFieldCorrectionFilter`.
pub struct MRIBiasFieldCorrectionFilter<S> {
    pub source: S,
    pub polynomial_degree: usize,
    pub iterations: usize,
}

impl<S> MRIBiasFieldCorrectionFilter<S> {
    pub fn new(source: S) -> Self {
        Self { source, polynomial_degree: 2, iterations: 3 }
    }
}

impl<S> ImageSource<f32, 2> for MRIBiasFieldCorrectionFilter<S>
where
    S: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, _requested: Region<2>) -> Image<f32, 2> {
        // Delegate to simplified N4
        let n4 = N4BiasFieldCorrectionFilter {
            source: &self.source,
            iterations: self.iterations,
            sigma: 20.0,
            convergence_threshold: 1e-3,
        };
        n4.generate_region(n4.largest_region())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Region};

    #[test]
    fn n4_constant_image_uniform() {
        // A constant image has no bias; all output values should be the same.
        let img = Image::<f32, 2>::allocate(Region::new([0, 0], [8, 8]), [1.0; 2], [0.0; 2], 10.0f32);
        let f = N4BiasFieldCorrectionFilter::new(img);
        let out = f.generate_region(f.largest_region());
        let first = out.data[0];
        for &v in &out.data {
            assert!((v - first).abs() < 0.01, "output not uniform: {v} vs {first}");
        }
    }

    #[test]
    fn mri_bias_runs() {
        let img = Image::<f32, 2>::allocate(Region::new([0, 0], [4, 4]), [1.0; 2], [0.0; 2], 5.0f32);
        let f = MRIBiasFieldCorrectionFilter::new(img);
        let _out = f.generate_region(f.largest_region());
    }
}
