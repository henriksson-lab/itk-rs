//! Image noise simulation filters.
//!
//! Analog to ITK's noise image filters. All filters use `rand::thread_rng()`
//! which is thread-local and compatible with rayon.

use rand::Rng;
use rayon::prelude::*;

use crate::image::{Image, Region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

// ===========================================================================
// Box-Muller Gaussian sample helper
// ===========================================================================

fn gaussian_sample(rng: &mut impl rand::Rng, mean: f64, std_dev: f64) -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rng.gen::<f64>().max(1e-300);
    let u2: f64 = rng.gen::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
    mean + std_dev * z
}

// ===========================================================================
// Additive Gaussian noise
// ===========================================================================

/// Add independent Gaussian noise `N(0, σ)` to each pixel.
/// Analog to `itk::AdditiveGaussianNoiseImageFilter`.
pub struct AdditiveGaussianNoiseFilter<S> {
    pub source: S,
    pub mean: f64,
    pub std_dev: f64,
}

impl<S> AdditiveGaussianNoiseFilter<S> {
    pub fn new(source: S, std_dev: f64) -> Self {
        Self { source, mean: 0.0, std_dev }
    }
    pub fn with_mean(mut self, m: f64) -> Self { self.mean = m; self }
}

impl<P, S, const D: usize> ImageSource<P, D> for AdditiveGaussianNoiseFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let (mean, std_dev) = (self.mean, self.std_dev);
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| {
                let mut rng = rand::thread_rng();
                let noise = gaussian_sample(&mut rng, mean, std_dev);
                P::from_f64(p.to_f64() + noise)
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Salt and pepper noise
// ===========================================================================

/// Randomly replace a fraction of pixels with `salt_value` or `pepper_value`.
/// Analog to `itk::SaltAndPepperNoiseImageFilter`.
pub struct SaltAndPepperNoiseFilter<S> {
    pub source: S,
    /// Probability that any given pixel is replaced (default 0.01).
    pub probability: f64,
    pub salt_value: f64,
    pub pepper_value: f64,
}

impl<S> SaltAndPepperNoiseFilter<S> {
    pub fn new(source: S, probability: f64) -> Self {
        Self { source, probability, salt_value: 255.0, pepper_value: 0.0 }
    }
    pub fn with_values(mut self, salt: f64, pepper: f64) -> Self {
        self.salt_value = salt;
        self.pepper_value = pepper;
        self
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for SaltAndPepperNoiseFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let (prob, salt, pepper) = (self.probability, self.salt_value, self.pepper_value);
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| {
                let mut rng = rand::thread_rng();
                let r: f64 = rng.gen();
                if r < prob / 2.0 {
                    P::from_f64(salt)
                } else if r < prob {
                    P::from_f64(pepper)
                } else {
                    p
                }
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Shot (Poisson) noise
// ===========================================================================

/// Model photon counting noise: output is Poisson-distributed with mean = input.
/// Uses Knuth's algorithm for small λ; normal approximation for λ > 50.
/// Analog to `itk::ShotNoiseImageFilter`.
pub struct ShotNoiseFilter<S> {
    pub source: S,
    /// Scale factor applied to input before computing Poisson random variable.
    pub scale: f64,
}

impl<S> ShotNoiseFilter<S> {
    pub fn new(source: S) -> Self { Self { source, scale: 1.0 } }
    pub fn with_scale(mut self, s: f64) -> Self { self.scale = s; self }
}

fn poisson_sample(rng: &mut impl rand::Rng, lambda: f64) -> f64 {
    if lambda <= 0.0 { return 0.0; }
    if lambda > 50.0 {
        // Normal approximation
        let g = gaussian_sample(rng, lambda, lambda.sqrt());
        return g.max(0.0).round();
    }
    // Knuth algorithm
    let limit = (-lambda).exp();
    let mut k = 0i64;
    let mut p = 1.0f64;
    loop {
        p *= rng.gen::<f64>();
        if p <= limit { break; }
        k += 1;
    }
    k as f64
}

impl<P, S, const D: usize> ImageSource<P, D> for ShotNoiseFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let scale = self.scale;
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| {
                let mut rng = rand::thread_rng();
                let lambda = p.to_f64() * scale;
                P::from_f64(poisson_sample(&mut rng, lambda) / scale)
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Speckle noise
// ===========================================================================

/// Multiplicative Gaussian noise: `output = input * N(1, σ)`.
/// Models speckle noise as in ultrasound or SAR images.
/// Analog to `itk::SpeckleNoiseImageFilter`.
pub struct SpeckleNoiseFilter<S> {
    pub source: S,
    pub std_dev: f64,
}

impl<S> SpeckleNoiseFilter<S> {
    pub fn new(source: S, std_dev: f64) -> Self { Self { source, std_dev } }
}

impl<P, S, const D: usize> ImageSource<P, D> for SpeckleNoiseFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let std_dev = self.std_dev;
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| {
                let mut rng = rand::thread_rng();
                let factor = gaussian_sample(&mut rng, 1.0, std_dev);
                P::from_f64(p.to_f64() * factor.max(0.0))
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Region};
    use crate::filters::statistics::compute_statistics;

    fn const_image(val: f32, n: usize) -> Image<f32, 1> {
        Image::<f32,1>::allocate(Region::new([0],[n]),[1.0],[0.0],val)
    }

    #[test]
    fn additive_gaussian_mean_near_signal() {
        // With many pixels and zero-mean noise, sample mean should be near signal
        let img = const_image(100.0, 10_000);
        let f = AdditiveGaussianNoiseFilter::new(img, 5.0);
        let out = f.generate_region(f.largest_region());
        let stats = compute_statistics(&out);
        assert!((stats.mean - 100.0).abs() < 1.0, "mean={}", stats.mean);
        assert!(stats.sigma > 0.1, "sigma should be non-zero");
    }

    #[test]
    fn salt_and_pepper_fraction() {
        let img = const_image(50.0, 10_000);
        let f = SaltAndPepperNoiseFilter::new(img, 0.10);
        let out = f.generate_region(f.largest_region());
        let corrupted = out.data.iter().filter(|&&v| (v - 50.0).abs() > 1.0).count();
        // Expect roughly 10% corrupted, allow ±5%
        assert!(corrupted > 500 && corrupted < 1500,
            "expected ~1000 corrupted, got {corrupted}");
    }

    #[test]
    fn shot_noise_preserves_order_of_magnitude() {
        let img = const_image(100.0, 1_000);
        let f = ShotNoiseFilter::new(img);
        let out = f.generate_region(f.largest_region());
        let stats = compute_statistics(&out);
        assert!(stats.mean > 80.0 && stats.mean < 120.0, "mean={}", stats.mean);
    }

    #[test]
    fn speckle_noise_multiplicative() {
        let img = const_image(0.0, 100);
        let f = SpeckleNoiseFilter::new(img, 0.1);
        let out = f.generate_region(f.largest_region());
        // Zero * anything = 0
        for &v in &out.data { assert!(v.abs() < 1e-10); }
    }
}
