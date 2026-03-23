//! Image sources (synthetic image generators).
//!
//! These implement `ImageSource` without an upstream source, generating pixels
//! from analytical functions of the pixel's physical coordinates.
//!
//! | Source | ITK analog |
//! |---|---|
//! | [`GaussianImageSource`] | `itk::GaussianImageSource` |
//! | [`GaborImageSource`]    | `itk::GaborImageSource` |
//! | [`GridImageSource`]     | `itk::GridImageSource` |
//! | [`PhysicalPointImageSource`] | `itk::PhysicalPointImageSource` |

use rayon::prelude::*;

use crate::image::{Image, Region, Index, iter_region};
use crate::source::ImageSource;

// ===========================================================================
// Gaussian image source
// ===========================================================================

/// Generate an N-D Gaussian function as an image.
/// Analog to `itk::GaussianImageSource`.
///
/// `f(x) = scale * exp(−sum_d((x_d − mean_d)² / (2 σ_d²)))`
pub struct GaussianImageSource<const D: usize> {
    pub region: Region<D>,
    pub spacing: [f64; D],
    pub origin: [f64; D],
    /// Mean (centre) in physical coordinates.
    pub mean: [f64; D],
    /// Standard deviation per axis in physical coordinates.
    pub sigma: [f64; D],
    /// Amplitude scaling factor.
    pub scale: f64,
    /// If true, normalise so the integral over all space = 1.
    pub normalized: bool,
}

impl<const D: usize> GaussianImageSource<D> {
    pub fn new(
        region: Region<D>,
        spacing: [f64; D],
        origin: [f64; D],
        mean: [f64; D],
        sigma: [f64; D],
    ) -> Self {
        Self { region, spacing, origin, mean, sigma, scale: 1.0, normalized: false }
    }
    pub fn with_scale(mut self, s: f64) -> Self { self.scale = s; self }
    pub fn with_normalized(mut self, n: bool) -> Self { self.normalized = n; self }
}

impl<const D: usize> ImageSource<f32, D> for GaussianImageSource<D> {
    fn largest_region(&self) -> Region<D> { self.region }
    fn spacing(&self) -> [f64; D] { self.spacing }
    fn origin(&self) -> [f64; D] { self.origin }

    fn generate_region(&self, requested: Region<D>) -> Image<f32, D> {
        let mean = self.mean;
        let sigma = self.sigma;
        let scale = self.scale;
        let origin = self.origin;
        let spacing = self.spacing;

        // Normalisation factor: (2π)^(D/2) * prod(σ_d)
        let norm = if self.normalized {
            use std::f64::consts::PI;
            let prod_sigma: f64 = sigma.iter().product();
            (2.0_f64 * PI).powi(D as i32 / 2) * prod_sigma
        } else {
            1.0
        };

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<f32> = out_indices.par_iter()
            .map(|&idx| {
                let mut exponent = 0.0f64;
                for d in 0..D {
                    let x = origin[d] + idx.0[d] as f64 * spacing[d];
                    let diff = (x - mean[d]) / sigma[d];
                    exponent += diff * diff;
                }
                ((scale / norm) * (-0.5 * exponent).exp()) as f32
            })
            .collect();
        Image { region: requested, spacing: self.spacing, origin: self.origin, data }
    }
}

// ===========================================================================
// Gabor image source
// ===========================================================================

/// Generate a Gabor function: Gaussian envelope × sinusoidal carrier.
/// Analog to `itk::GaborImageSource`.
///
/// `f(x) = amplitude * exp(−sum(dx²/σ²)) * cos(2π freq · x + phase)`
pub struct GaborImageSource<const D: usize> {
    pub region: Region<D>,
    pub spacing: [f64; D],
    pub origin: [f64; D],
    /// Centre of the Gaussian envelope (physical coords).
    pub mean: [f64; D],
    /// Standard deviation of the envelope per axis.
    pub sigma: [f64; D],
    /// Spatial frequency along each axis (cycles per unit length).
    pub frequency: [f64; D],
    pub phase: f64,
    pub amplitude: f64,
}

impl<const D: usize> GaborImageSource<D> {
    pub fn new(
        region: Region<D>,
        spacing: [f64; D],
        origin: [f64; D],
        mean: [f64; D],
        sigma: [f64; D],
        frequency: [f64; D],
    ) -> Self {
        Self { region, spacing, origin, mean, sigma, frequency,
               phase: 0.0, amplitude: 1.0 }
    }
}

impl<const D: usize> ImageSource<f32, D> for GaborImageSource<D> {
    fn largest_region(&self) -> Region<D> { self.region }
    fn spacing(&self) -> [f64; D] { self.spacing }
    fn origin(&self) -> [f64; D] { self.origin }

    fn generate_region(&self, requested: Region<D>) -> Image<f32, D> {
        use std::f64::consts::PI;
        let mean = self.mean;
        let sigma = self.sigma;
        let frequency = self.frequency;
        let origin = self.origin;
        let spacing = self.spacing;
        let phase = self.phase;
        let amplitude = self.amplitude;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<f32> = out_indices.par_iter()
            .map(|&idx| {
                let mut envelope = 0.0f64;
                let mut carrier_arg = phase;
                for d in 0..D {
                    let x = origin[d] + idx.0[d] as f64 * spacing[d];
                    let diff = (x - mean[d]) / sigma[d];
                    envelope += diff * diff;
                    carrier_arg += 2.0 * PI * frequency[d] * (x - mean[d]);
                }
                (amplitude * (-0.5 * envelope).exp() * carrier_arg.cos()) as f32
            })
            .collect();
        Image { region: requested, spacing: self.spacing, origin: self.origin, data }
    }
}

// ===========================================================================
// Grid image source
// ===========================================================================

/// Generate a grid pattern: 1 where any axis is on a grid line, 0 otherwise.
/// Analog to `itk::GridImageSource`.
pub struct GridImageSource<const D: usize> {
    pub region: Region<D>,
    pub spacing: [f64; D],
    pub origin: [f64; D],
    /// Grid line spacing per axis (physical units).
    pub grid_spacing: [f64; D],
    /// Grid line sigma (width in physical units; set to 0 for infinitely thin lines).
    pub grid_sigma: [f64; D],
    pub foreground_value: f64,
    pub background_value: f64,
}

impl<const D: usize> GridImageSource<D> {
    pub fn new(
        region: Region<D>,
        spacing: [f64; D],
        origin: [f64; D],
        grid_spacing: [f64; D],
        grid_sigma: [f64; D],
    ) -> Self {
        Self { region, spacing, origin, grid_spacing, grid_sigma,
               foreground_value: 1.0, background_value: 0.0 }
    }
}

impl<const D: usize> ImageSource<f32, D> for GridImageSource<D> {
    fn largest_region(&self) -> Region<D> { self.region }
    fn spacing(&self) -> [f64; D] { self.spacing }
    fn origin(&self) -> [f64; D] { self.origin }

    fn generate_region(&self, requested: Region<D>) -> Image<f32, D> {
        let origin = self.origin;
        let spacing = self.spacing;
        let grid_spacing = self.grid_spacing;
        let grid_sigma = self.grid_sigma;
        let fg = self.foreground_value;
        let bg = self.background_value;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<f32> = out_indices.par_iter()
            .map(|&idx| {
                // Product of cosine-based grids per axis
                // Each axis contributes: 0.5 * (1 - cos(2π x / grid_spacing)) smoothed by sigma
                let mut on_grid = false;
                for d in 0..D {
                    let x = origin[d] + idx.0[d] as f64 * spacing[d];
                    let gs = grid_spacing[d];
                    if gs <= 0.0 { continue; }
                    let sig = grid_sigma[d];
                    // Distance to nearest grid line
                    let phase = (x / gs).rem_euclid(1.0);
                    let dist_frac = (phase - phase.round()).abs();
                    let dist_phys = dist_frac * gs;
                    let val = if sig < 1e-10 {
                        if dist_phys < spacing[d] / 2.0 { 1.0 } else { 0.0 }
                    } else {
                        (-dist_phys * dist_phys / (2.0 * sig * sig)).exp()
                    };
                    if val > 0.5 { on_grid = true; break; }
                }
                (if on_grid { fg } else { bg }) as f32
            })
            .collect();
        Image { region: requested, spacing: self.spacing, origin: self.origin, data }
    }
}

// ===========================================================================
// Physical point image source
// ===========================================================================

/// Each pixel value encodes the physical coordinate along a chosen axis.
/// Analog to `itk::PhysicalPointImageSource`.
pub struct PhysicalPointImageSource<const D: usize> {
    pub region: Region<D>,
    pub spacing: [f64; D],
    pub origin: [f64; D],
    /// Which axis coordinate to store as the pixel value.
    pub component: usize,
}

impl<const D: usize> PhysicalPointImageSource<D> {
    pub fn new(region: Region<D>, spacing: [f64; D], origin: [f64; D], component: usize) -> Self {
        Self { region, spacing, origin, component }
    }
}

impl<const D: usize> ImageSource<f32, D> for PhysicalPointImageSource<D> {
    fn largest_region(&self) -> Region<D> { self.region }
    fn spacing(&self) -> [f64; D] { self.spacing }
    fn origin(&self) -> [f64; D] { self.origin }

    fn generate_region(&self, requested: Region<D>) -> Image<f32, D> {
        let origin = self.origin;
        let spacing = self.spacing;
        let comp = self.component;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<f32> = out_indices.par_iter()
            .map(|&idx| {
                (origin[comp] + idx.0[comp] as f64 * spacing[comp]) as f32
            })
            .collect();
        Image { region: requested, spacing: self.spacing, origin: self.origin, data }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Region;

    #[test]
    fn gaussian_source_peak_at_center() {
        let region = Region::new([0, 0], [11, 11]);
        let spacing = [1.0, 1.0];
        let origin = [0.0, 0.0];
        let mean = [5.0, 5.0];
        let sigma = [2.0, 2.0];
        let src = GaussianImageSource::new(region, spacing, origin, mean, sigma)
            .with_scale(1.0);
        let img = src.generate_region(region);
        let center = img.get_pixel(crate::image::Index([5i64, 5]));
        let corner = img.get_pixel(crate::image::Index([0i64, 0]));
        assert!(center > corner, "center {center} should exceed corner {corner}");
        assert!((center - 1.0).abs() < 1e-5, "peak should be 1.0 got {center}");
    }

    #[test]
    fn physical_point_encodes_coordinate() {
        let region = Region::new([0], [5]);
        let src = PhysicalPointImageSource::new(region, [2.0], [10.0], 0);
        let img = src.generate_region(region);
        // pixel at index 3 → physical = 10 + 3*2 = 16
        let v = img.get_pixel(crate::image::Index([3i64]));
        assert!((v - 16.0).abs() < 1e-5, "expected 16 got {v}");
    }

    #[test]
    fn grid_source_has_grid_lines() {
        let region = Region::new([0, 0], [20, 20]);
        let src = GridImageSource::new(region, [1.0, 1.0], [0.0, 0.0], [5.0, 5.0], [0.5, 0.5]);
        let img = src.generate_region(region);
        // pixel at x=0 should be on grid (x mod 5 = 0)
        let on_line = img.get_pixel(crate::image::Index([0i64, 3]));
        // pixel at x=2 should be off-grid
        let off_line = img.get_pixel(crate::image::Index([2i64, 3]));
        assert!(on_line > off_line, "on_line={on_line} off_line={off_line}");
    }
}
