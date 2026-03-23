//! Bilateral filter. Analog to `itk::BilateralImageFilter`.
//!
//! Edge-preserving smoother: weights each neighbour by both spatial distance
//! and intensity (range) similarity.
//!
//! `w(x, y) = exp(−‖x−y‖² / (2 σ_s²)) · exp(−(I(x)−I(y))² / (2 σ_r²))`
//!
//! `I_out(x) = Σ_y w(x,y)·I(y) / Σ_y w(x,y)`
//!
//! Non-separable; O(n · (2r+1)^D) per axis where r = ⌈3σ_s⌉.
//! Boundary: zero-flux Neumann (clamp-to-edge).

use rayon::prelude::*;

use crate::image::{Image, Index, Region, iter_region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

/// Bilateral filter.
pub struct BilateralFilter<S> {
    pub source: S,
    /// Spatial (domain) Gaussian standard deviation in **physical units**.
    pub sigma_spatial: f64,
    /// Range (photometric) Gaussian standard deviation in **intensity units**.
    pub sigma_range: f64,
}

impl<S> BilateralFilter<S> {
    pub fn new(source: S, sigma_spatial: f64, sigma_range: f64) -> Self {
        Self { source, sigma_spatial, sigma_range }
    }

    fn radius_for_spacing(&self, spacing: f64) -> usize {
        let sigma_px = self.sigma_spatial / spacing;
        (3.0 * sigma_px).ceil() as usize
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for BilateralFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let spacing = self.source.spacing();
        let radii = {
            let mut r = [0usize; D];
            for d in 0..D { r[d] = self.radius_for_spacing(spacing[d]); }
            r
        };
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let spacing = self.source.spacing();
        let bounds = self.source.largest_region();

        let radii: [usize; D] = {
            let mut r = [0usize; D];
            for d in 0..D { r[d] = self.radius_for_spacing(spacing[d]); }
            r
        };

        let input_region = requested.padded_per_axis(&radii).clipped_to(&bounds);
        let input = self.source.generate_region(input_region);

        // Precompute 1/sigma² constants
        let inv_2ss2: Vec<f64> = (0..D)
            .map(|d| {
                let sp = spacing[d];
                1.0 / (2.0 * (self.sigma_spatial / sp).powi(2))
            })
            .collect();
        let inv_2sr2 = 1.0 / (2.0 * self.sigma_range * self.sigma_range);

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices
            .par_iter()
            .map(|&center_idx| {
                let center_val = input.get_pixel(center_idx).to_f64();

                let nbr_size = {
                    let mut s = [0usize; D];
                    for d in 0..D { s[d] = 2 * radii[d] + 1; }
                    s
                };
                let nbr_start = {
                    let mut s = center_idx.0;
                    for d in 0..D { s[d] -= radii[d] as i64; }
                    s
                };
                let nbr_region = Region::new(nbr_start, nbr_size);

                let mut acc = 0.0f64;
                let mut weight_sum = 0.0f64;

                iter_region(&nbr_region, |nbr_idx| {
                    // Clamp to input bounds
                    let mut clamped = nbr_idx.0;
                    for d in 0..D {
                        clamped[d] = clamped[d]
                            .max(input.region.index.0[d])
                            .min(input.region.index.0[d] + input.region.size.0[d] as i64 - 1);
                    }
                    let nbr_val = input.get_pixel(Index(clamped)).to_f64();

                    // Spatial weight (in pixel space)
                    let mut spatial_dist2 = 0.0f64;
                    for d in 0..D {
                        let diff = (nbr_idx.0[d] - center_idx.0[d]) as f64;
                        spatial_dist2 += diff * diff * inv_2ss2[d];
                    }
                    let w_s = (-spatial_dist2).exp();

                    // Range weight
                    let range_diff2 = (nbr_val - center_val).powi(2) * inv_2sr2;
                    let w_r = (-range_diff2).exp();

                    let w = w_s * w_r;
                    acc += w * nbr_val;
                    weight_sum += w;
                });

                P::from_f64(if weight_sum > 0.0 { acc / weight_sum } else { center_val })
            })
            .collect();

        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    #[test]
    fn constant_image_preserved() {
        let img = Image::<f32, 2>::allocate(
            Region::new([0, 0], [10, 10]), [1.0; 2], [0.0; 2], 5.0f32,
        );
        let f = BilateralFilter::new(img, 1.5, 50.0);
        let out = f.generate_region(f.largest_region());
        for &v in &out.data {
            assert!((v - 5.0).abs() < 0.01, "expected 5.0, got {v}");
        }
    }

    #[test]
    fn edge_preserved() {
        // Step edge: left half = 0, right half = 100.
        // The bilateral filter should preserve the step (not blur across it).
        let size = 20usize;
        let mid = size / 2;
        let mut img = Image::<f32, 1>::allocate(Region::new([0], [size]), [1.0], [0.0], 0.0f32);
        for i in mid as i64..size as i64 {
            img.set_pixel(Index([i]), 100.0);
        }

        // Small range sigma → edge-preserving
        let f = BilateralFilter::new(img, 2.0, 10.0);
        let out = f.generate_region(f.largest_region());

        // Pixels deep inside each region should retain their values
        let left  = out.get_pixel(Index([2]));
        let right = out.get_pixel(Index([size as i64 - 3]));
        assert!(left < 5.0,  "left region blurred too much: {left}");
        assert!(right > 95.0, "right region blurred too much: {right}");
    }
}
