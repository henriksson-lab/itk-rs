//! Recursive (IIR) Gaussian smoothing.
//! Analog to `itk::RecursiveGaussianImageFilter` /
//! `itk::SmoothingRecursiveGaussianImageFilter`.
//!
//! Uses the Young & van Vliet (1995) 3rd-order recursive approximation of the
//! Gaussian. Unlike the truncated direct-kernel approach in [`GaussianFilter`],
//! this runs in **O(n)** regardless of σ and introduces no truncation error.
//!
//! # Algorithm
//!
//! Coefficients `(B, a1, a2, a3)` are derived from σ following
//! Young & van Vliet (Signal Processing 44, 1995, §3).  For each axis a
//! causal (forward) pass is applied to the signal, followed by an anticausal
//! (backward) pass on the result.  Together they form a zero-phase symmetric
//! filter whose frequency response approximates a Gaussian.
//!
//! Boundary condition: constant extension (the signal is assumed to equal its
//! edge value outside the image domain).

use rayon::prelude::*;

use crate::image::{Image, Index, Region, iter_region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

// ---------------------------------------------------------------------------
// Coefficient computation (Young & van Vliet 1995)
// ---------------------------------------------------------------------------

/// Compute Young-van Vliet 3rd-order IIR coefficients for smoothing.
/// Returns `(B, a1, a2, a3)` such that
/// `y[n] = B·x[n] + a1·y[n−1] + a2·y[n−2] + a3·y[n−3]`.
fn yvv_coeffs(sigma: f64) -> (f64, f64, f64, f64) {
    let sigma = sigma.max(0.5); // filter undefined for very small sigma
    let q = if sigma >= 2.5 {
        0.98711 * sigma - 0.96330
    } else {
        3.97156 - 4.14554 * (1.0 - 0.26891 * sigma).sqrt()
    };
    let q2 = q * q;
    let q3 = q2 * q;
    let b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
    let b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3;
    let b2 = -(1.4281 * q2 + 1.26661 * q3);
    let b3 = 0.422205 * q3;
    let big_b = 1.0 - (b1 + b2 + b3) / b0;
    (big_b, b1 / b0, b2 / b0, b3 / b0)
}

// ---------------------------------------------------------------------------
// IIR line filter
// ---------------------------------------------------------------------------

/// Apply a single causal IIR pass along `data[line[0..n]]` (indices into data).
fn causal_pass(data: &mut Vec<f64>, line: &[usize], big_b: f64, a1: f64, a2: f64, a3: f64) {
    let n = line.len();
    // Boundary: steady-state = first sample (since B + a1+a2+a3 = 1)
    let init = data[line[0]];
    let mut ym1 = init;
    let mut ym2 = init;
    let mut ym3 = init;
    for i in 0..n {
        let v = big_b * data[line[i]] + a1 * ym1 + a2 * ym2 + a3 * ym3;
        ym3 = ym2;
        ym2 = ym1;
        ym1 = v;
        data[line[i]] = v;
    }
}

/// Apply a single anticausal IIR pass (runs backward over the line).
fn anticausal_pass(data: &mut Vec<f64>, line: &[usize], big_b: f64, a1: f64, a2: f64, a3: f64) {
    let n = line.len();
    let init = data[line[n - 1]];
    let mut ym1 = init;
    let mut ym2 = init;
    let mut ym3 = init;
    for i in (0..n).rev() {
        let v = big_b * data[line[i]] + a1 * ym1 + a2 * ym2 + a3 * ym3;
        ym3 = ym2;
        ym2 = ym1;
        ym1 = v;
        data[line[i]] = v;
    }
}

// ---------------------------------------------------------------------------
// Axis filter
// ---------------------------------------------------------------------------

/// Apply the full zero-phase recursive Gaussian filter along one axis of `img`.
fn filter_axis<P, const D: usize>(img: &mut Image<f64, D>, axis: usize, sigma_px: f64)
where
    P: NumericPixel,
{
    let size = img.region.size.0[axis];
    if size <= 1 {
        return;
    }
    let (big_b, a1, a2, a3) = yvv_coeffs(sigma_px);

    // Build a stripe region (size 1 along `axis`) to iterate over all 1-D lines
    let mut stripe_size = img.region.size.0;
    stripe_size[axis] = 1;
    let stripe_region = Region::new(img.region.index.0, stripe_size);

    // Collect all lines
    let mut lines: Vec<Vec<usize>> = Vec::new();
    iter_region(&stripe_region, |start| {
        let line: Vec<usize> = (0..size)
            .map(|k| {
                let mut idx = start.0;
                idx[axis] += k as i64;
                img.flat_index(Index(idx))
            })
            .collect();
        lines.push(line);
    });

    // Apply causal then anticausal pass per line (sequential — rayon can't easily
    // borrow disjoint lines of a single Vec without unsafe)
    for line in &lines {
        causal_pass(&mut img.data, line, big_b, a1, a2, a3);
        anticausal_pass(&mut img.data, line, big_b, a1, a2, a3);
    }
}

// ---------------------------------------------------------------------------
// Public filter structs
// ---------------------------------------------------------------------------

/// IIR Gaussian smoothing applied along **all** axes.
/// Analog to `itk::SmoothingRecursiveGaussianImageFilter`.
pub struct RecursiveGaussianFilter<S> {
    pub source: S,
    /// Standard deviation in physical units.
    pub sigma: f64,
}

impl<S> RecursiveGaussianFilter<S> {
    pub fn new(source: S, sigma: f64) -> Self {
        Self { source, sigma }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for RecursiveGaussianFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    // IIR has infinite support; we need the full image to initialise the filter
    // (or at least enough context). Request the largest region as input.
    fn input_region_for_output(&self, _output_region: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let spacing = self.source.spacing();

        // Work in f64 internally
        let full = self.source.generate_region(self.source.largest_region());
        let mut coeffs = Image::<f64, D> {
            region: full.region,
            spacing: full.spacing,
            origin: full.origin,
            data: full.data.iter().map(|p| p.to_f64()).collect(),
        };

        for axis in 0..D {
            let sigma_px = self.sigma / spacing[axis];
            filter_axis::<P, D>(&mut coeffs, axis, sigma_px);
        }

        // Crop to requested region and convert back to P
        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices
            .par_iter()
            .map(|&idx| P::from_f64(coeffs.get_pixel(idx)))
            .collect();

        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

/// Single-axis IIR Gaussian. Analog to `itk::RecursiveGaussianImageFilter`
/// with a specific `Direction` set.
pub struct RecursiveGaussianAxisFilter<S> {
    pub source: S,
    pub sigma: f64,
    /// Which axis to smooth (0 = first/x axis).
    pub axis: usize,
}

impl<S> RecursiveGaussianAxisFilter<S> {
    pub fn new(source: S, sigma: f64, axis: usize) -> Self {
        Self { source, sigma, axis }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for RecursiveGaussianAxisFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let spacing = self.source.spacing();
        let full = self.source.generate_region(self.source.largest_region());
        let mut coeffs = Image::<f64, D> {
            region: full.region,
            spacing: full.spacing,
            origin: full.origin,
            data: full.data.iter().map(|p| p.to_f64()).collect(),
        };

        let sigma_px = self.sigma / spacing[self.axis];
        filter_axis::<P, D>(&mut coeffs, self.axis, sigma_px);

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<P> = out_indices
            .par_iter()
            .map(|&idx| P::from_f64(coeffs.get_pixel(idx)))
            .collect();

        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    fn impulse_1d(n: usize) -> Image<f32, 1> {
        let mid = (n / 2) as i64;
        let mut img = Image::<f32, 1>::allocate(Region::new([0], [n]), [1.0], [0.0], 0.0f32);
        img.set_pixel(Index([mid]), 1.0);
        img
    }

    #[test]
    fn energy_conserved_1d() {
        let img = impulse_1d(51);
        let f = RecursiveGaussianFilter::new(img, 2.0);
        let out = f.generate_region(f.largest_region());
        let sum: f32 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 0.02, "energy={sum}");
    }

    #[test]
    fn peak_at_centre() {
        let img = impulse_1d(51);
        let mid = 25i64;
        let f = RecursiveGaussianFilter::new(img, 2.0);
        let out = f.generate_region(f.largest_region());
        let peak = out.get_pixel(Index([mid]));
        let neighbour = out.get_pixel(Index([mid + 3]));
        assert!(peak > neighbour, "peak={peak} neighbour={neighbour}");
    }

    #[test]
    fn constant_image_unchanged() {
        let img = Image::<f32, 2>::allocate(
            Region::new([0, 0], [15, 15]), [1.0; 2], [0.0; 2], 4.0f32,
        );
        let f = RecursiveGaussianFilter::new(img, 1.5);
        let out = f.generate_region(f.largest_region());
        // Interior pixels should remain 4.0 (DC gain = 1)
        for i in 3..12i64 {
            for j in 3..12i64 {
                let v = out.get_pixel(Index([i, j]));
                assert!((v - 4.0).abs() < 0.05, "at [{i},{j}]: {v}");
            }
        }
    }

    #[test]
    fn single_axis_filter() {
        let img = impulse_1d(31);
        let f = RecursiveGaussianAxisFilter::new(img, 2.0, 0);
        let out = f.generate_region(f.largest_region());
        let sum: f32 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 0.02, "energy={sum}");
    }
}
