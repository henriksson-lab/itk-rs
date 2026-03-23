//! Thresholding filters.
//!
//! Analogs to ITK's `itk::BinaryThresholdImageFilter`,
//! `itk::ThresholdImageFilter`, and `itk::OtsuThresholdImageFilter`.

use rayon::prelude::*;

use crate::image::{Image, Region, iter_region, Index};
use crate::pixel::{NumericPixel, Pixel};
use crate::source::ImageSource;

// ===========================================================================
// Binary threshold
// ===========================================================================

/// Map pixels to `inside_value` or `outside_value` based on whether they fall
/// in `[lower_threshold, upper_threshold]`.
/// Analog to `itk::BinaryThresholdImageFilter`.
pub struct BinaryThresholdFilter<S, Q, P> {
    pub source: S,
    pub lower_threshold: f64,
    pub upper_threshold: f64,
    pub inside_value: Q,
    pub outside_value: Q,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, Q: Copy, P> BinaryThresholdFilter<S, Q, P> {
    pub fn new(source: S, lower: f64, upper: f64, inside: Q, outside: Q) -> Self {
        Self {
            source,
            lower_threshold: lower,
            upper_threshold: upper,
            inside_value: inside,
            outside_value: outside,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<P, Q, S, const D: usize> ImageSource<Q, D> for BinaryThresholdFilter<S, Q, P>
where
    P: NumericPixel,
    Q: Pixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<Q, D> {
        let input = self.source.generate_region(requested);
        let (lo, hi, iv, ov) = (
            self.lower_threshold, self.upper_threshold,
            self.inside_value, self.outside_value,
        );
        let data: Vec<Q> = input.data.par_iter()
            .map(|&p| {
                let v = p.to_f64();
                if v >= lo && v <= hi { iv } else { ov }
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Threshold (in-place style)
// ===========================================================================

/// Replace pixels **outside** `[lower, upper]` with `outside_value`, leaving
/// in-range pixels unchanged.
/// Analog to `itk::ThresholdImageFilter` (default `OutsideValue = 0`).
pub struct ThresholdFilter<S> {
    pub source: S,
    pub lower: f64,
    pub upper: f64,
    pub outside_value: f64,
}

impl<S> ThresholdFilter<S> {
    pub fn new(source: S, lower: f64, upper: f64) -> Self {
        Self { source, lower, upper, outside_value: 0.0 }
    }

    pub fn with_outside_value(mut self, v: f64) -> Self {
        self.outside_value = v;
        self
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for ThresholdFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let (lo, hi, ov) = (self.lower, self.upper, self.outside_value);
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| {
                let v = p.to_f64();
                if v >= lo && v <= hi { p } else { P::from_f64(ov) }
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Otsu threshold
// ===========================================================================

/// Automatic threshold via Otsu's method (maximises inter-class variance).
/// Analog to `itk::OtsuThresholdImageFilter`.
///
/// The threshold is computed from the **full** source image's intensity
/// histogram and applied as a binary threshold.
pub struct OtsuThresholdFilter<S, Q, P> {
    pub source: S,
    pub inside_value: Q,
    pub outside_value: Q,
    /// Number of histogram bins used for threshold search (default: 256).
    pub num_bins: usize,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, Q: Copy, P> OtsuThresholdFilter<S, Q, P> {
    pub fn new(source: S, inside: Q, outside: Q) -> Self {
        Self { source, inside_value: inside, outside_value: outside, num_bins: 256,
               _phantom: std::marker::PhantomData }
    }

    pub fn with_num_bins(mut self, bins: usize) -> Self {
        self.num_bins = bins;
        self
    }
}

/// Compute Otsu threshold from a histogram.
/// Returns the intensity value that maximises inter-class variance.
fn otsu_threshold(histogram: &[u64], min_val: f64, max_val: f64) -> f64 {
    let n = histogram.len();
    let total: u64 = histogram.iter().sum();
    if total == 0 {
        return (min_val + max_val) / 2.0;
    }

    // Precompute per-bin intensity centres
    let bin_width = (max_val - min_val) / n as f64;

    // Total weighted sum
    let total_sum: f64 = histogram.iter().enumerate()
        .map(|(i, &c)| c as f64 * (min_val + (i as f64 + 0.5) * bin_width))
        .sum();

    let mut w0 = 0u64;       // weight class 0
    let mut sum0 = 0.0f64;   // sum class 0

    let mut best_var = f64::NEG_INFINITY;
    let mut best_t = min_val;

    for i in 0..n {
        let intensity = min_val + (i as f64 + 0.5) * bin_width;
        w0 += histogram[i];
        sum0 += histogram[i] as f64 * intensity;

        let w1 = total - w0;
        if w0 == 0 || w1 == 0 {
            continue;
        }

        let mu0 = sum0 / w0 as f64;
        let mu1 = (total_sum - sum0) / w1 as f64;
        let inter = w0 as f64 * w1 as f64 * (mu0 - mu1).powi(2);

        if inter > best_var {
            best_var = inter;
            best_t = intensity;
        }
    }
    best_t
}

impl<P, Q, S, const D: usize> ImageSource<Q, D> for OtsuThresholdFilter<S, Q, P>
where
    P: NumericPixel,
    Q: Pixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<Q, D> {
        let full = self.source.generate_region(self.source.largest_region());

        // Find intensity range
        let (imin, imax) = full.data.iter().fold((f64::MAX, f64::MIN), |(mn, mx), &p| {
            let v = p.to_f64();
            (mn.min(v), mx.max(v))
        });

        // Build histogram
        let nb = self.num_bins;
        let range = imax - imin;
        let mut histogram = vec![0u64; nb];
        for &p in &full.data {
            let v = p.to_f64();
            let bin = if range < 1e-15 {
                0
            } else {
                ((v - imin) / range * nb as f64) as usize
            }.min(nb - 1);
            histogram[bin] += 1;
        }

        let threshold = otsu_threshold(&histogram, imin, imax);
        let (iv, ov) = (self.inside_value, self.outside_value);

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<Q> = out_indices.par_iter()
            .map(|&idx| {
                if full.get_pixel(idx).to_f64() >= threshold { iv } else { ov }
            })
            .collect();

        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

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
    fn binary_threshold_basic() {
        let img = ramp_1d(10); // 0..9
        let f = BinaryThresholdFilter::<_, u8, f32>::new(img, 3.0, 7.0, 1u8, 0u8);
        let out = f.generate_region(f.largest_region());
        assert_eq!(out.data[2], 0); // 2 < 3 → outside
        assert_eq!(out.data[3], 1); // 3 in range → inside
        assert_eq!(out.data[7], 1); // 7 in range → inside
        assert_eq!(out.data[8], 0); // 8 > 7 → outside
    }

    #[test]
    fn threshold_filter_replaces_outside() {
        let img = ramp_1d(10);
        let f = ThresholdFilter::new(img, 3.0, 7.0).with_outside_value(-1.0);
        let out = f.generate_region(f.largest_region());
        assert!((out.get_pixel(Index([2])) - (-1.0)).abs() < 1e-6);
        assert!((out.get_pixel(Index([5])) - 5.0).abs() < 1e-6);
        assert!((out.get_pixel(Index([9])) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn otsu_bimodal() {
        // Build a bimodal image: 5 pixels at 10 and 5 at 90
        let mut img = Image::<f32, 1>::allocate(Region::new([0], [10]), [1.0], [0.0], 0.0f32);
        for i in 0..5i64   { img.set_pixel(Index([i]),   10.0); }
        for i in 5..10i64  { img.set_pixel(Index([i]),   90.0); }

        let f = OtsuThresholdFilter::<_, u8, f32>::new(img, 1u8, 0u8);
        let out = f.generate_region(f.largest_region());

        // Pixels at value 10 should be 0, at 90 should be 1
        for i in 0..5  { assert_eq!(out.data[i], 0, "pixel {i} should be below threshold"); }
        for i in 5..10 { assert_eq!(out.data[i], 1, "pixel {i} should be above threshold"); }
    }

    #[test]
    fn otsu_constant_image_does_not_panic() {
        let img = Image::<f32, 1>::allocate(Region::new([0], [8]), [1.0], [0.0], 5.0f32);
        let f = OtsuThresholdFilter::<_, u8, f32>::new(img, 1u8, 0u8);
        let _ = f.generate_region(f.largest_region());
    }
}
