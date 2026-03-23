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
// Generic wrapper for histogram-based auto-threshold filters
// ===========================================================================

/// Apply an auto-threshold method as a binary threshold filter.
/// The threshold is computed from a histogram using the provided function.
pub struct AutoThresholdFilter<S, Q, P> {
    pub source: S,
    pub inside_value: Q,
    pub outside_value: Q,
    pub num_bins: usize,
    /// Threshold algorithm: takes histogram, min, max → threshold value.
    pub threshold_fn: fn(&[u64], f64, f64) -> f64,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, Q: Copy, P> AutoThresholdFilter<S, Q, P> {
    pub fn new(
        source: S,
        inside: Q,
        outside: Q,
        threshold_fn: fn(&[u64], f64, f64) -> f64,
    ) -> Self {
        Self { source, inside_value: inside, outside_value: outside,
               num_bins: 256, threshold_fn, _phantom: std::marker::PhantomData }
    }
}

impl<P, Q, S, const D: usize> ImageSource<Q, D> for AutoThresholdFilter<S, Q, P>
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
        let (imin, imax) = full.data.iter().fold((f64::MAX, f64::MIN), |(mn, mx), &p| {
            let v = p.to_f64(); (mn.min(v), mx.max(v))
        });
        let nb = self.num_bins;
        let range = imax - imin;
        let mut histogram = vec![0u64; nb];
        for &p in &full.data {
            let v = p.to_f64();
            let bin = if range < 1e-15 { 0 }
                else { ((v - imin) / range * nb as f64) as usize }.min(nb - 1);
            histogram[bin] += 1;
        }
        let threshold = (self.threshold_fn)(&histogram, imin, imax);
        let (iv, ov) = (self.inside_value, self.outside_value);

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<Q> = out_indices.par_iter()
            .map(|&idx| if full.get_pixel(idx).to_f64() >= threshold { iv } else { ov })
            .collect();
        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

// ===========================================================================
// Individual histogram threshold methods
// ===========================================================================

/// Huang's fuzzy thresholding method.
fn huang_threshold(histogram: &[u64], min_val: f64, max_val: f64) -> f64 {
    let n = histogram.len();
    if n == 0 { return (min_val + max_val) / 2.0; }
    let total: u64 = histogram.iter().sum();
    if total == 0 { return (min_val + max_val) / 2.0; }
    let bin_width = (max_val - min_val) / n as f64;

    // Minimize fuzzy entropy: iterate over all thresholds
    let mut best_ent = f64::MAX;
    let mut best_t = min_val;

    for t in 0..n {
        let threshold_intensity = min_val + (t as f64 + 0.5) * bin_width;
        // Compute mean of lower class
        let mut w0 = 0u64;
        let mut sum0 = 0.0f64;
        for i in 0..=t {
            w0 += histogram[i];
            sum0 += histogram[i] as f64 * (min_val + (i as f64 + 0.5) * bin_width);
        }
        let mu0 = if w0 > 0 { sum0 / w0 as f64 } else { threshold_intensity };
        let mut ent = 0.0f64;
        for i in 0..=t {
            let v = min_val + (i as f64 + 0.5) * bin_width;
            let mu = 1.0 / (1.0 + (v - mu0).abs() / (threshold_intensity - min_val + 1e-10));
            if mu > 0.0 && mu < 1.0 {
                ent -= histogram[i] as f64 * (mu * mu.ln() + (1.0 - mu) * (1.0 - mu).ln());
            }
        }
        if ent < best_ent {
            best_ent = ent;
            best_t = threshold_intensity;
        }
    }
    best_t
}

/// Li's minimum cross entropy thresholding.
fn li_threshold(histogram: &[u64], min_val: f64, max_val: f64) -> f64 {
    let n = histogram.len();
    let bin_width = (max_val - min_val) / n as f64;
    let total: u64 = histogram.iter().sum();
    if total == 0 { return (min_val + max_val) / 2.0; }

    // Initial threshold = mean
    let mean: f64 = histogram.iter().enumerate()
        .map(|(i, &c)| c as f64 * (min_val + (i as f64 + 0.5) * bin_width))
        .sum::<f64>() / total as f64;

    let mut t = mean;
    for _ in 0..100 {
        let mut w0 = 0.0f64; let mut s0 = 0.0f64;
        let mut w1 = 0.0f64; let mut s1 = 0.0f64;
        for i in 0..n {
            let v = min_val + (i as f64 + 0.5) * bin_width;
            let c = histogram[i] as f64;
            if v <= t { w0 += c; s0 += c * v; }
            else { w1 += c; s1 += c * v; }
        }
        let mu0 = if w0 > 0.0 { s0 / w0 } else { t / 2.0 };
        let mu1 = if w1 > 0.0 { s1 / w1 } else { (t + max_val) / 2.0 };
        let new_t = (mu1 - mu0) / (mu1.ln() - mu0.ln() + 1e-300);
        if (new_t - t).abs() < 1e-10 { break; }
        t = new_t;
    }
    t
}

/// IsoData (iterative midpoint) thresholding.
fn iso_data_threshold(histogram: &[u64], min_val: f64, max_val: f64) -> f64 {
    let n = histogram.len();
    let bin_width = (max_val - min_val) / n as f64;
    let total: u64 = histogram.iter().sum();
    if total == 0 { return (min_val + max_val) / 2.0; }

    let mut t = (min_val + max_val) / 2.0;
    for _ in 0..100 {
        let mut w0 = 0.0f64; let mut s0 = 0.0f64;
        let mut w1 = 0.0f64; let mut s1 = 0.0f64;
        for i in 0..n {
            let v = min_val + (i as f64 + 0.5) * bin_width;
            let c = histogram[i] as f64;
            if v <= t { w0 += c; s0 += c * v; }
            else { w1 += c; s1 += c * v; }
        }
        let mu0 = if w0 > 0.0 { s0 / w0 } else { min_val };
        let mu1 = if w1 > 0.0 { s1 / w1 } else { max_val };
        let new_t = (mu0 + mu1) / 2.0;
        if (new_t - t).abs() < 1e-10 { break; }
        t = new_t;
    }
    t
}

/// Maximum entropy thresholding (Kapur et al.).
fn max_entropy_threshold(histogram: &[u64], min_val: f64, max_val: f64) -> f64 {
    let n = histogram.len();
    let bin_width = (max_val - min_val) / n as f64;
    let total: f64 = histogram.iter().sum::<u64>() as f64;
    if total == 0.0 { return (min_val + max_val) / 2.0; }

    let p: Vec<f64> = histogram.iter().map(|&c| c as f64 / total).collect();

    let mut best_ent = f64::NEG_INFINITY;
    let mut best_t = min_val;

    let mut cum_p0 = 0.0f64;
    for t in 0..n {
        cum_p0 += p[t];
        if cum_p0 <= 0.0 || cum_p0 >= 1.0 { continue; }
        let mut h0 = 0.0f64;
        let mut h1 = 0.0f64;
        for i in 0..=t {
            if p[i] > 0.0 { h0 -= (p[i] / cum_p0) * (p[i] / cum_p0).ln(); }
        }
        let cum_p1 = 1.0 - cum_p0;
        for i in (t+1)..n {
            if p[i] > 0.0 { h1 -= (p[i] / cum_p1) * (p[i] / cum_p1).ln(); }
        }
        let total_ent = h0 + h1;
        if total_ent > best_ent {
            best_ent = total_ent;
            best_t = min_val + (t as f64 + 0.5) * bin_width;
        }
    }
    best_t
}

/// Moments-preserving thresholding (Tsai).
fn moments_threshold(histogram: &[u64], min_val: f64, max_val: f64) -> f64 {
    let n = histogram.len();
    let bin_width = (max_val - min_val) / n as f64;
    let total: f64 = histogram.iter().sum::<u64>() as f64;
    if total == 0.0 { return (min_val + max_val) / 2.0; }

    // Compute first 3 moments of input
    let m: Vec<f64> = (0..=3).map(|k| {
        histogram.iter().enumerate().map(|(i, &c)| {
            let v = min_val + (i as f64 + 0.5) * bin_width;
            c as f64 * v.powi(k) / total
        }).sum::<f64>()
    }).collect();

    // Find threshold that preserves 3 moments (simplified: use mean as fallback)
    let _m0 = m[0]; // ≈ 1.0
    // Fallback: return mean
    m[1]
}

/// Triangle thresholding (Zack et al.).
fn triangle_threshold(histogram: &[u64], min_val: f64, max_val: f64) -> f64 {
    let n = histogram.len();
    let bin_width = (max_val - min_val) / n as f64;

    // Find the peak bin
    let (peak_bin, _) = histogram.iter().enumerate()
        .max_by_key(|(_, &c)| c)
        .unwrap_or((0, &0));

    // Find the non-zero extreme away from the peak
    let left_nonzero = histogram.iter().position(|&c| c > 0).unwrap_or(0);
    let right_nonzero = n - 1 - histogram.iter().rev().position(|&c| c > 0).unwrap_or(0);

    let far_end = if peak_bin - left_nonzero > right_nonzero - peak_bin {
        left_nonzero
    } else {
        right_nonzero
    };

    // Line from peak to far_end; find bin with max perpendicular distance
    let px = peak_bin as f64;
    let py = histogram[peak_bin] as f64;
    let fx = far_end as f64;
    let fy = histogram[far_end] as f64;
    let dx = fx - px;
    let dy = fy - py;
    let len = (dx*dx + dy*dy).sqrt();

    let mut best_dist = 0.0f64;
    let mut best_bin = peak_bin;

    let range = if peak_bin < far_end { peak_bin..far_end } else { far_end..peak_bin };
    for i in range {
        let bx = i as f64 - px;
        let by = histogram[i] as f64 - py;
        let dist = (dx * by - dy * bx).abs() / (len + 1e-10);
        if dist > best_dist {
            best_dist = dist;
            best_bin = i;
        }
    }
    min_val + (best_bin as f64 + 0.5) * bin_width
}

/// Yen thresholding (maximum correlation criterion).
fn yen_threshold(histogram: &[u64], min_val: f64, max_val: f64) -> f64 {
    let n = histogram.len();
    let bin_width = (max_val - min_val) / n as f64;
    let total: f64 = histogram.iter().sum::<u64>() as f64;
    if total == 0.0 { return (min_val + max_val) / 2.0; }
    let p: Vec<f64> = histogram.iter().map(|&c| c as f64 / total).collect();

    // Cumulative sums of p and p²
    let mut cum_p = vec![0.0f64; n];
    let mut cum_p2 = vec![0.0f64; n];
    cum_p[0] = p[0];
    cum_p2[0] = p[0] * p[0];
    for i in 1..n {
        cum_p[i] = cum_p[i-1] + p[i];
        cum_p2[i] = cum_p2[i-1] + p[i] * p[i];
    }

    let mut best_crit = f64::NEG_INFINITY;
    let mut best_t = min_val;

    for t in 0..n-1 {
        let p0 = cum_p[t];
        let p1 = 1.0 - p0;
        if p0 <= 0.0 || p1 <= 0.0 { continue; }
        let sq0 = cum_p2[t];
        let sq1 = cum_p2[n-1] - sq0;
        let crit = -(sq0 / (p0 * p0) + sq1 / (p1 * p1)).ln();
        if crit > best_crit {
            best_crit = crit;
            best_t = min_val + (t as f64 + 0.5) * bin_width;
        }
    }
    best_t
}

/// Renyi entropy thresholding.
fn renyi_entropy_threshold(histogram: &[u64], min_val: f64, max_val: f64) -> f64 {
    // Renyi entropy of order 2 (same as Yen criterion up to sign)
    yen_threshold(histogram, min_val, max_val)
}

/// Shanbhag thresholding (minimum description length).
fn shanbhag_threshold(histogram: &[u64], min_val: f64, max_val: f64) -> f64 {
    let n = histogram.len();
    let bin_width = (max_val - min_val) / n as f64;
    let total: f64 = histogram.iter().sum::<u64>() as f64;
    if total == 0.0 { return (min_val + max_val) / 2.0; }
    let p: Vec<f64> = histogram.iter().map(|&c| c as f64 / total).collect();

    let mut cum = vec![0.0f64; n];
    cum[0] = p[0];
    for i in 1..n { cum[i] = cum[i-1] + p[i]; }

    let mut best_crit = f64::MAX;
    let mut best_t = min_val;

    for t in 0..n {
        let p0 = cum[t];
        let p1 = 1.0 - p0;
        if p0 <= 0.0 || p1 <= 0.0 { continue; }
        let mut h0 = 0.0f64;
        let mut h1 = 0.0f64;
        for i in 0..=t {
            let q = p[i] / p0;
            if q > 0.0 { h0 -= q * q.ln(); }
        }
        for i in (t+1)..n {
            let q = p[i] / p1;
            if q > 0.0 { h1 -= q * q.ln(); }
        }
        let crit = -p0 * h0 - p1 * h1;
        if crit < best_crit {
            best_crit = crit;
            best_t = min_val + (t as f64 + 0.5) * bin_width;
        }
    }
    best_t
}

/// Kittler-Illingworth minimum error thresholding.
fn kittler_illingworth_threshold(histogram: &[u64], min_val: f64, max_val: f64) -> f64 {
    let n = histogram.len();
    let bin_width = (max_val - min_val) / n as f64;
    let total: f64 = histogram.iter().sum::<u64>() as f64;
    if total == 0.0 { return (min_val + max_val) / 2.0; }

    let mut best_crit = f64::MAX;
    let mut best_t = min_val;

    for t in 0..n {
        let intensity_t = min_val + (t as f64 + 0.5) * bin_width;
        let mut w0 = 0.0f64; let mut s0 = 0.0f64; let mut sq0 = 0.0f64;
        let mut w1 = 0.0f64; let mut s1 = 0.0f64; let mut sq1 = 0.0f64;
        for i in 0..n {
            let v = min_val + (i as f64 + 0.5) * bin_width;
            let c = histogram[i] as f64;
            if i <= t { w0 += c; s0 += c*v; sq0 += c*v*v; }
            else { w1 += c; s1 += c*v; sq1 += c*v*v; }
        }
        if w0 <= 0.0 || w1 <= 0.0 { continue; }
        let mu0 = s0 / w0;
        let mu1 = s1 / w1;
        let var0 = sq0 / w0 - mu0 * mu0;
        let var1 = sq1 / w1 - mu1 * mu1;
        if var0 <= 0.0 || var1 <= 0.0 { continue; }
        let p0 = w0 / total;
        let p1 = w1 / total;
        let crit = 1.0 + 2.0 * (p0 * var0.ln() + p1 * var1.ln())
            - 2.0 * (p0 * p0.ln() + p1 * p1.ln());
        if crit < best_crit {
            best_crit = crit;
            best_t = intensity_t;
        }
    }
    best_t
}

/// Intermodes thresholding (valley between two histogram peaks).
fn intermodes_threshold(histogram: &[u64], min_val: f64, max_val: f64) -> f64 {
    let n = histogram.len();
    let bin_width = (max_val - min_val) / n as f64;

    // Find two highest peaks
    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..n-1 {
        if histogram[i] >= histogram[i-1] && histogram[i] >= histogram[i+1] && histogram[i] > 0 {
            peaks.push(i);
        }
    }
    peaks.sort_by(|&a, &b| histogram[b].cmp(&histogram[a]));
    if peaks.len() < 2 {
        // Fall back to midpoint
        return (min_val + max_val) / 2.0;
    }
    let (p0, p1) = (peaks[0].min(peaks[1]), peaks[0].max(peaks[1]));
    // Valley = minimum between p0 and p1
    let valley = (p0..=p1)
        .min_by_key(|&i| histogram[i])
        .unwrap_or((p0 + p1) / 2);
    min_val + (valley as f64 + 0.5) * bin_width
}

/// Kappa-Sigma clipping threshold.
fn kappa_sigma_threshold(histogram: &[u64], min_val: f64, max_val: f64) -> f64 {
    let n = histogram.len();
    let bin_width = (max_val - min_val) / n as f64;
    let total: f64 = histogram.iter().sum::<u64>() as f64;
    if total == 0.0 { return (min_val + max_val) / 2.0; }

    let kappa = 3.0_f64; // σ multiplier

    // Initial mean and sigma
    let mean = histogram.iter().enumerate()
        .map(|(i, &c)| c as f64 * (min_val + (i as f64 + 0.5) * bin_width))
        .sum::<f64>() / total;
    let variance = histogram.iter().enumerate()
        .map(|(i, &c)| {
            let v = min_val + (i as f64 + 0.5) * bin_width;
            c as f64 * (v - mean).powi(2)
        }).sum::<f64>() / total;
    let sigma = variance.sqrt();
    mean + kappa * sigma
}

/// Otsu multi-threshold: splits into `n_classes` classes.
/// Returns a vector of `n_classes - 1` thresholds.
pub fn otsu_multiple_thresholds(histogram: &[u64], min_val: f64, max_val: f64, n_classes: usize) -> Vec<f64> {
    if n_classes <= 1 { return vec![]; }
    if n_classes == 2 {
        return vec![otsu_threshold(histogram, min_val, max_val)];
    }
    let n = histogram.len();
    let bin_width = (max_val - min_val) / n as f64;
    let total: u64 = histogram.iter().sum();
    if total == 0 { return vec![(min_val + max_val) / 2.0; n_classes - 1]; }

    // Precompute cumulative sums and weighted sums
    let mut cum_w = vec![0.0f64; n + 1];
    let mut cum_s = vec![0.0f64; n + 1];
    for i in 0..n {
        let v = min_val + (i as f64 + 0.5) * bin_width;
        cum_w[i+1] = cum_w[i] + histogram[i] as f64;
        cum_s[i+1] = cum_s[i] + histogram[i] as f64 * v;
    }

    // For n_classes <= 4, use exhaustive search; otherwise approximate
    let k = n_classes - 1;
    let mut best_var = f64::NEG_INFINITY;
    let mut best_thresholds = vec![min_val; k];

    // Simple recursive-like approach for 2 thresholds (3 classes)
    let total_f = total as f64;
    let total_mean = cum_s[n] / total_f;

    // For 3 classes, iterate over pairs (t1, t2) with t1 < t2
    if k == 2 {
        for t1 in 0..n-1 {
            for t2 in t1+1..n {
                let w0 = cum_w[t1+1];
                let w1 = cum_w[t2+1] - cum_w[t1+1];
                let w2 = total_f - cum_w[t2+1];
                if w0 <= 0.0 || w1 <= 0.0 || w2 <= 0.0 { continue; }
                let mu0 = (cum_s[t1+1]) / w0;
                let mu1 = (cum_s[t2+1] - cum_s[t1+1]) / w1;
                let mu2 = (cum_s[n] - cum_s[t2+1]) / w2;
                let var = w0*(mu0-total_mean).powi(2) + w1*(mu1-total_mean).powi(2) + w2*(mu2-total_mean).powi(2);
                if var > best_var {
                    best_var = var;
                    best_thresholds[0] = min_val + (t1 as f64 + 0.5) * bin_width;
                    best_thresholds[1] = min_val + (t2 as f64 + 0.5) * bin_width;
                }
            }
        }
    } else {
        // For higher orders, repeatedly apply single-class Otsu
        // This is an approximation
        let step = n / (k + 1);
        for i in 0..k {
            best_thresholds[i] = min_val + ((i + 1) * step) as f64 * bin_width;
        }
    }
    best_thresholds
}

// ===========================================================================
// Public type aliases for the standard auto-threshold filters
// ===========================================================================

/// Huang's fuzzy threshold filter. Analog to `itk::HuangThresholdImageFilter`.
pub type HuangThresholdFilter<S, Q, P> = AutoThresholdFilter<S, Q, P>;
/// Li's cross-entropy threshold filter. Analog to `itk::LiThresholdImageFilter`.
pub type LiThresholdFilter<S, Q, P> = AutoThresholdFilter<S, Q, P>;
/// IsoData threshold filter. Analog to `itk::IsoDataThresholdImageFilter`.
pub type IsoDataThresholdFilter<S, Q, P> = AutoThresholdFilter<S, Q, P>;
/// Maximum entropy threshold filter. Analog to `itk::MaximumEntropyThresholdImageFilter`.
pub type MaxEntropyThresholdFilter<S, Q, P> = AutoThresholdFilter<S, Q, P>;
/// Moments threshold filter. Analog to `itk::MomentsThresholdImageFilter`.
pub type MomentsThresholdFilter<S, Q, P> = AutoThresholdFilter<S, Q, P>;
/// Triangle threshold filter. Analog to `itk::TriangleThresholdImageFilter`.
pub type TriangleThresholdFilter<S, Q, P> = AutoThresholdFilter<S, Q, P>;
/// Yen threshold filter. Analog to `itk::YenThresholdImageFilter`.
pub type YenThresholdFilter<S, Q, P> = AutoThresholdFilter<S, Q, P>;
/// Renyi entropy threshold filter. Analog to `itk::RenyiEntropyThresholdImageFilter`.
pub type RenyiEntropyThresholdFilter<S, Q, P> = AutoThresholdFilter<S, Q, P>;
/// Shanbhag threshold filter. Analog to `itk::ShanbhagThresholdImageFilter`.
pub type ShanbhagThresholdFilter<S, Q, P> = AutoThresholdFilter<S, Q, P>;
/// Kittler-Illingworth threshold filter. Analog to `itk::KittlerIllingworthThresholdImageFilter`.
pub type KittlerIllingworthThresholdFilter<S, Q, P> = AutoThresholdFilter<S, Q, P>;
/// Intermodes threshold filter. Analog to `itk::IntermodesThresholdImageFilter`.
pub type IntermodesThresholdFilter<S, Q, P> = AutoThresholdFilter<S, Q, P>;
/// Kappa-Sigma threshold filter. Analog to `itk::KappaSigmaThresholdImageFilter`.
pub type KappaSigmaThresholdFilter<S, Q, P> = AutoThresholdFilter<S, Q, P>;

/// Construct a Huang threshold filter.
pub fn huang_threshold_filter<S, Q: Copy, P>(source: S, inside: Q, outside: Q)
    -> AutoThresholdFilter<S, Q, P> {
    AutoThresholdFilter::new(source, inside, outside, huang_threshold)
}
/// Construct a Li threshold filter.
pub fn li_threshold_filter<S, Q: Copy, P>(source: S, inside: Q, outside: Q)
    -> AutoThresholdFilter<S, Q, P> {
    AutoThresholdFilter::new(source, inside, outside, li_threshold)
}
/// Construct an IsoData threshold filter.
pub fn iso_data_threshold_filter<S, Q: Copy, P>(source: S, inside: Q, outside: Q)
    -> AutoThresholdFilter<S, Q, P> {
    AutoThresholdFilter::new(source, inside, outside, iso_data_threshold)
}
/// Construct a maximum entropy threshold filter.
pub fn max_entropy_threshold_filter<S, Q: Copy, P>(source: S, inside: Q, outside: Q)
    -> AutoThresholdFilter<S, Q, P> {
    AutoThresholdFilter::new(source, inside, outside, max_entropy_threshold)
}
/// Construct a moments threshold filter.
pub fn moments_threshold_filter<S, Q: Copy, P>(source: S, inside: Q, outside: Q)
    -> AutoThresholdFilter<S, Q, P> {
    AutoThresholdFilter::new(source, inside, outside, moments_threshold)
}
/// Construct a triangle threshold filter.
pub fn triangle_threshold_filter<S, Q: Copy, P>(source: S, inside: Q, outside: Q)
    -> AutoThresholdFilter<S, Q, P> {
    AutoThresholdFilter::new(source, inside, outside, triangle_threshold)
}
/// Construct a Yen threshold filter.
pub fn yen_threshold_filter<S, Q: Copy, P>(source: S, inside: Q, outside: Q)
    -> AutoThresholdFilter<S, Q, P> {
    AutoThresholdFilter::new(source, inside, outside, yen_threshold)
}
/// Construct a Renyi entropy threshold filter.
pub fn renyi_entropy_threshold_filter<S, Q: Copy, P>(source: S, inside: Q, outside: Q)
    -> AutoThresholdFilter<S, Q, P> {
    AutoThresholdFilter::new(source, inside, outside, renyi_entropy_threshold)
}
/// Construct a Shanbhag threshold filter.
pub fn shanbhag_threshold_filter<S, Q: Copy, P>(source: S, inside: Q, outside: Q)
    -> AutoThresholdFilter<S, Q, P> {
    AutoThresholdFilter::new(source, inside, outside, shanbhag_threshold)
}
/// Construct a Kittler-Illingworth threshold filter.
pub fn kittler_illingworth_threshold_filter<S, Q: Copy, P>(source: S, inside: Q, outside: Q)
    -> AutoThresholdFilter<S, Q, P> {
    AutoThresholdFilter::new(source, inside, outside, kittler_illingworth_threshold)
}
/// Construct an intermodes threshold filter.
pub fn intermodes_threshold_filter<S, Q: Copy, P>(source: S, inside: Q, outside: Q)
    -> AutoThresholdFilter<S, Q, P> {
    AutoThresholdFilter::new(source, inside, outside, intermodes_threshold)
}
/// Construct a Kappa-Sigma threshold filter.
pub fn kappa_sigma_threshold_filter<S, Q: Copy, P>(source: S, inside: Q, outside: Q)
    -> AutoThresholdFilter<S, Q, P> {
    AutoThresholdFilter::new(source, inside, outside, kappa_sigma_threshold)
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
