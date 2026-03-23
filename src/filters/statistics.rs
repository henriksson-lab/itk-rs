//! Image statistics filters.
//!
//! Covers summary statistics, projections, and accumulation filters.

use rayon::prelude::*;

use crate::image::{Image, Region, Index, iter_region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

// ===========================================================================
// Image statistics result struct
// ===========================================================================

/// Summary statistics for an image.
#[derive(Debug, Clone, Copy)]
pub struct ImageStatistics {
    pub minimum: f64,
    pub maximum: f64,
    pub mean: f64,
    pub variance: f64,
    pub sigma: f64,
    pub sum: f64,
    pub count: usize,
}

/// Compute summary statistics for an image (serial, exact).
pub fn compute_statistics<P: NumericPixel, const D: usize>(image: &Image<P, D>) -> ImageStatistics {
    let n = image.data.len();
    if n == 0 {
        return ImageStatistics {
            minimum: 0.0, maximum: 0.0, mean: 0.0,
            variance: 0.0, sigma: 0.0, sum: 0.0, count: 0,
        };
    }
    let mut mn = f64::MAX;
    let mut mx = f64::MIN;
    let mut sum = 0.0f64;
    for &p in &image.data {
        let v = p.to_f64();
        if v < mn { mn = v; }
        if v > mx { mx = v; }
        sum += v;
    }
    let mean = sum / n as f64;
    let variance = image.data.iter()
        .map(|&p| { let d = p.to_f64() - mean; d * d })
        .sum::<f64>() / n as f64;
    ImageStatistics {
        minimum: mn, maximum: mx, mean,
        variance, sigma: variance.sqrt(), sum, count: n,
    }
}

// ===========================================================================
// StatisticsImageFilter wrapper
// ===========================================================================

/// Computes summary statistics from an image source.
/// Call `statistics()` after `update()`.
/// Analog to `itk::StatisticsImageFilter`.
pub struct StatisticsImageFilter<S> {
    pub source: S,
}

impl<S> StatisticsImageFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }

    /// Pull the full image and compute statistics.
    pub fn compute<P: NumericPixel, const D: usize>(&self) -> ImageStatistics
    where
        S: ImageSource<P, D>,
    {
        let full = self.source.generate_region(self.source.largest_region());
        compute_statistics(&full)
    }
}

// ===========================================================================
// MinimumMaximumImageFilter
// ===========================================================================

/// Computes only min and max. Analog to `itk::MinimumMaximumImageFilter`.
pub struct MinimumMaximumImageFilter<S> {
    pub source: S,
}

impl<S> MinimumMaximumImageFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }

    pub fn compute<P: NumericPixel, const D: usize>(&self) -> (f64, f64)
    where
        S: ImageSource<P, D>,
    {
        let full = self.source.generate_region(self.source.largest_region());
        full.data.iter().fold((f64::MAX, f64::MIN), |(mn, mx), &p| {
            let v = p.to_f64();
            (mn.min(v), mx.max(v))
        })
    }
}

// ===========================================================================
// Accumulate filter (sum along one axis → same-D image with axis size=1)
// ===========================================================================

/// Sum (accumulate) pixel values along one axis.
/// Output has the same number of dimensions but the accumulated axis has size 1.
/// Analog to `itk::AccumulateImageFilter`.
pub struct AccumulateFilter<S> {
    pub source: S,
    pub axis: usize,
}

impl<S> AccumulateFilter<S> {
    pub fn new(source: S, axis: usize) -> Self { Self { source, axis } }
}

impl<P, S, const D: usize> ImageSource<P, D> for AccumulateFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> {
        let src = self.source.largest_region();
        let mut size = src.size.0;
        size[self.axis] = 1;
        Region { index: src.index, size: crate::image::Size(size) }
    }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let src_full = self.source.generate_region(self.source.largest_region());
        let src_region = src_full.region;
        let axis = self.axis;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let mut acc = P::zero();
                let lo = src_region.index.0[axis];
                let len = src_region.size.0[axis] as i64;
                for i in 0..len {
                    let mut sidx = idx;
                    sidx.0[axis] = lo + i;
                    acc = acc + src_full.get_pixel(sidx);
                }
                acc
            })
            .collect();
        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

// ===========================================================================
// Projection filters (project along one axis)
// ===========================================================================

fn project_axis<P, S, F, const D: usize>(
    source: &S,
    requested: Region<D>,
    axis: usize,
    combine: F,
) -> Image<P, D>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
    F: Fn(f64, f64) -> f64 + Sync + Send,
{
    let src_full = source.generate_region(source.largest_region());
    let src_region = src_full.region;

    let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
    iter_region(&requested, |idx| out_indices.push(idx));

    let data: Vec<P> = out_indices.par_iter()
        .map(|&idx| {
            let lo = src_region.index.0[axis];
            let len = src_region.size.0[axis] as i64;
            let mut result = f64::NAN;
            for i in 0..len {
                let mut sidx = idx;
                sidx.0[axis] = lo + i;
                let v = src_full.get_pixel(sidx).to_f64();
                result = if result.is_nan() { v } else { combine(result, v) };
            }
            P::from_f64(if result.is_nan() { 0.0 } else { result })
        })
        .collect();
    Image { region: requested, spacing: source.spacing(), origin: source.origin(), data }
}

macro_rules! projection_filter {
    ($name:ident, $doc:literal, $combine:expr) => {
        #[doc = $doc]
        pub struct $name<S> {
            pub source: S,
            pub axis: usize,
        }

        impl<S> $name<S> {
            pub fn new(source: S, axis: usize) -> Self { Self { source, axis } }
        }

        impl<P, S, const D: usize> ImageSource<P, D> for $name<S>
        where
            P: NumericPixel,
            S: ImageSource<P, D> + Sync,
        {
            fn largest_region(&self) -> Region<D> {
                let src = self.source.largest_region();
                let mut size = src.size.0;
                size[self.axis] = 1;
                Region { index: src.index, size: crate::image::Size(size) }
            }
            fn spacing(&self) -> [f64; D] { self.source.spacing() }
            fn origin(&self) -> [f64; D] { self.source.origin() }
            fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
                self.source.largest_region()
            }
            fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
                project_axis(&self.source, requested, self.axis, $combine)
            }
        }
    };
}

projection_filter!(MaxProjectionFilter, "Maximum projection along one axis. Analog to `itk::MaximumProjectionImageFilter`.", f64::max);
projection_filter!(MinProjectionFilter, "Minimum projection along one axis. Analog to `itk::MinimumProjectionImageFilter`.", f64::min);
projection_filter!(SumProjectionFilter, "Sum projection along one axis. Analog to `itk::SumProjectionImageFilter`.", |a, b| a + b);
projection_filter!(MeanProjectionFilter, "Mean projection. Note: computes sum; divide externally for true mean. Analog to `itk::MeanProjectionImageFilter`.", |a, b| a + b);

/// Standard deviation projection along one axis.
/// Analog to `itk::StandardDeviationProjectionImageFilter`.
pub struct StdDevProjectionFilter<S> {
    pub source: S,
    pub axis: usize,
}

impl<S> StdDevProjectionFilter<S> {
    pub fn new(source: S, axis: usize) -> Self { Self { source, axis } }
}

impl<P, S, const D: usize> ImageSource<P, D> for StdDevProjectionFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> {
        let src = self.source.largest_region();
        let mut size = src.size.0;
        size[self.axis] = 1;
        Region { index: src.index, size: crate::image::Size(size) }
    }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let src_full = self.source.generate_region(self.source.largest_region());
        let src_region = src_full.region;
        let axis = self.axis;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let lo = src_region.index.0[axis];
                let len = src_region.size.0[axis] as i64;
                let vals: Vec<f64> = (0..len).map(|i| {
                    let mut sidx = idx;
                    sidx.0[axis] = lo + i;
                    src_full.get_pixel(sidx).to_f64()
                }).collect();
                let n = vals.len() as f64;
                let mean = vals.iter().sum::<f64>() / n;
                let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                P::from_f64(var.sqrt())
            })
            .collect();
        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

/// Median projection along one axis.
/// Analog to `itk::MedianProjectionImageFilter`.
pub struct MedianProjectionFilter<S> {
    pub source: S,
    pub axis: usize,
}

impl<S> MedianProjectionFilter<S> {
    pub fn new(source: S, axis: usize) -> Self { Self { source, axis } }
}

impl<P, S, const D: usize> ImageSource<P, D> for MedianProjectionFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> {
        let src = self.source.largest_region();
        let mut size = src.size.0;
        size[self.axis] = 1;
        Region { index: src.index, size: crate::image::Size(size) }
    }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let src_full = self.source.generate_region(self.source.largest_region());
        let src_region = src_full.region;
        let axis = self.axis;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let lo = src_region.index.0[axis];
                let len = src_region.size.0[axis] as i64;
                let mut vals: Vec<f64> = (0..len).map(|i| {
                    let mut sidx = idx;
                    sidx.0[axis] = lo + i;
                    src_full.get_pixel(sidx).to_f64()
                }).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let med = if vals.len() % 2 == 0 {
                    (vals[vals.len()/2 - 1] + vals[vals.len()/2]) / 2.0
                } else {
                    vals[vals.len()/2]
                };
                P::from_f64(med)
            })
            .collect();
        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

// ===========================================================================
// SimilarityIndexImageFilter
// ===========================================================================

/// Dice similarity coefficient between two binary images.
/// Analog to `itk::SimilarityIndexImageFilter`.
///
/// `DSC = 2 * |A ∩ B| / (|A| + |B|)` where membership is determined by
/// pixels above `threshold`.  Returns a value in `[0, 1]`.
pub struct SimilarityIndexFilter<S1, S2> {
    pub source1: S1,
    pub source2: S2,
    /// Pixel values > threshold are considered foreground.
    pub threshold: f64,
}

impl<S1, S2> SimilarityIndexFilter<S1, S2> {
    pub fn new(source1: S1, source2: S2, threshold: f64) -> Self {
        Self { source1, source2, threshold }
    }

    /// Compute and return the Dice similarity coefficient.
    pub fn compute<P, const D: usize>(&self) -> f64
    where
        P: NumericPixel,
        S1: ImageSource<P, D> + Sync,
        S2: ImageSource<P, D> + Sync,
    {
        let region = self.source1.largest_region();
        let img1 = self.source1.generate_region(region);
        let img2 = self.source2.generate_region(region);
        let t = self.threshold;
        let (mut a, mut b, mut inter) = (0u64, 0u64, 0u64);
        for i in 0..img1.data.len() {
            let v1 = img1.data[i].to_f64() > t;
            let v2 = img2.data[i].to_f64() > t;
            if v1 { a += 1; }
            if v2 { b += 1; }
            if v1 && v2 { inter += 1; }
        }
        let denom = a + b;
        if denom == 0 { 1.0 } else { 2.0 * inter as f64 / denom as f64 }
    }
}

// ===========================================================================
// AdaptiveHistogramEqualizationImageFilter
// ===========================================================================

/// Adaptive (local) histogram equalization.
/// Analog to `itk::AdaptiveHistogramEqualizationImageFilter`.
///
/// For each pixel, a local histogram within `[-radius, +radius]^D` is built
/// and used to remap the pixel.  `alpha` blends between identity (0) and full
/// equalization (1).  Output values are clamped to `[min_val, max_val]`.
pub struct AdaptiveHistogramEqualizationFilter<S, const D: usize> {
    pub source: S,
    pub radius: usize,
    /// Blend factor: 0 = identity, 1 = full equalization.
    pub alpha: f64,
    /// Number of histogram bins.
    pub num_bins: usize,
}

impl<S, const D: usize> AdaptiveHistogramEqualizationFilter<S, D> {
    pub fn new(source: S, radius: usize) -> Self {
        Self { source, radius, alpha: 0.3, num_bins: 256 }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for AdaptiveHistogramEqualizationFilter<S, D>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let radii = {
            let mut a = [0usize; D];
            a.iter_mut().for_each(|v| *v = self.radius);
            a
        };
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let bounds = self.source.largest_region();
        let radii = {
            let mut a = [0usize; D];
            a.iter_mut().for_each(|v| *v = self.radius);
            a
        };
        let input_region = requested.padded_per_axis(&radii).clipped_to(&bounds);
        let input = self.source.generate_region(input_region);

        // Global min/max for bin mapping
        let min_v = input.data.iter().map(|p| p.to_f64()).fold(f64::MAX, f64::min);
        let max_v = input.data.iter().map(|p| p.to_f64()).fold(f64::MIN, f64::max);
        let range = (max_v - min_v).max(1e-12);
        let nb = self.num_bins;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let alpha = self.alpha;

        let data: Vec<P> = out_indices.par_iter().map(|&out_idx| {
            // Build local histogram
            let mut hist = vec![0u64; nb];
            let mut total = 0u64;
            let mut nb_arr = [0i64; D];
            for d in 0..D { nb_arr[d] = -(self.radius as i64); }
            loop {
                let mut s = out_idx.0;
                for d in 0..D {
                    s[d] = (out_idx.0[d] + nb_arr[d])
                        .max(bounds.index.0[d])
                        .min(bounds.index.0[d] + bounds.size.0[d] as i64 - 1);
                }
                let v = input.get_pixel(Index(s)).to_f64();
                let bin = (((v - min_v) / range) * (nb as f64 - 1.0)).round() as usize;
                let bin = bin.min(nb - 1);
                hist[bin] += 1;
                total += 1;
                let mut carry = true;
                for d in 0..D {
                    if carry {
                        nb_arr[d] += 1;
                        if nb_arr[d] > self.radius as i64 {
                            nb_arr[d] = -(self.radius as i64);
                        } else {
                            carry = false;
                        }
                    }
                }
                if carry { break; }
            }
            // CDF
            let mut cdf = vec![0u64; nb];
            cdf[0] = hist[0];
            for i in 1..nb { cdf[i] = cdf[i-1] + hist[i]; }

            let v = out_idx.0;
            let pixel_val = input.get_pixel(Index(v)).to_f64();
            let bin = (((pixel_val - min_v) / range) * (nb as f64 - 1.0)).round() as usize;
            let bin = bin.min(nb - 1);
            let equalized = min_v + (cdf[bin] as f64 / total as f64) * range;
            let blended = (1.0 - alpha) * pixel_val + alpha * equalized;
            P::from_f64(blended.clamp(min_v, max_v))
        }).collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// LabelStatisticsImageFilter
// ===========================================================================

/// Compute per-label statistics (min, max, mean, std).
/// Analog to `itk::LabelStatisticsImageFilter`.
pub struct LabelStatisticsResult {
    pub label: u32,
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub sum: f64,
}

pub struct LabelStatisticsFilter<SI, SL> {
    pub intensity: SI,
    pub label_map: SL,
}

impl<SI, SL> LabelStatisticsFilter<SI, SL> {
    pub fn new(intensity: SI, label_map: SL) -> Self { Self { intensity, label_map } }
}

impl<SI, SL> LabelStatisticsFilter<SI, SL>
{
    /// Compute label statistics. Requires sources with compatible regions.
    pub fn compute<P, const D: usize>(&self) -> Vec<LabelStatisticsResult>
    where
        P: crate::pixel::NumericPixel,
        SI: crate::source::ImageSource<P, D>,
        SL: crate::source::ImageSource<u32, D>,
    {
        use crate::image::iter_region;
        use std::collections::HashMap;

        let intensity = self.intensity.generate_region(self.intensity.largest_region());
        let labels = self.label_map.generate_region(self.label_map.largest_region());

        let mut stats: HashMap<u32, (usize, f64, f64, f64)> = HashMap::new(); // (count, sum, sum2, min, max)
        let mut minmax: HashMap<u32, (f64, f64)> = HashMap::new();

        iter_region(&intensity.region, |idx| {
            let v = intensity.get_pixel(idx).to_f64();
            let lbl = labels.get_pixel(idx);
            let entry = stats.entry(lbl).or_insert((0, 0.0, 0.0, 0.0));
            entry.0 += 1;
            entry.1 += v;
            entry.2 += v * v;
            let mm = minmax.entry(lbl).or_insert((f64::MAX, f64::MIN));
            if v < mm.0 { mm.0 = v; }
            if v > mm.1 { mm.1 = v; }
        });

        let mut results: Vec<LabelStatisticsResult> = stats.iter().map(|(&label, &(count, sum, sum2, _))| {
            let mean = sum / count as f64;
            let var = (sum2 / count as f64 - mean * mean).max(0.0);
            let (min, max) = minmax[&label];
            LabelStatisticsResult { label, count, min, max, mean, std: var.sqrt(), sum }
        }).collect();
        results.sort_by_key(|r| r.label);
        results
    }
}

// ===========================================================================
// ImageMomentsCalculator
// ===========================================================================

/// Compute spatial moments of an image (M00, M10, M01, centroid).
/// Analog to `itk::ImageMomentsCalculator`.
pub struct ImageMomentsCalculator<S> {
    pub source: S,
}

impl<S> ImageMomentsCalculator<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

pub struct ImageMoments2D {
    pub m00: f64,
    pub centroid: [f64; 2],
    pub central_moments: [[f64; 2]; 2],
    pub principal_moments: [f64; 2],
}

impl<S> ImageMomentsCalculator<S>
{
    pub fn compute<P>(&self) -> ImageMoments2D
    where
        P: crate::pixel::NumericPixel,
        S: crate::source::ImageSource<P, 2>,
    {
        use crate::image::iter_region;
        let img = self.source.generate_region(self.source.largest_region());
        let sp = img.spacing;

        let mut m00 = 0.0f64;
        let mut m10 = 0.0f64;
        let mut m01 = 0.0f64;

        iter_region(&img.region, |idx| {
            let v = img.get_pixel(idx).to_f64();
            let x = idx.0[0] as f64 * sp[0];
            let y = idx.0[1] as f64 * sp[1];
            m00 += v;
            m10 += v * x;
            m01 += v * y;
        });

        let cx = if m00 != 0.0 { m10 / m00 } else { 0.0 };
        let cy = if m00 != 0.0 { m01 / m00 } else { 0.0 };

        // Central moments
        let mut mu20 = 0.0f64;
        let mut mu02 = 0.0f64;
        let mut mu11 = 0.0f64;
        iter_region(&img.region, |idx| {
            let v = img.get_pixel(idx).to_f64();
            let x = idx.0[0] as f64 * sp[0] - cx;
            let y = idx.0[1] as f64 * sp[1] - cy;
            mu20 += v * x * x;
            mu02 += v * y * y;
            mu11 += v * x * y;
        });
        if m00 != 0.0 {
            mu20 /= m00; mu02 /= m00; mu11 /= m00;
        }

        // Principal moments (eigenvalues of covariance)
        let tr = mu20 + mu02;
        let det = mu20 * mu02 - mu11 * mu11;
        let disc = ((tr * tr / 4.0) - det).max(0.0).sqrt();
        let l1 = tr / 2.0 - disc;
        let l2 = tr / 2.0 + disc;

        ImageMoments2D {
            m00,
            centroid: [cx, cy],
            central_moments: [[mu20, mu11], [mu11, mu02]],
            principal_moments: [l1, l2],
        }
    }
}

// ===========================================================================
// LabelOverlapMeasuresImageFilter
// ===========================================================================

/// Per-label Dice/Jaccard/Sensitivity/Specificity overlap measures.
/// Analog to `itk::LabelOverlapMeasuresImageFilter`.
pub struct LabelOverlapMeasures {
    pub label: u32,
    pub dice: f64,
    pub jaccard: f64,
    pub sensitivity: f64,
    pub specificity: f64,
}

pub struct LabelOverlapMeasuresFilter<SA, SB> {
    pub source_a: SA,
    pub source_b: SB,
}

impl<SA, SB> LabelOverlapMeasuresFilter<SA, SB> {
    pub fn new(source_a: SA, source_b: SB) -> Self { Self { source_a, source_b } }
}

impl<SA, SB> LabelOverlapMeasuresFilter<SA, SB>
{
    pub fn compute<const D: usize>(&self) -> Vec<LabelOverlapMeasures>
    where
        SA: crate::source::ImageSource<u32, D>,
        SB: crate::source::ImageSource<u32, D>,
    {
        use crate::image::iter_region;
        use std::collections::{HashMap, HashSet};

        let a = self.source_a.generate_region(self.source_a.largest_region());
        let b = self.source_b.generate_region(self.source_b.largest_region());

        // Collect all labels
        let labels: HashSet<u32> = a.data.iter().chain(b.data.iter())
            .filter(|&&l| l != 0)
            .cloned().collect();

        let mut results = Vec::new();
        for &label in &labels {
            let mut tp = 0usize; let mut fp = 0usize;
            let mut fn_ = 0usize; let mut tn = 0usize;
            iter_region(&a.region, |idx| {
                let av = a.get_pixel(idx) == label;
                let bv = b.get_pixel(idx) == label;
                match (av, bv) {
                    (true, true) => tp += 1,
                    (false, true) => fp += 1,
                    (true, false) => fn_ += 1,
                    (false, false) => tn += 1,
                }
            });
            let dice = if tp + fp + fn_ > 0 { 2.0 * tp as f64 / (2 * tp + fp + fn_) as f64 } else { 1.0 };
            let jaccard = if tp + fp + fn_ > 0 { tp as f64 / (tp + fp + fn_) as f64 } else { 1.0 };
            let sensitivity = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 1.0 };
            let specificity = if tn + fp > 0 { tn as f64 / (tn + fp) as f64 } else { 1.0 };
            results.push(LabelOverlapMeasures { label, dice, jaccard, sensitivity, specificity });
        }
        results.sort_by_key(|r| r.label);
        results
    }
}

// ===========================================================================
// STAPLEImageFilter
// ===========================================================================

/// STAPLE (Simultaneous Truth and Performance Level Estimation) consensus.
/// Analog to `itk::STAPLEImageFilter`.
/// Produces a probability image from multiple binary label images.
pub struct STAPLEFilter {
    pub sources: Vec<crate::image::Image<u32, 2>>,
    pub iterations: usize,
}

impl STAPLEFilter {
    pub fn new(sources: Vec<crate::image::Image<u32, 2>>) -> Self {
        Self { sources, iterations: 20 }
    }

    pub fn compute(&self) -> crate::image::Image<f32, 2> {
        if self.sources.is_empty() {
            return crate::image::Image { region: crate::image::Region::new([0,0],[0,0]), spacing: [1.0;2], origin: [0.0;2], data: vec![] };
        }
        let region = self.sources[0].region;
        let n_pix = region.linear_len();
        let n_raters = self.sources.len();

        // Initialize W (probability of true positive) to fraction of raters agreeing
        let mut w: Vec<f64> = (0..n_pix).map(|i| {
            let votes: usize = self.sources.iter().map(|s| if s.data[i] != 0 { 1 } else { 0 }).sum();
            votes as f64 / n_raters as f64
        }).collect();

        // EM iterations
        let mut p = vec![0.99f64; n_raters]; // sensitivity
        let mut q = vec![0.99f64; n_raters]; // specificity

        for _ in 0..self.iterations {
            // E-step: compute posterior W given current p, q
            let mut new_w: Vec<f64> = vec![0.0; n_pix];
            for i in 0..n_pix {
                let mut num = w[i];
                let mut den = w[i];
                for r in 0..n_raters {
                    let d = (self.sources[r].data[i] != 0) as usize;
                    num *= if d == 1 { p[r] } else { 1.0 - p[r] };
                    den *= if d == 1 { p[r] } else { 1.0 - p[r] };
                }
                let mut neg = 1.0 - w[i];
                for r in 0..n_raters {
                    let d = (self.sources[r].data[i] != 0) as usize;
                    neg *= if d == 0 { q[r] } else { 1.0 - q[r] };
                }
                new_w[i] = num / (num + neg).max(1e-12);
            }
            w = new_w;

            // M-step: update p and q
            for r in 0..n_raters {
                let mut sum_w = 0.0f64;
                let mut sum_wd = 0.0f64;
                let mut sum_1mw = 0.0f64;
                let mut sum_1mwd = 0.0f64;
                for i in 0..n_pix {
                    let d = if self.sources[r].data[i] != 0 { 1.0f64 } else { 0.0f64 };
                    sum_w += w[i];
                    sum_wd += w[i] * d;
                    sum_1mw += 1.0 - w[i];
                    sum_1mwd += (1.0 - w[i]) * d;
                }
                p[r] = (sum_wd / sum_w.max(1e-12)).clamp(1e-6, 1.0 - 1e-6);
                q[r] = (1.0 - sum_1mwd / sum_1mw.max(1e-12)).clamp(1e-6, 1.0 - 1e-6);
            }
        }

        let data: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        crate::image::Image { region, spacing: self.sources[0].spacing, origin: self.sources[0].origin, data }
    }
}

// ===========================================================================
// MultiLabelSTAPLEImageFilter
// ===========================================================================

/// Multi-label STAPLE: majority vote fusion over multiple label images.
/// Analog to `itk::MultiLabelSTAPLEImageFilter`.
pub struct MultiLabelSTAPLEFilter {
    pub sources: Vec<crate::image::Image<u32, 2>>,
}

impl MultiLabelSTAPLEFilter {
    pub fn new(sources: Vec<crate::image::Image<u32, 2>>) -> Self { Self { sources } }

    pub fn compute(&self) -> crate::image::Image<u32, 2> {
        if self.sources.is_empty() {
            return crate::image::Image { region: crate::image::Region::new([0,0],[0,0]), spacing: [1.0;2], origin: [0.0;2], data: vec![] };
        }
        let region = self.sources[0].region;
        let n_pix = region.linear_len();
        use std::collections::HashMap;

        let data: Vec<u32> = (0..n_pix).map(|i| {
            let mut votes: HashMap<u32, usize> = HashMap::new();
            for s in &self.sources {
                *votes.entry(s.data[i]).or_insert(0) += 1;
            }
            votes.into_iter().max_by_key(|&(_, c)| c).map(|(l, _)| l).unwrap_or(0)
        }).collect();

        crate::image::Image { region, spacing: self.sources[0].spacing, origin: self.sources[0].origin, data }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    fn make_3d_ramp() -> Image<f32, 3> {
        let mut img = Image::<f32,3>::allocate(Region::new([0,0,0],[4,4,4]),[1.0,1.0,1.0],[0.0,0.0,0.0],0.0);
        for z in 0..4i64 {
            for y in 0..4i64 {
                for x in 0..4i64 {
                    img.set_pixel(Index([x,y,z]), (x + y * 4 + z * 16) as f32);
                }
            }
        }
        img
    }

    #[test]
    fn statistics_ramp() {
        let img = make_3d_ramp();
        let stats = compute_statistics(&img);
        assert_eq!(stats.count, 64);
        assert!((stats.minimum - 0.0).abs() < 1e-6);
        assert!((stats.maximum - 63.0).abs() < 1e-6);
        assert!((stats.mean - 31.5).abs() < 1e-4);
    }

    #[test]
    fn max_projection_3d() {
        let img = make_3d_ramp();
        let f = MaxProjectionFilter::new(img, 2); // project along z
        let out = f.generate_region(f.largest_region());
        assert_eq!(out.region.size.0, [4, 4, 1]);
        // Max over z at (x,y,0) = x + y*4 + 3*16
        let v = out.get_pixel(Index([0i64,0,0]));
        assert!((v - 48.0).abs() < 1e-6, "expected 48 got {v}");
    }

    #[test]
    fn min_projection_3d() {
        let img = make_3d_ramp();
        let f = MinProjectionFilter::new(img, 2);
        let out = f.generate_region(f.largest_region());
        // Min over z at (x=0,y=0) = 0
        let v = out.get_pixel(Index([0i64,0,0]));
        assert!(v.abs() < 1e-6);
    }

    #[test]
    fn accumulate_axis() {
        let img = Image::<f32,2>::allocate(Region::new([0,0],[3,3]),[1.0,1.0],[0.0,0.0],2.0f32);
        let f = AccumulateFilter::new(img, 1); // sum along y
        let out = f.generate_region(f.largest_region());
        assert_eq!(out.region.size.0, [3, 1]);
        // Sum of 3 pixels each = 2.0 → 6.0
        let v = out.get_pixel(Index([1i64,0]));
        assert!((v - 6.0).abs() < 1e-6, "expected 6 got {v}");
    }
}
