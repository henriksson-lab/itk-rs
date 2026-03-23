//! Image intensity filters.
//!
//! Covers ITK's pointwise math, intensity mapping, two-image arithmetic,
//! and mask filters.
//!
//! # Organisation
//!
//! | Category | Types |
//! |---|---|
//! | Pointwise math | [`PointwiseMathFilter`], [`PowFilter`], [`ModulusFilter`] |
//! | Intensity mapping | [`SigmoidFilter`], [`ShiftScaleFilter`], [`RescaleIntensityFilter`], [`ClampFilter`], [`NormalizeFilter`], [`IntensityWindowingFilter`], [`InvertIntensityFilter`], [`BoundedReciprocalFilter`] |
//! | Two-image arithmetic | [`BinaryFilter`] + helpers (`add_images`, …) |
//! | Mask | [`MaskFilter`], [`MaskNegatedFilter`] |
//!
//! Simple math constructors: [`abs_filter`], [`square_filter`], [`sqrt_filter`],
//! [`exp_filter`], [`log_filter`], [`log10_filter`], [`sin_filter`], [`cos_filter`],
//! [`atan_filter`], [`round_filter`].

use std::marker::PhantomData;

use rayon::prelude::*;

use crate::image::{Image, Region, iter_region, Index};
use crate::pixel::{Pixel, NumericPixel};
use crate::source::ImageSource;

// ===========================================================================
// Pointwise math filter (single image, f64 round-trip)
// ===========================================================================

/// Applies a `f64 → f64` function to every pixel via `to_f64` / `from_f64`.
/// Analog to `itk::UnaryFunctorImageFilter` specialised for math operations.
///
/// Use the free-function constructors ([`abs_filter`], [`sqrt_filter`], …)
/// rather than constructing this type directly.
pub struct PointwiseMathFilter<S, P> {
    pub source: S,
    func: fn(f64) -> f64,
    _phantom: PhantomData<P>,
}

impl<S, P> PointwiseMathFilter<S, P> {
    pub fn new(source: S, func: fn(f64) -> f64) -> Self {
        Self { source, func, _phantom: PhantomData }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for PointwiseMathFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let f = self.func;
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| P::from_f64(f(p.to_f64())))
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// --- Constructor functions ---

/// `|x| → |x|`. Analog to `itk::AbsImageFilter`.
pub fn abs_filter<P: NumericPixel, S>(source: S) -> PointwiseMathFilter<S, P> {
    PointwiseMathFilter::new(source, f64::abs)
}
/// `x²`. Analog to `itk::SquareImageFilter`.
pub fn square_filter<P: NumericPixel, S>(source: S) -> PointwiseMathFilter<S, P> {
    PointwiseMathFilter::new(source, |x| x * x)
}
/// `√x`. Analog to `itk::SqrtImageFilter`.
pub fn sqrt_filter<P: NumericPixel, S>(source: S) -> PointwiseMathFilter<S, P> {
    PointwiseMathFilter::new(source, f64::sqrt)
}
/// `eˣ`. Analog to `itk::ExpImageFilter`.
pub fn exp_filter<P: NumericPixel, S>(source: S) -> PointwiseMathFilter<S, P> {
    PointwiseMathFilter::new(source, f64::exp)
}
/// `ln(x)`. Analog to `itk::LogImageFilter`.
pub fn log_filter<P: NumericPixel, S>(source: S) -> PointwiseMathFilter<S, P> {
    PointwiseMathFilter::new(source, f64::ln)
}
/// `log₁₀(x)`. Analog to `itk::Log10ImageFilter`.
pub fn log10_filter<P: NumericPixel, S>(source: S) -> PointwiseMathFilter<S, P> {
    PointwiseMathFilter::new(source, f64::log10)
}
/// `sin(x)`. Analog to `itk::SinImageFilter`.
pub fn sin_filter<P: NumericPixel, S>(source: S) -> PointwiseMathFilter<S, P> {
    PointwiseMathFilter::new(source, f64::sin)
}
/// `cos(x)`. Analog to `itk::CosImageFilter`.
pub fn cos_filter<P: NumericPixel, S>(source: S) -> PointwiseMathFilter<S, P> {
    PointwiseMathFilter::new(source, f64::cos)
}
/// `arctan(x)`. Analog to `itk::AtanImageFilter`.
pub fn atan_filter<P: NumericPixel, S>(source: S) -> PointwiseMathFilter<S, P> {
    PointwiseMathFilter::new(source, f64::atan)
}
/// Round to nearest integer. Analog to `itk::RoundImageFilter`.
pub fn round_filter<P: NumericPixel, S>(source: S) -> PointwiseMathFilter<S, P> {
    PointwiseMathFilter::new(source, f64::round)
}
/// `1 / (1 + x)`. Analog to `itk::BoundedReciprocalImageFilter`.
pub fn bounded_reciprocal_filter<P: NumericPixel, S>(source: S) -> PointwiseMathFilter<S, P> {
    PointwiseMathFilter::new(source, |x| 1.0 / (1.0 + x))
}

// ===========================================================================
// Power filter
// ===========================================================================

/// `xᵉ` with a runtime exponent. Analog to `itk::PowImageFilter`.
pub struct PowFilter<S> {
    pub source: S,
    pub exponent: f64,
}

impl<S> PowFilter<S> {
    pub fn new(source: S, exponent: f64) -> Self { Self { source, exponent } }
}

impl<P, S, const D: usize> ImageSource<P, D> for PowFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let e = self.exponent;
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| P::from_f64(p.to_f64().powf(e)))
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Modulus filter
// ===========================================================================

/// `x mod divisor`. Analog to `itk::ModulusImageFilter`.
pub struct ModulusFilter<S> {
    pub source: S,
    pub divisor: f64,
}

impl<S> ModulusFilter<S> {
    pub fn new(source: S, divisor: f64) -> Self { Self { source, divisor } }
}

impl<P, S, const D: usize> ImageSource<P, D> for ModulusFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let d = self.divisor;
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| P::from_f64(p.to_f64() % d))
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Sigmoid filter
// ===========================================================================

/// `1 / (1 + exp(−(x − beta) / alpha))` scaled to `[output_min, output_max]`.
/// Analog to `itk::SigmoidImageFilter`.
pub struct SigmoidFilter<S> {
    pub source: S,
    /// Controls steepness (larger = more gradual).
    pub alpha: f64,
    /// Inflection point.
    pub beta: f64,
    pub output_min: f64,
    pub output_max: f64,
}

impl<S> SigmoidFilter<S> {
    pub fn new(source: S, alpha: f64, beta: f64) -> Self {
        Self { source, alpha, beta, output_min: 0.0, output_max: 1.0 }
    }
    pub fn with_output_range(mut self, min: f64, max: f64) -> Self {
        self.output_min = min;
        self.output_max = max;
        self
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for SigmoidFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let (alpha, beta, omin, omax) = (self.alpha, self.beta, self.output_min, self.output_max);
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| {
                let s = 1.0 / (1.0 + (-(p.to_f64() - beta) / alpha).exp());
                P::from_f64(omin + s * (omax - omin))
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Shift-scale filter
// ===========================================================================

/// `(x + shift) * scale`. Analog to `itk::ShiftScaleImageFilter`.
pub struct ShiftScaleFilter<S> {
    pub source: S,
    pub shift: f64,
    pub scale: f64,
}

impl<S> ShiftScaleFilter<S> {
    pub fn new(source: S, shift: f64, scale: f64) -> Self {
        Self { source, shift, scale }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for ShiftScaleFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let (sh, sc) = (self.shift, self.scale);
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| P::from_f64((p.to_f64() + sh) * sc))
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Rescale intensity filter
// ===========================================================================

/// Linear map from `[input_min, input_max]` → `[output_min, output_max]`.
/// If `input_min == input_max` (constant image), output = `output_min`.
/// Analog to `itk::RescaleIntensityImageFilter`.
///
/// `input_min` / `input_max` are auto-detected from the **full** source image
/// at `generate_region` time (requires pulling the full image once for stats).
pub struct RescaleIntensityFilter<S> {
    pub source: S,
    pub output_min: f64,
    pub output_max: f64,
}

impl<S> RescaleIntensityFilter<S> {
    pub fn new(source: S, output_min: f64, output_max: f64) -> Self {
        Self { source, output_min, output_max }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for RescaleIntensityFilter<S>
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
        let full = self.source.generate_region(self.source.largest_region());
        let (imin, imax) = full.data.iter().fold((f64::MAX, f64::MIN), |(mn, mx), &p| {
            let v = p.to_f64();
            (mn.min(v), mx.max(v))
        });
        let range = imax - imin;
        let (omin, omax) = (self.output_min, self.output_max);

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let v = full.get_pixel(idx).to_f64();
                let scaled = if range.abs() < 1e-15 {
                    omin
                } else {
                    omin + (v - imin) / range * (omax - omin)
                };
                P::from_f64(scaled)
            })
            .collect();
        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

// ===========================================================================
// Clamp filter
// ===========================================================================

/// Clamp values to `[lower, upper]`. Analog to `itk::ClampImageFilter`.
pub struct ClampFilter<S> {
    pub source: S,
    pub lower: f64,
    pub upper: f64,
}

impl<S> ClampFilter<S> {
    pub fn new(source: S, lower: f64, upper: f64) -> Self {
        Self { source, lower, upper }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for ClampFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let (lo, hi) = (self.lower, self.upper);
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| P::from_f64(p.to_f64().clamp(lo, hi)))
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Normalize filter
// ===========================================================================

/// Subtract mean, divide by standard deviation: `(x − μ) / σ`.
/// Analog to `itk::NormalizeImageFilter`.
///
/// Requires pulling the **full** source image to compute statistics.
pub struct NormalizeFilter<S> {
    pub source: S,
}

impl<S> NormalizeFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<P, S, const D: usize> ImageSource<P, D> for NormalizeFilter<S>
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
        let full = self.source.generate_region(self.source.largest_region());
        let n = full.data.len() as f64;
        let mean = full.data.iter().map(|p| p.to_f64()).sum::<f64>() / n;
        let variance = full.data.iter()
            .map(|p| (p.to_f64() - mean).powi(2))
            .sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let v = full.get_pixel(idx).to_f64();
                P::from_f64(if std_dev < 1e-15 { 0.0 } else { (v - mean) / std_dev })
            })
            .collect();
        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

// ===========================================================================
// Intensity windowing filter
// ===========================================================================

/// Clamp input to `[window_min, window_max]`, then linearly rescale to
/// `[output_min, output_max]`.
/// Analog to `itk::IntensityWindowingImageFilter`.
pub struct IntensityWindowingFilter<S> {
    pub source: S,
    pub window_min: f64,
    pub window_max: f64,
    pub output_min: f64,
    pub output_max: f64,
}

impl<S> IntensityWindowingFilter<S> {
    pub fn new(source: S, window_min: f64, window_max: f64, output_min: f64, output_max: f64) -> Self {
        Self { source, window_min, window_max, output_min, output_max }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for IntensityWindowingFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let (wmin, wmax, omin, omax) = (self.window_min, self.window_max, self.output_min, self.output_max);
        let wrange = wmax - wmin;
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| {
                let v = p.to_f64().clamp(wmin, wmax);
                let t = if wrange.abs() < 1e-15 { 0.0 } else { (v - wmin) / wrange };
                P::from_f64(omin + t * (omax - omin))
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Invert intensity filter
// ===========================================================================

/// `maximum − x`. Analog to `itk::InvertIntensityImageFilter`.
///
/// `maximum` defaults to 255.0 (suitable for 8-bit images).
pub struct InvertIntensityFilter<S> {
    pub source: S,
    pub maximum: f64,
}

impl<S> InvertIntensityFilter<S> {
    pub fn new(source: S, maximum: f64) -> Self { Self { source, maximum } }
}

impl<P, S, const D: usize> ImageSource<P, D> for InvertIntensityFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let m = self.maximum;
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| P::from_f64(m - p.to_f64()))
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Binary (two-image) filter
// ===========================================================================

/// Element-wise combination of two images with an arbitrary functor.
/// Analog to `itk::BinaryFunctorImageFilter`.
///
/// Metadata (spacing, origin, region) comes from `source1`.
/// Both sources must support the requested region.
pub struct BinaryFilter<S1, S2, F, P, Q, R> {
    pub source1: S1,
    pub source2: S2,
    pub func: F,
    _phantom: PhantomData<fn(P, Q) -> R>,
}

impl<S1, S2, F, P, Q, R> BinaryFilter<S1, S2, F, P, Q, R> {
    pub fn new(source1: S1, source2: S2, func: F) -> Self {
        Self { source1, source2, func, _phantom: PhantomData }
    }
}

impl<S1, S2, F, P, Q, R, const D: usize> ImageSource<R, D> for BinaryFilter<S1, S2, F, P, Q, R>
where
    S1: ImageSource<P, D> + Sync,
    S2: ImageSource<Q, D> + Sync,
    F: Fn(P, Q) -> R + Sync + Send,
    P: Pixel,
    Q: Pixel,
    R: Pixel,
{
    fn largest_region(&self) -> Region<D> { self.source1.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source1.spacing() }
    fn origin(&self) -> [f64; D] { self.source1.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<R, D> {
        let a = self.source1.generate_region(requested);
        let b = self.source2.generate_region(requested);
        let data: Vec<R> = a.data.par_iter().zip(b.data.par_iter())
            .map(|(&pa, &pb)| (self.func)(pa, pb))
            .collect();
        Image { region: requested, spacing: a.spacing, origin: a.origin, data }
    }
}

// --- Two-image arithmetic constructors ---

/// Pixel-wise addition. Analog to `itk::AddImageFilter`.
pub fn add_images<P, S1, S2>(s1: S1, s2: S2)
    -> BinaryFilter<S1, S2, fn(P, P) -> P, P, P, P>
where P: NumericPixel
{
    BinaryFilter::new(s1, s2, |a, b| P::from_f64(a.to_f64() + b.to_f64()))
}

/// Pixel-wise subtraction. Analog to `itk::SubtractImageFilter`.
pub fn subtract_images<P, S1, S2>(s1: S1, s2: S2)
    -> BinaryFilter<S1, S2, fn(P, P) -> P, P, P, P>
where P: NumericPixel
{
    BinaryFilter::new(s1, s2, |a, b| P::from_f64(a.to_f64() - b.to_f64()))
}

/// Pixel-wise multiplication. Analog to `itk::MultiplyImageFilter`.
pub fn multiply_images<P, S1, S2>(s1: S1, s2: S2)
    -> BinaryFilter<S1, S2, fn(P, P) -> P, P, P, P>
where P: NumericPixel
{
    BinaryFilter::new(s1, s2, |a, b| P::from_f64(a.to_f64() * b.to_f64()))
}

/// Pixel-wise division. Analog to `itk::DivideImageFilter`.
/// Division by zero produces 0.
pub fn divide_images<P, S1, S2>(s1: S1, s2: S2)
    -> BinaryFilter<S1, S2, fn(P, P) -> P, P, P, P>
where P: NumericPixel
{
    BinaryFilter::new(s1, s2, |a, b| {
        let bv = b.to_f64();
        P::from_f64(if bv.abs() < 1e-300 { 0.0 } else { a.to_f64() / bv })
    })
}

/// Pixel-wise maximum. Analog to `itk::MaximumImageFilter`.
pub fn maximum_images<P, S1, S2>(s1: S1, s2: S2)
    -> BinaryFilter<S1, S2, fn(P, P) -> P, P, P, P>
where P: NumericPixel
{
    BinaryFilter::new(s1, s2, |a, b| {
        if a.to_f64() >= b.to_f64() { a } else { b }
    })
}

/// Pixel-wise minimum. Analog to `itk::MinimumImageFilter`.
pub fn minimum_images<P, S1, S2>(s1: S1, s2: S2)
    -> BinaryFilter<S1, S2, fn(P, P) -> P, P, P, P>
where P: NumericPixel
{
    BinaryFilter::new(s1, s2, |a, b| {
        if a.to_f64() <= b.to_f64() { a } else { b }
    })
}

// ===========================================================================
// Weighted add filter
// ===========================================================================

/// `alpha * A + (1 − alpha) * B`. Analog to `itk::WeightedAddImageFilter`.
pub struct WeightedAddFilter<S1, S2> {
    pub source1: S1,
    pub source2: S2,
    /// Weight for `source1`; `source2` weight = `1 − alpha`.
    pub alpha: f64,
}

impl<S1, S2> WeightedAddFilter<S1, S2> {
    pub fn new(source1: S1, source2: S2, alpha: f64) -> Self {
        Self { source1, source2, alpha }
    }
}

impl<P, S1, S2, const D: usize> ImageSource<P, D> for WeightedAddFilter<S1, S2>
where
    P: NumericPixel,
    S1: ImageSource<P, D> + Sync,
    S2: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source1.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source1.spacing() }
    fn origin(&self) -> [f64; D] { self.source1.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let a = self.source1.generate_region(requested);
        let b = self.source2.generate_region(requested);
        let alpha = self.alpha;
        let data: Vec<P> = a.data.par_iter().zip(b.data.par_iter())
            .map(|(&pa, &pb)| {
                P::from_f64(alpha * pa.to_f64() + (1.0 - alpha) * pb.to_f64())
            })
            .collect();
        Image { region: requested, spacing: a.spacing, origin: a.origin, data }
    }
}

// ===========================================================================
// Mask filters
// ===========================================================================

/// Zero out pixels where the mask equals its default (zero) value.
/// Analog to `itk::MaskImageFilter`.
///
/// `outside_value` is the replacement for masked pixels.
pub struct MaskFilter<SI, SM, P, M> {
    pub image_source: SI,
    pub mask_source: SM,
    pub outside_value: P,
    _phantom: PhantomData<M>,
}

impl<SI, SM, P, M> MaskFilter<SI, SM, P, M> {
    pub fn new(image_source: SI, mask_source: SM, outside_value: P) -> Self {
        Self { image_source, mask_source, outside_value, _phantom: PhantomData }
    }
}

impl<P, M, SI, SM, const D: usize> ImageSource<P, D> for MaskFilter<SI, SM, P, M>
where
    P: Pixel,
    M: Pixel + Default + PartialEq,
    SI: ImageSource<P, D> + Sync,
    SM: ImageSource<M, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.image_source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.image_source.spacing() }
    fn origin(&self) -> [f64; D] { self.image_source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let img  = self.image_source.generate_region(requested);
        let mask = self.mask_source.generate_region(requested);
        let outside = self.outside_value;
        let zero = M::default();
        let data: Vec<P> = img.data.par_iter().zip(mask.data.par_iter())
            .map(|(&p, &m)| if m == zero { outside } else { p })
            .collect();
        Image { region: requested, spacing: img.spacing, origin: img.origin, data }
    }
}

/// Zero out pixels where the mask is **non-zero**.
/// Analog to `itk::MaskNegatedImageFilter`.
pub struct MaskNegatedFilter<SI, SM, P, M> {
    pub image_source: SI,
    pub mask_source: SM,
    pub outside_value: P,
    _phantom: PhantomData<M>,
}

impl<SI, SM, P, M> MaskNegatedFilter<SI, SM, P, M> {
    pub fn new(image_source: SI, mask_source: SM, outside_value: P) -> Self {
        Self { image_source, mask_source, outside_value, _phantom: PhantomData }
    }
}

impl<P, M, SI, SM, const D: usize> ImageSource<P, D> for MaskNegatedFilter<SI, SM, P, M>
where
    P: Pixel,
    M: Pixel + Default + PartialEq,
    SI: ImageSource<P, D> + Sync,
    SM: ImageSource<M, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.image_source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.image_source.spacing() }
    fn origin(&self) -> [f64; D] { self.image_source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let img  = self.image_source.generate_region(requested);
        let mask = self.mask_source.generate_region(requested);
        let outside = self.outside_value;
        let zero = M::default();
        let data: Vec<P> = img.data.par_iter().zip(mask.data.par_iter())
            .map(|(&p, &m)| if m != zero { outside } else { p })
            .collect();
        Image { region: requested, spacing: img.spacing, origin: img.origin, data }
    }
}

// ===========================================================================
// Atan2 filter (binary, two images)
// ===========================================================================

/// Pixel-wise `atan2(y, x)`. Analog to `itk::Atan2ImageFilter`.
pub fn atan2_images<P, S1, S2>(y: S1, x: S2)
    -> BinaryFilter<S1, S2, fn(P, P) -> P, P, P, P>
where P: NumericPixel
{
    BinaryFilter::new(y, x, |py, px| P::from_f64(py.to_f64().atan2(px.to_f64())))
}

// ===========================================================================
// Normalize to constant filter
// ===========================================================================

/// Scale image so the sum of all pixels equals `constant`.
/// Analog to `itk::NormalizeToConstantImageFilter`.
pub struct NormalizeToConstantFilter<S> {
    pub source: S,
    pub constant: f64,
}

impl<S> NormalizeToConstantFilter<S> {
    pub fn new(source: S, constant: f64) -> Self { Self { source, constant } }
}

impl<P, S, const D: usize> ImageSource<P, D> for NormalizeToConstantFilter<S>
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
        let full = self.source.generate_region(self.source.largest_region());
        let sum: f64 = full.data.iter().map(|p| p.to_f64()).sum();
        let scale = if sum.abs() < 1e-300 { 1.0 } else { self.constant / sum };

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| P::from_f64(full.get_pixel(idx).to_f64() * scale))
            .collect();
        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

// ===========================================================================
// Bitwise filters (And, Or, Xor, Not)
// ===========================================================================

/// Pixel-wise bitwise AND (via f64 → u64 round-trip).
/// Analog to `itk::AndImageFilter`.
pub fn and_images<P, S1, S2>(s1: S1, s2: S2)
    -> BinaryFilter<S1, S2, fn(P, P) -> P, P, P, P>
where P: NumericPixel
{
    BinaryFilter::new(s1, s2, |a, b| {
        let av = a.to_f64() as u64;
        let bv = b.to_f64() as u64;
        P::from_f64((av & bv) as f64)
    })
}

/// Pixel-wise bitwise OR. Analog to `itk::OrImageFilter`.
pub fn or_images<P, S1, S2>(s1: S1, s2: S2)
    -> BinaryFilter<S1, S2, fn(P, P) -> P, P, P, P>
where P: NumericPixel
{
    BinaryFilter::new(s1, s2, |a, b| {
        let av = a.to_f64() as u64;
        let bv = b.to_f64() as u64;
        P::from_f64((av | bv) as f64)
    })
}

/// Pixel-wise bitwise XOR. Analog to `itk::XorImageFilter`.
pub fn xor_images<P, S1, S2>(s1: S1, s2: S2)
    -> BinaryFilter<S1, S2, fn(P, P) -> P, P, P, P>
where P: NumericPixel
{
    BinaryFilter::new(s1, s2, |a, b| {
        let av = a.to_f64() as u64;
        let bv = b.to_f64() as u64;
        P::from_f64((av ^ bv) as f64)
    })
}

/// Pixel-wise bitwise NOT (complement w.r.t. `maximum`).
/// Analog to `itk::NotImageFilter`.
pub struct NotFilter<S> {
    pub source: S,
    /// The maximum value (all bits set). For u8: 255, for u16: 65535, etc.
    pub maximum: u64,
}

impl<S> NotFilter<S> {
    pub fn new(source: S, maximum: u64) -> Self { Self { source, maximum } }
}

impl<P, S, const D: usize> ImageSource<P, D> for NotFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let max = self.maximum;
        let data: Vec<P> = input.data.par_iter()
            .map(|&p| P::from_f64((max & !(p.to_f64() as u64)) as f64))
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// N-ary filters (multiple images, same type)
// ===========================================================================

/// Add any number of images together. Analog to `itk::NaryAddImageFilter`.
pub struct NaryAddFilter<P, const D: usize> {
    pub sources: Vec<Box<dyn crate::source::ImageSource<P, D> + Send + Sync>>,
}

impl<P: NumericPixel, const D: usize> NaryAddFilter<P, D> {
    pub fn new(sources: Vec<Box<dyn crate::source::ImageSource<P, D> + Send + Sync>>) -> Self {
        Self { sources }
    }
}

impl<P: NumericPixel, const D: usize> ImageSource<P, D> for NaryAddFilter<P, D> {
    fn largest_region(&self) -> Region<D> {
        self.sources.first().expect("at least one source").largest_region()
    }
    fn spacing(&self) -> [f64; D] { self.sources[0].spacing() }
    fn origin(&self) -> [f64; D] { self.sources[0].origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let images: Vec<_> = self.sources.iter()
            .map(|s| s.generate_region(requested))
            .collect();
        let len = images[0].data.len();
        let mut data = vec![P::zero(); len];
        for img in &images {
            for (i, &p) in img.data.iter().enumerate() {
                data[i] = data[i] + p;
            }
        }
        Image { region: requested, spacing: images[0].spacing, origin: images[0].origin, data }
    }
}

/// Pixelwise maximum over any number of images. Analog to `itk::NaryMaximumImageFilter`.
pub struct NaryMaximumFilter<P, const D: usize> {
    pub sources: Vec<Box<dyn crate::source::ImageSource<P, D> + Send + Sync>>,
}

impl<P: NumericPixel, const D: usize> NaryMaximumFilter<P, D> {
    pub fn new(sources: Vec<Box<dyn crate::source::ImageSource<P, D> + Send + Sync>>) -> Self {
        Self { sources }
    }
}

impl<P: NumericPixel, const D: usize> ImageSource<P, D> for NaryMaximumFilter<P, D> {
    fn largest_region(&self) -> Region<D> {
        self.sources.first().expect("at least one source").largest_region()
    }
    fn spacing(&self) -> [f64; D] { self.sources[0].spacing() }
    fn origin(&self) -> [f64; D] { self.sources[0].origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let images: Vec<_> = self.sources.iter()
            .map(|s| s.generate_region(requested))
            .collect();
        let len = images[0].data.len();
        let mut data = images[0].data.clone();
        for img in &images[1..] {
            for (i, &p) in img.data.iter().enumerate() {
                if p.to_f64() > data[i].to_f64() { data[i] = p; }
            }
        }
        Image { region: requested, spacing: images[0].spacing, origin: images[0].origin, data }
    }
}

// ===========================================================================
// Constrained value addition / difference
// ===========================================================================

/// `clamp(a + b, lower, upper)`. Analog to `itk::ConstrainedValueAdditionImageFilter`.
pub struct ConstrainedValueAdditionFilter<S1, S2> {
    pub source1: S1,
    pub source2: S2,
    pub lower: f64,
    pub upper: f64,
}

impl<S1, S2> ConstrainedValueAdditionFilter<S1, S2> {
    pub fn new(source1: S1, source2: S2, lower: f64, upper: f64) -> Self {
        Self { source1, source2, lower, upper }
    }
}

impl<P, S1, S2, const D: usize> ImageSource<P, D> for ConstrainedValueAdditionFilter<S1, S2>
where
    P: NumericPixel,
    S1: ImageSource<P, D> + Sync,
    S2: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source1.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source1.spacing() }
    fn origin(&self) -> [f64; D] { self.source1.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let a = self.source1.generate_region(requested);
        let b = self.source2.generate_region(requested);
        let (lo, hi) = (self.lower, self.upper);
        let data: Vec<P> = a.data.par_iter().zip(b.data.par_iter())
            .map(|(&pa, &pb)| P::from_f64((pa.to_f64() + pb.to_f64()).clamp(lo, hi)))
            .collect();
        Image { region: requested, spacing: a.spacing, origin: a.origin, data }
    }
}

/// `clamp(a − b, lower, upper)`. Analog to `itk::ConstrainedValueDifferenceImageFilter`.
pub struct ConstrainedValueDifferenceFilter<S1, S2> {
    pub source1: S1,
    pub source2: S2,
    pub lower: f64,
    pub upper: f64,
}

impl<S1, S2> ConstrainedValueDifferenceFilter<S1, S2> {
    pub fn new(source1: S1, source2: S2, lower: f64, upper: f64) -> Self {
        Self { source1, source2, lower, upper }
    }
}

impl<P, S1, S2, const D: usize> ImageSource<P, D> for ConstrainedValueDifferenceFilter<S1, S2>
where
    P: NumericPixel,
    S1: ImageSource<P, D> + Sync,
    S2: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source1.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source1.spacing() }
    fn origin(&self) -> [f64; D] { self.source1.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let a = self.source1.generate_region(requested);
        let b = self.source2.generate_region(requested);
        let (lo, hi) = (self.lower, self.upper);
        let data: Vec<P> = a.data.par_iter().zip(b.data.par_iter())
            .map(|(&pa, &pb)| P::from_f64((pa.to_f64() - pb.to_f64()).clamp(lo, hi)))
            .collect();
        Image { region: requested, spacing: a.spacing, origin: a.origin, data }
    }
}

// ===========================================================================
// Vector pixel filters
// ===========================================================================

use crate::pixel::VecPixel;

/// Compute the Euclidean magnitude of vector pixels.
/// Analog to `itk::VectorMagnitudeImageFilter`.
pub struct VectorMagnitudeFilter<S, T, const N: usize> {
    pub source: S,
    _phantom: std::marker::PhantomData<T>,
}

impl<S, T, const N: usize> VectorMagnitudeFilter<S, T, N> {
    pub fn new(source: S) -> Self { Self { source, _phantom: std::marker::PhantomData } }
}

impl<T, S, const D: usize, const N: usize> ImageSource<T, D> for VectorMagnitudeFilter<S, T, N>
where
    T: NumericPixel,
    S: ImageSource<VecPixel<T, N>, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<T, D> {
        let input = self.source.generate_region(requested);
        let data: Vec<T> = input.data.par_iter()
            .map(|p| {
                let mag2: f64 = p.0.iter().map(|c| { let v = c.to_f64(); v * v }).sum();
                T::from_f64(mag2.sqrt())
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

/// Extract one scalar component from vector pixels.
/// Analog to `itk::VectorIndexSelectionCastImageFilter`.
pub struct VectorIndexSelectionFilter<S, T, const N: usize> {
    pub source: S,
    pub index: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<S, T, const N: usize> VectorIndexSelectionFilter<S, T, N> {
    pub fn new(source: S, index: usize) -> Self {
        Self { source, index, _phantom: std::marker::PhantomData }
    }
}

impl<T, S, const D: usize, const N: usize> ImageSource<T, D>
    for VectorIndexSelectionFilter<S, T, N>
where
    T: NumericPixel,
    S: ImageSource<VecPixel<T, N>, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<T, D> {
        let input = self.source.generate_region(requested);
        let idx = self.index.min(N - 1);
        let data: Vec<T> = input.data.par_iter()
            .map(|p| p.0[idx])
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

/// Compose two scalar images into a `VecPixel<T, 2>` image.
/// Analog to `itk::ComposeImageFilter` for 2-component output.
pub struct Compose2Filter<S1, S2, T> {
    pub source1: S1,
    pub source2: S2,
    _phantom: std::marker::PhantomData<T>,
}

impl<S1, S2, T> Compose2Filter<S1, S2, T> {
    pub fn new(source1: S1, source2: S2) -> Self {
        Self { source1, source2, _phantom: std::marker::PhantomData }
    }
}

impl<T, S1, S2, const D: usize> ImageSource<VecPixel<T, 2>, D> for Compose2Filter<S1, S2, T>
where
    T: NumericPixel,
    S1: ImageSource<T, D> + Sync,
    S2: ImageSource<T, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source1.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source1.spacing() }
    fn origin(&self) -> [f64; D] { self.source1.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<VecPixel<T, 2>, D> {
        let a = self.source1.generate_region(requested);
        let b = self.source2.generate_region(requested);
        let data: Vec<VecPixel<T, 2>> = a.data.par_iter().zip(b.data.par_iter())
            .map(|(&pa, &pb)| VecPixel([pa, pb]))
            .collect();
        Image { region: requested, spacing: a.spacing, origin: a.origin, data }
    }
}

/// Compose three scalar images into a `VecPixel<T, 3>` image.
/// Analog to `itk::ComposeImageFilter` for 3-component output.
pub struct Compose3Filter<S1, S2, S3, T> {
    pub source1: S1,
    pub source2: S2,
    pub source3: S3,
    _phantom: std::marker::PhantomData<T>,
}

impl<S1, S2, S3, T> Compose3Filter<S1, S2, S3, T> {
    pub fn new(source1: S1, source2: S2, source3: S3) -> Self {
        Self { source1, source2, source3, _phantom: std::marker::PhantomData }
    }
}

impl<T, S1, S2, S3, const D: usize> ImageSource<VecPixel<T, 3>, D>
    for Compose3Filter<S1, S2, S3, T>
where
    T: NumericPixel,
    S1: ImageSource<T, D> + Sync,
    S2: ImageSource<T, D> + Sync,
    S3: ImageSource<T, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source1.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source1.spacing() }
    fn origin(&self) -> [f64; D] { self.source1.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<VecPixel<T, 3>, D> {
        let a = self.source1.generate_region(requested);
        let b = self.source2.generate_region(requested);
        let c = self.source3.generate_region(requested);
        let data: Vec<VecPixel<T, 3>> = a.data.par_iter()
            .zip(b.data.par_iter())
            .zip(c.data.par_iter())
            .map(|((&pa, &pb), &pc)| VecPixel([pa, pb, pc]))
            .collect();
        Image { region: requested, spacing: a.spacing, origin: a.origin, data }
    }
}

// ===========================================================================
// Histogram matching filter
// ===========================================================================

/// Match the histogram of `source` to that of a `reference` image.
/// Analog to `itk::HistogramMatchingImageFilter`.
///
/// Computes empirical CDFs from both images and applies the piecewise-linear
/// mapping that transfers `source` intensities to match the reference CDF.
pub struct HistogramMatchingFilter<S, R> {
    pub source: S,
    pub reference: R,
    pub num_bins: usize,
}

impl<S, R> HistogramMatchingFilter<S, R> {
    pub fn new(source: S, reference: R) -> Self {
        Self { source, reference, num_bins: 256 }
    }
    pub fn with_num_bins(mut self, n: usize) -> Self { self.num_bins = n; self }
}

impl<P, S, R, const D: usize> ImageSource<P, D> for HistogramMatchingFilter<S, R>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
    R: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let src_full = self.source.generate_region(self.source.largest_region());
        let ref_full = self.reference.generate_region(self.reference.largest_region());

        let nb = self.num_bins;

        // Helper: build cumulative distribution function from image data
        let build_cdf = |data: &[P]| -> (f64, f64, Vec<f64>) {
            let (imin, imax) = data.iter().fold((f64::MAX, f64::MIN), |(mn, mx), &p| {
                let v = p.to_f64(); (mn.min(v), mx.max(v))
            });
            let range = imax - imin;
            let mut hist = vec![0u64; nb];
            for &p in data {
                let v = p.to_f64();
                let bin = if range < 1e-15 { 0 }
                    else { ((v - imin) / range * nb as f64) as usize }.min(nb - 1);
                hist[bin] += 1;
            }
            let total = data.len() as f64;
            let mut cdf = vec![0.0f64; nb];
            let mut acc = 0.0f64;
            for i in 0..nb {
                acc += hist[i] as f64 / total;
                cdf[i] = acc;
            }
            (imin, imax, cdf)
        };

        let (src_min, src_max, src_cdf) = build_cdf(&src_full.data);
        let (ref_min, ref_max, ref_cdf) = build_cdf(&ref_full.data);

        // Build intensity-to-intensity LUT: for each source intensity, find the reference
        // intensity that has the same CDF value
        let lut: Vec<f64> = (0..nb).map(|i| {
            let src_intensity = src_min + (i as f64 + 0.5) / nb as f64 * (src_max - src_min);
            let cdf_val = src_cdf[i];
            // Find bin in reference CDF with matching value
            let ref_bin = ref_cdf.partition_point(|&c| c < cdf_val).min(nb - 1);
            ref_min + (ref_bin as f64 + 0.5) / nb as f64 * (ref_max - ref_min)
        }).collect();

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let src_range = src_max - src_min;
        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let v = src_full.get_pixel(idx).to_f64();
                let bin = if src_range < 1e-15 { 0 }
                    else { ((v - src_min) / src_range * nb as f64) as usize }.min(nb - 1);
                P::from_f64(lut[bin])
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

    fn const_1d(n: usize, v: f32) -> Image<f32, 1> {
        Image::<f32, 1>::allocate(Region::new([0], [n]), [1.0], [0.0], v)
    }

    // --- Pointwise math ---

    #[test]
    fn abs_negated() {
        let mut img = const_1d(5, -3.0);
        img.data = vec![-1.0, -2.0, 3.0, -4.0, 5.0];
        let f = abs_filter(img);
        let out = f.generate_region(f.largest_region());
        for &v in &out.data { assert!(v >= 0.0); }
        assert!((out.data[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn sqrt_squares() {
        let img = ramp_1d(5); // 0,1,2,3,4
        let squared = square_filter(img);
        let back = sqrt_filter(squared);
        let out = back.generate_region(back.largest_region());
        for i in 0..5i64 {
            let v = out.get_pixel(Index([i]));
            assert!((v - i as f32).abs() < 1e-4, "at {i}: {v}");
        }
    }

    #[test]
    fn exp_log_roundtrip() {
        let img = ramp_1d(5); // 0..4
        // skip 0 since log(0) = -inf
        let shifted = ShiftScaleFilter::new(img, 1.0, 1.0); // 1..5
        let fwd = exp_filter(shifted);
        let back = log_filter(fwd);
        let out = back.generate_region(back.largest_region());
        for i in 0..5i64 {
            let expected = 1.0 + i as f32;
            let v = out.get_pixel(Index([i]));
            assert!((v - expected).abs() < 1e-4, "at {i}: {v}");
        }
    }

    // --- Intensity mapping ---

    #[test]
    fn shift_scale() {
        let img = const_1d(4, 2.0);
        let f = ShiftScaleFilter::new(img, 3.0, 2.0); // (2+3)*2 = 10
        let out = f.generate_region(f.largest_region());
        for &v in &out.data { assert!((v - 10.0).abs() < 1e-6); }
    }

    #[test]
    fn clamp() {
        let mut img = const_1d(5, 0.0);
        img.data = vec![-5.0, 0.0, 3.0, 7.0, 12.0];
        let f = ClampFilter::new(img, 0.0, 10.0);
        let out = f.generate_region(f.largest_region());
        assert!((out.data[0] - 0.0).abs() < 1e-6);
        assert!((out.data[4] - 10.0).abs() < 1e-6);
        assert!((out.data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn rescale_intensity() {
        let img = ramp_1d(11); // 0..10
        let f = RescaleIntensityFilter::new(img, 0.0, 1.0);
        let out = f.generate_region(f.largest_region());
        assert!((out.data[0] - 0.0).abs() < 1e-6);
        assert!((out.data[10] - 1.0).abs() < 1e-6);
        assert!((out.data[5] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn normalize() {
        let img = ramp_1d(5); // 0,1,2,3,4 — mean=2, std≈sqrt(2)
        let f = NormalizeFilter::new(img);
        let out = f.generate_region(f.largest_region());
        let sum: f32 = out.data.iter().sum();
        assert!(sum.abs() < 1e-5, "mean should be 0, sum={sum}");
    }

    #[test]
    fn invert_intensity() {
        let img = const_1d(4, 30.0);
        let f = InvertIntensityFilter::new(img, 255.0);
        let out = f.generate_region(f.largest_region());
        for &v in &out.data { assert!((v - 225.0).abs() < 1e-5); }
    }

    #[test]
    fn intensity_windowing() {
        let mut img = const_1d(5, 0.0);
        img.data = vec![0.0, 25.0, 50.0, 75.0, 100.0];
        let f = IntensityWindowingFilter::new(img, 25.0, 75.0, 0.0, 1.0);
        let out = f.generate_region(f.largest_region());
        // Values outside window are clamped, then rescaled
        assert!((out.data[0] - 0.0).abs() < 1e-6); // clamped to 25 → 0
        assert!((out.data[2] - 0.5).abs() < 1e-6); // 50 → 0.5
        assert!((out.data[4] - 1.0).abs() < 1e-6); // clamped to 75 → 1
    }

    #[test]
    fn sigmoid_range() {
        let img = const_1d(3, 0.0);
        let f = SigmoidFilter::new(img, 1.0, 0.0); // beta=0, alpha=1
        // At x=0 (= beta): sigmoid = 0.5, scaled to [0,1] = 0.5
        let out = f.generate_region(f.largest_region());
        for &v in &out.data { assert!((v - 0.5).abs() < 1e-6); }
    }

    // --- Binary ---

    #[test]
    fn add_subtract_inverse() {
        let a = ramp_1d(8);
        let b = ramp_1d(8);
        let sum = add_images(a, b);
        let diff = subtract_images(sum, ramp_1d(8));
        let out = diff.generate_region(diff.largest_region());
        for i in 0..8i64 {
            let v = out.get_pixel(Index([i]));
            assert!((v - i as f32).abs() < 1e-5, "at {i}: {v}");
        }
    }

    #[test]
    fn multiply_divide() {
        let a = ramp_1d(5); // 0..4
        let b = ShiftScaleFilter::new(ramp_1d(5), 1.0, 1.0); // 1..5
        let prod = multiply_images(a, b);
        let quot = divide_images(prod, ShiftScaleFilter::new(ramp_1d(5), 1.0, 1.0));
        let out = quot.generate_region(quot.largest_region());
        for i in 0..5i64 {
            let expected = i as f32;
            let v = out.get_pixel(Index([i]));
            assert!((v - expected).abs() < 1e-4, "at {i}: {v}");
        }
    }

    #[test]
    fn maximum_minimum() {
        let a = ramp_1d(5);
        let b = {
            let mut img = const_1d(5, 0.0);
            img.data = vec![4.0, 3.0, 2.0, 1.0, 0.0];
            img
        };
        let mx = maximum_images(a, b);
        let out = mx.generate_region(mx.largest_region());
        assert!((out.data[0] - 4.0).abs() < 1e-6);
        assert!((out.data[4] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn weighted_add() {
        let a = const_1d(4, 10.0);
        let b = const_1d(4, 0.0);
        let f = WeightedAddFilter::new(a, b, 0.3); // 0.3*10 + 0.7*0 = 3
        let out = f.generate_region(f.largest_region());
        for &v in &out.data { assert!((v - 3.0).abs() < 1e-5); }
    }

    // --- Mask ---

    #[test]
    fn mask_zeros_outside() {
        let img  = const_1d(5, 7.0);
        let mut mask = Image::<u8, 1>::allocate(Region::new([0], [5]), [1.0], [0.0], 1u8);
        mask.set_pixel(Index([2]), 0u8); // pixel 2 is masked

        let f = MaskFilter::<_, _, f32, u8>::new(img, mask, 0.0);
        let out = f.generate_region(f.largest_region());
        assert!((out.data[2] - 0.0).abs() < 1e-6);
        assert!((out.data[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn mask_negated() {
        let img  = const_1d(5, 7.0);
        let mask = Image::<u8, 1>::allocate(Region::new([0], [5]), [1.0], [0.0], 1u8);
        // All mask pixels are 1, so all image pixels get replaced by outside_value
        let f = MaskNegatedFilter::<_, _, f32, u8>::new(img, mask, -1.0);
        let out = f.generate_region(f.largest_region());
        for &v in &out.data { assert!((v - (-1.0)).abs() < 1e-6); }
    }
}
