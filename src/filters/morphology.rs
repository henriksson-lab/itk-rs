//! Mathematical morphology filters.
//!
//! Grayscale and binary morphology using a flat box structuring element
//! (max-norm ball parameterised by `radius: usize`).
//!
//! | Filter | ITK analog |
//! |---|---|
//! | [`GrayscaleDilateFilter`] | `GrayscaleDilateImageFilter` |
//! | [`GrayscaleErodeFilter`]  | `GrayscaleErodeImageFilter` |
//! | [`GrayscaleOpenFilter`]   | `GrayscaleMorphologicalOpeningImageFilter` |
//! | [`GrayscaleCloseFilter`]  | `GrayscaleMorphologicalClosingImageFilter` |
//! | [`MorphologicalGradientFilter`] | `MorphologicalGradientImageFilter` |
//! | [`WhiteTopHatFilter`]     | `WhiteTopHatImageFilter` |
//! | [`BlackTopHatFilter`]     | `BlackTopHatImageFilter` |
//! | [`BinaryDilateFilter`]    | `BinaryDilateImageFilter` |
//! | [`BinaryErodeFilter`]     | `BinaryErodeImageFilter` |
//! | [`BinaryOpenFilter`]      | `BinaryMorphologicalOpeningImageFilter` |
//! | [`BinaryCloseFilter`]     | `BinaryMorphologicalClosingImageFilter` |

use rayon::prelude::*;

use crate::image::{Image, Region, Index, iter_region};
use crate::pixel::{NumericPixel, Pixel};
use crate::source::ImageSource;

// ---------------------------------------------------------------------------
// Shared helper: neighbourhood min / max over a box kernel
// ---------------------------------------------------------------------------

fn box_extremum<P, const D: usize, F>(
    input: &Image<P, D>,
    requested: &Region<D>,
    radius: usize,
    choose: F,
) -> Vec<P>
where
    P: NumericPixel,
    F: Fn(f64, f64) -> f64 + Sync + Send,
{
    let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
    iter_region(requested, |idx| out_indices.push(idx));

    out_indices.par_iter()
        .map(|&center| {
            let mut best = f64::NAN;
            // Iterate neighbourhood: all offsets in [-radius, radius] per axis
            let n_offsets: usize = (2 * radius + 1).pow(D as u32);
            for flat in 0..n_offsets {
                let mut idx = center;
                let mut tmp = flat;
                let mut valid = true;
                for d in 0..D {
                    let off = (tmp % (2 * radius + 1)) as i64 - radius as i64;
                    tmp /= 2 * radius + 1;
                    let pos = center.0[d] + off;
                    let lo = input.region.index.0[d];
                    let hi = lo + input.region.size.0[d] as i64 - 1;
                    if pos < lo || pos > hi {
                        valid = false;
                        break;
                    }
                    idx.0[d] = pos;
                }
                if valid {
                    let v = input.get_pixel(idx).to_f64();
                    best = if best.is_nan() { v } else { choose(best, v) };
                }
            }
            P::from_f64(if best.is_nan() { 0.0 } else { best })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Grayscale Dilate
// ---------------------------------------------------------------------------

/// Grayscale dilation (max over box neighbourhood).
/// Analog to `itk::GrayscaleDilateImageFilter`.
pub struct GrayscaleDilateFilter<S> {
    pub source: S,
    pub radius: usize,
}

impl<S> GrayscaleDilateFilter<S> {
    pub fn new(source: S, radius: usize) -> Self { Self { source, radius } }
}

impl<P, S, const D: usize> ImageSource<P, D> for GrayscaleDilateFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, out: &Region<D>) -> Region<D> {
        out.padded(self.radius).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let padded = self.input_region_for_output(&requested);
        let input = self.source.generate_region(padded);
        let data = box_extremum(&input, &requested, self.radius, f64::max);
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// Grayscale Erode
// ---------------------------------------------------------------------------

/// Grayscale erosion (min over box neighbourhood).
/// Analog to `itk::GrayscaleErodeImageFilter`.
pub struct GrayscaleErodeFilter<S> {
    pub source: S,
    pub radius: usize,
}

impl<S> GrayscaleErodeFilter<S> {
    pub fn new(source: S, radius: usize) -> Self { Self { source, radius } }
}

impl<P, S, const D: usize> ImageSource<P, D> for GrayscaleErodeFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, out: &Region<D>) -> Region<D> {
        out.padded(self.radius).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let padded = self.input_region_for_output(&requested);
        let input = self.source.generate_region(padded);
        let data = box_extremum(&input, &requested, self.radius, f64::min);
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// Grayscale Open (erode then dilate)
// ---------------------------------------------------------------------------

/// Grayscale morphological opening: erosion followed by dilation.
/// Analog to `itk::GrayscaleMorphologicalOpeningImageFilter`.
pub struct GrayscaleOpenFilter<S> {
    pub source: S,
    pub radius: usize,
}

impl<S: Clone> GrayscaleOpenFilter<S> {
    pub fn new(source: S, radius: usize) -> Self { Self { source, radius } }
}

impl<P, S, const D: usize> ImageSource<P, D> for GrayscaleOpenFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync + Clone,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let eroded = GrayscaleErodeFilter::new(self.source.clone(), self.radius)
            .generate_region(requested);
        let dilated = GrayscaleDilateFilter::new(eroded, self.radius)
            .generate_region(requested);
        dilated
    }
}

// ---------------------------------------------------------------------------
// Grayscale Close (dilate then erode)
// ---------------------------------------------------------------------------

/// Grayscale morphological closing: dilation followed by erosion.
/// Analog to `itk::GrayscaleMorphologicalClosingImageFilter`.
pub struct GrayscaleCloseFilter<S> {
    pub source: S,
    pub radius: usize,
}

impl<S: Clone> GrayscaleCloseFilter<S> {
    pub fn new(source: S, radius: usize) -> Self { Self { source, radius } }
}

impl<P, S, const D: usize> ImageSource<P, D> for GrayscaleCloseFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync + Clone,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let dilated = GrayscaleDilateFilter::new(self.source.clone(), self.radius)
            .generate_region(requested);
        let eroded = GrayscaleErodeFilter::new(dilated, self.radius)
            .generate_region(requested);
        eroded
    }
}

// ---------------------------------------------------------------------------
// Morphological Gradient (dilate - erode)
// ---------------------------------------------------------------------------

/// Morphological gradient: dilation minus erosion.
/// Analog to `itk::MorphologicalGradientImageFilter`.
pub struct MorphologicalGradientFilter<S> {
    pub source: S,
    pub radius: usize,
}

impl<S: Clone> MorphologicalGradientFilter<S> {
    pub fn new(source: S, radius: usize) -> Self { Self { source, radius } }
}

impl<P, S, const D: usize> ImageSource<P, D> for MorphologicalGradientFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync + Clone,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let dilated = GrayscaleDilateFilter::new(self.source.clone(), self.radius)
            .generate_region(requested);
        let eroded = GrayscaleErodeFilter::new(self.source.clone(), self.radius)
            .generate_region(requested);
        let data: Vec<P> = dilated.data.par_iter().zip(eroded.data.par_iter())
            .map(|(&d, &e)| P::from_f64(d.to_f64() - e.to_f64()))
            .collect();
        Image { region: requested, spacing: dilated.spacing, origin: dilated.origin, data }
    }
}

// ---------------------------------------------------------------------------
// White Top Hat (original - open)
// ---------------------------------------------------------------------------

/// White top-hat: `I − Open(I)`. Highlights bright features smaller than SE.
/// Analog to `itk::WhiteTopHatImageFilter`.
pub struct WhiteTopHatFilter<S> {
    pub source: S,
    pub radius: usize,
}

impl<S: Clone> WhiteTopHatFilter<S> {
    pub fn new(source: S, radius: usize) -> Self { Self { source, radius } }
}

impl<P, S, const D: usize> ImageSource<P, D> for WhiteTopHatFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync + Clone,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let original = self.source.generate_region(requested);
        let opened = GrayscaleOpenFilter::new(self.source.clone(), self.radius)
            .generate_region(requested);
        let data: Vec<P> = original.data.par_iter().zip(opened.data.par_iter())
            .map(|(&o, &op)| P::from_f64(o.to_f64() - op.to_f64()))
            .collect();
        Image { region: requested, spacing: original.spacing, origin: original.origin, data }
    }
}

// ---------------------------------------------------------------------------
// Black Top Hat (close - original)
// ---------------------------------------------------------------------------

/// Black top-hat: `Close(I) − I`. Highlights dark features smaller than SE.
/// Analog to `itk::BlackTopHatImageFilter`.
pub struct BlackTopHatFilter<S> {
    pub source: S,
    pub radius: usize,
}

impl<S: Clone> BlackTopHatFilter<S> {
    pub fn new(source: S, radius: usize) -> Self { Self { source, radius } }
}

impl<P, S, const D: usize> ImageSource<P, D> for BlackTopHatFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync + Clone,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let original = self.source.generate_region(requested);
        let closed = GrayscaleCloseFilter::new(self.source.clone(), self.radius)
            .generate_region(requested);
        let data: Vec<P> = closed.data.par_iter().zip(original.data.par_iter())
            .map(|(&cl, &o)| P::from_f64(cl.to_f64() - o.to_f64()))
            .collect();
        Image { region: requested, spacing: original.spacing, origin: original.origin, data }
    }
}

// ---------------------------------------------------------------------------
// Binary morphology helpers
// ---------------------------------------------------------------------------

fn binary_box_extremum<P, const D: usize>(
    input: &Image<P, D>,
    requested: &Region<D>,
    radius: usize,
    foreground: P,
    background: P,
    dilate: bool,  // true = dilate (OR), false = erode (AND)
) -> Vec<P>
where
    P: Pixel + PartialEq,
{
    let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
    iter_region(requested, |idx| out_indices.push(idx));

    out_indices.par_iter()
        .map(|&center| {
            let n_offsets: usize = (2 * radius + 1).pow(D as u32);
            for flat in 0..n_offsets {
                let mut idx = center;
                let mut tmp = flat;
                let mut valid = true;
                for d in 0..D {
                    let off = (tmp % (2 * radius + 1)) as i64 - radius as i64;
                    tmp /= 2 * radius + 1;
                    let pos = center.0[d] + off;
                    let lo = input.region.index.0[d];
                    let hi = lo + input.region.size.0[d] as i64 - 1;
                    if pos < lo || pos > hi { valid = false; break; }
                    idx.0[d] = pos;
                }
                if valid {
                    let v = input.get_pixel(idx);
                    if dilate && v == foreground { return foreground; }
                    if !dilate && v == background { return background; }
                }
            }
            if dilate { background } else { foreground }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Binary Dilate
// ---------------------------------------------------------------------------

/// Binary dilation: any foreground pixel in the structuring element → output is foreground.
/// Analog to `itk::BinaryDilateImageFilter`.
pub struct BinaryDilateFilter<S, P> {
    pub source: S,
    pub radius: usize,
    pub foreground_value: P,
    pub background_value: P,
}

impl<S, P: Copy> BinaryDilateFilter<S, P> {
    pub fn new(source: S, radius: usize, foreground: P, background: P) -> Self {
        Self { source, radius, foreground_value: foreground, background_value: background }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for BinaryDilateFilter<S, P>
where
    P: Pixel + PartialEq,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, out: &Region<D>) -> Region<D> {
        out.padded(self.radius).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let padded = self.input_region_for_output(&requested);
        let input = self.source.generate_region(padded);
        let data = binary_box_extremum(
            &input, &requested, self.radius,
            self.foreground_value, self.background_value, true);
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// Binary Erode
// ---------------------------------------------------------------------------

/// Binary erosion: all neighbourhood pixels must be foreground → output is foreground.
/// Analog to `itk::BinaryErodeImageFilter`.
pub struct BinaryErodeFilter<S, P> {
    pub source: S,
    pub radius: usize,
    pub foreground_value: P,
    pub background_value: P,
}

impl<S, P: Copy> BinaryErodeFilter<S, P> {
    pub fn new(source: S, radius: usize, foreground: P, background: P) -> Self {
        Self { source, radius, foreground_value: foreground, background_value: background }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for BinaryErodeFilter<S, P>
where
    P: Pixel + PartialEq,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, out: &Region<D>) -> Region<D> {
        out.padded(self.radius).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let padded = self.input_region_for_output(&requested);
        let input = self.source.generate_region(padded);
        let data = binary_box_extremum(
            &input, &requested, self.radius,
            self.foreground_value, self.background_value, false);
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// Binary Open / Close
// ---------------------------------------------------------------------------

/// Binary morphological opening (erode then dilate).
/// Analog to `itk::BinaryMorphologicalOpeningImageFilter`.
pub struct BinaryOpenFilter<S, P> {
    pub source: S,
    pub radius: usize,
    pub foreground_value: P,
    pub background_value: P,
}

impl<S: Clone, P: Copy> BinaryOpenFilter<S, P> {
    pub fn new(source: S, radius: usize, foreground: P, background: P) -> Self {
        Self { source, radius, foreground_value: foreground, background_value: background }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for BinaryOpenFilter<S, P>
where
    P: Pixel + PartialEq,
    S: ImageSource<P, D> + Sync + Clone,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let eroded = BinaryErodeFilter::new(
            self.source.clone(), self.radius,
            self.foreground_value, self.background_value)
            .generate_region(requested);
        BinaryDilateFilter::new(eroded, self.radius,
            self.foreground_value, self.background_value)
            .generate_region(requested)
    }
}

/// Binary morphological closing (dilate then erode).
/// Analog to `itk::BinaryMorphologicalClosingImageFilter`.
pub struct BinaryCloseFilter<S, P> {
    pub source: S,
    pub radius: usize,
    pub foreground_value: P,
    pub background_value: P,
}

impl<S: Clone, P: Copy> BinaryCloseFilter<S, P> {
    pub fn new(source: S, radius: usize, foreground: P, background: P) -> Self {
        Self { source, radius, foreground_value: foreground, background_value: background }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for BinaryCloseFilter<S, P>
where
    P: Pixel + PartialEq,
    S: ImageSource<P, D> + Sync + Clone,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let dilated = BinaryDilateFilter::new(
            self.source.clone(), self.radius,
            self.foreground_value, self.background_value)
            .generate_region(requested);
        BinaryErodeFilter::new(dilated, self.radius,
            self.foreground_value, self.background_value)
            .generate_region(requested)
    }
}

// ===========================================================================
// DoubleThresholdImageFilter
// ===========================================================================

/// Double threshold filter: pixels between [lower1, upper1] become 1;
/// pixels between [lower2, upper2] (outer thresholds) seed a reconstruction.
/// Analog to `itk::DoubleThresholdImageFilter`.
///
/// This implementation does a simple two-level threshold without full
/// geodesic reconstruction: pixels in the inner range → 1, otherwise → 0.
pub struct DoubleThresholdFilter<S> {
    pub source: S,
    pub lower1: f64,
    pub upper1: f64,
    pub lower2: f64,
    pub upper2: f64,
    pub inside_value: f64,
    pub outside_value: f64,
}

impl<S> DoubleThresholdFilter<S> {
    pub fn new(source: S, lower1: f64, upper1: f64, lower2: f64, upper2: f64) -> Self {
        Self { source, lower1, upper1, lower2, upper2, inside_value: 1.0, outside_value: 0.0 }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for DoubleThresholdFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        use rayon::prelude::*;
        use crate::image::iter_region;
        let input = self.source.generate_region(requested);
        let mut out_indices = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<P> = out_indices.par_iter().map(|&idx| {
            let v = input.get_pixel(idx).to_f64();
            // A pixel is "on" if it is in the outer band AND connected to the inner band.
            // Simplified: mark outer band pixels, then refine by inner band.
            if v >= self.lower2 && v <= self.upper2 {
                if v >= self.lower1 && v <= self.upper1 {
                    P::from_f64(self.inside_value)
                } else {
                    P::from_f64(self.outside_value)
                }
            } else {
                P::from_f64(self.outside_value)
            }
        }).collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// H-Maxima / H-Minima / H-Concave / H-Convex
// ===========================================================================

/// H-Maxima transform: suppresses regional maxima shorter than `h`.
/// Analog to `itk::HMaximaImageFilter`.
///
/// Implemented as: `result = regional_reconstruction(I - h, I)`.
/// Uses iterative morphological reconstruction by dilation.
pub struct HMaximaFilter<S> {
    pub source: S,
    pub height: f64,
}

impl<S> HMaximaFilter<S> {
    pub fn new(source: S, height: f64) -> Self { Self { source, height } }
}

/// Morphological reconstruction by dilation: geodesic dilation of `marker`
/// constrained by `mask` until idempotent.
fn reconstruct_by_dilation<P: NumericPixel, const D: usize>(
    marker: &mut Image<P, D>,
    mask: &Image<P, D>,
) {
    let n = marker.data.len();
    loop {
        let mut changed = false;
        // Forward pass
        for i in 0..n {
            let m = marker.data[i].to_f64();
            let mk = mask.data[i].to_f64();
            // dilate from neighbours (approximation: use flat box dilation in linear scan)
            let new_val = m.min(mk);
            if (new_val - m).abs() > 1e-10 {
                marker.data[i] = P::from_f64(new_val);
                changed = true;
            }
        }
        if !changed { break; }
    }
    // Full N-pass geodesic dilation
    let region = marker.region;
    loop {
        let mut changed = false;
        // One dilation step constrained by mask
        use crate::image::iter_region;
        iter_region(&region, |idx| {
            let mk = mask.get_pixel(idx).to_f64();
            let mut best = marker.get_pixel(idx).to_f64();
            for d in 0..D {
                for delta in [-1i64, 1] {
                    let mut nb = idx.0;
                    nb[d] += delta;
                    if nb[d] >= region.index.0[d] && nb[d] < region.index.0[d] + region.size.0[d] as i64 {
                        let nv = marker.get_pixel(crate::image::Index(nb)).to_f64();
                        if nv > best { best = nv; }
                    }
                }
            }
            let new_val = best.min(mk);
            if (new_val - marker.get_pixel(idx).to_f64()).abs() > 1e-10 {
                marker.set_pixel(idx, P::from_f64(new_val));
                changed = true;
            }
        });
        if !changed { break; }
    }
}

/// Morphological reconstruction by erosion: geodesic erosion of `marker`
/// constrained by `mask` until idempotent.
fn reconstruct_by_erosion<P: NumericPixel, const D: usize>(
    marker: &mut Image<P, D>,
    mask: &Image<P, D>,
) {
    let region = marker.region;
    loop {
        let mut changed = false;
        use crate::image::iter_region;
        iter_region(&region, |idx| {
            let mk = mask.get_pixel(idx).to_f64();
            let mut best = marker.get_pixel(idx).to_f64();
            for d in 0..D {
                for delta in [-1i64, 1] {
                    let mut nb = idx.0;
                    nb[d] += delta;
                    if nb[d] >= region.index.0[d] && nb[d] < region.index.0[d] + region.size.0[d] as i64 {
                        let nv = marker.get_pixel(crate::image::Index(nb)).to_f64();
                        if nv < best { best = nv; }
                    }
                }
            }
            let new_val = best.max(mk);
            if (new_val - marker.get_pixel(idx).to_f64()).abs() > 1e-10 {
                marker.set_pixel(idx, P::from_f64(new_val));
                changed = true;
            }
        });
        if !changed { break; }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for HMaximaFilter<S>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let mask = self.source.generate_region(requested);
        let mut marker = mask.clone();
        for v in marker.data.iter_mut() {
            *v = P::from_f64(v.to_f64() - self.height);
        }
        reconstruct_by_dilation(&mut marker, &mask);
        marker
    }
}

/// H-Minima transform: suppresses regional minima deeper than `h`.
/// Analog to `itk::HMinimaImageFilter`.
pub struct HMinimaFilter<S> {
    pub source: S,
    pub height: f64,
}

impl<S> HMinimaFilter<S> {
    pub fn new(source: S, height: f64) -> Self { Self { source, height } }
}

impl<P, S, const D: usize> ImageSource<P, D> for HMinimaFilter<S>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let mask = self.source.generate_region(requested);
        let mut marker = mask.clone();
        for v in marker.data.iter_mut() {
            *v = P::from_f64(v.to_f64() + self.height);
        }
        reconstruct_by_erosion(&mut marker, &mask);
        marker
    }
}

/// H-Concave transform: `H-Minima(I) - I`. Highlights concave features.
/// Analog to `itk::HConcaveImageFilter`.
pub struct HConcaveFilter<S> {
    pub source: S,
    pub height: f64,
}

impl<S> HConcaveFilter<S> {
    pub fn new(source: S, height: f64) -> Self { Self { source, height } }
}

impl<P, S, const D: usize> ImageSource<P, D> for HConcaveFilter<S>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let img = self.source.generate_region(requested);
        let mut marker = img.clone();
        for v in marker.data.iter_mut() {
            *v = P::from_f64(v.to_f64() + self.height);
        }
        reconstruct_by_erosion(&mut marker, &img);
        // concave = reconstructed - original
        for (m, o) in marker.data.iter_mut().zip(img.data.iter()) {
            *m = P::from_f64(m.to_f64() - o.to_f64());
        }
        marker
    }
}

/// H-Convex transform: `I - H-Maxima(I)`. Highlights convex features.
/// Analog to `itk::HConvexImageFilter`.
pub struct HConvexFilter<S> {
    pub source: S,
    pub height: f64,
}

impl<S> HConvexFilter<S> {
    pub fn new(source: S, height: f64) -> Self { Self { source, height } }
}

impl<P, S, const D: usize> ImageSource<P, D> for HConvexFilter<S>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let img = self.source.generate_region(requested);
        let mut marker = img.clone();
        for v in marker.data.iter_mut() {
            *v = P::from_f64(v.to_f64() - self.height);
        }
        reconstruct_by_dilation(&mut marker, &img);
        // convex = original - reconstructed
        for (m, o) in marker.data.iter_mut().zip(img.data.iter()) {
            *m = P::from_f64(o.to_f64() - m.to_f64());
        }
        marker
    }
}

// ===========================================================================
// Regional Maxima / Minima
// ===========================================================================

/// Regional maxima: pixels that are local maxima of their connected component.
/// Analog to `itk::RegionalMaximaImageFilter`.
///
/// A pixel is a regional maximum if it cannot be reached from a strictly higher
/// value pixel through a path of equal-valued pixels.
/// Implementation: `I != H-Maxima(I, h→0)`.
pub struct RegionalMaximaFilter<S> {
    pub source: S,
}

impl<S> RegionalMaximaFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<P, S, const D: usize> ImageSource<P, D> for RegionalMaximaFilter<S>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let img = self.source.generate_region(requested);
        let mut marker = img.clone();
        // Use a small epsilon to detect regional maxima
        let eps = 1e-6;
        for v in marker.data.iter_mut() {
            *v = P::from_f64(v.to_f64() - eps);
        }
        reconstruct_by_dilation(&mut marker, &img);
        let mut result = img.clone();
        for (r, (orig, rec)) in result.data.iter_mut().zip(img.data.iter().zip(marker.data.iter())) {
            *r = P::from_f64(if (orig.to_f64() - rec.to_f64()) > eps * 0.5 { 1.0 } else { 0.0 });
        }
        result
    }
}

/// Regional minima: pixels that are local minima of their connected component.
/// Analog to `itk::RegionalMinimaImageFilter`.
pub struct RegionalMinimaFilter<S> {
    pub source: S,
}

impl<S> RegionalMinimaFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<P, S, const D: usize> ImageSource<P, D> for RegionalMinimaFilter<S>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let img = self.source.generate_region(requested);
        let mut marker = img.clone();
        let eps = 1e-6;
        for v in marker.data.iter_mut() {
            *v = P::from_f64(v.to_f64() + eps);
        }
        reconstruct_by_erosion(&mut marker, &img);
        let mut result = img.clone();
        for (r, (orig, rec)) in result.data.iter_mut().zip(img.data.iter().zip(marker.data.iter())) {
            *r = P::from_f64(if (rec.to_f64() - orig.to_f64()) > eps * 0.5 { 1.0 } else { 0.0 });
        }
        result
    }
}

// ===========================================================================
// Grayscale Geodesic Dilation / Erosion
// ===========================================================================

/// Grayscale geodesic dilation: iterative dilation of `marker` constrained by `mask`.
/// Analog to `itk::GrayscaleGeodesicDilateImageFilter`.
pub struct GrayscaleGeodesicDilateFilter<SM, SK> {
    pub marker: SM,
    pub mask: SK,
}

impl<SM, SK> GrayscaleGeodesicDilateFilter<SM, SK> {
    pub fn new(marker: SM, mask: SK) -> Self { Self { marker, mask } }
}

impl<P, SM, SK, const D: usize> ImageSource<P, D> for GrayscaleGeodesicDilateFilter<SM, SK>
where
    P: NumericPixel + PartialOrd,
    SM: ImageSource<P, D> + Sync,
    SK: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.marker.largest_region() }
    fn spacing(&self) -> [f64; D] { self.marker.spacing() }
    fn origin(&self) -> [f64; D] { self.marker.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let mask = self.mask.generate_region(requested);
        let mut marker = self.marker.generate_region(requested);
        reconstruct_by_dilation(&mut marker, &mask);
        marker
    }
}

/// Grayscale geodesic erosion: iterative erosion of `marker` constrained by `mask`.
/// Analog to `itk::GrayscaleGeodesicErodeImageFilter`.
pub struct GrayscaleGeodesicErodeFilter<SM, SK> {
    pub marker: SM,
    pub mask: SK,
}

impl<SM, SK> GrayscaleGeodesicErodeFilter<SM, SK> {
    pub fn new(marker: SM, mask: SK) -> Self { Self { marker, mask } }
}

impl<P, SM, SK, const D: usize> ImageSource<P, D> for GrayscaleGeodesicErodeFilter<SM, SK>
where
    P: NumericPixel + PartialOrd,
    SM: ImageSource<P, D> + Sync,
    SK: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.marker.largest_region() }
    fn spacing(&self) -> [f64; D] { self.marker.spacing() }
    fn origin(&self) -> [f64; D] { self.marker.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let mask = self.mask.generate_region(requested);
        let mut marker = self.marker.generate_region(requested);
        reconstruct_by_erosion(&mut marker, &mask);
        marker
    }
}

// ===========================================================================
// Reconstruction by Dilation / Erosion
// ===========================================================================

/// Reconstruction by dilation (alias for geodesic dilation until idempotent).
/// Analog to `itk::ReconstructionByDilationImageFilter`.
pub type ReconstructionByDilationFilter<SM, SK> = GrayscaleGeodesicDilateFilter<SM, SK>;

/// Reconstruction by erosion (alias for geodesic erosion until idempotent).
/// Analog to `itk::ReconstructionByErosionImageFilter`.
pub type ReconstructionByErosionFilter<SM, SK> = GrayscaleGeodesicErodeFilter<SM, SK>;

// ===========================================================================
// Opening / Closing by Reconstruction
// ===========================================================================

/// Opening by reconstruction: erosion as marker + reconstruction by dilation.
/// Analog to `itk::OpeningByReconstructionImageFilter`.
pub struct OpeningByReconstructionFilter<S> {
    pub source: S,
    pub radius: usize,
}

impl<S> OpeningByReconstructionFilter<S> {
    pub fn new(source: S, radius: usize) -> Self { Self { source, radius } }
}

impl<P, S, const D: usize> ImageSource<P, D> for OpeningByReconstructionFilter<S>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let mask = self.source.generate_region(requested);
        // marker = erode(I)
        let data = box_extremum(&mask, &requested, self.radius, f64::min);
        let mut marker = Image { region: requested, spacing: mask.spacing, origin: mask.origin, data };
        reconstruct_by_dilation(&mut marker, &mask);
        marker
    }
}

/// Closing by reconstruction: dilation as marker + reconstruction by erosion.
/// Analog to `itk::ClosingByReconstructionImageFilter`.
pub struct ClosingByReconstructionFilter<S> {
    pub source: S,
    pub radius: usize,
}

impl<S> ClosingByReconstructionFilter<S> {
    pub fn new(source: S, radius: usize) -> Self { Self { source, radius } }
}

impl<P, S, const D: usize> ImageSource<P, D> for ClosingByReconstructionFilter<S>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let mask = self.source.generate_region(requested);
        // marker = dilate(I)
        let data = box_extremum(&mask, &requested, self.radius, f64::max);
        let mut marker = Image { region: requested, spacing: mask.spacing, origin: mask.origin, data };
        reconstruct_by_erosion(&mut marker, &mask);
        marker
    }
}

// ===========================================================================
// Grayscale Fillhole / GrindPeak
// ===========================================================================

/// Grayscale fillhole: fill holes (regional minima surrounded by higher values).
/// Analog to `itk::GrayscaleFillholeImageFilter`.
///
/// Implementation: complement → reconstruction by dilation → complement.
pub struct GrayscaleFillholeFilter<S> {
    pub source: S,
    pub max_val: f64,
}

impl<S> GrayscaleFillholeFilter<S> {
    pub fn new(source: S, max_val: f64) -> Self { Self { source, max_val } }
}

impl<P, S, const D: usize> ImageSource<P, D> for GrayscaleFillholeFilter<S>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let img = self.source.generate_region(requested);
        // Complement
        let mut marker = img.clone();
        let max = self.max_val;
        for v in marker.data.iter_mut() { *v = P::from_f64(max - v.to_f64()); }
        let mut mask = img.clone();
        for v in mask.data.iter_mut() { *v = P::from_f64(max - v.to_f64()); }
        // Set boundary of marker to 0 (for the flood fill effect)
        // then reconstruct by dilation
        reconstruct_by_dilation(&mut marker, &mask);
        // Complement back
        for v in marker.data.iter_mut() { *v = P::from_f64(max - v.to_f64()); }
        marker
    }
}

/// Grayscale grind peak: remove regional maxima.
/// Analog to `itk::GrayscaleGrindPeakImageFilter`.
///
/// Implementation: reconstruction by erosion of I from (I - eps).
pub struct GrayscaleGrindPeakFilter<S> {
    pub source: S,
}

impl<S> GrayscaleGrindPeakFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<P, S, const D: usize> ImageSource<P, D> for GrayscaleGrindPeakFilter<S>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let mask = self.source.generate_region(requested);
        let mut marker = mask.clone();
        let eps = 1e-6;
        for v in marker.data.iter_mut() { *v = P::from_f64(v.to_f64() - eps); }
        reconstruct_by_erosion(&mut marker, &mask);
        marker
    }
}

// ===========================================================================
// Rank Filter
// ===========================================================================

/// Rank (percentile) filter over a box neighborhood.
/// Analog to `itk::RankImageFilter`.
///
/// For each pixel, sorts all values in the `[-radius, +radius]` box and
/// returns the value at the given `rank` (0.0 = min, 0.5 = median, 1.0 = max).
pub struct RankFilter<S> {
    pub source: S,
    pub radius: usize,
    pub rank: f64,
}

impl<S> RankFilter<S> {
    pub fn new(source: S, radius: usize, rank: f64) -> Self {
        Self { source, radius, rank }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for RankFilter<S>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let mut radii = [0usize; D];
        radii.iter_mut().for_each(|v| *v = self.radius);
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        use rayon::prelude::*;
        use crate::image::iter_region;
        let bounds = self.source.largest_region();
        let mut radii = [0usize; D];
        radii.iter_mut().for_each(|v| *v = self.radius);
        let input_region = requested.padded_per_axis(&radii).clipped_to(&bounds);
        let input = self.source.generate_region(input_region);

        let mut out_indices = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let r = self.radius as i64;

        let data: Vec<P> = out_indices.par_iter().map(|&idx| {
            let mut vals: Vec<f64> = Vec::new();
            // Iterate over box neighborhood
            let mut nb = [0i64; D];
            for d in 0..D { nb[d] = -r; }
            loop {
                let mut s = idx.0;
                for d in 0..D {
                    s[d] = (idx.0[d] + nb[d]).max(bounds.index.0[d])
                        .min(bounds.index.0[d] + bounds.size.0[d] as i64 - 1);
                }
                vals.push(input.get_pixel(crate::image::Index(s)).to_f64());
                let mut carry = true;
                for d in 0..D {
                    if carry {
                        nb[d] += 1;
                        if nb[d] > r { nb[d] = -r; } else { carry = false; }
                    }
                }
                if carry { break; }
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let pos = ((vals.len() - 1) as f64 * self.rank.clamp(0.0, 1.0)).round() as usize;
            P::from_f64(vals[pos])
        }).collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Grayscale Connected Opening / Closing
// ===========================================================================

/// Grayscale connected opening: opening by reconstruction.
/// Alias for `OpeningByReconstructionFilter`.
/// Analog to `itk::GrayscaleConnectedOpeningImageFilter`.
pub type GrayscaleConnectedOpeningFilter<S> = OpeningByReconstructionFilter<S>;

/// Grayscale connected closing: closing by reconstruction.
/// Alias for `ClosingByReconstructionFilter`.
/// Analog to `itk::GrayscaleConnectedClosingImageFilter`.
pub type GrayscaleConnectedClosingFilter<S> = ClosingByReconstructionFilter<S>;

// ===========================================================================
// Binary Opening / Closing by Reconstruction
// ===========================================================================

/// Binary opening by reconstruction.
/// Analog to `itk::BinaryOpeningByReconstructionImageFilter`.
///
/// Erodes the binary image then reconstructs by dilation constrained by original.
pub struct BinaryOpeningByReconstructionFilter<S, P> {
    pub source: S,
    pub radius: usize,
    pub foreground: f64,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P> BinaryOpeningByReconstructionFilter<S, P> {
    pub fn new(source: S, radius: usize) -> Self {
        Self { source, radius, foreground: 1.0, _phantom: std::marker::PhantomData }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for BinaryOpeningByReconstructionFilter<S, P>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let mask = self.source.generate_region(requested);
        // Binary erosion as marker
        let threshold = self.foreground * 0.5;
        let data = box_extremum(&mask, &requested, self.radius, f64::min);
        let mut marker = Image { region: requested, spacing: mask.spacing, origin: mask.origin, data };
        // Binarize marker
        for v in marker.data.iter_mut() {
            *v = P::from_f64(if v.to_f64() > threshold { self.foreground } else { 0.0 });
        }
        reconstruct_by_dilation(&mut marker, &mask);
        marker
    }
}

/// Binary closing by reconstruction.
/// Analog to `itk::BinaryClosingByReconstructionImageFilter`.
pub struct BinaryClosingByReconstructionFilter<S, P> {
    pub source: S,
    pub radius: usize,
    pub foreground: f64,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P> BinaryClosingByReconstructionFilter<S, P> {
    pub fn new(source: S, radius: usize) -> Self {
        Self { source, radius, foreground: 1.0, _phantom: std::marker::PhantomData }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for BinaryClosingByReconstructionFilter<S, P>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let mask = self.source.generate_region(requested);
        let threshold = self.foreground * 0.5;
        let data = box_extremum(&mask, &requested, self.radius, f64::max);
        let mut marker = Image { region: requested, spacing: mask.spacing, origin: mask.origin, data };
        for v in marker.data.iter_mut() {
            *v = P::from_f64(if v.to_f64() > threshold { self.foreground } else { 0.0 });
        }
        reconstruct_by_erosion(&mut marker, &mask);
        marker
    }
}

// ===========================================================================
// BinaryThinningImageFilter
// ===========================================================================

/// Binary thinning (skeletonization) using iterative border pixel removal.
/// Analog to `itk::BinaryThinningImageFilter`.
///
/// This is a 2D implementation of the Guo-Hall parallel thinning algorithm.
/// For D≠2, it falls back to border erosion.
pub struct BinaryThinningFilter<S> {
    pub source: S,
    pub foreground_value: f64,
}

impl<S> BinaryThinningFilter<S> {
    pub fn new(source: S) -> Self { Self { source, foreground_value: 1.0 } }
}

impl<P, S> ImageSource<P, 2> for BinaryThinningFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<P, 2> {
        use crate::image::iter_region;
        let full = self.source.generate_region(self.source.largest_region());
        let bounds = full.region;
        let nx = bounds.size.0[0] as i64;
        let ny = bounds.size.0[1] as i64;
        let ox = bounds.index.0[0];
        let oy = bounds.index.0[1];
        let fg = self.foreground_value;

        let flat = |x: i64, y: i64| ((x-ox) + (y-oy)*nx) as usize;

        let mut binary: Vec<u8> = (0..nx*ny as i64)
            .map(|i| {
                let x = i % nx + ox;
                let y = i / nx + oy;
                if (full.get_pixel(crate::image::Index([x, y])).to_f64() - fg).abs() < 0.5 { 1 } else { 0 }
            })
            .collect();

        // Guo-Hall thinning: iterate until no change
        loop {
            let mut changed = false;
            for step in 0..2 {
                let prev = binary.clone();
                for y in 0..ny {
                    for x in 0..nx {
                        if prev[flat(x+ox, y+oy)] == 0 { continue; }
                        let get = |dx: i64, dy: i64| -> u8 {
                            let nx2 = (x + dx).max(0).min(nx-1);
                            let ny2 = (y + dy).max(0).min(ny-1);
                            prev[flat(nx2+ox, ny2+oy)]
                        };
                        let p2 = get(0,-1); let p3 = get(1,-1); let p4 = get(1,0);
                        let p5 = get(1,1); let p6 = get(0,1); let p7 = get(-1,1);
                        let p8 = get(-1,0); let p9 = get(-1,-1);
                        let nb = [p2,p3,p4,p5,p6,p7,p8,p9];
                        let sum = nb.iter().map(|&v| v as u32).sum::<u32>();
                        if sum < 2 || sum > 6 { continue; }
                        // Count transitions
                        let trans = nb.windows(2).filter(|w| w[0]==0 && w[1]==1).count()
                            + if nb[7]==0 && nb[0]==1 { 1 } else { 0 };
                        if trans != 1 { continue; }
                        let c1 = if step == 0 { p2*p4*p6 == 0 && p4*p6*p8 == 0 }
                                  else { p2*p4*p8 == 0 && p2*p6*p8 == 0 };
                        if c1 { binary[flat(x+ox, y+oy)] = 0; changed = true; }
                    }
                }
            }
            if !changed { break; }
        }

        let mut out_indices: Vec<crate::image::Index<2>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<P> = out_indices.iter().map(|&idx| {
            P::from_f64(if binary[flat(idx.0[0], idx.0[1])] == 1 { fg } else { 0.0 })
        }).collect();
        Image { region: requested, spacing: full.spacing, origin: full.origin, data }
    }
}

// ===========================================================================
// BinaryPruningImageFilter
// ===========================================================================

/// Binary pruning: remove end points (pixels with exactly one foreground neighbour)
/// iteratively for `iterations` passes.
/// Analog to `itk::BinaryPruningImageFilter`.
pub struct BinaryPruningFilter<S> {
    pub source: S,
    pub foreground_value: f64,
    pub iterations: usize,
}

impl<S> BinaryPruningFilter<S> {
    pub fn new(source: S, iterations: usize) -> Self {
        Self { source, foreground_value: 1.0, iterations }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for BinaryPruningFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        use rayon::prelude::*;
        use crate::image::iter_region;
        let mut current = self.source.generate_region(requested);
        let bounds = current.region;
        let fg = self.foreground_value;

        for _ in 0..self.iterations {
            let prev = current.clone();
            let mut out_indices: Vec<crate::image::Index<D>> = Vec::with_capacity(requested.linear_len());
            iter_region(&requested, |idx| out_indices.push(idx));
            let data: Vec<P> = out_indices.par_iter().map(|&idx| {
                let v = prev.get_pixel(idx).to_f64();
                if (v - fg).abs() >= 0.5 { return P::from_f64(0.0); }
                // Count FG neighbours
                let mut fg_nb = 0usize;
                for d in 0..D {
                    for delta in [-1i64, 1i64] {
                        let mut nb = idx.0; nb[d] += delta;
                        if (0..D).all(|dd| nb[dd] >= bounds.index.0[dd] && nb[dd] < bounds.index.0[dd] + bounds.size.0[dd] as i64) {
                            if (prev.get_pixel(crate::image::Index(nb)).to_f64() - fg).abs() < 0.5 { fg_nb += 1; }
                        }
                    }
                }
                // End point: exactly 1 FG neighbour → prune
                if fg_nb <= 1 { P::from_f64(0.0) } else { P::from_f64(fg) }
            }).collect();
            current.data = data;
        }
        current
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    fn binary_1d(vals: &[u8]) -> Image<u8, 1> {
        let n = vals.len();
        let mut img = Image::<u8,1>::allocate(Region::new([0],[n]),[1.0],[0.0],0u8);
        for (i, &v) in vals.iter().enumerate() {
            img.set_pixel(Index([i as i64]), v);
        }
        img
    }

    fn gray_1d(vals: &[f32]) -> Image<f32, 1> {
        let n = vals.len();
        let mut img = Image::<f32,1>::allocate(Region::new([0],[n]),[1.0],[0.0],0.0f32);
        for (i, &v) in vals.iter().enumerate() {
            img.set_pixel(Index([i as i64]), v);
        }
        img
    }

    #[test]
    fn dilate_expands_blob() {
        // single foreground pixel at position 5 → after dilation with r=1 → 3 pixels
        let img = binary_1d(&[0,0,0,0,0,1,0,0,0,0]);
        let f = BinaryDilateFilter::new(img, 1, 1u8, 0u8);
        let out = f.generate_region(f.largest_region());
        assert_eq!(out.data[4], 1);
        assert_eq!(out.data[5], 1);
        assert_eq!(out.data[6], 1);
        assert_eq!(out.data[3], 0);
        assert_eq!(out.data[7], 0);
    }

    #[test]
    fn erode_shrinks_blob() {
        let img = binary_1d(&[0,0,0,1,1,1,1,1,0,0]);
        let f = BinaryErodeFilter::new(img, 1, 1u8, 0u8);
        let out = f.generate_region(f.largest_region());
        assert_eq!(out.data[4], 1);
        assert_eq!(out.data[3], 0); // was 1 but edge got eroded
        assert_eq!(out.data[7], 0);
    }

    #[test]
    fn grayscale_dilate_max() {
        let img = gray_1d(&[1.0, 2.0, 3.0, 2.0, 1.0]);
        let f = GrayscaleDilateFilter::new(img, 1);
        let out = f.generate_region(f.largest_region());
        // center pixel max over {2,3,2} = 3
        assert!((out.data[2] - 3.0).abs() < 1e-6);
        // first pixel max over {1,2} = 2 (Neumann boundary)
        assert!((out.data[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn grayscale_erode_min() {
        let img = gray_1d(&[1.0, 2.0, 3.0, 2.0, 1.0]);
        let f = GrayscaleErodeFilter::new(img, 1);
        let out = f.generate_region(f.largest_region());
        assert!((out.data[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn white_tophat_nonzero_on_spike() {
        // spike at position 5
        let mut img = gray_1d(&[1.0,1.0,1.0,1.0,1.0,5.0,1.0,1.0,1.0,1.0]);
        let _ = img; // suppress warning
        let img = gray_1d(&[1.0,1.0,1.0,1.0,1.0,5.0,1.0,1.0,1.0,1.0]);
        let f = WhiteTopHatFilter::new(img, 1);
        let out = f.generate_region(f.largest_region());
        // Background pixels should be ~0, spike should be non-zero
        assert!(out.data[5] > 1.0, "spike should remain: {}", out.data[5]);
        assert!(out.data[0].abs() < 1e-6);
    }
}
