//! Image grid / resampling filters.
//!
//! Covers ITK's geometric transformation, padding, cropping, and tiling filters.

use rayon::prelude::*;
use std::marker::PhantomData;

use crate::image::{Image, Region, Index, Size, iter_region};
use crate::pixel::{Pixel, NumericPixel};
use crate::source::ImageSource;
use crate::interpolate::{Interpolate, LinearInterpolator};
use crate::transform::Transform;

// ===========================================================================
// Resample image filter
// ===========================================================================

/// Resamples an image via an invertible geometric transform.
/// Metadata (output region/spacing/origin) is set explicitly.
pub struct ResampleImageFilterD<S, T, I, P, const D: usize> {
    pub source: S,
    pub transform: T,
    pub interpolator: I,
    pub output_region: Region<D>,
    pub output_spacing: [f64; D],
    pub output_origin: [f64; D],
    pub default_value: P,
}

impl<S, T, I, P, const D: usize> ResampleImageFilterD<S, T, I, P, D>
where P: Copy
{
    pub fn new(
        source: S,
        transform: T,
        interpolator: I,
        output_region: Region<D>,
        output_spacing: [f64; D],
        output_origin: [f64; D],
        default_value: P,
    ) -> Self {
        Self { source, transform, interpolator, output_region,
               output_spacing, output_origin, default_value }
    }
}

impl<P, S, T, I, const D: usize> ImageSource<P, D> for ResampleImageFilterD<S, T, I, P, D>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
    T: Transform<D> + Sync,
    I: Interpolate<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.output_region }
    fn spacing(&self) -> [f64; D] { self.output_spacing }
    fn origin(&self) -> [f64; D] { self.output_origin }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let src_full = self.source.generate_region(self.source.largest_region());
        let src_origin = self.source.origin();
        let src_spacing = self.source.spacing();

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let dv = self.default_value;
        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                // Physical position of output pixel
                let mut phys_out = [0.0f64; D];
                for d in 0..D {
                    phys_out[d] = self.output_origin[d]
                        + idx.0[d] as f64 * self.output_spacing[d];
                }
                // Inverse transform → input physical position
                let phys_in = match self.transform.inverse_transform_point(phys_out) {
                    Some(p) => p,
                    None => return dv,
                };
                // Convert to continuous index in source
                let mut cont_idx = [0.0f64; D];
                for d in 0..D {
                    cont_idx[d] = (phys_in[d] - src_origin[d]) / src_spacing[d];
                }
                // Check bounds (allow slight margin)
                let src_region = src_full.region;
                let mut in_bounds = true;
                for d in 0..D {
                    let lo = src_region.index.0[d] as f64 - 0.5;
                    let hi = (src_region.index.0[d] + src_region.size.0[d] as i64) as f64 - 0.5;
                    if cont_idx[d] < lo || cont_idx[d] > hi {
                        in_bounds = false;
                        break;
                    }
                }
                if !in_bounds { return dv; }
                self.interpolator.evaluate(&src_full, cont_idx)
            })
            .collect();
        Image { region: requested, spacing: self.output_spacing, origin: self.output_origin, data }
    }
}

/// Type alias for the common 2-D case.
pub type ResampleImageFilter2D<S, T, I, P> = ResampleImageFilterD<S, T, I, P, 2>;
/// Type alias for the common 3-D case.
pub type ResampleImageFilter3D<S, T, I, P> = ResampleImageFilterD<S, T, I, P, 3>;

// ===========================================================================
// Warp image filter (displacement-field based)
// ===========================================================================

/// Warps an image using a per-pixel displacement field.
/// Analog to `itk::WarpImageFilter`.
///
/// The displacement image has `VecPixel<f64, D>` pixels giving the physical
/// displacement to add to the sampling position.
pub struct WarpImageFilter<S, SD, I, P> {
    pub source: S,
    pub displacement: SD,
    pub interpolator: I,
    _phantom: PhantomData<P>,
}

impl<S, SD, I, P> WarpImageFilter<S, SD, I, P> {
    pub fn new(source: S, displacement: SD, interpolator: I) -> Self {
        Self { source, displacement, interpolator, _phantom: PhantomData }
    }
}

use crate::pixel::VecPixel;

impl<P, S, SD, I, const D: usize> ImageSource<P, D> for WarpImageFilter<S, SD, I, P>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
    SD: ImageSource<VecPixel<f64, D>, D> + Sync,
    I: Interpolate<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let src_full = self.source.generate_region(self.source.largest_region());
        let disp = self.displacement.generate_region(requested);
        let src_origin = self.source.origin();
        let src_spacing = self.source.spacing();

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                // Get displacement
                let dp = disp.get_pixel(idx);
                // Compute displaced continuous index
                let mut cont = [0.0f64; D];
                for d in 0..D {
                    let phys = src_origin[d] + idx.0[d] as f64 * src_spacing[d] + dp.0[d];
                    cont[d] = (phys - src_origin[d]) / src_spacing[d];
                }
                self.interpolator.evaluate(&src_full, cont)
            })
            .collect();
        Image { region: requested, spacing: disp.spacing, origin: disp.origin, data }
    }
}

// ===========================================================================
// Flip filter
// ===========================================================================

/// Flip image along specified axes.
/// Analog to `itk::FlipImageFilter`.
pub struct FlipImageFilter<S> {
    pub source: S,
    /// One entry per axis: `true` = flip that axis.
    pub flip_axes: Vec<bool>,
}

impl<S> FlipImageFilter<S> {
    pub fn new(source: S, flip_axes: Vec<bool>) -> Self {
        Self { source, flip_axes }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for FlipImageFilter<S>
where
    P: Pixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let full = self.source.generate_region(self.source.largest_region());
        let region = full.region;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let flip = &self.flip_axes;
        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let mut src_idx = idx;
                for d in 0..D {
                    if *flip.get(d).unwrap_or(&false) {
                        let lo = region.index.0[d];
                        let hi = lo + region.size.0[d] as i64 - 1;
                        src_idx.0[d] = hi - (idx.0[d] - lo);
                    }
                }
                full.get_pixel(src_idx)
            })
            .collect();
        Image { region: requested, spacing: full.spacing, origin: full.origin, data }
    }
}

// ===========================================================================
// Shrink filter
// ===========================================================================

/// Subsample by integer shrink factors (one per axis).
/// Output pixel `p_out[d]` reads input pixel `floor(p_out[d] * factor[d])`.
/// Analog to `itk::ShrinkImageFilter`.
pub struct ShrinkImageFilter<S> {
    pub source: S,
    pub shrink_factors: Vec<usize>,
}

impl<S> ShrinkImageFilter<S> {
    pub fn new(source: S, shrink_factors: Vec<usize>) -> Self {
        Self { source, shrink_factors }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for ShrinkImageFilter<S>
where
    P: Pixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> {
        let src = self.source.largest_region();
        let mut size = src.size.0;
        for d in 0..D {
            let f = *self.shrink_factors.get(d).unwrap_or(&1);
            size[d] = (size[d] + f - 1) / f;
        }
        Region { index: src.index, size: Size(size) }
    }
    fn spacing(&self) -> [f64; D] {
        let mut sp = self.source.spacing();
        for d in 0..D {
            sp[d] *= *self.shrink_factors.get(d).unwrap_or(&1) as f64;
        }
        sp
    }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        // Need to generate the corresponding input region
        let src_full = self.source.generate_region(self.source.largest_region());

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let factors = &self.shrink_factors;
        let src_region = src_full.region;
        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let mut src_idx = idx;
                for d in 0..D {
                    let f = *factors.get(d).unwrap_or(&1) as i64;
                    let src_pos = (idx.0[d] - src_region.index.0[d]) * f
                        + src_region.index.0[d];
                    let hi = src_region.index.0[d] + src_region.size.0[d] as i64 - 1;
                    src_idx.0[d] = src_pos.clamp(src_region.index.0[d], hi);
                }
                src_full.get_pixel(src_idx)
            })
            .collect();
        Image { region: requested, spacing: self.spacing(), origin: self.origin(), data }
    }
}

// ===========================================================================
// Expand filter
// ===========================================================================

/// Upsample by integer expand factors, with linear interpolation fill.
/// Analog to `itk::ExpandImageFilter`.
pub struct ExpandImageFilter<S> {
    pub source: S,
    pub expand_factors: Vec<usize>,
}

impl<S> ExpandImageFilter<S> {
    pub fn new(source: S, expand_factors: Vec<usize>) -> Self {
        Self { source, expand_factors }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for ExpandImageFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> {
        let src = self.source.largest_region();
        let mut size = src.size.0;
        for d in 0..D {
            size[d] *= *self.expand_factors.get(d).unwrap_or(&1);
        }
        Region { index: src.index, size: Size(size) }
    }
    fn spacing(&self) -> [f64; D] {
        let mut sp = self.source.spacing();
        for d in 0..D {
            sp[d] /= *self.expand_factors.get(d).unwrap_or(&1) as f64;
        }
        sp
    }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let src_full = self.source.generate_region(self.source.largest_region());
        let interp = LinearInterpolator;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let factors = &self.expand_factors;
        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let mut cont = [0.0f64; D];
                for d in 0..D {
                    let f = *factors.get(d).unwrap_or(&1) as f64;
                    cont[d] = src_full.region.index.0[d] as f64
                        + (idx.0[d] - src_full.region.index.0[d]) as f64 / f;
                }
                interp.evaluate(&src_full, cont)
            })
            .collect();
        Image { region: requested, spacing: self.spacing(), origin: self.origin(), data }
    }
}

// ===========================================================================
// Region of interest / Crop
// ===========================================================================

/// Extract a sub-region of the image.
/// Analog to `itk::RegionOfInterestImageFilter`.
pub struct RegionOfInterestFilterD<S, const D: usize> {
    pub source: S,
    pub region: Region<D>,
}

impl<S, const D: usize> RegionOfInterestFilterD<S, D> {
    pub fn new(source: S, region: Region<D>) -> Self { Self { source, region } }
}

impl<P, S, const D: usize> ImageSource<P, D> for RegionOfInterestFilterD<S, D>
where
    P: Pixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.region }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        // Just ask the source for the requested sub-region
        self.source.generate_region(requested)
    }
}

/// Crop filter: shrink the image by removing pixels from each border.
/// Analog to `itk::CropImageFilter`.
pub struct CropImageFilter<S, const D: usize> {
    pub source: S,
    /// Pixels to remove from the lower end of each axis.
    pub lower_crop: [usize; D],
    /// Pixels to remove from the upper end of each axis.
    pub upper_crop: [usize; D],
}

impl<S, const D: usize> CropImageFilter<S, D> {
    pub fn new(source: S, lower: [usize; D], upper: [usize; D]) -> Self {
        Self { source, lower_crop: lower, upper_crop: upper }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for CropImageFilter<S, D>
where
    P: Pixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> {
        let src = self.source.largest_region();
        let mut index = src.index.0;
        let mut size = src.size.0;
        for d in 0..D {
            index[d] += self.lower_crop[d] as i64;
            size[d] = size[d].saturating_sub(self.lower_crop[d] + self.upper_crop[d]);
        }
        Region::new(index, size)
    }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        self.source.generate_region(requested)
    }
}

// ===========================================================================
// Pad filters
// ===========================================================================

/// Pad image with a constant value on all sides.
/// Analog to `itk::ConstantPadImageFilter`.
pub struct ConstantPadFilter<S, P, const D: usize> {
    pub source: S,
    pub pad_lower: [usize; D],
    pub pad_upper: [usize; D],
    pub constant: P,
}

impl<S, P: Copy, const D: usize> ConstantPadFilter<S, P, D> {
    pub fn new(source: S, pad: [usize; D], constant: P) -> Self {
        Self { source, pad_lower: pad, pad_upper: pad, constant }
    }
    pub fn with_asymmetric_padding(source: S, lower: [usize; D], upper: [usize; D], constant: P) -> Self {
        Self { source, pad_lower: lower, pad_upper: upper, constant }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for ConstantPadFilter<S, P, D>
where
    P: Pixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> {
        let src = self.source.largest_region();
        let mut index = src.index.0;
        let mut size = src.size.0;
        for d in 0..D {
            index[d] -= self.pad_lower[d] as i64;
            size[d] += self.pad_lower[d] + self.pad_upper[d];
        }
        Region::new(index, size)
    }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let src_region = self.source.largest_region();
        let src = self.source.generate_region(src_region);
        let constant = self.constant;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                if src_region.contains(&idx) { src.get_pixel(idx) } else { constant }
            })
            .collect();
        Image { region: requested, spacing: src.spacing, origin: src.origin, data }
    }
}

/// Mirror (reflect) padding.
/// Analog to `itk::MirrorPadImageFilter`.
pub struct MirrorPadFilter<S, const D: usize> {
    pub source: S,
    pub pad_lower: [usize; D],
    pub pad_upper: [usize; D],
}

impl<S, const D: usize> MirrorPadFilter<S, D> {
    pub fn new(source: S, pad: [usize; D]) -> Self {
        Self { source, pad_lower: pad, pad_upper: pad }
    }
}

fn mirror_index(pos: i64, lo: i64, hi: i64) -> i64 {
    if hi <= lo { return lo; }
    let len = hi - lo;
    let mut p = (pos - lo).rem_euclid(2 * len);
    if p >= len { p = 2 * len - 1 - p; }
    p + lo
}

impl<P, S, const D: usize> ImageSource<P, D> for MirrorPadFilter<S, D>
where
    P: Pixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> {
        let src = self.source.largest_region();
        let mut index = src.index.0;
        let mut size = src.size.0;
        for d in 0..D {
            index[d] -= self.pad_lower[d] as i64;
            size[d] += self.pad_lower[d] + self.pad_upper[d];
        }
        Region::new(index, size)
    }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let src_region = self.source.largest_region();
        let src = self.source.generate_region(src_region);

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let mut mapped = idx;
                for d in 0..D {
                    let lo = src_region.index.0[d];
                    let hi = lo + src_region.size.0[d] as i64;
                    mapped.0[d] = mirror_index(idx.0[d], lo, hi);
                }
                src.get_pixel(mapped)
            })
            .collect();
        Image { region: requested, spacing: src.spacing, origin: src.origin, data }
    }
}

/// Wrap-around (periodic) padding.
/// Analog to `itk::WrapPadImageFilter`.
pub struct WrapPadFilter<S, const D: usize> {
    pub source: S,
    pub pad_lower: [usize; D],
    pub pad_upper: [usize; D],
}

impl<S, const D: usize> WrapPadFilter<S, D> {
    pub fn new(source: S, pad: [usize; D]) -> Self {
        Self { source, pad_lower: pad, pad_upper: pad }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for WrapPadFilter<S, D>
where
    P: Pixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> {
        let src = self.source.largest_region();
        let mut index = src.index.0;
        let mut size = src.size.0;
        for d in 0..D {
            index[d] -= self.pad_lower[d] as i64;
            size[d] += self.pad_lower[d] + self.pad_upper[d];
        }
        Region::new(index, size)
    }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let src_region = self.source.largest_region();
        let src = self.source.generate_region(src_region);

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let mut mapped = idx;
                for d in 0..D {
                    let lo = src_region.index.0[d];
                    let len = src_region.size.0[d] as i64;
                    if len == 0 { continue; }
                    mapped.0[d] = (idx.0[d] - lo).rem_euclid(len) + lo;
                }
                src.get_pixel(mapped)
            })
            .collect();
        Image { region: requested, spacing: src.spacing, origin: src.origin, data }
    }
}

/// Zero-flux Neumann padding (replicate border pixel).
/// Analog to `itk::ZeroFluxNeumannPadImageFilter`.
pub struct ZeroFluxNeumannPadFilter<S, const D: usize> {
    pub source: S,
    pub pad_lower: [usize; D],
    pub pad_upper: [usize; D],
}

impl<S, const D: usize> ZeroFluxNeumannPadFilter<S, D> {
    pub fn new(source: S, pad: [usize; D]) -> Self {
        Self { source, pad_lower: pad, pad_upper: pad }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for ZeroFluxNeumannPadFilter<S, D>
where
    P: Pixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> {
        let src = self.source.largest_region();
        let mut index = src.index.0;
        let mut size = src.size.0;
        for d in 0..D {
            index[d] -= self.pad_lower[d] as i64;
            size[d] += self.pad_lower[d] + self.pad_upper[d];
        }
        Region::new(index, size)
    }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let src_region = self.source.largest_region();
        let src = self.source.generate_region(src_region);

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let mut mapped = idx;
                for d in 0..D {
                    let lo = src_region.index.0[d];
                    let hi = lo + src_region.size.0[d] as i64 - 1;
                    mapped.0[d] = idx.0[d].clamp(lo, hi);
                }
                src.get_pixel(mapped)
            })
            .collect();
        Image { region: requested, spacing: src.spacing, origin: src.origin, data }
    }
}

// ===========================================================================
// Permute axes filter
// ===========================================================================

/// Reorder the axes of an image.
/// `order[d]` gives the source axis for output axis `d`.
/// Analog to `itk::PermuteAxesImageFilter`.
pub struct PermuteAxesFilter<S, const D: usize> {
    pub source: S,
    pub order: [usize; D],
}

impl<S, const D: usize> PermuteAxesFilter<S, D> {
    pub fn new(source: S, order: [usize; D]) -> Self { Self { source, order } }
}

impl<P, S, const D: usize> ImageSource<P, D> for PermuteAxesFilter<S, D>
where
    P: Pixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> {
        let src = self.source.largest_region();
        let mut index = [0i64; D];
        let mut size = [0usize; D];
        for d in 0..D {
            index[d] = src.index.0[self.order[d]];
            size[d] = src.size.0[self.order[d]];
        }
        Region::new(index, size)
    }
    fn spacing(&self) -> [f64; D] {
        let sp = self.source.spacing();
        let mut out = sp;
        for d in 0..D { out[d] = sp[self.order[d]]; }
        out
    }
    fn origin(&self) -> [f64; D] {
        let orig = self.source.origin();
        let mut out = orig;
        for d in 0..D { out[d] = orig[self.order[d]]; }
        out
    }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let src_full = self.source.generate_region(self.source.largest_region());

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let order = self.order;
        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let mut src_idx = idx;
                for d in 0..D { src_idx.0[order[d]] = idx.0[d]; }
                src_full.get_pixel(src_idx)
            })
            .collect();
        Image { region: requested, spacing: self.spacing(), origin: self.origin(), data }
    }
}

// ===========================================================================
// Paste filter
// ===========================================================================

/// Paste `source2` into `source1` at a given destination index.
/// Analog to `itk::PasteImageFilter`.
pub struct PasteFilter<S1, S2, const D: usize> {
    pub destination: S1,
    pub source_patch: S2,
    /// Top-left corner in the destination image where the patch is pasted.
    pub destination_index: Index<D>,
}

impl<S1, S2, const D: usize> PasteFilter<S1, S2, D> {
    pub fn new(destination: S1, source_patch: S2, destination_index: Index<D>) -> Self {
        Self { destination, source_patch, destination_index }
    }
}

impl<P, S1, S2, const D: usize> ImageSource<P, D> for PasteFilter<S1, S2, D>
where
    P: Pixel,
    S1: ImageSource<P, D> + Sync,
    S2: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.destination.largest_region() }
    fn spacing(&self) -> [f64; D] { self.destination.spacing() }
    fn origin(&self) -> [f64; D] { self.destination.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.destination.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let dst = self.destination.generate_region(requested);
        let patch_src_region = self.source_patch.largest_region();
        let patch = self.source_patch.generate_region(patch_src_region);

        // Compute the destination region for the patch
        let mut paste_region = patch_src_region;
        for d in 0..D {
            paste_region.index.0[d] = self.destination_index.0[d];
        }

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                if paste_region.contains(&idx) {
                    // Translate back to patch coordinates
                    let mut patch_idx = idx;
                    for d in 0..D {
                        patch_idx.0[d] = patch_src_region.index.0[d]
                            + (idx.0[d] - self.destination_index.0[d]);
                    }
                    if patch_src_region.contains(&patch_idx) {
                        return patch.get_pixel(patch_idx);
                    }
                }
                dst.get_pixel(idx)
            })
            .collect();
        Image { region: requested, spacing: dst.spacing, origin: dst.origin, data }
    }
}

// ===========================================================================
// Cyclic shift filter
// ===========================================================================

/// Cyclically shift image pixels by a given offset.
/// Analog to `itk::CyclicShiftImageFilter`.
pub struct CyclicShiftFilter<S, const D: usize> {
    pub source: S,
    pub shift: [i64; D],
}

impl<S, const D: usize> CyclicShiftFilter<S, D> {
    pub fn new(source: S, shift: [i64; D]) -> Self { Self { source, shift } }
}

impl<P, S, const D: usize> ImageSource<P, D> for CyclicShiftFilter<S, D>
where
    P: Pixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let full = self.source.generate_region(self.source.largest_region());
        let region = full.region;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let shift = self.shift;
        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let mut src_idx = idx;
                for d in 0..D {
                    let lo = region.index.0[d];
                    let len = region.size.0[d] as i64;
                    if len == 0 { continue; }
                    src_idx.0[d] = (idx.0[d] - lo - shift[d]).rem_euclid(len) + lo;
                }
                full.get_pixel(src_idx)
            })
            .collect();
        Image { region: requested, spacing: full.spacing, origin: full.origin, data }
    }
}

// ===========================================================================
// Change information filter
// ===========================================================================

/// Override spacing, origin, and/or direction of an image without changing pixels.
/// Analog to `itk::ChangeInformationImageFilter`.
pub struct ChangeInformationFilter<S, const D: usize> {
    pub source: S,
    pub new_spacing: Option<[f64; D]>,
    pub new_origin: Option<[f64; D]>,
}

impl<S, const D: usize> ChangeInformationFilter<S, D> {
    pub fn new(source: S) -> Self {
        Self { source, new_spacing: None, new_origin: None }
    }
    pub fn with_spacing(mut self, s: [f64; D]) -> Self { self.new_spacing = Some(s); self }
    pub fn with_origin(mut self, o: [f64; D]) -> Self { self.new_origin = Some(o); self }
}

impl<P, S, const D: usize> ImageSource<P, D> for ChangeInformationFilter<S, D>
where
    P: Pixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.new_spacing.unwrap_or_else(|| self.source.spacing()) }
    fn origin(&self) -> [f64; D] { self.new_origin.unwrap_or_else(|| self.source.origin()) }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let mut img = self.source.generate_region(requested);
        if let Some(sp) = self.new_spacing { img.spacing = sp; }
        if let Some(or) = self.new_origin { img.origin = or; }
        img
    }
}

// ===========================================================================
// Bin shrink filter
// ===========================================================================

/// Shrink by integer factors while averaging contributing input pixels.
/// Analog to `itk::BinShrinkImageFilter`.
pub struct BinShrinkImageFilter<S> {
    pub source: S,
    pub shrink_factors: Vec<usize>,
}

impl<S> BinShrinkImageFilter<S> {
    pub fn new(source: S, shrink_factors: Vec<usize>) -> Self {
        Self { source, shrink_factors }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for BinShrinkImageFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> {
        let src = self.source.largest_region();
        let mut size = src.size.0;
        for d in 0..D {
            let f = *self.shrink_factors.get(d).unwrap_or(&1);
            size[d] = (size[d] + f - 1) / f;
        }
        Region { index: src.index, size: Size(size) }
    }
    fn spacing(&self) -> [f64; D] {
        let mut sp = self.source.spacing();
        for d in 0..D {
            sp[d] *= *self.shrink_factors.get(d).unwrap_or(&1) as f64;
        }
        sp
    }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let src_full = self.source.generate_region(self.source.largest_region());
        let src_region = src_full.region;
        let factors = &self.shrink_factors;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                // Average all source pixels in the bin
                let mut sum = P::zero();
                let mut count = 0.0f64;
                // Enumerate all offsets within the bin
                let n_bin: usize = factors[..D.min(factors.len())].iter().map(|&f| f).product();
                let _ = n_bin; // used below
                // Iterate product space
                let total: usize = (0..D).map(|d| *factors.get(d).unwrap_or(&1)).product();
                for flat in 0..total {
                    let mut src_idx = src_region.index;
                    let mut tmp = flat;
                    let mut valid = true;
                    for d in 0..D {
                        let f = *factors.get(d).unwrap_or(&1);
                        let off = (tmp % f) as i64;
                        tmp /= f;
                        let src_base = src_region.index.0[d]
                            + (idx.0[d] - src_region.index.0[d]) * f as i64;
                        let pos = src_base + off;
                        let hi = src_region.index.0[d] + src_region.size.0[d] as i64 - 1;
                        if pos < src_region.index.0[d] || pos > hi {
                            valid = false; break;
                        }
                        src_idx.0[d] = pos;
                    }
                    if valid {
                        sum = sum + src_full.get_pixel(src_idx);
                        count += 1.0;
                    }
                }
                if count > 0.0 { sum.scale(1.0 / count) } else { P::zero() }
            })
            .collect();
        Image { region: requested, spacing: self.spacing(), origin: self.origin(), data }
    }
}

// ===========================================================================
// TileImageFilter
// ===========================================================================

/// Tile multiple images into a single larger image.
/// Analog to `itk::TileImageFilter`.
///
/// `layout` specifies how many tiles to place on each axis.
/// Images are placed left-to-right, bottom-to-top (axis-0 fastest).
/// All input images must have the same size.
pub struct TileImageFilter<P, const D: usize> {
    pub sources: Vec<Box<dyn crate::source::ImageSource<P, D> + Send + Sync>>,
    /// Number of tiles along each axis. Product must be >= sources.len().
    pub layout: [usize; D],
    pub default_value: P,
}

impl<P: Pixel, const D: usize> TileImageFilter<P, D> {
    pub fn new(
        sources: Vec<Box<dyn crate::source::ImageSource<P, D> + Send + Sync>>,
        layout: [usize; D],
        default_value: P,
    ) -> Self {
        Self { sources, layout, default_value }
    }

    fn tile_size(&self) -> [usize; D] {
        if self.sources.is_empty() {
            return [0usize; D];
        }
        self.sources[0].largest_region().size.0
    }
}

impl<P, const D: usize> ImageSource<P, D> for TileImageFilter<P, D>
where
    P: Pixel,
{
    fn largest_region(&self) -> Region<D> {
        let ts = self.tile_size();
        let mut size = [0usize; D];
        for d in 0..D {
            size[d] = self.layout[d] * ts[d];
        }
        Region { index: Index([0i64; D]), size: Size(size) }
    }
    fn spacing(&self) -> [f64; D] {
        if self.sources.is_empty() { return [1.0; D]; }
        self.sources[0].spacing()
    }
    fn origin(&self) -> [f64; D] {
        if self.sources.is_empty() { return [0.0; D]; }
        self.sources[0].origin()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let ts = self.tile_size();
        // Pre-generate all tiles
        let tiles: Vec<Image<P, D>> = self.sources.iter()
            .map(|s| s.generate_region(s.largest_region()))
            .collect();

        let mut out = Image::allocate(requested, self.spacing(), self.origin(), self.default_value.clone());

        iter_region(&requested, |idx| {
            // Which tile does this pixel belong to?
            let mut tile_coord = [0usize; D];
            let mut within = idx.0;
            for d in 0..D {
                tile_coord[d] = (idx.0[d] as usize) / ts[d];
                within[d] = idx.0[d] % ts[d] as i64;
            }
            // Flat tile index (axis-0 fastest)
            let mut tile_flat = 0usize;
            let mut stride = 1usize;
            for d in 0..D {
                tile_flat += tile_coord[d] * stride;
                stride *= self.layout[d];
            }
            if tile_flat < tiles.len() {
                let tile = &tiles[tile_flat];
                let src_idx = Index(within);
                if tile.region.contains(&src_idx) {
                    out.set_pixel(idx, tile.get_pixel(src_idx));
                }
            }
        });

        out
    }
}

// ===========================================================================
// CheckerBoardImageFilter
// ===========================================================================

/// Produce a checkerboard pattern from two input images.
/// Analog to `itk::CheckerBoardImageFilter`.
///
/// `pattern` specifies the number of checkerboard cells along each axis.
/// At each position, the cell index parity determines which source is used.
pub struct CheckerBoardFilter<S1, S2, const D: usize> {
    pub source1: S1,
    pub source2: S2,
    /// Number of checker tiles along each axis.
    pub pattern: [usize; D],
}

impl<S1, S2, const D: usize> CheckerBoardFilter<S1, S2, D> {
    pub fn new(source1: S1, source2: S2, pattern: [usize; D]) -> Self {
        Self { source1, source2, pattern }
    }
}

impl<P, S1, S2, const D: usize> ImageSource<P, D> for CheckerBoardFilter<S1, S2, D>
where
    P: Pixel,
    S1: ImageSource<P, D> + Sync,
    S2: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source1.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source1.spacing() }
    fn origin(&self) -> [f64; D] { self.source1.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let img1 = self.source1.generate_region(requested);
        let img2 = self.source2.generate_region(requested);
        let region = self.source1.largest_region();
        let size = region.size.0;

        let mut out = img1.clone();
        iter_region(&requested, |idx| {
            let mut parity = 0u32;
            for d in 0..D {
                let cells = self.pattern[d].max(1);
                let cell_size = size[d].max(1) / cells;
                let cell = if cell_size > 0 { (idx.0[d] as usize) / cell_size } else { 0 };
                parity ^= cell as u32;
            }
            if parity % 2 == 1 {
                out.set_pixel(idx, img2.get_pixel(idx));
            }
        });
        out
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    fn ramp_2d(w: usize, h: usize) -> Image<f32, 2> {
        let mut img = Image::<f32,2>::allocate(Region::new([0,0],[w,h]),[1.0,1.0],[0.0,0.0],0.0);
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                img.set_pixel(Index([x,y]), (x + y*10) as f32);
            }
        }
        img
    }

    #[test]
    fn flip_x_axis() {
        let img = ramp_2d(4, 4);
        let f = FlipImageFilter::new(img, vec![true, false]);
        let out = f.generate_region(f.largest_region());
        // pixel (0,0) should equal original (3,0)
        let v00 = out.get_pixel(Index([0i64,0]));
        assert!((v00 - 3.0).abs() < 1e-6, "expected 3 got {v00}");
    }

    #[test]
    fn shrink_2x() {
        let img = ramp_2d(4, 4);
        let f = ShrinkImageFilter::new(img, vec![2, 2]);
        let out = f.generate_region(f.largest_region());
        assert_eq!(out.region.size.0, [2, 2]);
    }

    #[test]
    fn constant_pad_increases_size() {
        let img = Image::<f32,2>::allocate(Region::new([0,0],[4,4]),[1.0,1.0],[0.0,0.0],1.0);
        let f = ConstantPadFilter::new(img, [2,2], 0.0f32);
        let out = f.generate_region(f.largest_region());
        assert_eq!(out.region.size.0, [8, 8]);
        // padded region starts at [-2,-2]; corners are constant 0
        assert!((out.get_pixel(Index([-2i64,-2])) - 0.0).abs() < 1e-6);
        // pixels inside original [0,4)×[0,4) should be 1
        assert!((out.get_pixel(Index([2i64,2])) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cyclic_shift_wraps() {
        let img = ramp_2d(4, 4);
        let original = img.clone();
        let f = CyclicShiftFilter::new(img, [4i64, 0]);
        let out = f.generate_region(f.largest_region());
        // Shift by full width = identity
        for y in 0..4i64 {
            for x in 0..4i64 {
                let a = out.get_pixel(Index([x,y]));
                let b = original.get_pixel(Index([x,y]));
                assert!((a-b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn crop_reduces_size() {
        let img = ramp_2d(6, 6);
        let f = CropImageFilter::new(img, [1usize,1], [1usize,1]);
        assert_eq!(f.largest_region().size.0, [4, 4]);
    }

    #[test]
    fn region_of_interest() {
        let img = ramp_2d(8, 8);
        let roi = Region::new([2,2],[4,4]);
        let f = RegionOfInterestFilterD::new(img, roi);
        let out = f.generate_region(f.largest_region());
        assert_eq!(out.region.size.0, [4, 4]);
        // pixel (2,2) should have value 2 + 2*10 = 22
        assert!((out.get_pixel(Index([2i64,2])) - 22.0).abs() < 1e-6);
    }

    #[test]
    fn mirror_pad_reflects() {
        let img = Image::<f32,1>::allocate(Region::new([0],[3]),[1.0],[0.0],0.0f32);
        let mut img2 = img;
        img2.set_pixel(Index([0i64]),1.0);
        img2.set_pixel(Index([1i64]),2.0);
        img2.set_pixel(Index([2i64]),3.0);
        let f = MirrorPadFilter::new(img2, [2usize]);
        let out = f.generate_region(f.largest_region());
        // padded region: [-2,-1,0,1,2,4,5] → mirror of [1,2,3]
        // idx -2 → mirror(−2, lo=0, hi=3) = mirror with len=3
        // p = (-2 - 0).rem_euclid(6) = 4 → 4 >= 3 → 5 - 4 = 1, +0 = 1 → pixel 1 = value 2
        assert!((out.get_pixel(Index([-2i64])) - 2.0).abs() < 1e-6, "got {}", out.get_pixel(Index([-2])));
    }
}

