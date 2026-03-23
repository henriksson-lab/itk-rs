//! Direct-space convolution and normalized-correlation filters.
//! Analog to `itk::ConvolutionImageFilter` and
//! `itk::NormalizedCorrelationImageFilter`.

use rayon::prelude::*;

use crate::image::{Image, Index, Region, iter_region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

// ---------------------------------------------------------------------------
// ConvolutionImageFilter
// ---------------------------------------------------------------------------

/// Direct-space convolution with a user-supplied kernel image.
/// Analog to `itk::ConvolutionImageFilter`.
///
/// The kernel origin defines its centre: for a kernel with `region.index =
/// [-r, -r]` and `size = [2r+1, 2r+1]` the centre is index `[0, 0]`.
/// If the kernel region is `[0, …, 0]`-indexed, the centre is taken as the
/// geometric centre of the kernel.
///
/// Out-of-bounds input pixels use zero-flux Neumann boundary.
pub struct ConvolutionFilter<S, const D: usize> {
    pub source: S,
    /// Kernel weights. Any numeric region is accepted; values are normalised
    /// unless `normalize` is `false`.
    pub kernel: Image<f64, D>,
    /// If true (default) the kernel weights are normalised to sum to 1.
    pub normalize: bool,
}

impl<S, const D: usize> ConvolutionFilter<S, D> {
    pub fn new(source: S, kernel: Image<f64, D>) -> Self {
        Self { source, kernel, normalize: true }
    }
    pub fn unnormalized(source: S, kernel: Image<f64, D>) -> Self {
        Self { source, kernel, normalize: false }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for ConvolutionFilter<S, D>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        // Radius is the max absolute index extent of the kernel
        let kr = self.kernel.region;
        let mut radii = [0usize; D];
        for d in 0..D {
            let lo = (-kr.index.0[d]).max(0) as usize;
            let hi = (kr.index.0[d] + kr.size.0[d] as i64 - 1).max(0) as usize;
            radii[d] = lo.max(hi);
        }
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let bounds = self.source.largest_region();
        let input_region = self.input_region_for_output(&requested);
        let input = self.source.generate_region(input_region);

        // Pre-compute normalisation factor
        let norm: f64 = if self.normalize {
            let s: f64 = self.kernel.data.iter().map(|v| v.abs()).sum();
            if s == 0.0 { 1.0 } else { s }
        } else {
            1.0
        };

        // Collect kernel (offset, weight) pairs
        let mut koffsets: Vec<([i64; D], f64)> = Vec::with_capacity(self.kernel.region.linear_len());
        iter_region(&self.kernel.region, |kidx| {
            let w = self.kernel.get_pixel(kidx) / norm;
            koffsets.push((kidx.0, w));
        });

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices
            .par_iter()
            .map(|&out_idx| {
                let mut acc = P::zero();
                for (koff, w) in &koffsets {
                    let mut s = out_idx.0;
                    for d in 0..D {
                        s[d] = (s[d] + koff[d])
                            .max(bounds.index.0[d])
                            .min(bounds.index.0[d] + bounds.size.0[d] as i64 - 1);
                    }
                    acc = acc + input.get_pixel(Index(s)).scale(*w);
                }
                acc
            })
            .collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// NormalizedCorrelationImageFilter
// ---------------------------------------------------------------------------

/// Normalized cross-correlation with a template (kernel) image.
/// Analog to `itk::NormalizedCorrelationImageFilter`.
///
/// For each position x the output is:
///   NCC(x) = Σ_k (I(x+k) - μ_I) * (T(k) - μ_T)
///            / sqrt( Σ_k (I(x+k) - μ_I)² * Σ_k (T(k) - μ_T)² )
///
/// Result is in [-1, 1].  If either patch or template is constant the output
/// is 0.
///
/// The template must fit inside the image at every requested location; pixels
/// outside the image use Neumann boundary.
pub struct NormalizedCorrelationFilter<S, P, const D: usize> {
    pub source: S,
    pub template: Image<f64, D>,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P, const D: usize> NormalizedCorrelationFilter<S, P, D> {
    pub fn new(source: S, template: Image<f64, D>) -> Self {
        Self { source, template, _phantom: std::marker::PhantomData }
    }
}

impl<P, S, const D: usize> ImageSource<f32, D> for NormalizedCorrelationFilter<S, P, D>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let kr = self.template.region;
        let mut radii = [0usize; D];
        for d in 0..D {
            let lo = (-kr.index.0[d]).max(0) as usize;
            let hi = (kr.index.0[d] + kr.size.0[d] as i64 - 1).max(0) as usize;
            radii[d] = lo.max(hi);
        }
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<f32, D> {
        let bounds = self.source.largest_region();
        let input_region = self.input_region_for_output(&requested);
        let input = self.source.generate_region(input_region);

        // Pre-compute template mean and std
        let t_n = self.template.data.len() as f64;
        let t_mean = self.template.data.iter().sum::<f64>() / t_n;
        let t_var: f64 = self.template.data.iter().map(|&v| (v - t_mean) * (v - t_mean)).sum::<f64>() / t_n;
        let t_std = t_var.sqrt();

        // Kernel offsets
        let mut koffsets: Vec<([i64; D], f64)> = Vec::with_capacity(self.template.region.linear_len());
        iter_region(&self.template.region, |kidx| {
            let tv = self.template.get_pixel(kidx) - t_mean;
            koffsets.push((kidx.0, tv));
        });

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<f32> = out_indices
            .par_iter()
            .map(|&out_idx| {
                // Collect local patch values
                let patch: Vec<f64> = koffsets
                    .iter()
                    .map(|(koff, _)| {
                        let mut s = out_idx.0;
                        for d in 0..D {
                            s[d] = (s[d] + koff[d])
                                .max(bounds.index.0[d])
                                .min(bounds.index.0[d] + bounds.size.0[d] as i64 - 1);
                        }
                        input.get_pixel(Index(s)).to_f64()
                    })
                    .collect();
                let p_n = patch.len() as f64;
                let p_mean = patch.iter().sum::<f64>() / p_n;
                let p_var: f64 = patch.iter().map(|&v| (v - p_mean) * (v - p_mean)).sum::<f64>() / p_n;
                let p_std = p_var.sqrt();

                if p_std < 1e-12 || t_std < 1e-12 {
                    return 0.0f32;
                }

                let ncc: f64 = koffsets
                    .iter()
                    .zip(patch.iter())
                    .map(|((_, tv), &pv)| (pv - p_mean) * tv)
                    .sum::<f64>()
                    / (p_n * p_std * t_std);

                ncc.clamp(-1.0, 1.0) as f32
            })
            .collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Region};

    fn constant_img(val: f64, size: usize) -> Image<f64, 1> {
        Image::allocate(Region::new([0], [size]), [1.0], [0.0], val)
    }

    #[test]
    fn convolution_box_is_mean() {
        // convolving with a box kernel should equal the mean
        let mut img = Image::<f32, 1>::allocate(Region::new([0], [10]), [1.0], [0.0], 0.0f32);
        for i in 0..10i64 {
            img.set_pixel(Index([i]), i as f32);
        }
        let radius = 2i64;
        let kr = Region::new([-radius], [(2 * radius + 1) as usize]);
        let kernel = Image::allocate(kr, [1.0], [0.0], 1.0f64);
        let f = ConvolutionFilter::new(img, kernel);
        let out = f.generate_region(f.largest_region());
        // interior pixel 5: mean of [3,4,5,6,7] = 5.0
        let v = out.get_pixel(Index([5]));
        assert!((v - 5.0).abs() < 1e-4, "expected 5.0 got {v}");
    }

    #[test]
    fn ncc_identical_patch_is_one() {
        // NCC of an image with itself = 1
        let mut img = Image::<f32, 1>::allocate(Region::new([0], [10]), [1.0], [0.0], 0.0f32);
        for i in 0..10i64 { img.set_pixel(Index([i]), i as f32); }
        // template: ramp 0..5 centred at 0
        let mut tmpl = Image::<f64, 1>::allocate(Region::new([-2], [5]), [1.0], [0.0], 0.0f64);
        for i in -2i64..=2 { tmpl.set_pixel(Index([i]), i as f64); }
        let f = NormalizedCorrelationFilter::<_, f32, 1>::new(img, tmpl);
        let out = f.generate_region(f.largest_region());
        let v = out.get_pixel(Index([5]));
        assert!((v - 1.0).abs() < 1e-4, "expected 1.0 got {v}");
    }

    #[test]
    fn ncc_constant_is_zero() {
        let img = Image::<f32, 1>::allocate(Region::new([0], [10]), [1.0], [0.0], 3.0f32);
        let tmpl = constant_img(1.0, 3);
        // reindex tmpl to centre-zero
        let tmpl = Image { region: Region::new([-1], [3]), ..tmpl };
        let f = NormalizedCorrelationFilter::<_, f32, 1>::new(img, tmpl);
        let out = f.generate_region(f.largest_region());
        let v = out.get_pixel(Index([5]));
        assert!(v.abs() < 1e-4, "expected 0.0 got {v}");
    }
}
