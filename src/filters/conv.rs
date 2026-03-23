//! Shared separable-convolution utilities used by multiple smoothing filters.

use rayon::prelude::*;

use crate::image::{Image, Index, Region, iter_region};
use crate::pixel::NumericPixel;

/// Apply a 1-D convolution kernel along `axis` of `src`, producing `requested`.
/// Out-of-bounds samples use zero-flux Neumann (clamp-to-edge) boundary.
/// Parallelised over output pixels with rayon.
pub(crate) fn convolve_axis<P, const D: usize>(
    src: &Image<P, D>,
    axis: usize,
    kernel: &[f64],
    requested: Region<D>,
) -> Image<P, D>
where
    P: NumericPixel,
{
    let radius = kernel.len() / 2;
    let mut indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
    iter_region(&requested, |idx| indices.push(idx));

    let data: Vec<P> = indices
        .par_iter()
        .map(|&out_idx| {
            let mut acc = P::zero();
            for (k_i, &w) in kernel.iter().enumerate() {
                let offset = k_i as i64 - radius as i64;
                let mut s = out_idx.0;
                s[axis] = (s[axis] + offset)
                    .max(src.region.index.0[axis])
                    .min(src.region.index.0[axis] + src.region.size.0[axis] as i64 - 1);
                acc = acc + src.get_pixel(Index(s)).scale(w);
            }
            acc
        })
        .collect();

    Image { region: requested, spacing: src.spacing, origin: src.origin, data }
}

