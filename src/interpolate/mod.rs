//! Image interpolation functions. Analog to ITK's ImageFunction hierarchy.
//!
//! All interpolators implement the [`Interpolate`] trait, which evaluates an
//! image at a continuous index (pixel coordinates, not physical coordinates).
//!
//! Interpolators that require arithmetic (Linear, BSpline, Gaussian, Sinc)
//! are generic over [`NumericPixel`] types. [`NearestNeighborInterpolator`]
//! and [`LabelGaussianInterpolator`] work with any [`Pixel`].

pub mod nearest;
pub mod linear;
pub mod bspline;
pub mod gaussian;
pub mod windowed_sinc;

pub use nearest::NearestNeighborInterpolator;
pub use linear::LinearInterpolator;
pub use bspline::BSplineInterpolator;
pub use gaussian::{GaussianInterpolator, LabelGaussianInterpolator};
pub use windowed_sinc::{WindowedSincInterpolator, WindowFunction};

use crate::image::{Image, Index};
use crate::pixel::Pixel;

/// Core interpolation trait. Analog to `itk::InterpolateImageFunction`.
///
/// Evaluates an image at a **continuous index** (real-valued pixel coordinates).
/// The index (0.0, 0.0, ...) corresponds to the centre of the first pixel.
/// Boundary handling is left to each implementation.
pub trait Interpolate<P: Pixel, const D: usize> {
    fn evaluate(&self, image: &Image<P, D>, index: [f64; D]) -> P;
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Clamp an integer index to the buffered region of an image.
pub(crate) fn clamp_index<const D: usize>(
    image: &Image<impl Pixel, D>,
    mut idx: [i64; D],
) -> Index<D> {
    for d in 0..D {
        let lo = image.region.index.0[d];
        let hi = lo + image.region.size.0[d] as i64 - 1;
        idx[d] = idx[d].clamp(lo, hi);
    }
    Index(idx)
}
