//! Dense displacement field transform.
//! Analog to `itk::DisplacementFieldTransform`.
//!
//! Each voxel stores a displacement vector `d(x)` such that
//! `y = x + d(x)`.
//!
//! The field is sampled with linear interpolation. The inverse is not computed
//! analytically; `inverse_transform_point` returns `None` unless an optional
//! inverse field is provided.

use crate::image::Image;
use crate::pixel::VecPixel;
use crate::interpolate::{Interpolate, linear::LinearInterpolator};
use super::Transform;

/// Displacement-field transform backed by an `Image<VecPixel<f64, D>, D>`.
pub struct DisplacementFieldTransform<const D: usize> {
    field: Image<VecPixel<f64, D>, D>,
    inverse_field: Option<Image<VecPixel<f64, D>, D>>,
}

impl<const D: usize> DisplacementFieldTransform<D> {
    pub fn new(field: Image<VecPixel<f64, D>, D>) -> Self {
        Self { field, inverse_field: None }
    }

    pub fn with_inverse(mut self, inverse_field: Image<VecPixel<f64, D>, D>) -> Self {
        self.inverse_field = Some(inverse_field);
        self
    }

    fn sample(field: &Image<VecPixel<f64, D>, D>, point: [f64; D]) -> VecPixel<f64, D> {
        let mut idx = [0.0f64; D];
        for d in 0..D {
            idx[d] = (point[d] - field.origin[d]) / field.spacing[d]
                     + field.region.index.0[d] as f64;
        }
        LinearInterpolator.evaluate(field, idx)
    }
}

impl<const D: usize> Transform<D> for DisplacementFieldTransform<D> {
    fn transform_point(&self, point: [f64; D]) -> [f64; D] {
        let disp = Self::sample(&self.field, point);
        let mut out = point;
        for d in 0..D { out[d] += disp.0[d]; }
        out
    }

    fn inverse_transform_point(&self, point: [f64; D]) -> Option<[f64; D]> {
        let inv = self.inverse_field.as_ref()?;
        let disp = Self::sample(inv, point);
        let mut out = point;
        for d in 0..D { out[d] += disp.0[d]; }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Region;

    fn constant_field_2d(dx: f64, dy: f64) -> Image<VecPixel<f64, 2>, 2> {
        Image::<VecPixel<f64, 2>, 2>::allocate(
            Region::new([0, 0], [10, 10]),
            [1.0; 2], [0.0; 2],
            VecPixel([dx, dy]),
        )
    }

    #[test]
    fn constant_displacement() {
        let t = DisplacementFieldTransform::new(constant_field_2d(1.5, -2.0));
        let p = t.transform_point([3.0, 4.0]);
        assert!((p[0] - 4.5).abs() < 1e-10 && (p[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn no_inverse_without_field() {
        let t = DisplacementFieldTransform::new(constant_field_2d(1.0, 0.0));
        assert!(t.inverse_transform_point([3.0, 4.0]).is_none());
    }
}
