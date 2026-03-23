//! Scale transform. Analog to `itk::ScaleTransform`.

use super::Transform;

/// Axis-aligned scaling about a center point: `y = center + scale * (x - center)`.
#[derive(Clone, Debug)]
pub struct ScaleTransform<const D: usize> {
    pub scale: [f64; D],
    pub center: [f64; D],
}

impl<const D: usize> ScaleTransform<D> {
    pub fn new(scale: [f64; D]) -> Self {
        Self { scale, center: [0.0; D] }
    }

    pub fn with_center(scale: [f64; D], center: [f64; D]) -> Self {
        Self { scale, center }
    }

    pub fn identity() -> Self {
        Self { scale: [1.0; D], center: [0.0; D] }
    }
}

impl<const D: usize> Transform<D> for ScaleTransform<D> {
    fn transform_point(&self, point: [f64; D]) -> [f64; D] {
        let mut out = point;
        for d in 0..D {
            out[d] = self.center[d] + self.scale[d] * (point[d] - self.center[d]);
        }
        out
    }

    fn inverse_transform_point(&self, point: [f64; D]) -> Option<[f64; D]> {
        for d in 0..D {
            if self.scale[d] == 0.0 {
                return None;
            }
        }
        let mut out = point;
        for d in 0..D {
            out[d] = self.center[d] + (point[d] - self.center[d]) / self.scale[d];
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_2d() {
        let t = ScaleTransform::with_center([2.0, 3.0], [1.0, 1.0]);
        let p = t.transform_point([3.0, 2.0]);
        // center=(1,1), scale=(2,3): y = (1,1) + (2,3)*(2,1) = (5, 4)
        assert!((p[0] - 5.0).abs() < 1e-10);
        assert!((p[1] - 4.0).abs() < 1e-10);
        let inv = t.inverse_transform_point(p).unwrap();
        assert!((inv[0] - 3.0).abs() < 1e-10);
        assert!((inv[1] - 2.0).abs() < 1e-10);
    }
}
