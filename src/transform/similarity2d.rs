//! 2-D similarity transform (isotropic scale + rotation + translation).
//! Analog to `itk::Similarity2DTransform`.
//!
//! `y = s · R(θ) · (x − center) + center + translation`
//!
//! Parameters: `[scale, angle, tx, ty]`.

use super::Transform;

/// 2-D similarity transform.
#[derive(Clone, Debug)]
pub struct Similarity2DTransform {
    /// Isotropic scale factor.
    pub scale: f64,
    /// Rotation angle in radians (CCW).
    pub angle: f64,
    pub translation: [f64; 2],
    /// Fixed center.
    pub center: [f64; 2],
}

impl Similarity2DTransform {
    pub fn new(scale: f64, angle: f64, translation: [f64; 2]) -> Self {
        Self { scale, angle, translation, center: [0.0; 2] }
    }

    pub fn identity() -> Self {
        Self::new(1.0, 0.0, [0.0; 2])
    }
}

impl Transform<2> for Similarity2DTransform {
    fn transform_point(&self, point: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.angle.sin_cos();
        let xc = [point[0] - self.center[0], point[1] - self.center[1]];
        [
            self.scale * (c * xc[0] - s * xc[1]) + self.center[0] + self.translation[0],
            self.scale * (s * xc[0] + c * xc[1]) + self.center[1] + self.translation[1],
        ]
    }

    fn inverse_transform_point(&self, point: [f64; 2]) -> Option<[f64; 2]> {
        if self.scale == 0.0 {
            return None;
        }
        let (s, c) = self.angle.sin_cos();
        // Inverse: (1/s) · R(-θ) · (y - center - translation) + center
        let v = [
            point[0] - self.center[0] - self.translation[0],
            point[1] - self.center[1] - self.translation[1],
        ];
        let inv_s = 1.0 / self.scale;
        Some([
            inv_s * ( c * v[0] + s * v[1]) + self.center[0],
            inv_s * (-s * v[0] + c * v[1]) + self.center[1],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn identity() {
        let t = Similarity2DTransform::identity();
        let p = [3.0, 7.0];
        let q = t.transform_point(p);
        assert!((q[0] - p[0]).abs() < 1e-10 && (q[1] - p[1]).abs() < 1e-10);
    }

    #[test]
    fn scale_only() {
        let t = Similarity2DTransform::new(2.0, 0.0, [0.0; 2]);
        let p = t.transform_point([3.0, 4.0]);
        assert!((p[0] - 6.0).abs() < 1e-10 && (p[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn inverse_round_trip() {
        let t = Similarity2DTransform { scale: 1.5, angle: PI/4.0, translation: [2.0, -1.0], center: [1.0, 1.0] };
        let p = [3.0, 5.0];
        let q = t.transform_point(p);
        let r = t.inverse_transform_point(q).unwrap();
        assert!((r[0] - p[0]).abs() < 1e-10 && (r[1] - p[1]).abs() < 1e-10);
    }
}
