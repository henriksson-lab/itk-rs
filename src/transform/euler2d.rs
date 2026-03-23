//! 2-D rigid body transform (rotation + translation).
//! Analog to `itk::Euler2DTransform`.
//!
//! `y = R(θ)·(x − center) + center + translation`
//!
//! Parameters: `[angle, tx, ty]`.

use super::Transform;

/// 2-D Euler (rigid-body) transform.
#[derive(Clone, Debug)]
pub struct Euler2DTransform {
    /// Rotation angle in radians (CCW).
    pub angle: f64,
    pub translation: [f64; 2],
    /// Fixed center of rotation.
    pub center: [f64; 2],
}

impl Euler2DTransform {
    pub fn new(angle: f64, translation: [f64; 2]) -> Self {
        Self { angle, translation, center: [0.0; 2] }
    }

    pub fn with_center(angle: f64, translation: [f64; 2], center: [f64; 2]) -> Self {
        Self { angle, translation, center }
    }

    pub fn identity() -> Self {
        Self { angle: 0.0, translation: [0.0; 2], center: [0.0; 2] }
    }

    fn rotation_matrix(&self) -> [[f64; 2]; 2] {
        let (s, c) = self.angle.sin_cos();
        [[c, -s], [s, c]]
    }
}

impl Transform<2> for Euler2DTransform {
    fn transform_point(&self, point: [f64; 2]) -> [f64; 2] {
        let r = self.rotation_matrix();
        let xc = [point[0] - self.center[0], point[1] - self.center[1]];
        [
            r[0][0] * xc[0] + r[0][1] * xc[1] + self.center[0] + self.translation[0],
            r[1][0] * xc[0] + r[1][1] * xc[1] + self.center[1] + self.translation[1],
        ]
    }

    fn inverse_transform_point(&self, point: [f64; 2]) -> Option<[f64; 2]> {
        // Inverse rotation: R(-θ)
        let r = self.rotation_matrix();
        // Transpose of rotation = inverse
        let rt = [[r[0][0], r[1][0]], [r[0][1], r[1][1]]];
        let v = [
            point[0] - self.center[0] - self.translation[0],
            point[1] - self.center[1] - self.translation[1],
        ];
        Some([
            rt[0][0] * v[0] + rt[0][1] * v[1] + self.center[0],
            rt[1][0] * v[0] + rt[1][1] * v[1] + self.center[1],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn identity() {
        let t = Euler2DTransform::identity();
        let p = t.transform_point([3.0, 5.0]);
        assert!((p[0] - 3.0).abs() < 1e-10 && (p[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_90_about_origin() {
        let t = Euler2DTransform::new(PI / 2.0, [0.0; 2]);
        let p = t.transform_point([1.0, 0.0]);
        assert!((p[0] - 0.0).abs() < 1e-10 && (p[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn inverse_round_trip() {
        let t = Euler2DTransform::with_center(0.7, [1.0, -2.0], [5.0, 5.0]);
        let p = [3.0, 4.0];
        let q = t.transform_point(p);
        let r = t.inverse_transform_point(q).unwrap();
        assert!((r[0] - p[0]).abs() < 1e-10 && (r[1] - p[1]).abs() < 1e-10);
    }
}
