//! 3-D versor (unit quaternion) rigid-body transform.
//! Analog to `itk::VersorRigid3DTransform`.
//!
//! `y = R(q) · (x − center) + center + translation`
//!
//! The rotation is stored as a unit quaternion `(w, x, y, z)` where
//! `w = sqrt(1 - x² - y² - z²)` (constrained). Parameters are the
//! three independent versor components `[vx, vy, vz, tx, ty, tz]`.

use super::Transform;

/// 3-D rigid-body transform parameterised by a unit quaternion.
#[derive(Clone, Debug)]
pub struct VersorRigid3DTransform {
    /// Unit quaternion (w, x, y, z).  Always kept normalised.
    pub quat: [f64; 4],
    pub translation: [f64; 3],
    pub center: [f64; 3],
}

impl VersorRigid3DTransform {
    /// Construct from unit quaternion `[w, x, y, z]`.
    pub fn from_quaternion(quat: [f64; 4], translation: [f64; 3]) -> Self {
        let norm = (quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]).sqrt();
        let q = [quat[0]/norm, quat[1]/norm, quat[2]/norm, quat[3]/norm];
        Self { quat: q, translation, center: [0.0; 3] }
    }

    /// Construct from axis-angle representation.
    /// `axis` need not be normalised.
    pub fn from_axis_angle(axis: [f64; 3], angle: f64, translation: [f64; 3]) -> Self {
        let norm = (axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]).sqrt();
        if norm < 1e-14 {
            return Self::identity();
        }
        let (s, c) = (angle / 2.0).sin_cos();
        let quat = [c, s * axis[0]/norm, s * axis[1]/norm, s * axis[2]/norm];
        Self { quat, translation, center: [0.0; 3] }
    }

    pub fn identity() -> Self {
        Self { quat: [1.0, 0.0, 0.0, 0.0], translation: [0.0; 3], center: [0.0; 3] }
    }

    /// Compute the 3×3 rotation matrix from `self.quat`.
    fn rotation_matrix(&self) -> [[f64; 3]; 3] {
        let [w, x, y, z] = self.quat;
        [
            [1.0 - 2.0*(y*y + z*z),  2.0*(x*y - w*z),       2.0*(x*z + w*y)      ],
            [2.0*(x*y + w*z),         1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)      ],
            [2.0*(x*z - w*y),         2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)],
        ]
    }
}

fn mat3_mul_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
        m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
        m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
    ]
}

fn mat3_transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut t = [[0.0f64; 3]; 3];
    for i in 0..3 { for j in 0..3 { t[j][i] = m[i][j]; } }
    t
}

impl Transform<3> for VersorRigid3DTransform {
    fn transform_point(&self, point: [f64; 3]) -> [f64; 3] {
        let r = self.rotation_matrix();
        let xc = [
            point[0] - self.center[0],
            point[1] - self.center[1],
            point[2] - self.center[2],
        ];
        let rot = mat3_mul_vec(&r, xc);
        [
            rot[0] + self.center[0] + self.translation[0],
            rot[1] + self.center[1] + self.translation[1],
            rot[2] + self.center[2] + self.translation[2],
        ]
    }

    fn inverse_transform_point(&self, point: [f64; 3]) -> Option<[f64; 3]> {
        let r = self.rotation_matrix();
        let rt = mat3_transpose(&r);
        let v = [
            point[0] - self.center[0] - self.translation[0],
            point[1] - self.center[1] - self.translation[1],
            point[2] - self.center[2] - self.translation[2],
        ];
        let out = mat3_mul_vec(&rt, v);
        Some([
            out[0] + self.center[0],
            out[1] + self.center[1],
            out[2] + self.center[2],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn identity() {
        let t = VersorRigid3DTransform::identity();
        let p = [1.0, 2.0, 3.0];
        let q = t.transform_point(p);
        for d in 0..3 { assert!((q[d] - p[d]).abs() < 1e-10); }
    }

    #[test]
    fn rotation_around_z() {
        // 90° around Z: (1,0,0) → (0,1,0)
        let t = VersorRigid3DTransform::from_axis_angle([0.0, 0.0, 1.0], PI/2.0, [0.0; 3]);
        let p = t.transform_point([1.0, 0.0, 0.0]);
        assert!(p[0].abs() < 1e-10 && (p[1] - 1.0).abs() < 1e-10 && p[2].abs() < 1e-10);
    }

    #[test]
    fn inverse_round_trip() {
        let t = VersorRigid3DTransform::from_axis_angle([1.0, 2.0, 3.0], 1.1, [0.5, -1.0, 2.0]);
        let p = [3.0, -1.0, 7.0];
        let q = t.transform_point(p);
        let r = t.inverse_transform_point(q).unwrap();
        for d in 0..3 { assert!((r[d] - p[d]).abs() < 1e-9, "axis {d}: {r:?}"); }
    }
}
