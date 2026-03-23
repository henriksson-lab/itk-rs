//! 3-D Euler angles rigid-body transform. Analog to `itk::Euler3DTransform`.
//!
//! `y = R_z · R_y · R_x · (x − center) + center + translation` (ZYX convention)
//!
//! ITK also supports ZXY; this is controlled by `compute_zyx`.
//! Parameters: `[angle_x, angle_y, angle_z, tx, ty, tz]`.

use super::Transform;

/// 3-D Euler rigid-body transform (ZYX or ZXY rotation order).
#[derive(Clone, Debug)]
pub struct Euler3DTransform {
    pub angle_x: f64,
    pub angle_y: f64,
    pub angle_z: f64,
    pub translation: [f64; 3],
    pub center: [f64; 3],
    /// If true, use Z-Y-X ordering (default). If false, use Z-X-Y.
    pub compute_zyx: bool,
}

impl Euler3DTransform {
    pub fn new(angle_x: f64, angle_y: f64, angle_z: f64, translation: [f64; 3]) -> Self {
        Self {
            angle_x, angle_y, angle_z, translation,
            center: [0.0; 3],
            compute_zyx: true,
        }
    }

    pub fn identity() -> Self {
        Self::new(0.0, 0.0, 0.0, [0.0; 3])
    }

    fn rotation_matrix(&self) -> [[f64; 3]; 3] {
        let (sx, cx) = self.angle_x.sin_cos();
        let (sy, cy) = self.angle_y.sin_cos();
        let (sz, cz) = self.angle_z.sin_cos();

        if self.compute_zyx {
            // R = Rz · Ry · Rx (ZYX — roll-pitch-yaw)
            [
                [ cy*cz,  cz*sx*sy - cx*sz,  cx*cz*sy + sx*sz],
                [ cy*sz,  cx*cz + sx*sy*sz, -cz*sx + cx*sy*sz],
                [-sy,     cy*sx,             cx*cy            ],
            ]
        } else {
            // R = Rz · Rx · Ry (ZXY)
            [
                [ cy*cz - sx*sy*sz, -cx*sz,  cz*sy + cy*sx*sz],
                [ cy*sz + cz*sx*sy,  cx*cz,  sy*sz - cy*cz*sx],
                [-cx*sy,             sx,     cx*cy            ],
            ]
        }
    }
}

/// Matrix-vector multiply (3×3 × 3).
fn mat3_mul_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
        m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
        m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
    ]
}

/// Transpose of a 3×3 matrix (rotation inverse = transpose).
fn mat3_transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut t = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[j][i] = m[i][j];
        }
    }
    t
}

impl Transform<3> for Euler3DTransform {
    fn transform_point(&self, point: [f64; 3]) -> [f64; 3] {
        let r = self.rotation_matrix();
        let xc = [
            point[0] - self.center[0],
            point[1] - self.center[1],
            point[2] - self.center[2],
        ];
        let rotated = mat3_mul_vec(&r, xc);
        [
            rotated[0] + self.center[0] + self.translation[0],
            rotated[1] + self.center[1] + self.translation[1],
            rotated[2] + self.center[2] + self.translation[2],
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
        let t = Euler3DTransform::identity();
        let p = [1.0, 2.0, 3.0];
        let q = t.transform_point(p);
        for d in 0..3 {
            assert!((q[d] - p[d]).abs() < 1e-10);
        }
    }

    #[test]
    fn rotation_z_90() {
        // 90° around Z-axis: (1,0,0) → (0,1,0)
        let t = Euler3DTransform::new(0.0, 0.0, PI / 2.0, [0.0; 3]);
        let p = t.transform_point([1.0, 0.0, 0.0]);
        assert!(p[0].abs() < 1e-10 && (p[1] - 1.0).abs() < 1e-10 && p[2].abs() < 1e-10);
    }

    #[test]
    fn inverse_round_trip() {
        let t = Euler3DTransform::new(0.3, -0.5, 1.2, [1.0, 2.0, -3.0]);
        let p = [4.0, -1.0, 7.0];
        let q = t.transform_point(p);
        let r = t.inverse_transform_point(q).unwrap();
        for d in 0..3 {
            assert!((r[d] - p[d]).abs() < 1e-9, "axis {d}: {r:?} vs {p:?}");
        }
    }
}
