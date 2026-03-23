//! Affine transform. Analog to `itk::AffineTransform`.
//!
//! `y = M·(x − center) + center + translation`
//!
//! where `M` is a D×D matrix stored in **row-major** order (row 0 is the
//! first D elements). The inverse is computed via Gaussian elimination.

use super::Transform;

/// General affine transform: linear map + translation, optionally about a center.
#[derive(Clone, Debug)]
pub struct AffineTransform<const D: usize> {
    /// Row-major D×D matrix.
    pub matrix: [[f64; D]; D],
    pub translation: [f64; D],
    pub center: [f64; D],
}

impl<const D: usize> AffineTransform<D> {
    /// Construct from matrix and translation; center = origin.
    pub fn new(matrix: [[f64; D]; D], translation: [f64; D]) -> Self {
        Self { matrix, translation, center: [0.0; D] }
    }

    pub fn identity() -> Self {
        let mut m = [[0.0f64; D]; D];
        for i in 0..D {
            m[i][i] = 1.0;
        }
        Self { matrix: m, translation: [0.0; D], center: [0.0; D] }
    }

    /// Apply the linear part `M·v` (no translation / center).
    fn apply_matrix(&self, v: [f64; D]) -> [f64; D] {
        let mut out = [0.0f64; D];
        for i in 0..D {
            for j in 0..D {
                out[i] += self.matrix[i][j] * v[j];
            }
        }
        out
    }

    /// Invert the D×D matrix using Gaussian elimination with partial pivoting.
    /// Returns `None` if singular.
    fn invert_matrix(&self) -> Option<[[f64; D]; D]> {
        let n = D;
        // Augmented matrix [M | I]
        let mut aug = [[0.0f64; D]; D];
        let mut inv = [[0.0f64; D]; D];
        for i in 0..n {
            aug[i] = self.matrix[i];
            inv[i][i] = 1.0;
        }

        for col in 0..n {
            // Partial pivot
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in col + 1..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-14 {
                return None;
            }
            aug.swap(col, max_row);
            inv.swap(col, max_row);

            let pivot = aug[col][col];
            for j in 0..n {
                aug[col][j] /= pivot;
                inv[col][j] /= pivot;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[row][col];
                for j in 0..n {
                    aug[row][j] -= factor * aug[col][j];
                    inv[row][j] -= factor * inv[col][j];
                }
            }
        }
        Some(inv)
    }
}

impl<const D: usize> Transform<D> for AffineTransform<D> {
    fn transform_point(&self, point: [f64; D]) -> [f64; D] {
        // x_c = x - center
        let mut xc = point;
        for d in 0..D {
            xc[d] -= self.center[d];
        }
        // M · x_c
        let mut out = self.apply_matrix(xc);
        // + center + translation
        for d in 0..D {
            out[d] += self.center[d] + self.translation[d];
        }
        out
    }

    fn inverse_transform_point(&self, point: [f64; D]) -> Option<[f64; D]> {
        let inv_m = self.invert_matrix()?;
        // Temporarily build an AffineTransform with inv_m and no center
        let inv_t = AffineTransform {
            matrix: inv_m,
            translation: [0.0; D],
            center: [0.0; D],
        };
        // y - center - translation
        let mut v = point;
        for d in 0..D {
            v[d] -= self.center[d] + self.translation[d];
        }
        // M⁻¹ · v + center
        let mut out = inv_t.apply_matrix(v);
        for d in 0..D {
            out[d] += self.center[d];
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_2d() {
        let t = AffineTransform::<2>::identity();
        let p = [3.0, 5.0];
        let q = t.transform_point(p);
        assert!((q[0] - 3.0).abs() < 1e-10 && (q[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_90_2d() {
        // 90° CCW rotation: [0,-1; 1, 0]
        let m = [[0.0, -1.0], [1.0, 0.0]];
        let t = AffineTransform::new(m, [0.0; 2]);
        let p = t.transform_point([1.0, 0.0]);
        assert!((p[0] - 0.0).abs() < 1e-10 && (p[1] - 1.0).abs() < 1e-10);
        let inv = t.inverse_transform_point([0.0, 1.0]).unwrap();
        assert!((inv[0] - 1.0).abs() < 1e-10 && (inv[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn singular_returns_none() {
        let m = [[1.0, 2.0], [2.0, 4.0]]; // det = 0
        let t = AffineTransform::new(m, [0.0; 2]);
        assert!(t.inverse_transform_point([1.0, 2.0]).is_none());
    }
}
