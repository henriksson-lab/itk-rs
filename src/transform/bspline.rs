//! B-spline deformable transform. Analog to `itk::BSplineTransform`.
//!
//! A regular grid of control points defines a smooth displacement field via
//! cubic B-spline interpolation.  Each control point stores a D-component
//! displacement.
//!
//! `y = x + Σ B(x; k) · d_k`
//!
//! where `B(x; k)` is the product of 1-D cubic B-spline basis functions and
//! `d_k` is the displacement at control point `k`.
//!
//! The inverse is not analytically available; `inverse_transform_point`
//! returns `None`.

use super::Transform;

/// B-spline deformable transform with a uniform control-point grid.
///
/// The grid is described by its `origin`, `spacing`, and `size` (number of
/// control points per axis).  Displacements are stored in a flat row-major
/// Vec of `[f64; D]` arrays.
pub struct BSplineTransform<const D: usize> {
    /// Origin of the control-point grid.
    pub grid_origin: [f64; D],
    /// Spacing between adjacent control points.
    pub grid_spacing: [f64; D],
    /// Number of control points per axis.
    pub grid_size: [usize; D],
    /// Flat, row-major displacement vectors (len = product of grid_size).
    pub displacements: Vec<[f64; D]>,
}

impl<const D: usize> BSplineTransform<D> {
    /// Create a zero-displacement (identity) transform.
    pub fn identity(grid_origin: [f64; D], grid_spacing: [f64; D], grid_size: [usize; D]) -> Self {
        let n: usize = grid_size.iter().product();
        Self {
            grid_origin,
            grid_spacing,
            grid_size,
            displacements: vec![[0.0; D]; n],
        }
    }

    /// Convert a physical point to a continuous control-point grid index.
    fn point_to_grid_index(&self, point: [f64; D]) -> [f64; D] {
        let mut gi = [0.0f64; D];
        for d in 0..D {
            gi[d] = (point[d] - self.grid_origin[d]) / self.grid_spacing[d];
        }
        gi
    }

    /// Flat index into `self.displacements` for a grid point `[i0, i1, …]`.
    fn flat_idx(&self, gp: [i64; D]) -> Option<usize> {
        let mut idx = 0usize;
        let mut stride = 1usize;
        for d in (0..D).rev() {
            if gp[d] < 0 || gp[d] >= self.grid_size[d] as i64 {
                return None;
            }
            idx += gp[d] as usize * stride;
            stride *= self.grid_size[d];
        }
        Some(idx)
    }

    /// Evaluate cubic B-spline weight for `t` in [0,1) relative to basis
    /// function starting at integer offset `k` (k = 0..4).
    ///
    /// Uses the standard symmetric cubic B-spline evaluated at `t + (1 - k)`.
    fn b3(k: usize, t: f64) -> f64 {
        // Shift so that kernel is evaluated at u = t - (k as f64 - 1.0)
        let u = t - (k as f64 - 1.0);
        cubic_bspline(u)
    }
}

/// Evaluate the uniform cubic B-spline kernel at `u` (support on [-2, 2]).
fn cubic_bspline(u: f64) -> f64 {
    let u = u.abs();
    if u < 1.0 {
        (2.0/3.0) - u*u + 0.5*u*u*u
    } else if u < 2.0 {
        (2.0 - u).powi(3) / 6.0
    } else {
        0.0
    }
}

impl<const D: usize> Transform<D> for BSplineTransform<D> {
    fn transform_point(&self, point: [f64; D]) -> [f64; D] {
        let gi = self.point_to_grid_index(point);

        // Starting integer control-point index (support window: 4 points per axis)
        let start: [i64; D] = {
            let mut s = [0i64; D];
            for d in 0..D { s[d] = gi[d].floor() as i64 - 1; }
            s
        };

        // Per-axis weights (4 each)
        let weights: Vec<[f64; 4]> = (0..D)
            .map(|d| {
                let t = gi[d] - gi[d].floor();
                [
                    BSplineTransform::<D>::b3(0, t),
                    BSplineTransform::<D>::b3(1, t),
                    BSplineTransform::<D>::b3(2, t),
                    BSplineTransform::<D>::b3(3, t),
                ]
            })
            .collect();

        let mut disp = [0.0f64; D];

        // Iterate over 4^D support hypercube
        let total: usize = 4usize.pow(D as u32);
        for flat in 0..total {
            // Decode flat index → per-axis offsets
            let mut offsets = [0usize; D];
            let mut tmp = flat;
            for d in (0..D).rev() {
                offsets[d] = tmp % 4;
                tmp /= 4;
            }

            // Compute grid coordinates and weight
            let mut gp = [0i64; D];
            let mut w = 1.0f64;
            for d in 0..D {
                gp[d] = start[d] + offsets[d] as i64;
                w *= weights[d][offsets[d]];
            }

            if let Some(fi) = self.flat_idx(gp) {
                let d_cp = self.displacements[fi];
                for d in 0..D {
                    disp[d] += w * d_cp[d];
                }
            }
        }

        let mut out = point;
        for d in 0..D { out[d] += disp[d]; }
        out
    }

    fn inverse_transform_point(&self, _point: [f64; D]) -> Option<[f64; D]> {
        // Not analytically invertible
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_transform_2d() {
        let t = BSplineTransform::<2>::identity([0.0; 2], [1.0; 2], [8, 8]);
        let p = [3.5, 2.7];
        let q = t.transform_point(p);
        assert!((q[0] - p[0]).abs() < 1e-10 && (q[1] - p[1]).abs() < 1e-10);
    }

    #[test]
    fn uniform_displacement_2d() {
        let mut t = BSplineTransform::<2>::identity([0.0; 2], [1.0; 2], [8, 8]);
        // Set all control points to [1.0, 0.0]
        for d in t.displacements.iter_mut() { *d = [1.0, 0.0]; }
        let p = [3.5, 2.5];
        let q = t.transform_point(p);
        // With uniform displacement and partition-of-unity, x += 1.0
        assert!((q[0] - (p[0] + 1.0)).abs() < 1e-6, "got {}", q[0]);
        assert!((q[1] - p[1]).abs() < 1e-6);
    }
}
