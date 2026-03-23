//! Displacement field filters.
//!
//! | Filter | ITK analog |
//! |---|---|
//! | [`TransformToDisplacementFieldFilter`]     | `TransformToDisplacementFieldFilter` |
//! | [`ComposeDisplacementFieldsFilter`]         | `ComposeDisplacementFieldsImageFilter` |
//! | [`DisplacementFieldJacobianDeterminantFilter`] | `DisplacementFieldJacobianDeterminantFilter` |
//! | [`InvertDisplacementFieldFilter`]           | `InvertDisplacementFieldImageFilter` |
//! | [`ExponentialDisplacementFieldFilter`]      | `ExponentialDisplacementFieldImageFilter` |

use rayon::prelude::*;

use crate::image::{Image, Index, Region, iter_region};
use crate::pixel::{NumericPixel, VecPixel};
use crate::source::ImageSource;

// Displacement field type aliases (2D: VecPixel<f32, 2>, 3D: VecPixel<f32, 3>)

// ===========================================================================
// TransformToDisplacementFieldFilter
// ===========================================================================

/// Samples a parametric transform at every grid point to produce a dense
/// displacement field.
/// Analog to `itk::TransformToDisplacementFieldFilter`.
///
/// The transform function maps a physical point `[f64; D]` (in world
/// coordinates: `origin + index * spacing`) to an output point; the
/// displacement is `output − input`.
///
/// For 2D the output pixel is `VecPixel<f32, 2>`.
pub struct TransformToDisplacementField2D<F> {
    pub region: Region<2>,
    pub spacing: [f64; 2],
    pub origin: [f64; 2],
    /// Maps physical input point `[x, y]` → output point `[x', y']`.
    pub transform: F,
}

impl<F: Fn([f64; 2]) -> [f64; 2] + Sync> ImageSource<VecPixel<f32, 2>, 2>
    for TransformToDisplacementField2D<F>
{
    fn largest_region(&self) -> Region<2> { self.region }
    fn spacing(&self) -> [f64; 2] { self.spacing }
    fn origin(&self) -> [f64; 2] { self.origin }

    fn generate_region(&self, requested: Region<2>) -> Image<VecPixel<f32, 2>, 2> {
        let mut out_indices = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<VecPixel<f32, 2>> = out_indices.par_iter().map(|&idx| {
            let px = self.origin[0] + idx.0[0] as f64 * self.spacing[0];
            let py = self.origin[1] + idx.0[1] as f64 * self.spacing[1];
            let [ox, oy] = (self.transform)([px, py]);
            VecPixel([(ox - px) as f32, (oy - py) as f32])
        }).collect();

        Image { region: requested, spacing: self.spacing, origin: self.origin, data }
    }
}

// ===========================================================================
// ComposeDisplacementFieldsImageFilter
// ===========================================================================

/// Compose two displacement fields: `d_out(x) = d1(x) + d2(x + d1(x))`.
/// Analog to `itk::ComposeDisplacementFieldsImageFilter`.
pub struct ComposeDisplacementFields2D<S1, S2> {
    pub field1: S1,
    pub field2: S2,
}

impl<S1, S2> ComposeDisplacementFields2D<S1, S2> {
    pub fn new(field1: S1, field2: S2) -> Self { Self { field1, field2 } }
}

impl<S1, S2> ImageSource<VecPixel<f32, 2>, 2> for ComposeDisplacementFields2D<S1, S2>
where
    S1: ImageSource<VecPixel<f32, 2>, 2> + Sync,
    S2: ImageSource<VecPixel<f32, 2>, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.field1.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.field1.spacing() }
    fn origin(&self) -> [f64; 2] { self.field1.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<VecPixel<f32, 2>, 2> {
        let f1 = self.field1.generate_region(requested);
        let f2 = self.field2.generate_region(self.field2.largest_region());
        let bounds2 = f2.region;
        let sp = f2.spacing;
        let or = f2.origin;

        let mut out_indices = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<VecPixel<f32, 2>> = out_indices.par_iter().map(|&idx| {
            let d1 = f1.get_pixel(idx);
            // Sample f2 at (idx + d1) via bilinear interpolation
            let x2 = idx.0[0] as f64 + d1.0[0] as f64 / sp[0];
            let y2 = idx.0[1] as f64 + d1.0[1] as f64 / sp[1];
            let xi = x2.floor() as i64;
            let yi = y2.floor() as i64;
            let fx = x2 - xi as f64;
            let fy = y2 - yi as f64;
            let clamp_x = |v: i64| v.max(bounds2.index.0[0]).min(bounds2.index.0[0] + bounds2.size.0[0] as i64 - 1);
            let clamp_y = |v: i64| v.max(bounds2.index.0[1]).min(bounds2.index.0[1] + bounds2.size.0[1] as i64 - 1);
            let p00 = f2.get_pixel(Index([clamp_x(xi), clamp_y(yi)]));
            let p10 = f2.get_pixel(Index([clamp_x(xi+1), clamp_y(yi)]));
            let p01 = f2.get_pixel(Index([clamp_x(xi), clamp_y(yi+1)]));
            let p11 = f2.get_pixel(Index([clamp_x(xi+1), clamp_y(yi+1)]));
            let interp = |c: usize| -> f32 {
                let top = p00.0[c] * (1.0 - fx as f32) + p10.0[c] * fx as f32;
                let bot = p01.0[c] * (1.0 - fx as f32) + p11.0[c] * fx as f32;
                top * (1.0 - fy as f32) + bot * fy as f32
            };
            VecPixel([d1.0[0] + interp(0), d1.0[1] + interp(1)])
        }).collect();

        Image { region: requested, spacing: f1.spacing, origin: f1.origin, data }
    }
}

// ===========================================================================
// DisplacementFieldJacobianDeterminantFilter
// ===========================================================================

/// Computes the Jacobian determinant of a displacement field.
/// Analog to `itk::DisplacementFieldJacobianDeterminantFilter`.
///
/// For D=2: `det(J) = (1+∂u/∂x)(1+∂v/∂y) − (∂u/∂y)(∂v/∂x)`.
pub struct DisplacementFieldJacobianDeterminantFilter2D<S> {
    pub source: S,
}

impl<S> DisplacementFieldJacobianDeterminantFilter2D<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<S> ImageSource<f32, 2> for DisplacementFieldJacobianDeterminantFilter2D<S>
where
    S: ImageSource<VecPixel<f32, 2>, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        let field = self.source.generate_region(requested);
        let bounds = field.region;
        let hx = field.spacing[0];
        let hy = field.spacing[1];

        let clamp_x = |v: i64| v.max(bounds.index.0[0]).min(bounds.index.0[0] + bounds.size.0[0] as i64 - 1);
        let clamp_y = |v: i64| v.max(bounds.index.0[1]).min(bounds.index.0[1] + bounds.size.0[1] as i64 - 1);

        let mut out_indices = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<f32> = out_indices.par_iter().map(|&idx| {
            let [x, y] = idx.0;
            let get = |xi: i64, yi: i64| field.get_pixel(Index([clamp_x(xi), clamp_y(yi)]));
            let dux = (get(x+1,y).0[0] - get(x-1,y).0[0]) as f64 / (2.0 * hx);
            let duy = (get(x,y+1).0[0] - get(x,y-1).0[0]) as f64 / (2.0 * hy);
            let dvx = (get(x+1,y).0[1] - get(x-1,y).0[1]) as f64 / (2.0 * hx);
            let dvy = (get(x,y+1).0[1] - get(x,y-1).0[1]) as f64 / (2.0 * hy);
            ((1.0 + dux) * (1.0 + dvy) - duy * dvx) as f32
        }).collect();

        Image { region: requested, spacing: field.spacing, origin: field.origin, data }
    }
}

// ===========================================================================
// InvertDisplacementFieldFilter
// ===========================================================================

/// Iterative inversion of a displacement field.
/// Analog to `itk::InvertDisplacementFieldImageFilter`.
///
/// Solves: `d_inv(x + d(x)) = -d(x)` iteratively.
pub struct InvertDisplacementFieldFilter2D<S> {
    pub source: S,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl<S> InvertDisplacementFieldFilter2D<S> {
    pub fn new(source: S) -> Self {
        Self { source, max_iterations: 20, tolerance: 0.01 }
    }
}

impl<S> ImageSource<VecPixel<f32, 2>, 2> for InvertDisplacementFieldFilter2D<S>
where
    S: ImageSource<VecPixel<f32, 2>, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<VecPixel<f32, 2>, 2> {
        let field = self.source.generate_region(requested);
        let bounds = field.region;
        let sp = field.spacing;

        // Initialize inverse as negated field
        let mut inv_data: Vec<VecPixel<f32, 2>> = field.data.iter()
            .map(|d| VecPixel([-d.0[0], -d.0[1]])).collect();

        let n = inv_data.len();
        let mut out_indices = Vec::with_capacity(n);
        iter_region(&requested, |idx| out_indices.push(idx));

        for _ in 0..self.max_iterations {
            let prev = inv_data.clone();
            // For each point x: d_inv_new(x) = -d(x + d_inv(x))
            let new_data: Vec<VecPixel<f32, 2>> = out_indices.par_iter().enumerate().map(|(i, &idx)| {
                let di = prev[i];
                // Sample field at (idx + di)
                let sx = idx.0[0] as f64 + di.0[0] as f64 / sp[0];
                let sy = idx.0[1] as f64 + di.0[1] as f64 / sp[1];
                let xi = sx.floor() as i64;
                let yi = sy.floor() as i64;
                let clamp_x = |v: i64| v.max(bounds.index.0[0]).min(bounds.index.0[0] + bounds.size.0[0] as i64 - 1);
                let clamp_y = |v: i64| v.max(bounds.index.0[1]).min(bounds.index.0[1] + bounds.size.0[1] as i64 - 1);
                let fx = (sx - xi as f64) as f32;
                let fy = (sy - yi as f64) as f32;
                let p00 = field.get_pixel(Index([clamp_x(xi), clamp_y(yi)]));
                let p10 = field.get_pixel(Index([clamp_x(xi+1), clamp_y(yi)]));
                let p01 = field.get_pixel(Index([clamp_x(xi), clamp_y(yi+1)]));
                let p11 = field.get_pixel(Index([clamp_x(xi+1), clamp_y(yi+1)]));
                let interp = |c: usize| -> f32 {
                    let top = p00.0[c] * (1.0 - fx) + p10.0[c] * fx;
                    let bot = p01.0[c] * (1.0 - fx) + p11.0[c] * fx;
                    top * (1.0 - fy) + bot * fy
                };
                VecPixel([-interp(0), -interp(1)])
            }).collect();
            let max_change = new_data.iter().zip(prev.iter())
                .map(|(a, b)| ((a.0[0]-b.0[0]).powi(2) + (a.0[1]-b.0[1]).powi(2)).sqrt())
                .fold(0.0f32, f32::max);
            inv_data = new_data;
            if max_change < self.tolerance as f32 { break; }
        }

        Image { region: requested, spacing: field.spacing, origin: field.origin, data: inv_data }
    }
}

// ===========================================================================
// ExponentialDisplacementFieldImageFilter
// ===========================================================================

/// Computes the exponential of a velocity field: `exp(v)` via scaling and squaring.
/// Analog to `itk::ExponentialDisplacementFieldImageFilter`.
pub struct ExponentialDisplacementFieldFilter2D<S> {
    pub source: S,
    pub compute_inverse: bool,
    pub max_iter: usize,
}

impl<S> ExponentialDisplacementFieldFilter2D<S> {
    pub fn new(source: S) -> Self {
        Self { source, compute_inverse: false, max_iter: 10 }
    }
}

impl<S> ImageSource<VecPixel<f32, 2>, 2> for ExponentialDisplacementFieldFilter2D<S>
where
    S: ImageSource<VecPixel<f32, 2>, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<VecPixel<f32, 2>, 2> {
        let v = self.source.generate_region(requested);
        let n_iter = self.max_iter;
        let sign = if self.compute_inverse { -1.0f32 } else { 1.0f32 };

        // Scale v by 2^(-n_iter)
        let scale = 1.0f32 / (1 << n_iter) as f32;
        let mut result: Vec<VecPixel<f32, 2>> = v.data.iter()
            .map(|d| VecPixel([d.0[0] * scale * sign, d.0[1] * scale * sign])).collect();

        // Squaring steps
        let bounds = v.region;
        let sp = v.spacing;
        let mut out_indices = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        for _ in 0..n_iter {
            let prev = result.clone();
            let tmp = Image { region: requested, spacing: sp, origin: v.origin, data: prev.clone() };
            result = out_indices.par_iter().enumerate().map(|(i, &idx)| {
                let d = prev[i];
                // Sample at idx + d (compose with itself)
                let sx = idx.0[0] as f64 + d.0[0] as f64 / sp[0];
                let sy = idx.0[1] as f64 + d.0[1] as f64 / sp[1];
                let xi = sx.floor() as i64;
                let yi = sy.floor() as i64;
                let clamp_x = |v: i64| v.max(bounds.index.0[0]).min(bounds.index.0[0] + bounds.size.0[0] as i64 - 1);
                let clamp_y = |v: i64| v.max(bounds.index.0[1]).min(bounds.index.0[1] + bounds.size.0[1] as i64 - 1);
                let fx = (sx - xi as f64) as f32;
                let fy = (sy - yi as f64) as f32;
                let p00 = tmp.get_pixel(Index([clamp_x(xi), clamp_y(yi)]));
                let p10 = tmp.get_pixel(Index([clamp_x(xi+1), clamp_y(yi)]));
                let p01 = tmp.get_pixel(Index([clamp_x(xi), clamp_y(yi+1)]));
                let p11 = tmp.get_pixel(Index([clamp_x(xi+1), clamp_y(yi+1)]));
                let interp = |c: usize| -> f32 {
                    let top = p00.0[c] * (1.0 - fx) + p10.0[c] * fx;
                    let bot = p01.0[c] * (1.0 - fx) + p11.0[c] * fx;
                    top * (1.0 - fy) + bot * fy
                };
                VecPixel([d.0[0] + interp(0), d.0[1] + interp(1)])
            }).collect();
        }

        Image { region: requested, spacing: v.spacing, origin: v.origin, data: result }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    #[test]
    fn identity_transform_zero_displacement() {
        let f = TransformToDisplacementField2D {
            region: Region::new([0,0],[5,5]),
            spacing: [1.0,1.0],
            origin: [0.0,0.0],
            transform: |p: [f64;2]| p, // identity
        };
        let out = f.generate_region(f.largest_region());
        for v in &out.data {
            assert!(v.0[0].abs() < 1e-6 && v.0[1].abs() < 1e-6);
        }
    }

    #[test]
    fn jacobian_of_zero_field_is_one() {
        let zero_field = Image::<VecPixel<f32, 2>, 2>::allocate(
            Region::new([0,0],[5,5]), [1.0,1.0], [0.0,0.0], VecPixel([0.0f32,0.0f32]),
        );
        let f = DisplacementFieldJacobianDeterminantFilter2D::new(zero_field);
        let out = f.generate_region(f.largest_region());
        for &v in &out.data {
            assert!((v - 1.0).abs() < 1e-5, "expected 1 got {v}");
        }
    }
}
