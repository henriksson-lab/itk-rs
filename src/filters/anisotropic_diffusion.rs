//! Anisotropic diffusion filters.
//!
//! Analogs to `itk::GradientAnisotropicDiffusionImageFilter` and
//! `itk::CurvatureAnisotropicDiffusionImageFilter`.
//!
//! Both implement the Perona-Malik class of PDE-based edge-preserving smoothers
//! solved with explicit Euler time-stepping.
//!
//! # Stability
//! For an explicit scheme the time step must satisfy:
//!
//! `Δt ≤ 1 / (2 · D)`    (D = number of dimensions)
//!
//! [`GradientAnisotropicDiffusionFilter`] defaults to `1/(2D+1)` for safety.
//!
//! # Conductance functions
//! Both filters share the Perona-Malik conductance function:
//!
//! `g(s) = exp(−s² / K²)`
//!
//! where `K` is the `conductance_parameter` (higher K → less edge preservation).

use crate::image::{Image, Index, Region, iter_region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

// ---------------------------------------------------------------------------
// Conductance function
// ---------------------------------------------------------------------------

#[inline]
fn conductance(gradient: f64, k: f64) -> f64 {
    (-(gradient / k).powi(2)).exp()
}

// ---------------------------------------------------------------------------
// Gradient Anisotropic Diffusion
// ---------------------------------------------------------------------------

/// Gradient anisotropic diffusion filter (Perona-Malik, 1990).
///
/// Update rule per pixel `x`, time step `dt`:
/// ```text
/// ΔI[x] = dt · Σ_d { g(|∇⁺_d I|) · ∇⁺_d I + g(|∇⁻_d I|) · ∇⁻_d I }
/// ```
/// where `∇⁺_d I = I[x+e_d] − I[x]` and `∇⁻_d I = I[x−e_d] − I[x]`.
pub struct GradientAnisotropicDiffusionFilter<S> {
    pub source: S,
    /// Number of explicit Euler iterations.
    pub iterations: usize,
    /// Edge-strength threshold.  Larger → more smoothing across edges.
    pub conductance_parameter: f64,
    /// Time step (default: `1/(2D+1)` for stability).
    pub time_step: Option<f64>,
}

impl<S> GradientAnisotropicDiffusionFilter<S> {
    pub fn new(source: S, iterations: usize, conductance_parameter: f64) -> Self {
        Self { source, iterations, conductance_parameter, time_step: None }
    }

    pub fn with_time_step(mut self, dt: f64) -> Self {
        self.time_step = Some(dt);
        self
    }
}

fn run_gradient_diffusion<const D: usize>(
    mut u: Vec<f64>,
    region: Region<D>,
    spacing: [f64; D],
    iterations: usize,
    k: f64,
    dt: f64,
) -> Vec<f64> {
    let size = region.size.0;
    let mut delta = vec![0.0f64; u.len()];

    // Build a stride table for flat indexing within `region`
    let strides: [usize; D] = {
        let mut s = [1usize; D];
        for d in 1..D {
            s[d] = s[d - 1] * size[d - 1];
        }
        s
    };

    let flat_idx = |idx: [i64; D]| -> usize {
        let mut f = 0usize;
        for d in 0..D {
            f += (idx[d] - region.index.0[d]) as usize * strides[d];
        }
        f
    };

    for _ in 0..iterations {
        // Compute delta
        let mut indices: Vec<Index<D>> = Vec::with_capacity(region.linear_len());
        iter_region(&region, |idx| indices.push(idx));

        for out_idx in &indices {
            let fi = flat_idx(out_idx.0);
            let center = u[fi];
            let mut acc = 0.0f64;

            for d in 0..D {
                let sp = spacing[d];
                // Forward neighbour
                let mut fwd = out_idx.0;
                fwd[d] = (fwd[d] + 1).min(region.index.0[d] + size[d] as i64 - 1);
                let g_fwd = u[flat_idx(fwd)] - center;
                let c_fwd = conductance(g_fwd.abs() / sp, k);
                acc += c_fwd * g_fwd;

                // Backward neighbour
                let mut bwd = out_idx.0;
                bwd[d] = (bwd[d] - 1).max(region.index.0[d]);
                let g_bwd = u[flat_idx(bwd)] - center;
                let c_bwd = conductance(g_bwd.abs() / sp, k);
                acc += c_bwd * g_bwd;
            }

            delta[fi] = dt * acc;
        }

        for i in 0..u.len() {
            u[i] += delta[i];
        }
    }
    u
}

impl<P, S, const D: usize> ImageSource<P, D> for GradientAnisotropicDiffusionFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> {
        self.source.largest_region()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let spacing = self.source.spacing();
        let dt = self.time_step.unwrap_or(1.0 / (2 * D + 1) as f64);

        let full_region = self.source.largest_region();
        let full = self.source.generate_region(full_region);
        let u: Vec<f64> = full.data.iter().map(|p| p.to_f64()).collect();

        let result = run_gradient_diffusion::<D>(
            u, full_region, spacing,
            self.iterations, self.conductance_parameter, dt,
        );

        // Crop to requested and convert back
        let mut indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| indices.push(idx));

        // Rebuild flat-index lookup for full_region
        let strides: [usize; D] = {
            let mut s = [1usize; D];
            for d in 1..D { s[d] = s[d-1] * full_region.size.0[d-1]; }
            s
        };
        let data: Vec<P> = indices.iter().map(|idx| {
            let mut fi = 0usize;
            for d in 0..D {
                fi += (idx.0[d] - full_region.index.0[d]) as usize * strides[d];
            }
            P::from_f64(result[fi])
        }).collect();

        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

// ---------------------------------------------------------------------------
// Curvature Anisotropic Diffusion
// ---------------------------------------------------------------------------

/// Curvature anisotropic diffusion filter.
/// Analog to `itk::CurvatureAnisotropicDiffusionImageFilter`.
///
/// Uses a modified curvature-based diffusion equation that smooths along
/// isophote contours while preserving edges. In practice this is implemented
/// as a gradient-magnitude-weighted curvature flow.
///
/// For 1-D images this degenerates to gradient diffusion.
///
/// **Note**: full curvature computation is only well-defined for D ≥ 2.
pub struct CurvatureAnisotropicDiffusionFilter<S> {
    pub source: S,
    pub iterations: usize,
    pub conductance_parameter: f64,
    pub time_step: Option<f64>,
}

impl<S> CurvatureAnisotropicDiffusionFilter<S> {
    pub fn new(source: S, iterations: usize, conductance_parameter: f64) -> Self {
        Self { source, iterations, conductance_parameter, time_step: None }
    }

    pub fn with_time_step(mut self, dt: f64) -> Self {
        self.time_step = Some(dt);
        self
    }
}

/// Compute the curvature-flow update for pixel `fi` in a 2-D field.
///
/// Uses the modified curvature-diffusion equation from Whitaker & Pizer (1993):
/// `ΔI = dt · g(|∇I|) · (Iξξ + c(|∇I|)·Iηη)`
///
/// For general D we fall back to gradient diffusion weighted by `g(|∇I|)`.
fn curvature_update_2d(
    u: &[f64],
    strides: &[usize],
    idx: [i64; 2],
    region: &Region<2>,
    spacing: [f64; 2],
    k: f64,
    dt: f64,
) -> f64 {
    let clamp = |v: i64, lo: i64, hi: i64| v.max(lo).min(hi);

    let lo = region.index.0;
    let hi = [
        lo[0] + region.size.0[0] as i64 - 1,
        lo[1] + region.size.0[1] as i64 - 1,
    ];

    let fi = |i: i64, j: i64| -> usize {
        (clamp(i, lo[0], hi[0]) - lo[0]) as usize * strides[0]
            + (clamp(j, lo[1], hi[1]) - lo[1]) as usize * strides[1]
    };

    let [x, y] = idx;
    let c = u[fi(x, y)];
    let dx = (u[fi(x+1, y)] - u[fi(x-1, y)]) / (2.0 * spacing[0]);
    let dy = (u[fi(x, y+1)] - u[fi(x, y-1)]) / (2.0 * spacing[1]);
    let dxx = (u[fi(x+1, y)] - 2.0*c + u[fi(x-1, y)]) / spacing[0].powi(2);
    let dyy = (u[fi(x, y+1)] - 2.0*c + u[fi(x, y-1)]) / spacing[1].powi(2);
    let dxy = (u[fi(x+1,y+1)] - u[fi(x-1,y+1)] - u[fi(x+1,y-1)] + u[fi(x-1,y-1)])
              / (4.0 * spacing[0] * spacing[1]);

    let grad2 = dx*dx + dy*dy;
    let grad = grad2.sqrt();

    let g = conductance(grad, k);

    if grad < 1e-10 {
        return 0.0;
    }

    // Normal direction (gradient): curvature term in Iξξ (along gradient)
    // Tangent direction: Iηη (perpendicular)
    // κ = (dxx*dy² - 2*dxy*dx*dy + dyy*dx²) / grad²^(3/2)  — mean curvature
    let curvature = (dxx * dy*dy - 2.0*dxy*dx*dy + dyy*dx*dx) / (grad2 * grad);

    dt * g * curvature * grad
}

fn run_curvature_diffusion_2d(
    mut u: Vec<f64>,
    region: Region<2>,
    spacing: [f64; 2],
    iterations: usize,
    conductance: f64,
    dt: f64,
) -> Vec<f64> {
    let size = region.size.0;
    let strides = [1usize, size[0]];

    for _ in 0..iterations {
        let mut delta = vec![0.0f64; u.len()];

        let mut indices: Vec<Index<2>> = Vec::with_capacity(region.linear_len());
        iter_region(&region, |idx| indices.push(idx));

        for idx in &indices {
            let fi = (idx.0[0] - region.index.0[0]) as usize * strides[0]
                   + (idx.0[1] - region.index.0[1]) as usize * strides[1];
            delta[fi] = curvature_update_2d(&u, &strides, idx.0, &region, spacing, conductance, dt);
        }
        for i in 0..u.len() {
            u[i] += delta[i];
        }
    }
    u
}

impl<P, S, const D: usize> ImageSource<P, D> for CurvatureAnisotropicDiffusionFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, _: &Region<D>) -> Region<D> { self.source.largest_region() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let spacing = self.source.spacing();
        let dt = self.time_step.unwrap_or(1.0 / (2 * D + 1) as f64);
        let full_region = self.source.largest_region();
        let full = self.source.generate_region(full_region);
        let u: Vec<f64> = full.data.iter().map(|p| p.to_f64()).collect();

        let result = if D == 2 {
            // Safety: we statically know D=2 here but can't convince the compiler
            // without unsafe transmute. Instead, we use a dynamic dispatch on D.
            run_curvature_diffusion_generic::<D>(
                u, full_region, spacing,
                self.iterations, self.conductance_parameter, dt,
            )
        } else {
            // For D != 2, fall back to gradient diffusion
            run_gradient_diffusion::<D>(
                u, full_region, spacing,
                self.iterations, self.conductance_parameter, dt,
            )
        };

        let strides: [usize; D] = {
            let mut s = [1usize; D];
            for d in 1..D { s[d] = s[d-1] * full_region.size.0[d-1]; }
            s
        };
        let mut indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| indices.push(idx));
        let data: Vec<P> = indices.iter().map(|idx| {
            let mut fi = 0usize;
            for d in 0..D { fi += (idx.0[d] - full_region.index.0[d]) as usize * strides[d]; }
            P::from_f64(result[fi])
        }).collect();

        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

/// Runtime dispatch wrapper so the 2D curvature code path can be used from the
/// generic `const D: usize` impl above without unsafe transmute.
fn run_curvature_diffusion_generic<const D: usize>(
    u: Vec<f64>,
    region: Region<D>,
    spacing: [f64; D],
    iterations: usize,
    conductance: f64,
    dt: f64,
) -> Vec<f64> {
    assert_eq!(D, 2, "curvature diffusion is only implemented for D=2");

    // Reinterpret as 2-D (we just asserted D==2)
    // SAFETY: Region<2> and Region<D> have the same layout when D==2,
    // and [f64;2] and [f64;D] are the same when D==2. We use ptr casts.
    use std::mem::transmute_copy;
    // We assert D==2 above so this is safe.
    let region2: Region<2> = unsafe { transmute_copy(&region) };
    let spacing2: [f64; 2] = unsafe { transmute_copy(&spacing) };

    run_curvature_diffusion_2d(u, region2, spacing2, iterations, conductance, dt)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    fn step_image_1d(n: usize) -> Image<f32, 1> {
        let mut img = Image::<f32, 1>::allocate(Region::new([0], [n]), [1.0], [0.0], 0.0f32);
        for i in (n / 2) as i64..n as i64 {
            img.set_pixel(Index([i]), 100.0);
        }
        img
    }

    #[test]
    fn gradient_diffusion_smooths_step() {
        // Use conductance > step magnitude so diffusion crosses the edge.
        let img = step_image_1d(30);
        let f = GradientAnisotropicDiffusionFilter::new(img, 30, 200.0);
        let out = f.generate_region(f.largest_region());
        // After many iterations with large K, the step should be noticeably blurred.
        let mid = 15i64;
        let before = out.get_pixel(Index([mid - 3]));
        let after  = out.get_pixel(Index([mid + 3]));
        assert!(before < after, "step not present: {before} vs {after}");
        // At least one side should have moved away from its original extreme value.
        assert!(before > 1.0 || after < 99.0, "diffusion had no effect");
    }

    #[test]
    fn gradient_diffusion_constant_image_unchanged() {
        let img = Image::<f32, 2>::allocate(
            Region::new([0, 0], [10, 10]), [1.0; 2], [0.0; 2], 3.0f32,
        );
        let f = GradientAnisotropicDiffusionFilter::new(img, 10, 5.0);
        let out = f.generate_region(f.largest_region());
        for &v in &out.data {
            assert!((v - 3.0).abs() < 1e-5, "expected 3.0 got {v}");
        }
    }

    #[test]
    fn curvature_diffusion_2d_constant_unchanged() {
        let img = Image::<f32, 2>::allocate(
            Region::new([0, 0], [12, 12]), [1.0; 2], [0.0; 2], 2.0f32,
        );
        let f = CurvatureAnisotropicDiffusionFilter::new(img, 5, 3.0);
        let out = f.generate_region(f.largest_region());
        for &v in &out.data {
            assert!((v - 2.0).abs() < 0.01, "expected 2.0 got {v}");
        }
    }
}
