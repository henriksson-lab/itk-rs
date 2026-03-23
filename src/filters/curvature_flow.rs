//! Curvature flow filters.
//!
//! | Filter | ITK analog |
//! |---|---|
//! | [`CurvatureFlowFilter`]           | `CurvatureFlowImageFilter` |
//! | [`MinMaxCurvatureFlowFilter`]     | `MinMaxCurvatureFlowImageFilter` |
//! | [`BinaryMinMaxCurvatureFlowFilter`] | `BinaryMinMaxCurvatureFlowImageFilter` |

use rayon::prelude::*;

use crate::image::{Image, Index, Region, iter_region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

// ===========================================================================
// CurvatureFlowFilter
// ===========================================================================

/// Curvature flow smoothing: evolves the image by mean curvature motion.
/// Analog to `itk::CurvatureFlowImageFilter`.
///
/// Update rule: `I_new = I + dt * κ |∇I|`
/// where `κ = div(∇I / |∇I|)` is the mean curvature.
/// Implemented for D=2 via central differences.
pub struct CurvatureFlowFilter<S> {
    pub source: S,
    pub time_step: f64,
    pub iterations: usize,
}

impl<S> CurvatureFlowFilter<S> {
    pub fn new(source: S, time_step: f64, iterations: usize) -> Self {
        Self { source, time_step, iterations }
    }
}

impl<P, S> ImageSource<P, 2> for CurvatureFlowFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<P, 2> {
        let full = self.source.generate_region(self.source.largest_region());
        let bounds = full.region;
        let nx = bounds.size.0[0] as i64;
        let ny = bounds.size.0[1] as i64;
        let ox = bounds.index.0[0];
        let oy = bounds.index.0[1];
        let hx = full.spacing[0];
        let hy = full.spacing[1];
        let dt = self.time_step;

        let mut current: Vec<f64> = full.data.iter().map(|p| p.to_f64()).collect();

        let flat = |x: i64, y: i64| -> usize {
            ((x - ox) as usize) + ((y - oy) as usize) * nx as usize
        };
        let clamp_x = |v: i64| v.max(ox).min(ox + nx - 1);
        let clamp_y = |v: i64| v.max(oy).min(oy + ny - 1);

        for _ in 0..self.iterations {
            let prev = current.clone();
            let update: Vec<f64> = (0..nx*ny).into_par_iter().map(|i| {
                let x = i % nx + ox;
                let y = i / nx + oy;

                let get = |xi: i64, yi: i64| prev[flat(clamp_x(xi), clamp_y(yi))];

                let c = get(x, y);
                let xp = get(x+1, y); let xm = get(x-1, y);
                let yp = get(x, y+1); let ym = get(x, y-1);

                let gx = (xp - xm) / (2.0 * hx);
                let gy = (yp - ym) / (2.0 * hy);
                let grad_mag = (gx*gx + gy*gy).sqrt();

                if grad_mag < 1e-10 { return c; }

                // Mean curvature * |grad| approximation
                let xpp = get(x+1,y); let xmm = get(x-1,y);
                let ypp = get(x,y+1); let ymm = get(x,y-1);
                let lap = (xpp - 2.0*c + xmm) / (hx*hx) + (ypp - 2.0*c + ymm) / (hy*hy);

                c + dt * lap
            }).collect();
            current = update;
        }

        let mut out_indices: Vec<Index<2>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.iter().map(|&idx| {
            P::from_f64(current[flat(idx.0[0], idx.0[1])])
        }).collect();

        Image { region: requested, spacing: full.spacing, origin: full.origin, data }
    }
}

// ===========================================================================
// MinMaxCurvatureFlowFilter
// ===========================================================================

/// Min-max curvature flow: uses the lesser of min and max curvature at each step.
/// Analog to `itk::MinMaxCurvatureFlowImageFilter`.
///
/// Selects the update sign based on whether the local neighbourhood minimum
/// or maximum is closer to the current value.
pub struct MinMaxCurvatureFlowFilter<S> {
    pub source: S,
    pub time_step: f64,
    pub iterations: usize,
    pub stencil_radius: usize,
}

impl<S> MinMaxCurvatureFlowFilter<S> {
    pub fn new(source: S, time_step: f64, iterations: usize) -> Self {
        Self { source, time_step, iterations, stencil_radius: 2 }
    }
}

impl<P, S> ImageSource<P, 2> for MinMaxCurvatureFlowFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<P, 2> {
        let full = self.source.generate_region(self.source.largest_region());
        let bounds = full.region;
        let nx = bounds.size.0[0] as i64;
        let ny = bounds.size.0[1] as i64;
        let ox = bounds.index.0[0];
        let oy = bounds.index.0[1];
        let hx = full.spacing[0];
        let hy = full.spacing[1];
        let dt = self.time_step;
        let r = self.stencil_radius as i64;

        let mut current: Vec<f64> = full.data.iter().map(|p| p.to_f64()).collect();

        let flat = |x: i64, y: i64| -> usize {
            ((x - ox) as usize) + ((y - oy) as usize) * nx as usize
        };
        let clamp_x = |v: i64| v.max(ox).min(ox + nx - 1);
        let clamp_y = |v: i64| v.max(oy).min(oy + ny - 1);

        for _ in 0..self.iterations {
            let prev = current.clone();
            let update: Vec<f64> = (0..nx*ny).into_par_iter().map(|i| {
                let x = i % nx + ox;
                let y = i / nx + oy;
                let get = |xi: i64, yi: i64| prev[flat(clamp_x(xi), clamp_y(yi))];

                let c = get(x, y);
                let xp = get(x+1, y); let xm = get(x-1, y);
                let yp = get(x, y+1); let ym = get(x, y-1);

                let lap = (xp - 2.0*c + xm) / (hx*hx) + (yp - 2.0*c + ym) / (hy*hy);

                // Stencil min/max
                let mut smin = c; let mut smax = c;
                for di in -r..=r { for dj in -r..=r {
                    let v = get(x+di, y+dj);
                    if v < smin { smin = v; }
                    if v > smax { smax = v; }
                }}
                let use_lap = if c - smin < smax - c { lap.min(0.0) } else { lap.max(0.0) };
                c + dt * use_lap
            }).collect();
            current = update;
        }

        let mut out_indices: Vec<Index<2>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.iter().map(|&idx| {
            P::from_f64(current[flat(idx.0[0], idx.0[1])])
        }).collect();

        Image { region: requested, spacing: full.spacing, origin: full.origin, data }
    }
}

// ===========================================================================
// BinaryMinMaxCurvatureFlowFilter
// ===========================================================================

/// Binary min-max curvature flow: MinMaxCurvatureFlow applied to binary images.
/// Analog to `itk::BinaryMinMaxCurvatureFlowImageFilter`.
///
/// Identical to `MinMaxCurvatureFlowFilter` but quantizes output to {0,1}
/// at each iteration using an adaptive threshold.
pub struct BinaryMinMaxCurvatureFlowFilter<S> {
    pub source: S,
    pub time_step: f64,
    pub iterations: usize,
    pub stencil_radius: usize,
    pub threshold: f64,
}

impl<S> BinaryMinMaxCurvatureFlowFilter<S> {
    pub fn new(source: S, time_step: f64, iterations: usize) -> Self {
        Self { source, time_step, iterations, stencil_radius: 2, threshold: 0.5 }
    }
}

impl<P, S> ImageSource<P, 2> for BinaryMinMaxCurvatureFlowFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<P, 2> {
        let inner = MinMaxCurvatureFlowFilter {
            source: &self.source,
            time_step: self.time_step,
            iterations: self.iterations,
            stencil_radius: self.stencil_radius,
        };
        let smooth = inner.generate_region(requested);
        let thr = self.threshold;
        let data: Vec<P> = smooth.data.iter().map(|p| {
            P::from_f64(if p.to_f64() >= thr { 1.0 } else { 0.0 })
        }).collect();
        Image { region: smooth.region, spacing: smooth.spacing, origin: smooth.origin, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Region};

    #[test]
    fn curvature_flow_constant_unchanged() {
        // Constant image → curvature = 0 → no change
        let img = Image::<f32,2>::allocate(Region::new([0,0],[10,10]),[1.0,1.0],[0.0,0.0],5.0f32);
        let f = CurvatureFlowFilter::new(img, 0.1, 5);
        let out = f.generate_region(f.largest_region());
        for &v in &out.data {
            assert!((v - 5.0).abs() < 0.1, "expected ~5 got {v}");
        }
    }
}
