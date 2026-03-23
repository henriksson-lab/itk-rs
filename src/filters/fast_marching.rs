//! Fast Marching method filters.
//!
//! | Filter | ITK analog |
//! |---|---|
//! | [`FastMarchingFilter`] | `FastMarchingImageFilter` |
//! | [`FastMarchingUpwindGradientFilter`] | `FastMarchingUpwindGradientImageFilter` |

use std::collections::BinaryHeap;
use std::cmp::Ordering;

use crate::image::{Image, Index, Region, iter_region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

// Priority queue entry for the narrow band
#[derive(PartialEq)]
struct HeapEntry {
    distance: f64,
    index: usize,
}

impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller distance = higher priority
        other.distance.partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Fast Marching method: computes the arrival time of a front emanating from seed points.
/// Analog to `itk::FastMarchingImageFilter`.
///
/// The speed function is provided as a source image. Seeds are given as `(index, initial_value)`
/// pairs. Output is the travel time T(x) such that `||∇T|| = 1/F(x)` where F is the speed.
///
/// This implementation uses a simple eikonal solver (Sethian's FMM) with 4-connectivity
/// (or 6-connectivity in 3D) upwind finite differences.
pub struct FastMarchingFilter<SF, P> {
    pub speed: SF,
    /// Seed points: (index coordinates, initial value).
    pub seeds: Vec<([i64; 3], f64)>,
    /// Value assigned to pixels that are never reached.
    pub stopping_value: f64,
    _phantom: std::marker::PhantomData<P>,
}

impl<SF, P> FastMarchingFilter<SF, P> {
    pub fn new(speed: SF) -> Self {
        Self {
            speed,
            seeds: Vec::new(),
            stopping_value: f64::MAX,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn add_seed(&mut self, index: [i64; 3], value: f64) {
        self.seeds.push((index, value));
    }
}

// Implement for D=2 and D=3 via specialization on array size
impl<P, SF> ImageSource<f32, 2> for FastMarchingFilter<SF, P>
where
    P: NumericPixel,
    SF: ImageSource<P, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.speed.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.speed.spacing() }
    fn origin(&self) -> [f64; 2] { self.speed.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        let speed_img = self.speed.generate_region(self.speed.largest_region());
        let bounds = speed_img.region;
        let nx = bounds.size.0[0];
        let ny = bounds.size.0[1];
        let n = nx * ny;

        let mut dist = vec![f64::MAX; n];
        let mut frozen = vec![false; n];

        let flat2 = |x: i64, y: i64| -> usize {
            ((x - bounds.index.0[0]) as usize) + ((y - bounds.index.0[1]) as usize) * nx
        };

        let mut heap = BinaryHeap::new();

        // Initialize seeds
        for (seed_idx, val) in &self.seeds {
            let x = seed_idx[0];
            let y = seed_idx[1];
            if x >= bounds.index.0[0] && x < bounds.index.0[0] + nx as i64
                && y >= bounds.index.0[1] && y < bounds.index.0[1] + ny as i64
            {
                let flat = flat2(x, y);
                dist[flat] = *val;
                heap.push(HeapEntry { distance: *val, index: flat });
            }
        }

        while let Some(entry) = heap.pop() {
            let flat = entry.index;
            if frozen[flat] { continue; }
            frozen[flat] = true;

            if entry.distance > self.stopping_value { break; }

            // Compute x, y from flat index
            let xi = (flat % nx) as i64 + bounds.index.0[0];
            let yi = (flat / nx) as i64 + bounds.index.0[1];

            for (dx, dy) in [(-1i64,0),(1,0),(0,-1i64),(0,1)] {
                let nx_i = xi + dx;
                let ny_i = yi + dy;
                if nx_i < bounds.index.0[0] || nx_i >= bounds.index.0[0] + nx as i64
                    || ny_i < bounds.index.0[1] || ny_i >= bounds.index.0[1] + ny as i64
                { continue; }
                let nflat = flat2(nx_i, ny_i);
                if frozen[nflat] { continue; }

                // Upwind eikonal update
                let f = speed_img.get_pixel(Index([nx_i, ny_i])).to_f64().max(1e-10);
                let hx = speed_img.spacing[0];
                let hy = speed_img.spacing[1];

                // Upwind values along each axis
                let tx = {
                    let mut t = f64::MAX;
                    if nx_i > bounds.index.0[0] { t = t.min(dist[flat2(nx_i-1, ny_i)]); }
                    if nx_i < bounds.index.0[0] + nx as i64 - 1 { t = t.min(dist[flat2(nx_i+1, ny_i)]); }
                    t
                };
                let ty = {
                    let mut t = f64::MAX;
                    if ny_i > bounds.index.0[1] { t = t.min(dist[flat2(nx_i, ny_i-1)]); }
                    if ny_i < bounds.index.0[1] + ny as i64 - 1 { t = t.min(dist[flat2(nx_i, ny_i+1)]); }
                    t
                };

                // Solve 2D eikonal: min over possible solutions
                let new_t = solve_eikonal_2d(tx, ty, hx, hy, f);

                if new_t < dist[nflat] {
                    dist[nflat] = new_t;
                    heap.push(HeapEntry { distance: new_t, index: nflat });
                }
            }
        }

        // Build output
        let mut out_indices: Vec<Index<2>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<f32> = out_indices.iter().map(|&idx| {
            let flat = flat2(idx.0[0], idx.0[1]);
            let v = dist[flat];
            if v == f64::MAX { self.stopping_value as f32 } else { v as f32 }
        }).collect();

        Image { region: requested, spacing: speed_img.spacing, origin: speed_img.origin, data }
    }
}

fn solve_eikonal_2d(tx: f64, ty: f64, hx: f64, hy: f64, f: f64) -> f64 {
    let inv_f = 1.0 / f;
    if tx == f64::MAX && ty == f64::MAX { return f64::MAX; }
    if ty == f64::MAX { return tx + hx * inv_f; }
    if tx == f64::MAX { return ty + hy * inv_f; }
    // Solve: ((t-tx)/hx)^2 + ((t-ty)/hy)^2 = (1/f)^2
    let a = 1.0/(hx*hx) + 1.0/(hy*hy);
    let b = -2.0*(tx/(hx*hx) + ty/(hy*hy));
    let c = tx*tx/(hx*hx) + ty*ty/(hy*hy) - inv_f*inv_f;
    let disc = b*b - 4.0*a*c;
    if disc < 0.0 {
        // Fall back to 1D
        tx.min(ty) + hx.min(hy) * inv_f
    } else {
        (-b + disc.sqrt()) / (2.0 * a)
    }
}

/// Fast Marching Upwind Gradient filter.
/// Analog to `itk::FastMarchingUpwindGradientImageFilter`.
///
/// Computes the upwind gradient of the arrival time T. Output is a 2D image of
/// VecPixel<f32, 2> (gradient vector per pixel).
pub struct FastMarchingUpwindGradientFilter<SF, P> {
    pub speed: SF,
    pub seeds: Vec<([i64; 3], f64)>,
    pub stopping_value: f64,
    _phantom: std::marker::PhantomData<P>,
}

impl<SF, P> FastMarchingUpwindGradientFilter<SF, P> {
    pub fn new(speed: SF) -> Self {
        Self { speed, seeds: Vec::new(), stopping_value: f64::MAX, _phantom: std::marker::PhantomData }
    }
    pub fn add_seed(&mut self, index: [i64; 3], value: f64) {
        self.seeds.push((index, value));
    }
}

impl<P, SF> ImageSource<crate::pixel::VecPixel<f32, 2>, 2>
    for FastMarchingUpwindGradientFilter<SF, P>
where
    P: NumericPixel,
    SF: ImageSource<P, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.speed.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.speed.spacing() }
    fn origin(&self) -> [f64; 2] { self.speed.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<crate::pixel::VecPixel<f32, 2>, 2> {
        // First run FMM
        let fmm = FastMarchingFilter::<_, P> {
            speed: &self.speed,
            seeds: self.seeds.clone(),
            stopping_value: self.stopping_value,
            _phantom: std::marker::PhantomData,
        };
        let t_img = fmm.generate_region(self.speed.largest_region());
        let bounds = t_img.region;

        let mut out_indices: Vec<Index<2>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<crate::pixel::VecPixel<f32, 2>> = out_indices.iter().map(|&idx| {
            let [x, y] = idx.0;
            let hx = t_img.spacing[0];
            let hy = t_img.spacing[1];
            let clamp_x = |v: i64| v.max(bounds.index.0[0]).min(bounds.index.0[0] + bounds.size.0[0] as i64 - 1);
            let clamp_y = |v: i64| v.max(bounds.index.0[1]).min(bounds.index.0[1] + bounds.size.0[1] as i64 - 1);
            let gx = (t_img.get_pixel(Index([clamp_x(x+1), y])) - t_img.get_pixel(Index([clamp_x(x-1), y]))) / (2.0 * hx as f32);
            let gy = (t_img.get_pixel(Index([x, clamp_y(y+1)])) - t_img.get_pixel(Index([x, clamp_y(y-1)]))) / (2.0 * hy as f32);
            crate::pixel::VecPixel([gx, gy])
        }).collect();

        Image { region: requested, spacing: t_img.spacing, origin: t_img.origin, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    #[test]
    fn fast_marching_from_center() {
        // Constant speed=1, seed at center → circular wavefront
        let speed = Image::<f32, 2>::allocate(
            Region::new([0, 0], [11, 11]), [1.0, 1.0], [0.0, 0.0], 1.0f32,
        );
        let mut fmm = FastMarchingFilter::<_, f32>::new(speed);
        fmm.add_seed([5, 5, 0], 0.0);
        let out = fmm.generate_region(fmm.largest_region());
        // Distance from center to (5,5) = 0
        let v = out.get_pixel(Index([5, 5]));
        assert!(v.abs() < 0.1, "expected 0 got {v}");
        // Distance from center to (8,5) ≈ 3
        let v2 = out.get_pixel(Index([8, 5]));
        assert!((v2 - 3.0).abs() < 0.5, "expected ~3 got {v2}");
    }
}
