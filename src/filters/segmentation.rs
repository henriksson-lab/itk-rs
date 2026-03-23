//! Segmentation filters.
//!
//! | Filter | ITK analog |
//! |---|---|
//! | [`ConnectedThresholdFilter`]     | `ConnectedThresholdImageFilter` |
//! | [`ConfidenceConnectedFilter`]    | `ConfidenceConnectedImageFilter` |
//! | [`NeighborhoodConnectedFilter`]  | `NeighborhoodConnectedImageFilter` |
//! | [`OtsuMultipleThresholdSegFilter`] | simple Otsu-based multi-class |

use std::collections::VecDeque;

use crate::image::{Image, Index, Region, iter_region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

// ===========================================================================
// ConnectedThresholdImageFilter
// ===========================================================================

/// Region growing: flood-fills pixels whose values lie in [lower, upper].
/// Analog to `itk::ConnectedThresholdImageFilter`.
pub struct ConnectedThresholdFilter<S> {
    pub source: S,
    pub seeds: Vec<[i64; 3]>,
    pub lower: f64,
    pub upper: f64,
    pub replace_value: f64,
}

impl<S> ConnectedThresholdFilter<S> {
    pub fn new(source: S, lower: f64, upper: f64) -> Self {
        Self { source, seeds: Vec::new(), lower, upper, replace_value: 1.0 }
    }
    pub fn add_seed(&mut self, index: [i64; 3]) { self.seeds.push(index); }
}

impl<P, S, const D: usize> ImageSource<P, D> for ConnectedThresholdFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(self.source.largest_region());
        let bounds = input.region;
        let n = bounds.linear_len();
        let mut output = vec![0.0f64; n];

        let flat = |idx: &[i64; D]| -> usize {
            let mut f = 0usize;
            let mut stride = 1usize;
            for d in 0..D {
                f += (idx[d] - bounds.index.0[d]) as usize * stride;
                stride *= bounds.size.0[d];
            }
            f
        };

        let mut visited = vec![false; n];
        let mut queue: VecDeque<[i64; D]> = VecDeque::new();

        for seed in &self.seeds {
            let mut idx = [0i64; D];
            for d in 0..D { idx[d] = seed[d]; }
            if (0..D).all(|d| idx[d] >= bounds.index.0[d] && idx[d] < bounds.index.0[d] + bounds.size.0[d] as i64) {
                let f = flat(&idx);
                if !visited[f] {
                    visited[f] = true;
                    queue.push_back(idx);
                }
            }
        }

        while let Some(idx) = queue.pop_front() {
            let v = input.get_pixel(Index(idx)).to_f64();
            if v < self.lower || v > self.upper { continue; }
            let f = flat(&idx);
            output[f] = self.replace_value;

            for d in 0..D {
                for delta in [-1i64, 1i64] {
                    let mut nb = idx;
                    nb[d] += delta;
                    if (0..D).all(|dd| nb[dd] >= bounds.index.0[dd] && nb[dd] < bounds.index.0[dd] + bounds.size.0[dd] as i64) {
                        let nf = flat(&nb);
                        if !visited[nf] {
                            visited[nf] = true;
                            queue.push_back(nb);
                        }
                    }
                }
            }
        }

        // Extract requested region
        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<P> = out_indices.iter().map(|&idx| {
            let v = if (0..D).all(|d| idx.0[d] >= bounds.index.0[d] && idx.0[d] < bounds.index.0[d] + bounds.size.0[d] as i64) {
                output[flat(&idx.0)]
            } else { 0.0 };
            P::from_f64(v)
        }).collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// ConfidenceConnectedImageFilter
// ===========================================================================

/// Confidence-connected region growing: adapts threshold from neighbourhood stats.
/// Analog to `itk::ConfidenceConnectedImageFilter`.
///
/// Iteratively: compute mean±multiplier*sigma in currently labelled region,
/// then re-grow from seeds with that threshold.
pub struct ConfidenceConnectedFilter<S> {
    pub source: S,
    pub seeds: Vec<[i64; 3]>,
    pub multiplier: f64,
    pub iterations: usize,
    pub initial_radius: usize,
    pub replace_value: f64,
}

impl<S> ConfidenceConnectedFilter<S> {
    pub fn new(source: S) -> Self {
        Self {
            source, seeds: Vec::new(), multiplier: 2.5,
            iterations: 4, initial_radius: 1, replace_value: 1.0,
        }
    }
    pub fn add_seed(&mut self, index: [i64; 3]) { self.seeds.push(index); }
}

impl<P, S, const D: usize> ImageSource<P, D> for ConfidenceConnectedFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(self.source.largest_region());
        let bounds = input.region;

        // Compute initial statistics from the initial radius neighbourhood around seeds
        let r = self.initial_radius as i64;
        let mut vals: Vec<f64> = Vec::new();
        for seed in &self.seeds {
            let mut nb = [0i64; D];
            for d in 0..D { nb[d] = -r; }
            loop {
                let mut ok = true;
                let mut idx = [0i64; D];
                for d in 0..D {
                    idx[d] = seed[d] + nb[d];
                    if idx[d] < bounds.index.0[d] || idx[d] >= bounds.index.0[d] + bounds.size.0[d] as i64 { ok = false; break; }
                }
                if ok { vals.push(input.get_pixel(Index(idx)).to_f64()); }
                let mut carry = true;
                for d in 0..D {
                    if carry { nb[d] += 1; if nb[d] > r { nb[d] = -r; } else { carry = false; } }
                }
                if carry { break; }
            }
        }

        let (mut lower, mut upper) = if vals.is_empty() {
            (0.0, 0.0)
        } else {
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|&v| (v-mean)*(v-mean)).sum::<f64>() / vals.len() as f64;
            let sigma = var.sqrt();
            (mean - self.multiplier * sigma, mean + self.multiplier * sigma)
        };

        let mut result_data = vec![0.0f64; bounds.linear_len()];
        let flat = |idx: [i64; D]| -> usize {
            let mut f = 0usize; let mut stride = 1usize;
            for d in 0..D { f += (idx[d] - bounds.index.0[d]) as usize * stride; stride *= bounds.size.0[d]; }
            f
        };

        for _iter in 0..=self.iterations {
            // Grow region with current thresholds
            result_data.fill(0.0);
            let mut visited = vec![false; bounds.linear_len()];
            let mut queue: VecDeque<[i64; D]> = VecDeque::new();
            for seed in &self.seeds {
                let mut idx = [0i64; D];
                for d in 0..D { idx[d] = seed[d]; }
                let f = flat(idx);
                if !visited[f] { visited[f] = true; queue.push_back(idx); }
            }
            while let Some(idx) = queue.pop_front() {
                let v = input.get_pixel(Index(idx)).to_f64();
                if v < lower || v > upper { continue; }
                result_data[flat(idx)] = self.replace_value;
                for d in 0..D {
                    for delta in [-1i64, 1i64] {
                        let mut nb = idx; nb[d] += delta;
                        if (0..D).all(|dd| nb[dd] >= bounds.index.0[dd] && nb[dd] < bounds.index.0[dd] + bounds.size.0[dd] as i64) {
                            let nf = flat(nb);
                            if !visited[nf] { visited[nf] = true; queue.push_back(nb); }
                        }
                    }
                }
            }

            // Update statistics from labelled region
            let mut region_vals: Vec<f64> = Vec::new();
            iter_region(&bounds, |idx| {
                if result_data[flat(idx.0)] > 0.5 { region_vals.push(input.get_pixel(idx).to_f64()); }
            });
            if region_vals.is_empty() { break; }
            let mean = region_vals.iter().sum::<f64>() / region_vals.len() as f64;
            let var = region_vals.iter().map(|&v| (v-mean)*(v-mean)).sum::<f64>() / region_vals.len() as f64;
            let sigma = var.sqrt();
            lower = mean - self.multiplier * sigma;
            upper = mean + self.multiplier * sigma;
        }

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<P> = out_indices.iter().map(|&idx| {
            let v = if (0..D).all(|d| idx.0[d] >= bounds.index.0[d] && idx.0[d] < bounds.index.0[d] + bounds.size.0[d] as i64) {
                result_data[flat(idx.0)]
            } else { 0.0 };
            P::from_f64(v)
        }).collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// NeighborhoodConnectedImageFilter
// ===========================================================================

/// Neighbourhood-connected: pixel is included if ALL pixels in its
/// neighbourhood are within [lower, upper].
/// Analog to `itk::NeighborhoodConnectedImageFilter`.
pub struct NeighborhoodConnectedFilter<S> {
    pub source: S,
    pub seeds: Vec<[i64; 3]>,
    pub lower: f64,
    pub upper: f64,
    pub radius: usize,
    pub replace_value: f64,
}

impl<S> NeighborhoodConnectedFilter<S> {
    pub fn new(source: S, lower: f64, upper: f64) -> Self {
        Self { source, seeds: Vec::new(), lower, upper, radius: 1, replace_value: 1.0 }
    }
    pub fn add_seed(&mut self, index: [i64; 3]) { self.seeds.push(index); }
}

impl<P, S, const D: usize> ImageSource<P, D> for NeighborhoodConnectedFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(self.source.largest_region());
        let bounds = input.region;
        let n = bounds.linear_len();
        let flat = |idx: [i64; D]| -> usize {
            let mut f = 0usize; let mut stride = 1usize;
            for d in 0..D { f += (idx[d] - bounds.index.0[d]) as usize * stride; stride *= bounds.size.0[d]; }
            f
        };

        // A pixel passes if ALL pixels in its radius neighbourhood are in [lower, upper]
        let passes = |idx: [i64; D]| -> bool {
            let r = self.radius as i64;
            let mut nb = [0i64; D];
            for d in 0..D { nb[d] = -r; }
            loop {
                let mut s = [0i64; D];
                for d in 0..D {
                    s[d] = (idx[d] + nb[d]).max(bounds.index.0[d])
                        .min(bounds.index.0[d] + bounds.size.0[d] as i64 - 1);
                }
                let v = input.get_pixel(Index(s)).to_f64();
                if v < self.lower || v > self.upper { return false; }
                let mut carry = true;
                for d in 0..D {
                    if carry { nb[d] += 1; if nb[d] > r { nb[d] = -r; } else { carry = false; } }
                }
                if carry { break; }
            }
            true
        };

        let mut output = vec![0.0f64; n];
        let mut visited = vec![false; n];
        let mut queue: VecDeque<[i64; D]> = VecDeque::new();

        for seed in &self.seeds {
            let mut idx = [0i64; D];
            for d in 0..D { idx[d] = seed[d]; }
            if (0..D).all(|d| idx[d] >= bounds.index.0[d] && idx[d] < bounds.index.0[d] + bounds.size.0[d] as i64) {
                let f = flat(idx);
                if !visited[f] { visited[f] = true; queue.push_back(idx); }
            }
        }

        while let Some(idx) = queue.pop_front() {
            if !passes(idx) { continue; }
            output[flat(idx)] = self.replace_value;
            for d in 0..D {
                for delta in [-1i64, 1i64] {
                    let mut nb = idx; nb[d] += delta;
                    if (0..D).all(|dd| nb[dd] >= bounds.index.0[dd] && nb[dd] < bounds.index.0[dd] + bounds.size.0[dd] as i64) {
                        let nf = flat(nb);
                        if !visited[nf] { visited[nf] = true; queue.push_back(nb); }
                    }
                }
            }
        }

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<P> = out_indices.iter().map(|&idx| {
            let v = if (0..D).all(|d| idx.0[d] >= bounds.index.0[d] && idx.0[d] < bounds.index.0[d] + bounds.size.0[d] as i64) {
                output[flat(idx.0)]
            } else { 0.0 };
            P::from_f64(v)
        }).collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    fn constant_region(val: f32, n: usize) -> Image<f32, 1> {
        Image::allocate(Region::new([0], [n]), [1.0], [0.0], val)
    }

    #[test]
    fn connected_threshold_1d() {
        // Image: 0 0 0 1 1 1 0 0 0; threshold [0.5, 1.5]; seed at 4
        let mut img = constant_region(0.0, 9);
        for i in 3..6i64 { img.set_pixel(Index([i]), 1.0f32); }
        let mut f = ConnectedThresholdFilter::new(img, 0.5, 1.5);
        f.add_seed([4, 0, 0]);
        let out = f.generate_region(f.largest_region());
        // Only indices 3,4,5 should be labeled
        for i in 0..9i64 {
            let v = out.get_pixel(Index([i]));
            let expected = if i >= 3 && i <= 5 { 1.0 } else { 0.0 };
            assert!((v - expected).abs() < 0.1, "at {i}: got {v} expected {expected}");
        }
    }
}
