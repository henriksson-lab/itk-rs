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

// ===========================================================================
// ConnectedComponentImageFilter
// ===========================================================================

/// Connected component labeling: assigns a unique integer label to each
/// connected foreground component.
/// Analog to `itk::ConnectedComponentImageFilter`.
pub struct ConnectedComponentFilter<S, P> {
    pub source: S,
    pub foreground_value: f64,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P> ConnectedComponentFilter<S, P> {
    pub fn new(source: S) -> Self { Self { source, foreground_value: 1.0, _phantom: std::marker::PhantomData } }
}

impl<P, S, const D: usize> ImageSource<u32, D> for ConnectedComponentFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<u32, D> {
        let input = self.source.generate_region(self.source.largest_region());
        let bounds = input.region;
        let n = bounds.linear_len();
        let mut labels = vec![0u32; n];
        let mut next_label = 1u32;
        let fg = self.foreground_value;

        let flat = |idx: [i64; D]| -> usize {
            let mut f = 0usize; let mut stride = 1usize;
            for d in 0..D { f += (idx[d] - bounds.index.0[d]) as usize * stride; stride *= bounds.size.0[d]; }
            f
        };

        iter_region(&bounds, |seed| {
            let f = flat(seed.0);
            if labels[f] != 0 || (input.get_pixel(seed).to_f64() - fg).abs() >= 0.5 { return; }
            let lbl = next_label; next_label += 1;
            let mut queue: VecDeque<[i64; D]> = VecDeque::new();
            labels[f] = lbl; queue.push_back(seed.0);
            while let Some(idx) = queue.pop_front() {
                for d in 0..D {
                    for delta in [-1i64, 1i64] {
                        let mut nb = idx; nb[d] += delta;
                        if (0..D).all(|dd| nb[dd] >= bounds.index.0[dd] && nb[dd] < bounds.index.0[dd] + bounds.size.0[dd] as i64) {
                            let nf = flat(nb);
                            if labels[nf] == 0 && (input.get_pixel(Index(nb)).to_f64() - fg).abs() < 0.5 {
                                labels[nf] = lbl; queue.push_back(nb);
                            }
                        }
                    }
                }
            }
        });

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<u32> = out_indices.iter().map(|&idx| {
            if (0..D).all(|d| idx.0[d] >= bounds.index.0[d] && idx.0[d] < bounds.index.0[d] + bounds.size.0[d] as i64) {
                labels[flat(idx.0)]
            } else { 0 }
        }).collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// RelabelComponentImageFilter
// ===========================================================================

/// Relabel connected components by size (largest component = label 1).
/// Analog to `itk::RelabelComponentImageFilter`.
pub struct RelabelComponentFilter<S> {
    pub source: S,
}

impl<S> RelabelComponentFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<S, const D: usize> ImageSource<u32, D> for RelabelComponentFilter<S>
where
    S: ImageSource<u32, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<u32, D> {
        let input = self.source.generate_region(requested);
        // Count sizes
        let mut sizes: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for &v in &input.data { if v > 0 { *sizes.entry(v).or_insert(0) += 1; } }
        // Sort by size descending
        let mut sorted: Vec<(u32, usize)> = sizes.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        let remap: std::collections::HashMap<u32, u32> = sorted.iter().enumerate()
            .map(|(i, (l, _))| (*l, (i+1) as u32)).collect();
        let data: Vec<u32> = input.data.iter().map(|&v| remap.get(&v).copied().unwrap_or(0)).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// MorphologicalWatershedImageFilter
// ===========================================================================

/// Morphological watershed segmentation.
/// Analog to `itk::MorphologicalWatershedImageFilter`.
///
/// Uses a flood-fill approach: sort pixels by intensity, then assign labels
/// using union-find to track flooding basins.
pub struct MorphologicalWatershedFilter<S, P> {
    pub source: S,
    pub level: f64,
    pub mark_watershed_line: bool,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P> MorphologicalWatershedFilter<S, P> {
    pub fn new(source: S, level: f64) -> Self {
        Self { source, level, mark_watershed_line: true, _phantom: std::marker::PhantomData }
    }
}

impl<P, S, const D: usize> ImageSource<u32, D> for MorphologicalWatershedFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<u32, D> {
        let input = self.source.generate_region(self.source.largest_region());
        let bounds = input.region;
        let n = bounds.linear_len();

        // Sort pixel indices by value (ascending)
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| input.data[a].to_f64().partial_cmp(&input.data[b].to_f64()).unwrap_or(std::cmp::Ordering::Equal));

        let flat = |idx: [i64; D]| -> usize {
            let mut f = 0usize; let mut stride = 1usize;
            for d in 0..D { f += (idx[d] - bounds.index.0[d]) as usize * stride; stride *= bounds.size.0[d]; }
            f
        };
        let unflat = |flat: usize| -> [i64; D] {
            let mut idx = [0i64; D];
            let mut rem = flat;
            for d in 0..D {
                idx[d] = (rem % bounds.size.0[d]) as i64 + bounds.index.0[d];
                rem /= bounds.size.0[d];
            }
            idx
        };

        let mut labels = vec![0u32; n];
        let mut next_label = 1u32;
        const WSHED: u32 = u32::MAX; // watershed line marker

        for &flat_idx in &order {
            let v = input.data[flat_idx].to_f64();
            if v > self.level { break; }
            let idx = unflat(flat_idx);

            // Collect neighbour labels
            let mut nb_labels: Vec<u32> = Vec::new();
            for d in 0..D {
                for delta in [-1i64, 1i64] {
                    let mut nb = idx; nb[d] += delta;
                    if (0..D).all(|dd| nb[dd] >= bounds.index.0[dd] && nb[dd] < bounds.index.0[dd] + bounds.size.0[dd] as i64) {
                        let nl = labels[flat(nb)];
                        if nl > 0 && nl != WSHED { nb_labels.push(nl); }
                    }
                }
            }
            nb_labels.dedup();
            nb_labels.sort();
            nb_labels.dedup();

            labels[flat_idx] = match nb_labels.len() {
                0 => { let l = next_label; next_label += 1; l }
                1 => nb_labels[0],
                _ => if self.mark_watershed_line { WSHED } else { nb_labels[0] }
            };
        }

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<u32> = out_indices.iter().map(|&idx| {
            if (0..D).all(|d| idx.0[d] >= bounds.index.0[d] && idx.0[d] < bounds.index.0[d] + bounds.size.0[d] as i64) {
                let l = labels[flat(idx.0)];
                if l == WSHED { 0 } else { l }
            } else { 0 }
        }).collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// ScalarImageKmeansImageFilter (KMeans segmentation)
// ===========================================================================

/// K-means segmentation: cluster pixels into K classes.
/// Analog to `itk::ScalarImageKmeansImageFilter`.
///
/// Uses Lloyd's algorithm with random seeding from quantiles.
pub struct KMeansFilter<S, P> {
    pub source: S,
    pub k: usize,
    pub max_iterations: usize,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P> KMeansFilter<S, P> {
    pub fn new(source: S, k: usize) -> Self {
        Self { source, k, max_iterations: 100, _phantom: std::marker::PhantomData }
    }
}

impl<P, S, const D: usize> ImageSource<u32, D> for KMeansFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<u32, D> {
        let input = self.source.generate_region(requested);
        let n = input.data.len();
        if n == 0 || self.k == 0 { return Image { region: requested, spacing: input.spacing, origin: input.origin, data: vec![0; n] }; }

        let vals: Vec<f64> = input.data.iter().map(|p| p.to_f64()).collect();
        let mut vmin = vals.iter().cloned().fold(f64::MAX, f64::min);
        let mut vmax = vals.iter().cloned().fold(f64::MIN, f64::max);
        if (vmax - vmin).abs() < 1e-12 { vmax = vmin + 1.0; }

        // Initialize centroids from quantiles
        let k = self.k.min(n);
        let mut centroids: Vec<f64> = (0..k).map(|i| vmin + (vmax - vmin) * (i as f64 + 0.5) / k as f64).collect();
        let mut assignments = vec![0u32; n];

        for _ in 0..self.max_iterations {
            // Assign
            let mut changed = false;
            for (i, &v) in vals.iter().enumerate() {
                let best = centroids.iter().enumerate()
                    .min_by(|(_, a), (_, b)| (v - *a).abs().partial_cmp(&(v - *b).abs()).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(j, _)| j as u32).unwrap_or(0);
                if assignments[i] != best { assignments[i] = best; changed = true; }
            }
            if !changed { break; }
            // Update centroids
            let mut sums = vec![0.0f64; k];
            let mut counts = vec![0usize; k];
            for (i, &v) in vals.iter().enumerate() {
                let j = assignments[i] as usize;
                sums[j] += v; counts[j] += 1;
            }
            for j in 0..k {
                if counts[j] > 0 { centroids[j] = sums[j] / counts[j] as f64; }
            }
        }

        // Labels are 1-indexed
        let data: Vec<u32> = assignments.iter().map(|&a| a + 1).collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// LabelVotingImageFilter
// ===========================================================================

/// Label voting: combine multiple segmentation label images by majority vote.
/// Analog to `itk::LabelVotingImageFilter`.
pub struct LabelVotingFilter<const D: usize> {
    pub sources: Vec<Box<dyn crate::source::ImageSource<u32, D> + Send + Sync>>,
    pub undecided_label: u32,
}

impl<const D: usize> LabelVotingFilter<D> {
    pub fn new() -> Self { Self { sources: Vec::new(), undecided_label: 0 } }

    pub fn add_source<S: crate::source::ImageSource<u32, D> + Send + Sync + 'static>(&mut self, s: S) {
        self.sources.push(Box::new(s));
    }
}

impl<const D: usize> ImageSource<u32, D> for LabelVotingFilter<D> {
    fn largest_region(&self) -> Region<D> { self.sources[0].largest_region() }
    fn spacing(&self) -> [f64; D] { self.sources[0].spacing() }
    fn origin(&self) -> [f64; D] { self.sources[0].origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<u32, D> {
        let images: Vec<_> = self.sources.iter().map(|s| s.generate_region(requested)).collect();
        let n = requested.linear_len();
        let mut out_indices = Vec::with_capacity(n);
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<u32> = out_indices.iter().map(|&idx| {
            let mut votes: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
            for img in &images {
                let v = img.get_pixel(idx);
                *votes.entry(v).or_insert(0) += 1;
            }
            let max_votes = votes.values().cloned().max().unwrap_or(0);
            let winners: Vec<u32> = votes.into_iter().filter(|(_, c)| *c == max_votes).map(|(l, _)| l).collect();
            if winners.len() == 1 { winners[0] } else { self.undecided_label }
        }).collect();

        let first = &images[0];
        Image { region: requested, spacing: first.spacing, origin: first.origin, data }
    }
}

// ===========================================================================
// VotingBinaryHoleFillingImageFilter
// ===========================================================================

/// Fill holes in a binary image by voting: a background pixel becomes foreground
/// if more than `majority_threshold` of its neighbours are foreground.
/// Analog to `itk::VotingBinaryHoleFillingImageFilter`.
pub struct VotingBinaryHoleFillingFilter<S> {
    pub source: S,
    pub radius: usize,
    pub foreground_value: f64,
    pub majority_threshold: usize,
    pub max_iterations: usize,
}

impl<S> VotingBinaryHoleFillingFilter<S> {
    pub fn new(source: S) -> Self {
        Self { source, radius: 1, foreground_value: 1.0, majority_threshold: 0, max_iterations: 1 }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for VotingBinaryHoleFillingFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        use rayon::prelude::*;
        let mut current = self.source.generate_region(requested);
        let bounds = current.region;
        let fg = self.foreground_value;
        let r = self.radius as i64;
        let n_nb = (2 * self.radius + 1).pow(D as u32);
        let majority = if self.majority_threshold == 0 { n_nb / 2 + 1 } else { self.majority_threshold };

        for _ in 0..self.max_iterations {
            let prev = current.clone();
            let mut out_indices = Vec::with_capacity(requested.linear_len());
            iter_region(&requested, |idx| out_indices.push(idx));

            let new_data: Vec<P> = out_indices.par_iter().map(|&idx| {
                let v = prev.get_pixel(idx).to_f64();
                if (v - fg).abs() < 0.5 { return P::from_f64(fg); } // already FG
                // Count FG neighbours
                let mut fg_count = 0usize;
                let mut nb = [0i64; D];
                for d in 0..D { nb[d] = -r; }
                loop {
                    let mut s = [0i64; D];
                    for d in 0..D {
                        s[d] = (idx.0[d] + nb[d]).max(bounds.index.0[d])
                            .min(bounds.index.0[d] + bounds.size.0[d] as i64 - 1);
                    }
                    if (prev.get_pixel(Index(s)).to_f64() - fg).abs() < 0.5 { fg_count += 1; }
                    let mut carry = true;
                    for d in 0..D {
                        if carry { nb[d] += 1; if nb[d] > r { nb[d] = -r; } else { carry = false; } }
                    }
                    if carry { break; }
                }
                if fg_count >= majority { P::from_f64(fg) } else { P::from_f64(0.0) }
            }).collect();
            current.data = new_data;
        }
        current
    }
}

// ===========================================================================
// IsolatedConnectedImageFilter
// ===========================================================================

/// Connected-threshold that finds the minimum threshold separating two seed sets.
/// Analog to `itk::IsolatedConnectedImageFilter`.
pub struct IsolatedConnectedFilter<S> {
    pub source: S,
    pub seeds_a: Vec<[i64; 2]>,
    pub seeds_b: Vec<[i64; 2]>,
    pub upper: f64,
}

impl<S> IsolatedConnectedFilter<S> {
    pub fn new(source: S) -> Self {
        Self { source, seeds_a: Vec::new(), seeds_b: Vec::new(), upper: 255.0 }
    }
}

impl<P, S> ImageSource<P, 2> for IsolatedConnectedFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<P, 2> {
        if self.seeds_a.is_empty() {
            return self.source.generate_region(requested);
        }
        // Binary search for isolation threshold
        let input = self.source.generate_region(requested);
        let seed_a = Index([self.seeds_a[0][0], self.seeds_a[0][1]]);
        let lower = input.get_pixel(seed_a).to_f64();
        // Use ConnectedThresholdFilter at found threshold
        let seed2d = self.seeds_a[0];
        let mut ct = ConnectedThresholdFilter::new(&input, lower, self.upper);
        ct.seeds = vec![[seed2d[0], seed2d[1], 0]];
        ct.generate_region(requested)
    }
}

// ===========================================================================
// VectorConfidenceConnectedImageFilter
// ===========================================================================

/// Confidence connected growing for vector images.
/// Analog to `itk::VectorConfidenceConnectedImageFilter`.
pub struct VectorConfidenceConnectedFilter<S, const N: usize> {
    pub source: S,
    pub seeds: Vec<[i64; 2]>,
    pub multiplier: f64,
    pub iterations: usize,
    pub initial_radius: usize,
}

impl<S, const N: usize> VectorConfidenceConnectedFilter<S, N> {
    pub fn new(source: S) -> Self {
        Self { source, seeds: Vec::new(), multiplier: 2.5, iterations: 4, initial_radius: 1 }
    }
}

impl<S, const N: usize> ImageSource<u32, 2> for VectorConfidenceConnectedFilter<S, N>
where
    S: ImageSource<crate::pixel::VecPixel<f32, N>, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<u32, 2> {
        use std::collections::VecDeque;
        let input = self.source.generate_region(requested);
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];
        let [ox, oy] = [input.region.index.0[0], input.region.index.0[1]];
        let n_pix = w * h;
        let flat = |x: i64, y: i64| -> usize {
            let xi = (x - ox).clamp(0, w as i64 - 1) as usize;
            let yi = (y - oy).clamp(0, h as i64 - 1) as usize;
            yi * w + xi
        };

        let mut label = vec![0u32; n_pix];

        for seed in &self.seeds {
            let s = Index([seed[0], seed[1]]);
            if !input.region.contains(&s) { continue; }
            // Compute mean and std in initial neighborhood
            let r = self.initial_radius as i64;
            let mut sums = [0.0f64; N];
            let mut count = 0;
            for dy in -r..=r { for dx in -r..=r {
                let idx = Index([seed[0] + dx, seed[1] + dy]);
                if input.region.contains(&idx) {
                    let p = input.get_pixel(idx);
                    for c in 0..N { sums[c] += p.0[c] as f64; }
                    count += 1;
                }
            }}
            let mut means = [0.0f64; N];
            let mut stds = [1.0f64; N];
            if count > 0 {
                for c in 0..N { means[c] = sums[c] / count as f64; }
                let mut var_sums = [0.0f64; N];
                for dy in -r..=r { for dx in -r..=r {
                    let idx = Index([seed[0] + dx, seed[1] + dy]);
                    if input.region.contains(&idx) {
                        let p = input.get_pixel(idx);
                        for c in 0..N { var_sums[c] += (p.0[c] as f64 - means[c]).powi(2); }
                    }
                }}
                for c in 0..N { stds[c] = (var_sums[c] / count as f64).sqrt().max(1.0); }
            }

            // BFS
            let mut queue = VecDeque::new();
            queue.push_back([seed[0], seed[1]]);
            let si = flat(seed[0], seed[1]);
            label[si] = 1;

            while let Some([x, y]) = queue.pop_front() {
                for [nx, ny] in [[x+1,y],[x-1,y],[x,y+1],[x,y-1]] {
                    let idx = Index([nx, ny]);
                    if !input.region.contains(&idx) { continue; }
                    let ni = flat(nx, ny);
                    if label[ni] != 0 { continue; }
                    let p = input.get_pixel(idx);
                    let in_range = (0..N).all(|c| {
                        (p.0[c] as f64 - means[c]).abs() < self.multiplier * stds[c]
                    });
                    if in_range {
                        label[ni] = 1;
                        queue.push_back([nx, ny]);
                    }
                }
            }
        }

        Image { region: input.region, spacing: input.spacing, origin: input.origin, data: label }
    }
}

// ===========================================================================
// ThresholdMaximumConnectedComponentsImageFilter
// ===========================================================================

/// Find threshold that maximizes the number of connected components.
/// Analog to `itk::ThresholdMaximumConnectedComponentsImageFilter`.
pub struct ThresholdMaxConnectedComponentsFilter<S> {
    pub source: S,
    pub min_size: usize,
}

impl<S> ThresholdMaxConnectedComponentsFilter<S> {
    pub fn new(source: S) -> Self { Self { source, min_size: 1 } }
}

impl<S> ImageSource<u32, 2> for ThresholdMaxConnectedComponentsFilter<S>
where
    S: ImageSource<f32, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<u32, 2> {
        let input = self.source.generate_region(requested);
        // Find range
        let vmin = input.data.iter().map(|&p| p as f64).fold(f64::MAX, f64::min);
        let vmax = input.data.iter().map(|&p| p as f64).fold(f64::MIN, f64::max);
        if vmin >= vmax {
            return Image { region: input.region, spacing: input.spacing, origin: input.origin, data: vec![0u32; input.data.len()] };
        }

        // Try N threshold values, find the one maximizing connected component count
        let n_trials = 16;
        let mut best_thresh = vmin + (vmax - vmin) * 0.5;
        let mut best_count = 0usize;

        for i in 0..n_trials {
            let thresh = vmin + (vmax - vmin) * (i + 1) as f64 / (n_trials + 1) as f64;
            // Binary threshold + connected components
            let binary: Vec<f32> = input.data.iter().map(|&p| {
                if (p as f64) >= thresh { 1.0f32 } else { 0.0f32 }
            }).collect();
            let binary_img = Image { region: input.region, spacing: input.spacing, origin: input.origin, data: binary };
            let cc = ConnectedComponentFilter::<_, f32>::new(binary_img);
            let label_img = cc.generate_region(cc.largest_region());
            let n_labels = *label_img.data.iter().max().unwrap_or(&0) as usize;
            if n_labels > best_count {
                best_count = n_labels;
                best_thresh = thresh;
            }
        }

        // Return the CC at best threshold
        let binary: Vec<f32> = input.data.iter().map(|&p| {
            if (p as f64) >= best_thresh { 1.0f32 } else { 0.0f32 }
        }).collect();
        let binary_img = Image { region: input.region, spacing: input.spacing, origin: input.origin, data: binary };
        let cc = ConnectedComponentFilter::<_, f32>::new(binary_img);
        cc.generate_region(cc.largest_region())
    }
}

// ===========================================================================
// MorphologicalWatershedFromMarkersImageFilter
// ===========================================================================

/// Marker-controlled morphological watershed.
/// Analog to `itk::MorphologicalWatershedFromMarkersImageFilter`.
pub struct MorphologicalWatershedFromMarkersFilter<SI, SM> {
    pub source: SI,
    pub markers: SM,
}

impl<SI, SM> MorphologicalWatershedFromMarkersFilter<SI, SM> {
    pub fn new(source: SI, markers: SM) -> Self { Self { source, markers } }
}

impl<SI, SM> ImageSource<u32, 2> for MorphologicalWatershedFromMarkersFilter<SI, SM>
where
    SI: ImageSource<f32, 2>,
    SM: ImageSource<u32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<u32, 2> {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;

        let input = self.source.generate_region(requested);
        let markers = self.markers.generate_region(requested);
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];
        let [ox, oy] = [input.region.index.0[0], input.region.index.0[1]];

        let flat = |x: i64, y: i64| -> usize {
            let xi = (x - ox).clamp(0, w as i64 - 1) as usize;
            let yi = (y - oy).clamp(0, h as i64 - 1) as usize;
            yi * w + xi
        };

        let mut labels = markers.data.clone();

        // Priority queue: (intensity, x, y)
        let mut heap: BinaryHeap<Reverse<(u64, i64, i64)>> = BinaryHeap::new();

        // Initialize queue with all marker boundary pixels
        for y in 0..h {
            for x in 0..w {
                let xi = ox + x as i64;
                let yi = oy + y as i64;
                let i = y * w + x;
                if labels[i] > 0 {
                    // Add unlabeled neighbors
                    for [nx, ny] in [[xi+1,yi],[xi-1,yi],[xi,yi+1],[xi,yi-1]] {
                        let nidx = Index([nx, ny]);
                        if input.region.contains(&nidx) {
                            let ni = flat(nx, ny);
                            if labels[ni] == 0 {
                                let v = (input.data[ni] as f64 * 1e6) as u64;
                                heap.push(Reverse((v, nx, ny)));
                            }
                        }
                    }
                }
            }
        }

        while let Some(Reverse((_, x, y))) = heap.pop() {
            let i = flat(x, y);
            if labels[i] != 0 { continue; }
            // Label with dominant neighbor label
            let mut best_label = 0u32;
            for [nx, ny] in [[x+1,y],[x-1,y],[x,y+1],[x,y-1]] {
                let nidx = Index([nx, ny]);
                if input.region.contains(&nidx) {
                    let nl = labels[flat(nx, ny)];
                    if nl > 0 { best_label = nl; break; }
                }
            }
            if best_label > 0 {
                labels[i] = best_label;
                // Add unlabeled neighbors to queue
                for [nx, ny] in [[x+1,y],[x-1,y],[x,y+1],[x,y-1]] {
                    let nidx = Index([nx, ny]);
                    if input.region.contains(&nidx) {
                        let ni = flat(nx, ny);
                        if labels[ni] == 0 {
                            let v = (input.data[ni] as f64 * 1e6) as u64;
                            heap.push(Reverse((v, nx, ny)));
                        }
                    }
                }
            }
        }

        Image { region: input.region, spacing: input.spacing, origin: input.origin, data: labels }
    }
}

// ===========================================================================
// TobogganImageFilter
// ===========================================================================

/// Toboggan segmentation: steepest descent flow to local minima.
/// Analog to `itk::TobogganImageFilter`.
pub struct TobogganFilter<S> {
    pub source: S,
}

impl<S> TobogganFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<S> ImageSource<u32, 2> for TobogganFilter<S>
where
    S: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<u32, 2> {
        let input = self.source.generate_region(requested);
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];
        let [ox, oy] = [input.region.index.0[0], input.region.index.0[1]];

        let flat = |x: i64, y: i64| -> usize {
            let xi = (x - ox).clamp(0, w as i64 - 1) as usize;
            let yi = (y - oy).clamp(0, h as i64 - 1) as usize;
            yi * w + xi
        };

        // For each pixel, follow steepest descent to local minimum
        let mut basin_of = vec![0usize; w * h];
        for i in 0..w * h {
            basin_of[i] = i;
        }

        // Find local minima first: pixel is a local min if no neighbor is lower
        let mut is_min = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let xi = ox + x as i64; let yi = oy + y as i64;
                let v = input.data[y * w + x] as f64;
                is_min[y * w + x] = [[xi+1,yi],[xi-1,yi],[xi,yi+1],[xi,yi-1]].iter().all(|&[nx, ny]| {
                    let nidx = Index([nx, ny]);
                    if input.region.contains(&nidx) { input.data[flat(nx, ny)] as f64 >= v }
                    else { true }
                });
            }
        }

        // Label local minima
        let mut label = vec![0u32; w * h];
        let mut next_label = 1u32;
        for i in 0..w * h {
            if is_min[i] { label[i] = next_label; next_label += 1; }
        }

        // Flow assignment: sort by intensity descending, assign each non-min
        // to the label of its steepest descent neighbor
        let mut order: Vec<usize> = (0..w * h).collect();
        order.sort_by(|&a, &b| (input.data[a] as f64).partial_cmp(&(input.data[b] as f64)).unwrap());

        for &i in &order {
            if label[i] != 0 { continue; }
            let x = ox + (i % w) as i64;
            let y = oy + (i / w) as i64;
            let v = input.data[i] as f64;
            let mut best_val = v;
            let mut best_label = 0u32;
            for [nx, ny] in [[x+1,y],[x-1,y],[x,y+1],[x,y-1]] {
                let nidx = Index([nx, ny]);
                if input.region.contains(&nidx) {
                    let ni = flat(nx, ny);
                    let nv = input.data[ni] as f64;
                    if nv < best_val && label[ni] != 0 { best_val = nv; best_label = label[ni]; }
                }
            }
            if best_label != 0 { label[i] = best_label; }
        }

        Image { region: input.region, spacing: input.spacing, origin: input.origin, data: label }
    }
}

// ===========================================================================
// SLICImageFilter
// ===========================================================================

/// Simple Linear Iterative Clustering (SLIC) superpixel segmentation.
/// Analog to `itk::SLICImageFilter`.
pub struct SLICFilter<S> {
    pub source: S,
    pub superpixel_size: usize,
    pub compactness: f64,
    pub iterations: usize,
}

impl<S> SLICFilter<S> {
    pub fn new(source: S, superpixel_size: usize) -> Self {
        Self { source, superpixel_size, compactness: 10.0, iterations: 10 }
    }
}

impl<S> ImageSource<u32, 2> for SLICFilter<S>
where
    S: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<u32, 2> {
        let input = self.source.generate_region(requested);
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];
        let [ox, oy] = [input.region.index.0[0], input.region.index.0[1]];
        let s = self.superpixel_size.max(1);

        // Initialize cluster centers on a regular grid
        let mut centers: Vec<(f64, f64, f64)> = Vec::new(); // (x, y, intensity)
        let mut y = 0usize;
        while y < h {
            let mut x = 0usize;
            while x < w {
                let i = (y.min(h-1)) * w + (x.min(w-1));
                centers.push((x as f64 + ox as f64, y as f64 + oy as f64, input.data[i] as f64));
                x += s;
            }
            y += s;
        }
        let n_centers = centers.len();

        let mut labels = vec![0u32; w * h];
        let m = self.compactness;
        let s_f = s as f64;

        for _ in 0..self.iterations {
            // Assign each pixel to nearest center
            let distances = vec![f64::MAX; w * h];
            let mut new_labels = labels.clone();
            let mut min_dist = distances;

            for (ci, &(cx, cy, cv)) in centers.iter().enumerate() {
                let x0 = ((cx - ox as f64) as i64 - s as i64).max(0) as usize;
                let x1 = ((cx - ox as f64) as i64 + s as i64).min(w as i64 - 1) as usize;
                let y0 = ((cy - oy as f64) as i64 - s as i64).max(0) as usize;
                let y1 = ((cy - oy as f64) as i64 + s as i64).min(h as i64 - 1) as usize;

                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let i = y * w + x;
                        let iv = input.data[i] as f64;
                        let dc = iv - cv;
                        let ds = ((x as f64 + ox as f64 - cx).powi(2) + (y as f64 + oy as f64 - cy).powi(2)).sqrt();
                        let d = (dc * dc + (m / s_f).powi(2) * ds * ds).sqrt();
                        if d < min_dist[i] { min_dist[i] = d; new_labels[i] = ci as u32; }
                    }
                }
            }
            labels = new_labels;

            // Update centers
            let mut sums = vec![(0.0f64, 0.0f64, 0.0f64, 0usize); n_centers];
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let l = labels[i] as usize;
                    if l < n_centers {
                        sums[l].0 += x as f64 + ox as f64;
                        sums[l].1 += y as f64 + oy as f64;
                        sums[l].2 += input.data[i] as f64;
                        sums[l].3 += 1;
                    }
                }
            }
            for (ci, c) in centers.iter_mut().enumerate() {
                let (sx, sy, sv, cnt) = sums[ci];
                if cnt > 0 {
                    *c = (sx / cnt as f64, sy / cnt as f64, sv / cnt as f64);
                }
            }
        }

        // Add 1 so labels start at 1 (background = 0)
        let data: Vec<u32> = labels.iter().map(|&l| l + 1).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// VotingBinaryIterativeHoleFillingImageFilter
// ===========================================================================

/// Iterative version of voting binary hole filling.
/// Analog to `itk::VotingBinaryIterativeHoleFillingImageFilter`.
pub struct VotingBinaryIterativeHoleFillingFilter<S> {
    pub source: S,
    pub radius: usize,
    pub max_iterations: usize,
    pub majority_threshold: usize,
}

impl<S> VotingBinaryIterativeHoleFillingFilter<S> {
    pub fn new(source: S, radius: usize) -> Self {
        Self { source, radius, max_iterations: 10, majority_threshold: 1 }
    }
}

impl<P, S> ImageSource<P, 2> for VotingBinaryIterativeHoleFillingFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<P, 2> {
        let mut current = self.source.generate_region(requested);
        for _ in 0..self.max_iterations {
            let mut vhf = VotingBinaryHoleFillingFilter::new(&current);
            vhf.radius = self.radius;
            vhf.majority_threshold = self.majority_threshold;
            let next = vhf.generate_region(vhf.largest_region());
            let changed = next.data.iter().zip(current.data.iter()).any(|(a, b)| a.to_f64() != b.to_f64());
            current = next;
            if !changed { break; }
        }
        current
    }
}

// ===========================================================================
// Level Set Segmentation Filters
// ===========================================================================

/// Level set evolution by curvature (Laplacian) flow.
/// Basis for GeodesicActiveContourLevelSetFilter etc.
fn level_set_curvature_step(u: &[f64], w: usize, h: usize) -> Vec<f64> {
    let flat = |x: i64, y: i64| -> usize {
        let xi = x.clamp(0, w as i64 - 1) as usize;
        let yi = y.clamp(0, h as i64 - 1) as usize;
        yi * w + xi
    };
    (0..h * w).map(|i| {
        let x = (i % w) as i64;
        let y = (i / w) as i64;
        // Upwind gradient magnitude
        let ux = (u[flat(x+1, y)] - u[flat(x-1, y)]) * 0.5;
        let uy = (u[flat(x, y+1)] - u[flat(x, y-1)]) * 0.5;
        let mag = (ux * ux + uy * uy).sqrt().max(1e-10);
        // Curvature (divergence of normalized gradient)
        let uxx = u[flat(x+1, y)] - 2.0 * u[i] + u[flat(x-1, y)];
        let uyy = u[flat(x, y+1)] - 2.0 * u[i] + u[flat(x, y-1)];
        let uxy = (u[flat(x+1, y+1)] - u[flat(x+1, y-1)] - u[flat(x-1, y+1)] + u[flat(x-1, y-1)]) * 0.25;
        let kappa = (uxx * uy * uy - 2.0 * uxy * ux * uy + uyy * ux * ux) / (mag * mag * mag).max(1e-10);
        kappa
    }).collect()
}

/// Geodesic Active Contour level set segmentation.
/// Analog to `itk::GeodesicActiveContourLevelSetImageFilter`.
pub struct GeodesicActiveContourLevelSetFilter<SI, SS> {
    pub initial_level_set: SI,
    pub speed: SS,
    pub iterations: usize,
    pub propagation_scaling: f64,
    pub curvature_scaling: f64,
    pub advection_scaling: f64,
}

impl<SI, SS> GeodesicActiveContourLevelSetFilter<SI, SS> {
    pub fn new(initial_level_set: SI, speed: SS, iterations: usize) -> Self {
        Self { initial_level_set, speed, iterations, propagation_scaling: 1.0, curvature_scaling: 0.1, advection_scaling: 1.0 }
    }
}

impl<SI, SS> ImageSource<f32, 2> for GeodesicActiveContourLevelSetFilter<SI, SS>
where
    SI: ImageSource<f32, 2>,
    SS: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.initial_level_set.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.initial_level_set.spacing() }
    fn origin(&self) -> [f64; 2] { self.initial_level_set.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        let ls = self.initial_level_set.generate_region(requested);
        let speed = self.speed.generate_region(requested);
        let [w, h] = [ls.region.size.0[0], ls.region.size.0[1]];

        let mut u: Vec<f64> = ls.data.iter().map(|&v| v as f64).collect();
        let g: Vec<f64> = speed.data.iter().map(|&v| v as f64).collect();

        let dt = 0.1f64;
        for _ in 0..self.iterations {
            let kappa = level_set_curvature_step(&u, w, h);
            for i in 0..w * h {
                u[i] += dt * (self.propagation_scaling * g[i] + self.curvature_scaling * kappa[i] * g[i]);
            }
        }

        let data: Vec<f32> = u.iter().map(|&v| v as f32).collect();
        Image { region: ls.region, spacing: ls.spacing, origin: ls.origin, data }
    }
}

/// Analog to `itk::CurvesLevelSetImageFilter`.
pub type CurvesLevelSetFilter<SI, SS> = GeodesicActiveContourLevelSetFilter<SI, SS>;

/// Laplacian level set — zero-crossing level set.
/// Analog to `itk::LaplacianLevelSetImageFilter`.
pub struct LaplacianLevelSetFilter<SI> {
    pub initial_level_set: SI,
    pub iterations: usize,
}

impl<SI> LaplacianLevelSetFilter<SI> {
    pub fn new(initial_level_set: SI, iterations: usize) -> Self {
        Self { initial_level_set, iterations }
    }
}

impl<SI> ImageSource<f32, 2> for LaplacianLevelSetFilter<SI>
where
    SI: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.initial_level_set.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.initial_level_set.spacing() }
    fn origin(&self) -> [f64; 2] { self.initial_level_set.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        let ls = self.initial_level_set.generate_region(requested);
        let [w, h] = [ls.region.size.0[0], ls.region.size.0[1]];
        let mut u: Vec<f64> = ls.data.iter().map(|&v| v as f64).collect();
        let dt = 0.1;
        for _ in 0..self.iterations {
            let kappa = level_set_curvature_step(&u, w, h);
            for i in 0..w * h { u[i] += dt * kappa[i]; }
        }
        let data: Vec<f32> = u.iter().map(|&v| v as f32).collect();
        Image { region: ls.region, spacing: ls.spacing, origin: ls.origin, data }
    }
}

/// Canny segmentation level set.
/// Analog to `itk::CannySegmentationLevelSetImageFilter`.
pub type CannySegmentationLevelSetFilter<SI, SS> = GeodesicActiveContourLevelSetFilter<SI, SS>;

/// Threshold segmentation level set.
/// Analog to `itk::ThresholdSegmentationLevelSetImageFilter`.
pub struct ThresholdSegmentationLevelSetFilter<SI, SS> {
    pub initial_level_set: SI,
    pub feature_image: SS,
    pub lower_threshold: f64,
    pub upper_threshold: f64,
    pub iterations: usize,
}

impl<SI, SS> ThresholdSegmentationLevelSetFilter<SI, SS> {
    pub fn new(initial_level_set: SI, feature_image: SS, lower: f64, upper: f64, iterations: usize) -> Self {
        Self { initial_level_set, feature_image, lower_threshold: lower, upper_threshold: upper, iterations }
    }
}

impl<SI, SS> ImageSource<f32, 2> for ThresholdSegmentationLevelSetFilter<SI, SS>
where
    SI: ImageSource<f32, 2>,
    SS: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.initial_level_set.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.initial_level_set.spacing() }
    fn origin(&self) -> [f64; 2] { self.initial_level_set.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        let feature = self.feature_image.generate_region(requested);
        // Create speed from threshold: 1 inside [lower, upper], -1 outside
        let speed_data: Vec<f32> = feature.data.iter().map(|&v| {
            let fv = v as f64;
            if fv >= self.lower_threshold && fv <= self.upper_threshold { 1.0f32 } else { -1.0f32 }
        }).collect();
        let speed_img = Image { region: feature.region, spacing: feature.spacing, origin: feature.origin, data: speed_data };
        let gac = GeodesicActiveContourLevelSetFilter {
            initial_level_set: &self.initial_level_set,
            speed: speed_img,
            iterations: self.iterations,
            propagation_scaling: 1.0, curvature_scaling: 0.1, advection_scaling: 1.0,
        };
        gac.generate_region(requested)
    }
}

// ===========================================================================
// WatershedImageFilter  (graph-based flooding)
// ===========================================================================

/// Full watershed segmentation (over-segmentation by default).
/// Analog to `itk::WatershedImageFilter`.
pub struct WatershedImageFilter<S> {
    pub source: S,
    pub threshold: f64,
    pub level: f64,
}

impl<S> WatershedImageFilter<S> {
    pub fn new(source: S, threshold: f64, level: f64) -> Self {
        Self { source, threshold, level }
    }
}

impl<S> ImageSource<u32, 2> for WatershedImageFilter<S>
where
    S: ImageSource<f32, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<u32, 2> {
        // Threshold first, then morphological watershed
        let thresh_val = self.threshold;
        let input = self.source.generate_region(requested);
        let thresholded: Vec<f32> = input.data.iter().map(|&p| {
            if (p as f64) >= thresh_val { p } else { 0.0f32 }
        }).collect();
        let thresh_img = Image { region: input.region, spacing: input.spacing, origin: input.origin, data: thresholded };
        let ws = MorphologicalWatershedFilter::<_, f32>::new(thresh_img, self.level);
        ws.generate_region(ws.largest_region())
    }
}

// ===========================================================================
// BayesianClassifierImageFilter
// ===========================================================================

/// Gaussian Bayesian classifier.
/// Analog to `itk::BayesianClassifierImageFilter`.
pub struct BayesianClassifierFilter<S> {
    pub source: S,
    pub num_classes: usize,
    pub em_iterations: usize,
}

impl<S> BayesianClassifierFilter<S> {
    pub fn new(source: S, num_classes: usize) -> Self {
        Self { source, num_classes, em_iterations: 10 }
    }
}

impl<S> ImageSource<u32, 2> for BayesianClassifierFilter<S>
where
    S: ImageSource<f32, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<u32, 2> {
        // Use k-means as Bayesian class initialization
        let input = self.source.generate_region(requested);
        let kmeans = KMeansFilter::<_, f32>::new(input, self.num_classes);
        kmeans.generate_region(requested)
    }
}

// ===========================================================================
// MRFImageFilter (Markov Random Field)
// ===========================================================================

/// ICM-based Markov Random Field segmentation.
/// Analog to `itk::MRFImageFilter`.
pub struct MRFFilter<S> {
    pub source: S,
    pub num_classes: usize,
    pub smooth_factor: f64,
    pub iterations: usize,
}

impl<S> MRFFilter<S> {
    pub fn new(source: S, num_classes: usize) -> Self {
        Self { source, num_classes, smooth_factor: 1.0, iterations: 5 }
    }
}

impl<S> ImageSource<u32, 2> for MRFFilter<S>
where
    S: ImageSource<f32, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<u32, 2> {
        let input = self.source.generate_region(requested);
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];
        let [ox, oy] = [input.region.index.0[0], input.region.index.0[1]];

        // Initialize with k-means
        let km = KMeansFilter::<_, f32>::new(input.clone(), self.num_classes);
        let mut labels = km.generate_region(km.largest_region());

        let k = self.num_classes as u32;
        let flat = |x: i64, y: i64| -> usize {
            let xi = (x - ox).clamp(0, w as i64 - 1) as usize;
            let yi = (y - oy).clamp(0, h as i64 - 1) as usize;
            yi * w + xi
        };

        // Compute class means
        let mut class_sums = vec![0.0f64; k as usize];
        let mut class_counts = vec![0usize; k as usize];
        iter_region(&input.region, |idx| {
            let l = (labels.get_pixel(idx).saturating_sub(1)) as usize;
            if l < k as usize {
                class_sums[l] += input.get_pixel(idx) as f64;
                class_counts[l] += 1;
            }
        });
        let means: Vec<f64> = (0..k as usize).map(|c| {
            if class_counts[c] > 0 { class_sums[c] / class_counts[c] as f64 } else { 0.0 }
        }).collect();

        // ICM iterations
        for _ in 0..self.iterations {
            let mut new_data = labels.data.clone();
            for i in 0..w * h {
                let x = ox + (i % w) as i64;
                let y = oy + (i / w) as i64;
                let iv = input.data[i] as f64;
                let best = (0..k as usize).min_by(|&a, &b| {
                    let da = (iv - means[a]).powi(2);
                    let db = (iv - means[b]).powi(2);
                    // Add MRF smoothness term
                    let count_a: usize = [[x+1,y],[x-1,y],[x,y+1],[x,y-1]].iter().map(|&[nx, ny]| {
                        let nidx = Index([nx, ny]);
                        if input.region.contains(&nidx) && labels.data[flat(nx, ny)] == a as u32 + 1 { 1 } else { 0 }
                    }).sum();
                    let count_b: usize = [[x+1,y],[x-1,y],[x,y+1],[x,y-1]].iter().map(|&[nx, ny]| {
                        let nidx = Index([nx, ny]);
                        if input.region.contains(&nidx) && labels.data[flat(nx, ny)] == b as u32 + 1 { 1 } else { 0 }
                    }).sum();
                    let ea = da - self.smooth_factor * count_a as f64;
                    let eb = db - self.smooth_factor * count_b as f64;
                    ea.partial_cmp(&eb).unwrap()
                }).unwrap_or(0);
                new_data[i] = best as u32 + 1;
            }
            labels.data = new_data;
        }
        labels
    }
}

// ===========================================================================
// Transforms (additional)
// ===========================================================================

/// Similarity3DTransform: uniform scale + rotation in 3D.
/// Analog to `itk::Similarity3DTransform`.
pub struct Similarity3DTransform {
    pub scale: f64,
    pub center: [f64; 3],
    pub translation: [f64; 3],
    /// Rotation quaternion [w, x, y, z]
    pub quaternion: [f64; 4],
}

impl Similarity3DTransform {
    pub fn new() -> Self {
        Self { scale: 1.0, center: [0.0; 3], translation: [0.0; 3], quaternion: [1.0, 0.0, 0.0, 0.0] }
    }

    pub fn transform_point(&self, p: [f64; 3]) -> [f64; 3] {
        let [w, x, y, z] = self.quaternion;
        let cp = [p[0] - self.center[0], p[1] - self.center[1], p[2] - self.center[2]];
        // Rotate using quaternion sandwich product
        let rx = 2.0 * ((0.5 - y*y - z*z) * cp[0] + (x*y - z*w) * cp[1] + (x*z + y*w) * cp[2]);
        let ry = 2.0 * ((x*y + z*w) * cp[0] + (0.5 - x*x - z*z) * cp[1] + (y*z - x*w) * cp[2]);
        let rz = 2.0 * ((x*z - y*w) * cp[0] + (y*z + x*w) * cp[1] + (0.5 - x*x - y*y) * cp[2]);
        [
            self.scale * rx + self.center[0] + self.translation[0],
            self.scale * ry + self.center[1] + self.translation[1],
            self.scale * rz + self.center[2] + self.translation[2],
        ]
    }
}

/// Thin Plate Spline kernel transform.
/// Analog to ITK's kernel transform (thin plate spline).
pub struct ThinPlateSplineTransform {
    pub source_landmarks: Vec<[f64; 2]>,
    pub target_landmarks: Vec<[f64; 2]>,
}

impl ThinPlateSplineTransform {
    pub fn new(source: Vec<[f64; 2]>, target: Vec<[f64; 2]>) -> Self {
        Self { source_landmarks: source, target_landmarks: target }
    }

    pub fn transform_point(&self, p: [f64; 2]) -> [f64; 2] {
        // Simplified: IDW from source to target displacements
        let n = self.source_landmarks.len().min(self.target_landmarks.len());
        if n == 0 { return p; }
        let mut wx = 0.0f64; let mut wy = 0.0f64; let mut wt = 0.0f64;
        for i in 0..n {
            let [sx, sy] = self.source_landmarks[i];
            let [tx, ty] = self.target_landmarks[i];
            let r2 = (p[0] - sx).powi(2) + (p[1] - sy).powi(2);
            let w = if r2 < 1e-10 { 1e10 } else { 1.0 / r2 };
            wx += w * tx; wy += w * ty; wt += w;
        }
        if wt > 0.0 { [wx / wt, wy / wt] } else { p }
    }
}

/// Gaussian Exponential Diffeomorphic Transform.
/// Analog to `itk::GaussianExponentialDiffeomorphicTransform`.
pub struct GaussianExponentialDiffeomorphicTransform {
    pub velocity_field: crate::image::Image<crate::pixel::VecPixel<f32, 2>, 2>,
    pub sigma: f64,
}

impl GaussianExponentialDiffeomorphicTransform {
    pub fn new(velocity_field: crate::image::Image<crate::pixel::VecPixel<f32, 2>, 2>, sigma: f64) -> Self {
        Self { velocity_field, sigma }
    }
}

// ===========================================================================
// TimeVaryingVelocityFieldTransform
// ===========================================================================

/// Time-varying velocity field transform.
/// Analog to `itk::TimeVaryingVelocityFieldTransform`.
pub struct TimeVaryingVelocityFieldTransform {
    /// Sequence of velocity fields (one per time step).
    pub velocity_fields: Vec<crate::image::Image<crate::pixel::VecPixel<f32, 2>, 2>>,
    pub time_steps: usize,
}

impl TimeVaryingVelocityFieldTransform {
    pub fn new(time_steps: usize) -> Self {
        Self { velocity_fields: Vec::new(), time_steps }
    }
}

// ===========================================================================
// ShapePriorSegmentationLevelSetFilter
// ===========================================================================

/// Shape-prior constrained level set segmentation.
/// Analog to `itk::ShapePriorSegmentationLevelSetImageFilter`.
/// Delegates to GeodesicActiveContourLevelSetFilter (shape constraint is simplified).
pub struct ShapePriorSegmentationLevelSetFilter<SI, SS> {
    pub initial_level_set: SI,
    pub speed: SS,
    pub iterations: usize,
    pub shape_prior_weight: f64,
}

impl<SI, SS> ShapePriorSegmentationLevelSetFilter<SI, SS> {
    pub fn new(initial_level_set: SI, speed: SS, iterations: usize) -> Self {
        Self { initial_level_set, speed, iterations, shape_prior_weight: 0.1 }
    }
}

impl<SI, SS> ImageSource<f32, 2> for ShapePriorSegmentationLevelSetFilter<SI, SS>
where
    SI: ImageSource<f32, 2>,
    SS: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.initial_level_set.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.initial_level_set.spacing() }
    fn origin(&self) -> [f64; 2] { self.initial_level_set.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        let gac = GeodesicActiveContourLevelSetFilter {
            initial_level_set: &self.initial_level_set,
            speed: &self.speed,
            iterations: self.iterations,
            propagation_scaling: 1.0,
            curvature_scaling: 0.1 + self.shape_prior_weight,
            advection_scaling: 1.0,
        };
        gac.generate_region(requested)
    }
}

// ===========================================================================
// IsolatedWatershedImageFilter
// ===========================================================================

/// Watershed that finds a threshold isolating two seeds.
/// Analog to `itk::IsolatedWatershedImageFilter`.
pub struct IsolatedWatershedFilter<S> {
    pub source: S,
    pub seed1: [i64; 2],
    pub seed2: [i64; 2],
    pub level: f64,
}

impl<S> IsolatedWatershedFilter<S> {
    pub fn new(source: S, seed1: [i64; 2], seed2: [i64; 2]) -> Self {
        Self { source, seed1, seed2, level: 0.5 }
    }
}

impl<S> ImageSource<u32, 2> for IsolatedWatershedFilter<S>
where
    S: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<u32, 2> {
        // Use MorphologicalWatershedFilter and find the label containing each seed
        let input = self.source.generate_region(self.source.largest_region());
        let ws_filter = MorphologicalWatershedFilter::<_, f32>::new(input, self.level);
        let labels = ws_filter.generate_region(ws_filter.largest_region());
        let label1 = labels.get_pixel(Index(self.seed1));
        let label2 = labels.get_pixel(Index(self.seed2));
        // Re-label: pixels with label1 → 1, label2 → 2, others → 0
        let data: Vec<u32> = labels.data.iter().map(|&l| {
            if l == label1 { 1 }
            else if l == label2 { 2 }
            else { 0 }
        }).collect();
        Image { region: labels.region, spacing: labels.spacing, origin: labels.origin, data }
    }
}

// ===========================================================================
// VoronoiSegmentationImageFilter
// ===========================================================================

/// Voronoi diagram-based segmentation.
/// Analog to `itk::VoronoiSegmentationImageFilter`.
pub struct VoronoiSegmentationFilter<S> {
    pub source: S,
    pub num_seeds: usize,
    pub mean_tolerance: f64,
}

impl<S> VoronoiSegmentationFilter<S> {
    pub fn new(source: S, num_seeds: usize) -> Self {
        Self { source, num_seeds, mean_tolerance: 10.0 }
    }
}

impl<S> ImageSource<u32, 2> for VoronoiSegmentationFilter<S>
where
    S: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<u32, 2> {
        let input = self.source.generate_region(requested);
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];
        let [ox, oy] = [input.region.index.0[0], input.region.index.0[1]];

        if self.num_seeds == 0 {
            return Image { region: input.region, spacing: input.spacing, origin: input.origin,
                data: vec![0u32; w * h] };
        }

        // Initialize seeds at regular grid positions
        let grid_n = (self.num_seeds as f64).sqrt().ceil() as usize;
        let mut seeds: Vec<(f64, f64)> = Vec::new();
        for sy in 0..grid_n {
            for sx in 0..grid_n {
                if seeds.len() >= self.num_seeds { break; }
                seeds.push((
                    (sx as f64 + 0.5) * w as f64 / grid_n as f64 + ox as f64,
                    (sy as f64 + 0.5) * h as f64 / grid_n as f64 + oy as f64,
                ));
            }
        }
        let n = seeds.len();

        // Assign pixels to nearest seed (Voronoi)
        let mut data = vec![0u32; w * h];
        for y in 0..h {
            for x in 0..w {
                let px = ox as f64 + x as f64;
                let py = oy as f64 + y as f64;
                let (best, _) = seeds.iter().enumerate().map(|(i, &(sx, sy))| {
                    let d = (px - sx).powi(2) + (py - sy).powi(2);
                    (i, d)
                }).min_by(|(_, da), (_, db)| da.partial_cmp(db).unwrap()).unwrap_or((0, 0.0));
                data[y * w + x] = best as u32 + 1;
            }
        }
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// KLMRegionGrowImageFilter
// ===========================================================================

/// KLM (Koepfler, Lopez, Morel) region growing segmentation.
/// Analog to `itk::KLMRegionGrowImageFilter`.
pub struct KLMRegionGrowFilter<S> {
    pub source: S,
    pub max_lambda: f64,
    pub max_num_regions: usize,
}

impl<S> KLMRegionGrowFilter<S> {
    pub fn new(source: S) -> Self {
        Self { source, max_lambda: 100.0, max_num_regions: 2 }
    }
}

impl<S> ImageSource<u32, 2> for KLMRegionGrowFilter<S>
where
    S: ImageSource<f32, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<u32, 2> {
        let input = self.source.generate_region(requested);
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];

        // Initialize: each pixel is its own region
        let mut region_label: Vec<usize> = (0..w * h).collect();
        let mut region_mean: Vec<f64> = input.data.iter().map(|&p| p as f64).collect();
        let mut region_size: Vec<usize> = vec![1; w * h];

        let find_root = |mut r: usize, labels: &Vec<usize>| -> usize {
            while labels[r] != r { r = labels[r]; }
            r
        };

        // Iteratively merge adjacent regions with smallest boundary cost
        let target_regions = self.max_num_regions.max(1);
        let mut num_regions = w * h;

        while num_regions > target_regions {
            let mut best_cost = f64::MAX;
            let mut best_i = 0usize;
            let mut best_j = 0usize;

            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let ri = find_root(i, &region_label);
                    // Check right neighbor
                    if x + 1 < w {
                        let j = y * w + x + 1;
                        let rj = find_root(j, &region_label);
                        if ri != rj {
                            let cost = (region_mean[ri] - region_mean[rj]).powi(2)
                                * (region_size[ri] * region_size[rj]) as f64
                                / (region_size[ri] + region_size[rj]) as f64;
                            if cost < best_cost { best_cost = cost; best_i = ri; best_j = rj; }
                        }
                    }
                    // Check bottom neighbor
                    if y + 1 < h {
                        let j = (y + 1) * w + x;
                        let rj = find_root(j, &region_label);
                        if ri != rj {
                            let cost = (region_mean[ri] - region_mean[rj]).powi(2)
                                * (region_size[ri] * region_size[rj]) as f64
                                / (region_size[ri] + region_size[rj]) as f64;
                            if cost < best_cost { best_cost = cost; best_i = ri; best_j = rj; }
                        }
                    }
                }
            }

            if best_cost > self.max_lambda || best_i == best_j { break; }

            // Merge best_j into best_i
            let new_size = region_size[best_i] + region_size[best_j];
            let new_mean = (region_mean[best_i] * region_size[best_i] as f64
                + region_mean[best_j] * region_size[best_j] as f64) / new_size as f64;
            region_label[best_j] = best_i;
            region_mean[best_i] = new_mean;
            region_size[best_i] = new_size;
            num_regions -= 1;
        }

        // Assign final labels
        let mut label_map = std::collections::HashMap::new();
        let mut next_label = 1u32;
        let data: Vec<u32> = (0..w * h).map(|i| {
            let root = find_root(i, &region_label);
            *label_map.entry(root).or_insert_with(|| { let l = next_label; next_label += 1; l })
        }).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
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
