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
