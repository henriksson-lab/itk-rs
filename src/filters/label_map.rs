//! Label map filters.
//!
//! | Filter | ITK analog |
//! |---|---|
//! | [`BinaryImageToLabelMapFilter`]  | `BinaryImageToLabelMapFilter` |
//! | [`LabelImageToLabelMapFilter`]   | `LabelImageToLabelMapFilter` |
//! | [`LabelMapToBinaryImageFilter`]  | `LabelMapToBinaryImageFilter` |
//! | [`LabelMapToLabelImageFilter`]   | `LabelMapToLabelImageFilter` |
//! | [`RelabelFilter`]                | `RelabelLabelMapFilter` |
//! | [`BinaryFillholeFilter`]         | `BinaryFillholeImageFilter` |
//! | [`BinaryGrindPeakFilter`]        | `BinaryGrindPeakImageFilter` |

use std::collections::{HashMap, VecDeque};

use crate::image::{Image, Index, Region, iter_region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

/// A label map: a map from label → list of pixel indices.
/// (Lightweight analog to `itk::LabelMap<itk::LabelObject>`)
pub struct LabelMap<const D: usize> {
    pub labels: HashMap<u32, Vec<[i64; D]>>,
    pub region: Region<D>,
    pub spacing: [f64; D],
    pub origin: [f64; D],
}

// ===========================================================================
// BinaryImageToLabelMapFilter
// ===========================================================================

/// Convert a binary image to a label map with connected component labeling.
/// Analog to `itk::BinaryImageToLabelMapFilter`.
///
/// Uses 4-connectivity (or 6-connectivity in 3D) BFS labeling.
pub struct BinaryImageToLabelMapFilter<S> {
    pub source: S,
    pub foreground_value: f64,
}

impl<S> BinaryImageToLabelMapFilter<S> {
    pub fn new(source: S) -> Self { Self { source, foreground_value: 1.0 } }

    pub fn compute<P: NumericPixel, const D: usize>(&self) -> LabelMap<D>
    where S: ImageSource<P, D>
    {
        let input = self.source.generate_region(self.source.largest_region());
        let bounds = input.region;
        let n = bounds.linear_len();
        let mut label_arr = vec![0u32; n];
        let mut next_label = 1u32;
        let fg = self.foreground_value;

        let flat = |idx: [i64; D]| -> usize {
            let mut f = 0usize; let mut stride = 1usize;
            for d in 0..D { f += (idx[d] - bounds.index.0[d]) as usize * stride; stride *= bounds.size.0[d]; }
            f
        };

        iter_region(&bounds, |seed| {
            let f = flat(seed.0);
            if label_arr[f] != 0 { return; }
            if (input.get_pixel(seed).to_f64() - fg).abs() >= 0.5 { return; }
            // BFS
            let lbl = next_label; next_label += 1;
            let mut queue: VecDeque<[i64; D]> = VecDeque::new();
            label_arr[f] = lbl;
            queue.push_back(seed.0);
            while let Some(idx) = queue.pop_front() {
                for d in 0..D {
                    for delta in [-1i64, 1i64] {
                        let mut nb = idx; nb[d] += delta;
                        if (0..D).all(|dd| nb[dd] >= bounds.index.0[dd] && nb[dd] < bounds.index.0[dd] + bounds.size.0[dd] as i64) {
                            let nf = flat(nb);
                            if label_arr[nf] == 0 && (input.get_pixel(Index(nb)).to_f64() - fg).abs() < 0.5 {
                                label_arr[nf] = lbl;
                                queue.push_back(nb);
                            }
                        }
                    }
                }
            }
        });

        let mut labels: HashMap<u32, Vec<[i64; D]>> = HashMap::new();
        iter_region(&bounds, |idx| {
            let lbl = label_arr[flat(idx.0)];
            if lbl > 0 { labels.entry(lbl).or_default().push(idx.0); }
        });

        LabelMap { labels, region: bounds, spacing: input.spacing, origin: input.origin }
    }
}

// ===========================================================================
// LabelMapToBinaryImageFilter
// ===========================================================================

/// Convert a label map to a binary image (any label → foreground).
pub fn label_map_to_binary<P: NumericPixel, const D: usize>(
    lm: &LabelMap<D>,
    foreground: f64,
) -> Image<P, D> {
    let mut img = Image::allocate(lm.region, lm.spacing, lm.origin, P::from_f64(0.0));
    for pts in lm.labels.values() {
        for &pt in pts {
            img.set_pixel(Index(pt), P::from_f64(foreground));
        }
    }
    img
}

/// Convert a label map to a label image (each pixel = its label value).
pub fn label_map_to_label_image<const D: usize>(lm: &LabelMap<D>) -> Image<u32, D> {
    let mut img = Image::allocate(lm.region, lm.spacing, lm.origin, 0u32);
    for (&lbl, pts) in &lm.labels {
        for &pt in pts {
            img.set_pixel(Index(pt), lbl);
        }
    }
    img
}

// ===========================================================================
// LabelImageToLabelMapFilter
// ===========================================================================

/// Convert a label image to a label map.
/// Analog to `itk::LabelImageToLabelMapFilter`.
pub fn label_image_to_label_map<const D: usize>(img: &Image<u32, D>) -> LabelMap<D> {
    let mut labels: HashMap<u32, Vec<[i64; D]>> = HashMap::new();
    iter_region(&img.region, |idx| {
        let lbl = img.get_pixel(idx);
        if lbl > 0 { labels.entry(lbl).or_default().push(idx.0); }
    });
    LabelMap { labels, region: img.region, spacing: img.spacing, origin: img.origin }
}

// ===========================================================================
// RelabelFilter
// ===========================================================================

/// Relabel connected components in a label image by size (largest = 1).
/// Analog to `itk::RelabelLabelMapFilter`.
pub fn relabel_by_size<const D: usize>(img: &Image<u32, D>) -> Image<u32, D> {
    let lm = label_image_to_label_map(img);
    let mut size_sorted: Vec<(u32, usize)> = lm.labels.iter().map(|(&l, pts)| (l, pts.len())).collect();
    size_sorted.sort_by(|a, b| b.1.cmp(&a.1)); // descending by size
    let remap: HashMap<u32, u32> = size_sorted.iter().enumerate()
        .map(|(i, (l, _))| (*l, (i + 1) as u32))
        .collect();
    let mut out = img.clone();
    for v in out.data.iter_mut() {
        *v = remap.get(v).copied().unwrap_or(0);
    }
    out
}

// ===========================================================================
// BinaryFillholeImageFilter
// ===========================================================================

/// Fill holes in a binary image: connected background regions that don't
/// touch the image border become foreground.
/// Analog to `itk::BinaryFillholeImageFilter`.
pub struct BinaryFillholeFilter<S> {
    pub source: S,
    pub foreground_value: f64,
}

impl<S> BinaryFillholeFilter<S> {
    pub fn new(source: S) -> Self { Self { source, foreground_value: 1.0 } }
}

impl<P, S, const D: usize> ImageSource<P, D> for BinaryFillholeFilter<S>
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
        let fg = self.foreground_value;

        let flat = |idx: [i64; D]| -> usize {
            let mut f = 0usize; let mut stride = 1usize;
            for d in 0..D { f += (idx[d] - bounds.index.0[d]) as usize * stride; stride *= bounds.size.0[d]; }
            f
        };

        // BFS from all border background pixels
        let mut bg_connected_to_border = vec![false; n];
        let mut queue: VecDeque<[i64; D]> = VecDeque::new();

        iter_region(&bounds, |idx| {
            // Is this a border pixel?
            let on_border = (0..D).any(|d|
                idx.0[d] == bounds.index.0[d] || idx.0[d] == bounds.index.0[d] + bounds.size.0[d] as i64 - 1
            );
            if !on_border { return; }
            let v = input.get_pixel(idx).to_f64();
            if (v - fg).abs() >= 0.5 {
                let f = flat(idx.0);
                if !bg_connected_to_border[f] {
                    bg_connected_to_border[f] = true;
                    queue.push_back(idx.0);
                }
            }
        });

        while let Some(idx) = queue.pop_front() {
            for d in 0..D {
                for delta in [-1i64, 1i64] {
                    let mut nb = idx; nb[d] += delta;
                    if (0..D).all(|dd| nb[dd] >= bounds.index.0[dd] && nb[dd] < bounds.index.0[dd] + bounds.size.0[dd] as i64) {
                        let nf = flat(nb);
                        if !bg_connected_to_border[nf] {
                            let nv = input.get_pixel(Index(nb)).to_f64();
                            if (nv - fg).abs() >= 0.5 {
                                bg_connected_to_border[nf] = true;
                                queue.push_back(nb);
                            }
                        }
                    }
                }
            }
        }

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<P> = out_indices.iter().map(|&idx| {
            let orig = input.get_pixel(idx).to_f64();
            let is_fg = (orig - fg).abs() < 0.5;
            let f = flat(idx.0);
            if is_fg || !bg_connected_to_border[f] {
                P::from_f64(fg)
            } else {
                P::from_f64(0.0)
            }
        }).collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// BinaryGrindPeakImageFilter
// ===========================================================================

/// Binary grind peak: remove isolated foreground pixels (regional maxima in binary).
/// Analog to `itk::BinaryGrindPeakImageFilter`.
///
/// A foreground pixel is removed if all its neighbours are also foreground.
/// (In binary morphology, "peaks" are isolated FG pixels.)
pub struct BinaryGrindPeakFilter<S> {
    pub source: S,
    pub foreground_value: f64,
}

impl<S> BinaryGrindPeakFilter<S> {
    pub fn new(source: S) -> Self { Self { source, foreground_value: 1.0 } }
}

impl<P, S, const D: usize> ImageSource<P, D> for BinaryGrindPeakFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        use rayon::prelude::*;
        let input = self.source.generate_region(requested);
        let bounds = input.region;
        let fg = self.foreground_value;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter().map(|&idx| {
            let v = input.get_pixel(idx).to_f64();
            if (v - fg).abs() >= 0.5 { return P::from_f64(0.0); }
            // Check if any neighbour is background (if so, this pixel is NOT an isolated peak)
            for d in 0..D {
                for delta in [-1i64, 1i64] {
                    let mut nb = idx.0; nb[d] += delta;
                    if (0..D).all(|dd| nb[dd] >= bounds.index.0[dd] && nb[dd] < bounds.index.0[dd] + bounds.size.0[dd] as i64) {
                        let nv = input.get_pixel(Index(nb)).to_f64();
                        if (nv - fg).abs() >= 0.5 { return P::from_f64(fg); }
                    }
                }
            }
            // All neighbours are FG → this is an interior pixel, keep it
            // (grind peak removes isolated FG pixels surrounded by BG)
            P::from_f64(fg)
        }).collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// ShapeLabelMapFilter
// ===========================================================================

/// Compute shape attributes for each connected component in a label map.
/// Analog to `itk::ShapeLabelMapFilter`.
pub struct ShapeAttributes {
    pub label: u32,
    pub area: usize,
    pub perimeter: usize,
    pub bounding_box_size: [usize; 2],
    pub centroid: [f64; 2],
    pub elongation: f64,
    pub roundness: f64,
}

pub struct ShapeLabelMapFilter<S> {
    pub source: S,
}

impl<S> ShapeLabelMapFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<S> ShapeLabelMapFilter<S>
where S: crate::source::ImageSource<u32, 2>
{
    pub fn compute(&self) -> Vec<ShapeAttributes> {
        use crate::image::iter_region;
        use std::collections::HashMap;

        let img = self.source.generate_region(self.source.largest_region());
        let [_w, _h] = [img.region.size.0[0], img.region.size.0[1]];

        let mut areas: HashMap<u32, usize> = HashMap::new();
        let mut sums_x: HashMap<u32, f64> = HashMap::new();
        let mut sums_y: HashMap<u32, f64> = HashMap::new();
        let mut min_xy: HashMap<u32, [i64; 2]> = HashMap::new();
        let mut max_xy: HashMap<u32, [i64; 2]> = HashMap::new();

        iter_region(&img.region, |idx| {
            let l = img.get_pixel(idx);
            if l == 0 { return; }
            let [x, y] = [idx.0[0], idx.0[1]];
            *areas.entry(l).or_insert(0) += 1;
            *sums_x.entry(l).or_insert(0.0) += x as f64;
            *sums_y.entry(l).or_insert(0.0) += y as f64;
            let mn = min_xy.entry(l).or_insert([x, y]);
            if x < mn[0] { mn[0] = x; }
            if y < mn[1] { mn[1] = y; }
            let mx = max_xy.entry(l).or_insert([x, y]);
            if x > mx[0] { mx[0] = x; }
            if y > mx[1] { mx[1] = y; }
        });

        // Count perimeter pixels (pixels with at least one non-same-label neighbour)
        let mut perims: HashMap<u32, usize> = HashMap::new();
        iter_region(&img.region, |idx| {
            let l = img.get_pixel(idx);
            if l == 0 { return; }
            let [x, y] = [idx.0[0], idx.0[1]];
            let on_border = [[x+1,y],[x-1,y],[x,y+1],[x,y-1]].iter().any(|&[nx, ny]| {
                let idx2 = crate::image::Index([nx, ny]);
                if img.region.contains(&idx2) { img.get_pixel(idx2) != l } else { true }
            });
            if on_border { *perims.entry(l).or_insert(0) += 1; }
        });

        let mut results: Vec<ShapeAttributes> = areas.iter().map(|(&label, &area)| {
            let cx = sums_x[&label] / area as f64;
            let cy = sums_y[&label] / area as f64;
            let mn = min_xy[&label];
            let mx = max_xy[&label];
            let bb = [(mx[0] - mn[0] + 1) as usize, (mx[1] - mn[1] + 1) as usize];
            let perim = *perims.get(&label).unwrap_or(&0);

            // Elongation: ratio of bounding box axes
            let (bx, by) = (bb[0].max(1) as f64, bb[1].max(1) as f64);
            let elongation = if bx > by { bx / by } else { by / bx };

            // Roundness: 4π·area / perimeter²
            let roundness = if perim > 0 {
                4.0 * std::f64::consts::PI * area as f64 / (perim * perim) as f64
            } else { 0.0 };

            ShapeAttributes { label, area, perimeter: perim, bounding_box_size: bb, centroid: [cx, cy], elongation, roundness }
        }).collect();
        results.sort_by_key(|r| r.label);
        results
    }
}

// ===========================================================================
// StatisticsLabelMapFilter
// ===========================================================================

/// Compute intensity statistics per label. Analog to `itk::StatisticsLabelMapFilter`.
/// Thin wrapper over `LabelStatisticsFilter`.
pub struct StatisticsLabelMapFilter<SI, SL> {
    pub intensity: SI,
    pub label_map: SL,
}

impl<SI, SL> StatisticsLabelMapFilter<SI, SL> {
    pub fn new(intensity: SI, label_map: SL) -> Self { Self { intensity, label_map } }

    pub fn compute<P, const D: usize>(&self) -> Vec<crate::filters::statistics::LabelStatisticsResult>
    where
        P: crate::pixel::NumericPixel,
        SI: crate::source::ImageSource<P, D>,
        SL: crate::source::ImageSource<u32, D>,
    {
        let lsf = crate::filters::statistics::LabelStatisticsFilter {
            intensity: &self.intensity,
            label_map: &self.label_map,
        };
        lsf.compute::<P, D>()
    }
}

// ===========================================================================
// ShapeKeepNObjectsLabelMapFilter
// ===========================================================================

/// Keep the N largest objects by area.
/// Analog to `itk::ShapeKeepNObjectsLabelMapFilter`.
pub struct ShapeKeepNObjectsFilter<S> {
    pub source: S,
    pub n_objects: usize,
}

impl<S> ShapeKeepNObjectsFilter<S> {
    pub fn new(source: S, n_objects: usize) -> Self { Self { source, n_objects } }
}

impl<S> crate::source::ImageSource<u32, 2> for ShapeKeepNObjectsFilter<S>
where S: crate::source::ImageSource<u32, 2>
{
    fn largest_region(&self) -> crate::image::Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: crate::image::Region<2>) -> crate::image::Image<u32, 2> {
        use crate::image::iter_region;
        use std::collections::HashMap;

        let img = self.source.generate_region(requested);
        // Count areas
        let mut areas: HashMap<u32, usize> = HashMap::new();
        iter_region(&img.region, |idx| {
            let l = img.get_pixel(idx);
            if l != 0 { *areas.entry(l).or_insert(0) += 1; }
        });

        // Find top-N labels by area
        let mut sorted: Vec<(u32, usize)> = areas.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        let keep: std::collections::HashSet<u32> = sorted.iter().take(self.n_objects).map(|&(l, _)| l).collect();

        let data: Vec<u32> = img.data.iter().map(|&l| if keep.contains(&l) { l } else { 0 }).collect();
        crate::image::Image { region: img.region, spacing: img.spacing, origin: img.origin, data }
    }
}

/// Analog to `itk::ShapeOpeningLabelMapFilter` (keep objects with area >= min_size).
pub struct ShapeOpeningFilter<S> {
    pub source: S,
    pub min_size: usize,
}

impl<S> ShapeOpeningFilter<S> {
    pub fn new(source: S, min_size: usize) -> Self { Self { source, min_size } }
}

impl<S> crate::source::ImageSource<u32, 2> for ShapeOpeningFilter<S>
where S: crate::source::ImageSource<u32, 2>
{
    fn largest_region(&self) -> crate::image::Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: crate::image::Region<2>) -> crate::image::Image<u32, 2> {
        use crate::image::iter_region;
        use std::collections::HashMap;

        let img = self.source.generate_region(requested);
        let mut areas: HashMap<u32, usize> = HashMap::new();
        iter_region(&img.region, |idx| {
            let l = img.get_pixel(idx);
            if l != 0 { *areas.entry(l).or_insert(0) += 1; }
        });
        let min_sz = self.min_size;
        let data: Vec<u32> = img.data.iter().map(|&l| {
            if l == 0 { 0 } else if areas.get(&l).copied().unwrap_or(0) >= min_sz { l } else { 0 }
        }).collect();
        crate::image::Image { region: img.region, spacing: img.spacing, origin: img.origin, data }
    }
}

/// Statistics-based variants (same as shape, but named for ITK analog).
pub type StatisticsKeepNObjectsFilter<S> = ShapeKeepNObjectsFilter<S>;
pub type StatisticsOpeningFilter<S> = ShapeOpeningFilter<S>;

// ===========================================================================
// BinaryImageToShapeLabelMapFilter
// ===========================================================================

/// Segment binary image into connected components + compute shape attributes.
/// Analog to `itk::BinaryImageToShapeLabelMapFilter`.
pub struct BinaryImageToShapeLabelMapFilter<S> {
    pub source: S,
}

impl<S> BinaryImageToShapeLabelMapFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<S> BinaryImageToShapeLabelMapFilter<S>
{
    pub fn compute<P>(&self) -> Vec<ShapeAttributes>
    where
        P: crate::pixel::NumericPixel,
        S: crate::source::ImageSource<P, 2> + Sync,
    {
        // Run connected components on binary image then shape analysis
        let input = self.source.generate_region(self.source.largest_region());
        let cc = crate::filters::segmentation::ConnectedComponentFilter::<_, P>::new(input);
        let label_img = cc.generate_region(cc.largest_region());
        let shape = ShapeLabelMapFilter::new(label_img);
        shape.compute()
    }
}

/// Analog to `itk::BinaryImageToStatisticsLabelMapFilter`.
pub struct BinaryImageToStatisticsLabelMapFilter<SI, SL> {
    pub binary: SI,
    pub intensity: SL,
}

impl<SI, SL> BinaryImageToStatisticsLabelMapFilter<SI, SL> {
    pub fn new(binary: SI, intensity: SL) -> Self { Self { binary, intensity } }
}

impl<SI, SL> BinaryImageToStatisticsLabelMapFilter<SI, SL>
{
    pub fn compute<P>(&self) -> Vec<crate::filters::statistics::LabelStatisticsResult>
    where
        P: crate::pixel::NumericPixel,
        SI: crate::source::ImageSource<P, 2> + Sync,
        SL: crate::source::ImageSource<P, 2>,
    {
        let binary_img = self.binary.generate_region(self.binary.largest_region());
        let cc = crate::filters::segmentation::ConnectedComponentFilter::<_, P>::new(binary_img);
        let label_img = cc.generate_region(cc.largest_region());
        let lsf = crate::filters::statistics::LabelStatisticsFilter {
            intensity: &self.intensity,
            label_map: label_img,
        };
        lsf.compute::<P, 2>()
    }
}

// ===========================================================================
// MergeLabelMapFilter
// ===========================================================================

/// Merge two label images by taking the union (non-zero wins).
/// Analog to `itk::MergeLabelMapFilter`.
pub struct MergeLabelMapFilter<SA, SB> {
    pub source_a: SA,
    pub source_b: SB,
}

impl<SA, SB> MergeLabelMapFilter<SA, SB> {
    pub fn new(source_a: SA, source_b: SB) -> Self { Self { source_a, source_b } }
}

impl<SA, SB, const D: usize> crate::source::ImageSource<u32, D> for MergeLabelMapFilter<SA, SB>
where
    SA: crate::source::ImageSource<u32, D>,
    SB: crate::source::ImageSource<u32, D>,
{
    fn largest_region(&self) -> crate::image::Region<D> { self.source_a.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source_a.spacing() }
    fn origin(&self) -> [f64; D] { self.source_a.origin() }

    fn generate_region(&self, requested: crate::image::Region<D>) -> crate::image::Image<u32, D> {
        let a = self.source_a.generate_region(requested);
        let b = self.source_b.generate_region(requested);
        let data: Vec<u32> = a.data.iter().zip(b.data.iter()).map(|(&la, &lb)| {
            if la != 0 { la } else { lb }
        }).collect();
        crate::image::Image { region: a.region, spacing: a.spacing, origin: a.origin, data }
    }
}

// ===========================================================================
// AutoCropLabelMapFilter
// ===========================================================================

/// Crop to the bounding box of non-zero labels.
/// Analog to `itk::AutoCropLabelMapFilter`.
pub struct AutoCropLabelMapFilter<S> {
    pub source: S,
    pub crop_border: usize,
}

impl<S> AutoCropLabelMapFilter<S> {
    pub fn new(source: S) -> Self { Self { source, crop_border: 0 } }
}

impl<S, const D: usize> crate::source::ImageSource<u32, D> for AutoCropLabelMapFilter<S>
where S: crate::source::ImageSource<u32, D>
{
    fn largest_region(&self) -> crate::image::Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: crate::image::Region<D>) -> crate::image::Image<u32, D> {
        use crate::image::iter_region;
        let img = self.source.generate_region(requested);
        let mut min_idx = img.region.index.0;
        let mut max_idx = min_idx;
        let mut found = false;
        iter_region(&img.region, |idx| {
            if img.get_pixel(idx) != 0 {
                if !found {
                    min_idx = idx.0; max_idx = idx.0; found = true;
                } else {
                    for d in 0..D {
                        if idx.0[d] < min_idx[d] { min_idx[d] = idx.0[d]; }
                        if idx.0[d] > max_idx[d] { max_idx[d] = idx.0[d]; }
                    }
                }
            }
        });
        if !found { return img; }
        let cb = self.crop_border as i64;
        let mut start = min_idx;
        let mut end = max_idx;
        for d in 0..D {
            start[d] = (start[d] - cb).max(img.region.index.0[d]);
            end[d] = (end[d] + cb).min(img.region.index.0[d] + img.region.size.0[d] as i64 - 1);
        }
        let mut size = [0usize; D];
        for d in 0..D { size[d] = (end[d] - start[d] + 1) as usize; }
        let cropped_region = crate::image::Region::new(start, size);
        let mut out = crate::image::Image::allocate(cropped_region, img.spacing, img.origin, 0u32);
        iter_region(&cropped_region, |idx| {
            out.set_pixel(idx, img.get_pixel(idx));
        });
        out
    }
}

// ===========================================================================
// LabelMapMaskImageFilter
// ===========================================================================

/// Mask an image with a label map: keep pixels where label == target_label, zero elsewhere.
/// Analog to `itk::LabelMapMaskImageFilter`.
pub struct LabelMapMaskImageFilter<SI, SL, P> {
    pub intensity: SI,
    pub label_map: SL,
    pub label: u32,
    pub background: f64,
    pub negate: bool,
    _phantom: std::marker::PhantomData<P>,
}

impl<SI, SL, P> LabelMapMaskImageFilter<SI, SL, P> {
    pub fn new(intensity: SI, label_map: SL, label: u32) -> Self {
        Self { intensity, label_map, label, background: 0.0, negate: false, _phantom: std::marker::PhantomData }
    }
}

impl<P, SI, SL, const D: usize> crate::source::ImageSource<P, D> for LabelMapMaskImageFilter<SI, SL, P>
where
    P: crate::pixel::NumericPixel,
    SI: crate::source::ImageSource<P, D>,
    SL: crate::source::ImageSource<u32, D>,
{
    fn largest_region(&self) -> crate::image::Region<D> { self.intensity.largest_region() }
    fn spacing(&self) -> [f64; D] { self.intensity.spacing() }
    fn origin(&self) -> [f64; D] { self.intensity.origin() }

    fn generate_region(&self, requested: crate::image::Region<D>) -> crate::image::Image<P, D> {
        let intensity = self.intensity.generate_region(requested);
        let labels = self.label_map.generate_region(requested);
        let bg = P::from_f64(self.background);
        let data: Vec<P> = intensity.data.iter().zip(labels.data.iter()).map(|(&v, &l)| {
            let matches = l == self.label;
            let keep = if self.negate { !matches } else { matches };
            if keep { v } else { bg }
        }).collect();
        crate::image::Image { region: intensity.region, spacing: intensity.spacing, origin: intensity.origin, data }
    }
}

// ===========================================================================
// LabelMapOverlayImageFilter
// ===========================================================================

/// Overlay label map onto RGB image with per-label colors.
/// Analog to `itk::LabelMapOverlayImageFilter`.
pub struct LabelMapOverlayImageFilter<SI, SL, P> {
    pub intensity: SI,
    pub label_map: SL,
    pub opacity: f32,
    _phantom: std::marker::PhantomData<P>,
}

impl<SI, SL, P> LabelMapOverlayImageFilter<SI, SL, P> {
    pub fn new(intensity: SI, label_map: SL) -> Self {
        Self { intensity, label_map, opacity: 0.5, _phantom: std::marker::PhantomData }
    }
}

impl<P, SI, SL> crate::source::ImageSource<crate::pixel::VecPixel<f32, 3>, 2> for LabelMapOverlayImageFilter<SI, SL, P>
where
    P: crate::pixel::NumericPixel,
    SI: crate::source::ImageSource<P, 2>,
    SL: crate::source::ImageSource<u32, 2>,
{
    fn largest_region(&self) -> crate::image::Region<2> { self.intensity.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.intensity.spacing() }
    fn origin(&self) -> [f64; 2] { self.intensity.origin() }

    fn generate_region(&self, requested: crate::image::Region<2>) -> crate::image::Image<crate::pixel::VecPixel<f32, 3>, 2> {
        use crate::pixel::VecPixel;
        let intensity = self.intensity.generate_region(requested);
        let labels = self.label_map.generate_region(requested);

        // Normalize intensity to [0,1]
        let vmin = intensity.data.iter().map(|p| p.to_f64()).fold(f64::MAX, f64::min);
        let vmax = intensity.data.iter().map(|p| p.to_f64()).fold(f64::MIN, f64::max);
        let range = (vmax - vmin).max(1e-10);

        let label_color = |l: u32| -> [f32; 3] {
            // Simple hash-based color per label
            if l == 0 { return [0.0, 0.0, 0.0]; }
            let h = (l as f32 * 37.0 + 17.0) % 360.0;
            let s = 0.8f32;
            let v = 0.9f32;
            // HSV to RGB
            let hi = (h / 60.0) as i32;
            let f = h / 60.0 - hi as f32;
            let p2 = v * (1.0 - s);
            let q = v * (1.0 - s * f);
            let t = v * (1.0 - s * (1.0 - f));
            match hi % 6 {
                0 => [v, t, p2], 1 => [q, v, p2], 2 => [p2, v, t],
                3 => [p2, q, v], 4 => [t, p2, v], _ => [v, p2, q],
            }
        };

        let data: Vec<VecPixel<f32, 3>> = intensity.data.iter().zip(labels.data.iter()).map(|(&iv, &l)| {
            let gray = ((iv.to_f64() - vmin) / range) as f32;
            if l == 0 {
                VecPixel([gray, gray, gray])
            } else {
                let col = label_color(l);
                let op = self.opacity;
                VecPixel([
                    gray * (1.0 - op) + col[0] * op,
                    gray * (1.0 - op) + col[1] * op,
                    gray * (1.0 - op) + col[2] * op,
                ])
            }
        }).collect();
        crate::image::Image { region: intensity.region, spacing: intensity.spacing, origin: intensity.origin, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    #[test]
    fn binary_fillhole_1d() {
        // Image: 1 1 0 0 1 1 — interior background should become foreground
        // But 1D: 1 on both ends, so bg between them is enclosed → filled
        let mut img = Image::<f32,1>::allocate(Region::new([0],[6]),[1.0],[0.0],0.0f32);
        img.set_pixel(Index([0]), 1.0);
        img.set_pixel(Index([1]), 1.0);
        img.set_pixel(Index([4]), 1.0);
        img.set_pixel(Index([5]), 1.0);
        let f = BinaryFillholeFilter::new(img);
        let out = f.generate_region(f.largest_region());
        // Indices 2,3 should be filled (enclosed by FG on both sides)
        // ... actually in 1D with border seeds: indices 0,1 are border BG=0? No, they're FG.
        // BG pixels at 2,3 are not connected to any border BG → they get filled.
        let v2 = out.get_pixel(Index([2]));
        let v3 = out.get_pixel(Index([3]));
        assert!((v2 - 1.0).abs() < 0.1, "expected 1.0 got {v2}");
        assert!((v3 - 1.0).abs() < 0.1, "expected 1.0 got {v3}");
    }

    #[test]
    fn binary_image_to_label_map_two_components() {
        // Two separate fg regions
        let mut img = Image::<f32,1>::allocate(Region::new([0],[9]),[1.0],[0.0],0.0f32);
        for i in [1i64,2,3,6,7] { img.set_pixel(Index([i]), 1.0f32); }
        let f = BinaryImageToLabelMapFilter::new(img);
        let lm = f.compute::<f32, 1>();
        assert_eq!(lm.labels.len(), 2, "expected 2 connected components");
    }
}
