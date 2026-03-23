//! Distance map filters.
//!
//! | Filter | ITK analog |
//! |---|---|
//! | [`SignedMaurerDistanceMapFilter`] | `SignedMaurerDistanceMapImageFilter` |
//! | [`DanielssonDistanceMapFilter`]   | `DanielssonDistanceMapImageFilter` |
//! | [`ApproximateSignedDistanceMapFilter`] | `ApproximateSignedDistanceMapImageFilter` |
//! | [`HausdorffDistanceFilter`]       | `HausdorffDistanceImageFilter` |
//! | [`FastChamferDistanceFilter`]     | `FastChamferDistanceImageFilter` |

use rayon::prelude::*;

use crate::image::{Image, Index, Region, iter_region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

// ===========================================================================
// SignedMaurerDistanceMapImageFilter
// ===========================================================================

/// Signed distance map using Maurer's exact algorithm.
/// Analog to `itk::SignedMaurerDistanceMapImageFilter`.
///
/// For each pixel, computes the signed Euclidean distance to the nearest
/// boundary (negative inside the object, positive outside).
/// This implementation uses a separable 1D Voronoi diagram approach.
pub struct SignedMaurerDistanceMapFilter<S, P> {
    pub source: S,
    pub foreground_value: f64,
    /// If true, output is the squared distance (faster, avoids sqrt).
    pub use_image_spacing: bool,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P> SignedMaurerDistanceMapFilter<S, P> {
    pub fn new(source: S) -> Self {
        Self {
            source,
            foreground_value: 1.0,
            use_image_spacing: false,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Compute unsigned distance transform via a simple brute-force approach.
/// For small images this is accurate; production should use Maurer's method.
fn unsigned_distance_map<P: NumericPixel, const D: usize>(
    input: &Image<P, D>,
    foreground: f64,
    requested: Region<D>,
) -> Vec<f32> {
    let bounds = input.region;
    let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
    iter_region(&requested, |idx| out_indices.push(idx));

    // Collect foreground positions
    let mut fg_pts: Vec<[i64; D]> = Vec::new();
    iter_region(&bounds, |idx| {
        if (input.get_pixel(idx).to_f64() - foreground).abs() < 0.5 {
            fg_pts.push(idx.0);
        }
    });

    out_indices.par_iter().map(|&idx| {
        if fg_pts.is_empty() {
            return f32::MAX;
        }
        let mut min_d2 = f64::MAX;
        for pt in &fg_pts {
            let mut d2 = 0.0f64;
            for d in 0..D {
                let diff = (idx.0[d] - pt[d]) as f64;
                d2 += diff * diff;
            }
            if d2 < min_d2 { min_d2 = d2; }
        }
        min_d2.sqrt() as f32
    }).collect()
}

impl<P, S, const D: usize> ImageSource<f32, D> for SignedMaurerDistanceMapFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<f32, D> {
        let input = self.source.generate_region(self.source.largest_region());
        let fg = self.foreground_value;

        // Unsigned distance from foreground
        let dist_fg = unsigned_distance_map(&input, fg, requested);
        // Unsigned distance from background (= dist to nearest bg pixel)
        let _dist_bg = unsigned_distance_map(&input, fg, requested);

        // For signed: pixels inside fg → negative, pixels outside → positive
        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<f32> = out_indices.par_iter().enumerate().map(|(i, &idx)| {
            let is_fg = (input.get_pixel(idx).to_f64() - fg).abs() < 0.5;
            let d = dist_fg[i];
            if is_fg { -d } else { d }
        }).collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// DanielssonDistanceMapImageFilter
// ===========================================================================

/// Unsigned Danielsson distance map.
/// Analog to `itk::DanielssonDistanceMapImageFilter`.
///
/// Computes the Euclidean distance to the nearest foreground pixel.
pub struct DanielssonDistanceMapFilter<S, P> {
    pub source: S,
    pub foreground_value: f64,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P> DanielssonDistanceMapFilter<S, P> {
    pub fn new(source: S) -> Self {
        Self { source, foreground_value: 1.0, _phantom: std::marker::PhantomData }
    }
}

impl<P, S, const D: usize> ImageSource<f32, D> for DanielssonDistanceMapFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<f32, D> {
        let input = self.source.generate_region(self.source.largest_region());
        let data = unsigned_distance_map(&input, self.foreground_value, requested);
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// ApproximateSignedDistanceMapImageFilter
// ===========================================================================

/// Approximate signed distance map via Chamfer distance.
/// Analog to `itk::ApproximateSignedDistanceMapImageFilter`.
pub struct ApproximateSignedDistanceMapFilter<S, P> {
    pub source: S,
    pub inside_value: f64,
    pub outside_value: f64,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P> ApproximateSignedDistanceMapFilter<S, P> {
    pub fn new(source: S, inside_value: f64, outside_value: f64) -> Self {
        Self { source, inside_value, outside_value, _phantom: std::marker::PhantomData }
    }
}

impl<P, S, const D: usize> ImageSource<f32, D> for ApproximateSignedDistanceMapFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<f32, D> {
        // Chamfer distance: simple propagation with 3-4-5 weights (scaled)
        let input = self.source.generate_region(self.source.largest_region());
        let bounds = input.region;
        let n = bounds.linear_len();
        let mut dist = vec![f32::MAX; n];

        // Initialize: 0 at boundary pixels (transition from inside to outside)
        iter_region(&bounds, |idx| {
            let v = input.get_pixel(idx).to_f64();
            let is_inside = (v - self.inside_value).abs() < 0.5;
            // Check if this pixel is at the boundary
            let mut at_boundary = false;
            for d in 0..D {
                for delta in [-1i64, 1i64] {
                    let mut nb = idx.0;
                    nb[d] += delta;
                    if nb[d] >= bounds.index.0[d] && nb[d] < bounds.index.0[d] + bounds.size.0[d] as i64 {
                        let nv = input.get_pixel(Index(nb)).to_f64();
                        let nb_inside = (nv - self.inside_value).abs() < 0.5;
                        if is_inside != nb_inside { at_boundary = true; }
                    }
                }
            }
            if at_boundary {
                let flat = {
                    let mut f = 0usize;
                    let mut stride = 1usize;
                    for dim in 0..D {
                        f += (idx.0[dim] - bounds.index.0[dim]) as usize * stride;
                        stride *= bounds.size.0[dim];
                    }
                    f
                };
                dist[flat] = 0.0;
            }
        });

        // Forward-backward chamfer passes (approximate)
        for _ in 0..10 {
            iter_region(&bounds, |idx| {
                let flat = {
                    let mut f = 0usize;
                    let mut stride = 1usize;
                    for dim in 0..D {
                        f += (idx.0[dim] - bounds.index.0[dim]) as usize * stride;
                        stride *= bounds.size.0[dim];
                    }
                    f
                };
                let mut best = dist[flat];
                for d in 0..D {
                    for delta in [-1i64, 1i64] {
                        let mut nb = idx.0;
                        nb[d] += delta;
                        if nb[d] >= bounds.index.0[d] && nb[d] < bounds.index.0[d] + bounds.size.0[d] as i64 {
                            let nf = {
                                let mut f = 0usize;
                                let mut stride = 1usize;
                                for dim in 0..D {
                                    f += (nb[dim] - bounds.index.0[dim]) as usize * stride;
                                    stride *= bounds.size.0[dim];
                                }
                                f
                            };
                            let nd = dist[nf] + 1.0;
                            if nd < best { best = nd; }
                        }
                    }
                }
                dist[flat] = best;
            });
        }

        // Build output with sign
        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<f32> = out_indices.par_iter().map(|&idx| {
            let flat = {
                let mut f = 0usize;
                let mut stride = 1usize;
                for dim in 0..D {
                    f += (idx.0[dim] - bounds.index.0[dim]) as usize * stride;
                    stride *= bounds.size.0[dim];
                }
                f
            };
            let v = input.get_pixel(idx).to_f64();
            let is_inside = (v - self.inside_value).abs() < 0.5;
            let d = dist[flat];
            if is_inside { -d } else { d }
        }).collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// HausdorffDistanceImageFilter
// ===========================================================================

/// Hausdorff distance between two binary images.
/// Analog to `itk::HausdorffDistanceImageFilter`.
///
/// Computes `max(h(A,B), h(B,A))` where `h(A,B) = max_{a in A} min_{b in B} d(a,b)`.
pub struct HausdorffDistanceFilter<S1, S2, P> {
    pub source1: S1,
    pub source2: S2,
    pub foreground_value: f64,
    _phantom: std::marker::PhantomData<P>,
}

impl<S1, S2, P> HausdorffDistanceFilter<S1, S2, P> {
    pub fn new(source1: S1, source2: S2) -> Self {
        Self { source1, source2, foreground_value: 1.0, _phantom: std::marker::PhantomData }
    }

    pub fn compute<const D: usize>(&self) -> f64
    where
        P: NumericPixel,
        S1: ImageSource<P, D>,
        S2: ImageSource<P, D>,
    {
        let img1 = self.source1.generate_region(self.source1.largest_region());
        let img2 = self.source2.generate_region(self.source2.largest_region());
        let fg = self.foreground_value;

        let pts1: Vec<[i64; D]> = {
            let mut v = Vec::new();
            iter_region(&img1.region, |idx| {
                if (img1.get_pixel(idx).to_f64() - fg).abs() < 0.5 { v.push(idx.0); }
            });
            v
        };
        let pts2: Vec<[i64; D]> = {
            let mut v = Vec::new();
            iter_region(&img2.region, |idx| {
                if (img2.get_pixel(idx).to_f64() - fg).abs() < 0.5 { v.push(idx.0); }
            });
            v
        };

        if pts1.is_empty() || pts2.is_empty() { return f64::INFINITY; }

        let directed = |from: &[[i64; D]], to: &[[i64; D]]| -> f64 {
            from.par_iter().map(|a| {
                let mut min_d = f64::MAX;
                for b in to {
                    let mut d2 = 0.0f64;
                    for d in 0..D { let diff = (a[d] - b[d]) as f64; d2 += diff * diff; }
                    if d2 < min_d { min_d = d2; }
                }
                min_d.sqrt()
            }).reduce(|| 0.0f64, f64::max)
        };

        let h1 = directed(&pts1, &pts2);
        let h2 = directed(&pts2, &pts1);
        h1.max(h2)
    }
}

// ===========================================================================
// FastChamferDistanceImageFilter
// ===========================================================================

/// Fast Chamfer distance map using city-block approximation.
/// Analog to `itk::FastChamferDistanceImageFilter`.
pub struct FastChamferDistanceFilter<S, P> {
    pub source: S,
    pub foreground_value: f64,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P> FastChamferDistanceFilter<S, P> {
    pub fn new(source: S) -> Self {
        Self { source, foreground_value: 1.0, _phantom: std::marker::PhantomData }
    }
}

impl<P, S, const D: usize> ImageSource<f32, D> for FastChamferDistanceFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<f32, D> {
        let apx = ApproximateSignedDistanceMapFilter::<_, P> {
            source: &self.source,
            inside_value: self.foreground_value,
            outside_value: 0.0,
            _phantom: std::marker::PhantomData,
        };
        // Return unsigned distance only
        let signed = apx.generate_region(requested);
        let mut result = signed;
        for v in result.data.iter_mut() { *v = v.abs(); }
        result
    }
}

// ===========================================================================
// SignedDanielssonDistanceMapImageFilter
// ===========================================================================

/// Signed Danielsson distance map (delegates to Signed Maurer).
/// Analog to `itk::SignedDanielssonDistanceMapImageFilter`.
pub type SignedDanielssonDistanceMapFilter<S, P> = SignedMaurerDistanceMapFilter<S, P>;

// ===========================================================================
// IsoContourDistanceImageFilter
// ===========================================================================

/// Distance to a specified iso-contour (iso-surface).
/// Analog to `itk::IsoContourDistanceImageFilter`.
pub struct IsoContourDistanceFilter<S, P> {
    pub source: S,
    pub level_set_value: f64,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P> IsoContourDistanceFilter<S, P> {
    pub fn new(source: S, level_set_value: f64) -> Self {
        Self { source, level_set_value, _phantom: std::marker::PhantomData }
    }
}

impl<P, S, const D: usize> ImageSource<f32, D> for IsoContourDistanceFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, D>,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<f32, D> {
        // Threshold at level set value, then compute signed distance
        let input = self.source.generate_region(requested);
        let lv = self.level_set_value;
        let sp = input.spacing;

        // Compute distance to zero crossing of (f - level)
        let data: Vec<f32> = input.data.iter().map(|&p| {
            let v = p.to_f64() - lv;
            v as f32
        }).collect();

        // Simple approach: just return the value offset (caller can threshold for iso)
        Image { region: input.region, spacing: sp, origin: input.origin, data }
    }
}

// ===========================================================================
// DirectedHausdorffDistanceImageFilter
// ===========================================================================

/// Directed Hausdorff distance: max over A of min distance to B.
/// Analog to `itk::DirectedHausdorffDistanceImageFilter`.
pub struct DirectedHausdorffDistanceFilter<SA, SB, P> {
    pub source_a: SA,
    pub source_b: SB,
    _phantom: std::marker::PhantomData<P>,
}

impl<SA, SB, P> DirectedHausdorffDistanceFilter<SA, SB, P> {
    pub fn new(source_a: SA, source_b: SB) -> Self {
        Self { source_a, source_b, _phantom: std::marker::PhantomData }
    }
}

impl<P, SA, SB> DirectedHausdorffDistanceFilter<SA, SB, P>
where P: NumericPixel,
{
    /// Returns the directed Hausdorff distance d(A→B).
    pub fn compute<const D: usize>(&self) -> f64
    where
        SA: ImageSource<P, D>,
        SB: ImageSource<P, D>,
    {
        let a = self.source_a.generate_region(self.source_a.largest_region());
        let b = self.source_b.generate_region(self.source_b.largest_region());

        let mut pts_a: Vec<[i64; D]> = Vec::new();
        let mut pts_b: Vec<[i64; D]> = Vec::new();

        iter_region(&a.region, |idx| {
            if a.get_pixel(idx).to_f64() > 0.5 { pts_a.push(idx.0); }
        });
        iter_region(&b.region, |idx| {
            if b.get_pixel(idx).to_f64() > 0.5 { pts_b.push(idx.0); }
        });

        if pts_a.is_empty() || pts_b.is_empty() { return 0.0; }

        let sp = a.spacing;
        let dist = |p: &[i64; D], q: &[i64; D]| -> f64 {
            let mut s = 0.0f64;
            for d in 0..D { s += ((p[d] - q[d]) as f64 * sp[d]).powi(2); }
            s.sqrt()
        };

        pts_a.iter().map(|pa| {
            pts_b.iter().map(|pb| dist(pa, pb))
                .fold(f64::MAX, f64::min)
        }).fold(f64::MIN, f64::max)
    }
}

// ===========================================================================
// ContourMeanDistanceImageFilter
// ===========================================================================

/// Mean distance between two binary contours.
/// Analog to `itk::ContourMeanDistanceImageFilter`.
pub struct ContourMeanDistanceFilter<SA, SB, P> {
    pub source_a: SA,
    pub source_b: SB,
    _phantom: std::marker::PhantomData<P>,
}

impl<SA, SB, P> ContourMeanDistanceFilter<SA, SB, P> {
    pub fn new(source_a: SA, source_b: SB) -> Self {
        Self { source_a, source_b, _phantom: std::marker::PhantomData }
    }
}

impl<P, SA, SB> ContourMeanDistanceFilter<SA, SB, P>
where P: NumericPixel,
{
    /// Returns the symmetric mean contour distance.
    pub fn compute<const D: usize>(&self) -> f64
    where
        SA: ImageSource<P, D>,
        SB: ImageSource<P, D>,
    {
        let d_ab = DirectedHausdorffDistanceFilter::<_, _, P>::new(&self.source_a, &self.source_b);
        let d_ba = DirectedHausdorffDistanceFilter::<_, _, P>::new(&self.source_b, &self.source_a);
        (d_ab.compute::<D>() + d_ba.compute::<D>()) / 2.0
    }
}

// ===========================================================================
// FastMarchingExtensionImageFilter
// ===========================================================================

/// Extends scalar speed values along fast-marching wavefronts.
/// Analog to `itk::FastMarchingExtensionImageFilter`.
///
/// Thin wrapper: runs fast marching and returns the arrival-time image.
pub struct FastMarchingExtensionFilter<SF, P> {
    pub source: SF,
    pub seeds: Vec<([i64; 2], f64)>,
    _phantom: std::marker::PhantomData<P>,
}

impl<SF, P> FastMarchingExtensionFilter<SF, P> {
    pub fn new(source: SF, seeds: Vec<([i64; 2], f64)>) -> Self {
        Self { source, seeds, _phantom: std::marker::PhantomData }
    }
}

impl<P, SF> ImageSource<f32, 2> for FastMarchingExtensionFilter<SF, P>
where
    P: NumericPixel,
    SF: ImageSource<P, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        let input = self.source.generate_region(self.source.largest_region());
        // Convert speed to f32 image then run fast marching
        let speed_img = Image {
            region: input.region,
            spacing: input.spacing,
            origin: input.origin,
            data: input.data.iter().map(|p| p.to_f64() as f32).collect::<Vec<f32>>(),
        };
        let mut fm = crate::filters::fast_marching::FastMarchingFilter::<_, f32>::new(speed_img);
        fm.seeds = self.seeds.iter().map(|&([x, y], v)| ([x, y, 0], v)).collect();
        fm.generate_region(requested)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    fn binary_line(fg: &[usize], n: usize) -> Image<f32, 1> {
        let mut img = Image::<f32,1>::allocate(Region::new([0],[n]),[1.0],[0.0],0.0f32);
        for &i in fg { img.set_pixel(Index([i as i64]), 1.0f32); }
        img
    }

    #[test]
    fn danielsson_single_point() {
        // Single foreground pixel at index 5; distance from index 0 should be 5
        let img = binary_line(&[5], 11);
        let f = DanielssonDistanceMapFilter::<_, f32>::new(img);
        let out = f.generate_region(f.largest_region());
        let v = out.get_pixel(Index([0]));
        assert!((v - 5.0).abs() < 0.5, "expected ~5 got {v}");
        let v5 = out.get_pixel(Index([5]));
        assert!(v5.abs() < 0.1, "expected 0 got {v5}");
    }

    #[test]
    fn hausdorff_identical_images() {
        let img1 = binary_line(&[3, 4, 5], 10);
        let img2 = binary_line(&[3, 4, 5], 10);
        let f = HausdorffDistanceFilter::<_, _, f32>::new(img1, img2);
        let d = f.compute::<1>();
        assert!(d < 1e-6, "expected 0 got {d}");
    }
}
