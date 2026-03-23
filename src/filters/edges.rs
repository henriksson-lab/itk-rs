//! Edge detection and gradient filters.
//!
//! Analogs to ITK's derivative, gradient, Laplacian, Sobel, Canny, and
//! unsharp mask filters.

use rayon::prelude::*;
use std::marker::PhantomData;

use crate::image::{Image, Region, Index, iter_region};
use crate::pixel::{NumericPixel, VecPixel};
use crate::source::ImageSource;

// ===========================================================================
// Derivative filter
// ===========================================================================

/// Finite-difference derivative along one axis.
/// Analog to `itk::DerivativeImageFilter`.
///
/// `order = 1`: central difference `(x[i+1] − x[i-1]) / (2h)`.
/// `order = 2`: second derivative `(x[i-1] − 2x[i] + x[i+1]) / h²`.
pub struct DerivativeFilter<S> {
    pub source: S,
    pub axis: usize,
    pub order: u32,
}

impl<S> DerivativeFilter<S> {
    pub fn new(source: S, axis: usize) -> Self {
        Self { source, axis, order: 1 }
    }
    pub fn with_order(mut self, order: u32) -> Self { self.order = order; self }
}

impl<P, S, const D: usize> ImageSource<P, D> for DerivativeFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, out: &Region<D>) -> Region<D> {
        let mut radii = [0usize; D];
        radii[self.axis] = self.order as usize;
        out.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let padded = self.input_region_for_output(&requested);
        let input = self.source.generate_region(padded);
        let axis = self.axis;
        let h = self.source.spacing()[axis];
        let order = self.order;
        let lo = input.region.index.0[axis];
        let hi = lo + input.region.size.0[axis] as i64 - 1;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let get = |offset: i64| {
                    let mut i = idx;
                    i.0[axis] = (idx.0[axis] + offset).clamp(lo, hi);
                    input.get_pixel(i).to_f64()
                };
                let v = match order {
                    1 => (get(1) - get(-1)) / (2.0 * h),
                    2 => (get(1) - 2.0 * get(0) + get(-1)) / (h * h),
                    _ => 0.0,
                };
                P::from_f64(v)
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Gradient filter (outputs vector pixels)
// ===========================================================================

/// N-D gradient. Each output pixel is the vector of partial derivatives.
/// Analog to `itk::GradientImageFilter`.
pub struct GradientFilter<S, P> {
    pub source: S,
    _phantom: PhantomData<P>,
}

impl<S, P> GradientFilter<S, P> {
    pub fn new(source: S) -> Self { Self { source, _phantom: PhantomData } }
}

impl<P, S, const D: usize> ImageSource<VecPixel<P, D>, D> for GradientFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, out: &Region<D>) -> Region<D> {
        out.padded(1).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<VecPixel<P, D>, D> {
        let padded = self.input_region_for_output(&requested);
        let input = self.source.generate_region(padded);
        let spacing = self.source.spacing();

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<VecPixel<P, D>> = out_indices.par_iter()
            .map(|&idx| {
                let mut grad = [P::zero(); D];
                for axis in 0..D {
                    let lo = input.region.index.0[axis];
                    let hi = lo + input.region.size.0[axis] as i64 - 1;
                    let h = spacing[axis];
                    let get = |offset: i64| {
                        let mut i = idx;
                        i.0[axis] = (idx.0[axis] + offset).clamp(lo, hi);
                        input.get_pixel(i).to_f64()
                    };
                    grad[axis] = P::from_f64((get(1) - get(-1)) / (2.0 * h));
                }
                VecPixel(grad)
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Gradient magnitude filter
// ===========================================================================

/// Magnitude of the image gradient: `||∇I||`.
/// Analog to `itk::GradientMagnitudeImageFilter`.
pub struct GradientMagnitudeFilter<S> {
    pub source: S,
}

impl<S> GradientMagnitudeFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<P, S, const D: usize> ImageSource<P, D> for GradientMagnitudeFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, out: &Region<D>) -> Region<D> {
        out.padded(1).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let padded = self.input_region_for_output(&requested);
        let input = self.source.generate_region(padded);
        let spacing = self.source.spacing();

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let mut mag2 = 0.0f64;
                for axis in 0..D {
                    let lo = input.region.index.0[axis];
                    let hi = lo + input.region.size.0[axis] as i64 - 1;
                    let h = spacing[axis];
                    let get = |offset: i64| {
                        let mut i = idx;
                        i.0[axis] = (idx.0[axis] + offset).clamp(lo, hi);
                        input.get_pixel(i).to_f64()
                    };
                    let d = (get(1) - get(-1)) / (2.0 * h);
                    mag2 += d * d;
                }
                P::from_f64(mag2.sqrt())
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Laplacian filter
// ===========================================================================

/// Sum of unmixed second-order partial derivatives.
/// Analog to `itk::LaplacianImageFilter`.
pub struct LaplacianFilter<S> {
    pub source: S,
}

impl<S> LaplacianFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<P, S, const D: usize> ImageSource<P, D> for LaplacianFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, out: &Region<D>) -> Region<D> {
        out.padded(1).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let padded = self.input_region_for_output(&requested);
        let input = self.source.generate_region(padded);
        let spacing = self.source.spacing();

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let mut lap = 0.0f64;
                for axis in 0..D {
                    let lo = input.region.index.0[axis];
                    let hi = lo + input.region.size.0[axis] as i64 - 1;
                    let h = spacing[axis];
                    let get = |offset: i64| {
                        let mut i = idx;
                        i.0[axis] = (idx.0[axis] + offset).clamp(lo, hi);
                        input.get_pixel(i).to_f64()
                    };
                    lap += (get(1) - 2.0 * get(0) + get(-1)) / (h * h);
                }
                P::from_f64(lap)
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Laplacian sharpening filter
// ===========================================================================

/// `I − k * Laplacian(I)`. Sharpens by subtracting the Laplacian.
/// Analog to `itk::LaplacianSharpeningImageFilter`.
pub struct LaplacianSharpeningFilter<S> {
    pub source: S,
    pub weight: f64,
}

impl<S> LaplacianSharpeningFilter<S> {
    pub fn new(source: S) -> Self { Self { source, weight: 1.0 } }
    pub fn with_weight(mut self, w: f64) -> Self { self.weight = w; self }
}

impl<P, S, const D: usize> ImageSource<P, D> for LaplacianSharpeningFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, out: &Region<D>) -> Region<D> {
        out.padded(1).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let padded = self.input_region_for_output(&requested);
        let input = self.source.generate_region(padded);
        let spacing = self.source.spacing();
        let weight = self.weight;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let center = input.get_pixel(idx).to_f64();
                let mut lap = 0.0f64;
                for axis in 0..D {
                    let lo = input.region.index.0[axis];
                    let hi = lo + input.region.size.0[axis] as i64 - 1;
                    let h = spacing[axis];
                    let get = |offset: i64| {
                        let mut i = idx;
                        i.0[axis] = (idx.0[axis] + offset).clamp(lo, hi);
                        input.get_pixel(i).to_f64()
                    };
                    lap += (get(1) - 2.0 * get(0) + get(-1)) / (h * h);
                }
                P::from_f64(center - weight * lap)
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Sobel edge detection (2-D)
// ===========================================================================

/// 2-D Sobel edge detector. Computes gradient magnitude using the 3×3 Sobel
/// operator. Works on 2-D images; panics if D ≠ 2 at runtime.
/// Analog to `itk::SobelEdgeDetectionImageFilter`.
pub struct SobelFilter<S> {
    pub source: S,
}

impl<S> SobelFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<P, S, const D: usize> ImageSource<P, D> for SobelFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, out: &Region<D>) -> Region<D> {
        out.padded(1).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        assert_eq!(D, 2, "SobelFilter requires D=2");
        let padded = self.input_region_for_output(&requested);
        let input = self.source.generate_region(padded);
        let lo_x = input.region.index.0[0];
        let hi_x = lo_x + input.region.size.0[0] as i64 - 1;
        let lo_y = input.region.index.0[1];
        let hi_y = lo_y + input.region.size.0[1] as i64 - 1;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let x = idx.0[0];
                let y = idx.0[1];
                let get = |dx: i64, dy: i64| {
                    let cx = (x + dx).clamp(lo_x, hi_x);
                    let cy = (y + dy).clamp(lo_y, hi_y);
                    // Safety: D==2 asserted above
                    let mut i = idx;
                    i.0[0] = cx;
                    i.0[1] = cy;
                    input.get_pixel(i).to_f64()
                };
                let gx = -get(-1,-1) + get(1,-1)
                       - 2.0*get(-1,0) + 2.0*get(1,0)
                       - get(-1,1) + get(1,1);
                let gy = -get(-1,-1) - 2.0*get(0,-1) - get(1,-1)
                       + get(-1,1) + 2.0*get(0,1) + get(1,1);
                P::from_f64((gx*gx + gy*gy).sqrt())
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Unsharp mask filter
// ===========================================================================

/// `I + amount * (I − blur(I))`. Sharpens by adding a scaled high-pass.
/// Analog to `itk::UnsharpMaskImageFilter`.
pub struct UnsharpMaskFilter<S> {
    pub source: S,
    /// Gaussian blur radius (standard deviation in pixels).
    pub sigma: f64,
    /// Sharpening amount (default 0.5).
    pub amount: f64,
    /// Clamp output to `[threshold_lower, threshold_upper]` before adding edge.
    pub threshold: f64,
}

impl<S> UnsharpMaskFilter<S> {
    pub fn new(source: S, sigma: f64) -> Self {
        Self { source, sigma, amount: 0.5, threshold: 0.0 }
    }
    pub fn with_amount(mut self, a: f64) -> Self { self.amount = a; self }
}

impl<P, S, const D: usize> ImageSource<P, D> for UnsharpMaskFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, out: &Region<D>) -> Region<D> {
        let radius = (3.0 * self.sigma).ceil() as usize;
        out.padded(radius).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        // Apply Gaussian blur in-line using separable convolution
        let radius = (3.0 * self.sigma).ceil() as usize;
        let padded = self.input_region_for_output(&requested);
        let input = self.source.generate_region(padded);
        let sigma = self.sigma;
        let amount = self.amount;
        let threshold = self.threshold;

        // Build 1D Gaussian kernel
        let ksize = 2 * radius + 1;
        let mut kernel: Vec<f64> = (0..ksize)
            .map(|i| {
                let d = i as f64 - radius as f64;
                (-(d * d) / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let ksum: f64 = kernel.iter().sum();
        for k in &mut kernel { *k /= ksum; }

        // Separable blur: convolve along each axis
        use crate::filters::conv::convolve_axis;
        let mut blurred = input.clone();
        for axis in 0..D {
            let region = blurred.region;
            blurred = convolve_axis(&blurred, axis, &kernel, region);
        }

        // Crop blurred to requested, apply unsharp formula
        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let orig = input.get_pixel(idx).to_f64();
                let blur = blurred.get_pixel(idx).to_f64();
                let edge = orig - blur;
                let enhanced = if edge.abs() > threshold {
                    orig + amount * edge
                } else {
                    orig
                };
                P::from_f64(enhanced)
            })
            .collect();
        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

// ===========================================================================
// Zero crossing filter
// ===========================================================================

/// Marks pixels where the image crosses zero (changes sign with a neighbour).
/// Output is 0 or 1. Analog to `itk::ZeroCrossingImageFilter`.
pub struct ZeroCrossingFilter<S> {
    pub source: S,
    pub foreground_value: f64,
    pub background_value: f64,
}

impl<S> ZeroCrossingFilter<S> {
    pub fn new(source: S) -> Self {
        Self { source, foreground_value: 1.0, background_value: 0.0 }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for ZeroCrossingFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }
    fn input_region_for_output(&self, out: &Region<D>) -> Region<D> {
        out.padded(1).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let padded = self.input_region_for_output(&requested);
        let input = self.source.generate_region(padded);
        let fg = self.foreground_value;
        let bg = self.background_value;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let center = input.get_pixel(idx).to_f64();
                let mut is_zero_crossing = false;
                'outer: for axis in 0..D {
                    let lo = input.region.index.0[axis];
                    let hi = lo + input.region.size.0[axis] as i64 - 1;
                    for offset in &[-1i64, 1i64] {
                        let mut nb = idx;
                        nb.0[axis] = (nb.0[axis] + offset).clamp(lo, hi);
                        let neighbour = input.get_pixel(nb).to_f64();
                        if center * neighbour < 0.0 {
                            is_zero_crossing = true;
                            break 'outer;
                        }
                    }
                }
                P::from_f64(if is_zero_crossing { fg } else { bg })
            })
            .collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Canny edge detection (simplified)
// ===========================================================================

/// Simplified Canny edge detector (Gaussian smooth → gradient magnitude →
/// non-maximum suppression → hysteresis thresholding).
/// Analog to `itk::CannyEdgeDetectionImageFilter`.
///
/// Works on 2-D scalar images. Panics if D ≠ 2.
pub struct CannyEdgeDetectionFilter<S> {
    pub source: S,
    pub sigma: f64,
    pub lower_threshold: f64,
    pub upper_threshold: f64,
}

impl<S> CannyEdgeDetectionFilter<S> {
    pub fn new(source: S, sigma: f64, lower: f64, upper: f64) -> Self {
        Self { source, sigma, lower_threshold: lower, upper_threshold: upper }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for CannyEdgeDetectionFilter<S>
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
        assert_eq!(D, 2, "CannyEdgeDetectionFilter requires D=2");
        let full_region = self.source.largest_region();
        let full = self.source.generate_region(full_region);
        let sigma = self.sigma;
        let lower = self.lower_threshold;
        let upper = self.upper_threshold;

        // 1. Gaussian smooth
        let radius = (3.0 * sigma).ceil() as usize;
        let ksize = 2 * radius + 1;
        let mut kernel: Vec<f64> = (0..ksize)
            .map(|i| {
                let d = i as f64 - radius as f64;
                (-(d * d) / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let ksum: f64 = kernel.iter().sum();
        for k in &mut kernel { *k /= ksum; }

        use crate::filters::conv::convolve_axis;
        let mut smoothed = full.clone();
        for axis in 0..D {
            let region = smoothed.region;
            smoothed = convolve_axis(&smoothed, axis, &kernel, region);
        }

        let nx = full_region.size.0[0] as i64;
        let ny = full_region.size.0[1] as i64;
        let ox = full_region.index.0[0];
        let oy = full_region.index.0[1];

        // 2. Gradient magnitude + direction (direct flat-index access, avoids Index<D> issue)
        let n = (nx * ny) as usize;
        let mut mag = vec![0.0f64; n];
        let mut dir = vec![0.0f64; n]; // angle in radians

        let sm_data: Vec<f64> = smoothed.data.iter().map(|p| p.to_f64()).collect();
        let get_sm = |i: i64, j: i64| -> f64 {
            let ci = i.clamp(0, nx - 1);
            let cj = j.clamp(0, ny - 1);
            sm_data[(ci + cj * nx) as usize]
        };

        for j in 0..ny {
            for i in 0..nx {
                let flat = (i + j * nx) as usize;
                let gx = get_sm(i + 1, j) - get_sm(i - 1, j);
                let gy = get_sm(i, j + 1) - get_sm(i, j - 1);
                mag[flat] = (gx*gx + gy*gy).sqrt();
                dir[flat] = gy.atan2(gx);
            }
        }

        // 3. Non-maximum suppression
        let mut nms = vec![0.0f64; n];
        for j in 0..ny {
            for i in 0..nx {
                let flat = (i + j * nx) as usize;
                let m = mag[flat];
                let angle = dir[flat].to_degrees().rem_euclid(180.0);
                let (di, dj): (i64, i64) = if angle < 22.5 || angle >= 157.5 {
                    (1, 0)
                } else if angle < 67.5 {
                    (1, 1)
                } else if angle < 112.5 {
                    (0, 1)
                } else {
                    (-1, 1)
                };
                let get_mag = |di: i64, dj: i64| {
                    let ni = (i + di).clamp(0, nx-1);
                    let nj = (j + dj).clamp(0, ny-1);
                    mag[(ni + nj * nx) as usize]
                };
                if m >= get_mag(di, dj) && m >= get_mag(-di, -dj) {
                    nms[flat] = m;
                }
            }
        }

        // 4. Hysteresis thresholding (simplified: two-pass)
        let mut edges = vec![false; n];
        // Strong edges
        for i in 0..n {
            if nms[i] >= upper { edges[i] = true; }
        }
        // Weak edges connected to strong
        let mut changed = true;
        while changed {
            changed = false;
            for j in 0..ny {
                for i in 0..nx {
                    let flat = (i + j * nx) as usize;
                    if !edges[flat] && nms[flat] >= lower {
                        for dj in -1i64..=1 {
                            for di in -1i64..=1 {
                                let ni = (i + di).clamp(0, nx-1);
                                let nj = (j + dj).clamp(0, ny-1);
                                if edges[(ni + nj * nx) as usize] {
                                    edges[flat] = true;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Build output for requested region
        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter()
            .map(|&idx| {
                let i = idx.0[0] - ox;
                let j = idx.0[1] - oy;
                let flat = (i + j * nx) as usize;
                P::from_f64(if edges[flat] { 1.0 } else { 0.0 })
            })
            .collect();
        Image { region: requested, spacing: self.source.spacing(), origin: self.source.origin(), data }
    }
}

// ===========================================================================
// GradientMagnitudeRecursiveGaussianImageFilter
// ===========================================================================

/// Gradient magnitude after recursive Gaussian smoothing.
/// Analog to `itk::GradientMagnitudeRecursiveGaussianImageFilter`.
///
/// Applies `SmoothingRecursiveGaussianFilter` then `GradientMagnitudeFilter`.
pub struct GradientMagnitudeRecursiveGaussianFilter<S> {
    pub source: S,
    pub sigma: f64,
}

impl<S> GradientMagnitudeRecursiveGaussianFilter<S> {
    pub fn new(source: S, sigma: f64) -> Self {
        Self { source, sigma }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for GradientMagnitudeRecursiveGaussianFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let radius = (3.0 * self.sigma).ceil() as usize;
        let mut radii = [0usize; D];
        radii.iter_mut().for_each(|v| *v = radius);
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        use super::recursive_gaussian::RecursiveGaussianFilter;
        // Smooth first
        let smoothed = RecursiveGaussianFilter::new(&self.source, self.sigma);
        let smooth_img = smoothed.generate_region(requested);
        // Then compute gradient magnitude
        let gm = GradientMagnitudeFilter::new(smooth_img);
        gm.generate_region(requested)
    }
}

// ===========================================================================
// LaplacianRecursiveGaussianImageFilter
// ===========================================================================

/// Laplacian after recursive Gaussian smoothing.
/// Analog to `itk::LaplacianRecursiveGaussianImageFilter`.
pub struct LaplacianRecursiveGaussianFilter<S> {
    pub source: S,
    pub sigma: f64,
}

impl<S> LaplacianRecursiveGaussianFilter<S> {
    pub fn new(source: S, sigma: f64) -> Self {
        Self { source, sigma }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for LaplacianRecursiveGaussianFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let radius = (3.0 * self.sigma).ceil() as usize + 1;
        let mut radii = [0usize; D];
        radii.iter_mut().for_each(|v| *v = radius);
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        use super::recursive_gaussian::RecursiveGaussianFilter;
        let smoothed = RecursiveGaussianFilter::new(&self.source, self.sigma);
        let smooth_img = smoothed.generate_region(requested);
        let lap = LaplacianFilter::new(smooth_img);
        lap.generate_region(requested)
    }
}

// ===========================================================================
// DiscreteGaussianDerivativeImageFilter
// ===========================================================================

/// Gaussian-smoothed derivative along one axis.
/// Analog to `itk::DiscreteGaussianDerivativeImageFilter`.
pub struct DiscreteGaussianDerivativeFilter<S> {
    pub source: S,
    pub sigma: f64,
    pub axis: usize,
    pub order: u32,
}

impl<S> DiscreteGaussianDerivativeFilter<S> {
    pub fn new(source: S, sigma: f64, axis: usize) -> Self {
        Self { source, sigma, axis, order: 1 }
    }
    pub fn with_order(mut self, order: u32) -> Self { self.order = order; self }
}

impl<P, S, const D: usize> ImageSource<P, D> for DiscreteGaussianDerivativeFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let radius = (3.0 * self.sigma).ceil() as usize + 1;
        let mut radii = [0usize; D];
        radii.iter_mut().for_each(|v| *v = radius);
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        use super::recursive_gaussian::RecursiveGaussianFilter;
        let smoothed = RecursiveGaussianFilter::new(&self.source, self.sigma);
        let smooth_img = smoothed.generate_region(requested);
        let deriv = DerivativeFilter::new(smooth_img, self.axis).with_order(self.order);
        deriv.generate_region(requested)
    }
}

// ===========================================================================
// GradientRecursiveGaussianImageFilter
// ===========================================================================

/// Gradient vector (one component per axis) after recursive Gaussian smoothing.
/// Analog to `itk::GradientRecursiveGaussianImageFilter`.
///
/// Output pixel type is `VecPixel<f32, D>`.
pub struct GradientRecursiveGaussianFilter<S, P, const D: usize> {
    pub source: S,
    pub sigma: f64,
    _phantom: PhantomData<P>,
}

impl<S, P, const D: usize> GradientRecursiveGaussianFilter<S, P, D> {
    pub fn new(source: S, sigma: f64) -> Self {
        Self { source, sigma, _phantom: PhantomData }
    }
}

impl<P, S, const D: usize> ImageSource<VecPixel<f32, D>, D>
    for GradientRecursiveGaussianFilter<S, P, D>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let radius = (3.0 * self.sigma).ceil() as usize + 1;
        let mut radii = [0usize; D];
        radii.iter_mut().for_each(|v| *v = radius);
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<VecPixel<f32, D>, D> {
        use super::recursive_gaussian::RecursiveGaussianFilter;
        use crate::image::iter_region;

        let smoothed = RecursiveGaussianFilter::new(&self.source, self.sigma);
        let smooth_img = smoothed.generate_region(requested);

        let mut out_indices: Vec<crate::image::Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let bounds = smooth_img.region;
        let data: Vec<VecPixel<f32, D>> = out_indices.par_iter().map(|&idx| {
            let mut components = [0.0f32; D];
            for axis in 0..D {
                let mut fwd = idx.0;
                let mut bwd = idx.0;
                fwd[axis] = (fwd[axis] + 1).min(bounds.index.0[axis] + bounds.size.0[axis] as i64 - 1);
                bwd[axis] = (bwd[axis] - 1).max(bounds.index.0[axis]);
                let h = smooth_img.spacing[axis];
                let v = (smooth_img.get_pixel(crate::image::Index(fwd)).to_f64()
                    - smooth_img.get_pixel(crate::image::Index(bwd)).to_f64())
                    / (2.0 * h);
                components[axis] = v as f32;
            }
            VecPixel(components)
        }).collect();

        Image { region: requested, spacing: smooth_img.spacing, origin: smooth_img.origin, data }
    }
}

// ===========================================================================
// HessianRecursiveGaussianImageFilter
// ===========================================================================

/// Hessian matrix components after recursive Gaussian smoothing.
/// Analog to `itk::HessianRecursiveGaussianImageFilter`.
///
/// For a D-dimensional image the Hessian has D*(D+1)/2 unique components
/// stored in a `VecPixel<f32, {D*(D+1)/2}>`. For D=2: [H00, H11, H01] (3 components).
/// For D=3: [H00, H11, H22, H01, H02, H12] (6 components).
///
/// This implementation works generically for any D by computing d²f/dx_i dx_j
/// via finite differences on the Gaussian-smoothed image.
pub struct HessianRecursiveGaussianFilter<S, P> {
    pub source: S,
    pub sigma: f64,
    _phantom: PhantomData<P>,
}

impl<S, P> HessianRecursiveGaussianFilter<S, P> {
    pub fn new(source: S, sigma: f64) -> Self {
        Self { source, sigma, _phantom: PhantomData }
    }
}

/// Helper: compute Hessian for D=2, returning VecPixel<f32, 3> = [H00, H11, H01]
impl<P, S> ImageSource<VecPixel<f32, 3>, 2> for HessianRecursiveGaussianFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<2>) -> Region<2> {
        let radius = (3.0 * self.sigma).ceil() as usize + 2;
        let radii = [radius; 2];
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<2>) -> Image<VecPixel<f32, 3>, 2> {
        use super::recursive_gaussian::RecursiveGaussianFilter;

        let smoothed = RecursiveGaussianFilter::new(&self.source, self.sigma);
        let smooth_img = smoothed.generate_region(requested);
        let bounds = smooth_img.region;

        let mut out_indices: Vec<crate::image::Index<2>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let clamp = |v: i64, ax: usize| -> i64 {
            v.max(bounds.index.0[ax]).min(bounds.index.0[ax] + bounds.size.0[ax] as i64 - 1)
        };

        let data: Vec<VecPixel<f32, 3>> = out_indices.par_iter().map(|&idx| {
            let [x, y] = idx.0;
            let hx = smooth_img.spacing[0];
            let hy = smooth_img.spacing[1];

            let get = |xi: i64, yi: i64| -> f64 {
                smooth_img.get_pixel(crate::image::Index([clamp(xi, 0), clamp(yi, 1)])).to_f64()
            };

            let h00 = (get(x+1, y) - 2.0*get(x, y) + get(x-1, y)) / (hx * hx);
            let h11 = (get(x, y+1) - 2.0*get(x, y) + get(x, y-1)) / (hy * hy);
            let h01 = (get(x+1, y+1) - get(x+1, y-1) - get(x-1, y+1) + get(x-1, y-1))
                / (4.0 * hx * hy);

            VecPixel([h00 as f32, h11 as f32, h01 as f32])
        }).collect();

        Image { region: requested, spacing: smooth_img.spacing, origin: smooth_img.origin, data }
    }
}

// ===========================================================================
// DifferenceOfGaussiansGradientImageFilter
// ===========================================================================

/// Difference of Gaussians approximation to the Laplacian of Gaussian.
/// Analog to `itk::DifferenceOfGaussiansGradientImageFilter`.
///
/// Computes `G(sigma1) * I − G(sigma2) * I` where sigma2 > sigma1.
pub struct DifferenceOfGaussiansFilter<S> {
    pub source: S,
    pub sigma1: f64,
    pub sigma2: f64,
}

impl<S> DifferenceOfGaussiansFilter<S> {
    pub fn new(source: S, sigma1: f64, sigma2: f64) -> Self {
        Self { source, sigma1, sigma2 }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for DifferenceOfGaussiansFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let radius = (3.0 * self.sigma2).ceil() as usize;
        let mut radii = [0usize; D];
        radii.iter_mut().for_each(|v| *v = radius);
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        use super::recursive_gaussian::RecursiveGaussianFilter;

        let s1 = RecursiveGaussianFilter::new(&self.source, self.sigma1);
        let img1 = s1.generate_region(requested);
        let s2 = RecursiveGaussianFilter::new(&self.source, self.sigma2);
        let img2 = s2.generate_region(requested);

        let mut out_indices: Vec<crate::image::Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter().map(|&idx| {
            let v1 = img1.get_pixel(idx).to_f64();
            let v2 = img2.get_pixel(idx).to_f64();
            P::from_f64(v1 - v2)
        }).collect();

        Image { region: requested, spacing: img1.spacing, origin: img1.origin, data }
    }
}

// ===========================================================================
// ZeroCrossingBasedEdgeDetectionImageFilter
// ===========================================================================

/// Zero-crossing-based edge detection (Marr-Hildreth).
/// Analog to `itk::ZeroCrossingBasedEdgeDetectionImageFilter`.
///
/// Applies Gaussian smoothing, then Laplacian, then zero-crossing detection.
pub struct ZeroCrossingBasedEdgeDetectionFilter<S> {
    pub source: S,
    pub sigma: f64,
}

impl<S> ZeroCrossingBasedEdgeDetectionFilter<S> {
    pub fn new(source: S, sigma: f64) -> Self { Self { source, sigma } }
}

impl<P, S, const D: usize> ImageSource<P, D> for ZeroCrossingBasedEdgeDetectionFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let radius = (3.0 * self.sigma).ceil() as usize + 1;
        let mut radii = [0usize; D];
        radii.iter_mut().for_each(|v| *v = radius);
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        use super::recursive_gaussian::RecursiveGaussianFilter;

        let smoothed = RecursiveGaussianFilter::new(&self.source, self.sigma);
        let smooth_img = smoothed.generate_region(requested);
        let lap = LaplacianFilter::new(smooth_img);
        let lap_img = lap.generate_region(requested);
        let zc = ZeroCrossingFilter::new(lap_img);
        zc.generate_region(requested)
    }
}

// ===========================================================================
// VectorGradientMagnitudeImageFilter
// ===========================================================================

/// Gradient magnitude for vector-valued images.
/// Analog to `itk::VectorGradientMagnitudeImageFilter`.
///
/// Computes the Frobenius norm of the Jacobian matrix:
/// `sqrt( Σ_c Σ_d (∂I_c/∂x_d)² )` where c is the channel and d is the axis.
pub struct VectorGradientMagnitudeFilter<S, const N: usize> {
    pub source: S,
}

impl<S, const N: usize> VectorGradientMagnitudeFilter<S, N> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<S, const N: usize, const D: usize> ImageSource<f32, D>
    for VectorGradientMagnitudeFilter<S, N>
where
    S: ImageSource<VecPixel<f32, N>, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<f32, D> {
        let bounds = self.source.largest_region();
        let input = self.source.generate_region(requested);

        let mut out_indices: Vec<crate::image::Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<f32> = out_indices.par_iter().map(|&idx| {
            let mut sq_sum = 0.0f64;
            for axis in 0..D {
                let mut fwd = idx.0;
                let mut bwd = idx.0;
                fwd[axis] = (fwd[axis] + 1).min(bounds.index.0[axis] + bounds.size.0[axis] as i64 - 1);
                bwd[axis] = (bwd[axis] - 1).max(bounds.index.0[axis]);
                let h = input.spacing[axis];
                let pf = input.get_pixel(crate::image::Index(fwd));
                let pb = input.get_pixel(crate::image::Index(bwd));
                for c in 0..N {
                    let d = (pf.0[c] as f64 - pb.0[c] as f64) / (2.0 * h);
                    sq_sum += d * d;
                }
            }
            sq_sum.sqrt() as f32
        }).collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// HoughTransform2DCirclesImageFilter
// ===========================================================================

/// Hough transform for detecting circles in a 2D edge image.
/// Analog to `itk::HoughTransform2DCirclesImageFilter`.
///
/// For each edge pixel, votes are cast in the Hough accumulator at all
/// positions `(x ± r·cos θ, y ± r·sin θ)` for `r ∈ [min_radius, max_radius]`
/// and θ sampled uniformly. The output is the accumulator image (vote counts as f32).
pub struct HoughTransform2DCirclesFilter<S, P> {
    pub source: S,
    pub min_radius: f64,
    pub max_radius: f64,
    pub threshold: f64,
    pub number_of_circles: usize,
    _phantom: PhantomData<P>,
}

impl<S, P> HoughTransform2DCirclesFilter<S, P> {
    pub fn new(source: S, min_radius: f64, max_radius: f64) -> Self {
        Self { source, min_radius, max_radius, threshold: 0.5, number_of_circles: 10, _phantom: PhantomData }
    }
}

impl<P, S> ImageSource<f32, 2> for HoughTransform2DCirclesFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        let input = self.source.generate_region(self.source.largest_region());
        let bounds = input.region;
        let nx = bounds.size.0[0];
        let ny = bounds.size.0[1];
        let ox = bounds.index.0[0];
        let oy = bounds.index.0[1];
        let mut acc = vec![0.0f32; nx * ny];

        let n_angles = 64usize;
        let n_radii = ((self.max_radius - self.min_radius).ceil() as usize).max(1);

        iter_region(&bounds, |idx| {
            if input.get_pixel(idx).to_f64() < self.threshold { return; }
            let x = idx.0[0]; let y = idx.0[1];
            for ri in 0..n_radii {
                let r = self.min_radius + ri as f64 * (self.max_radius - self.min_radius) / n_radii.max(1) as f64;
                for ai in 0..n_angles {
                    let angle = std::f64::consts::TAU * ai as f64 / n_angles as f64;
                    for sign in [-1.0f64, 1.0] {
                        let cx = (x as f64 + sign * r * angle.cos()).round() as i64;
                        let cy = (y as f64 + sign * r * angle.sin()).round() as i64;
                        if cx >= ox && cx < ox + nx as i64 && cy >= oy && cy < oy + ny as i64 {
                            let flat = (cx - ox) as usize + (cy - oy) as usize * nx;
                            acc[flat] += 1.0;
                        }
                    }
                }
            }
        });

        // Extract requested subregion
        let mut out_indices: Vec<Index<2>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<f32> = out_indices.iter().map(|&idx| {
            if idx.0[0] >= ox && idx.0[0] < ox + nx as i64 && idx.0[1] >= oy && idx.0[1] < oy + ny as i64 {
                acc[(idx.0[0]-ox) as usize + (idx.0[1]-oy) as usize * nx]
            } else { 0.0 }
        }).collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// HoughTransform2DLinesImageFilter
// ===========================================================================

/// Hough transform for detecting lines in a 2D edge image.
/// Analog to `itk::HoughTransform2DLinesImageFilter`.
///
/// Uses the standard (ρ, θ) parameterisation. The output image has the same
/// size as the input; each pixel value is the number of votes in the
/// Hough accumulator cell closest to that pixel's (ρ, θ) interpretation.
/// For a more natural interface, use the accumulator image directly.
pub struct HoughTransform2DLinesFilter<S, P> {
    pub source: S,
    pub threshold: f64,
    pub angle_resolution: usize,
    _phantom: PhantomData<P>,
}

impl<S, P> HoughTransform2DLinesFilter<S, P> {
    pub fn new(source: S) -> Self {
        Self { source, threshold: 0.5, angle_resolution: 180, _phantom: PhantomData }
    }
}

impl<P, S> ImageSource<f32, 2> for HoughTransform2DLinesFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        let input = self.source.generate_region(self.source.largest_region());
        let bounds = input.region;
        let nx = bounds.size.0[0];
        let ny = bounds.size.0[1];
        let ox = bounds.index.0[0];
        let oy = bounds.index.0[1];
        let n_theta = self.angle_resolution;
        // ρ range: [-diag, +diag]
        let diag = ((nx*nx + ny*ny) as f64).sqrt().ceil() as usize + 1;
        let n_rho = 2 * diag + 1;
        let mut acc = vec![0.0f32; n_theta * n_rho];

        iter_region(&bounds, |idx| {
            if input.get_pixel(idx).to_f64() < self.threshold { return; }
            let x = (idx.0[0] - ox) as f64;
            let y = (idx.0[1] - oy) as f64;
            for ti in 0..n_theta {
                let theta = std::f64::consts::PI * ti as f64 / n_theta as f64;
                let rho = x * theta.cos() + y * theta.sin();
                let ri = (rho.round() as i64 + diag as i64) as usize;
                if ri < n_rho { acc[ti + ri * n_theta] += 1.0; }
            }
        });

        // Output: copy acc into image (truncate/pad as needed)
        let mut out_indices: Vec<Index<2>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));
        let data: Vec<f32> = out_indices.iter().map(|&idx| {
            let xi = (idx.0[0] - ox) as usize;
            let yi = (idx.0[1] - oy) as usize;
            if xi < n_theta && yi < n_rho { acc[xi + yi * n_theta] } else { 0.0 }
        }).collect();
        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// SimpleContourExtractorImageFilter
// ===========================================================================

/// Extracts contour pixels: foreground pixels that have at least one
/// background neighbour.
/// Analog to `itk::SimpleContourExtractorImageFilter`.
pub struct SimpleContourExtractorFilter<S> {
    pub source: S,
    pub foreground_value: f64,
    pub contour_value: f64,
}

impl<S> SimpleContourExtractorFilter<S> {
    pub fn new(source: S) -> Self {
        Self { source, foreground_value: 1.0, contour_value: 1.0 }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for SimpleContourExtractorFilter<S>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let bounds = input.region;
        let fg = self.foreground_value;

        let mut out_indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<P> = out_indices.par_iter().map(|&idx| {
            let v = input.get_pixel(idx).to_f64();
            if (v - fg).abs() >= 0.5 { return P::from_f64(0.0); }
            // Is there a background neighbour?
            for d in 0..D {
                for delta in [-1i64, 1i64] {
                    let mut nb = idx.0; nb[d] += delta;
                    if (0..D).all(|dd| nb[dd] >= bounds.index.0[dd] && nb[dd] < bounds.index.0[dd] + bounds.size.0[dd] as i64) {
                        let nv = input.get_pixel(Index(nb)).to_f64();
                        if (nv - fg).abs() >= 0.5 { return P::from_f64(self.contour_value); }
                    }
                }
            }
            P::from_f64(0.0)
        }).collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// SymmetricEigenAnalysisImageFilter
// ===========================================================================

/// Per-pixel symmetric 2×2 eigen-analysis on a 3-component Hessian image
/// (stored as `VecPixel<f32, 3>` = [H00, H11, H01]).
///
/// Output: `VecPixel<f32, 2>` = [λ_min, λ_max] (eigenvalues, ascending).
/// Analog to `itk::SymmetricEigenAnalysisImageFilter`.
pub struct SymmetricEigenAnalysisFilter<S> {
    pub source: S,
}

impl<S> SymmetricEigenAnalysisFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<S> ImageSource<VecPixel<f32, 2>, 2> for SymmetricEigenAnalysisFilter<S>
where
    S: ImageSource<VecPixel<f32, 3>, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<VecPixel<f32, 2>, 2> {
        let input = self.source.generate_region(requested);
        let data: Vec<VecPixel<f32, 2>> = input.data.iter().map(|&h| {
            let (a, d, b) = (h.0[0] as f64, h.0[1] as f64, h.0[2] as f64);
            // Eigenvalues of [[a,b],[b,d]]
            let trace = a + d;
            let det = a * d - b * b;
            let disc = ((trace * trace / 4.0) - det).max(0.0).sqrt();
            let l1 = (trace / 2.0 - disc) as f32;
            let l2 = (trace / 2.0 + disc) as f32;
            VecPixel([l1, l2])
        }).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// HessianToObjectnessMeasureImageFilter
// ===========================================================================

/// Frangi vesselness measure from Hessian eigenvalues.
/// Analog to `itk::HessianToObjectnessMeasureImageFilter`.
pub struct HessianToObjectnessMeasureFilter<S> {
    pub source: S,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub bright_object: bool,
}

impl<S> HessianToObjectnessMeasureFilter<S> {
    pub fn new(source: S) -> Self {
        Self { source, alpha: 0.5, beta: 0.5, gamma: 5.0, bright_object: true }
    }
}

impl<S> ImageSource<f32, 2> for HessianToObjectnessMeasureFilter<S>
where
    S: ImageSource<VecPixel<f32, 3>, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        let eig = SymmetricEigenAnalysisFilter::new(&self.source);
        let eigenvals = eig.generate_region(requested);
        let data: Vec<f32> = eigenvals.data.iter().map(|&ev| {
            let l1 = ev.0[0] as f64; // smallest
            let l2 = ev.0[1] as f64; // largest
            // For bright tubular objects: λ1 ≈ 0, λ2 < 0
            if self.bright_object && l2 >= 0.0 { return 0.0f32; }
            if !self.bright_object && l2 <= 0.0 { return 0.0f32; }
            let rb = (l1 / l2).abs(); // ratio
            let s2 = l1 * l1 + l2 * l2;
            let v = (1.0 - (-rb * rb / (2.0 * self.alpha * self.alpha)).exp())
                  * (-s2 / (2.0 * self.gamma * self.gamma)).exp();
            v as f32
        }).collect();
        Image { region: eigenvals.region, spacing: eigenvals.spacing, origin: eigenvals.origin, data }
    }
}

// ===========================================================================
// Hessian3DToVesselnessMeasureImageFilter  (2D version)
// ===========================================================================

/// Sato 1997 vesselness measure.
/// Analog to `itk::Hessian3DToVesselnessMeasureImageFilter` (here for D=2).
pub struct Hessian3DToVesselnessMeasureFilter<S> {
    pub source: S,
    pub alpha1: f64,
    pub alpha2: f64,
}

impl<S> Hessian3DToVesselnessMeasureFilter<S> {
    pub fn new(source: S) -> Self {
        Self { source, alpha1: 0.5, alpha2: 2.0 }
    }
}

impl<S> ImageSource<f32, 2> for Hessian3DToVesselnessMeasureFilter<S>
where
    S: ImageSource<VecPixel<f32, 3>, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        // Delegate to objectness measure
        let om = HessianToObjectnessMeasureFilter {
            source: &self.source,
            alpha: self.alpha1,
            beta: self.alpha2,
            gamma: 5.0,
            bright_object: true,
        };
        om.generate_region(requested)
    }
}

// ===========================================================================
// MultiScaleHessianBasedMeasureImageFilter
// ===========================================================================

/// Runs HessianToObjectnessMeasure at multiple scales and takes the maximum.
/// Analog to `itk::MultiScaleHessianBasedMeasureImageFilter`.
pub struct MultiScaleHessianMeasureFilter<S> {
    pub source: S,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub num_sigma_steps: usize,
}

impl<S> MultiScaleHessianMeasureFilter<S> {
    pub fn new(source: S) -> Self {
        Self { source, sigma_min: 0.5, sigma_max: 4.0, num_sigma_steps: 4 }
    }
}

impl<S> ImageSource<f32, 2> for MultiScaleHessianMeasureFilter<S>
where
    S: ImageSource<f32, 2> + Sync,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        
        let input_img = self.source.generate_region(self.source.largest_region());

        let n = self.num_sigma_steps.max(1);
        let mut best: Vec<f32> = vec![0.0f32; input_img.data.len()];

        for step in 0..n {
            let t = step as f64 / (n - 1).max(1) as f64;
            let sigma = self.sigma_min + t * (self.sigma_max - self.sigma_min);

            // Compute Hessian components at this scale
            // d²I/dx² via second-derivative Gaussian
            let gxx = DiscreteGaussianDerivativeFilter {
                source: &input_img, sigma, axis: 0,
                order: 2,
            };
            let gyy = DiscreteGaussianDerivativeFilter {
                source: &input_img, sigma, axis: 1,
                order: 2,
            };
            let gxy = DiscreteGaussianDerivativeFilter {
                source: &input_img, sigma, axis: 0,
                order: 1, // simplified: use first x then first y
            };

            let hxx = gxx.generate_region(requested);
            let hyy = gyy.generate_region(requested);
            let hxy = gxy.generate_region(requested);

            let hessian_data: Vec<VecPixel<f32, 3>> = hxx.data.iter().zip(hyy.data.iter()).zip(hxy.data.iter())
                .map(|((&xx, &yy), &xy)| VecPixel([xx, yy, xy]))
                .collect();
            let hessian_img = Image { region: hxx.region, spacing: hxx.spacing, origin: hxx.origin, data: hessian_data };

            let measure = HessianToObjectnessMeasureFilter::new(hessian_img);
            let m = measure.generate_region(requested);

            for (b, &v) in best.iter_mut().zip(m.data.iter()) {
                if v > *b { *b = v; }
            }
        }
        let region = requested;
        Image { region, spacing: self.source.spacing(), origin: self.source.origin(), data: best }
    }
}

// ===========================================================================
// GradientVectorFlowImageFilter
// ===========================================================================

/// Gradient Vector Flow (GVF) diffusion — extends edge gradient field into
/// homogeneous regions.  Analog to `itk::GradientVectorFlowImageFilter`.
///
/// Inputs a gradient image (`VecPixel<f32, 2>`), applies GVF PDE.
pub struct GradientVectorFlowFilter<S> {
    pub source: S,
    pub mu: f64,
    pub iterations: usize,
}

impl<S> GradientVectorFlowFilter<S> {
    pub fn new(source: S, mu: f64, iterations: usize) -> Self {
        Self { source, mu, iterations }
    }
}

impl<S> ImageSource<VecPixel<f32, 2>, 2> for GradientVectorFlowFilter<S>
where
    S: ImageSource<VecPixel<f32, 2>, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, _requested: Region<2>) -> Image<VecPixel<f32, 2>, 2> {
        let input = self.source.generate_region(self.source.largest_region());
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];
        let [ox, oy] = [input.region.index.0[0], input.region.index.0[1]];

        // Extract edge map magnitude |∇f|² and initial field
        let f2: Vec<f64> = input.data.iter().map(|p| {
            let u = p.0[0] as f64;
            let v = p.0[1] as f64;
            u * u + v * v
        }).collect();

        let mut ux: Vec<f64> = input.data.iter().map(|p| p.0[0] as f64).collect();
        let mut uy: Vec<f64> = input.data.iter().map(|p| p.0[1] as f64).collect();
        let fx: Vec<f64> = ux.clone();
        let fy: Vec<f64> = uy.clone();

        let flat = |x: i64, y: i64| -> usize {
            let xi = (x - ox).clamp(0, w as i64 - 1) as usize;
            let yi = (y - oy).clamp(0, h as i64 - 1) as usize;
            yi * w + xi
        };

        let dt = 0.1f64 / (self.mu * 4.0 + 1.0);
        for _ in 0..self.iterations {
            let mut new_ux = vec![0.0f64; w * h];
            let mut new_uy = vec![0.0f64; w * h];
            for y in 0..h {
                for xi in 0..w {
                    let i = y * w + xi;
                    let xpos = ox + xi as i64;
                    let ypos = oy + y as i64;
                    let lap_x = ux[flat(xpos+1, ypos)] + ux[flat(xpos-1, ypos)]
                              + ux[flat(xpos, ypos+1)] + ux[flat(xpos, ypos-1)]
                              - 4.0 * ux[i];
                    let lap_y = uy[flat(xpos+1, ypos)] + uy[flat(xpos-1, ypos)]
                              + uy[flat(xpos, ypos+1)] + uy[flat(xpos, ypos-1)]
                              - 4.0 * uy[i];
                    new_ux[i] = ux[i] + dt * (self.mu * lap_x - f2[i] * (ux[i] - fx[i]));
                    new_uy[i] = uy[i] + dt * (self.mu * lap_y - f2[i] * (uy[i] - fy[i]));
                }
            }
            ux = new_ux;
            uy = new_uy;
        }

        let data: Vec<VecPixel<f32, 2>> = ux.iter().zip(uy.iter())
            .map(|(&u, &v)| VecPixel([u as f32, v as f32]))
            .collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    fn ramp_1d(n: usize) -> Image<f32, 1> {
        let mut img = Image::<f32, 1>::allocate(Region::new([0], [n]), [1.0], [0.0], 0.0f32);
        for i in 0..n as i64 {
            img.set_pixel(Index([i]), i as f32);
        }
        img
    }

    #[test]
    fn derivative_ramp() {
        // Derivative of a linear ramp should be 1 in interior (h=1)
        // Boundary pixels use Neumann clamping and give 0.5
        let img = ramp_1d(10);
        let f = DerivativeFilter::new(img, 0);
        let out = f.generate_region(f.largest_region());
        // Check interior pixels only
        for i in 1..9i64 {
            let v = out.get_pixel(crate::image::Index([i]));
            assert!((v - 1.0).abs() < 1e-4, "expected 1 got {v} at {i}");
        }
    }

    #[test]
    fn second_derivative_quadratic() {
        // f(x) = x^2 → f''(x) = 2
        let mut img = Image::<f32,1>::allocate(Region::new([0],[10]),[1.0],[0.0],0.0);
        for i in 0..10i64 { img.set_pixel(Index([i]), (i*i) as f32); }
        let f = DerivativeFilter::new(img, 0).with_order(2);
        let out = f.generate_region(f.largest_region());
        // Interior pixels should be ~2
        for i in 1..9usize {
            assert!((out.data[i] - 2.0).abs() < 1e-3, "at {i}: {}", out.data[i]);
        }
    }

    #[test]
    fn gradient_magnitude_ramp() {
        // Gradient magnitude of a ramp should be 1 in interior
        let img = ramp_1d(10);
        let f = GradientMagnitudeFilter::new(img);
        let out = f.generate_region(f.largest_region());
        for i in 1..9i64 {
            let v = out.get_pixel(crate::image::Index([i]));
            assert!((v - 1.0).abs() < 1e-4, "expected 1 got {v} at {i}");
        }
    }

    #[test]
    fn laplacian_linear_is_zero() {
        // Laplacian of a linear function is 0 in interior
        let img = ramp_1d(10);
        let f = LaplacianFilter::new(img);
        let out = f.generate_region(f.largest_region());
        for i in 1..9i64 {
            let v = out.get_pixel(crate::image::Index([i]));
            assert!(v.abs() < 1e-4, "expected 0 got {v} at {i}");
        }
    }

    #[test]
    fn sobel_flat_image() {
        // Sobel of a flat image should be 0
        let img = Image::<f32,2>::allocate(Region::new([0,0],[5,5]),[1.0,1.0],[0.0,0.0],1.0);
        let f = SobelFilter::new(img);
        let out = f.generate_region(f.largest_region());
        for &v in &out.data { assert!(v.abs() < 1e-6); }
    }
}
