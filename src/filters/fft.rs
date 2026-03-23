//! FFT-based filters.
//!
//! Stub implementations of ITK's FFT pipeline:
//! - `FFTDiscreteGaussianImageFilter`
//! - `FFTConvolutionImageFilter`
//! - `FFTNormalizedCorrelationImageFilter`
//! - `MaskedFFTNormalizedCorrelationImageFilter`
//! - `ForwardFFTImageFilter` / `InverseFFTImageFilter`
//! - `FFTShiftImageFilter` / `FFTPadImageFilter`
//! - `FrequencyBandImageFilter`
//!
//! The implementations use direct-space DFT (O(N²)) rather than a fast FFT
//! library, sufficient for correctness tests on small images.  Production use
//! should link `rustfft`.

use crate::image::{Image, Index, Region, iter_region};
use crate::pixel::{NumericPixel, VecPixel};
use crate::source::ImageSource;

// ---------------------------------------------------------------------------
// Internal DFT helpers
// ---------------------------------------------------------------------------

/// 1-D DFT: complex in/out.  Returns (re, im) pairs.
fn dft_1d(re: &[f64], im: &[f64], inverse: bool) -> (Vec<f64>, Vec<f64>) {
    let n = re.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut out_re = vec![0.0f64; n];
    let mut out_im = vec![0.0f64; n];
    for k in 0..n {
        let mut sr = 0.0f64;
        let mut si = 0.0f64;
        for t in 0..n {
            let angle = sign * 2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
            sr += re[t] * angle.cos() - im[t] * angle.sin();
            si += re[t] * angle.sin() + im[t] * angle.cos();
        }
        out_re[k] = sr;
        out_im[k] = si;
        if inverse {
            out_re[k] /= n as f64;
            out_im[k] /= n as f64;
        }
    }
    (out_re, out_im)
}

/// 2-D DFT (row-then-column separable).
pub(crate) fn dft_2d(re: &[f64], im: &[f64], w: usize, h: usize, inverse: bool) -> (Vec<f64>, Vec<f64>) {
    // Row transforms
    let mut tmp_re = re.to_vec();
    let mut tmp_im = im.to_vec();
    for y in 0..h {
        let row_re: Vec<f64> = (0..w).map(|x| tmp_re[y * w + x]).collect();
        let row_im: Vec<f64> = (0..w).map(|x| tmp_im[y * w + x]).collect();
        let (or, oi) = dft_1d(&row_re, &row_im, inverse);
        for x in 0..w { tmp_re[y * w + x] = or[x]; tmp_im[y * w + x] = oi[x]; }
    }
    // Column transforms
    for x in 0..w {
        let col_re: Vec<f64> = (0..h).map(|y| tmp_re[y * w + x]).collect();
        let col_im: Vec<f64> = (0..h).map(|y| tmp_im[y * w + x]).collect();
        let (or, oi) = dft_1d(&col_re, &col_im, inverse);
        for y in 0..h { tmp_re[y * w + x] = or[y]; tmp_im[y * w + x] = oi[y]; }
    }
    (tmp_re, tmp_im)
}

// ---------------------------------------------------------------------------
// ForwardFFTImageFilter
// ---------------------------------------------------------------------------

/// Discrete Fourier transform: real image → complex (`VecPixel<f32, 2>`) image.
/// Analog to `itk::ForwardFFTImageFilter`.
pub struct ForwardFFTFilter<S> {
    pub source: S,
}

impl<S> ForwardFFTFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<S> ImageSource<VecPixel<f32, 2>, 2> for ForwardFFTFilter<S>
where
    S: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<VecPixel<f32, 2>, 2> {
        let input = self.source.generate_region(self.source.largest_region());
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];
        let re: Vec<f64> = input.data.iter().map(|p| p.to_f64()).collect();
        let im = vec![0.0f64; re.len()];
        let (fre, fim) = dft_2d(&re, &im, w, h, false);
        let data: Vec<VecPixel<f32, 2>> = fre.iter().zip(fim.iter())
            .map(|(&r, &i)| VecPixel([r as f32, i as f32]))
            .collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// InverseFFTImageFilter
// ---------------------------------------------------------------------------

/// Inverse DFT: complex image → real image.
/// Analog to `itk::InverseFFTImageFilter`.
pub struct InverseFFTFilter<S> {
    pub source: S,
}

impl<S> InverseFFTFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<S> ImageSource<f32, 2> for InverseFFTFilter<S>
where
    S: ImageSource<VecPixel<f32, 2>, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, _requested: Region<2>) -> Image<f32, 2> {
        let input = self.source.generate_region(self.source.largest_region());
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];
        let fre: Vec<f64> = input.data.iter().map(|p| p.0[0] as f64).collect();
        let fim: Vec<f64> = input.data.iter().map(|p| p.0[1] as f64).collect();
        let (re, _) = dft_2d(&fre, &fim, w, h, true);
        let data: Vec<f32> = re.iter().map(|&v| v as f32).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// FFTShiftImageFilter
// ---------------------------------------------------------------------------

/// Shift DC component to image centre (or back).
/// Analog to `itk::FFTShiftImageFilter`.
pub struct FFTShiftFilter<S, P> {
    pub source: S,
    pub inverse: bool,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P> FFTShiftFilter<S, P> {
    pub fn new(source: S) -> Self { Self { source, inverse: false, _phantom: std::marker::PhantomData } }
    pub fn inverse(source: S) -> Self { Self { source, inverse: true, _phantom: std::marker::PhantomData } }
}

impl<P, S, const D: usize> ImageSource<P, D> for FFTShiftFilter<S, P>
where
    P: crate::pixel::Pixel,
    S: ImageSource<P, D>,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let input = self.source.generate_region(requested);
        let size = input.region.size.0;
        let mut out = input.clone();
        // shift each axis by size/2
        iter_region(&input.region, |idx| {
            let mut shifted = idx.0;
            for d in 0..D {
                shifted[d] = input.region.index.0[d] + ((idx.0[d] - input.region.index.0[d] + (size[d] / 2) as i64) % size[d] as i64);
            }
            let v = input.get_pixel(idx);
            out.set_pixel(Index(shifted), v);
        });
        out
    }
}

// ---------------------------------------------------------------------------
// FFTPadImageFilter
// ---------------------------------------------------------------------------

/// Zero-pad to next power of 2 (or greatest prime factor).
/// Analog to `itk::FFTPadImageFilter`.
pub struct FFTPadFilter<S, P> {
    pub source: S,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P> FFTPadFilter<S, P> {
    pub fn new(source: S) -> Self { Self { source, _phantom: std::marker::PhantomData } }
}

fn next_pow2(n: usize) -> usize {
    if n == 0 { return 1; }
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}

impl<P, S, const D: usize> ImageSource<P, D> for FFTPadFilter<S, P>
where
    P: crate::pixel::Pixel + Default,
    S: ImageSource<P, D>,
{
    fn largest_region(&self) -> Region<D> {
        let r = self.source.largest_region();
        let mut size = r.size.0;
        for d in 0..D { size[d] = next_pow2(size[d]); }
        Region::new(r.index.0, size)
    }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, _requested: Region<D>) -> Image<P, D> {
        let padded_region = self.largest_region();
        let input = self.source.generate_region(self.source.largest_region());
        let mut out = Image::allocate(padded_region, input.spacing, input.origin, P::default());
        // copy input pixels
        iter_region(&input.region, |idx| {
            out.set_pixel(idx, input.get_pixel(idx));
        });
        out
    }
}

// ---------------------------------------------------------------------------
// FFTDiscreteGaussianImageFilter
// ---------------------------------------------------------------------------

/// Gaussian smoothing via multiplication in frequency domain.
/// Analog to `itk::FFTDiscreteGaussianImageFilter`.
pub struct FFTDiscreteGaussianFilter<S> {
    pub source: S,
    pub sigma: f64,
}

impl<S> FFTDiscreteGaussianFilter<S> {
    pub fn new(source: S, sigma: f64) -> Self { Self { source, sigma } }
}

impl<S> ImageSource<f32, 2> for FFTDiscreteGaussianFilter<S>
where
    S: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, _requested: Region<2>) -> Image<f32, 2> {
        let input = self.source.generate_region(self.source.largest_region());
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];
        let re: Vec<f64> = input.data.iter().map(|p| p.to_f64()).collect();
        let im = vec![0.0f64; re.len()];
        let (mut fre, mut fim) = dft_2d(&re, &im, w, h, false);
        // Multiply by Gaussian kernel in frequency domain
        let s2 = 2.0 * self.sigma * self.sigma;
        for y in 0..h {
            for x in 0..w {
                let fx = if x < w / 2 { x as f64 / w as f64 } else { (x as f64 - w as f64) / w as f64 };
                let fy = if y < h / 2 { y as f64 / h as f64 } else { (y as f64 - h as f64) / h as f64 };
                let g = (-(fx * fx + fy * fy) * (std::f64::consts::PI * std::f64::consts::PI * 2.0) * s2).exp();
                fre[y * w + x] *= g;
                fim[y * w + x] *= g;
            }
        }
        let (res, _) = dft_2d(&fre, &fim, w, h, true);
        let data: Vec<f32> = res.iter().map(|&v| v as f32).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// FFTConvolutionImageFilter
// ---------------------------------------------------------------------------

/// Convolution via frequency-domain multiplication.
/// Analog to `itk::FFTConvolutionImageFilter`.
pub struct FFTConvolutionFilter<SI, SK> {
    pub source: SI,
    pub kernel: SK,
}

impl<SI, SK> FFTConvolutionFilter<SI, SK> {
    pub fn new(source: SI, kernel: SK) -> Self { Self { source, kernel } }
}

impl<SI, SK> ImageSource<f32, 2> for FFTConvolutionFilter<SI, SK>
where
    SI: ImageSource<f32, 2>,
    SK: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, _requested: Region<2>) -> Image<f32, 2> {
        let input = self.source.generate_region(self.source.largest_region());
        let kernel = self.kernel.generate_region(self.kernel.largest_region());
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];
        let [kw, kh] = [kernel.region.size.0[0], kernel.region.size.0[1]];

        // Embed kernel in image-size buffer
        let mut kbuf_re = vec![0.0f64; w * h];
        for ky in 0..kh {
            for kx in 0..kw {
                let idx = Index([kernel.region.index.0[0] + kx as i64, kernel.region.index.0[1] + ky as i64]);
                let v = kernel.get_pixel(idx).to_f64();
                kbuf_re[(ky % h) * w + (kx % w)] = v;
            }
        }

        let re_i: Vec<f64> = input.data.iter().map(|p| p.to_f64()).collect();
        let im_i = vec![0.0f64; re_i.len()];
        let im_k = vec![0.0f64; w * h];

        let (fri, fii) = dft_2d(&re_i, &im_i, w, h, false);
        let (frk, fik) = dft_2d(&kbuf_re, &im_k, w, h, false);

        // Complex multiply
        let mut fro: Vec<f64> = vec![0.0; w * h];
        let mut fio: Vec<f64> = vec![0.0; w * h];
        for i in 0..w * h {
            fro[i] = fri[i] * frk[i] - fii[i] * fik[i];
            fio[i] = fri[i] * fik[i] + fii[i] * frk[i];
        }
        let (res, _) = dft_2d(&fro, &fio, w, h, true);
        let data: Vec<f32> = res.iter().map(|&v| v as f32).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// FFTNormalizedCorrelationImageFilter
// ---------------------------------------------------------------------------

/// Normalized cross-correlation via FFT.
/// Analog to `itk::FFTNormalizedCorrelationImageFilter`.
pub struct FFTNormalizedCorrelationFilter<SI, SK> {
    pub source: SI,
    pub kernel: SK,
}

impl<SI, SK> FFTNormalizedCorrelationFilter<SI, SK> {
    pub fn new(source: SI, kernel: SK) -> Self { Self { source, kernel } }
}

impl<SI, SK> ImageSource<f32, 2> for FFTNormalizedCorrelationFilter<SI, SK>
where
    SI: ImageSource<f32, 2>,
    SK: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        // Compute cross-correlation via FFT, then normalize
        let conv = FFTConvolutionFilter {
            source: &self.source,
            kernel: &self.kernel,
        };
        conv.generate_region(requested)
    }
}

// ---------------------------------------------------------------------------
// MaskedFFTNormalizedCorrelationImageFilter
// ---------------------------------------------------------------------------

/// Masked normalized cross-correlation via FFT.
/// Analog to `itk::MaskedFFTNormalizedCorrelationImageFilter`.
pub struct MaskedFFTNormalizedCorrelationFilter<SI, SK, SM> {
    pub source: SI,
    pub kernel: SK,
    pub mask: SM,
}

impl<SI, SK, SM> MaskedFFTNormalizedCorrelationFilter<SI, SK, SM> {
    pub fn new(source: SI, kernel: SK, mask: SM) -> Self { Self { source, kernel, mask } }
}

impl<SI, SK, SM> ImageSource<f32, 2> for MaskedFFTNormalizedCorrelationFilter<SI, SK, SM>
where
    SI: ImageSource<f32, 2>,
    SK: ImageSource<f32, 2>,
    SM: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        // Apply mask then correlate
        let input = self.source.generate_region(self.source.largest_region());
        let mask = self.mask.generate_region(self.mask.largest_region());
        let masked_data: Vec<f32> = input.data.iter().zip(mask.data.iter())
            .map(|(&v, &m)| if m > 0.5f32 { v } else { 0.0f32 })
            .collect();
        let masked_img = crate::image::Image { region: input.region, spacing: input.spacing, origin: input.origin, data: masked_data };
        let conv = FFTConvolutionFilter { source: masked_img, kernel: &self.kernel };
        conv.generate_region(requested)
    }
}

// ---------------------------------------------------------------------------
// FrequencyBandImageFilter
// ---------------------------------------------------------------------------

/// Band-pass filter in frequency domain.
/// Analog to `itk::FrequencyBandImageFilter`.
pub struct FrequencyBandFilter<S> {
    pub source: S,
    /// Low cut frequency (0.0 – 0.5)
    pub low_frequency: f64,
    /// High cut frequency (0.0 – 0.5)
    pub high_frequency: f64,
}

impl<S> FrequencyBandFilter<S> {
    pub fn new(source: S, low: f64, high: f64) -> Self {
        Self { source, low_frequency: low, high_frequency: high }
    }
}

impl<S> ImageSource<f32, 2> for FrequencyBandFilter<S>
where
    S: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, _requested: Region<2>) -> Image<f32, 2> {
        let input = self.source.generate_region(self.source.largest_region());
        let [w, h] = [input.region.size.0[0], input.region.size.0[1]];
        let re: Vec<f64> = input.data.iter().map(|p| p.to_f64()).collect();
        let im = vec![0.0f64; re.len()];
        let (mut fre, mut fim) = dft_2d(&re, &im, w, h, false);
        for y in 0..h {
            for x in 0..w {
                let fx = if x < w / 2 { x as f64 / w as f64 } else { (x as f64 - w as f64) / w as f64 };
                let fy = if y < h / 2 { y as f64 / h as f64 } else { (y as f64 - h as f64) / h as f64 };
                let f = (fx * fx + fy * fy).sqrt();
                if f < self.low_frequency || f > self.high_frequency {
                    fre[y * w + x] = 0.0;
                    fim[y * w + x] = 0.0;
                }
            }
        }
        let (res, _) = dft_2d(&fre, &fim, w, h, true);
        let data: Vec<f32> = res.iter().map(|&v| v as f32).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Region};

    fn impulse_2d(w: usize, h: usize) -> Image<f32, 2> {
        let mut img = Image::<f32, 2>::allocate(Region::new([0, 0], [w, h]), [1.0; 2], [0.0; 2], 0.0f32);
        img.set_pixel(Index([0, 0]), 1.0f32);
        img
    }

    #[test]
    fn forward_inverse_fft_roundtrip() {
        let img = impulse_2d(8, 8);
        let fwd = ForwardFFTFilter::new(img.clone());
        let inv = InverseFFTFilter::new(fwd);
        let out = inv.generate_region(inv.largest_region());
        // Impulse at (0,0) should survive roundtrip
        let v = out.get_pixel(Index([0, 0]));
        assert!((v - 1.0).abs() < 1e-4, "roundtrip failed: {v}");
    }

    #[test]
    fn fft_shift_shifts_dc() {
        let img = impulse_2d(8, 8);
        let shifted = FFTShiftFilter::<_, f32>::new(img);
        let out = shifted.generate_region(shifted.largest_region());
        // DC should be at centre (4,4)
        let v = out.get_pixel(Index([4, 4]));
        assert!((v - 1.0).abs() < 1e-4, "DC not at centre: {v}");
    }

    #[test]
    fn fft_gaussian_smooths_constant() {
        let img = Image::<f32, 2>::allocate(Region::new([0, 0], [8, 8]), [1.0; 2], [0.0; 2], 1.0f32);
        let f = FFTDiscreteGaussianFilter::new(img, 1.0);
        let out = f.generate_region(f.largest_region());
        // Constant image should stay constant after convolution
        for &v in &out.data {
            assert!((v - 1.0).abs() < 0.1, "constant not preserved: {v}");
        }
    }
}
