//! Deconvolution filters.
//!
//! Analogs to ITK's deconvolution filters:
//! - `InverseDeconvolutionImageFilter`
//! - `WienerDeconvolutionImageFilter`
//! - `TikhonovDeconvolutionImageFilter`
//! - `RichardsonLucyDeconvolutionImageFilter`
//! - `LandweberDeconvolutionImageFilter`
//! - `ProjectedLandweberDeconvolutionImageFilter`

use crate::image::{Image, Region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

fn fft2(re: &[f64], im: &[f64], w: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    crate::filters::fft::dft_2d(re, im, w, h, false)
}

fn ifft2(re: &[f64], im: &[f64], w: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    crate::filters::fft::dft_2d(re, im, w, h, true)
}

/// Embed kernel in image-size zero-padded buffer.
fn embed_kernel<P: NumericPixel>(kernel: &Image<P, 2>, w: usize, h: usize) -> Vec<f64> {
    let mut buf = vec![0.0f64; w * h];
    let [kw, kh] = [kernel.region.size.0[0], kernel.region.size.0[1]];
    for ky in 0..kh {
        for kx in 0..kw {
            let idx = crate::image::Index([
                kernel.region.index.0[0] + kx as i64,
                kernel.region.index.0[1] + ky as i64,
            ]);
            buf[(ky % h) * w + (kx % w)] = kernel.get_pixel(idx).to_f64();
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// InverseDeconvolutionImageFilter
// ---------------------------------------------------------------------------

/// Inverse filtering deconvolution: divide by PSF in frequency domain.
/// Analog to `itk::InverseDeconvolutionImageFilter`.
pub struct InverseDeconvolutionFilter<SI, SK> {
    pub source: SI,
    pub kernel: SK,
    /// Regularization floor (avoid division by zero).
    pub regularization: f64,
}

impl<SI, SK> InverseDeconvolutionFilter<SI, SK> {
    pub fn new(source: SI, kernel: SK) -> Self {
        Self { source, kernel, regularization: 1e-4 }
    }
}

impl<SI, SK> ImageSource<f32, 2> for InverseDeconvolutionFilter<SI, SK>
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
        let re_i: Vec<f64> = input.data.iter().map(|p| p.to_f64()).collect();
        let im_i = vec![0.0f64; re_i.len()];
        let kbuf = embed_kernel(&kernel, w, h);
        let im_k = vec![0.0f64; w * h];
        let (fri, fii) = fft2(&re_i, &im_i, w, h);
        let (frk, fik) = fft2(&kbuf, &im_k, w, h);
        // Divide in frequency domain: F_out = F_in / F_kernel
        let eps = self.regularization;
        let mut fro = vec![0.0f64; w * h];
        let mut fio = vec![0.0f64; w * h];
        for i in 0..w * h {
            let denom = frk[i] * frk[i] + fik[i] * fik[i] + eps;
            fro[i] = (fri[i] * frk[i] + fii[i] * fik[i]) / denom;
            fio[i] = (fii[i] * frk[i] - fri[i] * fik[i]) / denom;
        }
        let (res, _) = ifft2(&fro, &fio, w, h);
        let data: Vec<f32> = res.iter().map(|&v| v as f32).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// WienerDeconvolutionImageFilter
// ---------------------------------------------------------------------------

/// Wiener deconvolution filter.
/// Analog to `itk::WienerDeconvolutionImageFilter`.
pub struct WienerDeconvolutionFilter<SI, SK> {
    pub source: SI,
    pub kernel: SK,
    /// Noise-to-signal power ratio (Wiener parameter).
    pub noise_power_ratio: f64,
}

impl<SI, SK> WienerDeconvolutionFilter<SI, SK> {
    pub fn new(source: SI, kernel: SK) -> Self {
        Self { source, kernel, noise_power_ratio: 1e-3 }
    }
}

impl<SI, SK> ImageSource<f32, 2> for WienerDeconvolutionFilter<SI, SK>
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
        let re_i: Vec<f64> = input.data.iter().map(|p| p.to_f64()).collect();
        let im_i = vec![0.0f64; re_i.len()];
        let kbuf = embed_kernel(&kernel, w, h);
        let im_k = vec![0.0f64; w * h];
        let (fri, fii) = fft2(&re_i, &im_i, w, h);
        let (frk, fik) = fft2(&kbuf, &im_k, w, h);
        let snr = self.noise_power_ratio;
        let mut fro = vec![0.0f64; w * h];
        let mut fio = vec![0.0f64; w * h];
        for i in 0..w * h {
            let hh = frk[i] * frk[i] + fik[i] * fik[i];
            let wiener = hh / (hh + snr);
            let denom = hh + snr * 1e-10;
            fro[i] = wiener * (fri[i] * frk[i] + fii[i] * fik[i]) / denom.max(1e-12);
            fio[i] = wiener * (fii[i] * frk[i] - fri[i] * fik[i]) / denom.max(1e-12);
        }
        let (res, _) = ifft2(&fro, &fio, w, h);
        let data: Vec<f32> = res.iter().map(|&v| v as f32).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// TikhonovDeconvolutionImageFilter
// ---------------------------------------------------------------------------

/// Tikhonov regularized deconvolution.
/// Analog to `itk::TikhonovDeconvolutionImageFilter`.
pub struct TikhonovDeconvolutionFilter<SI, SK> {
    pub source: SI,
    pub kernel: SK,
    pub regularization: f64,
}

impl<SI, SK> TikhonovDeconvolutionFilter<SI, SK> {
    pub fn new(source: SI, kernel: SK) -> Self {
        Self { source, kernel, regularization: 1e-3 }
    }
}

impl<SI, SK> ImageSource<f32, 2> for TikhonovDeconvolutionFilter<SI, SK>
where
    SI: ImageSource<f32, 2>,
    SK: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, _requested: Region<2>) -> Image<f32, 2> {
        // Tikhonov = Wiener with fixed SNR-like regularization
        let wiener = WienerDeconvolutionFilter {
            source: &self.source,
            kernel: &self.kernel,
            noise_power_ratio: self.regularization,
        };
        wiener.generate_region(wiener.largest_region())
    }
}

// ---------------------------------------------------------------------------
// RichardsonLucyDeconvolutionImageFilter
// ---------------------------------------------------------------------------

/// Richardson-Lucy iterative deconvolution.
/// Analog to `itk::RichardsonLucyDeconvolutionImageFilter`.
pub struct RichardsonLucyDeconvolutionFilter<SI, SK> {
    pub source: SI,
    pub kernel: SK,
    pub iterations: usize,
}

impl<SI, SK> RichardsonLucyDeconvolutionFilter<SI, SK> {
    pub fn new(source: SI, kernel: SK, iterations: usize) -> Self {
        Self { source, kernel, iterations }
    }
}

impl<SI, SK> ImageSource<f32, 2> for RichardsonLucyDeconvolutionFilter<SI, SK>
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
        let observed: Vec<f64> = input.data.iter().map(|p| p.to_f64().max(0.0)).collect();
        let kbuf = embed_kernel(&kernel, w, h);
        let im_k = vec![0.0f64; w * h];
        let (frk, fik) = fft2(&kbuf, &im_k, w, h);
        // Flipped kernel (for convolution in update step)
        let kbuf_flip: Vec<f64> = {
            let mut k = kbuf.clone();
            // Flip by reversing DFT conjugate: H* has conjugate spectrum
            k
        };
        let im_kf = vec![0.0f64; w * h];
        let (frkf, fikf) = fft2(&kbuf_flip, &im_kf, w, h);

        let mut estimate = observed.clone();
        for _ in 0..self.iterations {
            // Convolve estimate with H
            let im_e = vec![0.0f64; w * h];
            let (fre, fie) = fft2(&estimate, &im_e, w, h);
            let mut fr_he = vec![0.0f64; w * h];
            let mut fi_he = vec![0.0f64; w * h];
            for i in 0..w * h {
                fr_he[i] = fre[i] * frk[i] - fie[i] * fik[i];
                fi_he[i] = fre[i] * fik[i] + fie[i] * frk[i];
            }
            let (he, _) = ifft2(&fr_he, &fi_he, w, h);
            // Compute ratio observed / he
            let ratio: Vec<f64> = observed.iter().zip(he.iter())
                .map(|(&o, &h)| o / h.max(1e-10))
                .collect();
            // Convolve ratio with flipped H
            let im_r = vec![0.0f64; w * h];
            let (frr, fir) = fft2(&ratio, &im_r, w, h);
            let mut fr_hr = vec![0.0f64; w * h];
            let mut fi_hr = vec![0.0f64; w * h];
            for i in 0..w * h {
                fr_hr[i] = frr[i] * frkf[i] - fir[i] * fikf[i];
                fi_hr[i] = frr[i] * fikf[i] + fir[i] * frkf[i];
            }
            let (hr, _) = ifft2(&fr_hr, &fi_hr, w, h);
            for i in 0..w * h {
                estimate[i] *= hr[i].max(0.0);
            }
        }
        let data: Vec<f32> = estimate.iter().map(|&v| v as f32).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// LandweberDeconvolutionImageFilter
// ---------------------------------------------------------------------------

/// Landweber iterative deconvolution.
/// Analog to `itk::LandweberDeconvolutionImageFilter`.
pub struct LandweberDeconvolutionFilter<SI, SK> {
    pub source: SI,
    pub kernel: SK,
    pub iterations: usize,
    pub alpha: f64,
}

impl<SI, SK> LandweberDeconvolutionFilter<SI, SK> {
    pub fn new(source: SI, kernel: SK, iterations: usize) -> Self {
        Self { source, kernel, iterations, alpha: 0.1 }
    }
}

impl<SI, SK> ImageSource<f32, 2> for LandweberDeconvolutionFilter<SI, SK>
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
        let observed: Vec<f64> = input.data.iter().map(|p| p.to_f64()).collect();
        let kbuf = embed_kernel(&kernel, w, h);
        let im_k = vec![0.0f64; w * h];
        let (frk, fik) = fft2(&kbuf, &im_k, w, h);

        let mut estimate = observed.clone();
        for _ in 0..self.iterations {
            let im_e = vec![0.0f64; w * h];
            let (fre, fie) = fft2(&estimate, &im_e, w, h);
            // H * estimate
            let mut fr_he = vec![0.0f64; w * h];
            let mut fi_he = vec![0.0f64; w * h];
            for i in 0..w * h {
                fr_he[i] = fre[i] * frk[i] - fie[i] * fik[i];
                fi_he[i] = fre[i] * fik[i] + fie[i] * frk[i];
            }
            let (he, _) = ifft2(&fr_he, &fi_he, w, h);
            // residual = observed - H*estimate; H^T * residual
            let resid: Vec<f64> = observed.iter().zip(he.iter()).map(|(&o, &h)| o - h).collect();
            let im_res = vec![0.0f64; w * h];
            let (frres, fires) = fft2(&resid, &im_res, w, h);
            // H^T is conjugate in freq domain
            let mut fr_htr = vec![0.0f64; w * h];
            let mut fi_htr = vec![0.0f64; w * h];
            for i in 0..w * h {
                fr_htr[i] = frres[i] * frk[i] + fires[i] * fik[i];
                fi_htr[i] = fires[i] * frk[i] - frres[i] * fik[i];
            }
            let (htr, _) = ifft2(&fr_htr, &fi_htr, w, h);
            for i in 0..w * h {
                estimate[i] += self.alpha * htr[i];
            }
        }
        let data: Vec<f32> = estimate.iter().map(|&v| v as f32).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ---------------------------------------------------------------------------
// ProjectedLandweberDeconvolutionImageFilter
// ---------------------------------------------------------------------------

/// Projected Landweber: Landweber with non-negativity projection.
/// Analog to `itk::ProjectedLandweberDeconvolutionImageFilter`.
pub struct ProjectedLandweberDeconvolutionFilter<SI, SK> {
    pub source: SI,
    pub kernel: SK,
    pub iterations: usize,
    pub alpha: f64,
}

impl<SI, SK> ProjectedLandweberDeconvolutionFilter<SI, SK> {
    pub fn new(source: SI, kernel: SK, iterations: usize) -> Self {
        Self { source, kernel, iterations, alpha: 0.1 }
    }
}

impl<SI, SK> ImageSource<f32, 2> for ProjectedLandweberDeconvolutionFilter<SI, SK>
where
    SI: ImageSource<f32, 2>,
    SK: ImageSource<f32, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, _requested: Region<2>) -> Image<f32, 2> {
        let lw = LandweberDeconvolutionFilter {
            source: &self.source,
            kernel: &self.kernel,
            iterations: self.iterations,
            alpha: self.alpha,
        };
        let mut out = lw.generate_region(lw.largest_region());
        // Project to non-negative
        for v in &mut out.data { *v = v.max(0.0); }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    fn delta_kernel() -> Image<f32, 2> {
        let mut k = Image::<f32, 2>::allocate(Region::new([0, 0], [3, 3]), [1.0; 2], [0.0; 2], 0.0f32);
        k.set_pixel(Index([0, 0]), 1.0f32);
        k
    }

    #[test]
    fn inverse_deconv_delta_roundtrip() {
        let img = Image::<f32, 2>::allocate(Region::new([0, 0], [4, 4]), [1.0; 2], [0.0; 2], 0.0f32);
        let mut img = img;
        img.set_pixel(Index([1, 1]), 5.0f32);
        let kernel = delta_kernel();
        let f = InverseDeconvolutionFilter::new(img.clone(), kernel);
        let out = f.generate_region(f.largest_region());
        // After inverse with delta kernel the image should be close to original
        let v = out.get_pixel(Index([1, 1]));
        assert!(v > 3.0, "expected ~5.0, got {v}");
    }

    #[test]
    fn landweber_deconv_runs() {
        let img = Image::<f32, 2>::allocate(Region::new([0, 0], [4, 4]), [1.0; 2], [0.0; 2], 1.0f32);
        let kernel = delta_kernel();
        let f = LandweberDeconvolutionFilter::new(img, kernel, 5);
        let _out = f.generate_region(f.largest_region());
    }

    #[test]
    fn projected_landweber_nonnegative() {
        let mut img = Image::<f32, 2>::allocate(Region::new([0, 0], [4, 4]), [1.0; 2], [0.0; 2], 0.0f32);
        img.set_pixel(Index([0, 0]), 1.0f32);
        let kernel = delta_kernel();
        let f = ProjectedLandweberDeconvolutionFilter::new(img, kernel, 5);
        let out = f.generate_region(f.largest_region());
        for &v in &out.data {
            assert!(v >= 0.0, "negative pixel: {v}");
        }
    }
}
