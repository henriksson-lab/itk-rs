//! Diffusion tensor imaging (DTI) filters.
//!
//! | Filter | ITK analog |
//! |---|---|
//! | [`DiffusionTensor3DReconstructionFilter`] | `DiffusionTensor3DReconstructionImageFilter` |
//! | [`FractionalAnisotropyFilter`] | `TensorFractionalAnisotropyImageFilter` |
//! | [`RelativeAnisotropyFilter`] | `TensorRelativeAnisotropyImageFilter` |

use crate::image::{Image, Region, iter_region};
use crate::pixel::VecPixel;
use crate::source::ImageSource;

// ===========================================================================
// DiffusionTensor3DReconstructionImageFilter
// ===========================================================================

/// Reconstruct a diffusion tensor from DWI images.
/// Input: N+1 DWI volumes (one b=0 baseline + N diffusion weighted).
/// Output: symmetric tensor stored as `VecPixel<f32, 6>` = [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz].
/// Analog to `itk::DiffusionTensor3DReconstructionImageFilter`.
pub struct DiffusionTensor3DReconstructionFilter {
    /// b=0 reference image
    pub baseline: Image<f32, 2>,
    /// Diffusion-weighted images, one per gradient direction
    pub dwi: Vec<Image<f32, 2>>,
    /// Gradient directions (unit vectors), one per DWI
    pub gradients: Vec<[f64; 3]>,
    pub b_value: f64,
}

impl DiffusionTensor3DReconstructionFilter {
    pub fn new(baseline: Image<f32, 2>, dwi: Vec<Image<f32, 2>>, gradients: Vec<[f64; 3]>, b_value: f64) -> Self {
        Self { baseline, dwi, gradients, b_value }
    }
}

impl DiffusionTensor3DReconstructionFilter {
    /// Reconstruct the diffusion tensor image.
    /// Uses least-squares fit: log(S/S0) = -b * g^T * D * g
    pub fn compute(&self) -> Image<VecPixel<f32, 6>, 2> {
        let region = self.baseline.region;
        let n_grad = self.dwi.len().min(self.gradients.len());

        // Build design matrix B: each row is b * [gx², gy², gz², 2gxgy, 2gxgz, 2gygz]
        let design: Vec<[f64; 6]> = (0..n_grad).map(|i| {
            let g = self.gradients[i];
            let b = self.b_value;
            [
                b * g[0] * g[0],
                b * g[1] * g[1],
                b * g[2] * g[2],
                b * 2.0 * g[0] * g[1],
                b * 2.0 * g[0] * g[2],
                b * 2.0 * g[1] * g[2],
            ]
        }).collect();

        let mut tensor_data: Vec<VecPixel<f32, 6>> = Vec::with_capacity(region.linear_len());
        iter_region(&region, |idx| {
            let s0 = self.baseline.get_pixel(idx).max(1e-6) as f64;
            // Measurements vector: log(Si/S0)
            let b_vec: Vec<f64> = (0..n_grad).map(|i| {
                let si = self.dwi[i].get_pixel(idx).max(1e-6) as f64;
                -(si / s0).ln()
            }).collect();

            // Least squares: (B^T B)^{-1} B^T b_vec  (simplified: 6 unknowns)
            // Use simple pseudo-inverse via normal equations
            let tensor = if n_grad >= 6 {
                lstsq_6x6(&design, &b_vec)
            } else {
                [0.0f64; 6]
            };

            tensor_data.push(VecPixel([
                tensor[0] as f32, tensor[1] as f32, tensor[2] as f32,
                tensor[3] as f32, tensor[4] as f32, tensor[5] as f32,
            ]));
        });

        Image { region, spacing: self.baseline.spacing, origin: self.baseline.origin, data: tensor_data }
    }
}

/// Minimal least-squares solver for 6-component tensor (Ax = b, A is n×6).
fn lstsq_6x6(a: &[[f64; 6]], b: &[f64]) -> [f64; 6] {
    let n = a.len().min(b.len());
    // Compute A^T A (6×6) and A^T b (6×1)
    let mut ata = [[0.0f64; 6]; 6];
    let mut atb = [0.0f64; 6];
    for i in 0..n {
        for j in 0..6 {
            atb[j] += a[i][j] * b[i];
            for k in 0..6 {
                ata[j][k] += a[i][j] * a[i][k];
            }
        }
    }
    // Solve via Gaussian elimination with partial pivoting
    gaussian_elim_6x6(ata, atb)
}

fn gaussian_elim_6x6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> [f64; 6] {
    let n = 6;
    for col in 0..n {
        // Partial pivot
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap()).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        let p = a[col][col];
        if p.abs() < 1e-12 { continue; }
        for row in (col + 1)..n {
            let factor = a[row][col] / p;
            for c in col..n { a[row][c] -= factor * a[col][c]; }
            b[row] -= factor * b[col];
        }
    }
    // Back substitution
    let mut x = [0.0f64; 6];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n { s -= a[i][j] * x[j]; }
        x[i] = if a[i][i].abs() > 1e-12 { s / a[i][i] } else { 0.0 };
    }
    x
}

// ===========================================================================
// TensorFractionalAnisotropyImageFilter
// ===========================================================================

/// Compute fractional anisotropy (FA) from diffusion tensor.
/// FA = sqrt(3/2) · ||D - D̄I|| / ||D||
/// Analog to `itk::TensorFractionalAnisotropyImageFilter`.
pub struct FractionalAnisotropyFilter<S> {
    pub source: S,
}

impl<S> FractionalAnisotropyFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<S> ImageSource<f32, 2> for FractionalAnisotropyFilter<S>
where
    S: ImageSource<VecPixel<f32, 6>, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        let input = self.source.generate_region(requested);
        let data: Vec<f32> = input.data.iter().map(|t| {
            let [dxx, dyy, dzz, dxy, dxz, dyz] = t.0;
            let dxx = dxx as f64; let dyy = dyy as f64; let dzz = dzz as f64;
            let dxy = dxy as f64; let dxz = dxz as f64; let dyz = dyz as f64;
            let trace = dxx + dyy + dzz;
            let mean = trace / 3.0;
            // Frobenius norm of deviatoric part
            let dev_xx = dxx - mean; let dev_yy = dyy - mean; let dev_zz = dzz - mean;
            let dev_norm2 = dev_xx * dev_xx + dev_yy * dev_yy + dev_zz * dev_zz
                          + 2.0 * (dxy * dxy + dxz * dxz + dyz * dyz);
            let full_norm2 = dxx * dxx + dyy * dyy + dzz * dzz
                           + 2.0 * (dxy * dxy + dxz * dxz + dyz * dyz);
            let fa = if full_norm2 > 0.0 {
                ((3.0 / 2.0) * dev_norm2 / full_norm2).sqrt()
            } else { 0.0 };
            fa.clamp(0.0, 1.0) as f32
        }).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}

// ===========================================================================
// TensorRelativeAnisotropyImageFilter
// ===========================================================================

/// Compute relative anisotropy (RA) from diffusion tensor.
/// RA = ||D - D̄I|| / (√3 · D̄)
/// Analog to `itk::TensorRelativeAnisotropyImageFilter`.
pub struct RelativeAnisotropyFilter<S> {
    pub source: S,
}

impl<S> RelativeAnisotropyFilter<S> {
    pub fn new(source: S) -> Self { Self { source } }
}

impl<S> ImageSource<f32, 2> for RelativeAnisotropyFilter<S>
where
    S: ImageSource<VecPixel<f32, 6>, 2>,
{
    fn largest_region(&self) -> Region<2> { self.source.largest_region() }
    fn spacing(&self) -> [f64; 2] { self.source.spacing() }
    fn origin(&self) -> [f64; 2] { self.source.origin() }

    fn generate_region(&self, requested: Region<2>) -> Image<f32, 2> {
        let input = self.source.generate_region(requested);
        let data: Vec<f32> = input.data.iter().map(|t| {
            let [dxx, dyy, dzz, dxy, dxz, dyz] = t.0;
            let dxx = dxx as f64; let dyy = dyy as f64; let dzz = dzz as f64;
            let dxy = dxy as f64; let dxz = dxz as f64; let dyz = dyz as f64;
            let trace = dxx + dyy + dzz;
            let mean = trace / 3.0;
            let dev_xx = dxx - mean; let dev_yy = dyy - mean; let dev_zz = dzz - mean;
            let dev_norm2 = dev_xx * dev_xx + dev_yy * dev_yy + dev_zz * dev_zz
                          + 2.0 * (dxy * dxy + dxz * dxz + dyz * dyz);
            let ra = if mean.abs() > 1e-12 {
                dev_norm2.sqrt() / (3.0f64.sqrt() * mean.abs())
            } else { 0.0 };
            ra.clamp(0.0, f64::MAX) as f32
        }).collect();
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

    fn isotropic_tensor(d: f32) -> Image<VecPixel<f32, 6>, 2> {
        let region = Region::new([0, 0], [4, 4]);
        Image::allocate(region, [1.0; 2], [0.0; 2], VecPixel([d, d, d, 0.0, 0.0, 0.0]))
    }

    #[test]
    fn fa_isotropic_is_zero() {
        let tensor = isotropic_tensor(1e-3);
        let fa = FractionalAnisotropyFilter::new(tensor);
        let out = fa.generate_region(fa.largest_region());
        for &v in &out.data {
            assert!(v < 0.01, "FA of isotropic tensor should be ~0, got {v}");
        }
    }

    #[test]
    fn ra_isotropic_is_zero() {
        let tensor = isotropic_tensor(1e-3);
        let ra = RelativeAnisotropyFilter::new(tensor);
        let out = ra.generate_region(ra.largest_region());
        for &v in &out.data {
            assert!(v < 0.01, "RA of isotropic tensor should be ~0, got {v}");
        }
    }
}
