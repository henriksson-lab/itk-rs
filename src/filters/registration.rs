//! Image registration metrics.
//!
//! | Metric | ITK analog |
//! |---|---|
//! | [`MeanSquaresMetric`]        | `MeanSquaresImageToImageMetricv4` |
//! | [`CorrelationMetric`]        | `CorrelationImageToImageMetricv4` |
//! | [`MattesMutualInformation`]  | `MattesMutualInformationImageToImageMetricv4` |
//! | [`ANTSCorrelationMetric`]    | `ANTSNeighborhoodCorrelationImageToImageMetricv4` |

use crate::image::{Image, Region, iter_region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

// ===========================================================================
// MeanSquaresMetric
// ===========================================================================

/// Mean squared difference metric.
/// Analog to `itk::MeanSquaresImageToImageMetricv4`.
///
/// Returns the mean of `(I_fixed(x) - I_moving(x))²` over the overlap region.
pub struct MeanSquaresMetric<SF, SM, P> {
    pub fixed: SF,
    pub moving: SM,
    _phantom: std::marker::PhantomData<P>,
}

impl<SF, SM, P> MeanSquaresMetric<SF, SM, P> {
    pub fn new(fixed: SF, moving: SM) -> Self {
        Self { fixed, moving, _phantom: std::marker::PhantomData }
    }

    pub fn compute<const D: usize>(&self) -> f64
    where
        P: NumericPixel,
        SF: ImageSource<P, D>,
        SM: ImageSource<P, D>,
    {
        let region = self.fixed.largest_region();
        let fixed_img = self.fixed.generate_region(region);
        let moving_img = self.moving.generate_region(region);
        let n = fixed_img.data.len() as f64;
        if n == 0.0 { return 0.0; }
        fixed_img.data.iter().zip(moving_img.data.iter())
            .map(|(f, m)| { let d = f.to_f64() - m.to_f64(); d * d })
            .sum::<f64>() / n
    }
}

// ===========================================================================
// CorrelationMetric
// ===========================================================================

/// Normalized correlation metric.
/// Analog to `itk::CorrelationImageToImageMetricv4`.
///
/// Returns `1 - NCC(fixed, moving)` (minimize → maximize NCC).
pub struct CorrelationMetric<SF, SM, P> {
    pub fixed: SF,
    pub moving: SM,
    _phantom: std::marker::PhantomData<P>,
}

impl<SF, SM, P> CorrelationMetric<SF, SM, P> {
    pub fn new(fixed: SF, moving: SM) -> Self {
        Self { fixed, moving, _phantom: std::marker::PhantomData }
    }

    pub fn compute<const D: usize>(&self) -> f64
    where
        P: NumericPixel,
        SF: ImageSource<P, D>,
        SM: ImageSource<P, D>,
    {
        let region = self.fixed.largest_region();
        let fixed_img = self.fixed.generate_region(region);
        let moving_img = self.moving.generate_region(region);
        let n = fixed_img.data.len() as f64;
        if n == 0.0 { return 0.0; }

        let f_mean = fixed_img.data.iter().map(|v| v.to_f64()).sum::<f64>() / n;
        let m_mean = moving_img.data.iter().map(|v| v.to_f64()).sum::<f64>() / n;
        let f_std = (fixed_img.data.iter().map(|v| { let d = v.to_f64() - f_mean; d*d }).sum::<f64>() / n).sqrt();
        let m_std = (moving_img.data.iter().map(|v| { let d = v.to_f64() - m_mean; d*d }).sum::<f64>() / n).sqrt();

        if f_std < 1e-12 || m_std < 1e-12 { return 1.0; }

        let ncc = fixed_img.data.iter().zip(moving_img.data.iter())
            .map(|(f, m)| (f.to_f64() - f_mean) * (m.to_f64() - m_mean))
            .sum::<f64>() / (n * f_std * m_std);

        1.0 - ncc.clamp(-1.0, 1.0)
    }
}

// ===========================================================================
// MattesMutualInformationMetric
// ===========================================================================

/// Mattes mutual information metric.
/// Analog to `itk::MattesMutualInformationImageToImageMetricv4`.
///
/// Estimates joint and marginal histograms with `num_bins` bins, then computes
/// `H(F) + H(M) - H(F,M)` (negative MI to minimize).
pub struct MattesMutualInformationMetric<SF, SM, P> {
    pub fixed: SF,
    pub moving: SM,
    pub num_bins: usize,
    _phantom: std::marker::PhantomData<P>,
}

impl<SF, SM, P> MattesMutualInformationMetric<SF, SM, P> {
    pub fn new(fixed: SF, moving: SM) -> Self {
        Self { fixed, moving, num_bins: 50, _phantom: std::marker::PhantomData }
    }

    /// Returns negative mutual information (minimizing = maximizing MI).
    pub fn compute<const D: usize>(&self) -> f64
    where
        P: NumericPixel,
        SF: ImageSource<P, D>,
        SM: ImageSource<P, D>,
    {
        let region = self.fixed.largest_region();
        let fixed_img = self.fixed.generate_region(region);
        let moving_img = self.moving.generate_region(region);
        let n = fixed_img.data.len();
        if n == 0 { return 0.0; }

        let nb = self.num_bins;
        let f_vals: Vec<f64> = fixed_img.data.iter().map(|v| v.to_f64()).collect();
        let m_vals: Vec<f64> = moving_img.data.iter().map(|v| v.to_f64()).collect();

        let f_min = f_vals.iter().cloned().fold(f64::MAX, f64::min);
        let f_max = f_vals.iter().cloned().fold(f64::MIN, f64::max);
        let m_min = m_vals.iter().cloned().fold(f64::MAX, f64::min);
        let m_max = m_vals.iter().cloned().fold(f64::MIN, f64::max);
        let f_range = (f_max - f_min).max(1e-12);
        let m_range = (m_max - m_min).max(1e-12);

        let mut joint = vec![0u64; nb * nb];
        let mut f_hist = vec![0u64; nb];
        let mut m_hist = vec![0u64; nb];

        for (&fv, &mv) in f_vals.iter().zip(m_vals.iter()) {
            let fi = ((fv - f_min) / f_range * (nb - 1) as f64).round() as usize;
            let mi = ((mv - m_min) / m_range * (nb - 1) as f64).round() as usize;
            let fi = fi.min(nb - 1);
            let mi = mi.min(nb - 1);
            joint[fi * nb + mi] += 1;
            f_hist[fi] += 1;
            m_hist[mi] += 1;
        }

        let n_f = n as f64;
        let entropy = |h: &[u64]| -> f64 {
            h.iter().map(|&c| if c > 0 { let p = c as f64 / n_f; -p * p.ln() } else { 0.0 }).sum()
        };
        let h_f = entropy(&f_hist);
        let h_m = entropy(&m_hist);
        let h_joint = entropy(&joint);

        -(h_f + h_m - h_joint) // negative MI
    }
}

// ===========================================================================
// ANTSCorrelationMetric (Neighborhood Correlation)
// ===========================================================================

/// ANTS neighborhood cross-correlation metric.
/// Analog to `itk::ANTSNeighborhoodCorrelationImageToImageMetricv4`.
///
/// Computes mean local NCC over windows of size `(2*radius+1)^D`.
pub struct ANTSCorrelationMetric<SF, SM, P, const D: usize> {
    pub fixed: SF,
    pub moving: SM,
    pub radius: usize,
    _phantom: std::marker::PhantomData<P>,
}

impl<SF, SM, P, const D: usize> ANTSCorrelationMetric<SF, SM, P, D> {
    pub fn new(fixed: SF, moving: SM, radius: usize) -> Self {
        Self { fixed, moving, radius, _phantom: std::marker::PhantomData }
    }

    /// Returns negative mean local NCC (1 - mean_NCC).
    pub fn compute(&self) -> f64
    where
        P: NumericPixel,
        SF: ImageSource<P, D>,
        SM: ImageSource<P, D>,
    {
        let region = self.fixed.largest_region();
        let fixed_img = self.fixed.generate_region(region);
        let moving_img = self.moving.generate_region(region);
        let bounds = fixed_img.region;
        let r = self.radius as i64;
        let n_nb = (2 * self.radius + 1).pow(D as u32) as f64;

        let mut total_ncc = 0.0f64;
        let mut count = 0usize;

        iter_region(&bounds, |idx| {
            let mut sf = 0.0f64; let mut sm = 0.0f64;
            let mut sf2 = 0.0f64; let mut sm2 = 0.0f64; let mut sfm = 0.0f64;

            let mut nb = [0i64; D];
            for d in 0..D { nb[d] = -r; }
            loop {
                let mut s = [0i64; D];
                for d in 0..D {
                    s[d] = (idx.0[d] + nb[d]).max(bounds.index.0[d])
                        .min(bounds.index.0[d] + bounds.size.0[d] as i64 - 1);
                }
                let fv = fixed_img.get_pixel(crate::image::Index(s)).to_f64();
                let mv = moving_img.get_pixel(crate::image::Index(s)).to_f64();
                sf += fv; sm += mv; sf2 += fv*fv; sm2 += mv*mv; sfm += fv*mv;
                let mut carry = true;
                for d in 0..D {
                    if carry { nb[d] += 1; if nb[d] > r { nb[d] = -r; } else { carry = false; } }
                }
                if carry { break; }
            }
            let f_mean = sf / n_nb;
            let m_mean = sm / n_nb;
            let f_var = (sf2 / n_nb - f_mean * f_mean).max(0.0);
            let m_var = (sm2 / n_nb - m_mean * m_mean).max(0.0);
            let f_std = f_var.sqrt();
            let m_std = m_var.sqrt();
            if f_std < 1e-12 || m_std < 1e-12 { return; }
            let ncc = (sfm / n_nb - f_mean * m_mean) / (f_std * m_std);
            total_ncc += ncc.clamp(-1.0, 1.0);
            count += 1;
        });

        if count == 0 { return 0.0; }
        1.0 - total_ncc / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Region};

    #[test]
    fn mean_squares_identical() {
        let img = Image::<f32,1>::allocate(Region::new([0],[10]),[1.0],[0.0],3.0f32);
        let m = MeanSquaresMetric::<_,_,f32>::new(img.clone(), img);
        assert!(m.compute::<1>() < 1e-10);
    }

    #[test]
    fn correlation_identical_is_zero() {
        let img = Image::<f32,1>::allocate(Region::new([0],[10]),[1.0],[0.0],0.0f32);
        // Ramp to avoid zero-std
        let mut img2 = img.clone();
        for i in 0..10i64 { img2.set_pixel(crate::image::Index([i]), i as f32); }
        let m = CorrelationMetric::<_,_,f32>::new(img2.clone(), img2);
        let v = m.compute::<1>();
        assert!(v.abs() < 1e-6, "expected 0 got {v}");
    }
}
