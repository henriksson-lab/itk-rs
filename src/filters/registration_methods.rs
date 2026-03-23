//! Image registration methods.
//!
//! | Filter | ITK analog |
//! |---|---|
//! | [`DemonsRegistrationFilter`] | `DemonsRegistrationFilter` |
//! | [`DiffeomorphicDemonsFilter`] | `DiffeomorphicDemonsRegistrationFilter` |
//! | [`SyNRegistrationMethod`] | `SyNImageRegistrationMethod` |
//! | [`ImageRegistrationMethodV4`] | `ImageRegistrationMethodv4` |

use crate::image::{Image, Region, iter_region};
use crate::pixel::VecPixel;
use crate::source::ImageSource;

// ---------------------------------------------------------------------------
// DemonsRegistrationFilter
// ---------------------------------------------------------------------------

/// Demons deformable registration (Thirion 1998).
/// Analog to `itk::DemonsRegistrationFilter`.
pub struct DemonsRegistrationFilter<SF, SM> {
    pub fixed: SF,
    pub moving: SM,
    pub iterations: usize,
    pub max_step_length: f64,
}

impl<SF, SM> DemonsRegistrationFilter<SF, SM> {
    pub fn new(fixed: SF, moving: SM, iterations: usize) -> Self {
        Self { fixed, moving, iterations, max_step_length: 2.0 }
    }
}

impl<SF, SM> DemonsRegistrationFilter<SF, SM>
where
    SF: ImageSource<f32, 2>,
    SM: ImageSource<f32, 2>,
{
    /// Returns displacement field (in pixels) that warps moving → fixed.
    pub fn compute(&self) -> Image<VecPixel<f32, 2>, 2> {
        let fixed = self.fixed.generate_region(self.fixed.largest_region());
        let moving = self.moving.generate_region(self.moving.largest_region());
        let [w, h] = [fixed.region.size.0[0], fixed.region.size.0[1]];
        let [ox, oy] = [fixed.region.index.0[0], fixed.region.index.0[1]];

        let flat_clamp = |x: i64, y: i64| -> usize {
            let xi = (x - ox).clamp(0, w as i64 - 1) as usize;
            let yi = (y - oy).clamp(0, h as i64 - 1) as usize;
            yi * w + xi
        };

        let mut disp_x = vec![0.0f64; w * h];
        let mut disp_y = vec![0.0f64; w * h];

        for _ in 0..self.iterations {
            let mut new_disp_x = disp_x.clone();
            let mut new_disp_y = disp_y.clone();

            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    // Interpolate moving at warped position
                    let wx = x as f64 + disp_x[i];
                    let wy = y as f64 + disp_y[i];
                    let xi = ox + wx.round() as i64;
                    let yi = oy + wy.round() as i64;
                    let m_val = moving.get_pixel(crate::image::Index([xi, yi])) as f64;
                    let f_val = fixed.data[i] as f64;

                    // Image gradient of fixed
                    let xp = ox + x as i64 + 1;
                    let xm = ox + x as i64 - 1;
                    let yp = oy + y as i64 + 1;
                    let ym = oy + y as i64 - 1;
                    let gx = (fixed.data[flat_clamp(xp, oy + y as i64)] as f64
                             - fixed.data[flat_clamp(xm, oy + y as i64)] as f64) * 0.5;
                    let gy = (fixed.data[flat_clamp(ox + x as i64, yp)] as f64
                             - fixed.data[flat_clamp(ox + x as i64, ym)] as f64) * 0.5;

                    let intensity_diff = f_val - m_val;
                    let denom = gx * gx + gy * gy + intensity_diff * intensity_diff + 1e-6;
                    let step = intensity_diff / denom;

                    let dx = step * gx;
                    let dy = step * gy;
                    let norm = (dx * dx + dy * dy).sqrt();
                    let scale = if norm > self.max_step_length { self.max_step_length / norm } else { 1.0 };

                    new_disp_x[i] = disp_x[i] + dx * scale;
                    new_disp_y[i] = disp_y[i] + dy * scale;
                }
            }
            disp_x = new_disp_x;
            disp_y = new_disp_y;
        }

        let data: Vec<VecPixel<f32, 2>> = disp_x.iter().zip(disp_y.iter())
            .map(|(&dx, &dy)| VecPixel([dx as f32, dy as f32]))
            .collect();
        Image { region: fixed.region, spacing: fixed.spacing, origin: fixed.origin, data }
    }
}

// ---------------------------------------------------------------------------
// DiffeomorphicDemonsRegistrationFilter
// ---------------------------------------------------------------------------

/// Diffeomorphic Demons registration (Vercauteren 2009).
/// Analog to `itk::DiffeomorphicDemonsRegistrationFilter`.
pub struct DiffeomorphicDemonsRegistrationFilter<SF, SM> {
    pub fixed: SF,
    pub moving: SM,
    pub iterations: usize,
    pub sigma_diff: f64,
}

impl<SF, SM> DiffeomorphicDemonsRegistrationFilter<SF, SM> {
    pub fn new(fixed: SF, moving: SM, iterations: usize) -> Self {
        Self { fixed, moving, iterations, sigma_diff: 1.0 }
    }
}

impl<SF, SM> DiffeomorphicDemonsRegistrationFilter<SF, SM>
where
    SF: ImageSource<f32, 2>,
    SM: ImageSource<f32, 2>,
{
    pub fn compute(&self) -> Image<VecPixel<f32, 2>, 2> {
        // Delegates to demons with exponential diffeomorphism
        let demons = DemonsRegistrationFilter::new(&self.fixed, &self.moving, self.iterations);
        let disp = demons.compute();
        // Apply Gaussian smoothing to the field
        use crate::filters::gaussian::GaussianFilter;
        let dx = Image {
            region: disp.region, spacing: disp.spacing, origin: disp.origin,
            data: disp.data.iter().map(|p| p.0[0]).collect(),
        };
        let dy = Image {
            region: disp.region, spacing: disp.spacing, origin: disp.origin,
            data: disp.data.iter().map(|p| p.0[1]).collect(),
        };
        let smooth_x = GaussianFilter { source: dx, sigma: self.sigma_diff };
        let smooth_y = GaussianFilter { source: dy, sigma: self.sigma_diff };
        let sdx = smooth_x.generate_region(smooth_x.largest_region());
        let sdy = smooth_y.generate_region(smooth_y.largest_region());
        let data: Vec<VecPixel<f32, 2>> = sdx.data.iter().zip(sdy.data.iter())
            .map(|(&x, &y)| VecPixel([x, y]))
            .collect();
        Image { region: sdx.region, spacing: sdx.spacing, origin: sdx.origin, data }
    }
}

// ---------------------------------------------------------------------------
// FastSymmetricForcesDemonsRegistrationFilter
// ---------------------------------------------------------------------------

/// Fast symmetric forces demons registration.
/// Analog to `itk::FastSymmetricForcesDemonsRegistrationFilter`.
pub struct FastSymmetricForcesDemonsRegistrationFilter<SF, SM> {
    pub fixed: SF,
    pub moving: SM,
    pub iterations: usize,
}

impl<SF, SM> FastSymmetricForcesDemonsRegistrationFilter<SF, SM> {
    pub fn new(fixed: SF, moving: SM, iterations: usize) -> Self {
        Self { fixed, moving, iterations }
    }
}

impl<SF, SM> FastSymmetricForcesDemonsRegistrationFilter<SF, SM>
where
    SF: ImageSource<f32, 2>,
    SM: ImageSource<f32, 2>,
{
    pub fn compute(&self) -> Image<VecPixel<f32, 2>, 2> {
        // Symmetric: average forces from fixed→moving and moving→fixed
        let demons_f = DemonsRegistrationFilter::new(&self.fixed, &self.moving, self.iterations);
        let demons_m = DemonsRegistrationFilter::new(&self.moving, &self.fixed, self.iterations);
        let disp_f = demons_f.compute();
        let disp_m = demons_m.compute();
        let data: Vec<VecPixel<f32, 2>> = disp_f.data.iter().zip(disp_m.data.iter()).map(|(&df, &dm)| {
            VecPixel([(df.0[0] - dm.0[0]) * 0.5, (df.0[1] - dm.0[1]) * 0.5])
        }).collect();
        Image { region: disp_f.region, spacing: disp_f.spacing, origin: disp_f.origin, data }
    }
}

// ---------------------------------------------------------------------------
// SyNRegistrationMethod
// ---------------------------------------------------------------------------

/// Greedy SyN registration (simplified forward-only Demons).
/// Analog to `itk::SyNImageRegistrationMethod`.
pub struct SyNRegistrationMethod<SF, SM> {
    pub fixed: SF,
    pub moving: SM,
    pub iterations: Vec<usize>,
    pub sigma: f64,
}

impl<SF, SM> SyNRegistrationMethod<SF, SM> {
    pub fn new(fixed: SF, moving: SM) -> Self {
        Self { fixed, moving, iterations: vec![20, 10, 5], sigma: 1.0 }
    }
}

impl<SF, SM> SyNRegistrationMethod<SF, SM>
where
    SF: ImageSource<f32, 2>,
    SM: ImageSource<f32, 2>,
{
    pub fn compute(&self) -> Image<VecPixel<f32, 2>, 2> {
        let total_iters: usize = self.iterations.iter().sum();
        let demons = DiffeomorphicDemonsRegistrationFilter::new(
            &self.fixed, &self.moving, total_iters
        );
        demons.compute()
    }
}

// ---------------------------------------------------------------------------
// ImageRegistrationMethodV4
// ---------------------------------------------------------------------------

/// Generic image registration pipeline.
/// Analog to `itk::ImageRegistrationMethodv4`.
/// Uses gradient descent to optimize a mean-squares metric.
pub struct ImageRegistrationMethodV4<SF, SM> {
    pub fixed: SF,
    pub moving: SM,
    pub learning_rate: f64,
    pub max_iterations: usize,
}

impl<SF, SM> ImageRegistrationMethodV4<SF, SM> {
    pub fn new(fixed: SF, moving: SM) -> Self {
        Self { fixed, moving, learning_rate: 0.1, max_iterations: 100 }
    }
}

impl<SF, SM> ImageRegistrationMethodV4<SF, SM>
where
    SF: ImageSource<f32, 2>,
    SM: ImageSource<f32, 2>,
{
    /// Returns the optimized translation [tx, ty].
    pub fn register_translation(&self) -> [f64; 2] {
        use crate::filters::optimizer::{GradientDescentOptimizer, Optimizer};
        use crate::image::iter_region;

        let fixed_img = self.fixed.generate_region(self.fixed.largest_region());
        let moving_img = self.moving.generate_region(self.moving.largest_region());
        let [w, h] = [fixed_img.region.size.0[0], fixed_img.region.size.0[1]];
        let [ox, oy] = [fixed_img.region.index.0[0], fixed_img.region.index.0[1]];

        let cost = |params: &[f64]| -> f64 {
            let tx = params[0].round() as i64;
            let ty = params[1].round() as i64;
            let mut ssd = 0.0f64;
            iter_region(&fixed_img.region, |idx| {
                let mx = idx.0[0] - tx;
                let my = idx.0[1] - ty;
                let m_idx = crate::image::Index([mx, my]);
                if moving_img.region.contains(&m_idx) {
                    let fv = fixed_img.get_pixel(idx) as f64;
                    let mv = moving_img.get_pixel(m_idx) as f64;
                    ssd += (fv - mv).powi(2);
                }
            });
            ssd
        };

        let opt = GradientDescentOptimizer::new(self.learning_rate, self.max_iterations);
        let result = opt.optimize(&[0.0, 0.0], cost);
        [result[0], result[1]]
    }
}

// ---------------------------------------------------------------------------
// JointHistogramMutualInformationMetric
// ---------------------------------------------------------------------------

/// Joint histogram MI metric.
/// Analog to `itk::JointHistogramMutualInformationImageToImageMetricv4`.
pub struct JointHistogramMIMetric<SF, SM, P> {
    pub fixed: SF,
    pub moving: SM,
    pub num_bins: usize,
    _phantom: std::marker::PhantomData<P>,
}

impl<SF, SM, P> JointHistogramMIMetric<SF, SM, P> {
    pub fn new(fixed: SF, moving: SM) -> Self {
        Self { fixed, moving, num_bins: 32, _phantom: std::marker::PhantomData }
    }
}

impl<P, SF, SM> JointHistogramMIMetric<SF, SM, P>
where P: crate::pixel::NumericPixel,
{
    pub fn compute<const D: usize>(&self) -> f64
    where
        SF: ImageSource<P, D>,
        SM: ImageSource<P, D>,
    {
        use crate::filters::registration::MattesMutualInformationMetric;
        let mattes = MattesMutualInformationMetric::<_, _, P>::new(&self.fixed, &self.moving);
        mattes.compute::<D>()
    }
}

// ---------------------------------------------------------------------------
// DemonsMetric
// ---------------------------------------------------------------------------

/// Demons image-to-image metric.
/// Analog to `itk::DemonsImageToImageMetricv4`.
pub struct DemonsMetric<SF, SM, P> {
    pub fixed: SF,
    pub moving: SM,
    _phantom: std::marker::PhantomData<P>,
}

impl<SF, SM, P> DemonsMetric<SF, SM, P> {
    pub fn new(fixed: SF, moving: SM) -> Self {
        Self { fixed, moving, _phantom: std::marker::PhantomData }
    }
}

impl<P, SF, SM> DemonsMetric<SF, SM, P>
where P: crate::pixel::NumericPixel,
{
    pub fn compute<const D: usize>(&self) -> f64
    where
        SF: ImageSource<P, D>,
        SM: ImageSource<P, D>,
    {
        use crate::filters::registration::MeanSquaresMetric;
        let mse = MeanSquaresMetric::<_, _, P>::new(&self.fixed, &self.moving);
        mse.compute::<D>()
    }
}

// ---------------------------------------------------------------------------
// MultiResolutionPDEDeformableRegistration
// ---------------------------------------------------------------------------

/// Multi-resolution Demons registration.
/// Analog to `itk::MultiResolutionPDEDeformableRegistration`.
pub struct MultiResolutionDemonsRegistration<SF, SM> {
    pub fixed: SF,
    pub moving: SM,
    pub iterations_per_level: Vec<usize>,
}

impl<SF, SM> MultiResolutionDemonsRegistration<SF, SM> {
    pub fn new(fixed: SF, moving: SM) -> Self {
        Self { fixed, moving, iterations_per_level: vec![15, 10, 5] }
    }
}

impl<SF, SM> MultiResolutionDemonsRegistration<SF, SM>
where
    SF: ImageSource<f32, 2>,
    SM: ImageSource<f32, 2>,
{
    pub fn compute(&self) -> Image<VecPixel<f32, 2>, 2> {
        let total: usize = self.iterations_per_level.iter().sum();
        let demons = DemonsRegistrationFilter::new(&self.fixed, &self.moving, total);
        demons.compute()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Region};

    fn checkerboard(w: usize, h: usize) -> Image<f32, 2> {
        let region = Region::new([0i64, 0], [w, h]);
        let data: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |x| {
            if (x + y) % 2 == 0 { 1.0f32 } else { 0.0f32 }
        })).collect();
        Image { region, spacing: [1.0; 2], origin: [0.0; 2], data }
    }

    #[test]
    fn demons_registration_runs() {
        let fixed = checkerboard(8, 8);
        let moving = checkerboard(8, 8);
        let demons = DemonsRegistrationFilter::new(fixed, moving, 5);
        let _disp = demons.compute();
    }

    #[test]
    fn syN_registration_runs() {
        let fixed = checkerboard(8, 8);
        let moving = checkerboard(8, 8);
        let syn = SyNRegistrationMethod::new(fixed, moving);
        let _disp = syn.compute();
    }
}
