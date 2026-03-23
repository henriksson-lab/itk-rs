//! Windowed sinc interpolation.
//! Analog to `itk::WindowedSincInterpolateImageFunction`.
//!
//! Approximates the ideal (infinite) sinc filter by multiplying sinc with a
//! finite support window. The window type and radius are compile-time or
//! run-time parameters.
//!
//! Supported window functions (matching ITK):
//! - [`WindowFunction::Hamming`]  — `0.54 + 0.46 cos(πx/m)`  (default)
//! - [`WindowFunction::Cosine`]   — `cos(πx / 2m)`
//! - [`WindowFunction::Welch`]    — `1 − (x/m)²`
//! - [`WindowFunction::Lanczos`]  — `sinc(x/m)`
//! - [`WindowFunction::Blackman`] — `0.42 + 0.5 cos(πx/m) + 0.08 cos(2πx/m)`
//!
//! The kernel `K(t) = window(t) × sinc(t)` is applied separably to each axis.
//! Boundary: zero-flux Neumann (clamp-to-edge).

use std::f64::consts::PI;

use crate::image::Image;
use crate::pixel::NumericPixel;

use super::{Interpolate, clamp_index};

/// Choice of window function applied to the sinc kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowFunction {
    /// `0.54 + 0.46 cos(πx/m)` — good general-purpose window.
    Hamming,
    /// `cos(πx / 2m)` — narrow transition band.
    Cosine,
    /// `1 − (x/m)²` — smooth parabolic window.
    Welch,
    /// `sinc(x/m)` — Lanczos / sinc-in-sinc; often the best quality.
    Lanczos,
    /// `0.42 + 0.5 cos(πx/m) + 0.08 cos(2πx/m)` — lowest sidelobe level.
    Blackman,
}

/// Windowed sinc interpolator.
///
/// - `radius`: half-width of the kernel (number of neighbours each side).
///   Larger radius → higher quality but more computation.
///   Recommended: 3–5.
/// - `window`: window function applied to the sinc kernel.
pub struct WindowedSincInterpolator {
    pub radius: usize,
    pub window: WindowFunction,
}

impl WindowedSincInterpolator {
    pub fn new(radius: usize, window: WindowFunction) -> Self {
        Self { radius, window }
    }

    /// Evaluate the window function at distance `x` from centre, with half-width `m`.
    fn window_value(&self, x: f64, m: f64) -> f64 {
        if x.abs() >= m {
            return 0.0;
        }
        match self.window {
            WindowFunction::Hamming  => 0.54 + 0.46 * (PI * x / m).cos(),
            WindowFunction::Cosine   => (PI * x / (2.0 * m)).cos(),
            WindowFunction::Welch    => 1.0 - (x / m).powi(2),
            WindowFunction::Lanczos  => sinc(x / m),
            WindowFunction::Blackman =>
                0.42 + 0.5 * (PI * x / m).cos() + 0.08 * (2.0 * PI * x / m).cos(),
        }
    }

    /// Kernel value `K(t) = window(t) × sinc(t)`.
    fn kernel(&self, t: f64) -> f64 {
        self.window_value(t, self.radius as f64) * sinc(t)
    }
}

impl<P: NumericPixel, const D: usize> Interpolate<P, D> for WindowedSincInterpolator {
    fn evaluate(&self, image: &Image<P, D>, index: [f64; D]) -> P {
        let r = self.radius as i64;

        // Per-axis kernel weights over [floor(x) - r + 1, floor(x) + r]
        let mut per_axis: Vec<(i64, Vec<f64>)> = Vec::with_capacity(D);
        for d in 0..D {
            let floor = index[d].floor() as i64;
            let frac = index[d] - floor as f64;   // ∈ [0, 1)
            let start = floor - r + 1;
            let weights: Vec<f64> = (0..2 * r as usize)
                .map(|k| {
                    let t = k as f64 - (r as f64 - 1.0) - frac; // distance from query
                    self.kernel(t)
                })
                .collect();
            per_axis.push((start, weights));
        }

        // Accumulate over the 2r×…×2r support hypercube
        accumulate(image, &per_axis, 0, [0i64; D])
    }
}

/// Recursively accumulate contributions from each axis via separable convolution.
fn accumulate<P: NumericPixel, const D: usize>(
    image: &Image<P, D>,
    per_axis: &[(i64, Vec<f64>)],
    axis: usize,
    mut idx: [i64; D],
) -> P {
    if axis == D {
        return image.get_pixel(clamp_index(image, idx));
    }
    let (start, weights) = &per_axis[axis];
    let mut acc = P::zero();
    for (k, &w) in weights.iter().enumerate() {
        idx[axis] = start + k as i64;
        let sub = accumulate(image, per_axis, axis + 1, idx);
        acc = acc + sub.scale(w);
    }
    acc
}

/// Normalised sinc: `sin(πx) / (πx)`, with `sinc(0) = 1`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Index, Region};

    fn ramp_2d(size: usize) -> Image<f32, 2> {
        let mut img = Image::<f32, 2>::allocate(
            Region::new([0, 0], [size, size]), [1.0; 2], [0.0; 2], 0.0,
        );
        for y in 0..size as i64 {
            for x in 0..size as i64 {
                img.set_pixel(Index([x, y]), x as f32);
            }
        }
        img
    }

    #[test]
    fn sinc_at_integer_reproduces_pixel() {
        // For a ramp image, sinc at integer positions must return exact values
        let img = ramp_2d(10);
        let interp = WindowedSincInterpolator::new(4, WindowFunction::Hamming);
        for x in 2..8 {
            let v = interp.evaluate(&img, [x as f64, 4.0]);
            assert!((v - x as f32).abs() < 0.01, "at x={x}: got {v}, expected {x}");
        }
    }

    #[test]
    fn all_window_functions_compile_and_run() {
        let img = ramp_2d(12);
        for wf in [
            WindowFunction::Hamming,
            WindowFunction::Cosine,
            WindowFunction::Welch,
            WindowFunction::Lanczos,
            WindowFunction::Blackman,
        ] {
            let interp = WindowedSincInterpolator::new(3, wf);
            let v = interp.evaluate(&img, [5.5, 5.0]);
            assert!(v.is_finite(), "{wf:?} returned non-finite value");
        }
    }

    #[test]
    fn lanczos_midpoint_close_to_linear() {
        // Midpoint of a ramp: sinc should give ≈ linear result
        let img = ramp_2d(12);
        let interp = WindowedSincInterpolator::new(4, WindowFunction::Lanczos);
        let v = interp.evaluate(&img, [3.5, 4.0]);
        assert!((v - 3.5).abs() < 0.05, "expected ~3.5, got {v}");
    }
}
