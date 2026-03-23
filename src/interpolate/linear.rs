//! N-dimensional linear interpolation. Analog to `itk::LinearInterpolateImageFunction`.
//!
//! Uses 2^D neighbouring pixels with weights = products of per-axis distances.
//! Zero precomputation required.

use crate::image::Image;
use crate::pixel::NumericPixel;

use super::{Interpolate, clamp_index};

/// N-D linear (bilinear / trilinear / …) interpolation.
pub struct LinearInterpolator;

impl<P: NumericPixel, const D: usize> Interpolate<P, D> for LinearInterpolator {
    fn evaluate(&self, image: &Image<P, D>, index: [f64; D]) -> P {
        // Floor index and fractional distances
        let mut floor = [0i64; D];
        let mut frac = [0f64; D];
        for d in 0..D {
            floor[d] = index[d].floor() as i64;
            frac[d] = index[d] - floor[d] as f64;
        }

        // Sum over 2^D corners using bitmask to select floor/ceil per axis
        let num_corners = 1usize << D;
        let mut acc = P::zero();
        for corner in 0..num_corners {
            let mut weight = 1.0f64;
            let mut idx = floor;
            for d in 0..D {
                if (corner >> d) & 1 == 1 {
                    idx[d] += 1;
                    weight *= frac[d];
                } else {
                    weight *= 1.0 - frac[d];
                }
            }
            let pixel = image.get_pixel(clamp_index(image, idx));
            acc = acc + pixel.scale(weight);
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Index, Region};

    fn ramp_2d() -> Image<f32, 2> {
        // pixel value = x + y * 10 for easy checking
        let mut img =
            Image::<f32, 2>::allocate(Region::new([0, 0], [4, 4]), [1.0; 2], [0.0; 2], 0.0);
        for y in 0..4i64 {
            for x in 0..4i64 {
                img.set_pixel(Index([x, y]), (x + y * 10) as f32);
            }
        }
        img
    }

    #[test]
    fn linear_exact_pixel() {
        let img = ramp_2d();
        let interp = LinearInterpolator;
        assert_eq!(interp.evaluate(&img, [2.0, 1.0]), 12.0);
    }

    #[test]
    fn linear_midpoint_x() {
        let img = ramp_2d();
        let interp = LinearInterpolator;
        // Midpoint between x=1 and x=2 at y=0: (1 + 2) / 2 = 1.5
        assert!((interp.evaluate(&img, [1.5, 0.0]) - 1.5).abs() < 1e-5);
    }

    #[test]
    fn linear_midpoint_xy() {
        let img = ramp_2d();
        let interp = LinearInterpolator;
        // Centre of 4 pixels at (0,0),(1,0),(0,1),(1,1): values 0,1,10,11 → mean=5.5
        assert!((interp.evaluate(&img, [0.5, 0.5]) - 5.5).abs() < 1e-5);
    }
}
