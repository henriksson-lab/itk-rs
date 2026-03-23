//! Nearest-neighbour interpolation. Analog to `itk::NearestNeighborInterpolateImageFunction`.

use crate::image::Image;
use crate::pixel::Pixel;

use super::{Interpolate, clamp_index};

/// Returns the value of the pixel nearest to the given continuous index.
///
/// Zero precomputation required; works with any pixel type.
pub struct NearestNeighborInterpolator;

impl<P: Pixel, const D: usize> Interpolate<P, D> for NearestNeighborInterpolator {
    fn evaluate(&self, image: &Image<P, D>, index: [f64; D]) -> P {
        let mut idx = [0i64; D];
        for d in 0..D {
            idx[d] = index[d].round() as i64;
        }
        image.get_pixel(clamp_index(image, idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Index, Region};

    #[test]
    fn nearest_exact_pixel() {
        let mut img = Image::<u8, 2>::allocate(Region::new([0, 0], [4, 4]), [1.0; 2], [0.0; 2], 0);
        img.set_pixel(Index([2, 1]), 42);
        let interp = NearestNeighborInterpolator;
        assert_eq!(interp.evaluate(&img, [2.0, 1.0]), 42);
    }

    #[test]
    fn nearest_rounds_to_closest() {
        let mut img = Image::<u8, 2>::allocate(Region::new([0, 0], [4, 4]), [1.0; 2], [0.0; 2], 0);
        img.set_pixel(Index([2, 1]), 99);
        let interp = NearestNeighborInterpolator;
        // 2.4 rounds to 2, 1.4 rounds to 1
        assert_eq!(interp.evaluate(&img, [2.4, 1.4]), 99);
        // 2.6 rounds to 3
        assert_eq!(interp.evaluate(&img, [2.6, 1.0]), 0);
    }

    #[test]
    fn nearest_clamps_to_boundary() {
        let mut img = Image::<u8, 2>::allocate(Region::new([0, 0], [4, 4]), [1.0; 2], [0.0; 2], 0);
        img.set_pixel(Index([3, 3]), 7);
        let interp = NearestNeighborInterpolator;
        // Index beyond bounds clamps to edge
        assert_eq!(interp.evaluate(&img, [10.0, 10.0]), 7);
    }
}
