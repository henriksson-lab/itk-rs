//! Median filter. Analog to `itk::MedianImageFilter`.
//!
//! Collects all pixels in a D-dimensional box neighbourhood and returns the
//! median value.  Non-linear — cannot be decomposed into separable 1-D passes.
//!
//! Boundary: zero-flux Neumann (clamp-to-edge).

use rayon::prelude::*;

use crate::image::{Image, Index, Region, iter_region};
use crate::pixel::NumericPixel;
use crate::source::ImageSource;

/// Median filter with a symmetric box neighbourhood.
///
/// Requires `P: NumericPixel + PartialOrd` so that values can be sorted.
pub struct MedianFilter<S> {
    pub source: S,
    /// Box half-width in voxels (same on every axis).
    pub radius: usize,
}

impl<S> MedianFilter<S> {
    pub fn new(source: S, radius: usize) -> Self {
        Self { source, radius }
    }
}

impl<P, S, const D: usize> ImageSource<P, D> for MedianFilter<S>
where
    P: NumericPixel + PartialOrd,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        let r = self.radius;
        let radii = { let mut a = [0usize; D]; a.iter_mut().for_each(|v| *v = r); a };
        output_region.padded_per_axis(&radii).clipped_to(&self.source.largest_region())
    }

    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        let r = self.radius;
        let radii = { let mut a = [0usize; D]; a.iter_mut().for_each(|v| *v = r); a };
        let bounds = self.source.largest_region();
        let input_region = requested.padded_per_axis(&radii).clipped_to(&bounds);
        let input = self.source.generate_region(input_region);

        // Collect output indices
        let mut indices: Vec<Index<D>> = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| indices.push(idx));

        let data: Vec<P> = indices
            .par_iter()
            .map(|&out_idx| {
                // Build the neighbourhood box
                let nbr_size = { let mut s = [0usize; D]; s.iter_mut().for_each(|v| *v = 2*r+1); s };
                let nbr_start: [i64; D] = {
                    let mut s = out_idx.0;
                    for d in 0..D { s[d] -= r as i64; }
                    s
                };
                let nbr_region = Region::new(nbr_start, nbr_size);

                let mut values: Vec<P> = Vec::with_capacity((2*r+1).pow(D as u32));
                iter_region(&nbr_region, |idx| {
                    // Clamp to input bounds (Neumann)
                    let mut clamped = idx.0;
                    for d in 0..D {
                        clamped[d] = clamped[d]
                            .max(input.region.index.0[d])
                            .min(input.region.index.0[d] + input.region.size.0[d] as i64 - 1);
                    }
                    values.push(input.get_pixel(Index(clamped)));
                });

                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                values[values.len() / 2]
            })
            .collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    #[test]
    fn median_constant_image() {
        let img = Image::<f32, 2>::allocate(
            Region::new([0, 0], [10, 10]), [1.0; 2], [0.0; 2], 7.0f32,
        );
        let f = MedianFilter::new(img, 1);
        let out = f.generate_region(f.largest_region());
        for &v in &out.data {
            assert!((v - 7.0).abs() < 1e-6);
        }
    }

    #[test]
    fn median_removes_spike() {
        // Place an outlier spike in a constant image; median should suppress it.
        let mut img = Image::<f32, 1>::allocate(
            Region::new([0], [9]), [1.0], [0.0], 0.0f32,
        );
        img.set_pixel(Index([4]), 100.0);

        let f = MedianFilter::new(img, 1);
        let out = f.generate_region(f.largest_region());
        let centre = out.get_pixel(Index([4]));
        assert!(centre < 5.0, "spike not suppressed, got {centre}");
    }
}
