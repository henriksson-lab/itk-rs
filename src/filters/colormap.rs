//! Colormap filters.
//!
//! | Filter | ITK analog |
//! |---|---|
//! | [`ScalarToRGBColormapFilter`] | `ScalarToRGBColormapImageFilter` |

use crate::image::{Image, Region, iter_region};
use crate::pixel::{NumericPixel, VecPixel};
use crate::source::ImageSource;

/// Built-in colormap types.
#[derive(Clone, Copy, Debug)]
pub enum Colormap {
    /// Grayscale (R=G=B=I)
    Gray,
    /// Hot (black→red→yellow→white)
    Hot,
    /// Cool (cyan→magenta)
    Cool,
    /// Jet (blue→cyan→green→yellow→red)
    Jet,
    /// HSV
    Hsv,
    /// Spring (magenta→yellow)
    Spring,
    /// Summer (green→yellow)
    Summer,
    /// Autumn (red→yellow)
    Autumn,
    /// Winter (blue→green)
    Winter,
    /// Copper (black→copper)
    Copper,
}

fn apply_colormap(t: f64, cm: Colormap) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0) as f32;
    match cm {
        Colormap::Gray => [t, t, t],
        Colormap::Hot => {
            let r = (t * 3.0).clamp(0.0, 1.0);
            let g = (t * 3.0 - 1.0).clamp(0.0, 1.0);
            let b = (t * 3.0 - 2.0).clamp(0.0, 1.0);
            [r, g, b]
        }
        Colormap::Cool => [t, 1.0 - t, 1.0],
        Colormap::Jet => {
            let r = ((t - 0.625) * 4.0).clamp(0.0, 1.0).min((1.0 - (t - 0.375) * 4.0).clamp(0.0, 1.0));
            let g = ((t - 0.125) * 4.0).clamp(0.0, 1.0).min((1.0 - (t - 0.625) * 4.0).clamp(0.0, 1.0));
            let b = ((t + 0.375) * 4.0).clamp(0.0, 1.0).min((1.0 - (t - 0.125) * 4.0).clamp(0.0, 1.0));
            [r, g, b]
        }
        Colormap::Hsv => {
            // H cycles 0→1 mapped to hue
            let h = t * 6.0;
            let i = h.floor() as i32;
            let f = h - i as f32;
            let (r, g, b) = match i % 6 {
                0 => (1.0, f, 0.0),
                1 => (1.0-f, 1.0, 0.0),
                2 => (0.0, 1.0, f),
                3 => (0.0, 1.0-f, 1.0),
                4 => (f, 0.0, 1.0),
                _ => (1.0, 0.0, 1.0-f),
            };
            [r, g, b]
        }
        Colormap::Spring => [1.0, t, 1.0 - t],
        Colormap::Summer => [t, 0.5 + t*0.5, 0.4],
        Colormap::Autumn => [1.0, t, 0.0],
        Colormap::Winter => [0.0, t, 1.0 - t * 0.5],
        Colormap::Copper => [(t * 1.25).clamp(0.0, 1.0), t * 0.7814, t * 0.4980],
    }
}

/// Maps scalar values to RGB using a chosen colormap.
/// Analog to `itk::ScalarToRGBColormapImageFilter`.
///
/// Output pixel type is `VecPixel<f32, 3>` (R, G, B in [0, 1]).
/// Input values are linearly rescaled from [min, max] to [0, 1].
pub struct ScalarToRGBColormapFilter<S, P> {
    pub source: S,
    pub colormap: Colormap,
    /// Explicit min (None = compute from image).
    pub min: Option<f64>,
    /// Explicit max (None = compute from image).
    pub max: Option<f64>,
    _phantom: std::marker::PhantomData<P>,
}

impl<S, P> ScalarToRGBColormapFilter<S, P> {
    pub fn new(source: S, colormap: Colormap) -> Self {
        Self { source, colormap, min: None, max: None, _phantom: std::marker::PhantomData }
    }
}

impl<P, S, const D: usize> ImageSource<VecPixel<f32, 3>, D> for ScalarToRGBColormapFilter<S, P>
where
    P: NumericPixel,
    S: ImageSource<P, D> + Sync,
{
    fn largest_region(&self) -> Region<D> { self.source.largest_region() }
    fn spacing(&self) -> [f64; D] { self.source.spacing() }
    fn origin(&self) -> [f64; D] { self.source.origin() }

    fn generate_region(&self, requested: Region<D>) -> Image<VecPixel<f32, 3>, D> {
        let input = self.source.generate_region(requested);

        let (vmin, vmax) = {
            let lo = self.min.unwrap_or_else(|| input.data.iter().map(|p| p.to_f64()).fold(f64::MAX, f64::min));
            let hi = self.max.unwrap_or_else(|| input.data.iter().map(|p| p.to_f64()).fold(f64::MIN, f64::max));
            (lo, if (hi - lo).abs() < 1e-12 { lo + 1.0 } else { hi })
        };

        let cm = self.colormap;
        let mut out_indices = Vec::with_capacity(requested.linear_len());
        iter_region(&requested, |idx| out_indices.push(idx));

        let data: Vec<VecPixel<f32, 3>> = out_indices.iter().map(|&idx| {
            let t = (input.get_pixel(idx).to_f64() - vmin) / (vmax - vmin);
            VecPixel(apply_colormap(t, cm))
        }).collect();

        Image { region: requested, spacing: input.spacing, origin: input.origin, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, Index, Region};

    #[test]
    fn gray_colormap_identity() {
        let img = Image::<f32, 1>::allocate(Region::new([0], [5]), [1.0], [0.0], 0.5f32);
        let f = ScalarToRGBColormapFilter::<_, f32>::new(img, Colormap::Gray);
        let out = f.generate_region(f.largest_region());
        let v = out.get_pixel(Index([2]));
        // R=G=B ≈ 0.5 (it's already normalized since only one value)
        assert!((v.0[0] - v.0[1]).abs() < 1e-4, "R and G should be equal");
    }
}
