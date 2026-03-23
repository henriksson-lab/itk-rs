pub(crate) mod conv;
pub mod gaussian;
pub mod mean;
pub mod median;
pub mod binomial_blur;
pub mod recursive_gaussian;
pub mod bilateral;
pub mod anisotropic_diffusion;
pub mod intensity;
pub mod threshold;
pub mod edges;
pub mod morphology;
pub mod spatial;
pub mod statistics;
pub mod noise;
pub mod sources;
pub mod convolution;
pub mod distance;
pub mod fast_marching;
pub mod curvature_flow;
pub mod segmentation;
pub mod colormap;
pub mod label_map;
pub mod displacement;
pub mod registration;
pub mod fft;
pub mod deconvolution;

use std::marker::PhantomData;

use crate::image::{Image, Region};
use crate::pixel::Pixel;
use crate::source::ImageSource;
use rayon::prelude::*;

/// Per-pixel transform filter. Analog to itk::UnaryFunctorImageFilter.
///
/// `F: Fn(P) -> Q` is the functor; can be a closure capturing configuration.
/// Computation is parallelised over pixels with rayon.
///
/// `P` and `Q` are the input and output pixel types. They are carried as
/// PhantomData so the impl can be unambiguous.
pub struct UnaryFilter<S, F, P, Q> {
    pub source: S,
    pub func: F,
    _phantom: PhantomData<fn(P) -> Q>,
}

impl<S, F, P, Q> UnaryFilter<S, F, P, Q> {
    pub fn new(source: S, func: F) -> Self {
        Self { source, func, _phantom: PhantomData }
    }
}

impl<S, F, P, Q, const D: usize> ImageSource<Q, D> for UnaryFilter<S, F, P, Q>
where
    S: ImageSource<P, D> + Sync,
    F: Fn(P) -> Q + Sync + Send,
    P: Pixel,
    Q: Pixel,
{
    fn largest_region(&self) -> Region<D> {
        self.source.largest_region()
    }
    fn spacing(&self) -> [f64; D] {
        self.source.spacing()
    }
    fn origin(&self) -> [f64; D] {
        self.source.origin()
    }

    fn generate_region(&self, requested: Region<D>) -> Image<Q, D> {
        let input = self.source.generate_region(requested);
        let data: Vec<Q> = input.data.par_iter().map(|&p| (self.func)(p)).collect();
        Image { region: input.region, spacing: input.spacing, origin: input.origin, data }
    }
}
