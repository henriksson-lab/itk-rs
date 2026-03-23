use crate::image::{Image, Region, iter_region};
use crate::pixel::Pixel;
use crate::region_splitter::split_region;

/// The core pipeline trait. Analog to itk::ImageSource + itk::ImageToImageFilter.
///
/// Every pipeline node — raw images, file readers, and filters — implements this.
/// Three methods correspond to ITK's three pipeline phases:
///
/// - Phase 1 (UpdateOutputInformation): `largest_region`, `spacing`, `origin`
/// - Phase 2 (PropagateRequestedRegion): `input_region_for_output`
/// - Phase 3 (UpdateOutputData): `generate_region`
pub trait ImageSource<P, const D: usize> {
    /// The full extent of data this source can produce (LargestPossibleRegion).
    fn largest_region(&self) -> Region<D>;
    fn spacing(&self) -> [f64; D];
    fn origin(&self) -> [f64; D];

    /// What input region does this source require to produce `output_region`?
    ///
    /// Default: identity — sufficient for pixel-wise filters.
    /// Neighborhood filters override this to pad the region.
    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        *output_region
    }

    /// Allocate and compute pixels for `requested` (BufferedRegion = RequestedRegion).
    fn generate_region(&self, requested: Region<D>) -> Image<P, D>;
}

/// `Image` is both a buffer and a source — it can serve as a pipeline leaf.
impl<P: Pixel, const D: usize> ImageSource<P, D> for Image<P, D> {
    fn largest_region(&self) -> Region<D> {
        self.region
    }
    fn spacing(&self) -> [f64; D] {
        self.spacing
    }
    fn origin(&self) -> [f64; D] {
        self.origin
    }
    fn generate_region(&self, requested: Region<D>) -> Image<P, D> {
        // Crop to the intersection of requested and self.region
        let clipped = requested.clipped_to(&self.region);
        let mut out = Image {
            region: clipped,
            spacing: self.spacing,
            origin: self.origin,
            data: Vec::with_capacity(clipped.linear_len()),
        };
        // Fill with default then copy
        out.data.resize(clipped.linear_len(), P::default());
        iter_region(&clipped, |idx| {
            let v = self.get_pixel(idx);
            out.set_pixel(idx, v);
        });
        out
    }
}

/// Drive a source through all its data in `num_pieces` streaming chunks.
///
/// Analog to itk::StreamingImageFilter: allocates the full output buffer once,
/// then fills it piece by piece so upstream memory is bounded by one chunk.
pub fn stream_full<P, const D: usize, S>(source: &S, num_pieces: usize) -> Image<P, D>
where
    P: Pixel,
    S: ImageSource<P, D>,
{
    let full = source.largest_region();
    let mut output = Image {
        region: full,
        spacing: source.spacing(),
        origin: source.origin(),
        data: vec![P::default(); full.linear_len()],
    };
    for piece in split_region(full, num_pieces) {
        let piece_data = source.generate_region(piece);
        output.copy_from(&piece_data);
    }
    output
}
