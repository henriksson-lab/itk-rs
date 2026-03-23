use nitk::filters::gaussian::GaussianFilter;
use nitk::filters::UnaryFilter;
use nitk::source::{stream_full, ImageSource};
use nitk::{Image, Index, Region, VecPixel};

fn solid_image<const D: usize>(size: [usize; D], fill: f32) -> Image<f32, D> {
    Image::allocate(
        Region::new([0i64; D], size),
        [1.0f64; D],
        [0.0f64; D],
        fill,
    )
}

// ---------------------------------------------------------------------------
// ImageSource identity: Image.generate_region returns a correct subregion
// ---------------------------------------------------------------------------

#[test]
fn test_image_source_identity_2d() {
    let mut img = solid_image([8, 8], 0.0);
    // Mark a known pixel
    img.set_pixel(Index([3, 5]), 42.0);

    let sub = Region::new([2, 4], [4, 3]);
    let cropped = img.generate_region(sub);

    assert_eq!(cropped.region, sub);
    assert_eq!(cropped.get_pixel(Index([3, 5])), 42.0);
    assert_eq!(cropped.get_pixel(Index([2, 4])), 0.0);
}

// ---------------------------------------------------------------------------
// Gaussian impulse response: energy conserved, peak at centre
// ---------------------------------------------------------------------------

#[test]
fn test_gaussian_impulse_2d() {
    let size = 21usize;
    let mid = (size / 2) as i64;
    let mut img = solid_image([size, size], 0.0);
    img.set_pixel(Index([mid, mid]), 1.0);

    let filter = GaussianFilter::new(img, 1.5);
    let out = filter.generate_region(filter.largest_region());

    let sum: f32 = out.data.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "energy not conserved: sum={sum}");

    let peak = out.get_pixel(Index([mid, mid]));
    for &v in &out.data {
        assert!(v <= peak + 1e-6, "pixel {v} > peak {peak}");
    }
}

// ---------------------------------------------------------------------------
// Gaussian 3D: const generic D=3 compiles and runs
// ---------------------------------------------------------------------------

#[test]
fn test_gaussian_3d() {
    let size = 11usize;
    let mid = (size / 2) as i64;
    let mut img = solid_image([size, size, size], 0.0);
    img.set_pixel(Index([mid, mid, mid]), 1.0);

    let filter = GaussianFilter::new(img, 1.0);
    let out = filter.generate_region(filter.largest_region());

    let sum: f32 = out.data.iter().sum();
    assert!((sum - 1.0).abs() < 0.05, "3D energy not conserved: sum={sum}");
}

// ---------------------------------------------------------------------------
// Vector pixel: Image<VecPixel<f32,3>, 2> passes through GaussianFilter
// ---------------------------------------------------------------------------

#[test]
fn test_vector_pixel_gaussian() {
    let size = 11usize;
    let mid = (size / 2) as i64;
    let mut img: Image<VecPixel<f32, 3>, 2> = Image::allocate(
        Region::new([0, 0], [size, size]),
        [1.0, 1.0],
        [0.0, 0.0],
        VecPixel::default(),
    );
    img.set_pixel(Index([mid, mid]), VecPixel([1.0, 2.0, 3.0]));

    let filter = GaussianFilter::new(img, 1.0);
    let out = filter.generate_region(filter.largest_region());

    // Each channel should sum to its impulse weight
    let sum0: f32 = out.data.iter().map(|p| p.0[0]).sum();
    let sum1: f32 = out.data.iter().map(|p| p.0[1]).sum();
    let sum2: f32 = out.data.iter().map(|p| p.0[2]).sum();
    assert!((sum0 - 1.0).abs() < 0.05, "ch0 sum={sum0}");
    assert!((sum1 - 2.0).abs() < 0.05, "ch1 sum={sum1}");
    assert!((sum2 - 3.0).abs() < 0.05, "ch2 sum={sum2}");
}

// ---------------------------------------------------------------------------
// Streaming: 4-piece stream == single generate_region on full image
// ---------------------------------------------------------------------------

#[test]
fn test_streaming_matches_full() {
    let size = 20usize;
    let mid = (size / 2) as i64;
    let mut img = solid_image([size, size], 0.0);
    img.set_pixel(Index([mid, mid]), 1.0);

    let filter = GaussianFilter::new(img, 1.5);
    let full = filter.generate_region(filter.largest_region());

    // Rebuild source (GaussianFilter is not Clone, so reconstruct)
    let size2 = 20usize;
    let mid2 = (size2 / 2) as i64;
    let mut img2 = solid_image([size2, size2], 0.0);
    img2.set_pixel(Index([mid2, mid2]), 1.0);
    let filter2 = GaussianFilter::new(img2, 1.5);

    let streamed = stream_full(&filter2, 4);

    assert_eq!(full.data.len(), streamed.data.len());
    for (a, b) in full.data.iter().zip(streamed.data.iter()) {
        assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
    }
}

// ---------------------------------------------------------------------------
// Chained pipeline: GaussianFilter followed by UnaryFilter (threshold)
// ---------------------------------------------------------------------------

#[test]
fn test_unary_filter_pipeline() {
    let size = 21usize;
    let mid = (size / 2) as i64;
    let mut img = solid_image([size, size], 0.0);
    img.set_pixel(Index([mid, mid]), 1.0);

    let gaussian = GaussianFilter::new(img, 1.5);
    let threshold = UnaryFilter::<_, _, f32, f32>::new(
        gaussian,
        |p: f32| if p > 0.01 { 1.0f32 } else { 0.0f32 },
    );

    let out = threshold.generate_region(threshold.largest_region());
    // Should have some lit pixels (near centre) and some dark (far from centre)
    let lit: usize = out.data.iter().filter(|&&v| v > 0.5).count();
    assert!(lit > 0, "no pixels above threshold");
    assert!(lit < size * size, "all pixels above threshold");
}
