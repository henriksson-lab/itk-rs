#![cfg(feature = "io-png")]

use std::env::temp_dir;

use nitk::filters::gaussian::GaussianFilter;
use nitk::filters::UnaryFilter;
use nitk::image::{Image, Index, Region};
use nitk::io::png::{read_gray8, write_gray8};
use nitk::source::ImageSource;

/// Write a synthetic grayscale PNG (impulse at centre), blur it, save result,
/// reload and verify that:
///   - The peak moved from maximum to a smoothed value
///   - Total intensity is conserved (within rounding)
///   - The output file is a valid PNG with the correct dimensions
#[test]
fn blur_png_roundtrip() {
    let size = 32usize;
    let mid = (size / 2) as i64;

    // --- Build a synthetic image ---
    let mut img = Image::<u8, 2>::allocate(
        Region::new([0, 0], [size, size]),
        [1.0, 1.0],
        [0.0, 0.0],
        0u8,
    );
    img.set_pixel(Index([mid, mid]), 255u8);

    // --- Save to a temp PNG ---
    let input_path = temp_dir().join("nitk_blur_input.png");
    let output_path = temp_dir().join("nitk_blur_output.png");
    write_gray8(&img, &input_path).expect("failed to write input PNG");

    // --- Read back, blur, write output ---
    let loaded = read_gray8(&input_path).expect("failed to read input PNG");

    let as_f32 = UnaryFilter::new(loaded, |p: u8| p as f32);
    let blurred = GaussianFilter::new(as_f32, 2.0);
    let as_u8 = UnaryFilter::new(blurred, |p: f32| p.round().clamp(0.0, 255.0) as u8);

    let result = as_u8.generate_region(as_u8.largest_region());
    write_gray8(&result, &output_path).expect("failed to write output PNG");

    // --- Reload and verify ---
    let output = read_gray8(&output_path).expect("failed to read output PNG");

    assert_eq!(output.region.size.0, [size, size], "output dimensions changed");

    // Original peak was 255; after blur it must be lower
    let peak_before = 255u8;
    let peak_after = *output.data.iter().max().unwrap();
    assert!(
        peak_after < peak_before,
        "blur should reduce the peak: before={peak_before}, after={peak_after}"
    );

    // The blurred peak should still be at or near the centre
    let centre_val = output.get_pixel(Index([mid, mid]));
    assert!(
        centre_val == peak_after,
        "peak should remain at centre: centre={centre_val}, max={peak_after}"
    );

    // Total intensity should be roughly conserved.
    // Rounding f32→u8 discards small tail values, so allow up to ~15% loss.
    let sum_before: u32 = 255;
    let sum_after: u32 = output.data.iter().map(|&v| v as u32).sum();
    assert!(
        sum_after >= sum_before * 85 / 100,
        "too much intensity lost to rounding: before={sum_before}, after={sum_after}"
    );
}
