//! Read a PNG, apply Gaussian blur, and write the result.
//!
//! Usage:
//!   cargo run --example blur_png -- input.png output.png [sigma]
//!
//! sigma defaults to 2.0 if not provided.

use itk_rs::filters::gaussian::GaussianFilter;
use itk_rs::io::png::{read_gray8, write_gray8};
use itk_rs::source::ImageSource;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: blur_png <input.png> <output.png> [sigma]");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let sigma: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2.0);

    // Read
    let img = read_gray8(input_path).expect("failed to read input PNG");
    println!(
        "Read {}x{} image",
        img.region.size.0[0],
        img.region.size.0[1]
    );

    // The PNG reader gives u8 pixels; Gaussian needs f32.
    // Cast u8 → f32, blur, cast f32 → u8.
    use itk_rs::filters::UnaryFilter;

    let as_f32 = UnaryFilter::new(img, |p: u8| p as f32);
    let blurred = GaussianFilter::new(as_f32, sigma);
    let as_u8 = UnaryFilter::new(blurred, |p: f32| p.round().clamp(0.0, 255.0) as u8);

    // Pull the full result (single pass, no streaming)
    let result = as_u8.generate_region(as_u8.largest_region());

    // Write
    write_gray8(&result, output_path).expect("failed to write output PNG");
    println!("Wrote blurred image to {output_path}");
}
