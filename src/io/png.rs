//! PNG image I/O. Analog to ITK's PNGImageIO.
//!
//! Supported pixel types:
//! - `u8`  — 8-bit grayscale
//! - `u16` — 16-bit grayscale
//! - `VecPixel<u8,  3>` — 8-bit RGB
//! - `VecPixel<u8,  4>` — 8-bit RGBA
//! - `VecPixel<u16, 3>` — 16-bit RGB
//! - `VecPixel<u16, 4>` — 16-bit RGBA

use std::fs::File;
use std::io::{self, BufWriter, ErrorKind};
use std::path::Path;

use png::{BitDepth, ColorType, Decoder, Encoder};

use crate::image::{Image, Region};
use crate::pixel::VecPixel;

// ---------------------------------------------------------------------------
// Public read API
// ---------------------------------------------------------------------------

/// Read an 8-bit grayscale PNG.
pub fn read_gray8(path: impl AsRef<Path>) -> io::Result<Image<u8, 2>> {
    let (bytes, w, h) = decode8(path, ColorType::Grayscale)?;
    Ok(make_image(bytes, w, h))
}

/// Read a 16-bit grayscale PNG.
pub fn read_gray16(path: impl AsRef<Path>) -> io::Result<Image<u16, 2>> {
    let (bytes, w, h) = decode16(path, ColorType::Grayscale)?;
    let pixels = bytes_to_u16_be(&bytes);
    Ok(make_image(pixels, w, h))
}

/// Read an 8-bit RGB PNG.
pub fn read_rgb8(path: impl AsRef<Path>) -> io::Result<Image<VecPixel<u8, 3>, 2>> {
    let (bytes, w, h) = decode8(path, ColorType::Rgb)?;
    let pixels = bytes.chunks_exact(3).map(|c| VecPixel([c[0], c[1], c[2]])).collect();
    Ok(make_image(pixels, w, h))
}

/// Read an 8-bit RGBA PNG.
pub fn read_rgba8(path: impl AsRef<Path>) -> io::Result<Image<VecPixel<u8, 4>, 2>> {
    let (bytes, w, h) = decode8(path, ColorType::Rgba)?;
    let pixels = bytes.chunks_exact(4).map(|c| VecPixel([c[0], c[1], c[2], c[3]])).collect();
    Ok(make_image(pixels, w, h))
}

/// Read a 16-bit RGB PNG.
pub fn read_rgb16(path: impl AsRef<Path>) -> io::Result<Image<VecPixel<u16, 3>, 2>> {
    let (bytes, w, h) = decode16(path, ColorType::Rgb)?;
    let pixels = bytes
        .chunks_exact(6)
        .map(|c| VecPixel([u16_be(c, 0), u16_be(c, 2), u16_be(c, 4)]))
        .collect();
    Ok(make_image(pixels, w, h))
}

/// Read a 16-bit RGBA PNG.
pub fn read_rgba16(path: impl AsRef<Path>) -> io::Result<Image<VecPixel<u16, 4>, 2>> {
    let (bytes, w, h) = decode16(path, ColorType::Rgba)?;
    let pixels = bytes
        .chunks_exact(8)
        .map(|c| VecPixel([u16_be(c, 0), u16_be(c, 2), u16_be(c, 4), u16_be(c, 6)]))
        .collect();
    Ok(make_image(pixels, w, h))
}

// ---------------------------------------------------------------------------
// Public write API
// ---------------------------------------------------------------------------

/// Write an 8-bit grayscale PNG.
pub fn write_gray8(image: &Image<u8, 2>, path: impl AsRef<Path>) -> io::Result<()> {
    let (w, h) = dims(image);
    let mut enc = make_encoder(path, w, h, ColorType::Grayscale, BitDepth::Eight)?;
    enc.write_image_data(&image.data).map_err(encode_err)
}

/// Write a 16-bit grayscale PNG.
pub fn write_gray16(image: &Image<u16, 2>, path: impl AsRef<Path>) -> io::Result<()> {
    let (w, h) = dims(image);
    let mut enc = make_encoder(path, w, h, ColorType::Grayscale, BitDepth::Sixteen)?;
    let bytes: Vec<u8> = image.data.iter().flat_map(|&v| v.to_be_bytes()).collect();
    enc.write_image_data(&bytes).map_err(encode_err)
}

/// Write an 8-bit RGB PNG.
pub fn write_rgb8(image: &Image<VecPixel<u8, 3>, 2>, path: impl AsRef<Path>) -> io::Result<()> {
    let (w, h) = dims(image);
    let mut enc = make_encoder(path, w, h, ColorType::Rgb, BitDepth::Eight)?;
    let bytes: Vec<u8> = image.data.iter().flat_map(|p| p.0).collect();
    enc.write_image_data(&bytes).map_err(encode_err)
}

/// Write an 8-bit RGBA PNG.
pub fn write_rgba8(image: &Image<VecPixel<u8, 4>, 2>, path: impl AsRef<Path>) -> io::Result<()> {
    let (w, h) = dims(image);
    let mut enc = make_encoder(path, w, h, ColorType::Rgba, BitDepth::Eight)?;
    let bytes: Vec<u8> = image.data.iter().flat_map(|p| p.0).collect();
    enc.write_image_data(&bytes).map_err(encode_err)
}

/// Write a 16-bit RGB PNG.
pub fn write_rgb16(
    image: &Image<VecPixel<u16, 3>, 2>,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let (w, h) = dims(image);
    let mut enc = make_encoder(path, w, h, ColorType::Rgb, BitDepth::Sixteen)?;
    let bytes: Vec<u8> =
        image.data.iter().flat_map(|p| p.0.iter().flat_map(|v| v.to_be_bytes())).collect();
    enc.write_image_data(&bytes).map_err(encode_err)
}

/// Write a 16-bit RGBA PNG.
pub fn write_rgba16(
    image: &Image<VecPixel<u16, 4>, 2>,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let (w, h) = dims(image);
    let mut enc = make_encoder(path, w, h, ColorType::Rgba, BitDepth::Sixteen)?;
    let bytes: Vec<u8> =
        image.data.iter().flat_map(|p| p.0.iter().flat_map(|v| v.to_be_bytes())).collect();
    enc.write_image_data(&bytes).map_err(encode_err)
}

// ---------------------------------------------------------------------------
// Internal decode helpers
// ---------------------------------------------------------------------------

/// Decode to 8-bit output, converting and expanding as needed.
/// Returns (raw bytes, width, height). Each pixel channel is one byte.
fn decode8(path: impl AsRef<Path>, want_color: ColorType) -> io::Result<(Vec<u8>, usize, usize)> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(file);
    decoder.set_transformations(
        png::Transformations::normalize_to_color8()
            | png::Transformations::EXPAND
            | png::Transformations::STRIP_16,
    );
    let mut reader = decoder.read_info().map_err(decode_err)?;
    let (w, h, got_color) = {
        let info = reader.info();
        (info.width as usize, info.height as usize, info.color_type)
    };
    if got_color != want_color {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("PNG color type mismatch: expected {:?}, got {:?}", want_color, got_color),
        ));
    }
    let mut buf = vec![0u8; reader.output_buffer_size()];
    reader.next_frame(&mut buf).map_err(decode_err)?;
    Ok((buf, w, h))
}

/// Decode keeping full 16-bit precision (no STRIP_16 transform).
/// Returns (raw bytes in big-endian, width, height). Each channel is 2 bytes.
fn decode16(path: impl AsRef<Path>, want_color: ColorType) -> io::Result<(Vec<u8>, usize, usize)> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(file);
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().map_err(decode_err)?;
    let (w, h, got_color, got_depth) = {
        let info = reader.info();
        (info.width as usize, info.height as usize, info.color_type, info.bit_depth)
    };
    if got_color != want_color {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("PNG color type mismatch: expected {:?}, got {:?}", want_color, got_color),
        ));
    }
    if got_depth != BitDepth::Sixteen {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("expected 16-bit PNG, got {:?}", got_depth),
        ));
    }
    let mut buf = vec![0u8; reader.output_buffer_size()];
    reader.next_frame(&mut buf).map_err(decode_err)?;
    Ok((buf, w, h))
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

fn make_encoder(
    path: impl AsRef<Path>,
    w: usize,
    h: usize,
    color: ColorType,
    depth: BitDepth,
) -> io::Result<png::Writer<BufWriter<File>>> {
    let file = File::create(path)?;
    let mut enc = Encoder::new(BufWriter::new(file), w as u32, h as u32);
    enc.set_color(color);
    enc.set_depth(depth);
    enc.write_header().map_err(encode_err)
}

fn make_image<P: Default + Clone>(data: Vec<P>, w: usize, h: usize) -> Image<P, 2> {
    Image {
        region: Region::new([0, 0], [w, h]),
        spacing: [1.0, 1.0],
        origin: [0.0, 0.0],
        data,
    }
}

fn dims<P, const D: usize>(image: &Image<P, D>) -> (usize, usize) {
    (image.region.size.0[0], image.region.size.0[1])
}

fn bytes_to_u16_be(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|c| u16::from_be_bytes([c[0], c[1]])).collect()
}

fn u16_be(bytes: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes([bytes[offset], bytes[offset + 1]])
}

fn decode_err(e: png::DecodingError) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, e.to_string())
}

fn encode_err(e: png::EncodingError) -> io::Error {
    io::Error::new(ErrorKind::Other, e.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Index;
    use std::env::temp_dir;

    fn tmp(name: &str) -> std::path::PathBuf {
        temp_dir().join(name)
    }

    #[test]
    fn roundtrip_gray8() {
        let mut img = Image::<u8, 2>::allocate(
            Region::new([0, 0], [4, 4]),
            [1.0, 1.0],
            [0.0, 0.0],
            0u8,
        );
        img.set_pixel(Index([1, 2]), 200u8);
        let path = tmp("nitk_test_gray8.png");
        write_gray8(&img, &path).unwrap();
        let loaded = read_gray8(&path).unwrap();
        assert_eq!(loaded.get_pixel(Index([1, 2])), 200u8);
        assert_eq!(loaded.get_pixel(Index([0, 0])), 0u8);
        assert_eq!(loaded.region.size.0, [4, 4]);
    }

    #[test]
    fn roundtrip_rgb8() {
        let mut img = Image::<VecPixel<u8, 3>, 2>::allocate(
            Region::new([0, 0], [3, 3]),
            [1.0, 1.0],
            [0.0, 0.0],
            VecPixel::default(),
        );
        img.set_pixel(Index([0, 0]), VecPixel([10, 20, 30]));
        let path = tmp("nitk_test_rgb8.png");
        write_rgb8(&img, &path).unwrap();
        let loaded = read_rgb8(&path).unwrap();
        assert_eq!(loaded.get_pixel(Index([0, 0])), VecPixel([10, 20, 30]));
    }

    #[test]
    fn roundtrip_rgba8() {
        let mut img = Image::<VecPixel<u8, 4>, 2>::allocate(
            Region::new([0, 0], [2, 2]),
            [1.0, 1.0],
            [0.0, 0.0],
            VecPixel::default(),
        );
        img.set_pixel(Index([1, 1]), VecPixel([255, 128, 64, 32]));
        let path = tmp("nitk_test_rgba8.png");
        write_rgba8(&img, &path).unwrap();
        let loaded = read_rgba8(&path).unwrap();
        assert_eq!(loaded.get_pixel(Index([1, 1])), VecPixel([255, 128, 64, 32]));
    }

    #[test]
    fn roundtrip_gray16() {
        let mut img = Image::<u16, 2>::allocate(
            Region::new([0, 0], [4, 4]),
            [1.0, 1.0],
            [0.0, 0.0],
            0u16,
        );
        img.set_pixel(Index([2, 3]), 60000u16);
        let path = tmp("nitk_test_gray16.png");
        write_gray16(&img, &path).unwrap();
        let loaded = read_gray16(&path).unwrap();
        assert_eq!(loaded.get_pixel(Index([2, 3])), 60000u16);
    }
}
