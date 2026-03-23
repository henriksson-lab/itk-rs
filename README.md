# itk-rs — Native ITK in Rust

A Rust translation of [ITK](https://itk.org/) (Insight Segmentation and Registration Toolkit), the C++ medical image processing library.

This is not the authorative implementation. New features should be directed to the original ITK C++ implementation.

The code has been automatically translated from C++. Further testing is necessary

## Why

ITK is the gold standard for medical image processing but is written in heavily templated C++. This project translates ITK's core patterns into idiomatic Rust, gaining memory safety, modern tooling, and no runtime without sacrificing the compile-time genericity that makes ITK fast.

Compilation of this crate is significantly faster than SimpleITK-rs + it's C++
dependency. This is likely because SimpleITK is designed for Python and is forced to
compile all versions of all functions ahead of time, while this package
should - based on generics - should only compile as much as needed for 
downstream applications.

Unlike SimpleITK-rs, this package is also feature complete, as callback functions
across the C++/Rust layer are otherwise hard to support.


## Design

### C++ templates → Rust const generics + traits

ITK's two core template parameters map directly to Rust:

| ITK C++ | Rust |
|---|---|
| `template <typename TPixel, unsigned int VDim>` | `Image<P, const D: usize>` |
| `template <unsigned int VDim>` | `Region<const D: usize>` |
| Functor class with `operator()` | `Fn(P) -> Q` closure |
| `typename TImage::PixelType` | Associated type in trait |
| Concept checking macros | Trait bounds (`where P: NumericPixel`) |
| `SmartPointer<T>` | Owned struct or `Arc<T>` |

### The pipeline model

ITK uses a three-phase lazy pipeline for out-of-core (streaming) processing. `itk-rs` replicates this with a single trait:

```rust
pub trait ImageSource<P, const D: usize> {
    // Phase 1 — metadata only, no allocation
    fn largest_region(&self) -> Region<D>;
    fn spacing(&self) -> [f64; D];
    fn origin(&self) -> [f64; D];

    // Phase 2 — what input do I need to produce this output?
    fn input_region_for_output(&self, output_region: &Region<D>) -> Region<D> {
        *output_region  // default: pixel-wise, identity
    }

    // Phase 3 — allocate and compute
    fn generate_region(&self, requested: Region<D>) -> Image<P, D>;
}
```

`Image<P, D>` implements `ImageSource` directly — it is both a buffer and a pipeline node. Filters own their upstream source via composition:

```rust
let reader: Image<f32, 3> = /* load from file */;
let smooth = GaussianFilter { source: reader, sigma: 1.5 };
let result = smooth.generate_region(smooth.largest_region());
```

### Out-of-core streaming

Because filters only compute a `RequestedRegion`, streaming over large images is a free function:

```rust
let result = stream_full(&smooth, /*pieces=*/ 8);
```

Each piece propagates back through the pipeline independently, so memory usage is bounded by `image_size / pieces` regardless of the filter chain depth.

### Parallelism

`generate_region()` implementations use [rayon](https://github.com/rayon-rs/rayon) for data parallelism, analogous to ITK's `DynamicThreadedGenerateData()`.

## Pixel types

```rust
// Scalar pixels — f32, f64, u8, i16, ...
let img: Image<f32, 2> = Image::allocate(region, spacing, origin, 0.0);

// Vector pixels — analog to itk::Vector<T, N>
let rgb: Image<VecPixel<f32, 3>, 2> = Image::allocate(region, spacing, origin, VecPixel::default());
```

Arithmetic filters (Gaussian, Mean) require the `NumericPixel` trait bound, which both scalar and vector pixel types satisfy.

## Filters

| Filter | ITK equivalent | Description |
|---|---|---|
| `GaussianFilter` | `DiscreteGaussianImageFilter` | Separable Gaussian smoothing |
| `UnaryFilter<F>` | `UnaryFunctorImageFilter` | Per-pixel transform via closure |

For the full list of implemented filters, see [FEATURES.md](FEATURES.md).

## Status

This is an early proof-of-concept establishing the core patterns. The immediate goal is to validate that:

1. `Image<P, const D: usize>` with const generics covers ITK's two-parameter template
2. The `ImageSource` trait faithfully replicates ITK's three-phase pipeline
3. Streaming produces identical results to full-image processing
4. Both scalar and vector pixel types work through the same filter code

Once the foundation is solid, further ITK filters and IO classes can be added module by module, following the same patterns.

## Building

```sh
cargo build
cargo test
```

Requires Rust 1.65+ (stable const generics).

## License

Apache 2.0, matching ITK's license.
