# ITK Feature Implementation Status

Legend: ✅ Done | 🚧 In progress | ⬜ Not started

---

## Core Pipeline Infrastructure

| Feature | Status | Notes |
|---|---|---|
| `Image<P, const D>` | ✅ | `src/image.rs` |
| `Region<D>`, `Index<D>`, `Size<D>` | ✅ | `src/image.rs` |
| `ImageSource` trait (3-phase pipeline) | ✅ | `src/source.rs` |
| `stream_full()` (out-of-core streaming) | ✅ | `src/source.rs` |
| `split_region()` (axis-aligned tiling) | ✅ | `src/region_splitter.rs` |
| `Pixel` trait | ✅ | `src/pixel.rs` |
| `NumericPixel` trait | ✅ | `src/pixel.rs` |
| `VecPixel<T, N>` (vector pixel) | ✅ | `src/pixel.rs` |
| `UnaryFilter<S, F>` (functor filter) | ✅ | `src/filters/mod.rs` |

---

## Filtering

### Smoothing

| Filter | ITK class | Status |
|---|---|---|
| Gaussian (separable) | `DiscreteGaussianImageFilter` | ✅ `src/filters/gaussian.rs` |
| Recursive Gaussian | `RecursiveGaussianImageFilter` | ✅ `src/filters/recursive_gaussian.rs` |
| Smoothing Recursive Gaussian | `SmoothingRecursiveGaussianImageFilter` | ✅ `src/filters/recursive_gaussian.rs` |
| Mean | `MeanImageFilter` | ✅ `src/filters/mean.rs` |
| Median | `MedianImageFilter` | ✅ `src/filters/median.rs` |
| Box Mean | `BoxMeanImageFilter` | ✅ `src/filters/mean.rs` |
| Box Sigma | `BoxSigmaImageFilter` | ✅ `src/filters/mean.rs` |
| Binomial Blur | `BinomialBlurImageFilter` | ✅ `src/filters/binomial_blur.rs` |
| FFT Discrete Gaussian | `FFTDiscreteGaussianImageFilter` | ✅ `src/filters/fft.rs` |

### Anisotropic Smoothing

| Filter | ITK class | Status |
|---|---|---|
| Gradient Anisotropic Diffusion | `GradientAnisotropicDiffusionImageFilter` | ✅ `src/filters/anisotropic_diffusion.rs` |
| Curvature Anisotropic Diffusion | `CurvatureAnisotropicDiffusionImageFilter` | ✅ `src/filters/anisotropic_diffusion.rs` (D=2) |
| Vector Gradient Anisotropic Diffusion | `VectorGradientAnisotropicDiffusionImageFilter` | ✅ `src/filters/anisotropic_diffusion.rs` |
| Vector Curvature Anisotropic Diffusion | `VectorCurvatureAnisotropicDiffusionImageFilter` | ✅ `src/filters/anisotropic_diffusion.rs` |

### Denoising

| Filter | ITK class | Status |
|---|---|---|
| Patch-Based Denoising | `PatchBasedDenoisingImageFilter` | ⬜ |

### Image Intensity

| Filter | ITK class | Status |
|---|---|---|
| Add | `AddImageFilter` | ✅ `intensity::add_images` |
| Subtract | `SubtractImageFilter` | ✅ `intensity::subtract_images` |
| Multiply | `MultiplyImageFilter` | ✅ `intensity::multiply_images` |
| Divide | `DivideImageFilter` | ✅ `intensity::divide_images` |
| Abs | `AbsImageFilter` | ✅ `intensity::abs_filter` |
| Square | `SquareImageFilter` | ✅ `intensity::square_filter` |
| Sqrt | `SqrtImageFilter` | ✅ `intensity::sqrt_filter` |
| Exp | `ExpImageFilter` | ✅ `intensity::exp_filter` |
| Log | `LogImageFilter` | ✅ `intensity::log_filter` |
| Log10 | `Log10ImageFilter` | ✅ `intensity::log10_filter` |
| Sin | `SinImageFilter` | ✅ `intensity::sin_filter` |
| Cos | `CosImageFilter` | ✅ `intensity::cos_filter` |
| Atan | `AtanImageFilter` | ✅ `intensity::atan_filter` |
| Atan2 | `Atan2ImageFilter` | ✅ `intensity::atan2_images` |
| Pow | `PowImageFilter` | ✅ `intensity::PowFilter` |
| Sigmoid | `SigmoidImageFilter` | ✅ `intensity::SigmoidFilter` |
| Invert Intensity | `InvertIntensityImageFilter` | ✅ `intensity::InvertIntensityFilter` |
| Rescale Intensity | `RescaleIntensityImageFilter` | ✅ `intensity::RescaleIntensityFilter` |
| Shift Scale | `ShiftScaleImageFilter` | ✅ `intensity::ShiftScaleFilter` |
| Clamp | `ClampImageFilter` | ✅ `intensity::ClampFilter` |
| Normalize | `NormalizeImageFilter` | ✅ `intensity::NormalizeFilter` |
| Normalize to Constant | `NormalizeToConstantImageFilter` | ✅ `intensity::NormalizeToConstantFilter` |
| Intensity Windowing | `IntensityWindowingImageFilter` | ✅ `intensity::IntensityWindowingFilter` |
| Histogram Matching | `HistogramMatchingImageFilter` | ✅ `intensity::HistogramMatchingFilter` |
| Mask | `MaskImageFilter` | ✅ `intensity::MaskFilter` |
| Mask Negated | `MaskNegatedImageFilter` | ✅ `intensity::MaskNegatedFilter` |
| Maximum (pixelwise) | `MaximumImageFilter` | ✅ `intensity::maximum_images` |
| Minimum (pixelwise) | `MinimumImageFilter` | ✅ `intensity::minimum_images` |
| And | `AndImageFilter` | ✅ `intensity::and_images` |
| Or | `OrImageFilter` | ✅ `intensity::or_images` |
| Xor | `XorImageFilter` | ✅ `intensity::xor_images` |
| Not | `NotImageFilter` | ✅ `intensity::NotFilter` |
| Weighted Add | `WeightedAddImageFilter` | ✅ `intensity::WeightedAddFilter` |
| N-ary Add | `NaryAddImageFilter` | ✅ `intensity::NaryAddFilter` |
| N-ary Maximum | `NaryMaximumImageFilter` | ✅ `intensity::NaryMaximumFilter` |
| Constrained Value Addition | `ConstrainedValueAdditionImageFilter` | ✅ `intensity::ConstrainedValueAdditionFilter` |
| Constrained Value Difference | `ConstrainedValueDifferenceImageFilter` | ✅ `intensity::ConstrainedValueDifferenceFilter` |
| Bounded Reciprocal | `BoundedReciprocalImageFilter` | ✅ `intensity::bounded_reciprocal_filter` |
| Modulus | `ModulusImageFilter` | ✅ `intensity::ModulusFilter` |
| Round | `RoundImageFilter` | ✅ `intensity::round_filter` |
| Cast | `CastImageFilter` | ✅ (via `UnaryFilter` with `from_f64`/`to_f64`) |
| Complex to Real | `ComplexToRealImageFilter` | ✅ `intensity::complex_to_real` |
| Complex to Imaginary | `ComplexToImaginaryImageFilter` | ✅ `intensity::complex_to_imaginary` |
| Complex to Modulus | `ComplexToModulusImageFilter` | ✅ `intensity::complex_to_modulus` |
| Complex to Phase | `ComplexToPhaseImageFilter` | ✅ `intensity::complex_to_phase` |
| Magnitude and Phase to Complex | `MagnitudeAndPhaseToComplexImageFilter` | ⬜ |
| Vector Magnitude | `VectorMagnitudeImageFilter` | ✅ `intensity::VectorMagnitudeFilter` |
| Vector Index Selection Cast | `VectorIndexSelectionCastImageFilter` | ✅ `intensity::VectorIndexSelectionFilter` |
| Compose Image | `ComposeImageFilter` | ✅ `intensity::Compose2Filter`, `Compose3Filter` |
| Scalar to RGB Pixel | `ScalarToRGBPixelFunctor` | ⬜ |
| Symmetric Eigen Analysis | `SymmetricEigenAnalysisImageFilter` | ✅ `src/filters/edges.rs` |

### Thresholding

| Filter | ITK class | Status |
|---|---|---|
| Binary Threshold | `BinaryThresholdImageFilter` | ✅ `threshold::BinaryThresholdFilter` |
| Threshold (in-place) | `ThresholdImageFilter` | ✅ `threshold::ThresholdFilter` |
| Otsu Threshold | `OtsuThresholdImageFilter` | ✅ `threshold::OtsuThresholdFilter` |
| Otsu Multiple Thresholds | `OtsuMultipleThresholdsImageFilter` | ✅ `threshold::otsu_multiple_thresholds` |
| Huang Threshold | `HuangThresholdImageFilter` | ✅ `threshold::huang_threshold_filter` |
| Li Threshold | `LiThresholdImageFilter` | ✅ `threshold::li_threshold_filter` |
| IsoData Threshold | `IsoDataThresholdImageFilter` | ✅ `threshold::iso_data_threshold_filter` |
| MaxEntropy Threshold | `MaximumEntropyThresholdImageFilter` | ✅ `threshold::max_entropy_threshold_filter` |
| Moments Threshold | `MomentsThresholdImageFilter` | ✅ `threshold::moments_threshold_filter` |
| Triangle Threshold | `TriangleThresholdImageFilter` | ✅ `threshold::triangle_threshold_filter` |
| Yen Threshold | `YenThresholdImageFilter` | ✅ `threshold::yen_threshold_filter` |
| Renyi Entropy Threshold | `RenyiEntropyThresholdImageFilter` | ✅ `threshold::renyi_entropy_threshold_filter` |
| Shanbhag Threshold | `ShanbhagThresholdImageFilter` | ✅ `threshold::shanbhag_threshold_filter` |
| Kittler-Illingworth Threshold | `KittlerIllingworthThresholdImageFilter` | ✅ `threshold::kittler_illingworth_threshold_filter` |
| Intermodes Threshold | `IntermodesThresholdImageFilter` | ✅ `threshold::intermodes_threshold_filter` |
| Kappa-Sigma Threshold | `KappaSigmaThresholdImageFilter` | ✅ `threshold::kappa_sigma_threshold_filter` |

### Image Features & Edges

| Filter | ITK class | Status |
|---|---|---|
| Canny Edge Detection | `CannyEdgeDetectionImageFilter` | ✅ `edges::CannyEdgeDetectionFilter` (D=2) |
| Sobel Edge Detection | `SobelEdgeDetectionImageFilter` | ✅ `edges::SobelFilter` (D=2) |
| Laplacian | `LaplacianImageFilter` | ✅ `edges::LaplacianFilter` |
| Laplacian Recursive Gaussian | `LaplacianRecursiveGaussianImageFilter` | ✅ `edges::LaplacianRecursiveGaussianFilter` |
| Laplacian Sharpening | `LaplacianSharpeningImageFilter` | ✅ `edges::LaplacianSharpeningFilter` |
| Zero Crossing | `ZeroCrossingImageFilter` | ✅ `edges::ZeroCrossingFilter` |
| Zero Crossing Based Edge Detection | `ZeroCrossingBasedEdgeDetectionImageFilter` | ✅ `edges::ZeroCrossingBasedEdgeDetectionFilter` |
| Derivative | `DerivativeImageFilter` | ✅ `edges::DerivativeFilter` |
| Discrete Gaussian Derivative | `DiscreteGaussianDerivativeImageFilter` | ✅ `edges::DiscreteGaussianDerivativeFilter` |
| Hessian Recursive Gaussian | `HessianRecursiveGaussianImageFilter` | ✅ `edges::HessianRecursiveGaussianFilter` (D=2) |
| Hessian to Objectness Measure | `HessianToObjectnessMeasureImageFilter` | ✅ `src/filters/edges.rs` |
| Multi-Scale Hessian Measure | `MultiScaleHessianBasedMeasureImageFilter` | ✅ `src/filters/edges.rs` |
| Hessian 3D to Vesselness | `Hessian3DToVesselnessMeasureImageFilter` | ✅ `src/filters/edges.rs` |
| Bilateral | `BilateralImageFilter` | ✅ `src/filters/bilateral.rs` |
| Unsharp Mask | `UnsharpMaskImageFilter` | ✅ `edges::UnsharpMaskFilter` |
| Gradient Vector Flow | `GradientVectorFlowImageFilter` | ✅ `src/filters/edges.rs` |
| Hough Transform 2D Circles | `HoughTransform2DCirclesImageFilter` | ✅ `edges::HoughTransform2DCirclesFilter` |
| Hough Transform 2D Lines | `HoughTransform2DLinesImageFilter` | ✅ `edges::HoughTransform2DLinesFilter` |
| Simple Contour Extractor | `SimpleContourExtractorImageFilter` | ✅ `edges::SimpleContourExtractorFilter` |

### Gradients

| Filter | ITK class | Status |
|---|---|---|
| Gradient | `GradientImageFilter` | ✅ `edges::GradientFilter` |
| Gradient Magnitude | `GradientMagnitudeImageFilter` | ✅ `edges::GradientMagnitudeFilter` |
| Gradient Magnitude Recursive Gaussian | `GradientMagnitudeRecursiveGaussianImageFilter` | ✅ `edges::GradientMagnitudeRecursiveGaussianFilter` |
| Gradient Recursive Gaussian | `GradientRecursiveGaussianImageFilter` | ✅ `edges::GradientRecursiveGaussianFilter` |
| Difference of Gaussians Gradient | `DifferenceOfGaussiansGradientImageFilter` | ✅ `edges::DifferenceOfGaussiansFilter` |
| Vector Gradient Magnitude | `VectorGradientMagnitudeImageFilter` | ✅ `edges::VectorGradientMagnitudeFilter` |

### Mathematical Morphology

| Filter | ITK class | Status |
|---|---|---|
| Grayscale Dilate | `GrayscaleDilateImageFilter` | ✅ `morphology::GrayscaleDilateFilter` |
| Grayscale Erode | `GrayscaleErodeImageFilter` | ✅ `morphology::GrayscaleErodeFilter` |
| Grayscale Morphological Opening | `GrayscaleMorphologicalOpeningImageFilter` | ✅ `morphology::GrayscaleOpenFilter` |
| Grayscale Morphological Closing | `GrayscaleMorphologicalClosingImageFilter` | ✅ `morphology::GrayscaleCloseFilter` |
| Morphological Gradient | `MorphologicalGradientImageFilter` | ✅ `morphology::MorphologicalGradientFilter` |
| White Top Hat | `WhiteTopHatImageFilter` | ✅ `morphology::WhiteTopHatFilter` |
| Black Top Hat | `BlackTopHatImageFilter` | ✅ `morphology::BlackTopHatFilter` |
| Double Threshold | `DoubleThresholdImageFilter` | ✅ `morphology::DoubleThresholdFilter` |
| H-Maxima | `HMaximaImageFilter` | ✅ `morphology::HMaximaFilter` |
| H-Minima | `HMinimaImageFilter` | ✅ `morphology::HMinimaFilter` |
| H-Concave | `HConcaveImageFilter` | ✅ `morphology::HConcaveFilter` |
| H-Convex | `HConvexImageFilter` | ✅ `morphology::HConvexFilter` |
| Regional Maxima | `RegionalMaximaImageFilter` | ✅ `morphology::RegionalMaximaFilter` |
| Regional Minima | `RegionalMinimaImageFilter` | ✅ `morphology::RegionalMinimaFilter` |
| Grayscale Geodesic Dilate | `GrayscaleGeodesicDilateImageFilter` | ✅ `morphology::GrayscaleGeodesicDilateFilter` |
| Grayscale Geodesic Erode | `GrayscaleGeodesicErodeImageFilter` | ✅ `morphology::GrayscaleGeodesicErodeFilter` |
| Closing by Reconstruction | `ClosingByReconstructionImageFilter` | ✅ `morphology::ClosingByReconstructionFilter` |
| Opening by Reconstruction | `OpeningByReconstructionImageFilter` | ✅ `morphology::OpeningByReconstructionFilter` |
| Grayscale Connected Opening | `GrayscaleConnectedOpeningImageFilter` | ✅ `morphology::GrayscaleConnectedOpeningFilter` |
| Grayscale Connected Closing | `GrayscaleConnectedClosingImageFilter` | ✅ `morphology::GrayscaleConnectedClosingFilter` |
| Grayscale Fillhole | `GrayscaleFillholeImageFilter` | ✅ `morphology::GrayscaleFillholeFilter` |
| Grayscale Grind Peak | `GrayscaleGrindPeakImageFilter` | ✅ `morphology::GrayscaleGrindPeakFilter` |
| Rank | `RankImageFilter` | ✅ `morphology::RankFilter` |
| Reconstruction by Dilation | `ReconstructionByDilationImageFilter` | ✅ `morphology::ReconstructionByDilationFilter` |
| Reconstruction by Erosion | `ReconstructionByErosionImageFilter` | ✅ `morphology::ReconstructionByErosionFilter` |

### Binary Morphology

| Filter | ITK class | Status |
|---|---|---|
| Binary Dilate | `BinaryDilateImageFilter` | ✅ `morphology::BinaryDilateFilter` |
| Binary Erode | `BinaryErodeImageFilter` | ✅ `morphology::BinaryErodeFilter` |
| Binary Morphological Opening | `BinaryMorphologicalOpeningImageFilter` | ✅ `morphology::BinaryOpenFilter` |
| Binary Morphological Closing | `BinaryMorphologicalClosingImageFilter` | ✅ `morphology::BinaryCloseFilter` |
| Binary Opening by Reconstruction | `BinaryOpeningByReconstructionImageFilter` | ✅ `morphology::BinaryOpeningByReconstructionFilter` |
| Binary Closing by Reconstruction | `BinaryClosingByReconstructionImageFilter` | ✅ `morphology::BinaryClosingByReconstructionFilter` |
| Binary Thinning | `BinaryThinningImageFilter` | ✅ `morphology::BinaryThinningFilter` (D=2) |
| Binary Pruning | `BinaryPruningImageFilter` | ✅ `morphology::BinaryPruningFilter` |

### Convolution & Frequency Domain

| Filter | ITK class | Status |
|---|---|---|
| Convolution | `ConvolutionImageFilter` | ✅ `convolution::ConvolutionFilter` |
| FFT Convolution | `FFTConvolutionImageFilter` | ✅ `src/filters/fft.rs` |
| Normalized Correlation | `NormalizedCorrelationImageFilter` | ✅ `convolution::NormalizedCorrelationFilter` |
| FFT Normalized Correlation | `FFTNormalizedCorrelationImageFilter` | ✅ `src/filters/fft.rs` |
| Masked FFT Normalized Correlation | `MaskedFFTNormalizedCorrelationImageFilter` | ✅ `src/filters/fft.rs` |
| Forward FFT | `ForwardFFTImageFilter` | ✅ `src/filters/fft.rs` |
| Inverse FFT | `InverseFFTImageFilter` | ✅ `src/filters/fft.rs` |
| FFT Shift | `FFTShiftImageFilter` | ✅ `src/filters/fft.rs` |
| FFT Pad | `FFTPadImageFilter` | ✅ `src/filters/fft.rs` |
| Frequency Band Filter | `FrequencyBandImageFilter` | ✅ `src/filters/fft.rs` |

### Deconvolution

| Filter | ITK class | Status |
|---|---|---|
| Inverse Deconvolution | `InverseDeconvolutionImageFilter` | ✅ `src/filters/deconvolution.rs` |
| Wiener Deconvolution | `WienerDeconvolutionImageFilter` | ✅ `src/filters/deconvolution.rs` |
| Tikhonov Deconvolution | `TikhonovDeconvolutionImageFilter` | ✅ `src/filters/deconvolution.rs` |
| Richardson-Lucy Deconvolution | `RichardsonLucyDeconvolutionImageFilter` | ✅ `src/filters/deconvolution.rs` |
| Landweber Deconvolution | `LandweberDeconvolutionImageFilter` | ✅ `src/filters/deconvolution.rs` |
| Projected Landweber Deconvolution | `ProjectedLandweberDeconvolutionImageFilter` | ✅ `src/filters/deconvolution.rs` |

### Distance Maps

| Filter | ITK class | Status |
|---|---|---|
| Signed Maurer Distance Map | `SignedMaurerDistanceMapImageFilter` | ✅ `distance::SignedMaurerDistanceMapFilter` |
| Danielsson Distance Map | `DanielssonDistanceMapImageFilter` | ✅ `distance::DanielssonDistanceMapFilter` |
| Signed Danielsson Distance Map | `SignedDanielssonDistanceMapImageFilter` | ✅ `distance::SignedDanielssonDistanceMapFilter` |
| Fast Chamfer Distance | `FastChamferDistanceImageFilter` | ✅ `distance::FastChamferDistanceFilter` |
| IsoContour Distance | `IsoContourDistanceImageFilter` | ✅ `distance::IsoContourDistanceFilter` |
| Approximate Signed Distance Map | `ApproximateSignedDistanceMapImageFilter` | ✅ `distance::ApproximateSignedDistanceMapFilter` |
| Hausdorff Distance | `HausdorffDistanceImageFilter` | ✅ `distance::HausdorffDistanceFilter` |
| Directed Hausdorff Distance | `DirectedHausdorffDistanceImageFilter` | ✅ `distance::DirectedHausdorffDistanceFilter` |
| Contour Mean Distance | `ContourMeanDistanceImageFilter` | ✅ `distance::ContourMeanDistanceFilter` |

### Fast Marching

| Filter | ITK class | Status |
|---|---|---|
| Fast Marching Image Filter | `FastMarchingImageFilter` | ✅ `fast_marching::FastMarchingFilter` (D=2) |
| Fast Marching Upwind Gradient | `FastMarchingUpwindGradientImageFilter` | ✅ `fast_marching::FastMarchingUpwindGradientFilter` (D=2) |
| Fast Marching Extension | `FastMarchingExtensionImageFilter` | ✅ `distance::FastMarchingExtensionFilter` |

### Image Grid / Resampling

| Filter | ITK class | Status |
|---|---|---|
| Resample Image | `ResampleImageFilter` | ✅ `spatial::ResampleImageFilter2D`, `ResampleImageFilter3D` |
| Warp Image | `WarpImageFilter` | ✅ `spatial::WarpImageFilter` |
| Shrink | `ShrinkImageFilter` | ✅ `spatial::ShrinkImageFilter` |
| Expand | `ExpandImageFilter` | ✅ `spatial::ExpandImageFilter` |
| BSpline Downsample | `BSplineDownsampleImageFilter` | ✅ `spatial::BSplineDownsampleFilter` |
| BSpline Upsample | `BSplineUpsampleImageFilter` | ✅ `spatial::BSplineUpsampleFilter` |
| Crop | `CropImageFilter` | ✅ `spatial::CropImageFilter` |
| Region of Interest | `RegionOfInterestImageFilter` | ✅ `spatial::RegionOfInterestFilterD` |
| Flip | `FlipImageFilter` | ✅ `spatial::FlipImageFilter` |
| Permute Axes | `PermuteAxesImageFilter` | ✅ `spatial::PermuteAxesFilter` |
| Constant Pad | `ConstantPadImageFilter` | ✅ `spatial::ConstantPadFilter` |
| Mirror Pad | `MirrorPadImageFilter` | ✅ `spatial::MirrorPadFilter` |
| Wrap Pad | `WrapPadImageFilter` | ✅ `spatial::WrapPadFilter` |
| Zero Flux Neumann Pad | `ZeroFluxNeumannPadImageFilter` | ✅ `spatial::ZeroFluxNeumannPadFilter` |
| Tile | `TileImageFilter` | ✅ `spatial::TileImageFilter` |
| Paste | `PasteImageFilter` | ✅ `spatial::PasteFilter` |
| Change Information | `ChangeInformationImageFilter` | ✅ `spatial::ChangeInformationFilter` |
| Orient Image | `OrientImageFilter` | ✅ `spatial::OrientImageFilter` |
| Cyclic Shift | `CyclicShiftImageFilter` | ✅ `spatial::CyclicShiftFilter` |
| Bin Shrink | `BinShrinkImageFilter` | ✅ `spatial::BinShrinkImageFilter` |
| Slice by Slice | `SliceBySliceImageFilter` | ✅ `spatial::SliceBySliceFilter` |
| Interpolate Image | `InterpolateImageFilter` | ✅ `spatial::InterpolateImageFilter` |
| BSpline Scattered Data Point Set to Image | `BSplineScatteredDataPointSetToImageFilter` | ✅ `spatial::BSplineScatteredDataFilter` |

### Image Statistics

| Filter | ITK class | Status |
|---|---|---|
| Statistics | `StatisticsImageFilter` | ✅ `statistics::StatisticsImageFilter` |
| Label Statistics | `LabelStatisticsImageFilter` | ✅ `statistics::LabelStatisticsFilter` |
| Minimum Maximum | `MinimumMaximumImageFilter` | ✅ `statistics::MinimumMaximumImageFilter` |
| Image Moments | `ImageMomentsCalculator` | ✅ `statistics::ImageMomentsCalculator` |
| Image PCA Shape Model | `ImagePCAShapeModelEstimator` | ⬜ |
| Accumulate | `AccumulateImageFilter` | ✅ `statistics::AccumulateFilter` |
| Max Projection | `MaximumProjectionImageFilter` | ✅ `statistics::MaxProjectionFilter` |
| Min Projection | `MinimumProjectionImageFilter` | ✅ `statistics::MinProjectionFilter` |
| Mean Projection | `MeanProjectionImageFilter` | ✅ `statistics::MeanProjectionFilter` |
| Sum Projection | `SumProjectionImageFilter` | ✅ `statistics::SumProjectionFilter` |
| Median Projection | `MedianProjectionImageFilter` | ✅ `statistics::MedianProjectionFilter` |
| StdDev Projection | `StandardDeviationProjectionImageFilter` | ✅ `statistics::StdDevProjectionFilter` |
| Adaptive Histogram Equalization | `AdaptiveHistogramEqualizationImageFilter` | ✅ `statistics::AdaptiveHistogramEqualizationFilter` |
| Label Overlap Measures | `LabelOverlapMeasuresImageFilter` | ✅ `statistics::LabelOverlapMeasuresFilter` |
| STAPLE | `STAPLEImageFilter` | ✅ `statistics::STAPLEFilter` |
| Similarity Index | `SimilarityIndexImageFilter` | ✅ `statistics::SimilarityIndexFilter` |
| Checker Board | `CheckerBoardImageFilter` | ✅ `spatial::CheckerBoardFilter` |

### Image Noise Simulation

| Filter | ITK class | Status |
|---|---|---|
| Additive Gaussian Noise | `AdditiveGaussianNoiseImageFilter` | ✅ `noise::AdditiveGaussianNoiseFilter` |
| Salt and Pepper Noise | `SaltAndPepperNoiseImageFilter` | ✅ `noise::SaltAndPepperNoiseFilter` |
| Shot Noise | `ShotNoiseImageFilter` | ✅ `noise::ShotNoiseFilter` |
| Speckle Noise | `SpeckleNoiseImageFilter` | ✅ `noise::SpeckleNoiseFilter` |

### Image Sources

| Filter | ITK class | Status |
|---|---|---|
| Gaussian Image Source | `GaussianImageSource` | ✅ `sources::GaussianImageSource` |
| Gabor Image Source | `GaborImageSource` | ✅ `sources::GaborImageSource` |
| Grid Image Source | `GridImageSource` | ✅ `sources::GridImageSource` |
| Physical Point Image Source | `PhysicalPointImageSource` | ✅ `sources::PhysicalPointImageSource` |

### Bias Correction

| Filter | ITK class | Status |
|---|---|---|
| N4 Bias Field Correction | `N4BiasFieldCorrectionImageFilter` | ✅ `bias_correction::N4BiasFieldCorrectionFilter` |
| MRI Bias Field Correction | `MRIBiasFieldCorrectionFilter` | ✅ `bias_correction::MRIBiasFieldCorrectionFilter` |

### Displacement Fields

| Filter | ITK class | Status |
|---|---|---|
| Transform to Displacement Field | `TransformToDisplacementFieldFilter` | ✅ `displacement::TransformToDisplacementField2D` |
| Compose Displacement Fields | `ComposeDisplacementFieldsImageFilter` | ✅ `displacement::ComposeDisplacementFields2D` |
| Invert Displacement Field | `InvertDisplacementFieldImageFilter` | ✅ `displacement::InvertDisplacementFieldFilter2D` |
| Exponential Displacement Field | `ExponentialDisplacementFieldImageFilter` | ✅ `displacement::ExponentialDisplacementFieldFilter2D` |
| Displacement Field to BSpline | `DisplacementFieldToBSplineImageFilter` | ✅ `spatial::DisplacementFieldToBSplineFilter` |
| Displacement Field Jacobian Determinant | `DisplacementFieldJacobianDeterminantFilter` | ✅ `displacement::DisplacementFieldJacobianDeterminantFilter2D` |
| Landmark Displacement Field Source | `LandmarkDisplacementFieldSource` | ✅ `spatial::LandmarkDisplacementFieldSource` |

### Diffusion Tensor

| Filter | ITK class | Status |
|---|---|---|
| DTI Reconstruction | `DiffusionTensor3DReconstructionImageFilter` | ✅ `dti::DiffusionTensor3DReconstructionFilter` |
| Fractional Anisotropy | `TensorFractionalAnisotropyImageFilter` | ✅ `dti::FractionalAnisotropyFilter` |
| Relative Anisotropy | `TensorRelativeAnisotropyImageFilter` | ✅ `dti::RelativeAnisotropyFilter` |

### Curvature Flow

| Filter | ITK class | Status |
|---|---|---|
| Curvature Flow | `CurvatureFlowImageFilter` | ✅ `curvature_flow::CurvatureFlowFilter` (D=2) |
| Min-Max Curvature Flow | `MinMaxCurvatureFlowImageFilter` | ✅ `curvature_flow::MinMaxCurvatureFlowFilter` (D=2) |
| Binary Min-Max Curvature Flow | `BinaryMinMaxCurvatureFlowImageFilter` | ✅ `curvature_flow::BinaryMinMaxCurvatureFlowFilter` (D=2) |

### Colormap

| Filter | ITK class | Status |
|---|---|---|
| Scalar to RGB Colormap | `ScalarToRGBColormapImageFilter` | ✅ `colormap::ScalarToRGBColormapFilter` |

### Label Map

| Filter | ITK class | Status |
|---|---|---|
| Binary Image to Label Map | `BinaryImageToLabelMapFilter` | ✅ `label_map::BinaryImageToLabelMapFilter` |
| Label Image to Label Map | `LabelImageToLabelMapFilter` | ✅ `label_map::label_image_to_label_map` |
| Label Map to Binary Image | `LabelMapToBinaryImageFilter` | ✅ `label_map::label_map_to_binary` |
| Label Map to Label Image | `LabelMapToLabelImageFilter` | ✅ `label_map::label_map_to_label_image` |
| Shape Label Map Filter | `ShapeLabelMapFilter` | ✅ `label_map::ShapeLabelMapFilter` |
| Statistics Label Map Filter | `StatisticsLabelMapFilter` | ✅ `label_map::StatisticsLabelMapFilter` |
| Shape Keep N Objects | `ShapeKeepNObjectsLabelMapFilter` | ✅ `label_map::ShapeKeepNObjectsFilter` |
| Shape Opening | `ShapeOpeningLabelMapFilter` | ✅ `label_map::ShapeOpeningFilter` |
| Statistics Keep N Objects | `StatisticsKeepNObjectsLabelMapFilter` | ✅ `label_map::StatisticsKeepNObjectsFilter` |
| Statistics Opening | `StatisticsOpeningLabelMapFilter` | ✅ `label_map::StatisticsOpeningFilter` |
| Binary Image to Shape Label Map | `BinaryImageToShapeLabelMapFilter` | ✅ `label_map::BinaryImageToShapeLabelMapFilter` |
| Binary Image to Statistics Label Map | `BinaryImageToStatisticsLabelMapFilter` | ✅ `label_map::BinaryImageToStatisticsLabelMapFilter` |
| Relabel Label Map | `RelabelLabelMapFilter` | ✅ `label_map::relabel_by_size` |
| Merge Label Map | `MergeLabelMapFilter` | ✅ `label_map::MergeLabelMapFilter` |
| Auto Crop Label Map | `AutoCropLabelMapFilter` | ✅ `label_map::AutoCropLabelMapFilter` |
| Label Map Mask Image | `LabelMapMaskImageFilter` | ✅ `label_map::LabelMapMaskImageFilter` |
| Label Map Overlay Image | `LabelMapOverlayImageFilter` | ✅ `label_map::LabelMapOverlayImageFilter` |
| Binary Fillhole | `BinaryFillholeImageFilter` | ✅ `label_map::BinaryFillholeFilter` |
| Binary Grind Peak | `BinaryGrindPeakImageFilter` | ✅ `label_map::BinaryGrindPeakFilter` |

---

## Segmentation

| Filter | ITK class | Status |
|---|---|---|
| Connected Threshold | `ConnectedThresholdImageFilter` | ✅ `segmentation::ConnectedThresholdFilter` |
| Confidence Connected | `ConfidenceConnectedImageFilter` | ✅ `segmentation::ConfidenceConnectedFilter` |
| Neighborhood Connected | `NeighborhoodConnectedImageFilter` | ✅ `segmentation::NeighborhoodConnectedFilter` |
| Isolated Connected | `IsolatedConnectedImageFilter` | ✅ `segmentation::IsolatedConnectedFilter` |
| Vector Confidence Connected | `VectorConfidenceConnectedImageFilter` | ✅ `segmentation::VectorConfidenceConnectedFilter` |
| Connected Component | `ConnectedComponentImageFilter` | ✅ `segmentation::ConnectedComponentFilter` |
| Relabel Component | `RelabelComponentImageFilter` | ✅ `segmentation::RelabelComponentFilter` |
| Threshold Maximum Connected Components | `ThresholdMaximumConnectedComponentsImageFilter` | ✅ `segmentation::ThresholdMaxConnectedComponentsFilter` |
| Morphological Watershed | `MorphologicalWatershedImageFilter` | ✅ `segmentation::MorphologicalWatershedFilter` |
| Morphological Watershed from Markers | `MorphologicalWatershedFromMarkersImageFilter` | ✅ `segmentation::MorphologicalWatershedFromMarkersFilter` |
| Watershed | `WatershedImageFilter` | ✅ `segmentation::WatershedImageFilter` |
| Toboggan | `TobogganImageFilter` | ✅ `segmentation::TobogganFilter` |
| SLIC Superpixel | `SLICImageFilter` | ✅ `segmentation::SLICFilter` |
| Bayesian Classifier | `BayesianClassifierImageFilter` | ✅ `segmentation::BayesianClassifierFilter` |
| KMeans | `ScalarImageKmeansImageFilter` | ✅ `segmentation::KMeansFilter` |
| MRF | `MRFImageFilter` | ✅ `segmentation::MRFFilter` |
| Label Voting | `LabelVotingImageFilter` | ✅ `segmentation::LabelVotingFilter` |
| Multi-Label STAPLE | `MultiLabelSTAPLEImageFilter` | ✅ `statistics::MultiLabelSTAPLEFilter` |
| Voting Binary Hole Filling | `VotingBinaryHoleFillingImageFilter` | ✅ `segmentation::VotingBinaryHoleFillingFilter` |
| Voting Binary Iterative Hole Filling | `VotingBinaryIterativeHoleFillingImageFilter` | ✅ `segmentation::VotingBinaryIterativeHoleFillingFilter` |
| Geodesic Active Contour Level Set | `GeodesicActiveContourLevelSetImageFilter` | ✅ `segmentation::GeodesicActiveContourLevelSetFilter` |
| Curves Level Set | `CurvesLevelSetImageFilter` | ✅ `segmentation::CurvesLevelSetFilter` |
| Laplacian Level Set | `LaplacianLevelSetImageFilter` | ✅ `segmentation::LaplacianLevelSetFilter` |
| Canny Segmentation Level Set | `CannySegmentationLevelSetImageFilter` | ✅ `segmentation::CannySegmentationLevelSetFilter` |
| Threshold Segmentation Level Set | `ThresholdSegmentationLevelSetImageFilter` | ✅ `segmentation::ThresholdSegmentationLevelSetFilter` |
| Shape Prior Level Set | `ShapePriorSegmentationLevelSetImageFilter` | ⬜ |
| Isolated Watershed | `IsolatedWatershedImageFilter` | ⬜ |
| Voronoi Segmentation | `VoronoiSegmentationImageFilter` | ⬜ |
| KLM Region Growing | `KLMRegionGrowImageFilter` | ⬜ |

---

## Registration

| Method | ITK class | Status |
|---|---|---|
| Image Registration v4 | `ImageRegistrationMethodv4` | ✅ `registration_methods::ImageRegistrationMethodV4` |
| SyN Registration | `SyNImageRegistrationMethod` | ✅ `registration_methods::SyNRegistrationMethod` |
| BSpline SyN Registration | `BSplineSyNImageRegistrationMethod` | ⬜ |
| Time Varying Velocity Field | `TimeVaryingVelocityFieldImageRegistrationMethodv4` | ⬜ |
| Demons Registration | `DemonsRegistrationFilter` | ✅ `registration_methods::DemonsRegistrationFilter` |
| Diffeomorphic Demons | `DiffeomorphicDemonsRegistrationFilter` | ✅ `registration_methods::DiffeomorphicDemonsRegistrationFilter` |
| Fast Symmetric Forces Demons | `FastSymmetricForcesDemonsRegistrationFilter` | ✅ `registration_methods::FastSymmetricForcesDemonsRegistrationFilter` |
| Multi-Resolution PDE Deformable | `MultiResolutionPDEDeformableRegistration` | ✅ `registration_methods::MultiResolutionDemonsRegistration` |
| FEM Registration | `FEMRegistrationFilter` | ⬜ |

### Metrics

| Metric | ITK class | Status |
|---|---|---|
| Mean Squares | `MeanSquaresImageToImageMetricv4` | ✅ `registration::MeanSquaresMetric` |
| Normalized Correlation | `CorrelationImageToImageMetricv4` | ✅ `registration::CorrelationMetric` |
| Mattes Mutual Information | `MattesMutualInformationImageToImageMetricv4` | ✅ `registration::MattesMutualInformationMetric` |
| Joint Histogram Mutual Information | `JointHistogramMutualInformationImageToImageMetricv4` | ✅ `registration_methods::JointHistogramMIMetric` |
| ANTS Neighborhood Correlation | `ANTSNeighborhoodCorrelationImageToImageMetricv4` | ✅ `registration::ANTSCorrelationMetric` |
| Demons | `DemonsImageToImageMetricv4` | ✅ `registration_methods::DemonsMetric` |

---

## Transforms

| Transform | ITK class | Status |
|---|---|---|
| Translation | `TranslationTransform` | ✅ |
| Rigid 2D | `Rigid2DTransform` | ✅ (via Euler2D) |
| Euler 2D | `Euler2DTransform` | ✅ |
| Rigid 3D | (Versor-based) | ✅ (via VersorRigid3D) |
| Euler 3D | `Euler3DTransform` | ✅ |
| Similarity 2D | `Similarity2DTransform` | ✅ |
| Similarity 3D | `Similarity3DTransform` | ✅ `segmentation::Similarity3DTransform` |
| Affine | `AffineTransform` | ✅ |
| Scale | `ScaleTransform` | ✅ |
| BSpline Deformable | `BSplineTransform` | ✅ |
| Displacement Field | `DisplacementFieldTransform` | ✅ |
| Composite Transform | `CompositeTransform` | ✅ |
| Thin Plate Spline | (Kernel transform) | ✅ `segmentation::ThinPlateSplineTransform` |
| Gaussian Exponential Diffeomorphic | `GaussianExponentialDiffeomorphicTransform` | ✅ `segmentation::GaussianExponentialDiffeomorphicTransform` |
| Time Varying Velocity Field | `TimeVaryingVelocityFieldTransform` | ⬜ |

---

## Interpolators

| Interpolator | ITK class | Status |
|---|---|---|
| Nearest Neighbor | `NearestNeighborInterpolateImageFunction` | ✅ `src/interpolate/nearest.rs` |
| Linear | `LinearInterpolateImageFunction` | ✅ `src/interpolate/linear.rs` — N-D, 2^D corners |
| BSpline | `BSplineInterpolateImageFunction` | ✅ `src/interpolate/bspline.rs` — orders 0–5, IIR prefilter |
| Gaussian | `GaussianInterpolateImageFunction` | ✅ `src/interpolate/gaussian.rs` — erf-weighted |
| Label Gaussian | `LabelImageGaussianInterpolateImageFunction` | ✅ `src/interpolate/gaussian.rs` |
| Windowed Sinc | `WindowedSincInterpolateImageFunction` | ✅ `src/interpolate/windowed_sinc.rs` — Hamming/Cosine/Welch/Lanczos/Blackman |

---

## Numerics

### Optimizers

| Optimizer | ITK class | Status |
|---|---|---|
| Gradient Descent | `GradientDescentOptimizerv4` | ✅ `optimizer::GradientDescentOptimizer` |
| Regular Step Gradient Descent | `RegularStepGradientDescentOptimizerv4` | ✅ `optimizer::RegularStepGradientDescentOptimizer` |
| LBFGS | `LBFGSOptimizerv4` | ✅ `optimizer::LBFGSOptimizer` |
| LBFGSB | `LBFGSBOptimizerv4` | ✅ `optimizer::LBFGSBOptimizer` |
| Amoeba (Nelder-Mead) | `AmoebaOptimizer` | ✅ `optimizer::AmoebaOptimizer` |
| Powell | `PowellOptimizer` | ✅ `optimizer::PowellOptimizer` |
| Exhaustive | `ExhaustiveOptimizer` | ✅ `optimizer::ExhaustiveOptimizer` |
| Conjugate Gradient | `ConjugateGradientLineSearchOptimizerv4` | ✅ `optimizer::ConjugateGradientOptimizer` |

---

## I/O

| Format | Read | Write | Status |
|---|---|---|---|
| NIfTI (.nii, .nii.gz) | ✓ | ✓ | ⬜ |
| NRRD (.nrrd, .nhdr) | ✓ | ✓ | ⬜ |
| DICOM (GDCM) | ✓ | ✓ | ⬜ |
| DICOM (DCMTK) | ✓ | ✓ | ⬜ |
| MRC (cryo-EM) | ✓ | ✓ | ⬜ |
| HDF5 | ✓ | ✓ | ⬜ |
| TIFF | ✓ | ✓ | ⬜ |
| PNG | ✓ | ✓ | ✅ `src/io/png.rs` — gray8/16, rgb8/16, rgba8/16 |
| JPEG | ✓ | ✓ | ⬜ |
| BMP | ✓ | ✓ | ⬜ |
| VTK image | ✓ | ✓ | ⬜ |
| MINC | ✓ | ✓ | ⬜ |
| Zeiss LSM | ✓ | — | ⬜ |
| GE (4/5/Adw) | ✓ | — | ⬜ |
| Siemens | ✓ | — | ⬜ |
| Bruker 2dseq | ✓ | — | ⬜ |
| Philips PAR/REC | ✓ | — | ⬜ |
| Stimulate | ✓ | — | ⬜ |
| GIPL | ✓ | ✓ | ⬜ |
| RAW | ✓ | ✓ | ⬜ |
| Mesh VTK PolyData | ✓ | ✓ | ⬜ |
| Mesh OBJ | ✓ | ✓ | ⬜ |
| Mesh OFF | ✓ | ✓ | ⬜ |
| Mesh FreeSurfer | ✓ | ✓ | ⬜ |
| Mesh GIFTI | ✓ | ✓ | ⬜ |
| Transform (ITK legacy) | ✓ | ✓ | ⬜ |
| Transform (HDF5) | ✓ | ✓ | ⬜ |
