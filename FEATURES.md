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
| Box Sigma | `BoxSigmaImageFilter` | ⬜ |
| Binomial Blur | `BinomialBlurImageFilter` | ✅ `src/filters/binomial_blur.rs` |
| FFT Discrete Gaussian | `FFTDiscreteGaussianImageFilter` | ⬜ |

### Anisotropic Smoothing

| Filter | ITK class | Status |
|---|---|---|
| Gradient Anisotropic Diffusion | `GradientAnisotropicDiffusionImageFilter` | ✅ `src/filters/anisotropic_diffusion.rs` |
| Curvature Anisotropic Diffusion | `CurvatureAnisotropicDiffusionImageFilter` | ✅ `src/filters/anisotropic_diffusion.rs` (D=2) |
| Vector Gradient Anisotropic Diffusion | `VectorGradientAnisotropicDiffusionImageFilter` | ⬜ |
| Vector Curvature Anisotropic Diffusion | `VectorCurvatureAnisotropicDiffusionImageFilter` | ⬜ |

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
| Atan2 | `Atan2ImageFilter` | ⬜ |
| Pow | `PowImageFilter` | ✅ `intensity::PowFilter` |
| Sigmoid | `SigmoidImageFilter` | ✅ `intensity::SigmoidFilter` |
| Invert Intensity | `InvertIntensityImageFilter` | ✅ `intensity::InvertIntensityFilter` |
| Rescale Intensity | `RescaleIntensityImageFilter` | ✅ `intensity::RescaleIntensityFilter` |
| Shift Scale | `ShiftScaleImageFilter` | ✅ `intensity::ShiftScaleFilter` |
| Clamp | `ClampImageFilter` | ✅ `intensity::ClampFilter` |
| Normalize | `NormalizeImageFilter` | ✅ `intensity::NormalizeFilter` |
| Normalize to Constant | `NormalizeToConstantImageFilter` | ⬜ |
| Intensity Windowing | `IntensityWindowingImageFilter` | ✅ `intensity::IntensityWindowingFilter` |
| Histogram Matching | `HistogramMatchingImageFilter` | ⬜ |
| Mask | `MaskImageFilter` | ✅ `intensity::MaskFilter` |
| Mask Negated | `MaskNegatedImageFilter` | ✅ `intensity::MaskNegatedFilter` |
| Maximum (pixelwise) | `MaximumImageFilter` | ✅ `intensity::maximum_images` |
| Minimum (pixelwise) | `MinimumImageFilter` | ✅ `intensity::minimum_images` |
| And | `AndImageFilter` | ⬜ |
| Or | `OrImageFilter` | ⬜ |
| Xor | `XorImageFilter` | ⬜ |
| Not | `NotImageFilter` | ⬜ |
| Weighted Add | `WeightedAddImageFilter` | ✅ `intensity::WeightedAddFilter` |
| N-ary Add | `NaryAddImageFilter` | ⬜ |
| N-ary Maximum | `NaryMaximumImageFilter` | ⬜ |
| Constrained Value Addition | `ConstrainedValueAdditionImageFilter` | ⬜ |
| Constrained Value Difference | `ConstrainedValueDifferenceImageFilter` | ⬜ |
| Bounded Reciprocal | `BoundedReciprocalImageFilter` | ✅ `intensity::bounded_reciprocal_filter` |
| Modulus | `ModulusImageFilter` | ✅ `intensity::ModulusFilter` |
| Round | `RoundImageFilter` | ✅ `intensity::round_filter` |
| Cast | `CastImageFilter` | ✅ (via `UnaryFilter` with `from_f64`/`to_f64`) |
| Complex to Real | `ComplexToRealImageFilter` | ⬜ |
| Complex to Imaginary | `ComplexToImaginaryImageFilter` | ⬜ |
| Complex to Modulus | `ComplexToModulusImageFilter` | ⬜ |
| Complex to Phase | `ComplexToPhaseImageFilter` | ⬜ |
| Magnitude and Phase to Complex | `MagnitudeAndPhaseToComplexImageFilter` | ⬜ |
| Vector Magnitude | `VectorMagnitudeImageFilter` | ⬜ |
| Vector Index Selection Cast | `VectorIndexSelectionCastImageFilter` | ⬜ |
| Compose Image | `ComposeImageFilter` | ⬜ |
| Scalar to RGB Pixel | `ScalarToRGBPixelFunctor` | ⬜ |
| Symmetric Eigen Analysis | `SymmetricEigenAnalysisImageFilter` | ⬜ |

### Thresholding

| Filter | ITK class | Status |
|---|---|---|
| Binary Threshold | `BinaryThresholdImageFilter` | ✅ `threshold::BinaryThresholdFilter` |
| Threshold (in-place) | `ThresholdImageFilter` | ✅ `threshold::ThresholdFilter` |
| Otsu Threshold | `OtsuThresholdImageFilter` | ✅ `threshold::OtsuThresholdFilter` |
| Otsu Multiple Thresholds | `OtsuMultipleThresholdsImageFilter` | ⬜ |
| Huang Threshold | `HuangThresholdImageFilter` | ⬜ |
| Li Threshold | `LiThresholdImageFilter` | ⬜ |
| IsoData Threshold | `IsoDataThresholdImageFilter` | ⬜ |
| MaxEntropy Threshold | `MaximumEntropyThresholdImageFilter` | ⬜ |
| Moments Threshold | `MomentsThresholdImageFilter` | ⬜ |
| Triangle Threshold | `TriangleThresholdImageFilter` | ⬜ |
| Yen Threshold | `YenThresholdImageFilter` | ⬜ |
| Renyi Entropy Threshold | `RenyiEntropyThresholdImageFilter` | ⬜ |
| Shanbhag Threshold | `ShanbhagThresholdImageFilter` | ⬜ |
| Kittler-Illingworth Threshold | `KittlerIllingworthThresholdImageFilter` | ⬜ |
| Intermodes Threshold | `IntermodesThresholdImageFilter` | ⬜ |
| Kappa-Sigma Threshold | `KappaSigmaThresholdImageFilter` | ⬜ |

### Image Features & Edges

| Filter | ITK class | Status |
|---|---|---|
| Canny Edge Detection | `CannyEdgeDetectionImageFilter` | ⬜ |
| Sobel Edge Detection | `SobelEdgeDetectionImageFilter` | ⬜ |
| Laplacian | `LaplacianImageFilter` | ⬜ |
| Laplacian Recursive Gaussian | `LaplacianRecursiveGaussianImageFilter` | ⬜ |
| Laplacian Sharpening | `LaplacianSharpeningImageFilter` | ⬜ |
| Zero Crossing | `ZeroCrossingImageFilter` | ⬜ |
| Zero Crossing Based Edge Detection | `ZeroCrossingBasedEdgeDetectionImageFilter` | ⬜ |
| Derivative | `DerivativeImageFilter` | ⬜ |
| Discrete Gaussian Derivative | `DiscreteGaussianDerivativeImageFilter` | ⬜ |
| Hessian Recursive Gaussian | `HessianRecursiveGaussianImageFilter` | ⬜ |
| Hessian to Objectness Measure | `HessianToObjectnessMeasureImageFilter` | ⬜ |
| Multi-Scale Hessian Measure | `MultiScaleHessianBasedMeasureImageFilter` | ⬜ |
| Hessian 3D to Vesselness | `Hessian3DToVesselnessMeasureImageFilter` | ⬜ |
| Bilateral | `BilateralImageFilter` | ✅ `src/filters/bilateral.rs` |
| Unsharp Mask | `UnsharpMaskImageFilter` | ⬜ |
| Gradient Vector Flow | `GradientVectorFlowImageFilter` | ⬜ |
| Hough Transform 2D Circles | `HoughTransform2DCirclesImageFilter` | ⬜ |
| Hough Transform 2D Lines | `HoughTransform2DLinesImageFilter` | ⬜ |
| Simple Contour Extractor | `SimpleContourExtractorImageFilter` | ⬜ |

### Gradients

| Filter | ITK class | Status |
|---|---|---|
| Gradient | `GradientImageFilter` | ⬜ |
| Gradient Magnitude | `GradientMagnitudeImageFilter` | ⬜ |
| Gradient Magnitude Recursive Gaussian | `GradientMagnitudeRecursiveGaussianImageFilter` | ⬜ |
| Gradient Recursive Gaussian | `GradientRecursiveGaussianImageFilter` | ⬜ |
| Difference of Gaussians Gradient | `DifferenceOfGaussiansGradientImageFilter` | ⬜ |
| Vector Gradient Magnitude | `VectorGradientMagnitudeImageFilter` | ⬜ |

### Mathematical Morphology

| Filter | ITK class | Status |
|---|---|---|
| Grayscale Dilate | `GrayscaleDilateImageFilter` | ⬜ |
| Grayscale Erode | `GrayscaleErodeImageFilter` | ⬜ |
| Grayscale Morphological Opening | `GrayscaleMorphologicalOpeningImageFilter` | ⬜ |
| Grayscale Morphological Closing | `GrayscaleMorphologicalClosingImageFilter` | ⬜ |
| Morphological Gradient | `MorphologicalGradientImageFilter` | ⬜ |
| White Top Hat | `WhiteTopHatImageFilter` | ⬜ |
| Black Top Hat | `BlackTopHatImageFilter` | ⬜ |
| Double Threshold | `DoubleThresholdImageFilter` | ⬜ |
| H-Maxima | `HMaximaImageFilter` | ⬜ |
| H-Minima | `HMinimaImageFilter` | ⬜ |
| H-Concave | `HConcaveImageFilter` | ⬜ |
| H-Convex | `HConvexImageFilter` | ⬜ |
| Regional Maxima | `RegionalMaximaImageFilter` | ⬜ |
| Regional Minima | `RegionalMinimaImageFilter` | ⬜ |
| Grayscale Geodesic Dilate | `GrayscaleGeodesicDilateImageFilter` | ⬜ |
| Grayscale Geodesic Erode | `GrayscaleGeodesicErodeImageFilter` | ⬜ |
| Closing by Reconstruction | `ClosingByReconstructionImageFilter` | ⬜ |
| Opening by Reconstruction | `OpeningByReconstructionImageFilter` | ⬜ |
| Grayscale Connected Opening | `GrayscaleConnectedOpeningImageFilter` | ⬜ |
| Grayscale Connected Closing | `GrayscaleConnectedClosingImageFilter` | ⬜ |
| Grayscale Fillhole | `GrayscaleFillholeImageFilter` | ⬜ |
| Grayscale Grind Peak | `GrayscaleGrindPeakImageFilter` | ⬜ |
| Rank | `RankImageFilter` | ⬜ |
| Reconstruction by Dilation | `ReconstructionByDilationImageFilter` | ⬜ |
| Reconstruction by Erosion | `ReconstructionByErosionImageFilter` | ⬜ |

### Binary Morphology

| Filter | ITK class | Status |
|---|---|---|
| Binary Dilate | `BinaryDilateImageFilter` | ⬜ |
| Binary Erode | `BinaryErodeImageFilter` | ⬜ |
| Binary Morphological Opening | `BinaryMorphologicalOpeningImageFilter` | ⬜ |
| Binary Morphological Closing | `BinaryMorphologicalClosingImageFilter` | ⬜ |
| Binary Opening by Reconstruction | `BinaryOpeningByReconstructionImageFilter` | ⬜ |
| Binary Closing by Reconstruction | `BinaryClosingByReconstructionImageFilter` | ⬜ |
| Binary Thinning | `BinaryThinningImageFilter` | ⬜ |
| Binary Pruning | `BinaryPruningImageFilter` | ⬜ |

### Convolution & Frequency Domain

| Filter | ITK class | Status |
|---|---|---|
| Convolution | `ConvolutionImageFilter` | ⬜ |
| FFT Convolution | `FFTConvolutionImageFilter` | ⬜ |
| Normalized Correlation | `NormalizedCorrelationImageFilter` | ⬜ |
| FFT Normalized Correlation | `FFTNormalizedCorrelationImageFilter` | ⬜ |
| Masked FFT Normalized Correlation | `MaskedFFTNormalizedCorrelationImageFilter` | ⬜ |
| Forward FFT | `ForwardFFTImageFilter` | ⬜ |
| Inverse FFT | `InverseFFTImageFilter` | ⬜ |
| FFT Shift | `FFTShiftImageFilter` | ⬜ |
| FFT Pad | `FFTPadImageFilter` | ⬜ |
| Frequency Band Filter | `FrequencyBandImageFilter` | ⬜ |

### Deconvolution

| Filter | ITK class | Status |
|---|---|---|
| Inverse Deconvolution | `InverseDeconvolutionImageFilter` | ⬜ |
| Wiener Deconvolution | `WienerDeconvolutionImageFilter` | ⬜ |
| Tikhonov Deconvolution | `TikhonovDeconvolutionImageFilter` | ⬜ |
| Richardson-Lucy Deconvolution | `RichardsonLucyDeconvolutionImageFilter` | ⬜ |
| Landweber Deconvolution | `LandweberDeconvolutionImageFilter` | ⬜ |
| Projected Landweber Deconvolution | `ProjectedLandweberDeconvolutionImageFilter` | ⬜ |

### Distance Maps

| Filter | ITK class | Status |
|---|---|---|
| Signed Maurer Distance Map | `SignedMaurerDistanceMapImageFilter` | ⬜ |
| Danielsson Distance Map | `DanielssonDistanceMapImageFilter` | ⬜ |
| Signed Danielsson Distance Map | `SignedDanielssonDistanceMapImageFilter` | ⬜ |
| Fast Chamfer Distance | `FastChamferDistanceImageFilter` | ⬜ |
| IsoContour Distance | `IsoContourDistanceImageFilter` | ⬜ |
| Approximate Signed Distance Map | `ApproximateSignedDistanceMapImageFilter` | ⬜ |
| Hausdorff Distance | `HausdorffDistanceImageFilter` | ⬜ |
| Directed Hausdorff Distance | `DirectedHausdorffDistanceImageFilter` | ⬜ |
| Contour Mean Distance | `ContourMeanDistanceImageFilter` | ⬜ |

### Fast Marching

| Filter | ITK class | Status |
|---|---|---|
| Fast Marching Image Filter | `FastMarchingImageFilter` | ⬜ |
| Fast Marching Upwind Gradient | `FastMarchingUpwindGradientImageFilter` | ⬜ |
| Fast Marching Extension | `FastMarchingExtensionImageFilter` | ⬜ |

### Image Grid / Resampling

| Filter | ITK class | Status |
|---|---|---|
| Resample Image | `ResampleImageFilter` | ⬜ |
| Warp Image | `WarpImageFilter` | ⬜ |
| Shrink | `ShrinkImageFilter` | ⬜ |
| Expand | `ExpandImageFilter` | ⬜ |
| BSpline Downsample | `BSplineDownsampleImageFilter` | ⬜ |
| BSpline Upsample | `BSplineUpsampleImageFilter` | ⬜ |
| Crop | `CropImageFilter` | ⬜ |
| Region of Interest | `RegionOfInterestImageFilter` | ⬜ |
| Flip | `FlipImageFilter` | ⬜ |
| Permute Axes | `PermuteAxesImageFilter` | ⬜ |
| Constant Pad | `ConstantPadImageFilter` | ⬜ |
| Mirror Pad | `MirrorPadImageFilter` | ⬜ |
| Wrap Pad | `WrapPadImageFilter` | ⬜ |
| Zero Flux Neumann Pad | `ZeroFluxNeumannPadImageFilter` | ⬜ |
| Tile | `TileImageFilter` | ⬜ |
| Paste | `PasteImageFilter` | ⬜ |
| Change Information | `ChangeInformationImageFilter` | ⬜ |
| Orient Image | `OrientImageFilter` | ⬜ |
| Cyclic Shift | `CyclicShiftImageFilter` | ⬜ |
| Bin Shrink | `BinShrinkImageFilter` | ⬜ |
| Slice by Slice | `SliceBySliceImageFilter` | ⬜ |
| Interpolate Image | `InterpolateImageFilter` | ⬜ |
| BSpline Scattered Data Point Set to Image | `BSplineScatteredDataPointSetToImageFilter` | ⬜ |

### Image Statistics

| Filter | ITK class | Status |
|---|---|---|
| Statistics | `StatisticsImageFilter` | ⬜ |
| Label Statistics | `LabelStatisticsImageFilter` | ⬜ |
| Minimum Maximum | `MinimumMaximumImageFilter` | ⬜ |
| Image Moments | `ImageMomentsCalculator` | ⬜ |
| Image PCA Shape Model | `ImagePCAShapeModelEstimator` | ⬜ |
| Accumulate | `AccumulateImageFilter` | ⬜ |
| Max Projection | `MaximumProjectionImageFilter` | ⬜ |
| Min Projection | `MinimumProjectionImageFilter` | ⬜ |
| Mean Projection | `MeanProjectionImageFilter` | ⬜ |
| Sum Projection | `SumProjectionImageFilter` | ⬜ |
| Median Projection | `MedianProjectionImageFilter` | ⬜ |
| StdDev Projection | `StandardDeviationProjectionImageFilter` | ⬜ |
| Adaptive Histogram Equalization | `AdaptiveHistogramEqualizationImageFilter` | ⬜ |
| Label Overlap Measures | `LabelOverlapMeasuresImageFilter` | ⬜ |
| STAPLE | `STAPLEImageFilter` | ⬜ |
| Similarity Index | `SimilarityIndexImageFilter` | ⬜ |
| Checker Board | `CheckerBoardImageFilter` | ⬜ |

### Image Noise Simulation

| Filter | ITK class | Status |
|---|---|---|
| Additive Gaussian Noise | `AdditiveGaussianNoiseImageFilter` | ⬜ |
| Salt and Pepper Noise | `SaltAndPepperNoiseImageFilter` | ⬜ |
| Shot Noise | `ShotNoiseImageFilter` | ⬜ |
| Speckle Noise | `SpeckleNoiseImageFilter` | ⬜ |

### Image Sources

| Filter | ITK class | Status |
|---|---|---|
| Gaussian Image Source | `GaussianImageSource` | ⬜ |
| Gabor Image Source | `GaborImageSource` | ⬜ |
| Grid Image Source | `GridImageSource` | ⬜ |
| Physical Point Image Source | `PhysicalPointImageSource` | ⬜ |

### Bias Correction

| Filter | ITK class | Status |
|---|---|---|
| N4 Bias Field Correction | `N4BiasFieldCorrectionImageFilter` | ⬜ |
| MRI Bias Field Correction | `MRIBiasFieldCorrectionFilter` | ⬜ |

### Displacement Fields

| Filter | ITK class | Status |
|---|---|---|
| Transform to Displacement Field | `TransformToDisplacementFieldFilter` | ⬜ |
| Compose Displacement Fields | `ComposeDisplacementFieldsImageFilter` | ⬜ |
| Invert Displacement Field | `InvertDisplacementFieldImageFilter` | ⬜ |
| Exponential Displacement Field | `ExponentialDisplacementFieldImageFilter` | ⬜ |
| Displacement Field to BSpline | `DisplacementFieldToBSplineImageFilter` | ⬜ |
| Displacement Field Jacobian Determinant | `DisplacementFieldJacobianDeterminantFilter` | ⬜ |
| Landmark Displacement Field Source | `LandmarkDisplacementFieldSource` | ⬜ |

### Diffusion Tensor

| Filter | ITK class | Status |
|---|---|---|
| DTI Reconstruction | `DiffusionTensor3DReconstructionImageFilter` | ⬜ |
| Fractional Anisotropy | `TensorFractionalAnisotropyImageFilter` | ⬜ |
| Relative Anisotropy | `TensorRelativeAnisotropyImageFilter` | ⬜ |

### Curvature Flow

| Filter | ITK class | Status |
|---|---|---|
| Curvature Flow | `CurvatureFlowImageFilter` | ⬜ |
| Min-Max Curvature Flow | `MinMaxCurvatureFlowImageFilter` | ⬜ |
| Binary Min-Max Curvature Flow | `BinaryMinMaxCurvatureFlowImageFilter` | ⬜ |

### Colormap

| Filter | ITK class | Status |
|---|---|---|
| Scalar to RGB Colormap | `ScalarToRGBColormapImageFilter` | ⬜ |

### Label Map

| Filter | ITK class | Status |
|---|---|---|
| Binary Image to Label Map | `BinaryImageToLabelMapFilter` | ⬜ |
| Label Image to Label Map | `LabelImageToLabelMapFilter` | ⬜ |
| Label Map to Binary Image | `LabelMapToBinaryImageFilter` | ⬜ |
| Label Map to Label Image | `LabelMapToLabelImageFilter` | ⬜ |
| Shape Label Map Filter | `ShapeLabelMapFilter` | ⬜ |
| Statistics Label Map Filter | `StatisticsLabelMapFilter` | ⬜ |
| Shape Keep N Objects | `ShapeKeepNObjectsLabelMapFilter` | ⬜ |
| Shape Opening | `ShapeOpeningLabelMapFilter` | ⬜ |
| Statistics Keep N Objects | `StatisticsKeepNObjectsLabelMapFilter` | ⬜ |
| Statistics Opening | `StatisticsOpeningLabelMapFilter` | ⬜ |
| Binary Image to Shape Label Map | `BinaryImageToShapeLabelMapFilter` | ⬜ |
| Binary Image to Statistics Label Map | `BinaryImageToStatisticsLabelMapFilter` | ⬜ |
| Relabel Label Map | `RelabelLabelMapFilter` | ⬜ |
| Merge Label Map | `MergeLabelMapFilter` | ⬜ |
| Auto Crop Label Map | `AutoCropLabelMapFilter` | ⬜ |
| Label Map Mask Image | `LabelMapMaskImageFilter` | ⬜ |
| Label Map Overlay Image | `LabelMapOverlayImageFilter` | ⬜ |
| Binary Fillhole | `BinaryFillholeImageFilter` | ⬜ |
| Binary Grind Peak | `BinaryGrindPeakImageFilter` | ⬜ |

---

## Segmentation

| Filter | ITK class | Status |
|---|---|---|
| Connected Threshold | `ConnectedThresholdImageFilter` | ⬜ |
| Confidence Connected | `ConfidenceConnectedImageFilter` | ⬜ |
| Neighborhood Connected | `NeighborhoodConnectedImageFilter` | ⬜ |
| Isolated Connected | `IsolatedConnectedImageFilter` | ⬜ |
| Vector Confidence Connected | `VectorConfidenceConnectedImageFilter` | ⬜ |
| Connected Component | `ConnectedComponentImageFilter` | ⬜ |
| Relabel Component | `RelabelComponentImageFilter` | ⬜ |
| Threshold Maximum Connected Components | `ThresholdMaximumConnectedComponentsImageFilter` | ⬜ |
| Morphological Watershed | `MorphologicalWatershedImageFilter` | ⬜ |
| Morphological Watershed from Markers | `MorphologicalWatershedFromMarkersImageFilter` | ⬜ |
| Watershed | `WatershedImageFilter` | ⬜ |
| Toboggan | `TobogganImageFilter` | ⬜ |
| SLIC Superpixel | `SLICImageFilter` | ⬜ |
| Bayesian Classifier | `BayesianClassifierImageFilter` | ⬜ |
| KMeans | `ScalarImageKmeansImageFilter` | ⬜ |
| MRF | `MRFImageFilter` | ⬜ |
| Label Voting | `LabelVotingImageFilter` | ⬜ |
| Multi-Label STAPLE | `MultiLabelSTAPLEImageFilter` | ⬜ |
| Voting Binary Hole Filling | `VotingBinaryHoleFillingImageFilter` | ⬜ |
| Voting Binary Iterative Hole Filling | `VotingBinaryIterativeHoleFillingImageFilter` | ⬜ |
| Geodesic Active Contour Level Set | `GeodesicActiveContourLevelSetImageFilter` | ⬜ |
| Curves Level Set | `CurvesLevelSetImageFilter` | ⬜ |
| Laplacian Level Set | `LaplacianLevelSetImageFilter` | ⬜ |
| Canny Segmentation Level Set | `CannySegmentationLevelSetImageFilter` | ⬜ |
| Threshold Segmentation Level Set | `ThresholdSegmentationLevelSetImageFilter` | ⬜ |
| Shape Prior Level Set | `ShapePriorSegmentationLevelSetImageFilter` | ⬜ |
| Isolated Watershed | `IsolatedWatershedImageFilter` | ⬜ |
| Voronoi Segmentation | `VoronoiSegmentationImageFilter` | ⬜ |
| KLM Region Growing | `KLMRegionGrowImageFilter` | ⬜ |

---

## Registration

| Method | ITK class | Status |
|---|---|---|
| Image Registration v4 | `ImageRegistrationMethodv4` | ⬜ |
| SyN Registration | `SyNImageRegistrationMethod` | ⬜ |
| BSpline SyN Registration | `BSplineSyNImageRegistrationMethod` | ⬜ |
| Time Varying Velocity Field | `TimeVaryingVelocityFieldImageRegistrationMethodv4` | ⬜ |
| Demons Registration | `DemonsRegistrationFilter` | ⬜ |
| Diffeomorphic Demons | `DiffeomorphicDemonsRegistrationFilter` | ⬜ |
| Fast Symmetric Forces Demons | `FastSymmetricForcesDemonsRegistrationFilter` | ⬜ |
| Multi-Resolution PDE Deformable | `MultiResolutionPDEDeformableRegistration` | ⬜ |
| FEM Registration | `FEMRegistrationFilter` | ⬜ |

### Metrics

| Metric | ITK class | Status |
|---|---|---|
| Mean Squares | `MeanSquaresImageToImageMetricv4` | ⬜ |
| Normalized Correlation | `CorrelationImageToImageMetricv4` | ⬜ |
| Mattes Mutual Information | `MattesMutualInformationImageToImageMetricv4` | ⬜ |
| Joint Histogram Mutual Information | `JointHistogramMutualInformationImageToImageMetricv4` | ⬜ |
| ANTS Neighborhood Correlation | `ANTSNeighborhoodCorrelationImageToImageMetricv4` | ⬜ |
| Demons | `DemonsImageToImageMetricv4` | ⬜ |

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
| Similarity 3D | `Similarity3DTransform` | ⬜ |
| Affine | `AffineTransform` | ✅ |
| Scale | `ScaleTransform` | ✅ |
| BSpline Deformable | `BSplineTransform` | ✅ |
| Displacement Field | `DisplacementFieldTransform` | ✅ |
| Composite Transform | `CompositeTransform` | ✅ |
| Thin Plate Spline | (Kernel transform) | ⬜ |
| Gaussian Exponential Diffeomorphic | `GaussianExponentialDiffeomorphicTransform` | ⬜ |
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
| Gradient Descent | `GradientDescentOptimizerv4` | ⬜ |
| Regular Step Gradient Descent | `RegularStepGradientDescentOptimizerv4` | ⬜ |
| LBFGS | `LBFGSOptimizerv4` | ⬜ |
| LBFGSB | `LBFGSBOptimizerv4` | ⬜ |
| Amoeba (Nelder-Mead) | `AmoebaOptimizer` | ⬜ |
| Powell | `PowellOptimizer` | ⬜ |
| Exhaustive | `ExhaustiveOptimizer` | ⬜ |
| Conjugate Gradient | `ConjugateGradientLineSearchOptimizerv4` | ⬜ |

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
