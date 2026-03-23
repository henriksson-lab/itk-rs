//! Geometric transforms, analog to ITK's `itk::Transform` hierarchy.
//!
//! Every transform maps a D-dimensional point to another D-dimensional point.
//! Transforms are pure value types (no pipeline integration) and are typically
//! used together with a `ResampleImageFilter`.
//!
//! # Available transforms
//!
//! | Rust type | ITK analog | DOF |
//! |---|---|---|
//! | [`TranslationTransform`] | `itk::TranslationTransform` | D |
//! | [`ScaleTransform`] | `itk::ScaleTransform` | D |
//! | [`AffineTransform`] | `itk::AffineTransform` | D²+D |
//! | [`Euler2DTransform`] | `itk::Euler2DTransform` | 3 |
//! | [`Euler3DTransform`] | `itk::Euler3DTransform` | 6 |
//! | [`Similarity2DTransform`] | `itk::Similarity2DTransform` | 4 |
//! | [`VersorRigid3DTransform`] | `itk::VersorRigid3DTransform` | 6 |
//! | [`CompositeTransform`] | `itk::CompositeTransform` | sum |
//! | [`DisplacementFieldTransform`] | `itk::DisplacementFieldTransform` | per-voxel |
//! | [`BSplineTransform`] | `itk::BSplineTransform` | per-control-point |

pub mod translation;
pub mod scale;
pub mod affine;
pub mod euler2d;
pub mod euler3d;
pub mod similarity2d;
pub mod versor_rigid3d;
pub mod composite;
pub mod displacement_field;
pub mod bspline;

pub use translation::TranslationTransform;
pub use scale::ScaleTransform;
pub use affine::AffineTransform;
pub use euler2d::Euler2DTransform;
pub use euler3d::Euler3DTransform;
pub use similarity2d::Similarity2DTransform;
pub use versor_rigid3d::VersorRigid3DTransform;
pub use composite::CompositeTransform;
pub use displacement_field::DisplacementFieldTransform;
pub use bspline::BSplineTransform;

/// Core transform trait. Analog to `itk::Transform<double, D, D>`.
///
/// All implementations work in physical-space **f64** coordinates.
pub trait Transform<const D: usize>: Send + Sync {
    /// Map an input point to an output point: `y = T(x)`.
    fn transform_point(&self, point: [f64; D]) -> [f64; D];

    /// Attempt the inverse mapping: `x = T⁻¹(y)`.
    /// Returns `None` if the transform is not invertible at this point.
    fn inverse_transform_point(&self, point: [f64; D]) -> Option<[f64; D]>;
}
