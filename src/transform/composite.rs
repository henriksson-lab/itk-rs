//! Composite transform — chain of sub-transforms.
//! Analog to `itk::CompositeTransform`.
//!
//! Transforms are stored in a `Vec` and applied in **forward** order, i.e.
//! `T_last( … T_1( T_0(x) ) … )`. This matches ITK's "last added = outermost"
//! convention when transforms are pushed with `add_transform`.
//!
//! The inverse applies sub-transforms in reverse order.

use super::Transform;

/// Ordered sequence of boxed transforms applied left-to-right.
pub struct CompositeTransform<const D: usize> {
    transforms: Vec<Box<dyn Transform<D>>>,
}

impl<const D: usize> CompositeTransform<D> {
    pub fn new() -> Self {
        Self { transforms: Vec::new() }
    }

    /// Append a transform. It will be applied after all previously added transforms.
    pub fn add_transform(&mut self, t: Box<dyn Transform<D>>) {
        self.transforms.push(t);
    }

    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl<const D: usize> Default for CompositeTransform<D> {
    fn default() -> Self { Self::new() }
}

impl<const D: usize> Transform<D> for CompositeTransform<D> {
    fn transform_point(&self, mut point: [f64; D]) -> [f64; D] {
        for t in &self.transforms {
            point = t.transform_point(point);
        }
        point
    }

    fn inverse_transform_point(&self, mut point: [f64; D]) -> Option<[f64; D]> {
        for t in self.transforms.iter().rev() {
            point = t.inverse_transform_point(point)?;
        }
        Some(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{TranslationTransform, ScaleTransform};

    #[test]
    fn two_translations() {
        let mut ct = CompositeTransform::<2>::new();
        ct.add_transform(Box::new(TranslationTransform::new([1.0, 0.0])));
        ct.add_transform(Box::new(TranslationTransform::new([0.0, 2.0])));
        let p = ct.transform_point([0.0, 0.0]);
        assert!((p[0] - 1.0).abs() < 1e-10 && (p[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn inverse_round_trip() {
        let mut ct = CompositeTransform::<2>::new();
        ct.add_transform(Box::new(ScaleTransform::new([2.0, 3.0])));
        ct.add_transform(Box::new(TranslationTransform::new([5.0, -1.0])));
        let p = [3.0, 4.0];
        let q = ct.transform_point(p);
        let r = ct.inverse_transform_point(q).unwrap();
        assert!((r[0] - p[0]).abs() < 1e-10 && (r[1] - p[1]).abs() < 1e-10);
    }
}
