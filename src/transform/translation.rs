//! Translation transform. Analog to `itk::TranslationTransform`.

use super::Transform;

/// Pure translation: `y = x + offset`.
#[derive(Clone, Debug)]
pub struct TranslationTransform<const D: usize> {
    pub offset: [f64; D],
}

impl<const D: usize> TranslationTransform<D> {
    pub fn new(offset: [f64; D]) -> Self {
        Self { offset }
    }

    pub fn identity() -> Self {
        Self { offset: [0.0; D] }
    }
}

impl<const D: usize> Transform<D> for TranslationTransform<D> {
    fn transform_point(&self, point: [f64; D]) -> [f64; D] {
        let mut out = point;
        for d in 0..D {
            out[d] += self.offset[d];
        }
        out
    }

    fn inverse_transform_point(&self, point: [f64; D]) -> Option<[f64; D]> {
        let mut out = point;
        for d in 0..D {
            out[d] -= self.offset[d];
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translation_2d() {
        let t = TranslationTransform::new([3.0, -1.0]);
        let p = t.transform_point([1.0, 2.0]);
        assert!((p[0] - 4.0).abs() < 1e-10);
        assert!((p[1] - 1.0).abs() < 1e-10);
        let inv = t.inverse_transform_point(p).unwrap();
        assert!((inv[0] - 1.0).abs() < 1e-10);
        assert!((inv[1] - 2.0).abs() < 1e-10);
    }
}
