use std::ops::Add;

/// Marker trait for types that can be pixel values.
pub trait Pixel: Copy + Default + Send + Sync + 'static {}
impl<T: Copy + Default + Send + Sync + 'static> Pixel for T {}

/// Pixel types that support the arithmetic needed by smoothing filters.
/// Uses explicit `scale` and `zero` methods to avoid the cross-type
/// `Mul<f64>` problem (f32 doesn't implement Mul<f64>).
pub trait NumericPixel: Pixel + Add<Output = Self> {
    fn zero() -> Self;
    fn scale(self, w: f64) -> Self;
}

impl NumericPixel for f32 {
    fn zero() -> Self { 0.0 }
    fn scale(self, w: f64) -> Self { self * w as f32 }
}

impl NumericPixel for f64 {
    fn zero() -> Self { 0.0 }
    fn scale(self, w: f64) -> Self { self * w }
}

/// Fixed-length vector pixel. Analog to itk::Vector<T, N>.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VecPixel<T, const N: usize>(pub [T; N]);

impl<T: Default + Copy, const N: usize> Default for VecPixel<T, N> {
    fn default() -> Self {
        Self([T::default(); N])
    }
}

impl<T: NumericPixel, const N: usize> Add for VecPixel<T, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut out = self;
        for i in 0..N {
            out.0[i] = self.0[i] + rhs.0[i];
        }
        out
    }
}

impl<T: NumericPixel, const N: usize> NumericPixel for VecPixel<T, N> {
    fn zero() -> Self {
        Self([T::zero(); N])
    }
    fn scale(self, w: f64) -> Self {
        let mut out = self;
        for i in 0..N {
            out.0[i] = self.0[i].scale(w);
        }
        out
    }
}
