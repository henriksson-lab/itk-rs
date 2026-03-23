/// N-dimensional index (signed to allow negative offsets during padding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Index<const D: usize>(pub [i64; D]);

/// N-dimensional size (number of pixels per axis).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size<const D: usize>(pub [usize; D]);

/// Axis-aligned rectangular region: index + size.
/// Analog to itk::ImageRegion<VDim>.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region<const D: usize> {
    pub index: Index<D>,
    pub size: Size<D>,
}

impl<const D: usize> Region<D> {
    pub fn new(index: [i64; D], size: [usize; D]) -> Self {
        Self { index: Index(index), size: Size(size) }
    }

    /// Total number of pixels in the region.
    pub fn linear_len(&self) -> usize {
        self.size.0.iter().product()
    }

    /// Expand each axis by its own radius (for per-axis padding in separable filters).
    pub fn padded_per_axis(&self, radii: &[usize; D]) -> Self {
        let mut index = self.index.0;
        let mut size = self.size.0;
        for d in 0..D {
            let r = radii[d] as i64;
            index[d] -= r;
            size[d] += 2 * radii[d];
        }
        Self { index: Index(index), size: Size(size) }
    }

    /// Expand each axis by `radius` on both sides (for neighborhood filters).
    pub fn padded(&self, radius: usize) -> Self {
        let r = radius as i64;
        let mut index = self.index.0;
        let mut size = self.size.0;
        for d in 0..D {
            index[d] -= r;
            size[d] += 2 * radius;
        }
        Self { index: Index(index), size: Size(size) }
    }

    /// Clip this region to fit within `bounds` (zero-flux Neumann boundary handling).
    pub fn clipped_to(&self, bounds: &Region<D>) -> Self {
        let mut index = self.index.0;
        let mut size = self.size.0;
        for d in 0..D {
            let self_end = index[d] + size[d] as i64;
            let bounds_end = bounds.index.0[d] + bounds.size.0[d] as i64;
            let new_start = index[d].max(bounds.index.0[d]);
            let new_end = self_end.min(bounds_end);
            index[d] = new_start;
            size[d] = (new_end - new_start).max(0) as usize;
        }
        Self { index: Index(index), size: Size(size) }
    }

    /// Check whether an index lies within this region.
    pub fn contains(&self, idx: &Index<D>) -> bool {
        for d in 0..D {
            let start = self.index.0[d];
            let end = start + self.size.0[d] as i64;
            if idx.0[d] < start || idx.0[d] >= end {
                return false;
            }
        }
        true
    }
}

/// In-memory image buffer. Analog to itk::Image<TPixel, VDim>.
///
/// `region` is the BufferedRegion — the extent of `data` in image coordinates.
pub struct Image<P, const D: usize> {
    pub region: Region<D>,
    pub spacing: [f64; D],
    pub origin: [f64; D],
    pub data: Vec<P>,
}

impl<P: Clone + Default, const D: usize> Image<P, D> {
    /// Allocate a new image filled with `fill`.
    pub fn allocate(
        region: Region<D>,
        spacing: [f64; D],
        origin: [f64; D],
        fill: P,
    ) -> Self {
        let len = region.linear_len();
        Self { region, spacing, origin, data: vec![fill; len] }
    }
}

impl<P: Clone, const D: usize> Image<P, D> {
    /// Convert an N-D index to a flat offset into `self.data`.
    /// Panics if `idx` is outside `self.region`.
    pub fn flat_index(&self, idx: Index<D>) -> usize {
        let mut offset = 0usize;
        let mut stride = 1usize;
        for d in 0..D {
            let local = (idx.0[d] - self.region.index.0[d]) as usize;
            offset += local * stride;
            stride *= self.region.size.0[d];
        }
        offset
    }

    pub fn get_pixel(&self, idx: Index<D>) -> P {
        let i = self.flat_index(idx);
        self.data[i].clone()
    }

    pub fn set_pixel(&mut self, idx: Index<D>, val: P) {
        let i = self.flat_index(idx);
        self.data[i] = val;
    }

    /// Copy pixels from `src` into `self`. `src.region` must be contained in `self.region`.
    pub fn copy_from(&mut self, src: &Image<P, D>) {
        // Iterate over all indices in src.region
        iter_region(&src.region, |idx| {
            let src_i = src.flat_index(idx);
            let dst_i = self.flat_index(idx);
            self.data[dst_i] = src.data[src_i].clone();
        });
    }
}

/// Call `f` for every index in `region`, in row-major order (axis 0 fastest).
pub fn iter_region<const D: usize, F: FnMut(Index<D>)>(region: &Region<D>, mut f: F) {
    let mut idx = region.index.0;
    let end: [i64; D] = {
        let mut e = region.index.0;
        for d in 0..D {
            e[d] += region.size.0[d] as i64;
        }
        e
    };
    if region.linear_len() == 0 {
        return;
    }
    loop {
        f(Index(idx));
        // Increment: axis 0 is fastest
        let mut carry = true;
        for d in 0..D {
            if carry {
                idx[d] += 1;
                if idx[d] < end[d] {
                    carry = false;
                } else {
                    idx[d] = region.index.0[d];
                }
            }
        }
        if carry {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_padded() {
        let r = Region::<2>::new([10, 20], [5, 8]);
        let p = r.padded(2);
        assert_eq!(p.index.0, [8, 18]);
        assert_eq!(p.size.0, [9, 12]);
    }

    #[test]
    fn test_region_clipped() {
        let bounds = Region::<2>::new([0, 0], [10, 10]);
        let r = Region::<2>::new([-2, 3], [8, 10]);
        let c = r.clipped_to(&bounds);
        assert_eq!(c.index.0, [0, 3]);
        assert_eq!(c.size.0, [6, 7]);
    }

    #[test]
    fn test_flat_index_2d() {
        let img = Image::<u8, 2>::allocate(Region::new([0, 0], [4, 3]), [1.0, 1.0], [0.0, 0.0], 0);
        // row-major: axis 0 fastest → flat = x + y*4
        assert_eq!(img.flat_index(Index([1, 2])), 1 + 2 * 4);
    }

    #[test]
    fn test_iter_region_count() {
        let r = Region::<3>::new([0, 0, 0], [2, 3, 4]);
        let mut count = 0;
        iter_region(&r, |_| count += 1);
        assert_eq!(count, 24);
    }
}
