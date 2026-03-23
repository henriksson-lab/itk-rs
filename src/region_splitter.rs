use crate::image::Region;

/// Split `region` into at most `num_pieces` non-overlapping sub-regions that
/// together tile `region` exactly.
///
/// Splits along the slowest-varying axis (axis D-1), matching ITK's default
/// `ImageRegionSplitterSlowDimension` strategy.
pub fn split_region<const D: usize>(
    region: Region<D>,
    num_pieces: usize,
) -> impl Iterator<Item = Region<D>> {
    let num_pieces = num_pieces.max(1);

    // Collect pieces eagerly into a Vec so we can return a simple iterator.
    let mut pieces = Vec::with_capacity(num_pieces);

    if D == 0 || region.linear_len() == 0 {
        return pieces.into_iter();
    }

    let slow_axis = D - 1;
    let total = region.size.0[slow_axis];
    let actual_pieces = num_pieces.min(total).max(1);
    let base_size = total / actual_pieces;
    let remainder = total % actual_pieces;

    let mut start = region.index.0[slow_axis];
    for i in 0..actual_pieces {
        let this_size = base_size + if i < remainder { 1 } else { 0 };
        let mut idx = region.index.0;
        let mut sz = region.size.0;
        idx[slow_axis] = start;
        sz[slow_axis] = this_size;
        pieces.push(Region::new(idx, sz));
        start += this_size as i64;
    }

    pieces.into_iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_exact() {
        let r = Region::<2>::new([0, 0], [4, 6]);
        let pieces: Vec<_> = split_region(r, 3).collect();
        assert_eq!(pieces.len(), 3);
        assert_eq!(pieces[0].size.0[1], 2);
        assert_eq!(pieces[1].size.0[1], 2);
        assert_eq!(pieces[2].size.0[1], 2);
        // Together they cover the full region
        let total: usize = pieces.iter().map(|p| p.size.0[1]).sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn test_split_uneven() {
        let r = Region::<2>::new([0, 0], [4, 7]);
        let pieces: Vec<_> = split_region(r, 3).collect();
        let total: usize = pieces.iter().map(|p| p.size.0[1]).sum();
        assert_eq!(total, 7);
        // No piece is empty
        for p in &pieces {
            assert!(p.size.0[1] > 0);
        }
    }

    #[test]
    fn test_split_more_pieces_than_rows() {
        let r = Region::<2>::new([0, 0], [4, 2]);
        let pieces: Vec<_> = split_region(r, 10).collect();
        // Can't have more pieces than rows
        assert_eq!(pieces.len(), 2);
        let total: usize = pieces.iter().map(|p| p.size.0[1]).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_split_1d() {
        let r = Region::<1>::new([5], [10]);
        let pieces: Vec<_> = split_region(r, 4).collect();
        let total: usize = pieces.iter().map(|p| p.size.0[0]).sum();
        assert_eq!(total, 10);
        assert_eq!(pieces[0].index.0[0], 5);
    }
}
