use std::simd::{LaneCount, Simd, SupportedLaneCount};

use crate::exp_simd;

use super::SimdFloatMath;

impl<const LANES: usize> SimdFloatMath for Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline(always)]
    fn exp(self) -> Self {
        exp_simd(self)
    }
}
