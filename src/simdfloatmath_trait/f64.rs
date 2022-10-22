use std::simd::{LaneCount, Simd, SupportedLaneCount};

use crate::{exp_simd, sin_simd, cos_simd, tan_simd, atan_simd, ln_simd};

use super::SimdFloatMath;

impl<const LANES: usize> SimdFloatMath for Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline(always)]
    fn exp(self) -> Self {
        exp_simd(self)
    }

    fn sin(self) -> Self {
        sin_simd(self)
    }

    fn cos(self) -> Self {
        cos_simd(self)
    }

    fn tan(self) -> Self {
        tan_simd(self)
    }

    fn atan(self) -> Self {
        atan_simd(self)
    }

    fn ln(self) -> Self {
        ln_simd(self)
    }
}
