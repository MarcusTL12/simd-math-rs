use std::simd::SimdFloat;

mod f64;

pub trait SimdFloatMath: SimdFloat {
    fn exp(self) -> Self;
}
