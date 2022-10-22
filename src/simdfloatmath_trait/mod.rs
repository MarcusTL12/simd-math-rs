use std::simd::SimdFloat;

mod f64;

pub trait SimdFloatMath: SimdFloat {
    fn exp(self) -> Self;

    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;

    fn atan(self) -> Self;
    fn atan2(self, x: Self) -> Self;

    fn ln(self) -> Self;
}
