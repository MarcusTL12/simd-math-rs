use std::{
    f64::consts::PI,
    simd::{LaneCount, Simd, SimdPartialEq, SupportedLaneCount},
};

use crate::{periodic_clamp, periodic_clamp_simd, polyval, polyval_simd};

const TAYLOR_COEFFS: [f64; 16] = [
    -5.407361331613617e-13,
    -8.111041997420426e-12,
    1.1355458796388596e-10,
    1.4762096435305176e-9,
    -1.771451572236621e-8,
    -1.948596729460283e-7,
    1.948596729460283e-6,
    1.7537370565142548e-5,
    -0.00014029896452114038,
    -0.0009820927516479827,
    0.005892556509887896,
    0.02946278254943948,
    -0.11785113019775792,
    -0.3535533905932738,
    0.7071067811865476,
    0.7071067811865476,
];

fn sin_shift(x: f64) -> f64 {
    let (mut u, n) = periodic_clamp(x, PI / 2.0);

    if n & 1 != 0 {
        u = -u;
    }

    let mut tl = polyval(&TAYLOR_COEFFS, u);

    if n & 2 != 0 {
        tl = -tl;
    }

    tl
}

pub fn sin(x: f64) -> f64 {
    sin_shift(x - PI / 4.0)
}

pub fn cos(x: f64) -> f64 {
    sin_shift(x + PI / 4.0)
}

pub fn tan(x: f64) -> f64 {
    sin(x) / cos(x)
}

#[inline(always)]
fn sin_shift_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let (u, n) = periodic_clamp_simd(x, PI / 2.0);

    let n: Simd<i64, LANES> = n.cast();

    let u = (n & Simd::splat(1)).simd_eq(Simd::splat(0)).select(u, -u);

    let tl = polyval_simd(&TAYLOR_COEFFS, u);

    (n & Simd::splat(2)).simd_eq(Simd::splat(0)).select(tl, -tl)
}

pub fn sin_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    sin_shift_simd(x - Simd::splat(PI / 4.0))
}

pub fn cos_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    sin_shift_simd(x + Simd::splat(PI / 4.0))
}

pub fn tan_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    sin_simd(x) / cos_simd(x)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use crate::{
        tests::{accuracy_test, speed_test_simd_iterated},
        trig::sin_shift,
        *,
    };

    #[test]
    fn test_sin_shift() {
        let x = [
            -49.27350894907335,
            33.556609699122156,
            6.616844076093664,
            33.07858625776914,
            34.14184641616258,
            41.87091444626816,
            -5.752401808482954,
            -5.339014479417625,
        ];

        accuracy_test(&x, |x| (x + PI / 4.0).sin(), sin_shift);
    }

    #[test]
    fn test_sin_small() {
        let x: [f64; 8] = [
            -4.725590455468264,
            -3.029442898943749,
            0.3449817636324948,
            -0.7537994616348398,
            0.7327486458751387,
            3.1200184229578154,
            -4.9415570981767365,
            -4.933437312855772,
        ];

        accuracy_test(&x, |x| x.sin(), sin);
    }

    #[test]
    fn test_sin_large() {
        let x: [f64; 8] = [
            -673.1445111913359,
            -4194.129748644129,
            2812.623289566699,
            -4882.861762124198,
            1815.6590613844326,
            -4318.334861343217,
            2955.7357268745563,
            2267.811391833918,
        ];

        accuracy_test(&x, |x| x.sin(), sin);
    }

    #[test]
    fn test_cos_small() {
        let x: [f64; 8] = [
            -4.725590455468264,
            -3.029442898943749,
            0.3449817636324948,
            -0.7537994616348398,
            0.7327486458751387,
            3.1200184229578154,
            -4.9415570981767365,
            -4.933437312855772,
        ];

        accuracy_test(&x, |x| x.cos(), cos);
    }

    #[test]
    fn test_cos_large() {
        let x: [f64; 8] = [
            -673.1445111913359,
            -4194.129748644129,
            2812.623289566699,
            -4882.861762124198,
            1815.6590613844326,
            -4318.334861343217,
            2955.7357268745563,
            2267.811391833918,
        ];

        accuracy_test(&x, |x| x.cos(), cos);
    }

    #[test]
    fn test_tan_small() {
        let x: [f64; 8] = [
            -4.725590455468264,
            -3.029442898943749,
            0.3449817636324948,
            -0.7537994616348398,
            0.7327486458751387,
            3.1200184229578154,
            -4.9415570981767365,
            -4.933437312855772,
        ];

        accuracy_test(&x, |x| x.tan(), tan);
    }

    #[test]
    fn test_tan_large() {
        let x: [f64; 8] = [
            -673.1445111913359,
            -4194.129748644129,
            2812.623289566699,
            -4882.861762124198,
            1815.6590613844326,
            -4318.334861343217,
            2955.7357268745563,
            2267.811391833918,
        ];

        accuracy_test(&x, |x| x.tan(), tan);
    }

    #[test]
    fn test_sin_small_simd() {
        let x: [f64; 8] = [
            -4.725590455468264,
            -3.029442898943749,
            0.3449817636324948,
            -0.7537994616348398,
            0.7327486458751387,
            3.1200184229578154,
            -4.9415570981767365,
            -4.933437312855772,
        ];

        const ITERS: usize = 1000000;

        speed_test_simd_iterated(x, |x| x.sin(), sin_simd, ITERS);
    }

    #[test]
    fn test_cos_small_simd() {
        let x: [f64; 8] = [
            -4.725590455468264,
            -3.029442898943749,
            0.3449817636324948,
            -0.7537994616348398,
            0.7327486458751387,
            3.1200184229578154,
            -4.9415570981767365,
            -4.933437312855772,
        ];

        const ITERS: usize = 1000000;

        speed_test_simd_iterated(x, |x| x.cos(), cos_simd, ITERS);
    }

    #[test]
    fn test_tan_small_simd() {
        let x: [f64; 8] = [
            -4.725590455468264,
            -3.029442898943749,
            0.3449817636324948,
            -0.7537994616348398,
            0.7327486458751387,
            3.1200184229578154,
            -4.9415570981767365,
            -4.933437312855772,
        ];

        const ITERS: usize = 1000000;

        speed_test_simd_iterated(x, |x| x.tan(), tan_simd, ITERS);
    }
}
