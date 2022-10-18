use std::simd::{
    LaneCount, Simd, SimdFloat, SimdInt, SimdPartialEq, StdFloat,
    SupportedLaneCount,
};

const EXP_PT2: f64 = 1.2214027581601698;

const INV_FAC: [f64; 10] = [
    1.0,
    0.5,
    0.16666666666666666,
    0.041666666666666664,
    0.008333333333333333,
    0.001388888888888889,
    0.0001984126984126984,
    2.48015873015873e-5,
    2.7557319223985893e-6,
    2.755731922398589e-7,
];

pub fn periodic_clamp(x: f64, a: f64) -> (f64, i32) {
    let n = unsafe { (x / a + 0.5 * x.signum()).to_int_unchecked() };
    (x - (n as f64) * a, n)
}

pub fn exp_pt1(x: f64) -> f64 {
    let mut xn = x;
    let mut acc = 1.0;

    for f in INV_FAC {
        acc += xn * f;
        xn *= x;
    }

    acc
}

pub fn powi(x: f64, n: i32) -> f64 {
    let mut x = if n < 0 { x.recip() } else { x };
    let mut n = n.abs();

    let mut acc = 1.0;

    while n != 0 {
        acc = if n & 1 != 0 { acc * x } else { acc };

        x *= x;
        n >>= 1;
    }

    acc
}

pub fn exp(x: f64) -> f64 {
    const A: f64 = 0.2;
    let (u, n) = periodic_clamp(x, A);

    let expu = exp_pt1(u);
    let fac = powi(EXP_PT2, n);

    expu * fac
}

#[inline(always)]
pub fn periodic_clamp_simd<const LANES: usize>(
    x: Simd<f64, LANES>,
    a: f64,
) -> (Simd<f64, LANES>, Simd<i32, LANES>)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let n = unsafe {
        (x / Simd::splat(a) + Simd::splat(0.5) * x.signum()).to_int_unchecked()
    };
    (x - (n.cast()) * Simd::splat(a), n)
}

#[inline(always)]
pub fn exp_pt1_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut xn = x;
    let mut acc = Simd::splat(1.0);

    for f in INV_FAC {
        acc = xn.mul_add(Simd::splat(f), acc);
        xn *= x;
    }

    acc
}

#[inline(always)]
pub fn powi_simd<const LANES: usize>(
    x: Simd<f64, LANES>,
    n: Simd<i32, LANES>,
) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut x = n.is_negative().cast().select(x.recip(), x);
    let mut n: Simd<u64, LANES> = n.abs().cast();

    let mut acc = Simd::splat(1.0);

    while !n.simd_eq(Simd::splat(0)).all() {
        acc = (n & Simd::splat(1))
            .simd_eq(Simd::splat(0))
            .select(acc, acc * x);

        x *= x;
        n >>= Simd::splat(1);
    }

    acc
}

#[inline(always)]
pub fn exp_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    const A: f64 = 0.2;
    let (u, n) = periodic_clamp_simd(x, A);

    let expu = exp_pt1_simd(u);
    let fac = powi_simd(Simd::splat(EXP_PT2), n);

    expu * fac
}
