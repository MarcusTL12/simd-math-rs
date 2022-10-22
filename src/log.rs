use std::simd::{LaneCount, Simd, SimdFloat, SimdInt, SupportedLaneCount};

use crate::{polyval, polyval_simd, powi_simd};

// f(x) = ln(x + 1)
// domain: (2^(-1/4) - 1, 2^(1/4) - 1)
const TAYLOR: [f64; 18] = [
    0.05830012414910911,
    -0.0724454848037986,
    0.06745456264338053,
    -0.07071157326414279,
    0.0768621149324746,
    -0.08336128638736695,
    0.09091112804841374,
    -0.09999937157336908,
    0.1111110784241662,
    -0.12500000775209813,
    0.14285714306426145,
    -0.1666666666256715,
    0.2,
    -0.25,
    0.3333333333333333,
    -0.5,
    1.0,
    -7.458340731200207e-155,
];

const LN2: f64 = 0.6931471805599453;
const LNSQRT2: f64 = 0.34657359027997264;
const LN2POW4TH: f64 = 0.17328679513998632;

const SQRT2: f64 = 1.4142135623730951;
const SQRT2_INV: f64 = 0.7071067811865476;

const TWOPOW4TH: f64 = 1.189207115002721;
const TWOPOW4TH_INV: f64 = 0.8408964152537145;

fn fake_log2(x: f64) -> i32 {
    const MASK: u64 = 0x7ff0000000000000;

    let x: u64 = unsafe { std::mem::transmute(x) };

    let exp2 = (x & MASK) >> 52;

    (exp2 - 1023) as i32
}

pub fn ln(x: f64) -> f64 {
    let n = fake_log2(x);

    let n = if n < 0 { n + 1 } else { n };

    let x = x * 2f64.powi(-n);

    let (nsq2, fsq2) = if x > 1.0 {
        (1.0, SQRT2_INV)
    } else {
        (-1.0, SQRT2)
    };

    let x = x * fsq2;

    let (n2p4, f2p4) = if x > 1.0 {
        (1.0, TWOPOW4TH_INV)
    } else {
        (-1.0, TWOPOW4TH)
    };

    let x = x * f2p4;

    polyval(&TAYLOR, x - 1.0)
        + (n as f64) * LN2
        + nsq2 * LNSQRT2
        + n2p4 * LN2POW4TH
}

#[inline(always)]
fn fake_log2_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<i32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    const MASK: u64 = 0x7ff0000000000000;

    let x: Simd<u64, LANES> = unsafe { std::mem::transmute_copy(&x) };

    let exp2 = (x & Simd::splat(MASK)) >> Simd::splat(52);

    (exp2 - Simd::splat(1023)).cast()
}

#[inline(always)]
pub fn ln_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let n = fake_log2_simd(x);

    let n = n.is_negative().select(n + Simd::splat(1), n);

    let x = x * powi_simd(Simd::splat(2.0), -n);

    let (nsq2, fsq2) = {
        let n = Simd::splat(1.0).copysign(x);
        let f = x
            .is_sign_positive()
            .select(Simd::splat(SQRT2_INV), Simd::splat(SQRT2));

        (n, f)
    };

    let x = x * fsq2;

    let (n2p4, f2p4) = {
        let n = Simd::splat(1.0).copysign(x);
        let f = x
            .is_sign_positive()
            .select(Simd::splat(TWOPOW4TH_INV), Simd::splat(TWOPOW4TH));

        (n, f)
    };

    let x = x * f2p4;

    polyval_simd(&TAYLOR, x - Simd::splat(1.0))
        + n.cast() * Simd::splat(LN2)
        + nsq2.cast() * Simd::splat(LNSQRT2)
        + n2p4.cast() * Simd::splat(LN2POW4TH)
}

#[cfg(test)]
mod tests {
    use std::simd::Simd;

    use crate::{
        tests::{accuracy_test, speed_test_simd_iterated},
        *,
    };

    #[test]
    fn test_ln_small() {
        let x: [f64; 8] = [
            5.155388558913315,
            1963.561314768797,
            18138.072812963892,
            0.005506141006060214,
            0.8485974262673789,
            3236.7191093391725,
            0.5895235440367635,
            16.565388066382837,
        ];

        accuracy_test(&x, |x| x.ln(), ln);
    }

    #[test]
    fn test_ln_simd() {
        let x: [f64; 8] = [
            5.155388558913315,
            1963.561314768797,
            18138.072812963892,
            0.005506141006060214,
            0.8485974262673789,
            3236.7191093391725,
            0.5895235440367635,
            16.565388066382837,
        ];

        const ITERS: usize = 1000000;

        speed_test_simd_iterated(
            x,
            |x| (x + 1.0).ln(),
            |x| ln_simd(x + Simd::splat(1.0)),
            ITERS,
        );
    }
}
