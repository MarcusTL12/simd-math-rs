use std::simd::{LaneCount, Simd, SimdFloat, SupportedLaneCount};

use crate::{polyval, polyval_simd};

// Domain: 0 <= x <= 0.25
const TAYLOR: [f64; 15] = [
    -0.05935190303616799,
    0.10193283675657623,
    -0.005670542509580847,
    -0.09038859634762113,
    6.982378651457376e-5,
    0.11108220495522635,
    4.222287252074105e-6,
    -0.14285747979512672,
    1.4572412077765227e-8,
    0.19999999973142432,
    0.0,
    -0.3333333333333333,
    0.0,
    1.0,
    0.0,
];

const TAN_4: f64 = 0.24497866312686414;
const TAN_2: f64 = 0.4636476090008061;
const TAN_1: f64 = 0.7853981633974483;

pub fn atan(x: f64) -> f64 {
    fn s(x: f64, n: i32) -> f64 {
        let f2 = 2f64.powi(-n);
        (x - f2) / (1.0 + f2 * x)
    }

    let s0 = x;
    let x0 = s0.abs();

    let s1 = s(x0, 0);
    let x1 = s1.abs(); // in [0, 1]

    let s2 = s(x1, 1);
    let x2 = s2.abs(); // in [0, 0.5]

    let s3 = s(x2, 2);
    let x3 = s3.abs(); // in [0, 0.25]

    let atx3 = polyval(&TAYLOR, x3);

    let p3 = atx3.copysign(s3) + TAN_4;
    let p2 = p3.copysign(s2) + TAN_2;
    let p1 = p2.copysign(s1) + TAN_1;
    let p0 = p1.copysign(s0);

    p0
}

#[inline(always)]
pub fn atan_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let s = |x, n: i32| {
        let f2 = Simd::splat(2f64.powi(-n));

        (x - f2) / (Simd::splat(1.0) + f2 * x)
    };

    let s0 = x;
    let x0 = s0.abs();

    let s1 = s(x0, 0);
    let x1 = s1.abs(); // in [0, 1]

    let s2 = s(x1, 1);
    let x2 = s2.abs(); // in [0, 0.5]

    let s3 = s(x2, 2);
    let x3 = s3.abs(); // in [0, 0.25]

    let atx3 = polyval_simd(&TAYLOR, x3);

    let p3 = atx3.copysign(s3) + Simd::splat(TAN_4);
    let p2 = p3.copysign(s2) + Simd::splat(TAN_2);
    let p1 = p2.copysign(s1) + Simd::splat(TAN_1);
    let p0 = p1.copysign(s0);

    p0
}

#[cfg(test)]
mod tests {
    use crate::{
        tests::{accuracy_test, speed_test_simd_iterated},
        *,
    };

    #[test]
    fn test_atan() {
        let x: [f64; 8] = [
            -6.470329170669899,
            7.608185328297425,
            3.03226005318477,
            1.6990497408119154,
            -5.422265238742455,
            -3.9968940734442704,
            5.683523314814294,
            -4.318069330898973,
        ];

        accuracy_test(&x, |x| x.atan(), atan);
    }

    #[test]
    fn test_atan_simd() {
        let x: [f64; 8] = [
            -6.470329170669899,
            7.608185328297425,
            3.03226005318477,
            1.6990497408119154,
            -5.422265238742455,
            -3.9968940734442704,
            5.683523314814294,
            -4.318069330898973,
        ];

        const ITERS: usize = 10000000;

        speed_test_simd_iterated(x, |x| x.atan(), atan_simd, ITERS);
    }
}
