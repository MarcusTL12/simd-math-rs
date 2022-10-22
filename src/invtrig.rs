use std::{
    f64::{consts::PI, NAN},
    simd::{
        LaneCount, Simd, SimdFloat, SimdPartialEq, StdFloat, SupportedLaneCount,
    },
};

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
        (x - f2) / f2.mul_add(x, 1.0)
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

pub fn atan2(y: f64, x: f64) -> f64 {
    if x != 0.0 {
        let atanyx = (y / x).atan();

        if x > 0.0 {
            atanyx
        } else if y.is_sign_positive() {
            atanyx + PI
        } else {
            atanyx - PI
        }
    } else if y > 0.0 {
        PI / 2.0
    } else if y < 0.0 {
        -PI / 2.0
    } else {
        NAN
    }
}

#[inline(always)]
pub fn atan_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let s = |x, n: i32| {
        let f2 = Simd::splat(2f64.powi(-n));

        (x - f2) / f2.mul_add(x, Simd::splat(1.0))
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

#[inline(always)]
pub fn atan2_simd<const LANES: usize>(
    y: Simd<f64, LANES>,
    x: Simd<f64, LANES>,
) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let atanyx = atan_simd(y / x);

    x.simd_eq(Simd::splat(0.0)).select(
        Simd::splat(PI / 2.0).copysign(y),
        atanyx
            + x.is_sign_positive()
                .select(Simd::splat(0.0), Simd::splat(PI).copysign(y)),
    )
}

#[cfg(test)]
mod tests {
    use std::simd::Simd;

    use crate::{
        tests::{
            accuracy_test, accuracy_test_simd, print_array,
            speed_test_simd_iterated,
        },
        *,
    };

    const X: [f64; 8] = [
        -6.470329170669899,
        7.608185328297425,
        3.03226005318477,
        1.6990497408119154,
        -5.422265238742455,
        -3.9968940734442704,
        5.683523314814294,
        -4.318069330898973,
    ];

    #[test]
    fn test_atan() {
        accuracy_test(&X, |x| x.atan(), atan);
    }

    #[test]
    fn test_atan_simd() {
        accuracy_test_simd(X, |x| x.atan(), |x| x.atan());
    }

    #[test]
    fn test_atan_simd_speed() {
        const ITERS: usize = 1000000;

        speed_test_simd_iterated(X, |x| x.atan(), |x| x.atan(), ITERS);
    }

    #[test]
    fn test_atan2_simd() {
        let x: [f64; 8] = [
            -3.040346321204024,
            -7.777732768220749,
            0.0,
            0.0,
            8.027398490685906,
            2.7258759638490715,
            1.031443408365491,
            9.780589683514657,
        ];
        let y: [f64; 8] = [
            3.4337362961327833,
            0.0,
            -3.9420415247878404,
            6.9871321446290535,
            0.0,
            -8.517577704672803,
            -2.9815685286318883,
            -9.880496852312818,
        ];

        let y_std: Vec<_> =
            x.iter().zip(&y).map(|(x, y)| y.atan2(*x)).collect();

        let y_lib = Simd::from(y).atan2(Simd::from(x)).to_array();

        let diff: Vec<_> =
            y_std.iter().zip(&y_lib).map(|(a, b)| a - b).collect();

        let rdiff: Vec<_> =
            diff.iter().zip(&y_std).map(|(a, b)| a / b).collect();

        print!("y:     ");
        print_array(&y);
        print!("x:     ");
        print_array(&x);
        print!("y_std: ");
        print_array(&y_std);
        print!("y_lib: ");
        print_array(&y_lib);
        print!("adiff: ");
        print_array(&diff);
        print!("rodiff:");
        print_array(&rdiff);
    }
}
