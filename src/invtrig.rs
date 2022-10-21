use std::simd::{Simd, LaneCount, SupportedLaneCount, SimdFloat};

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
    use std::{time::Instant, simd::Simd};

    use crate::{atan, atan_simd};

    fn print_array(a: &[f64]) {
        print!("[");
        let mut first = true;
        for x in a {
            print!("{}{:9.2e}", if first { "" } else { ", " }, x);
            first = false;
        }
        println!("]");
    }

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

        const ITERS: usize = 1000000;

        let t = Instant::now();
        let mut y_std = x;
        for _ in 0..ITERS {
            y_std = y_std.map(|x| x.atan());
        }
        let t1 = t.elapsed();

        let t = Instant::now();
        let mut y_tlr = x;
        for _ in 0..ITERS {
            y_tlr = y_tlr.map(atan);
        }
        let t2 = t.elapsed();

        println!("{y_std:9.5?} took {t1:?}\n{y_tlr:9.5?} took {t2:?}");

        let mut diff = [0.0; 8];
        for (y, d) in y_std.iter().zip(y_tlr).map(|(a, b)| a - b).zip(&mut diff)
        {
            *d = y;
        }

        let mut rdiff = [0.0; 8];
        for ((a, b), c) in diff.iter().zip(&y_std).zip(&mut rdiff) {
            *c = a / b;
        }

        let mut rdiff2 = [0.0; 8];
        for ((a, b), c) in diff.iter().zip(&x).zip(&mut rdiff2) {
            *c = a / b;
        }

        print_array(&diff);
        print_array(&rdiff);
        print_array(&rdiff2);
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

        const ITERS: usize = 1000000;

        let t = Instant::now();
        let mut y_std = x;
        for _ in 0..ITERS {
            y_std = y_std.map(|x| x.atan());
        }
        let t1 = t.elapsed();

        let t = Instant::now();
        let mut y_tlr = Simd::from(x);
        for _ in 0..ITERS {
            y_tlr = atan_simd(y_tlr);
        }
        let y_tlr = y_tlr.to_array();
        let t2 = t.elapsed();

        println!("{y_std:9.5?} took {t1:?}\n{y_tlr:9.5?} took {t2:?}");

        let mut diff = [0.0; 8];
        for (y, d) in y_std.iter().zip(y_tlr).map(|(a, b)| a - b).zip(&mut diff)
        {
            *d = y;
        }

        let mut rdiff = [0.0; 8];
        for ((a, b), c) in diff.iter().zip(&y_std).zip(&mut rdiff) {
            *c = a / b;
        }

        let mut rdiff2 = [0.0; 8];
        for ((a, b), c) in diff.iter().zip(&x).zip(&mut rdiff2) {
            *c = a / b;
        }

        print_array(&diff);
        print_array(&rdiff);
        print_array(&rdiff2);
    }
}
