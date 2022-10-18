#![feature(portable_simd)]

use std::simd::{LaneCount, Simd, SupportedLaneCount};

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

pub fn exp(x: f64) -> f64 {
    const A: f64 = 0.2;
    let (u, n) = periodic_clamp(x, A);

    let expu = exp_pt1(u);
    let fac = EXP_PT2.powi(n);

    expu * fac
}

pub fn exp_pt1_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut xn = x;
    let acc = Simd::splat(0.0);
    // let mut fac = Simd

    todo!()
}

#[cfg(test)]
mod tests {
    use std::{f64::consts::PI, time::Instant};

    use super::*;

    fn print_array(a: &[f64]) {
        print!("[");
        let mut first = false;
        for x in a {
            print!("{:.2e}{}", x, if first { "" } else { ", " });
            first = false;
        }
        println!();
    }

    #[test]
    fn test_pclamp() {
        let x = -PI;
        let y = -2.71;
        let z = 3.05;

        let a = 0.2;

        println!(
            "{:?}\n{:?}\n{:?}",
            periodic_clamp(x, a),
            periodic_clamp(y, a),
            periodic_clamp(z, a)
        );
    }

    #[test]
    fn test_exp() {
        let xs = [PI * 2.0, PI, -PI * 4.0, 1.78, PI * 8.0, 0.5, 1.0, -1.0];

        const ITERS: usize = 10000000;

        let t = Instant::now();
        let mut y1 = xs.map(exp);
        for _ in 1..ITERS {
            y1 = xs.map(exp);
        }
        let t1 = t.elapsed();
        let mut y2 = xs.map(|x| x.exp());
        for _ in 1..ITERS {
            y2 = xs.map(|x| x.exp())
        }
        let t2 = t.elapsed() - t1;

        let mut diff = [0.0; 8];
        for (y, d) in y1.iter().zip(y2).map(|(a, b)| a - b).zip(&mut diff) {
            *d = y;
        }

        let mut rdiff = [0.0; 8];
        for ((a, b), c) in diff.iter().zip(&y2).zip(&mut rdiff) {
            *c = a / b;
        }

        println!("{y1:.5?} took {t1:?}\n{y2:.5?} took {t2:?}");
        print_array(&diff);
        print_array(&rdiff);
    }

    #[test]
    fn test1() {
        println!("{}", (0.2f64).exp());
    }
}
