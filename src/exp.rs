use std::simd::{LaneCount, Simd, StdFloat, SupportedLaneCount};

use crate::{periodic_clamp, periodic_clamp_simd, powi, powi_simd};

const EXP_PT2: f64 = 1.2214027581601698;

const TAYLOR: [f64; 10] = [
    1.0,
    0.5,
    0.166666666666666666666666666666666666666666666666666666666666666666,
    0.041666666666666666666666666666666666666666666666666666666666666666,
    0.008333333333333333333333333333333333333333333333333333333333333333,
    0.001388888888888888888888888888888888888888888888888888888888888888,
    0.000198412698412698412698412698412698412698412698412698412698412698,
    2.480158730158730158730158730158730158730158730158730158730158730158e-05,
    2.755731922398589065255731922398589065255731922398589065255731922398e-06,
    2.755731922398589065255731922398589065255731922398589065255731922398e-07,
];

fn exp_pt1(x: f64) -> f64 {
    let mut xn = x;
    let mut acc = 1.0;

    for f in TAYLOR {
        acc += xn * f;
        xn *= x;
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
fn exp_pt1_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut xn = x;
    let mut acc = Simd::splat(1.0);

    for f in TAYLOR {
        acc = xn.mul_add(Simd::splat(f), acc);
        xn *= x;
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

#[cfg(test)]
mod tests {
    use std::{f64::consts::PI, simd::Simd, time::Instant};

    use crate::*;

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
    fn test_exp() {
        let xs = [PI * 2.0, PI, -PI * 4.0, 1.78, PI * 8.0, 0.5, 1.0, -1.0];

        const ITERS: usize = 1000000;

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
    fn test_exp_simd() {
        let xs = [PI * 2.0, PI, -PI * 4.0, 1.78, PI * 8.0, 0.5, 1.0, -1.0];

        const ITERS: usize = 1000000;

        let t = Instant::now();
        let mut y1 = xs.map(|x| x.exp());
        for _ in 1..ITERS {
            y1 = xs.map(|x| x.exp());
        }
        let t1 = t.elapsed();
        let mut y2 = exp_simd(Simd::from(xs)).to_array();
        for _ in 1..ITERS {
            y2 = exp_simd(Simd::from(xs)).to_array();
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
}
