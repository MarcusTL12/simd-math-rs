use std::simd::{
    LaneCount, Simd, SimdFloat, SimdInt, SimdPartialEq, StdFloat,
    SupportedLaneCount,
};

#[inline(always)]
pub fn periodic_clamp(x: f64, a: f64) -> (f64, i32) {
    let n = unsafe { (x / a + 0.5 * x.signum()).to_int_unchecked() };
    (x - (n as f64) * a, n)
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
        (x / Simd::splat(a) + Simd::splat(0.5).copysign(x)).to_int_unchecked()
    };
    (x - (n.cast()) * Simd::splat(a), n)
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
pub fn polyval<const N: usize>(cs: &[f64; N], x: f64) -> f64 {
    let mut acc = cs[0];

    for &c in &cs[1..] {
        acc = x.mul_add(acc, c);
    }

    acc
}

#[inline(always)]
pub fn polyval_simd<const N: usize, const LANES: usize>(
    cs: &[f64; N],
    x: Simd<f64, LANES>,
) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut acc = Simd::splat(cs[0]);

    for &c in &cs[1..] {
        acc = x.mul_add(acc, Simd::splat(c));
    }

    acc
}

#[cfg(test)]
pub mod tests {
    use std::{f64::consts::PI, simd::{Simd, LaneCount, SupportedLaneCount}, time::Instant};

    use crate::*;

    pub fn print_array(a: &[f64]) {
        print!("[");
        let mut first = true;
        for x in a {
            print!("{}{:9.2e}", if first { "" } else { ", " }, x);
            first = false;
        }
        println!("]");
    }

    pub fn accuracy_test<F1: Fn(f64) -> f64, F2: Fn(f64) -> f64>(
        x: &[f64],
        f_std: F1,
        f_lib: F2,
    ) {
        let y_std: Vec<_> = x.iter().map(|&x| f_std(x)).collect();
        let y_lib: Vec<_> = x.iter().map(|&x| f_lib(x)).collect();

        let diff: Vec<_> =
            y_std.iter().zip(&y_lib).map(|(a, b)| a - b).collect();

        let rdiff: Vec<_> =
            diff.iter().zip(&y_std).map(|(a, b)| a / b).collect();

        let rdiff2: Vec<_> = diff.iter().zip(x).map(|(a, b)| a / b).collect();

        print!("x:     ");
        print_array(x);
        print!("y_std: ");
        print_array(&y_std);
        print!("y_lib: ");
        print_array(&y_lib);
        print!("adiff: ");
        print_array(&diff);
        print!("rodiff:");
        print_array(&rdiff);
        print!("ridiff:");
        print_array(&rdiff2);
    }

    pub fn speed_test_simd_iterated<
        const LANES: usize,
        F1: Fn(f64) -> f64 + Copy,
        F2: Fn(Simd<f64, LANES>) -> Simd<f64, LANES>,
    >(
        x: [f64; LANES],
        f_std: F1,
        f_lib: F2,
        iters: usize,
    ) where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let t = Instant::now();
        let mut y_std = x;
        for _ in 0..iters {
            y_std = y_std.map(f_std);
        }
        let t_std = t.elapsed();
        let mut y_lib = Simd::from(x);
        for _ in 0..iters {
            y_lib = f_lib(y_lib);
        }
        let y_lib = y_lib.to_array();
        let t_lib = t.elapsed() - t_std;

        let mut diff = [0.0; 8];
        for (y, d) in y_std.iter().zip(y_lib).map(|(a, b)| a - b).zip(&mut diff) {
            *d = y;
        }

        let mut rdiff = [0.0; 8];
        for ((a, b), c) in diff.iter().zip(&y_lib).zip(&mut rdiff) {
            *c = a / b;
        }

        print!("x:     ");
        print_array(&x);
        print!("y_std (took {t_std:?}):\n       ");
        print_array(&y_std);
        print!("y_lib (took {t_lib:?}):\n       ");
        print_array(&y_lib);
        print!("adiff: ");
        print_array(&diff);
        print!("rdiff: ");
        print_array(&rdiff);
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
    fn test_powi_simd() {
        let x: [f64; 8] = [
            0.17597038874508864,
            0.6961527072146775,
            0.08641903575832555,
            0.4450507584836101,
            0.17911624971729656,
            0.2544160362424094,
            0.8876640976231621,
            0.25636976570942127,
        ];

        let n = [-7, 3, 8, 10, -6, -5, 8, -1];

        let t = Instant::now();
        let mut y1 = [0.0; 8];
        for ((a, b), c) in x.iter().zip(&n).zip(&mut y1) {
            *c = a.powi(*b);
        }
        let t1 = t.elapsed();

        let t = Instant::now();
        let y2 = powi_simd(Simd::from(x), Simd::from(n)).to_array();
        let t2 = t.elapsed();

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
