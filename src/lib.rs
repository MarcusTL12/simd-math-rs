#![feature(portable_simd)]

mod exp;
pub use exp::*;

#[cfg(test)]
mod tests {
    use std::{f64::consts::PI, simd::Simd, time::Instant};

    use super::*;

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

    #[test]
    fn test_exp_simd() {
        let xs = [PI * 2.0, PI, -PI * 4.0, 1.78, PI * 8.0, 0.5, 1.0, -1.0];

        const ITERS: usize = 10000000;

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
