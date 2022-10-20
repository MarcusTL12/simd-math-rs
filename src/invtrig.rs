use std::f64::consts::PI;

use crate::tan;

fn atan_newton_guess(x: f64) -> f64 {
    if x > 0.9134022578162978 {
        let xr = x.recip();
        PI / 2.0 - xr + (xr * xr * xr) / 3.0
    } else {
        x
    }
}

pub fn atan(x: f64) -> f64 {
    let s = x;
    let x = x.abs();

    let mut y = atan_newton_guess(x);

    for _ in 0..5 {
        let t = tan(y);
        let dt = t.mul_add(t, 1.0);

        y -= (t - x) / dt;
    }

    y.copysign(s)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::invtrig::atan;

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

        const ITERS: usize = 100000;

        let t = Instant::now();
        let mut y_std = x.map(|x| x.atan());
        for _ in 0..ITERS {
            y_std = x.map(|x| x.atan());
        }
        let t1 = t.elapsed();

        let t = Instant::now();
        let mut y_tlr = x.map(atan);
        for _ in 0..ITERS {
            y_tlr = x.map(atan);
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
}
