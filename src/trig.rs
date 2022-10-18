use std::f64::consts::PI;

use crate::periodic_clamp;

const TAYLOR_COEFFS: [f64; 16] = [
    0.7071067811865476,
    0.7071067811865476,
    -0.3535533905932738,
    -0.11785113019775792,
    0.02946278254943948,
    0.005892556509887896,
    -0.0009820927516479827,
    -0.00014029896452114038,
    1.7537370565142548e-5,
    1.948596729460283e-6,
    -1.948596729460283e-7,
    -1.771451572236621e-8,
    1.4762096435305176e-9,
    1.1355458796388596e-10,
    -8.111041997420426e-12,
    -5.407361331613617e-13,
];

fn sin_taylor(x: f64) -> f64 {
    let mut xn = 1.0;
    let mut acc = 0.0;

    for c in TAYLOR_COEFFS {
        acc += xn * c;
        xn *= x;
    }

    acc
}

fn sin_shift(x: f64) -> f64 {
    let (mut u, n) = periodic_clamp(x, PI / 2.0);

    if n & 1 != 0 {
        u = -u;
    }

    let mut tl = sin_taylor(u);

    if n & 2 != 0 {
        tl = -tl;
    }

    tl
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use crate::trig::sin_shift;

    use super::sin_taylor;

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
    fn test_taylor() {
        let x = [
            -0.7853981633974483,
            -0.5609986881410345,
            -0.3365992128846207,
            -0.1121997376282069,
            0.1121997376282069,
            0.3365992128846207,
            0.5609986881410345,
            0.7853981633974483,
        ];

        let y_std = x.map(|x| (x + PI / 4.0).sin());
        let y_tlr = x.map(sin_taylor);

        println!("{y_std:.5?}\n{y_tlr:.5?}");

        let mut diff = [0.0; 8];
        for (y, d) in y_std.iter().zip(y_tlr).map(|(a, b)| a - b).zip(&mut diff)
        {
            *d = y;
        }

        let mut rdiff = [0.0; 8];
        for ((a, b), c) in diff.iter().zip(&y_std).zip(&mut rdiff) {
            *c = a / b;
        }

        print_array(&diff);
        print_array(&rdiff);
    }

    #[test]
    fn test_sin_shift() {
        let x = [
            -49.27350894907335,
            33.556609699122156,
            6.616844076093664,
            33.07858625776914,
            34.14184641616258,
            41.87091444626816,
            -5.752401808482954,
            -5.339014479417625,
        ];

        let y_std = x.map(|x| (x + PI / 4.0).sin());
        let y_tlr = x.map(sin_shift);

        println!("{y_std:.5?}\n{y_tlr:.5?}");

        let mut diff = [0.0; 8];
        for (y, d) in y_std.iter().zip(y_tlr).map(|(a, b)| a - b).zip(&mut diff)
        {
            *d = y;
        }

        let mut rdiff = [0.0; 8];
        for ((a, b), c) in diff.iter().zip(&y_std).zip(&mut rdiff) {
            *c = a / b;
        }

        print_array(&diff);
        print_array(&rdiff);
    }
}
