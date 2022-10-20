use crate::polyval;

// Domain: 0 <= x <= 0.25
const TAYLOR: [f64; 12] = [
    1.4249583610776878e-5,
    9.128894812417154e-5,
    -0.0005608352726600329,
    0.0015816919558857214,
    -0.002984804150564612,
    0.003521396901036374,
    0.0005351473351693737,
    -0.017507834130453707,
    0.06272251067722215,
    -0.1597881665449233,
    0.33783783783783783,
    0.9505468408120752,
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

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::atan;

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
