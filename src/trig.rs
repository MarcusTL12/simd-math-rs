use std::f64::consts::PI;

use crate::periodic_clamp;

// Actual taylor coefficients
const TAYLOR_COEFFS: [f64; 16] = [
    0.707106781186547524400844362104849039284835937688474036588339868995,
    0.707106781186547524400844362104849039284835937688474036588339868995,
    -0.353553390593273762200422181052424519642417968844237018294169934497,
    -0.117851130197757920733474060350808173214139322948079006098056644832,
    0.029462782549439480183368515087702043303534830737019751524514161208,
    0.005892556509887896036673703017540408660706966147403950304902832241,
    -0.000982092751647982672778950502923401443451161024567325050817138706,
    -0.000140298964521140381825564357560485920493023003509617864402448386,
    1.753737056514254772819554469506074006162787543870223305030604833817e-05,
    1.948596729460283080910616077228971117958652826522470338922894259797e-06,
    -1.948596729460283080910616077228971117958652826522470338922894259797e-07,
    -1.771451572236620982646014615662701016326048024111336671748085690725e-08,
    1.476209643530517485538345513052250846938373353426113893123404742270e-09,
    1.135545879638859604260265779270962189952594887250856840864157494054e-10,
    -8.111041997420425744716184137649729928232820623220406006172553528961e-12,
    -5.407361331613617163144122758433153285488547082146937337448369019307e-13,
];
// Adjusted coefficients for better accuracy across the relevant interval
// Found by matching the first 4 derivatives (+ constant) at the three points
// [-π/4, 0, π/4]
// const TAYLOR_COEFFS: [f64; 15] = [
//     0.7071067811865476,
//     0.7071067811865476,
//     -0.3535533905932738,
//     -0.11785113019775792,
//     0.02946278254943948,
//     0.005892556509840148,
//     -0.0009820927516449947,
//     -0.00014029896413393086,
//     1.753737054091264e-5,
//     1.9485954731631532e-6,
//     -1.9485959433832494e-7,
//     -1.771247677471614e-8,
//     1.4760820807878849e-9,
//     1.1189812082865404e-10,
//     -8.007434599882055e-12,
// ];

// const TAYLOR_COEFFS: [f64; 9] = [
//     -0.00018674129144960250147810671482280951737239745446659873935030,
//     2.13070555710686364320941766164255816713103604831409716623819323120e-06,
//     6.63302633312050570015643055502560394981142222531187848143404260316e-06,
//     2.66196793492499333493375636648455569757062248991398931274590897114e-08,
//     -1.11039255565993959195953786378812778607568383899905473324706196165e-07,
//     -6.21619504515468757307947645521161327720209063860598015646305331448e-10,
//     9.68244793881132063924995035178355045757900947721441517282950473502e-10,
//     1.75814361609054096350867672268760855592746632087610570664525485865e-11,
//     -8.00743459988205434622423451223813076870124755257829339225668138144e-12,
// ];

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

pub fn sin(x: f64) -> f64 {
    sin_shift(x - PI / 4.0)
}

pub fn cos(x: f64) -> f64 {
    sin_shift(x + PI / 4.0)
}

pub fn tan(x: f64) -> f64 {
    sin(x) / cos(x)
}

#[cfg(test)]
mod tests {
    use std::{f64::consts::PI, time::Instant};

    use crate::{*, trig::sin_shift};

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

        println!("{y_std:9.5?}\n{y_tlr:9.5?}");

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
    fn test_sin_small() {
        let x: [f64; 8] = [
            -4.725590455468264,
            -3.029442898943749,
            0.3449817636324948,
            -0.7537994616348398,
            0.7327486458751387,
            3.1200184229578154,
            -4.9415570981767365,
            -4.933437312855772,
        ];

        const ITERS: usize = 1000000;

        let t = Instant::now();
        let mut y_std = x.map(|x| x.sin());
        for _ in 0..ITERS {
            y_std = x.map(|x| x.sin())
        }
        let t1 = t.elapsed();

        let t = Instant::now();
        let mut y_tlr = x.map(sin);
        for _ in 0..ITERS {
            y_tlr = x.map(sin)
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
    fn test_sin_large() {
        let x: [f64; 8] = [
            -673.1445111913359,
            -4194.129748644129,
            2812.623289566699,
            -4882.861762124198,
            1815.6590613844326,
            -4318.334861343217,
            2955.7357268745563,
            2267.811391833918,
        ];

        const ITERS: usize = 1000000;

        let t = Instant::now();
        let mut y_std = x.map(|x| x.sin());
        for _ in 0..ITERS {
            y_std = x.map(|x| x.sin())
        }
        let t1 = t.elapsed();

        let t = Instant::now();
        let mut y_tlr = x.map(sin);
        for _ in 0..ITERS {
            y_tlr = x.map(sin)
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
    fn test_cos_small() {
        let x: [f64; 8] = [
            -4.725590455468264,
            -3.029442898943749,
            0.3449817636324948,
            -0.7537994616348398,
            0.7327486458751387,
            3.1200184229578154,
            -4.9415570981767365,
            -4.933437312855772,
        ];

        const ITERS: usize = 1000000;

        let t = Instant::now();
        let mut y_std = x.map(|x| x.cos());
        for _ in 0..ITERS {
            y_std = x.map(|x| x.cos())
        }
        let t1 = t.elapsed();

        let t = Instant::now();
        let mut y_tlr = x.map(cos);
        for _ in 0..ITERS {
            y_tlr = x.map(cos)
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
    fn test_cos_large() {
        let x: [f64; 8] = [
            -673.1445111913359,
            -4194.129748644129,
            2812.623289566699,
            -4882.861762124198,
            1815.6590613844326,
            -4318.334861343217,
            2955.7357268745563,
            2267.811391833918,
        ];

        const ITERS: usize = 1000000;

        let t = Instant::now();
        let mut y_std = x.map(|x| x.cos());
        for _ in 0..ITERS {
            y_std = x.map(|x| x.cos())
        }
        let t1 = t.elapsed();

        let t = Instant::now();
        let mut y_tlr = x.map(cos);
        for _ in 0..ITERS {
            y_tlr = x.map(cos)
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
    fn test_tan_small() {
        let x: [f64; 8] = [
            -4.725590455468264,
            -3.029442898943749,
            0.3449817636324948,
            -0.7537994616348398,
            0.7327486458751387,
            3.1200184229578154,
            -4.9415570981767365,
            -4.933437312855772,
        ];

        const ITERS: usize = 1000000;

        let t = Instant::now();
        let mut y_std = x.map(|x| x.tan());
        for _ in 0..ITERS {
            y_std = x.map(|x| x.tan())
        }
        let t1 = t.elapsed();

        let t = Instant::now();
        let mut y_tlr = x.map(tan);
        for _ in 0..ITERS {
            y_tlr = x.map(tan)
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
    fn test_tan_large() {
        let x: [f64; 8] = [
            -673.1445111913359,
            -4194.129748644129,
            2812.623289566699,
            -4882.861762124198,
            1815.6590613844326,
            -4318.334861343217,
            2955.7357268745563,
            2267.811391833918,
        ];

        const ITERS: usize = 1000000;

        let t = Instant::now();
        let mut y_std = x.map(|x| x.tan());
        for _ in 0..ITERS {
            y_std = x.map(|x| x.tan())
        }
        let t1 = t.elapsed();

        let t = Instant::now();
        let mut y_tlr = x.map(tan);
        for _ in 0..ITERS {
            y_tlr = x.map(tan)
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
