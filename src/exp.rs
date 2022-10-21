use std::simd::{LaneCount, Simd, SupportedLaneCount};

use crate::{
    periodic_clamp, periodic_clamp_simd, polyval, polyval_simd, powi, powi_simd,
};

const EXP_PT2: f64 = 1.2214027581601698;

const TAYLOR: [f64; 11] = [
    2.755731922398589065255731922398589065255731922398589065255731922398e-07,
    2.755731922398589065255731922398589065255731922398589065255731922398e-06,
    2.480158730158730158730158730158730158730158730158730158730158730158e-05,
    0.000198412698412698412698412698412698412698412698412698412698412698,
    0.001388888888888888888888888888888888888888888888888888888888888888,
    0.008333333333333333333333333333333333333333333333333333333333333333,
    0.041666666666666666666666666666666666666666666666666666666666666666,
    0.166666666666666666666666666666666666666666666666666666666666666666,
    0.5,
    1.0,
    1.0,
];

pub fn exp(x: f64) -> f64 {
    const A: f64 = 0.2;
    let (u, n) = periodic_clamp(x, A);

    let expu = polyval(&TAYLOR, u);
    let fac = powi(EXP_PT2, n);

    expu * fac
}

#[inline(always)]
pub fn exp_simd<const LANES: usize>(x: Simd<f64, LANES>) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    const A: f64 = 0.2;
    let (u, n) = periodic_clamp_simd(x, A);

    let expu = polyval_simd(&TAYLOR, u);
    let fac = powi_simd(Simd::splat(EXP_PT2), n);

    expu * fac
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use crate::{
        tests::{accuracy_test, speed_test_simd_iterated},
        *,
    };

    #[test]
    fn test_exp() {
        let xs = [PI * 2.0, PI, -PI * 4.0, 1.78, PI * 8.0, 0.5, 1.0, -1.0];

        accuracy_test(&xs, |x: f64| x.exp(), exp);
    }

    #[test]
    fn test_exp_simd() {
        let xs = [PI * 2.0, PI, -PI * 4.0, 1.78, PI * 8.0, 0.5, 1.0, -1.0];

        const ITERS: usize = 1000000;

        speed_test_simd_iterated(
            xs,
            |x: f64| (-x * x).exp(),
            |x| exp_simd(-x * x),
            ITERS,
        );
    }
}
