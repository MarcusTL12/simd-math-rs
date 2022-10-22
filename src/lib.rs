#![feature(portable_simd)]

mod util;
pub use util::*;

mod exp;
pub use exp::*;

mod trig;
pub use trig::*;

mod invtrig;
pub use invtrig::*;

mod log;
pub use log::*;

mod simdfloatmath_trait;
pub use simdfloatmath_trait::SimdFloatMath;
