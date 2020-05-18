use packed_simd::f32x16;
extern crate rayon;
use rayon::prelude::*;

pub fn sigmoid(x: f32) -> f32 {
	let denominator = 1.0 + (-x).exp();
	1.0 / denominator
}

pub fn dot_prod(vec1: &[f32], vec2: &[f32], dim: u64) -> f32 {
    //let prod1 = efficient_dot(vec1, vec1);
    let prod2 = unrolled_dot_product(vec1, vec2, dim);
    //let prod3 = parallel_dot(vec1, vec1);
    //println!("Prods: {} {} {}", prod1, prod2, prod3);
    prod2
}
// DIFFERENT IMPLEMENTATIONS OF THE DOT PRODUCT
// Only works with multiplies of 16, panics otherwise
fn parallel_dot(x: &[f32], y: &[f32]) -> f32 {
    let res: f32 = x
        .par_chunks(16)
        .map(f32x16::from_slice_unaligned)
        .zip(y.par_chunks(16).map(f32x16::from_slice_unaligned))
        .map(|(a, b)| a * b)
        .sum::<f32x16>()
        .sum();
    res
}

// Fast but ad hoc
fn unrolled_dot_product(x: &[f32], y: &[f32], dim: u64) -> f32 {
    let n = dim as usize;
    let (mut x, mut y) = (&x[..n], &y[..n]);

    let mut sum = 0.0;
    while x.len() >= 16 {
        sum += x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
             + x[4] * y[4] + x[5] * y[5] + x[6] * y[6] + x[7] * y[7]
             + x[8] * y[8] + x[9] * y[9] + x[10] * y[10] + x[11] * y[11]
             + x[12] * y[12] + x[13] * y[13] + x[14] * y[14] + x[15] * y[15];
        x = &x[16..];
        y = &y[16..];
    }

    // Take care of any left over elements (if len is not divisible by 8).
    x.iter().zip(y.iter()).fold(sum, |sum, (&ex, &ey)| sum + (ex * ey))
}

// Only works with multiplies of 16, doesn't panic
fn efficient_dot(x: &[f32], y: &[f32]) -> f32 {

    let res: f32 = x
        .chunks_exact(16)
        .map(f32x16::from_slice_unaligned)
        .zip(y.chunks_exact(16).map(f32x16::from_slice_unaligned))
        .map(|(a, b)| a * b)
        .sum::<f32x16>()
        .sum();
    res
}
