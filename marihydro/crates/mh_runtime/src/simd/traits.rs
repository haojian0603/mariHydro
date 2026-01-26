// crates/mh_runtime/src/simd/traits.rs

//! SIMD 操作 trait 抽象

use crate::RuntimeScalar;

/// SIMD 向量化操作 trait
///
/// 定义可向量化的数值操作接口
pub trait SimdOps: RuntimeScalar {
    /// 向量化 AXPY: y = alpha * x + y
    fn simd_axpy(alpha: Self, x: &[Self], y: &mut [Self]);

    /// 向量化点积
    fn simd_dot(x: &[Self], y: &[Self]) -> Self;

    /// 向量化求和
    fn simd_sum(x: &[Self]) -> Self;

    /// 向量化最大值
    fn simd_max(x: &[Self]) -> Self;

    /// 向量化最小值
    fn simd_min(x: &[Self]) -> Self;

    /// 向量化缩放: x = alpha * x
    fn simd_scale(alpha: Self, x: &mut [Self]);

    /// 向量化强制正性
    fn simd_enforce_positivity(x: &mut [Self], min_val: Self);

    /// 向量化平方根
    fn simd_sqrt(x: &[Self], out: &mut [Self]);

    /// 向量化 FMA: out = a * b + c
    fn simd_fma(a: &[Self], b: &[Self], c: &[Self], out: &mut [Self]);
}

// f64 标量回退实现
impl SimdOps for f64 {
    fn simd_axpy(alpha: Self, x: &[Self], y: &mut [Self]) {
        debug_assert_eq!(x.len(), y.len());
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    }

    fn simd_dot(x: &[Self], y: &[Self]) -> Self {
        debug_assert_eq!(x.len(), y.len());
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn simd_sum(x: &[Self]) -> Self {
        x.iter().sum()
    }

    fn simd_max(x: &[Self]) -> Self {
        x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    fn simd_min(x: &[Self]) -> Self {
        x.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    fn simd_scale(alpha: Self, x: &mut [Self]) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    fn simd_enforce_positivity(x: &mut [Self], min_val: Self) {
        for xi in x.iter_mut() {
            if *xi < min_val {
                *xi = min_val;
            }
        }
    }

    fn simd_sqrt(x: &[Self], out: &mut [Self]) {
        debug_assert_eq!(x.len(), out.len());
        for (o, xi) in out.iter_mut().zip(x.iter()) {
            *o = xi.sqrt();
        }
    }

    fn simd_fma(a: &[Self], b: &[Self], c: &[Self], out: &mut [Self]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        debug_assert_eq!(a.len(), out.len());
        for i in 0..a.len() {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
    }
}

// f32 标量回退实现
impl SimdOps for f32 {
    fn simd_axpy(alpha: Self, x: &[Self], y: &mut [Self]) {
        debug_assert_eq!(x.len(), y.len());
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    }

    fn simd_dot(x: &[Self], y: &[Self]) -> Self {
        debug_assert_eq!(x.len(), y.len());
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn simd_sum(x: &[Self]) -> Self {
        x.iter().sum()
    }

    fn simd_max(x: &[Self]) -> Self {
        x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    fn simd_min(x: &[Self]) -> Self {
        x.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    fn simd_scale(alpha: Self, x: &mut [Self]) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    fn simd_enforce_positivity(x: &mut [Self], min_val: Self) {
        for xi in x.iter_mut() {
            if *xi < min_val {
                *xi = min_val;
            }
        }
    }

    fn simd_sqrt(x: &[Self], out: &mut [Self]) {
        debug_assert_eq!(x.len(), out.len());
        for (o, xi) in out.iter_mut().zip(x.iter()) {
            *o = xi.sqrt();
        }
    }

    fn simd_fma(a: &[Self], b: &[Self], c: &[Self], out: &mut [Self]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), c.len());
        debug_assert_eq!(a.len(), out.len());
        for i in 0..a.len() {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_axpy_f64() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![1.0, 1.0, 1.0, 1.0];
        f64::simd_axpy(2.0, &x, &mut y);
        assert_eq!(y, vec![3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_simd_dot_f64() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let result = f64::simd_dot(&x, &y);
        assert_eq!(result, 30.0); // 1 + 4 + 9 + 16
    }

    #[test]
    fn test_simd_fma_f64() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];
        f64::simd_fma(&a, &b, &c, &mut out);
        assert_eq!(out, vec![3.0, 5.0, 7.0, 9.0]); // a*b + c
    }
}
