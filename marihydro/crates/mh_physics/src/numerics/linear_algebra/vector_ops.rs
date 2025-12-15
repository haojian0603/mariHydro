// =============================================================================
// mh_physics/src/numerics/linear_algebra/vector_ops.rs
// =============================================================================
//! 向量运算（BLAS Level 1 风格）
//!
//! 提供高效的向量运算函数，这些是迭代求解器的基础。
//! 支持泛型标量类型 `S: RuntimeScalar`（f32 或 f64）。
//!
//! # 函数列表
//!
//! - [`dot`]: 点积 x·y
//! - [`norm2`]: 二范数 ||x||₂
//! - [`axpy`]: y = α*x + y
//! - [`xpay`]: y = x + α*y
//! - [`scale`]: x = α*x
//! - [`copy`]: y = x
//! - [`fill`]: x[:] = α
//! - [`axpy_inplace`]: 原地 y = α*x + y
//! - [`copy_bounded`]: 带边界检查的复制
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::numerics::linear_algebra::vector_ops::{dot, norm2, axpy};
//!
//! let x = vec![1.0, 2.0, 3.0];
//! let mut y = vec![4.0, 5.0, 6.0];
//!
//! let d = dot(&x, &y);  // 1*4 + 2*5 + 3*6 = 32
//! let n = norm2(&x);    // sqrt(1 + 4 + 9) ≈ 3.74
//!
//! axpy(2.0, &x, &mut y);  // y = [6, 9, 12]
//! ```
//!
//! # 性能优化
//!
//! - 使用 `chunks_exact` 提升向量化概率
//! - 内联提示 `#[inline(always)]` 关键路径
//! - SIMD 加速路径（AVX2/AVX-512）

use mh_runtime::RuntimeScalar;

/// 点积 x·y
///
/// # 参数
///
/// - `x`: 向量 x
/// - `y`: 向量 y
///
/// # 返回
///
/// 点积结果
#[inline(always)]
pub fn dot<S: RuntimeScalar>(x: &[S], y: &[S]) -> S {
    debug_assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum()
}

/// 二范数 ||x||₂
///
/// # 参数
///
/// - `x`: 向量
///
/// # 返回
///
/// 二范数
#[inline]
pub fn norm2<S: RuntimeScalar>(x: &[S]) -> S {
    dot(x, x).sqrt()
}

/// 无穷范数 ||x||∞
///
/// # 参数
///
/// - `x`: 向量
///
/// # 返回
///
/// 无穷范数（最大绝对值）
#[inline]
pub fn norm_inf<S: RuntimeScalar>(x: &[S]) -> S {
    x.iter().map(|&v| v.abs()).fold(S::ZERO, |a, b| a.max(b))
}

/// AXPY: y = α*x + y
///
/// # 参数
///
/// - `alpha`: 标量 α
/// - `x`: 向量 x
/// - `y`: 向量 y（将被修改）
#[inline(always)]
pub fn axpy<S: RuntimeScalar>(alpha: S, x: &[S], y: &mut [S]) {
    debug_assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    // 使用 chunks_exact 提升向量化概率
    const CHUNK_SIZE: usize = 4;
    for (yi, xi) in y.chunks_exact_mut(CHUNK_SIZE)
        .zip(x.chunks_exact(CHUNK_SIZE)) 
    {
        for i in 0..CHUNK_SIZE {
            yi[i] += alpha * xi[i];
        }
    }
    // 处理尾部元素
    let rem = x.len() % CHUNK_SIZE;
    if rem > 0 {
        let start = x.len() - rem;
        for i in start..x.len() {
            y[i] += alpha * x[i];
        }
    }
}

/// XPAY: y = x + α*y
///
/// # 参数
///
/// - `x`: 向量 x
/// - `alpha`: 标量 α
/// - `y`: 向量 y（将被修改）
#[inline(always)]
pub fn xpay<S: RuntimeScalar>(x: &[S], alpha: S, y: &mut [S]) {
    debug_assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = xi + alpha * *yi;
    }
}

/// 缩放: x = α*x
///
/// # 参数
///
/// - `alpha`: 标量 α
/// - `x`: 向量（将被修改）
#[inline(always)]
pub fn scale<S: RuntimeScalar>(alpha: S, x: &mut [S]) {
    for xi in x.iter_mut() {
        *xi *= alpha;
    }
}

/// 复制: y = x
///
/// # 参数
///
/// - `x`: 源向量
/// - `y`: 目标向量（将被覆盖）
#[inline(always)]
pub fn copy<S: RuntimeScalar>(x: &[S], y: &mut [S]) {
    debug_assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    y.copy_from_slice(x);
}

/// **新增**: AXPY 原地版本（内存高效）
///
/// 当 x 和 y 指向同一缓冲区时使用
#[inline(always)]
pub fn axpy_inplace<S: RuntimeScalar>(alpha: S, x: &mut [S], y: &mut [S]) {
    debug_assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    const UNROLL_FACTOR: usize = 4;
    let mut i = 0;
    while i + UNROLL_FACTOR <= x.len() {
        y[i] += alpha * x[i];
        y[i+1] += alpha * x[i+1];
        y[i+2] += alpha * x[i+2];
        y[i+3] += alpha * x[i+3];
        i += UNROLL_FACTOR;
    }
    for j in i..x.len() {
        y[j] += alpha * x[j];
    }
}

/// **新增**: 带边界检查的复制
///
/// # 参数
/// - `src`: 源向量
/// - `dst`: 目标向量
/// - `bound`: 最大复制长度
#[inline]
pub fn copy_bounded<S: RuntimeScalar>(src: &[S], dst: &mut [S], bound: usize) {
    let n = src.len().min(dst.len()).min(bound);
    dst[..n].copy_from_slice(&src[..n]);
}

/// 填充: x[:] = α
///
/// # 参数
///
/// - `alpha`: 填充值
/// - `x`: 向量（将被修改）
#[inline(always)]
pub fn fill<S: RuntimeScalar>(alpha: S, x: &mut [S]) {
    x.fill(alpha);
}

/// 线性组合: z = α*x + β*y
///
/// # 参数
///
/// - `alpha`: 标量 α
/// - `x`: 向量 x
/// - `beta`: 标量 β
/// - `y`: 向量 y
/// - `z`: 结果向量（将被覆盖）
#[inline(always)]
pub fn linear_combination<S: RuntimeScalar>(
    alpha: S, x: &[S], 
    beta: S, y: &[S], 
    z: &mut [S]
) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), z.len());
    for ((zi, &xi), &yi) in z.iter_mut().zip(x.iter()).zip(y.iter()) {
        *zi = alpha * xi + beta * yi;
    }
}

/// 向量差: z = x - y
///
/// # 参数
/// - `x`: 向量 x
/// - `y`: 向量 y
/// - `z`: 结果向量
#[inline(always)]
pub fn sub<S: RuntimeScalar>(x: &[S], y: &[S], z: &mut [S]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), z.len());
    for ((zi, &xi), &yi) in z.iter_mut().zip(x.iter()).zip(y.iter()) {
        *zi = xi - yi;
    }
}

/// 向量和: z = x + y
///
/// # 参数
/// - `x`: 向量 x
/// - `y`: 向量 y
/// - `z`: 结果向量
#[inline(always)]
pub fn add<S: RuntimeScalar>(x: &[S], y: &[S], z: &mut [S]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), z.len());
    for ((zi, &xi), &yi) in z.iter_mut().zip(x.iter()).zip(y.iter()) {
        *zi = xi + yi;
    }
}

/// 逐元素乘法: z = x .* y
///
/// # 参数
/// - `x`: 向量 x
/// - `y`: 向量 y
/// - `z`: 结果向量
#[inline(always)]
pub fn hadamard<S: RuntimeScalar>(x: &[S], y: &[S], z: &mut [S]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), z.len());
    for ((zi, &xi), &yi) in z.iter_mut().zip(x.iter()).zip(y.iter()) {
        *zi = xi * yi;
    }
}

/// 逐元素除法: z = x ./ y
///
/// # 注意
/// y中元素绝对值小于 S::EPSILON 时，z对应位置设为 0
#[inline(always)]
pub fn hadamard_div<S: RuntimeScalar>(x: &[S], y: &[S], z: &mut [S]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), z.len());
    for ((zi, &xi), &yi) in z.iter_mut().zip(x.iter()).zip(y.iter()) {
        *zi = if yi.abs() > S::EPSILON { xi / yi } else { S::ZERO };
    }
}

/// 计算残差范数的相对误差
///
/// # 参数
///
/// - `residual`: 残差向量
/// - `b`: 右端项向量
///
/// # 返回
///
/// 相对残差 ||r|| / ||b||，若 ||b|| <= S::MIN_POSITIVE 则返回绝对残差 ||r||
#[inline(always)]
pub fn relative_residual<S: RuntimeScalar>(residual: &[S], b: &[S]) -> S {
    let norm_r = norm2(residual);
    let norm_b = norm2(b);
    if norm_b <= S::MIN_POSITIVE {
        norm_r
    } else {
        norm_r / norm_b
    }
}

/// 缩放加法别名: y = y + alpha * x
#[inline(always)]
pub fn add_scaled<S: RuntimeScalar>(alpha: S, x: &[S], y: &mut [S]) {
    axpy(alpha, x, y);
}

// ============================================================================
// SIMD 加速路径（条件编译）
// ============================================================================

/// AVX2 加速的 f64 点积
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub unsafe fn dot_avx2_f64(x: &[f64], y: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_pd();
    for (a, b) in x.chunks_exact(4).zip(y.chunks_exact(4)) {
        let a_vec = _mm256_loadu_pd(a.as_ptr());
        let b_vec = _mm256_loadu_pd(b.as_ptr());
        sum = _mm256_add_pd(sum, _mm256_mul_pd(a_vec, b_vec));
    }
    // 水平求和
    let mut result = [0.0; 4];
    _mm256_storeu_pd(result.as_mut_ptr(), sum);
    let base: f64 = result.iter().sum();
    // 处理尾部
    let rem = x.len() % 4;
    let tail: f64 = x[x.len()-rem..].iter()
        .zip(&y[y.len()-rem..])
        .map(|(&a, &b)| a * b)
        .sum();
    base + tail
}

/// AVX2 加速的 f32 点积
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub unsafe fn dot_avx2_f32(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    for (a, b) in x.chunks_exact(8).zip(y.chunks_exact(8)) {
        let a_vec = _mm256_loadu_ps(a.as_ptr());
        let b_vec = _mm256_loadu_ps(b.as_ptr());
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
    }
    // 水平求和
    let mut result = [0.0; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let base: f32 = result.iter().sum();
    // 处理尾部
    let rem = x.len() % 8;
    let tail: f32 = x[x.len()-rem..].iter()
        .zip(&y[y.len()-rem..])
        .map(|(&a, &b)| a * b)
        .sum();
    base + tail
}

// ============================================================================
// 单元测试（生产级验证）
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    type Scalar = f64;

    #[test]
    fn test_dot() {
        let x: Vec<Scalar> = vec![1.0, 2.0, 3.0];
        let y: Vec<Scalar> = vec![4.0, 5.0, 6.0];
        let result = dot(&x, &y);
        assert!((result - 32.0).abs() < 1e-14);
        
        // 测试SIMD路径（如果支持）
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            let result_simd = dot_avx2_f64(&x, &y);
            assert!((result_simd - 32.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_norm2() {
        let x: Vec<Scalar> = vec![3.0, 4.0];
        assert!((norm2(&x) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_norm_inf() {
        let x: Vec<Scalar> = vec![-5.0, 2.0, 3.0];
        assert!((norm_inf(&x) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_axpy() {
        let x: Vec<Scalar> = vec![1.0, 2.0, 3.0];
        let mut y: Vec<Scalar> = vec![4.0, 5.0, 6.0];
        axpy(2.0, &x, &mut y);
        assert!((y[0] - 6.0).abs() < 1e-14);
        assert!((y[1] - 9.0).abs() < 1e-14);
        assert!((y[2] - 12.0).abs() < 1e-14);
    }

    #[test]
    fn test_axpy_inplace() {
        let mut x: Vec<Scalar> = vec![1.0, 2.0, 3.0];
        let mut y: Vec<Scalar> = vec![4.0, 5.0, 6.0];
        axpy_inplace(2.0, &mut x, &mut y);
        assert!((y[0] - 6.0).abs() < 1e-14);
        assert!((y[1] - 9.0).abs() < 1e-14);
        assert!((y[2] - 12.0).abs() < 1e-14);
    }

    #[test]
    fn test_copy_bounded() {
        let src: Vec<Scalar> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst: Vec<Scalar> = vec![0.0; 3];
        copy_bounded(&src, &mut dst, 3);
        assert_eq!(dst, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_xpay() {
        let x: Vec<Scalar> = vec![1.0, 2.0, 3.0];
        let mut y: Vec<Scalar> = vec![4.0, 5.0, 6.0];
        xpay(&x, 2.0, &mut y);
        // y = x + 2*y = [1+8, 2+10, 3+12] = [9, 12, 15]
        assert!((y[0] - 9.0).abs() < 1e-14);
        assert!((y[1] - 12.0).abs() < 1e-14);
        assert!((y[2] - 15.0).abs() < 1e-14);
    }

    #[test]
    fn test_scale() {
        let mut x: Vec<Scalar> = vec![1.0, 2.0, 3.0];
        scale(3.0, &mut x);
        assert!((x[0] - 3.0).abs() < 1e-14);
        assert!((x[1] - 6.0).abs() < 1e-14);
        assert!((x[2] - 9.0).abs() < 1e-14);
    }

    #[test]
    fn test_copy() {
        let x: Vec<Scalar> = vec![1.0, 2.0, 3.0];
        let mut y: Vec<Scalar> = vec![0.0; 3];
        copy(&x, &mut y);
        assert_eq!(y, x);
    }

    #[test]
    fn test_fill() {
        let mut x: Vec<Scalar> = vec![1.0, 2.0, 3.0];
        fill(7.0, &mut x);
        assert!(x.iter().all(|&v| (v - 7.0).abs() < 1e-14));
    }

    #[test]
    fn test_linear_combination() {
        let x: Vec<Scalar> = vec![1.0, 2.0];
        let y: Vec<Scalar> = vec![3.0, 4.0];
        let mut z: Vec<Scalar> = vec![0.0; 2];
        linear_combination(2.0, &x, 3.0, &y, &mut z);
        // z = 2*[1,2] + 3*[3,4] = [2,4] + [9,12] = [11, 16]
        assert!((z[0] - 11.0).abs() < 1e-14);
        assert!((z[1] - 16.0).abs() < 1e-14);
    }

    #[test]
    fn test_sub_add() {
        let x: Vec<Scalar> = vec![5.0, 6.0];
        let y: Vec<Scalar> = vec![2.0, 3.0];
        let mut z: Vec<Scalar> = vec![0.0; 2];

        sub(&x, &y, &mut z);
        assert!((z[0] - 3.0).abs() < 1e-14);
        assert!((z[1] - 3.0).abs() < 1e-14);

        add(&x, &y, &mut z);
        assert!((z[0] - 7.0).abs() < 1e-14);
        assert!((z[1] - 9.0).abs() < 1e-14);
    }

    #[test]
    fn test_hadamard() {
        let x: Vec<Scalar> = vec![2.0, 3.0];
        let y: Vec<Scalar> = vec![4.0, 5.0];
        let mut z: Vec<Scalar> = vec![0.0; 2];
        hadamard(&x, &y, &mut z);
        assert!((z[0] - 8.0).abs() < 1e-14);
        assert!((z[1] - 15.0).abs() < 1e-14);
    }

    #[test]
    fn test_hadamard_div() {
        let x: Vec<Scalar> = vec![8.0, 15.0, 1.0];
        let y: Vec<Scalar> = vec![2.0, 3.0, 0.0];
        let mut z: Vec<Scalar> = vec![0.0; 3];
        hadamard_div(&x, &y, &mut z);
        assert!((z[0] - 4.0).abs() < 1e-14);
        assert!((z[1] - 5.0).abs() < 1e-14);
        assert!(z[2].abs() < 1e-14); // 除零保护
    }

    #[test]
    fn test_relative_residual() {
        let r: Vec<Scalar> = vec![0.1, 0.1];
        let b: Vec<Scalar> = vec![1.0, 1.0];
        let rel = relative_residual(&r, &b);
        // ||r|| = sqrt(0.02) ≈ 0.1414
        // ||b|| = sqrt(2) ≈ 1.414
        // rel ≈ 0.1
        assert!((rel - 0.1).abs() < 0.01);
    }

    #[test]
    #[should_panic(expected = "向量化操作时维度不匹配")]
    fn test_dot_dimension_mismatch() {
        let x: Vec<Scalar> = vec![1.0, 2.0];
        let y: Vec<Scalar> = vec![1.0, 2.0, 3.0];
        dot(&x, &y);
    }
}