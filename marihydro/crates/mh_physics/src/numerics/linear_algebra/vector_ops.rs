// mh_physics/src/numerics/linear_algebra/vector_ops.rs

//! 向量运算
//!
//! 提供高效的向量运算函数，这些是迭代求解器的基础。
//! 支持泛型标量类型 `S: RuntimeScalar`（f32 或 f64）。
//!
//! # 函数列表
//!
//! - [`dot`]: 点积 x·y（返回 Result）
//! - [`dot_unchecked`]: 点积（无检查版本）
//! - [`norm2`]: 二范数 ||x||₂
//! - [`axpy`]: y = α*x + y（返回 Result）
//! - [`axpy_unchecked`]: AXPY（无检查版本）
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
//! use mh_runtime::CpuBackend;
//! let backend = CpuBackend::<f64>::new();
//! let mut x = backend.alloc(3);
//! let mut y = backend.alloc(3);
//! x.copy_from_slice(&[1.0, 2.0, 3.0]);
//! y.copy_from_slice(&[4.0, 5.0, 6.0]);
//!
//! let d = dot(&backend, &x, &y)?;  // 1*4 + 2*5 + 3*6 = 32
//! let n = norm2(&backend, &x);     // sqrt(1 + 4 + 9) ≈ 3.74
//!
//! axpy(&backend, 2.0, &x, &mut y)?;  // y = [6, 9, 12]
//! ```
//!
//! # 性能优化
//!
//! - 使用 `chunks_exact` 提升向量化概率
//! - 内联提示 `#[inline(always)]` 关键路径
//! - SIMD 加速路径（AVX2/AVX-512）

use mh_runtime::{Backend, RuntimeScalar};

// ============================================================================
// 向量运算错误类型
// ============================================================================

/// 向量运算错误类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorOpError {
    /// 维度不匹配
    DimensionMismatch {
        /// x 向量长度
        x_len: usize,
        /// y 向量长度
        y_len: usize,
    },
    /// 空向量
    EmptyVector,
    /// 数值错误（NaN/Inf）
    NumericalError(String),
}

impl std::fmt::Display for VectorOpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { x_len, y_len } => {
                write!(f, "向量维度不匹配: x.len()={}, y.len()={}", x_len, y_len)
            }
            Self::EmptyVector => write!(f, "空向量"),
            Self::NumericalError(msg) => write!(f, "数值错误: {}", msg),
        }
    }
}

impl std::error::Error for VectorOpError {}

// ============================================================================
// 核心向量运算（生产级错误检查）
// ============================================================================

/// 点积 x·y（返回 Result）
/// 
/// # 参数
/// 
/// - `x`: 向量 x
/// - `y`: 向量 y
/// 
/// # 返回
/// 
/// - `Ok(S)`: 点积结果
/// - `Err(VectorOpError)`: 维度不匹配错误
/// 
/// # 示例
/// 
/// ```ignore
/// let x = vec![1.0, 2.0, 3.0];
/// let y = vec![4.0, 5.0, 6.0];
/// let result = dot(&x, &y)?;  // 32.0
/// ```
#[inline(always)]
pub fn dot<B: Backend>(
    backend: &B,
    x: &B::Buffer<B::Scalar>,
    y: &B::Buffer<B::Scalar>,
) -> Result<B::Scalar, VectorOpError> {
    if x.len() != y.len() {
        return Err(VectorOpError::DimensionMismatch {
            x_len: x.len(),
            y_len: y.len(),
        });
    }
    Ok(backend.dot(x, y))
}

/// 点积（不检查版本，用于已验证的内部调用）
/// 
/// # 安全性
/// 
/// 调用者必须确保 `x.len() == y.len()`
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
pub fn dot_unchecked<B: Backend>(
    backend: &B,
    x: &B::Buffer<B::Scalar>,
    y: &B::Buffer<B::Scalar>,
) -> B::Scalar {
    debug_assert_eq!(x.len(), y.len(), "向量维度不匹配（调试断言）");
    backend.dot(x, y)
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
pub fn norm2<B: Backend>(backend: &B, x: &B::Buffer<B::Scalar>) -> B::Scalar {
    backend.norm2(x)
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
pub fn norm_inf<B: Backend>(_backend: &B, x: &B::Buffer<B::Scalar>) -> B::Scalar {
    let slice = x.as_slice();
    let mut max_val = B::Scalar::ZERO;
    for &v in slice {
        let abs_v = v.abs();
        if abs_v > max_val {
            max_val = abs_v;
        }
    }
    max_val
}

/// AXPY: y = α*x + y（返回 Result）
/// 
/// # 参数
/// 
/// - `alpha`: 标量 α
/// - `x`: 向量 x
/// - `y`: 向量 y（将被修改）
/// 
/// # 返回
/// 
/// - `Ok(())`: 操作成功
/// - `Err(VectorOpError)`: 维度不匹配错误
#[inline(always)]
pub fn axpy<B: Backend>(
    backend: &B,
    alpha: B::Scalar,
    x: &B::Buffer<B::Scalar>,
    y: &mut B::Buffer<B::Scalar>,
) -> Result<(), VectorOpError> {
    if x.len() != y.len() {
        return Err(VectorOpError::DimensionMismatch {
            x_len: x.len(),
            y_len: y.len(),
        });
    }
    axpy_unchecked(backend, alpha, x, y);
    Ok(())
}

/// AXPY（不检查版本）
/// 
/// # 安全性
/// 
/// 调用者必须确保 `x.len() == y.len()`
#[inline(always)]
pub fn axpy_unchecked<B: Backend>(
    backend: &B,
    alpha: B::Scalar,
    x: &B::Buffer<B::Scalar>,
    y: &mut B::Buffer<B::Scalar>,
) {
    backend.axpy(alpha, x, y);
}

/// XPAY: y = x + α*y
///
/// # 参数
///
/// - `x`: 向量 x
/// - `alpha`: 标量 α
/// - `y`: 向量 y（将被修改）
///
/// # 错误处理
/// 维度不匹配立即 panic
#[inline(always)]
pub fn xpay<B: Backend>(
    x: &B::Buffer<B::Scalar>,
    alpha: B::Scalar,
    y: &mut B::Buffer<B::Scalar>,
) {
    assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    let x_slice = x.as_slice();
    let y_slice = y.as_slice_mut();
    for (yi, &xi) in y_slice.iter_mut().zip(x_slice.iter()) {
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
pub fn scale<B: Backend>(backend: &B, alpha: B::Scalar, x: &mut B::Buffer<B::Scalar>) {
    backend.scale(alpha, x);
}

/// 复制: y = x
///
/// # 参数
///
/// - `x`: 源向量
/// - `y`: 目标向量（将被覆盖）
///
/// # 错误处理
/// 维度不匹配立即 panic
#[inline(always)]
pub fn copy<B: Backend>(backend: &B, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>) {
    assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    backend.copy(x, y);
}

/// **新增**: AXPY 原地版本（内存高效）
///
/// 当 x 和 y 指向同一缓冲区时使用
#[inline(always)]
pub fn axpy_inplace<B: Backend>(
    backend: &B,
    alpha: B::Scalar,
    x: &mut B::Buffer<B::Scalar>,
    y: &mut B::Buffer<B::Scalar>,
) {
    assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    backend.axpy(alpha, x, y);
}

/// **新增**: 带边界检查的复制
///
/// # 参数
/// - `src`: 源向量
/// - `dst`: 目标向量
/// - `bound`: 最大复制长度
#[inline]
pub fn copy_bounded<B: Backend>(
    src: &B::Buffer<B::Scalar>,
    dst: &mut B::Buffer<B::Scalar>,
    bound: usize,
) {
    let n = src.len().min(dst.len()).min(bound);
    let src_slice = src.as_slice();
    let dst_slice = dst.as_slice_mut();
    dst_slice[..n].copy_from_slice(&src_slice[..n]);
}

/// 填充: x[:] = α
///
/// # 参数
///
/// - `alpha`: 填充值
/// - `x`: 向量（将被修改）
#[inline(always)]
pub fn fill<B: Backend>(alpha: B::Scalar, x: &mut B::Buffer<B::Scalar>) {
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
///
/// # 错误处理
/// 所有向量维度必须匹配，否则 panic
#[inline(always)]
pub fn linear_combination<B: Backend>(
    alpha: B::Scalar,
    x: &B::Buffer<B::Scalar>,
    beta: B::Scalar,
    y: &B::Buffer<B::Scalar>,
    z: &mut B::Buffer<B::Scalar>,
) {
    assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    assert_eq!(x.len(), z.len(), "向量化操作时维度不匹配");
    let x_slice = x.as_slice();
    let y_slice = y.as_slice();
    let z_slice = z.as_slice_mut();
    for ((zi, &xi), &yi) in z_slice.iter_mut().zip(x_slice.iter()).zip(y_slice.iter()) {
        *zi = alpha * xi + beta * yi;
    }
}

/// 向量差: z = x - y
///
/// # 参数
/// - `x`: 向量 x
/// - `y`: 向量 y
/// - `z`: 结果向量
///
/// # 错误处理
/// 维度不匹配立即 panic
#[inline(always)]
pub fn sub<B: Backend>(x: &B::Buffer<B::Scalar>, y: &B::Buffer<B::Scalar>, z: &mut B::Buffer<B::Scalar>) {
    assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    assert_eq!(x.len(), z.len(), "向量化操作时维度不匹配");
    let x_slice = x.as_slice();
    let y_slice = y.as_slice();
    let z_slice = z.as_slice_mut();
    for ((zi, &xi), &yi) in z_slice.iter_mut().zip(x_slice.iter()).zip(y_slice.iter()) {
        *zi = xi - yi;
    }
}

/// 向量和: z = x + y
///
/// # 参数
/// - `x`: 向量 x
/// - `y`: 向量 y
/// - `z`: 结果向量
///
/// # 错误处理
/// 维度不匹配立即 panic
#[inline(always)]
pub fn add<B: Backend>(x: &B::Buffer<B::Scalar>, y: &B::Buffer<B::Scalar>, z: &mut B::Buffer<B::Scalar>) {
    assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    assert_eq!(x.len(), z.len(), "向量化操作时维度不匹配");
    let x_slice = x.as_slice();
    let y_slice = y.as_slice();
    let z_slice = z.as_slice_mut();
    for ((zi, &xi), &yi) in z_slice.iter_mut().zip(x_slice.iter()).zip(y_slice.iter()) {
        *zi = xi + yi;
    }
}

/// 逐元素乘法: z = x .* y
///
/// # 参数
/// - `x`: 向量 x
/// - `y`: 向量 y
/// - `z`: 结果向量
///
/// # 错误处理
/// 维度不匹配立即 panic
#[inline(always)]
pub fn hadamard<B: Backend>(x: &B::Buffer<B::Scalar>, y: &B::Buffer<B::Scalar>, z: &mut B::Buffer<B::Scalar>) {
    assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    assert_eq!(x.len(), z.len(), "向量化操作时维度不匹配");
    let x_slice = x.as_slice();
    let y_slice = y.as_slice();
    let z_slice = z.as_slice_mut();
    for ((zi, &xi), &yi) in z_slice.iter_mut().zip(x_slice.iter()).zip(y_slice.iter()) {
        *zi = xi * yi;
    }
}

/// 逐元素除法: z = x ./ y
///
/// # 注意
/// y中元素绝对值小于 S::EPSILON 时，z对应位置设为 0
///
/// # 错误处理
/// 维度不匹配立即 panic
#[inline(always)]
pub fn hadamard_div<B: Backend>(x: &B::Buffer<B::Scalar>, y: &B::Buffer<B::Scalar>, z: &mut B::Buffer<B::Scalar>) {
    assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    assert_eq!(x.len(), z.len(), "向量化操作时维度不匹配");
    let x_slice = x.as_slice();
    let y_slice = y.as_slice();
    let z_slice = z.as_slice_mut();
    for ((zi, &xi), &yi) in z_slice.iter_mut().zip(x_slice.iter()).zip(y_slice.iter()) {
        *zi = if yi.abs() > B::Scalar::EPSILON { xi / yi } else { B::Scalar::ZERO };
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
pub fn relative_residual<B: Backend>(
    backend: &B,
    residual: &B::Buffer<B::Scalar>,
    b: &B::Buffer<B::Scalar>,
) -> B::Scalar {
    let norm_r = norm2(backend, residual);
    let norm_b = norm2(backend, b);
    if norm_b <= B::Scalar::MIN_POSITIVE {
        norm_r
    } else {
        norm_r / norm_b
    }
}

/// 缩放加法别名: y = y + alpha * x
#[inline(always)]
pub fn add_scaled<B: Backend>(
    backend: &B,
    alpha: B::Scalar,
    x: &B::Buffer<B::Scalar>,
    y: &mut B::Buffer<B::Scalar>,
) {
    assert_eq!(x.len(), y.len(), "向量化操作时维度不匹配");
    axpy_unchecked(backend, alpha, x, y);
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
    use mh_runtime::CpuBackend;

    #[test]
    fn test_dot() {
        let backend = CpuBackend::<f64>::new();
        let mut x = backend.alloc(3);
        let mut y = backend.alloc(3);
        x.copy_from_slice(&[1.0, 2.0, 3.0]);
        y.copy_from_slice(&[4.0, 5.0, 6.0]);
        let result = dot(&backend, &x, &y).unwrap();
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
        let backend = CpuBackend::<f64>::new();
        let mut x = backend.alloc(2);
        x.copy_from_slice(&[3.0, 4.0]);
        assert!((norm2(&backend, &x) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_norm_inf() {
        let backend = CpuBackend::<f64>::new();
        let mut x = backend.alloc(3);
        x.copy_from_slice(&[-5.0, 2.0, 3.0]);
        assert!((norm_inf(&backend, &x) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_axpy() {
        let backend = CpuBackend::<Scalar>::new();
        let mut x = backend.alloc(3);
        let mut y = backend.alloc(3);
        x.copy_from_slice(&[1.0, 2.0, 3.0]);
        y.copy_from_slice(&[4.0, 5.0, 6.0]);
        axpy(&backend, 2.0, &x, &mut y).unwrap();
        assert!((y[0] - 6.0).abs() < 1e-14);
        assert!((y[1] - 9.0).abs() < 1e-14);
        assert!((y[2] - 12.0).abs() < 1e-14);
    }

    #[test]
    fn test_axpy_inplace() {
        let backend = CpuBackend::<Scalar>::new();
        let mut x = backend.alloc(3);
        let mut y = backend.alloc(3);
        x.copy_from_slice(&[1.0, 2.0, 3.0]);
        y.copy_from_slice(&[4.0, 5.0, 6.0]);
        axpy_inplace(&backend, 2.0, &mut x, &mut y);
        assert!((y[0] - 6.0).abs() < 1e-14);
        assert!((y[1] - 9.0).abs() < 1e-14);
        assert!((y[2] - 12.0).abs() < 1e-14);
    }

    #[test]
    fn test_copy_bounded() {
        let backend = CpuBackend::<Scalar>::new();
        let mut src = backend.alloc(5);
        let mut dst = backend.alloc(3);
        src.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        dst.fill(0.0);
        copy_bounded(&src, &mut dst, 3);
        assert_eq!(dst.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_xpay() {
        let backend = CpuBackend::<Scalar>::new();
        let mut x = backend.alloc(3);
        let mut y = backend.alloc(3);
        x.copy_from_slice(&[1.0, 2.0, 3.0]);
        y.copy_from_slice(&[4.0, 5.0, 6.0]);
        xpay(&x, 2.0, &mut y);
        // y = x + 2*y = [1+8, 2+10, 3+12] = [9, 12, 15]
        assert!((y[0] - 9.0).abs() < 1e-14);
        assert!((y[1] - 12.0).abs() < 1e-14);
        assert!((y[2] - 15.0).abs() < 1e-14);
    }

    #[test]
    fn test_scale() {
        let backend = CpuBackend::<Scalar>::new();
        let mut x = backend.alloc(3);
        x.copy_from_slice(&[1.0, 2.0, 3.0]);
        scale(&backend, 3.0, &mut x);
        assert!((x[0] - 3.0).abs() < 1e-14);
        assert!((x[1] - 6.0).abs() < 1e-14);
        assert!((x[2] - 9.0).abs() < 1e-14);
    }

    #[test]
    fn test_copy() {
        let backend = CpuBackend::<Scalar>::new();
        let mut x = backend.alloc(3);
        let mut y = backend.alloc(3);
        x.copy_from_slice(&[1.0, 2.0, 3.0]);
        y.fill(0.0);
        copy(&backend, &x, &mut y);
        assert_eq!(y.as_slice(), x.as_slice());
    }

    #[test]
    fn test_fill() {
        let backend = CpuBackend::<Scalar>::new();
        let mut x = backend.alloc(3);
        x.copy_from_slice(&[1.0, 2.0, 3.0]);
        fill(7.0, &mut x);
        assert!(x.as_slice().iter().all(|&v| (v - 7.0).abs() < 1e-14));
    }

    #[test]
    fn test_linear_combination() {
        let backend = CpuBackend::<Scalar>::new();
        let mut x = backend.alloc(2);
        let mut y = backend.alloc(2);
        let mut z = backend.alloc(2);
        x.copy_from_slice(&[1.0, 2.0]);
        y.copy_from_slice(&[3.0, 4.0]);
        z.fill(0.0);
        linear_combination(2.0, &x, 3.0, &y, &mut z);
        // z = 2*[1,2] + 3*[3,4] = [2,4] + [9,12] = [11, 16]
        assert!((z[0] - 11.0).abs() < 1e-14);
        assert!((z[1] - 16.0).abs() < 1e-14);
    }

    #[test]
    fn test_sub_add() {
        let backend = CpuBackend::<Scalar>::new();
        let mut x = backend.alloc(2);
        let mut y = backend.alloc(2);
        let mut z = backend.alloc(2);
        x.copy_from_slice(&[5.0, 6.0]);
        y.copy_from_slice(&[2.0, 3.0]);
        z.fill(0.0);

        sub(&x, &y, &mut z);
        assert!((z[0] - 3.0).abs() < 1e-14);
        assert!((z[1] - 3.0).abs() < 1e-14);

        add(&x, &y, &mut z);
        assert!((z[0] - 7.0).abs() < 1e-14);
        assert!((z[1] - 9.0).abs() < 1e-14);
    }

    #[test]
    fn test_hadamard() {
        let backend = CpuBackend::<Scalar>::new();
        let mut x = backend.alloc(2);
        let mut y = backend.alloc(2);
        let mut z = backend.alloc(2);
        x.copy_from_slice(&[2.0, 3.0]);
        y.copy_from_slice(&[4.0, 5.0]);
        z.fill(0.0);
        hadamard(&x, &y, &mut z);
        assert!((z[0] - 8.0).abs() < 1e-14);
        assert!((z[1] - 15.0).abs() < 1e-14);
    }

    #[test]
    fn test_hadamard_div() {
        let backend = CpuBackend::<Scalar>::new();
        let mut x = backend.alloc(3);
        let mut y = backend.alloc(3);
        let mut z = backend.alloc(3);
        x.copy_from_slice(&[8.0, 15.0, 1.0]);
        y.copy_from_slice(&[2.0, 3.0, 0.0]);
        z.fill(0.0);
        hadamard_div(&x, &y, &mut z);
        assert!((z[0] - 4.0).abs() < 1e-14);
        assert!((z[1] - 5.0).abs() < 1e-14);
        assert!(z[2].abs() < 1e-14); // 除零保护
    }

    #[test]
    fn test_relative_residual() {
        let backend = CpuBackend::<Scalar>::new();
        let mut r = backend.alloc(2);
        let mut b = backend.alloc(2);
        r.copy_from_slice(&[0.1, 0.1]);
        b.copy_from_slice(&[1.0, 1.0]);
        let rel = relative_residual(&backend, &r, &b);
        // ||r|| = sqrt(0.02) ≈ 0.1414
        // ||b|| = sqrt(2) ≈ 1.414
        // rel ≈ 0.1
        assert!((rel - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_dot_dimension_mismatch() {
        let backend = CpuBackend::<Scalar>::new();
        let mut x = backend.alloc(2);
        let mut y = backend.alloc(3);
        x.copy_from_slice(&[1.0, 2.0]);
        y.copy_from_slice(&[1.0, 2.0, 3.0]);
        assert!(dot(&backend, &x, &y).is_err());
    }
}