// crates/mh_physics/src/numerics/linear_algebra/vector_ops.rs

//! 向量运算（BLAS Level 1 风格）
//!
//! 提供高效的向量运算函数，这些是迭代求解器的基础。
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
#[inline]
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum()
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
pub fn norm2(x: &[f64]) -> f64 {
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
pub fn norm_inf(x: &[f64]) -> f64 {
    x.iter().map(|&v| v.abs()).fold(0.0, f64::max)
}

/// AXPY: y = α*x + y
///
/// # 参数
///
/// - `alpha`: 标量 α
/// - `x`: 向量 x
/// - `y`: 向量 y（将被修改）
#[inline]
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// XPAY: y = x + α*y
///
/// # 参数
///
/// - `x`: 向量 x
/// - `alpha`: 标量 α
/// - `y`: 向量 y（将被修改）
#[inline]
pub fn xpay(x: &[f64], alpha: f64, y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
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
#[inline]
pub fn scale(alpha: f64, x: &mut [f64]) {
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
#[inline]
pub fn copy(x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    y.copy_from_slice(x);
}

/// 填充: x[:] = α
///
/// # 参数
///
/// - `alpha`: 填充值
/// - `x`: 向量（将被修改）
#[inline]
pub fn fill(alpha: f64, x: &mut [f64]) {
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
#[inline]
pub fn linear_combination(alpha: f64, x: &[f64], beta: f64, y: &[f64], z: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), z.len());
    for ((zi, &xi), &yi) in z.iter_mut().zip(x.iter()).zip(y.iter()) {
        *zi = alpha * xi + beta * yi;
    }
}

/// 向量差: z = x - y
///
/// # 参数
///
/// - `x`: 向量 x
/// - `y`: 向量 y
/// - `z`: 结果向量（将被覆盖）
#[inline]
pub fn sub(x: &[f64], y: &[f64], z: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), z.len());
    for ((zi, &xi), &yi) in z.iter_mut().zip(x.iter()).zip(y.iter()) {
        *zi = xi - yi;
    }
}

/// 向量和: z = x + y
///
/// # 参数
///
/// - `x`: 向量 x
/// - `y`: 向量 y
/// - `z`: 结果向量（将被覆盖）
#[inline]
pub fn add(x: &[f64], y: &[f64], z: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), z.len());
    for ((zi, &xi), &yi) in z.iter_mut().zip(x.iter()).zip(y.iter()) {
        *zi = xi + yi;
    }
}

/// 逐元素乘法: z = x .* y
///
/// # 参数
///
/// - `x`: 向量 x
/// - `y`: 向量 y
/// - `z`: 结果向量（将被覆盖）
#[inline]
pub fn hadamard(x: &[f64], y: &[f64], z: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), z.len());
    for ((zi, &xi), &yi) in z.iter_mut().zip(x.iter()).zip(y.iter()) {
        *zi = xi * yi;
    }
}

/// 逐元素除法: z = x ./ y
///
/// # 参数
///
/// - `x`: 向量 x
/// - `y`: 向量 y
/// - `z`: 结果向量（将被覆盖）
///
/// # 注意
///
/// y 中元素绝对值小于 f64::EPSILON 时，z 对应位置设为 0
#[inline]
pub fn hadamard_div(x: &[f64], y: &[f64], z: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), z.len());
    for ((zi, &xi), &yi) in z.iter_mut().zip(x.iter()).zip(y.iter()) {
        *zi = if yi.abs() > f64::EPSILON { xi / yi } else { 0.0 };
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
/// 相对残差 ||r|| / ||b||，若 ||b|| <= f64::MIN_POSITIVE 则返回绝对残差 ||r||
#[inline]
pub fn relative_residual(residual: &[f64], b: &[f64]) -> f64 {
    let norm_r = norm2(residual);
    let norm_b = norm2(b);
    if norm_b <= f64::MIN_POSITIVE {
        norm_r
    } else {
        norm_r / norm_b
    }
}

/// 缩放加法: y = y + alpha * x
///
/// 与 axpy 相同，提供语义更清晰的别名
#[inline]
pub fn add_scaled(alpha: f64, x: &[f64], y: &mut [f64]) {
    axpy(alpha, x, y);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let result = dot(&x, &y);
        assert!((result - 32.0).abs() < 1e-14);
    }

    #[test]
    fn test_norm2() {
        let x = vec![3.0, 4.0];
        assert!((norm2(&x) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_norm_inf() {
        let x = vec![-5.0, 2.0, 3.0];
        assert!((norm_inf(&x) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_axpy() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];
        axpy(2.0, &x, &mut y);
        assert!((y[0] - 6.0).abs() < 1e-14);
        assert!((y[1] - 9.0).abs() < 1e-14);
        assert!((y[2] - 12.0).abs() < 1e-14);
    }

    #[test]
    fn test_xpay() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];
        xpay(&x, 2.0, &mut y);
        // y = x + 2*y = [1+8, 2+10, 3+12] = [9, 12, 15]
        assert!((y[0] - 9.0).abs() < 1e-14);
        assert!((y[1] - 12.0).abs() < 1e-14);
        assert!((y[2] - 15.0).abs() < 1e-14);
    }

    #[test]
    fn test_scale() {
        let mut x = vec![1.0, 2.0, 3.0];
        scale(3.0, &mut x);
        assert!((x[0] - 3.0).abs() < 1e-14);
        assert!((x[1] - 6.0).abs() < 1e-14);
        assert!((x[2] - 9.0).abs() < 1e-14);
    }

    #[test]
    fn test_copy() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        copy(&x, &mut y);
        assert_eq!(y, x);
    }

    #[test]
    fn test_fill() {
        let mut x = vec![1.0, 2.0, 3.0];
        fill(7.0, &mut x);
        assert!(x.iter().all(|&v| (v - 7.0).abs() < 1e-14));
    }

    #[test]
    fn test_linear_combination() {
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let mut z = vec![0.0; 2];
        linear_combination(2.0, &x, 3.0, &y, &mut z);
        // z = 2*[1,2] + 3*[3,4] = [2,4] + [9,12] = [11, 16]
        assert!((z[0] - 11.0).abs() < 1e-14);
        assert!((z[1] - 16.0).abs() < 1e-14);
    }

    #[test]
    fn test_sub_add() {
        let x = vec![5.0, 6.0];
        let y = vec![2.0, 3.0];
        let mut z = vec![0.0; 2];

        sub(&x, &y, &mut z);
        assert!((z[0] - 3.0).abs() < 1e-14);
        assert!((z[1] - 3.0).abs() < 1e-14);

        add(&x, &y, &mut z);
        assert!((z[0] - 7.0).abs() < 1e-14);
        assert!((z[1] - 9.0).abs() < 1e-14);
    }

    #[test]
    fn test_hadamard() {
        let x = vec![2.0, 3.0];
        let y = vec![4.0, 5.0];
        let mut z = vec![0.0; 2];
        hadamard(&x, &y, &mut z);
        assert!((z[0] - 8.0).abs() < 1e-14);
        assert!((z[1] - 15.0).abs() < 1e-14);
    }

    #[test]
    fn test_hadamard_div() {
        let x = vec![8.0, 15.0, 1.0];
        let y = vec![2.0, 3.0, 0.0];
        let mut z = vec![0.0; 3];
        hadamard_div(&x, &y, &mut z);
        assert!((z[0] - 4.0).abs() < 1e-14);
        assert!((z[1] - 5.0).abs() < 1e-14);
        assert!(z[2].abs() < 1e-14); // 除零保护
    }

    #[test]
    fn test_relative_residual() {
        let r = vec![0.1, 0.1];
        let b = vec![1.0, 1.0];
        let rel = relative_residual(&r, &b);
        // ||r|| = sqrt(0.02) ≈ 0.1414
        // ||b|| = sqrt(2) ≈ 1.414
        // rel ≈ 0.1
        assert!((rel - 0.1).abs() < 0.01);
    }
}
