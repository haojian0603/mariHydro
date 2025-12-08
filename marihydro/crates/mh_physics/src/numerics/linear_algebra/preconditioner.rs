// crates/mh_physics/src/numerics/linear_algebra/preconditioner.rs

//! 预条件器模块
//!
//! 预条件器用于加速迭代求解器的收敛。核心思想是将原问题 Ax = b
//! 转换为条件数更好的问题 M⁻¹Ax = M⁻¹b。
//!
//! # 预条件器类型
//!
//! - [`IdentityPreconditioner`]: 恒等预条件器（无预条件）
//! - [`JacobiPreconditioner`]: Jacobi 预条件器（对角预条件）
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::numerics::linear_algebra::{
//!     CsrMatrix, JacobiPreconditioner, Preconditioner,
//! };
//!
//! let matrix: CsrMatrix = /* ... */;
//! let precond = JacobiPreconditioner::from_matrix(&matrix);
//!
//! let r = vec![1.0, 2.0, 3.0];
//! let mut z = vec![0.0; 3];
//! precond.apply(&r, &mut z);  // z = M⁻¹ * r
//! ```

use super::csr::CsrMatrix;
use mh_foundation::Scalar;

/// 预条件器 trait
///
/// 预条件器的核心操作是 `apply`: z = M⁻¹ * r
pub trait Preconditioner: Send + Sync {
    /// 应用预条件器: z = M⁻¹ * r
    ///
    /// # 参数
    ///
    /// - `r`: 输入向量（通常是残差）
    /// - `z`: 输出向量（预条件后的方向）
    fn apply(&self, r: &[Scalar], z: &mut [Scalar]);

    /// 获取预条件器名称
    fn name(&self) -> &'static str;
}

/// 恒等预条件器（无预条件）
///
/// M = I，即 z = r
#[derive(Debug, Clone, Default)]
pub struct IdentityPreconditioner;

impl IdentityPreconditioner {
    /// 创建恒等预条件器
    pub fn new() -> Self {
        Self
    }
}

impl Preconditioner for IdentityPreconditioner {
    fn apply(&self, r: &[Scalar], z: &mut [Scalar]) {
        z.copy_from_slice(r);
    }

    fn name(&self) -> &'static str {
        "Identity"
    }
}

/// Jacobi 预条件器（对角预条件）
///
/// M = diag(A)，即 z_i = r_i / A_ii
///
/// 这是最简单的预条件器，计算开销极低，但效果有限。
/// 适用于对角占优矩阵。
#[derive(Debug, Clone)]
pub struct JacobiPreconditioner {
    /// 对角元素的倒数
    inv_diag: Vec<Scalar>,
}

impl JacobiPreconditioner {
    /// 从 CSR 矩阵创建 Jacobi 预条件器
    pub fn from_matrix(matrix: &CsrMatrix) -> Self {
        let n = matrix.n_rows();
        let mut inv_diag = vec![1.0; n];

        for i in 0..n {
            if let Some(diag) = matrix.diagonal_value(i) {
                if diag.abs() > 1e-14 {
                    inv_diag[i] = 1.0 / diag;
                }
            }
        }

        Self { inv_diag }
    }

    /// 从对角向量创建 Jacobi 预条件器
    pub fn from_diagonal(diag: &[Scalar]) -> Self {
        let inv_diag: Vec<_> = diag
            .iter()
            .map(|&d| if d.abs() > 1e-14 { 1.0 / d } else { 1.0 })
            .collect();
        Self { inv_diag }
    }

    /// 更新预条件器（矩阵值变化但结构不变时）
    pub fn update(&mut self, matrix: &CsrMatrix) {
        for i in 0..self.inv_diag.len().min(matrix.n_rows()) {
            if let Some(diag) = matrix.diagonal_value(i) {
                if diag.abs() > 1e-14 {
                    self.inv_diag[i] = 1.0 / diag;
                }
            }
        }
    }

    /// 获取对角元素倒数引用
    pub fn inv_diagonal(&self) -> &[Scalar] {
        &self.inv_diag
    }
}

impl Preconditioner for JacobiPreconditioner {
    fn apply(&self, r: &[Scalar], z: &mut [Scalar]) {
        debug_assert_eq!(r.len(), z.len());
        debug_assert_eq!(r.len(), self.inv_diag.len());

        for ((zi, &ri), &inv_d) in z.iter_mut().zip(r.iter()).zip(self.inv_diag.iter()) {
            *zi = ri * inv_d;
        }
    }

    fn name(&self) -> &'static str {
        "Jacobi"
    }
}

/// SSOR 预条件器（对称逐次超松弛）
///
/// M = (D + ωL) D⁻¹ (D + ωU)
///
/// 其中 L、U 分别是 A 的严格下三角和严格上三角部分，
/// D 是对角部分，ω 是松弛因子。
#[derive(Debug, Clone)]
pub struct SsorPreconditioner {
    /// 矩阵引用（用于前向和后向扫描）
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<Scalar>,
    /// 对角元素
    diag: Vec<Scalar>,
    /// 松弛因子
    omega: Scalar,
    /// 临时工作向量
    work: Vec<Scalar>,
}

impl SsorPreconditioner {
    /// 从 CSR 矩阵创建 SSOR 预条件器
    ///
    /// # 参数
    ///
    /// - `matrix`: CSR 矩阵
    /// - `omega`: 松弛因子（通常取 1.0-1.8）
    pub fn from_matrix(matrix: &CsrMatrix, omega: Scalar) -> Self {
        let n = matrix.n_rows();
        let diag: Vec<_> = (0..n)
            .map(|i| matrix.diagonal_value(i).unwrap_or(1.0))
            .collect();

        Self {
            row_ptr: matrix.row_ptr().to_vec(),
            col_idx: matrix.col_idx().to_vec(),
            values: matrix.values().to_vec(),
            diag,
            omega,
            work: vec![0.0; n],
        }
    }

    /// 更新预条件器
    pub fn update(&mut self, matrix: &CsrMatrix) {
        self.values.copy_from_slice(matrix.values());
        for i in 0..self.diag.len().min(matrix.n_rows()) {
            self.diag[i] = matrix.diagonal_value(i).unwrap_or(1.0);
        }
    }

    /// 前向扫描 (D + ωL) y = r
    fn forward_sweep(&self, r: &[Scalar], y: &mut [Scalar]) {
        let n = self.diag.len();
        for i in 0..n {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            let mut sum = r[i];
            for idx in start..end {
                let j = self.col_idx[idx];
                if j < i {
                    sum -= self.omega * self.values[idx] * y[j];
                }
            }

            y[i] = sum / self.diag[i];
        }
    }

    /// 后向扫描 (D + ωU) z = D y
    fn backward_sweep(&self, y: &[Scalar], z: &mut [Scalar]) {
        let n = self.diag.len();
        for i in (0..n).rev() {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            let mut sum = self.diag[i] * y[i];
            for idx in start..end {
                let j = self.col_idx[idx];
                if j > i {
                    sum -= self.omega * self.values[idx] * z[j];
                }
            }

            z[i] = sum / self.diag[i];
        }
    }
}

impl Preconditioner for SsorPreconditioner {
    fn apply(&self, r: &[Scalar], z: &mut [Scalar]) {
        // 使用不可变借用完成所有计算
        let n = self.diag.len();

        // 前向扫描直接写入 z
        for i in 0..n {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            let mut sum = r[i];
            for idx in start..end {
                let j = self.col_idx[idx];
                if j < i {
                    sum -= self.omega * self.values[idx] * z[j];
                }
            }
            z[i] = sum / self.diag[i];
        }

        // 保存中间结果
        let y: Vec<_> = z.to_vec();

        // 后向扫描
        for i in (0..n).rev() {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            let mut sum = self.diag[i] * y[i];
            for idx in start..end {
                let j = self.col_idx[idx];
                if j > i {
                    sum -= self.omega * self.values[idx] * z[j];
                }
            }
            z[i] = sum / self.diag[i];
        }
    }

    fn name(&self) -> &'static str {
        "SSOR"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::linear_algebra::csr::CsrBuilder;

    fn create_test_matrix() -> CsrMatrix {
        let mut builder = CsrBuilder::new_square(3);
        builder.set(0, 0, 4.0);
        builder.set(0, 1, -1.0);
        builder.set(1, 0, -1.0);
        builder.set(1, 1, 4.0);
        builder.set(1, 2, -1.0);
        builder.set(2, 1, -1.0);
        builder.set(2, 2, 4.0);
        builder.build()
    }

    #[test]
    fn test_identity_preconditioner() {
        let precond = IdentityPreconditioner::new();
        let r = vec![1.0, 2.0, 3.0];
        let mut z = vec![0.0; 3];

        precond.apply(&r, &mut z);
        assert_eq!(z, r);
        assert_eq!(precond.name(), "Identity");
    }

    #[test]
    fn test_jacobi_preconditioner() {
        let matrix = create_test_matrix();
        let precond = JacobiPreconditioner::from_matrix(&matrix);

        let r = vec![4.0, 8.0, 12.0];
        let mut z = vec![0.0; 3];

        precond.apply(&r, &mut z);

        // z[i] = r[i] / diag[i] = r[i] / 4.0
        assert!((z[0] - 1.0).abs() < 1e-14);
        assert!((z[1] - 2.0).abs() < 1e-14);
        assert!((z[2] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_jacobi_from_diagonal() {
        let diag = vec![2.0, 4.0, 8.0];
        let precond = JacobiPreconditioner::from_diagonal(&diag);

        let r = vec![2.0, 4.0, 8.0];
        let mut z = vec![0.0; 3];

        precond.apply(&r, &mut z);

        assert!((z[0] - 1.0).abs() < 1e-14);
        assert!((z[1] - 1.0).abs() < 1e-14);
        assert!((z[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_jacobi_zero_diagonal() {
        let diag = vec![2.0, 0.0, 4.0]; // 中间为零
        let precond = JacobiPreconditioner::from_diagonal(&diag);

        let r = vec![2.0, 3.0, 4.0];
        let mut z = vec![0.0; 3];

        precond.apply(&r, &mut z);

        // 零对角用 1.0 替代
        assert!((z[0] - 1.0).abs() < 1e-14);
        assert!((z[1] - 3.0).abs() < 1e-14);
        assert!((z[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_ssor_preconditioner() {
        let matrix = create_test_matrix();
        let precond = SsorPreconditioner::from_matrix(&matrix, 1.0);

        let r = vec![1.0, 1.0, 1.0];
        let mut z = vec![0.0; 3];

        precond.apply(&r, &mut z);

        // SSOR 应该产生比 Jacobi 更好的结果
        // 这里只检查基本正确性
        assert!(!z[0].is_nan());
        assert!(!z[1].is_nan());
        assert!(!z[2].is_nan());
    }
}
