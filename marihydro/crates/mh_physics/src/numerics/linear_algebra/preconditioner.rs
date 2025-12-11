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

    /// 更新预条件器（矩阵值变化但结构不变时）
    ///
    /// # 参数
    ///
    /// - `matrix`: 更新后的系数矩阵
    fn update(&mut self, matrix: &CsrMatrix);
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

    fn update(&mut self, _matrix: &CsrMatrix) {
        // 恒等预条件器无需更新
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

    /// 从 CSR 矩阵创建 Jacobi 预条件器（带干单元检测）
    ///
    /// 对于对角元素小于 `h_dry * 1e-6` 的行，使用单位预条件。
    /// 这避免了干单元导致的数值不稳定。
    pub fn from_matrix_with_dry_detection(matrix: &CsrMatrix, h_dry: Scalar) -> Self {
        let n = matrix.n_rows();
        let mut inv_diag = vec![1.0; n];
        let threshold = h_dry * 1e-6;

        for i in 0..n {
            if let Some(diag) = matrix.diagonal_value(i) {
                if diag.abs() < threshold {
                    // 干单元：使用单位预条件
                    inv_diag[i] = 1.0;
                } else if diag.abs() > 1e-14 {
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

    /// 更新预条件器（带干单元检测）
    pub fn update_with_dry_detection(&mut self, matrix: &CsrMatrix, h_dry: Scalar) {
        let threshold = h_dry * 1e-6;
        for i in 0..self.inv_diag.len().min(matrix.n_rows()) {
            if let Some(diag) = matrix.diagonal_value(i) {
                if diag.abs() < threshold {
                    self.inv_diag[i] = 1.0;
                } else if diag.abs() > 1e-14 {
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

    fn update(&mut self, matrix: &CsrMatrix) {
        for i in 0..self.inv_diag.len().min(matrix.n_rows()) {
            if let Some(diag) = matrix.diagonal_value(i) {
                if diag.abs() > 1e-14 {
                    self.inv_diag[i] = 1.0 / diag;
                }
            }
        }
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
        let n = self.diag.len();

        // 前向扫描: (D + ωL) y = r
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

        // 对角缩放: y = D * (2 - ω) * y
        let scale = 2.0 - self.omega;
        for i in 0..n {
            z[i] *= self.diag[i] * scale;
        }

        // 后向扫描: (D + ωU) x = scaled_y
        for i in (0..n).rev() {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            let mut sum = z[i];
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

    fn update(&mut self, matrix: &CsrMatrix) {
        self.values.copy_from_slice(matrix.values());
        for i in 0..self.diag.len().min(matrix.n_rows()) {
            self.diag[i] = matrix.diagonal_value(i).unwrap_or(1.0);
        }
    }
}

/// ILU(0) 不完全 LU 分解预条件器
///
/// 保持原矩阵稀疏模式的不完全 LU 分解。
/// 比 Jacobi 更强但计算开销也更大。
#[derive(Debug, Clone)]
pub struct Ilu0Preconditioner {
    /// 矩阵维度
    n: usize,
    /// 行指针
    row_ptr: Vec<usize>,
    /// 列索引
    col_idx: Vec<usize>,
    /// LU 分解后的值（L 和 U 共用存储）
    lu_values: Vec<Scalar>,
    /// 对角元位置索引
    diag_ptr: Vec<usize>,
}

impl Ilu0Preconditioner {
    /// 从 CSR 矩阵创建 ILU(0) 预条件器
    ///
    /// # 参数
    ///
    /// - `matrix`: 系数矩阵
    pub fn new(matrix: &CsrMatrix) -> Self {
        let n = matrix.n_rows();
        let mut lu_values = matrix.values().to_vec();
        let row_ptr = matrix.row_ptr().to_vec();
        let col_idx = matrix.col_idx().to_vec();

        // 查找对角元位置
        let mut diag_ptr = vec![0usize; n];
        for i in 0..n {
            for k in row_ptr[i]..row_ptr[i + 1] {
                if col_idx[k] == i {
                    diag_ptr[i] = k;
                    break;
                }
            }
        }

        // 执行 ILU(0) 分解
        Self::factorize(&row_ptr, &col_idx, &mut lu_values, &diag_ptr, n);

        Self {
            n,
            row_ptr,
            col_idx,
            lu_values,
            diag_ptr,
        }
    }

    /// 执行 ILU(0) 分解
    ///
    /// 原地修改 lu 数组，使得 L 的严格下三角部分和 U 的上三角部分（含对角）
    /// 存储在同一数组中。
    ///
    /// 使用主元正则化和增长因子限制提高数值稳定性。
    fn factorize(
        row_ptr: &[usize],
        col_idx: &[usize],
        lu: &mut [Scalar],
        diag_ptr: &[usize],
        n: usize,
    ) {
        // 数值稳定性参数
        const PIVOT_TOL: Scalar = 1e-10;
        const GROWTH_LIMIT: Scalar = 1e3;

        for i in 1..n {
            // 遍历第 i 行的下三角部分 (j < i)
            for k_idx in row_ptr[i]..row_ptr[i + 1] {
                let k = col_idx[k_idx];
                if k >= i {
                    break;
                }

                // 主元正则化：避免除零
                let mut diag_k = lu[diag_ptr[k]];
                if diag_k.abs() < PIVOT_TOL {
                    diag_k = diag_k.signum() * PIVOT_TOL;
                    if diag_k == 0.0 {
                        diag_k = PIVOT_TOL;
                    }
                    lu[diag_ptr[k]] = diag_k;
                }

                // 计算因子并限制增长
                let mut factor = lu[k_idx] / diag_k;
                factor = factor.clamp(-GROWTH_LIMIT, GROWTH_LIMIT);
                lu[k_idx] = factor;

                // 更新第 i 行的其余元素
                for j_idx in (k_idx + 1)..row_ptr[i + 1] {
                    let j = col_idx[j_idx];
                    // 查找 A[k,j]
                    for m_idx in row_ptr[k]..row_ptr[k + 1] {
                        if col_idx[m_idx] == j {
                            let update = factor * lu[m_idx];
                            // 限制更新幅度
                            let limited_update = update.clamp(-GROWTH_LIMIT, GROWTH_LIMIT);
                            lu[j_idx] -= limited_update;
                            break;
                        }
                    }
                }
            }
        }
    }

    /// 前向替换: L * y = r
    fn forward_solve(&self, r: &[Scalar], y: &mut [Scalar]) {
        y.copy_from_slice(r);

        for i in 0..self.n {
            for k_idx in self.row_ptr[i]..self.diag_ptr[i] {
                let j = self.col_idx[k_idx];
                y[i] -= self.lu_values[k_idx] * y[j];
            }
        }
    }

    /// 后向替换: U * z = y
    fn backward_solve(&self, y: &[Scalar], z: &mut [Scalar]) {
        z.copy_from_slice(y);

        for i in (0..self.n).rev() {
            for k_idx in (self.diag_ptr[i] + 1)..self.row_ptr[i + 1] {
                let j = self.col_idx[k_idx];
                z[i] -= self.lu_values[k_idx] * z[j];
            }

            let diag = self.lu_values[self.diag_ptr[i]];
            if diag.abs() > 1e-14 {
                z[i] /= diag;
            }
        }
    }
}

impl Preconditioner for Ilu0Preconditioner {
    fn apply(&self, r: &[Scalar], z: &mut [Scalar]) {
        // 解 L * U * z = r
        // 分两步: L * y = r, 然后 U * z = y
        let mut y = vec![0.0; self.n];
        self.forward_solve(r, &mut y);
        self.backward_solve(&y, z);
    }

    fn name(&self) -> &'static str {
        "ILU(0)"
    }

    fn update(&mut self, matrix: &CsrMatrix) {
        // 重新复制矩阵值并重新分解
        self.lu_values.copy_from_slice(matrix.values());
        Self::factorize(
            &self.row_ptr,
            &self.col_idx,
            &mut self.lu_values,
            &self.diag_ptr,
            self.n,
        );
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
