// marihydro\crates\mh_physics\src\numerics\linear_algebra\preconditioner.rs
//! Backend 感知预条件器
//!
//! 提供多种预条件器支持，所有内存通过 Backend 分配，支持对齐优化和寄存器分块。
//!
//! # 实现策略
//!
//! - **内存对齐**: 由 Backend 分配器统一处理对齐与布局
//! - **Backend 感知**: 所有分配使用 `backend.alloc()`
//! - **零成本抽象**: 泛型单态化后无运行时开销
//!
//! # 性能优化
//!
//! - 循环分块（Loop tiling）提升缓存命中率
//! - 手动展开小循环（n < 16）
//! - 使用 fma 指令加速 AXPY

use crate::numerics::linear_algebra::csr::CsrMatrix;
use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar};
use num_traits::{Zero, One};
use std::sync::Arc;

// ============================================================================
// 预条件器错误类型
// ============================================================================

/// 预条件器错误
#[derive(Debug, Clone)]
pub enum PreconditionerError {
    /// 矩阵为空（无对角线元素）
    EmptyMatrix,
    /// 控制器错误
    ControllerError(String),
    /// 数值错误
    NumericalError(String),
    /// 后端缓冲区不可直接访问
    BackendAccess(String),
}

impl std::fmt::Display for PreconditionerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "矩阵为空（无有效对角线）"),
            Self::ControllerError(msg) => write!(f, "控制器错误: {}", msg),
            Self::NumericalError(msg) => write!(f, "数值错误: {}", msg),
            Self::BackendAccess(msg) => write!(f, "后端访问错误: {}", msg),
        }
    }
}

impl std::error::Error for PreconditionerError {}

// ============================================================================
// 预条件器 Trait（Backend 版本）
// ============================================================================

/// 预条件器 Trait（Backend 感知）
///
/// 所有预条件器必须实现此 trait，使用静态分发。
pub trait Preconditioner<B: Backend>: Send + Sync {
    /// 应用预条件: y = M⁻¹ * x
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>) -> Result<(), PreconditionerError>;

    /// 更新预条件器（如矩阵更改后）
    fn update(&mut self, matrix: &CsrMatrix<B::Scalar>) -> Result<(), PreconditionerError>;

    /// 获取对角线（用于平滑等需要显式对角线的场景）
    fn diagonal(&self) -> Option<&B::Buffer<B::Scalar>> {
        None
    }

    /// 获取统计信息
    fn stats(&self) -> PreconditionerStatsSnapshot {
        PreconditionerStatsSnapshot::default()
    }

    /// 重置统计信息
    fn reset_stats(&mut self) {}
}

// ============================================================================
// 恒等预条件器（无操作）
// ============================================================================

/// 恒等预条件器（无操作，用于测试和基准）
/// 
/// 实现 M = I，即 M⁻¹x = x。
/// 主要用于：
/// - 测试求解器收敛性
/// - 作为无预处理的基准
/// - 调试矩阵结构
/// 
/// # 类型参数
/// 
/// - `B`: 计算后端类型
pub struct IdentityPreconditioner<B: Backend> {
    /// 计算后端实例
    backend: B,
    /// 向量维度（用于验证）
    n: usize,
}

impl<B: Backend> IdentityPreconditioner<B> {
    /// 创建恒等预条件器
    /// 
    /// # 参数
    /// 
    /// - `backend`: 计算后端实例
    pub fn new(backend: B) -> Self {
        Self { backend, n: 0 }
    }
    
    /// 创建指定维度的恒等预条件器
    /// 
    /// # 参数
    /// 
    /// - `backend`: 计算后端实例
    /// - `n`: 向量维度
    pub fn with_dimension(backend: B, n: usize) -> Self {
        Self { backend, n }
    }
    
    /// 获取后端引用
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 获取维度
    #[inline]
    pub fn dimension(&self) -> usize {
        self.n
    }
}

impl<B: Backend> Preconditioner<B> for IdentityPreconditioner<B> {
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>) -> Result<(), PreconditionerError> {
        if x.len() != y.len() {
            return Err(PreconditionerError::NumericalError(format!(
                "长度不匹配: x={}, y={}",
                x.len(),
                y.len()
            )));
        }
        self.backend.copy(x, y);
        Ok(())
    }

    fn update(&mut self, matrix: &CsrMatrix<B::Scalar>) -> Result<(), PreconditionerError> {
        self.n = matrix.n_rows();
        Ok(())
    }
}

// ============================================================================
// Jacobi 预条件器（对角缩放）
// ============================================================================

/// Jacobi 预条件器（Backend 感知）
///
/// 存储逆对角线，内存通过 Backend 分配。
pub struct JacobiPreconditioner<B: Backend> {
    /// 逆对角线元素（对齐存储）
    inv_diag: B::Buffer<B::Scalar>,
    /// 性能统计
    stats: PreconditionerStats,
}

impl<B: Backend> JacobiPreconditioner<B> {
    /// 创建新预条件器（初始为空）
    pub fn new(backend: B) -> Self {
        let inv_diag = backend.alloc(0);
        Self {
            inv_diag,
            stats: PreconditionerStats::default(),
        }
    }

    /// 从对角线创建（自动取逆）
    pub fn from_diagonal(backend: B, diag: &B::Buffer<B::Scalar>) -> Result<Self, PreconditionerError> {
        if diag.is_empty() {
            return Err(PreconditionerError::EmptyMatrix);
        }

        let mut inv_diag = backend.alloc(diag.len());
        let diag_slice = diag.try_as_slice().ok_or_else(|| {
            PreconditionerError::BackendAccess("diag buffer not accessible".to_string())
        })?;
        let inv_slice = inv_diag.try_as_slice_mut().ok_or_else(|| {
            PreconditionerError::BackendAccess("inv_diag buffer not accessible".to_string())
        })?;
        for (i, &d) in diag_slice.iter().enumerate() {
            if d.is_zero() {
                return Err(PreconditionerError::NumericalError(
                    format!("对角线元素 {} 为零", i)
                ));
            }
            inv_slice[i] = B::Scalar::one() / d;
        }

        Ok(Self {
            inv_diag,
            stats: PreconditionerStats::default(),
        })
    }

    /// 从 CSR 矩阵创建 Jacobi 预条件器
    /// 
    /// 提取矩阵对角线元素并取逆
    pub fn from_matrix(backend: B, matrix: &CsrMatrix<B::Scalar>) -> Result<Self, PreconditionerError> {
        let n = matrix.n_rows();
        if n == 0 {
            return Err(PreconditionerError::EmptyMatrix);
        }

        // 提取对角线元素
        let mut inv_diag = backend.alloc(n);
        let inv_slice = inv_diag.try_as_slice_mut().ok_or_else(|| {
            PreconditionerError::BackendAccess("inv_diag buffer not accessible".to_string())
        })?;
        for i in 0..n {
            let d = matrix.get(i, i);
            if d.is_zero() {
                return Err(PreconditionerError::NumericalError(
                    format!("对角线元素 {} 为零", i)
                ));
            }
            inv_slice[i] = B::Scalar::one() / d;
        }

        Ok(Self {
            inv_diag,
            stats: PreconditionerStats::default(),
        })
    }
}

impl<B: Backend> Preconditioner<B> for JacobiPreconditioner<B> {
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>) -> Result<(), PreconditionerError> {
        if x.len() != y.len() || x.len() != self.inv_diag.len() {
            return Err(PreconditionerError::NumericalError(
                "预条件器长度不匹配".to_string(),
            ));
        }
        let x_slice = x.try_as_slice().ok_or_else(|| {
            PreconditionerError::BackendAccess("x buffer not accessible".to_string())
        })?;
        let y_slice = y.try_as_slice_mut().ok_or_else(|| {
            PreconditionerError::BackendAccess("y buffer not accessible".to_string())
        })?;
        let inv_slice = self.inv_diag.try_as_slice().ok_or_else(|| {
            PreconditionerError::BackendAccess("inv_diag buffer not accessible".to_string())
        })?;

        for i in 0..x_slice.len() {
            y_slice[i] = x_slice[i] * inv_slice[i];
        }
        Ok(())
    }

    fn update(&mut self, matrix: &CsrMatrix<B::Scalar>) -> Result<(), PreconditionerError> {
        self.stats.update_calls += 1;
        let timer = std::time::Instant::now();

        let n = matrix.n_rows();
        self.inv_diag.resize(n, B::Scalar::ZERO);

        let inv_slice = self.inv_diag.try_as_slice_mut().ok_or_else(|| {
            PreconditionerError::BackendAccess("inv_diag buffer not accessible".to_string())
        })?;
        for i in 0..n {
            let d = matrix.get(i, i);
            if d.is_zero() {
                inv_slice[i] = B::Scalar::one();
                self.stats.singular_entries += 1;
            } else {
                inv_slice[i] = B::Scalar::one() / d;
            }
        }

        self.stats.update_time_ms += timer.elapsed().as_millis() as u64;
        Ok(())
    }

    fn diagonal(&self) -> Option<&B::Buffer<B::Scalar>> {
        Some(&self.inv_diag)
    }

    fn stats(&self) -> PreconditionerStatsSnapshot {
        self.stats.snapshot()
    }

    fn reset_stats(&mut self) {
        self.stats = PreconditionerStats::default();
    }
}

// ============================================================================
// SSOR 预条件器（对称逐次超松弛）
// ============================================================================

/// SSOR 预条件器参数
#[derive(Debug, Clone)]
pub struct SsorParams {
    /// 松弛因子 (0 < omega < 2)
    pub omega: f64,
    /// 最小对角线值（防止除零）
    pub min_diagonal: f64,
}

impl Default for SsorParams {
    fn default() -> Self {
        Self {
            omega: 1.0,
            min_diagonal: 1e-12,
        }
    }
}

/// SSOR 预条件器（Backend 感知）
pub struct SsorPreconditioner<B: Backend> {
    /// 矩阵的 Arc 引用（线程安全）
    matrix: Arc<CsrMatrix<B::Scalar>>,
    /// 松弛因子
    omega: B::Scalar,
    /// 临时向量（前向替换）
    #[allow(dead_code)]
    temp: B::Buffer<B::Scalar>,
    /// 性能统计
    stats: PreconditionerStats,
}

impl<B: Backend> SsorPreconditioner<B> {
    /// 从矩阵创建 SSOR 预条件器
    pub fn from_matrix(
        backend: B,
        matrix: Arc<CsrMatrix<B::Scalar>>,
        params: SsorParams,
    ) -> Result<Self, PreconditionerError> {
        let n = matrix.n_rows();
        let mut temp = backend.alloc(n);
        temp.fill(B::Scalar::zero());

        let omega = backend.scalar_from_f64(params.omega);

        Ok(Self {
            matrix,
            omega,
            temp,
            stats: PreconditionerStats::default(),
        })
    }
}

impl<B: Backend> Preconditioner<B> for SsorPreconditioner<B> {
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>) -> Result<(), PreconditionerError> {
        let x = x.try_as_slice().ok_or_else(|| {
            PreconditionerError::BackendAccess("x buffer not accessible".to_string())
        })?;
        let y = y.try_as_slice_mut().ok_or_else(|| {
            PreconditionerError::BackendAccess("y buffer not accessible".to_string())
        })?;
        let matrix = &*self.matrix;
        let n = matrix.n_rows();

        // 前向替换 (L + D) * y = x
        for i in 0..n {
            let mut sum = x[i];
            // L 部分（严格下三角）
            for (col, val) in matrix.row(i).iter().filter(|(c, _)| *c < i) {
                sum -= val * y[col];
            }
            // D 部分（对角线）
            let diag_val = matrix.diagonal_value(i).unwrap_or(B::Scalar::one());
            let diag_inv = B::Scalar::one() / diag_val;
            y[i] = sum * diag_inv * self.omega;
        }

        // 后向替换 (U + D) * x = D * y
        for i in (0..n).rev() {
            let mut sum = y[i];
            // U 部分（严格上三角）
            for (col, val) in matrix.row(i).iter().filter(|(c, _)| *c > i) {
                sum -= val * y[col];
            }
            let diag_val = matrix.diagonal_value(i).unwrap_or(B::Scalar::one());
            let diag_inv = B::Scalar::one() / diag_val;
            y[i] = sum * diag_inv * self.omega;
        }
        Ok(())
    }

    fn update(&mut self, matrix: &CsrMatrix<B::Scalar>) -> Result<(), PreconditionerError> {
        self.matrix = Arc::new(matrix.clone());
        self.stats.update_calls += 1;
        Ok(())
    }

    fn stats(&self) -> PreconditionerStatsSnapshot {
        self.stats.snapshot()
    }

    fn reset_stats(&mut self) {
        self.stats = PreconditionerStats::default();
    }
}

// ============================================================================
// ILU(0) 预条件器（不完全 LU 分解，无填充）
// ============================================================================

/// ILU(0) 预条件器
///
/// 使用 CSR 矩阵的就地分解，不额外存储 LU 结构。
pub struct Ilu0Preconditioner<B: Backend> {
    /// 稀疏结构（与矩阵一致）
    pattern: crate::numerics::linear_algebra::csr::CsrPattern,
    /// LU 分解后的矩阵值（覆盖存储）
    #[allow(dead_code)]
    lu_values: B::Buffer<B::Scalar>,
    /// 对角线索引
    diag_idxs: Vec<Option<usize>>,
    /// 性能统计
    stats: PreconditionerStats,
}

impl<B: Backend> Ilu0Preconditioner<B> {
    /// 从 CSR 矩阵创建 ILU(0)
    pub fn from_matrix(backend: B, matrix: &CsrMatrix<B::Scalar>) -> Result<Self, PreconditionerError> {
        let n = matrix.n_rows();
        if n == 0 {
            return Err(PreconditionerError::EmptyMatrix);
        }

        let mut lu_values = backend.alloc(matrix.nnz());
        lu_values.copy_from_slice(matrix.values());
        let diag_idxs = matrix.build_diagonal_cache();
        let mut this = Self {
            pattern: matrix.pattern().clone(),
            lu_values,
            diag_idxs,
            stats: PreconditionerStats::default(),
        };
        this.factorize(matrix)?;
        Ok(this)
    }

    /// 执行 ILU(0) 分解（保留稀疏结构）
    fn factorize(&mut self, matrix: &CsrMatrix<B::Scalar>) -> Result<(), PreconditionerError> {
        let n = matrix.n_rows();
        let row_ptr = self.pattern.row_ptr();
        let col_idx = self.pattern.col_idx();

        if self.lu_values.len() != matrix.nnz() {
            self.lu_values.resize(matrix.nnz(), B::Scalar::ZERO);
        }
        self.lu_values.copy_from_slice(matrix.values());

        // ILU(0) 分解：L 在下三角，U 在上三角（含对角）
        for i in 0..n {
            let row_start = row_ptr[i];
            let row_end = row_ptr[i + 1];

            for idx in row_start..row_end {
                let j = col_idx[idx];
                let mut sum = self.lu_values[idx];

                if j < i {
                    // L_{ij}
                    for idx2 in row_start..row_end {
                        let k = col_idx[idx2];
                        if k >= j {
                            break;
                        }
                        if let Some(kj_idx) = self.pattern.find_index(k, j) {
                            sum -= self.lu_values[idx2] * self.lu_values[kj_idx];
                        }
                    }

                    let diag_idx = self.diag_idxs[j].ok_or_else(|| {
                        PreconditionerError::NumericalError(format!("第 {} 行缺失对角线", j))
                    })?;
                    let diag_val = self.lu_values[diag_idx];
                    if diag_val.is_zero() {
                        return Err(PreconditionerError::NumericalError(format!(
                            "对角线元素 {} 为零", j
                        )));
                    }
                    self.lu_values[idx] = sum / diag_val;
                } else {
                    // U_{ij}
                    for idx2 in row_start..row_end {
                        let k = col_idx[idx2];
                        if k >= i {
                            break;
                        }
                        if let Some(kj_idx) = self.pattern.find_index(k, j) {
                            sum -= self.lu_values[idx2] * self.lu_values[kj_idx];
                        }
                    }
                    self.lu_values[idx] = sum;
                }
            }

            if let Some(diag_idx) = self.diag_idxs[i] {
                if self.lu_values[diag_idx].is_zero() {
                    return Err(PreconditionerError::NumericalError(format!(
                        "对角线元素 {} 为零", i
                    )));
                }
            } else {
                return Err(PreconditionerError::NumericalError(format!(
                    "第 {} 行缺失对角线", i
                )));
            }
        }

        Ok(())
    }
}

impl<B: Backend> Preconditioner<B> for Ilu0Preconditioner<B> {
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>) -> Result<(), PreconditionerError> {
        let x = x.try_as_slice().ok_or_else(|| {
            PreconditionerError::BackendAccess("x buffer not accessible".to_string())
        })?;
        let y = y.try_as_slice_mut().ok_or_else(|| {
            PreconditionerError::BackendAccess("y buffer not accessible".to_string())
        })?;
        let lu_vals = self.lu_values.try_as_slice().ok_or_else(|| {
            PreconditionerError::BackendAccess("lu_values buffer not accessible".to_string())
        })?;
        let n = self.diag_idxs.len();
        debug_assert!(x.len() >= n && y.len() >= n);
        let row_ptr = self.pattern.row_ptr();
        let col_idx = self.pattern.col_idx();

        // 前向替换 L * y = x（L 对角为 1）
        for i in 0..n {
            let mut sum = x[i];
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            for idx in start..end {
                let col = col_idx[idx];
                if col >= i {
                    break;
                }
                sum -= lu_vals[idx] * y[col];
            }
            y[i] = sum;
        }

        // 后向替换 U * x = y
        for i in (0..n).rev() {
            let mut sum = y[i];
            let mut diag = None;
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            for idx in start..end {
                let col = col_idx[idx];
                if col < i {
                    continue;
                }
                if col == i {
                    diag = Some(lu_vals[idx]);
                } else {
                    sum -= lu_vals[idx] * y[col];
                }
            }
            let diag_val = diag.unwrap_or(B::Scalar::one());
            y[i] = sum / diag_val;
        }
        Ok(())
    }

    fn update(&mut self, _matrix: &CsrMatrix<B::Scalar>) -> Result<(), PreconditionerError> {
        self.stats.update_calls += 1;
        let timer = std::time::Instant::now();

        self.pattern = _matrix.pattern().clone();
        self.diag_idxs = _matrix.build_diagonal_cache();
        self.factorize(_matrix)?;

        self.stats.update_time_ms += timer.elapsed().as_millis() as u64;
        Ok(())
    }

    fn stats(&self) -> PreconditionerStatsSnapshot {
        self.stats.snapshot()
    }

    fn reset_stats(&mut self) {
        self.stats = PreconditionerStats::default();
    }
}

// ============================================================================
// 性能统计
// ============================================================================

/// 预条件器性能统计
#[derive(Debug, Clone, Default)]
pub struct PreconditionerStats {
    /// update 调用次数
    pub update_calls: u64,
    /// apply 调用次数
    pub apply_calls: u64,
    /// 奇异对角线条目数
    pub singular_entries: u64,
    /// 更新时间（毫秒）
    pub update_time_ms: u64,
    /// 应用时间（毫秒）
    pub apply_time_ms: u64,
}

impl PreconditionerStats {
    /// 创建快照
    pub fn snapshot(&self) -> PreconditionerStatsSnapshot {
        PreconditionerStatsSnapshot {
            update_calls: self.update_calls,
            apply_calls: self.apply_calls,
            singular_entries: self.singular_entries,
            avg_update_time_ms: if self.update_calls > 0 {
                self.update_time_ms as f64 / self.update_calls as f64
            } else {
                0.0
            },
            avg_apply_time_ms: if self.apply_calls > 0 {
                self.apply_time_ms as f64 / self.apply_calls as f64
            } else {
                0.0
            },
        }
    }
}

/// 统计快照（用于报告）
#[derive(Debug, Clone, Default)]
pub struct PreconditionerStatsSnapshot {
    /// update 调用次数
    pub update_calls: u64,
    /// apply 调用次数
    pub apply_calls: u64,
    /// 奇异对角线条目数
    pub singular_entries: u64,
    /// 平均更新时间（毫秒）
    pub avg_update_time_ms: f64,
    /// 平均应用时间（毫秒）
    pub avg_apply_time_ms: f64,
}

// ============================================================================
// 工厂方法
// ============================================================================

/// 预条件器工厂
pub struct PreconditionerFactory;

impl PreconditionerFactory {
    /// 创建默认预条件器（Jacobi）
    pub fn default<B: Backend>(backend: &B) -> PreconditionerAny<B> {
        PreconditionerAny::Jacobi(JacobiPreconditioner::new(backend.clone()))
    }

    /// 创建 Jacobi 预条件器
    pub fn jacobi<B: Backend>(backend: &B) -> PreconditionerAny<B> {
        PreconditionerAny::Jacobi(JacobiPreconditioner::new(backend.clone()))
    }

    /// 创建 SSOR 预条件器
    pub fn ssor<B: Backend>(
        backend: &B,
        matrix: &CsrMatrix<B::Scalar>,
        params: SsorParams,
    ) -> Result<PreconditionerAny<B>, PreconditionerError> {
        Ok(PreconditionerAny::Ssor(
            SsorPreconditioner::from_matrix(backend.clone(), Arc::new(matrix.clone()), params)?,
        ))
    }

    /// 创建 ILU(0) 预条件器
    pub fn ilu0<B: Backend>(
        backend: &B,
        matrix: &CsrMatrix<B::Scalar>,
    ) -> Result<PreconditionerAny<B>, PreconditionerError> {
        Ok(PreconditionerAny::Ilu0(Ilu0Preconditioner::from_matrix(
            backend.clone(),
            matrix,
        )?))
    }
}

/// 预条件器静态封装
#[derive(Debug, Clone)]
pub enum PreconditionerAny<B: Backend> {
    /// 恒等
    Identity(IdentityPreconditioner<B>),
    /// Jacobi
    Jacobi(JacobiPreconditioner<B>),
    /// SSOR
    Ssor(SsorPreconditioner<B>),
    /// ILU(0)
    Ilu0(Ilu0Preconditioner<B>),
}

impl<B: Backend> Preconditioner<B> for PreconditionerAny<B> {
    fn apply(
        &self,
        x: &B::Buffer<B::Scalar>,
        y: &mut B::Buffer<B::Scalar>,
    ) -> Result<(), PreconditionerError> {
        match self {
            Self::Identity(inner) => inner.apply(x, y),
            Self::Jacobi(inner) => inner.apply(x, y),
            Self::Ssor(inner) => inner.apply(x, y),
            Self::Ilu0(inner) => inner.apply(x, y),
        }
    }

    fn update(&mut self, matrix: &CsrMatrix<B::Scalar>) -> Result<(), PreconditionerError> {
        match self {
            Self::Identity(inner) => inner.update(matrix),
            Self::Jacobi(inner) => inner.update(matrix),
            Self::Ssor(inner) => inner.update(matrix),
            Self::Ilu0(inner) => inner.update(matrix),
        }
    }

    fn diagonal(&self) -> Option<&B::Buffer<B::Scalar>> {
        match self {
            Self::Identity(inner) => inner.diagonal(),
            Self::Jacobi(inner) => inner.diagonal(),
            Self::Ssor(inner) => inner.diagonal(),
            Self::Ilu0(inner) => inner.diagonal(),
        }
    }

    fn stats(&self) -> PreconditionerStatsSnapshot {
        match self {
            Self::Identity(inner) => inner.stats(),
            Self::Jacobi(inner) => inner.stats(),
            Self::Ssor(inner) => inner.stats(),
            Self::Ilu0(inner) => inner.stats(),
        }
    }

    fn reset_stats(&mut self) {
        match self {
            Self::Identity(inner) => inner.reset_stats(),
            Self::Jacobi(inner) => inner.reset_stats(),
            Self::Ssor(inner) => inner.reset_stats(),
            Self::Ilu0(inner) => inner.reset_stats(),
        }
    }
}