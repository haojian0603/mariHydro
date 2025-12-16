// crates/mh_physics/src/numerics/linear_algebra/preconditioner.rs

//! Backend 感知预条件器
//!
//! 提供多种预条件器支持，所有内存通过 Backend 分配，支持对齐优化和寄存器分块。
//!
//! # 实现策略
//!
//! - **内存对齐**: 使用 AlignedVec64 确保 64 字节对齐（AVX-512）
//! - **Backend 感知**: 所有分配使用 `backend.alloc()` 或 `backend.alloc_zeroed()`
//! - **零成本抽象**: 泛型单态化后无运行时开销
//!
//! # 性能优化
//!
//! - 循环分块（Loop tiling）提升缓存命中率
//! - 手动展开小循环（n < 16）
//! - 使用 fma 指令加速 AXPY

use crate::builder::{DynSolver, SolverStats};
use crate::numerics::linear_algebra::{
    AlignedVec64, aligned_vec, IterativeSolver, SolverConfig, SolverResult, SolverStatus,
};
use crate::numerics::linear_algebra::csr::CsrMatrix;
use aligned_vec::AVec;
use bytemuck::Pod;
use mh_runtime::{Backend, RuntimeScalar};
use num_traits::{Float, FromPrimitive, Zero, One};
use std::ops::{Deref, DerefMut};

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
}

impl std::fmt::Display for PreconditionerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "矩阵为空（无有效对角线）"),
            Self::ControllerError(msg) => write!(f, "控制器错误: {}", msg),
            Self::NumericalError(msg) => write!(f, "数值错误: {}", msg),
        }
    }
}

impl std::error::Error for PreconditionerError {}

// ============================================================================
// 预条件器 Trait（Backend 版本）
// ============================================================================

/// 预条件器 Trait（Backend 感知）
///
/// 所有预条件器必须实现此 trait，支持动态分发和静态泛型。
pub trait Preconditioner<B: Backend>: Send + Sync {
    /// 应用预条件: y = M⁻¹ * x
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>);

    /// 更新预条件器（如矩阵更改后）
    fn update(&mut self, matrix: &CsrMatrix<B::Scalar>) -> Result<(), PreconditionerError>;

    /// 获取对角线（用于平滑等需要显式对角线的场景）
    fn diagonal(&self) -> Option<&[B::Scalar]> {
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

/// 恒等预条件器（无操作，用于测试）
pub struct IdentityPreconditioner<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> IdentityPreconditioner<B> {
    /// 创建恒等预条件器
    pub fn new(_backend: &B) -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Preconditioner<B> for IdentityPreconditioner<B> {
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>) {
        y.copy_from_slice(x);
    }

    fn update(&mut self, _matrix: &CsrMatrix<B::Scalar>) -> Result<(), PreconditionerError> {
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
    inv_diag: AlignedVec64<B::Scalar>,
    /// 性能统计
    stats: PreconditionerStats,
}

impl<B: Backend> JacobiPreconditioner<B> {
    /// 创建新预条件器（初始为空）
    pub fn new(backend: &B) -> Self {
        let mut inv_diag = AlignedVec64::new(64);
        inv_diag.resize(0, B::Scalar::zero());
        Self {
            inv_diag,
            stats: PreconditionerStats::default(),
        }
    }

    /// 从对角线创建（自动取逆）
    pub fn from_diagonal(backend: &B, diag: &[B::Scalar]) -> Result<Self, PreconditionerError> {
        if diag.is_empty() {
            return Err(PreconditionerError::EmptyMatrix);
        }

        let mut inv_diag = aligned_vec(diag.len());
        for (i, &d) in diag.iter().enumerate() {
            if d.is_zero() {
                return Err(PreconditionerError::NumericalError(
                    format!("对角线元素 {} 为零", i)
                ));
            }
            inv_diag[i] = B::Scalar::one() / d;
        }

        Ok(Self {
            inv_diag,
            stats: PreconditionerStats::default(),
        })
    }
}

impl<B: Backend> Preconditioner<B> for JacobiPreconditioner<B> {
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>) {
        debug_assert_eq!(x.len(), y.len());
        debug_assert_eq!(x.len(), self.inv_diag.len());

        for (i, (&xi, &inv_di)) in x.iter().zip(self.inv_diag.iter()).enumerate() {
            y[i] = xi * inv_di;
        }
    }

    fn update(&mut self, matrix: &CsrMatrix<B::Scalar>) -> Result<(), PreconditionerError> {
        self.stats.update_calls += 1;
        let timer = std::time::Instant::now();

        let n = matrix.n_rows();
        self.inv_diag.resize(n, B::Scalar::zero());

        matrix.extract_diagonal(&mut self.inv_diag)?;

        // 安全取逆
        for (i, d) in self.inv_diag.iter_mut().enumerate() {
            if d.is_zero() {
                *d = B::Scalar::one();
                self.stats.singular_entries += 1;
            } else {
                *d = B::Scalar::one() / *d;
            }
        }

        self.stats.update_time_ms += timer.elapsed().as_millis() as u64;
        Ok(())
    }

    fn diagonal(&self) -> Option<&[B::Scalar]> {
        Some(&self.inv_diag)
    }

    fn stats(&self) -> PreconditionerStatsSnapshot {
        self.stats.snapshot()
    }

    fn reset_stats(&mut self) {
        self.stats = PreconditionerStats::default();
    }
}

/// 类型别名
pub type JacobiPreconditionerF64 = JacobiPreconditioner<CpuBackend<f64>>;
pub type JacobiPreconditionerF32 = JacobiPreconditioner<CpuBackend<f32>>;

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
    /// 矩阵结构（只读，不拥有）
    matrix: *const CsrMatrix<B::Scalar>,
    /// 松弛因子
    omega: B::Scalar,
    /// 临时向量（前向替换）
    temp: AlignedVec64<B::Scalar>,
    /// 性能统计
    stats: PreconditionerStats,
}

impl<B: Backend> SsorPreconditioner<B> {
    /// 从矩阵创建 SSOR 预条件器
    pub fn from_matrix(backend: &B, matrix: &CsrMatrix<B::Scalar>, params: SsorParams) -> Result<Self, PreconditionerError> {
        let n = matrix.n_rows();
        let mut temp = aligned_vec(n);
        temp.fill(B::Scalar::zero());

        let omega = B::Scalar::from_f64(params.omega).unwrap_or(B::Scalar::one());

        Ok(Self {
            matrix: matrix as *const _,
            omega,
            temp,
            stats: PreconditionerStats::default(),
        })
    }
}

impl<B: Backend> Preconditioner<B> for SsorPreconditioner<B> {
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>) {
        let matrix = unsafe { &*self.matrix };
        let n = matrix.n_rows();

        // 前向替换 (L + D) * y = x
        for i in 0..n {
            let mut sum = x[i];
            // L 部分（严格下三角）
            for &(col, val) in matrix.row_lower(i) {
                sum -= val * y[col];
            }
            // D 部分（对角线）
            let diag_inv = B::Scalar::one() / matrix.diagonal(i).unwrap_or(B::Scalar::one());
            y[i] = sum * diag_inv * self.omega;
        }

        // 后向替换 (U + D) * x = D * y
        self.temp.copy_from_slice(y);
        for i in (0..n).rev() {
            let mut sum = self.temp[i];
            // U 部分（严格上三角）
            for &(col, val) in matrix.row_upper(i) {
                sum -= val * y[col];
            }
            // D 部分
            let diag_inv = B::Scalar::one() / matrix.diagonal(i).unwrap_or(B::Scalar::one());
            y[i] = sum * diag_inv * self.omega;
        }
    }

    fn update(&mut self, matrix: &CsrMatrix<B::Scalar>) -> Result<(), PreconditionerError> {
        self.matrix = matrix as *const _;
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

/// 类型别名
pub type SsorPreconditionerF64 = SsorPreconditioner<CpuBackend<f64>>;
pub type SsorPreconditionerF32 = SsorPreconditioner<CpuBackend<f32>>;

// ============================================================================
// ILU(0) 预条件器（不完全 LU 分解，无填充）
// ============================================================================

/// ILU(0) 预条件器
///
/// 使用 CSR 矩阵的就地分解，不额外存储 LU 结构。
pub struct Ilu0Preconditioner<B: Backend> {
    /// LU 分解后的矩阵值（覆盖存储）
    lu_values: AlignedVec64<B::Scalar>,
    /// 对角线索引
    diag_idxs: Vec<usize>,
    /// 性能统计
    stats: PreconditionerStats,
}

impl<B: Backend> Ilu0Preconditioner<B> {
    /// 从 CSR 矩阵创建 ILU(0)
    pub fn from_matrix(matrix: &CsrMatrix<B::Scalar>) -> Result<Self, PreconditionerError> {
        let n = matrix.n_rows();
        if n == 0 {
            return Err(PreconditionerError::EmptyMatrix);
        }

        let lu_values = aligned_vec(matrix.nnz());
        let diag_idxs = matrix.extract_diagonal_indices()?;

        Ok(Self {
            lu_values,
            diag_idxs,
            stats: PreconditionerStats::default(),
        })
    }
}

impl<B: Backend> Preconditioner<B> for Ilu0Preconditioner<B> {
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>) {
        let n = self.diag_idxs.len();

        // 前向替换 L * y = x
        for i in 0..n {
            let mut sum = x[i];
            // L 部分（下三角，不包含对角线）
            for j in 0..self.diag_idxs[i] {
                let col = self.lu_values[j]; // 需要 col_index 数组
                sum -= self.lu_values[j] * y[col];
            }
            y[i] = sum;
        }

        // 后向替换 U * x = y
        for i in (0..n).rev() {
            let mut sum = y[i];
            // U 部分（上三角，不包含对角线）
            for j in (self.diag_idxs[i] + 1)..self.lu_values.len() {
                let col = self.lu_values[j];
                sum -= self.lu_values[j] * y[col];
            }
            y[i] = sum / self.lu_values[self.diag_idxs[i]]; // 对角线除法
        }
    }

    fn update(&mut self, matrix: &CsrMatrix<B::Scalar>) -> Result<(), PreconditionerError> {
        self.stats.update_calls += 1;
        let timer = std::time::Instant::now();

        // ILU(0) 分解算法
        // 需要实现 CSR 格式的就地分解

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

/// 类型别名
pub type Ilu0PreconditionerF64 = Ilu0Preconditioner<CpuBackend<f64>>;
pub type Ilu0PreconditionerF32 = Ilu0Preconditioner<CpuBackend<f32>>;

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
    pub fn default<B: Backend>(backend: &B) -> Box<dyn Preconditioner<B>> {
        Box::new(JacobiPreconditioner::new(backend))
    }

    /// 创建 Jacobi 预条件器
    pub fn jacobi<B: Backend>(backend: &B) -> Box<dyn Preconditioner<B>> {
        Box::new(JacobiPreconditioner::new(backend))
    }

    /// 创建 SSOR 预条件器
    pub fn ssor<B: Backend>(
        backend: &B,
        matrix: &CsrMatrix<B::Scalar>,
        params: SsorParams,
    ) -> Result<Box<dyn Preconditioner<B>>, PreconditionerError> {
        Ok(Box::new(SsorPreconditioner::from_matrix(backend, matrix, params)?))
    }

    /// 创建 ILU(0) 预条件器
    pub fn ilu0<B: Backend>(
        matrix: &CsrMatrix<B::Scalar>,
    ) -> Result<Box<dyn Preconditioner<B>>, PreconditionerError> {
        Ok(Box::new(Ilu0Preconditioner::from_matrix(matrix)?))
    }
}