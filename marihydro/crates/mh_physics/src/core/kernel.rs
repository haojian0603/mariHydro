//! Kernel 接口规范
//!
//! 定义 GPU kernel 的 Rust 侧接口，并提供 CPU 侧实现作为默认执行路径。

use crate::numerics::linear_algebra::csr::CsrMatrix;
use mh_runtime::RuntimeScalar;

/// Kernel 优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelPriority {
    /// P0: 核心计算（通量、状态更新）
    Critical,
    /// P1: 重要计算（源项、梯度）
    High,
    /// P2: 辅助计算（SpMV、剖面恢复）
    Medium,
    /// P3: 可选计算
    Low,
}

/// Kernel 规范
#[derive(Debug, Clone)]
pub struct KernelSpec {
    /// Kernel 名称
    pub name: &'static str,
    /// 优先级
    pub priority: KernelPriority,
    /// 预计加速比
    pub expected_speedup: f64,
    /// 是否已实现
    pub implemented: bool,
}

/// 核心 Kernel 列表
pub const CORE_KERNELS: &[KernelSpec] = &[
    KernelSpec {
        name: "flux_compute",
        priority: KernelPriority::Critical,
        expected_speedup: 30.0,
        implemented: false,
    },
    KernelSpec {
        name: "state_update",
        priority: KernelPriority::Critical,
        expected_speedup: 30.0,
        implemented: false,
    },
    KernelSpec {
        name: "source_batch",
        priority: KernelPriority::High,
        expected_speedup: 10.0,
        implemented: false,
    },
    KernelSpec {
        name: "gradient_compute",
        priority: KernelPriority::High,
        expected_speedup: 20.0,
        implemented: false,
    },
    KernelSpec {
        name: "spmv",
        priority: KernelPriority::Medium,
        expected_speedup: 5.0,
        implemented: true,
    },
    KernelSpec {
        name: "profile_restore",
        priority: KernelPriority::Medium,
        expected_speedup: 10.0,
        implemented: false,
    },
];

/// 传输策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum TransferPolicy {
    /// 延迟传输
    #[default]
    Lazy,
    /// 即时传输
    Eager,
    /// 流水线传输
    Pipelined,
}


/// 获取未实现的核心 kernel
pub fn unimplemented_kernels() -> Vec<&'static KernelSpec> {
    CORE_KERNELS.iter().filter(|k| !k.implemented).collect()
}

/// 获取按优先级排序的 kernel
pub fn kernels_by_priority(priority: KernelPriority) -> Vec<&'static KernelSpec> {
    CORE_KERNELS.iter().filter(|k| k.priority == priority).collect()
}

// ============================================================================
// CPU Kernel 实现
// ============================================================================

/// 稀疏矩阵乘向量 (SpMV) kernel: y = A * x
pub fn spmv_kernel<S: RuntimeScalar>(matrix: &CsrMatrix<S>, x: &[S], y: &mut [S]) {
    let n_rows = matrix.n_rows();
    assert_eq!(x.len(), matrix.n_cols(), "x 长度必须等于矩阵列数");
    assert_eq!(y.len(), n_rows, "y 长度必须等于矩阵行数");

    let row_ptr = matrix.row_ptr();
    let col_idx = matrix.col_idx();
    let values = matrix.values();

    for row in 0..n_rows {
        let start = row_ptr[row];
        let end = row_ptr[row + 1];
        let mut sum = S::ZERO;
        for idx in start..end {
            let col = col_idx[idx];
            sum += values[idx] * x[col];
        }
        y[row] = sum;
    }
}
