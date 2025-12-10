//! Kernel 接口规范
//!
//! 定义 GPU kernel 的 Rust 侧接口。

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
        implemented: false,
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
pub enum TransferPolicy {
    /// 延迟传输
    Lazy,
    /// 即时传输
    Eager,
    /// 流水线传输
    Pipelined,
}

impl Default for TransferPolicy {
    fn default() -> Self {
        Self::Lazy
    }
}

/// 获取未实现的核心 kernel
pub fn unimplemented_kernels() -> Vec<&'static KernelSpec> {
    CORE_KERNELS.iter().filter(|k| !k.implemented).collect()
}

/// 获取按优先级排序的 kernel
pub fn kernels_by_priority(priority: KernelPriority) -> Vec<&'static KernelSpec> {
    CORE_KERNELS.iter().filter(|k| k.priority == priority).collect()
}
