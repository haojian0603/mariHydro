// crates/mh_physics/src/gpu/backend.rs

//! 计算后端抽象trait
//!
//! 定义统一的计算后端接口，支持CPU和GPU实现。

use super::capabilities::DeviceCapabilities;
use mh_foundation::error::MhResult;

/// 计算操作类型
///
/// 用于性能估计和策略选择
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeOperation {
    /// 梯度计算（per-cell）
    Gradient,
    /// 限制器计算（per-cell）
    Limiter,
    /// Riemann求解（per-face）
    Riemann,
    /// 通量累加（per-face，需着色）
    FluxAccumulate,
    /// 源项计算（per-cell）
    SourceTerms,
    /// 时间积分（per-cell）
    TimeIntegrate,
    /// 归约操作（reduction）
    Reduction,
}

impl ComputeOperation {
    /// 是否为per-cell操作
    pub fn is_per_cell(&self) -> bool {
        matches!(
            self,
            ComputeOperation::Gradient
                | ComputeOperation::Limiter
                | ComputeOperation::SourceTerms
                | ComputeOperation::TimeIntegrate
        )
    }

    /// 是否为per-face操作
    pub fn is_per_face(&self) -> bool {
        matches!(
            self,
            ComputeOperation::Riemann | ComputeOperation::FluxAccumulate
        )
    }

    /// 是否需要着色并行
    pub fn requires_coloring(&self) -> bool {
        matches!(self, ComputeOperation::FluxAccumulate)
    }
}

/// 性能估计结果
#[derive(Debug, Clone)]
pub struct PerformanceEstimate {
    /// 预计执行时间（毫秒）
    pub estimated_time_ms: f64,
    /// 所需内存（字节）
    pub memory_required: usize,
    /// 是否推荐使用此后端
    pub recommended: bool,
}

impl PerformanceEstimate {
    /// 创建新的性能估计
    pub fn new(estimated_time_ms: f64, memory_required: usize, recommended: bool) -> Self {
        Self {
            estimated_time_ms,
            memory_required,
            recommended,
        }
    }

    /// 比较两个估计，返回更优的
    pub fn better_of(self, other: Self) -> Self {
        if self.recommended && !other.recommended {
            self
        } else if !self.recommended && other.recommended {
            other
        } else if self.estimated_time_ms <= other.estimated_time_ms {
            self
        } else {
            other
        }
    }
}

/// 计算后端抽象trait
///
/// 所有计算后端（CPU、GPU等）都需要实现此trait。
///
/// # 实现要求
///
/// - `Send + Sync`: 后端需要线程安全
/// - 所有方法不应panic，应返回适当的错误
pub trait ComputeBackend: Send + Sync {
    /// 后端名称
    ///
    /// 返回人类可读的后端标识，如 "CPU (rayon)" 或 "GPU (wgpu/Vulkan)"
    fn name(&self) -> &'static str;

    /// 检查后端是否可用
    ///
    /// 对于GPU后端，这可能涉及设备检测
    fn is_available(&self) -> bool;

    /// 获取设备能力描述
    fn capabilities(&self) -> &DeviceCapabilities;

    /// 估算特定操作的性能
    ///
    /// # 参数
    /// - `op`: 操作类型
    /// - `problem_size`: 问题规模（单元数或面数）
    ///
    /// # 返回
    /// 性能估计，包括预计时间和内存需求
    fn estimate_performance(
        &self,
        op: ComputeOperation,
        problem_size: usize,
    ) -> PerformanceEstimate;

    /// 同步等待所有操作完成
    ///
    /// 对于GPU后端，这会阻塞直到所有已提交的命令完成
    fn synchronize(&self) -> MhResult<()>;

    /// 是否支持f64双精度
    fn supports_f64(&self) -> bool {
        self.capabilities().supports_f64()
    }

    /// 获取推荐的工作组大小
    fn recommended_workgroup_size(&self) -> u32 {
        self.capabilities().max_workgroup_size.min(256)
    }
}

/// CPU后端（始终可用）
pub struct CpuBackend {
    capabilities: DeviceCapabilities,
}

impl CpuBackend {
    /// 创建新的CPU后端
    pub fn new() -> Self {
        Self {
            capabilities: DeviceCapabilities::cpu_default(),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "CPU (rayon)"
    }

    fn is_available(&self) -> bool {
        true // CPU始终可用
    }

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn estimate_performance(
        &self,
        op: ComputeOperation,
        problem_size: usize,
    ) -> PerformanceEstimate {
        // CPU性能估计（基于启发式）
        let ops_per_element = match op {
            ComputeOperation::Gradient => 20,
            ComputeOperation::Limiter => 30,
            ComputeOperation::Riemann => 100,
            ComputeOperation::FluxAccumulate => 10,
            ComputeOperation::SourceTerms => 50,
            ComputeOperation::TimeIntegrate => 20,
            ComputeOperation::Reduction => 5,
        };
        
        // 假设每核心每秒10亿浮点运算
        let total_ops = problem_size * ops_per_element;
        let gflops = self.capabilities.compute_units as f64 * 1.0; // 每核心1 GFLOP/s
        let time_ms = (total_ops as f64) / (gflops * 1e6);
        
        let memory = problem_size * 8 * 4; // 4个f64字段
        let recommended = problem_size < 100_000; // CPU对小问题更有效
        
        PerformanceEstimate::new(time_ms, memory, recommended)
    }

    fn synchronize(&self) -> MhResult<()> {
        // CPU操作是同步的，无需等待
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_operation_per_cell() {
        assert!(ComputeOperation::Gradient.is_per_cell());
        assert!(ComputeOperation::Limiter.is_per_cell());
        assert!(!ComputeOperation::Riemann.is_per_cell());
        assert!(!ComputeOperation::FluxAccumulate.is_per_cell());
    }

    #[test]
    fn test_compute_operation_per_face() {
        assert!(!ComputeOperation::Gradient.is_per_face());
        assert!(ComputeOperation::Riemann.is_per_face());
        assert!(ComputeOperation::FluxAccumulate.is_per_face());
    }

    #[test]
    fn test_compute_operation_requires_coloring() {
        assert!(!ComputeOperation::Riemann.requires_coloring());
        assert!(ComputeOperation::FluxAccumulate.requires_coloring());
    }

    #[test]
    fn test_performance_estimate_better_of() {
        let better = PerformanceEstimate::new(1.0, 1000, true);
        let worse = PerformanceEstimate::new(2.0, 1000, true);
        
        let result = better.clone().better_of(worse);
        assert_eq!(result.estimated_time_ms, 1.0);
    }

    #[test]
    fn test_cpu_backend_new() {
        let backend = CpuBackend::new();
        assert!(backend.is_available());
        assert_eq!(backend.name(), "CPU (rayon)");
    }

    #[test]
    fn test_cpu_backend_estimate() {
        let backend = CpuBackend::new();
        let estimate = backend.estimate_performance(ComputeOperation::Riemann, 10_000);
        
        assert!(estimate.estimated_time_ms > 0.0);
        assert!(estimate.memory_required > 0);
        assert!(estimate.recommended); // 小问题推荐CPU
    }

    #[test]
    fn test_cpu_backend_synchronize() {
        let backend = CpuBackend::new();
        assert!(backend.synchronize().is_ok());
    }
}