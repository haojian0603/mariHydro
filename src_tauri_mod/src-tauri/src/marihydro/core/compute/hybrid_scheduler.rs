//! 混合计算调度器
//!
//! 根据问题规模和设备能力自动选择CPU或GPU计算

use std::sync::Arc;

use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::parallel::strategy::ParallelStrategy;

use super::backend::{ComputeBackend, ComputeOperation, PerformanceEstimate};
use super::capabilities::{DeviceCapabilities, DeviceType};
use super::CpuBackend;

#[cfg(feature = "gpu")]
use super::{GpuSolver, GpuSolverConfig, WgpuBackend};

/// 混合计算策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridStrategy {
    /// 仅使用CPU
    CpuOnly,
    /// 仅使用GPU
    GpuOnly,
    /// 自动选择（基于问题规模）
    Auto,
    /// 强制使用指定设备
    ForceDevice(DeviceType),
}

impl Default for HybridStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

/// 混合计算配置
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// 计算策略
    pub strategy: HybridStrategy,
    /// GPU最小问题规模（单元数）
    pub gpu_min_cells: usize,
    /// GPU最大问题规模（受显存限制）
    pub gpu_max_cells: usize,
    /// 是否允许GPU回退到CPU
    pub allow_fallback: bool,
    /// GPU优先级（0-100，越高越倾向于使用GPU）
    pub gpu_priority: u32,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            strategy: HybridStrategy::Auto,
            gpu_min_cells: 10_000,      // 1万单元以下使用CPU
            gpu_max_cells: 10_000_000,  // 1000万单元以上可能显存不足
            allow_fallback: true,
            gpu_priority: 70,
        }
    }
}

/// 计算设备选择结果
#[derive(Debug, Clone)]
pub struct DeviceSelection {
    /// 选择的设备类型
    pub device_type: DeviceType,
    /// 推荐的并行策略
    pub parallel_strategy: ParallelStrategy,
    /// 选择原因
    pub reason: String,
    /// 预估性能
    pub estimated_speedup: f64,
}

/// 混合计算调度器
pub struct HybridScheduler {
    /// 配置
    config: HybridConfig,
    /// CPU后端
    cpu_backend: CpuBackend,
    /// GPU后端（可选）
    #[cfg(feature = "gpu")]
    gpu_backend: Option<Arc<WgpuBackend>>,
    /// GPU求解器（可选）
    #[cfg(feature = "gpu")]
    gpu_solver: Option<GpuSolver>,
    /// 当前选择的设备
    current_device: DeviceType,
    /// 问题规模
    problem_size: usize,
}

impl HybridScheduler {
    /// 创建混合调度器
    pub fn new(config: HybridConfig) -> MhResult<Self> {
        let cpu_backend = CpuBackend::new();
        
        #[cfg(feature = "gpu")]
        let (gpu_backend, gpu_solver) = {
            match WgpuBackend::new() {
                Ok(backend) => {
                    let backend = Arc::new(backend);
                    // 创建GPU求解器
                    let solver = GpuSolver::new(
                        backend.device(),
                        backend.queue(),
                        GpuSolverConfig::default(),
                    ).ok();
                    (Some(backend), solver)
                }
                Err(e) => {
                    log::warn!("GPU后端初始化失败: {}", e);
                    (None, None)
                }
            }
        };
        
        let current_device = DeviceType::Cpu;
        
        Ok(Self {
            config,
            cpu_backend,
            #[cfg(feature = "gpu")]
            gpu_backend,
            #[cfg(feature = "gpu")]
            gpu_solver,
            current_device,
            problem_size: 0,
        })
    }
    
    /// 选择计算设备
    pub fn select_device(&mut self, num_cells: usize) -> DeviceSelection {
        self.problem_size = num_cells;
        
        match self.config.strategy {
            HybridStrategy::CpuOnly => self.select_cpu("用户指定仅使用CPU"),
            HybridStrategy::GpuOnly => self.select_gpu_or_fallback("用户指定仅使用GPU"),
            HybridStrategy::ForceDevice(device_type) => {
                match device_type {
                    DeviceType::Cpu => self.select_cpu("强制使用CPU"),
                    _ => self.select_gpu_or_fallback("强制使用GPU"),
                }
            }
            HybridStrategy::Auto => self.auto_select(num_cells),
        }
    }
    
    /// 自动选择设备
    fn auto_select(&mut self, num_cells: usize) -> DeviceSelection {
        // 检查GPU是否可用
        #[cfg(feature = "gpu")]
        if let Some(ref gpu) = self.gpu_backend {
            if !gpu.is_available() {
                return self.select_cpu("GPU不可用");
            }
            
            // 检查问题规模
            if num_cells < self.config.gpu_min_cells {
                return self.select_cpu(&format!(
                    "问题规模{}小于GPU阈值{}",
                    num_cells, self.config.gpu_min_cells
                ));
            }
            
            if num_cells > self.config.gpu_max_cells {
                return self.select_cpu(&format!(
                    "问题规模{}超过GPU内存限制{}",
                    num_cells, self.config.gpu_max_cells
                ));
            }
            
            // 估算GPU性能收益
            let gpu_estimate = gpu.estimate_performance(
                ComputeOperation::Riemann, 
                num_cells
            );
            
            let cpu_estimate = self.cpu_backend.estimate_performance(
                ComputeOperation::Riemann,
                num_cells
            );
            
            if gpu_estimate.recommended {
                let speedup = cpu_estimate.estimated_time_ms / 
                    gpu_estimate.estimated_time_ms.max(0.001);
                return self.select_gpu(&format!(
                    "预估GPU加速比{:.1}x",
                    speedup
                ), speedup);
            } else {
                return self.select_cpu("GPU预估性能不佳");
            }
        }
        
        #[cfg(not(feature = "gpu"))]
        return self.select_cpu("GPU功能未编译");
        
        #[cfg(feature = "gpu")]
        self.select_cpu("GPU未初始化")
    }
    
    fn select_cpu(&mut self, reason: &str) -> DeviceSelection {
        self.current_device = DeviceType::Cpu;
        
        // 根据问题规模选择CPU并行策略
        let parallel_strategy = if self.problem_size < 1000 {
            ParallelStrategy::Sequential
        } else if self.problem_size < 100_000 {
            ParallelStrategy::StaticChunks { 
                chunk_size: (self.problem_size / rayon::current_num_threads()).max(100) 
            }
        } else {
            ParallelStrategy::Dynamic
        };
        
        DeviceSelection {
            device_type: DeviceType::Cpu,
            parallel_strategy,
            reason: reason.to_string(),
            estimated_speedup: 1.0,
        }
    }
    
    #[cfg(feature = "gpu")]
    fn select_gpu(&mut self, reason: &str, speedup: f64) -> DeviceSelection {
        self.current_device = DeviceType::DiscreteGpu;
        
        DeviceSelection {
            device_type: DeviceType::DiscreteGpu,
            parallel_strategy: ParallelStrategy::GpuCompute { 
                workgroup_size: 256 
            },
            reason: reason.to_string(),
            estimated_speedup: speedup,
        }
    }
    
    fn select_gpu_or_fallback(&mut self, reason: &str) -> DeviceSelection {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref gpu) = self.gpu_backend {
                if gpu.is_available() {
                    return self.select_gpu(reason, 1.0);
                }
            }
            
            if self.config.allow_fallback {
                return self.select_cpu("GPU不可用，回退到CPU");
            }
        }
        
        self.select_cpu("GPU功能未启用")
    }
    
    /// 获取当前设备
    pub fn current_device(&self) -> DeviceType {
        self.current_device
    }
    
    /// 获取CPU后端
    pub fn cpu_backend(&self) -> &CpuBackend {
        &self.cpu_backend
    }
    
    /// 获取GPU后端
    #[cfg(feature = "gpu")]
    pub fn gpu_backend(&self) -> Option<&Arc<WgpuBackend>> {
        self.gpu_backend.as_ref()
    }
    
    /// 获取GPU求解器
    #[cfg(feature = "gpu")]
    pub fn gpu_solver(&self) -> Option<&GpuSolver> {
        self.gpu_solver.as_ref()
    }
    
    /// 获取GPU求解器（可变）
    #[cfg(feature = "gpu")]
    pub fn gpu_solver_mut(&mut self) -> Option<&mut GpuSolver> {
        self.gpu_solver.as_mut()
    }
    
    /// GPU是否可用
    pub fn is_gpu_available(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.gpu_backend.as_ref().map(|g| g.is_available()).unwrap_or(false)
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }
    
    /// 获取设备信息
    pub fn device_info(&self) -> String {
        let mut info = format!("CPU: {}\n", self.cpu_backend.capabilities().name);
        
        #[cfg(feature = "gpu")]
        if let Some(ref gpu) = self.gpu_backend {
            let caps = gpu.capabilities();
            info.push_str(&format!(
                "GPU: {} ({:?})\n  显存: {:.0} MB\n  计算单元: {}\n",
                caps.name,
                caps.device_type,
                caps.memory.total_bytes as f64 / (1024.0 * 1024.0),
                caps.features.compute_units,
            ));
        }
        
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = HybridConfig::default();
        assert_eq!(config.strategy, HybridStrategy::Auto);
        assert!(config.allow_fallback);
    }
    
    #[test]
    fn test_scheduler_creation() {
        let scheduler = HybridScheduler::new(HybridConfig::default());
        assert!(scheduler.is_ok());
    }
    
    #[test]
    fn test_cpu_selection_small_problem() {
        let mut scheduler = HybridScheduler::new(HybridConfig::default()).unwrap();
        let selection = scheduler.select_device(1000);
        assert_eq!(selection.device_type, DeviceType::Cpu);
    }
}
