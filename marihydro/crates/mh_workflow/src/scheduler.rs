// crates/mh_workflow/src/scheduler.rs

//! 混合调度器模块
//!
//! 提供CPU/GPU计算设备的自动选择和调度。

use std::sync::Arc;

/// 设备类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU
    Cpu,
    /// 集成GPU
    IntegratedGpu,
    /// 独立GPU
    DiscreteGpu,
    /// 其他加速器
    Other,
}

impl Default for DeviceType {
    fn default() -> Self {
        Self::Cpu
    }
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::IntegratedGpu => write!(f, "Integrated GPU"),
            Self::DiscreteGpu => write!(f, "Discrete GPU"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// 混合计算策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridStrategy {
    /// 仅使用CPU
    CpuOnly,
    /// 仅使用GPU
    GpuOnly,
    /// 自动选择
    Auto,
    /// 强制指定设备
    ForceDevice(DeviceType),
}

impl Default for HybridStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::fmt::Display for HybridStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CpuOnly => write!(f, "CPU Only"),
            Self::GpuOnly => write!(f, "GPU Only"),
            Self::Auto => write!(f, "Auto"),
            Self::ForceDevice(device) => write!(f, "Force {}", device),
        }
    }
}

/// 混合调度配置
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// 计算策略
    pub strategy: HybridStrategy,
    /// GPU最小单元数阈值（低于此值使用CPU）
    pub gpu_min_cells: usize,
    /// GPU最大单元数阈值（超过此值分批处理）
    pub gpu_max_cells: usize,
    /// 允许回退到CPU
    pub allow_fallback: bool,
    /// GPU优先级权重 (0-100)
    pub gpu_priority: u32,
    /// CPU线程数 (0=自动)
    pub cpu_threads: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            strategy: HybridStrategy::Auto,
            gpu_min_cells: 10_000,
            gpu_max_cells: 10_000_000,
            allow_fallback: true,
            gpu_priority: 70,
            cpu_threads: 0,
        }
    }
}

impl HybridConfig {
    /// 创建CPU-only配置
    pub fn cpu_only() -> Self {
        Self {
            strategy: HybridStrategy::CpuOnly,
            ..Default::default()
        }
    }

    /// 创建GPU-only配置
    pub fn gpu_only() -> Self {
        Self {
            strategy: HybridStrategy::GpuOnly,
            allow_fallback: true,
            ..Default::default()
        }
    }

    /// 设置GPU阈值
    pub fn with_gpu_threshold(mut self, min: usize, max: usize) -> Self {
        self.gpu_min_cells = min;
        self.gpu_max_cells = max;
        self
    }
}

/// 设备选择结果
#[derive(Debug, Clone)]
pub struct DeviceSelection {
    /// 选择的设备类型
    pub device_type: DeviceType,
    /// 选择原因
    pub reason: String,
    /// 预估加速比
    pub estimated_speedup: f64,
    /// 是否为回退选择
    pub is_fallback: bool,
}

impl DeviceSelection {
    /// CPU选择
    pub fn cpu(reason: impl Into<String>) -> Self {
        Self {
            device_type: DeviceType::Cpu,
            reason: reason.into(),
            estimated_speedup: 1.0,
            is_fallback: false,
        }
    }

    /// GPU选择
    pub fn gpu(device_type: DeviceType, reason: impl Into<String>, speedup: f64) -> Self {
        Self {
            device_type,
            reason: reason.into(),
            estimated_speedup: speedup,
            is_fallback: false,
        }
    }

    /// 回退选择
    pub fn fallback(reason: impl Into<String>) -> Self {
        Self {
            device_type: DeviceType::Cpu,
            reason: reason.into(),
            estimated_speedup: 1.0,
            is_fallback: true,
        }
    }
}

/// GPU能力信息
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// 设备名称
    pub device_name: String,
    /// 设备类型
    pub device_type: DeviceType,
    /// 显存大小 (MB)
    pub memory_mb: u64,
    /// 最大工作组大小
    pub max_workgroup_size: u32,
    /// 最大缓冲区大小
    pub max_buffer_size: u64,
    /// 支持的特性
    pub features: GpuFeatures,
}

impl Default for GpuCapabilities {
    fn default() -> Self {
        Self {
            device_name: "Unknown".into(),
            device_type: DeviceType::DiscreteGpu,
            memory_mb: 0,
            max_workgroup_size: 256,
            max_buffer_size: 256 * 1024 * 1024,
            features: GpuFeatures::default(),
        }
    }
}

/// GPU特性
#[derive(Debug, Clone, Default)]
pub struct GpuFeatures {
    /// 支持float64
    pub float64: bool,
    /// 支持原子操作
    pub atomics: bool,
    /// 支持subgroup操作
    pub subgroups: bool,
}

/// 混合计算调度器
pub struct HybridScheduler {
    /// 配置
    config: HybridConfig,
    /// GPU能力
    gpu_capabilities: Option<Arc<GpuCapabilities>>,
    /// 当前设备
    current_device: DeviceType,
    /// 历史性能数据
    performance_history: parking_lot::RwLock<Vec<PerformanceRecord>>,
}

/// 性能记录
#[derive(Debug, Clone)]
struct PerformanceRecord {
    device: DeviceType,
    num_cells: usize,
    elapsed_secs: f64,
    timestamp: std::time::Instant,
}

impl HybridScheduler {
    /// 创建调度器
    pub fn new(config: HybridConfig) -> Self {
        Self {
            config,
            gpu_capabilities: None,
            current_device: DeviceType::Cpu,
            performance_history: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// 设置GPU能力
    pub fn with_gpu(mut self, capabilities: GpuCapabilities) -> Self {
        let device_type = capabilities.device_type;
        self.gpu_capabilities = Some(Arc::new(capabilities));
        self.current_device = device_type;
        self
    }

    /// 尝试检测GPU
    pub fn detect_gpu(&mut self) -> Option<&GpuCapabilities> {
        // TODO: 实际的GPU检测
        // 这里返回None表示未检测到
        self.gpu_capabilities.as_deref()
    }

    /// 选择计算设备
    pub fn select_device(&self, num_cells: usize) -> DeviceSelection {
        match self.config.strategy {
            HybridStrategy::CpuOnly => {
                DeviceSelection::cpu("Strategy: CPU only")
            }

            HybridStrategy::GpuOnly => {
                if let Some(gpu) = &self.gpu_capabilities {
                    let speedup = self.estimate_gpu_speedup(num_cells);
                    DeviceSelection::gpu(
                        gpu.device_type,
                        format!("Strategy: GPU only ({})", gpu.device_name),
                        speedup,
                    )
                } else if self.config.allow_fallback {
                    DeviceSelection::fallback("No GPU available, falling back to CPU")
                } else {
                    DeviceSelection::cpu("No GPU available")
                }
            }

            HybridStrategy::ForceDevice(device) => {
                if device == DeviceType::Cpu {
                    DeviceSelection::cpu("Forced CPU")
                } else if self.gpu_capabilities.is_some() {
                    let speedup = self.estimate_gpu_speedup(num_cells);
                    DeviceSelection::gpu(device, "Forced GPU", speedup)
                } else if self.config.allow_fallback {
                    DeviceSelection::fallback("Forced GPU not available, falling back to CPU")
                } else {
                    DeviceSelection::cpu("Forced GPU not available")
                }
            }

            HybridStrategy::Auto => {
                self.auto_select(num_cells)
            }
        }
    }

    /// 自动选择设备
    fn auto_select(&self, num_cells: usize) -> DeviceSelection {
        // 检查是否有GPU
        let gpu = match &self.gpu_capabilities {
            Some(gpu) => gpu,
            None => return DeviceSelection::cpu("No GPU available"),
        };

        // 检查单元数阈值
        if num_cells < self.config.gpu_min_cells {
            return DeviceSelection::cpu(format!(
                "Cell count ({}) below GPU threshold ({})",
                num_cells, self.config.gpu_min_cells
            ));
        }

        // 检查是否超过GPU最大处理能力
        if num_cells > self.config.gpu_max_cells {
            // 可能需要分批处理，但仍然使用GPU
            let speedup = self.estimate_gpu_speedup(num_cells);
            return DeviceSelection::gpu(
                gpu.device_type,
                format!(
                    "Large mesh ({}), will batch process on {}",
                    num_cells, gpu.device_name
                ),
                speedup,
            );
        }

        // 检查显存是否足够
        let estimated_memory_mb = self.estimate_memory_usage(num_cells);
        if estimated_memory_mb > gpu.memory_mb as f64 * 0.8 {
            if self.config.allow_fallback {
                return DeviceSelection::fallback(format!(
                    "Insufficient GPU memory: need {:.0} MB, have {} MB",
                    estimated_memory_mb, gpu.memory_mb
                ));
            }
        }

        // 估计加速比
        let speedup = self.estimate_gpu_speedup(num_cells);

        // 如果加速比太小，使用CPU
        if speedup < 1.5 {
            return DeviceSelection::cpu(format!(
                "Low GPU speedup ({:.2}x), using CPU instead",
                speedup
            ));
        }

        DeviceSelection::gpu(
            gpu.device_type,
            format!(
                "Auto-selected {} ({:.2}x speedup)",
                gpu.device_name, speedup
            ),
            speedup,
        )
    }

    /// 估计GPU加速比
    fn estimate_gpu_speedup(&self, num_cells: usize) -> f64 {
        // 简单的经验公式
        // 实际应该基于历史数据和设备特性

        if num_cells < 1000 {
            // 小网格，GPU开销较大
            0.5
        } else if num_cells < 10_000 {
            // 中等网格
            1.0 + (num_cells as f64 / 10_000.0) * 2.0
        } else if num_cells < 100_000 {
            // 大网格
            3.0 + (num_cells as f64 / 100_000.0) * 7.0
        } else {
            // 非常大的网格
            10.0 + (num_cells as f64 / 1_000_000.0).min(10.0) * 5.0
        }
    }

    /// 估计内存使用量 (MB)
    fn estimate_memory_usage(&self, num_cells: usize) -> f64 {
        // 每个单元大约需要：
        // - 状态: 4 * 8 bytes (h, hu, hv, eta)
        // - 梯度: 6 * 8 bytes
        // - 拓扑: 8 * 4 bytes (邻接关系)
        // - 其他: 约 100 bytes
        let bytes_per_cell = 4 * 8 + 6 * 8 + 8 * 4 + 100;
        (num_cells * bytes_per_cell) as f64 / (1024.0 * 1024.0)
    }

    /// 记录性能数据
    pub fn record_performance(&self, device: DeviceType, num_cells: usize, elapsed_secs: f64) {
        let record = PerformanceRecord {
            device,
            num_cells,
            elapsed_secs,
            timestamp: std::time::Instant::now(),
        };

        let mut history = self.performance_history.write();
        history.push(record);

        // 保留最近1000条记录
        if history.len() > 1000 {
            history.remove(0);
        }
    }

    /// 获取当前设备
    pub fn current_device(&self) -> DeviceType {
        self.current_device
    }

    /// 是否有GPU可用
    pub fn has_gpu(&self) -> bool {
        self.gpu_capabilities.is_some()
    }

    /// 获取GPU能力
    pub fn gpu_capabilities(&self) -> Option<&GpuCapabilities> {
        self.gpu_capabilities.as_deref()
    }

    /// 获取配置
    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    /// 获取性能统计
    pub fn performance_stats(&self) -> PerformanceStats {
        let history = self.performance_history.read();

        let cpu_records: Vec<_> = history
            .iter()
            .filter(|r| r.device == DeviceType::Cpu)
            .collect();

        let gpu_records: Vec<_> = history
            .iter()
            .filter(|r| r.device != DeviceType::Cpu)
            .collect();

        PerformanceStats {
            cpu_invocations: cpu_records.len(),
            gpu_invocations: gpu_records.len(),
            avg_cpu_cells_per_sec: Self::avg_throughput(&cpu_records),
            avg_gpu_cells_per_sec: Self::avg_throughput(&gpu_records),
        }
    }

    fn avg_throughput(records: &[&PerformanceRecord]) -> f64 {
        if records.is_empty() {
            return 0.0;
        }

        let total_cells: usize = records.iter().map(|r| r.num_cells).sum();
        let total_time: f64 = records.iter().map(|r| r.elapsed_secs).sum();

        if total_time > 0.0 {
            total_cells as f64 / total_time
        } else {
            0.0
        }
    }
}

/// 性能统计
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// CPU调用次数
    pub cpu_invocations: usize,
    /// GPU调用次数
    pub gpu_invocations: usize,
    /// 平均CPU吞吐量 (单元/秒)
    pub avg_cpu_cells_per_sec: f64,
    /// 平均GPU吞吐量 (单元/秒)
    pub avg_gpu_cells_per_sec: f64,
}

impl PerformanceStats {
    /// 计算GPU相对CPU的实际加速比
    pub fn actual_speedup(&self) -> Option<f64> {
        if self.avg_cpu_cells_per_sec > 0.0 && self.avg_gpu_cells_per_sec > 0.0 {
            Some(self.avg_gpu_cells_per_sec / self.avg_cpu_cells_per_sec)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_config() {
        let config = HybridConfig::default();
        assert_eq!(config.strategy, HybridStrategy::Auto);
        assert!(config.allow_fallback);

        let cpu_config = HybridConfig::cpu_only();
        assert_eq!(cpu_config.strategy, HybridStrategy::CpuOnly);

        let gpu_config = HybridConfig::gpu_only();
        assert_eq!(gpu_config.strategy, HybridStrategy::GpuOnly);
    }

    #[test]
    fn test_scheduler_cpu_only() {
        let config = HybridConfig::cpu_only();
        let scheduler = HybridScheduler::new(config);

        let selection = scheduler.select_device(100_000);
        assert_eq!(selection.device_type, DeviceType::Cpu);
        assert!(!selection.is_fallback);
    }

    #[test]
    fn test_scheduler_auto_no_gpu() {
        let scheduler = HybridScheduler::new(HybridConfig::default());

        let selection = scheduler.select_device(100_000);
        assert_eq!(selection.device_type, DeviceType::Cpu);
        assert!(selection.reason.contains("No GPU"));
    }

    #[test]
    fn test_scheduler_auto_with_gpu() {
        let gpu = GpuCapabilities {
            device_name: "Test GPU".into(),
            device_type: DeviceType::DiscreteGpu,
            memory_mb: 4096,
            ..Default::default()
        };

        let scheduler = HybridScheduler::new(HybridConfig::default()).with_gpu(gpu);

        // 小网格应该使用CPU
        let selection = scheduler.select_device(1000);
        assert_eq!(selection.device_type, DeviceType::Cpu);

        // 大网格应该使用GPU
        let selection = scheduler.select_device(100_000);
        assert_eq!(selection.device_type, DeviceType::DiscreteGpu);
        assert!(selection.estimated_speedup > 1.0);
    }

    #[test]
    fn test_estimate_speedup() {
        let scheduler = HybridScheduler::new(HybridConfig::default());

        // 小网格加速比应该较低
        let speedup = scheduler.estimate_gpu_speedup(500);
        assert!(speedup < 1.0);

        // 大网格加速比应该较高
        let speedup = scheduler.estimate_gpu_speedup(500_000);
        assert!(speedup > 5.0);
    }

    #[test]
    fn test_performance_recording() {
        let scheduler = HybridScheduler::new(HybridConfig::default());

        scheduler.record_performance(DeviceType::Cpu, 10000, 1.0);
        scheduler.record_performance(DeviceType::DiscreteGpu, 10000, 0.1);

        let stats = scheduler.performance_stats();
        assert_eq!(stats.cpu_invocations, 1);
        assert_eq!(stats.gpu_invocations, 1);

        if let Some(speedup) = stats.actual_speedup() {
            assert!(speedup > 1.0);
        }
    }
}
