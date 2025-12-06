// crates/mh_physics/src/gpu/hybrid_scheduler.rs

//! GPU/CPU 混合调度器
//!
//! 根据问题规模和设备能力，自动选择最优的计算后端。
//!
//! # 设计目标
//!
//! 1. **自动决策**: 基于历史性能数据自动选择 CPU 或 GPU
//! 2. **平滑切换**: 支持运行时动态切换计算后端
//! 3. **性能监控**: 实时收集性能指标，动态调整策略
//! 4. **负载均衡**: 大规模计算时支持 CPU+GPU 协同
//!
//! # 决策策略
//!
//! ```text
//! 问题规模 N:
//!   N < Threshold_min  → CPU (避免 GPU 启动开销)
//!   N > Threshold_max  → GPU (充分利用并行性)
//!   otherwise          → 根据历史性能数据选择
//! ```

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::capabilities::{DeviceCapabilities, DeviceType};
use super::GpuError;

// ============================================================================
// 核心类型定义
// ============================================================================

/// 计算后端类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    /// CPU 后端
    Cpu,
    /// GPU 后端
    Gpu,
    /// CPU+GPU 混合后端（大规模计算）
    Hybrid,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Cpu => write!(f, "CPU"),
            BackendType::Gpu => write!(f, "GPU"),
            BackendType::Hybrid => write!(f, "CPU+GPU"),
        }
    }
}

/// 调度决策
#[derive(Debug, Clone)]
pub struct SchedulingDecision {
    /// 选择的后端
    pub backend: BackendType,
    /// 预估执行时间（毫秒）
    pub estimated_time_ms: f64,
    /// 决策置信度 [0, 1]
    pub confidence: f64,
    /// 决策原因
    pub reason: DecisionReason,
}

/// 决策原因
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    /// 问题规模太小，不值得使用 GPU
    ProblemTooSmall,
    /// 问题规模较大，适合 GPU
    ProblemLargeEnough,
    /// GPU 不可用
    GpuUnavailable,
    /// 基于历史性能数据
    HistoricalPerformance,
    /// 显存不足
    InsufficientGpuMemory,
    /// 强制使用指定后端
    ForcedBackend,
    /// 超大规模，需要混合计算
    VeryLargeScale,
}

/// 性能样本
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    /// 后端类型
    pub backend: BackendType,
    /// 问题规模（单元数）
    pub problem_size: usize,
    /// 执行时间（微秒）
    pub execution_time_us: u64,
    /// 时间戳
    pub timestamp: Instant,
}

// ============================================================================
// 混合调度器配置
// ============================================================================

/// 混合调度器配置
#[derive(Debug, Clone)]
pub struct HybridSchedulerConfig {
    /// CPU/GPU 切换的最小阈值（单元数）
    ///
    /// 小于此值时始终使用 CPU
    pub min_gpu_threshold: usize,

    /// 强制使用 GPU 的阈值（单元数）
    ///
    /// 大于此值时始终使用 GPU（如果可用）
    pub max_cpu_threshold: usize,

    /// 启用混合计算的阈值（单元数）
    ///
    /// 大于此值时考虑使用 CPU+GPU 协同
    pub hybrid_threshold: usize,

    /// 是否启用自适应阈值调整
    pub adaptive_thresholds: bool,

    /// 性能历史窗口大小
    pub history_window_size: usize,

    /// 是否启用预热（首次运行时收集基准数据）
    pub enable_warmup: bool,

    /// 强制使用的后端（优先于自动决策）
    pub forced_backend: Option<BackendType>,

    /// GPU 显存使用上限（0.0-1.0）
    pub gpu_memory_limit: f64,

    /// 切换后端的最小间隔（避免频繁切换）
    pub min_switch_interval: Duration,
}

impl Default for HybridSchedulerConfig {
    fn default() -> Self {
        Self {
            min_gpu_threshold: 1_000,      // 1K 单元以下用 CPU
            max_cpu_threshold: 50_000,     // 50K 单元以上用 GPU
            hybrid_threshold: 1_000_000,   // 1M 单元以上考虑混合
            adaptive_thresholds: true,
            history_window_size: 100,
            enable_warmup: true,
            forced_backend: None,
            gpu_memory_limit: 0.8,
            min_switch_interval: Duration::from_millis(100),
        }
    }
}

impl HybridSchedulerConfig {
    /// 创建始终使用 CPU 的配置
    pub fn cpu_only() -> Self {
        Self {
            forced_backend: Some(BackendType::Cpu),
            ..Default::default()
        }
    }

    /// 创建始终使用 GPU 的配置
    pub fn gpu_only() -> Self {
        Self {
            forced_backend: Some(BackendType::Gpu),
            min_gpu_threshold: 0,
            ..Default::default()
        }
    }

    /// 创建保守配置（更倾向于使用 CPU）
    pub fn conservative() -> Self {
        Self {
            min_gpu_threshold: 10_000,
            max_cpu_threshold: 100_000,
            adaptive_thresholds: false,
            ..Default::default()
        }
    }

    /// 创建激进配置（更倾向于使用 GPU）
    pub fn aggressive() -> Self {
        Self {
            min_gpu_threshold: 500,
            max_cpu_threshold: 10_000,
            adaptive_thresholds: true,
            ..Default::default()
        }
    }
}

// ============================================================================
// 性能统计
// ============================================================================

/// 后端性能统计
#[derive(Debug, Clone)]
pub struct BackendStats {
    /// 总调用次数
    pub total_calls: u64,
    /// 总执行时间（微秒）
    pub total_time_us: u64,
    /// 平均执行时间（微秒）
    pub avg_time_us: f64,
    /// 最小执行时间（微秒）
    pub min_time_us: u64,
    /// 最大执行时间（微秒）
    pub max_time_us: u64,
    /// 处理的总单元数
    pub total_cells: u64,
    /// 平均吞吐量（单元/秒）
    pub avg_throughput: f64,
}

impl Default for BackendStats {
    fn default() -> Self {
        Self {
            total_calls: 0,
            total_time_us: 0,
            avg_time_us: 0.0,
            min_time_us: u64::MAX,
            max_time_us: 0,
            total_cells: 0,
            avg_throughput: 0.0,
        }
    }
}

impl BackendStats {
    /// 更新统计数据
    pub fn update(&mut self, execution_time_us: u64, cell_count: usize) {
        self.total_calls += 1;
        self.total_time_us += execution_time_us;
        self.min_time_us = self.min_time_us.min(execution_time_us);
        self.max_time_us = self.max_time_us.max(execution_time_us);
        self.total_cells += cell_count as u64;

        self.avg_time_us = self.total_time_us as f64 / self.total_calls as f64;

        if self.total_time_us > 0 {
            self.avg_throughput =
                (self.total_cells as f64 * 1_000_000.0) / self.total_time_us as f64;
        }
    }
}

/// 调度器统计
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// CPU 后端统计
    pub cpu_stats: BackendStats,
    /// GPU 后端统计
    pub gpu_stats: BackendStats,
    /// 混合后端统计
    pub hybrid_stats: BackendStats,
    /// 后端切换次数
    pub switch_count: u64,
    /// 当前后端
    pub current_backend: BackendType,
    /// 调度决策总数
    pub total_decisions: u64,
}

impl Default for SchedulerStats {
    fn default() -> Self {
        Self {
            cpu_stats: BackendStats::default(),
            gpu_stats: BackendStats::default(),
            hybrid_stats: BackendStats::default(),
            switch_count: 0,
            current_backend: BackendType::Cpu,
            total_decisions: 0,
        }
    }
}

// ============================================================================
// 混合调度器
// ============================================================================

/// GPU/CPU 混合调度器
///
/// 根据问题规模、设备能力和历史性能数据，自动选择最优的计算后端。
///
/// # 示例
///
/// ```ignore
/// use mh_physics::gpu::hybrid_scheduler::{HybridScheduler, HybridSchedulerConfig};
///
/// let config = HybridSchedulerConfig::default();
/// let scheduler = HybridScheduler::new(config, Some(device_caps));
///
/// // 获取调度决策
/// let decision = scheduler.decide(50_000, 1024 * 1024 * 100);
/// println!("使用后端: {:?}", decision.backend);
///
/// // 记录执行结果
/// scheduler.record_execution(decision.backend, 50_000, 2500);
/// ```
pub struct HybridScheduler {
    /// 配置
    config: HybridSchedulerConfig,

    /// 设备能力
    device_caps: Option<DeviceCapabilities>,

    /// 性能历史记录
    history: VecDeque<PerformanceSample>,

    /// 统计数据
    stats: SchedulerStats,

    /// 上次切换时间
    last_switch_time: Option<Instant>,

    /// 当前使用的后端
    current_backend: BackendType,

    /// 自适应阈值（从历史数据学习）
    adaptive_min_threshold: AtomicU64,
    adaptive_max_threshold: AtomicU64,
}

impl HybridScheduler {
    /// 创建新的混合调度器
    ///
    /// # 参数
    ///
    /// * `config` - 调度器配置
    /// * `device_caps` - GPU 设备能力（None 表示无 GPU）
    pub fn new(config: HybridSchedulerConfig, device_caps: Option<DeviceCapabilities>) -> Self {
        let initial_backend = if device_caps.is_none() {
            BackendType::Cpu
        } else if let Some(forced) = config.forced_backend {
            forced
        } else {
            BackendType::Cpu // 默认从 CPU 开始
        };

        Self {
            adaptive_min_threshold: AtomicU64::new(config.min_gpu_threshold as u64),
            adaptive_max_threshold: AtomicU64::new(config.max_cpu_threshold as u64),
            config,
            device_caps,
            history: VecDeque::with_capacity(128),
            stats: SchedulerStats::default(),
            last_switch_time: None,
            current_backend: initial_backend,
        }
    }

    /// 创建仅 CPU 的调度器
    pub fn cpu_only() -> Self {
        Self::new(HybridSchedulerConfig::cpu_only(), None)
    }

    /// 获取当前后端
    pub fn current_backend(&self) -> BackendType {
        self.current_backend
    }

    /// 获取统计数据
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// 检查 GPU 是否可用
    pub fn is_gpu_available(&self) -> bool {
        self.device_caps.is_some()
    }

    /// 获取设备能力
    pub fn device_capabilities(&self) -> Option<&DeviceCapabilities> {
        self.device_caps.as_ref()
    }

    // =========================================================================
    // 调度决策
    // =========================================================================

    /// 根据问题规模做出调度决策
    ///
    /// # 参数
    ///
    /// * `cell_count` - 单元数量
    /// * `memory_required` - 预估所需显存（字节）
    ///
    /// # 返回
    ///
    /// 调度决策，包含后端类型、预估时间和置信度
    pub fn decide(&self, cell_count: usize, memory_required: usize) -> SchedulingDecision {
        // 1. 检查强制后端
        if let Some(forced) = self.config.forced_backend {
            return SchedulingDecision {
                backend: forced,
                estimated_time_ms: self.estimate_time(forced, cell_count),
                confidence: 1.0,
                reason: DecisionReason::ForcedBackend,
            };
        }

        // 2. 检查 GPU 可用性
        if self.device_caps.is_none() {
            return SchedulingDecision {
                backend: BackendType::Cpu,
                estimated_time_ms: self.estimate_time(BackendType::Cpu, cell_count),
                confidence: 1.0,
                reason: DecisionReason::GpuUnavailable,
            };
        }

        // 3. 检查显存限制
        if !self.check_memory_available(memory_required) {
            return SchedulingDecision {
                backend: BackendType::Cpu,
                estimated_time_ms: self.estimate_time(BackendType::Cpu, cell_count),
                confidence: 0.9,
                reason: DecisionReason::InsufficientGpuMemory,
            };
        }

        // 4. 获取自适应阈值
        let min_threshold = if self.config.adaptive_thresholds {
            self.adaptive_min_threshold.load(Ordering::Relaxed) as usize
        } else {
            self.config.min_gpu_threshold
        };

        let max_threshold = if self.config.adaptive_thresholds {
            self.adaptive_max_threshold.load(Ordering::Relaxed) as usize
        } else {
            self.config.max_cpu_threshold
        };

        // 5. 基于问题规模的简单决策
        if cell_count < min_threshold {
            return SchedulingDecision {
                backend: BackendType::Cpu,
                estimated_time_ms: self.estimate_time(BackendType::Cpu, cell_count),
                confidence: 0.95,
                reason: DecisionReason::ProblemTooSmall,
            };
        }

        if cell_count > self.config.hybrid_threshold {
            return SchedulingDecision {
                backend: BackendType::Hybrid,
                estimated_time_ms: self.estimate_time(BackendType::Hybrid, cell_count),
                confidence: 0.8,
                reason: DecisionReason::VeryLargeScale,
            };
        }

        if cell_count > max_threshold {
            return SchedulingDecision {
                backend: BackendType::Gpu,
                estimated_time_ms: self.estimate_time(BackendType::Gpu, cell_count),
                confidence: 0.9,
                reason: DecisionReason::ProblemLargeEnough,
            };
        }

        // 6. 中间区域：基于历史性能数据决策
        self.decide_from_history(cell_count)
    }

    /// 基于历史数据做出决策
    fn decide_from_history(&self, cell_count: usize) -> SchedulingDecision {
        // 查找相似规模的历史样本
        let similar_samples: Vec<_> = self
            .history
            .iter()
            .filter(|s| {
                let ratio = s.problem_size as f64 / cell_count as f64;
                ratio > 0.5 && ratio < 2.0
            })
            .collect();

        if similar_samples.is_empty() {
            // 没有历史数据，使用启发式规则
            let backend = if cell_count > self.config.min_gpu_threshold * 5 {
                BackendType::Gpu
            } else {
                BackendType::Cpu
            };

            return SchedulingDecision {
                backend,
                estimated_time_ms: self.estimate_time(backend, cell_count),
                confidence: 0.5,
                reason: DecisionReason::HistoricalPerformance,
            };
        }

        // 计算 CPU 和 GPU 的平均性能
        let mut cpu_total_time = 0u64;
        let mut cpu_count = 0usize;
        let mut gpu_total_time = 0u64;
        let mut gpu_count = 0usize;

        for sample in &similar_samples {
            match sample.backend {
                BackendType::Cpu => {
                    cpu_total_time += sample.execution_time_us;
                    cpu_count += 1;
                }
                BackendType::Gpu => {
                    gpu_total_time += sample.execution_time_us;
                    gpu_count += 1;
                }
                BackendType::Hybrid => {}
            }
        }

        let cpu_avg = if cpu_count > 0 {
            cpu_total_time as f64 / cpu_count as f64
        } else {
            f64::INFINITY
        };

        let gpu_avg = if gpu_count > 0 {
            gpu_total_time as f64 / gpu_count as f64
        } else {
            f64::INFINITY
        };

        // 选择更快的后端
        let (backend, estimated_us) = if cpu_avg < gpu_avg {
            (BackendType::Cpu, cpu_avg)
        } else {
            (BackendType::Gpu, gpu_avg)
        };

        let confidence = if cpu_count + gpu_count > 10 { 0.85 } else { 0.7 };

        SchedulingDecision {
            backend,
            estimated_time_ms: estimated_us / 1000.0,
            confidence,
            reason: DecisionReason::HistoricalPerformance,
        }
    }

    /// 估算执行时间
    fn estimate_time(&self, backend: BackendType, cell_count: usize) -> f64 {
        // 基于统计数据估算
        let stats = match backend {
            BackendType::Cpu => &self.stats.cpu_stats,
            BackendType::Gpu => &self.stats.gpu_stats,
            BackendType::Hybrid => &self.stats.hybrid_stats,
        };

        if stats.avg_throughput > 0.0 {
            (cell_count as f64 / stats.avg_throughput) * 1000.0
        } else {
            // 默认估算（基于经验值）
            match backend {
                BackendType::Cpu => cell_count as f64 * 0.001, // 1μs per cell
                BackendType::Gpu => {
                    let overhead = 0.5; // 0.5ms 固定开销
                    overhead + cell_count as f64 * 0.0001 // 0.1μs per cell
                }
                BackendType::Hybrid => cell_count as f64 * 0.00005,
            }
        }
    }

    /// 检查显存是否足够
    fn check_memory_available(&self, required: usize) -> bool {
        if let Some(caps) = &self.device_caps {
            let available = (caps.memory.total_bytes as f64 * self.config.gpu_memory_limit) as usize;
            required <= available
        } else {
            false
        }
    }

    // =========================================================================
    // 执行记录和自适应
    // =========================================================================

    /// 记录执行结果
    ///
    /// 用于更新性能历史和统计数据。
    ///
    /// # 参数
    ///
    /// * `backend` - 使用的后端
    /// * `cell_count` - 处理的单元数
    /// * `execution_time_us` - 执行时间（微秒）
    pub fn record_execution(&mut self, backend: BackendType, cell_count: usize, execution_time_us: u64) {
        // 更新统计
        match backend {
            BackendType::Cpu => self.stats.cpu_stats.update(execution_time_us, cell_count),
            BackendType::Gpu => self.stats.gpu_stats.update(execution_time_us, cell_count),
            BackendType::Hybrid => self.stats.hybrid_stats.update(execution_time_us, cell_count),
        }

        // 添加到历史
        let sample = PerformanceSample {
            backend,
            problem_size: cell_count,
            execution_time_us,
            timestamp: Instant::now(),
        };

        self.history.push_back(sample);

        // 保持历史窗口大小
        while self.history.len() > self.config.history_window_size {
            self.history.pop_front();
        }

        // 自适应阈值调整
        if self.config.adaptive_thresholds && self.history.len() >= 20 {
            self.update_adaptive_thresholds();
        }

        // 更新当前后端
        if backend != self.current_backend {
            self.current_backend = backend;
            self.stats.switch_count += 1;
            self.last_switch_time = Some(Instant::now());
        }

        self.stats.total_decisions += 1;
        self.stats.current_backend = self.current_backend;
    }

    /// 更新自适应阈值
    fn update_adaptive_thresholds(&mut self) {
        // 分析历史数据，找到 CPU 和 GPU 的交叉点
        let mut crossover_sizes = Vec::new();

        // 按问题规模分桶
        let mut cpu_by_size: std::collections::HashMap<usize, Vec<u64>> =
            std::collections::HashMap::new();
        let mut gpu_by_size: std::collections::HashMap<usize, Vec<u64>> =
            std::collections::HashMap::new();

        for sample in &self.history {
            let bucket = (sample.problem_size / 1000) * 1000; // 1K 为一个桶
            match sample.backend {
                BackendType::Cpu => {
                    cpu_by_size.entry(bucket).or_default().push(sample.execution_time_us);
                }
                BackendType::Gpu => {
                    gpu_by_size.entry(bucket).or_default().push(sample.execution_time_us);
                }
                _ => {}
            }
        }

        // 找到交叉点
        for (&size, cpu_times) in &cpu_by_size {
            if let Some(gpu_times) = gpu_by_size.get(&size) {
                let cpu_avg: f64 = cpu_times.iter().sum::<u64>() as f64 / cpu_times.len() as f64;
                let gpu_avg: f64 = gpu_times.iter().sum::<u64>() as f64 / gpu_times.len() as f64;

                // 如果 GPU 开始变得更快，记录这个点
                if gpu_avg < cpu_avg * 0.9 {
                    crossover_sizes.push(size);
                }
            }
        }

        if !crossover_sizes.is_empty() {
            let min_crossover = *crossover_sizes.iter().min().unwrap();
            // 设置新阈值（略低于交叉点）
            let new_min = (min_crossover as f64 * 0.8) as u64;
            self.adaptive_min_threshold.store(new_min.max(100), Ordering::Relaxed);

            let max_crossover = *crossover_sizes.iter().max().unwrap();
            let new_max = (max_crossover as f64 * 1.2) as u64;
            self.adaptive_max_threshold.store(new_max, Ordering::Relaxed);
        }
    }

    // =========================================================================
    // 工具方法
    // =========================================================================

    /// 重置统计数据
    pub fn reset_stats(&mut self) {
        self.stats = SchedulerStats::default();
        self.history.clear();
        self.adaptive_min_threshold
            .store(self.config.min_gpu_threshold as u64, Ordering::Relaxed);
        self.adaptive_max_threshold
            .store(self.config.max_cpu_threshold as u64, Ordering::Relaxed);
    }

    /// 强制切换后端
    ///
    /// 临时覆盖自动决策。
    pub fn force_backend(&mut self, backend: BackendType) {
        self.config.forced_backend = Some(backend);
    }

    /// 取消强制后端
    pub fn unforce_backend(&mut self) {
        self.config.forced_backend = None;
    }

    /// 检查是否应该切换后端（考虑最小切换间隔）
    pub fn should_switch(&self, new_backend: BackendType) -> bool {
        if new_backend == self.current_backend {
            return false;
        }

        if let Some(last_switch) = self.last_switch_time {
            if last_switch.elapsed() < self.config.min_switch_interval {
                return false;
            }
        }

        true
    }

    /// 获取性能报告
    pub fn performance_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== 混合调度器性能报告 ===\n\n");

        report.push_str(&format!(
            "当前后端: {}\n",
            self.current_backend
        ));
        report.push_str(&format!(
            "总决策数: {}\n",
            self.stats.total_decisions
        ));
        report.push_str(&format!(
            "后端切换次数: {}\n\n",
            self.stats.switch_count
        ));

        // CPU 统计
        report.push_str("--- CPU 后端 ---\n");
        report.push_str(&format!(
            "  调用次数: {}\n",
            self.stats.cpu_stats.total_calls
        ));
        if self.stats.cpu_stats.total_calls > 0 {
            report.push_str(&format!(
                "  平均时间: {:.2} μs\n",
                self.stats.cpu_stats.avg_time_us
            ));
            report.push_str(&format!(
                "  吞吐量: {:.0} 单元/秒\n",
                self.stats.cpu_stats.avg_throughput
            ));
        }

        // GPU 统计
        report.push_str("\n--- GPU 后端 ---\n");
        report.push_str(&format!(
            "  调用次数: {}\n",
            self.stats.gpu_stats.total_calls
        ));
        if self.stats.gpu_stats.total_calls > 0 {
            report.push_str(&format!(
                "  平均时间: {:.2} μs\n",
                self.stats.gpu_stats.avg_time_us
            ));
            report.push_str(&format!(
                "  吞吐量: {:.0} 单元/秒\n",
                self.stats.gpu_stats.avg_throughput
            ));
        }

        // 自适应阈值
        if self.config.adaptive_thresholds {
            report.push_str("\n--- 自适应阈值 ---\n");
            report.push_str(&format!(
                "  最小 GPU 阈值: {} 单元\n",
                self.adaptive_min_threshold.load(Ordering::Relaxed)
            ));
            report.push_str(&format!(
                "  最大 CPU 阈值: {} 单元\n",
                self.adaptive_max_threshold.load(Ordering::Relaxed)
            ));
        }

        report
    }
}

// ============================================================================
// GPU 任务执行接口
// ============================================================================

/// GPU 计算任务 trait
///
/// 实现此 trait 可以通过 HybridScheduler 自动选择执行后端。
///
/// # 示例
///
/// ```ignore
/// struct MyComputeTask { ... }
///
/// impl GpuTask for MyComputeTask {
///     fn name(&self) -> &str { "my_compute" }
///     fn estimated_workload(&self) -> usize { self.cell_count }
///     fn estimated_memory(&self) -> usize { self.cell_count * 32 }
///     
///     fn execute_cpu(&self) -> Result<(), GpuError> { ... }
///     fn execute_gpu(&self, ctx: &GpuContext) -> Result<(), GpuError> { ... }
/// }
/// ```
pub trait GpuTask: Send + Sync {
    /// 任务名称（用于日志和调试）
    fn name(&self) -> &str;

    /// 估算计算工作量（单元数或等效计算量）
    fn estimated_workload(&self) -> usize;

    /// 估算所需内存（字节）
    fn estimated_memory(&self) -> usize;

    /// 在 CPU 上执行
    fn execute_cpu(&self) -> Result<(), GpuError>;

    /// 在 GPU 上执行
    ///
    /// # 参数
    /// - `ctx`: GPU 上下文，包含设备、队列等资源
    fn execute_gpu(&self, ctx: &super::GpuContext) -> Result<(), GpuError>;
}

/// 任务执行结果
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// 实际使用的后端
    pub backend_used: BackendType,
    /// 执行时间
    pub execution_time: Duration,
    /// 调度决策信息
    pub decision: SchedulingDecision,
    /// 任务名称
    pub task_name: String,
}

impl HybridScheduler {
    /// 执行计算任务（自动选择后端）
    ///
    /// 根据任务的工作量和当前调度策略，自动选择 CPU 或 GPU 后端执行。
    /// 执行完成后自动记录性能数据用于后续决策优化。
    ///
    /// # 参数
    /// - `task`: 实现 `GpuTask` trait 的任务对象
    /// - `gpu_ctx`: GPU 上下文（可选，用于 GPU 执行）
    ///
    /// # 返回
    /// 执行结果，包含使用的后端、执行时间等信息
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let result = scheduler.execute(&my_task, gpu_context.as_ref())?;
    /// println!("使用后端: {:?}, 耗时: {:?}", result.backend_used, result.execution_time);
    /// ```
    pub fn execute<T: GpuTask>(
        &mut self,
        task: &T,
        gpu_ctx: Option<&super::GpuContext>,
    ) -> Result<ExecutionResult, GpuError> {
        let workload = task.estimated_workload();
        let memory = task.estimated_memory();
        let decision = self.decide(workload, memory);

        let start = Instant::now();

        let actual_backend = match decision.backend {
            BackendType::Cpu => {
                task.execute_cpu()?;
                BackendType::Cpu
            }
            BackendType::Gpu => {
                if let Some(ctx) = gpu_ctx {
                    task.execute_gpu(ctx)?;
                    BackendType::Gpu
                } else {
                    // GPU 上下文不可用，回退到 CPU
                    task.execute_cpu()?;
                    BackendType::Cpu
                }
            }
            BackendType::Hybrid => {
                // TODO: 实现混合执行（CPU+GPU 并行）
                // 当前回退到 GPU 或 CPU
                if let Some(ctx) = gpu_ctx {
                    task.execute_gpu(ctx)?;
                    BackendType::Gpu
                } else {
                    task.execute_cpu()?;
                    BackendType::Cpu
                }
            }
        };

        let elapsed = start.elapsed();

        // 记录性能数据
        self.record_execution(actual_backend, workload, elapsed.as_micros() as u64);

        Ok(ExecutionResult {
            backend_used: actual_backend,
            execution_time: elapsed,
            decision,
            task_name: task.name().to_string(),
        })
    }

    /// 批量执行多个任务
    ///
    /// 对于多个小任务，可能会合并到同一后端执行以减少切换开销。
    pub fn execute_batch<T: GpuTask>(
        &mut self,
        tasks: &[T],
        gpu_ctx: Option<&super::GpuContext>,
    ) -> Result<Vec<ExecutionResult>, GpuError> {
        let mut results = Vec::with_capacity(tasks.len());

        for task in tasks {
            results.push(self.execute(task, gpu_ctx)?);
        }

        Ok(results)
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::capabilities::MemoryInfo;

    fn create_test_caps() -> DeviceCapabilities {
        DeviceCapabilities {
            device_name: "Test GPU".to_string(),
            device_type: DeviceType::DiscreteGpu,
            memory: MemoryInfo {
                total_bytes: 4 * 1024 * 1024 * 1024, // 4GB
                shared: false,
            },
            max_buffer_size: 256 * 1024 * 1024,
            max_workgroup_size: 256,
            max_workgroups: [65535, 65535, 65535],
            features: Default::default(),
        }
    }

    #[test]
    fn test_cpu_only_scheduler() {
        let scheduler = HybridScheduler::cpu_only();
        let decision = scheduler.decide(100_000, 0);

        assert_eq!(decision.backend, BackendType::Cpu);
        assert_eq!(decision.reason, DecisionReason::GpuUnavailable);
    }

    #[test]
    fn test_small_problem_uses_cpu() {
        let caps = create_test_caps();
        let scheduler = HybridScheduler::new(HybridSchedulerConfig::default(), Some(caps));

        let decision = scheduler.decide(500, 1024);

        assert_eq!(decision.backend, BackendType::Cpu);
        assert_eq!(decision.reason, DecisionReason::ProblemTooSmall);
    }

    #[test]
    fn test_large_problem_uses_gpu() {
        let caps = create_test_caps();
        let scheduler = HybridScheduler::new(HybridSchedulerConfig::default(), Some(caps));

        let decision = scheduler.decide(100_000, 10 * 1024 * 1024);

        assert_eq!(decision.backend, BackendType::Gpu);
        assert_eq!(decision.reason, DecisionReason::ProblemLargeEnough);
    }

    #[test]
    fn test_forced_backend() {
        let caps = create_test_caps();
        let config = HybridSchedulerConfig {
            forced_backend: Some(BackendType::Gpu),
            ..Default::default()
        };
        let scheduler = HybridScheduler::new(config, Some(caps));

        // 即使问题规模很小，也使用 GPU
        let decision = scheduler.decide(100, 1024);

        assert_eq!(decision.backend, BackendType::Gpu);
        assert_eq!(decision.reason, DecisionReason::ForcedBackend);
    }

    #[test]
    fn test_record_execution() {
        let caps = create_test_caps();
        let mut scheduler = HybridScheduler::new(HybridSchedulerConfig::default(), Some(caps));

        scheduler.record_execution(BackendType::Cpu, 10_000, 5000);
        scheduler.record_execution(BackendType::Gpu, 50_000, 3000);

        assert_eq!(scheduler.stats().cpu_stats.total_calls, 1);
        assert_eq!(scheduler.stats().gpu_stats.total_calls, 1);
        assert_eq!(scheduler.stats().total_decisions, 2);
    }

    #[test]
    fn test_very_large_scale_uses_hybrid() {
        let caps = create_test_caps();
        let scheduler = HybridScheduler::new(HybridSchedulerConfig::default(), Some(caps));

        let decision = scheduler.decide(2_000_000, 500 * 1024 * 1024);

        assert_eq!(decision.backend, BackendType::Hybrid);
        assert_eq!(decision.reason, DecisionReason::VeryLargeScale);
    }

    #[test]
    fn test_performance_report() {
        let mut scheduler = HybridScheduler::cpu_only();
        scheduler.record_execution(BackendType::Cpu, 10_000, 5000);
        scheduler.record_execution(BackendType::Cpu, 20_000, 8000);

        let report = scheduler.performance_report();

        assert!(report.contains("混合调度器性能报告"));
        assert!(report.contains("CPU 后端"));
        assert!(report.contains("调用次数: 2"));
    }
}
