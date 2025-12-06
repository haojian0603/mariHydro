// crates/mh_physics/src/gpu/profiler.rs

//! GPU 性能监控和分析工具
//!
//! 提供 GPU 计算的性能监控、瓶颈分析和优化建议。
//!
//! # 功能
//!
//! - 实时性能指标收集
//! - 计算管线性能分析
//! - 内存带宽监控
//! - 性能瓶颈检测
//! - 优化建议生成
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::gpu::profiler::{GpuProfiler, ProfilerConfig};
//!
//! let profiler = GpuProfiler::new(ProfilerConfig::default());
//!
//! // 开始记录
//! let token = profiler.begin_scope("flux_compute");
//!
//! // ... 执行 GPU 计算 ...
//!
//! profiler.end_scope(token);
//!
//! // 获取分析报告
//! let report = profiler.generate_report();
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// ============================================================================
// 核心类型
// ============================================================================

/// 性能指标类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// 执行时间
    ExecutionTime,
    /// 内存带宽 (GB/s)
    MemoryBandwidth,
    /// 计算吞吐量 (GFLOPS)
    ComputeThroughput,
    /// GPU 占用率
    GpuOccupancy,
    /// 内存使用量 (bytes)
    MemoryUsage,
    /// 缓冲区传输时间
    BufferTransferTime,
    /// 着色器编译时间
    ShaderCompileTime,
    /// 同步等待时间
    SyncWaitTime,
}

impl std::fmt::Display for MetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricType::ExecutionTime => write!(f, "执行时间"),
            MetricType::MemoryBandwidth => write!(f, "内存带宽"),
            MetricType::ComputeThroughput => write!(f, "计算吞吐量"),
            MetricType::GpuOccupancy => write!(f, "GPU占用率"),
            MetricType::MemoryUsage => write!(f, "内存使用"),
            MetricType::BufferTransferTime => write!(f, "传输时间"),
            MetricType::ShaderCompileTime => write!(f, "编译时间"),
            MetricType::SyncWaitTime => write!(f, "同步等待"),
        }
    }
}

/// 性能样本
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    /// 样本时间戳
    pub timestamp: Instant,
    /// 指标值
    pub value: f64,
    /// 关联的问题规模（可选）
    pub problem_size: Option<usize>,
}

/// 性能范围标记
#[derive(Debug, Clone)]
pub struct ScopeToken {
    /// 范围名称
    pub name: String,
    /// 开始时间
    pub start_time: Instant,
    /// 唯一标识
    pub id: u64,
}

/// 性能范围统计
#[derive(Debug, Clone)]
pub struct ScopeStats {
    /// 范围名称
    pub name: String,
    /// 调用次数
    pub call_count: u64,
    /// 总时间（微秒）
    pub total_time_us: u64,
    /// 平均时间（微秒）
    pub avg_time_us: f64,
    /// 最小时间（微秒）
    pub min_time_us: u64,
    /// 最大时间（微秒）
    pub max_time_us: u64,
    /// 时间方差
    pub variance_us: f64,
    /// 最近 N 次的时间
    pub recent_times: Vec<u64>,
}

impl Default for ScopeStats {
    fn default() -> Self {
        Self {
            name: String::new(),
            call_count: 0,
            total_time_us: 0,
            avg_time_us: 0.0,
            min_time_us: u64::MAX,
            max_time_us: 0,
            variance_us: 0.0,
            recent_times: Vec::with_capacity(100),
        }
    }
}

impl ScopeStats {
    /// 更新统计
    pub fn update(&mut self, elapsed_us: u64) {
        self.call_count += 1;
        self.total_time_us += elapsed_us;
        self.min_time_us = self.min_time_us.min(elapsed_us);
        self.max_time_us = self.max_time_us.max(elapsed_us);
        self.avg_time_us = self.total_time_us as f64 / self.call_count as f64;

        // 更新最近时间
        if self.recent_times.len() >= 100 {
            self.recent_times.remove(0);
        }
        self.recent_times.push(elapsed_us);

        // 计算方差
        if self.call_count > 1 {
            let mean = self.avg_time_us;
            let sum_sq: f64 = self
                .recent_times
                .iter()
                .map(|&t| {
                    let diff = t as f64 - mean;
                    diff * diff
                })
                .sum();
            self.variance_us = sum_sq / self.recent_times.len() as f64;
        }
    }

    /// 获取标准差
    pub fn std_dev_us(&self) -> f64 {
        self.variance_us.sqrt()
    }

    /// 获取变异系数
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.avg_time_us > 0.0 {
            self.std_dev_us() / self.avg_time_us
        } else {
            0.0
        }
    }
}

// ============================================================================
// 性能瓶颈分析
// ============================================================================

/// 瓶颈类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckType {
    /// 计算瓶颈
    Compute,
    /// 内存带宽瓶颈
    MemoryBandwidth,
    /// 数据传输瓶颈
    DataTransfer,
    /// 同步瓶颈
    Synchronization,
    /// 着色器编译瓶颈
    ShaderCompilation,
    /// 未知
    Unknown,
}

impl std::fmt::Display for BottleneckType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BottleneckType::Compute => write!(f, "计算"),
            BottleneckType::MemoryBandwidth => write!(f, "内存带宽"),
            BottleneckType::DataTransfer => write!(f, "数据传输"),
            BottleneckType::Synchronization => write!(f, "同步"),
            BottleneckType::ShaderCompilation => write!(f, "着色器编译"),
            BottleneckType::Unknown => write!(f, "未知"),
        }
    }
}

/// 瓶颈分析结果
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    /// 主要瓶颈
    pub primary_bottleneck: BottleneckType,
    /// 瓶颈严重程度 [0, 1]
    pub severity: f64,
    /// 受影响的范围
    pub affected_scopes: Vec<String>,
    /// 优化建议
    pub suggestions: Vec<String>,
}

// ============================================================================
// 配置
// ============================================================================

/// 性能监控配置
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// 是否启用
    pub enabled: bool,
    /// 历史样本保留数量
    pub history_size: usize,
    /// 自动分析间隔（毫秒）
    pub analysis_interval_ms: u64,
    /// 是否记录详细时序
    pub detailed_timing: bool,
    /// 是否启用内存追踪
    pub track_memory: bool,
    /// 警告阈值（毫秒）
    pub warning_threshold_ms: f64,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            history_size: 1000,
            analysis_interval_ms: 1000,
            detailed_timing: true,
            track_memory: true,
            warning_threshold_ms: 100.0,
        }
    }
}

impl ProfilerConfig {
    /// 禁用的配置
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// 轻量级配置
    pub fn lightweight() -> Self {
        Self {
            enabled: true,
            history_size: 100,
            detailed_timing: false,
            track_memory: false,
            ..Default::default()
        }
    }

    /// 详细配置
    pub fn detailed() -> Self {
        Self {
            enabled: true,
            history_size: 10000,
            detailed_timing: true,
            track_memory: true,
            analysis_interval_ms: 500,
            warning_threshold_ms: 50.0,
        }
    }
}

// ============================================================================
// GPU 性能监控器
// ============================================================================

/// GPU 性能监控器
///
/// 收集和分析 GPU 计算性能数据。
pub struct GpuProfiler {
    /// 配置
    config: ProfilerConfig,

    /// 范围统计
    scope_stats: RwLock<HashMap<String, ScopeStats>>,

    /// 指标历史
    metric_history: RwLock<HashMap<MetricType, Vec<PerformanceSample>>>,

    /// 下一个 token ID
    next_token_id: AtomicU64,

    /// 活动范围
    active_scopes: RwLock<HashMap<u64, ScopeToken>>,

    /// 总 GPU 时间（微秒）
    total_gpu_time_us: AtomicU64,

    /// 总传输时间（微秒）
    total_transfer_time_us: AtomicU64,

    /// 峰值内存使用（字节）
    peak_memory_bytes: AtomicU64,

    /// 当前内存使用（字节）
    current_memory_bytes: AtomicU64,

    /// 上次分析时间
    last_analysis_time: RwLock<Option<Instant>>,

    /// 缓存的瓶颈分析
    cached_analysis: RwLock<Option<BottleneckAnalysis>>,
}

impl GpuProfiler {
    /// 创建新的性能监控器
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            scope_stats: RwLock::new(HashMap::new()),
            metric_history: RwLock::new(HashMap::new()),
            next_token_id: AtomicU64::new(0),
            active_scopes: RwLock::new(HashMap::new()),
            total_gpu_time_us: AtomicU64::new(0),
            total_transfer_time_us: AtomicU64::new(0),
            peak_memory_bytes: AtomicU64::new(0),
            current_memory_bytes: AtomicU64::new(0),
            last_analysis_time: RwLock::new(None),
            cached_analysis: RwLock::new(None),
        }
    }

    /// 检查是否启用
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    // =========================================================================
    // 范围计时
    // =========================================================================

    /// 开始一个性能范围
    ///
    /// # 参数
    ///
    /// * `name` - 范围名称（如 "flux_compute", "time_step"）
    ///
    /// # 返回
    ///
    /// 范围标记，需要传给 `end_scope`
    pub fn begin_scope(&self, name: &str) -> ScopeToken {
        let id = self.next_token_id.fetch_add(1, Ordering::Relaxed);
        let token = ScopeToken {
            name: name.to_string(),
            start_time: Instant::now(),
            id,
        };

        if self.config.enabled {
            if let Ok(mut active) = self.active_scopes.write() {
                active.insert(id, token.clone());
            }
        }

        token
    }

    /// 结束一个性能范围
    ///
    /// # 参数
    ///
    /// * `token` - 从 `begin_scope` 返回的标记
    pub fn end_scope(&self, token: ScopeToken) {
        if !self.config.enabled {
            return;
        }

        let elapsed = token.start_time.elapsed();
        let elapsed_us = elapsed.as_micros() as u64;

        // 移除活动范围
        if let Ok(mut active) = self.active_scopes.write() {
            active.remove(&token.id);
        }

        // 更新统计
        if let Ok(mut stats) = self.scope_stats.write() {
            let entry = stats.entry(token.name.clone()).or_insert_with(|| {
                let mut s = ScopeStats::default();
                s.name = token.name.clone();
                s
            });
            entry.update(elapsed_us);
        }

        // 更新总 GPU 时间
        self.total_gpu_time_us.fetch_add(elapsed_us, Ordering::Relaxed);

        // 检查是否超过警告阈值
        if elapsed.as_secs_f64() * 1000.0 > self.config.warning_threshold_ms {
            // 可以在这里添加日志或回调
        }
    }

    /// RAII 风格的范围计时
    pub fn scope(&self, name: &str) -> ProfilerGuard<'_> {
        ProfilerGuard {
            profiler: self,
            token: self.begin_scope(name),
        }
    }

    // =========================================================================
    // 指标记录
    // =========================================================================

    /// 记录性能指标
    ///
    /// # 参数
    ///
    /// * `metric` - 指标类型
    /// * `value` - 指标值
    /// * `problem_size` - 关联的问题规模
    pub fn record_metric(&self, metric: MetricType, value: f64, problem_size: Option<usize>) {
        if !self.config.enabled {
            return;
        }

        let sample = PerformanceSample {
            timestamp: Instant::now(),
            value,
            problem_size,
        };

        if let Ok(mut history) = self.metric_history.write() {
            let samples = history.entry(metric).or_insert_with(Vec::new);
            samples.push(sample);

            // 保持历史大小
            while samples.len() > self.config.history_size {
                samples.remove(0);
            }
        }

        // 更新特定指标的原子计数
        match metric {
            MetricType::MemoryUsage => {
                let bytes = value as u64;
                self.current_memory_bytes.store(bytes, Ordering::Relaxed);
                let mut peak = self.peak_memory_bytes.load(Ordering::Relaxed);
                while bytes > peak {
                    match self.peak_memory_bytes.compare_exchange_weak(
                        peak,
                        bytes,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(p) => peak = p,
                    }
                }
            }
            MetricType::BufferTransferTime => {
                self.total_transfer_time_us
                    .fetch_add(value as u64, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// 记录内存分配
    pub fn record_allocation(&self, bytes: usize) {
        if !self.config.track_memory {
            return;
        }
        let current = self.current_memory_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        let new_total = current + bytes as u64;

        let mut peak = self.peak_memory_bytes.load(Ordering::Relaxed);
        while new_total > peak {
            match self.peak_memory_bytes.compare_exchange_weak(
                peak,
                new_total,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    /// 记录内存释放
    pub fn record_deallocation(&self, bytes: usize) {
        if !self.config.track_memory {
            return;
        }
        self.current_memory_bytes
            .fetch_sub(bytes as u64, Ordering::Relaxed);
    }

    // =========================================================================
    // 统计查询
    // =========================================================================

    /// 获取范围统计
    pub fn get_scope_stats(&self, name: &str) -> Option<ScopeStats> {
        self.scope_stats
            .read()
            .ok()
            .and_then(|stats| stats.get(name).cloned())
    }

    /// 获取所有范围统计
    pub fn get_all_scope_stats(&self) -> HashMap<String, ScopeStats> {
        self.scope_stats
            .read()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// 获取指标历史
    pub fn get_metric_history(&self, metric: MetricType) -> Vec<PerformanceSample> {
        self.metric_history
            .read()
            .ok()
            .and_then(|h| h.get(&metric).cloned())
            .unwrap_or_default()
    }

    /// 获取总 GPU 时间
    pub fn total_gpu_time(&self) -> Duration {
        Duration::from_micros(self.total_gpu_time_us.load(Ordering::Relaxed))
    }

    /// 获取总传输时间
    pub fn total_transfer_time(&self) -> Duration {
        Duration::from_micros(self.total_transfer_time_us.load(Ordering::Relaxed))
    }

    /// 获取峰值内存使用
    pub fn peak_memory_bytes(&self) -> u64 {
        self.peak_memory_bytes.load(Ordering::Relaxed)
    }

    /// 获取当前内存使用
    pub fn current_memory_bytes(&self) -> u64 {
        self.current_memory_bytes.load(Ordering::Relaxed)
    }

    // =========================================================================
    // 瓶颈分析
    // =========================================================================

    /// 分析性能瓶颈
    pub fn analyze_bottleneck(&self) -> BottleneckAnalysis {
        let stats = self.get_all_scope_stats();

        // 计算各类时间占比
        let total_gpu = self.total_gpu_time_us.load(Ordering::Relaxed) as f64;
        let total_transfer = self.total_transfer_time_us.load(Ordering::Relaxed) as f64;
        let total = total_gpu + total_transfer;

        if total < 1.0 {
            return BottleneckAnalysis {
                primary_bottleneck: BottleneckType::Unknown,
                severity: 0.0,
                affected_scopes: vec![],
                suggestions: vec!["数据不足，请运行更多计算".to_string()],
            };
        }

        // 查找最耗时的范围
        let mut affected_scopes: Vec<_> = stats
            .iter()
            .map(|(name, s)| (name.clone(), s.total_time_us))
            .collect();
        affected_scopes.sort_by(|a, b| b.1.cmp(&a.1));
        let top_scopes: Vec<_> = affected_scopes
            .into_iter()
            .take(3)
            .map(|(name, _)| name)
            .collect();

        // 判断瓶颈类型
        let transfer_ratio = total_transfer / total;
        let (bottleneck, severity) = if transfer_ratio > 0.4 {
            (BottleneckType::DataTransfer, transfer_ratio)
        } else {
            // 检查是否有高方差的范围（可能是同步问题）
            let high_variance_scopes: Vec<_> = stats
                .values()
                .filter(|s| s.coefficient_of_variation() > 0.5)
                .collect();

            if !high_variance_scopes.is_empty() {
                (BottleneckType::Synchronization, 0.6)
            } else {
                (BottleneckType::Compute, 0.5)
            }
        };

        // 生成建议
        let suggestions = self.generate_suggestions(&bottleneck, &stats);

        BottleneckAnalysis {
            primary_bottleneck: bottleneck,
            severity,
            affected_scopes: top_scopes,
            suggestions,
        }
    }

    /// 生成优化建议
    fn generate_suggestions(
        &self,
        bottleneck: &BottleneckType,
        stats: &HashMap<String, ScopeStats>,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        match bottleneck {
            BottleneckType::DataTransfer => {
                suggestions.push("考虑使用双缓冲减少传输等待".to_string());
                suggestions.push("尝试合并多次小传输为单次大传输".to_string());
                suggestions.push("使用 staging buffer 优化主机-设备传输".to_string());
            }
            BottleneckType::Synchronization => {
                suggestions.push("减少不必要的 GPU 同步点".to_string());
                suggestions.push("使用异步计算和传输".to_string());
                suggestions.push("考虑批量提交多个计算命令".to_string());
            }
            BottleneckType::Compute => {
                suggestions.push("检查着色器是否存在分支发散".to_string());
                suggestions.push("优化工作组大小以提高 GPU 占用率".to_string());
                suggestions.push("考虑使用共享内存减少全局内存访问".to_string());
            }
            BottleneckType::MemoryBandwidth => {
                suggestions.push("优化数据布局以提高缓存命中率".to_string());
                suggestions.push("使用纹理采样器利用硬件缓存".to_string());
                suggestions.push("减少每个线程的内存访问量".to_string());
            }
            BottleneckType::ShaderCompilation => {
                suggestions.push("预编译着色器并缓存管线".to_string());
                suggestions.push("减少运行时的动态着色器变体".to_string());
            }
            BottleneckType::Unknown => {
                suggestions.push("收集更多性能数据以进行分析".to_string());
            }
        }

        // 基于具体范围统计的建议
        for (name, scope) in stats {
            if scope.coefficient_of_variation() > 0.8 {
                suggestions.push(format!(
                    "范围 '{}' 时间波动大 (CV={:.2})，检查是否有条件执行路径",
                    name,
                    scope.coefficient_of_variation()
                ));
            }
        }

        suggestions
    }

    // =========================================================================
    // 报告生成
    // =========================================================================

    /// 生成性能报告
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        report.push_str("║               GPU 性能分析报告                               ║\n");
        report.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");

        // 总体统计
        report.push_str("【总体统计】\n");
        report.push_str(&format!(
            "  总 GPU 时间: {:.2} ms\n",
            self.total_gpu_time().as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "  总传输时间: {:.2} ms\n",
            self.total_transfer_time().as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "  峰值内存: {:.2} MB\n",
            self.peak_memory_bytes() as f64 / (1024.0 * 1024.0)
        ));
        report.push_str(&format!(
            "  当前内存: {:.2} MB\n\n",
            self.current_memory_bytes() as f64 / (1024.0 * 1024.0)
        ));

        // 范围统计
        let stats = self.get_all_scope_stats();
        if !stats.is_empty() {
            report.push_str("【范围统计】\n");
            report.push_str("  ─────────────────────────────────────────────────────────\n");
            report.push_str("  范围名称              | 调用次数 | 平均时间 | 总时间 | CV\n");
            report.push_str("  ─────────────────────────────────────────────────────────\n");

            let mut sorted_stats: Vec<_> = stats.iter().collect();
            sorted_stats.sort_by(|a, b| b.1.total_time_us.cmp(&a.1.total_time_us));

            for (name, s) in sorted_stats.iter().take(10) {
                report.push_str(&format!(
                    "  {:<22} | {:>8} | {:>6.1}μs | {:>5.1}ms | {:.2}\n",
                    truncate_string(name, 22),
                    s.call_count,
                    s.avg_time_us,
                    s.total_time_us as f64 / 1000.0,
                    s.coefficient_of_variation()
                ));
            }
            report.push_str("\n");
        }

        // 瓶颈分析
        let analysis = self.analyze_bottleneck();
        report.push_str("【瓶颈分析】\n");
        report.push_str(&format!(
            "  主要瓶颈: {} (严重程度: {:.0}%)\n",
            analysis.primary_bottleneck,
            analysis.severity * 100.0
        ));

        if !analysis.affected_scopes.is_empty() {
            report.push_str(&format!(
                "  影响范围: {}\n",
                analysis.affected_scopes.join(", ")
            ));
        }

        report.push_str("\n【优化建议】\n");
        for (i, suggestion) in analysis.suggestions.iter().take(5).enumerate() {
            report.push_str(&format!("  {}. {}\n", i + 1, suggestion));
        }

        report
    }

    /// 重置所有统计
    pub fn reset(&self) {
        if let Ok(mut stats) = self.scope_stats.write() {
            stats.clear();
        }
        if let Ok(mut history) = self.metric_history.write() {
            history.clear();
        }
        self.total_gpu_time_us.store(0, Ordering::Relaxed);
        self.total_transfer_time_us.store(0, Ordering::Relaxed);
        self.peak_memory_bytes.store(0, Ordering::Relaxed);
        self.current_memory_bytes.store(0, Ordering::Relaxed);
    }
}

/// 截断字符串
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        format!("{:<width$}", s, width = max_len)
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

// ============================================================================
// RAII Guard
// ============================================================================

/// RAII 风格的性能范围守卫
pub struct ProfilerGuard<'a> {
    profiler: &'a GpuProfiler,
    token: ScopeToken,
}

impl<'a> Drop for ProfilerGuard<'a> {
    fn drop(&mut self) {
        self.profiler.end_scope(std::mem::replace(
            &mut self.token,
            ScopeToken {
                name: String::new(),
                start_time: Instant::now(),
                id: 0,
            },
        ));
    }
}

// ============================================================================
// 全局 Profiler
// ============================================================================

/// 全局 profiler 实例
static GLOBAL_PROFILER: std::sync::OnceLock<GpuProfiler> = std::sync::OnceLock::new();

/// 初始化全局 profiler
pub fn init_global_profiler(config: ProfilerConfig) {
    let _ = GLOBAL_PROFILER.set(GpuProfiler::new(config));
}

/// 获取全局 profiler
pub fn global_profiler() -> Option<&'static GpuProfiler> {
    GLOBAL_PROFILER.get()
}

/// 便捷宏：记录范围
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _guard = $crate::gpu::profiler::global_profiler()
            .map(|p| p.scope($name));
    };
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_scope_timing() {
        let profiler = GpuProfiler::new(ProfilerConfig::default());

        let token = profiler.begin_scope("test_scope");
        thread::sleep(Duration::from_millis(10));
        profiler.end_scope(token);

        let stats = profiler.get_scope_stats("test_scope").unwrap();
        assert_eq!(stats.call_count, 1);
        assert!(stats.avg_time_us >= 10_000); // 至少 10ms
    }

    #[test]
    fn test_raii_scope() {
        let profiler = GpuProfiler::new(ProfilerConfig::default());

        {
            let _guard = profiler.scope("raii_test");
            thread::sleep(Duration::from_millis(5));
        }

        let stats = profiler.get_scope_stats("raii_test").unwrap();
        assert_eq!(stats.call_count, 1);
    }

    #[test]
    fn test_metric_recording() {
        let profiler = GpuProfiler::new(ProfilerConfig::default());

        profiler.record_metric(MetricType::MemoryBandwidth, 100.5, Some(10000));
        profiler.record_metric(MetricType::MemoryBandwidth, 105.0, Some(10000));

        let history = profiler.get_metric_history(MetricType::MemoryBandwidth);
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_memory_tracking() {
        let config = ProfilerConfig {
            track_memory: true,
            ..Default::default()
        };
        let profiler = GpuProfiler::new(config);

        profiler.record_allocation(1024);
        profiler.record_allocation(2048);
        assert_eq!(profiler.current_memory_bytes(), 3072);
        assert_eq!(profiler.peak_memory_bytes(), 3072);

        profiler.record_deallocation(1024);
        assert_eq!(profiler.current_memory_bytes(), 2048);
        assert_eq!(profiler.peak_memory_bytes(), 3072); // 峰值不变
    }

    #[test]
    fn test_bottleneck_analysis() {
        let profiler = GpuProfiler::new(ProfilerConfig::default());

        // 模拟一些计算
        for _ in 0..10 {
            let token = profiler.begin_scope("compute");
            thread::sleep(Duration::from_micros(100));
            profiler.end_scope(token);
        }

        let analysis = profiler.analyze_bottleneck();
        assert!(!analysis.suggestions.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let profiler = GpuProfiler::new(ProfilerConfig::default());

        let token = profiler.begin_scope("test");
        profiler.end_scope(token);

        let report = profiler.generate_report();
        assert!(report.contains("GPU 性能分析报告"));
        assert!(report.contains("总体统计"));
    }

    #[test]
    fn test_disabled_profiler() {
        let config = ProfilerConfig::disabled();
        let profiler = GpuProfiler::new(config);

        let token = profiler.begin_scope("disabled_test");
        profiler.end_scope(token);

        // 禁用时不应收集数据
        assert!(profiler.get_scope_stats("disabled_test").is_none());
    }
}
