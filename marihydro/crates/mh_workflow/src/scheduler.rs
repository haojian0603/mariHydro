// crates/mh_workflow/src/scheduler.rs

//! 混合调度器模块
//!
//! 提供CPU/GPU计算设备的自动选择和调度。

// GPU 相关引用已移除

/// 设备类型（现在仅保留 CPU）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Default)]
pub enum DeviceType {
    /// CPU
    #[default]
    Cpu,
    /// 其他 / 未知（保留以便将来扩展）
    Other,
}


impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// 混合计算策略（GPU相关策略移除）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum HybridStrategy {
    /// 仅使用CPU
    CpuOnly,
    /// 自动选择（当前等同于 CPU）
    #[default]
    Auto,
}


impl std::fmt::Display for HybridStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CpuOnly => write!(f, "CPU Only"),
            Self::Auto => write!(f, "Auto"),
        }
    }
}

/// 混合调度配置
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// 计算策略
    pub strategy: HybridStrategy,
    /// CPU线程数 (0=自动)
    pub cpu_threads: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            strategy: HybridStrategy::Auto,
            cpu_threads: 0,
        }
    }
}

impl HybridConfig {
    /// 创建 CPU-only 配置
    pub fn cpu_only() -> Self {
        Self {
            strategy: HybridStrategy::CpuOnly,
            cpu_threads: 0,
            ..Default::default()
        }
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

    /// GPU选择 (legacy; remains for compatibility but unused)
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

// GPU support was removed; these types have been deleted.

/// 混合计算调度器
pub struct HybridScheduler {
    /// 配置
    config: HybridConfig,
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
    _timestamp: std::time::Instant,
}

impl HybridScheduler {
    /// 创建调度器
    pub fn new(config: HybridConfig) -> Self {
        Self {
            config,
            current_device: DeviceType::Cpu,
            performance_history: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// 选择计算设备
    pub fn select_device(&self, _num_cells: usize) -> DeviceSelection {
        // GPU support removed; always choose CPU for now.
        DeviceSelection::cpu("GPU support removed; CPU only")
    }

    /// 自动选择设备
    // Automatic selection removed (CPU only)
    #[allow(dead_code)]
    fn auto_select(&self, _num_cells: usize) -> DeviceSelection {
        DeviceSelection::cpu("GPU support removed; auto -> CPU")
    }

    // 记录性能数据
    pub fn record_performance(&self, device: DeviceType, num_cells: usize, elapsed_secs: f64) {
        let record = PerformanceRecord {
            device,
            num_cells,
            elapsed_secs,
            _timestamp: std::time::Instant::now(),
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
        false
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
        PerformanceStats {
            cpu_invocations: cpu_records.len(),
            avg_cpu_cells_per_sec: Self::avg_throughput(&cpu_records),
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
    /// 平均CPU吞吐量 (单元/秒)
    pub avg_cpu_cells_per_sec: f64,
}

impl PerformanceStats {
    /// 计算GPU相对CPU的实际加速比
    pub fn actual_speedup(&self) -> Option<f64> {
        // GPU removed; return None
        None
    }
}

// ============================================================
// 调度器诊断
// ============================================================

/// 调度器诊断报告
///
/// 提供详细的调度器运行状态和性能信息，用于调试和优化。
#[derive(Debug, Clone)]
pub struct SchedulerDiagnostics {
    /// 配置信息
    pub config_summary: String,
    /// 性能统计
    pub performance: PerformanceStats,
    /// 设备选择历史统计
    pub selection_stats: SelectionStats,
    /// 建议
    pub recommendations: Vec<String>,
}

/// GPU 诊断信息
// GPU diagnostics removed since GPU support was removed.

/// 设备选择统计
#[derive(Debug, Clone, Default)]
pub struct SelectionStats {
    /// 总选择次数
    pub total_selections: u64,
    /// CPU 选择次数
    pub cpu_selections: u64,
    /// 回退次数
    pub fallback_count: u64,
    /// 因阈值不满足选择CPU次数
    pub threshold_cpu_selections: u64,
}

impl SelectionStats {
    /// GPU 选择率
    pub fn gpu_selection_rate(&self) -> f64 {
        0.0
    }

    /// 回退率
    pub fn fallback_rate(&self) -> f64 {
        if self.total_selections == 0 {
            0.0
        } else {
            self.fallback_count as f64 / self.total_selections as f64
        }
    }
}

impl HybridScheduler {
    /// 生成调度器诊断报告
    ///
    /// 返回详细的诊断信息，包括配置、GPU状态、性能统计和建议。
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let diagnostics = scheduler.diagnostics();
    /// println!("GPU状态: {}", diagnostics.gpu_status.health_status);
    /// for rec in &diagnostics.recommendations {
    ///     println!("建议: {}", rec);
    /// }
    /// ```
    pub fn diagnostics(&self) -> SchedulerDiagnostics {
        let performance = self.performance_stats();
        let selection_stats = self.build_selection_stats();
        let recommendations = self.build_recommendations(&performance, &selection_stats);

        SchedulerDiagnostics {
            config_summary: self.config_summary(),
            performance,
            selection_stats,
            recommendations,
        }
    }

    /// 生成配置摘要
    fn config_summary(&self) -> String {
        format!(
            "策略: {}, 线程: {}",
            self.config.strategy,
            if self.config.cpu_threads == 0 { "自动".to_string() } else { self.config.cpu_threads.to_string() }
        )
    }

    // GPU diagnostics removed

    /// 构建设备选择统计
    fn build_selection_stats(&self) -> SelectionStats {
        let history = self.performance_history.read();

        let mut stats = SelectionStats::default();
        stats.total_selections = history.len() as u64;

        for record in history.iter() {
            match record.device {
                DeviceType::Cpu => stats.cpu_selections += 1,
                _ => { /* No GPU support: nothing to count here */ }
            }
        }

        stats
    }

    /// 生成优化建议
    fn build_recommendations(&self, performance: &PerformanceStats, stats: &SelectionStats) -> Vec<String> {
        let mut recommendations = Vec::new();

        // GPU support removed; no GPU-specific recommendations.

        // 性能相关建议
        // CPU-only recommendations
        if performance.avg_cpu_cells_per_sec < 10.0 && stats.total_selections > 10 {
            recommendations.push("CPU 吞吐量低，建议检查数据分割或线程配置".to_string());
        }

        // 回退相关建议
        if stats.fallback_rate() > 0.1 {
            recommendations.push(format!(
                "回退率较高 ({:.1}%)，建议检查 GPU 内存配置或调整阈值",
                stats.fallback_rate() * 100.0
            ));
        }

        // 阈值相关建议: 如果大多数选择仍然使用 CPU, 建议优化线程或任务划分
        if stats.total_selections > 100 {
            let cpu_rate = stats.cpu_selections as f64 / stats.total_selections as f64;
            if cpu_rate > 0.8 {
                recommendations.push("CPU 选择率高，建议优化线程或任务划分".to_string());
            }
        }

        if recommendations.is_empty() {
            recommendations.push("调度器运行正常，暂无优化建议".to_string());
        }

        recommendations
    }

    /// 打印诊断报告到日志
    pub fn log_diagnostics(&self) {
        let diag = self.diagnostics();

        println!("========== 调度器诊断报告 ==========");
        println!("配置: {}", diag.config_summary);
        println!();
        // GPU support removed; skip GPU diagnostics
        println!("性能统计:");
        println!("  CPU 调用: {}", diag.performance.cpu_invocations);
        println!("  CPU 吞吐: {:.0} cells/s", diag.performance.avg_cpu_cells_per_sec);
        if let Some(speedup) = diag.performance.actual_speedup() {
            println!("  实测加速比: {:.2}x", speedup);
        }
        println!();
        println!("选择统计:");
        // GPU selection rate not available in CPU-only mode
        println!("  回退率: {:.1}%", diag.selection_stats.fallback_rate() * 100.0);
        println!();
        println!("建议:");
        for (i, rec) in diag.recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, rec);
        }
        println!("=====================================");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_config() {
        let config = HybridConfig::default();
        assert_eq!(config.strategy, HybridStrategy::Auto);

        let cpu_config = HybridConfig::cpu_only();
        assert_eq!(cpu_config.strategy, HybridStrategy::CpuOnly);

        // GPU-only config removed
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
        assert!(selection.reason.contains("GPU support removed"));
    }

    // GPU-related tests removed — scheduler is now CPU-only.

    #[test]
    fn test_performance_recording() {
        let scheduler = HybridScheduler::new(HybridConfig::default());

        scheduler.record_performance(DeviceType::Cpu, 10000, 1.0);

        let stats = scheduler.performance_stats();
        assert_eq!(stats.cpu_invocations, 1);
        // GPU stats removed; actual_speedup returns None
        assert!(stats.actual_speedup().is_none());
    }
}
