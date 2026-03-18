// crates/mh_runtime/src/metrics.rs

//! 运行时性能指标收集系统
//!
//! 提供线程安全的性能监控功能，用于物理引擎的性能分析和调优。
//! 与 Foundation 层的 `Counter` 不同，本模块包含时间度量和物理相关统计。

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// 原子计数器（增强版）
///
/// 相比 Foundation 层的 `Counter`，增加了序列化和更多统计功能。
#[derive(Debug)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// 创建零值计数器
    pub const fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// 增加计数
    #[inline]
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// 增加指定值
    #[inline]
    pub fn add(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// 获取当前值
    #[inline]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// 重置为零
    #[inline]
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 计时器
// =============================================================================

/// 高精度计时器（累积时间）
///
/// 用于测量代码块执行时间，支持嵌套调用。
#[derive(Debug)]
pub struct Timer {
    total_ns: AtomicU64,
    count: AtomicU64,
}

impl Timer {
    /// 创建新计时器
    pub const fn new() -> Self {
        Self {
            total_ns: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// 开始计时，返回守卫
    ///
    /// # 示例
    ///
    /// ```rust
    /// use mh_runtime::metrics::Timer;
    ///
    /// let timer = Timer::new();
    /// {
    ///     let _guard = timer.start();
    ///     // ... 被测量的代码 ...
    /// }
    /// println!("总时间: {:?} ms", timer.total_ms());
    /// ```
    pub fn start(&self) -> TimerGuard<'_> {
        TimerGuard {
            timer: self,
            start: Instant::now(),
        }
    }

    /// 记录一次计时（手动）
    pub fn record(&self, elapsed: Duration) {
        self.total_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// 获取总时间（纳秒）
    pub fn total_ns(&self) -> u64 {
        self.total_ns.load(Ordering::Relaxed)
    }

    /// 获取总时间（微秒）
    pub fn total_us(&self) -> f64 {
        self.total_ns.load(Ordering::Relaxed) as f64 / 1_000.0
    }

    /// 获取总时间（毫秒）
    pub fn total_ms(&self) -> f64 {
        self.total_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// 获取总时间（秒）
    pub fn total_sec(&self) -> f64 {
        self.total_ns.load(Ordering::Relaxed) as f64 / 1_000_000_000.0
    }

    /// 获取调用次数
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// 获取平均时间（纳秒）
    pub fn avg_ns(&self) -> f64 {
        let c = self.count();
        if c == 0 {
            0.0
        } else {
            self.total_ns() as f64 / c as f64
        }
    }

    /// 获取平均时间（毫秒）
    pub fn avg_ms(&self) -> f64 {
        self.avg_ns() / 1_000_000.0
    }

    /// 重置计时器
    pub fn reset(&self) {
        self.total_ns.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// 计时守卫
///
/// 当守卫被 drop 时，自动记录时间。
pub struct TimerGuard<'a> {
    timer: &'a Timer,
    start: Instant,
}

impl Drop for TimerGuard<'_> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        self.timer.record(elapsed);
    }
}

// =============================================================================
// 性能指标快照
// =============================================================================

/// 性能指标快照
///
/// 用于导出和序列化当前性能状态。
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MetricsSnapshot {
    /// 时间步数
    pub time_steps: u64,
    /// 通量计算次数
    pub flux_evals: u64,
    /// 源项计算次数
    pub source_evals: u64,
    /// 边界条件计算次数
    pub boundary_evals: u64,
    /// 总计算时间（秒）
    pub total_time_sec: f64,
    /// 通量计算时间（秒）
    pub flux_time_sec: f64,
    /// 源项计算时间（秒）
    pub source_time_sec: f64,
    /// 边界条件计算时间（秒）
    pub boundary_time_sec: f64,
    /// 每步平均时间（毫秒）
    pub avg_step_time_ms: f64,
}

impl MetricsSnapshot {
    /// 创建空快照
    pub fn new() -> Self {
        Self::default()
    }

    /// 格式化为人类可读的摘要
    pub fn summary(&self) -> String {
        format!(
            "Steps: {}, Flux: {}, Source: {}, Boundary: {}, Total: {:.3}s, Avg: {:.2}ms/step",
            self.time_steps,
            self.flux_evals,
            self.source_evals,
            self.boundary_evals,
            self.total_time_sec,
            self.avg_step_time_ms
        )
    }

    /// 计算每步平均时间（毫秒）
    pub fn avg_step_time_ms(&self) -> f64 {
        if self.time_steps == 0 {
            0.0
        } else {
            self.total_time_sec * 1000.0 / self.time_steps as f64
        }
    }

    /// 计算通量计算占比
    pub fn flux_time_ratio(&self) -> f64 {
        if self.total_time_sec > 0.0 {
            self.flux_time_sec / self.total_time_sec
        } else {
            0.0
        }
    }

    /// 计算源项计算占比
    pub fn source_time_ratio(&self) -> f64 {
        if self.total_time_sec > 0.0 {
            self.source_time_sec / self.total_time_sec
        } else {
            0.0
        }
    }
}

impl Default for MetricsSnapshot {
    fn default() -> Self {
        Self {
            time_steps: 0,
            flux_evals: 0,
            source_evals: 0,
            boundary_evals: 0,
            total_time_sec: 0.0,
            flux_time_sec: 0.0,
            source_time_sec: 0.0,
            boundary_time_sec: 0.0,
            avg_step_time_ms: 0.0,
        }
    }
}

// =============================================================================
// 指标收集器
// =============================================================================

/// 全局/局部指标收集器
///
/// 用于聚合多个计时器和计数器，提供统一的性能监控接口。
#[derive(Debug)]
pub struct MetricsCollector {
    /// 时间步计数
    pub time_steps: Counter,
    /// 通量计算计数
    pub flux_evals: Counter,
    /// 源项计算计数
    pub source_evals: Counter,
    /// 边界条件计算计数
    pub boundary_evals: Counter,
    /// 总计时器
    pub total_timer: Timer,
    /// 通量计算计时器
    pub flux_timer: Timer,
    /// 源项计算计时器
    pub source_timer: Timer,
    /// 边界条件计时器
    pub boundary_timer: Timer,
}

impl MetricsCollector {
    /// 创建新的收集器
    pub fn new() -> Self {
        Self::default()
    }

    /// 生成快照
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            time_steps: self.time_steps.get(),
            flux_evals: self.flux_evals.get(),
            source_evals: self.source_evals.get(),
            boundary_evals: self.boundary_evals.get(),
            total_time_sec: self.total_timer.total_sec(),
            flux_time_sec: self.flux_timer.total_sec(),
            source_time_sec: self.source_timer.total_sec(),
            boundary_time_sec: self.boundary_timer.total_sec(),
            avg_step_time_ms: self.total_timer.avg_ms(),
        }
    }

    /// 重置所有指标
    pub fn reset(&self) {
        self.time_steps.reset();
        self.flux_evals.reset();
        self.source_evals.reset();
        self.boundary_evals.reset();
        self.total_timer.reset();
        self.flux_timer.reset();
        self.source_timer.reset();
        self.boundary_timer.reset();
    }

    /// 记录一个完整时间步
    pub fn record_step(&self, step_time: Duration) {
        self.time_steps.inc();
        self.total_timer.record(step_time);
    }

    /// 开始通量计算计时
    pub fn start_flux(&self) -> TimerGuard<'_> {
        self.flux_evals.inc();
        self.flux_timer.start()
    }

    /// 开始源项计算计时
    pub fn start_source(&self) -> TimerGuard<'_> {
        self.source_evals.inc();
        self.source_timer.start()
    }

    /// 开始边界条件计算计时
    pub fn start_boundary(&self) -> TimerGuard<'_> {
        self.boundary_evals.inc();
        self.boundary_timer.start()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self {
            time_steps: Counter::new(),
            flux_evals: Counter::new(),
            source_evals: Counter::new(),
            boundary_evals: Counter::new(),
            total_timer: Timer::new(),
            flux_timer: Timer::new(),
            source_timer: Timer::new(),
            boundary_timer: Timer::new(),
        }
    }
}

// =============================================================================
// 测试
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_counter() {
        let counter = Counter::new();
        assert_eq!(counter.get(), 0);
        counter.inc();
        counter.inc();
        counter.add(3);
        assert_eq!(counter.get(), 5);
        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::new();
        {
            let _guard = timer.start();
            sleep(std::time::Duration::from_millis(10));
        }
        assert!(timer.total_ms() > 0.0);
        assert_eq!(timer.count(), 1);
        assert!(timer.avg_ms() > 0.0);
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        collector.time_steps.add(100);
        collector.flux_evals.add(1000);
        
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.time_steps, 100);
        assert_eq!(snapshot.flux_evals, 1000);
        assert_eq!(snapshot.source_evals, 0);
    }

    #[test]
    fn test_metrics_snapshot_summary() {
        let snapshot = MetricsSnapshot {
            time_steps: 100,
            flux_evals: 1000,
            source_evals: 500,
            boundary_evals: 200,
            total_time_sec: 10.5,
            flux_time_sec: 5.0,
            source_time_sec: 3.0,
            boundary_time_sec: 2.0,
            avg_step_time_ms: 105.0,
        };
        
        let summary = snapshot.summary();
        assert!(summary.contains("Steps: 100"));
        assert!(summary.contains("Flux: 1000"));
        assert!(summary.contains("Total: 10.500s"));
    }

    #[test]
    fn test_time_ratios() {
        let snapshot = MetricsSnapshot {
            time_steps: 10,
            flux_evals: 100,
            source_evals: 50,
            boundary_evals: 20,
            total_time_sec: 10.0,
            flux_time_sec: 5.0,
            source_time_sec: 3.0,
            boundary_time_sec: 2.0,
            avg_step_time_ms: 1000.0,
        };
        
        assert_eq!(snapshot.flux_time_ratio(), 0.5);
        assert_eq!(snapshot.source_time_ratio(), 0.3);
    }
}