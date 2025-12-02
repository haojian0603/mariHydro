//! 性能监控指标
//!
//! 提供并行执行的性能监控和统计功能。

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// 性能指标收集器
///
/// 用于跟踪并行操作的性能数据。
pub struct PerfMetrics {
    /// 总执行次数
    total_calls: AtomicU64,
    /// 并行执行次数
    parallel_calls: AtomicU64,
    /// 串行执行次数
    serial_calls: AtomicU64,
    /// 总处理的工作项数
    total_work_items: AtomicU64,
    /// 总执行时间（纳秒）
    total_duration_ns: AtomicU64,
}

impl PerfMetrics {
    /// 创建新的性能指标收集器
    pub const fn new() -> Self {
        Self {
            total_calls: AtomicU64::new(0),
            parallel_calls: AtomicU64::new(0),
            serial_calls: AtomicU64::new(0),
            total_work_items: AtomicU64::new(0),
            total_duration_ns: AtomicU64::new(0),
        }
    }

    /// 记录一次执行
    pub fn record(&self, work_items: usize, is_parallel: bool, duration_ns: u64) {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        self.total_work_items
            .fetch_add(work_items as u64, Ordering::Relaxed);
        self.total_duration_ns
            .fetch_add(duration_ns, Ordering::Relaxed);

        if is_parallel {
            self.parallel_calls.fetch_add(1, Ordering::Relaxed);
        } else {
            self.serial_calls.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// 获取统计摘要
    pub fn summary(&self) -> MetricsSummary {
        let total = self.total_calls.load(Ordering::Relaxed);
        let parallel = self.parallel_calls.load(Ordering::Relaxed);
        let serial = self.serial_calls.load(Ordering::Relaxed);
        let work_items = self.total_work_items.load(Ordering::Relaxed);
        let duration = self.total_duration_ns.load(Ordering::Relaxed);

        MetricsSummary {
            total_calls: total,
            parallel_calls: parallel,
            serial_calls: serial,
            total_work_items: work_items,
            total_duration_ns: duration,
            parallel_ratio: if total > 0 {
                parallel as f64 / total as f64
            } else {
                0.0
            },
            avg_duration_ns: if total > 0 {
                duration / total
            } else {
                0
            },
        }
    }

    /// 重置所有指标
    pub fn reset(&self) {
        self.total_calls.store(0, Ordering::Relaxed);
        self.parallel_calls.store(0, Ordering::Relaxed);
        self.serial_calls.store(0, Ordering::Relaxed);
        self.total_work_items.store(0, Ordering::Relaxed);
        self.total_duration_ns.store(0, Ordering::Relaxed);
    }
}

impl Default for PerfMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// 性能统计摘要
#[derive(Debug, Clone, Default)]
pub struct MetricsSummary {
    /// 总调用次数
    pub total_calls: u64,
    /// 并行调用次数
    pub parallel_calls: u64,
    /// 串行调用次数
    pub serial_calls: u64,
    /// 总处理的工作项数
    pub total_work_items: u64,
    /// 总执行时间（纳秒）
    pub total_duration_ns: u64,
    /// 并行比例
    pub parallel_ratio: f64,
    /// 平均执行时间（纳秒）
    pub avg_duration_ns: u64,
}

impl MetricsSummary {
    /// 格式化为人类可读的字符串
    pub fn format(&self) -> String {
        format!(
            "Calls: {} (parallel: {:.1}%), Items: {}, Avg duration: {:.2}ms",
            self.total_calls,
            self.parallel_ratio * 100.0,
            self.total_work_items,
            self.avg_duration_ns as f64 / 1_000_000.0
        )
    }
}

/// 计时器辅助结构
///
/// 用于测量代码块的执行时间。
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// 开始计时
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// 获取经过的时间（纳秒）
    pub fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    /// 获取经过的时间（微秒）
    pub fn elapsed_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    /// 获取经过的时间（毫秒）
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// 停止计时并返回纳秒数
    pub fn stop(self) -> u64 {
        self.elapsed_ns()
    }
}

/// 作用域计时器
///
/// 在离开作用域时自动记录执行时间。
pub struct ScopedTimer<'a> {
    timer: Timer,
    metrics: &'a PerfMetrics,
    work_items: usize,
    is_parallel: bool,
}

impl<'a> ScopedTimer<'a> {
    /// 创建作用域计时器
    pub fn new(metrics: &'a PerfMetrics, work_items: usize, is_parallel: bool) -> Self {
        Self {
            timer: Timer::start(),
            metrics,
            work_items,
            is_parallel,
        }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        let duration = self.timer.elapsed_ns();
        self.metrics
            .record(self.work_items, self.is_parallel, duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_metrics() {
        let metrics = PerfMetrics::new();

        metrics.record(1000, true, 1000000);
        metrics.record(500, false, 500000);

        let summary = metrics.summary();
        assert_eq!(summary.total_calls, 2);
        assert_eq!(summary.parallel_calls, 1);
        assert_eq!(summary.serial_calls, 1);
        assert_eq!(summary.total_work_items, 1500);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 9.0 && elapsed < 50.0);
    }

    #[test]
    fn test_scoped_timer() {
        let metrics = PerfMetrics::new();

        {
            let _timer = ScopedTimer::new(&metrics, 100, true);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        let summary = metrics.summary();
        assert_eq!(summary.total_calls, 1);
        assert!(summary.total_duration_ns > 0);
    }
}
