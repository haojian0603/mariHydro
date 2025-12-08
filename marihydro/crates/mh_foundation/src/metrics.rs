//! 性能计数器和指标系统
//!
//! 提供轻量级的性能监控功能，用于识别热点和优化。
//!
//! # 设计说明
//!
//! - 零开销：release 模式下可完全关闭
//! - 线程安全：使用原子计数
//! - 层次化：支持模块/功能/细节多级指标

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use serde::{Deserialize, Serialize};

/// 计数器（原子无锁）
#[derive(Debug, Default)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// 创建零值计数器
    pub const fn new() -> Self {
        Self { value: AtomicU64::new(0) }
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

/// 计时器（累积时间）
#[derive(Debug, Default)]
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
    pub fn start(&self) -> TimerGuard<'_> {
        TimerGuard {
            timer: self,
            start: Instant::now(),
        }
    }

    /// 记录一次计时
    pub fn record(&self, elapsed_ns: u64) {
        self.total_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// 获取总时间（纳秒）
    pub fn total_ns(&self) -> u64 {
        self.total_ns.load(Ordering::Relaxed)
    }

    /// 获取调用次数
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// 获取平均时间（纳秒）
    pub fn avg_ns(&self) -> f64 {
        let c = self.count();
        if c == 0 { 0.0 } else { self.total_ns() as f64 / c as f64 }
    }

    /// 重置
    pub fn reset(&self) {
        self.total_ns.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }
}

/// 计时守卫，drop 时自动记录
pub struct TimerGuard<'a> {
    timer: &'a Timer,
    start: Instant,
}

impl Drop for TimerGuard<'_> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_nanos() as u64;
        self.timer.record(elapsed);
    }
}

/// 性能指标快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// 时间步数
    pub time_steps: u64,
    /// 通量计算次数
    pub flux_evals: u64,
    /// 源项计算次数
    pub source_evals: u64,
    /// 总计算时间（毫秒）
    pub total_time_ms: f64,
    /// 通量计算时间（毫秒）
    pub flux_time_ms: f64,
    /// 源项计算时间（毫秒）
    pub source_time_ms: f64,
}

impl Default for MetricsSnapshot {
    fn default() -> Self {
        Self {
            time_steps: 0,
            flux_evals: 0,
            source_evals: 0,
            total_time_ms: 0.0,
            flux_time_ms: 0.0,
            source_time_ms: 0.0,
        }
    }
}

impl MetricsSnapshot {
    /// 格式化为人类可读的字符串
    pub fn summary(&self) -> String {
        format!(
            "Steps: {}, Flux: {}, Source: {}, Time: {:.2}ms",
            self.time_steps, self.flux_evals, self.source_evals, self.total_time_ms
        )
    }
}

/// 全局指标收集器
#[derive(Debug, Default)]
pub struct MetricsCollector {
    /// 时间步计数
    pub time_steps: Counter,
    /// 通量计算计数
    pub flux_evals: Counter,
    /// 源项计算计数
    pub source_evals: Counter,
    /// 总计时器
    pub total_timer: Timer,
    /// 通量计时器
    pub flux_timer: Timer,
    /// 源项计时器
    pub source_timer: Timer,
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
            total_time_ms: self.total_timer.total_ns() as f64 / 1_000_000.0,
            flux_time_ms: self.flux_timer.total_ns() as f64 / 1_000_000.0,
            source_time_ms: self.source_timer.total_ns() as f64 / 1_000_000.0,
        }
    }

    /// 重置所有指标
    pub fn reset(&self) {
        self.time_steps.reset();
        self.flux_evals.reset();
        self.source_evals.reset();
        self.total_timer.reset();
        self.flux_timer.reset();
        self.source_timer.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

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
            sleep(Duration::from_millis(1));
        }
        assert!(timer.total_ns() > 0);
        assert_eq!(timer.count(), 1);
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        collector.time_steps.add(100);
        collector.flux_evals.add(1000);
        
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.time_steps, 100);
        assert_eq!(snapshot.flux_evals, 1000);
    }

    #[test]
    fn test_metrics_summary() {
        let snapshot = MetricsSnapshot {
            time_steps: 100,
            flux_evals: 1000,
            source_evals: 500,
            total_time_ms: 150.5,
            flux_time_ms: 100.0,
            source_time_ms: 50.0,
        };
        let summary = snapshot.summary();
        assert!(summary.contains("100"));
        assert!(summary.contains("1000"));
    }
}
