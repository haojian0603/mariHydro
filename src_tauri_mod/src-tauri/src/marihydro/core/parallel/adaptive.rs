//! 自适应并行调度器
//!
//! 根据历史性能数据动态选择最优并行策略。

use super::config::ParallelConfig;
use super::strategy::ParallelStrategy;
use parking_lot::Mutex;
use std::collections::VecDeque;

/// 性能统计记录
#[derive(Debug, Clone)]
struct PerfRecord {
    /// 工作项数量
    work_items: usize,
    /// 使用的策略
    strategy: ParallelStrategy,
    /// 执行时间（纳秒）
    duration_ns: u64,
    /// 加速比
    speedup: f64,
}

/// 自适应并行调度器
///
/// 基于历史性能记录动态调整并行策略。
pub struct AdaptiveScheduler {
    /// 性能历史记录
    history: Mutex<VecDeque<PerfRecord>>,
    /// 窗口大小
    window_size: usize,
    /// 串行基准时间（用于计算加速比）
    serial_baseline: Mutex<Option<u64>>,
}

impl AdaptiveScheduler {
    /// 创建新的自适应调度器
    ///
    /// # Arguments
    /// - `window_size`: 历史记录窗口大小
    pub fn new(window_size: usize) -> Self {
        Self {
            history: Mutex::new(VecDeque::with_capacity(window_size)),
            window_size,
            serial_baseline: Mutex::new(None),
        }
    }

    /// 使用默认窗口大小创建
    pub fn default_scheduler() -> Self {
        let config = ParallelConfig::global();
        Self::new(config.monitoring_window)
    }

    /// 根据历史性能推荐策略
    pub fn recommend(&self, work_items: usize) -> ParallelStrategy {
        let history = self.history.lock();

        if history.len() < 10 {
            // 数据不足，使用默认策略
            return if ParallelConfig::global().should_parallelize(work_items) {
                ParallelStrategy::Dynamic
            } else {
                ParallelStrategy::Sequential
            };
        }

        // 计算各策略的平均加速比
        let mut strategy_stats: std::collections::HashMap<
            ParallelStrategyKey,
            (f64, usize), // (sum_speedup, count)
        > = std::collections::HashMap::new();

        for record in history.iter().rev().take(20) {
            let key = ParallelStrategyKey::from(record.strategy);
            let entry = strategy_stats.entry(key).or_insert((0.0, 0));
            entry.0 += record.speedup;
            entry.1 += 1;
        }

        // 选择平均加速比最高的策略
        strategy_stats
            .iter()
            .max_by(|a, b| {
                let avg_a = a.1 .0 / a.1 .1 as f64;
                let avg_b = b.1 .0 / b.1 .1 as f64;
                avg_a.partial_cmp(&avg_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(&key, _)| key.to_strategy())
            .unwrap_or(ParallelStrategy::Dynamic)
    }

    /// 记录执行结果
    ///
    /// # Arguments
    /// - `work_items`: 工作项数量
    /// - `strategy`: 使用的策略
    /// - `duration_ns`: 执行时间（纳秒）
    /// - `serial_duration_ns`: 串行执行时间（用于计算加速比，可选）
    pub fn record(
        &self,
        work_items: usize,
        strategy: ParallelStrategy,
        duration_ns: u64,
        serial_duration_ns: Option<u64>,
    ) {
        // 计算加速比
        let speedup = if let Some(serial) = serial_duration_ns {
            if duration_ns > 0 {
                serial as f64 / duration_ns as f64
            } else {
                1.0
            }
        } else if let Some(baseline) = *self.serial_baseline.lock() {
            if duration_ns > 0 {
                baseline as f64 / duration_ns as f64
            } else {
                1.0
            }
        } else {
            1.0
        };

        // 更新串行基准
        if strategy == ParallelStrategy::Sequential {
            *self.serial_baseline.lock() = Some(duration_ns);
        }

        let mut history = self.history.lock();
        if history.len() >= self.window_size {
            history.pop_front();
        }
        history.push_back(PerfRecord {
            work_items,
            strategy,
            duration_ns,
            speedup,
        });
    }

    /// 获取统计摘要
    pub fn summary(&self) -> SchedulerSummary {
        let history = self.history.lock();

        if history.is_empty() {
            return SchedulerSummary::default();
        }

        let total_records = history.len();
        let avg_speedup = history.iter().map(|r| r.speedup).sum::<f64>() / total_records as f64;

        let mut strategy_counts: std::collections::HashMap<ParallelStrategyKey, usize> =
            std::collections::HashMap::new();
        for record in history.iter() {
            *strategy_counts
                .entry(ParallelStrategyKey::from(record.strategy))
                .or_insert(0) += 1;
        }

        let most_used = strategy_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&key, _)| key.to_strategy())
            .unwrap_or(ParallelStrategy::Sequential);

        SchedulerSummary {
            total_records,
            avg_speedup,
            most_used_strategy: most_used,
        }
    }

    /// 清除历史记录
    pub fn reset(&self) {
        self.history.lock().clear();
        *self.serial_baseline.lock() = None;
    }
}

/// 调度器统计摘要
#[derive(Debug, Clone, Default)]
pub struct SchedulerSummary {
    /// 总记录数
    pub total_records: usize,
    /// 平均加速比
    pub avg_speedup: f64,
    /// 最常用策略
    pub most_used_strategy: ParallelStrategy,
}

/// 策略键（用于HashMap）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ParallelStrategyKey {
    Sequential,
    StaticChunks,
    Dynamic,
    WorkStealing,
    Colored,
}

impl From<ParallelStrategy> for ParallelStrategyKey {
    fn from(s: ParallelStrategy) -> Self {
        match s {
            ParallelStrategy::Sequential => ParallelStrategyKey::Sequential,
            ParallelStrategy::StaticChunks { .. } => ParallelStrategyKey::StaticChunks,
            ParallelStrategy::Dynamic => ParallelStrategyKey::Dynamic,
            ParallelStrategy::WorkStealing => ParallelStrategyKey::WorkStealing,
            ParallelStrategy::Colored => ParallelStrategyKey::Colored,
        }
    }
}

impl ParallelStrategyKey {
    fn to_strategy(self) -> ParallelStrategy {
        match self {
            ParallelStrategyKey::Sequential => ParallelStrategy::Sequential,
            ParallelStrategyKey::StaticChunks => ParallelStrategy::StaticChunks { chunk_size: 256 },
            ParallelStrategyKey::Dynamic => ParallelStrategy::Dynamic,
            ParallelStrategyKey::WorkStealing => ParallelStrategy::WorkStealing,
            ParallelStrategyKey::Colored => ParallelStrategy::Colored,
        }
    }
}

impl Default for ParallelStrategy {
    fn default() -> Self {
        ParallelStrategy::Sequential
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_scheduler_default_recommendation() {
        let scheduler = AdaptiveScheduler::new(50);

        // 无历史数据时，小任务应该串行
        let strategy = scheduler.recommend(100);
        assert_eq!(strategy, ParallelStrategy::Sequential);
    }

    #[test]
    fn test_adaptive_scheduler_record() {
        let scheduler = AdaptiveScheduler::new(10);

        // 记录一些执行
        scheduler.record(1000, ParallelStrategy::Dynamic, 1000000, None);
        scheduler.record(1000, ParallelStrategy::Sequential, 4000000, None);
        scheduler.record(1000, ParallelStrategy::Dynamic, 1100000, Some(4000000));

        let summary = scheduler.summary();
        assert_eq!(summary.total_records, 3);
    }

    #[test]
    fn test_window_size_limit() {
        let scheduler = AdaptiveScheduler::new(5);

        for i in 0..10 {
            scheduler.record(1000, ParallelStrategy::Dynamic, 1000 * i, None);
        }

        let summary = scheduler.summary();
        assert_eq!(summary.total_records, 5);
    }
}
