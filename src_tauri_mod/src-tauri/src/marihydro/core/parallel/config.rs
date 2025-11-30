//! 全局并行配置
//!
//! 提供并行计算的全局配置管理。

use std::sync::OnceLock;

/// 全局并行配置（单例）
static PARALLEL_CONFIG: OnceLock<ParallelConfig> = OnceLock::new();

/// 并行配置
///
/// 控制系统中所有并行操作的行为。
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// 是否启用并行
    pub enabled: bool,
    /// 最小并行化单元数（小于此数串行执行）
    pub min_work_items: usize,
    /// 每线程最小工作量（用于动态分块）
    pub min_items_per_thread: usize,
    /// 最大线程数
    pub max_threads: usize,
    /// 启用自适应调度
    pub adaptive_scheduling: bool,
    /// 性能监控窗口大小
    pub monitoring_window: usize,
    /// 是否启用性能日志
    pub enable_perf_logging: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_work_items: 1000,
            min_items_per_thread: 256,
            max_threads: rayon::current_num_threads(),
            adaptive_scheduling: true,
            monitoring_window: 50,
            enable_perf_logging: false,
        }
    }
}

impl ParallelConfig {
    /// 获取全局配置
    ///
    /// 如果尚未初始化，将使用默认配置。
    pub fn global() -> &'static Self {
        PARALLEL_CONFIG.get_or_init(Self::default)
    }

    /// 初始化全局配置
    ///
    /// 必须在程序启动时调用，只能调用一次。
    /// 如果已经初始化，返回 Err 包含传入的配置。
    pub fn init(config: Self) -> Result<(), Self> {
        PARALLEL_CONFIG.set(config)
    }

    /// 创建用于测试的配置（禁用并行）
    pub fn for_testing() -> Self {
        Self {
            enabled: false,
            min_work_items: usize::MAX,
            ..Default::default()
        }
    }

    /// 创建高性能配置
    pub fn high_performance() -> Self {
        Self {
            enabled: true,
            min_work_items: 500,
            min_items_per_thread: 128,
            adaptive_scheduling: true,
            ..Default::default()
        }
    }

    /// 决定是否对给定工作量并行化
    #[inline]
    pub fn should_parallelize(&self, work_items: usize) -> bool {
        self.enabled && work_items >= self.min_work_items
    }

    /// 计算推荐的块大小
    #[inline]
    pub fn recommended_chunk_size(&self, work_items: usize) -> usize {
        let threads = self
            .max_threads
            .min(work_items / self.min_items_per_thread)
            .max(1);
        (work_items / threads).max(1)
    }

    /// 获取有效线程数
    #[inline]
    pub fn effective_threads(&self, work_items: usize) -> usize {
        if !self.enabled {
            return 1;
        }
        let ideal = work_items / self.min_items_per_thread;
        ideal.clamp(1, self.max_threads)
    }
}

/// 构建器模式
impl ParallelConfig {
    /// 设置是否启用并行
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// 设置最小并行化单元数
    pub fn with_min_work_items(mut self, min: usize) -> Self {
        self.min_work_items = min;
        self
    }

    /// 设置每线程最小工作量
    pub fn with_min_items_per_thread(mut self, min: usize) -> Self {
        self.min_items_per_thread = min;
        self
    }

    /// 设置最大线程数
    pub fn with_max_threads(mut self, max: usize) -> Self {
        self.max_threads = max.max(1);
        self
    }

    /// 设置是否启用自适应调度
    pub fn with_adaptive_scheduling(mut self, enabled: bool) -> Self {
        self.adaptive_scheduling = enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_parallelize() {
        let config = ParallelConfig::default();
        assert!(!config.should_parallelize(100));
        assert!(config.should_parallelize(2000));

        let disabled = ParallelConfig::for_testing();
        assert!(!disabled.should_parallelize(10000));
    }

    #[test]
    fn test_recommended_chunk_size() {
        let config = ParallelConfig {
            max_threads: 4,
            min_items_per_thread: 100,
            ..Default::default()
        };

        // 1000 items / 4 threads = 250 per thread
        assert_eq!(config.recommended_chunk_size(1000), 250);

        // 100 items -> 1 thread -> chunk = 100
        assert_eq!(config.recommended_chunk_size(100), 100);
    }

    #[test]
    fn test_effective_threads() {
        let config = ParallelConfig {
            enabled: true,
            max_threads: 8,
            min_items_per_thread: 100,
            ..Default::default()
        };

        // 1000 items / 100 = 10 ideal, clamped to 8
        assert_eq!(config.effective_threads(1000), 8);

        // 200 items / 100 = 2
        assert_eq!(config.effective_threads(200), 2);

        // 50 items / 100 = 0, clamped to 1
        assert_eq!(config.effective_threads(50), 1);
    }
}
