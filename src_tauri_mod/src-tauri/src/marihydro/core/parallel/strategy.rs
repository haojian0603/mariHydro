//! 并行执行策略
//!
//! 定义不同的并行执行策略和执行函数。

use super::config::ParallelConfig;
use rayon::prelude::*;

/// 并行执行策略
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParallelStrategy {
    /// 串行执行
    Sequential,
    /// 静态分块并行
    StaticChunks {
        /// 每块大小
        chunk_size: usize,
    },
    /// 动态调度（rayon默认）
    Dynamic,
    /// 工作窃取
    WorkStealing,
    /// 着色并行（用于网格面循环）
    Colored,
}

impl ParallelStrategy {
    /// 判断是否为并行策略
    pub fn is_parallel(&self) -> bool {
        !matches!(self, ParallelStrategy::Sequential)
    }
}

/// 策略选择器
///
/// 根据问题特征选择合适的并行策略。
pub struct StrategySelector {
    config: &'static ParallelConfig,
}

impl StrategySelector {
    /// 创建新的策略选择器
    pub fn new() -> Self {
        Self {
            config: ParallelConfig::global(),
        }
    }

    /// 使用自定义配置创建
    pub fn with_config(config: &'static ParallelConfig) -> Self {
        Self { config }
    }

    /// 为通量计算选择策略
    ///
    /// # Arguments
    /// - `n_faces`: 面数量
    /// - `has_coloring`: 是否有着色信息
    pub fn for_flux_computation(&self, n_faces: usize, has_coloring: bool) -> ParallelStrategy {
        if !self.config.should_parallelize(n_faces) {
            return ParallelStrategy::Sequential;
        }

        if has_coloring {
            ParallelStrategy::Colored
        } else {
            ParallelStrategy::Dynamic
        }
    }

    /// 为梯度计算选择策略
    ///
    /// 梯度计算无数据竞争，使用静态分块。
    pub fn for_gradient(&self, n_cells: usize) -> ParallelStrategy {
        if !self.config.should_parallelize(n_cells) {
            return ParallelStrategy::Sequential;
        }

        let chunk = self.config.recommended_chunk_size(n_cells);
        ParallelStrategy::StaticChunks { chunk_size: chunk }
    }

    /// 为规约操作选择策略
    ///
    /// 规约操作使用动态调度以平衡负载。
    pub fn for_reduction(&self, n_items: usize) -> ParallelStrategy {
        if !self.config.should_parallelize(n_items) {
            return ParallelStrategy::Sequential;
        }
        ParallelStrategy::Dynamic
    }

    /// 为源项计算选择策略
    ///
    /// 源项计算通常是独立的，可以使用工作窃取。
    pub fn for_source_terms(&self, n_cells: usize) -> ParallelStrategy {
        if !self.config.should_parallelize(n_cells) {
            return ParallelStrategy::Sequential;
        }
        ParallelStrategy::WorkStealing
    }

    /// 为时间步长计算选择策略
    ///
    /// 时间步长是规约操作。
    pub fn for_timestep(&self, n_cells: usize) -> ParallelStrategy {
        self.for_reduction(n_cells)
    }

    /// 为限制器计算选择策略
    pub fn for_limiter(&self, n_cells: usize) -> ParallelStrategy {
        self.for_gradient(n_cells)
    }
}

impl Default for StrategySelector {
    fn default() -> Self {
        Self::new()
    }
}

/// 根据策略执行并行循环（不可变引用）
///
/// # Arguments
/// - `items`: 要处理的数据切片
/// - `strategy`: 并行策略
/// - `f`: 处理函数，接收索引和元素引用
///
/// # Panics
/// 如果使用 `Colored` 策略会 panic，因为需要着色数据。
pub fn parallel_for<T, F>(items: &[T], strategy: ParallelStrategy, f: F)
where
    T: Sync,
    F: Fn(usize, &T) + Sync,
{
    match strategy {
        ParallelStrategy::Sequential => {
            for (i, item) in items.iter().enumerate() {
                f(i, item);
            }
        }
        ParallelStrategy::StaticChunks { chunk_size } => {
            items
                .par_chunks(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    for (local_idx, item) in chunk.iter().enumerate() {
                        f(chunk_idx * chunk_size + local_idx, item);
                    }
                });
        }
        ParallelStrategy::Dynamic | ParallelStrategy::WorkStealing => {
            items.par_iter().enumerate().for_each(|(i, item)| f(i, item));
        }
        ParallelStrategy::Colored => {
            panic!("Colored strategy requires coloring data, use parallel_for_colored");
        }
    }
}

/// 根据策略执行并行循环（可变引用）
///
/// # Arguments
/// - `items`: 要处理的数据切片
/// - `strategy`: 并行策略
/// - `f`: 处理函数，接收索引和元素可变引用
pub fn parallel_for_mut<T, F>(items: &mut [T], strategy: ParallelStrategy, f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Sync,
{
    match strategy {
        ParallelStrategy::Sequential => {
            for (i, item) in items.iter_mut().enumerate() {
                f(i, item);
            }
        }
        ParallelStrategy::StaticChunks { chunk_size } => {
            items
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    for (local_idx, item) in chunk.iter_mut().enumerate() {
                        f(chunk_idx * chunk_size + local_idx, item);
                    }
                });
        }
        ParallelStrategy::Dynamic | ParallelStrategy::WorkStealing => {
            items
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, item)| f(i, item));
        }
        ParallelStrategy::Colored => {
            panic!("Colored strategy requires coloring data, use parallel_for_colored_mut");
        }
    }
}

/// 并行规约操作
pub fn parallel_reduce<T, I, F, R>(items: I, identity: T, map: F, reduce: R) -> T
where
    T: Clone + Send + Sync,
    I: IntoParallelIterator,
    I::Item: Send,
    F: Fn(I::Item) -> T + Sync,
    R: Fn(T, T) -> T + Sync,
{
    items
        .into_par_iter()
        .map(map)
        .reduce(|| identity.clone(), reduce)
}

/// 并行求最大值
pub fn parallel_max<T, F>(items: &[T], f: F) -> Option<f64>
where
    T: Sync,
    F: Fn(&T) -> f64 + Sync,
{
    if items.is_empty() {
        return None;
    }

    let max = items
        .par_iter()
        .map(|item| f(item))
        .reduce(|| f64::NEG_INFINITY, f64::max);

    if max.is_finite() {
        Some(max)
    } else {
        None
    }
}

/// 并行求最小值
pub fn parallel_min<T, F>(items: &[T], f: F) -> Option<f64>
where
    T: Sync,
    F: Fn(&T) -> f64 + Sync,
{
    if items.is_empty() {
        return None;
    }

    let min = items
        .par_iter()
        .map(|item| f(item))
        .reduce(|| f64::INFINITY, f64::min);

    if min.is_finite() {
        Some(min)
    } else {
        None
    }
}

/// 并行求和
pub fn parallel_sum<T, F>(items: &[T], f: F) -> f64
where
    T: Sync,
    F: Fn(&T) -> f64 + Sync,
{
    items.par_iter().map(|item| f(item)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_selection() {
        let selector = StrategySelector::new();

        // 小数据量应该串行
        let strategy = selector.for_gradient(100);
        assert_eq!(strategy, ParallelStrategy::Sequential);

        // 大数据量应该并行
        let strategy = selector.for_gradient(10000);
        assert!(strategy.is_parallel());
    }

    #[test]
    fn test_parallel_for() {
        let data: Vec<i32> = (0..100).collect();
        let sum = std::sync::atomic::AtomicI32::new(0);

        parallel_for(&data, ParallelStrategy::Sequential, |_, &x| {
            sum.fetch_add(x, std::sync::atomic::Ordering::Relaxed);
        });

        assert_eq!(sum.load(std::sync::atomic::Ordering::Relaxed), 4950);
    }

    #[test]
    fn test_parallel_max() {
        let data = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let max = parallel_max(&data, |&x| x);
        assert_eq!(max, Some(5.0));
    }

    #[test]
    fn test_parallel_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = parallel_sum(&data, |&x| x);
        assert!((sum - 15.0).abs() < 1e-10);
    }
}
