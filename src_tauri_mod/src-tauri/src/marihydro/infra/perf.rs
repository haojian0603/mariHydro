// src-tauri/src/marihydro/infra/perf.rs
//! 性能优化工具模块
//!
//! 提供并行化辅助函数、SIMD 支持和性能监控工具。

use std::time::{Duration, Instant};

/// 并行阈值：低于此数量使用串行
pub const PARALLEL_THRESHOLD: usize = 1000;

/// 判断是否应使用并行计算
#[inline]
pub fn should_parallelize(n: usize) -> bool {
    n >= PARALLEL_THRESHOLD
}

/// 性能计时器
#[derive(Debug, Clone)]
pub struct PerfTimer {
    name: &'static str,
    start: Instant,
    elapsed: Duration,
    count: u64,
}

impl PerfTimer {
    /// 创建新的计时器
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            start: Instant::now(),
            elapsed: Duration::ZERO,
            count: 0,
        }
    }

    /// 开始计时
    #[inline]
    pub fn start(&mut self) {
        self.start = Instant::now();
    }

    /// 停止计时并累加
    #[inline]
    pub fn stop(&mut self) {
        self.elapsed += self.start.elapsed();
        self.count += 1;
    }

    /// 重置计时器
    pub fn reset(&mut self) {
        self.elapsed = Duration::ZERO;
        self.count = 0;
    }

    /// 获取总耗时
    pub fn total(&self) -> Duration {
        self.elapsed
    }

    /// 获取平均耗时
    pub fn average(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.elapsed / self.count as u32
        }
    }

    /// 获取调用次数
    pub fn count(&self) -> u64 {
        self.count
    }

    /// 获取名称
    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// 性能统计收集器
#[derive(Debug, Default)]
pub struct PerfStats {
    timers: Vec<PerfTimer>,
}

impl PerfStats {
    /// 创建新的统计收集器
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加计时器
    pub fn add_timer(&mut self, name: &'static str) -> usize {
        let idx = self.timers.len();
        self.timers.push(PerfTimer::new(name));
        idx
    }

    /// 获取计时器（可变引用）
    pub fn timer_mut(&mut self, idx: usize) -> Option<&mut PerfTimer> {
        self.timers.get_mut(idx)
    }

    /// 获取计时器（不可变引用）
    pub fn timer(&self, idx: usize) -> Option<&PerfTimer> {
        self.timers.get(idx)
    }

    /// 重置所有计时器
    pub fn reset_all(&mut self) {
        for timer in &mut self.timers {
            timer.reset();
        }
    }

    /// 生成报告
    pub fn report(&self) -> String {
        let mut lines = vec!["Performance Report:".to_string()];
        let mut total = Duration::ZERO;

        for timer in &self.timers {
            let elapsed = timer.total();
            total += elapsed;
            lines.push(format!(
                "  {}: {:.3}ms ({} calls, avg {:.3}µs)",
                timer.name(),
                elapsed.as_secs_f64() * 1000.0,
                timer.count(),
                timer.average().as_secs_f64() * 1e6
            ));
        }

        lines.push(format!("  Total: {:.3}ms", total.as_secs_f64() * 1000.0));
        lines.join("\n")
    }
}

/// 内存对齐辅助函数
#[inline]
pub fn align_to<T>(ptr: *const T, align: usize) -> *const T {
    let addr = ptr as usize;
    let aligned = (addr + align - 1) & !(align - 1);
    aligned as *const T
}

/// 缓存友好的分块大小（64KB L1 cache / 8 bytes per f64 / 2 arrays）
pub const CACHE_BLOCK_SIZE: usize = 4096;

/// 分块迭代辅助
pub struct ChunkIter {
    start: usize,
    end: usize,
    chunk_size: usize,
}

impl ChunkIter {
    /// 创建分块迭代器
    pub fn new(len: usize, chunk_size: usize) -> Self {
        Self {
            start: 0,
            end: len,
            chunk_size,
        }
    }

    /// 使用默认缓存友好大小
    pub fn cache_friendly(len: usize) -> Self {
        Self::new(len, CACHE_BLOCK_SIZE)
    }
}

impl Iterator for ChunkIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            return None;
        }
        let chunk_end = (self.start + self.chunk_size).min(self.end);
        let result = (self.start, chunk_end);
        self.start = chunk_end;
        Some(result)
    }
}

/// 向量化辅助：检查是否可以使用 SIMD
#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2() -> bool {
    false
}

/// 预取提示（可能被编译器忽略）
#[inline]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::_mm_prefetch;
        _mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
}

/// 并行求和
pub fn parallel_sum(data: &[f64]) -> f64 {
    if data.len() < PARALLEL_THRESHOLD {
        data.iter().sum()
    } else {
        use rayon::prelude::*;
        data.par_iter().sum()
    }
}

/// 并行最大值
pub fn parallel_max(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NEG_INFINITY;
    }
    if data.len() < PARALLEL_THRESHOLD {
        data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    } else {
        use rayon::prelude::*;
        data.par_iter().cloned().reduce(|| f64::NEG_INFINITY, f64::max)
    }
}

/// 并行最小值
pub fn parallel_min(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::INFINITY;
    }
    if data.len() < PARALLEL_THRESHOLD {
        data.iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        use rayon::prelude::*;
        data.par_iter().cloned().reduce(|| f64::INFINITY, f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_timer() {
        let mut timer = PerfTimer::new("test");
        timer.start();
        std::thread::sleep(Duration::from_millis(10));
        timer.stop();

        assert!(timer.total() >= Duration::from_millis(10));
        assert_eq!(timer.count(), 1);
    }

    #[test]
    fn test_chunk_iter() {
        let chunks: Vec<_> = ChunkIter::new(10, 3).collect();
        assert_eq!(chunks, vec![(0, 3), (3, 6), (6, 9), (9, 10)]);
    }

    #[test]
    fn test_parallel_sum() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let sum = parallel_sum(&data);
        assert!((sum - 4950.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_max_min() {
        let data = vec![1.0, 5.0, 3.0, 9.0, 2.0];
        assert!((parallel_max(&data) - 9.0).abs() < 1e-10);
        assert!((parallel_min(&data) - 1.0).abs() < 1e-10);
    }
}
