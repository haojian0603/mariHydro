// src-tauri/src/marihydro/physics/engine/perf_utils.rs
//! 性能工具模块
//! 
//! 集中管理性能优化相关的工具函数：
//! - 并行归约操作
//! - SmallVec 优化
//! - 缓存友好的遍历模式
//!
//! ## 并行阈值
//! 
//! 默认并行阈值为 1000，可通过 `ParallelConfig` 配置。
//! 对于小规模计算，串行执行通常更快（避免线程调度开销）。

use glam::DVec2;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// 并行配置
///
/// 控制并行执行的阈值和参数
#[derive(Debug, Clone, Copy)]
pub struct ParallelConfig {
    /// 基础阈值（元素数量低于此值使用串行）
    pub base_threshold: usize,
    /// 每线程额外阈值
    pub per_thread: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            base_threshold: 500,
            per_thread: 200,
        }
    }
}

impl ParallelConfig {
    /// 创建自定义配置
    pub fn new(base_threshold: usize, per_thread: usize) -> Self {
        Self {
            base_threshold,
            per_thread,
        }
    }

    /// 保守配置（更倾向于并行）
    pub fn aggressive() -> Self {
        Self {
            base_threshold: 200,
            per_thread: 100,
        }
    }

    /// 保守配置（更倾向于串行）
    pub fn conservative() -> Self {
        Self {
            base_threshold: 2000,
            per_thread: 500,
        }
    }

    /// 计算当前有效阈值
    ///
    /// 根据可用线程数动态调整
    pub fn threshold(&self) -> usize {
        let threads = rayon::current_num_threads();
        self.base_threshold + self.per_thread * threads
    }

    /// 判断是否应使用并行
    #[inline]
    pub fn should_parallelize(&self, n: usize) -> bool {
        n >= self.threshold()
    }
}

/// 全局默认并行阈值
///
/// 简单情况下使用固定阈值 1000
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 1000;

/// 原子方式更新 f64 最大值
///
/// 使用 CAS 循环正确比较 f64 值（而非位模式）
/// P2-005 修复: fetch_max 对 f64 位模式比较不正确
#[inline]
fn atomic_f64_max(atomic: &AtomicU64, new_val: f64) {
    let mut old_bits = atomic.load(Ordering::Relaxed);
    loop {
        let old_val = f64::from_bits(old_bits);
        if new_val <= old_val {
            break;
        }
        match atomic.compare_exchange_weak(
            old_bits,
            new_val.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(x) => old_bits = x,
        }
    }
}

/// 原子方式更新 f64 最小值
///
/// 使用 CAS 循环正确比较 f64 值
#[inline]
fn atomic_f64_min(atomic: &AtomicU64, new_val: f64) {
    let mut old_bits = atomic.load(Ordering::Relaxed);
    loop {
        let old_val = f64::from_bits(old_bits);
        if new_val >= old_val {
            break;
        }
        match atomic.compare_exchange_weak(
            old_bits,
            new_val.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(x) => old_bits = x,
        }
    }
}

/// 并行计算最大值
#[inline]
pub fn parallel_max(values: &[f64]) -> f64 {
    if values.len() < 1000 {
        values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    } else {
        let max = AtomicU64::new(f64::NEG_INFINITY.to_bits());
        values.par_iter().for_each(|&v| {
            atomic_f64_max(&max, v);
        });
        f64::from_bits(max.load(Ordering::Relaxed))
    }
}

/// 并行计算最小值
#[inline]
pub fn parallel_min(values: &[f64]) -> f64 {
    if values.len() < 1000 {
        values.iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        let min = AtomicU64::new(f64::INFINITY.to_bits());
        values.par_iter().for_each(|&v| {
            atomic_f64_min(&min, v);
        });
        f64::from_bits(min.load(Ordering::Relaxed))
    }
}

/// 并行计算总和
#[inline]
pub fn parallel_sum(values: &[f64]) -> f64 {
    if values.len() < DEFAULT_PARALLEL_THRESHOLD {
        values.iter().sum()
    } else {
        values.par_iter().sum()
    }
}

/// 带配置的并行计算最大值
#[inline]
pub fn parallel_max_with_config(values: &[f64], config: &ParallelConfig) -> f64 {
    if !config.should_parallelize(values.len()) {
        values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    } else {
        let max = AtomicU64::new(f64::NEG_INFINITY.to_bits());
        values.par_iter().for_each(|&v| {
            atomic_f64_max(&max, v);
        });
        f64::from_bits(max.load(Ordering::Relaxed))
    }
}

/// 带配置的并行计算最小值
#[inline]
pub fn parallel_min_with_config(values: &[f64], config: &ParallelConfig) -> f64 {
    if !config.should_parallelize(values.len()) {
        values.iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        let min = AtomicU64::new(f64::INFINITY.to_bits());
        values.par_iter().for_each(|&v| {
            atomic_f64_min(&min, v);
        });
        f64::from_bits(min.load(Ordering::Relaxed))
    }
}

/// 带配置的并行计算总和
#[inline]
pub fn parallel_sum_with_config(values: &[f64], config: &ParallelConfig) -> f64 {
    if !config.should_parallelize(values.len()) {
        values.iter().sum()
    } else {
        values.par_iter().sum()
    }
}

/// 并行计算最大速度（从 hu, hv, h 数组）
pub fn parallel_max_velocity(hu: &[f64], hv: &[f64], h: &[f64], h_dry: f64) -> f64 {
    let n = hu.len().min(hv.len()).min(h.len());
    
    if n < 1000 {
        (0..n).fold(0.0f64, |max, i| {
            if h[i] <= h_dry {
                return max;
            }
            let u = hu[i] / h[i];
            let v = hv[i] / h[i];
            let speed = (u * u + v * v).sqrt();
            max.max(speed)
        })
    } else {
        let max = AtomicU64::new(0u64);
        (0..n).into_par_iter().for_each(|i| {
            if h[i] <= h_dry {
                return;
            }
            let u = hu[i] / h[i];
            let v = hv[i] / h[i];
            let speed = (u * u + v * v).sqrt();
            atomic_f64_max(&max, speed);
        });
        f64::from_bits(max.load(Ordering::Relaxed))
    }
}

/// 并行计算最大波速
pub fn parallel_max_wave_speed(hu: &[f64], hv: &[f64], h: &[f64], g: f64, h_dry: f64) -> f64 {
    let n = hu.len().min(hv.len()).min(h.len());
    
    if n < 1000 {
        (0..n).fold(0.0f64, |max, i| {
            if h[i] <= h_dry {
                return max;
            }
            let u = hu[i] / h[i];
            let v = hv[i] / h[i];
            let speed = (u * u + v * v).sqrt();
            let c = (g * h[i]).sqrt();
            max.max(speed + c)
        })
    } else {
        let max = AtomicU64::new(0u64);
        (0..n).into_par_iter().for_each(|i| {
            if h[i] <= h_dry {
                return;
            }
            let u = hu[i] / h[i];
            let v = hv[i] / h[i];
            let speed = (u * u + v * v).sqrt();
            let c = (g * h[i]).sqrt();
            atomic_f64_max(&max, speed + c);
        });
        f64::from_bits(max.load(Ordering::Relaxed))
    }
}

/// 缓存友好的批量操作
pub struct BatchOperator {
    /// 批次大小
    batch_size: usize,
}

impl BatchOperator {
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }
    
    /// 批量应用操作
    pub fn apply<F>(&self, n: usize, mut f: F)
    where
        F: FnMut(usize, usize),
    {
        let mut start = 0;
        while start < n {
            let end = (start + self.batch_size).min(n);
            f(start, end);
            start = end;
        }
    }
    
    /// 并行批量应用
    pub fn apply_parallel<F>(&self, n: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        let batch_size = self.batch_size;
        let n_batches = (n + batch_size - 1) / batch_size;
        
        (0..n_batches).into_par_iter().for_each(|batch_idx| {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(n);
            f(start, end);
        });
    }
}

impl Default for BatchOperator {
    fn default() -> Self {
        Self::new(256) // 适合 L1 缓存
    }
}

/// 小型固定容量向量（避免堆分配）
/// 
/// 对于已知最大容量的小型集合，比 Vec 更高效
#[derive(Clone, Debug)]
pub struct SmallVec<T, const N: usize> {
    data: [Option<T>; N],
    len: usize,
}

impl<T: Copy + Default, const N: usize> SmallVec<T, N> {
    pub fn new() -> Self {
        Self {
            data: [None; N],
            len: 0,
        }
    }
    
    pub fn push(&mut self, value: T) -> bool {
        if self.len < N {
            self.data[self.len] = Some(value);
            self.len += 1;
            true
        } else {
            false
        }
    }
    
    pub fn len(&self) -> usize {
        self.len
    }
    
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    pub fn clear(&mut self) {
        for i in 0..self.len {
            self.data[i] = None;
        }
        self.len = 0;
    }
    
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data[..self.len].iter().filter_map(|x| x.as_ref())
    }
    
    pub fn as_slice(&self) -> &[Option<T>] {
        &self.data[..self.len]
    }
}

impl<T: Copy + Default, const N: usize> Default for SmallVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// 面索引的小型向量（单元通常最多6-8个面）
pub type FaceSmallVec = SmallVec<usize, 8>;

/// 单元索引的小型向量
pub type CellSmallVec = SmallVec<usize, 8>;

/// 内存对齐的向量（用于 SIMD 优化）
#[repr(align(64))]
pub struct AlignedVec<T> {
    data: Vec<T>,
}

impl<T: Clone> AlignedVec<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }
    
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }
    
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T: Clone + Default> AlignedVec<T> {
    pub fn new_filled(len: usize, value: T) -> Self {
        Self {
            data: vec![value; len],
        }
    }
}

/// 预分配的结果缓冲区
pub struct ResultBuffer<T> {
    data: Vec<T>,
    valid: Vec<bool>,
}

impl<T: Clone + Default> ResultBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            valid: vec![false; capacity],
        }
    }
    
    pub fn set(&mut self, idx: usize, value: T) {
        if idx < self.data.len() {
            self.data[idx] = value;
            self.valid[idx] = true;
        }
    }
    
    pub fn get(&self, idx: usize) -> Option<&T> {
        if idx < self.data.len() && self.valid[idx] {
            Some(&self.data[idx])
        } else {
            None
        }
    }
    
    pub fn reset(&mut self) {
        self.valid.fill(false);
    }
    
    pub fn iter_valid(&self) -> impl Iterator<Item = (usize, &T)> {
        self.data.iter().enumerate().filter(|(i, _)| self.valid[*i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_max() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert_eq!(parallel_max(&values), 99.0);
    }
    
    #[test]
    fn test_parallel_min() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert_eq!(parallel_min(&values), 0.0);
    }
    
    #[test]
    fn test_small_vec() {
        let mut sv: SmallVec<i32, 4> = SmallVec::new();
        assert!(sv.push(1));
        assert!(sv.push(2));
        assert!(sv.push(3));
        assert!(sv.push(4));
        assert!(!sv.push(5)); // 容量已满
        assert_eq!(sv.len(), 4);
    }
    
    #[test]
    fn test_batch_operator() {
        let op = BatchOperator::new(10);
        let mut count = 0;
        op.apply(25, |start, end| {
            count += end - start;
        });
        assert_eq!(count, 25);
    }
}
