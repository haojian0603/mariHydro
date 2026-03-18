// marihydro\crates\mh_foundation\src\metrics.rs
//! 基础性能计数器
//!
//! 提供轻量级的原子计数功能，仅用于基础统计。
//! 运行时性能指标（如耗时、物理引擎相关计数）应在 mh_runtime 层实现。

use std::sync::atomic::{AtomicU64, Ordering};

/// 原子计数器（无锁）
///
/// 仅提供基础递增/读取功能，无运行时概念。
/// 用于 Foundation 层内部统计，不暴露给上层。
#[derive(Debug, Default)]
pub struct Counter(AtomicU64);

impl Counter {
    /// 创建零值计数器
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// 增加计数
    #[inline]
    pub fn inc(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    /// 增加指定值
    #[inline]
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }

    /// 获取当前值
    #[inline]
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    /// 重置为零
    #[inline]
    pub fn reset(&self) {
        self.0.store(0, Ordering::Relaxed);
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
}