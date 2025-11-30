//! 安全原子浮点数
//!
//! 提供 `AtomicF64` 类型，支持线程安全的原子浮点操作。
//! 在原子操作时自动过滤 NaN/Inf 值，防止数值污染。

use std::sync::atomic::{AtomicU64, Ordering};

/// 安全的原子浮点数
///
/// 解决 NaN 污染原子累加器的问题。在原子操作时：
/// - 过滤 NaN/Inf 输入值
/// - 检查运算结果的有效性
/// - 防止溢出导致的 Inf
///
/// # Example
/// ```rust
/// use marihydro::core::numerical::AtomicF64;
/// use std::sync::atomic::Ordering;
///
/// let counter = AtomicF64::new(0.0);
/// counter.fetch_add_safe(1.0, Ordering::SeqCst);
/// assert_eq!(counter.load(Ordering::SeqCst), 1.0);
///
/// // NaN 值会被忽略
/// counter.fetch_add_safe(f64::NAN, Ordering::SeqCst);
/// assert_eq!(counter.load(Ordering::SeqCst), 1.0);
/// ```
pub struct AtomicF64 {
    bits: AtomicU64,
}

impl AtomicF64 {
    /// 创建新的原子浮点数
    ///
    /// # Panics
    /// 在 debug 模式下，如果初始值不是有限数，会 panic。
    #[inline]
    pub fn new(value: f64) -> Self {
        debug_assert!(value.is_finite(), "AtomicF64 初始值必须有限");
        Self {
            bits: AtomicU64::new(value.to_bits()),
        }
    }

    /// 创建值为零的原子浮点数
    #[inline]
    pub fn zero() -> Self {
        Self::new(0.0)
    }

    /// 安全的原子加法 - 跳过非有限值
    ///
    /// 如果 `delta` 不是有限数，操作会被忽略。
    /// 如果加法结果不是有限数，操作也会被忽略。
    ///
    /// # Returns
    /// 操作前的旧值
    #[inline]
    pub fn fetch_add_safe(&self, delta: f64, ordering: Ordering) -> f64 {
        // 过滤无效输入
        if !delta.is_finite() {
            #[cfg(debug_assertions)]
            log::warn!("AtomicF64::fetch_add_safe 收到非有限值: {}", delta);
            return f64::from_bits(self.bits.load(ordering));
        }

        loop {
            let old_bits = self.bits.load(Ordering::Relaxed);
            let old = f64::from_bits(old_bits);
            let new = old + delta;

            // 检查结果有效性
            if !new.is_finite() {
                #[cfg(debug_assertions)]
                log::warn!(
                    "AtomicF64: 累加结果溢出 {} + {} = {}",
                    old,
                    delta,
                    new
                );
                return old;
            }

            match self.bits.compare_exchange_weak(
                old_bits,
                new.to_bits(),
                ordering,
                Ordering::Relaxed,
            ) {
                Ok(_) => return old,
                Err(_) => continue,
            }
        }
    }

    /// 安全的原子最大值 - 使用可比较位模式
    ///
    /// 对于非有限或负值，操作会被忽略。
    ///
    /// # Returns
    /// 操作前的旧值
    #[inline]
    pub fn fetch_max_safe(&self, value: f64, ordering: Ordering) -> f64 {
        if !value.is_finite() || value < 0.0 {
            return f64::from_bits(self.bits.load(ordering));
        }

        // 对于正有限浮点数，bits 比较与值比较一致
        let new_bits = value.to_bits();
        loop {
            let old_bits = self.bits.load(Ordering::Relaxed);
            if new_bits <= old_bits {
                return f64::from_bits(old_bits);
            }
            match self.bits.compare_exchange_weak(
                old_bits,
                new_bits,
                ordering,
                Ordering::Relaxed,
            ) {
                Ok(_) => return f64::from_bits(old_bits),
                Err(_) => continue,
            }
        }
    }

    /// 安全的原子最小值
    ///
    /// 对于非有限或负值，操作会被忽略。
    ///
    /// # Returns
    /// 操作前的旧值
    #[inline]
    pub fn fetch_min_safe(&self, value: f64, ordering: Ordering) -> f64 {
        if !value.is_finite() {
            return f64::from_bits(self.bits.load(ordering));
        }

        let new_bits = value.to_bits();
        loop {
            let old_bits = self.bits.load(Ordering::Relaxed);
            // 对于正数，较小的值有较小的 bits
            if new_bits >= old_bits {
                return f64::from_bits(old_bits);
            }
            match self.bits.compare_exchange_weak(
                old_bits,
                new_bits,
                ordering,
                Ordering::Relaxed,
            ) {
                Ok(_) => return f64::from_bits(old_bits),
                Err(_) => continue,
            }
        }
    }

    /// 原子加载
    #[inline]
    pub fn load(&self, ordering: Ordering) -> f64 {
        f64::from_bits(self.bits.load(ordering))
    }

    /// 原子存储
    ///
    /// # Panics
    /// 在 debug 模式下，如果值不是有限数，会 panic。
    #[inline]
    pub fn store(&self, value: f64, ordering: Ordering) {
        debug_assert!(value.is_finite(), "AtomicF64::store 值必须有限");
        self.bits.store(value.to_bits(), ordering);
    }

    /// 安全的原子存储 - 忽略非有限值
    #[inline]
    pub fn store_safe(&self, value: f64, ordering: Ordering) -> bool {
        if value.is_finite() {
            self.bits.store(value.to_bits(), ordering);
            true
        } else {
            false
        }
    }

    /// 原子交换
    #[inline]
    pub fn swap(&self, value: f64, ordering: Ordering) -> f64 {
        debug_assert!(value.is_finite(), "AtomicF64::swap 值必须有限");
        f64::from_bits(self.bits.swap(value.to_bits(), ordering))
    }

    /// 比较并交换
    #[inline]
    pub fn compare_exchange(
        &self,
        current: f64,
        new: f64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<f64, f64> {
        match self.bits.compare_exchange(
            current.to_bits(),
            new.to_bits(),
            success,
            failure,
        ) {
            Ok(bits) => Ok(f64::from_bits(bits)),
            Err(bits) => Err(f64::from_bits(bits)),
        }
    }

    /// 重置为零
    #[inline]
    pub fn reset(&self) {
        self.store(0.0, Ordering::Release);
    }
}

impl Default for AtomicF64 {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::fmt::Debug for AtomicF64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AtomicF64({})", self.load(Ordering::Relaxed))
    }
}

// AtomicF64 不能 Clone，因为原子类型不应该被复制
// 但可以 Send + Sync
unsafe impl Send for AtomicF64 {}
unsafe impl Sync for AtomicF64 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_load() {
        let a = AtomicF64::new(42.0);
        assert_eq!(a.load(Ordering::SeqCst), 42.0);
    }

    #[test]
    fn test_fetch_add_safe() {
        let a = AtomicF64::new(10.0);

        // 正常加法
        let old = a.fetch_add_safe(5.0, Ordering::SeqCst);
        assert_eq!(old, 10.0);
        assert_eq!(a.load(Ordering::SeqCst), 15.0);

        // NaN 被忽略
        let old = a.fetch_add_safe(f64::NAN, Ordering::SeqCst);
        assert_eq!(old, 15.0);
        assert_eq!(a.load(Ordering::SeqCst), 15.0);

        // Inf 被忽略
        let old = a.fetch_add_safe(f64::INFINITY, Ordering::SeqCst);
        assert_eq!(old, 15.0);
        assert_eq!(a.load(Ordering::SeqCst), 15.0);
    }

    #[test]
    fn test_fetch_max_safe() {
        let a = AtomicF64::new(10.0);

        // 更大的值
        a.fetch_max_safe(20.0, Ordering::SeqCst);
        assert_eq!(a.load(Ordering::SeqCst), 20.0);

        // 更小的值不改变
        a.fetch_max_safe(5.0, Ordering::SeqCst);
        assert_eq!(a.load(Ordering::SeqCst), 20.0);

        // NaN 被忽略
        a.fetch_max_safe(f64::NAN, Ordering::SeqCst);
        assert_eq!(a.load(Ordering::SeqCst), 20.0);
    }

    #[test]
    fn test_store_safe() {
        let a = AtomicF64::new(0.0);

        assert!(a.store_safe(42.0, Ordering::SeqCst));
        assert_eq!(a.load(Ordering::SeqCst), 42.0);

        assert!(!a.store_safe(f64::NAN, Ordering::SeqCst));
        assert_eq!(a.load(Ordering::SeqCst), 42.0);
    }

    #[test]
    fn test_reset() {
        let a = AtomicF64::new(100.0);
        a.reset();
        assert_eq!(a.load(Ordering::SeqCst), 0.0);
    }
}
