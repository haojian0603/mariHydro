// crates/mh_runtime/src/numerics/kahan_sum.rs

//! Kahan 求和算法（泛型版）

use crate::scalar::RuntimeScalar;

/// Kahan 求和算法（泛型版）
///
/// 使用 Kahan 算法减少浮点累加误差，适用于 `RuntimeScalar` 类型。
///
/// # 示例
///
/// ```rust
/// use mh_runtime::{RuntimeScalar, numerics::KahanSum};
///
/// fn sum_precise<S: RuntimeScalar>(data: &[S]) -> S {
///     KahanSum::sum_iter(data.iter().cloned())
/// }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct KahanSum<S: RuntimeScalar> {
    sum: S,
    compensation: S,
}

impl<S: RuntimeScalar> KahanSum<S> {
    /// 创建新的求和器
    pub fn new() -> Self {
        Self {
            sum: S::ZERO,
            compensation: S::ZERO,
        }
    }

    /// 添加一个值
    #[inline]
    pub fn add(&mut self, value: S) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    /// 获取当前求和值
    #[inline]
    pub fn value(&self) -> S {
        self.sum
    }

    /// 重置求和器
    #[inline]
    pub fn reset(&mut self) {
        self.sum = S::ZERO;
        self.compensation = S::ZERO;
    }

    /// 从迭代器求和
    pub fn sum_iter<I: IntoIterator<Item = S>>(iter: I) -> S {
        let mut kahan = Self::new();
        for v in iter {
            kahan.add(v);
        }
        kahan.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_sum_f32() {
        let data = vec![0.1f32; 1000];
        let sum = KahanSum::sum_iter(data.iter().cloned());
        assert!((sum - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_kahan_sum_f64() {
        let data = vec![0.1f64; 1000];
        let sum = KahanSum::sum_iter(data.iter().cloned());
        assert!((sum - 100.0).abs() < 1e-10);
    }
}