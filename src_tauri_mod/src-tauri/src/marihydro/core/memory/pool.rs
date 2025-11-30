//! 缓冲池实现
//!
//! 提供线程安全的缓冲区池管理。

use parking_lot::Mutex;
use std::ops::{Deref, DerefMut};

/// 通用缓冲区池
///
/// 提供线程安全的缓冲区复用，减少热路径上的内存分配。
///
/// # 使用示例
///
/// ```rust
/// use marihydro::core::memory::BufferPool;
///
/// let pool: BufferPool<f64> = BufferPool::new(1000);
///
/// {
///     let mut buffer = pool.acquire();
///     buffer[0] = 1.0;
///     // buffer 在离开作用域时自动归还
/// }
/// ```
pub struct BufferPool<T: Clone + Default + Send> {
    buffers: Mutex<Vec<Vec<T>>>,
    element_count: usize,
    max_pool_size: usize,
}

impl<T: Clone + Default + Send> BufferPool<T> {
    /// 创建新的缓冲池
    ///
    /// # Arguments
    /// - `element_count`: 每个缓冲区的元素数量
    pub fn new(element_count: usize) -> Self {
        Self::with_capacity(element_count, 16)
    }

    /// 创建带容量限制的缓冲池
    ///
    /// # Arguments
    /// - `element_count`: 每个缓冲区的元素数量
    /// - `max_pool_size`: 池中最大缓冲区数量
    pub fn with_capacity(element_count: usize, max_pool_size: usize) -> Self {
        Self {
            buffers: Mutex::new(Vec::with_capacity(max_pool_size.min(8))),
            element_count,
            max_pool_size,
        }
    }

    /// 获取缓冲区
    ///
    /// 优先从池中获取，如果池为空则新建。
    pub fn acquire(&self) -> PooledBuffer<'_, T> {
        let buffer = self
            .buffers
            .lock()
            .pop()
            .unwrap_or_else(|| vec![T::default(); self.element_count]);

        PooledBuffer {
            buffer: Some(buffer),
            pool: self,
        }
    }

    /// 获取已清零的缓冲区
    pub fn acquire_zeroed(&self) -> PooledBuffer<'_, T> {
        let mut buffer = self.acquire();
        buffer.iter_mut().for_each(|x| *x = T::default());
        buffer
    }

    /// 归还缓冲区
    fn release(&self, mut buffer: Vec<T>) {
        let mut pool = self.buffers.lock();
        if pool.len() < self.max_pool_size {
            // 清零并归还
            buffer.iter_mut().for_each(|x| *x = T::default());
            pool.push(buffer);
        }
        // 否则让 buffer 被 drop
    }

    /// 预热池（预先分配缓冲区）
    pub fn warm_up(&self, count: usize) {
        let mut pool = self.buffers.lock();
        let to_add = (self.max_pool_size - pool.len()).min(count);
        for _ in 0..to_add {
            pool.push(vec![T::default(); self.element_count]);
        }
    }

    /// 清空池
    pub fn clear(&self) {
        self.buffers.lock().clear();
    }

    /// 获取池中缓冲区数量
    pub fn available(&self) -> usize {
        self.buffers.lock().len()
    }

    /// 获取每个缓冲区的元素数量
    pub fn element_count(&self) -> usize {
        self.element_count
    }
}

/// 自动归还的缓冲区
///
/// 在离开作用域时自动归还到池中。
pub struct PooledBuffer<'a, T: Clone + Default + Send> {
    buffer: Option<Vec<T>>,
    pool: &'a BufferPool<T>,
}

impl<T: Clone + Default + Send> PooledBuffer<'_, T> {
    /// 填充指定值
    pub fn fill(&mut self, value: T) {
        if let Some(ref mut buf) = self.buffer {
            buf.iter_mut().for_each(|x| *x = value.clone());
        }
    }

    /// 获取长度
    pub fn len(&self) -> usize {
        self.buffer.as_ref().map_or(0, |b| b.len())
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Clone + Default + Send> Deref for PooledBuffer<'_, T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Vec<T> {
        self.buffer.as_ref().expect("Buffer already released")
    }
}

impl<T: Clone + Default + Send> DerefMut for PooledBuffer<'_, T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        self.buffer.as_mut().expect("Buffer already released")
    }
}

impl<T: Clone + Default + Send> Drop for PooledBuffer<'_, T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}

/// 专用的 f64 缓冲池
pub type ScalarPool = BufferPool<f64>;

/// 专用的向量缓冲池
pub type VectorPool = BufferPool<glam::DVec2>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_acquire_release() {
        let pool: BufferPool<f64> = BufferPool::new(100);

        assert_eq!(pool.available(), 0);

        {
            let mut buf = pool.acquire();
            buf[0] = 1.0;
            assert_eq!(pool.available(), 0);
        }

        // 归还后池中应该有一个缓冲区
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn test_pool_reuse() {
        let pool: BufferPool<f64> = BufferPool::new(100);

        {
            let mut buf = pool.acquire();
            buf[0] = 42.0;
        }

        // 再次获取应该是同一个（已清零的）缓冲区
        let buf = pool.acquire();
        assert_eq!(buf[0], 0.0);
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_pool_max_size() {
        let pool: BufferPool<f64> = BufferPool::with_capacity(10, 2);

        {
            let _b1 = pool.acquire();
            let _b2 = pool.acquire();
            let _b3 = pool.acquire();
        }

        // 只保留 2 个
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn test_warm_up() {
        let pool: BufferPool<f64> = BufferPool::with_capacity(100, 4);
        pool.warm_up(3);
        assert_eq!(pool.available(), 3);
    }
}
