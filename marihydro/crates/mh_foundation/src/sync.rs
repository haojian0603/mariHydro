// mh_foundation/src/sync.rs

//! 同步原语辅助模块
//!
//! 提供线程安全的锁获取工具，处理 Mutex 毒化等边缘情况。
//!
//! # 设计目标
//!
//! - 提供安全的 Mutex 锁获取，避免毒化时 panic
//! - 支持可配置的锁获取策略
//! - 记录毒化恢复日志便于调试
//!
//! # 示例
//!
//! ```
//! use std::sync::Mutex;
//! use mh_foundation::sync::lock_or_recover;
//!
//! let mutex = Mutex::new(42);
//! let guard = lock_or_recover(&mutex);
//! assert_eq!(*guard, 42);
//! ```

use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

// ============================================================================
// 锁获取策略
// ============================================================================

/// 锁获取策略
///
/// 定义当 Mutex 被毒化时的处理方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LockStrategy {
    /// 毒化时 panic（默认 std 行为）
    Panic,
    /// 毒化时恢复（忽略毒化状态，继续使用数据）
    #[default]
    Recover,
    /// 毒化时返回错误
    Error,
}

// ============================================================================
// Mutex 辅助函数
// ============================================================================

/// 安全获取 Mutex 锁，毒化时自动恢复
///
/// 当 Mutex 被毒化（持有锁的线程 panic）时，此函数会：
/// 1. 记录警告日志
/// 2. 恢复内部数据并返回锁守卫
///
/// # 参数
///
/// - `mutex`: 要获取锁的 Mutex 引用
///
/// # 返回
///
/// 锁守卫，即使 Mutex 被毒化也能获取
///
/// # 示例
///
/// ```
/// use std::sync::Mutex;
/// use mh_foundation::sync::lock_or_recover;
///
/// let mutex = Mutex::new(100);
/// let guard = lock_or_recover(&mutex);
/// assert_eq!(*guard, 100);
/// ```
#[inline]
pub fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        // 在生产环境中记录警告
        #[cfg(feature = "tracing")]
        tracing::warn!("Mutex was poisoned, recovering data");
        
        // 非 tracing 环境使用 eprintln
        #[cfg(not(feature = "tracing"))]
        eprintln!("[WARN] Mutex was poisoned, recovering data");
        
        poisoned.into_inner()
    })
}

/// 使用指定策略获取 Mutex 锁
///
/// # 参数
///
/// - `mutex`: 要获取锁的 Mutex 引用
/// - `strategy`: 锁获取策略
///
/// # 返回
///
/// 根据策略返回锁守卫或错误
///
/// # 策略说明
///
/// - `Panic`: 毒化时 panic（等同于 `lock().unwrap()`）
/// - `Recover`: 毒化时恢复数据
/// - `Error`: 毒化时返回错误
pub fn lock_with_strategy<T>(
    mutex: &Mutex<T>,
    strategy: LockStrategy,
) -> Result<MutexGuard<'_, T>, PoisonError<MutexGuard<'_, T>>> {
    match strategy {
        LockStrategy::Panic => Ok(mutex.lock().expect("Mutex poisoned")),
        LockStrategy::Recover => Ok(lock_or_recover(mutex)),
        LockStrategy::Error => mutex.lock(),
    }
}

/// 尝试获取 Mutex 锁，毒化时恢复
///
/// 非阻塞版本，如果锁不可用立即返回 None。
#[inline]
pub fn try_lock_or_recover<T>(mutex: &Mutex<T>) -> Option<MutexGuard<'_, T>> {
    match mutex.try_lock() {
        Ok(guard) => Some(guard),
        Err(std::sync::TryLockError::Poisoned(poisoned)) => {
            #[cfg(feature = "tracing")]
            tracing::warn!("Mutex was poisoned during try_lock, recovering");
            #[cfg(not(feature = "tracing"))]
            eprintln!("[WARN] Mutex was poisoned during try_lock, recovering");
            
            Some(poisoned.into_inner())
        }
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

// ============================================================================
// RwLock 辅助函数
// ============================================================================

/// 安全获取 RwLock 读锁，毒化时恢复
#[inline]
pub fn read_or_recover<T>(rwlock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    rwlock.read().unwrap_or_else(|poisoned| {
        #[cfg(feature = "tracing")]
        tracing::warn!("RwLock was poisoned (read), recovering");
        #[cfg(not(feature = "tracing"))]
        eprintln!("[WARN] RwLock was poisoned (read), recovering");
        
        poisoned.into_inner()
    })
}

/// 安全获取 RwLock 写锁，毒化时恢复
#[inline]
pub fn write_or_recover<T>(rwlock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    rwlock.write().unwrap_or_else(|poisoned| {
        #[cfg(feature = "tracing")]
        tracing::warn!("RwLock was poisoned (write), recovering");
        #[cfg(not(feature = "tracing"))]
        eprintln!("[WARN] RwLock was poisoned (write), recovering");
        
        poisoned.into_inner()
    })
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_lock_or_recover_normal() {
        let mutex = Mutex::new(42);
        let guard = lock_or_recover(&mutex);
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_lock_or_recover_poisoned() {
        let mutex = Arc::new(Mutex::new(42));
        let mutex_clone = mutex.clone();

        // 在另一个线程中故意 panic 来毒化 mutex
        let handle = thread::spawn(move || {
            let _guard = mutex_clone.lock().unwrap();
            std::panic::panic_any("intentional panic to poison mutex");
        });

        // 等待线程结束（会 panic）
        let _ = handle.join();

        // 验证 mutex 被毒化
        assert!(mutex.lock().is_err());

        // 使用 lock_or_recover 应该能恢复
        let guard = lock_or_recover(&mutex);
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_try_lock_or_recover() {
        let mutex = Mutex::new(100);
        
        // 正常情况
        let guard = try_lock_or_recover(&mutex);
        assert!(guard.is_some());
        assert_eq!(*guard.unwrap(), 100);
    }

    #[test]
    fn test_lock_with_strategy_recover() {
        let mutex = Mutex::new(200);
        let result = lock_with_strategy(&mutex, LockStrategy::Recover);
        assert!(result.is_ok());
        assert_eq!(*result.unwrap(), 200);
    }

    #[test]
    fn test_rwlock_helpers() {
        let rwlock = RwLock::new(300);
        
        let read_guard = read_or_recover(&rwlock);
        assert_eq!(*read_guard, 300);
        drop(read_guard);
        
        let mut write_guard = write_or_recover(&rwlock);
        *write_guard = 400;
        drop(write_guard);
        
        let read_guard = read_or_recover(&rwlock);
        assert_eq!(*read_guard, 400);
    }
}
