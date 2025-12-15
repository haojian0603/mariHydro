// crates/mh_foundation/src/error.rs
//! 基础错误类型
//!
//! 定义整个项目的基础错误类型，仅包含与基础设施相关的错误。
//! 本模块是 Layer 1 的核心组件，禁止引入任何运行时或业务领域概念。
//!
//! # 设计原则
//!
//! 1. **纯净性**：仅包含 IO、索引、内存等基础错误，无投影、网格、计算等高层概念
//! 2. **可转换性**：所有高层错误最终可转换为 [`MhError::Internal`] 或具体的基础错误
//! 3. **零依赖**：不依赖项目内其他 crate，可独立使用
//! 4. **易用性**：提供丰富的便捷构造函数和类型转换实现
//!
//! # 错误分层
//!
//! ```text
//! 高层错误 (mh_physics, mh_mesh, mh_geo)
//!        ↓ (转换)
//! 运行时错误 (mh_runtime::RuntimeError)
//!        ↓ (转换)
//! 配置错误 (mh_config::ConfigError)
//!        ↓ (转换)
//! 基础错误 ← 你在这里 (mh_foundation::MhError)
//! ```
//!
//! # 转换示例
//!
//! ```
//! # use mh_foundation::error::MhError;
//! // 高层错误（如网格错误）会转换为 MhError::Internal
//! let mesh_error = MhError::invalid_input("非流形边检测到");
//! let base_error: MhError = mesh_error; // 已经是基础错误
//! assert!(matches!(base_error, MhError::InvalidInput { .. }));
//! ```

use std::{fmt, io, path::PathBuf, sync::PoisonError, sync::mpsc::SendError};

/// 统一结果类型别名
///
/// 用于简化函数签名，等价于 `Result<T, MhError>`。
///
/// # 示例
///
/// ```
/// use mh_foundation::error::MhResult;
///
/// fn read_data() -> MhResult<Vec<f64>> {
///     // ...
/// #   Ok(vec![])
/// }
/// ```
pub type MhResult<T> = Result<T, MhError>;

/// Foundation 层基础错误
///
/// 包含所有基础设施级别的错误，是错误体系的根基。
/// 高层错误必须通过转换为 [`MhError::Internal`] 来向下兼容。
///
/// # 错误分类
///
/// - **IO 错误**：文件、网络等输入输出操作失败
/// - **索引错误**：越界、类型不匹配等内存访问问题
/// - **资源错误**：锁、通道等并发原语失败
/// - **逻辑错误**：无效输入、未实现、未找到等业务无关错误
#[derive(Debug)]
pub enum MhError {
    /// IO 操作失败
    ///
    /// 包含文件读写、网络通信等底层失败信息。
    ///
    /// # 字段
    ///
    /// - `message`：人类可读的错误描述
    /// - `source`：可选的底层 [`std::io::Error`]
    ///
    /// # 示例
    ///
    /// ```
    /// # use mh_foundation::error::MhError;
    /// let err = MhError::io("磁盘已满");
    /// assert!(err.to_string().contains("IO错误"));
    /// ```
    Io {
        /// 描述性错误信息
        message: String,
        /// 底层 IO 错误源
        source: Option<io::Error>,
    },

    /// 文件不存在
    ///
    /// 当尝试访问不存在的文件路径时返回。
    ///
    /// # 字段
    ///
    /// - `path`：请求的文件路径
    FileNotFound {
        /// 未找到的文件路径
        path: PathBuf,
    },

    /// 数组或集合大小不匹配
    ///
    /// 在需要严格大小一致性的操作中触发（如向量相加）。
    ///
    /// # 示例
    ///
    /// ```
    /// # use mh_foundation::error::{MhError, MhResult};
    /// fn add_vectors(a: &[f64], b: &[f64]) -> MhResult<Vec<f64>> {
    ///     if a.len() != b.len() {
    ///         return Err(MhError::size_mismatch("vectors", a.len(), b.len()));
    ///     }
    ///     Ok(a.iter().zip(b).map(|(x, y)| x + y).collect())
    /// }
    /// ```
    SizeMismatch {
        /// 数据名称（用于调试）
        name: &'static str,
        /// 期望大小
        expected: usize,
        /// 实际大小
        actual: usize,
    },

    /// 索引访问越界
    ///
    /// 当索引值大于等于容器长度时触发。
    IndexOutOfBounds {
        /// 索引类别（如 "Cell", "Node"）
        index_type: &'static str,
        /// 访问的索引值
        index: usize,
        /// 容器长度（上界）
        len: usize,
    },

    /// 输入数据验证失败
    ///
    /// 用于参数校验、前置条件检查等场景。
    InvalidInput {
        /// 说明无效原因
        message: String,
    },

    /// 内部实现错误
    ///
    /// 当程序进入不应到达的状态时使用。
    /// 通常表示 bug 或不变量被破坏。
    Internal {
        /// 内部错误描述
        message: String,
    },

    /// 功能或资源未找到
    ///
    /// 用于注册表、工厂模式等资源查找失败场景。
    NotFound {
        /// 资源名称或标识
        resource: String,
    },

    /// 功能未实现
    ///
    /// 用于占位符或条件编译场景。
    NotImplemented {
        /// 未实现的功能描述
        feature: String,
    },

    /// 锁获取失败
    ///
    /// 通常由于锁被 poisoned（线程 panic 导致）。
    LockError {
        /// 失败的资源名称
        resource: String,
    },

    /// 通道发送失败
    ///
    /// 当接收端已关闭时触发。
    ChannelSendError,
}

// ============================================================================
// 便捷构造方法
// ============================================================================

impl MhError {
    /// 创建 IO 错误
    ///
    /// # 参数
    ///
    /// - `message`：可读的错误描述
    ///
    /// # 示例
    ///
    /// ```
    /// # use mh_foundation::error::MhError;
    /// let err = MhError::io("磁盘已满");
    /// assert!(err.to_string().contains("IO错误"));
    /// ```
    #[inline]
    pub fn io(message: impl Into<String>) -> Self {
        Self::Io {
            message: message.into(),
            source: None,
        }
    }

    /// 创建带源的 IO 错误
    ///
    /// 当需要保留底层 [`std::io::Error`] 时使用。
    #[inline]
    pub fn io_with_source(message: impl Into<String>, source: io::Error) -> Self {
        Self::Io {
            message: message.into(),
            source: Some(source),
        }
    }

    /// 创建文件未找到错误
    ///
    /// # 参数
    ///
    /// - `path`：请求的文件路径（任何可转换为 [`PathBuf`] 的类型）
    #[inline]
    pub fn file_not_found(path: impl Into<PathBuf>) -> Self {
        Self::FileNotFound { path: path.into() }
    }

    /// 创建大小不匹配错误
    ///
    /// # 参数
    ///
    /// - `name`：数据名称（用于调试）
    /// - `expected`：期望大小
    /// - `actual`：实际大小
    #[inline]
    pub fn size_mismatch(name: &'static str, expected: usize, actual: usize) -> Self {
        Self::SizeMismatch {
            name,
            expected,
            actual,
        }
    }

    /// 创建索引越界错误
    #[inline]
    pub fn index_out_of_bounds(index_type: &'static str, index: usize, len: usize) -> Self {
        Self::IndexOutOfBounds {
            index_type,
            index,
            len,
        }
    }

    /// 创建无效输入错误
    #[inline]
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// 创建内部错误
    #[inline]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// 创建资源未找到错误
    #[inline]
    pub fn not_found(resource: impl Into<String>) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }

    /// 创建功能未实现错误
    #[inline]
    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Self::NotImplemented {
            feature: feature.into(),
        }
    }

    /// 创建锁获取失败错误
    #[inline]
    pub fn lock_error(resource: impl Into<String>) -> Self {
        Self::LockError {
            resource: resource.into(),
        }
    }

    /// 检查条件，不满足则返回错误
    ///
    /// # 参数
    ///
    /// - `cond`：检查条件
    /// - `err`：条件失败时返回的错误
    ///
    /// # 示例
    ///
    /// ```
    /// # use mh_foundation::error::{MhError, MhResult};
    /// fn divide(a: f64, b: f64) -> MhResult<f64> {
    ///     mh_foundation::ensure!(b != 0.0, MhError::invalid_input("除数不能为零"));
    ///     Ok(a / b)
    /// }
    /// ```
    #[inline]
    pub fn ensure(cond: bool, err: Self) -> Result<(), Self> {
        if cond {
            Ok(())
        } else {
            Err(err)
        }
    }
}

// ============================================================================
// 验证辅助方法
// ============================================================================

impl MhError {
    /// 验证数组大小是否匹配，不匹配则返回 [`MhError::SizeMismatch`]
    ///
    /// # 示例
    ///
    /// ```
    /// # use mh_foundation::error::{MhError, MhResult};
    /// fn process(a: &[f64], b: &[f64]) -> MhResult<()> {
    ///     MhError::check_size("vectors", a.len(), b.len())?;
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn check_size(name: &'static str, expected: usize, actual: usize) -> MhResult<()> {
        if expected != actual {
            Err(Self::size_mismatch(name, expected, actual))
        } else {
            Ok(())
        }
    }

    /// 验证索引是否在有效范围内，越界则返回 [`MhError::IndexOutOfBounds`]
    #[inline]
    pub fn check_index(index_type: &'static str, index: usize, len: usize) -> MhResult<()> {
        if index >= len {
            Err(Self::index_out_of_bounds(index_type, index, len))
        } else {
            Ok(())
        }
    }
}

// ============================================================================
// 标准库错误转换实现
// ============================================================================

impl From<io::Error> for MhError {
    /// 将 [`std::io::Error`] 转换为 [`MhError::Io`]
    fn from(err: io::Error) -> Self {
        Self::io_with_source("IO 操作失败", err)
    }
}

impl<T> From<PoisonError<T>> for MhError {
    /// 将 [`std::sync::PoisonError`] 转换为 [`MhError::LockError`]
    fn from(_: PoisonError<T>) -> Self {
        Self::lock_error("mutex")
    }
}

impl<T> From<SendError<T>> for MhError {
    /// 将 [`std::sync::mpsc::SendError`] 转换为 [`MhError::ChannelSendError`]
    fn from(_: SendError<T>) -> Self {
        Self::ChannelSendError
    }
}

// ============================================================================
// 核心 Trait 实现
// ============================================================================

impl fmt::Display for MhError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { message, .. } => {
                write!(f, "IO错误: {}", message)
            }
            Self::FileNotFound { path } => {
                write!(f, "文件不存在: {}", path.display())
            }
            Self::SizeMismatch {
                name,
                expected,
                actual,
            } => {
                write!(f, "数组大小不匹配: {} 期望{}, 实际{}", name, expected, actual)
            }
            Self::IndexOutOfBounds {
                index_type,
                index,
                len,
            } => {
                write!(f, "索引越界: {} 索引{} 超出范围 0..{}", index_type, index, len)
            }
            Self::InvalidInput { message } => write!(f, "无效的输入数据: {}", message),
            Self::Internal { message } => write!(f, "内部错误: {}", message),
            Self::NotFound { resource } => write!(f, "资源未找到: {}", resource),
            Self::NotImplemented { feature } => write!(f, "功能未实现: {}", feature),
            Self::LockError { resource } => write!(f, "锁获取失败: {}", resource),
            Self::ChannelSendError => write!(f, "通道发送失败"),
        }
    }
}

impl std::error::Error for MhError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => source.as_ref().map(|e| e as _),
            _ => None,
        }
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as StdError;

    #[test]
    fn test_error_display() {
        let err = MhError::io("磁盘已满");
        assert!(err.to_string().contains("IO错误"));
    }

    #[test]
    fn test_io_error_with_source() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "文件未找到");
        let err = MhError::io_with_source("读取配置失败", io_err);
        assert!(err.to_string().contains("读取配置失败"));
        assert!(err.source().is_some());
    }

    #[test]
    fn test_file_not_found() {
        let err = MhError::file_not_found("/path/to/config.json");
        assert!(err.to_string().contains("/path/to/config.json"));
    }

    #[test]
    fn test_size_mismatch() {
        let err = MhError::size_mismatch("velocity", 100, 50);
        assert!(err.to_string().contains("velocity"));
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
    }

    #[test]
    fn test_check_size_success() {
        assert!(MhError::check_size("test", 10, 10).is_ok());
    }

    #[test]
    fn test_check_size_failure() {
        let result = MhError::check_size("test", 10, 5);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MhError::SizeMismatch { .. }));
    }

    #[test]
    fn test_check_index_success() {
        assert!(MhError::check_index("Cell", 5, 10).is_ok());
    }

    #[test]
    fn test_check_index_failure() {
        let result = MhError::check_index("Cell", 10, 10);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MhError::IndexOutOfBounds { .. }));
    }

    #[test]
    fn test_ensure_macro_success() {
        let result = MhError::ensure(true, MhError::invalid_input("不应失败"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_ensure_macro_failure() {
        let result = MhError::ensure(false, MhError::invalid_input("条件失败"));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MhError::InvalidInput { .. }));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "无权限");
        let mh_err: MhError = io_err.into();
        assert!(matches!(mh_err, MhError::Io { .. }));
    }

    #[test]
    fn test_poison_error_conversion() {
        use std::sync::Mutex;
        let lock = Mutex::new(0);
        
        // 在闭包中持有锁并 panic，使锁被 poison
        let _ = std::panic::catch_unwind(|| {
            let _g = lock.lock().unwrap();
            panic!("poison");
        });
        
        // 现在主线程尝试获取锁会得到 PoisonError
        let poison_err = lock.lock().unwrap_err();
        let mh_err: MhError = poison_err.into();
        assert!(matches!(mh_err, MhError::LockError { .. }));
    }

    #[test]
    fn test_mh_result_type() {
        fn success() -> MhResult<i32> {
            Ok(42)
        }
        fn failure() -> MhResult<i32> {
            Err(MhError::not_found("resource"))
        }

        assert!(success().is_ok());
        assert_eq!(success().unwrap(), 42);
        assert!(failure().is_err());
    }
}