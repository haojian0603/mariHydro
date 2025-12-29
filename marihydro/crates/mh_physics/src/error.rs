// mh_physics/src/error.rs

//! 物理求解器统一错误类型
//!
//! 提供 mh_physics 模块的统一错误处理框架。
//!
//! # 设计目标
//!
//! - 统一所有物理相关错误类型
//! - 提供丰富的错误上下文信息
//! - 支持错误链和原因追踪
//! - 便于调试和诊断
//!
//! # 示例
//!
//! ```ignore
//! use mh_physics::error::{PhysicsError, PhysicsResult};
//!
//! fn compute_flux(h: f64) -> PhysicsResult<f64> {
//!     if h < 0.0 {
//!         return Err(PhysicsError::NonPhysical {
//!             description: "负水深".to_string(),
//!         });
//!     }
//!     Ok(h * h * 9.81)
//! }
//! ```

use thiserror::Error;

// ============================================================================
// 主错误类型
// ============================================================================

/// 物理求解器错误
///
/// 涵盖所有物理计算过程中可能发生的错误类型。
#[derive(Error, Debug)]
pub enum PhysicsError {
    // === 配置错误 ===
    
    /// 无效参数
    ///
    /// 当输入参数超出有效范围或不满足约束时抛出。
    #[error("无效参数 '{name}': 值={value}, 原因={reason}")]
    InvalidParameter {
        /// 参数名称
        name: &'static str,
        /// 参数值
        value: f64,
        /// 无效原因
        reason: &'static str,
    },

    /// 配置错误
    #[error("配置错误: {message}")]
    Configuration {
        /// 错误消息
        message: String,
    },

    // === 数值错误 ===

    /// 数值溢出
    ///
    /// 当计算结果超出表示范围时抛出。
    #[error("数值溢出: {context}")]
    NumericalOverflow {
        /// 溢出发生的上下文描述
        context: String,
    },

    /// 数值不收敛
    ///
    /// 迭代求解器未能在指定次数内收敛。
    #[error("数值不收敛: 迭代次数={iterations}, 残差={residual}")]
    NotConverged {
        /// 已执行的迭代次数
        iterations: usize,
        /// 最终残差
        residual: f64,
    },

    /// 非物理状态
    ///
    /// 计算结果违反物理约束（如负水深、负能量等）。
    #[error("非物理状态: {description}")]
    NonPhysical {
        /// 非物理状态的描述
        description: String,
    },

    /// 除零错误
    #[error("除零错误: {context}")]
    DivisionByZero {
        /// 发生除零的上下文
        context: String,
    },

    // === 网格错误 ===

    /// 网格错误
    #[error("网格错误: {message}")]
    MeshError {
        /// 错误消息
        message: String,
    },

    /// 无效索引
    #[error("无效索引: {index_type} 索引 {index} 超出范围 [0, {max})")]
    InvalidIndex {
        /// 索引类型（如 "单元"、"面"、"节点"）
        index_type: &'static str,
        /// 无效索引值
        index: usize,
        /// 最大有效值
        max: usize,
    },

    // === 边界错误 ===

    /// 边界条件错误
    #[error("边界条件错误: {message}")]
    BoundaryError {
        /// 错误消息
        message: String,
    },

    /// 缺少边界数据
    #[error("缺少边界数据: 边界 '{boundary_name}' 在时间 {time} 处没有强迫数据")]
    MissingBoundaryData {
        /// 边界名称
        boundary_name: String,
        /// 查询时间
        time: f64,
    },

    // === 求解器错误 ===

    /// 求解器失败
    #[error("求解器失败: 阶段={stage}, 消息={message}")]
    SolverFailed {
        /// 失败阶段
        stage: &'static str,
        /// 详细消息
        message: String,
    },

    /// CFL 条件违反
    #[error("CFL 条件违反: CFL={cfl:.4} > 允许值={max_cfl:.4}")]
    CflViolation {
        /// 当前 CFL 数
        cfl: f64,
        /// 允许的最大 CFL 数
        max_cfl: f64,
    },

    /// 时间步长过小
    #[error("时间步长过小: dt={dt:.2e} < 最小值={min_dt:.2e}")]
    TimestepTooSmall {
        /// 当前时间步长
        dt: f64,
        /// 最小允许值
        min_dt: f64,
    },

    // === IO 错误 ===

    /// IO 错误
    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    // === 守恒性错误 ===

    /// 守恒性违反
    #[error("守恒性违反: {quantity} 变化 {change:.4e} 超过容差 {tolerance:.4e}")]
    ConservationViolation {
        /// 守恒量名称
        quantity: &'static str,
        /// 变化量
        change: f64,
        /// 容差
        tolerance: f64,
    },

    /// 能量增加（非物理）
    #[error("能量非物理增加: 变化前={before:.4e}, 变化后={after:.4e}, 相对增加={relative_increase:.4e}")]
    EnergyIncreased {
        /// 变化前的总能量
        before: f64,
        /// 变化后的总能量
        after: f64,
        /// 相对增加量
        relative_increase: f64,
    },

    // === 并发错误 ===

    /// 锁获取失败
    #[error("锁获取失败: {resource}")]
    LockFailed {
        /// 资源名称
        resource: String,
    },

    // === 其他错误 ===

    /// 未实现功能
    #[error("未实现: {feature}")]
    NotImplemented {
        /// 未实现的功能描述
        feature: String,
    },

    /// 内部错误
    #[error("内部错误: {message}")]
    Internal {
        /// 错误消息
        message: String,
    },
}

// ============================================================================
// 结果类型别名
// ============================================================================

/// 物理计算结果类型
pub type PhysicsResult<T> = Result<T, PhysicsError>;

// ============================================================================
// 辅助实现
// ============================================================================

impl PhysicsError {
    /// 创建无效参数错误
    pub fn invalid_param(name: &'static str, value: f64, reason: &'static str) -> Self {
        Self::InvalidParameter { name, value, reason }
    }

    /// 创建非物理状态错误
    pub fn non_physical(description: impl Into<String>) -> Self {
        Self::NonPhysical {
            description: description.into(),
        }
    }

    /// 创建数值溢出错误
    pub fn overflow(context: impl Into<String>) -> Self {
        Self::NumericalOverflow {
            context: context.into(),
        }
    }

    /// 创建不收敛错误
    pub fn not_converged(iterations: usize, residual: f64) -> Self {
        Self::NotConverged { iterations, residual }
    }

    /// 创建求解器失败错误
    pub fn solver_failed(stage: &'static str, message: impl Into<String>) -> Self {
        Self::SolverFailed {
            stage,
            message: message.into(),
        }
    }

    /// 创建除零错误
    pub fn div_by_zero(context: impl Into<String>) -> Self {
        Self::DivisionByZero {
            context: context.into(),
        }
    }

    /// 创建无效索引错误
    pub fn invalid_index(index_type: &'static str, index: usize, max: usize) -> Self {
        Self::InvalidIndex { index_type, index, max }
    }

    /// 判断是否为可恢复错误
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::CflViolation { .. }
                | Self::NotConverged { .. }
                | Self::TimestepTooSmall { .. }
        )
    }

    /// 判断是否为严重错误
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            Self::NonPhysical { .. }
                | Self::NumericalOverflow { .. }
                | Self::Internal { .. }
        )
    }
}

// ============================================================================
// 从其他错误类型转换
// ============================================================================

impl From<mh_foundation::MhError> for PhysicsError {
    fn from(err: mh_foundation::MhError) -> Self {
        Self::Internal {
            message: err.to_string(),
        }
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_parameter() {
        let err = PhysicsError::invalid_param("dt", -0.001, "必须为正值");
        assert!(err.to_string().contains("dt"));
        assert!(err.to_string().contains("-0.001"));
    }

    #[test]
    fn test_not_converged() {
        let err = PhysicsError::not_converged(100, 1e-5);
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("不收敛"));
    }

    #[test]
    fn test_is_recoverable() {
        let recoverable = PhysicsError::CflViolation { cfl: 1.5, max_cfl: 1.0 };
        let critical = PhysicsError::non_physical("负水深");

        assert!(recoverable.is_recoverable());
        assert!(!critical.is_recoverable());
        assert!(critical.is_critical());
    }

    #[test]
    fn test_error_chain() {
        fn inner() -> PhysicsResult<()> {
            Err(PhysicsError::div_by_zero("水深为零"))
        }

        fn outer() -> PhysicsResult<()> {
            inner()?;
            Ok(())
        }

        let result = outer();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("除零"));
    }
}
