//! 配置转换桥梁（Layer 4 → Layer 3）
//!
//! 本模块提供从应用层配置（mh_config::SolverConfig）到引擎层
//! 泛型配置的显式转换，确保类型安全和精度一致性。

use mh_config::SolverConfig as Layer4Config;
use crate::types::NumericalParams;
use crate::engine::solver::{NumericalScheme, FallbackStrategy, TimeIntegrator, StabilityOptions};
use mh_runtime::RuntimeScalar;
use num_traits::FromPrimitive;

/// 配置转换错误
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigBridgeError {
    #[error("数值转换失败：字段'{field}'的值{value}超出目标类型范围")]
    ConversionFailed { field: &'static str, value: f64 },
    #[error("不支持的精度类型：{0:?}")]
    UnsupportedPrecision(mh_config::Precision),
}

/// Layer 3 内部配置（Backend 泛型）
///
/// 这是引擎层内部使用的配置结构，所有数值字段已转换为 Backend 标量类型。
#[derive(Debug, Clone)]
pub struct Layer3Config<S: RuntimeScalar> {
    /// 数值参数（泛型化）
    pub params: NumericalParams<S>,
    /// 重力加速度（Backend 标量）
    pub gravity: S,
    /// 是否启用静水重构
    pub use_hydrostatic_reconstruction: bool,
    /// 并行化阈值（面数）
    pub parallel_threshold: usize,
    /// 是否启用隐式摩擦
    pub implicit_friction: bool,
    /// 数值格式
    pub scheme: NumericalScheme,
    /// 回退策略
    pub fallback: FallbackStrategy,
    /// 稳定性检查选项
    pub stability: StabilityOptions,
    /// 最大回退次数
    pub max_fallback_attempts: u32,
    /// 时间步减小因子
    pub timestep_reduction_factor: S,
    /// 时间积分器类型
    pub integrator: TimeIntegrator,
}

impl<S> Layer3Config<S>
where
    S: RuntimeScalar + FromPrimitive,
{
    /// 从 Layer 4 配置转换到指定精度
    ///
    /// # 参数
    /// - `config`: Layer 4 配置（mh_config::SolverConfig）
    ///
    /// # 返回
    /// - `Ok(Self)`: 转换成功
    /// - `Err(ConfigBridgeError)`: 转换失败（数值溢出或无法转换）
    pub fn from_layer4(config: &Layer4Config) -> Result<Self, ConfigBridgeError> {
        // 转换数值参数
        let params = NumericalParams::<S>::from_f64_params(&config.numerical)
            .map_err(|_| ConfigBridgeError::ConversionFailed {
                field: "numerical_params", 
                value: 0.0,
            })?;

        // 转换重力加速度
        let gravity = S::from_f64(config.physics.gravity)
            .ok_or(ConfigBridgeError::ConversionFailed {
                field: "gravity",
                value: config.physics.gravity,
            })?;
        // 转换时间步减小因子
        let timestep_reduction_factor = S::from_f64(config.timestep_reduction_factor)
            .ok_or(ConfigBridgeError::ConversionFailed {
                field: "timestep_reduction_factor",
                value: config.timestep_reduction_factor,
            })?;

        // 判断是否为二阶格式
        let second_order = matches!(
            config.scheme,
            mh_config::RiemannSolverType::Hllc | mh_config::RiemannSolverType::Roe
        );

        Ok(Self {
            params,
            gravity,
            use_hydrostatic_reconstruction: config.use_hydrostatic_reconstruction,
            parallel_threshold: config.parallel_threshold,
            implicit_friction: config.implicit_friction,
            scheme: config.scheme.into(),
            fallback: config.fallback.into(),
            stability: StabilityOptions {
                check_nan: config.stability.check_nan,
                check_negative_depth: config.stability.check_negative_depth,
                check_extreme_velocity: config.stability.check_extreme_velocity,
                velocity_limit: config.stability.velocity_limit,
                depth_limit: config.stability.depth_limit,
            },
            max_fallback_attempts: config.max_fallback_attempts,
            timestep_reduction_factor,
            integrator: config.time_integration.into(),
            second_order,
        })
    }
}

// 转换 trait 实现
impl From<mh_config::RiemannSolverType> for NumericalScheme {
    fn from(value: mh_config::RiemannSolverType) -> Self {
        match value {
            mh_config::RiemannSolverType::Hllc => NumericalScheme::SecondOrderMuscl,
            mh_config::RiemannSolverType::Roe => NumericalScheme::SecondOrderMuscl,
            mh_config::RiemannSolverType::Rusanov => NumericalScheme::FirstOrder,
            mh_config::RiemannSolverType::Central => NumericalScheme::FirstOrder,
        }
    }
}

impl From<mh_config::FallbackStrategy> for FallbackStrategy {
    fn from(value: mh_config::FallbackStrategy) -> Self {
        match value {
            mh_config::FallbackStrategy::NoFallback => FallbackStrategy::NoFallback,
            mh_config::FallbackStrategy::FallbackToFirstOrder => FallbackStrategy::FallbackToFirstOrder,
            mh_config::FallbackStrategy::ReduceTimestep => FallbackStrategy::ReduceTimestep,
            mh_config::FallbackStrategy::Progressive => FallbackStrategy::Progressive,
        }
    }
}

impl From<mh_config::TimeIntegrationMethod> for TimeIntegrator {
    fn from(value: mh_config::TimeIntegrationMethod) -> Self {
        match value {
            mh_config::TimeIntegrationMethod::ForwardEuler => TimeIntegrator::Explicit,
            mh_config::TimeIntegrationMethod::SspRk2 => TimeIntegrator::Explicit,
            mh_config::TimeIntegrationMethod::SspRk3 => TimeIntegrator::Explicit,
        }
    }
}