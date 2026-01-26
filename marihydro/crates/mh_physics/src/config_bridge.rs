// crates/mh_physics/src/config_bridge.rs
//! 配置转换桥梁（Layer 4 → Layer 3）
//!
//! 本模块提供从应用层配置（mh_config::SolverConfig）到引擎层
//! 泛型配置的显式转换，确保类型安全和精度一致性。
//!
//! # 精度转换说明
//!
//! 从f64转换到f32时，由于IEEE 754二进制浮点表示限制，无法保证完全相等。
//! 例如：9.81 → f32 → f64后为9.8100004196167。测试中使用epsilon比较。

use mh_config::SolverConfig as Layer4Config;
use mh_config::solver_config::{RiemannSolverType, TimeIntegrationMethod};
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
    /// 黎曼求解器类型
    pub riemann_solver: RiemannSolverType,
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

impl<S> Default for Layer3Config<S>
where
    S: RuntimeScalar + FromPrimitive,
{
    /// 创建默认配置，用于测试和简单场景
    fn default() -> Self {
        Self {
            params: NumericalParams::<S>::default(),
            gravity: S::from_f64(9.81).unwrap_or_else(|| S::ZERO),
            use_hydrostatic_reconstruction: true,
            parallel_threshold: 1000,
            implicit_friction: true,
            riemann_solver: RiemannSolverType::Hllc,
            scheme: NumericalScheme::SecondOrderMuscl,
            fallback: FallbackStrategy::default(),
            stability: StabilityOptions::default(),
            max_fallback_attempts: 3,
            timestep_reduction_factor: S::from_f64(0.5).unwrap_or_else(|| S::ZERO),
            integrator: TimeIntegrator::Explicit,
        }
    }
}

/// Layer3Config 构建器
#[derive(Debug, Clone)]
pub struct Layer3ConfigBuilder<S: RuntimeScalar> {
    config: Layer3Config<S>,
}

impl<S> Layer3ConfigBuilder<S>
where
    S: RuntimeScalar + FromPrimitive,
{
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            config: Layer3Config::default(),
        }
    }

    /// 设置数值格式
    pub fn scheme(mut self, scheme: NumericalScheme) -> Self {
        self.config.scheme = scheme;
        self
    }

    /// 设置黎曼求解器类型
    pub fn riemann_solver(mut self, solver: RiemannSolverType) -> Self {
        self.config.riemann_solver = solver;
        self
    }

    /// 设置 CFL 数
    pub fn cfl(mut self, cfl: f64) -> Self {
        if let Some(cfl_s) = S::from_f64(cfl) {
            self.config.params.cfl = cfl_s;
        }
        self
    }

    /// 设置重力加速度
    pub fn gravity(mut self, g: f64) -> Self {
        if let Some(g_s) = S::from_f64(g) {
            self.config.gravity = g_s;
        }
        self
    }

    /// 设置数值参数
    pub fn params(mut self, params: NumericalParams<S>) -> Self {
        self.config.params = params;
        self
    }

    /// 设置静水重构
    pub fn use_hydrostatic_reconstruction(mut self, value: bool) -> Self {
        self.config.use_hydrostatic_reconstruction = value;
        self
    }

    /// 设置并行阈值
    pub fn parallel_threshold(mut self, value: usize) -> Self {
        self.config.parallel_threshold = value;
        self
    }

    /// 设置隐式摩擦
    pub fn implicit_friction(mut self, value: bool) -> Self {
        self.config.implicit_friction = value;
        self
    }

    /// 设置NaN检测
    pub fn nan_detection_enabled(mut self, enabled: bool) -> Self {
        self.config.stability.check_nan = enabled;
        self
    }

    /// 设置稳定性选项
    pub fn stability_options(mut self, options: StabilityOptions) -> Self {
        self.config.stability = options;
        self
    }

    /// 构建配置
    pub fn build(self) -> Layer3Config<S> {
        self.config
    }
}

impl<S> Default for Layer3ConfigBuilder<S>
where
    S: RuntimeScalar + FromPrimitive,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Layer3Config<S>
where
    S: RuntimeScalar + FromPrimitive,
{
    /// 创建配置构建器
    pub fn builder() -> Layer3ConfigBuilder<S> {
        Layer3ConfigBuilder::new()
    }

    /// 从 Layer 4 配置转换到指定精度
    ///
    /// # 参数
    /// - `config`: Layer 4 配置（mh_config::SolverConfig）
    ///
    /// # 返回
    /// - `Ok(Self)`: 转换成功
    /// - `Err(ConfigBridgeError)`: 转换失败（数值溢出或无法转换）
    pub fn from_layer4(config: &Layer4Config) -> Result<Self, ConfigBridgeError> {
        let params_f64 = NumericalParams::<f64> {
            h_min: config.physics.h_min,
            h_dry: config.physics.h_dry,
            cfl: config.physics.cfl,
            vel_max: config.physics.velocity_cap,
            flux_eps: config.physics.flux_eps,
            min_wave_speed: config.physics.min_wave_speed,
            ..NumericalParams::<f64>::default()
        };

        let params = NumericalParams::<S>::from_f64_params(&params_f64)
            .map_err(|_| ConfigBridgeError::ConversionFailed {
                field: "numerical_params", 
                value: 0.0,
            })?;

        let gravity = S::from_f64(config.physics.gravity)
            .ok_or(ConfigBridgeError::ConversionFailed {
                field: "gravity",
                value: config.physics.gravity,
            })?;
        let timestep_reduction_factor = S::from_f64(config.numerical.timestep_reduction_factor)
            .ok_or(ConfigBridgeError::ConversionFailed {
                field: "timestep_reduction_factor",
                value: config.numerical.timestep_reduction_factor,
            })?;

        let scheme = match config.numerical.riemann_solver {
            RiemannSolverType::Hllc => NumericalScheme::SecondOrderMuscl,
            RiemannSolverType::Roe => NumericalScheme::SecondOrderMuscl,
            RiemannSolverType::Rusanov => NumericalScheme::FirstOrder,
            RiemannSolverType::Central => NumericalScheme::FirstOrder,
        };

        let integrator = match config.numerical.time_integration {
            TimeIntegrationMethod::ForwardEuler => TimeIntegrator::Explicit,
            TimeIntegrationMethod::SspRk2 => TimeIntegrator::Explicit,
            TimeIntegrationMethod::SspRk3 => TimeIntegrator::Explicit,
        };

        Ok(Self {
            params,
            gravity,
            use_hydrostatic_reconstruction: config.numerical.use_hydrostatic_reconstruction,
            parallel_threshold: config.parallel.threshold,
            implicit_friction: config.numerical.friction,
            riemann_solver: config.numerical.riemann_solver,
            scheme,
            fallback: FallbackStrategy::default(),
            stability: StabilityOptions::default(),
            max_fallback_attempts: config.numerical.max_fallback_attempts,
            timestep_reduction_factor,
            integrator,
        })
    }
}

impl From<RiemannSolverType> for NumericalScheme {
    fn from(value: RiemannSolverType) -> Self {
        match value {
            RiemannSolverType::Hllc => NumericalScheme::SecondOrderMuscl,
            RiemannSolverType::Roe => NumericalScheme::SecondOrderMuscl,
            RiemannSolverType::Rusanov => NumericalScheme::FirstOrder,
            RiemannSolverType::Central => NumericalScheme::FirstOrder,
        }
    }
}

impl From<TimeIntegrationMethod> for TimeIntegrator {
    fn from(value: TimeIntegrationMethod) -> Self {
        match value {
            TimeIntegrationMethod::ForwardEuler => TimeIntegrator::Explicit,
            TimeIntegrationMethod::SspRk2 => TimeIntegrator::Explicit,
            TimeIntegrationMethod::SspRk3 => TimeIntegrator::Explicit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_config::SolverConfig;
    use num_traits::ToPrimitive;

    /// f64精度容差（相对误差约1e-15）
    const EPSILON_F64: f64 = 1e-10;

    #[test]
    fn test_from_layer4_f64() {
        let layer4 = SolverConfig::default();
        let layer3: Layer3Config<f64> = Layer3Config::from_layer4(&layer4).unwrap();
        
        // 使用epsilon比较，容忍f64-f64转换的微小误差
        assert!((layer3.gravity.to_f64().unwrap() - layer4.physics.gravity).abs() < EPSILON_F64);
        assert!((layer3.params.cfl.to_f64().unwrap() - layer4.physics.cfl).abs() < EPSILON_F64);
    }

    #[test]
    fn test_from_layer4_f32() {
        let layer4 = SolverConfig::default();
        let layer3: Layer3Config<f32> = Layer3Config::from_layer4(&layer4).unwrap();
        
        // f32转换后应有合理精度误差，不能期望完全相等
        let expected_gravity = f32::from_f64(layer4.physics.gravity).unwrap();
        assert!((layer3.gravity.to_f64().unwrap() - expected_gravity.to_f64().unwrap()).abs() < EPSILON_F64);
        
        let expected_cfl = f32::from_f64(layer4.physics.cfl).unwrap();
        assert!((layer3.params.cfl.to_f64().unwrap() - expected_cfl.to_f64().unwrap()).abs() < EPSILON_F64);
    }

    #[test]
    fn test_riemann_solver_conversion() {
        assert!(matches!(
            NumericalScheme::from(RiemannSolverType::Hllc),
            NumericalScheme::SecondOrderMuscl
        ));
        assert!(matches!(
            NumericalScheme::from(RiemannSolverType::Rusanov),
            NumericalScheme::FirstOrder
        ));
    }

    #[test]
    fn test_time_integration_conversion() {
        assert!(matches!(
            TimeIntegrator::from(TimeIntegrationMethod::SspRk3),
            TimeIntegrator::Explicit
        ));
    }
}