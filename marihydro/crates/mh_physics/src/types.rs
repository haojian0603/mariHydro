// crates/mh_physics/src/types.rs

//! 物理计算核心类型定义
//!
//! 本模块提供物理求解器所需的类型系统，包括：
//! - **类型安全索引**：从`mh_runtime`重新导出的计算层索引
//! - **安全包装类型**：`SafeDepth<S>`和`SafeVelocity<S>`提供数值安全保护
//! - **泛型数值参数**：`NumericalParams<S>`通过泛型参数实现f32/f64运行时切换
//! - **物理常数**：`PhysicalConstants`保持f64（自然常数不随计算精度改变）
//! - **求解器配置**：`SolverConfig`作为Layer 4无泛型配置结构
//!
//! # 架构分层
//!
//! - **Layer 3 (Engine)**：`NumericalParams<S>`、`SafeVelocity<S>`、`SafeDepth<S>`全泛型化
//! - **Layer 4 (Config)**：`SolverConfig`保持f64，作为运行时配置接口
//! - **Layer 5 (App)**：不直接使用本模块泛型类型，通过`DynSolver`交互
//!
//! # 使用规范
//!
//! ```rust
//! // ✅ 正确：Layer 3引擎层使用泛型参数
//! fn compute_flux<S: RuntimeScalar>(h: S, u: S) -> S { h * u }
//!
//! // ❌ 错误：Layer 4/5不应直接使用RuntimeScalar约束
//! // fn app_level<S: RuntimeScalar>(config: SolverConfig) { ... }
//! ```

use crate::fields::{FieldMeta, FieldRegistry};
use bytemuck::Pod;
use num_traits::{Float, FromPrimitive};

// 从 mh_runtime 重新导出索引类型（公开）
pub use mh_runtime::{
    BoundaryIndex, CellIndex, EdgeIndex, FaceIndex, HalfEdgeIndex, LayerIndex, NodeIndex,
    RuntimeScalar, VertexIndex, INVALID_INDEX,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Mul, Sub};

// ============================================================
// 索引类型说明
// ============================================================

// 注：索引扩展trait已删除（T=1清理）
// 直接使用 CellIndex::get(), FaceIndex::get(), NodeIndex::get() 进行 usize 转换
// 直接使用 CellIndex::new(idx), FaceIndex::new(idx), NodeIndex::new(idx) 创建

// ============================================================
// 安全包装类型（泛型化改造）
// ============================================================

/// 安全水深（保证≥h_min）
///
/// 提供正则化处理，避免浅水方程中除以接近零的水深导致数值不稳定。
///
/// # 类型参数
/// - `S`: 运行时标量类型（f32或f64）
///
/// # 示例
/// ```
/// use mh_physics::types::SafeDepth;
/// use mh_runtime::RuntimeScalar;
///
/// let depth_f64 = SafeDepth::<f64>::new(1e-10, 1e-9);
/// assert!(depth_f64.get() >= 1e-9);
///
/// let depth_f32 = SafeDepth::<f32>::new(0.1, 1e-4);
/// assert_eq!(depth_f32.get(), 0.1);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SafeDepth<S: RuntimeScalar> {
    /// 安全处理后的水深值
    value: S,
    /// 最小水深阈值
    h_min: S,
}

impl<S: RuntimeScalar> SafeDepth<S> {
    /// 从原始水深创建安全水深
    ///
    /// # 参数
    /// - `h`: 原始水深值
    /// - `h_min`: 最小水深阈值，低于此值将使用h_min
    ///
    /// # 返回
    /// 包装后的SafeDepth，其value = max(h, h_min)
    #[inline]
    pub fn new(h: S, h_min: S) -> Self {
        Self {
            value: h.max(h_min),
            h_min,
        }
    }

    /// 获取安全水深值（用于除法等运算）
    #[inline]
    pub fn get(self) -> S {
        self.value
    }

    /// 获取h_min阈值
    #[inline]
    pub fn h_min(self) -> S {
        self.h_min
    }

    /// 判断原始水深是否干（低于干湿阈值）
    ///
    /// # 参数
    /// - `h`: 原始水深值
    /// - `h_dry`: 干燥判定阈值
    ///
    /// # 返回
    /// `true`表示干单元（h < h_dry）
    #[inline]
    pub fn is_originally_dry(h: S, h_dry: S) -> bool {
        h < h_dry
    }

    /// 安全除法运算
    ///
    /// 执行numerator / h，但使用max(h, h_min)作为分母保护
    ///
    /// # 参数
    /// - `numerator`: 分子值
    /// - `h`: 原始分母（水深）
    /// - `h_min`: 最小分母保护值
    ///
    /// # 返回
    /// numerator / max(h, h_min)
    #[inline]
    pub fn safe_divide(numerator: S, h: S, h_min: S) -> S {
        numerator / Self::new(h, h_min).get()
    }

    /// 计算安全速度分量
    ///
    /// 执行momentum / h，但使用max(h, h_min)作为分母
    #[inline]
    pub fn velocity_component(momentum: S, h: S, h_min: S) -> S {
        if h < h_min {
            S::ZERO
        } else {
            momentum / h.max(h_min)
        }
    }
}

impl<S: RuntimeScalar> Default for SafeDepth<S> {
    /// 使用S::Epsilon作为默认最小水深
    fn default() -> Self {
        Self {
            value: S::EPSILON,
            h_min: S::EPSILON,
        }
    }
}

/// 安全速度（避免除零导致的无穷大）
///
/// 从动量和水深计算速度时，自动处理干单元（h ≈ 0）情况。
///
/// # 类型参数
/// - `S`: 运行时标量类型（f32或f64）
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SafeVelocity<S: RuntimeScalar> {
    /// x方向速度 [m/s]
    pub u: S,
    /// y方向速度 [m/s]
    pub v: S,
}

impl<S: RuntimeScalar> SafeVelocity<S> {
    /// 零速度常量
    pub const ZERO: Self = Self { u: S::ZERO, v: S::ZERO };

    /// 从动量和水深计算安全速度
    ///
    /// # 参数
    /// - `hu`, `hv`: 动量分量 [m²/s]
    /// - `h`: 水深 [m]
    /// - `h_dry`: 干湿判断阈值
    /// - `h_min`: 最小计算水深
    ///
    /// # 返回
    /// 当`h < h_dry`时返回ZERO，否则返回`(hu/h, hv/h)`
    #[inline]
    pub fn from_momentum(hu: S, hv: S, h: S, h_dry: S, h_min: S) -> Self {
        if SafeDepth::<S>::is_originally_dry(h, h_dry) {
            Self::ZERO
        } else {
            let h_safe = SafeDepth::new(h, h_min);
            Self {
                u: hu / h_safe.get(),
                v: hv / h_safe.get(),
            }
        }
    }

    /// 从分量创建速度
    #[inline]
    pub const fn new(u: S, v: S) -> Self {
        Self { u, v }
    }

    /// 速度大小（模长）
    #[inline]
    pub fn speed(&self) -> S {
        (self.u * self.u + self.v * self.v).sqrt()
    }

    /// 速度平方
    #[inline]
    pub fn speed_squared(&self) -> S {
        self.u * self.u + self.v * self.v
    }

    /// 限制最大速度
    ///
    /// # 参数
    /// - `max_speed`: 最大允许速度值，必须与`S`同类型
    ///
    /// # 返回
    /// 如果|v| > max_speed，则缩放至max_speed，否则返回原值
    #[inline]
    pub fn clamp_speed(self, max_speed: S) -> Self {
        let speed = self.speed();
        if speed > max_speed && speed > S::from_f64(1e-14).unwrap() {
            let factor = max_speed / speed;
            Self {
                u: self.u * factor,
                v: self.v * factor,
            }
        } else {
            self
        }
    }

    /// 检查速度是否有效（非NaN/Inf）
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.u.is_finite() && self.v.is_finite()
    }

    /// 动能（单位质量）
    #[inline]
    pub fn kinetic_energy_per_mass(&self) -> S {
        S::from_f64(0.5).unwrap() * self.speed_squared()
    }
}

// SafeVelocity算术运算实现
impl<S: RuntimeScalar> Add for SafeVelocity<S> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            u: self.u + rhs.u,
            v: self.v + rhs.v,
        }
    }
}

impl<S: RuntimeScalar> Sub for SafeVelocity<S> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            u: self.u - rhs.u,
            v: self.v - rhs.v,
        }
    }
}

impl<S: RuntimeScalar> Mul<S> for SafeVelocity<S> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: S) -> Self {
        Self {
            u: self.u * rhs,
            v: self.v * rhs,
        }
    }
}

impl<S: RuntimeScalar> fmt::Display for SafeVelocity<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}) m/s", self.u, self.v) // 添加单位
    }
}

// ============================================================
// 泛型数值参数配置（核心改造）
// ============================================================

/// 数值参数配置 - 泛型版本
///
/// 控制浅水方程求解器的各种阈值和参数，通过泛型参数`S`支持f32/f64运行时切换。
///
/// # 类型参数
/// - `S`: 运行时标量类型（f32或f64），必须实现`RuntimeScalar` + `PartialOrd`
///
/// # 使用示例
/// ```
/// use mh_physics::types::NumericalParams;
/// use mh_runtime::RuntimeScalar;
///
/// let params_f64: NumericalParams<f64> = NumericalParams::default();
/// let params_f32: NumericalParams<f32> = NumericalParams::default();
///
/// // 干湿判定
/// let is_dry = params_f64.is_dry(1e-8);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalParams<S>
where
    S: RuntimeScalar + PartialOrd,
{
    /// 最小水深 [m] - 用于除法保护
    pub h_min: S,
    /// 干单元阈值 [m] - 低于此值视为干
    pub h_dry: S,
    /// 摩擦计算水深阈值 [m]
    pub h_friction: S,
    /// 湿单元阈值 [m] - 高于此值视为完全湿
    pub h_wet: S,
    /// 通量计算零阈值
    pub flux_eps: S,
    /// 熵修正比例因子
    pub entropy_ratio: S,
    /// 最小波速 [m/s]
    pub min_wave_speed: S,
    /// 行列式最小阈值
    pub det_min: S,
    /// 限制器K参数（Venkatakrishnan）
    pub limiter_k: S,
    /// 最小速度阈值 [m/s]
    pub vel_min: S,
    /// 最大允许速度 [m/s]
    pub vel_max: S,
    /// 最小涡粘系数 [m²/s]
    pub nu_min: S,
    /// 最大涡粘系数 [m²/s]
    pub nu_max: S,
    /// CFL数
    pub cfl: S,
    /// 最小时间步 [s]
    pub dt_min: S,
    /// 最大时间步 [s]
    pub dt_max: S,
    /// 水位容差 [m]
    pub eta_tolerance: S,
    /// 流量容差 [m³/s]
    pub flux_tolerance: S,
    /// 守恒检查容差
    pub conservation_tolerance: S,
}

/// f64参数类型别名
pub type NumericalParamsF64 = NumericalParams<f64>;

/// f32参数类型别名
pub type NumericalParamsF32 = NumericalParams<f32>;

impl<S> Default for NumericalParams<S>
where
    S: RuntimeScalar + PartialOrd + FromPrimitive,
{
    /// 使用标准物理默认值初始化
    fn default() -> Self {
        Self {
            h_min: S::from_f64(1e-9).unwrap(),
            h_dry: S::from_f64(1e-6).unwrap(),
            h_friction: S::from_f64(1e-4).unwrap(),
            h_wet: S::from_f64(1e-3).unwrap(),
            flux_eps: S::from_f64(1e-14).unwrap(),
            entropy_ratio: S::from_f64(0.1).unwrap(),
            min_wave_speed: S::from_f64(1e-6).unwrap(),
            det_min: S::from_f64(1e-14).unwrap(),
            limiter_k: S::from_f64(5.0).unwrap(),
            vel_min: S::from_f64(1e-8).unwrap(),
            vel_max: S::from_f64(100.0).unwrap(),
            nu_min: S::from_f64(1e-6).unwrap(),
            nu_max: S::from_f64(1e3).unwrap(),
            cfl: S::from_f64(0.5).unwrap(),
            dt_min: S::from_f64(1e-8).unwrap(),
            dt_max: S::from_f64(3600.0).unwrap(),
            eta_tolerance: S::from_f64(1e-6).unwrap(),
            flux_tolerance: S::from_f64(1e-10).unwrap(),
            conservation_tolerance: S::from_f64(1e-8).unwrap(),
        }
    }
}

/// 配置转换错误
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    #[error("配置参数'{0}'转换失败：无法从f64转换到目标类型")]
    Conversion(&'static str),
}

impl<S> NumericalParams<S>
where
    S: RuntimeScalar + PartialOrd + FromPrimitive,
{
    /// 从SolverConfig（Layer 4全f64配置）转换到泛型参数
    ///
    /// # 参数
    /// - `config`: Layer 4配置结构（全f64）
    ///
    /// # 返回
    /// - `Ok(Self)`: 转换成功
    /// - `Err(ConfigError)`: 转换失败（数值溢出或无法转换）
    pub fn from_config(config: &crate::builder::SolverConfig) -> Result<Self, ConfigError> {
        // 先使用默认值填充所有字段
        let mut params = Self::default();
        
        // TODO 仅覆盖 builder::SolverConfig 中存在的字段,其他字段可以后续根据需要补充，当前的任务是将代码测报错解决
        params.h_min = S::from_f64(config.h_min)
            .ok_or(ConfigError::Conversion("h_min"))?;
        params.h_dry = S::from_f64(config.h_dry)
            .ok_or(ConfigError::Conversion("h_dry"))?;
        params.cfl = S::from_f64(config.cfl)
            .ok_or(ConfigError::Conversion("cfl"))?;
        params.vel_max = S::from_f64(config.max_velocity)
            .ok_or(ConfigError::Conversion("max_velocity"))?;
        params.h_friction = S::from_f64(1e-4) // 默认值
            .ok_or(ConfigError::Conversion("h_friction"))?;
        
        // 其他字段保持默认值
        Ok(params)
    }

    /// 判断是否为干单元
    ///
    /// # 参数
    /// - `h`: 水深值
    ///
    /// # 返回
    /// `true`当h < self.h_dry
    #[inline]
    pub fn is_dry(&self, h: S) -> bool {
        h < self.h_dry
    }

    /// 判断是否为湿单元
    #[inline]
    pub fn is_wet(&self, h: S) -> bool {
        h >= self.h_wet
    }

    /// 判断是否在过渡区
    #[inline]
    pub fn is_transition(&self, h: S) -> bool {
        h >= self.h_dry && h < self.h_wet
    }

    /// 干湿过渡权重（线性）
    ///
    /// 返回值 ∈ [0, 1]：0=完全干，1=完全湿
    #[inline]
    pub fn wet_fraction(&self, h: S) -> S {
        if h <= self.h_dry {
            S::ZERO
        } else if h >= self.h_wet {
            S::ONE
        } else {
            (h - self.h_dry) / (self.h_wet - self.h_dry)
        }
    }

    /// 干湿过渡权重（Hermite平滑）
    #[inline]
    pub fn wet_fraction_smooth(&self, h: S) -> S {
        let t = self.wet_fraction(h);
        t * t * (S::from_f64(3.0).unwrap() - S::from_f64(2.0).unwrap() * t)
    }

    /// 创建安全水深
    #[inline]
    pub fn safe_depth(&self, h: S) -> S {
        h.max(self.h_min)
    }

    /// 创建摩擦安全水深
    #[inline]
    pub fn friction_safe_depth(&self, h: S) -> S {
        h.max(self.h_friction)
    }

    /// 计算波速（浅水方程）
    #[inline]
    pub fn wave_speed(&self, h: S, g: S) -> S {
        (g * h.max(S::ZERO)).sqrt().max(self.min_wave_speed)
    }

    /// 动态熵修正阈值
    #[inline]
    pub fn entropy_threshold(&self, local_wave_speed: S) -> S {
        (self.entropy_ratio * local_wave_speed.abs()).max(self.flux_eps)
    }

    /// 限制速度
    #[inline]
    pub fn clamp_velocity(&self, vel: SafeVelocity<S>) -> SafeVelocity<S> {
        vel.clamp_speed(self.vel_max)
    }

    /// 限制涡粘系数
    #[inline]
    pub fn clamp_nu(&self, nu: S) -> S {
        nu.clamp(self.nu_min, self.nu_max)
    }

    /// 计算最大允许时间步（基于CFL）
    #[inline]
    pub fn max_dt_from_cfl(&self, dx: S, max_wave_speed: S) -> S {
        let wave_speed = max_wave_speed.max(self.min_wave_speed);
        let dt = self.cfl * dx / wave_speed;
        dt.clamp(self.dt_min, self.dt_max)
    }

    /// 检查速度是否超过警告阈值
    #[inline]
    pub fn is_velocity_excessive(&self, speed: S) -> bool {
        speed > self.vel_max
    }

    /// 计算安全速度分量
    ///
    /// # 参数
    /// - `hu`, `hv`: 动量分量
    /// - `h`: 水深
    ///
    /// # 返回
    /// 当`h < h_dry`时返回(0,0)，否则返回安全速度
    #[inline]
    pub fn safe_velocity_components(&self, hu: S, hv: S, h: S) -> (S, S) {
        if self.is_dry(h) {
            (S::ZERO, S::ZERO)
        } else {
            let h_safe = self.safe_depth(h);
            // 使用正则化公式避免除零
            let h2 = h_safe * h_safe;
            let h4 = h2 * h2;
            let eps4 = self.h_min.powi(4);
            let denom = (h4 + eps4).sqrt();
            let u = hu * h_safe / denom;
            let v = hv * h_safe / denom;
            
            // 限制最大速度
            let speed = (u * u + v * v).sqrt();
            if speed > self.vel_max && speed > S::from_f64(1e-14).unwrap() {
                let factor = self.vel_max / speed;
                (u * factor, v * factor)
            } else {
                (u, v)
            }
        }
    }

    /// 计算安全速度
    #[inline]
    pub fn safe_velocity(&self, hu: S, hv: S, h: S) -> SafeVelocity<S> {
        let (u, v) = self.safe_velocity_components(hu, hv, h);
        SafeVelocity::new(u, v)
    }

    /// 验证参数有效性
    ///
    /// 检查阈值层级关系和正数约束
    pub fn validate(&self) -> Result<(), ParamsValidationError> {
        // 验证阈值层级
        if self.h_min >= self.h_dry {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "h_min",
                constraint: "h_min < h_dry",
            });
        }
        if self.h_dry >= self.h_friction {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "h_dry",
                constraint: "h_dry < h_friction",
            });
        }
        if self.h_friction >= self.h_wet {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "h_friction",
                constraint: "h_friction < h_wet",
            });
        }

        // 验证正数参数
        if self.cfl <= S::ZERO || self.cfl > S::ONE {
            return Err(ParamsValidationError::OutOfRange {
                field: "cfl",
                min: 0.0,
                max: 1.0,
            });
        }
        if self.dt_min <= S::ZERO {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "dt_min",
                constraint: "dt_min > 0",
            });
        }
        if self.dt_min >= self.dt_max {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "dt_min",
                constraint: "dt_min < dt_max",
            });
        }

        Ok(())
    }
}

/// 参数验证错误
#[derive(Debug, Clone, thiserror::Error)]
pub enum ParamsValidationError {
    /// 阈值约束违反
    #[error("参数{field}违反约束: {constraint}")]
    InvalidThreshold {
        field: &'static str,
        constraint: &'static str,
    },
    /// 数值超出允许范围
    #[error("参数{field}超出范围[{min}, {max}]")]
    OutOfRange {
        field: &'static str,
        min: f64,
        max: f64,
    },
}

// ============================================================
// 物理常数（保持f64，自然常数不随计算精度改变）
// ============================================================

/// 物理常数
///
/// 包含地球物理、流体性质等自然界常数。这些常量的值不随算法、场景、网格变化，
/// 仅取决于物理现实，因此保持为f64。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalConstants {
    /// 重力加速度 [m/s²]
    pub g: f64,
    /// 地球角速度 [rad/s]
    pub omega: f64,
    /// 水密度 [kg/m³]
    pub rho_water: f64,
    /// 空气密度 [kg/m³]
    pub rho_air: f64,
    /// 水的运动粘度 [m²/s]
    pub nu_water: f64,
    /// 风拖曳系数（默认值）
    pub wind_drag_coefficient: f64,
    /// 大气压 [Pa]
    pub atmospheric_pressure: f64,
    /// 地球平均半径 [m]
    pub earth_radius: f64,
}

impl Default for PhysicalConstants {
    /// 默认使用海水常数
    fn default() -> Self {
        Self::seawater()
    }
}

impl PhysicalConstants {
    /// 标准海水常数（3.5%盐度，15°C）
    pub fn seawater() -> Self {
        Self {
            g: 9.81,
            omega: 7.2921e-5,
            rho_water: 1025.0,
            rho_air: 1.225,
            nu_water: 1.0e-6,
            wind_drag_coefficient: 1.3e-3,
            atmospheric_pressure: 101325.0,
            earth_radius: 6_371_000.0,
        }
    }

    /// 淡水常数
    pub fn freshwater() -> Self {
        Self {
            rho_water: 1000.0,
            ..Self::seawater()
        }
    }

    /// 计算科里奥利参数 f = 2Ω sin(φ)
    ///
    /// # 参数
    /// - `latitude_rad`: 纬度 [弧度]
    #[inline]
    pub fn coriolis_parameter(&self, latitude_rad: f64) -> f64 {
        2.0 * self.omega * latitude_rad.sin()
    }

    /// 从纬度（度）计算科里奥利参数
    #[inline]
    pub fn coriolis_parameter_deg(&self, latitude_deg: f64) -> f64 {
        self.coriolis_parameter(latitude_deg.to_radians())
    }

    /// 计算风应力系数
    ///
    /// 风应力 τ = ρ_air * C_d * |W|² / ρ_water
    /// 返回 ρ_air * C_d / ρ_water
    #[inline]
    pub fn wind_stress_coefficient(&self) -> f64 {
        self.rho_air * self.wind_drag_coefficient / self.rho_water
    }
}

// ============================================================
// 求解器配置（Layer 4，保持f64）
// ============================================================

/// 黎曼求解器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RiemannSolverType {
    /// HLL求解器（鲁棒性好）
    Hll,
    /// HLLC求解器（更精确）
    #[default]
    Hllc,
    /// Roe求解器（需熵修正）
    Roe,
}

/// 时间积分方案
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TimeIntegration {
    /// 前向欧拉（一阶）
    ForwardEuler,
    /// SSP-RK2（二阶）
    #[default]
    SspRk2,
    /// SSP-RK3（三阶）
    SspRk3,
}

/// 梯度限制器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LimiterType {
    /// 无限制器（一阶精度）
    None,
    /// Barth-Jespersen限制器
    BarthJespersen,
    /// Venkatakrishnan限制器
    #[default]
    Venkatakrishnan,
    /// Minmod限制器
    Minmod,
}

/// 求解器配置（Layer 4，保持f64）
///
/// 本结构体属于应用层配置，所有参数使用f64存储。
/// 在构建求解器时，会转换到Layer 3的泛型参数。
/// 这使得CLI/Editor层完全无泛型语法。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// 数值参数（f64配置）
    pub numerical: NumericalParamsF64,
    /// 物理常数
    pub physics: PhysicalConstants,
    /// 黎曼求解器类型
    pub riemann_solver: RiemannSolverType,
    /// 时间积分方案
    pub time_integration: TimeIntegration,
    /// 梯度限制器
    pub limiter: LimiterType,
    /// 是否启用二阶精度
    pub second_order: bool,
    /// 是否启用干湿处理
    pub wetting_drying: bool,
    /// 是否启用摩擦
    pub friction: bool,
    /// 是否启用科里奥利力
    pub coriolis: bool,
    /// 是否启用风应力
    pub wind_stress: bool,
}

impl Default for SolverConfig {
    /// 使用标准默认值
    fn default() -> Self {
        Self {
            numerical: NumericalParamsF64::default(),
            physics: PhysicalConstants::default(),
            riemann_solver: RiemannSolverType::default(),
            time_integration: TimeIntegration::default(),
            limiter: LimiterType::default(),
            second_order: true,
            wetting_drying: true,
            friction: true,
            coriolis: false,
            wind_stress: false,
        }
    }
}

impl SolverConfig {
    /// 创建新配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 验证配置有效性
    pub fn validate(&self) -> Result<(), ParamsValidationError> {
        self.numerical.validate()
    }

    /// 创建用于稳定性测试的配置
    pub fn for_stability_test() -> Self {
        Self {
            numerical: NumericalParamsF64 {
                cfl: 0.3,
                ..NumericalParamsF64::default()
            },
            second_order: false,
            ..Self::default()
        }
    }

    /// 创建高精度配置
    pub fn high_accuracy() -> Self {
        Self {
            numerical: NumericalParamsF64 {
                cfl: 0.3,
                ..NumericalParamsF64::default()
            },
            time_integration: TimeIntegration::SspRk3,
            limiter: LimiterType::Venkatakrishnan,
            second_order: true,
            ..Self::default()
        }
    }
}

// ============================================================
// 边界值提供者基础trait
// ============================================================

/// 边界值提供者基础trait
///
/// 用于提供边界条件的时变值，支持水位、流量、浓度等。
/// 实现者需要是Send + Sync以支持并行计算。
///
/// # 类型参数
/// - `T`: 边界值的类型（如f64表示标量，[f64; 2]表示向量）
pub trait BoundaryValueProvider<T>: Send + Sync {
    /// 获取指定边界面在给定时间的边界值
    ///
    /// # 参数
    /// - `face_idx`: 边界面索引
    /// - `time`: 模拟时间 [s]
    ///
    /// # 返回
    /// 边界值，若该面无边界值则返回None
    fn get_value(&self, face_idx: usize, time: f64) -> Option<T>;

    /// 批量获取边界值
    ///
    /// 默认实现逐个调用`get_value`，可重写以优化性能。
    fn get_values_batch(&self, face_indices: &[usize], time: f64, out: &mut [Option<T>])
    where
        T: Clone,
    {
        for (i, &face_idx) in face_indices.iter().enumerate() {
            out[i] = self.get_value(face_idx, time);
        }
    }

    /// 检查是否为指定面提供边界值
    fn provides_for(&self, face_idx: usize) -> bool {
        // 默认实现：尝试获取t=0时的值
        self.get_value(face_idx, 0.0).is_some()
    }
}

/// 常量边界值提供者
///
/// 为所有边界面提供相同的常量值。
#[derive(Debug, Clone)]
pub struct ConstantBoundaryProvider<T>
where
    T: Clone + Send + Sync,
{
    value: T,
}

impl<T> ConstantBoundaryProvider<T>
where
    T: Clone + Send + Sync,
{
    /// 创建常量边界值提供者
    pub fn new(value: T) -> Self {
        Self { value }
    }

    /// 获取常量值
    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T> BoundaryValueProvider<T> for ConstantBoundaryProvider<T>
where
    T: Clone + Send + Sync,
{
    fn get_value(&self, _face_idx: usize, _time: f64) -> Option<T> {
        Some(self.value.clone())
    }

    fn provides_for(&self, _face_idx: usize) -> bool {
        true
    }
}

/// 零边界值提供者
///
/// 为所有边界面提供零值（适用于标量）。
#[derive(Debug, Clone, Copy, Default)]
pub struct ZeroBoundaryProvider;

impl BoundaryValueProvider<f64> for ZeroBoundaryProvider {
    fn get_value(&self, _face_idx: usize, _time: f64) -> Option<f64> {
        Some(0.0)
    }

    fn provides_for(&self, _face_idx: usize) -> bool {
        true
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::RuntimeScalar;

    #[test]
    fn test_cell_index_basic() {
        // T=1清理后直接使用 get() 和 new()
        let idx = CellIndex::new(42);
        assert_eq!(idx.get(), 42);
        let idx2 = CellIndex::new(42);
        assert_eq!(idx2, idx);
    }

    #[test]
    fn test_safe_depth_f64() {
        let depth = SafeDepth::<f64>::new(1e-10, 1e-9);
        assert!(depth.get() >= 1e-9);

        let depth2 = SafeDepth::<f64>::new(0.1, 1e-4);
        assert_eq!(depth2.get(), 0.1);
    }

    #[test]
    fn test_safe_depth_f32() {
        let depth = SafeDepth::<f32>::new(1e-4, 1e-3);
        assert!(depth.get() >= 1e-3);
    }

    #[test]
    fn test_safe_velocity_f64() {
        let v = SafeVelocity::<f64>::from_momentum(10.0, 20.0, 2.0, 1e-6, 1e-9);
        assert!((v.u - 5.0).abs() < 1e-10);
        assert!((v.v - 10.0).abs() < 1e-10);

        let v_dry = SafeVelocity::<f64>::from_momentum(10.0, 20.0, 1e-8, 1e-6, 1e-9);
        assert_eq!(v_dry, SafeVelocity::ZERO);
    }

    #[test]
    fn test_safe_velocity_f32() {
        let v = SafeVelocity::<f32>::from_momentum(10.0, 20.0, 2.0, 1e-6, 1e-9);
        assert!((v.u - 5.0f32).abs() < 1e-6f32);
    }

    #[test]
    fn test_safe_velocity_clamp() {
        let v = SafeVelocity::<f64>::new(100.0, 0.0);
        let clamped = v.clamp_speed(50.0);
        assert!((clamped.speed() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_params_default() {
        let params_f64: NumericalParams<f64> = NumericalParams::default();
        assert_eq!(params_f64.h_dry, 1e-6f64);

        let params_f32: NumericalParams<f32> = NumericalParams::default();
        assert_eq!(params_f32.h_dry, 1e-6f32);
    }

    #[test]
    fn test_numerical_params_f64_from_config() {
        let config = SolverConfig::default();
        let params = NumericalParams::<f64>::from_config(&config).unwrap();
        assert_eq!(params.h_dry, config.numerical.h_dry);
    }

    #[test]
    fn test_numerical_params_f32_from_config() {
        let mut config = SolverConfig::default();
        config.numerical.cfl = 0.8;
        let params = NumericalParams::<f32>::from_config(&config).unwrap();
        assert_eq!(params.cfl, 0.8f32);
    }

    #[test]
    fn test_numerical_params_validate() {
        let params = NumericalParams::<f64>::default();
        assert!(params.validate().is_ok());

        let invalid = NumericalParams::<f64> {
            h_min: 1e-3,
            h_dry: 1e-6,
            ..NumericalParams::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_numerical_params_wet_fraction() {
        let params: NumericalParams<f64> = NumericalParams::default();
        assert_eq!(params.wet_fraction(0.0f64), 0.0f64);
        assert_eq!(params.wet_fraction(1e-6f64), 0.0f64);
        assert_eq!(params.wet_fraction(1e-3f64), 1.0f64);

        let mid = 5.5e-4f64;
        let frac = params.wet_fraction(mid);
        assert!(frac > 0.0 && frac < 1.0);
    }

    #[test]
    fn test_physical_constants_seawater() {
        let sea = PhysicalConstants::seawater();
        assert_eq!(sea.g, 9.81f64);
        assert_eq!(sea.rho_water, 1025.0f64);
    }

    #[test]
    fn test_coriolis_parameter() {
        let consts = PhysicalConstants::default();
        let f = consts.coriolis_parameter_deg(45.0f64);
        assert!((f - 1.03e-4f64).abs() < 1e-6f64);
    }

    #[test]
    fn test_solver_config_validate() {
        let config = SolverConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.second_order);
        assert!(config.wetting_drying);
    }

    #[test]
    fn test_boundary_provider_constant() {
        let provider = ConstantBoundaryProvider::new(10.0f64);
        assert_eq!(provider.get_value(0, 0.0), Some(10.0f64));
        assert_eq!(provider.provides_for(999), true);
    }

    #[test]
    fn test_config_conversion_from_builder() {
        let builder_config = crate::builder::SolverConfig::default();
        let params = NumericalParams::<f64>::from_config(&builder_config).unwrap();
        assert_eq!(params.h_min, builder_config.h_min);
        assert_eq!(params.h_dry, builder_config.h_dry);
        assert_eq!(params.cfl, builder_config.cfl);
    }
}