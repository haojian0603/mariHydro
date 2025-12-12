// crates/mh_physics/src/types.rs

//! 物理计算核心类型定义
//!
//! 本模块提供物理求解器所需的类型定义，包括：
//! - 类型安全索引 (从 mh_core 重新导出)
//! - 安全包装类型 (SafeDepth, SafeVelocity)
//! - 数值参数配置 (NumericalParams)
//! - 物理常数 (PhysicalConstants)
//!

use glam::DVec2;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Mul, Sub};

// ============================================================
// 类型安全索引 (从 mh_core 重新导出)
// ============================================================

// 重新导出 mh_core 中统一定义的索引类型
pub use mh_core::{
    CellIndex, FaceIndex, NodeIndex, BoundaryIndex,
    INVALID_INDEX,
};

// ============================================================
// 索引类型扩展 - 为物理引擎添加额外方法
// ============================================================

/// CellIndex 的物理引擎扩展 trait
pub trait CellIndexExt {
    /// 转换为 u32 (用于新架构兼容)
    fn as_u32(self) -> u32;
    /// 从 u32 创建 (用于新架构兼容)
    fn from_u32(idx: u32) -> Self;
}

impl CellIndexExt for CellIndex {
    #[inline]
    fn as_u32(self) -> u32 {
        self.0 as u32
    }

    #[inline]
    fn from_u32(idx: u32) -> CellIndex {
        CellIndex(idx as usize)
    }
}

/// FaceIndex 的物理引擎扩展 trait
pub trait FaceIndexExt {
    /// 转换为 u32 (用于新架构兼容)
    fn as_u32(self) -> u32;
    /// 从 u32 创建 (用于新架构兼容)
    fn from_u32(idx: u32) -> Self;
}

impl FaceIndexExt for FaceIndex {
    #[inline]
    fn as_u32(self) -> u32 {
        self.0 as u32
    }

    #[inline]
    fn from_u32(idx: u32) -> FaceIndex {
        FaceIndex(idx as usize)
    }
}

/// NodeIndex 的物理引擎扩展 trait
pub trait NodeIndexExt {
    /// 转换为 u32 (用于新架构兼容)
    fn as_u32(self) -> u32;
    /// 从 u32 创建 (用于新架构兼容)
    fn from_u32(idx: u32) -> Self;
}

impl NodeIndexExt for NodeIndex {
    #[inline]
    fn as_u32(self) -> u32 {
        self.0 as u32
    }

    #[inline]
    fn from_u32(idx: u32) -> NodeIndex {
        NodeIndex(idx as usize)
    }
}

// ============================================================
// 安全包装类型
// ============================================================

/// 安全水深（保证 >= h_min）
///
/// # 用途
///
/// 在浅水方程计算中，避免除以接近零的水深导致数值不稳定。
#[derive(Debug, Clone, Copy)]
pub struct SafeDepth {
    value: f64,
    h_min: f64,
}

impl SafeDepth {
    /// 从原始水深创建
    #[inline]
    pub fn new(h: f64, h_min: f64) -> Self {
        Self {
            value: h.max(h_min),
            h_min,
        }
    }

    /// 获取安全值（用于除法等运算）
    #[inline]
    pub fn get(self) -> f64 {
        self.value
    }

    /// 获取 h_min 阈值
    #[inline]
    pub fn h_min(self) -> f64 {
        self.h_min
    }

    /// 判断原始水深是否干（低于干湿阈值）
    #[inline]
    pub fn is_originally_dry(h: f64, h_dry: f64) -> bool {
        h < h_dry
    }

    /// 安全除法
    #[inline]
    pub fn safe_divide(numerator: f64, h: f64, h_min: f64) -> f64 {
        numerator / Self::new(h, h_min).get()
    }

    /// 计算安全速度分量
    #[inline]
    pub fn velocity_component(momentum: f64, h: f64, h_min: f64) -> f64 {
        if h < h_min {
            0.0
        } else {
            momentum / h.max(h_min)
        }
    }
}

impl Default for SafeDepth {
    fn default() -> Self {
        Self {
            value: 1e-9,
            h_min: 1e-9,
        }
    }
}

/// 安全速度（避免除零导致的无穷大）
///
/// # 用途
///
/// 从动量和水深计算速度时，处理干单元（h ≈ 0）的情况。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SafeVelocity {
    /// x 方向速度 [m/s]
    pub u: f64,
    /// y 方向速度 [m/s]
    pub v: f64,
}

impl SafeVelocity {
    /// 零速度
    pub const ZERO: Self = Self { u: 0.0, v: 0.0 };

    /// 从动量和水深计算速度
    ///
    /// # 参数
    ///
    /// - `hu`, `hv`: 动量分量 [m²/s]
    /// - `h`: 水深 [m]
    /// - `h_dry`: 干湿判断阈值 [m]
    /// - `h_min`: 最小计算水深 [m]
    #[inline]
    pub fn from_momentum(hu: f64, hv: f64, h: f64, h_dry: f64, h_min: f64) -> Self {
        if SafeDepth::is_originally_dry(h, h_dry) {
            Self::ZERO
        } else {
            let h_safe = SafeDepth::new(h, h_min);
            Self {
                u: hu / h_safe.get(),
                v: hv / h_safe.get(),
            }
        }
    }

    /// 从分量创建
    #[inline]
    pub const fn new(u: f64, v: f64) -> Self {
        Self { u, v }
    }

    /// 从 DVec2 创建
    #[inline]
    pub fn from_vec(v: DVec2) -> Self {
        Self { u: v.x, v: v.y }
    }

    /// 速度大小
    #[inline]
    pub fn speed(self) -> f64 {
        (self.u * self.u + self.v * self.v).sqrt()
    }

    /// 速度平方
    #[inline]
    pub fn speed_squared(self) -> f64 {
        self.u * self.u + self.v * self.v
    }

    /// 转换为 DVec2
    #[inline]
    pub fn as_dvec2(self) -> DVec2 {
        DVec2::new(self.u, self.v)
    }

    /// 法向分量
    #[inline]
    pub fn normal_component(self, normal: DVec2) -> f64 {
        self.u * normal.x + self.v * normal.y
    }

    /// 切向分量
    #[inline]
    pub fn tangent_component(self, normal: DVec2) -> f64 {
        -self.u * normal.y + self.v * normal.x
    }

    /// 旋转到局部坐标系（法向、切向）
    #[inline]
    pub fn to_local(self, normal: DVec2) -> (f64, f64) {
        let un = self.normal_component(normal);
        let ut = self.tangent_component(normal);
        (un, ut)
    }

    /// 从局部坐标系转换回全局
    #[inline]
    pub fn from_local(un: f64, ut: f64, normal: DVec2) -> Self {
        Self {
            u: un * normal.x - ut * normal.y,
            v: un * normal.y + ut * normal.x,
        }
    }

    /// 限制最大速度
    #[inline]
    pub fn clamp_speed(self, max_speed: f64) -> Self {
        let speed = self.speed();
        if speed > max_speed && speed > 1e-14 {
            let factor = max_speed / speed;
            Self {
                u: self.u * factor,
                v: self.v * factor,
            }
        } else {
            self
        }
    }

    /// 检查速度是否有效
    #[inline]
    pub fn is_valid(self) -> bool {
        self.u.is_finite() && self.v.is_finite()
    }

    /// 动能（单位质量）
    #[inline]
    pub fn kinetic_energy_per_mass(self) -> f64 {
        0.5 * self.speed_squared()
    }
}

impl Add for SafeVelocity {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            u: self.u + rhs.u,
            v: self.v + rhs.v,
        }
    }
}

impl Sub for SafeVelocity {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            u: self.u - rhs.u,
            v: self.v - rhs.v,
        }
    }
}

impl Mul<f64> for SafeVelocity {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            u: self.u * rhs,
            v: self.v * rhs,
        }
    }
}

impl fmt::Display for SafeVelocity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}) m/s", self.u, self.v)
    }
}

// ============================================================
// 数值参数配置
// ============================================================

/// 数值参数配置
///
/// 控制浅水方程求解器的各种阈值和参数。
/// 采用保守的默认值，适用于大多数海洋/河流模拟。
///
/// # 阈值层级关系
///
/// ```text
/// h_min < h_dry < h_friction < h_wet
/// 1e-9    1e-6    1e-4         1e-3
/// ```
///
/// - `h_min`: 数值安全最小水深（用于除法保护）
/// - `h_dry`: 干单元判断阈值（完全干）
/// - `h_friction`: 摩擦计算水深阈值
/// - `h_wet`: 湿单元判断阈值（完全湿）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalParams {
    // ========== 水深阈值 ==========
    /// 最小水深 [m] - 用于除法保护
    pub h_min: f64,

    /// 干单元阈值 [m] - 低于此值视为干
    pub h_dry: f64,

    /// 摩擦计算水深阈值 [m]
    pub h_friction: f64,

    /// 湿单元阈值 [m] - 高于此值视为完全湿
    pub h_wet: f64,

    // ========== 通量计算参数 ==========
    /// 通量计算零阈值
    pub flux_eps: f64,

    /// 熵修正比例因子
    pub entropy_ratio: f64,

    /// 最小波速 [m/s]
    pub min_wave_speed: f64,

    // ========== 梯度限制器参数 ==========
    /// 最小行列式阈值
    pub det_min: f64,

    /// 限制器 K 参数（Venkatakrishnan）
    pub limiter_k: f64,

    // ========== 速度限制 ==========
    /// 最小速度阈值 [m/s]
    pub vel_min: f64,

    /// 最大允许速度 [m/s]
    pub vel_max: f64,

    // ========== 湍流参数 ==========
    /// 最小涡粘系数 [m²/s]
    pub nu_min: f64,

    /// 最大涡粘系数 [m²/s]
    pub nu_max: f64,

    // ========== 时间步参数 ==========
    /// CFL 数
    pub cfl: f64,

    /// 最小时间步 [s]
    pub dt_min: f64,

    /// 最大时间步 [s]
    pub dt_max: f64,

    // ========== 容差参数 ==========
    /// 水位容差 [m]
    pub eta_tolerance: f64,

    /// 流量容差 [m³/s]
    pub flux_tolerance: f64,

    /// 守恒检查容差
    pub conservation_tolerance: f64,
}

impl Default for NumericalParams {
    fn default() -> Self {
        Self {
            // 水深阈值（层级递增）
            h_min: 1e-9,
            h_dry: 1e-6,
            h_friction: 1e-4,
            h_wet: 1e-3,

            // 通量计算
            flux_eps: 1e-14,
            entropy_ratio: 0.1,
            min_wave_speed: 1e-6,

            // 梯度限制器
            det_min: 1e-14,
            limiter_k: 5.0,

            // 速度限制
            vel_min: 1e-8,
            vel_max: 100.0,

            // 湍流
            nu_min: 1e-6,
            nu_max: 1e3,

            // 时间步
            cfl: 0.5,
            dt_min: 1e-8,
            dt_max: 3600.0,

            // 容差
            eta_tolerance: 1e-6,
            flux_tolerance: 1e-10,
            conservation_tolerance: 1e-8,
        }
    }
}

impl NumericalParams {
    /// 创建新的参数实例（使用默认值）
    pub fn new() -> Self {
        Self::default()
    }

    /// 创建高精度参数配置
    pub fn high_precision() -> Self {
        Self {
            h_min: 1e-12,
            h_dry: 1e-9,
            h_friction: 1e-6,
            h_wet: 1e-4,
            flux_eps: 1e-16,
            cfl: 0.3,
            ..Default::default()
        }
    }

    /// 创建快速计算参数配置
    pub fn fast() -> Self {
        Self {
            h_min: 1e-6,
            h_dry: 1e-4,
            h_friction: 1e-3,
            h_wet: 1e-2,
            cfl: 0.8,
            ..Default::default()
        }
    }

    /// 验证参数有效性
    pub fn validate(&self) -> Result<(), ParamsValidationError> {
        // 验证阈值层级
        if !(self.h_min < self.h_dry) {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "h_min",
                constraint: "h_min < h_dry",
                value: self.h_min,
            });
        }
        if !(self.h_dry < self.h_friction) {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "h_dry",
                constraint: "h_dry < h_friction",
                value: self.h_dry,
            });
        }
        if !(self.h_friction < self.h_wet) {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "h_friction",
                constraint: "h_friction < h_wet",
                value: self.h_friction,
            });
        }

        // 验证正数参数
        if self.cfl <= 0.0 || self.cfl > 1.0 {
            return Err(ParamsValidationError::OutOfRange {
                field: "cfl",
                min: 0.0,
                max: 1.0,
                value: self.cfl,
            });
        }
        if self.dt_min <= 0.0 {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "dt_min",
                constraint: "dt_min > 0",
                value: self.dt_min,
            });
        }
        if self.dt_min >= self.dt_max {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "dt_min",
                constraint: "dt_min < dt_max",
                value: self.dt_min,
            });
        }

        Ok(())
    }

    // ========== 干湿判断方法 ==========

    /// 判断是否为干单元
    #[inline]
    pub fn is_dry(&self, h: f64) -> bool {
        h < self.h_dry
    }

    /// 判断是否为湿单元
    #[inline]
    pub fn is_wet(&self, h: f64) -> bool {
        h >= self.h_wet
    }

    /// 判断是否在过渡区
    #[inline]
    pub fn is_transition(&self, h: f64) -> bool {
        h >= self.h_dry && h < self.h_wet
    }

    /// 干湿过渡权重（线性）
    ///
    /// 返回值 ∈ [0, 1]：0=干，1=湿
    #[inline]
    pub fn wet_fraction(&self, h: f64) -> f64 {
        if h <= self.h_dry {
            0.0
        } else if h >= self.h_wet {
            1.0
        } else {
            (h - self.h_dry) / (self.h_wet - self.h_dry)
        }
    }

    /// 干湿过渡权重（Hermite 平滑）
    #[inline]
    pub fn wet_fraction_smooth(&self, h: f64) -> f64 {
        if h <= self.h_dry {
            0.0
        } else if h >= self.h_wet {
            1.0
        } else {
            let t = (h - self.h_dry) / (self.h_wet - self.h_dry);
            t * t * (3.0 - 2.0 * t)
        }
    }

    // ========== 安全计算方法 ==========

    /// 创建安全水深
    #[inline]
    pub fn safe_depth(&self, h: f64) -> SafeDepth {
        SafeDepth::new(h, self.h_min)
    }

    /// 创建摩擦安全水深
    #[inline]
    pub fn friction_safe_depth(&self, h: f64) -> SafeDepth {
        SafeDepth::new(h, self.h_friction)
    }

    /// 计算安全速度
    #[inline]
    pub fn safe_velocity(&self, hu: f64, hv: f64, h: f64) -> SafeVelocity {
        SafeVelocity::from_momentum(hu, hv, h, self.h_dry, self.h_min)
    }

    /// 动态熵修正阈值
    #[inline]
    pub fn entropy_threshold(&self, local_wave_speed: f64) -> f64 {
        (self.entropy_ratio * local_wave_speed.abs()).max(self.flux_eps)
    }

    /// 计算最大允许时间步（基于CFL）
    #[inline]
    pub fn max_dt_from_cfl(&self, dx: f64, max_wave_speed: f64) -> f64 {
        let wave_speed = max_wave_speed.max(self.min_wave_speed);
        let dt = self.cfl * dx / wave_speed;
        dt.clamp(self.dt_min, self.dt_max)
    }

    /// 限制涡粘系数
    #[inline]
    pub fn clamp_nu(&self, nu: f64) -> f64 {
        nu.clamp(self.nu_min, self.nu_max)
    }

    /// 限制速度
    #[inline]
    pub fn clamp_velocity(&self, vel: SafeVelocity) -> SafeVelocity {
        vel.clamp_speed(self.vel_max)
    }

    /// 检查速度是否超过警告阈值
    #[inline]
    pub fn is_velocity_excessive(&self, speed: f64) -> bool {
        speed > self.vel_max
    }

    /// 计算波速（浅水）
    #[inline]
    pub fn wave_speed(&self, h: f64, g: f64) -> f64 {
        (g * h.max(0.0)).sqrt().max(self.min_wave_speed)
    }

    /// 创建参数构建器
    #[inline]
    pub fn builder() -> NumericalParamsBuilder {
        NumericalParamsBuilder::new()
    }

    /// 计算安全速度分量（返回 (u, v) 元组）
    ///
    /// 用于从动量计算速度，处理干单元的数值问题
    #[inline]
    pub fn safe_velocity_components(&self, hu: f64, hv: f64, h: f64) -> (f64, f64) {
        if h < self.h_dry {
            // 干单元：速度为零
            (0.0, 0.0)
        } else {
            // 使用正则化公式避免除零
            let h_safe = h.max(self.h_min);
            let h2 = h_safe * h_safe;
            let h4 = h2 * h2;
            let eps4 = self.h_min.powi(4);
            let denom = (h4 + eps4).sqrt();
            
            let u = hu * h_safe / denom;
            let v = hv * h_safe / denom;
            
            // 限制最大速度
            let speed = (u * u + v * v).sqrt();
            if speed > self.vel_max {
                let scale = self.vel_max / speed;
                (u * scale, v * scale)
            } else {
                (u, v)
            }
        }
    }
}

/// 参数验证错误
#[derive(Debug, Clone)]
pub enum ParamsValidationError {
    InvalidThreshold {
        field: &'static str,
        constraint: &'static str,
        value: f64,
    },
    OutOfRange {
        field: &'static str,
        min: f64,
        max: f64,
        value: f64,
    },
}

impl fmt::Display for ParamsValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidThreshold {
                field,
                constraint,
                value,
            } => {
                write!(f, "参数 {} = {} 违反约束: {}", field, value, constraint)
            }
            Self::OutOfRange {
                field,
                min,
                max,
                value,
            } => {
                write!(f, "参数 {} = {} 超出范围 [{}, {}]", field, value, min, max)
            }
        }
    }
}

impl std::error::Error for ParamsValidationError {}

/// 参数构建器
#[derive(Default)]
pub struct NumericalParamsBuilder {
    params: NumericalParams,
}

impl NumericalParamsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    // 水深阈值
    pub fn h_min(mut self, v: f64) -> Self {
        self.params.h_min = v;
        self
    }
    pub fn h_dry(mut self, v: f64) -> Self {
        self.params.h_dry = v;
        self
    }
    pub fn h_friction(mut self, v: f64) -> Self {
        self.params.h_friction = v;
        self
    }
    pub fn h_wet(mut self, v: f64) -> Self {
        self.params.h_wet = v;
        self
    }

    // 通量阈值
    pub fn flux_eps(mut self, v: f64) -> Self {
        self.params.flux_eps = v;
        self
    }
    pub fn entropy_ratio(mut self, v: f64) -> Self {
        self.params.entropy_ratio = v;
        self
    }

    // 时间步
    pub fn cfl(mut self, v: f64) -> Self {
        self.params.cfl = v;
        self
    }
    pub fn dt_min(mut self, v: f64) -> Self {
        self.params.dt_min = v;
        self
    }
    pub fn dt_max(mut self, v: f64) -> Self {
        self.params.dt_max = v;
        self
    }

    // 速度限制
    pub fn vel_max(mut self, v: f64) -> Self {
        self.params.vel_max = v;
        self
    }

    /// 构建参数（带验证）
    pub fn build(self) -> Result<NumericalParams, ParamsValidationError> {
        self.params.validate()?;
        Ok(self.params)
    }

    /// 构建参数（不验证）
    pub fn build_unchecked(self) -> NumericalParams {
        self.params
    }
}

/// 物理常数
///
/// 包含地球物理、流体性质等自然界常数。
/// 如果常量的值不随算法、场景、网格变化，仅取决于物理现实，则必须放入 PhysicalConstants
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
    fn default() -> Self {
        Self::seawater()
    }
}

impl PhysicalConstants {
    /// 标准海水常数（3.5% 盐度，15°C）
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
    ///
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
// 求解器配置
// ============================================================

/// 黎曼求解器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RiemannSolverType {
    /// HLL 求解器（鲁棒性好）
    Hll,
    /// HLLC 求解器（更精确）
    #[default]
    Hllc,
    /// Roe 求解器（需熵修正）
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
    /// Barth-Jespersen 限制器
    BarthJespersen,
    /// Venkatakrishnan 限制器
    #[default]
    Venkatakrishnan,
    /// Minmod 限制器
    Minmod,
}

/// 求解器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// 数值参数
    pub numerical: NumericalParams,

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
    fn default() -> Self {
        Self {
            numerical: NumericalParams::default(),
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
    /// 创建新的求解器配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 验证配置
    pub fn validate(&self) -> Result<(), ParamsValidationError> {
        self.numerical.validate()
    }

    /// 创建用于稳定性测试的配置
    pub fn for_stability_test() -> Self {
        Self {
            numerical: NumericalParams {
                cfl: 0.3,
                ..Default::default()
            },
            second_order: false,
            ..Default::default()
        }
    }

    /// 创建高精度配置
    pub fn high_accuracy() -> Self {
        Self {
            numerical: NumericalParams::high_precision(),
            time_integration: TimeIntegration::SspRk3,
            limiter: LimiterType::Venkatakrishnan,
            second_order: true,
            ..Default::default()
        }
    }
}

// ============================================================
// 边界值提供者基础 trait
// ============================================================

/// 边界值提供者基础 trait
///
/// 用于提供边界条件的时变值，可用于水位、流量、浓度等。
/// 实现者需要是 Send + Sync 以支持并行计算。
///
/// # 泛型参数
///
/// - `T`: 边界值的类型（如 f64 表示标量，DVec2 表示向量）
///
/// # 示例
///
/// ```ignore
/// use mh_physics::types::BoundaryValueProvider;
///
/// struct ConstantWaterLevel(f64);
///
/// impl BoundaryValueProvider<f64> for ConstantWaterLevel {
///     fn get_value(&self, _face_idx: usize, _time: f64) -> Option<f64> {
///         Some(self.0)
///     }
/// }
/// ```
pub trait BoundaryValueProvider<T>: Send + Sync {
    /// 获取指定边界面在给定时间的边界值
    ///
    /// # 参数
    ///
    /// - `face_idx`: 边界面索引
    /// - `time`: 模拟时间 [s]
    ///
    /// # 返回
    ///
    /// 边界值，若该面无边界值则返回 None
    fn get_value(&self, face_idx: usize, time: f64) -> Option<T>;

    /// 批量获取边界值
    ///
    /// 默认实现逐个调用 `get_value`，可重写以优化性能。
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
        // 默认实现：尝试获取 t=0 时的值
        self.get_value(face_idx, 0.0).is_some()
    }
}

/// 常量边界值提供者
///
/// 为所有边界面提供相同的常量值。
#[derive(Debug, Clone)]
pub struct ConstantBoundaryProvider<T: Clone + Send + Sync> {
    value: T,
}

impl<T: Clone + Send + Sync> ConstantBoundaryProvider<T> {
    /// 创建常量边界值提供者
    pub fn new(value: T) -> Self {
        Self { value }
    }

    /// 获取常量值
    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T: Clone + Send + Sync> BoundaryValueProvider<T> for ConstantBoundaryProvider<T> {
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

    #[test]
    fn test_cell_index() {
        let idx = CellIndex::new(42);
        assert_eq!(idx.get(), 42);
        assert!(idx.is_valid());
        assert!(!CellIndex::INVALID.is_valid());
    }

    #[test]
    fn test_safe_depth() {
        let sd = SafeDepth::new(1e-10, 1e-9);
        assert!(sd.get() >= 1e-9);

        let sd2 = SafeDepth::new(1.0, 1e-9);
        assert_eq!(sd2.get(), 1.0);
    }

    #[test]
    fn test_safe_velocity() {
        // 正常水深
        let v = SafeVelocity::from_momentum(10.0, 20.0, 2.0, 1e-6, 1e-9);
        assert!((v.u - 5.0).abs() < 1e-10);
        assert!((v.v - 10.0).abs() < 1e-10);

        // 干单元
        let v_dry = SafeVelocity::from_momentum(10.0, 20.0, 1e-8, 1e-6, 1e-9);
        assert_eq!(v_dry, SafeVelocity::ZERO);
    }

    #[test]
    fn test_numerical_params_validation() {
        let params = NumericalParams::default();
        assert!(params.validate().is_ok());

        // 无效阈值层级
        let invalid = NumericalParams {
            h_min: 1e-3,
            h_dry: 1e-6,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_numerical_params_dry_wet() {
        let params = NumericalParams::default();

        assert!(params.is_dry(1e-8));
        assert!(params.is_wet(1e-2));
        assert!(params.is_transition(5e-4));

        let frac = params.wet_fraction(5e-4);
        assert!(frac > 0.0 && frac < 1.0);
    }

    #[test]
    fn test_physical_constants() {
        let sea = PhysicalConstants::seawater();
        assert_eq!(sea.g, 9.81);
        assert_eq!(sea.rho_water, 1025.0);

        let fresh = PhysicalConstants::freshwater();
        assert_eq!(fresh.rho_water, 1000.0);
    }

    #[test]
    fn test_coriolis() {
        let consts = PhysicalConstants::default();
        let f = consts.coriolis_parameter_deg(45.0);
        // f ≈ 2 * 7.2921e-5 * sin(45°) ≈ 1.03e-4
        assert!((f - 1.03e-4).abs() < 1e-6);
    }

    #[test]
    fn test_solver_config() {
        let config = SolverConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.second_order);
        assert!(config.wetting_drying);
    }
}
