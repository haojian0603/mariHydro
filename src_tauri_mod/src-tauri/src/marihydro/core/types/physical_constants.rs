// src-tauri/src/marihydro/core/types/physical_constants.rs

//! 物理常数
//!
//! 包含地球物理、流体性质等自然界常数。

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// 物理常数（不可变）
///
/// # 使用方式
///
/// ```
/// use marihydro::core::types::PhysicalConstants;
///
/// let consts = PhysicalConstants::default();
/// let f = consts.coriolis_parameter(30.0_f64.to_radians());
/// ```
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
    ///
    /// # 返回
    ///
    /// 科里奥利参数 [1/s]
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
    pub fn wind_stress_factor(&self) -> f64 {
        self.rho_air * self.wind_drag_coefficient / self.rho_water
    }

    /// 计算风应力 [N/m²] → 加速度项 [m/s²]
    ///
    /// # 参数
    ///
    /// - `wind_speed`: 风速大小 [m/s]
    /// - `h`: 水深 [m]
    #[inline]
    pub fn wind_acceleration(&self, wind_speed: f64, h: f64) -> f64 {
        if h < 1e-6 {
            return 0.0;
        }
        self.wind_stress_factor() * wind_speed * wind_speed / h
    }

    /// 计算浅水波速 c = sqrt(g * h)
    #[inline]
    pub fn shallow_water_speed(&self, h: f64) -> f64 {
        (self.g * h.max(0.0)).sqrt()
    }

    /// 计算 Froude 数 Fr = V / c
    #[inline]
    pub fn froude_number(&self, velocity: f64, h: f64) -> f64 {
        let c = self.shallow_water_speed(h);
        if c < 1e-10 {
            return 0.0;
        }
        velocity.abs() / c
    }

    /// 判断是否为超临界流
    #[inline]
    pub fn is_supercritical(&self, velocity: f64, h: f64) -> bool {
        self.froude_number(velocity, h) > 1.0
    }

    /// 计算平衡水位梯度（用于大气压驱动）
    ///
    /// dη/dx = -(1/(ρg)) * dp/dx
    #[inline]
    pub fn pressure_gradient_to_slope(&self, dp_dx: f64) -> f64 {
        -dp_dx / (self.rho_water * self.g)
    }
}

// ============================================================
// 辅助常量模块
// ============================================================

/// 天文与地理常数
pub mod astronomy {
    use super::PI;

    /// 地球自转角速度 [rad/s] (GRS80)
    pub const EARTH_ROTATION_RATE: f64 = 7.292115e-5;

    /// 一天的秒数
    pub const SECONDS_PER_DAY: f64 = 86400.0;

    /// 一小时的秒数
    pub const SECONDS_PER_HOUR: f64 = 3600.0;

    /// 角度转弧度系数
    pub const DEG_TO_RAD: f64 = PI / 180.0;

    /// 弧度转角度系数
    pub const RAD_TO_DEG: f64 = 180.0 / PI;

    /// 将角度转换为弧度
    #[inline]
    pub fn to_radians(deg: f64) -> f64 {
        deg * DEG_TO_RAD
    }

    /// 将弧度转换为角度
    #[inline]
    pub fn to_degrees(rad: f64) -> f64 {
        rad * RAD_TO_DEG
    }
}

/// 标准值常数
pub mod standard {
    /// 标准重力加速度 [m/s²]
    pub const GRAVITY: f64 = 9.80665;

    /// 标准海水密度 [kg/m³]
    pub const SEAWATER_DENSITY: f64 = 1025.0;

    /// 标准淡水密度 [kg/m³]
    pub const FRESHWATER_DENSITY: f64 = 1000.0;

    /// 标准空气密度 [kg/m³]
    pub const AIR_DENSITY: f64 = 1.225;

    /// 标准大气压 [Pa]
    pub const ATMOSPHERIC_PRESSURE: f64 = 101325.0;

    /// 地球平均半径 [m]
    pub const EARTH_RADIUS: f64 = 6_371_000.0;
}

/// 验证阈值（用于 Fail-Fast 机制）
pub mod validation {
    /// 物理上合理的最大风速 [m/s]
    pub const MAX_REASONABLE_WIND_SPEED: f64 = 130.0;

    /// 物理上合理的最大水深 [m]
    pub const MAX_REASONABLE_DEPTH: f64 = 15_000.0;

    /// 物理上合理的最大流速 [m/s]
    pub const MAX_REASONABLE_VELOCITY: f64 = 100.0;

    /// 可疑地形高程上限 [m]
    pub const SUSPICIOUS_ELEVATION_HIGH: f64 = 8900.0;

    /// 可疑地形高程下限 [m]
    pub const SUSPICIOUS_ELEVATION_LOW: f64 = -12000.0;

    /// 检查风速是否合理
    #[inline]
    pub fn is_valid_wind_speed(speed: f64) -> bool {
        speed >= 0.0 && speed <= MAX_REASONABLE_WIND_SPEED
    }

    /// 检查水深是否合理
    #[inline]
    pub fn is_valid_depth(h: f64) -> bool {
        h >= 0.0 && h <= MAX_REASONABLE_DEPTH
    }

    /// 检查流速是否合理
    #[inline]
    pub fn is_valid_velocity(v: f64) -> bool {
        v.abs() <= MAX_REASONABLE_VELOCITY
    }

    /// 检查高程是否合理
    #[inline]
    pub fn is_valid_elevation(z: f64) -> bool {
        z >= SUSPICIOUS_ELEVATION_LOW && z <= SUSPICIOUS_ELEVATION_HIGH
    }
}

/// 数值容差
pub mod tolerance {
    /// 通用浮点比较极小值
    pub const EPSILON: f64 = 1e-9;

    /// 坐标变换容差
    pub const EPSILON_TRANSFORM: f64 = 1e-9;

    /// 权重归一化阈值
    pub const EPSILON_WEIGHT: f64 = 1e-6;

    /// 时间比较阈值 [s]
    pub const EPSILON_TIME: f64 = 1e-6;

    /// 机器精度时间步长底限 [s]
    pub const MIN_DT_MACHINE: f64 = 1e-10;

    /// 检查两个浮点数是否近似相等
    #[inline]
    pub fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    /// 检查两个浮点数是否近似相等（指定容差）
    #[inline]
    pub fn approx_eq_with_tol(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }
}

/// 默认物理参数值
pub mod defaults {
    /// 默认曼宁粗糙率 [s/m^(1/3)]
    pub const MANNING_N: f64 = 0.025;

    /// 默认水平涡粘系数 [m²/s]
    pub const EDDY_VISCOSITY: f64 = 1.0;

    /// 默认风拖曳系数
    pub const WIND_DRAG: f64 = 1.3e-3;

    /// 默认参考纬度 [deg]
    pub const LATITUDE_REF: f64 = 30.0;

    /// 默认最小水深阈值 [m]
    pub const H_MIN: f64 = 0.05;

    /// 默认地形高程 [m]（未指定时）
    pub const ELEVATION: f64 = -10.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coriolis_parameter() {
        let consts = PhysicalConstants::default();

        // 赤道处 f 应为 0
        let f_equator = consts.coriolis_parameter(0.0);
        assert!(f_equator.abs() < 1e-10);

        // 极地处 f 应最大
        let f_pole = consts.coriolis_parameter(PI / 2.0);
        assert!((f_pole - 2.0 * consts.omega).abs() < 1e-10);

        // 中纬度（30°）
        let f_mid = consts.coriolis_parameter_deg(30.0);
        assert!((f_mid - consts.omega).abs() < 1e-10); // sin(30°) = 0.5
    }

    #[test]
    fn test_wind_stress_factor() {
        let consts = PhysicalConstants::default();
        let factor = consts.wind_stress_factor();

        // 应该是一个小的正数
        assert!(factor > 0.0);
        assert!(factor < 1e-4);
    }

    #[test]
    fn test_shallow_water_speed() {
        let consts = PhysicalConstants::default();

        let c = consts.shallow_water_speed(10.0);
        // c = sqrt(9.81 * 10) ≈ 9.9
        assert!((c - (9.81 * 10.0_f64).sqrt()).abs() < 1e-6);

        // 负水深应返回 0
        let c_neg = consts.shallow_water_speed(-1.0);
        assert!((c_neg - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_froude_number() {
        let consts = PhysicalConstants::default();

        // 亚临界流
        let fr_sub = consts.froude_number(2.0, 10.0);
        assert!(fr_sub < 1.0);

        // 超临界流
        let fr_super = consts.froude_number(20.0, 1.0);
        assert!(fr_super > 1.0);
    }

    #[test]
    fn test_freshwater_constants() {
        let fresh = PhysicalConstants::freshwater();
        assert!((fresh.rho_water - 1000.0).abs() < 1e-10);

        let sea = PhysicalConstants::seawater();
        assert!((sea.rho_water - 1025.0).abs() < 1e-10);
    }

    #[test]
    fn test_angle_conversion() {
        use astronomy::{to_degrees, to_radians};

        let deg = 45.0;
        let rad = to_radians(deg);
        let deg_back = to_degrees(rad);

        assert!((deg - deg_back).abs() < 1e-10);
        assert!((rad - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_validation() {
        use validation::*;

        assert!(is_valid_wind_speed(10.0));
        assert!(!is_valid_wind_speed(200.0));

        assert!(is_valid_depth(100.0));
        assert!(!is_valid_depth(-1.0));

        assert!(is_valid_velocity(5.0));
        assert!(!is_valid_velocity(150.0));
    }

    #[test]
    fn test_tolerance() {
        use tolerance::*;

        assert!(approx_eq(1.0, 1.0 + 1e-10));
        assert!(!approx_eq(1.0, 1.1));

        assert!(approx_eq_with_tol(1.0, 1.5, 1.0));
        assert!(!approx_eq_with_tol(1.0, 1.5, 0.1));
    }
}
