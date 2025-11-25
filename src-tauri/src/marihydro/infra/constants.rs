// src-tauri/src/marihydro/infra/constants.rs

use std::f64::consts::PI;

/// 物理常数 (Immutable Physics)
/// 包含地球物理、流体性质等自然界铁律。
pub mod physics {
    use super::PI;

    // --- 天文与地理 ---

    /// 地球自转角速度 [rad/s] (GRS80)
    /// 用于计算科氏参数 f = 2Ω sin(φ)
    pub const EARTH_ROTATION_RATE_RAD: f64 = 7.292115e-5;

    /// 标准重力加速度 [m/s^2]
    pub const STANDARD_GRAVITY: f64 = 9.80665;

    /// 地球平均半径 [m]
    pub const EARTH_RADIUS: f64 = 6_371_000.0;

    /// 一天的秒数
    pub const SECONDS_PER_DAY: f64 = 86400.0;

    // --- 角度转换 ---

    /// 角度转弧度系数 (π / 180)
    pub const DEG_TO_RAD: f64 = PI / 180.0;

    /// 弧度转角度系数 (1 / DEG_TO_RAD)
    /// 使用倒数定义以保证数学一致性
    pub const RAD_TO_DEG: f64 = 1.0 / DEG_TO_RAD;

    // --- 流体标准性质 ---

    /// 标准海水密度 [kg/m^3] (3.5% 盐度, 15°C)
    pub const STD_SEAWATER_DENSITY: f64 = 1025.0;

    /// 标准淡水密度 [kg/m^3] (4°C)
    pub const STD_FRESHWATER_DENSITY: f64 = 1000.0;

    /// 标准空气密度 [kg/m^3] (海平面, 15°C)
    pub const STD_AIR_DENSITY: f64 = 1.225;

    /// 标准大气压 [Pa]
    pub const STD_ATM_PRESSURE: f64 = 101325.0;
}

/// 验证阈值 (Validation Thresholds)
/// 用于 Fail-Fast 机制，检测输入数据或计算结果是否违背物理常识。
pub mod validation {
    /// 物理上合理的最大风速 [m/s]
    /// 地球表面实测最大阵风约 113 m/s (408 km/h)
    pub const MAX_REASONABLE_WIND_SPEED: f64 = 130.0;

    /// 物理上合理的最大水深 [m]
    /// 马里亚纳海沟深约 11km，设置 15km 为安全上限
    pub const MAX_REASONABLE_DEPTH: f64 = 15_000.0;

    /// 物理上合理的最大流速 [m/s]
    /// 超过 100 m/s 必定是数值爆炸
    pub const MAX_REASONABLE_VELOCITY: f64 = 100.0;

    /// 可疑地形高程上限 [m]
    /// 如果地形高于此值，系统会发出警告（可能是单位错误或坐标系错误）
    pub const SUSPICIOUS_ELEVATION_HIGH: f64 = 8900.0; // 略高于珠峰

    /// 线性变换系数上限
    /// 防止 VariableMapping 中输入了错误的 scale_factor
    pub const MAX_SCALE_FACTOR: f64 = 1e6;
    pub const MIN_SCALE_FACTOR: f64 = 1e-6;

    /// 网格激活率警告阈值
    /// 如果有效计算单元占比低于此值，可能意味着掩膜设置错误
    pub const MIN_ACTIVE_RATIO: f64 = 0.001;
}

/// 系统默认值 (Defaults)
/// 用于 `ProjectManifest` 初始化或数据缺失时的回退。
pub mod defaults {
    /// 幽灵层厚度 (Ghost Cell Width)
    /// MUSCL 二阶重构需要 2 层
    pub const GHOST_WIDTH: usize = 2;

    /// 最小水深阈值 [m] (Dry/Wet threshold)
    pub const H_MIN: f64 = 0.05;

    /// 默认地形高程 [m]
    pub const ELEVATION: f64 = -10.0;

    /// 默认曼宁粗糙率 [s/m^(1/3)]
    pub const MANNING_N: f64 = 0.025;

    /// 默认水平涡粘系数 [m^2/s]
    pub const EDDY_VISCOSITY: f64 = 1.0;

    /// 默认泥沙沉降速度 [m/s]
    pub const SEDIMENT_W_S: f64 = 0.001;

    /// 默认泥沙临界切应力 [N/m^2]
    pub const SEDIMENT_TAU_CR: f64 = 0.1;

    /// 默认参考纬度 [deg]
    pub const LATITUDE_REF: f64 = 30.0;

    /// 默认时间步长 (CFL 计算前) [s]
    pub const INITIAL_DT: f64 = 1.0;
}

/// 数值容差 (Numerical Tolerances)
/// 用于浮点数比较和收敛判断。
pub mod tolerances {
    /// 通用浮点比较极小值
    pub const EPSILON: f64 = 1e-9;

    /// 坐标变换容差
    pub const EPSILON_TRANSFORM: f64 = 1e-9;

    /// 权重归一化阈值
    /// 空间插值时，如果总权重小于此值，视为插值无效
    pub const EPSILON_WEIGHT: f64 = 1e-6;

    /// 时间比较阈值 [s]
    pub const EPSILON_TIME: f64 = 1e-6;

    /// 机器精度时间步长底限 [s]
    /// 防止 dt 过小导致停滞
    pub const MIN_DT_MACHINE: f64 = 1e-10;
}

// --- 辅助函数 ---

/// 将角度转换为弧度
#[inline(always)]
pub fn to_radians(deg: f64) -> f64 {
    deg * physics::DEG_TO_RAD
}

/// 将弧度转换为角度
#[inline(always)]
pub fn to_degrees(rad: f64) -> f64 {
    rad * physics::RAD_TO_DEG
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_consistency() {
        // 测试角度转换的互逆性
        let deg = 45.0;
        let rad = to_radians(deg);
        let deg_back = to_degrees(rad);
        assert!((deg - deg_back).abs() < tolerances::EPSILON);

        // 测试乘积为 1
        assert!((physics::DEG_TO_RAD * physics::RAD_TO_DEG - 1.0).abs() < tolerances::EPSILON);
    }

    #[test]
    fn test_coriolis_magnitude() {
        // 赤道处 f 应为 0
        let f_equator = 2.0 * physics::EARTH_ROTATION_RATE_RAD * to_radians(0.0).sin();
        assert!(f_equator.abs() < tolerances::EPSILON);

        // 极地处 f 应最大
        let f_pole = 2.0 * physics::EARTH_ROTATION_RATE_RAD * to_radians(90.0).sin();
        assert!((f_pole - 2.0 * physics::EARTH_ROTATION_RATE_RAD).abs() < tolerances::EPSILON);
    }
}
