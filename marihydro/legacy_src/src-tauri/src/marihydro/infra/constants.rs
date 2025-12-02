// src-tauri/src/marihydro/infra/constants.rs
use std::f64::consts::PI;

pub mod physics {
    use super::PI;
    
    // === 地球与天文 ===
    pub const EARTH_ROTATION_RATE_RAD: f64 = 7.292115e-5;
    pub const STANDARD_GRAVITY: f64 = 9.80665;
    pub const EARTH_RADIUS: f64 = 6_371_000.0;
    pub const SECONDS_PER_DAY: f64 = 86400.0;
    pub const DEG_TO_RAD: f64 = PI / 180.0;
    pub const RAD_TO_DEG: f64 = 180.0 / PI;
    
    // === 流体密度 ===
    pub const STD_SEAWATER_DENSITY: f64 = 1025.0;
    pub const STD_FRESHWATER_DENSITY: f64 = 1000.0;
    pub const STD_AIR_DENSITY: f64 = 1.225;
    pub const STD_ATM_PRESSURE: f64 = 101325.0;
    
    // === 流体粘性（动力粘度，Pa·s） ===
    pub const FRESHWATER_VISCOSITY: f64 = 0.001002;  // 20°C
    pub const SEAWATER_VISCOSITY: f64 = 0.00108;     // 20°C, 35 PSU
    pub const AIR_VISCOSITY: f64 = 1.825e-5;         // 20°C
    
    // === 运动粘度（m²/s）===
    pub const FRESHWATER_KINEMATIC_VISCOSITY: f64 = 1.002e-6;  // 20°C
    pub const SEAWATER_KINEMATIC_VISCOSITY: f64 = 1.054e-6;    // 20°C, 35 PSU
    
    // === 泥沙相关 ===
    pub const QUARTZ_DENSITY: f64 = 2650.0;           // 石英密度 kg/m³
    pub const CLAY_DENSITY: f64 = 2600.0;             // 黏土密度 kg/m³
    pub const ORGANIC_MATTER_DENSITY: f64 = 1500.0;   // 有机质密度 kg/m³
    
    // Ferguson-Church 沉速公式系数
    // w_s = R * g * d² / (C1 * ν + (0.75 * C2 * R * g * d³)^0.5)
    pub const FERGUSON_CHURCH_C1: f64 = 18.0;   // 层流系数
    pub const FERGUSON_CHURCH_C2: f64 = 1.0;    // 紊流系数（天然泥沙）
    pub const FERGUSON_CHURCH_C2_SMOOTH: f64 = 0.4;  // 光滑球形颗粒
    
    // van Rijn 沉速公式系数
    pub const VAN_RIJN_A: f64 = 10.0;
    pub const VAN_RIJN_B: f64 = 0.01;
    
    // Shields 参数
    pub const SHIELDS_CRITICAL: f64 = 0.047;    // 临界希尔兹数（中值）
    pub const SHIELDS_CRITICAL_MIN: f64 = 0.03; // 细沙
    pub const SHIELDS_CRITICAL_MAX: f64 = 0.06; // 粗沙
    
    // === 湍流常数 ===
    pub const VON_KARMAN: f64 = 0.41;           // 卡门常数
    pub const KOLMOGOROV_CONSTANT: f64 = 1.5;   // Kolmogorov 常数
    
    // k-ε 模型标准常数
    pub const K_EPSILON_C_MU: f64 = 0.09;
    pub const K_EPSILON_C1: f64 = 1.44;
    pub const K_EPSILON_C2: f64 = 1.92;
    pub const K_EPSILON_SIGMA_K: f64 = 1.0;
    pub const K_EPSILON_SIGMA_E: f64 = 1.3;
    
    // === 波浪相关 ===
    pub const JONSWAP_GAMMA: f64 = 3.3;         // JONSWAP 峰值增强因子
    pub const PIERSON_MOSKOWITZ_ALPHA: f64 = 0.0081;  // P-M 谱常数
    
    // === 风应力 ===
    // Wu (1982) 公式系数
    // Cd = (0.8 + 0.065 * U10) * 1e-3, for U10 > 1 m/s
    pub const WU_DRAG_A: f64 = 0.8e-3;
    pub const WU_DRAG_B: f64 = 0.065e-3;
    
    // Large-Pond (1981) 公式
    pub const LARGE_POND_C1: f64 = 1.2e-3;      // U10 < 11 m/s
    pub const LARGE_POND_C2: f64 = 0.49e-3;     // 11 <= U10 < 25 m/s
    pub const LARGE_POND_SLOPE: f64 = 0.065e-3; // 斜率
    
    // === 热力学 ===
    pub const WATER_SPECIFIC_HEAT: f64 = 4186.0;  // J/(kg·K)
    pub const AIR_SPECIFIC_HEAT: f64 = 1005.0;    // J/(kg·K)
    pub const LATENT_HEAT_VAPORIZATION: f64 = 2.45e6;  // J/kg at 20°C
    pub const STEFAN_BOLTZMANN: f64 = 5.67e-8;    // W/(m²·K⁴)
}

pub mod validation {
    pub const MAX_REASONABLE_WIND_SPEED: f64 = 130.0;
    pub const MAX_REASONABLE_DEPTH: f64 = 15_000.0;
    pub const MAX_REASONABLE_VELOCITY: f64 = 100.0;
    pub const SUSPICIOUS_ELEVATION_HIGH: f64 = 8900.0;
    pub const MAX_SCALE_FACTOR: f64 = 1e6;
    pub const MIN_SCALE_FACTOR: f64 = 1e-6;
    pub const MIN_ACTIVE_RATIO: f64 = 0.001;
    
    // 泥沙验证
    pub const MAX_SEDIMENT_CONCENTRATION: f64 = 0.6;  // 最大体积浓度
    pub const MAX_BED_SLOPE: f64 = 0.5;               // 最大床面坡度 (约26.6°)
    pub const MIN_GRAIN_SIZE: f64 = 1e-6;             // 最小粒径 1μm
    pub const MAX_GRAIN_SIZE: f64 = 0.1;              // 最大粒径 100mm
}

pub mod defaults {
    pub const GHOST_WIDTH: usize = 2;
    pub const H_MIN: f64 = 0.05;
    pub const ELEVATION: f64 = -10.0;
    pub const MANNING_N: f64 = 0.025;
    pub const EDDY_VISCOSITY: f64 = 1.0;
    pub const SEDIMENT_W_S: f64 = 0.001;
    pub const SEDIMENT_TAU_CR: f64 = 0.1;
    pub const CFL: f64 = 0.9;
    pub const MAX_DT: f64 = 60.0;
    pub const MIN_DT: f64 = 1e-6;
    
    // 干湿处理默认值
    pub const DRY_DEPTH: f64 = 0.01;
    pub const WET_DEPTH: f64 = 0.05;
    pub const TRANSITION_DEPTH: f64 = 0.03;
    
    // 湍流默认值
    pub const MIN_TKE: f64 = 1e-10;
    pub const MIN_EPSILON: f64 = 1e-14;
}
