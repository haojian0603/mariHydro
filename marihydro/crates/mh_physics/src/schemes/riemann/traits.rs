// crates/mh_physics/src/schemes/riemann/traits.rs

//! 黎曼求解器统一接口

use glam::DVec2;

use crate::types::NumericalParams;

/// 黎曼求解结果通量
#[derive(Debug, Clone, Copy, Default)]
pub struct RiemannFlux {
    /// 质量通量 [m²/s]
    pub mass: f64,
    /// x方向动量通量 [m³/s²]
    pub momentum_x: f64,
    /// y方向动量通量 [m³/s²]
    pub momentum_y: f64,
    /// 最大波速 [m/s]
    pub max_wave_speed: f64,
}

impl RiemannFlux {
    /// 零通量
    pub const ZERO: Self = Self {
        mass: 0.0,
        momentum_x: 0.0,
        momentum_y: 0.0,
        max_wave_speed: 0.0,
    };

    /// 创建通量
    pub fn new(mass: f64, momentum_x: f64, momentum_y: f64, max_wave_speed: f64) -> Self {
        Self {
            mass,
            momentum_x,
            momentum_y,
            max_wave_speed,
        }
    }

    /// 动量向量
    #[inline]
    pub fn momentum(&self) -> DVec2 {
        DVec2::new(self.momentum_x, self.momentum_y)
    }

    /// 从旋转坐标系转换到全局坐标系
    ///
    /// # 参数
    /// - `mass`: 质量通量
    /// - `flux_n`: 法向动量通量
    /// - `flux_t`: 切向动量通量
    /// - `normal`: 法向量
    /// - `max_wave_speed`: 最大波速
    pub fn from_rotated(
        mass: f64,
        flux_n: f64,
        flux_t: f64,
        normal: DVec2,
        max_wave_speed: f64,
    ) -> Self {
        let tangent = DVec2::new(-normal.y, normal.x);
        let momentum = normal * flux_n + tangent * flux_t;
        Self {
            mass,
            momentum_x: momentum.x,
            momentum_y: momentum.y,
            max_wave_speed,
        }
    }

    /// 缩放通量
    pub fn scaled(self, factor: f64) -> Self {
        Self {
            mass: self.mass * factor,
            momentum_x: self.momentum_x * factor,
            momentum_y: self.momentum_y * factor,
            max_wave_speed: self.max_wave_speed,
        }
    }

    /// 检查数值有效性
    pub fn is_valid(&self) -> bool {
        self.mass.is_finite()
            && self.momentum_x.is_finite()
            && self.momentum_y.is_finite()
            && self.max_wave_speed.is_finite()
            && self.max_wave_speed >= 0.0
    }
}

/// 求解器能力标志
#[derive(Debug, Clone, Copy, Default)]
pub struct SolverCapabilities {
    /// 是否支持干湿处理
    pub handles_dry_wet: bool,
    /// 是否包含熵修正
    pub has_entropy_fix: bool,
    /// 是否支持静水重构
    pub supports_hydrostatic: bool,
    /// 精度阶数
    pub order: u8,
    /// 是否保正
    pub positivity_preserving: bool,
}

/// 求解器参数
#[derive(Debug, Clone, Copy)]
pub struct SolverParams {
    /// 重力加速度 [m/s²]
    pub gravity: f64,
    /// 干湿阈值 [m]
    pub h_dry: f64,
    /// 最小水深 [m]
    pub h_min: f64,
    /// 通量零阈值
    pub flux_eps: f64,
    /// 熵修正比例
    pub entropy_ratio: f64,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            h_dry: 1e-6,
            h_min: 1e-9,
            flux_eps: 1e-14,
            entropy_ratio: 0.1,
        }
    }
}

impl SolverParams {
    /// 从 NumericalParams 创建
    pub fn from_numerical(params: &NumericalParams, gravity: f64) -> Self {
        Self {
            gravity,
            h_dry: params.h_dry,
            h_min: params.h_min,
            flux_eps: params.flux_eps,
            entropy_ratio: params.entropy_ratio,
        }
    }

    /// 计算熵修正阈值
    #[inline]
    pub fn entropy_threshold(&self, wave_speed_range: f64) -> f64 {
        (self.entropy_ratio * wave_speed_range.abs()).max(self.flux_eps)
    }
}

/// 黎曼求解器 trait
pub trait RiemannSolver: Send + Sync {
    /// 求解器名称
    fn name(&self) -> &'static str;

    /// 求解器能力
    fn capabilities(&self) -> SolverCapabilities;

    /// 求解黎曼问题
    ///
    /// # 参数
    /// - `h_left`: 左侧水深
    /// - `h_right`: 右侧水深
    /// - `vel_left`: 左侧速度向量
    /// - `vel_right`: 右侧速度向量
    /// - `normal`: 界面法向量（指向右侧）
    ///
    /// # 返回
    /// 界面数值通量
    fn solve(
        &self,
        h_left: f64,
        h_right: f64,
        vel_left: DVec2,
        vel_right: DVec2,
        normal: DVec2,
    ) -> Result<RiemannFlux, RiemannError>;

    /// 重力加速度
    fn gravity(&self) -> f64;

    /// 干湿阈值
    fn dry_threshold(&self) -> f64;
}

/// 黎曼求解器错误
#[derive(Debug, Clone)]
pub enum RiemannError {
    /// 数值错误
    Numerical { message: String },
    /// 无效输入
    InvalidInput { message: String },
}

impl std::fmt::Display for RiemannError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Numerical { message } => write!(f, "Numerical error: {}", message),
            Self::InvalidInput { message } => write!(f, "Invalid input: {}", message),
        }
    }
}

impl std::error::Error for RiemannError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_riemann_flux_zero() {
        let flux = RiemannFlux::ZERO;
        assert_eq!(flux.mass, 0.0);
        assert_eq!(flux.momentum_x, 0.0);
        assert_eq!(flux.momentum_y, 0.0);
        assert!(flux.is_valid());
    }

    #[test]
    fn test_riemann_flux_from_rotated() {
        // 沿 x 轴法向
        let flux = RiemannFlux::from_rotated(1.0, 2.0, 3.0, DVec2::X, 5.0);
        assert_eq!(flux.mass, 1.0);
        assert_eq!(flux.momentum_x, 2.0);
        assert_eq!(flux.momentum_y, 3.0);

        // 沿 y 轴法向
        let flux = RiemannFlux::from_rotated(1.0, 2.0, 3.0, DVec2::Y, 5.0);
        assert_eq!(flux.mass, 1.0);
        assert_eq!(flux.momentum_y, 2.0);
        assert!((flux.momentum_x - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_solver_params() {
        let params = SolverParams::default();
        assert_eq!(params.gravity, 9.81);

        let threshold = params.entropy_threshold(10.0);
        assert!(threshold > 0.0);
    }
}
