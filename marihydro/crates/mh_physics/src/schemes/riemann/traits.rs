// crates/mh_physics/src/schemes/riemann/traits.rs

//! 黎曼求解器统一接口（泛型化）
//!
//! T=3 改造：将所有核心类型泛型化以支持 f32/f64 精度切换

use mh_runtime::{Backend, RuntimeScalar};
use mh_runtime::Vector2D;

/// 黎曼求解结果通量（泛型化）
#[derive(Debug, Clone, Copy)]
pub struct RiemannFlux<S: RuntimeScalar> {
    /// 质量通量 [m²/s]
    pub mass: S,
    /// x方向动量通量 [m³/s²]
    pub momentum_x: S,
    /// y方向动量通量 [m³/s²]
    pub momentum_y: S,
    /// 最大波速 [m/s]
    pub max_wave_speed: S,
}

impl<S: RuntimeScalar> Default for RiemannFlux<S> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<S: RuntimeScalar> RiemannFlux<S> {
    /// 零通量
    pub fn zero() -> Self {
        Self {
            mass: S::ZERO,
            momentum_x: S::ZERO,
            momentum_y: S::ZERO,
            max_wave_speed: S::ZERO,
        }
    }

    /// 创建通量
    pub fn new(mass: S, momentum_x: S, momentum_y: S, max_wave_speed: S) -> Self {
        Self {
            mass,
            momentum_x,
            momentum_y,
            max_wave_speed,
        }
    }

    /// 从旋转坐标系转换到全局坐标系
    ///
    /// # 参数
    /// - `mass`: 质量通量
    /// - `flux_n`: 法向动量通量
    /// - `flux_t`: 切向动量通量
    /// - `normal`: 法向量 (使用 Backend 的 Vector2D)
    /// - `max_wave_speed`: 最大波速
    /// - `_gravity`: 重力加速度（用于类型推断，某些实现可能需要）
    pub fn from_rotated<B: Backend<Scalar = S>>(
        mass: S,
        flux_n: S,
        flux_t: S,
        normal: B::Vector2D,
        max_wave_speed: S,
        _gravity: S,
    ) -> Self {
        // 计算切向量
        let tangent = B::vec2_new(-normal.y(), normal.x());
        // 动量 = normal * flux_n + tangent * flux_t
        let momentum_x = normal.x() * flux_n + tangent.x() * flux_t;
        let momentum_y = normal.y() * flux_n + tangent.y() * flux_t;
        Self {
            mass,
            momentum_x,
            momentum_y,
            max_wave_speed,
        }
    }

    /// 缩放通量
    pub fn scaled(self, factor: S) -> Self {
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
            && self.max_wave_speed >= S::ZERO
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

/// 求解器参数（泛型化）
#[derive(Debug, Clone, Copy)]
pub struct SolverParams<S: RuntimeScalar> {
    /// 重力加速度 [m/s²]
    pub gravity: S,
    /// 干湿阈值 [m]
    pub h_dry: S,
    /// 最小水深 [m]
    pub h_min: S,
    /// 通量零阈值
    pub flux_eps: S,
    /// 熵修正比例
    pub entropy_ratio: S,
}

impl<S: RuntimeScalar> Default for SolverParams<S> {
    fn default() -> Self {
        Self {
            gravity: S::from_f64(9.81).unwrap_or(S::ZERO),
            h_dry: S::from_f64(1e-6).unwrap_or(S::ZERO),
            h_min: S::from_f64(1e-9).unwrap_or(S::ZERO),
            flux_eps: S::from_f64(1e-14).unwrap_or(S::ZERO),
            entropy_ratio: S::from_f64(0.1).unwrap_or(S::ZERO),
        }
    }
}

impl<S: RuntimeScalar> SolverParams<S> {
    /// 从 NumericalParams 创建
    pub fn from_numerical(params: &crate::types::NumericalParams<S>, gravity: S) -> Self {
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
    pub fn entropy_threshold(&self, wave_speed_range: S) -> S {
        (self.entropy_ratio * wave_speed_range.abs()).max(self.flux_eps)
    }
}

/// 黎曼求解器 trait（泛型化，使用 Backend）
///
/// 实现此 trait 的求解器可以处理任意 Backend 的几何类型
pub trait RiemannSolver: Send + Sync {
    /// 求解器使用的标量类型
    type Scalar: RuntimeScalar;
    /// 求解器使用的二维向量类型
    type Vector2D: Copy + Send + Sync;

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
        h_left: Self::Scalar,
        h_right: Self::Scalar,
        vel_left: Self::Vector2D,
        vel_right: Self::Vector2D,
        normal: Self::Vector2D,
    ) -> Result<RiemannFlux<Self::Scalar>, RiemannError>;

    /// 重力加速度
    fn gravity(&self) -> Self::Scalar;

    /// 干湿阈值
    fn dry_threshold(&self) -> Self::Scalar;
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

// ============================================================================
// 向后兼容类型别名
// ============================================================================

/// f64 版本的 RiemannFlux（向后兼容）
pub type RiemannFluxF64 = RiemannFlux<f64>;

/// f32 版本的 RiemannFlux
pub type RiemannFluxF32 = RiemannFlux<f32>;

/// f64 版本的 SolverParams（向后兼容）
pub type SolverParamsF64 = SolverParams<f64>;

/// f32 版本的 SolverParams
pub type SolverParamsF32 = SolverParams<f32>;

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_riemann_flux_zero_f64() {
        let flux = RiemannFlux::<f64>::zero();
        assert_eq!(flux.mass, 0.0);
        assert_eq!(flux.momentum_x, 0.0);
        assert_eq!(flux.momentum_y, 0.0);
        assert!(flux.is_valid());
    }

    #[test]
    fn test_riemann_flux_zero_f32() {
        let flux = RiemannFlux::<f32>::zero();
        assert_eq!(flux.mass, 0.0f32);
        assert_eq!(flux.momentum_x, 0.0f32);
        assert_eq!(flux.momentum_y, 0.0f32);
        assert!(flux.is_valid());
    }

    #[test]
    fn test_riemann_flux_from_rotated_f64() {
        // 沿 x 轴法向
        let normal = CpuBackend::<f64>::vec2_new(1.0, 0.0);
        let flux = RiemannFlux::from_rotated::<CpuBackend<f64>>(1.0, 2.0, 3.0, normal, 5.0, 9.81);
        assert_eq!(flux.mass, 1.0);
        assert!((flux.momentum_x - 2.0).abs() < 1e-10);
        assert!((flux.momentum_y - 3.0).abs() < 1e-10);

        // 沿 y 轴法向
        let normal_y = CpuBackend::<f64>::vec2_new(0.0, 1.0);
        let flux_y = RiemannFlux::from_rotated::<CpuBackend<f64>>(1.0, 2.0, 3.0, normal_y, 5.0, 9.81);
        assert_eq!(flux_y.mass, 1.0);
        assert!((flux_y.momentum_y - 2.0).abs() < 1e-10);
        assert!((flux_y.momentum_x - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_solver_params_f64() {
        let params = SolverParams::<f64>::default();
        assert_eq!(params.gravity, 9.81);

        let threshold = params.entropy_threshold(10.0);
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_solver_params_f32() {
        let params = SolverParams::<f32>::default();
        assert!((params.gravity - 9.81f32).abs() < 1e-5);
    }
}
