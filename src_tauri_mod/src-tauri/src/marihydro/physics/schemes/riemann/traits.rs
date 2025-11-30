// src-tauri/src/marihydro/physics/schemes/riemann/traits.rs

//! 黎曼求解器统一接口
//!
//! 定义所有黎曼求解器必须实现的 trait。

use super::InterfaceState;
use crate::marihydro::core::error::MhResult;
use glam::DVec2;

/// 黎曼求解结果通量
#[derive(Debug, Clone, Copy, Default)]
pub struct RiemannFlux {
    /// 质量通量 (m²/s)
    pub mass: f64,
    /// x方向动量通量 (m³/s²)
    pub momentum_x: f64,
    /// y方向动量通量 (m³/s²)
    pub momentum_y: f64,
    /// 最大波速 (m/s)
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
    /// - `flux_n`: 法向通量
    /// - `flux_t`: 切向通量
    /// - `normal`: 法向量
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
    /// 理论精度阶数
    pub order: u8,
    /// 是否保持正性
    pub positivity_preserving: bool,
}

impl SolverCapabilities {
    /// 基本能力（一阶）
    pub fn basic() -> Self {
        Self {
            handles_dry_wet: false,
            has_entropy_fix: false,
            supports_hydrostatic: false,
            order: 1,
            positivity_preserving: false,
        }
    }

    /// 完整能力（用于 HLLC）
    pub fn full() -> Self {
        Self {
            handles_dry_wet: true,
            has_entropy_fix: true,
            supports_hydrostatic: true,
            order: 2,
            positivity_preserving: true,
        }
    }
}

/// 黎曼求解器 trait
///
/// 所有黎曼求解器都必须实现此 trait。
pub trait RiemannSolver: Send + Sync {
    /// 求解器名称
    fn name(&self) -> &'static str;

    /// 求解器能力
    fn capabilities(&self) -> SolverCapabilities;

    /// 求解界面黎曼问题
    ///
    /// # 参数
    /// - `h_left`: 左侧水深
    /// - `h_right`: 右侧水深
    /// - `vel_left`: 左侧速度向量
    /// - `vel_right`: 右侧速度向量
    /// - `normal`: 界面法向量（从左指向右）
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
    ) -> MhResult<RiemannFlux>;

    /// 使用界面状态求解
    fn solve_state(&self, state: &InterfaceState) -> MhResult<RiemannFlux> {
        self.solve(
            state.h_left,
            state.h_right,
            state.vel_left,
            state.vel_right,
            state.normal,
        )
    }

    /// 重力加速度
    fn gravity(&self) -> f64;

    /// 干燥阈值
    fn dry_threshold(&self) -> f64;

    /// 批量求解（可优化实现）
    fn solve_batch(
        &self,
        states: &[InterfaceState],
        fluxes: &mut [RiemannFlux],
    ) -> MhResult<()> {
        debug_assert_eq!(states.len(), fluxes.len());

        for (state, flux) in states.iter().zip(fluxes.iter_mut()) {
            *flux = self.solve_state(state)?;
        }
        Ok(())
    }
}

/// 可配置的求解器参数
#[derive(Debug, Clone, Copy)]
pub struct SolverParams {
    /// 重力加速度
    pub gravity: f64,
    /// 干燥阈值
    pub h_dry: f64,
    /// 最小水深
    pub h_min: f64,
    /// 熵修正比例
    pub entropy_ratio: f64,
    /// 通量稳定化参数
    pub flux_eps: f64,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            h_dry: 1e-4,
            h_min: 1e-6,
            entropy_ratio: 0.1,
            flux_eps: 1e-12,
        }
    }
}

impl SolverParams {
    /// 从 NumericalParams 转换
    pub fn from_numerical(params: &crate::marihydro::core::types::NumericalParams, g: f64) -> Self {
        Self {
            gravity: g,
            h_dry: params.h_dry,
            h_min: params.h_min,
            entropy_ratio: params.entropy_ratio,
            flux_eps: params.flux_eps,
        }
    }

    /// 计算熵阈值
    #[inline]
    pub fn entropy_threshold(&self, speed_range: f64) -> f64 {
        self.entropy_ratio * speed_range + self.flux_eps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_riemann_flux() {
        let flux = RiemannFlux::new(1.0, 2.0, 3.0, 5.0);
        assert!(flux.is_valid());

        let scaled = flux.scaled(0.5);
        assert!((scaled.mass - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_from_rotated() {
        let normal = DVec2::X;
        let flux = RiemannFlux::from_rotated(1.0, 2.0, 3.0, normal, 5.0);

        assert!((flux.momentum_x - 2.0).abs() < 1e-10);
        assert!((flux.momentum_y - 3.0).abs() < 1e-10);
    }
}
