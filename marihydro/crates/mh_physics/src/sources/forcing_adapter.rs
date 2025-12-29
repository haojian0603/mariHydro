// crates/mh_physics/src/sources/forcing_adapter.rs

//! 强迫项适配器
//!
//! 将 `forcing` 模块的数据提供者适配为 `sources` 模块的源项接口。
//!
//! # 设计思路
//!
//! - WindForcingAdapter: 将 WindProvider 包装为 SourceTerm
//! - TideForcingAdapter: 将 TideProvider 包装为边界源项
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::sources::forcing_adapter::WindForcingAdapter;
//! use mh_physics::forcing::WindProvider;
//! use mh_physics::sources::atmosphere::DragCoefficientMethod;
//!
//! let wind_provider = WindProvider::constant(10.0, 225.0);
//! let adapter = WindForcingAdapter::new(wind_provider, DragCoefficientMethod::Wu1982);
//! ```

use crate::forcing::wind::WindProvider;
use crate::sources::atmosphere::DragCoefficientMethod;
use crate::sources::traits::{
    SourceContribution, SourceContext, SourceTerm,
    SourceContributionGeneric, SourceContextGeneric, SourceStiffness, SourceTermGeneric,
};
use crate::state::{ShallowWaterStateF64, ShallowWaterStateGeneric};
use crate::core::CpuBackend;

/// 风场强迫适配器
///
/// 将 WindProvider 与 WindStress 源项结合
pub struct WindForcingAdapter {
    /// 风场数据提供者
    provider: WindProvider,
    /// 风阻系数计算方法
    drag_method: DragCoefficientMethod,
    /// 空气密度 [kg/m³]
    rho_air: f64,
    /// 水密度 [kg/m³] (预留用于潜在扩展)
    #[allow(dead_code)]
    rho_water: f64,
    /// 缓存的风速 (u, v)
    cached_wind: (f64, f64),
    /// 是否启用
    enabled: bool,
}

impl WindForcingAdapter {
    /// 创建新的风场强迫适配器
    pub fn new(provider: WindProvider, drag_method: DragCoefficientMethod) -> Self {
        Self {
            provider,
            drag_method,
            rho_air: 1.225,
            rho_water: 1000.0,
            cached_wind: (0.0, 0.0),
            enabled: true,
        }
    }

    /// 从恒定风场创建
    pub fn constant(speed: f64, direction_deg: f64, drag_method: DragCoefficientMethod) -> Self {
        Self::new(WindProvider::constant(speed, direction_deg), drag_method)
    }

    /// 更新风场到指定时间
    pub fn update(&mut self, time: f64) {
        self.cached_wind = self.provider.get_wind_at(time);
    }

    /// 获取当前风速
    pub fn wind(&self) -> (f64, f64) {
        self.cached_wind
    }

    /// 计算风阻系数
    fn drag_coefficient(&self, wind_speed: f64) -> f64 {
        self.drag_method.compute(wind_speed)
    }

    /// 计算风应力
    fn compute_stress(&self, wind_u: f64, wind_v: f64) -> (f64, f64) {
        let wind_speed = (wind_u * wind_u + wind_v * wind_v).sqrt();
        if wind_speed < 1e-6 {
            return (0.0, 0.0);
        }
        
        let cd = self.drag_coefficient(wind_speed);
        
        // τ = ρ_air × C_d × |W| × W
        let tau = self.rho_air * cd * wind_speed;
        (tau * wind_u, tau * wind_v)
    }
}

impl SourceTerm for WindForcingAdapter {
    fn name(&self) -> &'static str {
        "WindForcing"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        state: &ShallowWaterStateF64,
        cell: usize,
        ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h[cell];
        if ctx.is_dry(h) {
            return SourceContribution::ZERO;
        }

        let (wind_u, wind_v) = self.cached_wind;
        let (tau_x, tau_y) = self.compute_stress(wind_u, wind_v);

        // 动量源项 = τ / ρ_water (转换为 m²/s²)
        let rho_water = 1025.0;
        SourceContribution::momentum(tau_x / rho_water, tau_y / rho_water)
    }

    fn is_explicit(&self) -> bool {
        true
    }
}

/// 泛型版本
pub struct WindForcingAdapterGeneric<S> {
    /// 缓存的风应力 (τ_x/ρ, τ_y/ρ) [m²/s²]
    cached_stress: (S, S),
    /// 是否启用
    enabled: bool,
}

impl WindForcingAdapterGeneric<f64> {
    /// 创建并设置风应力
    pub fn new(tau_x: f64, tau_y: f64) -> Self {
        Self {
            cached_stress: (tau_x, tau_y),
            enabled: true,
        }
    }

    /// 更新风应力（从外部计算后设置）
    pub fn set_stress(&mut self, tau_x: f64, tau_y: f64) {
        self.cached_stress = (tau_x, tau_y);
    }
}

impl SourceTermGeneric<CpuBackend<f64>> for WindForcingAdapterGeneric<f64> {
    fn name(&self) -> &'static str {
        "WindForcing"
    }

    fn stiffness(&self) -> SourceStiffness {
        SourceStiffness::Explicit
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        ctx: &SourceContextGeneric<f64>,
    ) -> SourceContributionGeneric<f64> {
        let h = state.h[cell];
        if ctx.is_dry(h) {
            return SourceContributionGeneric::default();
        }

        let (tau_x, tau_y) = self.cached_stress;
        SourceContributionGeneric::momentum(tau_x, tau_y)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        _rhs_h: &mut Vec<f64>,
        rhs_hu: &mut Vec<f64>,
        rhs_hv: &mut Vec<f64>,
        ctx: &SourceContextGeneric<f64>,
    ) {
        if !self.enabled {
            return;
        }

        let (tau_x, tau_y) = self.cached_stress;
        for cell in 0..state.n_cells() {
            let h = state.h[cell];
            if !ctx.is_dry(h) {
                rhs_hu[cell] += tau_x;
                rhs_hv[cell] += tau_y;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::atmosphere::DragCoefficientMethod;

    #[test]
    fn test_wind_forcing_adapter() {
        let adapter = WindForcingAdapter::constant(10.0, 180.0, DragCoefficientMethod::Wu1982);
        
        assert_eq!(adapter.name(), "WindForcing");
        assert!(adapter.is_enabled());
    }
}
