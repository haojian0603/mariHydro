// src-tauri/src/marihydro/core/traits/flux.rs

//! 数值通量格式接口
//!
//! 定义 Riemann 求解器的统一抽象。

use super::state::ConservedState;
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::types::NumericalParams;
use glam::DVec2;

/// 面通量结果
#[derive(Debug, Clone, Copy, Default)]
pub struct FaceFlux {
    /// 质量通量 [m³/s]
    pub f_h: f64,
    /// x动量通量 [m³/s²]
    pub f_hu: f64,
    /// y动量通量 [m³/s²]
    pub f_hv: f64,
    /// 最大波速（用于CFL）
    pub max_wave_speed: f64,
}

impl FaceFlux {
    pub const ZERO: Self = Self {
        f_h: 0.0,
        f_hu: 0.0,
        f_hv: 0.0,
        max_wave_speed: 0.0,
    };

    /// 乘以面长度
    pub fn scale_by_length(&self, length: f64) -> Self {
        Self {
            f_h: self.f_h * length,
            f_hu: self.f_hu * length,
            f_hv: self.f_hv * length,
            max_wave_speed: self.max_wave_speed,
        }
    }
}

/// 面重构状态
#[derive(Debug, Clone, Copy)]
pub struct ReconstructedState {
    /// 水深
    pub h: f64,
    /// 速度
    pub vel: DVec2,
    /// 底床高程
    pub z: f64,
}

impl ReconstructedState {
    pub const fn new(h: f64, u: f64, v: f64, z: f64) -> Self {
        Self {
            h,
            vel: DVec2::new(u, v),
            z,
        }
    }

    /// 水位
    pub fn eta(&self) -> f64 {
        self.h + self.z
    }

    /// 法向速度分量
    pub fn normal_velocity(&self, normal: DVec2) -> f64 {
        self.vel.dot(normal)
    }

    /// 切向速度分量
    pub fn tangent_velocity(&self, normal: DVec2) -> f64 {
        // 切向量 = (-ny, nx)
        -self.vel.x * normal.y + self.vel.y * normal.x
    }
}

/// 通量计算格式接口
///
/// # 实现要求
///
/// 1. 必须处理干湿单元（h ≈ 0 的情况）
/// 2. 必须保证数值稳定性（避免除零、NaN等）
/// 3. 应支持静水重构以保持湖泊静止
pub trait FluxScheme: Send + Sync {
    /// 计算单个面的通量
    ///
    /// # 参数
    ///
    /// - `left`: 左侧（owner）重构状态
    /// - `right`: 右侧（neighbor）重构状态
    /// - `normal`: 面外法向量（从left指向right）
    /// - `g`: 重力加速度
    /// - `params`: 数值参数
    ///
    /// # 返回
    ///
    /// 法向数值通量（已考虑静水压力项）
    fn compute_flux(
        &self,
        left: &ReconstructedState,
        right: &ReconstructedState,
        normal: DVec2,
        g: f64,
        params: &NumericalParams,
    ) -> FaceFlux;

    /// 批量计算通量（可选优化实现）
    fn compute_fluxes_batch(
        &self,
        left_states: &[ReconstructedState],
        right_states: &[ReconstructedState],
        normals: &[DVec2],
        g: f64,
        params: &NumericalParams,
        output: &mut [FaceFlux],
    ) -> MhResult<()> {
        if left_states.len() != right_states.len()
            || left_states.len() != normals.len()
            || left_states.len() != output.len()
        {
            return Err(crate::marihydro::core::MhError::size_mismatch(
                "flux arrays",
                left_states.len(),
                output.len(),
            ));
        }

        for i in 0..left_states.len() {
            output[i] = self.compute_flux(&left_states[i], &right_states[i], normals[i], g, params);
        }
        Ok(())
    }

    /// 格式名称（用于日志和调试）
    fn name(&self) -> &'static str;

    /// 是否支持静水重构
    fn supports_hydrostatic_reconstruction(&self) -> bool {
        true
    }
}

/// 通量限制器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxLimiterType {
    /// 无限制（一阶精度）
    None,
    /// MinMod 限制器
    MinMod,
    /// Van Leer 限制器
    VanLeer,
    /// Superbee 限制器
    Superbee,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_flux() {
        let flux = FaceFlux {
            f_h: 1.0,
            f_hu: 2.0,
            f_hv: 3.0,
            max_wave_speed: 5.0,
        };

        let scaled = flux.scale_by_length(2.0);
        assert!((scaled.f_h - 2.0).abs() < 1e-10);
        assert!((scaled.f_hu - 4.0).abs() < 1e-10);
        assert!((scaled.max_wave_speed - 5.0).abs() < 1e-10); // 波速不缩放
    }

    #[test]
    fn test_reconstructed_state() {
        let state = ReconstructedState::new(2.0, 1.0, 0.5, -1.0);
        assert!((state.eta() - 1.0).abs() < 1e-10);

        let normal = DVec2::new(1.0, 0.0);
        assert!((state.normal_velocity(normal) - 1.0).abs() < 1e-10);
        assert!((state.tangent_velocity(normal) - 0.5).abs() < 1e-10);
    }
}
