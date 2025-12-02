// src-tauri/src/marihydro/domain/boundary/ghost.rs

//! Ghost 状态计算

use glam::DVec2;

use super::types::{BoundaryKind, BoundaryParams, ExternalForcing};
use crate::marihydro::core::traits::state::ConservedState;
use crate::marihydro::core::types::NumericalParams;

/// Ghost 状态计算器
pub struct GhostStateCalculator {
    params: BoundaryParams,
}

impl GhostStateCalculator {
    /// 创建计算器
    pub fn new(params: BoundaryParams) -> Self {
        Self { params }
    }

    /// 从数值参数创建
    pub fn from_numerical_params(numerical_params: &NumericalParams, gravity: f64) -> Self {
        Self::new(BoundaryParams::new(gravity, numerical_params.h_min))
    }

    /// 计算 ghost 状态
    pub fn compute_ghost_state(
        &self,
        interior_state: ConservedState,
        interior_z: f64,
        normal: DVec2,
        kind: BoundaryKind,
        forcing: Option<&ExternalForcing>,
    ) -> ConservedState {
        match kind {
            BoundaryKind::Wall | BoundaryKind::Symmetry => self.wall_ghost(interior_state, normal),
            BoundaryKind::OpenSea => {
                if let Some(f) = forcing {
                    self.open_sea_ghost(interior_state, interior_z, normal, f)
                } else {
                    self.outflow_ghost(interior_state)
                }
            }
            BoundaryKind::Outflow => self.outflow_ghost(interior_state),
            BoundaryKind::RiverInflow => {
                if let Some(f) = forcing {
                    self.inflow_ghost(interior_state, f)
                } else {
                    self.outflow_ghost(interior_state)
                }
            }
            BoundaryKind::Periodic => {
                // 周期边界需要查找对应单元
                interior_state
            }
        }
    }

    /// 固壁边界 ghost 状态（反射）
    fn wall_ghost(&self, interior: ConservedState, normal: DVec2) -> ConservedState {
        let h = interior.h;

        if h < self.params.h_min {
            return ConservedState::ZERO;
        }

        // 计算速度
        let u = interior.hu / h;
        let v = interior.hv / h;

        // 法向速度分量
        let un = u * normal.x + v * normal.y;

        // 反射法向速度，保留切向速度
        let u_ghost = u - 2.0 * un * normal.x;
        let v_ghost = v - 2.0 * un * normal.y;

        ConservedState::new(h, h * u_ghost, h * v_ghost)
    }

    /// 开海边界 ghost 状态（Flather）
    fn open_sea_ghost(
        &self,
        interior: ConservedState,
        interior_z: f64,
        normal: DVec2,
        forcing: &ExternalForcing,
    ) -> ConservedState {
        let h_int = interior.h;

        if h_int < self.params.h_min {
            // 干单元，使用外部水位
            let h_ext = (forcing.eta - interior_z).max(0.0);
            return ConservedState::new(h_ext, 0.0, 0.0);
        }

        let u_int = interior.hu / h_int;
        let v_int = interior.hv / h_int;

        // 特征速度
        let c = self.params.sqrt_g * h_int.sqrt();

        // 内部水位
        let eta_int = h_int + interior_z;

        // Flather: 使用特征关系
        let un_int = u_int * normal.x + v_int * normal.y;
        let un_ext = forcing.u * normal.x + forcing.v * normal.y;

        // ghost 水位
        let eta_ghost = forcing.eta + (h_int / c) * (un_int - un_ext);
        let h_ghost = (eta_ghost - interior_z).max(self.params.h_min);

        // ghost 速度（保持切向速度）
        let ut_int = -u_int * normal.y + v_int * normal.x; // 切向分量
        let un_ghost = un_ext; // 使用外部法向速度

        let u_ghost = un_ghost * normal.x - ut_int * normal.y;
        let v_ghost = un_ghost * normal.y + ut_int * normal.x;

        ConservedState::new(h_ghost, h_ghost * u_ghost, h_ghost * v_ghost)
    }

    /// 自由出流 ghost 状态（零梯度）
    fn outflow_ghost(&self, interior: ConservedState) -> ConservedState {
        interior
    }

    /// 入流边界 ghost 状态
    fn inflow_ghost(&self, interior: ConservedState, forcing: &ExternalForcing) -> ConservedState {
        let h = interior.h.max(self.params.h_min);

        // 使用外部速度
        ConservedState::new(h, h * forcing.u, h * forcing.v)
    }

    /// 批量计算 ghost 状态（用于重构）
    pub fn compute_ghost_states_batch(
        &self,
        interior_states: &[ConservedState],
        interior_z: &[f64],
        normals: &[DVec2],
        kinds: &[BoundaryKind],
        forcings: &[Option<ExternalForcing>],
        output: &mut [ConservedState],
    ) {
        debug_assert_eq!(interior_states.len(), output.len());

        for i in 0..interior_states.len() {
            output[i] = self.compute_ghost_state(
                interior_states[i],
                interior_z[i],
                normals[i],
                kinds[i],
                forcings[i].as_ref(),
            );
        }
    }
}

impl Default for GhostStateCalculator {
    fn default() -> Self {
        Self::new(BoundaryParams::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wall_ghost() {
        let calc = GhostStateCalculator::default();

        // 向右流动的水碰到向右的壁
        let interior = ConservedState::new(1.0, 1.0, 0.0); // h=1, u=1, v=0
        let normal = DVec2::new(1.0, 0.0); // 向右的法向

        let ghost = calc.wall_ghost(interior, normal);

        assert!((ghost.h - 1.0).abs() < 1e-10);
        assert!((ghost.hu - (-1.0)).abs() < 1e-10); // 速度反向
        assert!((ghost.hv - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_outflow_ghost() {
        let calc = GhostStateCalculator::default();

        let interior = ConservedState::new(2.0, 4.0, 6.0);
        let ghost = calc.outflow_ghost(interior);

        assert!((ghost.h - interior.h).abs() < 1e-10);
        assert!((ghost.hu - interior.hu).abs() < 1e-10);
        assert!((ghost.hv - interior.hv).abs() < 1e-10);
    }

    #[test]
    fn test_dry_wall() {
        let calc = GhostStateCalculator::default();

        let interior = ConservedState::new(1e-10, 0.0, 0.0);
        let normal = DVec2::new(1.0, 0.0);

        let ghost = calc.wall_ghost(interior, normal);

        assert!((ghost.h - 0.0).abs() < 1e-10);
    }
}
