// marihydro\crates\mh_physics\src\engine\strategy\explicit.rs
//! 显式时间积分策略
//!
//! 基于 Godunov 格式的显式有限体积法。

use super::{ExplicitConfig, StepResult, TimeIntegrationStrategy};
use super::workspace::SolverWorkspaceGeneric;
use crate::core::{Backend, CpuBackend, Scalar};
use crate::mesh::MeshTopology;
use crate::state::ShallowWaterStateGeneric;

/// 显式时间积分策略
#[allow(dead_code)]
pub struct ExplicitStrategy<B: Backend> {
    /// 配置
    config: ExplicitConfig,
    /// 重力加速度
    gravity: B::Scalar,
    /// 干单元阈值
    h_dry: B::Scalar,
}

impl<B: Backend> ExplicitStrategy<B> {
    /// 创建显式策略
    pub fn new(config: ExplicitConfig) -> Self {
        Self {
            gravity: B::Scalar::from_f64(config.gravity),
            h_dry: B::Scalar::from_f64(config.h_dry),
            config,
        }
    }
}

impl TimeIntegrationStrategy<CpuBackend<f64>> for ExplicitStrategy<CpuBackend<f64>> {
    fn name(&self) -> &'static str {
        "Explicit Godunov"
    }
    
    fn step(
        &mut self,
        state: &mut ShallowWaterStateGeneric<CpuBackend<f64>>,
        mesh: &dyn MeshTopology<CpuBackend<f64>>,
        workspace: &mut SolverWorkspaceGeneric<CpuBackend<f64>>,
        dt: f64,
    ) -> StepResult<f64> {
        // 1. 重置工作区
        workspace.reset();
        
        let n_cells = mesh.n_cells();
        // 使用 Vec 的原生 as_slice 方法
        let h: &[f64] = &state.h;
        let hu: &[f64] = &state.hu;
        let hv: &[f64] = &state.hv;
        let z: &[f64] = &state.z;
        
        let flux_h: &mut [f64] = &mut workspace.flux_h;
        let flux_hu: &mut [f64] = &mut workspace.flux_hu;
        let flux_hv: &mut [f64] = &mut workspace.flux_hv;
        
        let h_dry = self.config.h_dry;
        let gravity = self.config.gravity;
        
        let mut max_wave_speed = 0.0f64;
        let mut dry_cells = 0usize;
        
        // 2. 计算内部面通量
        for face in mesh.interior_faces() {
            let owner = mesh.face_owner(*face);
            let neighbor = mesh.face_neighbor(*face).unwrap();
            
            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);
            
            // Owner 状态
            let h_l = h[owner];
            let hu_l = hu[owner];
            let hv_l = hv[owner];
            let z_l = z[owner];
            
            // Neighbor 状态
            let h_r = h[neighbor];
            let hu_r = hu[neighbor];
            let hv_r = hv[neighbor];
            let z_r = z[neighbor];
            
            // 计算法向速度
            let (u_l, v_l) = if h_l > h_dry {
                (hu_l / h_l, hv_l / h_l)
            } else {
                (0.0, 0.0)
            };
            
            let (u_r, v_r) = if h_r > h_dry {
                (hu_r / h_r, hv_r / h_r)
            } else {
                (0.0, 0.0)
            };
            
            // 投影到法向
            let un_l = u_l * normal[0] + v_l * normal[1];
            let un_r = u_r * normal[0] + v_r * normal[1];
            
            // 静水重构
            let eta_l = h_l + z_l;
            let eta_r = h_r + z_r;
            let z_star = z_l.max(z_r);
            
            let h_l_star = (eta_l - z_star).max(0.0);
            let h_r_star = (eta_r - z_star).max(0.0);
            
            // 波速估计
            let c_l = if h_l_star > h_dry { (gravity * h_l_star).sqrt() } else { 0.0 };
            let c_r = if h_r_star > h_dry { (gravity * h_r_star).sqrt() } else { 0.0 };
            
            let s_l = un_l.min(un_r) - c_l.max(c_r);
            let s_r = un_l.max(un_r) + c_l.max(c_r);
            
            // 更新最大波速
            max_wave_speed = max_wave_speed.max(s_l.abs()).max(s_r.abs());
            
            // HLL 通量
            let (f_h, f_hu, f_hv) = if s_l >= 0.0 {
                // 全部来自左侧
                let f_h = h_l_star * un_l;
                let f_hu = h_l_star * u_l * un_l + 0.5 * gravity * h_l_star * h_l_star * normal[0];
                let f_hv = h_l_star * v_l * un_l + 0.5 * gravity * h_l_star * h_l_star * normal[1];
                (f_h, f_hu, f_hv)
            } else if s_r <= 0.0 {
                // 全部来自右侧
                let f_h = h_r_star * un_r;
                let f_hu = h_r_star * u_r * un_r + 0.5 * gravity * h_r_star * h_r_star * normal[0];
                let f_hv = h_r_star * v_r * un_r + 0.5 * gravity * h_r_star * h_r_star * normal[1];
                (f_h, f_hu, f_hv)
            } else {
                // HLL 公式
                let denom = s_r - s_l;
                if denom.abs() < 1e-14 {
                    (0.0, 0.0, 0.0)
                } else {
                    let f_l_h = h_l_star * un_l;
                    let f_r_h = h_r_star * un_r;
                    let f_l_hu = h_l_star * u_l * un_l + 0.5 * gravity * h_l_star * h_l_star * normal[0];
                    let f_r_hu = h_r_star * u_r * un_r + 0.5 * gravity * h_r_star * h_r_star * normal[0];
                    let f_l_hv = h_l_star * v_l * un_l + 0.5 * gravity * h_l_star * h_l_star * normal[1];
                    let f_r_hv = h_r_star * v_r * un_r + 0.5 * gravity * h_r_star * h_r_star * normal[1];
                    
                    let f_h = (s_r * f_l_h - s_l * f_r_h + s_l * s_r * (h_r_star - h_l_star)) / denom;
                    let f_hu = (s_r * f_l_hu - s_l * f_r_hu + s_l * s_r * (h_r_star * u_r - h_l_star * u_l)) / denom;
                    let f_hv = (s_r * f_l_hv - s_l * f_r_hv + s_l * s_r * (h_r_star * v_r - h_l_star * v_l)) / denom;
                    (f_h, f_hu, f_hv)
                }
            };
            
            // 累加通量
            let flux_mag_h = f_h * length;
            let flux_mag_hu = f_hu * length;
            let flux_mag_hv = f_hv * length;
            
            flux_h[owner] -= flux_mag_h;
            flux_h[neighbor] += flux_mag_h;
            flux_hu[owner] -= flux_mag_hu;
            flux_hu[neighbor] += flux_mag_hu;
            flux_hv[owner] -= flux_mag_hv;
            flux_hv[neighbor] += flux_mag_hv;
        }
        
        // 3. 边界面处理（反射边界）
        for face in mesh.boundary_faces() {
            let owner = mesh.face_owner(*face);
            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);
            
            let h_l = h[owner];
            let _z_l = z[owner];
            
            if h_l <= h_dry {
                continue;
            }
            
            let u_l = hu[owner] / h_l;
            let v_l = hv[owner] / h_l;
            
            // 反射边界：法向速度取反
            let un_l = u_l * normal[0] + v_l * normal[1];
            
            // 静止壁面通量（仅压力）
            let f_hu = 0.5 * gravity * h_l * h_l * normal[0] * length;
            let f_hv = 0.5 * gravity * h_l * h_l * normal[1] * length;
            
            flux_hu[owner] -= f_hu;
            flux_hv[owner] -= f_hv;
            
            let c = (gravity * h_l).sqrt();
            max_wave_speed = max_wave_speed.max(un_l.abs() + c);
        }
        
        // 4. 更新状态
        let h_mut: &mut [f64] = &mut state.h;
        let hu_mut: &mut [f64] = &mut state.hu;
        let hv_mut: &mut [f64] = &mut state.hv;
        
        for i in 0..n_cells {
            let area = mesh.cell_area(i);
            if area <= 0.0 {
                continue;
            }
            let inv_area = 1.0 / area;
            
            h_mut[i] += dt * flux_h[i] * inv_area;
            hu_mut[i] += dt * flux_hu[i] * inv_area;
            hv_mut[i] += dt * flux_hv[i] * inv_area;
            
            // 干单元处理
            if h_mut[i] < h_dry {
                h_mut[i] = 0.0;
                hu_mut[i] = 0.0;
                hv_mut[i] = 0.0;
                dry_cells += 1;
            }
        }
        
        StepResult {
            dt_used: dt,
            max_wave_speed,
            dry_cells,
            limited_cells: 0,
            converged: true,
            iterations: 0,
        }
    }
    
    fn compute_stable_dt(
        &self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        mesh: &dyn MeshTopology<CpuBackend<f64>>,
        cfl: f64,
    ) -> f64 {
        let h: &[f64] = &state.h;
        let hu: &[f64] = &state.hu;
        let hv: &[f64] = &state.hv;
        
        let h_dry = self.config.h_dry;
        let gravity = self.config.gravity;
        
        let mut dt_min = f64::MAX;
        
        for i in 0..mesh.n_cells() {
            if h[i] <= h_dry {
                continue;
            }
            
            let u = hu[i] / h[i];
            let v = hv[i] / h[i];
            let c = (gravity * h[i]).sqrt();
            let speed = (u * u + v * v).sqrt() + c;
            
            if speed > 1e-10 {
                let area = mesh.cell_area(i);
                let dx = area.sqrt();
                let dt_local = cfl * dx / speed;
                dt_min = dt_min.min(dt_local);
            }
        }
        
        if dt_min == f64::MAX {
            dt_min = 1e-6;
        }
        
        dt_min
    }
    
    fn recommended_cfl(&self) -> f64 {
        self.config.cfl.max(0.5)
    }
}
