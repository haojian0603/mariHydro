// src-tauri/src/marihydro/physics/engine/parallel.rs
//! 并行通量计算模块
//! 
//! 提供两种并行策略：
//! 1. ParallelFluxCalculator - 原子操作累加（简单但有竞争开销）
//! 2. ColoredFluxCalculator - 基于图着色的无竞争并行（推荐）

use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, FaceIndex, NumericalParams};
use crate::marihydro::core::Workspace;
use crate::marihydro::domain::mesh::MeshColoring;
use crate::marihydro::domain::state::ShallowWaterState;
use crate::marihydro::physics::schemes::{HllcSolver, HydrostaticReconstruction, InterfaceFlux};
use glam::DVec2;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub struct ParallelFluxCalculator {
    g: f64,
    params: NumericalParams,
}

impl ParallelFluxCalculator {
    pub fn new(g: f64, params: NumericalParams) -> Self {
        Self { g, params }
    }

    pub fn compute_fluxes<M: MeshAccess + Sync>(
        &self,
        state: &ShallowWaterState,
        mesh: &M,
        workspace: &mut Workspace,
    ) -> MhResult<f64> {
        let hllc = HllcSolver::new(self.params.clone(), self.g);
        let hydro = HydrostaticReconstruction::new(&self.params, self.g);
        let max_speed = AtomicU64::new(0u64);
        let n_faces = mesh.n_faces();
        let face_fluxes: Vec<(InterfaceFlux, DVec2)> = (0..n_faces)
            .into_par_iter()
            .map(|face_idx| {
                let face = FaceIndex(face_idx);
                let owner = mesh.face_owner(face);
                let neighbor = mesh.face_neighbor(face);
                let normal = mesh.face_normal(face);
                let h_l = state.h[owner.0];
                let vel_l = self.params.safe_velocity(state.hu[owner.0], state.hv[owner.0], h_l).to_vec();
                let z_l = state.z[owner.0];
                let (h_r, vel_r, z_r) = if neighbor.is_valid() {
                    let h = state.h[neighbor.0];
                    let v = self.params.safe_velocity(state.hu[neighbor.0], state.hv[neighbor.0], h).to_vec();
                    (h, v, state.z[neighbor.0])
                } else {
                    self.compute_ghost_state(h_l, vel_l, z_l, normal, face_idx, mesh)
                };
                let recon = hydro.reconstruct_face_simple(h_l, h_r, z_l, z_r, vel_l, vel_r);
                let flux = hllc.solve(recon.h_left, recon.h_right, recon.vel_left, recon.vel_right, normal)
                    .unwrap_or(InterfaceFlux::ZERO);
                let bed_src = hydro.bed_slope_correction(h_l, h_r, z_l, z_r, normal, mesh.face_length(face));
                let spd = flux.max_wave_speed.to_bits();
                max_speed.fetch_max(spd, Ordering::Relaxed);
                (flux, bed_src.as_vec())
            })
            .collect();
        workspace.flux_h.fill(0.0);
        workspace.flux_hu.fill(0.0);
        workspace.flux_hv.fill(0.0);
        workspace.source_hu.fill(0.0);
        workspace.source_hv.fill(0.0);
        for (face_idx, (flux, bed_src)) in face_fluxes.iter().enumerate() {
            let face = FaceIndex(face_idx);
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            let length = mesh.face_length(face);
            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;
            workspace.flux_h[owner.0] -= fh;
            workspace.flux_hu[owner.0] -= fhu;
            workspace.flux_hv[owner.0] -= fhv;
            workspace.source_hu[owner.0] += bed_src.x;
            workspace.source_hv[owner.0] += bed_src.y;
            if neighbor.is_valid() {
                workspace.flux_h[neighbor.0] += fh;
                workspace.flux_hu[neighbor.0] += fhu;
                workspace.flux_hv[neighbor.0] += fhv;
                workspace.source_hu[neighbor.0] -= bed_src.x;
                workspace.source_hv[neighbor.0] -= bed_src.y;
            }
        }
        Ok(f64::from_bits(max_speed.load(Ordering::Relaxed)))
    }

    fn compute_ghost_state<M: MeshAccess>(&self, h: f64, vel: DVec2, z: f64, normal: DVec2, _face_idx: usize, _mesh: &M) -> (f64, DVec2, f64) {
        let vn = vel.dot(normal);
        let vel_ghost = vel - 2.0 * vn * normal;
        (h, vel_ghost, z)
    }
}

pub struct CellBasedFluxCalculator {
    g: f64,
    params: NumericalParams,
}

impl CellBasedFluxCalculator {
    pub fn new(g: f64, params: NumericalParams) -> Self {
        Self { g, params }
    }

    pub fn compute_cell_updates<M: MeshAccess + Sync>(
        &self,
        state: &ShallowWaterState,
        mesh: &M,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        let n_cells = mesh.n_cells();
        let hllc = HllcSolver::new(self.params.clone(), self.g);
        let hydro = HydrostaticReconstruction::new(&self.params, self.g);
        let max_speed = AtomicU64::new(0u64);
        let updates: Vec<(f64, f64, f64)> = (0..n_cells)
            .into_par_iter()
            .map(|cell_idx| {
                let cell = CellIndex(cell_idx);
                let area = mesh.cell_area(cell);
                if area < 1e-14 { return (0.0, 0.0, 0.0); }
                let h_c = state.h[cell_idx];
                let vel_c = self.params.safe_velocity(state.hu[cell_idx], state.hv[cell_idx], h_c).to_vec();
                let z_c = state.z[cell_idx];
                let mut dh = 0.0;
                let mut dhu = 0.0;
                let mut dhv = 0.0;
                for &face in mesh.cell_faces(cell) {
                    let owner = mesh.face_owner(face);
                    let neighbor = mesh.face_neighbor(face);
                    let normal = mesh.face_normal(face);
                    let length = mesh.face_length(face);
                    let is_owner = owner.0 == cell_idx;
                    let sign = if is_owner { -1.0 } else { 1.0 };
                    let (h_o, vel_o, z_o) = if is_owner { (h_c, vel_c, z_c) } else {
                        let h = state.h[owner.0];
                        (h, self.params.safe_velocity(state.hu[owner.0], state.hv[owner.0], h).to_vec(), state.z[owner.0])
                    };
                    let (h_n, vel_n, z_n) = if !neighbor.is_valid() {
                        let vn = vel_o.dot(normal);
                        (h_o, vel_o - 2.0 * vn * normal, z_o)
                    } else if is_owner {
                        let h = state.h[neighbor.0];
                        (h, self.params.safe_velocity(state.hu[neighbor.0], state.hv[neighbor.0], h).to_vec(), state.z[neighbor.0])
                    } else { (h_c, vel_c, z_c) };
                    let recon = hydro.reconstruct_face_simple(h_o, h_n, z_o, z_n, vel_o, vel_n);
                    if let Ok(flux) = hllc.solve(recon.h_left, recon.h_right, recon.vel_left, recon.vel_right, normal) {
                        dh += sign * flux.mass * length;
                        dhu += sign * flux.momentum_x * length;
                        dhv += sign * flux.momentum_y * length;
                        let bed = hydro.bed_slope_correction(h_o, h_n, z_o, z_n, normal, length);
                        dhu += sign * bed.source_x;
                        dhv += sign * bed.source_y;
                        let spd = flux.max_wave_speed.to_bits();
                        max_speed.fetch_max(spd, Ordering::Relaxed);
                    }
                }
                (dh / area, dhu / area, dhv / area)
            })
            .collect();
        let (dh, dhu, dhv): (Vec<_>, (Vec<_>, Vec<_>)) = updates.into_iter().map(|(a,b,c)| (a,(b,c))).unzip();
        let (dhu, dhv): (Vec<_>, Vec<_>) = dhu.into_iter().unzip();
        (dh, dhu, dhv, f64::from_bits(max_speed.load(Ordering::Relaxed)))
    }
}

/// 基于图着色的并行通量计算器
/// 
/// 利用网格着色保证同一颜色组内的面不共享单元，
/// 从而实现无竞争的并行累加。
pub struct ColoredFluxCalculator {
    g: f64,
    params: NumericalParams,
    coloring: Arc<MeshColoring>,
}

impl ColoredFluxCalculator {
    /// 创建着色通量计算器
    pub fn new<M: MeshAccess>(g: f64, params: NumericalParams, mesh: &M) -> Self {
        let coloring = MeshColoring::build(mesh);
        Self {
            g,
            params,
            coloring: Arc::new(coloring),
        }
    }
    
    /// 使用预计算的着色创建
    pub fn with_coloring(g: f64, params: NumericalParams, coloring: Arc<MeshColoring>) -> Self {
        Self { g, params, coloring }
    }
    
    /// 获取着色信息
    pub fn coloring(&self) -> &MeshColoring {
        &self.coloring
    }
    
    /// 计算通量（基于着色的并行）
    pub fn compute_fluxes<M: MeshAccess + Sync>(
        &self,
        state: &ShallowWaterState,
        mesh: &M,
        workspace: &mut Workspace,
    ) -> MhResult<f64> {
        let hllc = HllcSolver::new(self.params.clone(), self.g);
        let hydro = HydrostaticReconstruction::new(&self.params, self.g);
        
        // 重置工作空间
        workspace.flux_h.fill(0.0);
        workspace.flux_hu.fill(0.0);
        workspace.flux_hv.fill(0.0);
        workspace.source_hu.fill(0.0);
        workspace.source_hv.fill(0.0);
        
        let mut max_wave_speed: f64 = 0.0;
        
        // 按颜色组顺序处理，组内并行
        for (_, group) in self.coloring.iter_groups() {
            let group_max_speed = AtomicU64::new(0u64);
            
            // 组内面可以安全并行处理
            group.par_iter().for_each(|&face_idx| {
                let face = FaceIndex(face_idx);
                let owner = mesh.face_owner(face);
                let neighbor = mesh.face_neighbor(face);
                let normal = mesh.face_normal(face);
                let length = mesh.face_length(face);
                
                // 获取左右状态
                let h_l = state.h[owner.0];
                let vel_l = self.params.safe_velocity(state.hu[owner.0], state.hv[owner.0], h_l).to_vec();
                let z_l = state.z[owner.0];
                
                let (h_r, vel_r, z_r) = if neighbor.is_valid() {
                    let h = state.h[neighbor.0];
                    let v = self.params.safe_velocity(state.hu[neighbor.0], state.hv[neighbor.0], h).to_vec();
                    (h, v, state.z[neighbor.0])
                } else {
                    // 边界：反射条件
                    let vn = vel_l.dot(normal);
                    (h_l, vel_l - 2.0 * vn * normal, z_l)
                };
                
                // 静水重构
                let recon = hydro.reconstruct_face_simple(h_l, h_r, z_l, z_r, vel_l, vel_r);
                
                // 计算通量
                let flux = hllc
                    .solve(recon.h_left, recon.h_right, recon.vel_left, recon.vel_right, normal)
                    .unwrap_or(InterfaceFlux::ZERO);
                
                // 底坡源项
                let bed_src = hydro.bed_slope_correction(h_l, h_r, z_l, z_r, normal, length);
                
                // 更新最大波速
                let spd = flux.max_wave_speed.to_bits();
                group_max_speed.fetch_max(spd, Ordering::Relaxed);
                
                // 计算通量贡献
                let fh = flux.mass * length;
                let fhu = flux.momentum_x * length;
                let fhv = flux.momentum_y * length;
                
                // 由于同一组内面不共享单元，可以安全地直接访问
                // 使用 unsafe 进行原子性写入（实际上由于着色保证不会有竞争）
                unsafe {
                    let ptr_h = workspace.flux_h.as_ptr() as *mut f64;
                    let ptr_hu = workspace.flux_hu.as_ptr() as *mut f64;
                    let ptr_hv = workspace.flux_hv.as_ptr() as *mut f64;
                    let ptr_shu = workspace.source_hu.as_ptr() as *mut f64;
                    let ptr_shv = workspace.source_hv.as_ptr() as *mut f64;
                    
                    *ptr_h.add(owner.0) -= fh;
                    *ptr_hu.add(owner.0) -= fhu;
                    *ptr_hv.add(owner.0) -= fhv;
                    *ptr_shu.add(owner.0) += bed_src.source_x;
                    *ptr_shv.add(owner.0) += bed_src.source_y;
                    
                    if neighbor.is_valid() {
                        *ptr_h.add(neighbor.0) += fh;
                        *ptr_hu.add(neighbor.0) += fhu;
                        *ptr_hv.add(neighbor.0) += fhv;
                        *ptr_shu.add(neighbor.0) -= bed_src.source_x;
                        *ptr_shv.add(neighbor.0) -= bed_src.source_y;
                    }
                }
            });
            
            max_wave_speed = max_wave_speed.max(f64::from_bits(group_max_speed.load(Ordering::Relaxed)));
        }
        
        Ok(max_wave_speed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colored_flux_calculator_creation() {
        // 基本创建测试
        let params = NumericalParams::default();
        let coloring = MeshColoring {
            color_groups: vec![vec![0, 1], vec![2, 3]],
            face_colors: vec![0, 0, 1, 1],
            n_colors: 2,
        };
        let calc = ColoredFluxCalculator::with_coloring(9.81, params, Arc::new(coloring));
        assert_eq!(calc.coloring().n_colors, 2);
    }
}
