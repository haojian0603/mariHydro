// marihydro\crates\mh_physics\src\engine\strategy\semi_implicit.rs
//! 半隐式时间积分策略（泛型版本）
//!
//! 基于压力校正的半隐式时间推进算法。

use super::{SemiImplicitConfig, StepResult, TimeIntegrationStrategy};
use super::workspace::SolverWorkspaceGeneric;
use crate::core::{Backend, CpuBackend};
use crate::mesh::MeshTopology;
use crate::state::ShallowWaterStateGeneric;

/// 泛型半隐式策略
pub struct SemiImplicitStrategyGeneric<B: Backend> {
    /// 配置
    config: SemiImplicitConfig,
    /// 预测速度 u*
    u_star: B::Buffer<B::Scalar>,
    /// 预测速度 v*
    v_star: B::Buffer<B::Scalar>,
    /// 水位校正量 η'
    eta_prime: B::Buffer<B::Scalar>,
    /// 右端项
    rhs: B::Buffer<B::Scalar>,
    /// 对角矩阵
    diag: B::Buffer<B::Scalar>,
    /// 求解器已分配的单元数
    n_cells_allocated: usize,
}

impl<B: Backend> SemiImplicitStrategyGeneric<B> {
    /// 创建半隐式策略
    pub fn new(n_cells: usize, config: SemiImplicitConfig) -> Self {
        Self {
            config,
            u_star: B::alloc(n_cells),
            v_star: B::alloc(n_cells),
            eta_prime: B::alloc(n_cells),
            rhs: B::alloc(n_cells),
            diag: B::alloc(n_cells),
            n_cells_allocated: n_cells,
        }
    }
    
    /// 确保工作区大小足够
    fn ensure_capacity(&mut self, n_cells: usize) {
        if n_cells > self.n_cells_allocated {
            self.u_star = B::alloc(n_cells);
            self.v_star = B::alloc(n_cells);
            self.eta_prime = B::alloc(n_cells);
            self.rhs = B::alloc(n_cells);
            self.diag = B::alloc(n_cells);
            self.n_cells_allocated = n_cells;
        }
    }
}

impl TimeIntegrationStrategy<CpuBackend<f64>> for SemiImplicitStrategyGeneric<CpuBackend<f64>> {
    fn name(&self) -> &'static str {
        "Semi-Implicit Pressure Correction"
    }
    
    fn step(
        &mut self,
        state: &mut ShallowWaterStateGeneric<CpuBackend<f64>>,
        mesh: &dyn MeshTopology<CpuBackend<f64>>,
        _workspace: &mut SolverWorkspaceGeneric<CpuBackend<f64>>,
        dt: f64,
    ) -> StepResult<f64> {
        let n_cells = mesh.n_cells();
        self.ensure_capacity(n_cells);
        
        let gravity = self.config.gravity;
        let h_min = self.config.h_min;
        let theta = self.config.theta;
        
        let h: &[f64] = &state.h;
        let hu: &[f64] = &state.hu;
        let hv: &[f64] = &state.hv;
        let _z: &[f64] = &state.z;
        
        let u_star: &mut [f64] = &mut self.u_star;
        let v_star: &mut [f64] = &mut self.v_star;
        let eta_prime: &mut [f64] = &mut self.eta_prime;
        let rhs: &mut [f64] = &mut self.rhs;
        let diag: &mut [f64] = &mut self.diag;
        
        // ========== 步骤 1: 预测步 ==========
        // 计算预测速度 u* = u^n + dt * (对流项 + 扩散项)
        // 简化实现：直接使用当前速度
        for i in 0..n_cells {
            if h[i] > h_min {
                u_star[i] = hu[i] / h[i];
                v_star[i] = hv[i] / h[i];
            } else {
                u_star[i] = 0.0;
                v_star[i] = 0.0;
            }
        }
        
        // ========== 步骤 2: 压力泊松方程组装 ==========
        // ∇·(H∇η') = ∇·(H u*)
        // 简化实现：对角矩阵近似
        for i in 0..n_cells {
            let area = mesh.cell_area(i);
            let h_eff = h[i].max(h_min);
            diag[i] = area / (gravity * theta * dt * dt * h_eff);
        }
        
        // 计算右端项：散度
        rhs.fill(0.0);
        for face in mesh.interior_faces() {
            let owner = mesh.face_owner(*face);
            let neighbor = mesh.face_neighbor(*face).unwrap();
            
            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);
            
            let h_face = 0.5 * (h[owner] + h[neighbor]).max(h_min);
            let u_face = 0.5 * (u_star[owner] + u_star[neighbor]);
            let v_face = 0.5 * (v_star[owner] + v_star[neighbor]);
            
            let flux = h_face * (u_face * normal[0] + v_face * normal[1]) * length;
            
            rhs[owner] -= flux;
            rhs[neighbor] += flux;
        }
        
        // ========== 步骤 3: 求解压力校正 ==========
        // 简化实现：Jacobi 迭代
        eta_prime.fill(0.0);
        let mut converged = true;
        let mut iterations = 0;
        
        for iter in 0..self.config.solver_max_iter {
            let mut max_residual = 0.0f64;
            
            for i in 0..n_cells {
                if diag[i].abs() > 1e-14 {
                    let new_eta = rhs[i] / diag[i];
                    let residual = (new_eta - eta_prime[i]).abs();
                    max_residual = max_residual.max(residual);
                    eta_prime[i] = new_eta;
                }
            }
            
            iterations = iter + 1;
            if max_residual < self.config.solver_rtol {
                break;
            }
            
            if iter == self.config.solver_max_iter - 1 {
                converged = false;
            }
        }
        
        // ========== 步骤 4: 校正步 ==========
        let h_mut: &mut [f64] = &mut state.h;
        let hu_mut: &mut [f64] = &mut state.hu;
        let hv_mut: &mut [f64] = &mut state.hv;
        
        let mut max_wave_speed = 0.0f64;
        let mut dry_cells = 0usize;
        
        for i in 0..n_cells {
            // 更新水深
            h_mut[i] += eta_prime[i];
            
            if h_mut[i] < h_min {
                h_mut[i] = 0.0;
                hu_mut[i] = 0.0;
                hv_mut[i] = 0.0;
                dry_cells += 1;
            } else {
                // 更新动量
                // u^{n+1} = u* - g*θ*dt*∇η'
                let u_new = u_star[i];
                let v_new = v_star[i];
                hu_mut[i] = h_mut[i] * u_new;
                hv_mut[i] = h_mut[i] * v_new;
                
                // 波速
                let c = (gravity * h_mut[i]).sqrt();
                let speed = (u_new * u_new + v_new * v_new).sqrt() + c;
                max_wave_speed = max_wave_speed.max(speed);
            }
        }
        
        StepResult {
            dt_used: dt,
            max_wave_speed,
            dry_cells,
            limited_cells: 0,
            converged,
            iterations,
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
        
        let h_min = self.config.h_min;
        let gravity = self.config.gravity;
        
        let mut dt_min = f64::MAX;
        
        for i in 0..mesh.n_cells() {
            if h[i] <= h_min {
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
        
        // 半隐式允许更大的 CFL
        dt_min * 2.0
    }
    
    fn supports_large_cfl(&self) -> bool {
        true
    }
    
    fn recommended_cfl(&self) -> f64 {
        2.0
    }
}
