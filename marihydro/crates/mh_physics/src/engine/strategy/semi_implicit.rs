// marihydro\crates\mh_physics\src\engine\strategy\semi_implicit.rs
//! 半隐式时间积分策略（泛型版本）
//!
//! 基于压力校正的半隐式时间推进算法。
//!
//! # 算法概述
//!
//! 半隐式方法将浅水方程分为显式和隐式两部分：
//! 1. **预测步**：显式计算对流和扩散项，得到预测速度 u*, v*
//! 2. **压力校正步**：隐式求解压力泊松方程
//! 3. **校正步**：用压力梯度校正速度和水位
//!
//! 这种方法允许使用比显式方法更大的 CFL 数（通常 2-10 倍），
//! 因为重力波的传播是隐式处理的。
//!
//! # 优势
//!
//! - 对于重力波主导的流动，可以使用更大的时间步长
//! - 在低 Froude 数流动中特别有效
//! - 适合长时间尺度的模拟

use super::{SemiImplicitConfig, StepResult, TimeIntegrationStrategy};
use super::workspace::SolverWorkspaceGeneric;
use crate::core::Backend;
use crate::engine::pcg::{PcgConfig, PcgSolver, PoissonMatrixBuilder, PreconditionerType};
use crate::mesh::MeshTopology;
use crate::prelude::*;
use crate::state::ShallowWaterState;

/// 泛型半隐式策略
/// 
/// 使用压力校正法的半隐式时间积分策略。
/// 内部使用 PCG 求解器求解压力泊松方程。
/// 
/// # 类型参数
/// 
/// - `B`: 计算后端类型
pub struct SemiImplicitStrategyGeneric<B: Backend> {
    /// 计算后端实例
    backend: B,
    /// 配置
    config: SemiImplicitConfig,
    /// PCG 求解器
    pcg_solver: Option<PcgSolver<B>>,
    /// 预测速度 u*
    u_star: B::Buffer<B::Scalar>,
    /// 预测速度 v*
    v_star: B::Buffer<B::Scalar>,
    /// 水位校正量 η'
    eta_prime: B::Buffer<B::Scalar>,
    /// 右端项（散度）
    rhs: B::Buffer<B::Scalar>,
    /// 对角矩阵（预处理器）
    diag: B::Buffer<B::Scalar>,
    /// 压力梯度 x 分量
    grad_eta_x: B::Buffer<B::Scalar>,
    /// 压力梯度 y 分量
    grad_eta_y: B::Buffer<B::Scalar>,
    /// 求解器已分配的单元数
    n_cells_allocated: usize,
}

impl<B: Backend + Clone> SemiImplicitStrategyGeneric<B> {
    /// 使用后端实例创建半隐式策略
    /// 
    /// # 参数
    /// 
    /// - `backend`: 计算后端实例
    /// - `n_cells`: 单元数量
    /// - `config`: 半隐式策略配置
    pub fn new_with_backend(backend: B, n_cells: usize, config: SemiImplicitConfig) -> Self {
        // 创建 PCG 求解器配置
        let pcg_config = PcgConfig {
            rtol: config.solver_rtol,
            atol: 1e-14,
            max_iter: config.solver_max_iter,
            preconditioner: PreconditionerType::Jacobi,
            verbose: false,
        };
        
        Self {
            u_star: backend.alloc(n_cells),
            v_star: backend.alloc(n_cells),
            eta_prime: backend.alloc(n_cells),
            rhs: backend.alloc(n_cells),
            diag: backend.alloc(n_cells),
            grad_eta_x: backend.alloc(n_cells),
            grad_eta_y: backend.alloc(n_cells),
            pcg_solver: Some(PcgSolver::new_with_backend(backend.clone(), n_cells, pcg_config)),
            backend,
            config,
            n_cells_allocated: n_cells,
        }
    }
    
    /// 获取后端引用
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 获取配置引用
    #[inline]
    pub fn config(&self) -> &SemiImplicitConfig {
        &self.config
    }
    
    /// 确保工作区大小足够
    fn ensure_capacity(&mut self, n_cells: usize) {
        if n_cells > self.n_cells_allocated {
            self.u_star = self.backend.alloc(n_cells);
            self.v_star = self.backend.alloc(n_cells);
            self.eta_prime = self.backend.alloc(n_cells);
            self.rhs = self.backend.alloc(n_cells);
            self.diag = self.backend.alloc(n_cells);
            self.grad_eta_x = self.backend.alloc(n_cells);
            self.grad_eta_y = self.backend.alloc(n_cells);
            
            if let Some(ref mut solver) = self.pcg_solver {
                solver.ensure_capacity(n_cells);
            }
            
            self.n_cells_allocated = n_cells;
        }
    }
}

impl<B: Backend + Clone> TimeIntegrationStrategy<B> for SemiImplicitStrategyGeneric<B> {
    fn name(&self) -> &'static str {
        "半隐式压力校正法"
    }
    
    fn step(
        &mut self,
        state: &mut ShallowWaterState<B>,
        mesh: &dyn MeshTopology<B>,
        _workspace: &mut SolverWorkspaceGeneric<B>,
        dt: B::Scalar,
    ) -> StepResult<B::Scalar> {
        let n_cells = mesh.n_cells();
        self.ensure_capacity(n_cells);

        let gravity = self.backend.config_scalar(self.config.gravity, "semi_implicit.gravity");
        let h_min = self.backend.config_scalar(self.config.h_min, "semi_implicit.h_min");
        let theta = self.backend.config_scalar(self.config.theta, "semi_implicit.theta");
        let half = B::Scalar::HALF;

        let h: &[B::Scalar] = &state.h;
        let hu: &[B::Scalar] = &state.hu;
        let hv: &[B::Scalar] = &state.hv;

        let u_star: &mut [B::Scalar] = &mut self.u_star;
        let v_star: &mut [B::Scalar] = &mut self.v_star;
        let eta_prime: &mut [B::Scalar] = &mut self.eta_prime;
        let rhs: &mut [B::Scalar] = &mut self.rhs;
        let grad_eta_x: &mut [B::Scalar] = &mut self.grad_eta_x;
        let grad_eta_y: &mut [B::Scalar] = &mut self.grad_eta_y;

        for i in 0..n_cells {
            if h[i] > h_min {
                u_star[i] = hu[i] / h[i];
                v_star[i] = hv[i] / h[i];
            } else {
                u_star[i] = B::Scalar::ZERO;
                v_star[i] = B::Scalar::ZERO;
            }
        }

        let mut cell_areas = self.backend.alloc(n_cells);
        for i in 0..n_cells {
            let area = mesh.cell_area(i);
            if !area.is_finite() || area <= B::Scalar::ZERO {
                cell_areas[i] = B::Scalar::ZERO;
            } else {
                cell_areas[i] = area;
            }
        }

        let matrix = PoissonMatrixBuilder::new(n_cells).build_csr(
            mesh,
            cell_areas.as_slice(),
            dt,
            gravity,
            theta,
            h,
            h_min,
        );

        let diag_matrix = match PoissonMatrixBuilder::new(n_cells).build_diagonal(
            &self.backend,
            cell_areas.as_slice(),
            dt,
            gravity,
            theta,
            h,
            h_min,
        ) {
            Ok(matrix) => matrix,
            Err(_) => {
                return StepResult {
                    dt_used: dt,
                    max_wave_speed: B::Scalar::ZERO,
                    dry_cells: 0,
                    limited_cells: 0,
                    converged: false,
                    iterations: 0,
                };
            }
        };

        self.diag = diag_matrix.diag.clone();
        let diag: &[B::Scalar] = &self.diag;

        rhs.fill(B::Scalar::ZERO);
        for face in mesh.interior_faces() {
            let owner = mesh.face_owner(*face);
            let neighbor = mesh.face_neighbor(*face).unwrap();

            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);

            let h_face = half * (h[owner] + h[neighbor]).max(h_min);

            let u_face = half * (u_star[owner] + u_star[neighbor]);
            let v_face = half * (v_star[owner] + v_star[neighbor]);

            let flux = h_face * (u_face * normal[0] + v_face * normal[1]) * length;

            rhs[owner] = rhs[owner] - flux;
            rhs[neighbor] = rhs[neighbor] + flux;
        }

        eta_prime.fill(B::Scalar::ZERO);
        let mut converged = true;
        let mut iterations = 0;

        if let Some(ref mut pcg_solver) = self.pcg_solver {
            let mut eta_buf = self.backend.alloc(eta_prime.len());
            eta_buf.copy_from_slice(eta_prime);
            let mut rhs_buf = self.backend.alloc(rhs.len());
            rhs_buf.copy_from_slice(rhs);

            let result = pcg_solver.solve(&matrix, &mut eta_buf, &rhs_buf, Some(&diag_matrix));

            converged = result.converged;
            iterations = result.iterations;

            eta_prime.copy_from_slice(eta_buf.as_slice());
        } else {
            let eps = self.backend.config_scalar(1e-14, "semi_implicit.residual_eps");
            let rtol = self.backend.config_scalar(self.config.solver_rtol, "semi_implicit.solver_rtol");
            for iter in 0..self.config.solver_max_iter {
                let mut max_residual = B::Scalar::ZERO;

                for i in 0..n_cells {
                    if diag[i].abs() > eps {
                        let new_eta = rhs[i] / diag[i];
                        let residual = (new_eta - eta_prime[i]).abs();
                        if residual > max_residual {
                            max_residual = residual;
                        }
                        eta_prime[i] = new_eta;
                    }
                }

                iterations = iter + 1;
                if max_residual < rtol {
                    break;
                }

                if iter == self.config.solver_max_iter - 1 {
                    converged = false;
                }
            }
        }

        grad_eta_x.fill(B::Scalar::ZERO);
        grad_eta_y.fill(B::Scalar::ZERO);

        for face in mesh.interior_faces() {
            let owner = mesh.face_owner(*face);
            let neighbor = mesh.face_neighbor(*face).unwrap();

            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);

            let eta_face = half * (eta_prime[owner] + eta_prime[neighbor]);

            let contrib_x = eta_face * normal[0] * length;
            let contrib_y = eta_face * normal[1] * length;

            grad_eta_x[owner] = grad_eta_x[owner] + contrib_x;
            grad_eta_x[neighbor] = grad_eta_x[neighbor] - contrib_x;
            grad_eta_y[owner] = grad_eta_y[owner] + contrib_y;
            grad_eta_y[neighbor] = grad_eta_y[neighbor] - contrib_y;
        }

        for i in 0..n_cells {
            let area = mesh.cell_area(i);
            if area.is_finite() && area > eps_zero::<B::Scalar>() {
                let inv_area = B::Scalar::ONE / area;
                grad_eta_x[i] = grad_eta_x[i] * inv_area;
                grad_eta_y[i] = grad_eta_y[i] * inv_area;
            }
        }

        let h_mut: &mut [B::Scalar] = &mut state.h;
        let hu_mut: &mut [B::Scalar] = &mut state.hu;
        let hv_mut: &mut [B::Scalar] = &mut state.hv;

        let mut max_wave_speed = B::Scalar::ZERO;
        let mut dry_cells = 0usize;

        for i in 0..n_cells {
            h_mut[i] = h_mut[i] + eta_prime[i];

            if h_mut[i] < h_min {
                h_mut[i] = B::Scalar::ZERO;
                hu_mut[i] = B::Scalar::ZERO;
                hv_mut[i] = B::Scalar::ZERO;
                dry_cells += 1;
            } else {
                let u_new = u_star[i] - gravity * theta * dt * grad_eta_x[i];
                let v_new = v_star[i] - gravity * theta * dt * grad_eta_y[i];

                hu_mut[i] = h_mut[i] * u_new;
                hv_mut[i] = h_mut[i] * v_new;

                let c = (gravity * h_mut[i]).sqrt();
                let speed = (u_new * u_new + v_new * v_new).sqrt() + c;
                if speed > max_wave_speed {
                    max_wave_speed = speed;
                }
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
        state: &ShallowWaterState<B>,
        mesh: &dyn MeshTopology<B>,
        cfl: B::Scalar,
    ) -> B::Scalar {
        let h: &[B::Scalar] = &state.h;
        let hu: &[B::Scalar] = &state.hu;
        let hv: &[B::Scalar] = &state.hv;

        let h_min = B::Scalar::from_f64(self.config.h_min).unwrap_or(B::Scalar::ZERO);
        let gravity = B::Scalar::from_f64(self.config.gravity).unwrap_or(B::Scalar::ZERO);

        let mut dt_min = B::Scalar::MAX;

        for i in 0..mesh.n_cells() {
            if h[i] <= h_min {
                continue;
            }

            let u = hu[i] / h[i];
            let v = hv[i] / h[i];
            let c = (gravity * h[i]).sqrt();
            let speed = (u * u + v * v).sqrt() + c;

            let speed_eps = B::Scalar::from_f64(1e-10).unwrap_or(B::Scalar::EPSILON);
            if speed > speed_eps {
                let area = mesh.cell_area(i);
                if !area.is_finite() || area <= B::Scalar::ZERO {
                    continue;
                }
                let dx = area.sqrt();
                let dt_local = cfl * dx / speed;
                if dt_local < dt_min {
                    dt_min = dt_local;
                }
            }
        }

        if dt_min == B::Scalar::MAX {
            dt_min = B::Scalar::from_f64(1e-6).unwrap_or(B::Scalar::MIN_POSITIVE);
        }

        let two = B::Scalar::from_f64(2.0).unwrap_or(B::Scalar::TWO);
        dt_min * two
    }

    fn supports_large_cfl(&self) -> bool {
        true
    }

    fn recommended_cfl(&self) -> B::Scalar {
        B::Scalar::from_f64(2.0).unwrap_or(B::Scalar::TWO)
    }
}

#[inline]
fn eps_zero<S: mh_runtime::RuntimeScalar>() -> S {
    S::from_f64(1e-14).unwrap_or(S::MIN_POSITIVE)
}
