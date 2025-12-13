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
use crate::core::{Backend, CpuBackend};
use crate::engine::pcg::{PcgSolver, PcgConfig, DiagonalMatrix, PreconditionerType};
use crate::mesh::MeshTopology;
use crate::state::ShallowWaterStateGeneric;

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

// 为了向后兼容，保留旧的构造函数（但标记为废弃）
impl<B: Backend> SemiImplicitStrategyGeneric<B> {
    /// 创建半隐式策略（废弃，请使用 new_with_backend）
    #[deprecated(note = "请使用 new_with_backend 方法显式传入后端实例")]
    pub fn new(n_cells: usize, config: SemiImplicitConfig) -> Self
    where
        B: Default + Clone,
    {
        let backend = B::default();
        Self {
            u_star: backend.alloc(n_cells),
            v_star: backend.alloc(n_cells),
            eta_prime: backend.alloc(n_cells),
            rhs: backend.alloc(n_cells),
            diag: backend.alloc(n_cells),
            grad_eta_x: backend.alloc(n_cells),
            grad_eta_y: backend.alloc(n_cells),
            pcg_solver: None,  // 旧版本不使用 PCG
            backend,
            config,
            n_cells_allocated: n_cells,
        }
    }
}

impl TimeIntegrationStrategy<CpuBackend<f64>> for SemiImplicitStrategyGeneric<CpuBackend<f64>> {
    fn name(&self) -> &'static str {
        "半隐式压力校正法"
    }
    
    fn step(
        &mut self,
        state: &mut ShallowWaterStateGeneric<CpuBackend<f64>>,
        mesh: &dyn MeshTopology<CpuBackend<f64>>,
        _workspace: &mut SolverWorkspaceGeneric<CpuBackend<f64>>,
        dt: f64, // ALLOW_F64: 时间步长
    ) -> StepResult<f64> {
        let n_cells = mesh.n_cells();
        self.ensure_capacity(n_cells);
        
        let gravity = self.config.gravity;
        let h_min = self.config.h_min;
        let theta = self.config.theta;
        
        // 获取状态引用（只读）
        let h: &[f64] = &state.h;
        let hu: &[f64] = &state.hu;
        let hv: &[f64] = &state.hv;
        let _z: &[f64] = &state.z;
        
        // 获取工作缓冲区（可写）
        let u_star: &mut [f64] = &mut self.u_star;
        let v_star: &mut [f64] = &mut self.v_star;
        let eta_prime: &mut [f64] = &mut self.eta_prime;
        let rhs: &mut [f64] = &mut self.rhs;
        let diag: &mut [f64] = &mut self.diag;
        let grad_eta_x: &mut [f64] = &mut self.grad_eta_x;
        let grad_eta_y: &mut [f64] = &mut self.grad_eta_y;
        
        // ========== 第1步：预测步 ==========
        // 计算预测速度 u* = u^n + dt * (显式项)
        // 显式项包括：对流、扩散、床底坡度、摩擦等
        // 这里使用简化实现：直接从当前动量计算速度
        for i in 0..n_cells {
            if h[i] > h_min {
                u_star[i] = hu[i] / h[i];
                v_star[i] = hv[i] / h[i];
            } else {
                u_star[i] = 0.0;
                v_star[i] = 0.0;
            }
        }
        
        // ========== 第2步：组装压力泊松方程 ==========
        // 离散形式：A * η' = b
        // 其中 A 是拉普拉斯算子的离散化，b 是速度散度
        //
        // 对于简化的对角近似：
        // A_ii ≈ Σ_f (H_f * L_f / d_f)
        // 这里使用更简单的形式：A_ii = Area_i / (g * θ * dt² * H_i)
        
        for i in 0..n_cells {
            let area = mesh.cell_area(i);
            let h_eff = h[i].max(h_min);
            
            // 对角项：来自压力泊松方程的离散化
            // 系数与时间步长、重力和水深相关
            diag[i] = area / (gravity * theta * dt * dt * h_eff);
        }
        
        // 计算右端项：预测速度的散度
        // b_i = -∫∫ ∇·(H u*) dA ≈ -Σ_f (H_f * u*_f · n_f) * L_f
        rhs.fill(0.0);
        for face in mesh.interior_faces() {
            let owner = mesh.face_owner(*face);
            let neighbor = mesh.face_neighbor(*face).unwrap();
            
            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);
            
            // 界面处的水深（算术平均）
            let h_face = 0.5 * (h[owner] + h[neighbor]).max(h_min);
            
            // 界面处的预测速度（算术平均）
            let u_face = 0.5 * (u_star[owner] + u_star[neighbor]);
            let v_face = 0.5 * (v_star[owner] + v_star[neighbor]);
            
            // 通过界面的体积通量
            let flux = h_face * (u_face * normal[0] + v_face * normal[1]) * length;
            
            // 累加到相邻单元（守恒形式）
            rhs[owner] -= flux;
            rhs[neighbor] += flux;
        }
        
        // ========== 第3步：求解压力校正方程 ==========
        // 使用 PCG 求解器或简单的 Jacobi 迭代
        eta_prime.fill(0.0);
        let mut converged = true;
        let mut iterations = 0;
        
        if let Some(ref mut pcg_solver) = self.pcg_solver {
            // 使用 PCG 求解器
            let diag_matrix = DiagonalMatrix::new(diag.to_vec(), n_cells);
            let mut eta_vec = eta_prime.to_vec();
            let rhs_vec = rhs.to_vec();
            
            let result = pcg_solver.solve(&diag_matrix, &mut eta_vec, &rhs_vec, Some(&diag_matrix));
            
            converged = result.converged;
            iterations = result.iterations;
            
            // 复制结果回缓冲区
            for i in 0..n_cells {
                eta_prime[i] = eta_vec[i];
            }
        } else {
            // 回退到简单的 Jacobi 迭代
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
        }
        
        // ========== 第4步：计算压力梯度 ==========
        // ∇η' 通过 Green-Gauss 公式计算
        grad_eta_x.fill(0.0);
        grad_eta_y.fill(0.0);
        
        for face in mesh.interior_faces() {
            let owner = mesh.face_owner(*face);
            let neighbor = mesh.face_neighbor(*face).unwrap();
            
            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);
            
            // 界面处的水位校正（算术平均）
            let eta_face = 0.5 * (eta_prime[owner] + eta_prime[neighbor]);
            
            // 梯度贡献（Green-Gauss 定理）
            let contrib_x = eta_face * normal[0] * length;
            let contrib_y = eta_face * normal[1] * length;
            
            grad_eta_x[owner] += contrib_x;
            grad_eta_x[neighbor] -= contrib_x;
            grad_eta_y[owner] += contrib_y;
            grad_eta_y[neighbor] -= contrib_y;
        }
        
        // 除以单元面积得到梯度
        for i in 0..n_cells {
            let area = mesh.cell_area(i);
            if area > 1e-14 {
                let inv_area = 1.0 / area;
                grad_eta_x[i] *= inv_area;
                grad_eta_y[i] *= inv_area;
            }
        }
        
        // ========== 第5步：校正速度和水位 ==========
        // u^{n+1} = u* - g * θ * dt * ∇η'
        // η^{n+1} = η^n + η'
        let h_mut: &mut [f64] = &mut state.h;
        let hu_mut: &mut [f64] = &mut state.hu;
        let hv_mut: &mut [f64] = &mut state.hv;
        
        let mut max_wave_speed = 0.0f64;
        let mut dry_cells = 0usize;
        
        for i in 0..n_cells {
            // 更新水位
            h_mut[i] += eta_prime[i];
            
            if h_mut[i] < h_min {
                // 干单元处理
                h_mut[i] = 0.0;
                hu_mut[i] = 0.0;
                hv_mut[i] = 0.0;
                dry_cells += 1;
            } else {
                // 速度校正
                let u_new = u_star[i] - gravity * theta * dt * grad_eta_x[i];
                let v_new = v_star[i] - gravity * theta * dt * grad_eta_y[i];
                
                // 更新动量
                hu_mut[i] = h_mut[i] * u_new;
                hv_mut[i] = h_mut[i] * v_new;
                
                // 计算最大波速
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
    
    /// 计算稳定时间步长
    /// 
    /// 半隐式方法可以使用比显式方法更大的 CFL 数，
    /// 因为重力波是隐式处理的。
    fn compute_stable_dt(
        &self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        mesh: &dyn MeshTopology<CpuBackend<f64>>,
        cfl: f64, // ALLOW_F64: 物理参数
    ) -> f64 {
        let h: &[f64] = &state.h;
        let hu: &[f64] = &state.hu;
        let hv: &[f64] = &state.hv;
        
        let h_min = self.config.h_min;
        let gravity = self.config.gravity;
        
        let mut dt_min = f64::MAX;
        
        for i in 0..mesh.n_cells() {
            // 跳过干单元
            if h[i] <= h_min {
                continue;
            }
            
            let u = hu[i] / h[i];
            let v = hv[i] / h[i];
            let c = (gravity * h[i]).sqrt();
            
            // 对于半隐式方法，时间步长主要受对流速度限制
            // 重力波速度的影响较小
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
        
        // 半隐式方法允许更大的 CFL 数（通常可以是显式的 2-5 倍）
        dt_min * 2.0
    }
    
    /// 半隐式方法支持大 CFL 数
    fn supports_large_cfl(&self) -> bool {
        true
    }
    
    /// 推荐的 CFL 数
    fn recommended_cfl(&self) -> f64 {
        // 半隐式方法推荐使用 CFL ≈ 2.0
        2.0
    }
}
