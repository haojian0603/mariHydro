// crates/mh_physics/src/engine/semi_implicit.rs

//! 半隐式时间推进策略
//!
//! 实现基于压力校正的半隐式时间推进算法，适用于：
//! - 大时间步长（突破 CFL 限制）
//! - 低弗劳德数流动
//! - 具有复杂地形的浅水问题
//!
//! # 算法概述
//!
//! 1. **预测步**：显式计算对流项和扩散项，得到预测速度 u*
//! 2. **压力校正**：求解压力泊松方程，得到水位校正 η'
//! 3. **校正步**：更新速度和水深
//!
//! $$\vec{u}^* = \vec{u}^n + \Delta t \cdot \text{(advection + diffusion)}$$
//! $$\nabla \cdot (H \nabla \eta') = \nabla \cdot (H \vec{u}^*)$$
//! $$\vec{u}^{n+1} = \vec{u}^* - g \Delta t \nabla \eta'$$
//! $$h^{n+1} = h^n + \eta'$$
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::engine::semi_implicit::{SemiImplicitStrategy, SemiImplicitConfig};
//!
//! let config = SemiImplicitConfig::default();
//! let mut strategy = SemiImplicitStrategy::new(&mesh, config);
//!
//! // 时间推进
//! strategy.advance(&mut state, &mesh, dt);
//! ```

use crate::adapter::PhysicsMesh;
use crate::numerics::discretization::{
    CellFaceTopology, DepthCorrector, PressureMatrixAssembler, VelocityCorrector,
};
use crate::numerics::linear_algebra::{
    IterativeSolver, JacobiPreconditioner, PcgSolver, SolverConfig, SolverResult, SolverStatus,
};
use crate::state::ShallowWaterState;
use mh_foundation::{AlignedVec, Scalar};
use serde::{Deserialize, Serialize};

/// 半隐式配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemiImplicitConfig {
    /// 重力加速度 [m/s²]
    pub gravity: Scalar,
    /// 最小水深阈值 [m]
    pub h_min: Scalar,
    /// 干单元水深阈值 [m]
    pub h_dry: Scalar,
    /// 线性求解器相对容差
    pub solver_rtol: Scalar,
    /// 线性求解器绝对容差
    pub solver_atol: Scalar,
    /// 线性求解器最大迭代次数
    pub solver_max_iter: usize,
    /// 预测步的隐式因子 (0=显式, 1=全隐式, 0.5=Crank-Nicolson)
    pub theta: Scalar,
    /// 是否打印求解器信息
    pub verbose: bool,
}

impl Default for SemiImplicitConfig {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            h_min: 1e-6,
            h_dry: 1e-4,
            solver_rtol: 1e-8,
            solver_atol: 1e-14,
            solver_max_iter: 200,
            theta: 0.5,
            verbose: false,
        }
    }
}

impl SemiImplicitConfig {
    /// 创建保守配置（更严格的求解器容差）
    pub fn conservative() -> Self {
        Self {
            solver_rtol: 1e-10,
            solver_max_iter: 500,
            ..Default::default()
        }
    }

    /// 创建快速配置（更宽松的求解器容差）
    pub fn fast() -> Self {
        Self {
            solver_rtol: 1e-6,
            solver_max_iter: 100,
            ..Default::default()
        }
    }

    /// 创建全隐式配置
    pub fn fully_implicit() -> Self {
        Self {
            theta: 1.0,
            ..Default::default()
        }
    }
}

/// 半隐式求解统计
#[derive(Debug, Clone, Default)]
pub struct SemiImplicitStats {
    /// 压力求解迭代次数
    pub pressure_iterations: usize,
    /// 压力求解残差
    pub pressure_residual: Scalar,
    /// 压力求解状态
    pub pressure_converged: bool,
    /// 最大水深校正量
    pub max_depth_correction: Scalar,
    /// 最大速度校正量
    pub max_velocity_correction: Scalar,
    /// 湿单元数
    pub n_wet_cells: usize,
    /// 干单元数
    pub n_dry_cells: usize,
}

/// 半隐式时间推进策略
pub struct SemiImplicitStrategy {
    /// 配置
    config: SemiImplicitConfig,
    /// 网格拓扑
    topo: CellFaceTopology,
    /// 压力矩阵组装器
    pressure_assembler: PressureMatrixAssembler,
    /// 线性求解器
    solver: PcgSolver,
    /// 预条件器
    precond: JacobiPreconditioner,
    /// 水深校正器
    depth_corrector: DepthCorrector,
    /// 速度校正器
    velocity_corrector: VelocityCorrector,
    /// 预测速度 u*
    u_star: AlignedVec<Scalar>,
    /// 预测速度 v*
    v_star: AlignedVec<Scalar>,
    /// 水位校正量 η'
    eta_prime: AlignedVec<Scalar>,
    /// 右端项
    rhs: AlignedVec<Scalar>,
    /// 最新统计
    stats: SemiImplicitStats,
}

impl SemiImplicitStrategy {
    /// 创建半隐式策略
    pub fn new(mesh: &PhysicsMesh, config: SemiImplicitConfig) -> Self {
        let topo = CellFaceTopology::from_mesh(mesh);
        let n_cells = mesh.n_cells();

        let pressure_assembler = PressureMatrixAssembler::new(&topo);

        let solver_config = SolverConfig {
            rtol: config.solver_rtol,
            atol: config.solver_atol,
            max_iter: config.solver_max_iter,
            verbose: config.verbose,
        };
        let solver = PcgSolver::new(solver_config);

        // 初始化预条件器为单位矩阵
        let precond = JacobiPreconditioner::from_diagonal(&vec![1.0; n_cells]);

        let depth_corrector = DepthCorrector::new(n_cells).with_h_min(config.h_min);
        let velocity_corrector = VelocityCorrector::new(&topo).with_h_min(config.h_dry);

        Self {
            config,
            topo,
            pressure_assembler,
            solver,
            precond,
            depth_corrector,
            velocity_corrector,
            u_star: AlignedVec::zeros(n_cells),
            v_star: AlignedVec::zeros(n_cells),
            eta_prime: AlignedVec::zeros(n_cells),
            rhs: AlignedVec::zeros(n_cells),
            stats: SemiImplicitStats::default(),
        }
    }

    /// 获取配置引用
    pub fn config(&self) -> &SemiImplicitConfig {
        &self.config
    }

    /// 获取最新统计
    pub fn stats(&self) -> &SemiImplicitStats {
        &self.stats
    }

    /// 执行半隐式时间推进
    ///
    /// # 参数
    ///
    /// - `state`: 浅水状态（将被修改）
    /// - `mesh`: 物理网格
    /// - `dt`: 时间步长
    pub fn advance(&mut self, state: &mut ShallowWaterState, mesh: &PhysicsMesh, dt: Scalar) {
        // 重置统计
        self.stats = SemiImplicitStats::default();

        // 统计干湿单元
        for i in 0..state.n_cells() {
            if state.h[i] > self.config.h_dry {
                self.stats.n_wet_cells += 1;
            } else {
                self.stats.n_dry_cells += 1;
            }
        }

        // 1. 预测步：u* = u^n (简化版，不计算对流项)
        self.u_star.as_mut_slice().copy_from_slice(&state.u);
        self.v_star.as_mut_slice().copy_from_slice(&state.v);

        // 2. 组装压力矩阵
        self.pressure_assembler.assemble(
            mesh,
            &self.topo,
            state,
            dt,
            self.config.gravity,
        );

        // 3. 计算右端项 (散度)
        self.compute_divergence_rhs(mesh, state, dt);

        // 4. 更新预条件器
        self.precond.update(self.pressure_assembler.matrix());

        // 5. 求解压力校正方程
        self.eta_prime.as_mut_slice().fill(0.0);
        let result = self.solver.solve(
            self.pressure_assembler.matrix(),
            self.rhs.as_slice(),
            self.eta_prime.as_mut_slice(),
            &self.precond,
        );

        self.update_solver_stats(&result);

        // 6. 校正水深
        let correction_stats =
            self.depth_corrector
                .correct_with_stats(&mut state.h, self.eta_prime.as_slice());
        self.stats.max_depth_correction = correction_stats.max_correction;

        // 7. 校正速度
        self.velocity_corrector.correct(
            &mut state.u,
            &mut state.v,
            &state.h,
            self.eta_prime.as_slice(),
            &self.topo,
            mesh,
            dt,
            self.config.gravity,
        );

        // 计算速度校正统计
        self.compute_velocity_correction_stats(state);
    }

    /// 计算散度右端项
    fn compute_divergence_rhs(
        &mut self,
        mesh: &PhysicsMesh,
        state: &ShallowWaterState,
        dt: Scalar,
    ) {
        self.rhs.as_mut_slice().fill(0.0);

        // 遍历内部面计算通量
        for &face_idx in self.topo.interior_faces() {
            let face = self.topo.face(face_idx);
            let owner = face.owner;
            let neighbor = face.neighbor.expect("interior face");

            // 面水深
            let h_o = state.h[owner];
            let h_n = state.h[neighbor];
            let h_f = 0.5 * (h_o + h_n);

            if h_f < self.config.h_dry {
                continue;
            }

            // 面速度
            let u_f = 0.5 * (self.u_star[owner] + self.u_star[neighbor]);
            let v_f = 0.5 * (self.v_star[owner] + self.v_star[neighbor]);

            // 法向通量
            let flux = h_f * (u_f * face.normal.x + v_f * face.normal.y) * face.length;

            // 累加到右端项
            let area_o = mesh.cell_area_unchecked(owner);
            let area_n = mesh.cell_area_unchecked(neighbor);

            self.rhs[owner] += flux / area_o;
            self.rhs[neighbor] -= flux / area_n;
        }

        // 边界面贡献（假设固壁边界，通量为零）
        // 如果需要开边界，在此处理

        // 乘以 dt
        for v in self.rhs.as_mut_slice() {
            *v *= dt;
        }
    }

    /// 更新求解器统计
    fn update_solver_stats(&mut self, result: &SolverResult) {
        self.stats.pressure_iterations = result.iterations;
        self.stats.pressure_residual = result.relative_residual;
        self.stats.pressure_converged = result.status == SolverStatus::Converged;

        if !self.stats.pressure_converged && self.config.verbose {
            log::warn!(
                "Pressure solver did not converge: {:?}, iter={}, residual={:.2e}",
                result.status,
                result.iterations,
                result.residual_norm
            );
        }
    }

    /// 计算速度校正统计
    fn compute_velocity_correction_stats(&mut self, state: &ShallowWaterState) {
        let mut max_correction: Scalar = 0.0;

        for i in 0..state.n_cells() {
            let du = state.u[i] - self.u_star[i];
            let dv = state.v[i] - self.v_star[i];
            let correction = (du * du + dv * dv).sqrt();
            max_correction = max_correction.max(correction);
        }

        self.stats.max_velocity_correction = max_correction;
    }

    /// 获取拓扑引用
    pub fn topology(&self) -> &CellFaceTopology {
        &self.topo
    }

    /// 获取水位校正量引用
    pub fn eta_prime(&self) -> &[Scalar] {
        self.eta_prime.as_slice()
    }

    /// 重置策略（网格变化时调用）
    pub fn reset(&mut self, mesh: &PhysicsMesh) {
        let n_cells = mesh.n_cells();

        self.topo = CellFaceTopology::from_mesh(mesh);
        self.pressure_assembler = PressureMatrixAssembler::new(&self.topo);
        self.depth_corrector = DepthCorrector::new(n_cells).with_h_min(self.config.h_min);
        self.velocity_corrector = VelocityCorrector::new(&self.topo).with_h_min(self.config.h_dry);

        self.u_star = AlignedVec::zeros(n_cells);
        self.v_star = AlignedVec::zeros(n_cells);
        self.eta_prime = AlignedVec::zeros(n_cells);
        self.rhs = AlignedVec::zeros(n_cells);

        self.precond = JacobiPreconditioner::from_diagonal(&vec![1.0; n_cells]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SemiImplicitConfig::default();
        assert!((config.gravity - 9.81).abs() < 1e-10);
        assert!((config.theta - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_config_variants() {
        let conservative = SemiImplicitConfig::conservative();
        let fast = SemiImplicitConfig::fast();

        assert!(conservative.solver_rtol < fast.solver_rtol);
        assert!(conservative.solver_max_iter > fast.solver_max_iter);
    }

    #[test]
    fn test_stats_default() {
        let stats = SemiImplicitStats::default();
        assert_eq!(stats.pressure_iterations, 0);
        assert!(!stats.pressure_converged);
    }
}
