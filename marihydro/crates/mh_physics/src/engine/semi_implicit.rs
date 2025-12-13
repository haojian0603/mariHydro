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
    JacobiPreconditioner, PcgSolver, Preconditioner, SolverConfig, SolverResult, SolverStatus,
};
use crate::schemes::riemann::{HllcSolver, RiemannSolver, SolverParams as RiemannParams};
use crate::state::ShallowWaterState;
use crate::types::PhysicalConstants;
use glam::DVec2;
use mh_foundation::AlignedVec;
use serde::{Deserialize, Serialize};

/// 半隐式配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemiImplicitConfig {
    /// 物理常数（包含重力等权威值）
    pub constants: PhysicalConstants,
    /// 最小水深阈值 [m]
    pub h_min: f64, // ALLOW_F64: Layer 4 配置参数
    /// 干单元水深阈值 [m]
    pub h_dry: f64, // ALLOW_F64: Layer 4 配置参数
    /// 线性求解器相对容差
    pub solver_rtol: f64, // ALLOW_F64: Layer 4 配置参数
    /// 线性求解器绝对容差
    pub solver_atol: f64, // ALLOW_F64: Layer 4 配置参数
    /// 线性求解器最大迭代次数
    pub solver_max_iter: usize,
    /// 预测步的隐式因子 (0=显式, 1=全隐式, 0.5=Crank-Nicolson)
    pub theta: f64, // ALLOW_F64: Layer 4 配置参数
    /// 是否打印求解器信息
    pub verbose: bool,
}

impl Default for SemiImplicitConfig {
    fn default() -> Self {
        Self {
            constants: PhysicalConstants::seawater(),
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
    pub pressure_residual: f64, // ALLOW_F64: Layer 4 配置参数
    /// 压力求解状态
    pub pressure_converged: bool,
    /// 最大水深校正量
    pub max_depth_correction: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最大速度校正量
    pub max_velocity_correction: f64, // ALLOW_F64: Layer 4 配置参数
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
    solver: PcgSolver<f64>,
    /// 预条件器
    precond: JacobiPreconditioner<f64>,
    /// 求解器工作区
    workspace: crate::numerics::linear_algebra::CgWorkspace<f64>,
    /// 水深校正器
    depth_corrector: DepthCorrector,
    /// 速度校正器
    velocity_corrector: VelocityCorrector,
    /// 当前时刻速度 u^n
    u_n: AlignedVec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
    /// 当前时刻速度 v^n
    v_n: AlignedVec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
    /// 预测速度 u*
    u_star: AlignedVec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
    /// 预测速度 v*
    v_star: AlignedVec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
    /// 对流通量累加（u方向）
    advection_flux_u: AlignedVec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
    /// 对流通量累加（v方向）
    advection_flux_v: AlignedVec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
    /// 扩散通量累加（u方向）
    diffusion_flux_u: AlignedVec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
    /// 扩散通量累加（v方向）
    diffusion_flux_v: AlignedVec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
    /// 水位校正量 η'
    eta_prime: AlignedVec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
    /// 上一步水位校正量
    d_eta_prev: AlignedVec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
    /// 右端项
    rhs: AlignedVec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
    /// 上一步干湿掩码
    prev_wet_mask: Vec<bool>,
    /// CFL 历史记录
    cfl_history: Vec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
    /// 最新统计
    stats: SemiImplicitStats,
    /// Riemann 求解器（用于面通量计算）
    riemann_solver: HllcSolver,
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

        // 初始化求解器工作区
        let workspace = crate::numerics::linear_algebra::CgWorkspace::new(n_cells);

        let depth_corrector = DepthCorrector::new(n_cells).with_h_min(config.h_min);
        let velocity_corrector = VelocityCorrector::new(&topo).with_h_min(config.h_dry);

        // 初始化 Riemann 求解器
        let riemann_params = RiemannParams {
            gravity: config.constants.g,
            h_dry: config.h_dry,
            h_min: config.h_min,
            flux_eps: 1e-14,
            entropy_ratio: 0.1,
        };
        let riemann_solver = HllcSolver::from_params(riemann_params);

        Self {
            config,
            topo,
            pressure_assembler,
            solver,
            precond,
            workspace,
            depth_corrector,
            velocity_corrector,
            u_n: AlignedVec::zeros(n_cells),
            v_n: AlignedVec::zeros(n_cells),
            u_star: AlignedVec::zeros(n_cells),
            v_star: AlignedVec::zeros(n_cells),
            advection_flux_u: AlignedVec::zeros(n_cells),
            advection_flux_v: AlignedVec::zeros(n_cells),
            diffusion_flux_u: AlignedVec::zeros(n_cells),
            diffusion_flux_v: AlignedVec::zeros(n_cells),
            eta_prime: AlignedVec::zeros(n_cells),
            d_eta_prev: AlignedVec::zeros(n_cells),
            rhs: AlignedVec::zeros(n_cells),
            prev_wet_mask: vec![false; n_cells],
            cfl_history: Vec::new(),
            stats: SemiImplicitStats::default(),
            riemann_solver,
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

    /// 执行半隐式时间推进（返回是否成功收敛）
    ///
    /// # 参数
    ///
    /// - `state`: 浅水状态（将被修改）
    /// - `mesh`: 物理网格
    /// - `dt`: 时间步长
    ///
    /// # 返回
    ///
    /// 线性求解器是否收敛
    pub fn step(&mut self, mesh: &PhysicsMesh, state: &mut ShallowWaterState, dt: f64) -> bool { // ALLOW_F64: 时间步长参数
        // 重置统计
        self.stats = SemiImplicitStats::default();

        // 1. CFL 检查
        let cfl = self.compute_cfl(state, mesh, dt);
        self.cfl_history.push(cfl);
        if cfl > 1.0 {
            log::warn!("半隐式 CFL={:.2} > 1.0，可能影响精度", cfl);
        }

        // 统计干湿单元
        for i in 0..state.n_cells() {
            if state.h[i] > self.config.h_dry {
                self.stats.n_wet_cells += 1;
            } else {
                self.stats.n_dry_cells += 1;
            }
        }

        // 2. 从动量恢复速度
        self.extract_velocity(state);

        // 3. 预测步（对流+扩散）
        self.compute_prediction_step(mesh, state, dt);

        // 4. 组装压力矩阵
        self.pressure_assembler.assemble(
            mesh,
            &self.topo,
            state,
            dt,
            self.config.constants.g,
        );

        // 5. 计算散度 RHS
        self.compute_divergence_rhs(mesh, state, dt);

        // 6. 干单元边界条件
        let wet_mask = self.compute_wet_mask(state);
        self.apply_dry_cell_bc(&wet_mask);

        // 7. 更新预条件器并求解
        self.precond.update(self.pressure_assembler.matrix());

        self.eta_prime.as_mut_slice().fill(0.0);
        let result = self.solver.solve_with_workspace(
            self.pressure_assembler.matrix(),
            self.rhs.as_slice(),
            self.eta_prime.as_mut_slice(),
            &self.precond,
            &mut self.workspace,
        );

        self.update_solver_stats(&result);

        // 8. 速度和水深校正
        self.velocity_correction(mesh, state, dt);
        self.depth_correction(state);

        // 更新干湿掩码历史
        self.prev_wet_mask.copy_from_slice(&wet_mask);

        // 保存上一步校正量
        self.d_eta_prev
            .as_mut_slice()
            .copy_from_slice(self.eta_prime.as_slice());

        result.is_converged()
    }

    /// 执行半隐式时间推进（兼容旧接口）
    pub fn advance(&mut self, state: &mut ShallowWaterState, mesh: &PhysicsMesh, dt: f64) { // ALLOW_F64: 时间步长参数
        let _ = self.step(mesh, state, dt);
    }

    /// 计算 CFL 数
    fn compute_cfl(&self, state: &ShallowWaterState, mesh: &PhysicsMesh, dt: f64) -> f64 { // ALLOW_F64: 时间步长参数
        let g = self.config.constants.g;
        let h_dry = self.config.h_dry;

        (0..state.n_cells())
            .map(|i| {
                let h = state.h[i];
                if h < h_dry {
                    return 0.0;
                }
                let u = state.hu[i] / h;
                let v = state.hv[i] / h;
                let c = (g * h).sqrt();
                let vel = (u * u + v * v).sqrt() + c;
                // 使用 sqrt(面积) 作为特征尺寸
                let dx = mesh.cell_area_unchecked(i).sqrt();
                if dx > 1e-14 {
                    vel * dt / dx
                } else {
                    0.0
                }
            })
            .fold(0.0, f64::max)
    }

    /// 从动量恢复速度场
    fn extract_velocity(&mut self, state: &ShallowWaterState) {
        let h_dry = self.config.h_dry;
        for i in 0..state.n_cells() {
            let h = state.h[i];
            if h > h_dry {
                self.u_n[i] = state.hu[i] / h;
                self.v_n[i] = state.hv[i] / h;
            } else {
                self.u_n[i] = 0.0;
                self.v_n[i] = 0.0;
            }
        }
    }

    /// 计算预测步（对流+扩散）
    fn compute_prediction_step(&mut self, mesh: &PhysicsMesh, state: &ShallowWaterState, dt: f64) { // ALLOW_F64: 时间步长参数
        // 清零通量累加器
        self.advection_flux_u.as_mut_slice().fill(0.0);
        self.advection_flux_v.as_mut_slice().fill(0.0);
        self.diffusion_flux_u.as_mut_slice().fill(0.0);
        self.diffusion_flux_v.as_mut_slice().fill(0.0);

        let h_dry = self.config.h_dry;

        // 对流通量计算（使用 HLLC Riemann 求解器）
        for &face_idx in self.topo.interior_faces() {
            let face = self.topo.face(face_idx);
            let owner = face.owner;
            let neighbor = face.neighbor.expect("interior face");

            let h_o = state.h[owner];
            let h_n = state.h[neighbor];

            // 跳过干面
            if h_o < h_dry && h_n < h_dry {
                continue;
            }

            let vel_o = DVec2::new(self.u_n[owner], self.v_n[owner]);
            let vel_n = DVec2::new(self.u_n[neighbor], self.v_n[neighbor]);
            let normal = DVec2::new(face.normal.x, face.normal.y);

            // 使用 Riemann 求解器计算面通量
            let flux = match self.riemann_solver.solve(h_o, h_n, vel_o, vel_n, normal) {
                Ok(f) => f,
                Err(_) => continue, // 数值问题时跳过
            };

            // 通量乘以面长度
            let face_length = face.length;
            let cell_area_o = mesh.cell_area_unchecked(owner);
            let cell_area_n = mesh.cell_area_unchecked(neighbor);

            // 动量通量累加（注意：flux.momentum 已是守恒形式 h*u*u + g*h²/2）
            // 对于 owner: 通量为正（流出）时减小动量
            // 对于 neighbor: 通量为负（流入）时增加动量
            self.advection_flux_u[owner] -= flux.momentum_x * face_length / cell_area_o;
            self.advection_flux_v[owner] -= flux.momentum_y * face_length / cell_area_o;
            self.advection_flux_u[neighbor] += flux.momentum_x * face_length / cell_area_n;
            self.advection_flux_v[neighbor] += flux.momentum_y * face_length / cell_area_n;
        }

        // 边界面处理（自由滑移）
        for &face_idx in self.topo.boundary_faces() {
            let face = self.topo.face(face_idx);
            let owner = face.owner;

            if state.h[owner] < h_dry {
                continue;
            }

            let (u_o, v_o) = (self.u_n[owner], self.v_n[owner]);
            // 自由滑移：法向速度为零
            let vn = u_o * face.normal.x + v_o * face.normal.y;
            let _u_boundary = u_o - 2.0 * vn * face.normal.x;
            let _v_boundary = v_o - 2.0 * vn * face.normal.y;
            // 边界面不贡献对流通量（固壁）
        }

        // 更新预测速度
        for i in 0..state.n_cells() {
            if state.h[i] <= h_dry {
                self.u_star[i] = 0.0;
                self.v_star[i] = 0.0;
                continue;
            }
            let inv_area = 1.0 / mesh.cell_area_unchecked(i);
            self.u_star[i] = self.u_n[i]
                + dt * (self.advection_flux_u[i] + self.diffusion_flux_u[i]) * inv_area;
            self.v_star[i] = self.v_n[i]
                + dt * (self.advection_flux_v[i] + self.diffusion_flux_v[i]) * inv_area;
        }
    }

    /// 计算干湿掩码
    fn compute_wet_mask(&self, state: &ShallowWaterState) -> Vec<bool> {
        (0..state.n_cells())
            .map(|i| state.h[i] > self.config.h_dry)
            .collect()
    }

    /// 应用干单元边界条件
    fn apply_dry_cell_bc(&mut self, wet_mask: &[bool]) {
        for i in 0..wet_mask.len() {
            if !wet_mask[i] {
                self.rhs[i] = 0.0;
            }
        }
    }

    /// 速度校正（Green-Gauss 梯度）
    fn velocity_correction(&mut self, mesh: &PhysicsMesh, state: &mut ShallowWaterState, dt: f64) { // ALLOW_F64: 时间步长参数
        let g = self.config.constants.g;
        let theta = self.config.theta;
        let coeff = g * theta * dt;
        let h_dry = self.config.h_dry;

        for i in 0..mesh.n_cells() {
            if state.h[i] < h_dry {
                continue;
            }

            // Green-Gauss 梯度计算
            let (grad_x, grad_y) = self.compute_gradient(mesh, i);

            let u_new = self.u_star[i] - coeff * grad_x;
            let v_new = self.v_star[i] - coeff * grad_y;
            let h_new = (state.h[i] + self.eta_prime[i]).max(0.0);

            state.hu[i] = h_new * u_new;
            state.hv[i] = h_new * v_new;
        }

        // 计算速度校正统计
        self.compute_velocity_correction_stats(state);
    }

    /// 使用 Green-Gauss 方法计算水位校正量的梯度
    fn compute_gradient(&self, mesh: &PhysicsMesh, cell_idx: usize) -> (f64, f64) {
        let mut grad_x = 0.0;
        let mut grad_y = 0.0;

        let cell_faces = self.topo.cell_faces(cell_idx);
        let area = mesh.cell_area_unchecked(cell_idx);

        if area < 1e-14 {
            return (0.0, 0.0);
        }

        for &face_idx in cell_faces {
            let face = self.topo.face(face_idx);
            let owner = face.owner;

            // 面上的值（算术平均）
            let eta_face = if let Some(neigh) = face.neighbor {
                0.5 * (self.eta_prime[owner] + self.eta_prime[neigh])
            } else {
                self.eta_prime[owner]
            };

            // 根据面方向调整符号
            let sign = if owner == cell_idx { 1.0 } else { -1.0 };

            grad_x += sign * eta_face * face.normal.x * face.length;
            grad_y += sign * eta_face * face.normal.y * face.length;
        }

        (grad_x / area, grad_y / area)
    }

    /// 水深校正
    fn depth_correction(&self, state: &mut ShallowWaterState) {
        for i in 0..state.n_cells() {
            state.h[i] = (state.h[i] + self.eta_prime[i]).max(0.0);
        }
    }

    /// 计算散度右端项
    fn compute_divergence_rhs(
        &mut self,
        mesh: &PhysicsMesh,
        state: &ShallowWaterState,
        dt: f64, // ALLOW_F64: 时间步长参数
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
    fn update_solver_stats(&mut self, result: &SolverResult<f64>) {
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
        let mut max_correction: f64 = 0.0; // ALLOW_F64: 临时计算变量
        let h_dry = self.config.h_dry;

        for i in 0..state.n_cells() {
            let h = state.h[i];
            if h < h_dry {
                continue;
            }
            // 从动量计算当前速度
            let u_new = state.hu[i] / h;
            let v_new = state.hv[i] / h;
            let du = u_new - self.u_star[i];
            let dv = v_new - self.v_star[i];
            let correction = (du * du + dv * dv).sqrt();
            max_correction = max_correction.max(correction);
        }

        self.stats.max_velocity_correction = max_correction;
        self.stats.max_depth_correction = self
            .eta_prime
            .as_slice()
            .iter()
            .map(|x| x.abs())
            .fold(0.0, f64::max);
    }

    /// 获取拓扑引用
    pub fn topology(&self) -> &CellFaceTopology {
        &self.topo
    }

    /// 获取水位校正量引用
    pub fn eta_prime(&self) -> &[f64] {
        self.eta_prime.as_slice()
    }

    /// 重置策略（网格变化时调用）
    pub fn reset(&mut self, mesh: &PhysicsMesh) {
        let n_cells = mesh.n_cells();

        self.topo = CellFaceTopology::from_mesh(mesh);
        self.pressure_assembler = PressureMatrixAssembler::new(&self.topo);
        self.depth_corrector = DepthCorrector::new(n_cells).with_h_min(self.config.h_min);
        self.velocity_corrector = VelocityCorrector::new(&self.topo).with_h_min(self.config.h_dry);

        self.workspace = crate::numerics::linear_algebra::CgWorkspace::new(n_cells);

        self.u_n = AlignedVec::zeros(n_cells);
        self.v_n = AlignedVec::zeros(n_cells);
        self.u_star = AlignedVec::zeros(n_cells);
        self.v_star = AlignedVec::zeros(n_cells);
        self.advection_flux_u = AlignedVec::zeros(n_cells);
        self.advection_flux_v = AlignedVec::zeros(n_cells);
        self.diffusion_flux_u = AlignedVec::zeros(n_cells);
        self.diffusion_flux_v = AlignedVec::zeros(n_cells);
        self.eta_prime = AlignedVec::zeros(n_cells);
        self.d_eta_prev = AlignedVec::zeros(n_cells);
        self.rhs = AlignedVec::zeros(n_cells);
        self.prev_wet_mask = vec![false; n_cells];
        self.cfl_history.clear();

        self.precond = JacobiPreconditioner::from_diagonal(&vec![1.0; n_cells]);

        // 重新初始化 Riemann 求解器
        let riemann_params = RiemannParams {
            gravity: self.config.constants.g,
            h_dry: self.config.h_dry,
            h_min: self.config.h_min,
            flux_eps: 1e-14,
            entropy_ratio: 0.1,
        };
        self.riemann_solver = HllcSolver::from_params(riemann_params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SemiImplicitConfig::default();
        assert!((config.constants.g - 9.81).abs() < 1e-10);
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
