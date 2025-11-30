// src-tauri/src/marihydro/physics/sources/turbulence_ke.rs
//! k-ε 两方程湍流模型
//!
//! 实现标准k-ε模型，用于计算涡粘系数。相比Smagorinsky模型，
//! k-ε模型能更准确地模拟非平衡湍流场。

use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::state::StateAccess;
use crate::marihydro::core::types::{CellIndex, FaceIndex, NumericalParams};
use glam::DVec2;

/// k-ε 模型标准系数
#[derive(Debug, Clone, Copy)]
pub struct KEpsilonCoefficients {
    /// Cμ系数，涡粘系数公式中的常数
    pub c_mu: f64,
    /// C₁系数，ε方程生成项系数
    pub c1: f64,
    /// C₂系数，ε方程耗散项系数
    pub c2: f64,
    /// σₖ，k方程扩散系数
    pub sigma_k: f64,
    /// σε，ε方程扩散系数
    pub sigma_eps: f64,
}

impl Default for KEpsilonCoefficients {
    fn default() -> Self {
        Self {
            c_mu: 0.09,
            c1: 1.44,
            c2: 1.92,
            sigma_k: 1.0,
            sigma_eps: 1.3,
        }
    }
}

impl KEpsilonCoefficients {
    /// Launder-Spalding 标准系数
    pub fn standard() -> Self {
        Self::default()
    }

    /// RNG k-ε 系数
    pub fn rng() -> Self {
        Self {
            c_mu: 0.0845,
            c1: 1.42,
            c2: 1.68,
            sigma_k: 0.7179,
            sigma_eps: 0.7179,
        }
    }

    /// Realizable k-ε 系数
    pub fn realizable() -> Self {
        Self {
            c_mu: 0.09,  // 实际上 Cμ 在 realizable 中是可变的
            c1: 1.44,
            c2: 1.9,
            sigma_k: 1.0,
            sigma_eps: 1.2,
        }
    }
}

/// k-ε 湍流场状态
#[derive(Debug, Clone)]
pub struct KEpsilonState {
    /// 湍动能 k [m²/s²]
    pub k: Vec<f64>,
    /// 湍动能耗散率 ε [m²/s³]
    pub epsilon: Vec<f64>,
    /// 涡粘系数 νt [m²/s]
    pub nu_t: Vec<f64>,
    /// 湍流生成项 Pk
    pub production: Vec<f64>,
}

impl KEpsilonState {
    /// 创建新的状态场
    pub fn new(n_cells: usize) -> Self {
        Self {
            k: vec![1e-6; n_cells],
            epsilon: vec![1e-8; n_cells],
            nu_t: vec![1e-6; n_cells],
            production: vec![0.0; n_cells],
        }
    }

    /// 使用初始值创建
    pub fn with_initial(n_cells: usize, k0: f64, eps0: f64) -> Self {
        Self {
            k: vec![k0; n_cells],
            epsilon: vec![eps0; n_cells],
            nu_t: vec![1e-6; n_cells],
            production: vec![0.0; n_cells],
        }
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        self.k.resize(n_cells, 1e-6);
        self.epsilon.resize(n_cells, 1e-8);
        self.nu_t.resize(n_cells, 1e-6);
        self.production.resize(n_cells, 0.0);
    }
}

/// k-ε 边界条件类型
#[derive(Debug, Clone, Copy)]
pub enum KEpsilonBoundary {
    /// 壁面函数边界
    WallFunction {
        /// von Kármán 常数
        kappa: f64,
        /// 光滑壁面常数
        e_constant: f64,
        /// 壁面距离
        y_plus: f64,
    },
    /// 自由滑移（零法向梯度）
    Slip,
    /// 入流边界
    Inflow { k: f64, epsilon: f64 },
    /// 出流边界（零梯度）
    Outflow,
}

impl Default for KEpsilonBoundary {
    fn default() -> Self {
        Self::WallFunction {
            kappa: 0.41,
            e_constant: 9.793,
            y_plus: 30.0,
        }
    }
}

/// k-ε 模型配置
#[derive(Debug, Clone)]
pub struct KEpsilonConfig {
    /// 模型系数
    pub coefficients: KEpsilonCoefficients,
    /// 最小湍动能
    pub k_min: f64,
    /// 最大湍动能
    pub k_max: f64,
    /// 最小耗散率
    pub eps_min: f64,
    /// 最大耗散率
    pub eps_max: f64,
    /// 最小涡粘系数
    pub nu_t_min: f64,
    /// 最大涡粘系数
    pub nu_t_max: f64,
    /// 湍流强度（用于入流边界）
    pub turbulence_intensity: f64,
    /// 湍流长度尺度比（与水深的比值）
    pub length_scale_ratio: f64,
}

impl Default for KEpsilonConfig {
    fn default() -> Self {
        Self {
            coefficients: KEpsilonCoefficients::default(),
            k_min: 1e-10,
            k_max: 1e4,
            eps_min: 1e-12,
            eps_max: 1e6,
            nu_t_min: 1e-8,
            nu_t_max: 1e4,
            turbulence_intensity: 0.05,
            length_scale_ratio: 0.07,
        }
    }
}

impl KEpsilonConfig {
    /// 从速度和水深估算入流边界条件
    pub fn estimate_inflow_bc(&self, velocity_mag: f64, depth: f64) -> KEpsilonBoundary {
        // k = 1.5 * (I * U)²
        let k = 1.5 * (self.turbulence_intensity * velocity_mag).powi(2);
        // L = ratio * H
        let length_scale = self.length_scale_ratio * depth;
        // ε = Cμ^0.75 * k^1.5 / L
        let epsilon = self.coefficients.c_mu.powf(0.75) * k.powf(1.5) / length_scale.max(1e-6);

        KEpsilonBoundary::Inflow {
            k: k.max(self.k_min),
            epsilon: epsilon.max(self.eps_min),
        }
    }
}

/// k-ε 两方程湍流模型求解器
pub struct KEpsilonSolver {
    /// 配置
    config: KEpsilonConfig,
    /// 速度场工作缓冲区
    velocities: Vec<DVec2>,
    /// 速度梯度工作缓冲区
    du_dx: Vec<f64>,
    du_dy: Vec<f64>,
    dv_dx: Vec<f64>,
    dv_dy: Vec<f64>,
    /// k 梯度
    dk_dx: Vec<f64>,
    dk_dy: Vec<f64>,
    /// ε 梯度
    deps_dx: Vec<f64>,
    deps_dy: Vec<f64>,
    /// k 的 RHS
    rhs_k: Vec<f64>,
    /// ε 的 RHS
    rhs_eps: Vec<f64>,
}

impl KEpsilonSolver {
    /// 创建新的求解器
    pub fn new(config: KEpsilonConfig, n_cells: usize) -> Self {
        Self {
            config,
            velocities: vec![DVec2::ZERO; n_cells],
            du_dx: vec![0.0; n_cells],
            du_dy: vec![0.0; n_cells],
            dv_dx: vec![0.0; n_cells],
            dv_dy: vec![0.0; n_cells],
            dk_dx: vec![0.0; n_cells],
            dk_dy: vec![0.0; n_cells],
            deps_dx: vec![0.0; n_cells],
            deps_dy: vec![0.0; n_cells],
            rhs_k: vec![0.0; n_cells],
            rhs_eps: vec![0.0; n_cells],
        }
    }

    /// 使用默认配置创建
    pub fn with_default(n_cells: usize) -> Self {
        Self::new(KEpsilonConfig::default(), n_cells)
    }

    /// 调整工作缓冲区大小
    pub fn resize(&mut self, n_cells: usize) {
        self.velocities.resize(n_cells, DVec2::ZERO);
        self.du_dx.resize(n_cells, 0.0);
        self.du_dy.resize(n_cells, 0.0);
        self.dv_dx.resize(n_cells, 0.0);
        self.dv_dy.resize(n_cells, 0.0);
        self.dk_dx.resize(n_cells, 0.0);
        self.dk_dy.resize(n_cells, 0.0);
        self.deps_dx.resize(n_cells, 0.0);
        self.deps_dy.resize(n_cells, 0.0);
        self.rhs_k.resize(n_cells, 0.0);
        self.rhs_eps.resize(n_cells, 0.0);
    }

    /// 执行一个时间步的 k-ε 方程求解
    pub fn step<M: MeshAccess, S: StateAccess>(
        &mut self,
        mesh: &M,
        state: &S,
        params: &NumericalParams,
        ke_state: &mut KEpsilonState,
        dt: f64,
    ) -> MhResult<()> {
        let n = mesh.n_cells();
        if n != self.velocities.len() {
            self.resize(n);
        }
        if n != ke_state.k.len() {
            ke_state.resize(n);
        }

        // 1. 计算速度场
        self.compute_velocities(mesh, state, params);

        // 2. 计算速度梯度
        self.compute_velocity_gradient(mesh);

        // 3. 计算 k 和 ε 的梯度
        self.compute_ke_gradient(mesh, ke_state);

        // 4. 计算涡粘系数 νt = Cμ * k² / ε
        self.update_eddy_viscosity(mesh, params, ke_state);

        // 5. 计算生成项 Pk
        self.compute_production(mesh, params, ke_state);

        // 6. 求解 k 方程
        self.solve_k_equation(mesh, state, params, ke_state, dt);

        // 7. 求解 ε 方程
        self.solve_epsilon_equation(mesh, state, params, ke_state, dt);

        // 8. 更新涡粘系数
        self.update_eddy_viscosity(mesh, params, ke_state);

        Ok(())
    }

    /// 计算速度场
    fn compute_velocities<M: MeshAccess, S: StateAccess>(
        &mut self,
        mesh: &M,
        state: &S,
        params: &NumericalParams,
    ) {
        for i in 0..mesh.n_cells() {
            let h = state.h(i);
            self.velocities[i] = if params.is_dry(h) {
                DVec2::ZERO
            } else {
                DVec2::new(state.hu(i) / h, state.hv(i) / h)
            };
        }
    }

    /// 使用 Green-Gauss 方法计算速度梯度
    fn compute_velocity_gradient<M: MeshAccess>(&mut self, mesh: &M) {
        self.du_dx.fill(0.0);
        self.du_dy.fill(0.0);
        self.dv_dx.fill(0.0);
        self.dv_dy.fill(0.0);

        for i in 0..mesh.n_cells() {
            let cell = CellIndex(i);
            let area = mesh.cell_area(cell);
            if area < 1e-14 {
                continue;
            }

            let mut gu = DVec2::ZERO;
            let mut gv = DVec2::ZERO;

            for &face in mesh.cell_faces(cell) {
                let owner = mesh.face_owner(face);
                let neighbor = mesh.face_neighbor(face);
                let normal = mesh.face_normal(face);
                let length = mesh.face_length(face);
                let sign = if i == owner.0 { 1.0 } else { -1.0 };
                let ds = normal * length * sign;

                let vel_face = if !neighbor.is_valid() {
                    self.velocities[i]
                } else {
                    let o = if i == owner.0 { neighbor.0 } else { owner.0 };
                    (self.velocities[i] + self.velocities[o]) * 0.5
                };

                gu += ds * vel_face.x;
                gv += ds * vel_face.y;
            }

            self.du_dx[i] = gu.x / area;
            self.du_dy[i] = gu.y / area;
            self.dv_dx[i] = gv.x / area;
            self.dv_dy[i] = gv.y / area;
        }
    }

    /// 计算 k 和 ε 的梯度
    fn compute_ke_gradient<M: MeshAccess>(&mut self, mesh: &M, ke_state: &KEpsilonState) {
        self.dk_dx.fill(0.0);
        self.dk_dy.fill(0.0);
        self.deps_dx.fill(0.0);
        self.deps_dy.fill(0.0);

        for i in 0..mesh.n_cells() {
            let cell = CellIndex(i);
            let area = mesh.cell_area(cell);
            if area < 1e-14 {
                continue;
            }

            let mut gk = DVec2::ZERO;
            let mut geps = DVec2::ZERO;

            for &face in mesh.cell_faces(cell) {
                let owner = mesh.face_owner(face);
                let neighbor = mesh.face_neighbor(face);
                let normal = mesh.face_normal(face);
                let length = mesh.face_length(face);
                let sign = if i == owner.0 { 1.0 } else { -1.0 };
                let ds = normal * length * sign;

                let (k_face, eps_face) = if !neighbor.is_valid() {
                    (ke_state.k[i], ke_state.epsilon[i])
                } else {
                    let o = if i == owner.0 { neighbor.0 } else { owner.0 };
                    (
                        (ke_state.k[i] + ke_state.k[o]) * 0.5,
                        (ke_state.epsilon[i] + ke_state.epsilon[o]) * 0.5,
                    )
                };

                gk += ds * k_face;
                geps += ds * eps_face;
            }

            self.dk_dx[i] = gk.x / area;
            self.dk_dy[i] = gk.y / area;
            self.deps_dx[i] = geps.x / area;
            self.deps_dy[i] = geps.y / area;
        }
    }

    /// 更新涡粘系数
    fn update_eddy_viscosity<M: MeshAccess>(
        &self,
        mesh: &M,
        params: &NumericalParams,
        ke_state: &mut KEpsilonState,
    ) {
        let c_mu = self.config.coefficients.c_mu;

        for i in 0..mesh.n_cells() {
            let cell = CellIndex(i);
            let area = mesh.cell_area(cell);
            if area < 1e-14 {
                ke_state.nu_t[i] = self.config.nu_t_min;
                continue;
            }

            let k = ke_state.k[i];
            let eps = ke_state.epsilon[i].max(self.config.eps_min);

            // νt = Cμ * k² / ε
            let nu_t = c_mu * k * k / eps;
            ke_state.nu_t[i] = nu_t.clamp(self.config.nu_t_min, self.config.nu_t_max);
        }
    }

    /// 计算湍流生成项 Pk
    fn compute_production<M: MeshAccess>(
        &self,
        mesh: &M,
        params: &NumericalParams,
        ke_state: &mut KEpsilonState,
    ) {
        for i in 0..mesh.n_cells() {
            let cell = CellIndex(i);
            let area = mesh.cell_area(cell);
            if area < 1e-14 {
                ke_state.production[i] = 0.0;
                continue;
            }

            // 应变率张量 Sij
            let s11 = self.du_dx[i];
            let s22 = self.dv_dy[i];
            let s12 = 0.5 * (self.du_dy[i] + self.dv_dx[i]);

            // |S| = sqrt(2 * Sij * Sij)
            let s_mag_sq = 2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12);

            // Pk = νt * |S|²
            let nu_t = ke_state.nu_t[i];
            ke_state.production[i] = nu_t * s_mag_sq;
        }
    }

    /// 求解 k 方程
    /// ∂k/∂t + u·∇k = Pk - ε + ∇·(νt/σk ∇k)
    fn solve_k_equation<M: MeshAccess, S: StateAccess>(
        &mut self,
        mesh: &M,
        state: &S,
        params: &NumericalParams,
        ke_state: &mut KEpsilonState,
        dt: f64,
    ) {
        let sigma_k = self.config.coefficients.sigma_k;

        // 初始化 RHS
        self.rhs_k.fill(0.0);

        // 对流项：-u·∇k（使用迎风格式）
        for i in 0..mesh.n_cells() {
            let advection = self.velocities[i].x * self.dk_dx[i]
                + self.velocities[i].y * self.dk_dy[i];
            self.rhs_k[i] -= advection;
        }

        // 扩散项：∇·(νt/σk ∇k)
        for face_idx in 0..mesh.n_faces() {
            let face = FaceIndex(face_idx);
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);

            if !neighbor.is_valid() {
                continue;
            }

            let dist = (mesh.cell_centroid(neighbor) - mesh.cell_centroid(owner)).length();
            if dist < 1e-14 {
                continue;
            }

            let nu_face = 0.5 * (ke_state.nu_t[owner.0] + ke_state.nu_t[neighbor.0]) / sigma_k;
            let dk = ke_state.k[neighbor.0] - ke_state.k[owner.0];
            let length = mesh.face_length(face);

            let flux = nu_face * dk / dist * length;

            let area_o = mesh.cell_area(owner);
            let area_n = mesh.cell_area(neighbor);

            if area_o > 1e-14 {
                self.rhs_k[owner.0] += flux / area_o;
            }
            if area_n > 1e-14 {
                self.rhs_k[neighbor.0] -= flux / area_n;
            }
        }

        // 源项：Pk - ε
        for i in 0..mesh.n_cells() {
            self.rhs_k[i] += ke_state.production[i] - ke_state.epsilon[i];
        }

        // 时间推进（显式欧拉）
        for i in 0..mesh.n_cells() {
            let h = state.h(i);
            if params.is_dry(h) {
                ke_state.k[i] = self.config.k_min;
                continue;
            }

            let k_new = ke_state.k[i] + dt * self.rhs_k[i];
            ke_state.k[i] = k_new.clamp(self.config.k_min, self.config.k_max);
        }
    }

    /// 求解 ε 方程
    /// ∂ε/∂t + u·∇ε = C1 * ε/k * Pk - C2 * ε²/k + ∇·(νt/σε ∇ε)
    fn solve_epsilon_equation<M: MeshAccess, S: StateAccess>(
        &mut self,
        mesh: &M,
        state: &S,
        params: &NumericalParams,
        ke_state: &mut KEpsilonState,
        dt: f64,
    ) {
        let c1 = self.config.coefficients.c1;
        let c2 = self.config.coefficients.c2;
        let sigma_eps = self.config.coefficients.sigma_eps;

        // 初始化 RHS
        self.rhs_eps.fill(0.0);

        // 对流项：-u·∇ε
        for i in 0..mesh.n_cells() {
            let advection = self.velocities[i].x * self.deps_dx[i]
                + self.velocities[i].y * self.deps_dy[i];
            self.rhs_eps[i] -= advection;
        }

        // 扩散项：∇·(νt/σε ∇ε)
        for face_idx in 0..mesh.n_faces() {
            let face = FaceIndex(face_idx);
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);

            if !neighbor.is_valid() {
                continue;
            }

            let dist = (mesh.cell_centroid(neighbor) - mesh.cell_centroid(owner)).length();
            if dist < 1e-14 {
                continue;
            }

            let nu_face = 0.5 * (ke_state.nu_t[owner.0] + ke_state.nu_t[neighbor.0]) / sigma_eps;
            let deps = ke_state.epsilon[neighbor.0] - ke_state.epsilon[owner.0];
            let length = mesh.face_length(face);

            let flux = nu_face * deps / dist * length;

            let area_o = mesh.cell_area(owner);
            let area_n = mesh.cell_area(neighbor);

            if area_o > 1e-14 {
                self.rhs_eps[owner.0] += flux / area_o;
            }
            if area_n > 1e-14 {
                self.rhs_eps[neighbor.0] -= flux / area_n;
            }
        }

        // 源项：C1 * ε/k * Pk - C2 * ε²/k
        for i in 0..mesh.n_cells() {
            let k = ke_state.k[i].max(self.config.k_min);
            let eps = ke_state.epsilon[i];
            let pk = ke_state.production[i];

            let source = c1 * eps / k * pk - c2 * eps * eps / k;
            self.rhs_eps[i] += source;
        }

        // 时间推进（显式欧拉）
        for i in 0..mesh.n_cells() {
            let h = state.h(i);
            if params.is_dry(h) {
                ke_state.epsilon[i] = self.config.eps_min;
                continue;
            }

            let eps_new = ke_state.epsilon[i] + dt * self.rhs_eps[i];
            ke_state.epsilon[i] = eps_new.clamp(self.config.eps_min, self.config.eps_max);
        }
    }

    /// 获取涡粘系数（只读访问）
    pub fn eddy_viscosity<'a>(&'a self, ke_state: &'a KEpsilonState) -> &'a [f64] {
        &ke_state.nu_t
    }

    /// 估算稳定时间步长
    pub fn estimate_stable_dt<M: MeshAccess>(
        &self,
        mesh: &M,
        ke_state: &KEpsilonState,
    ) -> f64 {
        let mut dt_min = f64::MAX;
        let sigma_k = self.config.coefficients.sigma_k;

        for i in 0..mesh.n_cells() {
            let cell = CellIndex(i);
            let area = mesh.cell_area(cell);
            if area < 1e-14 {
                continue;
            }

            let nu_eff = ke_state.nu_t[i] / sigma_k;
            if nu_eff < 1e-14 {
                continue;
            }

            // 扩散稳定性条件：dt < dx² / (2 * ν)
            let dx = area.sqrt();
            let dt_diff = 0.5 * dx * dx / nu_eff;

            // 对流稳定性条件（CFL）
            let vel_mag = self.velocities[i].length();
            let dt_conv = if vel_mag > 1e-10 { dx / vel_mag } else { f64::MAX };

            dt_min = dt_min.min(dt_diff).min(dt_conv);
        }

        dt_min * 0.5 // 安全系数
    }
}

/// 湍流模型枚举，统一 Smagorinsky 和 k-ε 接口
pub enum TurbulenceModel {
    /// Smagorinsky 代数模型
    Smagorinsky { cs: f64 },
    /// k-ε 两方程模型
    KEpsilon {
        solver: KEpsilonSolver,
        state: KEpsilonState,
    },
}

impl TurbulenceModel {
    /// 创建 Smagorinsky 模型
    pub fn smagorinsky(cs: f64) -> Self {
        Self::Smagorinsky { cs }
    }

    /// 创建 k-ε 模型
    pub fn k_epsilon(n_cells: usize) -> Self {
        Self::KEpsilon {
            solver: KEpsilonSolver::with_default(n_cells),
            state: KEpsilonState::new(n_cells),
        }
    }

    /// 创建带配置的 k-ε 模型
    pub fn k_epsilon_with_config(n_cells: usize, config: KEpsilonConfig) -> Self {
        Self::KEpsilon {
            solver: KEpsilonSolver::new(config, n_cells),
            state: KEpsilonState::new(n_cells),
        }
    }

    /// 获取涡粘系数
    pub fn get_eddy_viscosity(&self) -> Option<&[f64]> {
        match self {
            Self::Smagorinsky { .. } => None,
            Self::KEpsilon { state, .. } => Some(&state.nu_t),
        }
    }

    /// 更新模型状态
    pub fn step<M: MeshAccess, S: StateAccess>(
        &mut self,
        mesh: &M,
        state: &S,
        params: &NumericalParams,
        dt: f64,
    ) -> MhResult<()> {
        match self {
            Self::Smagorinsky { .. } => Ok(()),
            Self::KEpsilon {
                solver,
                state: ke_state,
            } => solver.step(mesh, state, params, ke_state, dt),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ke_coefficients_default() {
        let coeff = KEpsilonCoefficients::default();
        assert!((coeff.c_mu - 0.09).abs() < 1e-10);
        assert!((coeff.c1 - 1.44).abs() < 1e-10);
        assert!((coeff.c2 - 1.92).abs() < 1e-10);
    }

    #[test]
    fn test_ke_state_creation() {
        let state = KEpsilonState::new(100);
        assert_eq!(state.k.len(), 100);
        assert_eq!(state.epsilon.len(), 100);
        assert_eq!(state.nu_t.len(), 100);
    }

    #[test]
    fn test_inflow_bc_estimation() {
        let config = KEpsilonConfig::default();
        let bc = config.estimate_inflow_bc(1.0, 2.0);
        match bc {
            KEpsilonBoundary::Inflow { k, epsilon } => {
                assert!(k > 0.0);
                assert!(epsilon > 0.0);
            }
            _ => panic!("Expected Inflow boundary"),
        }
    }

    #[test]
    fn test_turbulence_model_enum() {
        let smag = TurbulenceModel::smagorinsky(0.17);
        assert!(smag.get_eddy_viscosity().is_none());

        let ke = TurbulenceModel::k_epsilon(100);
        assert!(ke.get_eddy_viscosity().is_some());
    }
}
