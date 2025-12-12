// crates/mh_physics/src/numerics/operators/diffusion.rs

//! 扩散算子
//!
//! 实现标量场的显式扩散求解，支持多种边界条件。
//! 
//! # 模块说明
//!
//! 这是一个**数值算子**而非物理源项。可应用于任意标量场
//! （温度、盐度、示踪剂、动量等）的扩散计算。
//!
//! # 算法
//!
//! 使用基于面的有限体积方法：
//! ```text
//! dφ/dt = ν∇²φ ≈ (1/A) Σ ν (φ_n - φ) / d × L
//! ```
//!
//! # 边界条件
//!
//! - `ZeroFlux`: Neumann 边界，∂φ/∂n = 0
//! - `FixedValue`: Dirichlet 边界，φ = 指定值
//! - `Radiation`: 辐射边界，flux = α(φ_∞ - φ)
//! - `SpecifiedFlux`: 指定通量边界
//!
//! # 稳定性
//!
//! 显式方法需要满足 CFL 条件: dt < α × d_min² / ν
//! 使用 `estimate_stable_dt()` 或 `apply_diffusion_auto_substeps()` 自动处理。

use rayon::prelude::*;

use crate::adapter::PhysicsMesh;

/// 扩散边界条件类型
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum DiffusionBC {
    /// 零通量 (Neumann): ∂φ/∂n = 0
    #[default]
    ZeroFlux,
    /// 固定值 (Dirichlet): φ = value
    FixedValue(f64),
    /// 辐射边界: flux = α*(φ - φ_∞)
    Radiation {
        /// 传递系数 [1/s 或 m/s 取决于场类型]
        alpha: f64,
        /// 远场值
        phi_inf: f64,
    },
    /// 指定通量: flux = value
    SpecifiedFlux(f64),
}


impl DiffusionBC {
    /// 创建 Dirichlet 边界条件
    pub fn dirichlet(value: f64) -> Self {
        Self::FixedValue(value)
    }

    /// 创建辐射边界条件
    pub fn radiation(alpha: f64, phi_inf: f64) -> Self {
        Self::Radiation { alpha, phi_inf }
    }

    /// 创建指定通量边界条件
    pub fn specified_flux(flux: f64) -> Self {
        Self::SpecifiedFlux(flux)
    }

    /// 计算边界通量贡献
    ///
    /// # 参数
    /// - `phi_cell`: 边界单元的场值
    /// - `nu`: 扩散系数
    /// - `d`: 单元中心到边界的距离
    /// - `length`: 边界面长度
    ///
    /// # 返回
    /// 边界通量（正值表示进入单元）
    #[inline]
    pub fn compute_flux(&self, phi_cell: f64, nu: f64, d: f64, length: f64) -> f64 {
        match *self {
            Self::ZeroFlux => 0.0,
            Self::FixedValue(phi_bc) => {
                // Dirichlet: F = -ν * (φ_cell - φ_bc) / d * L
                // 使用单侧差分
                -nu * (phi_cell - phi_bc) / d * length
            }
            Self::Radiation { alpha, phi_inf } => {
                // 辐射: F = α * (φ_∞ - φ_cell) * L
                alpha * (phi_inf - phi_cell) * length
            }
            Self::SpecifiedFlux(flux) => {
                // 直接指定通量
                flux * length
            }
        }
    }
}

/// 扩散求解器配置
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// 扩散系数 [m²/s]
    pub nu: f64,
    /// 边界条件（按边界索引）
    pub boundary_conditions: Vec<DiffusionBC>,
    /// CFL 安全系数
    pub cfl_safety: f64,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            nu: 1.0,
            boundary_conditions: Vec::new(),
            cfl_safety: 0.25,
        }
    }
}

impl DiffusionConfig {
    /// 创建新配置
    pub fn new(nu: f64) -> Self {
        Self {
            nu,
            ..Default::default()
        }
    }

    /// 设置边界条件
    pub fn with_boundary_conditions(mut self, bcs: Vec<DiffusionBC>) -> Self {
        self.boundary_conditions = bcs;
        self
    }

    /// 设置 CFL 安全系数
    pub fn with_cfl_safety(mut self, safety: f64) -> Self {
        self.cfl_safety = safety.clamp(0.1, 0.5);
        self
    }
}

/// 扩散求解器
pub struct DiffusionSolver<'a> {
    mesh: &'a PhysicsMesh,
    config: DiffusionConfig,
    /// 缓存的面距离最小值平方
    min_dist_sq: f64,
}

impl<'a> DiffusionSolver<'a> {
    /// 创建求解器
    pub fn new(mesh: &'a PhysicsMesh, config: DiffusionConfig) -> Self {
        // 预计算最小面距离
        let min_dist_sq = Self::compute_min_dist_sq(mesh);

        Self {
            mesh,
            config,
            min_dist_sq,
        }
    }

    /// 计算最小面距离平方
    fn compute_min_dist_sq(mesh: &PhysicsMesh) -> f64 {
        let n_faces = mesh.n_faces();
        let mut min_sq = f64::MAX;

        for face in 0..n_faces {
            if let Some(dist) = mesh.face_distance(face) {
                if dist > 1e-14 {
                    min_sq = min_sq.min(dist * dist);
                }
            }
        }

        min_sq
    }

    /// 估计稳定时间步长
    ///
    /// 对于显式扩散，CFL 条件: dt < α * d_min² / ν
    pub fn estimate_stable_dt(&self) -> f64 {
        if self.config.nu < 1e-14 {
            return f64::MAX;
        }

        if self.min_dist_sq >= f64::MAX {
            return 1.0;
        }

        self.config.cfl_safety * self.min_dist_sq / self.config.nu
    }

    /// 计算所需子步数以保证稳定性
    pub fn required_substeps(&self, dt: f64) -> usize {
        let stable_dt = self.estimate_stable_dt();
        if stable_dt >= dt {
            1
        } else {
            (dt / stable_dt).ceil() as usize
        }
    }

    /// 计算扩散通量
    fn compute_fluxes(&self, field: &[f64]) -> Vec<f64> {
        let n_cells = self.mesh.n_cells();
        let n_faces = self.mesh.n_faces();
        let nu = self.config.nu;

        let mut flux_sum = vec![0.0; n_cells];

        // 内部面
        for face in 0..n_faces {
            let owner = self.mesh.face_owner(face);
            let neighbor_opt = self.mesh.face_neighbor(face);

            if let Some(neighbor) = neighbor_opt {
                // 内部面
                let dist = self.mesh.face_distance(face).unwrap_or(1e-14);
                if dist < 1e-14 {
                    continue;
                }

                let length = self.mesh.face_length(face);
                let phi_o = field[owner];
                let phi_n = field[neighbor];

                // F = -ν * (φ_n - φ_o) / d * L
                let flux = -nu * (phi_n - phi_o) / dist * length;

                flux_sum[owner] += flux;
                flux_sum[neighbor] -= flux;
            } else {
                // 边界面
                let bc = self.get_boundary_condition(face);
                let dist = self.mesh.face_distance(face).unwrap_or(1e-14).max(1e-14);
                let length = self.mesh.face_length(face);
                let phi_cell = field[owner];

                let flux = bc.compute_flux(phi_cell, nu, dist, length);
                flux_sum[owner] += flux;
            }
        }

        flux_sum
    }

    /// 获取边界条件
    ///
    /// 根据面索引查找对应的边界条件。使用网格的边界 ID 映射
    /// 到配置中的边界条件向量。
    ///
    /// # 参数
    /// - `face`: 边界面索引
    ///
    /// # 返回
    /// 该面对应的扩散边界条件。如果未找到映射或配置中
    /// 没有对应条件，则返回默认的零通量边界条件。
    fn get_boundary_condition(&self, face: usize) -> DiffusionBC {
        // 获取面的边界 ID（边界条件索引）
        if let Some(boundary_id) = self.mesh.face_boundary_id(face) {
            // 根据边界 ID 查找对应的边界条件
            self.config
                .boundary_conditions
                .get(boundary_id)
                .copied()
                .unwrap_or_default()
        } else {
            // 无边界 ID（内部面或未配置），使用默认零通量
            DiffusionBC::default()
        }
    }

    /// 显式扩散求解
    ///
    /// # 参数
    /// - `field`: 输入场值
    /// - `field_out`: 输出场值
    /// - `dt`: 时间步长
    pub fn apply_explicit(
        &self,
        field: &[f64],
        field_out: &mut [f64],
        dt: f64,
    ) -> Result<(), DiffusionError> {
        self.validate_params(dt)?;

        let n_cells = self.mesh.n_cells();
        if field.len() != n_cells || field_out.len() != n_cells {
            return Err(DiffusionError::SizeMismatch {
                expected: n_cells,
                field_in: field.len(),
                field_out: field_out.len(),
            });
        }

        let flux_sum = self.compute_fluxes(field);

        field_out
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, phi_out)| {
                let area = self.mesh.cell_area(i).unwrap_or(1.0);
                if area > 1e-14 {
                    *phi_out = field[i] + dt * flux_sum[i] / area;
                } else {
                    *phi_out = field[i];
                }
            });

        Ok(())
    }

    /// 原地扩散
    pub fn apply_inplace(&self, field: &mut [f64], dt: f64) -> Result<(), DiffusionError> {
        let mut temp = vec![0.0; field.len()];
        self.apply_explicit(field, &mut temp, dt)?;
        field.copy_from_slice(&temp);
        Ok(())
    }

    /// 多子步扩散
    pub fn apply_substeps(
        &self,
        field: &mut [f64],
        dt: f64,
        n_substeps: usize,
    ) -> Result<(), DiffusionError> {
        if n_substeps == 0 {
            return Ok(());
        }

        let sub_dt = dt / n_substeps as f64;
        let mut buffer = vec![0.0; field.len()];

        for step in 0..n_substeps {
            if step % 2 == 0 {
                self.apply_explicit(field, &mut buffer, sub_dt)?;
            } else {
                self.apply_explicit(&buffer, field, sub_dt)?;
            }
        }

        // 如果子步数是奇数，最终结果在 buffer 中
        if n_substeps % 2 == 1 {
            field.copy_from_slice(&buffer);
        }

        Ok(())
    }

    /// 自动子步扩散
    ///
    /// 自动计算所需子步数以保证稳定性
    pub fn apply_auto_substeps(&self, field: &mut [f64], dt: f64) -> Result<usize, DiffusionError> {
        let n_substeps = self.required_substeps(dt);

        if n_substeps > 1 {
            // 在需要时记录调试信息
            log::debug!(
                "扩散需要 {} 个子步以保证稳定性 (ν={:.2e}, dt={:.2e})",
                n_substeps,
                self.config.nu,
                dt
            );
        }

        self.apply_substeps(field, dt, n_substeps)?;
        Ok(n_substeps)
    }

    /// 验证参数
    fn validate_params(&self, dt: f64) -> Result<(), DiffusionError> {
        if self.config.nu < 0.0 {
            return Err(DiffusionError::InvalidParameter {
                name: "nu",
                value: self.config.nu,
                reason: "扩散系数不能为负".to_string(),
            });
        }

        if dt <= 0.0 {
            return Err(DiffusionError::InvalidParameter {
                name: "dt",
                value: dt,
                reason: "时间步长必须为正".to_string(),
            });
        }

        Ok(())
    }
}

/// 可变扩散系数求解器
pub struct VariableDiffusionSolver<'a> {
    mesh: &'a PhysicsMesh,
}

impl<'a> VariableDiffusionSolver<'a> {
    /// 创建求解器
    pub fn new(mesh: &'a PhysicsMesh) -> Self {
        Self { mesh }
    }

    /// 显式扩散求解（空间变化扩散系数）
    ///
    /// # 参数
    /// - `field`: 输入场值
    /// - `field_out`: 输出场值
    /// - `nu`: 扩散系数场
    /// - `dt`: 时间步长
    pub fn apply_explicit(
        &self,
        field: &[f64],
        field_out: &mut [f64],
        nu: &[f64],
        dt: f64,
    ) -> Result<(), DiffusionError> {
        let n_cells = self.mesh.n_cells();
        let n_faces = self.mesh.n_faces();

        if field.len() != n_cells || field_out.len() != n_cells || nu.len() != n_cells {
            return Err(DiffusionError::SizeMismatch {
                expected: n_cells,
                field_in: field.len(),
                field_out: field_out.len(),
            });
        }

        let mut flux_sum = vec![0.0; n_cells];

        for face in 0..n_faces {
            let owner = self.mesh.face_owner(face);
            let neighbor_opt = self.mesh.face_neighbor(face);

            if let Some(neighbor) = neighbor_opt {
                let dist = self.mesh.face_distance(face).unwrap_or(1e-14);
                if dist < 1e-14 {
                    continue;
                }

                let length = self.mesh.face_length(face);

                // 调和平均扩散系数 (保证正定性)
                let nu_o = nu[owner];
                let nu_n = nu[neighbor];
                let nu_face = if nu_o + nu_n > 1e-14 {
                    2.0 * nu_o * nu_n / (nu_o + nu_n)
                } else {
                    0.0
                };

                let phi_o = field[owner];
                let phi_n = field[neighbor];
                let flux = -nu_face * (phi_n - phi_o) / dist * length;

                flux_sum[owner] += flux;
                flux_sum[neighbor] -= flux;
            }
        }

        field_out
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, phi_out)| {
                let area = self.mesh.cell_area(i).unwrap_or(1.0);
                if area > 1e-14 {
                    *phi_out = field[i] + dt * flux_sum[i] / area;
                } else {
                    *phi_out = field[i];
                }
            });

        Ok(())
    }
}

/// 扩散求解错误
#[derive(Debug, Clone)]
pub enum DiffusionError {
    /// 数组尺寸不匹配
    SizeMismatch {
        expected: usize,
        field_in: usize,
        field_out: usize,
    },
    /// 无效参数
    InvalidParameter {
        name: &'static str,
        value: f64,
        reason: String,
    },
}

impl std::fmt::Display for DiffusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SizeMismatch {
                expected,
                field_in,
                field_out,
            } => {
                write!(
                    f,
                    "场数组尺寸不匹配: 期望 {}, 实际 in={}, out={}",
                    expected, field_in, field_out
                )
            }
            Self::InvalidParameter { name, value, reason } => {
                write!(f, "无效参数 {}: {} ({})", name, value, reason)
            }
        }
    }
}

impl std::error::Error for DiffusionError {}

// ============================================================================
// 便捷函数（兼容旧接口）
// ============================================================================

/// 估计稳定时间步长
pub fn estimate_stable_dt(mesh: &PhysicsMesh, nu: f64) -> f64 {
    let config = DiffusionConfig::new(nu);
    let solver = DiffusionSolver::new(mesh, config);
    solver.estimate_stable_dt()
}

/// 计算所需子步数
pub fn required_substeps(mesh: &PhysicsMesh, nu: f64, dt: f64) -> usize {
    let config = DiffusionConfig::new(nu);
    let solver = DiffusionSolver::new(mesh, config);
    solver.required_substeps(dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::PhysicsMesh;

    fn create_test_mesh(n_cells: usize) -> PhysicsMesh {
        // 创建简单测试网格
        PhysicsMesh::empty(n_cells)
    }

    #[test]
    fn test_diffusion_bc_default() {
        let bc = DiffusionBC::default();
        assert_eq!(bc, DiffusionBC::ZeroFlux);
    }

    #[test]
    fn test_diffusion_bc_constructors() {
        let dirichlet = DiffusionBC::dirichlet(1.0);
        assert!(matches!(dirichlet, DiffusionBC::FixedValue(1.0)));

        let radiation = DiffusionBC::radiation(0.5, 2.0);
        assert!(matches!(
            radiation,
            DiffusionBC::Radiation {
                alpha: 0.5,
                phi_inf: 2.0
            }
        ));

        let flux = DiffusionBC::specified_flux(0.1);
        assert!(matches!(flux, DiffusionBC::SpecifiedFlux(0.1)));
    }

    #[test]
    fn test_diffusion_bc_zero_flux() {
        let bc = DiffusionBC::ZeroFlux;
        let flux = bc.compute_flux(1.0, 1.0, 1.0, 1.0);
        assert_eq!(flux, 0.0);
    }

    #[test]
    fn test_diffusion_bc_fixed_value() {
        let bc = DiffusionBC::FixedValue(0.0);
        // phi_cell = 1.0, phi_bc = 0.0, nu = 1.0, d = 1.0, L = 1.0
        // flux = -1.0 * (1.0 - 0.0) / 1.0 * 1.0 = -1.0
        let flux = bc.compute_flux(1.0, 1.0, 1.0, 1.0);
        assert!((flux - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_diffusion_bc_radiation() {
        let bc = DiffusionBC::Radiation {
            alpha: 0.5,
            phi_inf: 2.0,
        };
        // phi_cell = 1.0, phi_inf = 2.0, alpha = 0.5, L = 1.0
        // flux = 0.5 * (2.0 - 1.0) * 1.0 = 0.5
        let flux = bc.compute_flux(1.0, 1.0, 1.0, 1.0);
        assert!((flux - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_diffusion_config_default() {
        let config = DiffusionConfig::default();
        assert_eq!(config.nu, 1.0);
        assert!(config.boundary_conditions.is_empty());
        assert!((config.cfl_safety - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_diffusion_config_builder() {
        let config = DiffusionConfig::new(0.5)
            .with_cfl_safety(0.3)
            .with_boundary_conditions(vec![DiffusionBC::ZeroFlux]);

        assert!((config.nu - 0.5).abs() < 1e-10);
        assert!((config.cfl_safety - 0.3).abs() < 1e-10);
        assert_eq!(config.boundary_conditions.len(), 1);
    }

    #[test]
    fn test_estimate_stable_dt_zero_nu() {
        let mesh = create_test_mesh(10);
        let dt = estimate_stable_dt(&mesh, 0.0);
        assert_eq!(dt, f64::MAX);
    }

    #[test]
    fn test_required_substeps_small_dt() {
        let mesh = create_test_mesh(10);
        // 小时间步应该不需要子步
        let n = required_substeps(&mesh, 1.0, 0.001);
        assert_eq!(n, 1);
    }

    #[test]
    fn test_diffusion_solver_creation() {
        let mesh = create_test_mesh(10);
        let config = DiffusionConfig::new(1.0);
        let _solver = DiffusionSolver::new(&mesh, config);
    }

    #[test]
    fn test_diffusion_error_display() {
        let err = DiffusionError::SizeMismatch {
            expected: 10,
            field_in: 5,
            field_out: 10,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("场数组尺寸不匹配"));
    }

    #[test]
    fn test_harmonic_mean() {
        // 验证调和平均公式
        let nu_o: f64 = 1.0;
        let nu_n: f64 = 2.0;
        let nu_face = 2.0 * nu_o * nu_n / (nu_o + nu_n);
        let expected: f64 = 2.0 * 1.0 * 2.0 / 3.0; // 4/3
        assert!((nu_face - expected).abs() < 1e-10);
    }
}
