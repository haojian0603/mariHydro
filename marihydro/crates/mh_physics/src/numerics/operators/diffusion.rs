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
use crate::prelude::*;
use crate::adapter::PhysicsMesh;

/// 扩散边界条件类型
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum DiffusionBC<S: RuntimeScalar> {
    /// 零通量 (Neumann): ∂φ/∂n = 0
    #[default]
    ZeroFlux,
    /// 固定值 (Dirichlet): φ = value
    FixedValue(S),
    /// 辐射边界: flux = α*(φ - φ_∞)
    Radiation {
        /// 传递系数 [1/s 或 m/s 取决于场类型]
        alpha: S,
        /// 远场值
        phi_inf: S,
    },
    /// 指定通量: flux = value
    SpecifiedFlux(S),
}

impl<S: RuntimeScalar> DiffusionBC<S> {
    /// 创建 Dirichlet 边界条件
    pub fn dirichlet(value: S) -> Self {
        Self::FixedValue(value)
    }

    /// 创建辐射边界条件
    pub fn radiation(alpha: S, phi_inf: S) -> Self {
        Self::Radiation { alpha, phi_inf }
    }

    /// 创建指定通量边界条件
    pub fn specified_flux(flux: S) -> Self {
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
    pub fn compute_flux(&self, phi_cell: S, nu: S, d: S, length: S) -> S {
        match *self {
            Self::ZeroFlux => S::ZERO,
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

impl DiffusionBC<f64> {
    /// 将 f64 边界条件转换为后端标量类型
    pub fn to_backend<B: Backend>(&self, backend: &B) -> DiffusionBC<B::Scalar> {
        match *self {
            DiffusionBC::ZeroFlux => DiffusionBC::ZeroFlux,
            DiffusionBC::FixedValue(v) => DiffusionBC::FixedValue(backend.scalar_from_f64(v)),
            DiffusionBC::Radiation { alpha, phi_inf } => DiffusionBC::Radiation {
                alpha: backend.scalar_from_f64(alpha),
                phi_inf: backend.scalar_from_f64(phi_inf),
            },
            DiffusionBC::SpecifiedFlux(f) => DiffusionBC::SpecifiedFlux(backend.scalar_from_f64(f)),
        }
    }
}

/// 扩散求解器配置
#[derive(Debug, Clone)]
pub struct DiffusionConfig<S: RuntimeScalar> {
    /// 扩散系数 [m²/s]
    pub nu: S,
    /// 边界条件（按边界索引）
    pub boundary_conditions: Vec<DiffusionBC<S>>,
    /// CFL 安全系数
    pub cfl_safety: S,
}

impl<S: RuntimeScalar> Default for DiffusionConfig<S> {
    fn default() -> Self {
        Self {
            nu: S::ONE,
            boundary_conditions: Vec::new(),
            cfl_safety: S::HALF * S::HALF,
        }
    }
}

impl<S: RuntimeScalar> DiffusionConfig<S> {
    /// 创建新配置
    pub fn new(nu: S) -> Self {
        Self {
            nu,
            ..Default::default()
        }
    }

    /// 设置边界条件
    pub fn with_boundary_conditions(mut self, bcs: Vec<DiffusionBC<S>>) -> Self {
        self.boundary_conditions = bcs;
        self
    }

    /// 设置 CFL 安全系数
    pub fn with_cfl_safety(mut self, safety: S) -> Self {
        let ten = S::ONE
            + S::ONE
            + S::ONE
            + S::ONE
            + S::ONE
            + S::ONE
            + S::ONE
            + S::ONE
            + S::ONE
            + S::ONE;
        let min = S::ONE / ten;
        let max = S::HALF;
        self.cfl_safety = if safety < min { min } else if safety > max { max } else { safety };
        self
    }
}

impl DiffusionConfig<f64> {
    /// 将 f64 配置转换为后端标量配置
    pub fn to_backend<B: Backend>(&self, backend: &B) -> DiffusionConfig<B::Scalar> {
        DiffusionConfig {
            nu: backend.scalar_from_f64(self.nu),
            boundary_conditions: self
                .boundary_conditions
                .iter()
                .map(|bc| bc.to_backend(backend))
                .collect(),
            cfl_safety: backend.scalar_from_f64(self.cfl_safety),
        }
    }
}

/// 扩散求解器（Backend 泛型）
pub struct DiffusionSolver<'a, B: Backend> {
    mesh: &'a PhysicsMesh,
    config: DiffusionConfig<B::Scalar>,
    /// 缓存的面距离最小值平方
    min_dist_sq: B::Scalar,
    backend: B,
}

impl<'a, B: Backend> DiffusionSolver<'a, B> {
    /// 创建求解器（已是泛型配置）
    pub fn new(mesh: &'a PhysicsMesh, backend: B, config: DiffusionConfig<B::Scalar>) -> Self {
        let min_dist_sq = Self::compute_min_dist_sq(mesh, &backend);

        Self {
            mesh,
            config,
            min_dist_sq,
            backend,
        }
    }

    /// 从 f64 配置创建求解器
    pub fn from_f64_config(mesh: &'a PhysicsMesh, backend: B, config: DiffusionConfig<f64>) -> Self {
        let cfg = config.to_backend(&backend);
        Self::new(mesh, backend, cfg)
    }

    /// 计算最小面距离平方
    fn compute_min_dist_sq(mesh: &PhysicsMesh, backend: &B) -> B::Scalar {
        let n_faces = mesh.face_count();
        let mut min_sq = B::Scalar::MAX;
        let eps = backend.scalar_from_f64(1e-14);

        for face in 0..n_faces {
            if let Some(dist) = mesh.face_distance(FaceIndex(face)) {
                let dist_s = backend.scalar_from_f64(dist);
                if dist_s > eps {
                    min_sq = min_sq.min(dist_s * dist_s);
                }
            }
        }

        min_sq
    }

    /// 估计稳定时间步长
    ///
    /// 对于显式扩散，CFL 条件: dt < α * d_min² / ν
    pub fn estimate_stable_dt(&self) -> B::Scalar {
        let eps = self.backend.scalar_from_f64(1e-14);
        if self.config.nu < eps {
            return B::Scalar::MAX;
        }

        if self.min_dist_sq >= B::Scalar::MAX {
            return B::Scalar::ONE;
        }

        self.config.cfl_safety * self.min_dist_sq / self.config.nu
    }

    /// 计算所需子步数以保证稳定性
    pub fn required_substeps(&self, dt: B::Scalar) -> usize {
        let stable_dt = self.estimate_stable_dt();
        if !stable_dt.is_finite() || stable_dt <= B::Scalar::ZERO {
            return 1;
        }
        if stable_dt >= dt {
            return 1;
        }

        let mut n = 1usize;
        let mut acc = stable_dt;
        while acc < dt {
            n += 1;
            acc += stable_dt;
            if !acc.is_finite() {
                break;
            }
        }
        n
    }

    /// 计算扩散通量
    fn compute_fluxes(&self, field: &[B::Scalar]) -> B::Buffer<B::Scalar> {
        let n_cells = self.mesh.cell_count();
        let n_faces = self.mesh.face_count();
        let nu = self.config.nu;

        let mut flux_sum = self.backend.alloc_init(n_cells, B::Scalar::ZERO);
        let eps = self.backend.scalar_from_f64(1e-14);

        // 内部面
        for face in 0..n_faces {
            let owner = self.mesh.face_owner(FaceIndex(face));
            let neighbor_opt = self.mesh.face_neighbor(FaceIndex(face));

            if let Some(neighbor) = neighbor_opt {
                // 内部面
                let dist = self
                    .mesh
                    .face_distance(FaceIndex(face))
                    .map(|d| self.backend.scalar_from_f64(d))
                    .unwrap_or(eps);
                if dist < eps {
                    continue;
                }

                let length = self.backend.scalar_from_f64(self.mesh.face_length(FaceIndex(face)));
                let phi_o = field[owner.get()];
                let phi_n = field[neighbor.get()];

                // F = -ν * (φ_n - φ_o) / d * L
                let flux = -nu * (phi_n - phi_o) / dist * length;

                flux_sum[owner.get()] += flux;
                flux_sum[neighbor.get()] -= flux;
            } else {
                // 边界面
                let bc = self.get_boundary_condition(face);
                let dist = self
                    .mesh
                    .face_distance(FaceIndex(face))
                    .map(|d| self.backend.scalar_from_f64(d))
                    .unwrap_or(eps)
                    .max(eps);
                let length = self.backend.scalar_from_f64(self.mesh.face_length(FaceIndex(face)));
                let phi_cell = field[owner.get()];

                let flux = bc.compute_flux(phi_cell, nu, dist, length);
                flux_sum[owner.get()] += flux;
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
    fn get_boundary_condition(&self, face: usize) -> DiffusionBC<B::Scalar> {
        // 获取面的边界 ID（边界条件索引）
        if let Some(boundary_id) = self.mesh.face_boundary_id(FaceIndex(face)) {
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
        field: &B::Buffer<B::Scalar>,
        field_out: &mut B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) -> Result<(), DiffusionError<B::Scalar>> {
        self.validate_params(dt)?;

        let n_cells = self.mesh.cell_count();
        if field.len() != n_cells || field_out.len() != n_cells {
            return Err(DiffusionError::SizeMismatch {
                expected: n_cells,
                field_in: field.len(),
                field_out: field_out.len(),
            });
        }

        let flux_sum = self.compute_fluxes(field.as_slice());

        let eps = self.backend.scalar_from_f64(1e-14);

        field_out
            .as_slice_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, phi_out)| {
                let area = self
                    .mesh
                    .cell_area(CellIndex(i))
                    .map(|a| self.backend.scalar_from_f64(a))
                    .unwrap_or(B::Scalar::ONE);
                if area > eps {
                    *phi_out = field[i] + dt * flux_sum[i] / area;
                } else {
                    *phi_out = field[i];
                }
            });

        Ok(())
    }

    /// 原地扩散
    pub fn apply_inplace(
        &self,
        field: &mut B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) -> Result<(), DiffusionError<B::Scalar>> {
        let mut temp = self.backend.alloc(field.len());
        self.apply_explicit(field, &mut temp, dt)?;
        field.copy_from_slice(temp.as_slice());
        Ok(())
    }

    /// 多子步扩散
    pub fn apply_substeps(
        &self,
        field: &mut B::Buffer<B::Scalar>,
        dt: B::Scalar,
        n_substeps: usize,
    ) -> Result<(), DiffusionError<B::Scalar>> {
        if n_substeps == 0 {
            return Ok(());
        }

        let sub_dt = dt / self.backend.scalar_from_f64(n_substeps as f64);
        let mut buffer = self.backend.alloc(field.len());

        for step in 0..n_substeps {
            if step % 2 == 0 {
                self.apply_explicit(field, &mut buffer, sub_dt)?;
            } else {
                self.apply_explicit(&buffer, field, sub_dt)?;
            }
        }

        // 如果子步数是奇数，最终结果在 buffer 中
        if n_substeps % 2 == 1 {
            field.copy_from_slice(buffer.as_slice());
        }

        Ok(())
    }

    /// 自动子步扩散
    ///
    /// 自动计算所需子步数以保证稳定性
    pub fn apply_auto_substeps(
        &self,
        field: &mut B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) -> Result<usize, DiffusionError<B::Scalar>> {
        let n_substeps = self.required_substeps(dt);

        if n_substeps > 1 {
            // 在需要时记录调试信息
            log::debug!(
                "扩散需要 {} 个子步以保证稳定性 (ν={}, dt={})",
                n_substeps,
                self.config.nu,
                dt
            );
        }

        self.apply_substeps(field, dt, n_substeps)?;
        Ok(n_substeps)
    }

    /// 验证参数
    fn validate_params(&self, dt: B::Scalar) -> Result<(), DiffusionError<B::Scalar>> {
        if self.config.nu < B::Scalar::ZERO {
            return Err(DiffusionError::InvalidParameter {
                name: "nu",
                value: self.config.nu,
                reason: "扩散系数不能为负".to_string(),
            });
        }

        if dt <= B::Scalar::ZERO {
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
pub struct VariableDiffusionSolver<'a, B: Backend> {
    mesh: &'a PhysicsMesh,
    backend: B,
}

impl<'a, B: Backend> VariableDiffusionSolver<'a, B> {
    /// 创建求解器
    pub fn new(mesh: &'a PhysicsMesh, backend: B) -> Self {
        Self { mesh, backend }
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
        field: &B::Buffer<B::Scalar>,
        field_out: &mut B::Buffer<B::Scalar>,
        nu: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) -> Result<(), DiffusionError<B::Scalar>> {
        let n_cells = self.mesh.cell_count();
        let n_faces = self.mesh.face_count();

        if field.len() != n_cells || field_out.len() != n_cells || nu.len() != n_cells {
            return Err(DiffusionError::SizeMismatch {
                expected: n_cells,
                field_in: field.len(),
                field_out: field_out.len(),
            });
        }

        let mut flux_sum = self.backend.alloc_init(n_cells, B::Scalar::ZERO);
        let eps = self.backend.scalar_from_f64(1e-14);

        for face in 0..n_faces {
            let owner = self.mesh.face_owner(FaceIndex(face));
            let neighbor_opt = self.mesh.face_neighbor(FaceIndex(face));

            if let Some(neighbor) = neighbor_opt {
                let dist = self
                    .mesh
                    .face_distance(FaceIndex(face))
                    .map(|d| self.backend.scalar_from_f64(d))
                    .unwrap_or(eps);
                if dist < eps {
                    continue;
                }

                let length = self.backend.scalar_from_f64(self.mesh.face_length(FaceIndex(face)));

                // 调和平均扩散系数 (保证正定性)
                let nu_o = nu[owner.get()];
                let nu_n = nu[neighbor.get()];
                let nu_face = if nu_o + nu_n > eps {
                    (B::Scalar::TWO * nu_o * nu_n).safe_div_eps(
                        nu_o + nu_n,
                        B::Scalar::MIN_POSITIVE,
                        B::Scalar::ZERO,
                    )
                } else {
                    B::Scalar::ZERO
                };

                let phi_o = field[owner.get()];
                let phi_n = field[neighbor.get()];
                let flux = -nu_face * (phi_n - phi_o) / dist * length;

                flux_sum[owner.get()] += flux;
                flux_sum[neighbor.get()] -= flux;
            }
        }

        field_out
            .as_slice_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, phi_out)| {
                let area = self
                    .mesh
                    .cell_area(CellIndex(i))
                    .map(|a| self.backend.scalar_from_f64(a))
                    .unwrap_or(B::Scalar::ONE);
                if area > eps {
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
pub enum DiffusionError<S: RuntimeScalar> {
    /// 数组尺寸不匹配
    SizeMismatch {
        expected: usize,
        field_in: usize,
        field_out: usize,
    },
    /// 无效参数
    InvalidParameter {
        name: &'static str,
        value: S,
        reason: String,
    },
}

impl<S> std::fmt::Display for DiffusionError<S>
where
    S: RuntimeScalar + std::fmt::Display,
{
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

impl<S> std::error::Error for DiffusionError<S>
where
    S: RuntimeScalar + std::fmt::Debug + std::fmt::Display,
{
}

// ============================================================================
// 便捷函数（兼容旧接口）
// ============================================================================

/// 估计稳定时间步长
pub fn estimate_stable_dt<B: Backend + Clone>(mesh: &PhysicsMesh, backend: &B, nu: f64) -> B::Scalar {
    let config = DiffusionConfig::new(backend.scalar_from_f64(nu));
    let solver = DiffusionSolver::new(mesh, backend.clone(), config);
    solver.estimate_stable_dt()
}

/// 计算所需子步数
pub fn required_substeps<B: Backend + Clone>(mesh: &PhysicsMesh, backend: &B, nu: f64, dt: f64) -> usize {
    let config = DiffusionConfig::new(backend.scalar_from_f64(nu));
    let solver = DiffusionSolver::new(mesh, backend.clone(), config);
    solver.required_substeps(backend.scalar_from_f64(dt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::PhysicsMesh;
    use mh_runtime::CpuBackend;

    fn create_test_mesh(n_cells: usize) -> PhysicsMesh {
        // 创建简单测试网格
        PhysicsMesh::empty(n_cells)
    }

    #[test]
    fn test_diffusion_bc_default() {
        let bc: DiffusionBC<f64> = DiffusionBC::default();
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
        let config: DiffusionConfig<f64> = DiffusionConfig::default();
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
        let backend = CpuBackend::<f64>::default();
        let dt = estimate_stable_dt(&mesh, &backend, 0.0);
        assert_eq!(dt, f64::MAX);
    }

    #[test]
    fn test_required_substeps_small_dt() {
        let mesh = create_test_mesh(10);
        let backend = CpuBackend::<f64>::default();
        // 小时间步应该不需要子步
        let n = required_substeps(&mesh, &backend, 1.0, 0.001);
        assert_eq!(n, 1);
    }

    #[test]
    fn test_diffusion_solver_creation() {
        let mesh = create_test_mesh(10);
        let config = DiffusionConfig::new(1.0);
        let backend = CpuBackend::<f64>::default();
        let _solver = DiffusionSolver::new(&mesh, backend, config);
    }

    #[test]
    fn test_diffusion_error_display() {
        let err = DiffusionError::<f64>::SizeMismatch {
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
