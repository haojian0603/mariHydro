/*
    #[test]
    fn test_constant_coefficient() {
        let coef: DiffusionCoefficient<f64> = DiffusionCoefficient::Constant(10.0);
        assert!((coef.effective_at(0, None) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_variable_coefficient() {
        let coef: DiffusionCoefficient<f64> = DiffusionCoefficient::Variable(vec![1.0, 2.0, 3.0]);
        assert!((coef.effective_at(0, None) - 1.0).abs() < 1e-10);
        assert!((coef.effective_at(1, None) - 2.0).abs() < 1e-10);
        assert!((coef.effective_at(2, None) - 3.0).abs() < 1e-10);
        assert!((coef.effective_at(10, None)).abs() < 1e-10); // 越界返回 0
    }

    #[test]
    fn test_anisotropic_coefficient() {
        let coef: DiffusionCoefficient<f64> = DiffusionCoefficient::Anisotropic {
            longitudinal: 100.0,
            transverse: 10.0,
        };
        let effective = coef.effective_at(0, None);
        // 几何平均 = sqrt(100 * 10) ≈ 31.62
        assert!((effective - 31.622776601683793).abs() < 1e-10);
    }

    #[test]
    fn test_turbulent_coefficient() {
        let coef: DiffusionCoefficient<f64> = DiffusionCoefficient::Turbulent {
            molecular: 1.0,
            schmidt_number: 0.7,
        };

        // 无涡粘度时只有分子扩散
        assert!((coef.effective_at(0, None) - 1.0).abs() < 1e-10);

        // 有涡粘度时 = molecular + nu_t / Sc
        let nu_t = 7.0;
        let expected = 1.0 + 7.0 / 0.7; // = 11.0
        assert!((coef.effective_at(0, Some(nu_t)) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_harmonic_mean() {
        assert!((harmonic_mean(2.0_f64, 2.0_f64) - 2.0_f64).abs() < 1e-10);
        assert!((harmonic_mean(1.0_f64, 3.0_f64) - 1.5_f64).abs() < 1e-10);
        assert!(harmonic_mean(0.0_f64, 1.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_config_disabled() {
        let config = DiffusionConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_config_constant_f64_to_f32() {
        let config = DiffusionConfig::constant(10.0);
        let operator = DiffusionOperator::<f32>::new(100, 200, config);
        assert_eq!(operator.face_diffusivity().len(), 200);
    }

    #[test]
    fn test_config_f64_to_f32_conversion() {
        let config = DiffusionConfig {
            coefficient: DiffusionCoefficientConfig::turbulent(1.5, 0.8),
            enabled: true,
            min_diffusivity: 0.1,
            max_diffusivity: 100.0,
        };

        let operator_f64 = DiffusionOperator::<f64>::new(50, 100, config.clone());
        let operator_f32 = DiffusionOperator::<f32>::new(50, 100, config);

        // f64 版本
        assert_eq!(operator_f64.face_diffusivity().len(), 100);

        // f32 版本
        assert_eq!(operator_f32.face_diffusivity().len(), 100);
    }
}
*/
//! marihydro\crates\mh_physics\src\tracer\diffusion.rs
//! 扩散算子模块
//!
//! 提供示踪剂输运的扩散计算，支持：
//! - 各向同性扩散
//! - 各向异性扩散（纵向/横向扩散系数不同）
//! - 湍流扩散（基于涡粘度和 Schmidt 数）
//! - 空间变化的扩散系数
//!
//! # 基本方程
//!
//! 各向同性扩散通量：
//! $$\vec{J} = -D \nabla c$$
//!
//! 各向异性扩散通量：
//! $$\vec{J} = -\mathbf{D} \nabla c$$
//!
//! 其中 $\mathbf{D}$ 是扩散系数张量。
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::tracer::diffusion::{DiffusionOperator, DiffusionCoefficient};
//!
//! // 创建各向同性扩散算子
//! let backend = mh_runtime::CpuBackend::<f64>::new();
//! let op = DiffusionOperator::new(backend, n_cells, n_faces, DiffusionConfig::constant(10.0));
//!
//! // 计算扩散通量
//! let fluxes = op.compute_face_fluxes(&mesh, &concentration).unwrap();
//! ```

use crate::adapter::PhysicsMesh;
use crate::prelude::*;
use serde::{Deserialize, Serialize};

/// 扩散系数类型（配置层，硬编码 f64）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffusionCoefficientConfig {
    /// 常数扩散系数 [m²/s]
    Constant(f64),

    /// 空间变化的扩散系数（每单元一个值）
    Variable(Vec<f64>),

    /// 各向异性扩散（纵向、横向系数）
    Anisotropic {
        /// 纵向扩散系数 [m²/s]
        longitudinal: f64,
        /// 横向扩散系数 [m²/s]
        transverse: f64,
    },

    /// 基于湍流的扩散（涡粘度 / Schmidt 数）
    Turbulent {
        /// 分子扩散系数 [m²/s]
        molecular: f64,
        /// 湍流 Schmidt 数（无量纲）
        schmidt_number: f64,
    },
}

impl Default for DiffusionCoefficientConfig {
    fn default() -> Self {
        Self::Constant(1.0)
    }
}

impl DiffusionCoefficientConfig {
    /// 创建常数扩散系数
    pub fn constant(d: f64) -> Self {
        Self::Constant(d)
    }

    /// 创建零扩散
    pub fn zero() -> Self {
        Self::Constant(0.0)
    }

    /// 创建各向异性扩散
    pub fn anisotropic(longitudinal: f64, transverse: f64) -> Self {
        Self::Anisotropic {
            longitudinal,
            transverse,
        }
    }

    /// 创建湍流扩散
    pub fn turbulent(molecular: f64, schmidt_number: f64) -> Self {
        Self::Turbulent {
            molecular,
            schmidt_number,
        }
    }

    /// 转换为运行时精度（供算子使用）
    pub fn to_precision<B: Backend>(&self, backend: &B) -> DiffusionCoefficient<B> {
        match *self {
            Self::Constant(d) => DiffusionCoefficient::Constant(
                backend.config_scalar(d, "DiffusionCoefficientConfig.constant"),
            ),
            Self::Variable(ref values) => {
                let mut buf = backend.alloc(values.len());
                let converted: Vec<B::Scalar> = values
                    .iter()
                    .map(|&v| backend.config_scalar(v, "DiffusionCoefficientConfig.variable"))
                    .collect();
                buf.copy_from_slice(&converted);
                DiffusionCoefficient::Variable(buf)
            }
            Self::Anisotropic {
                longitudinal,
                transverse,
            } => DiffusionCoefficient::Anisotropic {
                longitudinal: backend.config_scalar(
                    longitudinal,
                    "DiffusionCoefficientConfig.anisotropic.longitudinal",
                ),
                transverse: backend.config_scalar(
                    transverse,
                    "DiffusionCoefficientConfig.anisotropic.transverse",
                ),
            },
            Self::Turbulent {
                molecular,
                schmidt_number,
            } => DiffusionCoefficient::Turbulent {
                molecular: backend
                    .config_scalar(molecular, "DiffusionCoefficientConfig.turbulent.molecular"),
                schmidt_number: backend.config_scalar(
                    schmidt_number,
                    "DiffusionCoefficientConfig.turbulent.schmidt_number",
                ),
            },
        }
    }
}

/// 扩散系数类型（运行层，泛型化）
#[derive(Debug, Clone)]
pub enum DiffusionCoefficient<B: Backend> {
    /// 常数扩散系数 [m²/s]
    Constant(B::Scalar),

    /// 空间变化的扩散系数（每单元一个值）
    Variable(B::Buffer<B::Scalar>),

    /// 各向异性扩散（纵向、横向系数）
    Anisotropic {
        /// 纵向扩散系数 [m²/s]
        longitudinal: B::Scalar,
        /// 横向扩散系数 [m²/s]
        transverse: B::Scalar,
    },

    /// 基于湍流的扩散（涡粘度 / Schmidt 数）
    Turbulent {
        /// 分子扩散系数 [m²/s]
        molecular: B::Scalar,
        /// 湍流 Schmidt 数（无量纲）
        schmidt_number: B::Scalar,
    },
}

impl<B: Backend> DiffusionCoefficient<B> {
    /// 获取单元的有效扩散系数（各向同性等效）
    pub fn effective_at(&self, cell_idx: usize, eddy_viscosity: Option<B::Scalar>) -> B::Scalar {
        match *self {
            Self::Constant(d) => d,
            Self::Variable(ref values) => values.get(cell_idx).copied().unwrap_or(B::Scalar::ZERO),
            Self::Anisotropic {
                longitudinal,
                transverse,
            } => {
                // 几何平均作为各向同性等效
                (longitudinal * transverse).sqrt()
            }
            Self::Turbulent {
                molecular,
                schmidt_number,
            } => {
                let nu_t = eddy_viscosity.unwrap_or(B::Scalar::ZERO);
                molecular + nu_t / schmidt_number
            }
        }
    }
}

/// 扩散算子错误
#[derive(Debug, Clone)]
pub enum DiffusionError {
    /// 后端缓冲区不可访问
    BackendAccess(&'static str),
    /// 长度不匹配
    SizeMismatch { expected: usize, actual: usize },
}

/// 扩散配置（硬编码 f64，精度无关）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionConfig {
    /// 扩散系数（配置层，f64）
    pub coefficient: DiffusionCoefficientConfig,
    /// 是否启用扩散
    pub enabled: bool,
    /// 最小扩散系数（数值稳定性，f64）
    pub min_diffusivity: f64,
    /// 最大扩散系数（限制，f64）
    pub max_diffusivity: f64,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            coefficient: DiffusionCoefficientConfig::default(),
            enabled: true,
            min_diffusivity: 0.0,
            max_diffusivity: 1000.0,
        }
    }
}

impl DiffusionConfig {
    /// 创建禁用的扩散配置
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// 创建常数扩散配置
    pub fn constant(d: f64) -> Self {
        Self {
            coefficient: DiffusionCoefficientConfig::constant(d),
            ..Default::default()
        }
    }

    /// 创建各向异性扩散配置
    pub fn anisotropic(longitudinal: f64, transverse: f64) -> Self {
        Self {
            coefficient: DiffusionCoefficientConfig::anisotropic(longitudinal, transverse),
            ..Default::default()
        }
    }

    /// 创建湛流扩散配置
    pub fn turbulent(molecular: f64, schmidt_number: f64) -> Self {
        Self {
            coefficient: DiffusionCoefficientConfig::turbulent(molecular, schmidt_number),
            ..Default::default()
        }
    }
}

/// 扩散算子（泛型化，支持多精度）
///
/// 计算扩散通量和扩散项对浓度场的贡献
pub struct DiffusionOperator<B: Backend> {
    /// 配置（硬编码 f64，存储时）
    config: DiffusionConfig,
    /// 配置（运行时精度，计算时）
    coefficient: DiffusionCoefficient<B>,
    min_diffusivity: B::Scalar,
    max_diffusivity: B::Scalar,
    /// 面扩散系数缓存
    face_diffusivity: B::Buffer<B::Scalar>,
    /// 扩散通量缓存
    face_flux: B::Buffer<B::Scalar>,
    /// 扩散源项（单元体积分）
    cell_diffusion: B::Buffer<B::Scalar>,
    /// 后端
    backend: B,
}

impl<B: Backend> DiffusionOperator<B> {
    /// 创建新的扩散算子
    ///
    /// # 参数
    ///
    /// - `n_cells`: 单元数量
    /// - `n_faces`: 面数量
    /// - `config`: 扩散配置（f64）
    pub fn new(backend: B, n_cells: usize, n_faces: usize, config: DiffusionConfig) -> Self {
        // 转换配置到运行时精度
        let coefficient = config.coefficient.to_precision(&backend);
        let min_diffusivity = backend.config_scalar(
            config.min_diffusivity,
            "DiffusionOperatorConfig.min_diffusivity",
        );
        let max_diffusivity = backend.config_scalar(
            config.max_diffusivity,
            "DiffusionOperatorConfig.max_diffusivity",
        );

        Self {
            config,
            coefficient,
            min_diffusivity,
            max_diffusivity,
            face_diffusivity: backend.alloc(n_faces),
            face_flux: backend.alloc(n_faces),
            cell_diffusion: backend.alloc(n_cells),
            backend,
        }
    }

    /// 获取配置引用
    pub fn config(&self) -> &DiffusionConfig {
        &self.config
    }

    /// 获取扩散系数配置引用（运行时精度）
    pub fn coefficient(&self) -> &DiffusionCoefficient<B> {
        &self.coefficient
    }

    /// 更新面扩散系数
    ///
    /// # 参数
    ///
    /// - `mesh`: 物理网格
    /// - `eddy_viscosity`: 涡粘度场（可选，用于湍流扩散）
    pub fn update_face_diffusivity(
        &mut self,
        mesh: &PhysicsMesh,
        eddy_viscosity: Option<&B::Buffer<B::Scalar>>,
    ) -> Result<(), DiffusionError> {
        if !self.config.enabled {
            self.face_diffusivity.fill(B::Scalar::ZERO);
            return Ok(());
        }

        let eddy_slice = match eddy_viscosity {
            Some(buf) => Some(
                buf.try_as_slice()
                    .ok_or(DiffusionError::BackendAccess("eddy_viscosity"))?,
            ),
            None => None,
        };

        for face_idx in 0..mesh.face_count() {
            let face = FaceIndex::new(face_idx);
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);

            let nu_t_o = eddy_slice.map(|nu| nu[owner.get()]);
            let d_owner = self.coefficient.effective_at(owner.get(), nu_t_o);

            let d_face = if let Some(neigh) = neighbor {
                // 内部面：调和平均
                let nu_t_n = eddy_slice.map(|nu| nu[neigh.get()]);
                let d_neigh = self.coefficient.effective_at(neigh.get(), nu_t_n);
                harmonic_mean(d_owner, d_neigh)
            } else {
                // 边界面：使用内部值
                d_owner
            };

            // 应用限制
            self.face_diffusivity[face_idx] =
                d_face.max(self.min_diffusivity).min(self.max_diffusivity);
        }
        Ok(())
    }

    /// 计算扩散通量
    ///
    /// # 参数
    ///
    /// - `mesh`: 物理网格
    /// - `concentration`: 浓度场
    ///
    /// # 返回
    ///
    /// 面扩散通量 [单位/s]
    pub fn compute_face_fluxes(
        &mut self,
        mesh: &PhysicsMesh,
        concentration: &B::Buffer<B::Scalar>,
    ) -> Result<&B::Buffer<B::Scalar>, DiffusionError> {
        if !self.config.enabled {
            self.face_flux.fill(B::Scalar::ZERO);
            return Ok(&self.face_flux);
        }

        let conc = concentration
            .try_as_slice()
            .ok_or(DiffusionError::BackendAccess("concentration"))?;

        for face_idx in 0..mesh.face_count() {
            let face = FaceIndex::new(face_idx);
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);

            let d = self.face_diffusivity[face_idx];
            let length = self
                .backend
                .config_scalar(mesh.face_length(face), "DiffusionOperator.face_length");

            let flux = if let Some(neigh) = neighbor {
                // 内部面：中心差分
                let dist = self
                    .backend
                    .config_scalar(mesh.face_dist_o2n(face), "DiffusionOperator.face_distance");
                if dist
                    > self
                        .backend
                        .config_scalar(1e-14, "DiffusionOperator.distance_eps")
                {
                    let grad_n = (conc[neigh.get()] - conc[owner.get()]) / dist;
                    -d * grad_n * length
                } else {
                    B::Scalar::ZERO
                }
            } else {
                // 边界面：假设零梯度（由边界条件处理）
                B::Scalar::ZERO
            };

            self.face_flux[face_idx] = flux;
        }

        Ok(&self.face_flux)
    }

    /// 计算扩散对单元的贡献
    ///
    /// # 参数
    ///
    /// - `mesh`: 物理网格
    /// - `concentration`: 浓度场
    ///
    /// # 返回
    ///
    /// 单元扩散率 dc/dt [单位/s]
    pub fn compute_cell_diffusion(
        &mut self,
        mesh: &PhysicsMesh,
        concentration: &B::Buffer<B::Scalar>,
    ) -> Result<&B::Buffer<B::Scalar>, DiffusionError> {
        // 先计算面通量
        self.compute_face_fluxes(mesh, concentration)?;

        // 清零
        self.cell_diffusion.fill(B::Scalar::ZERO);

        // 累加面通量到单元
        for face_idx in 0..mesh.face_count() {
            let face = FaceIndex::new(face_idx);
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            let flux = self.face_flux[face_idx];

            let area_o = self.backend.config_scalar(
                mesh.cell_area_unchecked(owner),
                "DiffusionOperator.owner_area",
            );
            self.cell_diffusion[owner.get()] -= flux / area_o;

            if let Some(neigh) = neighbor {
                let area_n = self.backend.config_scalar(
                    mesh.cell_area_unchecked(neigh),
                    "DiffusionOperator.neighbor_area",
                );
                self.cell_diffusion[neigh.get()] += flux / area_n;
            }
        }

        Ok(&self.cell_diffusion)
    }

    /// 获取面扩散系数
    pub fn face_diffusivity(&self) -> &B::Buffer<B::Scalar> {
        &self.face_diffusivity
    }

    /// 获取面扩散通量
    pub fn face_flux(&self) -> &B::Buffer<B::Scalar> {
        &self.face_flux
    }

    /// 获取单元扩散率
    pub fn cell_diffusion(&self) -> &B::Buffer<B::Scalar> {
        &self.cell_diffusion
    }
}

/// 各向异性扩散算子（泛型化）
///
/// 考虑流向（纵向）和垂直流向（横向）的不同扩散系数
pub struct AnisotropicDiffusionOperator<B: Backend> {
    /// 纵向扩散系数 [m²/s]
    longitudinal: B::Scalar,
    /// 横向扩散系数 [m²/s]
    transverse: B::Scalar,
    /// 面扩散通量缓存
    face_flux: B::Buffer<B::Scalar>,
    /// 后端
    backend: B,
}

impl<B: Backend> AnisotropicDiffusionOperator<B> {
    /// 创建新的各向异性扩散算子
    pub fn new(backend: B, n_faces: usize, longitudinal: B::Scalar, transverse: B::Scalar) -> Self {
        Self {
            longitudinal,
            transverse,
            face_flux: backend.alloc(n_faces),
            backend,
        }
    }

    /// 计算各向异性扩散通量
    ///
    /// # 参数
    ///
    /// - `mesh`: 物理网格
    /// - `concentration`: 浓度场
    /// - `velocity_x`: x 方向速度
    /// - `velocity_y`: y 方向速度
    pub fn compute_face_fluxes(
        &mut self,
        mesh: &PhysicsMesh,
        concentration: &B::Buffer<B::Scalar>,
        velocity_x: &B::Buffer<B::Scalar>,
        velocity_y: &B::Buffer<B::Scalar>,
    ) -> Result<&B::Buffer<B::Scalar>, DiffusionError> {
        let conc = concentration
            .try_as_slice()
            .ok_or(DiffusionError::BackendAccess("concentration"))?;
        let u = velocity_x
            .try_as_slice()
            .ok_or(DiffusionError::BackendAccess("velocity_x"))?;
        let v = velocity_y
            .try_as_slice()
            .ok_or(DiffusionError::BackendAccess("velocity_y"))?;
        for face_idx in 0..mesh.face_count() {
            let face = FaceIndex::new(face_idx);
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);

            let flux = if let Some(neigh) = neighbor {
                let normal = mesh
                    .face_normal_generic::<B>(face)
                    .expect("face_normal out of range");
                let normal_x = normal.x();
                let normal_y = normal.y();
                let length = self.backend.config_scalar(
                    mesh.face_length(face),
                    "AnisotropicDiffusionOperator.face_length",
                );
                let dist = self.backend.config_scalar(
                    mesh.face_dist_o2n(face),
                    "AnisotropicDiffusionOperator.face_distance",
                );

                if dist
                    < self
                        .backend
                        .config_scalar(1e-14, "AnisotropicDiffusionOperator.distance_eps")
                {
                    B::Scalar::ZERO
                } else {
                    // 计算流向单位向量
                    let u_o = u[owner.get()];
                    let v_o = v[owner.get()];
                    let u_n = u[neigh.get()];
                    let v_n = v[neigh.get()];

                    let half = B::Scalar::HALF;
                    let u_avg = half * (u_o + u_n);
                    let v_avg = half * (v_o + v_n);
                    let speed = (u_avg * u_avg + v_avg * v_avg).sqrt();

                    // 有效扩散系数（投影到面法向）
                    let d_eff = if speed
                        > self
                            .backend
                            .config_scalar(1e-8, "AnisotropicDiffusionOperator.speed_eps")
                    {
                        let e_x = u_avg / speed;
                        let e_y = v_avg / speed;

                        // 法向方向的流向分量
                        let cos_theta = e_x * normal_x + e_y * normal_y;
                        let sin_theta = (B::Scalar::ONE - cos_theta * cos_theta).sqrt();

                        self.longitudinal * cos_theta.abs() + self.transverse * sin_theta
                    } else {
                        // 静水时使用几何平均
                        (self.longitudinal * self.transverse).sqrt()
                    };

                    let grad_n = (conc[neigh.get()] - conc[owner.get()]) / dist;
                    -d_eff * grad_n * length
                }
            } else {
                B::Scalar::ZERO
            };

            self.face_flux[face_idx] = flux;
        }

        Ok(&self.face_flux)
    }

    /// 获取面扩散通量
    pub fn face_flux(&self) -> &B::Buffer<B::Scalar> {
        &self.face_flux
    }
}

/// 计算调和平均
#[inline]
fn harmonic_mean<S: RuntimeScalar>(a: S, b: S) -> S {
    let eps = S::MIN_POSITIVE;
    let two = S::TWO;
    let denom = a + b;
    let mask = (a.abs() > eps) && (b.abs() > eps) && (denom.abs() > eps);
    if mask {
        two * a * b / denom
    } else {
        S::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_constant_coefficient() {
        let coef: DiffusionCoefficient<CpuBackend<f64>> = DiffusionCoefficient::Constant(10.0);
        assert!((coef.effective_at(0, None) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_variable_coefficient() {
        let backend = CpuBackend::<f64>::new();
        let mut buf = backend.alloc(3);
        buf.copy_from_slice(&[1.0, 2.0, 3.0]);
        let coef: DiffusionCoefficient<CpuBackend<f64>> = DiffusionCoefficient::Variable(buf);
        assert!((coef.effective_at(0, None) - 1.0).abs() < 1e-10);
        assert!((coef.effective_at(1, None) - 2.0).abs() < 1e-10);
        assert!((coef.effective_at(2, None) - 3.0).abs() < 1e-10);
        assert!((coef.effective_at(10, None)).abs() < 1e-10); // 越界返回 0
    }

    #[test]
    fn test_anisotropic_coefficient() {
        let coef: DiffusionCoefficient<CpuBackend<f64>> = DiffusionCoefficient::Anisotropic {
            longitudinal: 100.0,
            transverse: 10.0,
        };
        let effective = coef.effective_at(0, None);
        // 几何平均 = sqrt(100 * 10) ≈ 31.62
        assert!((effective - 31.622776601683793).abs() < 1e-10);
    }

    #[test]
    fn test_turbulent_coefficient() {
        let coef: DiffusionCoefficient<CpuBackend<f64>> = DiffusionCoefficient::Turbulent {
            molecular: 1.0,
            schmidt_number: 0.7,
        };

        // 无涡粘度时只有分子扩散
        assert!((coef.effective_at(0, None) - 1.0).abs() < 1e-10);

        // 有涡粘度时 = molecular + nu_t / Sc
        let nu_t = 7.0;
        let expected = 1.0 + 7.0 / 0.7; // = 11.0
        assert!((coef.effective_at(0, Some(nu_t)) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_harmonic_mean() {
        assert!((harmonic_mean(2.0_f64, 2.0_f64) - 2.0_f64).abs() < 1e-10);
        assert!((harmonic_mean(1.0_f64, 3.0_f64) - 1.5_f64).abs() < 1e-10);
        assert!(harmonic_mean(0.0_f64, 1.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_config_disabled() {
        let config = DiffusionConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_config_constant_f64_to_f32() {
        let config = DiffusionConfig::constant(10.0);
        let backend = CpuBackend::<f32>::new();
        let operator = DiffusionOperator::new(backend, 100, 200, config);
        assert_eq!(operator.face_diffusivity().len(), 200);
    }

    #[test]
    fn test_config_f64_to_f32_conversion() {
        let config = DiffusionConfig {
            coefficient: DiffusionCoefficientConfig::turbulent(1.5, 0.8),
            enabled: true,
            min_diffusivity: 0.1,
            max_diffusivity: 100.0,
        };

        let backend_f64 = CpuBackend::<f64>::new();
        let backend_f32 = CpuBackend::<f32>::new();
        let operator_f64 = DiffusionOperator::new(backend_f64, 50, 100, config.clone());
        let operator_f32 = DiffusionOperator::new(backend_f32, 50, 100, config);

        // f64 版本
        assert_eq!(operator_f64.face_diffusivity().len(), 100);

        // f32 版本
        assert_eq!(operator_f32.face_diffusivity().len(), 100);
    }
}
