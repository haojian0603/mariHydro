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
//! let op = DiffusionOperator::new(n_cells, n_faces, DiffusionConfig::constant(10.0));
//!
//! // 计算扩散通量
//! let fluxes = op.compute_face_fluxes(&mesh, &concentration);
//! ```

use crate::adapter::PhysicsMesh;
use mh_runtime::RuntimeScalar as Scalar;
use bytemuck::Pod;
use mh_foundation::AlignedVec;
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
    pub fn to_precision<S: Scalar>(&self) -> DiffusionCoefficient<S> {
        match *self {
            Self::Constant(d) => DiffusionCoefficient::Constant(S::from_config(d).unwrap_or(S::ZERO)),
            Self::Variable(ref values) => DiffusionCoefficient::Variable(
                values.iter().map(|&v| S::from_config(v).unwrap_or(S::ZERO)).collect()
            ),
            Self::Anisotropic { longitudinal, transverse } => {
                DiffusionCoefficient::Anisotropic {
                    longitudinal: S::from_config(longitudinal).unwrap_or(S::ZERO),
                    transverse: S::from_config(transverse).unwrap_or(S::ZERO),
                }
            }
            Self::Turbulent { molecular, schmidt_number } => {
                DiffusionCoefficient::Turbulent {
                    molecular: S::from_config(molecular).unwrap_or(S::ZERO),
                    schmidt_number: S::from_config(schmidt_number).unwrap_or(S::ZERO),
                }
            }
        }
    }
}

/// 扩散系数类型（运行层，泛型化）
#[derive(Debug, Clone)]
pub enum DiffusionCoefficient<S: Scalar> {
    /// 常数扩散系数 [m²/s]
    Constant(S),

    /// 空间变化的扩散系数（每单元一个值）
    Variable(Vec<S>),

    /// 各向异性扩散（纵向、横向系数）
    Anisotropic {
        /// 纵向扩散系数 [m²/s]
        longitudinal: S,
        /// 横向扩散系数 [m²/s]
        transverse: S,
    },

    /// 基于湍流的扩散（涡粘度 / Schmidt 数）
    Turbulent {
        /// 分子扩散系数 [m²/s]
        molecular: S,
        /// 湍流 Schmidt 数（无量纲）
        schmidt_number: S,
    },
}

impl<S: Scalar> DiffusionCoefficient<S> {
    /// 获取单元的有效扩散系数（各向同性等效）
    pub fn effective_at(&self, cell_idx: usize, eddy_viscosity: Option<S>) -> S {
        match *self {
            Self::Constant(d) => d,
            Self::Variable(ref values) => values.get(cell_idx).copied().unwrap_or(S::ZERO),
            Self::Anisotropic { longitudinal, transverse } => {
                // 几何平均作为各向同性等效
                (longitudinal * transverse).sqrt()
            }
            Self::Turbulent { molecular, schmidt_number } => {
                let nu_t = eddy_viscosity.unwrap_or(S::ZERO);
                molecular + nu_t / schmidt_number
            }
        }
    }
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
pub struct DiffusionOperator<S: Scalar + Pod + Default> {
    /// 配置（硬编码 f64，存储时）
    config: DiffusionConfig,
    /// 配置（运行时精度，计算时）
    coefficient: DiffusionCoefficient<S>,
    min_diffusivity: S,
    max_diffusivity: S,
    /// 面扩散系数缓存
    face_diffusivity: AlignedVec<S>,
    /// 扩散通量缓存
    face_flux: AlignedVec<S>,
    /// 扩散源项（单元体积分）
    cell_diffusion: AlignedVec<S>,
}

impl<S: Scalar + Pod + Default> DiffusionOperator<S> {
    /// 创建新的扩散算子
    ///
    /// # 参数
    ///
    /// - `n_cells`: 单元数量
    /// - `n_faces`: 面数量
    /// - `config`: 扩散配置（f64）
    pub fn new(n_cells: usize, n_faces: usize, config: DiffusionConfig) -> Self {
        // 转换配置到运行时精度
        let coefficient = config.coefficient.to_precision();
        let min_diffusivity = S::from_config(config.min_diffusivity).unwrap_or(S::ZERO);
        let max_diffusivity = S::from_config(config.max_diffusivity).unwrap_or(S::ZERO);
        
        Self {
            config,
            coefficient,
            min_diffusivity,
            max_diffusivity,
            face_diffusivity: AlignedVec::zeros(n_faces),
            face_flux: AlignedVec::zeros(n_faces),
            cell_diffusion: AlignedVec::zeros(n_cells),
        }
    }

    /// 获取配置引用
    pub fn config(&self) -> &DiffusionConfig {
        &self.config
    }

    /// 获取扩散系数配置引用（运行时精度）
    pub fn coefficient(&self) -> &DiffusionCoefficient<S> {
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
        eddy_viscosity: Option<&[S]>,
    ) {
        if !self.config.enabled {
            self.face_diffusivity.as_mut_slice().fill(S::ZERO);
            return;
        }

        for face_idx in 0..mesh.n_faces() {
            let owner = mesh.face_owner(face_idx);
            let neighbor = mesh.face_neighbor(face_idx);

            let nu_t_o = eddy_viscosity.map(|nu| nu[owner]);
            let d_owner = self.coefficient.effective_at(owner, nu_t_o);

            let d_face = if let Some(neigh) = neighbor {
                // 内部面：调和平均
                let nu_t_n = eddy_viscosity.map(|nu| nu[neigh]);
                let d_neigh = self.coefficient.effective_at(neigh, nu_t_n);
                harmonic_mean(d_owner, d_neigh)
            } else {
                // 边界面：使用内部值
                d_owner
            };

            // 应用限制
            self.face_diffusivity[face_idx] = d_face
                .max(self.min_diffusivity)
                .min(self.max_diffusivity);
        }
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
        concentration: &[S],
    ) -> &[S] {
        if !self.config.enabled {
            self.face_flux.as_mut_slice().fill(S::ZERO);
            return self.face_flux.as_slice();
        }

        for face_idx in 0..mesh.n_faces() {
            let owner = mesh.face_owner(face_idx);
            let neighbor = mesh.face_neighbor(face_idx);

            let d = self.face_diffusivity[face_idx];
            let length = S::from_config(mesh.face_length(face_idx) as f64).unwrap_or(S::ZERO);

            let flux = if let Some(neigh) = neighbor {
                // 内部面：中心差分
                let dist = S::from_config(mesh.face_dist_o2n(face_idx) as f64).unwrap_or(S::ZERO);
                if dist > S::from_config(1e-14).unwrap_or(S::ZERO) {
                    let grad_n = (concentration[neigh] - concentration[owner]) / dist;
                    -d * grad_n * length
                } else {
                    S::ZERO
                }
            } else {
                // 边界面：假设零梯度（由边界条件处理）
                S::ZERO
            };

            self.face_flux[face_idx] = flux;
        }

        self.face_flux.as_slice()
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
        concentration: &[S],
    ) -> &[S] {
        // 先计算面通量
        self.compute_face_fluxes(mesh, concentration);

        // 清零
        self.cell_diffusion.as_mut_slice().fill(S::ZERO);

        // 累加面通量到单元
        for face_idx in 0..mesh.n_faces() {
            let owner = mesh.face_owner(face_idx);
            let neighbor = mesh.face_neighbor(face_idx);
            let flux = self.face_flux[face_idx];

            let area_o = S::from_config(mesh.cell_area_unchecked(owner) as f64).unwrap_or(S::ZERO);
            self.cell_diffusion[owner] -= flux / area_o;

            if let Some(neigh) = neighbor {
                let area_n = S::from_config(mesh.cell_area_unchecked(neigh) as f64).unwrap_or(S::ZERO);
                self.cell_diffusion[neigh] += flux / area_n;
            }
        }

        self.cell_diffusion.as_slice()
    }

    /// 获取面扩散系数
    pub fn face_diffusivity(&self) -> &[S] {
        self.face_diffusivity.as_slice()
    }

    /// 获取面扩散通量
    pub fn face_flux(&self) -> &[S] {
        self.face_flux.as_slice()
    }

    /// 获取单元扩散率
    pub fn cell_diffusion(&self) -> &[S] {
        self.cell_diffusion.as_slice()
    }
}

/// 各向异性扩散算子（泛型化）
///
/// 考虑流向（纵向）和垂直流向（横向）的不同扩散系数
pub struct AnisotropicDiffusionOperator<S: Scalar + Pod + Default> {
    /// 纵向扩散系数 [m²/s]
    longitudinal: S,
    /// 横向扩散系数 [m²/s]
    transverse: S,
    /// 面扩散通量缓存
    face_flux: AlignedVec<S>,
}

impl<S: Scalar + Pod + Default> AnisotropicDiffusionOperator<S> {
    /// 创建新的各向异性扩散算子
    pub fn new(n_faces: usize, longitudinal: S, transverse: S) -> Self {
        Self {
            longitudinal,
            transverse,
            face_flux: AlignedVec::zeros(n_faces),
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
        concentration: &[S],
        velocity_x: &[S],
        velocity_y: &[S],
    ) -> &[S] {
        for face_idx in 0..mesh.n_faces() {
            let owner = mesh.face_owner(face_idx);
            let neighbor = mesh.face_neighbor(face_idx);

            let flux = if let Some(neigh) = neighbor {
                let normal = mesh.face_normal(face_idx);
                let normal_x = S::from_config(normal.x as f64).unwrap_or(S::ZERO);
                let normal_y = S::from_config(normal.y as f64).unwrap_or(S::ZERO);
                let length = S::from_config(mesh.face_length(face_idx) as f64).unwrap_or(S::ZERO);
                let dist = S::from_config(mesh.face_dist_o2n(face_idx) as f64).unwrap_or(S::ZERO);

                if dist < S::from_config(1e-14).unwrap_or(S::ZERO) {
                    S::ZERO
                } else {
                    // 计算流向单位向量
                    let u_o = velocity_x[owner];
                    let v_o = velocity_y[owner];
                    let u_n = velocity_x[neigh];
                    let v_n = velocity_y[neigh];

                    let half = S::from_config(0.5).unwrap_or(S::ZERO);
                    let u_avg = half * (u_o + u_n);
                    let v_avg = half * (v_o + v_n);
                    let speed = (u_avg * u_avg + v_avg * v_avg).sqrt();

                    // 有效扩散系数（投影到面法向）
                    let d_eff = if speed > S::from_config(1e-8).unwrap_or(S::ZERO) {
                        let e_x = u_avg / speed;
                        let e_y = v_avg / speed;

                        // 法向方向的流向分量
                        let cos_theta = e_x * normal_x + e_y * normal_y;
                        let sin_theta = (S::ONE - cos_theta * cos_theta).sqrt();

                        self.longitudinal * cos_theta.abs() + self.transverse * sin_theta
                    } else {
                        // 静水时使用几何平均
                        (self.longitudinal * self.transverse).sqrt()
                    };

                    let grad_n = (concentration[neigh] - concentration[owner]) / dist;
                    -d_eff * grad_n * length
                }
            } else {
                S::ZERO
            };

            self.face_flux[face_idx] = flux;
        }

        self.face_flux.as_slice()
    }

    /// 获取面扩散通量
    pub fn face_flux(&self) -> &[S] {
        self.face_flux.as_slice()
    }
}

/// 计算调和平均
#[inline]
fn harmonic_mean<S: Scalar>(a: S, b: S) -> S {
    if a.abs() < S::from_config(1e-14).unwrap_or(S::ZERO) || b.abs() < S::from_config(1e-14).unwrap_or(S::ZERO) {
        S::ZERO
    } else {
        S::from_config(2.0).unwrap_or(S::ZERO) * a * b / (a + b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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