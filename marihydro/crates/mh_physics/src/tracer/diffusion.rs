// crates/mh_physics/src/tracer/diffusion.rs

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
//! let op = DiffusionOperator::new(n_cells, DiffusionCoefficient::Constant(10.0));
//!
//! // 计算扩散通量
//! let fluxes = op.compute_face_fluxes(&mesh, &concentration);
//! ```

use crate::adapter::PhysicsMesh;
use mh_foundation::{AlignedVec, Scalar};
use serde::{Deserialize, Serialize};

/// 扩散系数类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffusionCoefficient {
    /// 常数扩散系数 [m²/s]
    Constant(Scalar),

    /// 空间变化的扩散系数（每单元一个值）
    Variable(Vec<Scalar>),

    /// 各向异性扩散（纵向、横向系数）
    Anisotropic {
        /// 纵向扩散系数 [m²/s]
        longitudinal: Scalar,
        /// 横向扩散系数 [m²/s]
        transverse: Scalar,
    },

    /// 基于湍流的扩散（涡粘度 / Schmidt 数）
    Turbulent {
        /// 分子扩散系数 [m²/s]
        molecular: Scalar,
        /// 湍流 Schmidt 数（无量纲）
        schmidt_number: Scalar,
    },
}

impl Default for DiffusionCoefficient {
    fn default() -> Self {
        Self::Constant(1.0)
    }
}

impl DiffusionCoefficient {
    /// 创建常数扩散系数
    pub fn constant(d: Scalar) -> Self {
        Self::Constant(d)
    }

    /// 创建零扩散
    pub fn zero() -> Self {
        Self::Constant(0.0)
    }

    /// 创建各向异性扩散
    pub fn anisotropic(longitudinal: Scalar, transverse: Scalar) -> Self {
        Self::Anisotropic {
            longitudinal,
            transverse,
        }
    }

    /// 创建湍流扩散
    pub fn turbulent(molecular: Scalar, schmidt_number: Scalar) -> Self {
        Self::Turbulent {
            molecular,
            schmidt_number,
        }
    }

    /// 获取单元的有效扩散系数（各向同性等效）
    pub fn effective_at(&self, cell_idx: usize, eddy_viscosity: Option<Scalar>) -> Scalar {
        match self {
            Self::Constant(d) => *d,
            Self::Variable(values) => values.get(cell_idx).copied().unwrap_or(0.0),
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
                let nu_t = eddy_viscosity.unwrap_or(0.0);
                molecular + nu_t / schmidt_number
            }
        }
    }
}

/// 扩散配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionConfig {
    /// 扩散系数
    pub coefficient: DiffusionCoefficient,
    /// 是否启用扩散
    pub enabled: bool,
    /// 最小扩散系数（数值稳定性）
    pub min_diffusivity: Scalar,
    /// 最大扩散系数（限制）
    pub max_diffusivity: Scalar,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            coefficient: DiffusionCoefficient::default(),
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
    pub fn constant(d: Scalar) -> Self {
        Self {
            coefficient: DiffusionCoefficient::Constant(d),
            ..Default::default()
        }
    }
}

/// 扩散算子
///
/// 计算扩散通量和扩散项对浓度场的贡献
pub struct DiffusionOperator {
    /// 配置
    config: DiffusionConfig,
    /// 面扩散系数缓存
    face_diffusivity: AlignedVec<Scalar>,
    /// 扩散通量缓存
    face_flux: AlignedVec<Scalar>,
    /// 扩散源项（单元体积分）
    cell_diffusion: AlignedVec<Scalar>,
}

impl DiffusionOperator {
    /// 创建新的扩散算子
    ///
    /// # 参数
    ///
    /// - `n_cells`: 单元数量
    /// - `n_faces`: 面数量
    /// - `config`: 扩散配置
    pub fn new(n_cells: usize, n_faces: usize, config: DiffusionConfig) -> Self {
        Self {
            config,
            face_diffusivity: AlignedVec::zeros(n_faces),
            face_flux: AlignedVec::zeros(n_faces),
            cell_diffusion: AlignedVec::zeros(n_cells),
        }
    }

    /// 获取配置引用
    pub fn config(&self) -> &DiffusionConfig {
        &self.config
    }

    /// 获取可变配置引用
    pub fn config_mut(&mut self) -> &mut DiffusionConfig {
        &mut self.config
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
        eddy_viscosity: Option<&[Scalar]>,
    ) {
        if !self.config.enabled {
            self.face_diffusivity.as_mut_slice().fill(0.0);
            return;
        }

        for face_idx in 0..mesh.n_faces() {
            let owner = mesh.face_owner(face_idx);
            let neighbor = mesh.face_neighbor(face_idx);

            let nu_t_o = eddy_viscosity.map(|nu| nu[owner]);
            let d_owner = self.config.coefficient.effective_at(owner, nu_t_o);

            let d_face = if let Some(neigh) = neighbor {
                // 内部面：调和平均
                let nu_t_n = eddy_viscosity.map(|nu| nu[neigh]);
                let d_neigh = self.config.coefficient.effective_at(neigh, nu_t_n);
                harmonic_mean(d_owner, d_neigh)
            } else {
                // 边界面：使用内部值
                d_owner
            };

            // 应用限制
            self.face_diffusivity[face_idx] = d_face
                .max(self.config.min_diffusivity)
                .min(self.config.max_diffusivity);
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
        concentration: &[Scalar],
    ) -> &[Scalar] {
        if !self.config.enabled {
            self.face_flux.as_mut_slice().fill(0.0);
            return self.face_flux.as_slice();
        }

        for face_idx in 0..mesh.n_faces() {
            let owner = mesh.face_owner(face_idx);
            let neighbor = mesh.face_neighbor(face_idx);

            let d = self.face_diffusivity[face_idx];
            let length = mesh.face_length(face_idx);

            let flux = if let Some(neigh) = neighbor {
                // 内部面：中心差分
                let dist = mesh.face_dist_o2n(face_idx);
                if dist > 1e-14 {
                    let grad_n = (concentration[neigh] - concentration[owner]) / dist;
                    -d * grad_n * length
                } else {
                    0.0
                }
            } else {
                // 边界面：假设零梯度（由边界条件处理）
                0.0
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
        concentration: &[Scalar],
    ) -> &[Scalar] {
        // 先计算面通量
        self.compute_face_fluxes(mesh, concentration);

        // 清零
        self.cell_diffusion.as_mut_slice().fill(0.0);

        // 累加面通量到单元
        for face_idx in 0..mesh.n_faces() {
            let owner = mesh.face_owner(face_idx);
            let neighbor = mesh.face_neighbor(face_idx);
            let flux = self.face_flux[face_idx];

            let area_o = mesh.cell_area_unchecked(owner);
            self.cell_diffusion[owner] -= flux / area_o;

            if let Some(neigh) = neighbor {
                let area_n = mesh.cell_area_unchecked(neigh);
                self.cell_diffusion[neigh] += flux / area_n;
            }
        }

        self.cell_diffusion.as_slice()
    }

    /// 获取面扩散系数
    pub fn face_diffusivity(&self) -> &[Scalar] {
        self.face_diffusivity.as_slice()
    }

    /// 获取面扩散通量
    pub fn face_flux(&self) -> &[Scalar] {
        self.face_flux.as_slice()
    }

    /// 获取单元扩散率
    pub fn cell_diffusion(&self) -> &[Scalar] {
        self.cell_diffusion.as_slice()
    }
}

/// 各向异性扩散算子
///
/// 考虑流向（纵向）和垂直流向（横向）的不同扩散系数
pub struct AnisotropicDiffusionOperator {
    /// 纵向扩散系数 [m²/s]
    longitudinal: Scalar,
    /// 横向扩散系数 [m²/s]
    transverse: Scalar,
    /// 面扩散通量缓存
    face_flux: AlignedVec<Scalar>,
}

impl AnisotropicDiffusionOperator {
    /// 创建新的各向异性扩散算子
    pub fn new(n_faces: usize, longitudinal: Scalar, transverse: Scalar) -> Self {
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
        concentration: &[Scalar],
        velocity_x: &[Scalar],
        velocity_y: &[Scalar],
    ) -> &[Scalar] {
        for face_idx in 0..mesh.n_faces() {
            let owner = mesh.face_owner(face_idx);
            let neighbor = mesh.face_neighbor(face_idx);

            let flux = if let Some(neigh) = neighbor {
                let normal = mesh.face_normal(face_idx);
                let length = mesh.face_length(face_idx);
                let dist = mesh.face_dist_o2n(face_idx);

                if dist < 1e-14 {
                    0.0
                } else {
                    // 计算流向单位向量
                    let u_o = velocity_x[owner];
                    let v_o = velocity_y[owner];
                    let u_n = velocity_x[neigh];
                    let v_n = velocity_y[neigh];

                    let u_avg = 0.5 * (u_o + u_n);
                    let v_avg = 0.5 * (v_o + v_n);
                    let speed = (u_avg * u_avg + v_avg * v_avg).sqrt();

                    // 有效扩散系数（投影到面法向）
                    let d_eff = if speed > 1e-8 {
                        let e_x = u_avg / speed;
                        let e_y = v_avg / speed;

                        // 法向方向的流向分量
                        let cos_theta = e_x * normal.x + e_y * normal.y;
                        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                        self.longitudinal * cos_theta.abs() + self.transverse * sin_theta
                    } else {
                        // 静水时使用几何平均
                        (self.longitudinal * self.transverse).sqrt()
                    };

                    let grad_n = (concentration[neigh] - concentration[owner]) / dist;
                    -d_eff * grad_n * length
                }
            } else {
                0.0
            };

            self.face_flux[face_idx] = flux;
        }

        self.face_flux.as_slice()
    }

    /// 获取面扩散通量
    pub fn face_flux(&self) -> &[Scalar] {
        self.face_flux.as_slice()
    }
}

/// 计算调和平均
#[inline]
fn harmonic_mean(a: Scalar, b: Scalar) -> Scalar {
    if a.abs() < 1e-14 || b.abs() < 1e-14 {
        0.0
    } else {
        2.0 * a * b / (a + b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_coefficient() {
        let coef = DiffusionCoefficient::constant(10.0);
        assert!((coef.effective_at(0, None) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_variable_coefficient() {
        let coef = DiffusionCoefficient::Variable(vec![1.0, 2.0, 3.0]);
        assert!((coef.effective_at(0, None) - 1.0).abs() < 1e-10);
        assert!((coef.effective_at(1, None) - 2.0).abs() < 1e-10);
        assert!((coef.effective_at(2, None) - 3.0).abs() < 1e-10);
        assert!((coef.effective_at(10, None)).abs() < 1e-10); // 越界返回 0
    }

    #[test]
    fn test_anisotropic_coefficient() {
        let coef = DiffusionCoefficient::anisotropic(100.0, 10.0);
        let effective = coef.effective_at(0, None);
        // 几何平均 = sqrt(100 * 10) ≈ 31.62
        assert!((effective - 31.622776601683793).abs() < 1e-10);
    }

    #[test]
    fn test_turbulent_coefficient() {
        let coef = DiffusionCoefficient::turbulent(1.0, 0.7);

        // 无涡粘度时只有分子扩散
        assert!((coef.effective_at(0, None) - 1.0).abs() < 1e-10);

        // 有涡粘度时 = molecular + nu_t / Sc
        let nu_t = 7.0;
        let expected = 1.0 + 7.0 / 0.7; // = 11.0
        assert!((coef.effective_at(0, Some(nu_t)) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_harmonic_mean() {
        assert!((harmonic_mean(2.0, 2.0) - 2.0).abs() < 1e-10);
        assert!((harmonic_mean(1.0, 3.0) - 1.5).abs() < 1e-10);
        assert!(harmonic_mean(0.0, 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_disabled() {
        let config = DiffusionConfig::disabled();
        assert!(!config.enabled);
    }
}
