//! Green-Gauss 梯度 - 消除 PERF-004

use super::traits::{GradientMethod, ScalarGradientStorage, VectorGradientStorage};
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, FaceIndex};
use glam::DVec2;
use rayon::prelude::*;

/// 面插值方法
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FaceInterpolation {
    /// 简单算术平均 (原始方法)
    #[default]
    Arithmetic,
    /// 距离加权插值 (更精确，适用于非均匀网格)
    DistanceWeighted,
}

pub struct GreenGaussGradient {
    parallel: bool,
    parallel_threshold: usize,
    /// P1-001: 面插值方法
    face_interpolation: FaceInterpolation,
}

impl Default for GreenGaussGradient {
    fn default() -> Self {
        Self {
            parallel: true,
            parallel_threshold: 1000,
            face_interpolation: FaceInterpolation::Arithmetic,
        }
    }
}

impl GreenGaussGradient {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }
    pub fn with_threshold(mut self, t: usize) -> Self {
        self.parallel_threshold = t;
        self
    }
    
    /// P1-001: 设置面插值方法
    pub fn with_face_interpolation(mut self, method: FaceInterpolation) -> Self {
        self.face_interpolation = method;
        self
    }
    
    /// 使用距离加权插值（推荐用于非均匀网格）
    pub fn with_distance_weighted(self) -> Self {
        self.with_face_interpolation(FaceInterpolation::DistanceWeighted)
    }
    
    /// 距离加权面插值
    /// 
    /// P1-001: 使用距离加权代替简单算术平均，提高非均匀网格精度
    /// 
    /// phi_face = (phi_n * d_o + phi_o * d_n) / (d_o + d_n)
    /// 其中 d_o, d_n 分别是 owner 和 neighbor 到面的距离
    #[inline]
    fn distance_weighted_interpolate(phi_o: f64, phi_n: f64, d_o: f64, d_n: f64) -> f64 {
        let d_total = d_o + d_n;
        if d_total < 1e-14 {
            0.5 * (phi_o + phi_n)  // 回退到算术平均
        } else {
            // 距离加权：离得近的权重大
            (phi_n * d_o + phi_o * d_n) / d_total
        }
    }

    fn compute_cell_scalar<M: MeshAccess>(
        &self,
        cell_idx: usize,
        field: &[f64],
        mesh: &M,
    ) -> DVec2 {
        let cell = CellIndex(cell_idx);
        let area = mesh.cell_area(cell);
        if area < 1e-14 {
            return DVec2::ZERO;
        }
        
        let center = mesh.cell_centroid(cell);

        let mut grad = DVec2::ZERO;
        for &face in mesh.cell_faces(cell) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            let normal = mesh.face_normal(face);
            let length = mesh.face_length(face);

            let sign = if cell_idx == owner.0 { 1.0 } else { -1.0 };
            let ds = normal * length * sign;

            let phi_face = if !neighbor.is_valid() {
                field[cell_idx]
            } else {
                let other = if cell_idx == owner.0 {
                    neighbor.0
                } else {
                    owner.0
                };
                
                match self.face_interpolation {
                    FaceInterpolation::Arithmetic => {
                        0.5 * (field[cell_idx] + field[other])
                    }
                    FaceInterpolation::DistanceWeighted => {
                        let face_center = mesh.face_centroid(face);
                        let center_other = mesh.cell_centroid(CellIndex(other));
                        let d_self = (face_center - center).length();
                        let d_other = (face_center - center_other).length();
                        Self::distance_weighted_interpolate(
                            field[cell_idx], field[other], d_self, d_other
                        )
                    }
                }
            };
            grad += ds * phi_face;
        }
        grad / area
    }

    fn compute_cell_vector<M: MeshAccess>(
        &self,
        cell_idx: usize,
        field: &[DVec2],
        mesh: &M,
    ) -> (DVec2, DVec2) {
        let cell = CellIndex(cell_idx);
        let area = mesh.cell_area(cell);
        if area < 1e-14 {
            return (DVec2::ZERO, DVec2::ZERO);
        }
        
        let center = mesh.cell_centroid(cell);

        let mut grad_u = DVec2::ZERO;
        let mut grad_v = DVec2::ZERO;

        for &face in mesh.cell_faces(cell) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            let normal = mesh.face_normal(face);
            let length = mesh.face_length(face);

            let sign = if cell_idx == owner.0 { 1.0 } else { -1.0 };
            let ds = normal * length * sign;

            let vel_face = if !neighbor.is_valid() {
                field[cell_idx]
            } else {
                let other = if cell_idx == owner.0 {
                    neighbor.0
                } else {
                    owner.0
                };
                
                match self.face_interpolation {
                    FaceInterpolation::Arithmetic => {
                        (field[cell_idx] + field[other]) * 0.5
                    }
                    FaceInterpolation::DistanceWeighted => {
                        let face_center = mesh.face_centroid(face);
                        let center_other = mesh.cell_centroid(CellIndex(other));
                        let d_self = (face_center - center).length();
                        let d_other = (face_center - center_other).length();
                        let vx = Self::distance_weighted_interpolate(
                            field[cell_idx].x, field[other].x, d_self, d_other
                        );
                        let vy = Self::distance_weighted_interpolate(
                            field[cell_idx].y, field[other].y, d_self, d_other
                        );
                        DVec2::new(vx, vy)
                    }
                }
            };
            grad_u += ds * vel_face.x;
            grad_v += ds * vel_face.y;
        }
        (grad_u / area, grad_v / area)
    }
}

impl GradientMethod for GreenGaussGradient {
    fn compute_scalar_gradient<M: MeshAccess>(
        &self,
        field: &[f64],
        mesh: &M,
        output: &mut ScalarGradientStorage,
    ) -> MhResult<()> {
        output.reset();
        for i in 0..mesh.n_cells() {
            output.set(i, self.compute_cell_scalar(i, field, mesh));
        }
        Ok(())
    }

    fn compute_vector_gradient<M: MeshAccess>(
        &self,
        field: &[DVec2],
        mesh: &M,
        output: &mut VectorGradientStorage,
    ) -> MhResult<()> {
        output.reset();
        for i in 0..mesh.n_cells() {
            let (gu, gv) = self.compute_cell_vector(i, field, mesh);
            output.set_grad_u(i, gu);
            output.set_grad_v(i, gv);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Green-Gauss"
    }
}

/// 专用并行版本
pub mod optimized {
    use super::*;
    use crate::marihydro::domain::mesh::UnstructuredMesh;

    pub fn compute_scalar_gradient_parallel(
        field: &[f64],
        mesh: &UnstructuredMesh,
        output: &mut ScalarGradientStorage,
    ) -> MhResult<()> {
        output.reset();
        let grads: Vec<DVec2> = (0..mesh.n_cells())
            .into_par_iter()
            .map(|i| {
                let cell = CellIndex(i);
                let area = mesh.cell_area(cell);
                if area < 1e-14 {
                    return DVec2::ZERO;
                }
                let mut g = DVec2::ZERO;
                for &face in mesh.cell_faces(cell) {
                    let owner = mesh.face_owner(face);
                    let neighbor = mesh.face_neighbor(face);
                    let normal = mesh.face_normal(face);
                    let length = mesh.face_length(face);
                    let sign = if i == owner.0 { 1.0 } else { -1.0 };
                    let ds = normal * length * sign;
                    let phi = if !neighbor.is_valid() {
                        field[i]
                    } else {
                        let o = if i == owner.0 { neighbor.0 } else { owner.0 };
                        0.5 * (field[i] + field[o])
                    };
                    g += ds * phi;
                }
                g / area
            })
            .collect();
        for (i, g) in grads.into_iter().enumerate() {
            output.set(i, g);
        }
        Ok(())
    }
}
