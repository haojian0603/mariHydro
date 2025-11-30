// src-tauri/src/marihydro/physics/numerics/reconstruction/muscl.rs
//! MUSCL 重构模块
//!
//! 实现 MUSCL (Monotonic Upstream-centered Scheme for Conservation Laws) 重构方法。
//! 使用 TVD (Total Variation Diminishing) 通量限制器来控制振荡。
//!
//! # 注意
//! 本模块中的 `FluxLimiter` trait 与 `limiter/` 模块中的 `Limiter` trait 不同：
//! - `FluxLimiter`: TVD 通量限制器，输入斜率比 r，输出 φ(r) ∈ [0, 2]
//! - `limiter::Limiter`: 梯度限制器，基于网格邻居信息计算限制因子 α ∈ [0, 1]

use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, FaceIndex};
use crate::marihydro::physics::numerics::gradient::{GradientMethod, GreenGaussGradient, ScalarGradientStorage};
use glam::DVec2;
use rayon::prelude::*;

/// TVD 通量限制器 trait
///
/// 实现 TVD 条件下的通量限制函数 φ(r)，其中 r 是连续单元间的斜率比。
/// 所有实现必须满足 TVD 条件：φ(r) ∈ [0, 2] 且 φ(1) = 1
pub trait FluxLimiter: Send + Sync {
    /// 计算限制因子 φ(r)
    ///
    /// # Arguments
    /// * `r` - 斜率比 (连续单元间梯度的比值)
    ///
    /// # Returns
    /// 限制因子，通常在 [0, 2] 范围内
    fn limit(&self, r: f64) -> f64;
    
    /// 限制器名称
    fn name(&self) -> &'static str;
}

// 为向后兼容保留别名
pub use FluxLimiter as Limiter;

/// Minmod 限制器 (TVD, 最保守)
///
/// φ(r) = max(0, min(1, r))
///
/// 最保守的限制器，数值耗散大但稳定性好。
#[derive(Debug, Clone, Copy, Default)]
pub struct MinmodLimiter;

impl FluxLimiter for MinmodLimiter {
    #[inline]
    fn limit(&self, r: f64) -> f64 { r.max(0.0).min(1.0) }
    fn name(&self) -> &'static str { "Minmod" }
}

/// Van Leer 限制器 (TVD, 平滑)
///
/// φ(r) = (r + |r|) / (1 + |r|)
///
/// 平滑的二阶限制器，在 TVD 区域内接近二阶精度。
#[derive(Debug, Clone, Copy, Default)]
pub struct VanLeerLimiter;

impl FluxLimiter for VanLeerLimiter {
    #[inline]
    fn limit(&self, r: f64) -> f64 {
        if r <= 0.0 { 0.0 } else { (r + r.abs()) / (1.0 + r.abs()) }
    }
    fn name(&self) -> &'static str { "VanLeer" }
}

/// Van Albada 限制器 (TVD, 平滑)
///
/// φ(r) = (r² + r) / (r² + 1)
///
/// 另一种平滑的二阶限制器。
#[derive(Debug, Clone, Copy, Default)]
pub struct VanAlbadaLimiter;

impl FluxLimiter for VanAlbadaLimiter {
    #[inline]
    fn limit(&self, r: f64) -> f64 {
        if r <= 0.0 { 0.0 } else { (r * r + r) / (r * r + 1.0) }
    }
    fn name(&self) -> &'static str { "VanAlbada" }
}

/// Superbee 限制器 (TVD, 激进)
///
/// φ(r) = max(0, min(2r, 1), min(r, 2))
///
/// 最激进的限制器，保留最多梯度信息，可能产生过冲。
#[derive(Debug, Clone, Copy)]
pub struct SuperbeeLimiter;

impl FluxLimiter for SuperbeeLimiter {
    #[inline]
    fn limit(&self, r: f64) -> f64 {
        let a = (2.0 * r).min(1.0);
        let b = r.min(2.0);
        a.max(b).max(0.0)
    }
    fn name(&self) -> &'static str { "Superbee" }
}

/// MUSCL 重构器
pub struct MusclReconstructor<L: FluxLimiter = MinmodLimiter> {
    limiter: L,
    grad_storage: ScalarGradientStorage,
    gradient_method: GreenGaussGradient,
}

impl<L: FluxLimiter> MusclReconstructor<L> {
    pub fn new(n_cells: usize, limiter: L) -> Self {
        Self {
            limiter,
            grad_storage: ScalarGradientStorage::new(n_cells),
            gradient_method: GreenGaussGradient::new(),
        }
    }

    pub fn reconstruct_face<M: MeshAccess>(
        &mut self, field: &[f64], mesh: &M,
    ) -> MhResult<Vec<(f64, f64)>> {
        self.gradient_method.compute_scalar_gradient(field, mesh, &mut self.grad_storage)?;
        let n_faces = mesh.n_faces();
        let mut result = vec![(0.0, 0.0); n_faces];
        for f in 0..n_faces {
            let face = FaceIndex(f);
            let (owner, neighbor_opt) = mesh.face_cells(face);
            if neighbor_opt.is_none() {
                let phi_o = field[owner.0];
                result[f] = (phi_o, phi_o);
                continue;
            }
            let neighbor = neighbor_opt.unwrap();
            let center_o = mesh.cell_center(owner);
            let center_n = mesh.cell_center(neighbor);
            let face_center = mesh.face_center(face);
            let r_of = face_center - center_o;
            let r_nf = face_center - center_n;
            let grad_o = self.grad_storage.get(owner.0);
            let grad_n = self.grad_storage.get(neighbor.0);
            let phi_o = field[owner.0];
            let phi_n = field[neighbor.0];
            let delta_o = grad_o.dot(r_of);
            let delta_n = grad_n.dot(r_nf);
            let central_diff = phi_n - phi_o;
            let r_o = if delta_o.abs() > 1e-14 { central_diff / delta_o } else { 1.0 };
            let r_n = if delta_n.abs() > 1e-14 { -central_diff / delta_n } else { 1.0 };
            let phi_l = phi_o + 0.5 * self.limiter.limit(r_o) * delta_o;
            let phi_r = phi_n + 0.5 * self.limiter.limit(r_n) * delta_n;
            result[f] = (phi_l, phi_r);
        }
        Ok(result)
    }
}

pub fn barth_jespersen_limiter<M: MeshAccess>(
    field: &[f64], grad: &mut ScalarGradientStorage, mesh: &M,
) {
    for i in 0..mesh.n_cells() {
        let cell = CellIndex(i);
        let phi_c = field[i];
        let center_c = mesh.cell_center(cell);
        let mut phi_min = phi_c;
        let mut phi_max = phi_c;
        for &face in mesh.cell_faces(cell) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            if !neighbor.is_valid() { continue; }
            let other = if i == owner.0 { neighbor.0 } else { owner.0 };
            phi_min = phi_min.min(field[other]);
            phi_max = phi_max.max(field[other]);
        }
        let g = grad.get(i);
        let mut alpha = 1.0;
        for &face in mesh.cell_faces(cell) {
            let face_center = mesh.face_center(face);
            let r = face_center - center_c;
            let delta = g.dot(r);
            if delta.abs() < 1e-14 { continue; }
            let alpha_f = if delta > 0.0 { ((phi_max - phi_c) / delta).min(1.0) }
                          else { ((phi_min - phi_c) / delta).min(1.0) };
            alpha = alpha.min(alpha_f.max(0.0));
        }
        grad.set(i, g * alpha);
    }
}

pub fn venkatakrishnan_limiter<M: MeshAccess>(
    field: &[f64], grad: &mut ScalarGradientStorage, mesh: &M, k: f64,
) {
    for i in 0..mesh.n_cells() {
        let cell = CellIndex(i);
        let phi_c = field[i];
        let center_c = mesh.cell_center(cell);
        let mut phi_min = phi_c;
        let mut phi_max = phi_c;
        for &face in mesh.cell_faces(cell) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            if !neighbor.is_valid() { continue; }
            let other = if i == owner.0 { neighbor.0 } else { owner.0 };
            phi_min = phi_min.min(field[other]);
            phi_max = phi_max.max(field[other]);
        }
        let eps_sq = (k * mesh.cell_area(cell).sqrt()).powi(3);
        let g = grad.get(i);
        let mut alpha = 1.0;
        for &face in mesh.cell_faces(cell) {
            let face_center = mesh.face_center(face);
            let delta = g.dot(face_center - center_c);
            if delta.abs() < 1e-14 { continue; }
            let dm = if delta > 0.0 { phi_max - phi_c } else { phi_min - phi_c };
            let r_sq = dm / delta;
            let num = (r_sq * r_sq + 2.0 * r_sq) * delta * delta + eps_sq;
            let den = (r_sq * r_sq + r_sq + 2.0) * delta * delta + eps_sq;
            alpha = alpha.min((num / den).clamp(0.0, 1.0));
        }
        grad.set(i, g * alpha);
    }
}
