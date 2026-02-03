// marihydro/crates/mh_physics/src/numerics/reconstruction/weno.rs
//! WENO 重构器实现（泛型版本）
//!
//! 当前提供 WENO2 的稳健实现，面向非结构网格采用简化模板：
//! - 左侧使用 owner + owner 的邻居
//! - 右侧使用 neighbor + neighbor 的邻居
//!
//! 当模板不完整时自动回退到一阶。

use std::marker::PhantomData;
use std::sync::Arc;

use mh_runtime::RuntimeScalar;

use super::traits::{ReconstructedStateGeneric, ReconstructorGeneric};
use crate::adapter::{FaceIndex, PhysicsMesh};

/// WENO 重构配置
#[derive(Debug, Clone)]
pub struct WenoConfig {
    /// WENO 阶数（当前支持 2）
    pub order: usize,
    /// 小数 ε 防止除零
    pub epsilon: f64,
    /// 非线性权重指数 p
    pub power: i32,
    /// 是否启用 Z-WENO（预留）
    pub use_z_weno: bool,
}

impl Default for WenoConfig {
    fn default() -> Self {
        Self {
            order: 2,
            epsilon: 1e-6,
            power: 2,
            use_z_weno: false,
        }
    }
}

/// WENO 模板缓存
#[derive(Debug, Clone, Copy)]
struct WenoStencil {
    owner: usize,
    neighbor: Option<usize>,
    left_extra: Option<usize>,
    right_extra: Option<usize>,
    complete: bool,
}

/// WENO 重构器（泛型）
pub struct WenoReconstructorGeneric<S: RuntimeScalar> {
    config: WenoConfig,
    mesh: Arc<PhysicsMesh>,
    stencils: Vec<WenoStencil>,
    _marker: PhantomData<S>,
}

impl<S: RuntimeScalar> WenoReconstructorGeneric<S> {
    /// 创建新的 WENO 重构器
    pub fn new(config: WenoConfig, mesh: Arc<PhysicsMesh>) -> Self {
        let config = Self::sanitize_config(config);
        let stencils = Self::build_stencils(&mesh);
        Self {
            config,
            mesh,
            stencils,
            _marker: PhantomData,
        }
    }

    /// 更新配置
    pub fn set_config(&mut self, config: WenoConfig) {
        self.config = Self::sanitize_config(config);
    }

    /// 更新网格并重建模板
    pub fn update_mesh(&mut self, mesh: Arc<PhysicsMesh>) {
        self.stencils = Self::build_stencils(&mesh);
        self.mesh = mesh;
    }

    fn sanitize_config(mut config: WenoConfig) -> WenoConfig {
        if config.order < 1 {
            config.order = 1;
        }
        if config.epsilon <= 0.0 || !config.epsilon.is_finite() {
            config.epsilon = 1e-6;
        }
        config
    }

    fn build_stencils(mesh: &PhysicsMesh) -> Vec<WenoStencil> {
        let n_faces = mesh.face_count();
        let mut stencils = Vec::with_capacity(n_faces);

        for face in 0..n_faces {
            let owner = mesh.face_owner(FaceIndex::new(face)).get();
            let neighbor = mesh.face_neighbor(FaceIndex::new(face)).map(|c| c.get());
            let left_extra = neighbor.and_then(|n| Self::find_extra_neighbor(mesh, owner, n));
            let right_extra = neighbor.and_then(|n| Self::find_extra_neighbor(mesh, n, owner));
            let complete = neighbor.is_some() && left_extra.is_some() && right_extra.is_some();

            stencils.push(WenoStencil {
                owner,
                neighbor,
                left_extra,
                right_extra,
                complete,
            });
        }

        stencils
    }

    fn find_extra_neighbor(mesh: &PhysicsMesh, cell: usize, exclude: usize) -> Option<usize> {
        for n in mesh.cell_neighbors(mh_runtime::CellIndex::new(cell)) {
            let idx = n.get();
            if idx != exclude {
                return Some(idx);
            }
        }
        None
    }

    #[inline]
    fn weno2_one_side(&self, v_far: S, v_near: S, v_center: S, eps: S) -> S {
        let d0 = S::from_f64(1.0 / 3.0).unwrap_or(S::ONE);
        let d1 = S::from_f64(2.0 / 3.0).unwrap_or(S::ONE);

        let half = S::from_f64(0.5).unwrap_or(S::HALF);
        let p0 = v_near + half * (v_near - v_far);
        let p1 = v_near + half * (v_center - v_near);

        let beta0 = (v_near - v_far) * (v_near - v_far);
        let beta1 = (v_center - v_near) * (v_center - v_near);

        let pw = self.config.power;
        let alpha0 = d0 / (eps + beta0).powi(pw);
        let alpha1 = d1 / (eps + beta1).powi(pw);
        let alpha_sum = alpha0 + alpha1;

        let w0 = alpha0 / alpha_sum;
        let w1 = alpha1 / alpha_sum;

        w0 * p0 + w1 * p1
    }
}

impl<S: RuntimeScalar> ReconstructorGeneric<S> for WenoReconstructorGeneric<S> {
    fn compute_gradients(&mut self, _values: &[S]) {
        // WENO 不需要显式梯度
    }

    fn reconstruct_scalar(&self, face_id: usize, values: &[S]) -> ReconstructedStateGeneric<S> {
        let stencil = &self.stencils[face_id];
        let owner = stencil.owner;

        let Some(neighbor) = stencil.neighbor else {
            let v = values[owner];
            return ReconstructedStateGeneric::from_values(v, v);
        };

        if self.config.order < 2 || !stencil.complete {
            return ReconstructedStateGeneric::from_values(values[owner], values[neighbor]);
        }

        let eps = S::from_f64(self.config.epsilon).unwrap_or(S::EPSILON);
        let v_l = values[owner];
        let v_r = values[neighbor];

        let v_ll = stencil.left_extra.map(|i| values[i]).unwrap_or(v_l);
        let v_rr = stencil.right_extra.map(|i| values[i]).unwrap_or(v_r);

        let q_left = self.weno2_one_side(v_ll, v_l, v_r, eps);
        let q_right = self.weno2_one_side(v_rr, v_r, v_l, eps);

        ReconstructedStateGeneric::new(q_left, q_right)
    }

    fn get_limited_gradient_tuple(&self, _cell_id: usize) -> (S, S) {
        (S::ZERO, S::ZERO)
    }

    fn is_second_order(&self) -> bool {
        self.config.order >= 2
    }

    fn name(&self) -> &'static str {
        match self.config.order {
            2 => "WENO2",
            3 => "WENO3",
            _ => "WENO",
        }
    }
}
