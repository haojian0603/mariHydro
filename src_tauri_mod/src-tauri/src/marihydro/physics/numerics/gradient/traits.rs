// src-tauri/src/marihydro/physics/numerics/gradient/traits.rs

//! 梯度计算泛型框架 - 消除 DUP-002

use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use glam::DVec2;

pub trait FieldValue: Clone + Copy + Send + Sync + Default {
    fn multiply_with_vector(&self, v: DVec2) -> GradientContribution;
    fn scale(&self, factor: f64) -> Self;
    fn add(&self, other: Self) -> Self;
    fn zero() -> Self;
}

impl FieldValue for f64 {
    #[inline]
    fn multiply_with_vector(&self, v: DVec2) -> GradientContribution {
        GradientContribution::Scalar(v * (*self))
    }
    #[inline]
    fn scale(&self, factor: f64) -> Self {
        *self * factor
    }
    #[inline]
    fn add(&self, other: Self) -> Self {
        *self + other
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
}

impl FieldValue for DVec2 {
    #[inline]
    fn multiply_with_vector(&self, v: DVec2) -> GradientContribution {
        GradientContribution::Vector {
            du: v * self.x,
            dv: v * self.y,
        }
    }
    #[inline]
    fn scale(&self, factor: f64) -> Self {
        *self * factor
    }
    #[inline]
    fn add(&self, other: Self) -> Self {
        *self + other
    }
    #[inline]
    fn zero() -> Self {
        DVec2::ZERO
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GradientContribution {
    Scalar(DVec2),
    Vector { du: DVec2, dv: DVec2 },
}

impl Default for GradientContribution {
    fn default() -> Self {
        Self::Scalar(DVec2::ZERO)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ScalarGradientStorage {
    pub grad_x: Vec<f64>,
    pub grad_y: Vec<f64>,
}

impl ScalarGradientStorage {
    pub fn new(n: usize) -> Self {
        Self {
            grad_x: vec![0.0; n],
            grad_y: vec![0.0; n],
        }
    }
    pub fn allocate(n: usize) -> Self {
        Self::new(n)
    }

    #[inline]
    pub fn get(&self, i: usize) -> DVec2 {
        DVec2::new(self.grad_x[i], self.grad_y[i])
    }
    #[inline]
    pub fn set(&mut self, i: usize, g: DVec2) {
        self.grad_x[i] = g.x;
        self.grad_y[i] = g.y;
    }
    pub fn reset(&mut self) {
        self.grad_x.fill(0.0);
        self.grad_y.fill(0.0);
    }
    pub fn len(&self) -> usize {
        self.grad_x.len()
    }
    pub fn is_empty(&self) -> bool {
        self.grad_x.is_empty()
    }

    pub fn apply_limiter(&mut self, limiters: &[f64]) {
        for (i, &alpha) in limiters.iter().enumerate() {
            self.grad_x[i] *= alpha;
            self.grad_y[i] *= alpha;
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct VectorGradientStorage {
    pub du_dx: Vec<f64>,
    pub du_dy: Vec<f64>,
    pub dv_dx: Vec<f64>,
    pub dv_dy: Vec<f64>,
}

impl VectorGradientStorage {
    pub fn new(n: usize) -> Self {
        Self {
            du_dx: vec![0.0; n],
            du_dy: vec![0.0; n],
            dv_dx: vec![0.0; n],
            dv_dy: vec![0.0; n],
        }
    }
    pub fn allocate(n: usize) -> Self {
        Self::new(n)
    }

    #[inline]
    pub fn grad_u(&self, i: usize) -> DVec2 {
        DVec2::new(self.du_dx[i], self.du_dy[i])
    }
    #[inline]
    pub fn grad_v(&self, i: usize) -> DVec2 {
        DVec2::new(self.dv_dx[i], self.dv_dy[i])
    }
    #[inline]
    pub fn set_grad_u(&mut self, i: usize, g: DVec2) {
        self.du_dx[i] = g.x;
        self.du_dy[i] = g.y;
    }
    #[inline]
    pub fn set_grad_v(&mut self, i: usize, g: DVec2) {
        self.dv_dx[i] = g.x;
        self.dv_dy[i] = g.y;
    }
    pub fn reset(&mut self) {
        self.du_dx.fill(0.0);
        self.du_dy.fill(0.0);
        self.dv_dx.fill(0.0);
        self.dv_dy.fill(0.0);
    }
    pub fn len(&self) -> usize {
        self.du_dx.len()
    }

    /// 应变率张量模 |S| = √(2·S_ij·S_ij)
    #[inline]
    pub fn strain_rate_magnitude(&self, i: usize) -> f64 {
        let s11 = self.du_dx[i];
        let s22 = self.dv_dy[i];
        let s12 = 0.5 * (self.du_dy[i] + self.dv_dx[i]);
        (2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12)).sqrt()
    }

    #[inline]
    pub fn vorticity(&self, i: usize) -> f64 {
        self.dv_dx[i] - self.du_dy[i]
    }
    #[inline]
    pub fn divergence(&self, i: usize) -> f64 {
        self.du_dx[i] + self.dv_dy[i]
    }
}

pub trait GradientMethod: Send + Sync {
    fn compute_scalar_gradient<M: MeshAccess>(
        &self,
        field: &[f64],
        mesh: &M,
        output: &mut ScalarGradientStorage,
    ) -> MhResult<()>;

    fn compute_vector_gradient<M: MeshAccess>(
        &self,
        field: &[DVec2],
        mesh: &M,
        output: &mut VectorGradientStorage,
    ) -> MhResult<()>;

    fn name(&self) -> &'static str;
    fn supports_parallel(&self) -> bool {
        true
    }
}
