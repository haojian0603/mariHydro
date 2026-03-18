// crates/mh_physics/src/numerics/gradient/traits.rs

//! 梯度计算 trait 和存储类型
//!
//! **层级**: Layer 3 - Engine Layer
//!
//! 本模块提供泛型化的梯度存储和计算接口，支持 f32/f64 精度切换。
//!
//! # 设计原则
//!
//! 1. **单轨泛型**: 所有接口基于 `RuntimeScalar` 泛型，无 Legacy f64 别名
//! 2. **Backend 无关**: 使用 `[S; 2]` 元组表示向量，不依赖 glam::DVec2

use crate::prelude::*;

// ============================================================
// 泛型梯度存储
// ============================================================


/// 标量场梯度存储 (SoA布局) - 泛型版本
#[derive(Debug, Clone)]
pub struct ScalarGradientStorage<B: Backend> {
    /// x方向梯度分量
    pub grad_x: B::Buffer<B::Scalar>,
    /// y方向梯度分量
    pub grad_y: B::Buffer<B::Scalar>,
}

impl<B: Backend> ScalarGradientStorage<B> {
    /// 使用后端创建指定大小的存储
    pub fn with_backend(backend: &B, n: usize) -> Self {
        let mut grad_x = backend.alloc(n);
        let mut grad_y = backend.alloc(n);
        grad_x.fill(B::Scalar::ZERO);
        grad_y.fill(B::Scalar::ZERO);
        Self { grad_x, grad_y }
    }

    /// 获取单元梯度 (返回元组)
    #[inline]
    pub fn get_tuple(&self, i: usize) -> (B::Scalar, B::Scalar) {
        (self.grad_x[i], self.grad_y[i])
    }

    /// 设置单元梯度 (从元组)
    #[inline]
    pub fn set_tuple(&mut self, i: usize, g: (B::Scalar, B::Scalar)) {
        self.grad_x[i] = g.0;
        self.grad_y[i] = g.1;
    }

    /// 重置所有梯度为零
    pub fn reset(&mut self) {
        self.grad_x.fill(B::Scalar::ZERO);
        self.grad_y.fill(B::Scalar::ZERO);
    }

    /// 存储大小
    pub fn len(&self) -> usize {
        self.grad_x.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.grad_x.is_empty()
    }

    /// 调整大小
    pub fn resize(&mut self, n: usize) {
        self.grad_x.resize(n, B::Scalar::ZERO);
        self.grad_y.resize(n, B::Scalar::ZERO);
    }

    /// 应用限制器 (梯度乘以限制因子)
    pub fn apply_limiter(&mut self, limiters: &B::Buffer<B::Scalar>) {
        let limiter_slice = limiters.as_slice();
        for (i, &alpha) in limiter_slice.iter().enumerate() {
            self.grad_x[i] = self.grad_x[i] * alpha;
            self.grad_y[i] = self.grad_y[i] * alpha;
        }
    }
}

// 注意：DVec2 兼容方法已删除
// 请使用 get_tuple() 和 set_tuple() 方法

/// 向量场梯度存储 (速度梯度张量) - 泛型版本
///
/// 存储 ∇u 和 ∇v:
/// ```text
/// ┌ du/dx  du/dy ┐
/// │              │
/// └ dv/dx  dv/dy ┘
/// ```
#[derive(Debug, Clone)]
pub struct VectorGradientStorage<B: Backend> {
    /// ∂u/∂x
    pub du_dx: B::Buffer<B::Scalar>,
    /// ∂u/∂y
    pub du_dy: B::Buffer<B::Scalar>,
    /// ∂v/∂x
    pub dv_dx: B::Buffer<B::Scalar>,
    /// ∂v/∂y
    pub dv_dy: B::Buffer<B::Scalar>,
}

impl<B: Backend> VectorGradientStorage<B> {
    /// 使用后端创建指定大小的存储
    pub fn with_backend(backend: &B, n: usize) -> Self {
        let mut du_dx = backend.alloc(n);
        let mut du_dy = backend.alloc(n);
        let mut dv_dx = backend.alloc(n);
        let mut dv_dy = backend.alloc(n);
        du_dx.fill(B::Scalar::ZERO);
        du_dy.fill(B::Scalar::ZERO);
        dv_dx.fill(B::Scalar::ZERO);
        dv_dy.fill(B::Scalar::ZERO);
        Self {
            du_dx,
            du_dy,
            dv_dx,
            dv_dy,
        }
    }

    /// 获取 u 的梯度 (元组版本)
    #[inline]
    pub fn grad_u_tuple(&self, i: usize) -> (B::Scalar, B::Scalar) {
        (self.du_dx[i], self.du_dy[i])
    }

    /// 获取 v 的梯度 (元组版本)
    #[inline]
    pub fn grad_v_tuple(&self, i: usize) -> (B::Scalar, B::Scalar) {
        (self.dv_dx[i], self.dv_dy[i])
    }

    /// 设置 u 的梯度 (元组版本)
    #[inline]
    pub fn set_grad_u_tuple(&mut self, i: usize, g: (B::Scalar, B::Scalar)) {
        self.du_dx[i] = g.0;
        self.du_dy[i] = g.1;
    }

    /// 设置 v 的梯度 (元组版本)
    #[inline]
    pub fn set_grad_v_tuple(&mut self, i: usize, g: (B::Scalar, B::Scalar)) {
        self.dv_dx[i] = g.0;
        self.dv_dy[i] = g.1;
    }

    /// 重置所有梯度为零
    pub fn reset(&mut self) {
        self.du_dx.fill(B::Scalar::ZERO);
        self.du_dy.fill(B::Scalar::ZERO);
        self.dv_dx.fill(B::Scalar::ZERO);
        self.dv_dy.fill(B::Scalar::ZERO);
    }

    /// 存储大小
    pub fn len(&self) -> usize {
        self.du_dx.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.du_dx.is_empty()
    }

    /// 调整大小
    pub fn resize(&mut self, n: usize) {
        self.du_dx.resize(n, B::Scalar::ZERO);
        self.du_dy.resize(n, B::Scalar::ZERO);
        self.dv_dx.resize(n, B::Scalar::ZERO);
        self.dv_dy.resize(n, B::Scalar::ZERO);
    }

    /// 应变率张量模 |S| = √(2·S_ij·S_ij)
    ///
    /// 用于湍流模型的应变率计算
    #[inline]
    pub fn strain_rate_magnitude(&self, i: usize) -> B::Scalar {
        let s11 = self.du_dx[i];
        let s22 = self.dv_dy[i];
        let s12 = B::Scalar::HALF * (self.du_dy[i] + self.dv_dx[i]);
        (B::Scalar::TWO * (s11 * s11 + s22 * s22 + B::Scalar::TWO * s12 * s12)).sqrt()
    }

    /// 涡量 (2D): ω = ∂v/∂x - ∂u/∂y
    #[inline]
    pub fn vorticity(&self, i: usize) -> B::Scalar {
        self.dv_dx[i] - self.du_dy[i]
    }

    /// 散度: ∇·v = ∂u/∂x + ∂v/∂y
    #[inline]
    pub fn divergence(&self, i: usize) -> B::Scalar {
        self.du_dx[i] + self.dv_dy[i]
    }
}

// 注意：DVec2 兼容方法已删除
// 请使用 grad_u_tuple() / grad_v_tuple() 和 set_grad_u_tuple() / set_grad_v_tuple() 方法

// ============================================================
// 梯度方法 Trait
// ============================================================

use crate::adapter::PhysicsMesh;

// ============================================================
// 梯度方法 Trait（强制显式类型标注）
// ============================================================

/// 泛型梯度计算方法 trait
/// 
/// 所有实现者必须显式实现 `supports_parallel()` 方法，不提供默认实现。
pub trait GradientMethod<B: Backend>: Send + Sync {
    /// 计算标量场梯度
    fn compute_scalar_gradient(
        &self,
        backend: &B,
        field: &B::Buffer<B::Scalar>,
        mesh: &PhysicsMesh,
        output: &mut ScalarGradientStorage<B>,
    );

    /// 计算向量场梯度
    fn compute_vector_gradient(
        &self,
        backend: &B,
        field_u: &B::Buffer<B::Scalar>,
        field_v: &B::Buffer<B::Scalar>,
        mesh: &PhysicsMesh,
        output: &mut VectorGradientStorage<B>,
    );

    /// 方法名称
    fn name(&self) -> &'static str;

    /// 🔥 删除默认实现，强制所有实现者必须指定 Backend 类型
    fn supports_parallel(&self) -> bool;
}


// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_scalar_gradient_storage() {
        let backend = CpuBackend::<f64>::new();
        let mut storage = ScalarGradientStorage::with_backend(&backend, 4);
        assert_eq!(storage.len(), 4);

        storage.set_tuple(0, (1.0, 2.0));
        storage.set_tuple(1, (3.0, 4.0));

        let (x, y) = storage.get_tuple(0);
        assert!((x - 1.0).abs() < 1e-10);
        assert!((y - 2.0).abs() < 1e-10);

        storage.reset();
        let (x, y) = storage.get_tuple(0);
        assert!(x.abs() < 1e-10);
        assert!(y.abs() < 1e-10);
    }

    #[test]
    fn test_vector_gradient_storage() {
        let backend = CpuBackend::<f64>::new();
        let mut storage = VectorGradientStorage::with_backend(&backend, 4);
        assert_eq!(storage.len(), 4);

        storage.set_grad_u_tuple(0, (1.0, 0.0));
        storage.set_grad_v_tuple(0, (0.0, 1.0));

        // 测试散度: div = du/dx + dv/dy = 1 + 1 = 2
        assert!((storage.divergence(0) - 2.0).abs() < 1e-10);

        // 测试涡量: omega = dv/dx - du/dy = 0 - 0 = 0
        assert!(storage.vorticity(0).abs() < 1e-10);
    }

    #[test]
    fn test_strain_rate() {
        let backend = CpuBackend::<f64>::new();
        let mut storage = VectorGradientStorage::with_backend(&backend, 1);
        
        // 纯剪切流: du/dy = 1, 其他为0
        storage.du_dy[0] = 1.0;
        
        // |S| = sqrt(2 * 2 * s12^2) = sqrt(2 * 2 * 0.25) = 1
        // s12 = 0.5 * (du/dy + dv/dx) = 0.5
        let s = storage.strain_rate_magnitude(0);
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_limiter() {
        let backend = CpuBackend::<f64>::new();
        let mut storage = ScalarGradientStorage::with_backend(&backend, 3);
        storage.set_tuple(0, (2.0, 4.0));
        storage.set_tuple(1, (6.0, 8.0));
        storage.set_tuple(2, (1.0, 1.0));

        let mut limiters = backend.alloc(3);
        limiters.copy_from_slice(&[0.5, 0.25, 1.0]);
        storage.apply_limiter(&limiters);

        let (x0, y0) = storage.get_tuple(0);
        let (x1, _y1) = storage.get_tuple(1);
        let (x2, _y2) = storage.get_tuple(2);
        
        assert!((x0 - 1.0).abs() < 1e-10);
        assert!((y0 - 2.0).abs() < 1e-10);
        assert!((x1 - 1.5).abs() < 1e-10);
        assert!((x2 - 1.0).abs() < 1e-10);
    }
}
