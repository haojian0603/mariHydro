// crates/mh_physics/src/numerics/gradient/traits.rs

//! 梯度计算 trait 和存储类型
//!
//! **层级**: Layer 3 - Engine Layer
//!
//! 本模块提供泛型化的梯度存储和计算接口，支持 f32/f64 精度切换。

use glam::DVec2;
use mh_runtime::RuntimeScalar;

// ============================================================
// 泛型梯度存储
// ============================================================

/// 标量场梯度存储 (SoA布局) - 泛型版本
#[derive(Debug, Clone, Default)]
pub struct ScalarGradientStorageGeneric<S: RuntimeScalar> {
    /// x方向梯度分量
    pub grad_x: Vec<S>,
    /// y方向梯度分量
    pub grad_y: Vec<S>,
}

/// 标量场梯度存储 (SoA布局) - Legacy f64 版本
pub type ScalarGradientStorage = ScalarGradientStorageGeneric<f64>;

impl<S: RuntimeScalar> ScalarGradientStorageGeneric<S> {
    /// 创建指定大小的存储
    pub fn new(n: usize) -> Self {
        Self {
            grad_x: vec![S::ZERO; n],
            grad_y: vec![S::ZERO; n],
        }
    }

    /// 获取单元梯度 (返回元组)
    #[inline]
    pub fn get_tuple(&self, i: usize) -> (S, S) {
        (self.grad_x[i], self.grad_y[i])
    }

    /// 设置单元梯度 (从元组)
    #[inline]
    pub fn set_tuple(&mut self, i: usize, g: (S, S)) {
        self.grad_x[i] = g.0;
        self.grad_y[i] = g.1;
    }

    /// 重置所有梯度为零
    pub fn reset(&mut self) {
        self.grad_x.fill(S::ZERO);
        self.grad_y.fill(S::ZERO);
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
        self.grad_x.resize(n, S::ZERO);
        self.grad_y.resize(n, S::ZERO);
    }

    /// 应用限制器 (梯度乘以限制因子)
    pub fn apply_limiter(&mut self, limiters: &[S]) {
        for (i, &alpha) in limiters.iter().enumerate() {
            self.grad_x[i] = self.grad_x[i] * alpha;
            self.grad_y[i] = self.grad_y[i] * alpha;
        }
    }
}

// f64 版本的 DVec2 兼容方法
impl ScalarGradientStorageGeneric<f64> {
    /// 获取单元梯度 (DVec2 版本，仅 f64)
    #[inline]
    pub fn get(&self, i: usize) -> DVec2 {
        DVec2::new(self.grad_x[i], self.grad_y[i])
    }

    /// 设置单元梯度 (DVec2 版本，仅 f64)
    #[inline]
    pub fn set(&mut self, i: usize, g: DVec2) {
        self.grad_x[i] = g.x;
        self.grad_y[i] = g.y;
    }
}

/// 向量场梯度存储 (速度梯度张量) - 泛型版本
///
/// 存储 ∇u 和 ∇v:
/// ```text
/// ┌ du/dx  du/dy ┐
/// │              │
/// └ dv/dx  dv/dy ┘
/// ```
#[derive(Debug, Clone, Default)]
pub struct VectorGradientStorageGeneric<S: RuntimeScalar> {
    /// ∂u/∂x
    pub du_dx: Vec<S>,
    /// ∂u/∂y
    pub du_dy: Vec<S>,
    /// ∂v/∂x
    pub dv_dx: Vec<S>,
    /// ∂v/∂y
    pub dv_dy: Vec<S>,
}

/// 向量场梯度存储 - Legacy f64 版本
pub type VectorGradientStorage = VectorGradientStorageGeneric<f64>;

impl<S: RuntimeScalar> VectorGradientStorageGeneric<S> {
    /// 创建指定大小的存储
    pub fn new(n: usize) -> Self {
        Self {
            du_dx: vec![S::ZERO; n],
            du_dy: vec![S::ZERO; n],
            dv_dx: vec![S::ZERO; n],
            dv_dy: vec![S::ZERO; n],
        }
    }

    /// 获取 u 的梯度 (元组版本)
    #[inline]
    pub fn grad_u_tuple(&self, i: usize) -> (S, S) {
        (self.du_dx[i], self.du_dy[i])
    }

    /// 获取 v 的梯度 (元组版本)
    #[inline]
    pub fn grad_v_tuple(&self, i: usize) -> (S, S) {
        (self.dv_dx[i], self.dv_dy[i])
    }

    /// 设置 u 的梯度 (元组版本)
    #[inline]
    pub fn set_grad_u_tuple(&mut self, i: usize, g: (S, S)) {
        self.du_dx[i] = g.0;
        self.du_dy[i] = g.1;
    }

    /// 设置 v 的梯度 (元组版本)
    #[inline]
    pub fn set_grad_v_tuple(&mut self, i: usize, g: (S, S)) {
        self.dv_dx[i] = g.0;
        self.dv_dy[i] = g.1;
    }

    /// 重置所有梯度为零
    pub fn reset(&mut self) {
        self.du_dx.fill(S::ZERO);
        self.du_dy.fill(S::ZERO);
        self.dv_dx.fill(S::ZERO);
        self.dv_dy.fill(S::ZERO);
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
        self.du_dx.resize(n, S::ZERO);
        self.du_dy.resize(n, S::ZERO);
        self.dv_dx.resize(n, S::ZERO);
        self.dv_dy.resize(n, S::ZERO);
    }

    /// 应变率张量模 |S| = √(2·S_ij·S_ij)
    ///
    /// 用于湍流模型的应变率计算
    #[inline]
    pub fn strain_rate_magnitude(&self, i: usize) -> S {
        let s11 = self.du_dx[i];
        let s22 = self.dv_dy[i];
        let s12 = S::HALF * (self.du_dy[i] + self.dv_dx[i]);
        (S::TWO * (s11 * s11 + s22 * s22 + S::TWO * s12 * s12)).sqrt()
    }

    /// 涡量 (2D): ω = ∂v/∂x - ∂u/∂y
    #[inline]
    pub fn vorticity(&self, i: usize) -> S {
        self.dv_dx[i] - self.du_dy[i]
    }

    /// 散度: ∇·v = ∂u/∂x + ∂v/∂y
    #[inline]
    pub fn divergence(&self, i: usize) -> S {
        self.du_dx[i] + self.dv_dy[i]
    }
}

// f64 版本的 DVec2 兼容方法
impl VectorGradientStorageGeneric<f64> {
    /// 获取 u 的梯度 (DVec2 版本，仅 f64)
    #[inline]
    pub fn grad_u(&self, i: usize) -> DVec2 {
        DVec2::new(self.du_dx[i], self.du_dy[i])
    }

    /// 获取 v 的梯度 (DVec2 版本，仅 f64)
    #[inline]
    pub fn grad_v(&self, i: usize) -> DVec2 {
        DVec2::new(self.dv_dx[i], self.dv_dy[i])
    }

    /// 设置 u 的梯度 (DVec2 版本，仅 f64)
    #[inline]
    pub fn set_grad_u(&mut self, i: usize, g: DVec2) {
        self.du_dx[i] = g.x;
        self.du_dy[i] = g.y;
    }

    /// 设置 v 的梯度 (DVec2 版本，仅 f64)
    #[inline]
    pub fn set_grad_v(&mut self, i: usize, g: DVec2) {
        self.dv_dx[i] = g.x;
        self.dv_dy[i] = g.y;
    }
}

// ============================================================
// 梯度方法 Trait
// ============================================================

use crate::adapter::PhysicsMesh;

/// 泛型梯度计算方法 trait
pub trait GradientMethodGeneric<S: RuntimeScalar>: Send + Sync {
    /// 计算标量场梯度
    fn compute_scalar_gradient(
        &self,
        field: &[S],
        mesh: &PhysicsMesh,
        output: &mut ScalarGradientStorageGeneric<S>,
    );

    /// 计算向量场梯度
    fn compute_vector_gradient(
        &self,
        field_u: &[S],
        field_v: &[S],
        mesh: &PhysicsMesh,
        output: &mut VectorGradientStorageGeneric<S>,
    );

    /// 方法名称
    fn name(&self) -> &'static str;

    /// 是否支持并行
    fn supports_parallel(&self) -> bool {
        true
    }
}

/// 梯度计算方法 trait - f64 版本别名
pub trait GradientMethod: GradientMethodGeneric<f64> {}

/// 为所有实现 GradientMethodGeneric<f64> 的类型自动实现 GradientMethod
impl<T: GradientMethodGeneric<f64>> GradientMethod for T {}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_gradient_storage() {
        let mut storage = ScalarGradientStorage::new(4);
        assert_eq!(storage.len(), 4);

        storage.set(0, DVec2::new(1.0, 2.0));
        storage.set(1, DVec2::new(3.0, 4.0));

        assert!((storage.get(0).x - 1.0).abs() < 1e-10);
        assert!((storage.get(0).y - 2.0).abs() < 1e-10);

        storage.reset();
        assert!(storage.get(0).length() < 1e-10);
    }

    #[test]
    fn test_vector_gradient_storage() {
        let mut storage = VectorGradientStorage::new(4);
        assert_eq!(storage.len(), 4);

        storage.set_grad_u(0, DVec2::new(1.0, 0.0));
        storage.set_grad_v(0, DVec2::new(0.0, 1.0));

        // 测试散度: div = du/dx + dv/dy = 1 + 1 = 2
        assert!((storage.divergence(0) - 2.0).abs() < 1e-10);

        // 测试涡量: omega = dv/dx - du/dy = 0 - 0 = 0
        assert!(storage.vorticity(0).abs() < 1e-10);
    }

    #[test]
    fn test_strain_rate() {
        let mut storage = VectorGradientStorage::new(1);
        
        // 纯剪切流: du/dy = 1, 其他为0
        storage.du_dy[0] = 1.0;
        
        // |S| = sqrt(2 * 2 * s12^2) = sqrt(2 * 2 * 0.25) = 1
        // s12 = 0.5 * (du/dy + dv/dx) = 0.5
        let s = storage.strain_rate_magnitude(0);
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_limiter() {
        let mut storage = ScalarGradientStorage::new(3);
        storage.set(0, DVec2::new(2.0, 4.0));
        storage.set(1, DVec2::new(6.0, 8.0));
        storage.set(2, DVec2::new(1.0, 1.0));

        let limiters = vec![0.5, 0.25, 1.0];
        storage.apply_limiter(&limiters);

        assert!((storage.get(0).x - 1.0).abs() < 1e-10);
        assert!((storage.get(0).y - 2.0).abs() < 1e-10);
        assert!((storage.get(1).x - 1.5).abs() < 1e-10);
        assert!((storage.get(2).x - 1.0).abs() < 1e-10);
    }
}
