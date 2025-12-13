// crates/mh_core/src/backend.rs

//! 计算后端抽象
//!
//! 提供CPU/GPU统一的计算接口。

use crate::buffer::DeviceBuffer;
use crate::scalar::RuntimeScalar;
use bytemuck::Pod;
use std::fmt::Debug;
use std::marker::PhantomData;

/// 计算后端内存位置
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// 主机内存 (CPU)
    Host,
    /// 设备内存 (GPU)
    Device(usize),
}

/// 计算后端trait
///
/// 统一CPU/GPU内存和算子接口。所有方法使用`&self`实例方法，
/// 以支持GPU后端持有设备状态。
///
/// # 示例
///
/// ```ignore
/// use mh_core::{Backend, CpuBackend};
///
/// let backend = CpuBackend::<f64>::new();
/// let x = backend.alloc_init(100, 1.0);
/// let mut y = backend.alloc_init(100, 2.0);
/// backend.axpy(0.5, &x, &mut y);
/// ```
pub trait Backend: Clone + Send + Sync + Debug + 'static {
    /// 标量类型 (f32 或 f64)
    type Scalar: RuntimeScalar + Pod + Default;

    /// 设备缓冲区类型
    type Buffer<T: Pod + Clone + Send + Sync>: DeviceBuffer<T>;

    /// 后端名称
    fn name(&self) -> &'static str;

    /// 内存位置
    fn memory_location(&self) -> MemoryLocation;

    // ========== 内存分配 ==========

    /// 分配零初始化缓冲区
    fn alloc<T: Pod + Clone + Default + Send + Sync>(&self, len: usize) -> Self::Buffer<T>;

    /// 分配并初始化缓冲区
    fn alloc_init<T: Pod + Clone + Send + Sync>(&self, len: usize, init: T) -> Self::Buffer<T>;

    /// 同步操作（GPU需要等待流完成，CPU空实现）
    fn synchronize(&self) {}

    // ========== BLAS Level 1 算子 ==========

    /// y = alpha * x + y (AXPY)
    fn axpy(
        &self,
        alpha: Self::Scalar,
        x: &Self::Buffer<Self::Scalar>,
        y: &mut Self::Buffer<Self::Scalar>,
    );

    /// dot = x · y (点积)
    fn dot(
        &self,
        x: &Self::Buffer<Self::Scalar>,
        y: &Self::Buffer<Self::Scalar>,
    ) -> Self::Scalar;

    /// dst = src (复制)
    fn copy(
        &self,
        src: &Self::Buffer<Self::Scalar>,
        dst: &mut Self::Buffer<Self::Scalar>,
    );

    /// x = alpha * x (缩放)
    fn scale(&self, alpha: Self::Scalar, x: &mut Self::Buffer<Self::Scalar>);

    /// 归约：最大值
    fn reduce_max(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;

    /// 归约：最小值
    fn reduce_min(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;

    /// 归约：求和
    fn reduce_sum(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;

    // ========== 物理专用算子 ==========

    /// 确保正性：x[i] = max(x[i], min_val)
    fn enforce_positivity(&self, x: &mut Self::Buffer<Self::Scalar>, min_val: Self::Scalar);

    /// 分配未初始化缓冲区（性能优化，谨慎使用）
    /// 
    /// # Safety
    /// 调用者必须在使用前初始化所有元素。对于 Pod 类型，
    /// 此实现通常会分配容量然后设置长度。
    fn alloc_uninit<T: Pod + Send + Sync>(&self, len: usize) -> Self::Buffer<T>;

    /// L2 范数：sqrt(sum(x^2))
    fn norm2(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;

    /// 逐元素乘法：z[i] = x[i] * y[i]
    fn elementwise_mul(
        &self,
        x: &Self::Buffer<Self::Scalar>,
        y: &Self::Buffer<Self::Scalar>,
        z: &mut Self::Buffer<Self::Scalar>,
    );

    /// 逐元素安全除法：z[i] = x[i] / max(y[i], eps)
    fn elementwise_div_safe(
        &self,
        x: &Self::Buffer<Self::Scalar>,
        y: &Self::Buffer<Self::Scalar>,
        z: &mut Self::Buffer<Self::Scalar>,
        eps: Self::Scalar,
    );
}

/// CPU后端（泛型精度，无状态）
///
/// CpuBackend是无状态的零大小类型，实例化零开销。
#[derive(Debug, Clone, Default)]
pub struct CpuBackend<S: RuntimeScalar> {
    _marker: PhantomData<S>,
}

impl<S: RuntimeScalar> CpuBackend<S> {
    /// 创建CPU后端实例
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

// ============================================================================
// CpuBackend<f32> 实现
// ============================================================================

impl Backend for CpuBackend<f32> {
    type Scalar = f32;
    type Buffer<T: Pod + Clone + Send + Sync> = Vec<T>;

    fn name(&self) -> &'static str {
        "CPU-f32"
    }

    fn memory_location(&self) -> MemoryLocation {
        MemoryLocation::Host
    }

    fn alloc<T: Pod + Clone + Default + Send + Sync>(&self, len: usize) -> Self::Buffer<T> {
        vec![T::default(); len]
    }

    fn alloc_init<T: Pod + Clone + Send + Sync>(&self, len: usize, init: T) -> Self::Buffer<T> {
        vec![init; len]
    }

    fn axpy(
        &self,
        alpha: f32,
        x: &Self::Buffer<f32>,
        y: &mut Self::Buffer<f32>,
    ) {
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    }

    fn dot(&self, x: &Self::Buffer<f32>, y: &Self::Buffer<f32>) -> f32 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn copy(&self, src: &Self::Buffer<f32>, dst: &mut Self::Buffer<f32>) {
        dst.copy_from_slice(src);
    }

    fn scale(&self, alpha: f32, x: &mut Self::Buffer<f32>) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    fn reduce_max(&self, x: &Self::Buffer<f32>) -> f32 {
        x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    fn reduce_min(&self, x: &Self::Buffer<f32>) -> f32 {
        x.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    fn reduce_sum(&self, x: &Self::Buffer<f32>) -> f32 {
        x.iter().sum()
    }

    fn enforce_positivity(&self, x: &mut Self::Buffer<f32>, min_val: f32) {
        for xi in x.iter_mut() {
            if *xi < min_val {
                *xi = min_val;
            }
        }
    }

    fn alloc_uninit<T: Pod + Send + Sync>(&self, len: usize) -> Self::Buffer<T> {
        let mut v = Vec::with_capacity(len);
        // 安全：Pod 类型允许任意位模式
        #[allow(clippy::uninit_vec)]
        unsafe { v.set_len(len); }
        v
    }

    fn norm2(&self, x: &Self::Buffer<f32>) -> f32 {
        x.iter().map(|xi| xi * xi).sum::<f32>().sqrt()
    }

    fn elementwise_mul(
        &self,
        x: &Self::Buffer<f32>,
        y: &Self::Buffer<f32>,
        z: &mut Self::Buffer<f32>,
    ) {
        debug_assert_eq!(x.len(), y.len(), "逐元素乘法: 向量长度不匹配");
        debug_assert_eq!(x.len(), z.len(), "逐元素乘法: 输出向量长度不匹配");
        for ((xi, yi), zi) in x.iter().zip(y.iter()).zip(z.iter_mut()) {
            *zi = xi * yi;
        }
    }

    fn elementwise_div_safe(
        &self,
        x: &Self::Buffer<f32>,
        y: &Self::Buffer<f32>,
        z: &mut Self::Buffer<f32>,
        eps: f32,
    ) {
        debug_assert_eq!(x.len(), y.len(), "逐元素除法: 向量长度不匹配");
        debug_assert_eq!(x.len(), z.len(), "逐元素除法: 输出向量长度不匹配");
        for ((xi, yi), zi) in x.iter().zip(y.iter()).zip(z.iter_mut()) {
            *zi = xi / yi.max(eps);
        }
    }
}

// ============================================================================
// CpuBackend<f64> 实现
// ============================================================================

impl Backend for CpuBackend<f64> {
    type Scalar = f64;
    type Buffer<T: Pod + Clone + Send + Sync> = Vec<T>;

    fn name(&self) -> &'static str {
        "CPU-f64"
    }

    fn memory_location(&self) -> MemoryLocation {
        MemoryLocation::Host
    }

    fn alloc<T: Pod + Clone + Default + Send + Sync>(&self, len: usize) -> Self::Buffer<T> {
        vec![T::default(); len]
    }

    fn alloc_init<T: Pod + Clone + Send + Sync>(&self, len: usize, init: T) -> Self::Buffer<T> {
        vec![init; len]
    }

    fn axpy(
        &self,
        alpha: f64,
        x: &Self::Buffer<f64>,
        y: &mut Self::Buffer<f64>,
    ) {
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    }

    fn dot(&self, x: &Self::Buffer<f64>, y: &Self::Buffer<f64>) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn copy(&self, src: &Self::Buffer<f64>, dst: &mut Self::Buffer<f64>) {
        dst.copy_from_slice(src);
    }

    fn scale(&self, alpha: f64, x: &mut Self::Buffer<f64>) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }

    fn reduce_max(&self, x: &Self::Buffer<f64>) -> f64 {
        x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    fn reduce_min(&self, x: &Self::Buffer<f64>) -> f64 {
        x.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    fn reduce_sum(&self, x: &Self::Buffer<f64>) -> f64 {
        x.iter().sum()
    }

    fn enforce_positivity(&self, x: &mut Self::Buffer<f64>, min_val: f64) {
        for xi in x.iter_mut() {
            if *xi < min_val {
                *xi = min_val;
            }
        }
    }

    fn alloc_uninit<T: Pod + Send + Sync>(&self, len: usize) -> Self::Buffer<T> {
        let mut v = Vec::with_capacity(len);
        // 安全：Pod 类型允许任意位模式
        #[allow(clippy::uninit_vec)]
        unsafe { v.set_len(len); }
        v
    }

    fn norm2(&self, x: &Self::Buffer<f64>) -> f64 {
        x.iter().map(|xi| xi * xi).sum::<f64>().sqrt()
    }

    fn elementwise_mul(
        &self,
        x: &Self::Buffer<f64>,
        y: &Self::Buffer<f64>,
        z: &mut Self::Buffer<f64>,
    ) {
        debug_assert_eq!(x.len(), y.len(), "逐元素乘法: 向量长度不匹配");
        debug_assert_eq!(x.len(), z.len(), "逐元素乘法: 输出向量长度不匹配");
        for ((xi, yi), zi) in x.iter().zip(y.iter()).zip(z.iter_mut()) {
            *zi = xi * yi;
        }
    }

    fn elementwise_div_safe(
        &self,
        x: &Self::Buffer<f64>,
        y: &Self::Buffer<f64>,
        z: &mut Self::Buffer<f64>,
        eps: f64,
    ) {
        debug_assert_eq!(x.len(), y.len(), "逐元素除法: 向量长度不匹配");
        debug_assert_eq!(x.len(), z.len(), "逐元素除法: 输出向量长度不匹配");
        for ((xi, yi), zi) in x.iter().zip(y.iter()).zip(z.iter_mut()) {
            *zi = xi / yi.max(eps);
        }
    }
}

/// 默认后端类型别名
#[cfg(feature = "precision-f64")]
pub type DefaultBackend = CpuBackend<f64>;

#[cfg(all(feature = "precision-f32", not(feature = "precision-f64")))]
pub type DefaultBackend = CpuBackend<f32>;

// 当没有启用任何精度feature时，默认使用f64
#[cfg(not(any(feature = "precision-f32", feature = "precision-f64")))]
pub type DefaultBackend = CpuBackend<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_f64() {
        let backend = CpuBackend::<f64>::new();
        assert_eq!(backend.name(), "CPU-f64");

        let x = backend.alloc_init(10, 1.0);
        let mut y = backend.alloc_init(10, 2.0);

        backend.axpy(0.5, &x, &mut y);
        assert!((y[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_backend_f32() {
        let backend = CpuBackend::<f32>::new();
        assert_eq!(backend.name(), "CPU-f32");

        let x = backend.alloc_init(10, 1.0f32);
        let y = backend.alloc_init(10, 2.0f32);

        let dot = backend.dot(&x, &y);
        assert!((dot - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_ops() {
        let backend = CpuBackend::<f64>::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(backend.reduce_max(&x), 5.0);
        assert_eq!(backend.reduce_min(&x), 1.0);
        assert_eq!(backend.reduce_sum(&x), 15.0);
    }

    #[test]
    fn test_enforce_positivity() {
        let backend = CpuBackend::<f64>::new();
        let mut x = vec![-1.0, 0.0, 1.0, -2.0];

        backend.enforce_positivity(&mut x, 0.0);
        assert_eq!(x, vec![0.0, 0.0, 1.0, 0.0]);
    }
}
