// marihydro\crates\mh_physics\src\core\backend.rs
//! 计算后端抽象
//!
//! 提供 CPU/GPU 统一的计算接口。Backend trait 使用实例方法设计，
//! 以支持 GPU 后端持有设备句柄和流。
//!
//! # 设计原则
//!
//! 1. **实例方法**: 所有操作通过 `&self` 调用，GPU 后端可持有 CudaDevice/Stream
//! 2. **零开销 CPU**: CpuBackend 是无状态的，实例化零开销
//! 3. **类型安全**: 缓冲区类型与后端绑定，防止跨后端误用
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::core::{Backend, CpuBackend};
//!
//! let backend = CpuBackend::<f64>::new();
//! let x = backend.alloc_init(100, 1.0);
//! let mut y = backend.alloc_init(100, 2.0);
//! backend.axpy(0.5, &x, &mut y);
//! assert!((y[0] - 2.5).abs() < 1e-10);
//! ```

use super::buffer::DeviceBuffer;
use super::scalar::Scalar;
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

/// 计算后端 trait - 统一 CPU/GPU 内存和算子接口
///
/// 所有方法使用 `&self` 实例方法，以支持 GPU 后端持有设备状态。
pub trait Backend: Clone + Send + Sync + Debug + 'static {
    /// 标量类型 (f32 或 f64)
    /// 注意：添加 Pod + Default 约束以支持 Buffer 操作
    type Scalar: Scalar + Pod + Default;
    
    /// 设备缓冲区类型
    type Buffer<T: Pod + Send + Sync>: DeviceBuffer<T>;
    
    /// 后端名称
    fn name(&self) -> &'static str;
    
    /// 内存位置
    fn memory_location(&self) -> MemoryLocation;
    
    // ========== 内存分配（实例方法）==========
    
    /// 分配零初始化缓冲区
    fn alloc<T: Pod + Clone + Default + Send + Sync>(&self, len: usize) -> Self::Buffer<T>;
    
    /// 分配并初始化缓冲区
    fn alloc_init<T: Pod + Clone + Send + Sync>(&self, len: usize, init: T) -> Self::Buffer<T>;
    
    /// 分配未初始化缓冲区（性能优化，谨慎使用）
    fn alloc_uninit<T: Pod + Send + Sync>(&self, len: usize) -> Self::Buffer<T>;
    
    /// 同步操作（GPU 需要等待流完成，CPU 空实现）
    fn synchronize(&self) {}
    
    // ========== BLAS Level 1 算子（实例方法）==========
    
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
    
    /// L2 范数：sqrt(sum(x^2))
    fn norm2(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    // ========== 物理专用算子 ==========
    
    /// 确保正性：x[i] = max(x[i], min_val)
    fn enforce_positivity(&self, x: &mut Self::Buffer<Self::Scalar>, min_val: Self::Scalar);
    
    /// 逐元素乘法：z[i] = x[i] * y[i]
    fn elementwise_mul(
        &self,
        x: &Self::Buffer<Self::Scalar>,
        y: &Self::Buffer<Self::Scalar>,
        z: &mut Self::Buffer<Self::Scalar>,
    );
    
    /// 逐元素除法（安全）：z[i] = x[i] / max(y[i], eps)
    fn elementwise_div_safe(
        &self,
        x: &Self::Buffer<Self::Scalar>,
        y: &Self::Buffer<Self::Scalar>,
        z: &mut Self::Buffer<Self::Scalar>,
        eps: Self::Scalar,
    );
}

/// CPU 后端（泛型精度，无状态）
///
/// CpuBackend 是无状态的零大小类型，实例化零开销。
/// 所有计算直接在 CPU 上执行，使用 Vec<T> 作为缓冲区。
#[derive(Debug, Clone, Default)]
pub struct CpuBackend<S: Scalar> {
    _marker: PhantomData<S>,
}

impl<S: Scalar> CpuBackend<S> {
    /// 创建 CPU 后端实例
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl Backend for CpuBackend<f32> {
    type Scalar = f32;
    type Buffer<T: Pod + Send + Sync> = Vec<T>;
    
    fn name(&self) -> &'static str { "CPU-f32" }
    
    fn memory_location(&self) -> MemoryLocation { MemoryLocation::Host }
    
    fn alloc<T: Pod + Clone + Default + Send + Sync>(&self, len: usize) -> Self::Buffer<T> {
        vec![T::default(); len]
    }
    
    fn alloc_init<T: Pod + Clone + Send + Sync>(&self, len: usize, init: T) -> Self::Buffer<T> {
        vec![init; len]
    }
    
    fn alloc_uninit<T: Pod + Send + Sync>(&self, len: usize) -> Self::Buffer<T> {
        let mut v = Vec::with_capacity(len);
        // 安全：Pod 类型允许未初始化，且我们立即初始化为零字节
        #[allow(clippy::uninit_vec)]
        unsafe { v.set_len(len); }
        v
    }
    
    fn axpy(&self, alpha: f32, x: &Vec<f32>, y: &mut Vec<f32>) {
        debug_assert_eq!(x.len(), y.len(), "AXPY: 向量长度不匹配");
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    }
    
    fn dot(&self, x: &Vec<f32>, y: &Vec<f32>) -> f32 {
        debug_assert_eq!(x.len(), y.len(), "DOT: 向量长度不匹配");
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
    
    fn copy(&self, src: &Vec<f32>, dst: &mut Vec<f32>) {
        debug_assert_eq!(src.len(), dst.len(), "COPY: 向量长度不匹配");
        dst.copy_from_slice(src);
    }
    
    fn scale(&self, alpha: f32, x: &mut Vec<f32>) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }
    
    fn reduce_max(&self, x: &Vec<f32>) -> f32 {
        x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }
    
    fn reduce_min(&self, x: &Vec<f32>) -> f32 {
        x.iter().cloned().fold(f32::INFINITY, f32::min)
    }
    
    fn reduce_sum(&self, x: &Vec<f32>) -> f32 {
        x.iter().sum()
    }
    
    fn norm2(&self, x: &Vec<f32>) -> f32 {
        x.iter().map(|xi| xi * xi).sum::<f32>().sqrt()
    }
    
    fn enforce_positivity(&self, x: &mut Vec<f32>, min_val: f32) {
        for xi in x.iter_mut() {
            *xi = xi.max(min_val);
        }
    }
    
    fn elementwise_mul(&self, x: &Vec<f32>, y: &Vec<f32>, z: &mut Vec<f32>) {
        debug_assert_eq!(x.len(), y.len(), "逐元素乘法: 向量长度不匹配");
        debug_assert_eq!(x.len(), z.len(), "逐元素乘法: 输出向量长度不匹配");
        for ((xi, yi), zi) in x.iter().zip(y.iter()).zip(z.iter_mut()) {
            *zi = xi * yi;
        }
    }
    
    fn elementwise_div_safe(&self, x: &Vec<f32>, y: &Vec<f32>, z: &mut Vec<f32>, eps: f32) {
        debug_assert_eq!(x.len(), y.len(), "逐元素除法: 向量长度不匹配");
        debug_assert_eq!(x.len(), z.len(), "逐元素除法: 输出向量长度不匹配");
        for ((xi, yi), zi) in x.iter().zip(y.iter()).zip(z.iter_mut()) {
            *zi = xi / yi.max(eps);
        }
    }
}

impl Backend for CpuBackend<f64> {
    type Scalar = f64;
    type Buffer<T: Pod + Send + Sync> = Vec<T>;
    
    fn name(&self) -> &'static str { "CPU-f64" }
    
    fn memory_location(&self) -> MemoryLocation { MemoryLocation::Host }
    
    fn alloc<T: Pod + Clone + Default + Send + Sync>(&self, len: usize) -> Self::Buffer<T> {
        vec![T::default(); len]
    }
    
    fn alloc_init<T: Pod + Clone + Send + Sync>(&self, len: usize, init: T) -> Self::Buffer<T> {
        vec![init; len]
    }
    
    fn alloc_uninit<T: Pod + Send + Sync>(&self, len: usize) -> Self::Buffer<T> {
        let mut v = Vec::with_capacity(len);
        // 安全：Pod 类型允许未初始化，且我们立即初始化为零字节
        #[allow(clippy::uninit_vec)]
        unsafe { v.set_len(len); }
        v
    }
    
    fn axpy(&self, alpha: f64, x: &Vec<f64>, y: &mut Vec<f64>) {
        debug_assert_eq!(x.len(), y.len(), "AXPY: 向量长度不匹配");
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    }
    
    fn dot(&self, x: &Vec<f64>, y: &Vec<f64>) -> f64 {
        debug_assert_eq!(x.len(), y.len(), "DOT: 向量长度不匹配");
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
    
    fn copy(&self, src: &Vec<f64>, dst: &mut Vec<f64>) {
        debug_assert_eq!(src.len(), dst.len(), "COPY: 向量长度不匹配");
        dst.copy_from_slice(src);
    }
    
    fn scale(&self, alpha: f64, x: &mut Vec<f64>) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }
    
    fn reduce_max(&self, x: &Vec<f64>) -> f64 {
        x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
    
    fn reduce_min(&self, x: &Vec<f64>) -> f64 {
        x.iter().cloned().fold(f64::INFINITY, f64::min)
    }
    
    fn reduce_sum(&self, x: &Vec<f64>) -> f64 {
        x.iter().sum()
    }
    
    fn norm2(&self, x: &Vec<f64>) -> f64 {
        x.iter().map(|xi| xi * xi).sum::<f64>().sqrt()
    }
    
    fn enforce_positivity(&self, x: &mut Vec<f64>, min_val: f64) {
        for xi in x.iter_mut() {
            *xi = xi.max(min_val);
        }
    }
    
    fn elementwise_mul(&self, x: &Vec<f64>, y: &Vec<f64>, z: &mut Vec<f64>) {
        debug_assert_eq!(x.len(), y.len(), "逐元素乘法: 向量长度不匹配");
        debug_assert_eq!(x.len(), z.len(), "逐元素乘法: 输出向量长度不匹配");
        for ((xi, yi), zi) in x.iter().zip(y.iter()).zip(z.iter_mut()) {
            *zi = xi * yi;
        }
    }
    
    fn elementwise_div_safe(&self, x: &Vec<f64>, y: &Vec<f64>, z: &mut Vec<f64>, eps: f64) {
        debug_assert_eq!(x.len(), y.len(), "逐元素除法: 向量长度不匹配");
        debug_assert_eq!(x.len(), z.len(), "逐元素除法: 输出向量长度不匹配");
        for ((xi, yi), zi) in x.iter().zip(y.iter()).zip(z.iter_mut()) {
            *zi = xi / yi.max(eps);
        }
    }
}

/// 类型别名：默认后端
pub type DefaultBackend = CpuBackend<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_f64_axpy() {
        let backend = CpuBackend::<f64>::new();
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut y: Vec<f64> = vec![4.0, 5.0, 6.0];
        backend.axpy(2.0, &x, &mut y);
        assert_eq!(y, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_cpu_backend_f64_dot() {
        let backend = CpuBackend::<f64>::new();
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![4.0, 5.0, 6.0];
        let result = backend.dot(&x, &y);
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_backend_f32_copy() {
        let backend = CpuBackend::<f32>::new();
        let x: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut y: Vec<f32> = vec![0.0, 0.0, 0.0];
        backend.copy(&x, &mut y);
        assert_eq!(y, x);
    }
    
    #[test]
    fn test_cpu_backend_alloc() {
        let backend = CpuBackend::<f64>::new();
        let zeros: Vec<f64> = backend.alloc(5);
        assert_eq!(zeros.len(), 5);
        assert!(zeros.iter().all(|&x| x == 0.0));
        
        let inited: Vec<f64> = backend.alloc_init(5, 1.234);
        assert_eq!(inited.len(), 5);
        assert!(inited.iter().all(|&x| (x - 1.234).abs() < 1e-10));
    }
    
    #[test]
    fn test_cpu_backend_reduce() {
        let backend = CpuBackend::<f64>::new();
        let x: Vec<f64> = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        
        assert!((backend.reduce_max(&x) - 5.0).abs() < 1e-10);
        assert!((backend.reduce_min(&x) - 1.0).abs() < 1e-10);
        assert!((backend.reduce_sum(&x) - 15.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_cpu_backend_norm2() {
        let backend = CpuBackend::<f64>::new();
        let x: Vec<f64> = vec![3.0, 4.0];
        assert!((backend.norm2(&x) - 5.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_cpu_backend_enforce_positivity() {
        let backend = CpuBackend::<f64>::new();
        let mut x: Vec<f64> = vec![-1.0, 0.5, -0.001, 2.0];
        backend.enforce_positivity(&mut x, 0.0);
        assert_eq!(x, vec![0.0, 0.5, 0.0, 2.0]);
    }
    
    #[test]
    fn test_cpu_backend_elementwise_ops() {
        let backend = CpuBackend::<f64>::new();
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![4.0, 5.0, 6.0];
        let mut z: Vec<f64> = vec![0.0, 0.0, 0.0];
        
        backend.elementwise_mul(&x, &y, &mut z);
        assert_eq!(z, vec![4.0, 10.0, 18.0]);
        
        backend.elementwise_div_safe(&x, &y, &mut z, 1e-10);
        assert!((z[0] - 0.25).abs() < 1e-10);
        assert!((z[1] - 0.4).abs() < 1e-10);
        assert!((z[2] - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_cpu_backend_memory_location() {
        let backend = CpuBackend::<f64>::new();
        assert_eq!(backend.memory_location(), MemoryLocation::Host);
        assert_eq!(backend.name(), "CPU-f64");
        
        let backend_f32 = CpuBackend::<f32>::new();
        assert_eq!(backend_f32.name(), "CPU-f32");
    }
}
