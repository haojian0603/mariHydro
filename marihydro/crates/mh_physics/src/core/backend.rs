// marihydro\crates\mh_physics\src\core\backend.rs
//! 计算后端抽象
//!
//! 提供 CPU/GPU 统一的计算接口。

use super::buffer::DeviceBuffer;
use super::scalar::Scalar;
use bytemuck::Pod;
use std::fmt::Debug;
use std::marker::PhantomData;

/// 计算后端 trait
pub trait Backend: Clone + Send + Sync + Debug + 'static {
    /// 标量类型
    type Scalar: Scalar;
    
    /// 缓冲区类型
    type Buffer<T: Pod + Send + Sync>: DeviceBuffer<T>;
    
    /// 后端名称
    fn name() -> &'static str;
    
    /// 分配缓冲区
    fn alloc<T: Pod + Clone + Default + Send + Sync>(len: usize) -> Self::Buffer<T>;
    
    /// 分配并初始化缓冲区
    fn alloc_init<T: Pod + Clone + Send + Sync>(len: usize, init: T) -> Self::Buffer<T>;
    
    /// 同步操作（GPU 需要，CPU 空实现）
    fn synchronize() {}
    
    // ========== BLAS Level 1 算子 ==========
    
    /// y = alpha * x + y (AXPY)
    fn axpy(
        alpha: Self::Scalar,
        x: &Self::Buffer<Self::Scalar>,
        y: &mut Self::Buffer<Self::Scalar>,
    );
    
    /// dot = x · y
    fn dot(
        x: &Self::Buffer<Self::Scalar>,
        y: &Self::Buffer<Self::Scalar>,
    ) -> Self::Scalar;
    
    /// y = x (复制)
    fn copy(
        src: &Self::Buffer<Self::Scalar>,
        dst: &mut Self::Buffer<Self::Scalar>,
    );
    
    /// 归约：最大值
    fn reduce_max(x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    /// 归约：求和
    fn reduce_sum(x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    /// 缩放: x = alpha * x
    fn scale(alpha: Self::Scalar, x: &mut Self::Buffer<Self::Scalar>);
}

/// CPU 后端（泛型精度）
#[derive(Debug, Clone, Default)]
pub struct CpuBackend<S: Scalar> {
    _marker: PhantomData<S>,
}

impl<S: Scalar> CpuBackend<S> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl Backend for CpuBackend<f32> {
    type Scalar = f32;
    type Buffer<T: Pod + Send + Sync> = Vec<T>;
    
    fn name() -> &'static str { "CPU-f32" }
    
    fn alloc<T: Pod + Clone + Default + Send + Sync>(len: usize) -> Self::Buffer<T> {
        vec![T::default(); len]
    }
    
    fn alloc_init<T: Pod + Clone + Send + Sync>(len: usize, init: T) -> Self::Buffer<T> {
        vec![init; len]
    }
    
    fn axpy(alpha: f32, x: &Vec<f32>, y: &mut Vec<f32>) {
        debug_assert_eq!(x.len(), y.len());
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    }
    
    fn dot(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
        debug_assert_eq!(x.len(), y.len());
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
    
    fn copy(src: &Vec<f32>, dst: &mut Vec<f32>) {
        dst.copy_from_slice(src);
    }
    
    fn reduce_max(x: &Vec<f32>) -> f32 {
        x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }
    
    fn reduce_sum(x: &Vec<f32>) -> f32 {
        x.iter().sum()
    }
    
    fn scale(alpha: f32, x: &mut Vec<f32>) {
        for xi in x.iter_mut() {
            *xi *= alpha;
        }
    }
}

impl Backend for CpuBackend<f64> {
    type Scalar = f64;
    type Buffer<T: Pod + Send + Sync> = Vec<T>;
    
    fn name() -> &'static str { "CPU-f64" }
    
    fn alloc<T: Pod + Clone + Default + Send + Sync>(len: usize) -> Self::Buffer<T> {
        vec![T::default(); len]
    }
    
    fn alloc_init<T: Pod + Clone + Send + Sync>(len: usize, init: T) -> Self::Buffer<T> {
        vec![init; len]
    }
    
    fn axpy(alpha: f64, x: &Vec<f64>, y: &mut Vec<f64>) {
        debug_assert_eq!(x.len(), y.len());
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    }
    
    fn dot(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
        debug_assert_eq!(x.len(), y.len());
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }
    
    fn copy(src: &Vec<f64>, dst: &mut Vec<f64>) {
        dst.copy_from_slice(src);
    }
    
    fn reduce_max(x: &Vec<f64>) -> f64 {
        x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
    
    fn reduce_sum(x: &Vec<f64>) -> f64 {
        x.iter().sum()
    }
    
    fn scale(alpha: f64, x: &mut Vec<f64>) {
        for xi in x.iter_mut() {
            *xi *= alpha;
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
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut y: Vec<f64> = vec![4.0, 5.0, 6.0];
        CpuBackend::<f64>::axpy(2.0, &x, &mut y);
        assert_eq!(y, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_cpu_backend_f64_dot() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![4.0, 5.0, 6.0];
        let result = CpuBackend::<f64>::dot(&x, &y);
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_backend_f32() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut y: Vec<f32> = vec![0.0, 0.0, 0.0];
        CpuBackend::<f32>::copy(&x, &mut y);
        assert_eq!(y, x);
    }
}
