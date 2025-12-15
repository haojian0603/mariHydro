// crates/mh_runtime/src/backend.rs

//! Backend - 计算后端抽象
//!
//! 提供统一的计算后端接口，支持 CPU 和未来的 GPU 后端。

use bytemuck::Pod;
use std::marker::PhantomData;

use crate::buffer::DeviceBuffer;
use crate::scalar::RuntimeScalar;

/// 内存位置
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// 主机内存 (CPU)
    Host,
    /// 设备内存 (GPU)
    Device(usize),
}

/// 计算后端 Trait
///
/// 抽象不同计算设备的操作，包括内存分配、BLAS 操作等。
/// 
/// # 类型参数
/// 
/// - `Scalar`: 标量类型（f32 或 f64）
/// - `Buffer<T>`: 关联的缓冲区类型
pub trait Backend: Clone + Send + Sync + 'static {
    /// 标量类型
    type Scalar: RuntimeScalar;
    /// 缓冲区类型
    type Buffer<T: Pod + Clone + Send + Sync>: DeviceBuffer<T>;

    /// 后端名称
    fn name(&self) -> &'static str;
    
    /// 内存位置
    fn memory_location(&self) -> MemoryLocation;
    
    /// 分配缓冲区
    fn alloc<T: Pod + Clone + Default + Send + Sync>(&self, len: usize) -> Self::Buffer<T>;
    
    /// 分配并初始化缓冲区
    fn alloc_init<T: Pod + Clone + Send + Sync>(&self, len: usize, value: T) -> Self::Buffer<T>
    where 
        T: Default 
    {
        let mut buf = self.alloc(len);
        buf.fill(value);
        buf
    }
    
    /// 从配置 f64 转换到标量类型
    #[inline]
    fn scalar_from_f64(&self, v: f64) -> Self::Scalar {
        Self::Scalar::from_config(v).unwrap_or(Self::Scalar::ZERO)
    }

    /// 同步操作（GPU 后端需要）
    fn synchronize(&self) {}

    // =========================================================================
    // BLAS Level 1 操作
    // =========================================================================

    /// y = alpha * x + y (AXPY)
    fn axpy(
        &self,
        alpha: Self::Scalar,
        x: &Self::Buffer<Self::Scalar>,
        y: &mut Self::Buffer<Self::Scalar>,
    );
    
    /// 点积: sum(x[i] * y[i])
    fn dot(
        &self,
        x: &Self::Buffer<Self::Scalar>,
        y: &Self::Buffer<Self::Scalar>,
    ) -> Self::Scalar;
    
    /// 复制: dst = src
    fn copy(
        &self,
        src: &Self::Buffer<Self::Scalar>,
        dst: &mut Self::Buffer<Self::Scalar>,
    );
    
    /// 缩放: x = alpha * x
    fn scale(&self, alpha: Self::Scalar, x: &mut Self::Buffer<Self::Scalar>);

    // =========================================================================
    // 规约操作
    // =========================================================================
    
    /// 最大值
    fn reduce_max(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    /// 最小值
    fn reduce_min(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    /// 求和
    fn reduce_sum(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    /// 2-范数
    fn norm2(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;

    // =========================================================================
    // 物理约束
    // =========================================================================
    
    /// 强制正性（水深等物理量）
    fn enforce_positivity(&self, x: &mut Self::Buffer<Self::Scalar>, min_val: Self::Scalar);
}

// =============================================================================
// CPU 后端
// =============================================================================

/// CPU 后端（零大小类型）
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend<S: RuntimeScalar> {
    _marker: PhantomData<S>,
}

impl<S: RuntimeScalar> CpuBackend<S> {
    /// 创建 CPU 后端
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

// f32 后端实现
impl Backend for CpuBackend<f32> {
    type Scalar = f32;
    type Buffer<T: Pod + Clone + Send + Sync> = Vec<T>;

    fn name(&self) -> &'static str { "CPU-f32" }
    fn memory_location(&self) -> MemoryLocation { MemoryLocation::Host }
    
    fn alloc<T: Pod + Clone + Default + Send + Sync>(&self, len: usize) -> Self::Buffer<T> {
        vec![T::default(); len]
    }

    fn axpy(&self, alpha: f32, x: &Vec<f32>, y: &mut Vec<f32>) {
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    }

    fn dot(&self, x: &Vec<f32>, y: &Vec<f32>) -> f32 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn copy(&self, src: &Vec<f32>, dst: &mut Vec<f32>) {
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
        self.dot(x, x).sqrt()
    }

    fn enforce_positivity(&self, x: &mut Vec<f32>, min_val: f32) {
        for xi in x.iter_mut() {
            if *xi < min_val {
                *xi = min_val;
            }
        }
    }
}

// f64 后端实现
impl Backend for CpuBackend<f64> {
    type Scalar = f64;
    type Buffer<T: Pod + Clone + Send + Sync> = Vec<T>;

    fn name(&self) -> &'static str { "CPU-f64" }
    fn memory_location(&self) -> MemoryLocation { MemoryLocation::Host }
    
    fn alloc<T: Pod + Clone + Default + Send + Sync>(&self, len: usize) -> Self::Buffer<T> {
        vec![T::default(); len]
    }

    fn axpy(&self, alpha: f64, x: &Vec<f64>, y: &mut Vec<f64>) {
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    }

    fn dot(&self, x: &Vec<f64>, y: &Vec<f64>) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn copy(&self, src: &Vec<f64>, dst: &mut Vec<f64>) {
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
        self.dot(x, x).sqrt()
    }

    fn enforce_positivity(&self, x: &mut Vec<f64>, min_val: f64) {
        for xi in x.iter_mut() {
            if *xi < min_val {
                *xi = min_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_f64() {
        let backend = CpuBackend::<f64>::new();
        assert_eq!(backend.name(), "CPU-f64");
        assert_eq!(backend.memory_location(), MemoryLocation::Host);
        
        let x: Vec<f64> = backend.alloc(10);
        assert_eq!(x.len(), 10);
    }

    #[test]
    fn test_axpy() {
        let backend = CpuBackend::<f64>::new();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![1.0, 1.0, 1.0];
        backend.axpy(2.0, &x, &mut y);
        assert_eq!(y, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_dot() {
        let backend = CpuBackend::<f64>::new();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 1.0, 1.0];
        assert_eq!(backend.dot(&x, &y), 6.0);
    }

    #[test]
    fn test_enforce_positivity() {
        let backend = CpuBackend::<f64>::new();
        let mut x = vec![-1.0, 0.5, -0.5, 2.0];
        backend.enforce_positivity(&mut x, 0.0);
        assert_eq!(x, vec![0.0, 0.5, 0.0, 2.0]);
    }
}