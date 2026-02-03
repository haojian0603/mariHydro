// crates/mh_runtime/src/backend.rs

//! 计算后端抽象
//!
//! 提供统一的计算后端接口，支持 CPU 和未来的 GPU 后端。
//! 使用宏 `impl_cpu_backend!` 生成 f32/f64 后端实现，消除代码重复。
//! 采用密封 trait 模式防止外部实现，确保向后兼容性和优化空间。

use bytemuck::Pod;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use crate::buffer::DeviceBuffer;
use crate::scalar::RuntimeScalar;
use crate::error::{RuntimeError, RuntimeResult};
use num_traits::{FromPrimitive, ToPrimitive};

/// 密封模块，限制 Backend 只能在库内部实现
mod private {
    pub trait Sealed {}
}

/// 内存位置
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// 主机内存（CPU）
    Host,
    /// 设备内存（GPU），包含设备 ID
    Device(usize),
}

/// 二维向量 trait
///
/// 抽象不同后端的二维向量类型，确保几何运算一致性
pub trait Vector2D: Copy + Clone + Send + Sync + 'static {
    /// 关联的标量类型
    type Scalar: RuntimeScalar;
    
    /// 获取 x 分量
    fn x(&self) -> Self::Scalar;
    
    /// 获取 y 分量
    fn y(&self) -> Self::Scalar;
}

/// 计算后端 Trait（密封）
///
/// 抽象不同计算设备的操作，包括内存分配、BLAS 操作等。
/// 只能由库内部类型实现，外部 crate 无法为自定义类型实现此 trait。
///
/// # 类型参数
/// - `Scalar`: 标量类型（f32 或 f64）
/// - `Buffer<T>`: 关联的缓冲区类型
/// - `Vector2D`: 二维向量类型
pub trait Backend: private::Sealed + Clone + Send + Sync + 'static {
    /// 标量类型
    type Scalar: RuntimeScalar;
    /// 关联的缓冲区类型
    type Buffer<T: Pod + Clone + Send + Sync>: DeviceBuffer<T> + Deref<Target = [T]> + DerefMut;
    /// 二维向量类型
    type Vector2D: Vector2D<Scalar = Self::Scalar>;

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
        match self.try_scalar_from_f64(v) {
            Ok(val) => val,
            Err(err) => {
                eprintln!("[mh_runtime::backend] scalar_from_f64 fallback: {err}");
                Self::Scalar::ZERO
            }
        }
    }

    /// 尝试从 f64 转换到标量类型（失败返回错误）
    #[inline]
    fn try_scalar_from_f64(&self, v: f64) -> RuntimeResult<Self::Scalar> {
        if !v.is_finite() {
            return Err(RuntimeError::NonFinite { value: v });
        }
        Self::Scalar::from_f64(v).ok_or_else(|| {
            let min = Self::Scalar::MIN.to_f64().unwrap_or(f64::MIN);
            let max = Self::Scalar::MAX.to_f64().unwrap_or(f64::MAX);
            RuntimeError::out_of_range(v, min, max)
        })
    }

    /// 同步操作（GPU 后端需要）
    fn synchronize(&self) {}

    /// y = alpha * x + y (AXPY)
    fn axpy(
        &self,
        alpha: Self::Scalar,
        x: &Self::Buffer<Self::Scalar>,
        y: &mut Self::Buffer<Self::Scalar>,
    );

    /// y = alpha * x + y (AXPY) with length check
    fn axpy_checked(
        &self,
        alpha: Self::Scalar,
        x: &Self::Buffer<Self::Scalar>,
        y: &mut Self::Buffer<Self::Scalar>,
    ) -> RuntimeResult<()> {
        if x.len() != y.len() {
            return Err(RuntimeError::size_mismatch("axpy", x.len(), y.len()));
        }
        self.axpy(alpha, x, y);
        Ok(())
    }
    
    /// 点积: sum(x[i] * y[i])
    fn dot(
        &self,
        x: &Self::Buffer<Self::Scalar>,
        y: &Self::Buffer<Self::Scalar>,
    ) -> Self::Scalar;

    /// 点积（带长度检查）
    fn dot_checked(
        &self,
        x: &Self::Buffer<Self::Scalar>,
        y: &Self::Buffer<Self::Scalar>,
    ) -> RuntimeResult<Self::Scalar> {
        if x.len() != y.len() {
            return Err(RuntimeError::size_mismatch("dot", x.len(), y.len()));
        }
        Ok(self.dot(x, y))
    }
    
    /// 复制: dst = src
    fn copy(
        &self,
        src: &Self::Buffer<Self::Scalar>,
        dst: &mut Self::Buffer<Self::Scalar>,
    );

    /// 复制（带长度检查）
    fn copy_checked(
        &self,
        src: &Self::Buffer<Self::Scalar>,
        dst: &mut Self::Buffer<Self::Scalar>,
    ) -> RuntimeResult<()> {
        if src.len() != dst.len() {
            return Err(RuntimeError::size_mismatch("copy", src.len(), dst.len()));
        }
        self.copy(src, dst);
        Ok(())
    }
    
    /// 缩放: x = alpha * x
    fn scale(&self, alpha: Self::Scalar, x: &mut Self::Buffer<Self::Scalar>);

    /// 最大值
    fn reduce_max(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    /// 最小值
    fn reduce_min(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    /// 求和
    fn reduce_sum(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    /// 2-范数
    fn norm2(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;

    /// 强制正性（水深等物理量）
    fn enforce_positivity(&self, x: &mut Self::Buffer<Self::Scalar>, min_val: Self::Scalar);

    /// 创建二维向量
    fn vec2_new(x: Self::Scalar, y: Self::Scalar) -> Self::Vector2D;
    
    /// 向量点积
    fn vec2_dot(a: &Self::Vector2D, b: &Self::Vector2D) -> Self::Scalar;
    
    /// 向量长度
    fn vec2_length(v: &Self::Vector2D) -> Self::Scalar;
    
    /// 向量减法
    fn vec2_sub(a: &Self::Vector2D, b: &Self::Vector2D) -> Self::Vector2D;
    
    /// 向量缩放
    fn vec2_scale(v: &Self::Vector2D, s: Self::Scalar) -> Self::Vector2D;

    /// 向量相加
    fn vec2_add(a: &Self::Vector2D, b: &Self::Vector2D) -> Self::Vector2D;

    /// 零向量
    fn vec2_zero() -> Self::Vector2D;

    /// 从切片收集一个向量（用于边界法向量表）
    fn vec2_gather(slice: &[Self::Vector2D], idx: usize) -> Self::Vector2D;

    /// 单块缓冲克隆（便于零拷贝视图）
    fn clone_buffer<T: Pod + Clone + Send + Sync>(&self, src: &Self::Buffer<T>) -> Self::Buffer<T>;

    /// 按字段交错拷贝 sources -> dst（SoA -> AoS 或打包快照）
    fn copy_interleaved<T: Pod + Clone + Send + Sync + Default>(&self, sources: &[&[T]], dst: &mut Self::Buffer<T>);
}

/// CPU 后端
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend<S: RuntimeScalar> {
    _marker: PhantomData<S>,
}

impl<S: RuntimeScalar> CpuBackend<S> {
    /// 创建 CPU 后端
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

/// 生成 CPU 后端 f32/f64 实现的宏
macro_rules! impl_cpu_backend {
    ($scalar:ty, $name:literal) => {
        impl private::Sealed for CpuBackend<$scalar> {}

        impl Backend for CpuBackend<$scalar> {
            type Scalar = $scalar;
            type Buffer<T: Pod + Clone + Send + Sync> = Vec<T>;
            type Vector2D = [$scalar; 2];

            fn name(&self) -> &'static str {
                $name
            }
            
            fn memory_location(&self) -> MemoryLocation {
                MemoryLocation::Host
            }
            
            fn alloc<T: Pod + Clone + Default + Send + Sync>(&self, len: usize) -> Self::Buffer<T> {
                vec![T::default(); len]
            }

            fn axpy(&self, alpha: $scalar, x: &Vec<$scalar>, y: &mut Vec<$scalar>) {
                if x.len() != y.len() {
                    panic!(
                        "[mh_runtime::backend] axpy length mismatch: x={}, y={}",
                        x.len(),
                        y.len()
                    );
                }
                for (yi, xi) in y.iter_mut().zip(x.iter()) {
                    *yi += alpha * xi;
                }
            }

            fn dot(&self, x: &Vec<$scalar>, y: &Vec<$scalar>) -> $scalar {
                if x.len() != y.len() {
                    panic!(
                        "[mh_runtime::backend] dot length mismatch: x={}, y={}",
                        x.len(),
                        y.len()
                    );
                }
                x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
            }

            fn copy(&self, src: &Vec<$scalar>, dst: &mut Vec<$scalar>) {
                if src.len() != dst.len() {
                    panic!(
                        "[mh_runtime::backend] copy length mismatch: src={}, dst={}",
                        src.len(),
                        dst.len()
                    );
                }
                dst.copy_from_slice(src);
            }

            fn scale(&self, alpha: $scalar, x: &mut Vec<$scalar>) {
                for xi in x.iter_mut() {
                    *xi *= alpha;
                }
            }

            fn reduce_max(&self, x: &Vec<$scalar>) -> $scalar {
                if x.is_empty() {
                    return <$scalar>::ZERO;
                }
                x.iter().cloned().fold(<$scalar>::NEG_INFINITY, <$scalar>::max)
            }

            fn reduce_min(&self, x: &Vec<$scalar>) -> $scalar {
                if x.is_empty() {
                    return <$scalar>::ZERO;
                }
                x.iter().cloned().fold(<$scalar>::INFINITY, <$scalar>::min)
            }

            fn reduce_sum(&self, x: &Vec<$scalar>) -> $scalar {
                x.iter().sum()
            }

            fn norm2(&self, x: &Vec<$scalar>) -> $scalar {
                if x.is_empty() {
                    return <$scalar>::ZERO;
                }
                self.dot(x, x).sqrt()
            }

            fn enforce_positivity(&self, x: &mut Vec<$scalar>, min_val: $scalar) {
                for xi in x.iter_mut() {
                    if *xi < min_val {
                        *xi = min_val;
                    }
                }
            }

            #[inline]
            fn vec2_new(x: $scalar, y: $scalar) -> Self::Vector2D {
                [x, y]
            }

            #[inline]
            fn vec2_dot(a: &Self::Vector2D, b: &Self::Vector2D) -> Self::Scalar {
                a[0] * b[0] + a[1] * b[1]
            }

            #[inline]
            fn vec2_length(v: &Self::Vector2D) -> Self::Scalar {
                (v[0] * v[0] + v[1] * v[1]).sqrt()
            }

            #[inline]
            fn vec2_sub(a: &Self::Vector2D, b: &Self::Vector2D) -> Self::Vector2D {
                [a[0] - b[0], a[1] - b[1]]
            }

            #[inline]
            fn vec2_scale(v: &Self::Vector2D, s: Self::Scalar) -> Self::Vector2D {
                [v[0] * s, v[1] * s]
            }

            #[inline]
            fn vec2_add(a: &Self::Vector2D, b: &Self::Vector2D) -> Self::Vector2D {
                [a[0] + b[0], a[1] + b[1]]
            }

            #[inline]
            fn vec2_zero() -> Self::Vector2D {
                [0 as $scalar, 0 as $scalar]
            }

            #[inline]
            fn vec2_gather(slice: &[Self::Vector2D], idx: usize) -> Self::Vector2D {
                slice.get(idx).copied().unwrap_or([0 as $scalar, 0 as $scalar])
            }

            #[inline]
            fn clone_buffer<T: Pod + Clone + Send + Sync>(&self, src: &Self::Buffer<T>) -> Self::Buffer<T> {
                src.clone()
            }

            fn copy_interleaved<T: Pod + Clone + Send + Sync + Default>(&self, sources: &[&[T]], dst: &mut Self::Buffer<T>) {
                let n_fields = sources.len();
                if n_fields == 0 {
                    return;
                }
                let len = sources[0].len();
                for (i, field) in sources.iter().enumerate() {
                    if field.len() != len {
                        panic!(
                            "[mh_runtime::backend] copy_interleaved length mismatch: field[0]={}, field[{}]={}",
                            len,
                            i,
                            field.len()
                        );
                    }
                }
                dst.clear();
                dst.reserve(len * n_fields);
                for i in 0..len {
                    for field in sources {
                        dst.push(field.get(i).cloned().unwrap_or_default());
                    }
                }
            }
        }

        impl Vector2D for [$scalar; 2] {
            type Scalar = $scalar;
            
            #[inline]
            fn x(&self) -> $scalar {
                self[0]
            }
            
            #[inline]
            fn y(&self) -> $scalar {
                self[1]
            }
        }
    };
}

impl_cpu_backend!(f32, "CPU-f32");
impl_cpu_backend!(f64, "CPU-f64");

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

    #[test]
    fn test_vec_operations_f32() {
        let v1 = <CpuBackend<f32>>::vec2_new(3.0f32, 4.0f32);
        let v2 = <CpuBackend<f32>>::vec2_new(1.0f32, 2.0f32);
        
        let dot = <CpuBackend<f32>>::vec2_dot(&v1, &v2);
        assert_eq!(dot, 11.0f32);
        
        let len = <CpuBackend<f32>>::vec2_length(&v1);
        assert_eq!(len, 5.0f32);
        
        let sub = <CpuBackend<f32>>::vec2_sub(&v1, &v2);
        assert_eq!(sub, [2.0f32, 2.0f32]);
        
        let scaled = <CpuBackend<f32>>::vec2_scale(&v1, 2.0f32);
        assert_eq!(scaled, [6.0f32, 8.0f32]);
    }

    #[test]
    fn test_vec_operations_f64() {
        let v1 = <CpuBackend<f64>>::vec2_new(3.0, 4.0);
        let v2 = <CpuBackend<f64>>::vec2_new(1.0, 2.0);
        
        let dot = <CpuBackend<f64>>::vec2_dot(&v1, &v2);
        assert_eq!(dot, 11.0);
        
        let len = <CpuBackend<f64>>::vec2_length(&v1);
        assert_eq!(len, 5.0);
        
        let sub = <CpuBackend<f64>>::vec2_sub(&v1, &v2);
        assert_eq!(sub, [2.0, 2.0]);
        
        let scaled = <CpuBackend<f64>>::vec2_scale(&v1, 2.0);
        assert_eq!(scaled, [6.0, 8.0]);
    }
}