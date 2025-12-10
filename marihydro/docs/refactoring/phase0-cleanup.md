# Phase 0: 清理与根基

## 目标

删除死代码，统一 Scalar 定义，修复 Backend 静态方法问题。

## 时间：第 1 周

## 任务清单

### 0.1 删除 3D 死代码

**目标**：移除未使用的 3D 相关代码，减少维护负担。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 删除 | `sources/turbulence/k_epsilon.rs` | 3D湍流模型，无2D支持 |
| 修改 | `sources/turbulence/mod.rs` | 删除k_epsilon引用 |
| 修改 | `sources/implicit.rs` | 删除未使用的`ImplicitMethod::CrankNicolson`变体 |

#### 具体改动

**sources/turbulence/mod.rs**

```rust
// 删除以下行（如果存在）
// pub mod k_epsilon;
// pub use k_epsilon::*;
```

**sources/implicit.rs**

```rust
// 删除 CrankNicolson 变体（如果存在且未使用）
pub enum ImplicitMethod {
    PointImplicit,
    // CrankNicolson,  // 删除此行
}
```

#### 验证

```bash
cargo check -p mh_physics
cargo test -p mh_physics
```

---

### 0.2 统一 Scalar 到 physics

**目标**：将 Scalar trait 的权威定义移至 `mh_physics::core::scalar`，`mh_foundation` 改为重导出。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 重构 | `mh_physics/src/core/scalar.rs` | 完整Scalar trait定义（添加物理常量） |
| 重构 | `mh_foundation/src/scalar.rs` | 删除Float/ScalarOps，改为重导出 |
| 修改 | `mh_foundation/src/lib.rs` | 更新导出 |
| 修改 | `mh_foundation/Cargo.toml` | 添加对mh_physics的依赖 |
| 标记 | `mh_foundation/src/memory.rs` | `#[deprecated]` AlignedVec |

#### 具体改动

**mh_physics/src/core/scalar.rs（扩展）**

```rust
// marihydro/crates/mh_physics/src/core/scalar.rs
//! 标量类型抽象 - 项目唯一权威定义
//!
//! 提供 f32/f64 的统一接口，支持编译期精度选择。

use bytemuck::Pod;
use num_traits::{Float, FromPrimitive, NumAssign};
use std::fmt::{Debug, Display};
use std::iter::Sum;

/// 标量类型约束 - 项目唯一权威定义
pub trait Scalar:
    Float
    + Pod
    + Default
    + Debug
    + Display
    + Send
    + Sync
    + NumAssign
    + FromPrimitive
    + Sum
    + 'static
{
    // ========== 常量 ==========
    
    /// 零值
    const ZERO: Self;
    /// 单位值
    const ONE: Self;
    /// 机器精度
    const EPSILON: Self;
    /// 最小正规数
    const MIN_POSITIVE: Self;
    /// 最大值
    const MAX: Self;
    /// 圆周率
    const PI: Self;
    /// 重力加速度 (m/s²)
    const GRAVITY: Self;
    /// 水密度 (kg/m³)
    const WATER_DENSITY: Self;
    /// 冯卡门常数
    const VON_KARMAN: Self;
    
    // ========== 类型信息 ==========
    
    /// 类型名称
    fn type_name() -> &'static str;
    
    // ========== 转换 ==========
    
    /// 从 f64 转换
    fn from_f64(v: f64) -> Self;
    
    /// 转换为 f64
    fn to_f64(self) -> f64;
    
    // ========== 数学运算 ==========
    
    /// 平方根
    fn sqrt(self) -> Self;
    
    /// 绝对值
    fn abs(self) -> Self;
    
    /// 最大值
    fn max(self, other: Self) -> Self;
    
    /// 最小值
    fn min(self, other: Self) -> Self;
    
    /// 钳位
    fn clamp(self, min: Self, max: Self) -> Self;
    
    /// 幂运算
    fn powf(self, n: Self) -> Self;
    
    /// 是否为有限数
    fn is_finite(self) -> bool;
    
    /// 是否为 NaN
    fn is_nan(self) -> bool;
    
    /// 符号函数
    fn signum(self) -> Self;
    
    /// 正弦
    fn sin(self) -> Self;
    
    /// 余弦
    fn cos(self) -> Self;
    
    /// 反正切
    fn atan2(self, other: Self) -> Self;
    
    /// 自然对数
    fn ln(self) -> Self;
    
    /// 指数
    fn exp(self) -> Self;
}

impl Scalar for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const EPSILON: f32 = 1e-6;
    const MIN_POSITIVE: f32 = f32::MIN_POSITIVE;
    const MAX: f32 = f32::MAX;
    const PI: f32 = std::f32::consts::PI;
    const GRAVITY: f32 = 9.81;
    const WATER_DENSITY: f32 = 1000.0;
    const VON_KARMAN: f32 = 0.41;
    
    fn type_name() -> &'static str { "f32" }
    fn from_f64(v: f64) -> Self { v as f32 }
    fn to_f64(self) -> f64 { self as f64 }
    fn sqrt(self) -> Self { f32::sqrt(self) }
    fn abs(self) -> Self { f32::abs(self) }
    fn max(self, other: Self) -> Self { f32::max(self, other) }
    fn min(self, other: Self) -> Self { f32::min(self, other) }
    fn clamp(self, min: Self, max: Self) -> Self { f32::clamp(self, min, max) }
    fn powf(self, n: Self) -> Self { f32::powf(self, n) }
    fn is_finite(self) -> bool { f32::is_finite(self) }
    fn is_nan(self) -> bool { f32::is_nan(self) }
    fn signum(self) -> Self { f32::signum(self) }
    fn sin(self) -> Self { f32::sin(self) }
    fn cos(self) -> Self { f32::cos(self) }
    fn atan2(self, other: Self) -> Self { f32::atan2(self, other) }
    fn ln(self) -> Self { f32::ln(self) }
    fn exp(self) -> Self { f32::exp(self) }
}

impl Scalar for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const EPSILON: f64 = 1e-12;
    const MIN_POSITIVE: f64 = f64::MIN_POSITIVE;
    const MAX: f64 = f64::MAX;
    const PI: f64 = std::f64::consts::PI;
    const GRAVITY: f64 = 9.81;
    const WATER_DENSITY: f64 = 1000.0;
    const VON_KARMAN: f64 = 0.41;
    
    fn type_name() -> &'static str { "f64" }
    fn from_f64(v: f64) -> Self { v }
    fn to_f64(self) -> f64 { self }
    fn sqrt(self) -> Self { f64::sqrt(self) }
    fn abs(self) -> Self { f64::abs(self) }
    fn max(self, other: Self) -> Self { f64::max(self, other) }
    fn min(self, other: Self) -> Self { f64::min(self, other) }
    fn clamp(self, min: Self, max: Self) -> Self { f64::clamp(self, min, max) }
    fn powf(self, n: Self) -> Self { f64::powf(self, n) }
    fn is_finite(self) -> bool { f64::is_finite(self) }
    fn is_nan(self) -> bool { f64::is_nan(self) }
    fn signum(self) -> Self { f64::signum(self) }
    fn sin(self) -> Self { f64::sin(self) }
    fn cos(self) -> Self { f64::cos(self) }
    fn atan2(self, other: Self) -> Self { f64::atan2(self, other) }
    fn ln(self) -> Self { f64::ln(self) }
    fn exp(self) -> Self { f64::exp(self) }
}

/// 物理常量模块（使用泛型）
pub mod constants {
    use super::Scalar;
    
    /// 获取重力加速度
    #[inline]
    pub fn gravity<S: Scalar>() -> S { S::GRAVITY }
    
    /// 获取水密度
    #[inline]
    pub fn water_density<S: Scalar>() -> S { S::WATER_DENSITY }
    
    /// 获取冯卡门常数
    #[inline]
    pub fn von_karman<S: Scalar>() -> S { S::VON_KARMAN }
}
```

**mh_foundation/src/scalar.rs（重构后）**

```rust
// marihydro/crates/mh_foundation/src/scalar.rs
//! Scalar类型重导出
//!
//! 权威定义位于 `mh_physics::core::scalar`
//! 此模块仅提供向后兼容重导出

// 注意：由于循环依赖问题，这里保留独立定义
// 但接口与 mh_physics::core::Scalar 保持一致

use std::ops::{Add, Sub, Mul, Div, Neg};

/// 计算用标量类型（默认 f64，启用 gpu-f32 feature 时为 f32）
#[cfg(not(feature = "gpu-f32"))]
pub type Scalar = f64;

#[cfg(feature = "gpu-f32")]
pub type Scalar = f32;

/// 标量 trait：所有物理量必须满足的约束
/// 
/// 注意：此 trait 与 `mh_physics::core::Scalar` 接口一致
#[deprecated(since = "0.5.0", note = "请使用 mh_physics::core::Scalar")]
pub trait ScalarOps: 
    Copy + Clone + Default + PartialOrd +
    Add<Output = Self> + Sub<Output = Self> + 
    Mul<Output = Self> + Div<Output = Self> +
    Neg<Output = Self> +
    Sized
{
    const ZERO: Self;
    const ONE: Self;
    const EPSILON: Self;
    const MIN_POSITIVE: Self;
    const MAX: Self;
    
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
    fn clamp(self, min: Self, max: Self) -> Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

// 保留实现以保持向后兼容
#[allow(deprecated)]
impl ScalarOps for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = 1e-12;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    const MAX: Self = f64::MAX;
    
    #[inline] fn abs(self) -> Self { f64::abs(self) }
    #[inline] fn sqrt(self) -> Self { f64::sqrt(self) }
    #[inline] fn max(self, other: Self) -> Self { f64::max(self, other) }
    #[inline] fn min(self, other: Self) -> Self { f64::min(self, other) }
    #[inline] fn powf(self, n: Self) -> Self { f64::powf(self, n) }
    #[inline] fn is_finite(self) -> bool { f64::is_finite(self) }
    #[inline] fn is_nan(self) -> bool { f64::is_nan(self) }
    #[inline] fn clamp(self, min: Self, max: Self) -> Self { f64::clamp(self, min, max) }
    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn to_f64(self) -> f64 { self }
}

#[allow(deprecated)]
impl ScalarOps for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = 1e-6;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;
    const MAX: Self = f32::MAX;
    
    #[inline] fn abs(self) -> Self { f32::abs(self) }
    #[inline] fn sqrt(self) -> Self { f32::sqrt(self) }
    #[inline] fn max(self, other: Self) -> Self { f32::max(self, other) }
    #[inline] fn min(self, other: Self) -> Self { f32::min(self, other) }
    #[inline] fn powf(self, n: Self) -> Self { f32::powf(self, n) }
    #[inline] fn is_finite(self) -> bool { f32::is_finite(self) }
    #[inline] fn is_nan(self) -> bool { f32::is_nan(self) }
    #[inline] fn clamp(self, min: Self, max: Self) -> Self { f32::clamp(self, min, max) }
    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
}

/// 物理常量（保持向后兼容）
pub mod constants {
    use super::Scalar;
    
    pub const GRAVITY: Scalar = 9.81;
    pub const WATER_DENSITY: Scalar = 1000.0;
    pub const KINEMATIC_VISCOSITY: Scalar = 1.0e-6;
    pub const VON_KARMAN: Scalar = 0.41;
    pub const PI: Scalar = std::f64::consts::PI as Scalar;
}
```

**mh_foundation/src/memory.rs（添加废弃标记）**

```rust
// 在文件开头添加
#![allow(deprecated)]

// 在 AlignedVec 结构体上添加
/// 对齐内存容器
/// 
/// # 废弃说明
/// 
/// 此类型将在 v0.6.0 移除，请使用 `mh_physics::core::DeviceBuffer`
#[deprecated(since = "0.5.0", note = "请使用 mh_physics::core::DeviceBuffer")]
pub struct AlignedVec<T, A: Alignment = CpuAlign> {
    // ... 现有实现
}
```

#### 验证

```bash
cargo check --workspace
cargo test --workspace
```

---

### 0.3 Backend 改为实例方法

**目标**：将 Backend trait 的静态方法改为实例方法，支持 GPU 持有设备状态。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 重构 | `core/backend.rs` | 所有方法添加`&self` |
| 新建 | `core/cpu_backend.rs` | CpuBackend<f32/f64>完整实现 |
| 修改 | `core/mod.rs` | 更新导出 |

#### 具体改动

**mh_physics/src/core/backend.rs（重构）**

```rust
// marihydro/crates/mh_physics/src/core/backend.rs
//! 计算后端抽象
//!
//! 提供 CPU/GPU 统一的计算接口。
//! 
//! # 设计变更 (v0.5.0)
//! 
//! 所有方法改为实例方法，支持 GPU 后端持有设备状态。

use super::buffer::DeviceBuffer;
use super::scalar::Scalar;
use bytemuck::Pod;
use std::fmt::Debug;

/// 计算后端内存位置
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// 主机内存
    Host,
    /// GPU设备内存
    Device(usize),
}

/// 计算后端 trait
/// 
/// # 设计原则
/// 
/// 1. 所有方法为实例方法（支持GPU持有设备状态）
/// 2. 泛型标量类型（f32/f64编译期确定）
/// 3. 零开销抽象（CPU后端无运行时开销）
pub trait Backend: Clone + Send + Sync + Debug + 'static {
    /// 标量类型
    type Scalar: Scalar;
    
    /// 缓冲区类型
    type Buffer<T: Pod + Send + Sync>: DeviceBuffer<T>;
    
    /// 后端名称
    fn name(&self) -> &'static str;
    
    /// 内存位置
    fn memory_location(&self) -> MemoryLocation;
    
    /// 分配缓冲区
    fn alloc<T: Pod + Clone + Default + Send + Sync>(&self, len: usize) -> Self::Buffer<T>;
    
    /// 分配并初始化缓冲区
    fn alloc_init<T: Pod + Clone + Send + Sync>(&self, len: usize, init: T) -> Self::Buffer<T>;
    
    /// 分配未初始化缓冲区（性能优化）
    fn alloc_uninit<T: Pod + Send + Sync>(&self, len: usize) -> Self::Buffer<T>;
    
    /// 同步操作（GPU 需要，CPU 空实现）
    fn synchronize(&self) {}
    
    // ========== BLAS Level 1 算子 ==========
    
    /// y = alpha * x + y (AXPY)
    fn axpy(
        &self,
        alpha: Self::Scalar,
        x: &Self::Buffer<Self::Scalar>,
        y: &mut Self::Buffer<Self::Scalar>,
    );
    
    /// dot = x · y
    fn dot(
        &self,
        x: &Self::Buffer<Self::Scalar>,
        y: &Self::Buffer<Self::Scalar>,
    ) -> Self::Scalar;
    
    /// y = x (复制)
    fn copy(
        &self,
        src: &Self::Buffer<Self::Scalar>,
        dst: &mut Self::Buffer<Self::Scalar>,
    );
    
    /// 归约：最大值
    fn reduce_max(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    /// 归约：求和
    fn reduce_sum(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    /// 缩放: x = alpha * x
    fn scale(&self, alpha: Self::Scalar, x: &mut Self::Buffer<Self::Scalar>);
    
    // ========== 物理专用算子 ==========
    
    /// 逐元素应用函数
    fn apply_elementwise<F>(&self, f: F, x: &mut Self::Buffer<Self::Scalar>)
    where
        F: Fn(Self::Scalar) -> Self::Scalar + Send + Sync;
    
    /// 确保正性：x[i] = max(x[i], min_val)
    fn enforce_positivity(&self, x: &mut Self::Buffer<Self::Scalar>, min_val: Self::Scalar);
    
    /// 填充常量
    fn fill(&self, x: &mut Self::Buffer<Self::Scalar>, value: Self::Scalar);
}

/// 类型别名：默认后端
pub type DefaultBackend = CpuBackend<f64>;

// CpuBackend 移至单独文件
pub use super::cpu_backend::CpuBackend;
```

**mh_physics/src/core/cpu_backend.rs（新建）**

```rust
// marihydro/crates/mh_physics/src/core/cpu_backend.rs
//! CPU 后端实现

use super::backend::{Backend, MemoryLocation};
use super::buffer::DeviceBuffer;
use super::scalar::Scalar;
use bytemuck::Pod;
use std::marker::PhantomData;

/// CPU 后端（泛型精度）
/// 
/// 无状态实现，实例化零开销。
#[derive(Debug, Clone, Default)]
pub struct CpuBackend<S: Scalar> {
    _marker: PhantomData<S>,
}

impl<S: Scalar> CpuBackend<S> {
    /// 创建新的 CPU 后端
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<S: Scalar> Backend for CpuBackend<S> {
    type Scalar = S;
    type Buffer<T: Pod + Send + Sync> = Vec<T>;
    
    fn name(&self) -> &'static str {
        if std::mem::size_of::<S>() == 4 { "CPU-f32" } else { "CPU-f64" }
    }
    
    fn memory_location(&self) -> MemoryLocation {
        MemoryLocation::Host
    }
    
    fn alloc<T: Pod + Clone + Default + Send + Sync>(&self, len: usize) -> Vec<T> {
        vec![T::default(); len]
    }
    
    fn alloc_init<T: Pod + Clone + Send + Sync>(&self, len: usize, init: T) -> Vec<T> {
        vec![init; len]
    }
    
    fn alloc_uninit<T: Pod + Send + Sync>(&self, len: usize) -> Vec<T> {
        let mut v = Vec::with_capacity(len);
        // SAFETY: Pod 类型可以安全地使用未初始化内存
        unsafe { v.set_len(len); }
        v
    }
    
    fn axpy(&self, alpha: S, x: &Vec<S>, y: &mut Vec<S>) {
        debug_assert_eq!(x.len(), y.len());
        for (yi, xi) in y.iter_mut().zip(x.iter()) {
            *yi = *yi + alpha * *xi;
        }
    }
    
    fn dot(&self, x: &Vec<S>, y: &Vec<S>) -> S {
        debug_assert_eq!(x.len(), y.len());
        x.iter().zip(y.iter()).fold(S::ZERO, |acc, (&xi, &yi)| acc + xi * yi)
    }
    
    fn copy(&self, src: &Vec<S>, dst: &mut Vec<S>) {
        dst.copy_from_slice(src);
    }
    
    fn reduce_max(&self, x: &Vec<S>) -> S {
        x.iter().cloned().fold(S::from_f64(f64::NEG_INFINITY), S::max)
    }
    
    fn reduce_sum(&self, x: &Vec<S>) -> S {
        x.iter().cloned().fold(S::ZERO, |a, b| a + b)
    }
    
    fn scale(&self, alpha: S, x: &mut Vec<S>) {
        for xi in x.iter_mut() {
            *xi = *xi * alpha;
        }
    }
    
    fn apply_elementwise<F>(&self, f: F, x: &mut Vec<S>)
    where
        F: Fn(S) -> S + Send + Sync
    {
        for xi in x.iter_mut() {
            *xi = f(*xi);
        }
    }
    
    fn enforce_positivity(&self, x: &mut Vec<S>, min_val: S) {
        for xi in x.iter_mut() {
            *xi = xi.max(min_val);
        }
    }
    
    fn fill(&self, x: &mut Vec<S>, value: S) {
        x.fill(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_instance_f64() {
        let backend = CpuBackend::<f64>::new();
        let x = backend.alloc_init(100, 1.0);
        let mut y = backend.alloc_init(100, 2.0);
        backend.axpy(0.5, &x, &mut y);
        assert!((y[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_backend_instance_f32() {
        let backend = CpuBackend::<f32>::new();
        let x = backend.alloc_init(100, 1.0f32);
        let mut y = backend.alloc_init(100, 2.0f32);
        backend.axpy(0.5, &x, &mut y);
        assert!((y[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_backend_dot() {
        let backend = CpuBackend::<f64>::new();
        let x = backend.alloc_init(3, 1.0);
        let y = backend.alloc_init(3, 2.0);
        let result = backend.dot(&x, &y);
        assert!((result - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_backend_reduce() {
        let backend = CpuBackend::<f64>::new();
        let x = vec![1.0, 5.0, 3.0, 2.0];
        assert!((backend.reduce_max(&x) - 5.0).abs() < 1e-10);
        assert!((backend.reduce_sum(&x) - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_backend_enforce_positivity() {
        let backend = CpuBackend::<f64>::new();
        let mut x = vec![-1.0, 0.5, -0.1, 2.0];
        backend.enforce_positivity(&mut x, 0.0);
        assert_eq!(x, vec![0.0, 0.5, 0.0, 2.0]);
    }
}
```

**mh_physics/src/core/mod.rs（更新）**

```rust
//! 核心抽象层

pub mod scalar;
pub mod buffer;
pub mod backend;
pub mod cpu_backend;  // 新增
pub mod dimension;
pub mod kernel;
pub mod gpu;

// 重导出常用类型
pub use scalar::Scalar;
pub use buffer::DeviceBuffer;
pub use backend::{Backend, MemoryLocation, DefaultBackend};
pub use cpu_backend::CpuBackend;
pub use dimension::{Dimension, D2, D3};
pub use kernel::{KernelSpec, KernelPriority, TransferPolicy, CORE_KERNELS};
pub use gpu::{CudaError, GpuDeviceInfo, available_gpus, has_cuda};
```

#### 验证

```rust
#[test]
fn test_backend_instance_methods() {
    let backend = CpuBackend::<f64>::new();
    
    // 测试实例方法调用
    let x = backend.alloc_init(100, 1.0);
    let mut y = backend.alloc_init(100, 2.0);
    
    backend.axpy(0.5, &x, &mut y);
    assert!((y[0] - 2.5).abs() < 1e-10);
    
    let dot_result = backend.dot(&x, &y);
    assert!((dot_result - 250.0).abs() < 1e-10);
}
```

```bash
cargo test -p mh_physics core::
```

---

## 验收标准

1. ✅ `cargo check -p mh_physics` 通过
2. ✅ `cargo check --workspace` 通过
3. ✅ 所有现有测试通过
4. ✅ Backend 方法可通过实例调用
5. ✅ Scalar trait 包含物理常量

## 依赖关系

```
Phase 0 完成后:
├── Phase 1 可以开始（状态泛型化依赖 Backend 实例方法）
└── 其他 Phase 等待 Phase 1
```

## 回滚计划

如果出现严重问题：

1. 保留旧的静态方法作为 `_static` 后缀版本
2. 新代码使用实例方法，旧代码逐步迁移
3. 使用 feature flag 控制新旧实现切换
