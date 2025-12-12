// crates/mh_core/src/lib.rs

//! MariHydro 核心抽象层
//!
//! 提供整个项目的基础抽象，包括：
//! - [`precision`]: 运行时精度选择（F32/F64）
//! - [`scalar`]: 统一标量trait（替代分散的f32/f64）
//! - [`indices`]: 统一索引类型（CellIndex, FaceIndex, NodeIndex等）
//! - [`tolerance`]: 泛型化容差配置
//! - [`backend`]: 计算后端抽象（CPU/GPU）
//! - [`buffer`]: 设备缓冲区抽象
//!
//! # 设计原则
//!
//! 1. **App层无泛型**: SolverConfig及以上层级禁止出现泛型参数
//! 2. **运行时精度可切换**: 通过Precision枚举动态分发
//! 3. **计算层零成本抽象**: 编译期单态化，无运行时开销
//! 4. **f32/f64双版本支持**: 通过Scalar trait统一接口
//!
//! # 层级架构
//!
//! ```text
//! Layer 5: Application (无泛型)
//!     └─> SolverConfig, Precision枚举
//! Layer 4: Builder (枚举→泛型桥接)
//!     └─> SolverBuilder -> Box<dyn DynSolver>
//! Layer 3: Engine (全泛型)
//!     └─> ShallowWaterSolver<B>, FluxCalculator<B>
//! Layer 2: Core Abstractions (trait定义)
//!     └─> Scalar, Backend, DeviceBuffer
//! Layer 1: Foundation (类型定义)
//!     └─> Precision, Indices, Tolerance<S>
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod precision;
pub mod scalar;
pub mod indices;
pub mod tolerance;
pub mod backend;
pub mod buffer;

// 统一导出
pub use precision::Precision;
pub use scalar::Scalar;
pub use indices::{CellIndex, FaceIndex, NodeIndex, BoundaryIndex, VertexIndex, HalfEdgeIndex, INVALID_INDEX};
pub use tolerance::Tolerance;
pub use backend::{Backend, CpuBackend};
pub use buffer::DeviceBuffer;

/// 编译期断言宏
#[macro_export]
macro_rules! assert_scalar {
    ($t:ty) => {
        const _: () = {
            fn _assert_scalar<T: $crate::Scalar>() {}
            fn _check() { _assert_scalar::<$t>(); }
        };
    };
}

/// 编译期断言：确保类型是Backend
#[macro_export]
macro_rules! assert_backend {
    ($t:ty) => {
        const _: () = {
            fn _assert_backend<B: $crate::Backend>() {}
            fn _check() { _assert_backend::<$t>(); }
        };
    };
}

/// Prelude模块
pub mod prelude {
    //! 常用类型预导入
    pub use crate::precision::Precision;
    pub use crate::scalar::Scalar;
    pub use crate::indices::{CellIndex, FaceIndex, NodeIndex, BoundaryIndex};
    pub use crate::tolerance::Tolerance;
    pub use crate::backend::{Backend, CpuBackend};
    pub use crate::buffer::DeviceBuffer;
}
