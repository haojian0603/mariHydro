//! marihydro\crates\mh_physics\src\core\scalar.rs
//! 统一标量类型抽象 - 项目唯一权威定义 - 禁止更改此代码文件
//!
//! 本模块提供**编译期精度选择**的唯一接口，支持物理算法在 `f32` 和 `f64` 之间零成本切换。
//! 这是项目中**唯一**的精度控制层，所有运行时精度相关代码必须通过此 trait 实现泛型化。
//!
//! # 核心理念
//!
//! **架构分层原则**：
//! - **配置层**：物理常量硬编码为 `f64`（精度无关，确保配置数据权威）
//! - **运行层**：算法实现泛型化为 `<S: Scalar>`（精度相关，编译期自动适配）
//!
//! # 设计哲学
//!
//! 1. **单一职责**：仅解决精度切换问题，不定义物理常量，不重复数学运算
//! 2. **零成本抽象**：`#[inline]` + 编译期单态化，性能与手写具体类型完全相同
//! 3. **架构红线**：
//!    - 本 trait 定义在 `mh_physics::core`，不属于 `mh_foundation`
//!    - `mh_mesh` 必须硬编码 `f64`，**禁止使用本 trait**（几何精度独立于计算精度）
//!    - **GPU 内存操作**：需在 `mh_gpu` 模块中显式添加 `+ Pod` 约束
//! 4. **可扩展性**：移除 `Pod` 后，未来可支持 `f16`、`rug::Float` 等类型
//!
//! # 正确用法示例
//!
//! ```
//! use mh_physics::core::Scalar;
//! use num_traits::Float;
//! 
//! // 运行层：泛型算法
//! fn compute_wave_speed<S: Scalar>(depth: S) -> S {
//!     let g = S::from_f64(9.81);  // 配置层f64 → 运行层S
//!     (g * depth).sqrt()          // sqrt() 来自 Float trait
//! }
//!
//! // 配置层：硬编码f64
//! const GRAVITY_F64: f64 = 9.81;  // 权威配置值
//! ```
//!
//! # GPU 内存操作
//!
//! 由于 trait 已移除 `Pod`，GPU 上传需显式约束：
//!
//! ```no_run
//! // mh_gpu/src/buffer.rs
//! use bytemuck::Pod;
//!
//! fn upload_to_gpu<S: Scalar + Pod>(data: &[S]) -> GpuBuffer {
//!     bytemuck::cast_slice(data) // 零拷贝
//! }
//! ```

use std::fmt::{Debug, Display};
use std::iter::Sum;

use num_traits::{Float, FromPrimitive, NumAssign};

/// 统一标量类型约束 - 项目唯一权威接口
///
/// 所有物理计算必须使用此 trait 作为泛型边界。**禁止**直接引用 `f32` 或 `f64` 类型。
/// 数学运算（`sqrt`/`sin` 等）全部继承自 `Float` trait，无需重复定义。
///
/// **特殊值获取**：使用 `Float` trait 的 `infinity()`、`neg_infinity()`、`min_positive_value()` 方法
///
/// # 架构约束
///
/// - **必须**：作为泛型约束使用，如 `<S: Scalar>`
/// - **禁止**：作为 trait 对象使用，如 `&dyn Scalar`
/// - **禁止**：`mh_mesh` 等几何库硬编码使用
///
/// # 实现类型
///
/// - `f32`：GPU 加速模式，内存占用减半，精度约 7 位有效数字
/// - `f64`：CPU 高精度模式（默认），精度约 15 位有效数字
pub trait Scalar:
    Float
    + FromPrimitive
    + NumAssign
    // + Pod // ❌ 已移除：职责分离，GPU 场景显式添加
    + Copy
    + Clone
    + Debug
    + Display
    + Send
    + Sync
    + Sum
    + 'static
{
    // ========== 基本数学常量 ==========

    /// 零值：`0.0`
    const ZERO: Self;

    /// 单位值：`1.0`
    const ONE: Self;

    /// 机器精度（Machine epsilon）
    ///
    /// - f32: `1.1920929e-7`
    /// - f64: `2.220446049250313e-16`
    const EPSILON: Self;

    // ========== 类型转换（唯一入口） ==========

    /// 从 **配置层** `f64` 转换到 **运行层** `S`（可能丢失精度）
    ///
    /// # Panics
    /// Debug 模式下，若 `v` 超出 `S` 范围则 panic
    fn from_f64(v: f64) -> Self;

    /// 转换回 `f64`（用于输出或跨模块接口）
    fn to_f64(self) -> f64;
}

// ============================================================================
// f32 实现
// ============================================================================

impl Scalar for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const EPSILON: f32 = f32::EPSILON; // 正确机器精度

    #[inline]
    fn from_f64(v: f64) -> Self {
        debug_assert!(
            v >= f64::from(f32::MIN) && v <= f64::from(f32::MAX),
            "f64 值 {} 超出 f32 范围 [{}, {}]",
            v,
            f32::MIN,
            f32::MAX
        );
        v as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

// ============================================================================
// f64 实现
// ============================================================================

impl Scalar for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const EPSILON: f64 = f64::EPSILON; 

    #[inline]
    fn from_f64(v: f64) -> Self {
        v // 精确转换
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

// ============================================================================
// 单元测试（系统化容差）
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Float;

    // 容差标准：f32用绝对误差，f64用相对误差
    const TOL_F32: f32 = 1e-6;
    const TOL_F64: f64 = 1e-14;

    #[test]
    fn test_f32_basic() {
        let v: f32 = <f32 as Scalar>::from_f64(1.23456789012345);
        assert!((v - 1.2345679).abs() < TOL_F32);
    }

    #[test]
    fn test_f64_basic() {
        let v: f64 = <f64 as Scalar>::from_f64(1.23456789012345);
        assert_eq!(v, 1.23456789012345);
    }

    #[test]
    fn test_float_operations() {
        // 验证 Float trait 方法可用
        let x: f32 = <f32 as Scalar>::from_f64(9.0);
        assert!((x.sqrt() - 3.0).abs() < TOL_F32); // sqrt() 来自 Float
        
        let y: f64 = <f64 as Scalar>::from_f64(2.0);
        assert!((y.powf(3.0) - 8.0).abs() < TOL_F64); // powf() 来自 Float
    }

    #[test]
    fn test_special_values() {
        // 验证 Float trait 的方法
        assert!(f32::neg_infinity().is_sign_negative());
        assert!(f64::infinity().is_sign_positive());
        assert!(f64::min_positive_value().is_normal());
    }

    #[test]
    fn test_pod_constraint_removed() {
        // 验证 Scalar 不再强制要求 Pod
        fn pure_computation<S: Scalar>(x: S) -> S {
            x * S::from_f64(2.0)
        }

        let result_f32 = pure_computation(10.0f32);
        let result_f64 = pure_computation(10.0f64);
        
        assert_eq!(result_f32, 20.0);
        assert_eq!(result_f64, 20.0);
    }
}