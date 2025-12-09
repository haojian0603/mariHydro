//! 悬移质输沙子模块
//!
//! 提供悬移质（悬浮在水中的泥沙）输运功能：
//! - [`SettlingVelocity`]: 沉降速度计算
//! - [`ResuspensionSource`]: 再悬浮/侵蚀源项
//! - [`SuspendedTransport`]: 悬移质输运求解器
//!
//! # 与 tracer 模块的关系
//!
//! 悬移质可以视为一种特殊的示踪剂（tracer），因此本模块复用
//! `tracer` 模块的对流-扩散求解器，只需实现泥沙特有的源项：
//! - 侵蚀（从床面进入水体）
//! - 沉降（从水体沉降到床面）

pub mod settling;
pub mod resuspension;
pub mod transport;

pub use settling::{SettlingFormula, SettlingVelocity, StokesSettling, DietrichSettling, VanRijnSettling};
pub use resuspension::{ErosionFormula, ResuspensionSource, SmithMcLean, GarciaParker};
pub use transport::SuspendedTransport;
