//! 维度类型重导出。

//! 该模块只负责从 `mh_foundation` 重导出统一的维度类型，避免 `mh_physics` 主链重复定义维度抽象。

pub use mh_foundation::dimension::{Dimension, D2, D3};
