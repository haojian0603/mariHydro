// crates/mh_physics/src/schemes/wetting_drying/mod.rs

//! 干湿处理模块
//!
//! 提供浅水方程的干湿处理功能：
//!
//! - [`WettingDryingHandler`]: 干湿状态判定和处理
//! - [`transitions`]: 光滑过渡函数
//!
//! # 设计原则
//!
//! 1. **物理一致性**: 干区无质量和动量
//! 2. **数值稳定性**: 避免除零和极大值
//! 3. **守恒性**: 不引入虚假的质量或动量
//! 4. **平滑过渡**: 使用光滑函数避免数值振荡
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::schemes::wetting_drying::{
//!     WettingDryingHandler, WettingDryingConfig,
//!     transitions::{smooth_heaviside, TransitionCalculator}
//! };
//!
//! // 基本干湿处理
//! let handler = WettingDryingHandler::new(WettingDryingConfig::default());
//! let state = handler.classify(0.005);
//!
//! // 使用过渡函数
//! let factor = smooth_heaviside(0.005, 0.01);
//! let vel_damped = vel * factor;
//! ```

mod handler;
pub mod transitions;

pub use handler::{WetState, WettingDryingConfig, WettingDryingHandler};

// 过渡函数快捷导出
pub use transitions::{
    friction_enhancement, porosity_factor, safe_froude, safe_velocity, smooth_heaviside,
    smooth_heaviside_with_width, velocity_damping, TransitionCalculator, TransitionConfig,
    TransitionFunctionType,
};

