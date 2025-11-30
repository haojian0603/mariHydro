// src-tauri/src/marihydro/domain/state/mod.rs

//! 状态模块

pub mod accessors;
pub mod shallow_water;
pub mod tracer;
pub mod view;

pub use accessors::StateAccessors;
pub use shallow_water::{Flux, GradientState, ShallowWaterState};
pub use tracer::{TracerField, TracerManager, TracerProperties, TracerSource, TracerType};
pub use view::{StateView, StateViewMut};

