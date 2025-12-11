// crates/mh_physics/src/tracer/mod.rs

//! 示踪剂模块
//!
//! 本模块提供水动力学模拟中的示踪剂输运功能：
//!
//! # 子模块
//!
//! - [`state`]: 示踪剂状态和场定义
//! - [`transport`]: 对流-扩散求解器
//! - [`boundary`]: 边界条件管理
//! - [`diffusion`]: 扩散算子
//!
//! # 主要类型
//!
//! ## 状态类型
//!
//! - [`TracerType`]: 示踪剂类型枚举（盐度、温度、泥沙等）
//! - [`TracerProperties`]: 示踪剂物理属性
//! - [`TracerField`]: 单个示踪剂的场数据
//! - [`TracerState`]: 多示踪剂集合状态
//!
//! ## 输运求解器
//!
//! - [`TracerTransportSolver`]: 单示踪剂输运求解器
//! - [`MultiTracerSolver`]: 多示踪剂输运求解器
//! - [`TracerAdvectionScheme`]: 对流格式选项
//! - [`TracerDiffusionConfig`]: 扩散配置
//!
//! ## 边界条件
//!
//! - [`TracerBoundaryManager`]: 边界条件管理器
//! - [`TracerBoundaryCondition`]: 边界条件类型
//! - [`TracerBoundaryType`]: 边界类型枚举
//!
//! ## 扩散算子
//!
//! - [`DiffusionOperator`]: 各向同性/湍流扩散算子
//! - [`AnisotropicDiffusionOperator`]: 各向异性扩散算子
//! - [`DiffusionCoefficient`]: 扩散系数类型
//!
//! # 使用示例
//!
//! ## 创建示踪剂状态
//!
//! ```ignore
//! use mh_physics::tracer::{TracerState, TracerProperties, TracerType};
//!
//! let mut state = TracerState::new(1000); // 1000 个计算单元
//!
//! // 添加盐度和温度
//! state.add_tracer(TracerProperties::salinity()).unwrap();
//! state.add_tracer(TracerProperties::temperature()).unwrap();
//!
//! // 访问特定示踪剂
//! if let Some(salinity) = state.get_mut(TracerType::Salinity) {
//!     salinity.set_concentration(0, 35.0);
//! }
//! ```
//!
//! ## 使用输运求解器
//!
//! ```ignore
//! use mh_physics::tracer::{
//!     TracerTransportSolver, TracerTransportConfig,
//!     TracerAdvectionScheme, TracerDiffusionConfig,
//! };
//!
//! let config = TracerTransportConfig {
//!     advection_scheme: TracerAdvectionScheme::TvdVanLeer,
//!     diffusion: TracerDiffusionConfig::constant(10.0),
//!     ..Default::default()
//! };
//!
//! let mut solver = TracerTransportSolver::new(config);
//! ```
//!
//! # 设计原则
//!
//! 1. **分离状态和求解**：TracerState 只存储数据，不包含求解逻辑
//! 2. **可扩展性**：支持多种示踪剂类型和对流格式
//! 3. **性能优先**：使用守恒形式，批量处理面通量
//! 4. **与水动力耦合**：共享相同的网格和时间积分器

pub mod boundary;
pub mod diffusion;
pub mod state;
pub mod transport;
pub mod settling;

// 从 state 模块导出
pub use state::{
    TracerError, TracerField, TracerFieldStats, TracerProperties, TracerState, TracerType,
};

// 从 transport 模块导出
pub use transport::{
    FaceFlowData, MultiTracerSolver, TracerAdvectionScheme, TracerDiffusionConfig,
    TracerFaceFlux, TracerTransportConfig, TracerTransportSolver,
};

pub use settling::{SettlingSolver, SettlingConfig, SettlingResult};

// 从 boundary 模块导出
pub use boundary::{
    ResolvedBoundaryValue, TracerBoundaryBuilder, TracerBoundaryCondition, TracerBoundaryManager,
    TracerBoundaryType,
};

// 从 diffusion 模块导出
pub use diffusion::{
    AnisotropicDiffusionOperator, DiffusionCoefficient, DiffusionConfig, DiffusionOperator,
};
