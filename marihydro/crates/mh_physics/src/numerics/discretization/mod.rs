// crates/mh_physics/src/numerics/discretization/mod.rs

//! 有限体积离散化模块
//!
//! 提供浅水方程隐式求解所需的离散化工具：
//!
//! # 子模块
//!
//! - [`topology`]: 网格拓扑信息（单元-面连接）
//! - [`assembler`]: 系数矩阵组装器
//! - [`back_sub`]: 回代水深更新
//!
//! # 主要类型
//!
//! ## 拓扑
//!
//! - [`CellFaceTopology`]: 单元-面拓扑关系
//! - [`FaceInfo`]: 面信息（法向、长度、距离等）
//!
//! ## 矩阵组装
//!
//! - [`PressureMatrixAssembler`]: 压力泊松方程矩阵组装
//! - [`ImplicitMomentumAssembler`]: 动量方程隐式组装
//!
//! ## 回代
//!
//! - [`DepthCorrector`]: 水深校正器
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::numerics::discretization::{
//!     CellFaceTopology, PressureMatrixAssembler,
//! };
//!
//! // 构建拓扑
//! let topo = CellFaceTopology::from_mesh(&mesh);
//!
//! // 组装压力矩阵
//! let mut assembler = PressureMatrixAssembler::new(&topo);
//! assembler.assemble(&mesh, &state, dt);
//! let matrix = assembler.matrix();
//! ```
//!
//! # 设计原则
//!
//! 1. **拓扑与几何分离**：拓扑关系一次构建，几何量每步更新
//! 2. **模式复用**：矩阵稀疏模式在初始化时确定，只更新值
//! 3. **高效组装**：基于面遍历的组装避免重复计算

pub mod assembler;
pub mod back_sub;
pub mod topology;

// 拓扑
pub use topology::{CellFaceTopology, FaceInfo, NeighborInfo};

// 组装器
pub use assembler::{ImplicitMomentumAssembler, PressureMatrixAssembler, AssemblerConfig};

// 回代
pub use back_sub::{DepthCorrector, VelocityCorrector};
