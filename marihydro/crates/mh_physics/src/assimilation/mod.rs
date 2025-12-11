// crates/mh_physics/src/assimilation/mod.rs

//! 数据同化桥接层
//! 
//! 提供AI代理层与物理核心之间的接口

mod bridge;
mod conservation;

pub use bridge::{AssimilableBridge, StateSnapshot};
pub use conservation::{ConservationChecker, ConservedQuantities, ConservationError};


use crate::tracer::TracerType;

/// 可同化状态接口（重新定义，因为mh_agent的trait不能直接用于mh_physics）
pub trait PhysicsAssimilable {
    /// 获取示踪剂可变引用
    fn get_tracer_mut(&mut self, tracer_type: TracerType) -> Option<&mut [f64]>;
    
    /// 获取速度场可变引用 (u, v)
    fn get_velocity_mut(&mut self) -> (&mut [f64], &mut [f64]);
    
    /// 获取水深可变引用
    fn get_depth_mut(&mut self) -> &mut [f64];
    
    /// 获取床面高程可变引用
    fn get_bed_elevation_mut(&mut self) -> &mut [f64];
    
    /// 单元数量
    fn n_cells(&self) -> usize;
    
    /// 单元面积
    fn cell_areas(&self) -> &[f64];
    
    /// 单元中心坐标
    fn cell_centers(&self) -> &[[f64; 2]];
    
    /// 创建状态快照（用于AI推理）
    fn create_snapshot(&self) -> StateSnapshot;
    
    /// 计算守恒量
    fn compute_conserved(&mut self) -> ConservedQuantities;
    
    /// 强制守恒
    fn enforce_conservation(&mut self, reference: &ConservedQuantities, tolerance: f64);
}
