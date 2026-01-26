// crates/mh_physics/src/assimilation/mod.rs

//! 数据同化桥接层
//! 
//! 提供AI代理层与物理核心之间的接口

mod bridge;
mod conservation;

pub use bridge::{AssimilableBridge, StateSnapshot};
pub use conservation::{ConservationChecker, ConservationError, ConservedQuantities};

use crate::tracer::TracerType;
use mh_runtime::{Backend, RuntimeScalar};
use bytemuck::Pod;

/// 可同化状态接口（Backend-first）
///
/// 彻底移除 f64 硬编码，所有字段均以 `B::Scalar`/`B::Vector2D` 表示。
pub trait PhysicsAssimilable<B: Backend>
where
    B::Vector2D: Pod,
{
    /// 获取示踪剂可变引用
    fn get_tracer_mut(&mut self, tracer_type: TracerType) -> Option<&mut [B::Scalar]>;

    /// 获取动量场可变引用 (hu, hv)
    fn get_momentum_mut(&mut self) -> (&mut [B::Scalar], &mut [B::Scalar]);

    /// 获取水深可变引用
    fn get_depth_mut(&mut self) -> &mut [B::Scalar];

    /// 获取床面高程可变引用
    fn get_bed_elevation_mut(&mut self) -> &mut [B::Scalar];

    /// 单元数量
    fn n_cells(&self) -> usize;

    /// 单元面积
    fn cell_areas(&self) -> &[B::Scalar];

    /// 单元中心坐标
    fn cell_centers(&self) -> &[B::Vector2D];

    /// 创建状态快照（用于 AI/同化管线）
    fn create_snapshot(&self) -> StateSnapshot<B>;

    /// 计算守恒量
    fn compute_conserved(&mut self) -> ConservedQuantities<B>;

    /// 强制守恒（基础版本）
    fn enforce_conservation(&mut self, reference: &ConservedQuantities<B>, tolerance: B::Scalar);

    /// 强制守恒（约束最小化版本）
    fn enforce_conservation_constrained(
        &mut self,
        reference: &ConservedQuantities<B>,
        tolerance: B::Scalar,
        constraints: &ConservationConstraints,
    ) {
        let _ = constraints;
        self.enforce_conservation(reference, tolerance)
    }
}

/// 守恒约束条件
#[derive(Debug, Clone)]
pub struct ConservationConstraints {
    /// 干单元索引（不参与修正）
    pub dry_cells: Vec<usize>,
    /// 边界单元索引（不缩放）
    pub boundary_cells: Vec<usize>,
    /// 最大速度限制
    pub max_velocity: f64,
    /// 最小水深（修正后不能低于此值）
    pub min_depth: f64,
}

impl Default for ConservationConstraints {
    fn default() -> Self {
        Self {
            dry_cells: Vec::new(),
            boundary_cells: Vec::new(),
            max_velocity: 50.0,
            min_depth: 1e-6,
        }
    }
}

/// 速度访问辅助（处理干单元）
pub struct VelocityAccessor<'a, B: Backend> {
    h: &'a [B::Scalar],
    hu: &'a [B::Scalar],
    hv: &'a [B::Scalar],
    min_depth: B::Scalar,
}

impl<'a, B: Backend> VelocityAccessor<'a, B>
where
    B::Scalar: RuntimeScalar,
{
    pub fn new(
        h: &'a [B::Scalar],
        hu: &'a [B::Scalar],
        hv: &'a [B::Scalar],
        min_depth: B::Scalar,
    ) -> Self {
        Self { h, hu, hv, min_depth }
    }

    #[inline]
    pub fn get_velocity(&self, idx: usize) -> (B::Scalar, B::Scalar) {
        let h = self.h.get(idx).copied().unwrap_or(B::Scalar::ZERO);
        if h > self.min_depth {
            let hu = self.hu.get(idx).copied().unwrap_or(B::Scalar::ZERO);
            let hv = self.hv.get(idx).copied().unwrap_or(B::Scalar::ZERO);
            (hu / h, hv / h)
        } else {
            (B::Scalar::ZERO, B::Scalar::ZERO)
        }
    }

    #[inline]
    pub fn get_speed(&self, idx: usize) -> B::Scalar {
        let (u, v) = self.get_velocity(idx);
        (u * u + v * v).safe_sqrt()
    }
}
