//! marihydro\crates\mh_physics\src\tracer\boundary.rs
//! 示踪剂边界条件模块
//!
//! 提供示踪剂输运的边界条件管理，支持：
//! - Dirichlet 边界（固定浓度）
//! - Neumann 边界（固定通量/梯度）
//! - Robin 边界（混合边界条件）
//! - 时变边界（支持 TimeSeries 驱动）
//!
//! # 使用示例
//!
//! ```rust
//! use mh_runtime::RuntimeScalar as Scalar;
//! use mh_physics::tracer::boundary::{
//!     TracerBoundaryManager, TracerBoundaryCondition, TracerBoundaryType
//! };
//!
//! // 显式指定运行时精度（推荐）
//! let mut manager: TracerBoundaryManager<f64> = TracerBoundaryManager::new(100);
//!
//! // 设置入口固定浓度（35 psu）
//! manager.set_boundary(0, TracerBoundaryCondition::dirichlet(35.0));
//!
//! // 设置出口零梯度
//! manager.set_boundary(99, TracerBoundaryCondition::zero_gradient());
//!
//! // 获取边界值（需先更新时间缓存）
//! let time = 0.0; // 模拟时间
//! manager.update_cache(time);
//! let bc = manager.get(0, time);
//! assert_eq!(bc.bc_type, TracerBoundaryType::Dirichlet);
//! assert!((bc.value - 35.0).abs() < 1e-10);
//!
//! // GPU 加速模式（f32）
//! let mut manager_f32: TracerBoundaryManager<f32> = TracerBoundaryManager::new(50);
//! manager_f32.set_boundary(0, TracerBoundaryCondition::dirichlet(35.0f32));
//! ```

use crate::forcing::timeseries::TimeSeries;
use crate::types::BoundaryValueProvider;
use mh_runtime::RuntimeScalar as Scalar;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// 边界条件类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TracerBoundaryType {
    /// Dirichlet 边界（固定值）
    Dirichlet,
    /// Neumann 边界（固定梯度/通量）
    Neumann,
    /// Robin 边界（混合条件）: α*c + β*∂c/∂n = γ
    Robin,
    /// 零梯度（自然边界）
    ZeroGradient,
    /// 内部边界（无边界条件）
    Internal,
}

/// 边界条件数据（配置层，硬编码 f64）
#[derive(Debug, Clone)]
pub enum TracerBoundaryConditionConfig {
    /// Dirichlet: 固定浓度值
    Dirichlet(f64),

    /// Dirichlet (时变): 浓度随时间变化
    DirichletTimeSeries(Arc<TimeSeries>),

    /// Neumann: 固定法向通量 [单位/s]
    Neumann(f64),

    /// Neumann (时变): 通量随时间变化
    NeumannTimeSeries(Arc<TimeSeries>),

    /// Robin: α*c + β*∂c/∂n = γ
    Robin {
        alpha: f64,
        beta: f64,
        gamma: f64,
    },

    /// 零梯度边界
    ZeroGradient,
}

impl TracerBoundaryConditionConfig {
    /// 创建 Dirichlet 边界条件（固定浓度）
    pub fn dirichlet(value: f64) -> Self {
        Self::Dirichlet(value)
    }

    /// 创建时变 Dirichlet 边界条件
    pub fn dirichlet_timeseries(ts: TimeSeries) -> Self {
        Self::DirichletTimeSeries(Arc::new(ts))
    }

    /// 创建 Neumann 边界条件（固定通量）
    pub fn neumann(flux: f64) -> Self {
        Self::Neumann(flux)
    }

    /// 创建时变 Neumann 边界条件
    pub fn neumann_timeseries(ts: TimeSeries) -> Self {
        Self::NeumannTimeSeries(Arc::new(ts))
    }

    /// 创建 Robin 边界条件
    pub fn robin(alpha: f64, beta: f64, gamma: f64) -> Self {
        Self::Robin { alpha, beta, gamma }
    }

    /// 创建零梯度边界条件
    pub fn zero_gradient() -> Self {
        Self::ZeroGradient
    }

    /// 获取边界类型
    pub fn boundary_type(&self) -> TracerBoundaryType {
        match self {
            Self::Dirichlet(_) | Self::DirichletTimeSeries(_) => TracerBoundaryType::Dirichlet,
            Self::Neumann(_) | Self::NeumannTimeSeries(_) => TracerBoundaryType::Neumann,
            Self::Robin { .. } => TracerBoundaryType::Robin,
            Self::ZeroGradient => TracerBoundaryType::ZeroGradient,
        }
    }

    /// 在给定时刻评估边界值
    ///
    /// 对于 Dirichlet: 返回浓度值
    /// 对于 Neumann: 返回通量值
    /// 对于 Robin: 返回 gamma 值
    /// 对于 ZeroGradient: 返回 0.0
    pub fn evaluate(&self, time: f64) -> f64 {
        match self {
            Self::Dirichlet(v) => *v,
            Self::DirichletTimeSeries(ts) => ts.get_value(time),
            Self::Neumann(flux) => *flux,
            Self::NeumannTimeSeries(ts) => ts.get_value(time),
            Self::Robin { gamma, .. } => *gamma,
            Self::ZeroGradient => 0.0,
        }
    }

    /// 获取 Robin 系数
    pub fn robin_coefficients(&self) -> Option<(f64, f64, f64)> {
        if let Self::Robin { alpha, beta, gamma } = self {
            Some((*alpha, *beta, *gamma))
        } else {
            None
        }
    }

    /// 转换为运行时精度
    pub fn to_precision<S: Scalar>(&self) -> TracerBoundaryCondition<S> {
        match *self {
            Self::Dirichlet(v) => TracerBoundaryCondition::Dirichlet(S::from_f64(v).unwrap_or(S::ZERO)),
            Self::DirichletTimeSeries(ref ts) => {
                TracerBoundaryCondition::DirichletTimeSeries(ts.clone())
            }
            Self::Neumann(flux) => TracerBoundaryCondition::Neumann(S::from_f64(flux).unwrap_or(S::ZERO)),
            Self::NeumannTimeSeries(ref ts) => {
                TracerBoundaryCondition::NeumannTimeSeries(ts.clone())
            }
            Self::Robin { alpha, beta, gamma } => TracerBoundaryCondition::Robin {
                alpha: S::from_f64(alpha).unwrap_or(S::ZERO),
                beta: S::from_f64(beta).unwrap_or(S::ZERO),
                gamma: S::from_f64(gamma).unwrap_or(S::ZERO),
            },
            Self::ZeroGradient => TracerBoundaryCondition::ZeroGradient,
        }
    }
}

/// 边界条件数据（运行层，泛型化）
#[derive(Debug, Clone)]
pub enum TracerBoundaryCondition<S: Scalar> {
    /// Dirichlet: 固定浓度值
    Dirichlet(S),

    /// Dirichlet (时变): 浓度随时间变化
    DirichletTimeSeries(Arc<TimeSeries>),

    /// Neumann: 固定法向通量 [单位/s]
    Neumann(S),

    /// Neumann (时变): 通量随时间变化
    NeumannTimeSeries(Arc<TimeSeries>),

    /// Robin: α*c + β*∂c/∂n = γ
    Robin {
        alpha: S,
        beta: S,
        gamma: S,
    },

    /// 零梯度边界
    ZeroGradient,
}

impl<S: Scalar> TracerBoundaryCondition<S> {
    /// 创建 Dirichlet 边界条件（固定浓度）
    pub fn dirichlet(value: S) -> Self {
        Self::Dirichlet(value)
    }

    /// 创建时变 Dirichlet 边界条件
    pub fn dirichlet_timeseries(ts: TimeSeries) -> Self {
        Self::DirichletTimeSeries(Arc::new(ts))
    }

    /// 创建 Neumann 边界条件（固定通量）
    pub fn neumann(flux: S) -> Self {
        Self::Neumann(flux)
    }

    /// 创建时变 Neumann 边界条件
    pub fn neumann_timeseries(ts: TimeSeries) -> Self {
        Self::NeumannTimeSeries(Arc::new(ts))
    }

    /// 创建 Robin 边界条件
    pub fn robin(alpha: S, beta: S, gamma: S) -> Self {
        Self::Robin { alpha, beta, gamma }
    }

    /// 创建零梯度边界条件
    pub fn zero_gradient() -> Self {
        Self::ZeroGradient
    }

    /// 获取边界类型
    pub fn boundary_type(&self) -> TracerBoundaryType {
        match self {
            Self::Dirichlet(_) | Self::DirichletTimeSeries(_) => TracerBoundaryType::Dirichlet,
            Self::Neumann(_) | Self::NeumannTimeSeries(_) => TracerBoundaryType::Neumann,
            Self::Robin { .. } => TracerBoundaryType::Robin,
            Self::ZeroGradient => TracerBoundaryType::ZeroGradient,
        }
    }

    /// 在给定时刻评估边界值
    ///
    /// 对于 Dirichlet: 返回浓度值
    /// 对于 Neumann: 返回通量值
    /// 对于 Robin: 返回 gamma 值
    /// 对于 ZeroGradient: 返回 0.0
    pub fn evaluate(&self, time: f64) -> S {
        match self {
            Self::Dirichlet(v) => *v,
            Self::DirichletTimeSeries(ts) => S::from_f64(ts.get_value(time)).unwrap_or(S::ZERO),
            Self::Neumann(flux) => *flux,
            Self::NeumannTimeSeries(ts) => S::from_f64(ts.get_value(time)).unwrap_or(S::ZERO),
            Self::Robin { gamma, .. } => *gamma,
            Self::ZeroGradient => S::ZERO,
        }
    }

    /// 获取 Robin 系数
    pub fn robin_coefficients(&self) -> Option<(S, S, S)> {
        if let Self::Robin { alpha, beta, gamma } = self {
            Some((*alpha, *beta, *gamma))
        } else {
            None
        }
    }
}

/// 已解析的边界值（泛型化）
#[derive(Debug, Clone, Copy)]
pub struct ResolvedBoundaryValue<S: Scalar> {
    /// 边界类型
    pub bc_type: TracerBoundaryType,
    /// 主值（浓度/通量/gamma）
    pub value: S,
    /// Robin alpha 系数（仅 Robin 类型有效）
    pub alpha: S,
    /// Robin beta 系数（仅 Robin 类型有效）
    pub beta: S,
}

impl<S: Scalar> Default for ResolvedBoundaryValue<S> {
    fn default() -> Self {
        Self {
            bc_type: TracerBoundaryType::ZeroGradient,
            value: S::ZERO,
            alpha: S::ZERO,
            beta: S::ZERO,
        }
    }
}

impl<S: Scalar> ResolvedBoundaryValue<S> {
    /// 创建零梯度边界
    pub fn zero_gradient() -> Self {
        Self::default()
    }

    /// 创建 Dirichlet 边界
    pub fn dirichlet(value: S) -> Self {
        Self {
            bc_type: TracerBoundaryType::Dirichlet,
            value,
            alpha: S::ZERO,
            beta: S::ZERO,
        }
    }

    /// 创建 Neumann 边界
    pub fn neumann(flux: S) -> Self {
        Self {
            bc_type: TracerBoundaryType::Neumann,
            value: flux,
            alpha: S::ZERO,
            beta: S::ZERO,
        }
    }

    /// 创建 Robin 边界
    pub fn robin(alpha: S, beta: S, gamma: S) -> Self {
        Self {
            bc_type: TracerBoundaryType::Robin,
            value: gamma,
            alpha,
            beta,
        }
    }

    /// 计算隐式矩阵贡献（对角项）
    ///
    /// 对于 Robin BC: αc + β∂c/∂n = γ
    /// 隐式化后对角修正为: 1 + dt * α / β
    pub fn implicit_diagonal_contribution(&self, _dt: S, dx: S) -> S {
        match self.bc_type {
            TracerBoundaryType::Robin => {
                if self.beta.abs() > S::from_f64(1e-14).unwrap_or(S::ZERO) {
                    // Robin: α/β * dx 贡献到对角
                    self.alpha / self.beta * dx
                } else {
                    // 退化为 Dirichlet
                    S::from_f64(1e14).unwrap_or(S::ZERO) // 强制约束
                }
            }
            TracerBoundaryType::Dirichlet => {
                // Dirichlet 施加于对角
                S::from_f64(1e14).unwrap_or(S::ZERO)
            }
            _ => S::ZERO,
        }
    }

    /// 计算隐式 RHS 贡献
    pub fn implicit_rhs_contribution(&self, dx: S, _c_interior: S) -> S {
        match self.bc_type {
            TracerBoundaryType::Robin => {
                if self.beta.abs() > S::from_f64(1e-14).unwrap_or(S::ZERO) {
                    // γ/β * dx
                    self.value / self.beta * dx
                } else {
                    self.value * S::from_f64(1e14).unwrap_or(S::ZERO)
                }
            }
            TracerBoundaryType::Dirichlet => {
                self.value * S::from_f64(1e14).unwrap_or(S::ZERO)
            }
            TracerBoundaryType::Neumann => {
                // 通量直接加入 RHS
                self.value * dx
            }
            _ => S::ZERO,
        }
    }

    /// 计算边界面浓度值
    ///
    /// 使用 Robin 公式反算边界浓度
    pub fn compute_boundary_concentration(
        &self,
        c_interior: S,
        grad_n: S,
    ) -> S {
        match self.bc_type {
            TracerBoundaryType::Dirichlet => self.value,
            TracerBoundaryType::ZeroGradient => c_interior,
            TracerBoundaryType::Neumann => {
                // c_b = c_i + grad_n * dx (需要外部提供 dx)
                c_interior
            }
            TracerBoundaryType::Robin => {
                // αc + β·grad_n = γ => c = (γ - β·grad_n) / α
                if self.alpha.abs() > S::from_f64(1e-14).unwrap_or(S::ZERO) {
                    (self.value - self.beta * grad_n) / self.alpha
                } else {
                    // 退化为 Neumann
                    c_interior
                }
            }
            TracerBoundaryType::Internal => c_interior,
        }
    }
}

impl BoundaryValueProvider<f64> for TracerBoundaryConditionConfig {
    fn get_value(&self, _face_idx: usize, time: f64) -> Option<f64> {
        Some(self.evaluate(time))
    }
}

impl<S: Scalar> BoundaryValueProvider<S> for TracerBoundaryCondition<S> {
    fn get_value(&self, _face_idx: usize, time: f64) -> Option<S> {
        Some(self.evaluate(time))
    }
}

/// 示踪剂边界条件管理器（泛型化）
///
/// 管理边界面上的示踪剂边界条件，支持时变边界
pub struct TracerBoundaryManager<S: Scalar> {
    /// 边界面数量
    n_boundary_faces: usize,
    /// 边界条件映射：面索引 -> 边界条件
    conditions: HashMap<usize, TracerBoundaryCondition<S>>,
    /// 默认边界条件
    default_condition: TracerBoundaryCondition<S>,
    /// 缓存的解析值
    resolved_cache: Vec<ResolvedBoundaryValue<S>>,
    /// 缓存时间戳
    cache_time: Option<f64>,
}

impl<S: Scalar> TracerBoundaryManager<S> {
    /// 创建新的边界条件管理器
    ///
    /// # 参数
    ///
    /// - `n_boundary_faces`: 边界面数量
    pub fn new(n_boundary_faces: usize) -> Self {
        Self {
            n_boundary_faces,
            conditions: HashMap::new(),
            default_condition: TracerBoundaryCondition::ZeroGradient,
            resolved_cache: vec![ResolvedBoundaryValue::zero_gradient(); n_boundary_faces],
            cache_time: None,
        }
    }

    /// 设置默认边界条件
    pub fn set_default(&mut self, condition: TracerBoundaryCondition<S>) {
        self.default_condition = condition;
        self.invalidate_cache();
    }

    /// 设置特定边界面的边界条件
    ///
    /// # 参数
    ///
    /// - `boundary_face_idx`: 边界面索引
    /// - `condition`: 边界条件
    pub fn set_boundary(&mut self, boundary_face_idx: usize, condition: TracerBoundaryCondition<S>) {
        if boundary_face_idx < self.n_boundary_faces {
            self.conditions.insert(boundary_face_idx, condition);
            self.invalidate_cache();
        }
    }

    /// 批量设置边界条件
    ///
    /// # 参数
    ///
    /// - `face_indices`: 边界面索引列表
    /// - `condition`: 边界条件
    pub fn set_boundaries(&mut self, face_indices: &[usize], condition: TracerBoundaryCondition<S>) {
        for &idx in face_indices {
            if idx < self.n_boundary_faces {
                self.conditions.insert(idx, condition.clone());
            }
        }
        self.invalidate_cache();
    }

    /// 移除边界条件（恢复默认）
    pub fn remove_boundary(&mut self, boundary_face_idx: usize) {
        self.conditions.remove(&boundary_face_idx);
        self.invalidate_cache();
    }

    /// 清空所有边界条件
    pub fn clear(&mut self) {
        self.conditions.clear();
        self.invalidate_cache();
    }

    /// 获取边界条件引用
    pub fn get_condition(&self, boundary_face_idx: usize) -> &TracerBoundaryCondition<S> {
        self.conditions
            .get(&boundary_face_idx)
            .unwrap_or(&self.default_condition)
    }

    /// 获取解析后的边界值
    ///
    /// # 参数
    ///
    /// - `boundary_face_idx`: 边界面索引
    /// - `time`: 当前时间
    pub fn get(&self, boundary_face_idx: usize, time: f64) -> ResolvedBoundaryValue<S> {
        self.resolve_at(boundary_face_idx, time)
    }

    /// 解析单个边界值
    fn resolve_at(&self, boundary_face_idx: usize, time: f64) -> ResolvedBoundaryValue<S> {
        let cond = self.get_condition(boundary_face_idx);
        match cond {
            TracerBoundaryCondition::Dirichlet(v) => ResolvedBoundaryValue::dirichlet(*v),
            TracerBoundaryCondition::DirichletTimeSeries(ts) => {
                ResolvedBoundaryValue::dirichlet(S::from_f64(ts.get_value(time)).unwrap_or(S::ZERO))
            }
            TracerBoundaryCondition::Neumann(flux) => ResolvedBoundaryValue::neumann(*flux),
            TracerBoundaryCondition::NeumannTimeSeries(ts) => {
                ResolvedBoundaryValue::neumann(S::from_f64(ts.get_value(time)).unwrap_or(S::ZERO))
            }
            TracerBoundaryCondition::Robin { alpha, beta, gamma } => {
                ResolvedBoundaryValue::robin(*alpha, *beta, *gamma)
            }
            TracerBoundaryCondition::ZeroGradient => ResolvedBoundaryValue::zero_gradient(),
        }
    }

    /// 更新缓存
    ///
    /// 对于时变边界，需要在每个时间步调用以更新缓存
    pub fn update_cache(&mut self, time: f64) {
        // 检查是否需要更新
        if let Some(cached_time) = self.cache_time {
            if (cached_time - time).abs() < 1e-14 {
                return;
            }
        }

        // 更新所有边界值
        for idx in 0..self.n_boundary_faces {
            self.resolved_cache[idx] = self.resolve_at(idx, time);
        }

        self.cache_time = Some(time);
    }

    /// 获取缓存的边界值切片
    ///
    /// 注意：需要先调用 `update_cache` 更新缓存
    pub fn cached_values(&self) -> &[ResolvedBoundaryValue<S>] {
        &self.resolved_cache
    }

    /// 使缓存无效
    fn invalidate_cache(&mut self) {
        self.cache_time = None;
    }

    /// 获取边界面数量
    pub fn n_boundary_faces(&self) -> usize {
        self.n_boundary_faces
    }

    /// 获取已设置的边界条件数量
    pub fn n_conditions(&self) -> usize {
        self.conditions.len()
    }

    /// 检查是否有时变边界条件
    pub fn has_time_varying(&self) -> bool {
        self.conditions.values().any(|c| {
            matches!(
                c,
                TracerBoundaryCondition::DirichletTimeSeries(_)
                    | TracerBoundaryCondition::NeumannTimeSeries(_)
            )
        })
    }

    /// 获取所有 Dirichlet 边界面索引
    pub fn dirichlet_faces(&self) -> Vec<usize> {
        self.conditions
            .iter()
            .filter(|(_, c)| c.boundary_type() == TracerBoundaryType::Dirichlet)
            .map(|(&idx, _)| idx)
            .collect()
    }

    /// 获取所有 Neumann 边界面索引
    pub fn neumann_faces(&self) -> Vec<usize> {
        self.conditions
            .iter()
            .filter(|(_, c)| c.boundary_type() == TracerBoundaryType::Neumann)
            .map(|(&idx, _)| idx)
            .collect()
    }
}

impl<S: Scalar> Default for TracerBoundaryManager<S> {
    fn default() -> Self {
        Self::new(0)
    }
}

/// 边界条件构建器
pub struct TracerBoundaryBuilder<S: Scalar> {
    manager: TracerBoundaryManager<S>,
}

impl<S: Scalar> TracerBoundaryBuilder<S> {
    /// 创建新的构建器
    pub fn new(n_boundary_faces: usize) -> Self {
        Self {
            manager: TracerBoundaryManager::new(n_boundary_faces),
        }
    }

    /// 设置默认边界条件
    pub fn default_condition(mut self, condition: TracerBoundaryCondition<S>) -> Self {
        self.manager.set_default(condition);
        self
    }

    /// 添加单个边界条件
    pub fn add(mut self, face_idx: usize, condition: TracerBoundaryCondition<S>) -> Self {
        self.manager.set_boundary(face_idx, condition);
        self
    }

    /// 添加批量边界条件
    pub fn add_many(mut self, face_indices: &[usize], condition: TracerBoundaryCondition<S>) -> Self {
        self.manager.set_boundaries(face_indices, condition);
        self
    }

    /// 构建边界管理器
    pub fn build(self) -> TracerBoundaryManager<S> {
        self.manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirichlet_boundary() {
        let bc: TracerBoundaryCondition<f64> = TracerBoundaryCondition::dirichlet(35.0);
        assert_eq!(bc.boundary_type(), TracerBoundaryType::Dirichlet);
        assert!((bc.evaluate(0.0) - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_neumann_boundary() {
        let bc: TracerBoundaryCondition<f64> = TracerBoundaryCondition::neumann(-0.01);
        assert_eq!(bc.boundary_type(), TracerBoundaryType::Neumann);
        assert!((bc.evaluate(0.0) - (-0.01)).abs() < 1e-10);
    }

    #[test]
    fn test_robin_boundary() {
        let bc: TracerBoundaryCondition<f64> = TracerBoundaryCondition::robin(1.0, 0.1, 5.0);
        assert_eq!(bc.boundary_type(), TracerBoundaryType::Robin);
        assert_eq!(bc.robin_coefficients(), Some((1.0, 0.1, 5.0)));
    }

    #[test]
    fn test_zero_gradient() {
        let bc: TracerBoundaryCondition<f64> = TracerBoundaryCondition::zero_gradient();
        assert_eq!(bc.boundary_type(), TracerBoundaryType::ZeroGradient);
        assert!((bc.evaluate(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_timeseries_dirichlet() {
        let ts = TimeSeries::from_points(vec![(0.0, 30.0), (10.0, 35.0), (20.0, 32.0)]);
        let bc: TracerBoundaryCondition<f64> = TracerBoundaryCondition::dirichlet_timeseries(ts);

        assert_eq!(bc.boundary_type(), TracerBoundaryType::Dirichlet);
        assert!((bc.evaluate(5.0) - 32.5).abs() < 1e-10);
    }

    #[test]
    fn test_manager_basic() {
        let mut manager: TracerBoundaryManager<f64> = TracerBoundaryManager::new(10);

        manager.set_boundary(0, TracerBoundaryCondition::dirichlet(35.0));
        manager.set_boundary(9, TracerBoundaryCondition::neumann(-0.01));

        let bc0 = manager.get(0, 0.0);
        assert_eq!(bc0.bc_type, TracerBoundaryType::Dirichlet);
        assert!((bc0.value - 35.0).abs() < 1e-10);

        let bc9 = manager.get(9, 0.0);
        assert_eq!(bc9.bc_type, TracerBoundaryType::Neumann);
        assert!((bc9.value - (-0.01)).abs() < 1e-10);

        // 未设置的边界使用默认值
        let bc5 = manager.get(5, 0.0);
        assert_eq!(bc5.bc_type, TracerBoundaryType::ZeroGradient);
    }

    #[test]
    fn test_manager_cache() {
        let mut manager: TracerBoundaryManager<f64> = TracerBoundaryManager::new(5);
        manager.set_boundary(0, TracerBoundaryCondition::dirichlet(10.0));

        manager.update_cache(0.0);
        let cached = manager.cached_values();
        assert!((cached[0].value - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder() {
        let manager: TracerBoundaryManager<f64> = TracerBoundaryBuilder::new(10)
            .default_condition(TracerBoundaryCondition::zero_gradient())
            .add(0, TracerBoundaryCondition::dirichlet(35.0))
            .add_many(&[8, 9], TracerBoundaryCondition::neumann(0.0))
            .build();

        assert_eq!(manager.n_conditions(), 3);
        assert!(manager.dirichlet_faces().contains(&0));
        assert!(manager.neumann_faces().contains(&8));
        assert!(manager.neumann_faces().contains(&9));
    }

    #[test]
    fn test_time_varying_detection() {
        let mut manager: TracerBoundaryManager<f64> = TracerBoundaryManager::new(5);
        assert!(!manager.has_time_varying());

        let ts = TimeSeries::from_points(vec![(0.0, 30.0), (10.0, 35.0)]);
        manager.set_boundary(0, TracerBoundaryCondition::dirichlet_timeseries(ts));
        assert!(manager.has_time_varying());
    }

    #[test]
    fn test_precision_conversion_f32() {
        let config_bc = TracerBoundaryConditionConfig::dirichlet(35.0);
        let runtime_bc: TracerBoundaryCondition<f32> = config_bc.to_precision();
        assert!((runtime_bc.evaluate(0.0) - 35.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_precision_conversion_f64() {
        let config_bc = TracerBoundaryConditionConfig::dirichlet(35.0);
        let runtime_bc: TracerBoundaryCondition<f64> = config_bc.to_precision();
        assert!((runtime_bc.evaluate(0.0) - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_robin_precision_conversion() {
        let config_bc = TracerBoundaryConditionConfig::robin(1.0, 0.1, 5.0);
        let runtime_bc: TracerBoundaryCondition<f32> = config_bc.to_precision();
        let coeffs = runtime_bc.robin_coefficients().unwrap();
        assert!((coeffs.0 - 1.0_f32).abs() < 1e-6);
        assert!((coeffs.1 - 0.1_f32).abs() < 1e-6);
        assert!((coeffs.2 - 5.0_f32).abs() < 1e-6);
    }
}