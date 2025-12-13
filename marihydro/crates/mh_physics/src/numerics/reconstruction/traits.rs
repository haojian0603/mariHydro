//! 重构 trait 定义
//!
//! **层级**: Layer 3 - Engine Layer
//!
//! 本模块提供泛型化的重构器接口，支持 f32/f64 精度切换。

use glam::DVec2;
use mh_core::RuntimeScalar;

// ============================================================
// 泛型重构状态
// ============================================================

/// 重构后的面状态值 - 泛型版本
///
/// 包含面两侧（左/右）的重构值，用于通量计算。
#[derive(Debug, Clone, Copy)]
pub struct ReconstructedStateGeneric<S: RuntimeScalar> {
    /// 左侧单元的重构值
    pub left: S,
    
    /// 右侧单元的重构值
    pub right: S,
}

/// 重构后的面状态值 - Legacy f64 版本
pub type ReconstructedState = ReconstructedStateGeneric<f64>;

impl<S: RuntimeScalar> ReconstructedStateGeneric<S> {
    /// 创建新的重构状态
    pub fn new(left: S, right: S) -> Self {
        Self { left, right }
    }
    
    /// 从单个值创建（一阶精度）
    pub fn from_values(left: S, right: S) -> Self {
        Self { left, right }
    }
    
    /// 计算面平均值
    pub fn average(&self) -> S {
        S::HALF * (self.left + self.right)
    }
    
    /// 计算跳跃 (right - left)
    pub fn jump(&self) -> S {
        self.right - self.left
    }
    
    /// 计算绝对最大值
    pub fn max_abs(&self) -> S {
        self.left.abs().max(self.right.abs())
    }
    
    /// 确保正定（用于水深）
    pub fn ensure_positive(&mut self, min_value: S) {
        if self.left < min_value {
            self.left = min_value;
        }
        if self.right < min_value {
            self.right = min_value;
        }
    }
}

impl<S: RuntimeScalar> Default for ReconstructedStateGeneric<S> {
    fn default() -> Self {
        Self { left: S::ZERO, right: S::ZERO }
    }
}

// ============================================================
// 泛型重构器 Trait
// ============================================================

/// 重构器 trait - 泛型版本
///
/// 所有重构方案实现此 trait。
pub trait ReconstructorGeneric<S: RuntimeScalar>: Send + Sync {
    /// 计算所有单元的梯度
    fn compute_gradients(&mut self, values: &[S]);
    
    /// 重构标量场的面值
    ///
    /// # Arguments
    /// * `face_id` - 面索引
    /// * `values` - 单元中心值
    ///
    /// # Returns
    /// 面两侧的重构值
    fn reconstruct_scalar(&self, face_id: usize, values: &[S]) -> ReconstructedStateGeneric<S>;
    
    /// 获取限制后的梯度 (返回元组)
    fn get_limited_gradient_tuple(&self, cell_id: usize) -> (S, S);
    
    /// 是否启用二阶精度
    fn is_second_order(&self) -> bool;
    
    /// 重构器名称
    fn name(&self) -> &'static str;
}

/// 重构器 trait - Legacy f64 版本
///
/// 所有重构方案实现此 trait。
pub trait Reconstructor: Send + Sync {
    /// 计算所有单元的梯度
    fn compute_gradients(&mut self, values: &[f64]);
    
    /// 重构标量场的面值
    fn reconstruct_scalar(&self, face_id: usize, values: &[f64]) -> ReconstructedState;
    
    /// 获取限制后的梯度
    fn get_limited_gradient(&self, cell_id: usize) -> DVec2;
    
    /// 是否启用二阶精度
    fn is_second_order(&self) -> bool;
    
    /// 重构器名称
    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reconstructed_state() {
        let state = ReconstructedState::new(1.0, 2.0);
        assert_eq!(state.left, 1.0);
        assert_eq!(state.right, 2.0);
        assert_eq!(state.average(), 1.5);
        assert_eq!(state.jump(), 1.0);
    }
    
    #[test]
    fn test_reconstructed_state_ensure_positive() {
        let mut state = ReconstructedState::new(-0.1, 0.5);
        state.ensure_positive(0.0);
        assert_eq!(state.left, 0.0);
        assert_eq!(state.right, 0.5);
    }
}
