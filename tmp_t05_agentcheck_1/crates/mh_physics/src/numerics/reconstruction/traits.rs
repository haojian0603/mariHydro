// marihydro/crates/mh_physics/src/numerics/reconstruction/traits.rs
//! 重构 trait 定义
//!
//! **层级**: Layer 3 - Engine Layer
//!
//! 本模块提供泛型化的重构器接口，支持 f32/f64 精度切换。
//!
//! # 设计原则
//!
//! 1. **单轨泛型**: 所有接口基于 `RuntimeScalar` 泛型，无 Legacy f64 别名
//! 2. **Backend 无关**: 使用 `(S, S)` 元组表示向量，不依赖 glam::DVec2

use mh_runtime::{Backend, RuntimeScalar};
use num_traits::Float;

// ============================================================
// 泛型重构状态
// ============================================================

/// 重构后的面状态值 - 泛型版本
///
/// 包含面两侧（左/右）的重构值，用于通量计算。
#[derive(Debug, Clone, Copy)]
pub struct ReconstructedState<B: Backend> {
    /// 左侧单元的重构值
    pub left: B::Scalar,
    
    /// 右侧单元的重构值
    pub right: B::Scalar,
}

impl<B: Backend> ReconstructedState<B> {
    /// 创建新的重构状态
    pub fn new(left: B::Scalar, right: B::Scalar) -> Self {
        Self { left, right }
    }
    
    /// 从单个值创建（一阶精度）
    pub fn from_values(left: B::Scalar, right: B::Scalar) -> Self {
        Self { left, right }
    }
    
    /// 计算面平均值
    pub fn average(&self) -> B::Scalar {
        B::Scalar::HALF * (self.left + self.right)
    }
    
    /// 计算跳跃 (right - left)
    pub fn jump(&self) -> B::Scalar {
        self.right - self.left
    }
    
    /// 计算绝对最大值
    pub fn max_abs(&self) -> B::Scalar {
        self.left.abs().max(self.right.abs())
    }
    
    /// 确保正定（用于水深）
    pub fn ensure_positive(&mut self, min_value: B::Scalar) {
        if self.left < min_value {
            self.left = min_value;
        }
        if self.right < min_value {
            self.right = min_value;
        }
    }
}

impl<B: Backend> Default for ReconstructedState<B> {
    fn default() -> Self {
        Self { left: B::Scalar::ZERO, right: B::Scalar::ZERO }
    }
}

// ============================================================
// 泛型重构器 Trait
// ============================================================

/// 重构器 trait - 泛型版本
///
/// 所有重构方案实现此 trait。
#[allow(dead_code)]
pub trait Reconstructor<B: Backend>: Send + Sync {
    /// 计算所有单元的梯度
    fn compute_gradients(&mut self, values: &B::Buffer<B::Scalar>);
    
    /// 重构标量场的面值
    ///
    /// # Arguments
    /// * `face_id` - 面索引
    /// * `values` - 单元中心值
    ///
    /// # Returns
    /// 面两侧的重构值
    fn reconstruct_scalar(
        &self,
        face_id: usize,
        values: &B::Buffer<B::Scalar>,
    ) -> ReconstructedState<B>;
    
    /// 获取限制后的梯度 (返回元组)
    fn get_limited_gradient_tuple(&self, cell_id: usize) -> (B::Scalar, B::Scalar);
    
    /// 是否启用二阶精度
    fn is_second_order(&self) -> bool;
    
    /// 重构器名称
    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;
    
    #[test]
    fn test_reconstructed_state() {
        let state = ReconstructedState::<CpuBackend<f64>>::new(1.0, 2.0);
        assert_eq!(state.left, 1.0);
        assert_eq!(state.right, 2.0);
        assert_eq!(state.average(), 1.5);
        assert_eq!(state.jump(), 1.0);
    }
    
    #[test]
    fn test_reconstructed_state_ensure_positive() {
        let mut state = ReconstructedState::<CpuBackend<f64>>::new(-0.1, 0.5);
        state.ensure_positive(0.0);
        assert_eq!(state.left, 0.0);
        assert_eq!(state.right, 0.5);
    }
}
