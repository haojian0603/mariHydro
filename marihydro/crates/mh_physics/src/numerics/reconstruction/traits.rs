//! 重构 trait 定义

use glam::DVec2;

/// 重构后的面状态值
///
/// 包含面两侧（左/右）的重构值，用于通量计算。
#[derive(Debug, Clone, Copy)]
pub struct ReconstructedState {
    /// 左侧单元的重构值
    pub left: f64,
    
    /// 右侧单元的重构值
    pub right: f64,
}

impl ReconstructedState {
    /// 创建新的重构状态
    pub fn new(left: f64, right: f64) -> Self {
        Self { left, right }
    }
    
    /// 从单个值创建（一阶精度）
    pub fn from_values(left: f64, right: f64) -> Self {
        Self { left, right }
    }
    
    /// 计算面平均值
    pub fn average(&self) -> f64 {
        0.5 * (self.left + self.right)
    }
    
    /// 计算跳跃 (right - left)
    pub fn jump(&self) -> f64 {
        self.right - self.left
    }
    
    /// 计算绝对最大值
    pub fn max_abs(&self) -> f64 {
        self.left.abs().max(self.right.abs())
    }
    
    /// 确保正定（用于水深）
    pub fn ensure_positive(&mut self, min_value: f64) {
        self.left = self.left.max(min_value);
        self.right = self.right.max(min_value);
    }
}

/// 重构器 trait
///
/// 所有重构方案实现此 trait。
pub trait Reconstructor: Send + Sync {
    /// 计算所有单元的梯度
    fn compute_gradients(&mut self, values: &[f64]);
    
    /// 重构标量场的面值
    ///
    /// # Arguments
    /// * `face_id` - 面索引
    /// * `values` - 单元中心值
    ///
    /// # Returns
    /// 面两侧的重构值
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
