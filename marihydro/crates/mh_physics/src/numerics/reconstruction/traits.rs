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

/// 向量重构状态 (2D 速度)
#[derive(Debug, Clone, Copy)]
pub struct ReconstructedVector {
    /// 左侧向量
    pub left: DVec2,
    
    /// 右侧向量
    pub right: DVec2,
}

impl ReconstructedVector {
    /// 创建新的向量重构状态
    pub fn new(left: DVec2, right: DVec2) -> Self {
        Self { left, right }
    }
    
    /// 从分量创建
    pub fn from_components(left_x: f64, left_y: f64, right_x: f64, right_y: f64) -> Self {
        Self {
            left: DVec2::new(left_x, left_y),
            right: DVec2::new(right_x, right_y),
        }
    }
    
    /// 计算平均向量
    pub fn average(&self) -> DVec2 {
        0.5 * (self.left + self.right)
    }
    
    /// 计算法向分量
    ///
    /// 将速度投影到面法向上
    pub fn normal_components(&self, normal: DVec2) -> ReconstructedState {
        ReconstructedState {
            left: self.left.dot(normal),
            right: self.right.dot(normal),
        }
    }
    
    /// 计算切向分量
    pub fn tangent_components(&self, normal: DVec2) -> ReconstructedState {
        let tangent = DVec2::new(-normal.y, normal.x);
        ReconstructedState {
            left: self.left.dot(tangent),
            right: self.right.dot(tangent),
        }
    }
}

/// 浅水方程完整重构状态
#[derive(Debug, Clone, Copy)]
pub struct ReconstructedShallowWater {
    /// 水深重构
    pub h: ReconstructedState,
    
    /// 速度重构
    pub velocity: ReconstructedVector,
    
    /// 底高程重构 (可选，用于静水重构)
    pub z: Option<ReconstructedState>,
}

impl ReconstructedShallowWater {
    /// 创建新的浅水重构状态
    pub fn new(h: ReconstructedState, velocity: ReconstructedVector) -> Self {
        Self { h, velocity, z: None }
    }
    
    /// 带底高程的构造
    pub fn with_bed(h: ReconstructedState, velocity: ReconstructedVector, z: ReconstructedState) -> Self {
        Self { h, velocity, z: Some(z) }
    }
    
    /// 确保水深正定
    pub fn ensure_positive_depth(&mut self, min_h: f64) {
        self.h.ensure_positive(min_h);
    }
    
    /// 获取动量通量所需的 hu 值
    pub fn hu(&self) -> ReconstructedVector {
        ReconstructedVector {
            left: self.h.left * self.velocity.left,
            right: self.h.right * self.velocity.right,
        }
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
    
    #[test]
    fn test_reconstructed_vector() {
        let left = DVec2::new(1.0, 2.0);
        let right = DVec2::new(3.0, 4.0);
        let vec = ReconstructedVector::new(left, right);
        
        assert_eq!(vec.average(), DVec2::new(2.0, 3.0));
    }
    
    #[test]
    fn test_reconstructed_vector_normal() {
        let left = DVec2::new(1.0, 0.0);
        let right = DVec2::new(2.0, 0.0);
        let vec = ReconstructedVector::new(left, right);
        
        let normal = DVec2::new(1.0, 0.0);
        let normal_comp = vec.normal_components(normal);
        
        assert_eq!(normal_comp.left, 1.0);
        assert_eq!(normal_comp.right, 2.0);
    }
    
    #[test]
    fn test_reconstructed_shallow_water() {
        let h = ReconstructedState::new(1.0, 1.2);
        let vel = ReconstructedVector::new(
            DVec2::new(0.5, 0.0),
            DVec2::new(0.6, 0.0),
        );
        
        let sw = ReconstructedShallowWater::new(h, vel);
        
        let hu = sw.hu();
        assert!((hu.left.x - 0.5).abs() < 1e-10);
        assert!((hu.right.x - 0.72).abs() < 1e-10);
    }
    
    #[test]
    fn test_reconstructed_shallow_water_positive() {
        let h = ReconstructedState::new(-0.01, 1.0);
        let vel = ReconstructedVector::new(DVec2::ZERO, DVec2::ZERO);
        
        let mut sw = ReconstructedShallowWater::new(h, vel);
        sw.ensure_positive_depth(0.0);
        
        assert_eq!(sw.h.left, 0.0);
    }
}
