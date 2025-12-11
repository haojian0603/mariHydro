// crates/mh_physics/src/sources/turbulence/traits.rs

//! 湍流闭合 trait
//!
//! 定义湍流模型的公共接口，为不同湍流闭合提供统一抽象。

/// 速度梯度张量
///
/// 用于计算应变率、涡度等湍流相关量。
///
/// # 2D 应变率张量
///
/// ```text
/// S = [S_11  S_12]   [∂u/∂x        (∂u/∂y+∂v/∂x)/2]
///     [S_21  S_22] = [(∂u/∂y+∂v/∂x)/2     ∂v/∂y    ]
/// ```
///
/// # 应变率模
///
/// ```text
/// |S| = √(2S_ij·S_ij) = √(2(∂u/∂x)² + 2(∂v/∂y)² + (∂u/∂y + ∂v/∂x)²)
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct VelocityGradient {
    /// ∂u/∂x
    pub du_dx: f64,
    /// ∂u/∂y
    pub du_dy: f64,
    /// ∂v/∂x
    pub dv_dx: f64,
    /// ∂v/∂y
    pub dv_dy: f64,
}

impl VelocityGradient {
    /// 创建新的速度梯度
    #[inline]
    pub fn new(du_dx: f64, du_dy: f64, dv_dx: f64, dv_dy: f64) -> Self {
        Self { du_dx, du_dy, dv_dx, dv_dy }
    }

    /// 计算应变率张量的模
    ///
    /// |S| = √(2*(∂u/∂x)² + 2*(∂v/∂y)² + (∂u/∂y + ∂v/∂x)²)
    #[inline]
    pub fn strain_rate_magnitude(&self) -> f64 {
        let s11 = self.du_dx;
        let s22 = self.dv_dy;
        let s12 = 0.5 * (self.du_dy + self.dv_dx);

        (2.0 * s11 * s11 + 2.0 * s22 * s22 + 4.0 * s12 * s12).sqrt()
    }

    /// 计算涡度（z 分量）
    ///
    /// ω_z = ∂v/∂x - ∂u/∂y
    #[inline]
    pub fn vorticity(&self) -> f64 {
        self.dv_dx - self.du_dy
    }

    /// 计算散度
    ///
    /// div(u) = ∂u/∂x + ∂v/∂y
    #[inline]
    pub fn divergence(&self) -> f64 {
        self.du_dx + self.dv_dy
    }
    
    /// 检查梯度是否有效
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.du_dx.is_finite() && self.du_dy.is_finite() 
            && self.dv_dx.is_finite() && self.dv_dy.is_finite()
    }
}

/// 湍流闭合模型 trait
///
/// 所有湍流模型（Smagorinsky, k-ε 等）的公共接口。
/// 
/// # 实现者
/// 
/// - [`SmagorinskySolver`](super::SmagorinskySolver): 2D Smagorinsky 模型
/// - [`KEpsilonModel`](super::KEpsilonModel): 3D k-ε 模型
pub trait TurbulenceClosure: Send + Sync {
    /// 模型名称
    fn name(&self) -> &'static str;
    
    /// 是否为 3D 模型
    ///
    /// 3D 模型需要完整的垂向结构，不适用于深度平均方程。
    fn is_3d(&self) -> bool;
    
    /// 获取涡粘性场
    fn eddy_viscosity(&self) -> &[f64];
    
    /// 获取单个单元的涡粘性
    fn get_eddy_viscosity(&self, cell: usize) -> f64 {
        self.eddy_viscosity().get(cell).copied().unwrap_or(0.0)
    }
    
    /// 更新涡粘性（基于当前速度场）
    ///
    /// # 参数
    /// - `velocity_gradients`: 速度梯度场
    /// - `cell_sizes`: 网格尺度（用于 Smagorinsky 等模型）
    fn update(&mut self, velocity_gradients: &[VelocityGradient], cell_sizes: &[f64]);
    
    /// 是否启用
    fn is_enabled(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_gradient_strain_rate() {
        // 纯剪切流: u = y, v = 0
        // ∂u/∂y = 1, 其他为 0
        let grad = VelocityGradient::new(0.0, 1.0, 0.0, 0.0);

        // |S| = √(4 * s12²) = √(4 * 0.25) = 1.0
        let strain = grad.strain_rate_magnitude();
        assert!((strain - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_velocity_gradient_vorticity() {
        let grad = VelocityGradient::new(0.0, 1.0, -1.0, 0.0);
        let vorticity = grad.vorticity();
        // ω = -1 - 1 = -2
        assert!((vorticity - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_velocity_gradient_divergence() {
        let grad = VelocityGradient::new(2.0, 0.0, 0.0, 3.0);
        let div = grad.divergence();
        assert!((div - 5.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_velocity_gradient_validity() {
        let valid = VelocityGradient::new(1.0, 2.0, 3.0, 4.0);
        assert!(valid.is_valid());
        
        let invalid = VelocityGradient::new(f64::NAN, 0.0, 0.0, 0.0);
        assert!(!invalid.is_valid());
    }
}
