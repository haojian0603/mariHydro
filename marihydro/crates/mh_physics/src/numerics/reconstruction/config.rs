//! MUSCL 重构配置

use crate::types::LimiterType;

/// 梯度计算方法选择
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GradientType {
    /// Green-Gauss 方法（基于面积分）
    #[default]
    GreenGauss,
    
    /// 最小二乘法（精度更高但计算量大）
    LeastSquares,
}

/// MUSCL 重构配置
#[derive(Debug, Clone)]
pub struct MusclConfig {
    /// 是否启用二阶精度
    pub second_order: bool,
    
    /// 梯度计算方法
    pub gradient_type: GradientType,
    
    /// 限制器类型
    pub limiter_type: LimiterType,
    
    /// Venkatakrishnan 限制器的 K 参数
    pub venkat_k: f64,
    
    /// 是否对水深使用正定限制
    ///
    /// 确保重构后的水深不为负
    pub positivity_preserving: bool,
    
    /// 干单元梯度容差
    ///
    /// 水深低于此值时将梯度设为零
    pub dry_tolerance: f64,
    
    /// 速度梯度额外限制
    ///
    /// 限制速度梯度以避免非物理值
    pub velocity_limiting: bool,
}

impl Default for MusclConfig {
    fn default() -> Self {
        Self {
            second_order: true,
            gradient_type: GradientType::GreenGauss,
            limiter_type: LimiterType::Venkatakrishnan,
            venkat_k: 5.0,
            positivity_preserving: true,
            dry_tolerance: 1e-6,
            velocity_limiting: true,
        }
    }
}

impl MusclConfig {
    /// 创建新配置
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 一阶精度配置（无梯度重构）
    pub fn first_order() -> Self {
        Self {
            second_order: false,
            ..Default::default()
        }
    }
    
    /// 高精度配置
    pub fn high_accuracy() -> Self {
        Self {
            second_order: true,
            gradient_type: GradientType::LeastSquares,
            limiter_type: LimiterType::Venkatakrishnan,
            venkat_k: 10.0, // 较高的 K 减少限制
            positivity_preserving: true,
            dry_tolerance: 1e-8,
            velocity_limiting: true,
        }
    }
    
    /// 高稳定性配置
    pub fn high_stability() -> Self {
        Self {
            second_order: true,
            gradient_type: GradientType::GreenGauss,
            limiter_type: LimiterType::Minmod,
            venkat_k: 1.0,
            positivity_preserving: true,
            dry_tolerance: 1e-4,
            velocity_limiting: true,
        }
    }
    
    /// 设置二阶精度
    pub fn with_second_order(mut self, enabled: bool) -> Self {
        self.second_order = enabled;
        self
    }
    
    /// 设置梯度类型
    pub fn with_gradient_type(mut self, gradient_type: GradientType) -> Self {
        self.gradient_type = gradient_type;
        self
    }
    
    /// 设置限制器类型
    pub fn with_limiter(mut self, limiter_type: LimiterType) -> Self {
        self.limiter_type = limiter_type;
        self
    }
    
    /// 设置 Venkatakrishnan K 参数
    pub fn with_venkat_k(mut self, k: f64) -> Self {
        self.venkat_k = k;
        self
    }
    
    /// 设置正定保持
    pub fn with_positivity_preserving(mut self, enabled: bool) -> Self {
        self.positivity_preserving = enabled;
        self
    }
    
    /// 设置干容差
    pub fn with_dry_tolerance(mut self, tol: f64) -> Self {
        self.dry_tolerance = tol;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = MusclConfig::default();
        assert!(config.second_order);
        assert_eq!(config.gradient_type, GradientType::GreenGauss);
        assert_eq!(config.limiter_type, LimiterType::Venkatakrishnan);
        assert_eq!(config.venkat_k, 5.0);
        assert!(config.positivity_preserving);
    }
    
    #[test]
    fn test_first_order_config() {
        let config = MusclConfig::first_order();
        assert!(!config.second_order);
    }
    
    #[test]
    fn test_high_accuracy_config() {
        let config = MusclConfig::high_accuracy();
        assert!(config.second_order);
        assert_eq!(config.gradient_type, GradientType::LeastSquares);
        assert_eq!(config.venkat_k, 10.0);
    }
    
    #[test]
    fn test_high_stability_config() {
        let config = MusclConfig::high_stability();
        assert!(config.second_order);
        assert_eq!(config.limiter_type, LimiterType::Minmod);
    }
    
    #[test]
    fn test_builder_pattern() {
        let config = MusclConfig::new()
            .with_second_order(true)
            .with_gradient_type(GradientType::LeastSquares)
            .with_limiter(LimiterType::BarthJespersen)
            .with_venkat_k(3.0)
            .with_dry_tolerance(1e-5);
        
        assert!(config.second_order);
        assert_eq!(config.gradient_type, GradientType::LeastSquares);
        assert_eq!(config.limiter_type, LimiterType::BarthJespersen);
        assert_eq!(config.venkat_k, 3.0);
        assert_eq!(config.dry_tolerance, 1e-5);
    }
    
    #[test]
    fn test_gradient_type_default() {
        assert_eq!(GradientType::default(), GradientType::GreenGauss);
    }
}
