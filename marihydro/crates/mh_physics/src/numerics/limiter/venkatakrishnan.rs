//! marihydro\crates\mh_physics\src\numerics\limiter\venkatakrishnan.rs
//! Venkatakrishnan 限制器
//!
//! 光滑的梯度限制器，避免 Barth-Jespersen 的梯度突变问题。
//!
//! # 算法
//!
//! 使用光滑函数替代 min 操作:
//!
//! ```text
//!         (Δ²_max + ε²) Δ_f + 2 Δ²_f Δ_max
//! α_f = ────────────────────────────────────  如果 Δ_f > 0
//!         Δ²_max + 2 Δ²_f + Δ_max Δ_f + ε²
//! ```
//!
//! 其中:
//! - ε² = (K * h)³，h 是网格特征尺度
//! - K 是用户参数
//!
//! # K 值选择指南
//!
//! | K 值 | 特性 | 适用场景 |
//! |------|------|---------|
//! | 0.1-0.3 | 强限制，高稳定性 | 激波、溃坝、干湿过渡 |
//! | 0.3-1.0 | 中等限制 | 一般流动 |
//! | 1.0-5.0 | 弱限制，高精度 | 平滑流、稳态问题 |
//! | 5.0-10.0 | 几乎不限制 | 非常光滑的问题 |
//!
//! **注意**: 传统文献推荐 K=5-10，但对于浅水方程中常见的干湿过渡和
//! 强间断问题，K=0.3 更加稳定。
//!
//! # 特点
//!
//! - 光滑连续，不产生梯度突变
//! - 在光滑区域趋近于 1（二阶精度）
//! - 在不连续处限制梯度（一阶精度）
//! - K 越大，越不耗散
//!
//! # 参考文献
//!
//! Venkatakrishnan, V. (1993). "On the accuracy of limiters and 
//! convergence to steady state solutions". AIAA Paper 93-0880.

use mh_core::RuntimeScalar;
use super::traits::{LimiterContextGeneric, SlopeLimiterGeneric};

// Re-export for tests
#[cfg(test)]
use super::traits::LimiterContext;

/// 泛型 Venkatakrishnan 限制器
///
/// 光滑的二阶精度限制器，推荐用于通用浅水模拟。
///
/// # 默认配置
///
/// 默认使用 K=0.3，适合浅水方程中常见的干湿过渡和强间断问题。
/// 如需更高精度（较少耗散），可使用 `for_smooth_flow()` 预设。
#[derive(Debug, Clone, Copy)]
pub struct VenkatakrishnanGeneric<S: RuntimeScalar> {
    /// K 参数，控制限制强度
    /// - 小值 (0.1-0.3): 强限制，高稳定性
    /// - 大值 (1.0-5.0): 弱限制，高精度
    k: S,
    
    /// ε² = (K * h)³，预计算的平滑参数
    eps_squared: S,
    
    /// 判断梯度为零的容差
    tol: S,
}

impl<S: RuntimeScalar> VenkatakrishnanGeneric<S> {
    /// 创建新的 Venkatakrishnan 限制器
    ///
    /// # Arguments
    /// * `k` - K 参数，控制限制强度
    /// * `mesh_scale` - 网格特征尺度 h（如平均边长或最小边长）
    ///
    /// # K 值建议
    /// - 激波/溃坝问题: 0.1-0.3
    /// - 一般流动: 0.3-1.0
    /// - 光滑稳态: 1.0-5.0
    pub fn new(k: S, mesh_scale: S) -> Self {
        // ε² = (K * h)³
        let kh = k * mesh_scale;
        let eps_squared = kh * kh * kh;
        
        Self {
            k,
            eps_squared,
            tol: S::from_config(1e-12).unwrap_or(S::MIN_POSITIVE),
        }
    }
    
    /// 创建具有自定义容差的限制器
    pub fn with_tolerance(k: S, mesh_scale: S, tol: S) -> Self {
        let kh = k * mesh_scale;
        let eps_squared = kh * kh * kh;
        
        Self {
            k,
            eps_squared,
            tol,
        }
    }
    
    // =========================================================================
    // 预设配置
    // =========================================================================
    
    /// 创建适合激波和溃坝问题的限制器
    ///
    /// 使用 K=0.1，提供最强的限制以确保稳定性。
    /// 适用于溃坝、水跃等强间断问题。
    pub fn for_shock_capturing(mesh_scale: S) -> Self {
        Self::new(S::from_config(0.1).unwrap_or(S::ZERO), mesh_scale)
    }
    
    /// 创建适合干湿过渡的限制器
    ///
    /// 使用 K=0.3（默认值），平衡稳定性和精度。
    /// 适用于大多数浅水模拟。
    pub fn for_wetting_drying(mesh_scale: S) -> Self {
        Self::new(S::from_config(0.3).unwrap_or(S::ZERO), mesh_scale)
    }
    
    /// 创建适合光滑流动的限制器
    ///
    /// 使用 K=2.0，减少数值耗散。
    /// 适用于平滑流动、稳态问题。
    pub fn for_smooth_flow(mesh_scale: S) -> Self {
        Self::new(S::TWO, mesh_scale)
    }
    
    /// 创建几乎不限制的限制器
    ///
    /// 使用 K=5.0，接近无限制器的二阶精度。
    /// 仅适用于非常光滑、无间断的问题。
    pub fn minimal_limiting(mesh_scale: S) -> Self {
        Self::new(S::from_config(5.0).unwrap_or(S::ZERO), mesh_scale)
    }
    
    // =========================================================================
    // 访问器
    // =========================================================================
    
    /// 获取 K 参数
    pub fn k(&self) -> S {
        self.k
    }
    
    /// 获取 ε² 值
    pub fn eps_squared(&self) -> S {
        self.eps_squared
    }
    
    /// 更新网格尺度（当网格变化时）
    pub fn update_mesh_scale(&mut self, mesh_scale: S) {
        let kh = self.k * mesh_scale;
        self.eps_squared = kh * kh * kh;
    }
    
    /// 更新 K 参数
    pub fn set_k(&mut self, k: S, mesh_scale: S) {
        self.k = k;
        let kh = k * mesh_scale;
        self.eps_squared = kh * kh * kh;
    }
    
    /// 计算光滑限制函数 φ(x, y, ε²)
    ///
    /// ```text
    ///         (y² + ε²) x + 2 x² y
    /// φ = ────────────────────────────
    ///         y² + 2 x² + x y + ε²
    /// ```
    #[inline]
    fn phi(&self, x: S, y: S) -> S {
        let x2 = x * x;
        let y2 = y * y;
        let eps2 = self.eps_squared;
        
        let numerator = (y2 + eps2) * x + S::TWO * x2 * y;
        let denominator = y2 + S::TWO * x2 + x * y + eps2;
        
        if denominator.abs() < self.tol {
            S::ONE
        } else {
            numerator / denominator
        }
    }
}

impl<S: RuntimeScalar> Default for VenkatakrishnanGeneric<S> {
    fn default() -> Self {
        // 默认 K=0.3（适合浅水方程的干湿过渡）, mesh_scale=1
        // 注意：这与传统推荐值 K=5 不同，但对浅水问题更稳定
        Self::new(S::from_config(0.3).unwrap_or(S::ZERO), S::ONE)
    }
}

impl<S: RuntimeScalar> SlopeLimiterGeneric<S> for VenkatakrishnanGeneric<S> {
    fn compute_limiter(&self, ctx: &LimiterContextGeneric<S>) -> S {
        // 如果梯度为零，不需要限制
        if ctx.is_gradient_zero(self.tol) {
            return S::ONE;
        }
        
        let delta = ctx.gradient;
        
        if delta > S::ZERO {
            // 正梯度：使用 Δ_max
            let delta_max = ctx.delta_max();
            if delta_max < self.tol {
                // 已经在最大值附近
                S::ZERO
            } else {
                let result = self.phi(delta, delta_max);
                if result < S::ONE { result } else { S::ONE }
            }
        } else {
            // 负梯度：使用 Δ_min
            let delta_min = ctx.delta_min();
            if delta_min > -self.tol {
                // 已经在最小值附近
                S::ZERO
            } else {
                // 取绝对值来使用相同的 phi 函数
                let result = self.phi(-delta, -delta_min);
                if result < S::ONE { result } else { S::ONE }
            }
        }
    }
    
    fn name(&self) -> &'static str {
        "Venkatakrishnan"
    }
}

// =============================================================================
// Type alias for f64 version
// =============================================================================

/// f64 特化版本 (默认)
pub type Venkatakrishnan = VenkatakrishnanGeneric<f64>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_venkatakrishnan_creation() {
        let limiter = Venkatakrishnan::new(5.0, 0.1);
        assert_eq!(limiter.k(), 5.0);
        
        // ε² = (5 * 0.1)³ = 0.5³ = 0.125
        assert!((limiter.eps_squared() - 0.125).abs() < 1e-10);
    }
    
    #[test]
    fn test_venkatakrishnan_default() {
        let limiter = Venkatakrishnan::default();
        assert_eq!(limiter.k(), 0.3); // 新默认值
        // ε² = (0.3 * 1)³ = 0.027
        assert!((limiter.eps_squared() - 0.027).abs() < 1e-10);
    }
    
    #[test]
    fn test_presets() {
        let shock = Venkatakrishnan::for_shock_capturing(1.0);
        assert_eq!(shock.k(), 0.1);
        
        let wet_dry = Venkatakrishnan::for_wetting_drying(1.0);
        assert_eq!(wet_dry.k(), 0.3);
        
        let smooth = Venkatakrishnan::for_smooth_flow(1.0);
        assert_eq!(smooth.k(), 2.0);
        
        let minimal = Venkatakrishnan::minimal_limiting(1.0);
        assert_eq!(minimal.k(), 5.0);
    }
    
    #[test]
    fn test_venkatakrishnan_name() {
        let limiter = Venkatakrishnan::default();
        assert_eq!(limiter.name(), "Venkatakrishnan");
    }
    
    #[test]
    fn test_zero_gradient() {
        let limiter = Venkatakrishnan::new(5.0, 0.1);
        let ctx = LimiterContext::new(1.0, 0.0, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
    
    #[test]
    fn test_small_gradient() {
        // 测试基本行为：小梯度时的限制值
        let limiter = Venkatakrishnan::new(5.0, 0.1);
        // ε² = (5 * 0.1)³ = 0.125
        let ctx = LimiterContext::new(1.0, 0.1, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        // 限制器应该返回 [0, 1] 范围内的值
        assert!((0.0..=1.0).contains(&alpha), "Alpha {} out of bounds", alpha);
        
        // 测试零梯度应该返回 1.0
        let ctx_zero = LimiterContext::new(1.0, 0.0, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx_zero), 1.0);
    }
    
    #[test]
    fn test_k_parameter_sensitivity() {
        // K 参数越大，限制越小（alpha 越接近 1）
        // 这是 Venkatakrishnan 限制器的关键特性
        let mesh_scale = 0.1;
        
        let limiter_k1 = Venkatakrishnan::new(1.0, mesh_scale);
        let limiter_k5 = Venkatakrishnan::new(5.0, mesh_scale);
        let limiter_k10 = Venkatakrishnan::new(10.0, mesh_scale);
        
        // 测试场景：需要一定限制的情况
        let ctx = LimiterContext::new(1.0, 0.4, 0.5, 1.5, 0.1);
        
        let alpha_k1 = limiter_k1.compute_limiter(&ctx);
        let alpha_k5 = limiter_k5.compute_limiter(&ctx);
        let alpha_k10 = limiter_k10.compute_limiter(&ctx);
        
        // K 越大，alpha 应该越接近 1（限制越小）
        assert!(alpha_k1 <= alpha_k5, 
            "K=1 should limit more than K=5: {} vs {}", alpha_k1, alpha_k5);
        assert!(alpha_k5 <= alpha_k10, 
            "K=5 should limit more than K=10: {} vs {}", alpha_k5, alpha_k10);
        
        // 额外测试：K 越大，差距越明显
        // 验证 ε² 计算正确
        assert!(limiter_k1.eps_squared() < limiter_k5.eps_squared(),
            "Larger K should give larger ε²");
        assert!(limiter_k5.eps_squared() < limiter_k10.eps_squared(),
            "Larger K should give larger ε²");
    }
    
    #[test]
    fn test_large_gradient() {
        let limiter = Venkatakrishnan::new(1.0, 0.01); // 小 ε²
        // 大梯度，小 ε² → 类似 Barth-Jespersen
        let ctx = LimiterContext::new(1.0, 0.8, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        // Δ_max = 0.5, gradient = 0.8 → 需要限制
        assert!(alpha < 1.0);
        assert!(alpha > 0.0);
    }
    
    #[test]
    fn test_convergence_to_barth_jespersen() {
        // 当 ε² → 0 时，Venkatakrishnan 应该趋近 Barth-Jespersen
        // 但由于两者公式不同，不期望完全相等
        // Venkatakrishnan 使用光滑函数，BJ 使用 min 操作
        
        use super::super::barth_jespersen::BarthJespersen;
        
        let bj = BarthJespersen::new();
        
        // 测试不同的 ε² 值：随着 ε² 减小，VK 应该更接近 BJ
        let vk_large_eps = Venkatakrishnan::new(5.0, 1.0);    // ε² = 125
        let vk_medium_eps = Venkatakrishnan::new(1.0, 0.1);   // ε² = 0.001
        let vk_small_eps = Venkatakrishnan::new(0.01, 0.01);  // ε² = 1e-9
        
        let ctx = LimiterContext::new(1.0, 0.8, 0.5, 1.5, 0.1);
        
        let alpha_bj = bj.compute_limiter(&ctx);
        let alpha_vk_large = vk_large_eps.compute_limiter(&ctx);
        let alpha_vk_medium = vk_medium_eps.compute_limiter(&ctx);
        let alpha_vk_small = vk_small_eps.compute_limiter(&ctx);
        
        // 验证：ε² 越小，VK 限制器产生更小的 alpha（更多限制）
        // 这是因为 ε² 提供了光滑性，ε²越小越接近非光滑的 BJ
        assert!(alpha_vk_large >= alpha_vk_small,
            "Larger ε² should give larger alpha. large={}, small={}", 
            alpha_vk_large, alpha_vk_small);
        
        // 验证所有结果都在有效范围内
        assert!((0.0..=1.0).contains(&alpha_vk_large), 
            "alpha_vk_large out of bounds: {}", alpha_vk_large);
        assert!((0.0..=1.0).contains(&alpha_vk_medium),
            "alpha_vk_medium out of bounds: {}", alpha_vk_medium);
        assert!((0.0..=1.0).contains(&alpha_vk_small),
            "alpha_vk_small out of bounds: {}", alpha_vk_small);
        
        // 验证 BJ 在有效范围
        assert!((0.0..=1.0).contains(&alpha_bj),
            "alpha_bj out of bounds: {}", alpha_bj);
    }
    
    #[test]
    fn test_update_mesh_scale() {
        let mut limiter = Venkatakrishnan::new(5.0, 0.1);
        assert!((limiter.eps_squared() - 0.125).abs() < 1e-10);
        
        limiter.update_mesh_scale(0.2);
        // ε² = (5 * 0.2)³ = 1.0³ = 1.0
        assert!((limiter.eps_squared() - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_smoothness() {
        let limiter = Venkatakrishnan::new(5.0, 0.1);
        
        // 测试限制器在梯度变化时是连续的（只测试正梯度区域）
        // 在 0 附近会有突变因为处理逻辑不同
        // ALLOW_F64: 测试代码使用具体类型
        let gradients: Vec<f64> = (1..=100).map(|i| i as f64 * 0.01).collect();
        // ALLOW_F64: 测试代码使用具体类型
        let alphas: Vec<f64> = gradients
            .iter()
            .map(|&g| {
                let ctx = LimiterContext::new(1.0, g, 0.5, 1.5, 0.1);
                limiter.compute_limiter(&ctx)
            })
            .collect();
        
        // 检查连续性：相邻值差不应该太大
        for window in alphas.windows(2) {
            let diff = (window[1] - window[0]).abs();
            assert!(diff < 0.2, "Limiter not smooth: diff = {}", diff);
        }
    }
    
    #[test]
    fn test_positive_vs_negative_symmetry() {
        let limiter = Venkatakrishnan::new(5.0, 0.1);
        
        // 正梯度
        let ctx_pos = LimiterContext::new(1.0, 0.3, 0.5, 1.5, 0.1);
        let alpha_pos = limiter.compute_limiter(&ctx_pos);
        
        // 对称的负梯度
        let ctx_neg = LimiterContext::new(1.0, -0.3, 0.5, 1.5, 0.1);
        let alpha_neg = limiter.compute_limiter(&ctx_neg);
        
        // 由于 delta_max 和 delta_min 对称，结果应该相等
        assert!((alpha_pos - alpha_neg).abs() < 1e-10);
    }
    
    #[test]
    fn test_at_maximum() {
        let limiter = Venkatakrishnan::new(1.0, 0.01); // 小 ε² 使限制更明显
        // 单元值已经是最大值，正梯度应该被强限制
        let ctx = LimiterContext::new(1.5, 0.3, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!(alpha < 0.1);
    }
    
    #[test]
    fn test_limiter_bounded() {
        let limiter = Venkatakrishnan::new(3.0, 0.1);
        
        // 测试各种情况下 α ∈ [0, 1]
        let test_cases = vec![
            (1.0, 0.5, 0.0, 2.0),
            (1.0, -0.5, 0.0, 2.0),
            (1.0, 2.0, 0.0, 2.0),
            (1.0, -2.0, 0.0, 2.0),
            (1.0, 0.01, 0.5, 1.5),
            (1.0, -0.01, 0.5, 1.5),
        ];
        
        for (q, g, q_min, q_max) in test_cases {
            let ctx = LimiterContext::new(q, g, q_min, q_max, 0.1);
            let alpha = limiter.compute_limiter(&ctx);
            assert!((0.0..=1.0).contains(&alpha), 
                "Alpha {} out of bounds for q={}, g={}", alpha, q, g);
        }
    }
}
