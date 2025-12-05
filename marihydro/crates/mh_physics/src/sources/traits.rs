// crates/mh_physics/src/sources/traits.rs

//! 源项 Trait 定义
//!
//! 定义源项的核心接口和数据结构。

use crate::state::ShallowWaterState;
use crate::types::NumericalParams;

/// 源项贡献
///
/// 表示单个单元的源项贡献，包括质量和动量变化率。
#[derive(Debug, Clone, Copy, Default)]
pub struct SourceContribution {
    /// 质量源 [m/s]
    pub s_h: f64,
    /// x动量源 [m²/s²]
    pub s_hu: f64,
    /// y动量源 [m²/s²]
    pub s_hv: f64,
}

impl SourceContribution {
    /// 零贡献常量
    pub const ZERO: Self = Self {
        s_h: 0.0,
        s_hu: 0.0,
        s_hv: 0.0,
    };

    /// 创建新的源项贡献
    #[inline]
    pub fn new(s_h: f64, s_hu: f64, s_hv: f64) -> Self {
        Self { s_h, s_hu, s_hv }
    }

    /// 创建仅动量贡献
    #[inline]
    pub fn momentum(s_hu: f64, s_hv: f64) -> Self {
        Self { s_h: 0.0, s_hu, s_hv }
    }

    /// 创建仅质量贡献
    #[inline]
    pub fn mass(s_h: f64) -> Self {
        Self { s_h, s_hu: 0.0, s_hv: 0.0 }
    }

    /// 加法
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            s_h: self.s_h + other.s_h,
            s_hu: self.s_hu + other.s_hu,
            s_hv: self.s_hv + other.s_hv,
        }
    }

    /// 原地加法
    #[inline]
    pub fn add_assign(&mut self, other: &Self) {
        self.s_h += other.s_h;
        self.s_hu += other.s_hu;
        self.s_hv += other.s_hv;
    }

    /// 缩放
    #[inline]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            s_h: self.s_h * factor,
            s_hu: self.s_hu * factor,
            s_hv: self.s_hv * factor,
        }
    }

    /// 检查是否有效（所有分量都是有限数）
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.s_h.is_finite() && self.s_hu.is_finite() && self.s_hv.is_finite()
    }

    /// 钳位到安全范围
    #[inline]
    pub fn clamp(&self, max_abs: f64) -> Self {
        Self {
            s_h: self.s_h.clamp(-max_abs, max_abs),
            s_hu: self.s_hu.clamp(-max_abs, max_abs),
            s_hv: self.s_hv.clamp(-max_abs, max_abs),
        }
    }
}

impl std::ops::Add for SourceContribution {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            s_h: self.s_h + rhs.s_h,
            s_hu: self.s_hu + rhs.s_hu,
            s_hv: self.s_hv + rhs.s_hv,
        }
    }
}

impl std::ops::AddAssign for SourceContribution {
    fn add_assign(&mut self, rhs: Self) {
        self.s_h += rhs.s_h;
        self.s_hu += rhs.s_hu;
        self.s_hv += rhs.s_hv;
    }
}

impl std::ops::Mul<f64> for SourceContribution {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self.scale(rhs)
    }
}

/// 源项计算上下文
///
/// 包含源项计算所需的时间和参数信息。
#[derive(Debug, Clone)]
pub struct SourceContext<'a> {
    /// 当前模拟时间 [s]
    pub time: f64,
    /// 时间步长 [s]
    pub dt: f64,
    /// 数值参数
    pub params: &'a NumericalParams,
}

impl<'a> SourceContext<'a> {
    /// 创建新的源项上下文
    pub fn new(time: f64, dt: f64, params: &'a NumericalParams) -> Self {
        Self { time, dt, params }
    }

    /// 检查单元是否干燥
    #[inline]
    pub fn is_dry(&self, h: f64) -> bool {
        h < self.params.h_dry
    }

    /// 检查单元是否湿润
    #[inline]
    pub fn is_wet(&self, h: f64) -> bool {
        h >= self.params.h_wet
    }
}

/// 源项 Trait
///
/// 定义源项计算的统一接口。
pub trait SourceTerm: Send + Sync {
    /// 获取源项名称
    fn name(&self) -> &'static str;

    /// 是否启用
    fn is_enabled(&self) -> bool;

    /// 计算单个单元的源项贡献
    fn compute_cell(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        ctx: &SourceContext,
    ) -> SourceContribution;

    /// 批量计算所有单元的源项
    ///
    /// 默认实现逐单元调用 `compute_cell`。
    /// 子类可以覆盖以提供优化的批量计算。
    fn compute_all(
        &self,
        state: &ShallowWaterState,
        ctx: &SourceContext,
        output_h: &mut [f64],
        output_hu: &mut [f64],
        output_hv: &mut [f64],
    ) {
        if !self.is_enabled() {
            return;
        }

        let n_cells = state.h.len();
        for i in 0..n_cells {
            let contrib = self.compute_cell(state, i, ctx);
            output_h[i] += contrib.s_h;
            output_hu[i] += contrib.s_hu;
            output_hv[i] += contrib.s_hv;
        }
    }

    /// 源项是否显式（需要CFL限制）
    fn is_explicit(&self) -> bool {
        true
    }

    /// 是否需要隐式处理
    fn requires_implicit_treatment(&self) -> bool {
        false
    }
}

/// 源项辅助函数
pub struct SourceHelpers;

impl SourceHelpers {
    /// 安全累加（忽略无效值）
    #[inline]
    pub fn safe_accumulate(acc: &mut f64, val: f64) {
        if val.is_finite() {
            *acc += val;
        }
    }

    /// 验证贡献值并钳位
    #[inline]
    pub fn validate_contribution(val: f64, max_abs: f64) -> f64 {
        if !val.is_finite() {
            return 0.0;
        }
        val.clamp(-max_abs, max_abs)
    }

    /// 光滑过渡函数 (干湿过渡)
    ///
    /// 返回 0.0 (完全干) 到 1.0 (完全湿) 之间的值
    #[inline]
    pub fn smooth_transition(h: f64, h_dry: f64, h_wet: f64) -> f64 {
        if h <= h_dry {
            0.0
        } else if h >= h_wet {
            1.0
        } else {
            (h - h_dry) / (h_wet - h_dry)
        }
    }

    /// 计算安全速度（避免除以零）
    #[inline]
    pub fn safe_velocity(hu: f64, hv: f64, h: f64, h_min: f64) -> (f64, f64) {
        let h_safe = h.max(h_min);
        (hu / h_safe, hv / h_safe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_contribution_zero() {
        let c = SourceContribution::ZERO;
        assert_eq!(c.s_h, 0.0);
        assert_eq!(c.s_hu, 0.0);
        assert_eq!(c.s_hv, 0.0);
    }

    #[test]
    fn test_source_contribution_add() {
        let c1 = SourceContribution::new(1.0, 2.0, 3.0);
        let c2 = SourceContribution::new(0.5, 1.0, 1.5);
        let c3 = c1.add(&c2);
        assert_eq!(c3.s_h, 1.5);
        assert_eq!(c3.s_hu, 3.0);
        assert_eq!(c3.s_hv, 4.5);
    }

    #[test]
    fn test_source_contribution_scale() {
        let c = SourceContribution::new(1.0, 2.0, 3.0);
        let scaled = c.scale(2.0);
        assert_eq!(scaled.s_h, 2.0);
        assert_eq!(scaled.s_hu, 4.0);
        assert_eq!(scaled.s_hv, 6.0);
    }

    #[test]
    fn test_source_contribution_operators() {
        let c1 = SourceContribution::new(1.0, 2.0, 3.0);
        let c2 = SourceContribution::new(0.5, 1.0, 1.5);
        
        let c3 = c1 + c2;
        assert_eq!(c3.s_h, 1.5);
        
        let c4 = c1 * 2.0;
        assert_eq!(c4.s_hu, 4.0);
    }

    #[test]
    fn test_source_contribution_validity() {
        let valid = SourceContribution::new(1.0, 2.0, 3.0);
        assert!(valid.is_valid());

        let invalid = SourceContribution::new(f64::NAN, 2.0, 3.0);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_source_contribution_clamp() {
        let c = SourceContribution::new(100.0, -200.0, 50.0);
        let clamped = c.clamp(75.0);
        assert_eq!(clamped.s_h, 75.0);
        assert_eq!(clamped.s_hu, -75.0);
        assert_eq!(clamped.s_hv, 50.0);
    }

    #[test]
    fn test_smooth_transition() {
        assert_eq!(SourceHelpers::smooth_transition(0.0, 0.01, 0.1), 0.0);
        assert_eq!(SourceHelpers::smooth_transition(0.1, 0.01, 0.1), 1.0);
        
        let mid = SourceHelpers::smooth_transition(0.055, 0.01, 0.1);
        assert!((mid - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_source_context() {
        let params = NumericalParams::default();
        let ctx = SourceContext::new(10.0, 0.1, &params);
        
        assert_eq!(ctx.time, 10.0);
        assert_eq!(ctx.dt, 0.1);
        // 默认 h_dry = 1e-6，所以 1e-7 是干的，1e-5 不是
        assert!(ctx.is_dry(1e-7));
        assert!(!ctx.is_dry(1e-5));
        assert!(ctx.is_wet(0.1));
    }
}
