// crates/mh_foundation/src/tolerance.rs

//! 数值容差配置（Legacy 模块）
//!
//! **⚠️ 已废弃：新代码应使用 `mh_core::Tolerance<S>` 泛型版本**
//!
//! 本模块保留 `NumericalTolerance` 结构体用于向后兼容。
//! 全局 TOLERANCE 静态变量已被移除，请使用参数注入模式。
//!
//! # 迁移指南
//!
//! ```ignore
//! // 旧代码（已废弃）
//! use mh_foundation::tolerance::TOLERANCE;
//! let h_dry = TOLERANCE.read().unwrap().h_dry;
//!
//! // 新代码（推荐）
//! use mh_core::Tolerance;
//! fn is_dry<S: RuntimeScalar>(h: S, tol: &Tolerance<S>) -> bool {
//!     h < tol.h_dry
//! }
//! ```

/// 数值容差配置（Legacy，建议使用 `mh_core::Tolerance<S>`）
///
/// 包含所有数值计算中使用的容差阈值。
/// 此结构体仅使用 f64，不支持泛型精度切换。
#[deprecated(
    since = "0.2.0",
    note = "使用 mh_core::Tolerance<S> 替代，支持泛型精度"
)]
#[derive(Debug, Clone)]
pub struct NumericalTolerance {
    /// 时间相对容差
    pub time_rel: f64,
    /// 时间绝对容差
    pub time_abs: f64,
    /// 空间容差
    pub spatial: f64,
    /// 权重求和容差
    pub weight_sum: f64,
    /// 矩阵对角元容差
    pub matrix_diag: f64,
    /// 迭代收敛容差
    pub convergence: f64,
    /// 干单元水深阈值 [m]
    pub h_dry: f64,
    /// 最小水深 [m]
    pub h_min: f64,
    /// 梯度计算容差
    pub gradient_eps: f64,
    /// 安全除法阈值
    pub safe_div: f64,
    /// 面积最小值 [m²]
    pub min_area: f64,
}

#[allow(deprecated)]
impl Default for NumericalTolerance {
    fn default() -> Self {
        Self {
            time_rel: 1e-12,
            time_abs: 1e-14,
            spatial: 1e-14,
            weight_sum: 1e-14,
            matrix_diag: 1e-14,
            convergence: 1e-8,
            h_dry: 1e-4,
            h_min: 1e-6,
            gradient_eps: 1e-12,
            safe_div: 1e-14,
            min_area: 1e-12,
        }
    }
}

#[allow(deprecated)]
impl NumericalTolerance {
    /// 创建保守配置（更严格的容差）
    pub fn conservative() -> Self {
        Self {
            convergence: 1e-10,
            h_dry: 1e-5,
            h_min: 1e-8,
            ..Default::default()
        }
    }

    /// 创建快速配置（更宽松的容差）
    pub fn fast() -> Self {
        Self {
            convergence: 1e-6,
            h_dry: 1e-3,
            h_min: 1e-5,
            ..Default::default()
        }
    }

    /// 混合相对/绝对容差判断时间接近
    #[inline]
    pub fn is_time_close(&self, a: f64, b: f64) -> bool {
        let diff = (a - b).abs();
        diff < self.time_abs || diff < self.time_rel * a.abs().max(b.abs()).max(1.0)
    }

    /// 判断空间值是否接近零
    #[inline]
    pub fn is_spatial_zero(&self, x: f64) -> bool {
        x.abs() < self.spatial
    }

    /// 判断水深是否为干
    #[inline]
    pub fn is_dry(&self, h: f64) -> bool {
        h < self.h_dry
    }

    /// 判断水深是否为湿
    #[inline]
    pub fn is_wet(&self, h: f64) -> bool {
        h >= self.h_dry
    }

    /// 安全除法判断分母是否过小
    #[inline]
    pub fn is_divisor_safe(&self, d: f64) -> bool {
        d.abs() >= self.safe_div
    }

    /// 判断矩阵对角元是否有效
    #[inline]
    pub fn is_diag_valid(&self, d: f64) -> bool {
        d.abs() >= self.matrix_diag
    }

    /// 判断迭代是否收敛
    #[inline]
    pub fn is_converged(&self, residual: f64, initial: f64) -> bool {
        residual < self.convergence * initial.max(1.0)
    }
}

// 全局 TOLERANCE 静态变量已删除
// 请使用 mh_core::Tolerance<S> 并通过参数传递

#[cfg(test)]
mod tests {
    #[allow(deprecated)]
    use super::*;

    #[test]
    fn test_default_tolerance() {
        let tol = NumericalTolerance::default();
        assert!((tol.h_dry - 1e-4).abs() < 1e-15);
        assert!((tol.convergence - 1e-8).abs() < 1e-15);
    }

    #[test]
    fn test_is_time_close() {
        let tol = NumericalTolerance::default();
        assert!(tol.is_time_close(1.0, 1.0));
        assert!(tol.is_time_close(1.0, 1.0 + 1e-13));
        assert!(tol.is_time_close(1e-15, 2e-15));
        assert!(!tol.is_time_close(1.0, 2.0));
    }

    #[test]
    fn test_is_dry_wet() {
        let tol = NumericalTolerance::default();
        assert!(tol.is_dry(1e-5));
        assert!(!tol.is_dry(1e-3));
        assert!(tol.is_wet(1e-3));
        assert!(!tol.is_wet(1e-5));
    }

    #[test]
    fn test_conservative_config() {
        let tol = NumericalTolerance::conservative();
        assert!(tol.convergence < NumericalTolerance::default().convergence);
    }

    #[test]
    fn test_fast_config() {
        let tol = NumericalTolerance::fast();
        assert!(tol.convergence > NumericalTolerance::default().convergence);
    }
}
