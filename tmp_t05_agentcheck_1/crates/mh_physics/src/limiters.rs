// crates/mh_physics/src/limiters.rs
//
// Legacy note:
// - The main engine path uses `crate::numerics::limiter` and
//   `crate::numerics::reconstruction`.
// - This module remains only as a scalar compatibility shim for older callers.
// - New engine work should not add fresh dependencies on this module.

//! 斜率限制器与重构方法
//!
//! 提供高阶空间重构所需的限制器，支持：
//! - 经典限制器（Minmod, Superbee, Van Leer）
//! - Venkatakrishnan 限制器（非结构网格）
//! - Barth-Jespersen 限制器
//! - MUSCL 重构
//!
//! # 设计原则
//!
//! 1. **TVD 保持**：确保总变差不增
//! 2. **高阶精度**：光滑区域保持二阶
//! 3. **无振荡**：间断附近无伪振荡
//! 4. **数值扩散**：最小化人工扩散

// ============================================================================
// 限制器函数
// ============================================================================

/// 限制器类型
#[deprecated(note = "Use crate::types::LimiterType for configuration and crate::numerics::limiter for the engine path.")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimiterType {
    /// 无限制（一阶迎风）
    None,
    /// Minmod 限制器（最耗散）
    Minmod,
    /// Superbee 限制器（最不耗散）
    Superbee,
    /// Van Leer 限制器（平滑）
    VanLeer,
    /// Van Albada 限制器（可微）
    VanAlbada,
    /// Koren 限制器（三阶）
    Koren,
    /// MC (Monotonized Central) 限制器
    Mc,
    /// Venkatakrishnan 限制器（非结构网格）
    Venkatakrishnan,
    /// Barth-Jespersen 限制器
    BarthJespersen,
}

impl Default for LimiterType {
    fn default() -> Self {
        Self::VanLeer
    }
}

/// 符号函数
#[inline]
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Minmod 函数
#[inline]
pub fn minmod(a: f64, b: f64) -> f64 {
    if a * b > 0.0 {
        sign(a) * a.abs().min(b.abs())
    } else {
        0.0
    }
}

/// 三参数 Minmod
#[inline]
pub fn minmod3(a: f64, b: f64, c: f64) -> f64 {
    if a * b > 0.0 && b * c > 0.0 {
        sign(a) * a.abs().min(b.abs()).min(c.abs())
    } else {
        0.0
    }
}

/// Maxmod 函数
#[inline]
pub fn maxmod(a: f64, b: f64) -> f64 {
    if a * b > 0.0 {
        sign(a) * a.abs().max(b.abs())
    } else {
        0.0
    }
}

// ============================================================================
// 经典限制器实现
// ============================================================================

/// Minmod 限制器函数
///
/// φ(r) = max(0, min(1, r))
#[inline]
pub fn limiter_minmod(r: f64) -> f64 {
    0.0f64.max(1.0f64.min(r))
}

/// Superbee 限制器函数
///
/// φ(r) = max(0, min(1, 2r), min(2, r))
#[inline]
pub fn limiter_superbee(r: f64) -> f64 {
    0.0f64.max((1.0f64.min(2.0 * r)).max(2.0f64.min(r)))
}

/// Van Leer 限制器函数
///
/// φ(r) = (r + |r|) / (1 + |r|)
#[inline]
pub fn limiter_van_leer(r: f64) -> f64 {
    (r + r.abs()) / (1.0 + r.abs())
}

/// Van Albada 限制器函数
///
/// φ(r) = (r² + r) / (r² + 1)
#[inline]
pub fn limiter_van_albada(r: f64) -> f64 {
    if r <= 0.0 {
        0.0
    } else {
        (r * r + r) / (r * r + 1.0)
    }
}

/// Koren 限制器函数（三阶）
///
/// φ(r) = max(0, min(2r, (2 + r)/3, 2))
#[inline]
pub fn limiter_koren(r: f64) -> f64 {
    0.0f64.max((2.0 * r).min((2.0 + r) / 3.0).min(2.0))
}

/// MC (Monotonized Central) 限制器函数
///
/// φ(r) = max(0, min(2, 2r, (1 + r)/2))
#[inline]
pub fn limiter_mc(r: f64) -> f64 {
    0.0f64.max(2.0f64.min((2.0 * r).min((1.0 + r) / 2.0)))
}

/// 根据类型选择限制器
#[inline]
pub fn apply_limiter(limiter_type: LimiterType, r: f64) -> f64 {
    match limiter_type {
        LimiterType::None => 1.0,
        LimiterType::Minmod => limiter_minmod(r),
        LimiterType::Superbee => limiter_superbee(r),
        LimiterType::VanLeer => limiter_van_leer(r),
        LimiterType::VanAlbada => limiter_van_albada(r),
        LimiterType::Koren => limiter_koren(r),
        LimiterType::Mc => limiter_mc(r),
        LimiterType::Venkatakrishnan | LimiterType::BarthJespersen => {
            // 这些需要额外参数，使用默认 Van Leer
            limiter_van_leer(r)
        }
    }
}

// ============================================================================
// 非结构网格限制器上下文支持
// ============================================================================

/// 限制器上下文（非结构网格）
#[derive(Debug, Clone)]
pub struct LimiterContext {
    /// 单元中心值
    pub q_center: f64,
    /// 邻域最小值
    pub q_min: f64,
    /// 邻域最大值
    pub q_max: f64,
    /// 面重构值
    pub q_face: f64,
    /// 网格特征尺寸
    pub length_scale: f64,
    /// Venkatakrishnan K 参数
    pub venkat_k: f64,
}

impl Default for LimiterContext {
    fn default() -> Self {
        Self {
            q_center: 0.0,
            q_min: 0.0,
            q_max: 0.0,
            q_face: 0.0,
            length_scale: 1.0,
            venkat_k: 5.0,
        }
    }
}

/// 带上下文的限制器（支持 Venkatakrishnan / Barth-Jespersen）
pub fn apply_limiter_with_context(
    limiter_type: LimiterType,
    r: f64,
    ctx: Option<&LimiterContext>,
) -> f64 {
    match limiter_type {
        LimiterType::Venkatakrishnan => {
            if let Some(ctx) = ctx {
                let limiter = VenkatakrishnanLimiter::new(ctx.venkat_k, ctx.length_scale);
                let delta_minus = ctx.q_min - ctx.q_center;
                let delta_plus = ctx.q_max - ctx.q_center;
                let delta_face = ctx.q_face - ctx.q_center;
                limiter.compute(delta_minus, delta_plus, delta_face)
            } else {
                limiter_van_leer(r)
            }
        }
        LimiterType::BarthJespersen => {
            if let Some(ctx) = ctx {
                BarthJespersenLimiter::compute(ctx.q_center, ctx.q_min, ctx.q_max, ctx.q_face)
            } else {
                limiter_van_leer(r)
            }
        }
        _ => apply_limiter(limiter_type, r),
    }
}

/// 非结构网格梯度限制器应用器
pub struct UnstructuredLimiterApplicator {
    limiter_type: LimiterType,
    venkat_k: f64,
}

impl UnstructuredLimiterApplicator {
    pub fn new(limiter_type: LimiterType) -> Self {
        Self {
            limiter_type,
            venkat_k: 5.0,
        }
    }

    pub fn with_venkat_k(mut self, k: f64) -> Self {
        self.venkat_k = k;
        self
    }

    /// 计算单元限制因子
    pub fn compute_cell_limiter(
        &self,
        q_center: f64,
        neighbor_values: &[f64],
        face_reconstructed: &[f64],
        cell_length: f64,
    ) -> f64 {
        let q_min = neighbor_values.iter().copied().fold(q_center, f64::min);
        let q_max = neighbor_values.iter().copied().fold(q_center, f64::max);

        let mut phi_min: f64 = 1.0;

        for &q_face in face_reconstructed {
            let delta_face = q_face - q_center;
            let delta_ref = if delta_face >= 0.0 {
                q_max - q_center
            } else {
                q_min - q_center
            };

            let r = if delta_ref.abs() > 1e-12 { delta_face / delta_ref } else { 0.0 };

            let ctx = LimiterContext {
                q_center,
                q_min,
                q_max,
                q_face,
                length_scale: cell_length,
                venkat_k: self.venkat_k,
            };

            let phi = apply_limiter_with_context(self.limiter_type, r, Some(&ctx));
            if phi < phi_min {
                phi_min = phi;
            }
        }

        phi_min
    }
}

// ============================================================================
// Venkatakrishnan 限制器
// ============================================================================

/// Venkatakrishnan 限制器
///
/// 非结构网格的可微限制器，避免极端值处的收敛问题
pub struct VenkatakrishnanLimiter {
    /// 平滑参数 K
    k: f64,
    /// 参考长度（网格尺寸）
    length_scale: f64,
    /// 预计算的 epsilon² = (K * h)³
    eps_sq: f64,
}

impl VenkatakrishnanLimiter {
    /// 创建限制器
    ///
    /// # 参数
    ///
    /// * `k` - 平滑参数（通常 1-10，越大越平滑）
    /// * `length_scale` - 参考长度尺寸
    pub fn new(k: f64, length_scale: f64) -> Self {
        let kh = k * length_scale;
        Self {
            k,
            length_scale,
            eps_sq: kh * kh * kh,
        }
    }

    /// 更新网格尺寸
    pub fn set_length_scale(&mut self, length_scale: f64) {
        self.length_scale = length_scale;
        let kh = self.k * length_scale;
        self.eps_sq = kh * kh * kh;
    }

    /// 计算单个面的限制因子
    ///
    /// # 参数
    ///
    /// * `delta_minus` - 负方向梯度增量 (q_min - q_i)
    /// * `delta_plus` - 正方向梯度增量 (q_max - q_i)
    /// * `delta_face` - 面重构增量 (q_face - q_i)
    pub fn compute(&self, delta_minus: f64, delta_plus: f64, delta_face: f64) -> f64 {
        if delta_face.abs() < 1e-12 {
            return 1.0;
        }

        let delta = if delta_face > 0.0 { delta_plus } else { delta_minus };

        if delta.abs() < 1e-12 {
            return 0.0;
        }

        let y = delta / delta_face;
        let y2 = y * y;
        
        // Venkatakrishnan 公式
        let num = y2 + 2.0 * y + self.eps_sq;
        let den = y2 + y + 2.0 + self.eps_sq;
        
        (num / den).clamp(0.0, 1.0)
    }

    /// 计算单元的限制因子（考虑所有面）
    pub fn compute_cell_limiter(
        &self,
        q_center: f64,
        q_min: f64,
        q_max: f64,
        q_face_values: &[f64],
    ) -> f64 {
        let mut phi: f64 = 1.0;

        let delta_minus = q_min - q_center;
        let delta_plus = q_max - q_center;

        for &q_face in q_face_values {
            let delta_face = q_face - q_center;
            let phi_face = self.compute(delta_minus, delta_plus, delta_face);
            phi = phi.min(phi_face);
        }

        phi
    }
}

// ============================================================================
// Barth-Jespersen 限制器
// ============================================================================

/// Barth-Jespersen 限制器
///
/// 确保重构值不超过邻居极值
pub struct BarthJespersenLimiter;

impl BarthJespersenLimiter {
    /// 计算限制因子
    ///
    /// # 参数
    ///
    /// * `q_center` - 单元中心值
    /// * `q_min` - 邻域最小值
    /// * `q_max` - 邻域最大值
    /// * `q_face` - 面重构值
    pub fn compute(q_center: f64, q_min: f64, q_max: f64, q_face: f64) -> f64 {
        let delta = q_face - q_center;

        if delta.abs() < 1e-12 {
            return 1.0;
        }

        if delta > 0.0 {
            let delta_max = q_max - q_center;
            if delta_max.abs() < 1e-12 {
                0.0
            } else {
                (delta_max / delta).min(1.0)
            }
        } else {
            let delta_min = q_min - q_center;
            if delta_min.abs() < 1e-12 {
                0.0
            } else {
                (delta_min / delta).min(1.0)
            }
        }
    }

    /// 计算单元的限制因子
    pub fn compute_cell_limiter(
        q_center: f64,
        q_min: f64,
        q_max: f64,
        q_face_values: &[f64],
    ) -> f64 {
        let mut phi: f64 = 1.0;

        for &q_face in q_face_values {
            let phi_face = Self::compute(q_center, q_min, q_max, q_face);
            phi = phi.min(phi_face);
        }

        phi
    }
}

// ============================================================================
// MUSCL 重构
// ============================================================================

/// MUSCL 重构配置
#[deprecated(note = "Use crate::numerics::reconstruction::MusclConfig on the main engine path.")]
#[derive(Debug, Clone)]
pub struct MusclConfig {
    /// 限制器类型
    pub limiter: LimiterType,
    /// kappa 参数（-1 = 完全迎风，0 = Fromm，1/3 = 三阶，1 = 中心）
    pub kappa: f64,
    /// 是否使用特征变量
    pub characteristic_decomposition: bool,
}

impl Default for MusclConfig {
    fn default() -> Self {
        Self {
            limiter: LimiterType::VanLeer,
            kappa: 1.0 / 3.0, // 三阶
            characteristic_decomposition: false,
        }
    }
}

/// MUSCL 重构器
#[deprecated(note = "Use crate::numerics::reconstruction::MusclReconstructor on the main engine path.")]
pub struct MusclReconstructor {
    config: MusclConfig,
}

impl MusclReconstructor {
    /// 创建重构器
    pub fn new(config: MusclConfig) -> Self {
        Self { config }
    }

    /// 重构面值
    ///
    /// # 参数
    ///
    /// * `q_left` - 左单元值 (i-1)
    /// * `q_center` - 中心单元值 (i)
    /// * `q_right` - 右单元值 (i+1)
    ///
    /// # 返回
    ///
    /// (左侧重构值, 右侧重构值) 即 (q_{i+1/2}^L, q_{i+1/2}^R)
    pub fn reconstruct(&self, q_left: f64, q_center: f64, q_right: f64) -> (f64, f64) {
        let delta_l = q_center - q_left;
        let delta_r = q_right - q_center;

        // 计算斜率比
        let r_l = if delta_r.abs() > 1e-12 {
            delta_l / delta_r
        } else {
            0.0
        };

        let r_r = if delta_l.abs() > 1e-12 {
            delta_r / delta_l
        } else {
            0.0
        };

        // 应用限制器
        let phi_l = apply_limiter(self.config.limiter, r_l);
        let phi_r = apply_limiter(self.config.limiter, r_r);

        // 重构
        let kappa = self.config.kappa;
        let q_l_face = q_center + 0.25 * ((1.0 - kappa) * phi_l * delta_l + (1.0 + kappa) * phi_r * delta_r);
        let q_r_face = q_right - 0.25 * ((1.0 + kappa) * phi_l * delta_l + (1.0 - kappa) * phi_r * delta_r);

        (q_l_face, q_r_face)
    }

    /// 简化版：只重构左侧
    pub fn reconstruct_left(&self, q_left: f64, q_center: f64, q_right: f64) -> f64 {
        let delta_l = q_center - q_left;
        let delta_r = q_right - q_center;

        let r = if delta_r.abs() > 1e-12 {
            delta_l / delta_r
        } else {
            0.0
        };

        let phi = apply_limiter(self.config.limiter, r);
        let kappa = self.config.kappa;

        q_center + 0.25 * ((1.0 - kappa) * phi * delta_l + (1.0 + kappa) * phi * delta_r)
    }
}

// ============================================================================
// 梯度限制
// ============================================================================

/// 对梯度应用限制
pub fn limit_gradient(
    grad_x: f64,
    grad_y: f64,
    phi: f64,
) -> (f64, f64) {
    (grad_x * phi, grad_y * phi)
}

/// 计算梯度限制因子（Green-Gauss 梯度 + Venkatakrishnan）
pub fn compute_gradient_limiter(
    q_center: f64,
    q_neighbors: &[f64],
    q_face_reconstructed: &[f64],
    k: f64,
    h: f64,
) -> f64 {
    let q_min = q_neighbors.iter().fold(f64::INFINITY, |a, &b| a.min(b)).min(q_center);
    let q_max = q_neighbors.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)).max(q_center);

    let limiter = VenkatakrishnanLimiter::new(k, h);
    limiter.compute_cell_limiter(q_center, q_min, q_max, q_face_reconstructed)
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmod() {
        assert_eq!(minmod(1.0, 2.0), 1.0);
        assert_eq!(minmod(-1.0, -2.0), -1.0);
        assert_eq!(minmod(1.0, -2.0), 0.0);
        assert_eq!(minmod(-1.0, 2.0), 0.0);
    }

    #[test]
    fn test_limiter_functions() {
        // r = 1 时，所有限制器应返回 1
        assert!((limiter_minmod(1.0) - 1.0).abs() < 1e-10);
        assert!((limiter_superbee(1.0) - 1.0).abs() < 0.01); // superbee 返回 min(2, 1) = 1
        assert!((limiter_van_leer(1.0) - 1.0).abs() < 1e-10);
        assert!((limiter_van_albada(1.0) - 1.0).abs() < 1e-10);

        // r < 0 时，应返回 0
        assert_eq!(limiter_minmod(-0.5), 0.0);
        assert_eq!(limiter_van_albada(-0.5), 0.0);
    }

    #[test]
    fn test_venkatakrishnan() {
        let limiter = VenkatakrishnanLimiter::new(5.0, 1.0);
        
        // 无梯度
        let phi = limiter.compute(0.0, 0.0, 0.0);
        assert_eq!(phi, 1.0);

        // 正常情况
        let phi = limiter.compute(-1.0, 1.0, 0.5);
        assert!((0.0..=1.0).contains(&phi));
    }

    #[test]
    fn test_barth_jespersen() {
        // 不超过极值
        let phi = BarthJespersenLimiter::compute(1.0, 0.0, 2.0, 1.5);
        assert_eq!(phi, 1.0);

        // 超过极值
        let phi = BarthJespersenLimiter::compute(1.0, 0.0, 2.0, 3.0);
        assert!((phi - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_muscl_reconstruct() {
        let config = MusclConfig {
            limiter: LimiterType::Minmod,
            kappa: 0.0, // Fromm
            ..Default::default()
        };
        let reconstructor = MusclReconstructor::new(config);

        // 线性场
        let (q_l, q_r) = reconstructor.reconstruct(0.0, 1.0, 2.0);
        
        // 应该产生二阶精度的重构
        assert!(q_l > 1.0);
        assert!(q_r < 2.0);
    }

    #[test]
    fn test_muscl_constant_field() {
        let config = MusclConfig::default();
        let reconstructor = MusclReconstructor::new(config);

        // 常数场
        let (q_l, q_r) = reconstructor.reconstruct(1.0, 1.0, 1.0);
        
        assert!((q_l - 1.0).abs() < 1e-10);
        assert!((q_r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_venkatakrishnan_with_context() {
        let ctx = LimiterContext {
            q_center: 1.0,
            q_min: 0.5,
            q_max: 1.5,
            q_face: 1.2,
            length_scale: 1.0,
            venkat_k: 5.0,
        };
        let phi = apply_limiter_with_context(LimiterType::Venkatakrishnan, 0.0, Some(&ctx));
        assert!((0.0..=1.0).contains(&phi));
    }

    #[test]
    fn test_barth_jespersen_with_context() {
        let ctx = LimiterContext {
            q_center: 1.0,
            q_min: 0.5,
            q_max: 1.5,
            q_face: 1.3,
            length_scale: 1.0,
            venkat_k: 5.0,
        };
        let phi = apply_limiter_with_context(LimiterType::BarthJespersen, 0.0, Some(&ctx));
        assert!((0.0..=1.0).contains(&phi));
    }

    #[test]
    fn test_limiter_symmetry() {
        // Van Leer 应该满足 φ(r) = r * φ(1/r)
        for r in [0.5, 1.0, 2.0, 4.0] {
            let phi_r = limiter_van_leer(r);
            let phi_inv = limiter_van_leer(1.0 / r);
            assert!((phi_r - r * phi_inv).abs() < 1e-10);
        }
    }
}
