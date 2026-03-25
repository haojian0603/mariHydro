use super::{apply_limiter, limiter_van_leer, LegacyLimiterType};

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
    limiter_type: LegacyLimiterType,
    r: f64,
    ctx: Option<&LimiterContext>,
) -> f64 {
    match limiter_type {
        LegacyLimiterType::Venkatakrishnan => {
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
        LegacyLimiterType::BarthJespersen => {
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
    limiter_type: LegacyLimiterType,
    venkat_k: f64,
}

impl UnstructuredLimiterApplicator {
    pub fn new(limiter_type: LegacyLimiterType) -> Self {
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
