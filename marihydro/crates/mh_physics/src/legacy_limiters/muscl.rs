use super::{apply_limiter, LegacyLimiterType, VenkatakrishnanLimiter};

// MUSCL 重构
// ============================================================================

/// MUSCL 重构配置
#[derive(Debug, Clone)]
pub struct LegacyMusclConfig {
    /// 限制器类型
    pub limiter: LegacyLimiterType,
    /// kappa 参数（-1 = 完全迎风，0 = Fromm，1/3 = 三阶，1 = 中心）
    pub kappa: f64,
    /// 是否使用特征变量
    pub characteristic_decomposition: bool,
}

#[deprecated(note = "Use crate::numerics::reconstruction::MusclConfig on the main engine path.")]
pub use LegacyMusclConfig as MusclConfig;

impl Default for LegacyMusclConfig {
    fn default() -> Self {
        Self {
            limiter: LegacyLimiterType::VanLeer,
            kappa: 1.0 / 3.0, // 三阶
            characteristic_decomposition: false,
        }
    }
}

/// MUSCL 重构器
pub struct LegacyMusclReconstructor {
    config: LegacyMusclConfig,
}

#[deprecated(note = "Use crate::numerics::reconstruction::MusclReconstructor on the main engine path.")]
pub use LegacyMusclReconstructor as MusclReconstructor;

impl LegacyMusclReconstructor {
    /// 创建重构器
    pub fn new(config: LegacyMusclConfig) -> Self {
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
