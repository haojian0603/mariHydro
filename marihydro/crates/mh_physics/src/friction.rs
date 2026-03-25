// crates/mh_physics/src/friction.rs

//! 底摩擦与阻力模型
//!
//! 提供工业级摩擦计算，支持：
//! - Manning 公式
//! - Chezy 公式
//! - 显式/隐式/半隐式处理
//! - 空间变化糙率
//!
//! # 设计原则
//!
//! 1. **物理准确**：正确的量纲和极限行为
//! 2. **数值稳定**：避免浅水处的刚性问题
//! 3. **灵活性**：支持多种公式和处理方式

// ============================================================================
// 摩擦公式类型
// ============================================================================

/// 摩擦公式类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrictionFormula {
    /// 无摩擦
    None,
    /// Manning 公式
    Manning,
    /// Chezy 公式
    Chezy,
    /// 达西-魏斯巴赫
    DarcyWeisbach,
    /// 线性阻力
    Linear,
    /// 二次阻力
    Quadratic,
}

impl Default for FrictionFormula {
    fn default() -> Self {
        Self::Manning
    }
}

/// 摩擦处理方式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrictionTreatment {
    /// 显式处理
    Explicit,
    /// 隐式处理（推荐）
    Implicit,
    /// 半隐式处理
    SemiImplicit,
    /// 点隐式
    PointImplicit,
}

impl Default for FrictionTreatment {
    fn default() -> Self {
        Self::Implicit // 推荐隐式处理以获得稳定性
    }
}

// ============================================================================
// 摩擦配置
// ============================================================================

/// 摩擦配置
#[derive(Debug, Clone)]
pub struct FrictionConfig {
    /// 摩擦公式
    pub formula: FrictionFormula,
    /// 处理方式
    pub treatment: FrictionTreatment,
    /// 全局 Manning 系数 (m^{-1/3} s)
    pub manning_n: f64,
    /// 全局 Chezy 系数 (m^{1/2}/s)
    pub chezy_c: f64,
    /// 达西-魏斯巴赫摩擦系数
    pub friction_factor: f64,
    /// 最小水深阈值（用于摩擦计算）
    pub min_depth: f64,
    /// 最大摩擦系数（限制器）
    pub max_friction_coefficient: f64,
    /// 是否使用空间变化糙率
    pub spatially_varying: bool,
}

impl Default for FrictionConfig {
    fn default() -> Self {
        Self {
            formula: FrictionFormula::Manning,
            treatment: FrictionTreatment::Implicit,
            manning_n: 0.025,   // 典型天然河道值
            chezy_c: 50.0,      // 对应 Manning n ≈ 0.025
            friction_factor: 0.02,
            min_depth: 1e-3,    // 1 mm
            max_friction_coefficient: 100.0,
            spatially_varying: false,
        }
    }
}

impl FrictionConfig {
    /// 光滑表面配置
    pub fn smooth() -> Self {
        Self {
            manning_n: 0.010,
            chezy_c: 80.0,
            ..Default::default()
        }
    }

    /// 天然河道配置
    pub fn natural_river() -> Self {
        Self {
            manning_n: 0.030,
            chezy_c: 45.0,
            ..Default::default()
        }
    }

    /// 洪泛区配置
    pub fn floodplain() -> Self {
        Self {
            manning_n: 0.050,
            chezy_c: 30.0,
            ..Default::default()
        }
    }

    /// 城市地区配置
    pub fn urban() -> Self {
        Self {
            manning_n: 0.020,
            chezy_c: 55.0,
            ..Default::default()
        }
    }
}

// ============================================================================
// 摩擦计算器
// ============================================================================

/// 摩擦计算器
pub struct FrictionCalculator {
    config: FrictionConfig,
    /// 空间变化 Manning 系数（如果启用）
    manning_field: Option<Vec<f64>>,
}

impl FrictionCalculator {
    /// 创建计算器
    pub fn new(config: FrictionConfig) -> Self {
        Self {
            config,
            manning_field: None,
        }
    }

    /// 设置空间变化的 Manning 系数
    pub fn set_manning_field(&mut self, field: Vec<f64>) {
        self.config.spatially_varying = true;
        self.manning_field = Some(field);
    }

    /// 获取局部 Manning 系数
    fn get_manning_n(&self, cell: usize) -> f64 {
        if let Some(ref field) = self.manning_field {
            field.get(cell).copied().unwrap_or(self.config.manning_n)
        } else {
            self.config.manning_n
        }
    }

    /// 计算摩擦系数 cf
    ///
    /// cf = g * n² / h^{1/3} (Manning)
    /// cf = g / C² (Chezy)
    ///
    /// # 参数
    ///
    /// * `h` - 水深 (m)
    /// * `cell` - 单元索引（用于空间变化糙率）
    /// * `g` - 重力加速度
    pub fn friction_coefficient(&self, h: f64, cell: usize, g: f64) -> f64 {
        if h < self.config.min_depth {
            return 0.0;
        }

        let cf = match self.config.formula {
            FrictionFormula::None => 0.0,
            FrictionFormula::Manning => {
                let n = self.get_manning_n(cell);
                g * n * n / h.powf(1.0 / 3.0)
            }
            FrictionFormula::Chezy => {
                let c = self.config.chezy_c;
                g / (c * c)
            }
            FrictionFormula::DarcyWeisbach => {
                self.config.friction_factor / (8.0 * h)
            }
            FrictionFormula::Linear => {
                self.config.friction_factor
            }
            FrictionFormula::Quadratic => {
                self.config.friction_factor / h
            }
        };

        cf.min(self.config.max_friction_coefficient)
    }

    /// 计算摩擦源项 (显式)
    ///
    /// S_f = -cf * |u| * u
    ///
    /// # 返回
    ///
    /// (S_hu, S_hv) 动量源项
    pub fn compute_explicit(
        &self,
        h: f64,
        hu: f64,
        hv: f64,
        cell: usize,
        g: f64,
    ) -> (f64, f64) {
        if h < self.config.min_depth {
            return (0.0, 0.0);
        }

        let u = hu / h;
        let v = hv / h;
        let speed = (u * u + v * v).sqrt();

        let cf = self.friction_coefficient(h, cell, g);

        let s_hu = -cf * speed * hu;
        let s_hv = -cf * speed * hv;

        (s_hu, s_hv)
    }

    /// 计算隐式摩擦因子
    ///
    /// 用于 hu^{n+1} = hu^n + dt * rhs - dt * cf * |u| * hu^{n+1}
    /// 求解得: hu^{n+1} = (hu^n + dt * rhs) / (1 + dt * cf * |u|)
    ///
    /// # 返回
    ///
    /// 隐式系数 1 / (1 + dt * cf * |u|)
    pub fn implicit_factor(
        &self,
        h: f64,
        speed: f64,
        cell: usize,
        g: f64,
        dt: f64,
    ) -> f64 {
        if h < self.config.min_depth {
            return 1.0;
        }

        let cf = self.friction_coefficient(h, cell, g);
        1.0 / (1.0 + dt * cf * speed)
    }

    /// 应用隐式摩擦
    ///
    /// # 参数
    ///
    /// * `h` - 水深
    /// * `hu_star, hv_star` - 未考虑摩擦的动量
    /// * `cell` - 单元索引
    /// * `g` - 重力加速度
    /// * `dt` - 时间步
    ///
    /// # 返回
    ///
    /// (hu_new, hv_new) 摩擦后的动量
    pub fn apply_implicit(
        &self,
        h: f64,
        hu_star: f64,
        hv_star: f64,
        cell: usize,
        g: f64,
        dt: f64,
    ) -> (f64, f64) {
        if h < self.config.min_depth {
            return (0.0, 0.0);
        }

        // 使用 hu_star 估计速度
        let u_star = hu_star / h;
        let v_star = hv_star / h;
        let speed_star = (u_star * u_star + v_star * v_star).sqrt();

        let factor = self.implicit_factor(h, speed_star, cell, g, dt);

        (hu_star * factor, hv_star * factor)
    }

    /// 批量应用隐式摩擦
    pub fn apply_implicit_batch(
        &self,
        h: &[f64],
        hu_star: &mut [f64],
        hv_star: &mut [f64],
        g: f64,
        dt: f64,
    ) {
        let n = h.len();
        for i in 0..n {
            let (hu_new, hv_new) = self.apply_implicit(
                h[i], hu_star[i], hv_star[i], i, g, dt
            );
            hu_star[i] = hu_new;
            hv_star[i] = hv_new;
        }
    }

    /// 半隐式处理（使用上一步速度）
    pub fn apply_semi_implicit(
        &self,
        h: f64,
        hu_old: f64,
        hv_old: f64,
        hu_star: f64,
        hv_star: f64,
        cell: usize,
        g: f64,
        dt: f64,
    ) -> (f64, f64) {
        if h < self.config.min_depth {
            return (0.0, 0.0);
        }

        // 使用旧速度
        let u_old = hu_old / h.max(self.config.min_depth);
        let v_old = hv_old / h.max(self.config.min_depth);
        let speed_old = (u_old * u_old + v_old * v_old).sqrt();

        let factor = self.implicit_factor(h, speed_old, cell, g, dt);

        (hu_star * factor, hv_star * factor)
    }
}

// ============================================================================
// 其他源项
// ============================================================================

/// 科里奥利力计算
///
/// # 参数
///
/// * `latitude` - 纬度 (度)
///
/// # 返回
///
/// 科里奥利参数 f (1/s)
pub fn coriolis_parameter(latitude: f64) -> f64 {
    const OMEGA: f64 = 7.2921e-5; // 地球自转角速度 (rad/s)
    2.0 * OMEGA * latitude.to_radians().sin()
}

/// 计算科里奥利源项
pub fn coriolis_source(f: f64, hu: f64, hv: f64) -> (f64, f64) {
    // S_hu = f * hv
    // S_hv = -f * hu
    (f * hv, -f * hu)
}

/// 风应力计算
///
/// τ = ρ_a * C_d * |W| * W
///
/// # 参数
///
/// * `wind_u, wind_v` - 风速分量 (m/s)
/// * `cd` - 拖曳系数 (默认约 0.0013)
/// * `rho_air` - 空气密度 (kg/m³)
/// * `rho_water` - 水密度 (kg/m³)
/// * `h` - 水深 (m)
pub fn wind_stress_source(
    wind_u: f64,
    wind_v: f64,
    cd: f64,
    rho_air: f64,
    rho_water: f64,
    h: f64,
) -> (f64, f64) {
    if h < 1e-3 {
        return (0.0, 0.0);
    }

    let wind_speed = (wind_u * wind_u + wind_v * wind_v).sqrt();
    let coef = rho_air * cd * wind_speed / (rho_water * h);

    (coef * wind_u, coef * wind_v)
}

/// 大气压梯度源项
pub fn pressure_gradient_source(
    dp_dx: f64,
    dp_dy: f64,
    rho_water: f64,
    h: f64,
) -> (f64, f64) {
    if h < 1e-3 {
        return (0.0, 0.0);
    }

    // S = -h/ρ * ∇p
    let coef = -1.0 / rho_water;
    (coef * dp_dx, coef * dp_dy)
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manning_friction() {
        let config = FrictionConfig::default();
        let calc = FrictionCalculator::new(config);

        // h = 1m, n = 0.025
        // cf = g * n² / h^{1/3} = 9.81 * 0.025² / 1^{1/3} ≈ 0.00613
        let cf = calc.friction_coefficient(1.0, 0, 9.81);
        assert!((cf - 0.00613).abs() < 0.0001);
    }

    #[test]
    fn test_chezy_friction() {
        let config = FrictionConfig {
            formula: FrictionFormula::Chezy,
            chezy_c: 50.0,
            ..Default::default()
        };
        let calc = FrictionCalculator::new(config);

        // cf = g / C² = 9.81 / 2500 ≈ 0.00392
        let cf = calc.friction_coefficient(1.0, 0, 9.81);
        assert!((cf - 0.003924).abs() < 0.0001);
    }

    #[test]
    fn test_explicit_friction() {
        let config = FrictionConfig::default();
        let calc = FrictionCalculator::new(config);

        // h = 1m, hu = 1 m²/s, hv = 0
        let (s_hu, s_hv) = calc.compute_explicit(1.0, 1.0, 0.0, 0, 9.81);

        // S_hu = -cf * |u| * hu = -0.00613 * 1 * 1 ≈ -0.00613
        assert!(s_hu < 0.0);
        assert!(s_hv.abs() < 1e-10);
    }

    #[test]
    fn test_implicit_friction() {
        let config = FrictionConfig::default();
        let calc = FrictionCalculator::new(config);

        let dt = 0.1;
        let (hu_new, _hv_new) = calc.apply_implicit(1.0, 1.0, 0.0, 0, 9.81, dt);

        // 隐式处理应该减小动量
        assert!(hu_new < 1.0);
        assert!(hu_new > 0.0);
    }

    #[test]
    fn test_dry_cell_friction() {
        let config = FrictionConfig::default();
        let calc = FrictionCalculator::new(config);

        // 干燥单元不应有摩擦
        let cf = calc.friction_coefficient(1e-6, 0, 9.81);
        assert_eq!(cf, 0.0);
    }

    #[test]
    fn test_coriolis() {
        // 赤道 f = 0
        let f_eq = coriolis_parameter(0.0);
        assert!(f_eq.abs() < 1e-10);

        // 北极 f ≈ 2Ω
        let f_pole = coriolis_parameter(90.0);
        assert!((f_pole - 2.0 * 7.2921e-5).abs() < 1e-10);

        // 30°N
        let f_30 = coriolis_parameter(30.0);
        assert!((f_30 - 7.2921e-5).abs() < 1e-10);
    }

    #[test]
    fn test_wind_stress() {
        // 10 m/s 风速
        let (s_hu, s_hv) = wind_stress_source(
            10.0, 0.0, 0.0013, 1.225, 1025.0, 1.0
        );

        // 应该有正的 x 方向加速度
        assert!(s_hu > 0.0);
        assert!(s_hv.abs() < 1e-10);
    }

    #[test]
    fn test_spatially_varying_manning() {
        let config = FrictionConfig {
            spatially_varying: true,
            ..Default::default()
        };
        let mut calc = FrictionCalculator::new(config);

        // 设置空间变化场
        calc.set_manning_field(vec![0.01, 0.02, 0.03]);

        let cf_0 = calc.friction_coefficient(1.0, 0, 9.81);
        let cf_1 = calc.friction_coefficient(1.0, 1, 9.81);
        let cf_2 = calc.friction_coefficient(1.0, 2, 9.81);

        // 应该不同
        assert!(cf_0 < cf_1);
        assert!(cf_1 < cf_2);
    }
}
