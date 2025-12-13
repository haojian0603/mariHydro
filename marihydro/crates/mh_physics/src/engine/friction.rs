//! Manning 摩擦源项计算模块
//!
//! 实现曼宁公式的底部摩擦源项，支持：
//! - 显式和半隐式时间离散
//! - 并行计算（使用 rayon）
//! - 干湿处理
//!
//! # 曼宁公式
//!
//! 摩擦源项:
//! ```text
//! S_f = -g * n^2 * |V| / h^(4/3) * q
//! ```
//! 其中:
//! - `n` - 曼宁糙率系数
//! - `V` - 速度大小 sqrt(u^2 + v^2)
//! - `h` - 水深
//! - `q` - 流量 (hu, hv)
//!
//! # 半隐式处理
//!
//! 为避免刚性问题（浅水时摩擦项可能很大），使用半隐式方法：
//! ```text
//! q^{n+1} = q^n + dt * (... - C_f * q^{n+1})
//! => q^{n+1} = q^n / (1 + dt * C_f)
//! ```
//! 其中 `C_f = g * n^2 * |V| / h^(4/3)`

use rayon::prelude::*;

/// 摩擦计算配置
#[derive(Debug, Clone)]
pub struct FrictionConfig {
    /// 是否使用半隐式时间离散
    pub semi_implicit: bool,
    /// 最小水深阈值（低于此值不计算摩擦）
    pub h_min: f64, // ALLOW_F64: Layer 4 配置参数
    /// 默认曼宁糙率（当未提供时使用）
    pub default_manning_n: f64, // ALLOW_F64: Layer 4 配置参数
    /// 是否启用并行计算
    pub parallel: bool,
    /// 并行阈值（单元数超过此值时使用并行）
    pub parallel_threshold: usize,
}

impl Default for FrictionConfig {
    fn default() -> Self {
        Self {
            semi_implicit: true,
            h_min: 1e-4,
            default_manning_n: 0.03,
            parallel: true,
            parallel_threshold: 1000,
        }
    }
}

impl FrictionConfig {
    /// 创建显式配置
    pub fn explicit() -> Self {
        Self {
            semi_implicit: false,
            ..Default::default()
        }
    }

    /// 创建半隐式配置
    pub fn semi_implicit() -> Self {
        Self {
            semi_implicit: true,
            ..Default::default()
        }
    }
}

/// Manning 摩擦源项计算器
#[derive(Debug, Clone)]
pub struct ManningFriction {
    /// 配置
    config: FrictionConfig,
    /// 重力加速度
    g: f64, // ALLOW_F64: Layer 4 配置参数
}

impl ManningFriction {
    /// 创建摩擦计算器
    pub fn new(g: f64) -> Self { // ALLOW_F64: 与 PhysicsMesh 配合/物理参数
        Self {
            config: FrictionConfig::default(),
            g,
        }
    }

    /// 使用指定配置创建
    pub fn with_config(g: f64, config: FrictionConfig) -> Self { // ALLOW_F64: 与 PhysicsMesh 配合/物理参数
        Self { config, g }
    }

    /// 设置配置
    pub fn set_config(&mut self, config: FrictionConfig) {
        self.config = config;
    }

    /// 获取配置
    pub fn config(&self) -> &FrictionConfig {
        &self.config
    }

    /// 计算单个单元的摩擦系数
    ///
    /// 返回 C_f = g * n^2 * |V| / h^(4/3)
    #[inline]
    pub fn compute_friction_coefficient(&self, h: f64, hu: f64, hv: f64, manning_n: f64) -> f64 { // ALLOW_F64: 与 PhysicsMesh 配合/物理参数
        if h < self.config.h_min {
            return 0.0;
        }

        // 计算速度大小
        let u = hu / h;
        let v = hv / h;
        let speed = (u * u + v * v).sqrt();

        // 曼宁摩擦系数
        // C_f = g * n^2 * |V| / h^(4/3)
        let h_pow = h.powf(4.0 / 3.0);
        if h_pow < 1e-12 {
            return 0.0;
        }

        self.g * manning_n * manning_n * speed / h_pow
    }

    /// 计算显式摩擦源项
    ///
    /// 返回 (source_hu, source_hv)
    #[inline]
    pub fn compute_explicit_source(&self, h: f64, hu: f64, hv: f64, manning_n: f64) -> (f64, f64) { // ALLOW_F64: 与 PhysicsMesh 配合/物理参数
        let cf = self.compute_friction_coefficient(h, hu, hv, manning_n);
        (-cf * hu, -cf * hv)
    }

    /// 应用半隐式摩擦更新
    ///
    /// 更新后的流量: q^{n+1} = q^n / (1 + dt * C_f)
    #[inline]
    // ALLOW_F64: 与 PhysicsMesh 配合/物理参数
    pub fn apply_semi_implicit(
        &self,
        h: f64, // ALLOW_F64: 与 PhysicsMesh 配合
        hu: f64, // ALLOW_F64: 与 PhysicsMesh 配合
        hv: f64, // ALLOW_F64: 与 PhysicsMesh 配合
        manning_n: f64, // ALLOW_F64: 物理参数
        dt: f64, // ALLOW_F64: 时间步长参数
    ) -> (f64, f64) {
        let cf = self.compute_friction_coefficient(h, hu, hv, manning_n);
        let factor = 1.0 / (1.0 + dt * cf);
        (hu * factor, hv * factor)
    }

    /// 批量计算显式摩擦源项
    ///
    /// # 参数
    /// - `h`: 水深数组
    /// - `hu`, `hv`: 流量数组
    /// - `manning_n`: 曼宁系数数组（可以是单元值，也可以是常数填充）
    /// - `source_hu`, `source_hv`: 输出源项数组（将累加到现有值）
    pub fn compute_sources_batch(
        &self,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        manning_n: &[f64],
        source_hu: &mut [f64],
        source_hv: &mut [f64],
    ) {
        let n = h.len();
        assert_eq!(n, hu.len());
        assert_eq!(n, hv.len());
        assert!(manning_n.len() == n || manning_n.len() == 1);
        assert_eq!(n, source_hu.len());
        assert_eq!(n, source_hv.len());

        let use_uniform_n = manning_n.len() == 1;
        let uniform_n = if use_uniform_n { manning_n[0] } else { 0.0 };

        if self.config.parallel && n >= self.config.parallel_threshold {
            self.compute_sources_parallel(h, hu, hv, manning_n, use_uniform_n, uniform_n, source_hu, source_hv);
        } else {
            self.compute_sources_serial(h, hu, hv, manning_n, use_uniform_n, uniform_n, source_hu, source_hv);
        }
    }

    /// 串行计算源项
    fn compute_sources_serial(
        &self,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        manning_n: &[f64],
        use_uniform_n: bool,
        uniform_n: f64, // ALLOW_F64: 与 PhysicsMesh 配合/物理参数
        source_hu: &mut [f64],
        source_hv: &mut [f64],
    ) {
        for i in 0..h.len() {
            let n_val = if use_uniform_n { uniform_n } else { manning_n[i] };
            let (s_hu, s_hv) = self.compute_explicit_source(h[i], hu[i], hv[i], n_val);
            source_hu[i] += s_hu;
            source_hv[i] += s_hv;
        }
    }

    /// 并行计算源项
    fn compute_sources_parallel(
        &self,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        manning_n: &[f64],
        use_uniform_n: bool,
        uniform_n: f64, // ALLOW_F64: 与 PhysicsMesh 配合/物理参数
        source_hu: &mut [f64],
        source_hv: &mut [f64],
    ) {
        // 并行计算，每个单元独立
        source_hu
            .par_iter_mut()
            .zip(source_hv.par_iter_mut())
            .zip(h.par_iter())
            .zip(hu.par_iter())
            .zip(hv.par_iter())
            .enumerate()
            .for_each(|(i, ((((s_hu, s_hv), &h_i), &hu_i), &hv_i))| {
                let n_val = if use_uniform_n { uniform_n } else { manning_n[i] };
                let (src_hu, src_hv) = self.compute_explicit_source(h_i, hu_i, hv_i, n_val);
                *s_hu += src_hu;
                *s_hv += src_hv;
            });
    }

    /// 批量应用半隐式摩擦
    ///
    /// 直接更新流量数组
    pub fn apply_semi_implicit_batch(
        &self,
        h: &[f64],
        hu: &mut [f64],
        hv: &mut [f64],
        manning_n: &[f64],
        dt: f64, // ALLOW_F64: 与 PhysicsMesh 配合/物理参数
    ) {
        let n = h.len();
        assert_eq!(n, hu.len());
        assert_eq!(n, hv.len());
        assert!(manning_n.len() == n || manning_n.len() == 1);

        let use_uniform_n = manning_n.len() == 1;
        let uniform_n = if use_uniform_n { manning_n[0] } else { 0.0 };

        if self.config.parallel && n >= self.config.parallel_threshold {
            self.apply_semi_implicit_parallel(h, hu, hv, manning_n, use_uniform_n, uniform_n, dt);
        } else {
            self.apply_semi_implicit_serial(h, hu, hv, manning_n, use_uniform_n, uniform_n, dt);
        }
    }

    /// 串行半隐式更新
    fn apply_semi_implicit_serial(
        &self,
        h: &[f64],
        hu: &mut [f64],
        hv: &mut [f64],
        manning_n: &[f64],
        use_uniform_n: bool,
        uniform_n: f64, // ALLOW_F64: 与 PhysicsMesh 配合/物理参数
        dt: f64, // ALLOW_F64: 与 PhysicsMesh 配合/物理参数
    ) {
        for i in 0..h.len() {
            let n_val = if use_uniform_n { uniform_n } else { manning_n[i] };
            let (new_hu, new_hv) = self.apply_semi_implicit(h[i], hu[i], hv[i], n_val, dt);
            hu[i] = new_hu;
            hv[i] = new_hv;
        }
    }

    /// 并行半隐式更新
    fn apply_semi_implicit_parallel(
        &self,
        h: &[f64],
        hu: &mut [f64],
        hv: &mut [f64],
        manning_n: &[f64],
        use_uniform_n: bool,
        uniform_n: f64, // ALLOW_F64: 与 PhysicsMesh 配合/物理参数
        dt: f64, // ALLOW_F64: 与 PhysicsMesh 配合/物理参数
    ) {
        hu.par_iter_mut()
            .zip(hv.par_iter_mut())
            .zip(h.par_iter())
            .enumerate()
            .for_each(|(i, ((hu_i, hv_i), &h_i))| {
                let n_val = if use_uniform_n { uniform_n } else { manning_n[i] };
                let (new_hu, new_hv) = self.apply_semi_implicit(h_i, *hu_i, *hv_i, n_val, dt);
                *hu_i = new_hu;
                *hv_i = new_hv;
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 9.81;
    const EPS: f64 = 1e-10;

    #[test]
    fn test_friction_coefficient_dry() {
        let friction = ManningFriction::new(G);
        let cf = friction.compute_friction_coefficient(1e-5, 0.0, 0.0, 0.03);
        assert!(cf.abs() < EPS, "干单元摩擦系数应为零");
    }

    #[test]
    fn test_friction_coefficient_wet() {
        let friction = ManningFriction::new(G);
        let h = 1.0;
        let hu = 1.0; // u = 1 m/s
        let hv = 0.0;
        let n = 0.03;

        let cf = friction.compute_friction_coefficient(h, hu, hv, n);
        
        // C_f = g * n^2 * |V| / h^(4/3)
        // = 9.81 * 0.03^2 * 1.0 / 1.0
        // = 9.81 * 0.0009
        // ≈ 0.008829
        let expected = G * n * n * 1.0 / 1.0_f64.powf(4.0 / 3.0);
        assert!((cf - expected).abs() < EPS, "摩擦系数计算错误: {} vs {}", cf, expected);
    }

    #[test]
    fn test_friction_coefficient_shallow() {
        let friction = ManningFriction::new(G);
        let h = 0.1;
        let hu = 0.1; // u = 1 m/s
        let hv = 0.0;
        let n = 0.03;

        let cf = friction.compute_friction_coefficient(h, hu, hv, n);
        
        // 浅水时摩擦系数更大
        let deep_cf = friction.compute_friction_coefficient(1.0, 1.0, 0.0, n);
        assert!(cf > deep_cf, "浅水摩擦系数应大于深水");
    }

    #[test]
    fn test_explicit_source() {
        let friction = ManningFriction::new(G);
        let h = 1.0;
        let hu = 1.0;
        let hv = 0.5;
        let n = 0.03;

        let (s_hu, s_hv) = friction.compute_explicit_source(h, hu, hv, n);
        
        // 源项应为负（减速）
        assert!(s_hu < 0.0, "x方向摩擦源项应为负");
        assert!(s_hv < 0.0, "y方向摩擦源项应为负");
        
        // 源项比例应与流量比例相同
        let ratio = s_hv / s_hu;
        let q_ratio = hv / hu;
        assert!((ratio - q_ratio).abs() < EPS, "源项比例应与流量比例相同");
    }

    #[test]
    fn test_semi_implicit() {
        let friction = ManningFriction::new(G);
        let h = 1.0;
        let hu = 1.0;
        let hv = 0.5;
        let n = 0.03;
        let dt = 0.1;

        let (new_hu, new_hv) = friction.apply_semi_implicit(h, hu, hv, n, dt);
        
        // 更新后流量应减小
        assert!(new_hu < hu, "半隐式更新后 hu 应减小");
        assert!(new_hv < hv, "半隐式更新后 hv 应减小");
        
        // 流量应保持非负
        assert!(new_hu >= 0.0, "流量不应变负");
        assert!(new_hv >= 0.0, "流量不应变负");
    }

    #[test]
    fn test_semi_implicit_large_dt() {
        let friction = ManningFriction::new(G);
        let h = 0.1; // 浅水，大摩擦
        let hu = 0.1;
        let hv = 0.0;
        let n = 0.1; // 高糙率
        let dt = 10.0; // 大时间步

        let (new_hu, new_hv) = friction.apply_semi_implicit(h, hu, hv, n, dt);
        
        // 即使时间步很大，流量也不会变负（半隐式的稳定性）
        assert!(new_hu >= 0.0, "半隐式应保持稳定");
        assert!(new_hv.abs() < EPS, "无y方向流量");
        
        // 流量应显著减小
        assert!(new_hu < 0.1 * hu, "大时间步应显著减速");
    }

    #[test]
    fn test_batch_compute() {
        let friction = ManningFriction::with_config(G, FrictionConfig {
            parallel: false,
            ..Default::default()
        });

        let n = 10;
        let h = vec![1.0; n];
        let hu = vec![1.0; n];
        let hv = vec![0.5; n];
        let manning = vec![0.03; n];
        let mut source_hu = vec![0.0; n];
        let mut source_hv = vec![0.0; n];

        friction.compute_sources_batch(&h, &hu, &hv, &manning, &mut source_hu, &mut source_hv);

        // 所有源项应相同
        for i in 1..n {
            assert!((source_hu[i] - source_hu[0]).abs() < EPS);
            assert!((source_hv[i] - source_hv[0]).abs() < EPS);
        }
    }

    #[test]
    fn test_batch_uniform_manning() {
        let friction = ManningFriction::with_config(G, FrictionConfig {
            parallel: false,
            ..Default::default()
        });

        let n = 10;
        let h = vec![1.0; n];
        let hu = vec![1.0; n];
        let hv = vec![0.5; n];
        let manning_uniform = vec![0.03]; // 只有一个值
        let manning_array = vec![0.03; n];
        
        let mut source_hu_1 = vec![0.0; n];
        let mut source_hv_1 = vec![0.0; n];
        let mut source_hu_2 = vec![0.0; n];
        let mut source_hv_2 = vec![0.0; n];

        friction.compute_sources_batch(&h, &hu, &hv, &manning_uniform, &mut source_hu_1, &mut source_hv_1);
        friction.compute_sources_batch(&h, &hu, &hv, &manning_array, &mut source_hu_2, &mut source_hv_2);

        // 两种方式结果应相同
        for i in 0..n {
            assert!((source_hu_1[i] - source_hu_2[i]).abs() < EPS);
            assert!((source_hv_1[i] - source_hv_2[i]).abs() < EPS);
        }
    }

    #[test]
    fn test_batch_semi_implicit() {
        let friction = ManningFriction::with_config(G, FrictionConfig {
            parallel: false,
            ..Default::default()
        });

        let n = 10;
        let h = vec![1.0; n];
        let mut hu = vec![1.0; n];
        let mut hv = vec![0.5; n];
        let manning = vec![0.03; n];
        let dt = 0.1;

        let old_hu = hu.clone();
        let old_hv = hv.clone();

        friction.apply_semi_implicit_batch(&h, &mut hu, &mut hv, &manning, dt);

        // 所有流量应减小
        for i in 0..n {
            assert!(hu[i] < old_hu[i], "hu[{}] 应减小", i);
            assert!(hv[i] < old_hv[i], "hv[{}] 应减小", i);
        }
    }
}
