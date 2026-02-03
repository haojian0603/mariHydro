// crates/mh_physics/src/engine/friction.rs
//! Manning摩擦源项计算模块
//!
//! 实现曼宁公式的底部摩擦源项，支持显式和半隐式时间离散、并行计算以及Backend泛型。

use rayon::prelude::*;
use mh_runtime::{Backend, RuntimeScalar};
use num_traits::{Zero, Float};

/// 摩擦计算配置（Layer 4，保持f64）
#[derive(Debug, Clone)]
pub struct FrictionConfig {
    pub semi_implicit: bool,
    pub h_min: f64,
    pub default_manning_n: f64,
    pub parallel: bool,
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

/// Manning 摩擦计算器（Backend 泛型）
/// 
/// 实现曼宁公式的底部摩擦源项计算，支持显式和半隐式离散。
/// 通过 Backend 泛型实现 f32/f64 精度切换和 GPU 加速。
/// 
/// # 类型参数
/// 
/// - `B`: 计算后端类型，必须实现 `Backend` trait
/// 
/// # 数学模型
/// 
/// Manning 摩擦公式:
/// $$\tau_b = \rho g n^2 \frac{|\mathbf{u}|}{h^{4/3}} \mathbf{u}$$
/// 
/// 其中:
/// - $n$: Manning 糙率系数
/// - $h$: 水深
/// - $\mathbf{u}$: 流速向量
#[derive(Debug, Clone)]
pub struct ManningFriction<B: Backend>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    /// 计算后端实例
    backend: B,
    /// 摩擦配置
    config: FrictionConfig,
    /// 重力加速度
    g: B::Scalar,
}

impl<B: Backend> ManningFriction<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
    B::Scalar: RuntimeScalar,
{
    /// 使用后端实例创建摩擦计算器
    /// 
    /// # 参数
    /// 
    /// - `backend`: 计算后端实例
    /// - `g`: 重力加速度 [m/s²]
    /// 
    /// # 返回
    /// 
    /// 返回初始化完成的摩擦计算器实例
    pub fn new(backend: B, g: f64) -> Self {
        Self {
            g: backend.scalar_from_f64(g),
            config: FrictionConfig::default(),
            backend,
        }
    }

    /// 使用自定义配置创建摩擦计算器
    /// 
    /// # 参数
    /// 
    /// - `backend`: 计算后端实例
    /// - `g`: 重力加速度 [m/s²]
    /// - `config`: 摩擦计算配置
    pub fn with_config(backend: B, g: f64, config: FrictionConfig) -> Self {
        Self {
            g: backend.scalar_from_f64(g),
            config,
            backend,
        }
    }

    /// 设置配置
    pub fn set_config(&mut self, config: FrictionConfig) {
        self.config = config;
    }

    /// 获取配置引用
    pub fn config(&self) -> &FrictionConfig {
        &self.config
    }
    
    /// 获取后端引用
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// 计算摩擦系数
    /// 
    /// # 参数
    /// 
    /// - `h`: 水深 [m]
    /// - `hu`: x 方向动量 [m²/s]
    /// - `hv`: y 方向动量 [m²/s]
    /// - `manning_n`: Manning 糙率系数
    /// 
    /// # 返回
    /// 
    /// 摩擦系数 C_f
    #[inline]
    pub fn compute_friction_coefficient(
        &self,
        h: B::Scalar,
        hu: B::Scalar,
        hv: B::Scalar,
        manning_n: B::Scalar,
    ) -> B::Scalar {
        let h_min = self.backend.scalar_from_f64(self.config.h_min);
        if h < h_min {
            return B::Scalar::ZERO;
        }

        let h_safe = h.max(h_min);
        let u = hu / h_safe;
        let v = hv / h_safe;
        let speed = (u * u + v * v).sqrt();

        if speed.is_zero() {
            return B::Scalar::ZERO;
        }

        let four_thirds = self.backend.scalar_from_f64(4.0 / 3.0);
        let h_pow = h_safe.powf(four_thirds);
        let eps = self.backend.scalar_from_f64(1e-12);
        if h_pow < eps {
            return B::Scalar::ZERO;
        }

        self.g * manning_n * manning_n * speed / h_pow
    }

    /// 计算显式摩擦源项
    /// 
    /// # 参数
    /// 
    /// - `h`: 水深 [m]
    /// - `hu`: x 方向动量 [m²/s]
    /// - `hv`: y 方向动量 [m²/s]
    /// - `manning_n`: Manning 糙率系数
    /// 
    /// # 返回
    /// 
    /// (S_hu, S_hv) 源项元组
    #[inline]
    pub fn compute_explicit_source(
        &self,
        h: B::Scalar,
        hu: B::Scalar,
        hv: B::Scalar,
        manning_n: B::Scalar,
    ) -> (B::Scalar, B::Scalar) {
        let cf = self.compute_friction_coefficient(h, hu, hv, manning_n);
        (-cf * hu, -cf * hv)
    }

    /// 应用半隐式摩擦离散
    /// 
    /// # 参数
    /// 
    /// - `h`: 水深 [m]
    /// - `hu`: x 方向动量 [m²/s]
    /// - `hv`: y 方向动量 [m²/s]
    /// - `manning_n`: Manning 糙率系数
    /// - `dt`: 时间步长 [s]
    /// 
    /// # 返回
    /// 
    /// 更新后的 (hu, hv) 元组
    #[inline]
    pub fn apply_semi_implicit(
        &self,
        h: B::Scalar,
        hu: B::Scalar,
        hv: B::Scalar,
        manning_n: B::Scalar,
        dt: B::Scalar,
    ) -> (B::Scalar, B::Scalar) {
        let cf = self.compute_friction_coefficient(h, hu, hv, manning_n);
        let factor = B::Scalar::ONE / (B::Scalar::ONE + dt * cf);
        (hu * factor, hv * factor)
    }

    pub fn compute_sources_batch(
        &self,
        h: &[B::Scalar],
        hu: &[B::Scalar],
        hv: &[B::Scalar],
        manning_n: &[B::Scalar],
        source_hu: &mut [B::Scalar],
        source_hv: &mut [B::Scalar],
    ) {
        let n = h.len();
        assert_eq!(n, hu.len());
        assert_eq!(n, hv.len());
        assert!(manning_n.len() == n || manning_n.len() == 1);
        assert_eq!(n, source_hu.len());
        assert_eq!(n, source_hv.len());

        let use_uniform_n = manning_n.len() == 1;
        let uniform_n = if use_uniform_n { manning_n[0] } else { B::Scalar::ZERO };

        if self.config.parallel && n >= self.config.parallel_threshold {
            self.compute_sources_parallel(h, hu, hv, manning_n, use_uniform_n, uniform_n, source_hu, source_hv);
        } else {
            self.compute_sources_serial(h, hu, hv, manning_n, use_uniform_n, uniform_n, source_hu, source_hv);
        }
    }

    fn compute_sources_serial(
        &self,
        h: &[B::Scalar],
        hu: &[B::Scalar],
        hv: &[B::Scalar],
        manning_n: &[B::Scalar],
        use_uniform_n: bool,
        uniform_n: B::Scalar,
        source_hu: &mut [B::Scalar],
        source_hv: &mut [B::Scalar],
    ) {
        for i in 0..h.len() {
            let n_val = if use_uniform_n { uniform_n } else { manning_n[i] };
            let (s_hu, s_hv) = self.compute_explicit_source(h[i], hu[i], hv[i], n_val);
            source_hu[i] = source_hu[i] + s_hu;
            source_hv[i] = source_hv[i] + s_hv;
        }
    }

    fn compute_sources_parallel(
        &self,
        h: &[B::Scalar],
        hu: &[B::Scalar],
        hv: &[B::Scalar],
        manning_n: &[B::Scalar],
        use_uniform_n: bool,
        uniform_n: B::Scalar,
        source_hu: &mut [B::Scalar],
        source_hv: &mut [B::Scalar],
    ) {
        source_hu
            .par_iter_mut()
            .zip(source_hv.par_iter_mut())
            .zip(h.par_iter())
            .zip(hu.par_iter())
            .zip(hv.par_iter())
            .enumerate()
            .for_each(|(_i, ((((s_hu, s_hv), &h_i), &hu_i), &hv_i))| {
                let n_val = if use_uniform_n { uniform_n } else { manning_n[_i] };
                let (src_hu, src_hv) = self.compute_explicit_source(h_i, hu_i, hv_i, n_val);
                *s_hu = *s_hu + src_hu;
                *s_hv = *s_hv + src_hv;
            });
    }

    pub fn apply_semi_implicit_batch(
        &self,
        h: &[B::Scalar],
        hu: &mut [B::Scalar],
        hv: &mut [B::Scalar],
        manning_n: &[B::Scalar],
        dt: B::Scalar,
    ) {
        let n = h.len();
        assert_eq!(n, hu.len());
        assert_eq!(n, hv.len());
        assert!(manning_n.len() == n || manning_n.len() == 1);

        let use_uniform_n = manning_n.len() == 1;
        let uniform_n = if use_uniform_n { manning_n[0] } else { B::Scalar::ZERO };

        if self.config.parallel && n >= self.config.parallel_threshold {
            self.apply_semi_implicit_parallel(h, hu, hv, manning_n, use_uniform_n, uniform_n, dt);
        } else {
            self.apply_semi_implicit_serial(h, hu, hv, manning_n, use_uniform_n, uniform_n, dt);
        }
    }

    fn apply_semi_implicit_serial(
        &self,
        h: &[B::Scalar],
        hu: &mut [B::Scalar],
        hv: &mut [B::Scalar],
        _manning_n: &[B::Scalar],
        use_uniform_n: bool,
        uniform_n: B::Scalar,
        dt: B::Scalar,
    ) {
        for i in 0..h.len() {
            let n_val = if use_uniform_n { uniform_n } else { B::Scalar::ZERO };
            let (new_hu, new_hv) = self.apply_semi_implicit(h[i], hu[i], hv[i], n_val, dt);
            hu[i] = new_hu;
            hv[i] = new_hv;
        }
    }

    fn apply_semi_implicit_parallel(
        &self,
        h: &[B::Scalar],
        hu: &mut [B::Scalar],
        hv: &mut [B::Scalar],
        _manning_n: &[B::Scalar],
        use_uniform_n: bool,
        uniform_n: B::Scalar,
        dt: B::Scalar,
    ) {
        hu.par_iter_mut()
            .zip(hv.par_iter_mut())
            .zip(h.par_iter())
            .enumerate()
            .for_each(|(_i, ((hu_i, hv_i), &h_i))| {
                let n_val = if use_uniform_n { uniform_n } else { B::Scalar::ZERO };
                let (new_hu, new_hv) = self.apply_semi_implicit(h_i, *hu_i, *hv_i, n_val, dt);
                *hu_i = new_hu;
                *hv_i = new_hv;
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    const G: f64 = 9.81;

    #[test]
    fn test_friction_coefficient_dry() {
        let backend = CpuBackend::<f64>::new();
        let friction = ManningFriction::new(backend, G);
        let cf = friction.compute_friction_coefficient(1e-5_f64, 0.0, 0.0, 0.03);
        assert!(cf.abs() < 1e-10);
    }

    #[test]
    fn test_friction_coefficient_wet() {
        let backend = CpuBackend::<f64>::new();
        let friction = ManningFriction::new(backend, G);
        let h = 1.0_f64;
        let hu = 1.0_f64;
        let hv = 0.0_f64;
        let n = 0.03_f64;
        let cf = friction.compute_friction_coefficient(h, hu, hv, n);
        let expected = G * n * n * 1.0;
        assert!((cf - expected).abs() < 1e-10);
    }

    #[test]
    fn test_explicit_source() {
        let backend = CpuBackend::<f64>::new();
        let friction = ManningFriction::new(backend, G);
        let h = 1.0_f64;
        let hu = 1.0_f64;
        let hv = 0.5_f64;
        let n = 0.03_f64;
        let (s_hu, s_hv) = friction.compute_explicit_source(h, hu, hv, n);
        assert!(s_hu < 0.0);
        assert!(s_hv < 0.0);
    }

    #[test]
    fn test_semi_implicit() {
        let backend = CpuBackend::<f64>::new();
        let friction = ManningFriction::new(backend, G);
        let h = 1.0_f64;
        let hu = 1.0_f64;
        let hv = 0.5_f64;
        let n = 0.03_f64;
        let dt = 0.1_f64;
        let (new_hu, new_hv) = friction.apply_semi_implicit(h, hu, hv, n, dt);
        assert!(new_hu < hu);
        assert!(new_hv < hv);
        assert!(new_hu >= 0.0);
        assert!(new_hv >= 0.0);
    }
}