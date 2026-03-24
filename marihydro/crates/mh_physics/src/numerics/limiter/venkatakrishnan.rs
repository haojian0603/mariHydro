// crates/mh_physics/src/numerics/limiter/venkatakrishnan.rs

//! Venkatakrishnan 闄愬埗鍣?- 娉涘瀷瀹炵幇
//!
//! 鍏夋粦鐨勬搴﹂檺鍒跺櫒锛岄伩鍏?Barth-Jespersen 鐨勬搴︾獊鍙橀棶棰樸€?
//! 浣跨敤鍏夋粦鍑芥暟鏇夸唬 min 鎿嶄綔锛屾彁渚涗簩闃剁簿搴﹀苟淇濇寔鏁板€肩ǔ瀹氭€с€?
//!
//! # 绫诲瀷鍙傛暟
//! - `S: RuntimeScalar` - 鏀寔 f32/f64 绮惧害
//!
//! # K 鍙傛暟閫夋嫨
//! - 0.1-0.3: 寮洪檺鍒讹紝閫傜敤浜庢縺娉?婧冨潩
//! - 0.3-1.0: 涓瓑闄愬埗锛岄€氱敤鍦烘櫙锛堥粯璁わ級
//! - 1.0-5.0: 寮遍檺鍒讹紝閫傜敤浜庡厜婊戞祦鍔?
//!
//! # 娉ㄦ剰浜嬮」
//! 榛樿鏋勯€犲櫒浣跨敤 `mesh_scale=1.0`锛屽疄闄呬娇鐢ㄦ椂蹇呴』璋冪敤 `update_mesh_scale()`
//! 鏍规嵁鐪熷疄缃戞牸灏哄害鏇存柊锛屽惁鍒欓檺鍒舵晥鏋滃彲鑳戒笉绗﹀悎棰勬湡銆?
//!
//! # 鍙傝€冩枃鐚?
//! Venkatakrishnan, V. (1993). "On the accuracy of limiters and convergence to steady state solutions".
//! AIAA Paper 93-0880.

use mh_runtime::{Backend, RuntimeScalar};
use num_traits::Float;
use super::traits::{LimiterContext, SlopeLimiter};

/// Venkatakrishnan 闄愬埗鍣?
#[derive(Clone, Copy)]
pub struct Venkatakrishnan<B: Backend> {
    k: B::Scalar,
    eps_squared: B::Scalar,
    tol: B::Scalar,
}

impl<B: Backend> std::fmt::Debug for Venkatakrishnan<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Venkatakrishnan")
            .field("k", &self.k)
            .field("eps_squared", &self.eps_squared)
            .field("tol", &self.tol)
            .finish()
    }
}

impl<B: Backend> Venkatakrishnan<B> {
    #[inline]
    fn preset_k(value: f64, context: &'static str) -> B::Scalar {
        B::Scalar::from_config(value).unwrap_or_else(|| {
            panic!("failed to convert preset limiter parameter for {context}: {value}")
        })
    }
    /// 鍒涘缓鏂扮殑闄愬埗鍣?
    ///
    /// # 鍙傛暟
    /// - `k`: K 鍙傛暟锛屾帶鍒堕檺鍒跺己搴?
    /// - `mesh_scale`: 缃戞牸鐗瑰緛灏哄害
    #[inline]
    pub fn new(k: B::Scalar, mesh_scale: B::Scalar) -> Self {
        let scale = if mesh_scale.is_finite() && mesh_scale > B::Scalar::ZERO {
            mesh_scale
        } else {
            B::Scalar::ONE
        };
        let kh = k * scale;
        let eps_squared = kh * kh * kh;
        
        Self {
            k,
            eps_squared,
            tol: B::Scalar::MIN_POSITIVE,
        }
    }

    /// 鍒涘缓鍏锋湁鑷畾涔夊宸殑闄愬埗鍣?
    #[inline]
    pub fn with_tolerance(k: B::Scalar, mesh_scale: B::Scalar, tol: B::Scalar) -> Self {
        let scale = if mesh_scale.is_finite() && mesh_scale > B::Scalar::ZERO {
            mesh_scale
        } else {
            B::Scalar::ONE
        };
        let kh = k * scale;
        let eps_squared = kh * kh * kh;
        
        Self {
            k,
            eps_squared,
            tol,
        }
    }

        /// 棰勮锛氭縺娉?寮洪棿鏂?
        pub fn for_shock_capturing(mesh_scale: B::Scalar) -> Self {
            let k = Self::preset_k(0.1, "venkatakrishnan.shock_capturing");
            Self::new(k, mesh_scale)
        }

        /// 棰勮锛氬共婀夸氦鐣?
        pub fn for_wetting_drying(mesh_scale: B::Scalar) -> Self {
            let k = Self::preset_k(0.3, "venkatakrishnan.wetting_drying");
            Self::new(k, mesh_scale)
        }

        /// 棰勮锛氬厜婊戞祦鍔?
        pub fn for_smooth_flow(mesh_scale: B::Scalar) -> Self {
            let k = Self::preset_k(2.0, "venkatakrishnan.smooth_flow");
            Self::new(k, mesh_scale)
        }

        /// 棰勮锛氭渶灏忛檺鍒?
        pub fn minimal_limiting(mesh_scale: B::Scalar) -> Self {
            let k = Self::preset_k(5.0, "venkatakrishnan.minimal_limiting");
            Self::new(k, mesh_scale)
        }

    /// 鑾峰彇 K 鍙傛暟
    #[inline]
    pub fn k(&self) -> B::Scalar {
        self.k
    }

    /// 鑾峰彇 蔚虏 鍊?
    #[inline]
    pub fn eps_squared(&self) -> B::Scalar {
        self.eps_squared
    }

    /// 鏇存柊缃戞牸灏哄害
    #[inline]
    pub fn update_mesh_scale(&mut self, mesh_scale: B::Scalar) {
        let scale = if mesh_scale.is_finite() && mesh_scale > B::Scalar::ZERO {
            mesh_scale
        } else {
            B::Scalar::ONE
        };
        let kh = self.k * scale;
        self.eps_squared = kh * kh * kh;
    }

    /// 璁＄畻鍏夋粦闄愬埗鍑芥暟
    #[inline]
    fn phi(&self, x: B::Scalar, y: B::Scalar) -> B::Scalar {
        let x2 = x * x;
        let y2 = y * y;
        let eps2 = self.eps_squared;
        
        let numerator = (y2 + eps2) * x + B::Scalar::TWO * x2 * y;
        let denominator = y2 + B::Scalar::TWO * x2 + x * y + eps2;
        
        if denominator.abs() < self.tol {
            B::Scalar::ONE
        } else {
            numerator / denominator
        }
    }
}

impl<B: Backend> SlopeLimiter<B> for Venkatakrishnan<B> {
    #[inline]
    fn compute_limiter(&self, ctx: &LimiterContext<B>) -> B::Scalar {
        if ctx.is_gradient_zero(self.tol) {
            return B::Scalar::ONE;
        }
        
        let delta = ctx.gradient;
        
        if delta > B::Scalar::ZERO {
            let delta_max = ctx.delta_max();
            if delta_max < self.tol {
                B::Scalar::ZERO
            } else {
                self.phi(delta, delta_max).min(B::Scalar::ONE)
            }
        } else {
            let delta_min = ctx.delta_min();
            if delta_min > -self.tol {
                B::Scalar::ZERO
            } else {
                self.phi(-delta, -delta_min).min(B::Scalar::ONE)
            }
        }
    }
    
    #[inline]
    fn name(&self) -> &'static str {
        "Venkatakrishnan"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;
    
    type BackendF64 = CpuBackend<f64>;
    type BackendF32 = CpuBackend<f32>;

    #[test]
    fn test_creation_f64() {
        let limiter = Venkatakrishnan::<BackendF64>::new(5.0, 0.1);
        assert_eq!(limiter.k(), 5.0);
        assert!((limiter.eps_squared() - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_creation_f32() {
        let limiter = Venkatakrishnan::<BackendF32>::new(5.0f32, 0.1f32);
        assert_eq!(limiter.k(), 5.0f32);
        assert!((limiter.eps_squared() - 0.125f32).abs() < 1e-6f32);
    }

    #[test]
    fn test_with_tolerance() {
        let limiter = Venkatakrishnan::<BackendF64>::with_tolerance(5.0, 0.1, 1e-8);
        let ctx = LimiterContext::<BackendF64>::new(1.0, 0.0, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }

    #[test]
    fn test_presets() {
        let shock = Venkatakrishnan::<BackendF64>::for_shock_capturing(1.0);
        assert_eq!(shock.k(), 0.1);

        let wet_dry = Venkatakrishnan::<BackendF64>::for_wetting_drying(1.0);
        assert_eq!(wet_dry.k(), 0.3);

        let smooth = Venkatakrishnan::<BackendF64>::for_smooth_flow(1.0);
        assert_eq!(smooth.k(), 2.0);

        let minimal = Venkatakrishnan::<BackendF64>::minimal_limiting(1.0);
        assert_eq!(minimal.k(), 5.0);
    }

    #[test]
    fn test_zero_gradient() {
        let limiter = Venkatakrishnan::<BackendF64>::new(5.0, 0.1);
        let ctx = LimiterContext::<BackendF64>::new(1.0, 0.0, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }

    #[test]
    fn test_small_gradient_f64() {
        let limiter = Venkatakrishnan::<BackendF64>::new(5.0, 0.1);
        let ctx = LimiterContext::<BackendF64>::new(1.0, 0.1, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!((0.0..=1.0).contains(&alpha));
    }

    #[test]
    fn test_small_gradient_f32() {
        let limiter = Venkatakrishnan::<BackendF32>::new(5.0f32, 0.1f32);
        let ctx = LimiterContext::<BackendF32>::new(
            1.0f32, 0.1f32, 0.5f32, 1.5f32, 0.1f32
        );
        let alpha = limiter.compute_limiter(&ctx);
        assert!((0.0f32..=1.0f32).contains(&alpha));
    }

    #[test]
    fn test_k_parameter_sensitivity() {
        let mesh_scale = 0.1;
        let limiter_k1 = Venkatakrishnan::<BackendF64>::new(1.0, mesh_scale);
        let limiter_k5 = Venkatakrishnan::<BackendF64>::new(5.0, mesh_scale);
        let limiter_k10 = Venkatakrishnan::<BackendF64>::new(10.0, mesh_scale);

        let ctx = LimiterContext::<BackendF64>::new(1.0, 0.4, 0.5, 1.5, 0.1);
        let alpha_k1 = limiter_k1.compute_limiter(&ctx);
        let alpha_k5 = limiter_k5.compute_limiter(&ctx);
        let alpha_k10 = limiter_k10.compute_limiter(&ctx);

        assert!(alpha_k1 <= alpha_k5);
        assert!(alpha_k5 <= alpha_k10);
        assert!(limiter_k1.eps_squared() < limiter_k5.eps_squared());
        assert!(limiter_k5.eps_squared() < limiter_k10.eps_squared());
    }

    #[test]
    fn test_large_gradient() {
        let limiter = Venkatakrishnan::<BackendF64>::new(1.0, 0.01);
        let ctx = LimiterContext::<BackendF64>::new(1.0, 0.8, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!(alpha < 1.0);
        assert!(alpha > 0.0);
    }

    #[test]
    fn test_update_mesh_scale() {
        let mut limiter = Venkatakrishnan::<BackendF64>::new(5.0, 0.1);
        assert!((limiter.eps_squared() - 0.125).abs() < 1e-10);

        limiter.update_mesh_scale(0.2);
        assert!((limiter.eps_squared() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_smoothness() {
        let limiter = Venkatakrishnan::<BackendF64>::new(5.0, 0.1);
        let gradients: Vec<f64> = (1..=100).map(|i| i as f64 * 0.01).collect();
        let alphas: Vec<f64> = gradients
            .iter()
            .map(|&g| {
                let ctx = LimiterContext::<BackendF64>::new(1.0, g, 0.5, 1.5, 0.1);
                limiter.compute_limiter(&ctx)
            })
            .collect();

        for window in alphas.windows(2) {
            let diff = (window[1] - window[0]).abs();
            assert!(diff < 0.2, "Limiter not smooth: diff = {}", diff);
        }
    }

    #[test]
    fn test_symmetry() {
        let limiter = Venkatakrishnan::<BackendF64>::new(5.0, 0.1);
        let ctx_pos = LimiterContext::<BackendF64>::new(1.0, 0.3, 0.5, 1.5, 0.1);
        let ctx_neg = LimiterContext::<BackendF64>::new(1.0, -0.3, 0.5, 1.5, 0.1);
        assert!((limiter.compute_limiter(&ctx_pos) - limiter.compute_limiter(&ctx_neg)).abs() < 1e-10);
    }

    #[test]
    fn test_at_maximum() {
        let limiter = Venkatakrishnan::<BackendF64>::new(1.0, 0.01);
        let ctx = LimiterContext::<BackendF64>::new(1.5, 0.3, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!(alpha < 0.1);
    }

    #[test]
    fn test_limiter_bounded() {
        let limiter = Venkatakrishnan::<BackendF64>::new(3.0, 0.1);
        let test_cases = vec![
            (1.0, 0.5, 0.0, 2.0),
            (1.0, -0.5, 0.0, 2.0),
            (1.0, 0.01, 0.5, 1.5),
            (1.0, -0.01, 0.5, 1.5),
        ];

        for (q, g, q_min, q_max) in test_cases {
            let ctx = LimiterContext::<BackendF64>::new(q, g, q_min, q_max, 0.1);
            let alpha = limiter.compute_limiter(&ctx);
            assert!((0.0..=1.0).contains(&alpha));
        }
    }
}
