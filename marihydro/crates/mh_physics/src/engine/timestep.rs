// crates/mh_physics/src/engine/timestep.rs

//! 鏃堕棿姝ラ暱鎺у埗妯″潡
//!
//! 鎻愪緵鍩轰簬 CFL 鏉′欢鐨勮嚜閫傚簲鏃堕棿姝ラ暱鎺у埗銆?
//!
//! ## CFL 鏉′欢
//!
//! 鏃堕棿姝ラ暱闇€婊¤冻 CFL 鏉′欢锛?
//!
//! $$ \Delta t \leq C \cdot \min_i \frac{\Delta x_i}{|u_i| + \sqrt{gh_i}} $$
//!
//! 鍏朵腑 $C$ 閫氬父鍙?0.4-0.8锛堝彇鍐充簬绌洪棿鏍煎紡闃舵暟锛夈€?
//!
//! ## 鐗规€?
//!
//! - 棰勮绠?dx_min锛岄伩鍏嶆瘡姝ラ噸澶嶈绠?
//! - 骞惰娉㈤€熻绠椾娇鐢ㄥ師瀛愭搷浣?
//! - 鍙€夌殑鑷€傚簲鏃堕棿姝ラ暱澧為暱
//!
//! # 杩佺Щ璇存槑
//!
//! 浠?history_src/physics/engine/timestep.rs 杩佺Щ銆?

use crate::adapter::PhysicsMesh;
use crate::state::ShallowWaterState;
use crate::types::NumericalParams;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// CFL 鏃堕棿姝ヨ绠楀櫒
///
/// 涓昏浼樺寲锛氶璁＄畻缃戞牸鏈€灏忕壒寰侀暱搴?
#[derive(Clone, Debug)]
pub struct CflCalculator {
    /// 閲嶅姏鍔犻€熷害
    g: f64,
    /// CFL 鏁?
    cfl: f64,
    /// 鏈€灏忔椂闂存闀?
    dt_min: f64,
    /// 鏈€澶ф椂闂存闀?
    dt_max: f64,
    /// 棰勮绠楃殑鏈€灏忕壒寰侀暱搴?
    cached_dx_min: Option<f64>,
    /// 鏈€灏忔尝閫熼槇鍊硷紙浣庝簬姝ゅ€艰涓洪潤姝級
    min_wave_speed: f64,
}

impl CflCalculator {
    /// 鍒涘缓璁＄畻鍣?
    pub fn new(g: f64, params: &NumericalParams) -> Self {
        Self {
            g,
            cfl: params.cfl,
            dt_min: params.dt_min,
            dt_max: params.dt_max,
            cached_dx_min: None,
            min_wave_speed: params.min_wave_speed,
        }
    }

    /// 棰勮绠楃綉鏍兼渶灏忕壒寰侀暱搴?
    ///
    /// 搴斿湪缃戞牸鍔犺浇鍚庤皟鐢ㄤ竴娆?
    pub fn precompute_dx_min(&mut self, mesh: &PhysicsMesh) {
        self.cached_dx_min = Some(self.compute_min_char_length(mesh));
    }

    /// 鑾峰彇缂撳瓨鐨?dx_min
    pub fn dx_min(&self) -> Option<f64> {
        self.cached_dx_min
    }

    /// 璁＄畻鏃堕棿姝ラ暱
    pub fn compute_dt(
        &self,
        state: &ShallowWaterState,
        mesh: &PhysicsMesh,
        params: &NumericalParams,
    ) -> f64 {
        let n_cells = mesh.n_cells();
        if n_cells == 0 {
            return self.dt_max;
        }

        // 浣跨敤棰勮绠楃殑 dx_min 鎴栫幇鍦鸿绠?
        let min_length = self
            .cached_dx_min
            .unwrap_or_else(|| self.compute_min_char_length(mesh));

        // 骞惰璁＄畻鏈€澶ф尝閫?
        let max_speed = self.compute_max_wave_speed_parallel(state, params);

        if max_speed < self.min_wave_speed {
            return self.dt_max;
        }

        let dt = self.cfl * min_length / max_speed;
        dt.clamp(self.dt_min, self.dt_max)
    }

    /// 浠庡凡鐭ユ渶澶ф尝閫熻绠楁椂闂存闀?
    ///
    /// 褰撻€氶噺璁＄畻宸插緱鍒版渶澶ф尝閫熸椂浣跨敤姝ゆ柟娉曪紝閬垮厤閲嶅璁＄畻
    pub fn compute_from_max_speed(&self, max_speed: f64) -> f64 {
        let min_length = self.cached_dx_min.unwrap_or(1.0);

        if max_speed < self.min_wave_speed {
            return self.dt_max;
        }

        let dt = self.cfl * min_length / max_speed;
        dt.clamp(self.dt_min, self.dt_max)
    }

    /// 骞惰璁＄畻鏈€澶ф尝閫燂紙浣跨敤鍘熷瓙鎿嶄綔锛?
    fn compute_max_wave_speed_parallel(
        &self,
        state: &ShallowWaterState,
        params: &NumericalParams,
    ) -> f64 {
        let n = state.h.len();
        if n == 0 {
            return 0.0;
        }

        // 浣跨敤鍘熷瓙鎿嶄綔鏀堕泦鏈€澶у€?
        let max_speed = AtomicU64::new(0u64);

        (0..n).into_par_iter().for_each(|i| {
            let h = state.h[i];
            if params.is_dry(h) {
                return;
            }

            let (u, v) = params.safe_velocity_components(state.hu[i], state.hv[i], h);
            let speed = (u * u + v * v).sqrt();
            let c = (self.g * h).sqrt();
            let wave_speed = speed + c;

            // 鍘熷瓙鏇存柊鏈€澶у€?
            let bits = wave_speed.to_bits();
            max_speed.fetch_max(bits, Ordering::Relaxed);
        });

        f64::from_bits(max_speed.load(Ordering::Relaxed))
    }

    /// 璁＄畻鏈€灏忕壒寰侀暱搴?
    fn compute_min_char_length(&self, mesh: &PhysicsMesh) -> f64 {
        let n = mesh.n_cells();
        if n == 0 {
            return f64::MAX;
        }

        // 浣跨敤鍘熷瓙鎿嶄綔鏀堕泦鏈€灏忓€?
        let min_dx = AtomicU64::new(f64::MAX.to_bits());

        (0..n).into_par_iter().for_each(|i| {
            let area = mesh.cell_area(i).unwrap_or(0.0);
            let perimeter = mesh.cell_perimeter(i).unwrap_or(0.0);

            if perimeter < 1e-14 {
                return;
            }

            // 姘村姏鐩村緞杩戜技
            let dx = 2.0 * area / perimeter;

            // 鍘熷瓙鏇存柊鏈€灏忓€?
            let bits = dx.to_bits();
            min_dx.fetch_min(bits, Ordering::Relaxed);
        });

        f64::from_bits(min_dx.load(Ordering::Relaxed))
    }
}

/// 鏃堕棿姝ラ暱鎺у埗鍣?
///
/// 鎻愪緵鑷€傚簲鏃堕棿姝ラ暱鎺у埗锛屾敮鎸侊細
/// - 棰勮绠?dx_min
/// - 鑷€傚簲澧為暱/鏀剁缉鍥犲瓙
/// - 鏃堕棿姝ラ暱鍘嗗彶杩借釜
pub struct TimeStepController {
    calculator: CflCalculator,
    /// 褰撳墠鏃堕棿姝ラ暱
    current_dt: f64,
    /// 澧為暱鍥犲瓙
    growth_factor: f64,
    /// 鏀剁缉鍥犲瓙
    shrink_factor: f64,
    /// 鏈€澶у厑璁稿闀垮洜瀛?
    max_growth_factor: f64,
    /// 杩炵画绋冲畾姝ユ暟
    stable_steps: usize,
    /// 绋冲畾澧為暱闃堝€?
    stable_growth_threshold: usize,
    /// 鏄惁鍚敤鑷€傚簲澧為暱
    adaptive_growth: bool,
}

impl TimeStepController {
    /// 鍒涘缓鎺у埗鍣?
    pub fn new(g: f64, params: &NumericalParams) -> Self {
        Self {
            calculator: CflCalculator::new(g, params),
            current_dt: params.dt_max,
            growth_factor: 1.1,
            shrink_factor: 0.5,
            max_growth_factor: 1.5,
            stable_steps: 0,
            stable_growth_threshold: 10,
            adaptive_growth: true,
        }
    }

    /// 棰勮绠楃綉鏍肩壒寰?
    pub fn precompute_mesh_characteristics(&mut self, mesh: &PhysicsMesh) {
        self.calculator.precompute_dx_min(mesh);
    }

    /// 鑾峰彇棰勮绠楃殑 dx_min
    pub fn dx_min(&self) -> Option<f64> {
        self.calculator.dx_min()
    }

    /// 鏇存柊鏃堕棿姝ラ暱
    pub fn update(
        &mut self,
        state: &ShallowWaterState,
        mesh: &PhysicsMesh,
        params: &NumericalParams,
    ) -> f64 {
        let suggested = self.calculator.compute_dt(state, mesh, params);

        // 璁＄畻澧為暱鍥犲瓙
        let growth = if self.adaptive_growth {
            self.compute_adaptive_growth()
        } else {
            self.growth_factor
        };

        let grown = self.current_dt * growth;
        let new_dt = suggested.min(grown);

        // 鏇存柊绋冲畾姝ユ暟
        if new_dt >= self.current_dt * 0.95 {
            self.stable_steps += 1;
        } else {
            self.stable_steps = 0;
        }

        self.current_dt = new_dt;
        self.current_dt
    }

    /// 浠庡凡鐭ユ渶澶ф尝閫熸洿鏂版椂闂存闀?
    pub fn update_from_max_speed(&mut self, max_speed: f64) -> f64 {
        let suggested = self.calculator.compute_from_max_speed(max_speed);

        let growth = if self.adaptive_growth {
            self.compute_adaptive_growth()
        } else {
            self.growth_factor
        };

        let grown = self.current_dt * growth;
        let new_dt = suggested.min(grown);

        if new_dt >= self.current_dt * 0.95 {
            self.stable_steps += 1;
        } else {
            self.stable_steps = 0;
        }

        self.current_dt = new_dt;
        self.current_dt
    }

    /// 璁＄畻鑷€傚簲澧為暱鍥犲瓙
    fn compute_adaptive_growth(&self) -> f64 {
        if self.stable_steps >= self.stable_growth_threshold {
            // 闀挎湡绋冲畾锛屽厑璁告洿澶у闀?
            self.growth_factor.min(self.max_growth_factor)
        } else if self.stable_steps >= self.stable_growth_threshold / 2 {
            // 涓瓑绋冲畾
            self.growth_factor
        } else {
            // 涓嶇ǔ瀹氾紝淇濆畧澧為暱
            1.0 + (self.growth_factor - 1.0) * 0.5
        }
    }

    /// 鏀剁缉鏃堕棿姝ラ暱锛堥亣鍒伴棶棰樻椂璋冪敤锛?
    pub fn shrink(&mut self) {
        self.current_dt *= self.shrink_factor;
        self.current_dt = self.current_dt.max(self.calculator.dt_min);
        self.stable_steps = 0;
    }

    /// 寮哄埗鏀剁缉锛堜弗閲嶉棶棰樻椂锛?
    pub fn force_shrink(&mut self, factor: f64) {
        self.current_dt *= factor;
        self.current_dt = self.current_dt.max(self.calculator.dt_min);
        self.stable_steps = 0;
    }

    /// 鑾峰彇褰撳墠鏃堕棿姝ラ暱
    pub fn current_dt(&self) -> f64 {
        self.current_dt
    }

    /// 璁剧疆鏃堕棿姝ラ暱锛堟墜鍔ㄨ鐩栵級
    pub fn set_dt(&mut self, dt: f64) {
        self.current_dt = dt.clamp(self.calculator.dt_min, self.calculator.dt_max);
        self.stable_steps = 0;
    }

    /// 璁剧疆澧為暱鍥犲瓙
    pub fn set_growth_factor(&mut self, factor: f64) {
        self.growth_factor = factor.max(1.0);
    }

    /// 璁剧疆鏀剁缉鍥犲瓙
    pub fn set_shrink_factor(&mut self, factor: f64) {
        self.shrink_factor = factor.clamp(0.1, 0.9);
    }

    /// 鍚敤/绂佺敤鑷€傚簲澧為暱
    pub fn set_adaptive_growth(&mut self, enabled: bool) {
        self.adaptive_growth = enabled;
    }

    /// 鑾峰彇缁熻淇℃伅
    pub fn stats(&self) -> TimeStepStats {
        TimeStepStats {
            current_dt: self.current_dt,
            dx_min: self.calculator.cached_dx_min,
            stable_steps: self.stable_steps,
            adaptive_growth_enabled: self.adaptive_growth,
        }
    }

    /// 鍗婇殣寮忔柟娉曡凯浠ｆ鏁拌嚜閫傚簲
    ///
    /// 鏍规嵁鍘嬪姏姹傝В鍣ㄨ凯浠ｆ鏁拌皟鏁存椂闂存闀裤€?
    ///
    /// # 鍙傛暟
    ///
    /// - `iterations`: 瀹為檯杩唬娆℃暟
    /// - `target_iterations`: 鐩爣杩唬娆℃暟锛堥€氬父涓烘眰瑙ｅ櫒鏈€澶ц凯浠ｇ殑 50%锛?
    pub fn adapt_from_iterations(
        &mut self,
        iterations: usize,
        target_iterations: usize,
    ) -> f64 {
        let ratio = iterations as f64 / target_iterations.max(1) as f64;

        if ratio < 0.5 {
            // 鏀舵暃澶揩锛屽彲浠ュ澶ф椂闂存闀?
            let growth = (1.0 + (1.0 - ratio * 2.0) * 0.2).min(self.max_growth_factor);
            self.current_dt *= growth;
            self.stable_steps += 1;
        } else if ratio > 1.5 {
            // 鏀舵暃澶參锛屽噺灏忔椂闂存闀?
            let shrink = (1.0 - (ratio - 1.5) * 0.3).max(0.5);
            self.current_dt *= shrink;
            self.stable_steps = 0;
        } else if ratio > 1.0 {
            // 鎺ヨ繎杈圭晫锛屼繚瀹堝闀?
            self.stable_steps = self.stable_steps.saturating_sub(1);
        }

        self.current_dt = self.current_dt.clamp(self.calculator.dt_min, self.calculator.dt_max);
        self.current_dt
    }

    /// 搴旂敤婧愰」绋冲畾鎬ч檺鍒?
    ///
    /// 灏嗘墍鏈夋簮椤圭殑绋冲畾鎬ч檺鍒跺簲鐢ㄤ簬鏃堕棿姝ラ暱銆?
    ///
    /// # 鍙傛暟
    ///
    /// - `limits`: 鍚勬簮椤硅繑鍥炵殑绋冲畾鎬ч檺鍒舵椂闂存闀?
    pub fn apply_source_limits(&mut self, limits: &[Option<f64>]) -> f64 {
        let mut min_dt = self.current_dt;

        for &limit in limits {
            if let Some(dt_limit) = limit {
                min_dt = min_dt.min(dt_limit);
            }
        }

        if min_dt < self.current_dt * 0.9 {
            self.stable_steps = 0;
        }

        self.current_dt = min_dt.clamp(self.calculator.dt_min, self.calculator.dt_max);
        self.current_dt
    }

    /// 璁＄畻绉戞皬鍔涚ǔ瀹氭€ч檺鍒?
    ///
    /// 杩斿洖 dt < 2蟺 / |f| 浠ヤ繚璇佹儻鎬ф尟鑽＄ǔ瀹?
    pub fn coriolis_stability_limit(&self, f: f64) -> Option<f64> {
        if f.abs() < 1e-14 {
            None
        } else {
            Some(std::f64::consts::PI / f.abs())
        }
    }

    /// 璁＄畻鎽╂摝绋冲畾鎬ч檺鍒?
    ///
    /// 瀵逛簬鏇煎畞鍏紡鐨勯殣寮忔懇鎿?
    pub fn friction_stability_limit(&self, max_cf: f64) -> Option<f64> {
        if max_cf < 1e-14 {
            None
        } else {
            // 鏄惧紡绋冲畾鎬ч檺鍒?
            Some(2.0 / max_cf)
        }
    }

    /// 鑾峰彇 CFL 鏁?
    pub fn cfl(&self) -> f64 {
        self.calculator.cfl
    }

    /// 璁剧疆 CFL 鏁?
    pub fn set_cfl(&mut self, cfl: f64) {
        self.calculator.cfl = cfl.clamp(0.1, 1.0);
    }
}

/// 鏃堕棿姝ラ暱缁熻
#[derive(Clone, Debug)]
pub struct TimeStepStats {
    /// 褰撳墠鏃堕棿姝ラ暱
    pub current_dt: f64,
    /// 鏈€灏忕壒寰侀暱搴?
    pub dx_min: Option<f64>,
    /// 杩炵画绋冲畾姝ユ暟
    pub stable_steps: usize,
    /// 鏄惁鍚敤鑷€傚簲澧為暱
    pub adaptive_growth_enabled: bool,
}

/// 鏃堕棿姝ラ暱鎺у埗鍣ㄦ瀯寤哄櫒
pub struct TimeStepControllerBuilder {
    g: f64,
    cfl: f64,
    dt_min: f64,
    dt_max: f64,
    growth_factor: f64,
    shrink_factor: f64,
    adaptive_growth: bool,
}

impl TimeStepControllerBuilder {
    /// 鍒涘缓鏋勫缓鍣?
    pub fn new(g: f64) -> Self {
        Self {
            g,
            cfl: 0.5,
            dt_min: 1e-6,
            dt_max: 1.0,
            growth_factor: 1.1,
            shrink_factor: 0.5,
            adaptive_growth: true,
        }
    }

    /// 璁剧疆 CFL 鏁?
    pub fn with_cfl(mut self, cfl: f64) -> Self {
        self.cfl = cfl;
        self
    }

    /// 璁剧疆鏃堕棿姝ラ檺鍒?
    pub fn with_dt_limits(mut self, dt_min: f64, dt_max: f64) -> Self {
        self.dt_min = dt_min;
        self.dt_max = dt_max;
        self
    }

    /// 璁剧疆澧為暱鍥犲瓙
    pub fn with_growth_factor(mut self, factor: f64) -> Self {
        self.growth_factor = factor;
        self
    }

    /// 璁剧疆鏀剁缉鍥犲瓙
    pub fn with_shrink_factor(mut self, factor: f64) -> Self {
        self.shrink_factor = factor;
        self
    }

    /// 璁剧疆鑷€傚簲澧為暱
    pub fn with_adaptive_growth(mut self, enabled: bool) -> Self {
        self.adaptive_growth = enabled;
        self
    }

    /// 鏋勫缓鎺у埗鍣?
    pub fn build(self) -> TimeStepController {
        let params = NumericalParams::builder()
            .cfl(self.cfl)
            .dt_min(self.dt_min)
            .dt_max(self.dt_max)
            .build()
            .unwrap_or_default();

        let mut controller = TimeStepController::new(self.g, &params);
        controller.growth_factor = self.growth_factor;
        controller.shrink_factor = self.shrink_factor;
        controller.adaptive_growth = self.adaptive_growth;

        controller
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfl_calculator_creation() {
        let params = NumericalParams::default();
        let calc = CflCalculator::new(9.81, &params);
        assert!(calc.cached_dx_min.is_none());
        assert!((calc.g - 9.81).abs() < 1e-10);
    }

    #[test]
    fn test_cfl_calculator_from_max_speed() {
        let params = NumericalParams::default();
        let mut calc = CflCalculator::new(9.81, &params);
        calc.cached_dx_min = Some(1.0);

        let dt = calc.compute_from_max_speed(10.0);
        // dt = cfl * dx_min / max_speed = 0.5 * 1.0 / 10.0 = 0.05
        assert!((dt - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_cfl_calculator_static_water() {
        let params = NumericalParams::default();
        let calc = CflCalculator::new(9.81, &params);

        // 闈欐按鏃讹紝max_speed < min_wave_speed锛岃繑鍥瀌t_max
        let dt = calc.compute_from_max_speed(1e-10);
        assert!((dt - params.dt_max).abs() < 1e-10);
    }

    #[test]
    fn test_controller_creation() {
        let params = NumericalParams::default();
        let controller = TimeStepController::new(9.81, &params);
        assert!(controller.adaptive_growth);
        assert_eq!(controller.stable_steps, 0);
    }

    #[test]
    fn test_controller_adaptive_growth() {
        let params = NumericalParams::default();
        let mut controller = TimeStepController::new(9.81, &params);

        // 妯℃嫙绋冲畾姝?
        for _ in 0..15 {
            controller.stable_steps += 1;
        }

        let growth = controller.compute_adaptive_growth();
        assert!(growth >= controller.growth_factor);
    }

    #[test]
    fn test_controller_shrink() {
        let params = NumericalParams::default();
        let mut controller = TimeStepController::new(9.81, &params);
        controller.current_dt = 0.1;

        controller.shrink();
        assert!(controller.current_dt < 0.1);
        assert_eq!(controller.stable_steps, 0);
    }

    #[test]
    fn test_controller_force_shrink() {
        let params = NumericalParams::default();
        let mut controller = TimeStepController::new(9.81, &params);
        controller.current_dt = 0.1;

        controller.force_shrink(0.1);
        assert!((controller.current_dt - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_builder() {
        let controller = TimeStepControllerBuilder::new(9.81)
            .with_cfl(0.3)
            .with_dt_limits(1e-8, 0.5)
            .with_adaptive_growth(false)
            .build();

        assert!(!controller.adaptive_growth);
    }

    #[test]
    fn test_stats() {
        let params = NumericalParams::default();
        let mut controller = TimeStepController::new(9.81, &params);
        controller.stable_steps = 5;

        let stats = controller.stats();
        assert_eq!(stats.stable_steps, 5);
        assert!(stats.adaptive_growth_enabled);
    }

    #[test]
    fn test_set_dt() {
        let params = NumericalParams::default();
        let mut controller = TimeStepController::new(9.81, &params);

        controller.set_dt(0.5);
        assert!((controller.current_dt() - 0.5).abs() < 1e-10);
        assert_eq!(controller.stable_steps, 0);
    }
}
