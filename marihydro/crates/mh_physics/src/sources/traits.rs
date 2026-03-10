// crates/mh_physics/src/sources/traits.rs

//! 婧愰」 Trait 瀹氫箟
//!
//! 瀹氫箟婧愰」鐨勬牳蹇冩帴鍙ｅ拰鏁版嵁缁撴瀯銆?

use crate::core::{Backend, Scalar};
use crate::state::{ShallowWaterState, ShallowWaterStateGeneric};
use crate::types::NumericalParams;

/// 婧愰」璐＄尞
///
/// 琛ㄧず鍗曚釜鍗曞厓鐨勬簮椤硅础鐚紝鍖呮嫭璐ㄩ噺鍜屽姩閲忓彉鍖栫巼銆?
#[derive(Debug, Clone, Copy, Default)]
pub struct SourceContribution {
    /// 璐ㄩ噺婧?[m/s]
    pub s_h: f64,
    /// x鍔ㄩ噺婧?[m虏/s虏]
    pub s_hu: f64,
    /// y鍔ㄩ噺婧?[m虏/s虏]
    pub s_hv: f64,
}

impl SourceContribution {
    /// 闆惰础鐚父閲?
    pub const ZERO: Self = Self {
        s_h: 0.0,
        s_hu: 0.0,
        s_hv: 0.0,
    };

    /// 鍒涘缓鏂扮殑婧愰」璐＄尞
    #[inline]
    pub fn new(s_h: f64, s_hu: f64, s_hv: f64) -> Self {
        Self { s_h, s_hu, s_hv }
    }

    /// 鍒涘缓浠呭姩閲忚础鐚?
    #[inline]
    pub fn momentum(s_hu: f64, s_hv: f64) -> Self {
        Self { s_h: 0.0, s_hu, s_hv }
    }

    /// 鍒涘缓浠呰川閲忚础鐚?
    #[inline]
    pub fn mass(s_h: f64) -> Self {
        Self { s_h, s_hu: 0.0, s_hv: 0.0 }
    }

    /// 鍔犳硶
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            s_h: self.s_h + other.s_h,
            s_hu: self.s_hu + other.s_hu,
            s_hv: self.s_hv + other.s_hv,
        }
    }

    /// 鍘熷湴鍔犳硶
    #[inline]
    pub fn add_assign(&mut self, other: &Self) {
        self.s_h += other.s_h;
        self.s_hu += other.s_hu;
        self.s_hv += other.s_hv;
    }

    /// 缂╂斁
    #[inline]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            s_h: self.s_h * factor,
            s_hu: self.s_hu * factor,
            s_hv: self.s_hv * factor,
        }
    }

    /// 妫€鏌ユ槸鍚︽湁鏁堬紙鎵€鏈夊垎閲忛兘鏄湁闄愭暟锛?
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.s_h.is_finite() && self.s_hu.is_finite() && self.s_hv.is_finite()
    }

    /// 閽充綅鍒板畨鍏ㄨ寖鍥?
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

/// 婧愰」璁＄畻涓婁笅鏂?
///
/// 鍖呭惈婧愰」璁＄畻鎵€闇€鐨勬椂闂村拰鍙傛暟淇℃伅銆?
#[derive(Debug, Clone)]
pub struct SourceContext<'a> {
    /// 褰撳墠妯℃嫙鏃堕棿 [s]
    pub time: f64,
    /// 鏃堕棿姝ラ暱 [s]
    pub dt: f64,
    /// 鏁板€煎弬鏁?
    pub params: &'a NumericalParams,
}

impl<'a> SourceContext<'a> {
    /// 鍒涘缓鏂扮殑婧愰」涓婁笅鏂?
    pub fn new(time: f64, dt: f64, params: &'a NumericalParams) -> Self {
        Self { time, dt, params }
    }

    /// 妫€鏌ュ崟鍏冩槸鍚﹀共鐕?
    #[inline]
    pub fn is_dry(&self, h: f64) -> bool {
        h < self.params.h_dry
    }

    /// 妫€鏌ュ崟鍏冩槸鍚︽箍娑?
    #[inline]
    pub fn is_wet(&self, h: f64) -> bool {
        h >= self.params.h_wet
    }
}

/// 婧愰」 Trait
///
/// 瀹氫箟婧愰」璁＄畻鐨勭粺涓€鎺ュ彛銆?
pub trait SourceTerm: Send + Sync {
    /// 鑾峰彇婧愰」鍚嶇О
    fn name(&self) -> &'static str;

    /// 鏄惁鍚敤
    fn is_enabled(&self) -> bool;

    /// 璁＄畻鍗曚釜鍗曞厓鐨勬簮椤硅础鐚?
    fn compute_cell(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        ctx: &SourceContext,
    ) -> SourceContribution;

    /// 鎵归噺璁＄畻鎵€鏈夊崟鍏冪殑婧愰」
    ///
    /// 榛樿瀹炵幇閫愬崟鍏冭皟鐢?`compute_cell`銆?
    /// 瀛愮被鍙互瑕嗙洊浠ユ彁渚涗紭鍖栫殑鎵归噺璁＄畻銆?
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

    /// 婧愰」鏄惁鏄惧紡锛堥渶瑕丆FL闄愬埗锛?
    fn is_explicit(&self) -> bool {
        true
    }

    /// 婧愰」鏄惁浣跨敤灞€閮ㄩ殣寮忓鐞?
    /// 
    /// 灞€閮ㄩ殣寮忔剰鍛崇潃婧愰」鍐呴儴澶勭悊鍒氭€э紙濡傛懇鎿︾殑 1/(1+dt*纬)锛夛紝
    /// 鑰岄潪闇€瑕佸叏灞€闅愬紡姹傝В鍣ㄣ€?
    fn is_locally_implicit(&self) -> bool {
        false
    }

    // ========== 鍗婇殣寮忓垎瑁傛柟娉?==========

    /// 璁＄畻棰勬祴姝ユ簮椤硅础鐚?
    ///
    /// 鐢ㄤ簬鍗婇殣寮忔柟娉曠殑棰勬祴闃舵銆傞粯璁よ繑鍥炲畬鏁存簮椤广€?
    fn compute_prediction(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        ctx: &SourceContext,
    ) -> SourceContribution {
        self.compute_cell(state, cell, ctx)
    }

    /// 璁＄畻鏍℃姝ユ簮椤硅础鐚?
    ///
    /// 鐢ㄤ簬鍗婇殣寮忔柟娉曠殑鏍℃闃舵銆傞粯璁よ繑鍥為浂璐＄尞銆?
    fn compute_correction(
        &self,
        _state: &ShallowWaterState,
        _cell: usize,
        _ctx: &SourceContext,
    ) -> SourceContribution {
        SourceContribution::ZERO
    }

    /// 鏍℃姝ユ槸鍚﹂渶瑕佹婧愰」
    fn requires_correction(&self) -> bool {
        false
    }

    /// 鑾峰彇闅愬紡鍥犲瓙
    ///
    /// 杩斿洖 0.0 琛ㄧず瀹屽叏鏄惧紡锛?.0 琛ㄧず瀹屽叏闅愬紡銆?
    /// 鐢ㄤ簬鏃堕棿姝ラ暱鎺у埗鍜岀ǔ瀹氭€у垎鏋愩€?
    fn implicit_factor(&self) -> f64 {
        if self.is_locally_implicit() {
            1.0
        } else {
            0.0
        }
    }

    /// 鑾峰彇绋冲畾鎬ч檺鍒舵椂闂存闀?
    ///
    /// 杩斿洖 None 琛ㄧず鏃犻檺鍒讹紝Some(dt) 琛ㄧず鏈€澶у厑璁告椂闂存闀裤€?
    fn stability_limit(&self, _state: &ShallowWaterState, _ctx: &SourceContext) -> Option<f64> {
        None
    }
}

/// 婧愰」杈呭姪鍑芥暟
pub struct SourceHelpers;

impl SourceHelpers {
    /// 瀹夊叏绱姞锛堝拷鐣ユ棤鏁堝€硷級
    #[inline]
    pub fn safe_accumulate(acc: &mut f64, val: f64) {
        if val.is_finite() {
            *acc += val;
        }
    }

    /// 楠岃瘉璐＄尞鍊煎苟閽充綅
    #[inline]
    pub fn validate_contribution(val: f64, max_abs: f64) -> f64 {
        if !val.is_finite() {
            return 0.0;
        }
        val.clamp(-max_abs, max_abs)
    }

    /// 鍏夋粦杩囨浮鍑芥暟 (骞叉箍杩囨浮)
    ///
    /// 杩斿洖 0.0 (瀹屽叏骞? 鍒?1.0 (瀹屽叏婀? 涔嬮棿鐨勫€?
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

    /// 璁＄畻瀹夊叏閫熷害锛堥伩鍏嶉櫎浠ラ浂锛?
    #[inline]
    pub fn safe_velocity(hu: f64, hv: f64, h: f64, h_min: f64) -> (f64, f64) {
        let h_safe = h.max(h_min);
        (hu / h_safe, hv / h_safe)
    }
}

// =============================================================================
// 娉涘瀷鐗堟湰锛堟帹鑽愪娇鐢級
// =============================================================================

/// 婧愰」鍒氭€у垎绫?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceStiffness {
    /// 鏄惧紡澶勭悊锛氭簮椤硅緝涓哄钩缂擄紝鍙互鏄惧紡绉垎
    Explicit,
    /// 灞€閮ㄩ殣寮忥細婧愰」鍙兘杈冨垰鎬э紙濡傛懇鎿︼級锛岄渶瑕佸眬閮ㄩ殣寮忓鐞?
    /// 浣跨敤 1/(1 + dt*纬) 褰㈠紡鐨勯殣寮忓洜瀛?
    LocallyImplicit,
    /// 鍏ㄩ殣寮忥細闇€瑕佸湪鍏ㄥ眬闅愬紡姹傝В鍣ㄤ腑澶勭悊
    FullyImplicit,
}

/// 娉涘瀷婧愰」璐＄尞
#[derive(Debug, Clone, Copy)]
pub struct SourceContributionGeneric<S: Scalar> {
    /// 璐ㄩ噺婧?[m/s]
    pub s_h: S,
    /// x 鏂瑰悜鍔ㄩ噺婧?[m虏/s虏]
    pub s_hu: S,
    /// y 鏂瑰悜鍔ㄩ噺婧?[m虏/s虏]
    pub s_hv: S,
}

impl<S: Scalar> Default for SourceContributionGeneric<S> {
    fn default() -> Self {
        Self {
            s_h: S::ZERO,
            s_hu: S::ZERO,
            s_hv: S::ZERO,
        }
    }
}

impl<S: Scalar> SourceContributionGeneric<S> {
    /// 闆惰础鐚?
    #[inline]
    pub fn zero() -> Self {
        Self { s_h: S::ZERO, s_hu: S::ZERO, s_hv: S::ZERO }
    }
    
    /// 鍒涘缓鏂扮殑婧愰」璐＄尞
    #[inline]
    pub fn new(s_h: S, s_hu: S, s_hv: S) -> Self {
        Self { s_h, s_hu, s_hv }
    }
    
    /// 鍒涘缓浠呭姩閲忚础鐚?
    #[inline]
    pub fn momentum(s_hu: S, s_hv: S) -> Self {
        Self { s_h: S::ZERO, s_hu, s_hv }
    }
    
    /// 鍒涘缓浠呰川閲忚础鐚?
    #[inline]
    pub fn mass(s_h: S) -> Self {
        Self { s_h, s_hu: S::ZERO, s_hv: S::ZERO }
    }
    
    /// 鍘熷湴鍔犳硶
    #[inline]
    pub fn add_assign(&mut self, other: &Self) {
        self.s_h += other.s_h;
        self.s_hu += other.s_hu;
        self.s_hv += other.s_hv;
    }
}

/// 娉涘瀷婧愰」璁＄畻涓婁笅鏂?
#[derive(Debug, Clone)]
pub struct SourceContextGeneric<S: Scalar> {
    /// 褰撳墠妯℃嫙鏃堕棿 [s]
    pub time: f64,
    /// 鏃堕棿姝ラ暱 [s]
    pub dt: S,
    /// 閲嶅姏鍔犻€熷害 [m/s虏]
    pub gravity: S,
    /// 骞插崟鍏冮槇鍊?[m]
    pub h_dry: S,
    /// 婀垮崟鍏冮槇鍊?[m]
    pub h_wet: S,
}


impl<S: Scalar> SourceContextGeneric<S> {
    /// 鍒涘缓鏂扮殑婧愰」涓婁笅鏂?
    pub fn new(time: f64, dt: S, gravity: S, h_dry: S, h_wet: S) -> Self {
        Self { time, dt, gravity, h_dry, h_wet }
    }
    
    /// 浣跨敤榛樿鐗╃悊鍙傛暟鍒涘缓
    pub fn with_defaults(time: f64, dt: S) -> Self {
        Self {
            time,
            dt,
            gravity: <S as Scalar>::from_f64(9.81),
            h_dry: <S as Scalar>::from_f64(1e-6),
            h_wet: <S as Scalar>::from_f64(1e-4),
        }
    }
    
    /// 妫€鏌ユ按娣辨槸鍚︿负骞?
    #[inline]
    pub fn is_dry(&self, h: S) -> bool { h < self.h_dry }
    
    /// 妫€鏌ユ按娣辨槸鍚︿负婀?
    #[inline]
    pub fn is_wet(&self, h: S) -> bool { h >= self.h_wet }
}

/// 娉涘瀷婧愰」 Trait
pub trait SourceTermGeneric<B: Backend>: Send + Sync {
    /// 鑾峰彇婧愰」鍚嶇О
    fn name(&self) -> &'static str;
    
    /// 鑾峰彇婧愰」鍒氭€у垎绫?
    fn stiffness(&self) -> SourceStiffness;
    
    /// 婧愰」鏄惁鍚敤
    fn is_enabled(&self) -> bool { true }
    
    /// 璁＄畻鍗曚釜鍗曞厓鐨勬簮椤硅础鐚?
    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterStateGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar>;
    
    /// 鎵归噺璁＄畻鎵€鏈夊崟鍏冪殑婧愰」
    fn compute_batch(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        contributions: &mut [SourceContributionGeneric<B::Scalar>],
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !self.is_enabled() { return; }
        for cell in 0..state.n_cells() {
            contributions[cell] = self.compute_cell(cell, state, ctx);
        }
    }
    
    /// 绱姞婧愰」鍒板彸绔」缂撳啿鍖?
    fn accumulate(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    );
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
        // 榛樿 h_dry = 1e-6锛屾墍浠?1e-7 鏄共鐨勶紝1e-5 涓嶆槸
        assert!(ctx.is_dry(1e-7));
        assert!(!ctx.is_dry(1e-5));
        assert!(ctx.is_wet(0.1));
    }
}
