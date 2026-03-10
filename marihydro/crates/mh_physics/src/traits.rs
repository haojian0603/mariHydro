// marihydro/crates/mh_physics/src/traits.rs

//! 鐘舵€佽闂娊璞℃帴鍙?
//!
//! 瀹氫箟娴呮按鏂圭▼瀹堟亽鍙橀噺鐨勮闂娊璞★紝鏀寔涓嶅悓鐘舵€佸瓨鍌ㄥ疄鐜扮殑浜掓崲銆?
//!
//! # 璁捐璇存槑
//!
//! 鏈ā鍧椾粠 `history_src/core/traits/state.rs` 杩佺Щ鑰屾潵锛屾彁渚涚粺涓€鐨勭姸鎬佽闂娊璞°€?
//! 浣跨敤 `usize` 浣滀负绱㈠紩绫诲瀷锛屼笌鍏朵粬妯″潡淇濇寔涓€鑷淬€?
//!
//! # 浣跨敤绀轰緥
//!
//! ```rust,ignore
//! use mh_physics::traits::{StateAccess, StateAccessMut};
//! use mh_physics::state::ShallowWaterState;
//!
//! fn compute_total_volume<S: StateAccess>(state: &S, areas: &[f64]) -> f64 {
//!     (0..state.n_cells())
//!         .map(|i| state.h(i) * areas[i])
//!         .sum()
//! }
//! ```

use crate::state::ConservedState;
use crate::types::{NumericalParams, SafeVelocity};

// ============================================================
// 鐘舵€佽闂?Trait
// ============================================================

/// 鐘舵€佸彧璇昏闂帴鍙?
///
/// 鎻愪緵瀵规祬姘存柟绋嬪畧鎭掑彉閲忕殑缁熶竴鍙璁块棶銆?
/// 瀹炵幇姝?trait 鐨勭被鍨嬪簲淇濊瘉绾跨▼瀹夊叏锛圫end + Sync锛夈€?
pub trait StateAccess: Send + Sync {
    /// 鍗曞厓鏁伴噺
    fn n_cells(&self) -> usize;

    /// 鑾峰彇鍗曞厓鐨勫畧鎭掔姸鎬?
    fn get(&self, cell: usize) -> ConservedState;

    /// 鑾峰彇姘存繁 [m]
    fn h(&self, cell: usize) -> f64;

    /// 鑾峰彇 x 鏂瑰悜鍔ㄩ噺 [m虏/s]
    fn hu(&self, cell: usize) -> f64;

    /// 鑾峰彇 y 鏂瑰悜鍔ㄩ噺 [m虏/s]
    fn hv(&self, cell: usize) -> f64;

    /// 鑾峰彇搴曞簥楂樼▼ [m]
    fn z(&self, cell: usize) -> f64;

    /// 鑾峰彇姘翠綅锛堟按娣?+ 搴曞簥锛塠m]
    #[inline]
    fn eta(&self, cell: usize) -> f64 {
        self.h(cell) + self.z(cell)
    }

    /// 璁＄畻閫熷害
    fn velocity(&self, cell: usize, params: &NumericalParams) -> SafeVelocity {
        params.safe_velocity(self.hu(cell), self.hv(cell), self.h(cell))
    }

    /// 鍒ゆ柇鏄惁涓哄共鍗曞厓
    fn is_dry(&self, cell: usize, params: &NumericalParams) -> bool {
        params.is_dry(self.h(cell))
    }

    /// 鍒ゆ柇鏄惁涓烘箍鍗曞厓
    #[inline]
    fn is_wet(&self, cell: usize, params: &NumericalParams) -> bool {
        !self.is_dry(cell, params)
    }

    // ===== 鎵归噺璁块棶锛堝垏鐗囧紩鐢級=====

    /// 姘存繁鏁扮粍寮曠敤
    fn h_slice(&self) -> &[f64];

    /// x 鍔ㄩ噺鏁扮粍寮曠敤
    fn hu_slice(&self) -> &[f64];

    /// y 鍔ㄩ噺鏁扮粍寮曠敤
    fn hv_slice(&self) -> &[f64];

    /// 搴曞簥楂樼▼鏁扮粍寮曠敤
    fn z_slice(&self) -> &[f64];
}

/// 鐘舵€佸彲鍙樿闂帴鍙?
///
/// 鎻愪緵瀵规祬姘存柟绋嬪畧鎭掑彉閲忕殑缁熶竴鍙彉璁块棶銆?
pub trait StateAccessMut: StateAccess {
    /// 璁剧疆鍗曞厓鐨勫畧鎭掔姸鎬?
    fn set(&mut self, cell: usize, state: ConservedState);

    /// 璁剧疆姘存繁 [m]
    fn set_h(&mut self, cell: usize, value: f64);

    /// 璁剧疆 x 鏂瑰悜鍔ㄩ噺 [m虏/s]
    fn set_hu(&mut self, cell: usize, value: f64);

    /// 璁剧疆 y 鏂瑰悜鍔ㄩ噺 [m虏/s]
    fn set_hv(&mut self, cell: usize, value: f64);

    /// 璁剧疆搴曞簥楂樼▼ [m]
    fn set_z(&mut self, cell: usize, value: f64);

    // ===== 鎵归噺鍙彉璁块棶 =====

    /// 姘存繁鏁扮粍鍙彉寮曠敤
    fn h_slice_mut(&mut self) -> &mut [f64];

    /// x 鍔ㄩ噺鏁扮粍鍙彉寮曠敤
    fn hu_slice_mut(&mut self) -> &mut [f64];

    /// y 鍔ㄩ噺鏁扮粍鍙彉寮曠敤
    fn hv_slice_mut(&mut self) -> &mut [f64];

    /// 搴曞簥楂樼▼鏁扮粍鍙彉寮曠敤
    fn z_slice_mut(&mut self) -> &mut [f64];

    // ===== 鎵归噺鎿嶄綔 =====

    /// 搴旂敤閫氶噺鏇存柊
    ///
    /// # 鍙傛暟
    /// - `dt`: 鏃堕棿姝ラ暱
    /// - `areas`: 鍗曞厓闈㈢Н
    /// - `flux_h`, `flux_hu`, `flux_hv`: 鍚勫崟鍏冪疮绉€氶噺
    fn apply_flux_update(
        &mut self,
        dt: f64,
        areas: &[f64],
        flux_h: &[f64],
        flux_hu: &[f64],
        flux_hv: &[f64],
    ) {
        let n = self.n_cells();
        debug_assert_eq!(areas.len(), n);
        debug_assert_eq!(flux_h.len(), n);
        debug_assert_eq!(flux_hu.len(), n);
        debug_assert_eq!(flux_hv.len(), n);

        for i in 0..n {
            let inv_area = 1.0 / areas[i];
            let h_new = self.h(i) + dt * flux_h[i] * inv_area;
            let hu_new = self.hu(i) + dt * flux_hu[i] * inv_area;
            let hv_new = self.hv(i) + dt * flux_hv[i] * inv_area;
            self.set_h(i, h_new);
            self.set_hu(i, hu_new);
            self.set_hv(i, hv_new);
        }
    }

    /// 搴旂敤婧愰」鏇存柊
    fn apply_source_update(
        &mut self,
        dt: f64,
        source_h: &[f64],
        source_hu: &[f64],
        source_hv: &[f64],
    ) {
        let n = self.n_cells();
        for i in 0..n {
            self.set_h(i, self.h(i) + dt * source_h[i]);
            self.set_hu(i, self.hu(i) + dt * source_hu[i]);
            self.set_hv(i, self.hv(i) + dt * source_hv[i]);
        }
    }

    /// 寮哄埗闈炶礋姘存繁
    fn enforce_non_negative_depth(&mut self, h_min: f64) {
        let h = self.h_slice_mut();
        for value in h.iter_mut() {
            if *value < h_min {
                *value = 0.0;
            }
        }
    }

    /// 浠庡彟涓€涓姸鎬佸鍒?
    ///
    /// # 閿欒
    /// 濡傛灉鍗曞厓鏁伴噺涓嶅尮閰嶅垯杩斿洖閿欒
    fn copy_from<S: StateAccess>(&mut self, other: &S) -> Result<(), &'static str> {
        if self.n_cells() != other.n_cells() {
            return Err("单元数量不匹配");
        }
        for i in 0..self.n_cells() {
            self.set(i, other.get(i));
            self.set_z(i, other.z(i));
        }
        Ok(())
    }
}

// ============================================================
// 杈呭姪绫诲瀷
// ============================================================

/// 鐘舵€佽鍥撅紙鍊熺敤鍒嗙锛岀敤浜庡悓鏃惰鍐欎笉鍚屽瓧娈碉級
pub struct StateView<'a> {
    /// 姘存繁鍒囩墖
    pub h: &'a [f64],
    /// x 鍔ㄩ噺鍒囩墖
    pub hu: &'a [f64],
    /// y 鍔ㄩ噺鍒囩墖
    pub hv: &'a [f64],
    /// 搴曞簥楂樼▼鍒囩墖
    pub z: &'a [f64],
}

impl<'a> StateView<'a> {
    /// 鍒涘缓鐘舵€佽鍥?
    pub fn new(h: &'a [f64], hu: &'a [f64], hv: &'a [f64], z: &'a [f64]) -> Self {
        Self { h, hu, hv, z }
    }

    /// 鍗曞厓鏁伴噺
    pub fn n_cells(&self) -> usize {
        self.h.len()
    }

    /// 鑾峰彇瀹堟亽鐘舵€?
    pub fn get(&self, cell: usize) -> ConservedState {
        ConservedState::new(self.h[cell], self.hu[cell], self.hv[cell])
    }

    /// 鑾峰彇姘翠綅
    pub fn eta(&self, cell: usize) -> f64 {
        self.h[cell] + self.z[cell]
    }
}

/// 鍙彉鐘舵€佽鍥?
pub struct StateViewMut<'a> {
    /// 姘存繁鍒囩墖
    pub h: &'a mut [f64],
    /// x 鍔ㄩ噺鍒囩墖
    pub hu: &'a mut [f64],
    /// y 鍔ㄩ噺鍒囩墖
    pub hv: &'a mut [f64],
    /// 搴曞簥楂樼▼鍒囩墖
    pub z: &'a mut [f64],
}

impl<'a> StateViewMut<'a> {
    /// 鍒涘缓鍙彉鐘舵€佽鍥?
    pub fn new(
        h: &'a mut [f64],
        hu: &'a mut [f64],
        hv: &'a mut [f64],
        z: &'a mut [f64],
    ) -> Self {
        Self { h, hu, hv, z }
    }

    /// 鍗曞厓鏁伴噺
    pub fn n_cells(&self) -> usize {
        self.h.len()
    }

    /// 璁剧疆瀹堟亽鐘舵€?
    pub fn set(&mut self, cell: usize, state: ConservedState) {
        self.h[cell] = state.h;
        self.hu[cell] = state.hu;
        self.hv[cell] = state.hv;
    }
}

// ============================================================
// StateAccess 鎵╁睍鏂规硶
// ============================================================

/// 鐘舵€佽闂墿灞曟柟娉?
///
/// 鎻愪緵鍩轰簬 StateAccess 鐨勪究鎹锋柟娉曪紝鏃犻渶鍗曠嫭瀹炵幇銆?
pub trait StateAccessExt: StateAccess {
    /// 璁＄畻鎬绘按閲忥紙浣撶Н锛?
    ///
    /// # 鍙傛暟
    ///
    /// - `areas`: 鍗曞厓闈㈢Н鏁扮粍
    fn total_volume(&self, areas: &[f64]) -> f64 {
        self.h_slice()
            .iter()
            .zip(areas.iter())
            .map(|(h, a)| h * a)
            .sum()
    }

    /// 璁＄畻婀垮崟鍏冩暟閲?
    fn wet_cell_count(&self, h_threshold: f64) -> usize {
        self.h_slice().iter().filter(|&&h| h > h_threshold).count()
    }

    /// 璁＄畻骞插崟鍏冩暟閲?
    fn dry_cell_count(&self, h_threshold: f64) -> usize {
        self.n_cells() - self.wet_cell_count(h_threshold)
    }

    /// 鑾峰彇鏈€澶ф按娣?
    fn max_depth(&self) -> f64 {
        self.h_slice()
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// 鑾峰彇鏈€灏忔按娣?
    fn min_depth(&self) -> f64 {
        self.h_slice()
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    }

    /// 鑾峰彇骞冲潎姘存繁
    fn mean_depth(&self) -> f64 {
        if self.n_cells() == 0 {
            return 0.0;
        }
        self.h_slice().iter().sum::<f64>() / self.n_cells() as f64
    }

    /// 妫€鏌ユ槸鍚﹀寘鍚?NaN 鎴?Inf
    fn has_invalid_values(&self) -> bool {
        self.h_slice().iter().any(|&v| !v.is_finite())
            || self.hu_slice().iter().any(|&v| !v.is_finite())
            || self.hv_slice().iter().any(|&v| !v.is_finite())
    }

    /// 鑾峰彇鏃犳晥鍊肩殑鍗曞厓绱㈠紩鍒楄〃
    fn invalid_cell_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        for i in 0..self.n_cells() {
            if !self.h(i).is_finite() || !self.hu(i).is_finite() || !self.hv(i).is_finite() {
                indices.push(i);
            }
        }
        indices
    }

    /// 璁＄畻鏈€澶ч€熷害锛堢敤浜?CFL 绾︽潫锛?
    fn max_velocity_magnitude(&self, params: &NumericalParams) -> f64 {
        let mut max_v = 0.0;
        for i in 0..self.n_cells() {
            if self.h(i) > params.h_dry {
                let vel = self.velocity(i, params);
                let mag = (vel.u * vel.u + vel.v * vel.v).sqrt();
                if mag > max_v {
                    max_v = mag;
                }
            }
        }
        max_v
    }

    /// 鑾峰彇鐘舵€佺粺璁′俊鎭?
    fn statistics(&self) -> StateStatistics {
        StateStatistics {
            n_cells: self.n_cells(),
            h_min: self.min_depth(),
            h_max: self.max_depth(),
            h_mean: self.mean_depth(),
        }
    }
}

/// 鐘舵€佺粺璁′俊鎭?
#[derive(Debug, Clone, Default)]
pub struct StateStatistics {
    /// 鍗曞厓鏁?
    pub n_cells: usize,
    /// 鏈€灏忔按娣?
    pub h_min: f64,
    /// 鏈€澶ф按娣?
    pub h_max: f64,
    /// 骞冲潎姘存繁
    pub h_mean: f64,
}

// 涓烘墍鏈夊疄鐜?StateAccess 鐨勭被鍨嬭嚜鍔ㄥ疄鐜?StateAccessExt
impl<T: StateAccess + ?Sized> StateAccessExt for T {}

// ============================================================
// 娴嬭瘯
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ShallowWaterState;

    #[test]
    fn test_state_access_basic() {
        let state = ShallowWaterState::new(10);
        
        // 閫氳繃 trait 璁块棶
        fn check_state<S: StateAccess>(s: &S) -> usize {
            s.n_cells()
        }
        
        assert_eq!(check_state(&state), 10);
    }

    #[test]
    fn test_state_access_mut() {
        let mut state = ShallowWaterState::new(10);
        
        // 閫氳繃 trait 淇敼
        fn modify_state<S: StateAccessMut>(s: &mut S) {
            s.set_h(0, 1.5);
        }
        
        modify_state(&mut state);
        assert!((state.h(0) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_state_view() {
        let h = vec![1.0, 2.0, 3.0];
        let hu = vec![0.1, 0.2, 0.3];
        let hv = vec![0.0, 0.0, 0.0];
        let z = vec![0.0, 1.0, 2.0];
        
        let view = StateView::new(&h, &hu, &hv, &z);
        
        assert_eq!(view.n_cells(), 3);
        assert!((view.eta(1) - 3.0).abs() < 1e-10); // h=2.0 + z=1.0
    }

    #[test]
    fn test_state_access_ext() {
        let mut state = ShallowWaterState::new(5);
        state.set_h(0, 1.0);
        state.set_h(1, 2.0);
        state.set_h(2, 3.0);
        state.set_h(3, 0.001);
        state.set_h(4, 0.0);

        // 娴嬭瘯鎵╁睍鏂规硶
        assert_eq!(state.wet_cell_count(0.01), 3);
        assert_eq!(state.dry_cell_count(0.01), 2);
        assert!((state.max_depth() - 3.0).abs() < 1e-10);
        assert!((state.min_depth() - 0.0).abs() < 1e-10);
        assert!(!state.has_invalid_values());
    }
}
