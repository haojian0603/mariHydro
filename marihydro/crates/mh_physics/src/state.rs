// crates/mh_physics/src/state.rs

//! 娴呮按鏂圭▼鐘舵€佺鐞?
//!
//! 鏈ā鍧楁彁渚涙祬姘存柟绋嬫眰瑙ｆ墍闇€鐨勭姸鎬佺鐞嗭紝鍖呮嫭锛?
//! - ShallowWaterState: 瀹堟亽鍙橀噺鐘舵€?(h, hu, hv, z)
//! - GradientState: 姊害鐘舵€?(grad_h, grad_hu, grad_hv)
//! - Flux: 鏁板€奸€氶噺
//! - RhsBuffers: 鍙崇椤圭紦鍐插尯
//!
//! # 甯冨眬璁捐
//!
//! 閲囩敤 SoA (Structure of Arrays) 甯冨眬浠ヤ紭鍖栫紦瀛樻€ц兘锛?
//! ```text
//! h:  [h_0,  h_1,  h_2,  ...]
//! hu: [hu_0, hu_1, hu_2, ...]
//! hv: [hv_0, hv_1, hv_2, ...]
//! z:  [z_0,  z_1,  z_2,  ...]
//! ```
//!


use glam::DVec2;
use mh_foundation::memory::AlignedVec;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use crate::fields::{FieldMeta, FieldRegistry};
use crate::types::{CellIndex, NumericalParams, SafeVelocity};

// ============================================================
// 瀹堟亽鐘舵€?
// ============================================================

/// 鍗曚釜鍗曞厓鐨勫畧鎭掔姸鎬?
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ConservedState {
    /// 姘存繁 [m]
    pub h: f64,
    /// x 鏂瑰悜鍔ㄩ噺 [m虏/s]
    pub hu: f64,
    /// y 鏂瑰悜鍔ㄩ噺 [m虏/s]
    pub hv: f64,
}

impl ConservedState {
    /// 鍒涘缓鏂扮殑瀹堟亽鐘舵€?
    #[inline]
    pub const fn new(h: f64, hu: f64, hv: f64) -> Self {
        Self { h, hu, hv }
    }

    /// 闆剁姸鎬?
    pub const ZERO: Self = Self {
        h: 0.0,
        hu: 0.0,
        hv: 0.0,
    };

    /// 浠庡師濮嬪彉閲忓垱寤?
    #[inline]
    pub fn from_primitive(h: f64, u: f64, v: f64) -> Self {
        Self {
            h,
            hu: h * u,
            hv: h * v,
        }
    }

    /// 鑾峰彇閫熷害 (浣跨敤瀹夊叏闄ゆ硶)
    #[inline]
    pub fn velocity(&self, params: &NumericalParams) -> SafeVelocity {
        params.safe_velocity(self.hu, self.hv, self.h)
    }

    /// 鐘舵€佹槸鍚︽湁鏁?
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.h.is_finite() && self.hu.is_finite() && self.hv.is_finite() && self.h >= 0.0
    }
}

impl Add for ConservedState {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            h: self.h + rhs.h,
            hu: self.hu + rhs.hu,
            hv: self.hv + rhs.hv,
        }
    }
}

impl Sub for ConservedState {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            h: self.h - rhs.h,
            hu: self.hu - rhs.hu,
            hv: self.hv - rhs.hv,
        }
    }
}

impl Mul<f64> for ConservedState {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            h: self.h * rhs,
            hu: self.hu * rhs,
            hv: self.hv * rhs,
        }
    }
}

// ============================================================
// 鍔ㄦ€佹爣閲忓満锛堢ず韪墏绛夛級
// ============================================================

/// 鍔ㄦ€佹爣閲忓満闆嗗悎锛屾寜鍚嶇О绠＄悊绀鸿釜鍓傜瓑鎵╁睍瀛楁
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DynamicScalars {
    /// 鍗曞厓鏁伴噺
    #[serde(default)]
    len: usize,
    /// 瀛楁鍚嶇О鍒楄〃锛堥『搴忓嵆瀛樺偍椤哄簭锛?
    #[serde(default)]
    names: Vec<String>,
    /// 鏁版嵁瀛樺偍
    #[serde(default)]
    data: Vec<AlignedVec<f64>>,
}

impl DynamicScalars {
    /// 鍒涘缓绌洪泦鍚?
    pub fn new(len: usize) -> Self {
        Self {
            len,
            names: Vec::new(),
            data: Vec::new(),
        }
    }

    /// 鍒涘缓鎸囧畾鏁伴噺鐨勫尶鍚嶇ず韪墏瀛楁锛堝悕绉颁负 tracer_i锛?
    pub fn with_count(len: usize, count: usize) -> Self {
        let mut scalars = Self::new(len);
        for i in 0..count {
            scalars.register(format!("tracer_{i}"));
        }
        scalars
    }

    /// 褰撳墠鍗曞厓鏁伴噺
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 瀛楁鏁伴噺
    #[inline]
    pub fn count(&self) -> usize {
        self.data.len()
    }

    /// 瀛楁鍚嶇О鍒楄〃
    #[inline]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// 娉ㄥ唽涓€涓柊瀛楁锛屽宸插瓨鍦ㄥ垯鐩存帴杩斿洖绱㈠紩
    pub fn register(&mut self, name: impl Into<String>) -> usize {
        let name = name.into();
        if let Some(pos) = self.names.iter().position(|n| n == &name) {
            // 纭繚闀垮害涓€鑷?
            self.data[pos].resize(self.len);
            return pos;
        }

        self.names.push(name);
        self.data.push(AlignedVec::zeros(self.len));
        self.data.len() - 1
    }

    /// 鎸夌储寮曡幏鍙栧彧璇诲垏鐗?
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&[f64]> {
        self.data.get(idx).map(|v| v.as_slice())
    }

    /// 鎸夌储寮曡幏鍙栧彲鍙樺垏鐗?
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut [f64]> {
        self.data.get_mut(idx).map(|v| v.as_mut_slice())
    }

    /// 鎸夊悕绉拌幏鍙栧彧璇诲垏鐗?
    pub fn get_by_name(&self, name: &str) -> Option<&[f64]> {
        self.names.iter().position(|n| n == name).and_then(|i| self.get(i))
    }

    /// 鎸夊悕绉拌幏鍙栧彲鍙樺垏鐗?
    pub fn get_mut_by_name(&mut self, name: &str) -> Option<&mut [f64]> {
        if let Some(pos) = self.names.iter().position(|n| n == name) {
            return self.get_mut(pos);
        }
        None
    }

    /// 灏嗘墍鏈夊瓧娈垫竻闆?
    pub fn clear_all(&mut self) {
        for field in &mut self.data {
            field.as_mut_slice().fill(0.0);
        }
    }

    /// 璋冩暣鍗曞厓闀垮害骞朵繚鎸佸凡鏈夋暟鎹紙鏂板閮ㄥ垎濉浂锛?
    pub fn resize_len(&mut self, len: usize) {
        self.len = len;
        for field in &mut self.data {
            field.resize(len);
        }
    }

    /// 鎸夊彟涓€涓泦鍚堢殑甯冨眬瀵归綈锛堝悕绉般€佹暟閲忋€侀暱搴︼級锛屼絾涓嶅鍒舵暟鎹?
    pub fn match_layout(&mut self, other: &Self) {
        if self.len != other.len || self.names != other.names {
            self.len = other.len;
            self.names = other.names.clone();
            self.data = other
                .data
                .iter()
                .map(|_| AlignedVec::zeros(other.len))
                .collect();
        } else {
            self.resize_len(other.len);
        }
    }

    /// 澶嶅埗鏁版嵁骞跺榻愬竷灞€
    pub fn copy_from(&mut self, other: &Self) {
        self.match_layout(other);
        for (dst, src) in self.data.iter_mut().zip(other.data.iter()) {
            dst.as_mut_slice().copy_from_slice(src.as_slice());
        }
    }

    /// self += scale * rhs
    pub fn add_scaled(&mut self, rhs: &Self, scale: f64) {
        self.match_layout(rhs);
        for (dst, src) in self.data.iter_mut().zip(rhs.data.iter()) {
            for (d, s) in dst.as_mut_slice().iter_mut().zip(src.as_slice()) {
                *d += scale * s;
            }
        }
    }

    /// 璁剧疆瀛楁鏁伴噺锛屽浣欑殑鎴柇锛屼笉瓒崇殑浠?tracer_i 濉厖
    pub fn set_count(&mut self, count: usize) {
        self.names.truncate(count);
        self.data.truncate(count);
        while self.data.len() < count {
            let idx = self.data.len();
            self.names.push(format!("tracer_{idx}"));
            self.data.push(AlignedVec::zeros(self.len));
        }
    }

    /// self = a * A + b * B
    pub fn linear_combine(&mut self, a: f64, state_a: &Self, b: f64, state_b: &Self) {
        debug_assert_eq!(state_a.names, state_b.names, "示踪剂字段布局不一致");
        self.match_layout(state_a);
        for ((dst, sa), sb) in self
            .data
            .iter_mut()
            .zip(state_a.data.iter())
            .zip(state_b.data.iter())
        {
            for ((d, a_val), b_val) in dst
                .as_mut_slice()
                .iter_mut()
                .zip(sa.as_slice())
                .zip(sb.as_slice())
            {
                *d = a * a_val + b * b_val;
            }
        }
    }

    /// self = a * self + b * other
    pub fn axpy(&mut self, a: f64, b: f64, other: &Self) {
        self.match_layout(other);
        for (dst, src) in self.data.iter_mut().zip(other.data.iter()) {
            for (d, s) in dst.as_mut_slice().iter_mut().zip(src.as_slice()) {
                *d = a * *d + b * s;
            }
        }
    }

    /// 杩唬鎵€鏈夊瓧娈电殑鍙彉瀛樺偍
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut AlignedVec<f64>> {
        self.data.iter_mut()
    }
}

// ============================================================
// 娴呮按鏂圭▼鐘舵€?(SoA 甯冨眬)
// ============================================================

/// 娴呮按鏂圭▼瀹堟亽鐘舵€侊紙SoA 甯冨眬锛?
///
/// 瀛樺偍鏁翠釜缃戞牸鐨勭姸鎬佸彉閲忥紝閲囩敤 SoA 甯冨眬浼樺寲缂撳瓨璁块棶銆?
/// 
/// 閫熷害鍦洪€氳繃 `velocity()` 鏂规硶浠庡姩閲忓拰姘存繁瀹炴椂璁＄畻锛?
/// 閬垮厤瀛樺偍鍐椾綑鏁版嵁骞剁‘淇濇暟鎹竴鑷存€с€?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShallowWaterState {
    /// 鍗曞厓鏁伴噺
    n_cells: usize,

    /// 姘存繁 [m]
    pub h: AlignedVec<f64>,
    /// x 鏂瑰悜鍔ㄩ噺 [m虏/s]
    pub hu: AlignedVec<f64>,
    /// y 鏂瑰悜鍔ㄩ噺 [m虏/s]
    pub hv: AlignedVec<f64>,
    /// 搴曞簥楂樼▼ [m]
    pub z: AlignedVec<f64>,

    /// 鍔ㄦ€佺ず韪墏瀛楁
    #[serde(default)]
    pub tracers: DynamicScalars,

    /// 瀛楁娉ㄥ唽琛紙鍏冩暟鎹級
    #[serde(default = "FieldRegistry::shallow_water")]
    pub field_registry: FieldRegistry,
}

impl ShallowWaterState {
    /// 鍒涘缓鏂扮姸鎬?
    pub fn new(n_cells: usize) -> Self {
        Self {
            n_cells,
            h: AlignedVec::zeros(n_cells),
            hu: AlignedVec::zeros(n_cells),
            hv: AlignedVec::zeros(n_cells),
            z: AlignedVec::zeros(n_cells),
            tracers: DynamicScalars::new(n_cells),
            field_registry: FieldRegistry::shallow_water(),
        }
    }

    /// 鍒涘缓甯︽爣閲忕殑鐘舵€?
    pub fn with_scalar(n_cells: usize) -> Self {
        let mut state = Self::new(n_cells);
        state.register_tracer("tracer_0", "");
        state
    }

    /// 浠庡垵濮嬫按浣嶅拰搴曞簥鍒涘缓锛堝喎鍚姩锛?
    pub fn cold_start(initial_eta: f64, z_bed: &[f64]) -> Self {
        let n_cells = z_bed.len();

        let h: Vec<f64> = z_bed.iter().map(|&z| (initial_eta - z).max(0.0)).collect();

        Self {
            n_cells,
            h: AlignedVec::from_vec(h),
            hu: AlignedVec::zeros(n_cells),
            hv: AlignedVec::zeros(n_cells),
            z: AlignedVec::from_vec(z_bed.to_vec()),
            tracers: DynamicScalars::new(n_cells),
            field_registry: FieldRegistry::shallow_water(),
        }
    }

    /// 鍏嬮殕缁撴瀯锛堜笉澶嶅埗鏁版嵁锛屽垱寤洪浂鍒濆鍖栫殑鐘舵€侊級
    pub fn clone_structure(&self) -> Self {
        let mut tracers = DynamicScalars::new(self.n_cells);
        tracers.match_layout(&self.tracers);
        Self {
            n_cells: self.n_cells,
            h: AlignedVec::zeros(self.n_cells),
            hu: AlignedVec::zeros(self.n_cells),
            hv: AlignedVec::zeros(self.n_cells),
            z: self.z.clone(),
            tracers,
            field_registry: self.field_registry.clone(),
        }
    }

    /// 鍗曞厓鏁伴噺
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 娉ㄥ唽涓€涓柊鐨勭ず韪墏瀛楁锛岃嫢宸插瓨鍦ㄥ垯杩斿洖鍏剁储寮?
    pub fn register_tracer(&mut self, name: impl Into<String>, unit: impl Into<String>) -> usize {
        let name = name.into();
        let idx = self.tracers.register(name.clone());
        if !self.field_registry.contains(&name) {
            self.field_registry
                .register(FieldMeta::cell_scalar(name.clone(), unit.into()).with_desc("示踪剂标量"));
        }
        idx
    }

    /// 鑾峰彇绀鸿釜鍓傛暟閲?
    #[inline]
    pub fn tracer_count(&self) -> usize {
        self.tracers.count()
    }

    /// 鑾峰彇鎵€鏈夌ず韪墏鍚嶇О
    #[inline]
    pub fn tracer_names(&self) -> &[String] {
        self.tracers.names()
    }

    /// 鎸夌储寮曡幏鍙栫ず韪墏鍒囩墖
    #[inline]
    pub fn tracer_slice(&self, idx: usize) -> Option<&[f64]> {
        self.tracers.get(idx)
    }

    /// 鎸夌储寮曡幏鍙栧彲鍙樼ず韪墏鍒囩墖
    #[inline]
    pub fn tracer_slice_mut(&mut self, idx: usize) -> Option<&mut [f64]> {
        self.tracers.get_mut(idx)
    }

    /// 鎸夊悕绉拌幏鍙栫ず韪墏鍒囩墖
    #[inline]
    pub fn tracer_by_name(&self, name: &str) -> Option<&[f64]> {
        self.tracers.get_by_name(name)
    }

    /// 鎸夊悕绉拌幏鍙栧彲鍙樼ず韪墏鍒囩墖
    #[inline]
    pub fn tracer_by_name_mut(&mut self, name: &str) -> Option<&mut [f64]> {
        self.tracers.get_mut_by_name(name)
    }

    // ========== 鐘舵€佽闂?==========

    /// 鑾峰彇鍗曞厓鐨勫畧鎭掔姸鎬?
    #[inline]
    pub fn get(&self, idx: usize) -> ConservedState {
        ConservedState::new(self.h[idx], self.hu[idx], self.hv[idx])
    }

    /// 鑾峰彇鍗曞厓鐨勫畧鎭掔姸鎬侊紙浣跨敤 CellIndex锛?
    #[inline]
    pub fn get_by_index(&self, cell: CellIndex) -> ConservedState {
        self.get(cell.get())
    }

    /// 鑾峰彇鍘熷鍙橀噺 (h, u, v)
    #[inline]
    pub fn primitive(&self, idx: usize, params: &NumericalParams) -> (f64, f64, f64) {
        let h = self.h[idx];
        let vel = params.safe_velocity(self.hu[idx], self.hv[idx], h);
        (h, vel.u, vel.v)
    }

    /// 鑾峰彇閫熷害
    #[inline]
    pub fn velocity(&self, idx: usize, params: &NumericalParams) -> SafeVelocity {
        params.safe_velocity(self.hu[idx], self.hv[idx], self.h[idx])
    }

    /// 鑾峰彇閫熷害锛堜娇鐢ㄩ槇鍊硷級
    #[inline]
    pub fn velocity_with_eps(&self, idx: usize, eps: f64) -> DVec2 {
        let h = self.h[idx];
        if h > eps {
            DVec2::new(self.hu[idx] / h, self.hv[idx] / h)
        } else {
            DVec2::ZERO
        }
    }

    /// 鑾峰彇姘翠綅 (eta = h + z)
    #[inline]
    pub fn water_level(&self, idx: usize) -> f64 {
        self.h[idx] + self.z[idx]
    }

    // ========== 鐘舵€佷慨鏀?==========

    /// 璁剧疆瀹堟亽鍙橀噺
    #[inline]
    pub fn set(&mut self, idx: usize, h: f64, hu: f64, hv: f64) {
        self.h[idx] = h;
        self.hu[idx] = hu;
        self.hv[idx] = hv;
    }

    /// 璁剧疆瀹堟亽鐘舵€?
    #[inline]
    pub fn set_state(&mut self, idx: usize, state: ConservedState) {
        self.h[idx] = state.h;
        self.hu[idx] = state.hu;
        self.hv[idx] = state.hv;
    }

    /// 浠庡師濮嬪彉閲忚缃?
    #[inline]
    pub fn set_from_primitive(&mut self, idx: usize, h: f64, u: f64, v: f64) {
        self.h[idx] = h;
        self.hu[idx] = h * u;
        self.hv[idx] = h * v;
    }

    /// 閲嶇疆涓洪浂
    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.hu.fill(0.0);
        self.hv.fill(0.0);
        self.tracers.clear_all();
    }

    // ========== 鍒囩墖璁块棶 ==========

    /// 鑾峰彇姘存繁鍒囩墖
    #[inline]
    pub fn h_slice(&self) -> &[f64] {
        &self.h
    }

    /// 鑾峰彇 x 鍔ㄩ噺鍒囩墖
    #[inline]
    pub fn hu_slice(&self) -> &[f64] {
        &self.hu
    }

    /// 鑾峰彇 y 鍔ㄩ噺鍒囩墖
    #[inline]
    pub fn hv_slice(&self) -> &[f64] {
        &self.hv
    }

    /// 鑾峰彇搴曞簥楂樼▼鍒囩墖
    #[inline]
    pub fn z_slice(&self) -> &[f64] {
        &self.z
    }

    /// 鑾峰彇鍙彉姘存繁鍒囩墖
    #[inline]
    pub fn h_slice_mut(&mut self) -> &mut [f64] {
        &mut self.h
    }

    /// 鑾峰彇鍙彉 x 鍔ㄩ噺鍒囩墖
    #[inline]
    pub fn hu_slice_mut(&mut self) -> &mut [f64] {
        &mut self.hu
    }

    /// 鑾峰彇鍙彉 y 鍔ㄩ噺鍒囩墖
    #[inline]
    pub fn hv_slice_mut(&mut self) -> &mut [f64] {
        &mut self.hv
    }

    /// 鑾峰彇鍙彉搴曞簥楂樼▼鍒囩墖
    #[inline]
    pub fn z_slice_mut(&mut self) -> &mut [f64] {
        &mut self.z
    }

    // ========== 绉垎璁＄畻 ==========

    /// 璁＄畻鎬昏川閲?
    pub fn total_mass(&self, cell_areas: &[f64]) -> f64 {
        self.h.iter().zip(cell_areas).map(|(h, a)| h * a).sum()
    }

    /// 璁＄畻鎬诲姩閲?
    pub fn total_momentum(&self, cell_areas: &[f64]) -> DVec2 {
        let hux: f64 = self.hu.iter().zip(cell_areas).map(|(hu, a)| hu * a).sum();
        let hvx: f64 = self.hv.iter().zip(cell_areas).map(|(hv, a)| hv * a).sum();
        DVec2::new(hux, hvx)
    }

    // ========== 鏃堕棿绉垎鏀寔 ==========

    /// 浠庡彟涓€涓姸鎬佸鍒舵暟鎹?
    pub fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.n_cells(), other.n_cells());
        self.h.copy_from_slice(&other.h);
        self.hu.copy_from_slice(&other.hu);
        self.hv.copy_from_slice(&other.hv);
        self.tracers.copy_from(&other.tracers);
    }

    /// 娣诲姞缂╂斁鐨?RHS: self += scale * rhs
    pub fn add_scaled_rhs(&mut self, rhs: &RhsBuffers, scale: f64) {
        for i in 0..self.n_cells {
            self.h[i] += scale * rhs.dh_dt[i];
            self.hu[i] += scale * rhs.dhu_dt[i];
            self.hv[i] += scale * rhs.dhv_dt[i];
        }
        self.tracers.add_scaled(&rhs.tracer_rhs, scale);
    }

    /// 浜屽厓绾挎€х粍鍚? self = a*A + b*B
    pub fn linear_combine(&mut self, a: f64, state_a: &Self, b: f64, state_b: &Self) {
        debug_assert_eq!(self.n_cells(), state_a.n_cells());
        debug_assert_eq!(self.n_cells(), state_b.n_cells());

        for i in 0..self.n_cells {
            self.h[i] = a * state_a.h[i] + b * state_b.h[i];
            self.hu[i] = a * state_a.hu[i] + b * state_b.hu[i];
            self.hv[i] = a * state_a.hv[i] + b * state_b.hv[i];
        }
        self.tracers.linear_combine(a, &state_a.tracers, b, &state_b.tracers);
    }

    /// 鑷嚎鎬х粍鍚? self = a * self + b * other
    pub fn axpy(&mut self, a: f64, b: f64, other: &Self) {
        debug_assert_eq!(self.n_cells(), other.n_cells());

        for i in 0..self.n_cells {
            self.h[i] = a * self.h[i] + b * other.h[i];
            self.hu[i] = a * self.hu[i] + b * other.hu[i];
            self.hv[i] = a * self.hv[i] + b * other.hv[i];
        }
        self.tracers.axpy(a, b, &other.tracers);
    }

    /// 寮哄埗姝ｆ€х害鏉?
    pub fn enforce_positivity(&mut self) {
        for h in self.h.iter_mut() {
            if *h < 0.0 {
                *h = 0.0;
            }
        }

        for tracer in self.tracers.iter_mut() {
            for v in tracer.as_mut_slice() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
    }

    // ========== 楠岃瘉 ==========

    /// 楠岃瘉鐘舵€佹湁鏁堟€?
    pub fn validate(&self, time: f64, params: &NumericalParams) -> Result<(), StateError> {
        for idx in 0..self.n_cells {
            // 妫€鏌?NaN/Inf
            if !self.h[idx].is_finite() {
                return Err(StateError::InvalidValue {
                    field: "h",
                    cell: idx,
                    value: self.h[idx],
                    time,
                });
            }

            if !self.hu[idx].is_finite() || !self.hv[idx].is_finite() {
                return Err(StateError::InvalidValue {
                    field: "momentum",
                    cell: idx,
                    value: if !self.hu[idx].is_finite() {
                        self.hu[idx]
                    } else {
                        self.hv[idx]
                    },
                    time,
                });
            }

            // 妫€鏌ヨ礋姘存繁
            if self.h[idx] < 0.0 {
                return Err(StateError::NegativeDepth {
                    cell: idx,
                    value: self.h[idx],
                    time,
                });
            }

            // 妫€鏌ラ€熷害
            if !params.is_dry(self.h[idx]) {
                let vel = self.velocity(idx, params);
                if params.is_velocity_excessive(vel.speed()) {
                    return Err(StateError::ExcessiveVelocity {
                        cell: idx,
                        speed: vel.speed(),
                        max_speed: params.vel_max,
                        time,
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================
// 鍙崇椤圭紦鍐插尯
// ============================================================

/// 鍙崇椤圭紦鍐插尯 (鐢ㄤ簬鏃堕棿绉垎)
#[derive(Debug, Clone)]
pub struct RhsBuffers {
    /// 姘存繁鍙樺寲鐜?[m/s]
    pub dh_dt: AlignedVec<f64>,
    /// x 鍔ㄩ噺鍙樺寲鐜?[m虏/s虏]
    pub dhu_dt: AlignedVec<f64>,
    /// y 鍔ㄩ噺鍙樺寲鐜?[m虏/s虏]
    pub dhv_dt: AlignedVec<f64>,
    /// 鏍囬噺绀鸿釜鍓傚彉鍖栫巼锛堝彲閫夛級
    pub tracer_rhs: DynamicScalars,
}

impl RhsBuffers {
    /// 鍒涘缓鏂扮殑 RHS 缂撳啿鍖?
    pub fn new(n_cells: usize) -> Self {
        Self {
            dh_dt: AlignedVec::zeros(n_cells),
            dhu_dt: AlignedVec::zeros(n_cells),
            dhv_dt: AlignedVec::zeros(n_cells),
            tracer_rhs: DynamicScalars::new(n_cells),
        }
    }

    /// 鍒涘缓甯︽湁绀鸿釜鍓傜殑 RHS 缂撳啿鍖?
    pub fn with_tracers(n_cells: usize, n_tracers: usize) -> Self {
        let mut rhs = Self::new(n_cells);
        rhs.tracer_rhs.set_count(n_tracers);
        rhs
    }

    /// 鑾峰彇鍗曞厓鏁伴噺
    pub fn n_cells(&self) -> usize {
        self.dh_dt.len()
    }

    /// 鑾峰彇绀鸿釜鍓傛暟閲?
    pub fn n_tracers(&self) -> usize {
        self.tracer_rhs.count()
    }

    /// 閲嶇疆涓洪浂
    pub fn reset(&mut self) {
        self.dh_dt.fill(0.0);
        self.dhu_dt.fill(0.0);
        self.dhv_dt.fill(0.0);
        self.tracer_rhs.clear_all();
    }

    /// 璋冩暣澶у皬
    pub fn resize(&mut self, n_cells: usize, n_tracers: usize) {
        self.dh_dt.resize(n_cells);
        self.dhu_dt.resize(n_cells);
        self.dhv_dt.resize(n_cells);
        self.tracer_rhs.resize_len(n_cells);
        self.tracer_rhs.set_count(n_tracers);
    }

    /// 灏嗙ず韪墏甯冨眬瀵归綈鍒扮粰瀹氱姸鎬?
    pub fn match_tracers(&mut self, layout: &DynamicScalars) {
        self.tracer_rhs.match_layout(layout);
    }

    /// 娣诲姞閫氶噺璐＄尞
    #[inline]
    pub fn add_flux(&mut self, cell: usize, flux: Flux, area_inv: f64) {
        self.dh_dt[cell] += flux.mass * area_inv;
        self.dhu_dt[cell] += flux.mom_x * area_inv;
        self.dhv_dt[cell] += flux.mom_y * area_inv;
    }

    /// 娣诲姞婧愰」璐＄尞
    #[inline]
    pub fn add_source(&mut self, cell: usize, source: ConservedState) {
        self.dh_dt[cell] += source.h;
        self.dhu_dt[cell] += source.hu;
        self.dhv_dt[cell] += source.hv;
    }
}

// ============================================================
// 姊害鐘舵€?
// ============================================================

/// 姊害鐘舵€?(鐢ㄤ簬浜岄樁閲嶆瀯)
#[derive(Debug, Clone)]
pub struct GradientState {
    /// 姘存繁姊害
    pub grad_h: Vec<DVec2>,
    /// x 鍔ㄩ噺姊害
    pub grad_hu: Vec<DVec2>,
    /// y 鍔ㄩ噺姊害
    pub grad_hv: Vec<DVec2>,
}

impl GradientState {
    /// 鍒涘缓鏂扮殑姊害鐘舵€?
    pub fn new(n_cells: usize) -> Self {
        Self {
            grad_h: vec![DVec2::ZERO; n_cells],
            grad_hu: vec![DVec2::ZERO; n_cells],
            grad_hv: vec![DVec2::ZERO; n_cells],
        }
    }

    /// 閲嶇疆涓洪浂
    pub fn reset(&mut self) {
        self.grad_h.fill(DVec2::ZERO);
        self.grad_hu.fill(DVec2::ZERO);
        self.grad_hv.fill(DVec2::ZERO);
    }

    /// 鑾峰彇鍗曞厓姊害
    #[inline]
    pub fn get(&self, cell: usize) -> (DVec2, DVec2, DVec2) {
        (self.grad_h[cell], self.grad_hu[cell], self.grad_hv[cell])
    }

    /// 璁剧疆鍗曞厓姊害
    #[inline]
    pub fn set(&mut self, cell: usize, grad_h: DVec2, grad_hu: DVec2, grad_hv: DVec2) {
        self.grad_h[cell] = grad_h;
        self.grad_hu[cell] = grad_hu;
        self.grad_hv[cell] = grad_hv;
    }
}

// ============================================================
// 鏁板€奸€氶噺
// ============================================================

/// 鏁板€奸€氶噺
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Flux {
    /// 璐ㄩ噺閫氶噺 [m虏/s]
    pub mass: f64,
    /// x 鍔ㄩ噺閫氶噺 [m鲁/s虏]
    pub mom_x: f64,
    /// y 鍔ㄩ噺閫氶噺 [m鲁/s虏]
    pub mom_y: f64,
}

impl Flux {
    /// 鍒涘缓鏂伴€氶噺
    #[inline]
    pub const fn new(mass: f64, mom_x: f64, mom_y: f64) -> Self {
        Self { mass, mom_x, mom_y }
    }

    /// 闆堕€氶噺
    pub const ZERO: Self = Self {
        mass: 0.0,
        mom_x: 0.0,
        mom_y: 0.0,
    };

    /// 缂╂斁閫氶噺
    #[inline]
    pub fn scale(self, factor: f64) -> Self {
        Self {
            mass: self.mass * factor,
            mom_x: self.mom_x * factor,
            mom_y: self.mom_y * factor,
        }
    }

    /// 閫氶噺澶у皬
    #[inline]
    pub fn magnitude(&self) -> f64 {
        (self.mass * self.mass + self.mom_x * self.mom_x + self.mom_y * self.mom_y).sqrt()
    }

    /// 妫€鏌ラ€氶噺鏄惁鏈夋晥
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.mass.is_finite() && self.mom_x.is_finite() && self.mom_y.is_finite()
    }
}

impl Add for Flux {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            mass: self.mass + rhs.mass,
            mom_x: self.mom_x + rhs.mom_x,
            mom_y: self.mom_y + rhs.mom_y,
        }
    }
}

impl AddAssign for Flux {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.mass += rhs.mass;
        self.mom_x += rhs.mom_x;
        self.mom_y += rhs.mom_y;
    }
}

impl Sub for Flux {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            mass: self.mass - rhs.mass,
            mom_x: self.mom_x - rhs.mom_x,
            mom_y: self.mom_y - rhs.mom_y,
        }
    }
}

impl SubAssign for Flux {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.mass -= rhs.mass;
        self.mom_x -= rhs.mom_x;
        self.mom_y -= rhs.mom_y;
    }
}

impl Neg for Flux {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            mass: -self.mass,
            mom_x: -self.mom_x,
            mom_y: -self.mom_y,
        }
    }
}

impl Mul<f64> for Flux {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        self.scale(rhs)
    }
}

impl Mul<Flux> for f64 {
    type Output = Flux;
    #[inline]
    fn mul(self, rhs: Flux) -> Flux {
        rhs.scale(self)
    }
}

// ============================================================
// 閿欒绫诲瀷
// ============================================================

/// 鐘舵€侀敊璇?
#[derive(Debug, Clone)]
pub enum StateError {
    /// 鏃犳晥鍊?(NaN/Inf)
    InvalidValue {
        field: &'static str,
        cell: usize,
        value: f64,
        time: f64,
    },
    /// 璐熸按娣?
    NegativeDepth {
        cell: usize,
        value: f64,
        time: f64,
    },
    /// 閫熷害杩囧ぇ
    ExcessiveVelocity {
        cell: usize,
        speed: f64,
        max_speed: f64,
        time: f64,
    },
    /// 灏哄涓嶅尮閰?
    SizeMismatch {
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidValue {
                field,
                cell,
                value,
                time,
            } => {
                write!(
                    f,
                    "Invalid {} at cell {} (value={}, time={})",
                    field, cell, value, time
                )
            }
            Self::NegativeDepth { cell, value, time } => {
                write!(
                    f,
                    "Negative depth at cell {} (h={}, time={})",
                    cell, value, time
                )
            }
            Self::ExcessiveVelocity {
                cell,
                speed,
                max_speed,
                time,
            } => {
                write!(
                    f,
                    "Excessive velocity at cell {} (speed={} > max={}, time={})",
                    cell, speed, max_speed, time
                )
            }
            Self::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Size mismatch: expected {} cells, got {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for StateError {}

// ============================================================
// StateAccess / StateAccessMut trait 瀹炵幇
// ============================================================

use crate::traits::{StateAccess, StateAccessMut};

impl StateAccess for ShallowWaterState {
    #[inline]
    fn n_cells(&self) -> usize {
        self.n_cells
    }

    #[inline]
    fn get(&self, cell: usize) -> ConservedState {
        ConservedState::new(self.h[cell], self.hu[cell], self.hv[cell])
    }

    #[inline]
    fn h(&self, cell: usize) -> f64 {
        self.h[cell]
    }

    #[inline]
    fn hu(&self, cell: usize) -> f64 {
        self.hu[cell]
    }

    #[inline]
    fn hv(&self, cell: usize) -> f64 {
        self.hv[cell]
    }

    #[inline]
    fn z(&self, cell: usize) -> f64 {
        self.z[cell]
    }

    #[inline]
    fn h_slice(&self) -> &[f64] {
        &self.h
    }

    #[inline]
    fn hu_slice(&self) -> &[f64] {
        &self.hu
    }

    #[inline]
    fn hv_slice(&self) -> &[f64] {
        &self.hv
    }

    #[inline]
    fn z_slice(&self) -> &[f64] {
        &self.z
    }
}

impl StateAccessMut for ShallowWaterState {
    #[inline]
    fn set(&mut self, cell: usize, state: ConservedState) {
        self.h[cell] = state.h;
        self.hu[cell] = state.hu;
        self.hv[cell] = state.hv;
    }

    #[inline]
    fn set_h(&mut self, cell: usize, value: f64) {
        self.h[cell] = value;
    }

    #[inline]
    fn set_hu(&mut self, cell: usize, value: f64) {
        self.hu[cell] = value;
    }

    #[inline]
    fn set_hv(&mut self, cell: usize, value: f64) {
        self.hv[cell] = value;
    }

    #[inline]
    fn set_z(&mut self, cell: usize, value: f64) {
        self.z[cell] = value;
    }

    #[inline]
    fn h_slice_mut(&mut self) -> &mut [f64] {
        &mut self.h
    }

    #[inline]
    fn hu_slice_mut(&mut self) -> &mut [f64] {
        &mut self.hu
    }

    #[inline]
    fn hv_slice_mut(&mut self) -> &mut [f64] {
        &mut self.hv
    }

    #[inline]
    fn z_slice_mut(&mut self) -> &mut [f64] {
        &mut self.z
    }
}

// ============================================================
// 鍗曞厓娴嬭瘯
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = ShallowWaterState::new(100);
        assert_eq!(state.n_cells(), 100);
        assert_eq!(state.h.len(), 100);
        assert_eq!(state.hu.len(), 100);
        assert_eq!(state.hv.len(), 100);
        assert_eq!(state.z.len(), 100);
    }

    #[test]
    fn test_cold_start() {
        let z_bed = vec![-10.0, -5.0, 0.0, 5.0];
        let state = ShallowWaterState::cold_start(0.0, &z_bed);

        assert_eq!(state.h[0], 10.0);
        assert_eq!(state.h[1], 5.0);
        assert_eq!(state.h[2], 0.0);
        assert_eq!(state.h[3], 0.0);
    }

    #[test]
    fn test_velocity_calculation() {
        let mut state = ShallowWaterState::new(1);
        state.h[0] = 2.0;
        state.hu[0] = 4.0;
        state.hv[0] = 6.0;

        let params = NumericalParams::default();
        let vel = state.velocity(0, &params);

        assert!((vel.u - 2.0).abs() < 1e-10);
        assert!((vel.v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_water_level() {
        let mut state = ShallowWaterState::new(1);
        state.h[0] = 3.0;
        state.z[0] = -5.0;

        assert_eq!(state.water_level(0), -2.0);
    }

    #[test]
    fn test_flux_operations() {
        let f1 = Flux::new(1.0, 2.0, 3.0);
        let f2 = Flux::new(0.5, 1.0, 1.5);

        let sum = f1 + f2;
        assert_eq!(sum.mass, 1.5);
        assert_eq!(sum.mom_x, 3.0);
        assert_eq!(sum.mom_y, 4.5);

        let scaled = f1 * 2.0;
        assert_eq!(scaled.mass, 2.0);
        assert_eq!(scaled.mom_x, 4.0);
        assert_eq!(scaled.mom_y, 6.0);

        let neg = -f1;
        assert_eq!(neg.mass, -1.0);
    }

    #[test]
    fn test_rhs_buffers() {
        let mut rhs = RhsBuffers::new(10);
        assert_eq!(rhs.dh_dt.len(), 10);

        rhs.add_flux(0, Flux::new(1.0, 2.0, 3.0), 0.5);
        assert_eq!(rhs.dh_dt[0], 0.5);
        assert_eq!(rhs.dhu_dt[0], 1.0);
        assert_eq!(rhs.dhv_dt[0], 1.5);
    }

    #[test]
    fn test_gradient_state() {
        let mut grad = GradientState::new(5);
        grad.set(0, DVec2::new(1.0, 2.0), DVec2::new(3.0, 4.0), DVec2::new(5.0, 6.0));

        let (gh, ghu, ghv) = grad.get(0);
        assert_eq!(gh, DVec2::new(1.0, 2.0));
        assert_eq!(ghu, DVec2::new(3.0, 4.0));
        assert_eq!(ghv, DVec2::new(5.0, 6.0));
    }

    #[test]
    fn test_state_linear_combine() {
        let mut result = ShallowWaterState::new(2);
        let mut a = ShallowWaterState::new(2);
        let mut b = ShallowWaterState::new(2);

        a.h[0] = 1.0;
        a.h[1] = 2.0;
        b.h[0] = 3.0;
        b.h[1] = 4.0;

        result.linear_combine(0.5, &a, 0.5, &b);

        assert_eq!(result.h[0], 2.0);
        assert_eq!(result.h[1], 3.0);
    }

    #[test]
    fn test_state_axpy() {
        let mut state = ShallowWaterState::new(2);
        let other = ShallowWaterState::cold_start(10.0, &[0.0, 5.0]);

        state.h[0] = 1.0;
        state.h[1] = 2.0;

        state.axpy(0.5, 0.5, &other);

        assert_eq!(state.h[0], 5.5); // 0.5 * 1.0 + 0.5 * 10.0
        assert_eq!(state.h[1], 3.5); // 0.5 * 2.0 + 0.5 * 5.0
    }

    #[test]
    fn test_conserved_state() {
        let state = ConservedState::new(2.0, 4.0, 6.0);
        let params = NumericalParams::default();
        let vel = state.velocity(&params);

        assert!((vel.u - 2.0).abs() < 1e-10);
        assert!((vel.v - 3.0).abs() < 1e-10);

        let from_prim = ConservedState::from_primitive(2.0, 2.0, 3.0);
        assert_eq!(from_prim, state);
    }
}

// ============================================================
// 娉涘瀷娴呮按鐘舵€?(Backend 鎶借薄)
// ============================================================

use crate::core::{Backend, CpuBackend, DeviceBuffer, Scalar};

/// 娉涘瀷娴呮按鐘舵€?
///
/// 浣跨敤 Backend trait 鎶借薄瀛樺偍锛屾敮鎸?CPU/GPU 鍚庣銆?
/// 姘歌繙鍙湁4涓牳蹇冨瓧娈碉細h, hu, hv, z銆?
///
/// # 璁捐璇存槑
///
/// 鐘舵€佹寔鏈?Backend 瀹炰緥鐨勫厠闅嗭紝鐢ㄤ簬鍚庣画鐨勭紦鍐插尯鎿嶄綔銆?
/// 鐢变簬 CpuBackend 鏄浂澶у皬绫诲瀷锛岃繖涓嶄細甯︽潵棰濆寮€閿€銆?
#[derive(Debug, Clone)]
pub struct ShallowWaterStateGeneric<B: Backend> {
    /// 鍗曞厓鏁伴噺
    n_cells: usize,
    /// 姘存繁 [m]
    pub h: B::Buffer<B::Scalar>,
    /// x 鏂瑰悜鍔ㄩ噺 [m虏/s]
    pub hu: B::Buffer<B::Scalar>,
    /// y 鏂瑰悜鍔ㄩ噺 [m虏/s]
    pub hv: B::Buffer<B::Scalar>,
    /// 搴曞簥楂樼▼ [m]
    pub z: B::Buffer<B::Scalar>,
    /// 鍚庣瀹炰緥
    backend: B,
}

impl<B: Backend> ShallowWaterStateGeneric<B> {
    /// 浣跨敤鍚庣瀹炰緥鍒涘缓鏂扮姸鎬?
    pub fn new_with_backend(backend: B, n_cells: usize) -> Self {
        Self {
            n_cells,
            h: backend.alloc(n_cells),
            hu: backend.alloc(n_cells),
            hv: backend.alloc(n_cells),
            z: backend.alloc(n_cells),
            backend,
        }
    }
    
    /// 鍗曞厓鏁伴噺
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
    
    /// 鑾峰彇鍚庣寮曠敤
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 閲嶇疆涓洪浂
    pub fn reset(&mut self) {
        self.h.fill(<B::Scalar as Scalar>::from_f64(0.0));
        self.hu.fill(<B::Scalar as Scalar>::from_f64(0.0));
        self.hv.fill(<B::Scalar as Scalar>::from_f64(0.0));
    }
    
    /// 楠岃瘉鐘舵€佹湁鏁堟€?
    pub fn is_valid(&self) -> bool {
        if let Some(h) = self.h.as_slice() {
            h.iter().all(|&x| x.to_f64().is_finite() && x.to_f64() >= 0.0)
        } else {
            // GPU 缂撳啿鍖洪渶瑕佸悓姝ユ鏌?
            true
        }
    }
}

/// CPU f64 鍚庣鐨勪究鎹锋柟娉?
impl ShallowWaterStateGeneric<CpuBackend<f64>> {
    /// 浣跨敤榛樿 CPU f64 鍚庣鍒涘缓锛堝悜鍚庡吋瀹癸級
    pub fn new(n_cells: usize) -> Self {
        Self::new_with_backend(CpuBackend::<f64>::new(), n_cells)
    }
    
}

/// CPU f32 鍚庣鐨勪究鎹锋柟娉?
impl ShallowWaterStateGeneric<CpuBackend<f32>> {
    /// 浣跨敤 CPU f32 鍚庣鍒涘缓
    pub fn new_f32(n_cells: usize) -> Self {
        Self::new_with_backend(CpuBackend::<f32>::new(), n_cells)
    }
}

/// 绫诲瀷鍒悕锛氶粯璁ゅ悗绔殑鐘舵€?
pub type ShallowWaterStateDefault = ShallowWaterStateGeneric<CpuBackend<f64>>;

// ============================================================
// 娉涘瀷鐘舵€佺殑缁熻璁＄畻
// ============================================================

/// 鐘舵€佺粺璁′俊鎭?
#[derive(Debug, Clone, Copy, Default)]
pub struct StateStatisticsData<S> {
    /// 鏈€澶ф按娣?
    pub h_max: S,
    /// 鏈€灏忔按娣憋紙闈為浂锛?
    pub h_min: S,
    /// 骞冲潎姘存繁
    pub h_mean: S,
    /// 鏈€澶ч€熷害
    pub velocity_max: S,
    /// 姘翠綋鎬讳綋绉?
    pub total_volume: S,
    /// 婀垮崟鍏冩暟閲?
    pub wet_cells: usize,
}

impl<B: Backend> ShallowWaterStateGeneric<B> {
    /// 璁＄畻鐘舵€佺粺璁′俊鎭紙浠?CPU 鍚庣鏈夋晥锛?
    pub fn compute_statistics(&self, cell_areas: &[B::Scalar], h_dry: B::Scalar) -> Option<StateStatisticsData<B::Scalar>> {
        let h_slice = self.h.as_slice()?;
        let hu_slice = self.hu.as_slice()?;
        let hv_slice = self.hv.as_slice()?;
        
        let mut stats = StateStatisticsData {
            h_max: <B::Scalar as Scalar>::from_f64(0.0),
            h_min: <B::Scalar as Scalar>::from_f64(f64::MAX),
            h_mean: <B::Scalar as Scalar>::from_f64(0.0),
            velocity_max: <B::Scalar as Scalar>::from_f64(0.0),
            total_volume: <B::Scalar as Scalar>::from_f64(0.0),
            wet_cells: 0,
        };
        
        for i in 0..self.n_cells {
            let h = h_slice[i];
            
            if h > h_dry {
                stats.wet_cells += 1;
                stats.h_max = Float::max(stats.h_max, h);
                stats.h_min = Float::min(stats.h_min, h);
                
                // 璁＄畻閫熷害
                let u = hu_slice[i] / h;
                let v = hv_slice[i] / h;
                let speed = Float::sqrt(u * u + v * v);
                stats.velocity_max = Float::max(stats.velocity_max, speed);
                
                // 绱姞浣撶Н
                if i < cell_areas.len() {
                    stats.total_volume = stats.total_volume + h * cell_areas[i];
                }
            }
        }
        
        if stats.wet_cells > 0 {
            stats.h_mean = stats.total_volume / <B::Scalar as Scalar>::from_f64(stats.wet_cells as f64);
        }
        
        Some(stats)
    }
    
    /// 澶嶅埗鐘舵€佹暟鎹埌鍙︿竴涓姸鎬?
    pub fn copy_to(&self, other: &mut Self) {
        debug_assert_eq!(self.n_cells, other.n_cells, "状态复制: 单元数量不匹配");
        
        // CPU 鍚庣鐩存帴澶嶅埗
        if let (Some(src_h), Some(dst_h)) = (self.h.as_slice(), other.h.as_slice_mut()) {
            dst_h.copy_from_slice(src_h);
        }
        if let (Some(src_hu), Some(dst_hu)) = (self.hu.as_slice(), other.hu.as_slice_mut()) {
            dst_hu.copy_from_slice(src_hu);
        }
        if let (Some(src_hv), Some(dst_hv)) = (self.hv.as_slice(), other.hv.as_slice_mut()) {
            dst_hv.copy_from_slice(src_hv);
        }
        if let (Some(src_z), Some(dst_z)) = (self.z.as_slice(), other.z.as_slice_mut()) {
            dst_z.copy_from_slice(src_z);
        }
    }
}

