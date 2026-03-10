// crates/mh_physics/src/tracer/transport.rs

//! 绀鸿釜鍓傝緭杩愭眰瑙ｅ櫒
//!
//! 鏈ā鍧楁彁渚涚ず韪墏瀵规祦-鎵╂暎鏂圭▼鐨勬眰瑙ｅ姛鑳斤細
//! - TracerTransportSolver: 涓绘眰瑙ｅ櫒
//! - TracerAdvectionScheme: 瀵规祦鏍煎紡
//! - TracerDiffusionConfig: 鎵╂暎閰嶇疆
//!
//! # 鍩烘湰鏂圭▼
//!
//! 绀鸿釜鍓傝緭杩愭柟绋嬶紙浜岀淮娣卞害骞冲潎锛夛細
//!
//! $$\frac{\partial (hC)}{\partial t} + \nabla \cdot (hC\vec{u}) = \nabla \cdot (hK\nabla C) + S$$
//!
//! 鍏朵腑锛?
//! - $C$: 绀鸿釜鍓傛祿搴?
//! - $h$: 姘存繁
//! - $\vec{u}$: 娣卞害骞冲潎娴侀€?
//! - $K$: 鎵╂暎绯绘暟寮犻噺
//! - $S$: 婧愭眹椤?
//!
//! # 杩佺Щ璇存槑
//!
//! 浠?history_src/tracer/tracer_transport.rs 杩佺Щ锛屾敼杩涳細
//! - 浣跨敤 trait 鎶借薄缃戞牸璁块棶
//! - 鏀寔澶氱瀵规祦鏍煎紡
//! - 涓庢柊鏋舵瀯鐨勬椂闂寸Н鍒嗗櫒闆嗘垚

use glam::DVec2;
use serde::{Deserialize, Serialize};
use super::state::{TracerField, TracerState};

// ============================================================
// 瀵规祦鏍煎紡
// ============================================================

/// 瀵规祦鏍煎紡绫诲瀷
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TracerAdvectionScheme {
    /// 涓€闃惰繋椋庢牸寮?
    ///
    /// 绠€鍗曠ǔ瀹氾紝浣嗘暟鍊兼墿鏁ｈ緝澶с€?
    #[default]
    FirstOrderUpwind,

    /// 浜岄樁 Lax-Wendroff 鏍煎紡
    ///
    /// 绮惧害鏇撮珮锛屼絾鍙兘浜х敓鎸崱銆?
    LaxWendroff,

    /// 浜岄樁 TVD 鏍煎紡锛圡inMod 闄愬埗鍣級
    ///
    /// 骞宠　绮惧害鍜岀ǔ瀹氭€с€?
    TvdMinmod,

    /// 浜岄樁 TVD 鏍煎紡锛圫uperbee 闄愬埗鍣級
    ///
    /// 鏇村皷閿愮殑闂存柇锛屼絾鍙兘杩囧害鍘嬬缉銆?
    TvdSuperbee,

    /// 浜岄樁 TVD 鏍煎紡锛圴an Leer 闄愬埗鍣級
    ///
    /// 骞虫粦鐨勯檺鍒讹紝閫傜敤浜庝竴鑸儏鍐点€?
    TvdVanLeer,
}

impl TracerAdvectionScheme {
    /// 鑾峰彇鏍煎紡鍚嶇О
    pub fn name(&self) -> &'static str {
        match self {
            Self::FirstOrderUpwind => "First-Order Upwind",
            Self::LaxWendroff => "Lax-Wendroff",
            Self::TvdMinmod => "TVD (MinMod)",
            Self::TvdSuperbee => "TVD (Superbee)",
            Self::TvdVanLeer => "TVD (Van Leer)",
        }
    }

    /// 鏄惁闇€瑕佹搴︿俊鎭?
    pub fn requires_gradient(&self) -> bool {
        matches!(
            self,
            Self::LaxWendroff | Self::TvdMinmod | Self::TvdSuperbee | Self::TvdVanLeer
        )
    }
}

// ============================================================
// 鎵╂暎閰嶇疆
// ============================================================

/// 鎵╂暎璁＄畻閰嶇疆
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerDiffusionConfig {
    /// 鏄惁鍚敤鎵╂暎
    pub enabled: bool,

    /// 姘村钩鎵╂暎绯绘暟 [m虏/s]
    ///
    /// 鍙互鏄父鏁版垨鍩轰簬缃戞牸灏哄害鐨?Smagorinsky 鍏紡銆?
    pub horizontal_diffusivity: f64,

    /// Smagorinsky 绯绘暟锛堢敤浜庤嚜閫傚簲鎵╂暎锛?
    ///
    /// K = C_s * dx虏 * |S|锛屽叾涓?|S| 鏄簲鍙樼巼銆?
    pub smagorinsky_coefficient: f64,

    /// 鏄惁浣跨敤 Smagorinsky 妯″瀷
    pub use_smagorinsky: bool,
}

impl Default for TracerDiffusionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            horizontal_diffusivity: 10.0, // 鍏稿瀷鍊?1-100 m虏/s
            smagorinsky_coefficient: 0.2,
            use_smagorinsky: false,
        }
    }
}

impl TracerDiffusionConfig {
    /// 浠呬娇鐢ㄥ父鏁版墿鏁?
    pub fn constant(diffusivity: f64) -> Self {
        Self {
            enabled: true,
            horizontal_diffusivity: diffusivity,
            use_smagorinsky: false,
            ..Default::default()
        }
    }

    /// 浣跨敤 Smagorinsky 妯″瀷
    pub fn smagorinsky(coefficient: f64) -> Self {
        Self {
            enabled: true,
            horizontal_diffusivity: 0.0,
            smagorinsky_coefficient: coefficient,
            use_smagorinsky: true,
        }
    }

    /// 绂佺敤鎵╂暎
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

// ============================================================
// 姹傝В鍣ㄩ厤缃?
// ============================================================

/// 绀鸿釜鍓傝緭杩愭眰瑙ｅ櫒閰嶇疆
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerTransportConfig {
    /// 瀵规祦鏍煎紡
    pub advection_scheme: TracerAdvectionScheme,

    /// 鎵╂暎閰嶇疆
    pub diffusion: TracerDiffusionConfig,

    /// 鏈€灏忔按娣遍槇鍊?[m]
    ///
    /// 姘存繁灏忎簬姝ゅ€肩殑鍗曞厓涓嶈绠楃ず韪墏銆?
    pub h_min: f64,

    /// 娴撳害闄愬埗鍣?
    ///
    /// 闃叉浜х敓璐熸祿搴︽垨瓒呰繃鐗╃悊鑼冨洿鐨勬祿搴︺€?
    pub enable_clipping: bool,

    /// 鏈€灏忔祿搴?
    pub c_min: f64,

    /// 鏈€澶ф祿搴︼紙鍙€夛級
    pub c_max: Option<f64>,
}

impl Default for TracerTransportConfig {
    fn default() -> Self {
        Self {
            advection_scheme: TracerAdvectionScheme::default(),
            diffusion: TracerDiffusionConfig::default(),
            h_min: 1e-6,
            enable_clipping: true,
            c_min: 0.0,
            c_max: None,
        }
    }
}

// ============================================================
// 闈㈤€氶噺鏁版嵁
// ============================================================

/// 闈㈢殑娴佸姩鏁版嵁锛堢敤浜庤绠楃ず韪墏閫氶噺锛?
#[derive(Debug, Clone, Copy)]
pub struct FaceFlowData {
    /// 闈㈢储寮?
    pub face_id: usize,
    /// 宸︿晶鍗曞厓绱㈠紩
    pub left_cell: usize,
    /// 鍙充晶鍗曞厓绱㈠紩锛堣竟鐣岄潰涓?None锛?
    pub right_cell: Option<usize>,
    /// 闈㈡硶鍚戦噺锛堜粠宸﹀埌鍙筹級
    pub normal: DVec2,
    /// 闈㈤暱搴?[m]
    pub length: f64,
    /// 闈笂鐨勬硶鍚戞祦閫?[m/s]
    pub un: f64,
    /// 闈笂鐨勬按娣?[m]
    pub h_face: f64,
}

/// 绀鸿釜鍓傞潰閫氶噺
#[derive(Debug, Clone, Copy, Default)]
pub struct TracerFaceFlux {
    /// 瀵规祦閫氶噺 [鍗曚綅/s]
    pub advective: f64,
    /// 鎵╂暎閫氶噺 [鍗曚綅/s]
    pub diffusive: f64,
}

impl TracerFaceFlux {
    /// 鎬婚€氶噺
    pub fn total(&self) -> f64 {
        self.advective + self.diffusive
    }
}

// ============================================================
// 绀鸿釜鍓傝緭杩愭眰瑙ｅ櫒
// ============================================================

/// 绀鸿釜鍓傝緭杩愭眰瑙ｅ櫒
///
/// 璐熻矗璁＄畻绀鸿釜鍓傜殑瀵规祦鍜屾墿鏁ｉ€氶噺锛屽苟鏇存柊娴撳害鍦恒€?
///
/// # 浣跨敤娴佺▼
///
/// 1. 鍒涘缓姹傝В鍣ㄥ疄渚?
/// 2. 鍑嗗闈㈡祦鍔ㄦ暟鎹紙浠庢按鍔ㄥ姏姹傝В鍣ㄨ幏鍙栵級
/// 3. 璁＄畻閫氶噺骞舵洿鏂?RHS
/// 4. 浣跨敤鏃堕棿绉垎鍣ㄦ洿鏂板畧鎭掗噺
/// 5. 浠庡畧鎭掗噺鍙嶇畻娴撳害
///
/// # 绀轰緥
///
/// ```ignore
/// use mh_physics::tracer::{TracerTransportSolver, TracerTransportConfig};
///
/// let solver = TracerTransportSolver::new(TracerTransportConfig::default());
/// ```
pub struct TracerTransportSolver {
    config: TracerTransportConfig,
    
    /// 涓存椂宸ヤ綔鏁扮粍锛氶潰閫氶噺
    face_fluxes: Vec<TracerFaceFlux>,
}

impl TracerTransportSolver {
    /// 鍒涘缓鏂扮殑姹傝В鍣?
    pub fn new(config: TracerTransportConfig) -> Self {
        Self {
            config,
            face_fluxes: Vec::new(),
        }
    }

    /// 鑾峰彇閰嶇疆寮曠敤
    pub fn config(&self) -> &TracerTransportConfig {
        &self.config
    }

    /// 璁剧疆閰嶇疆
    pub fn set_config(&mut self, config: TracerTransportConfig) {
        self.config = config;
    }

    /// 璁＄畻鍗曚釜闈㈢殑瀵规祦閫氶噺锛堜竴闃惰繋椋庯級
    ///
    /// # 鍙傛暟
    /// - `c_left`: 宸︿晶鍗曞厓娴撳害
    /// - `c_right`: 鍙充晶鍗曞厓娴撳害
    /// - `h_face`: 闈笂姘存繁
    /// - `un`: 闈㈡硶鍚戦€熷害锛堟鍊间粠宸﹀埌鍙筹級
    /// - `face_length`: 闈㈤暱搴?
    ///
    /// # 杩斿洖
    /// 瀵规祦閫氶噺锛堟鍊艰〃绀轰粠宸﹀埌鍙宠緭閫侊級
    pub fn compute_advective_flux_upwind(
        &self,
        c_left: f64,
        c_right: f64,
        h_face: f64,
        un: f64,
        face_length: f64,
    ) -> f64 {
        // 杩庨閫夋嫨
        let c_upwind = if un >= 0.0 { c_left } else { c_right };
        h_face * un * c_upwind * face_length
    }

    /// 璁＄畻鍗曚釜闈㈢殑鎵╂暎閫氶噺
    ///
    /// # 鍙傛暟
    /// - `c_left`: 宸︿晶鍗曞厓娴撳害
    /// - `c_right`: 鍙充晶鍗曞厓娴撳害
    /// - `h_face`: 闈笂姘存繁
    /// - `distance`: 鍗曞厓涓績闂磋窛
    /// - `face_length`: 闈㈤暱搴?
    /// - `diffusivity`: 鎵╂暎绯绘暟 [m虏/s]
    ///
    /// # 杩斿洖
    /// 鎵╂暎閫氶噺锛堟鍊艰〃绀轰粠宸﹀埌鍙宠緭閫侊級
    pub fn compute_diffusive_flux(
        &self,
        c_left: f64,
        c_right: f64,
        h_face: f64,
        distance: f64,
        face_length: f64,
        diffusivity: f64,
    ) -> f64 {
        if !self.config.diffusion.enabled || diffusivity <= 0.0 {
            return 0.0;
        }

        // 鎵╂暎閫氶噺: F = -h * K * dC/dx
        let dc_dx = (c_right - c_left) / distance.max(1e-10);
        -h_face * diffusivity * dc_dx * face_length
    }

    /// 璁＄畻鎵€鏈夐潰鐨勯€氶噺骞剁疮鍔犲埌 RHS
    ///
    /// # 鍙傛暟
    /// - `field`: 绀鸿釜鍓傚満
    /// - `flow_data`: 闈㈡祦鍔ㄦ暟鎹?
    /// - `cell_volumes`: 鍗曞厓浣撶Н
    /// - `face_distances`: 闈㈠搴旂殑鍗曞厓涓績闂磋窛
    pub fn compute_rhs(
        &mut self,
        field: &mut TracerField,
        flow_data: &[FaceFlowData],
        cell_volumes: &[f64],
        face_distances: &[f64],
    ) {
        field.clear_rhs();

        let diffusivity = if self.config.diffusion.use_smagorinsky {
            // TODO: 璁＄畻 Smagorinsky 鎵╂暎绯绘暟
            self.config.diffusion.horizontal_diffusivity
        } else {
            self.config.diffusion.horizontal_diffusivity
        };

        // 纭繚宸ヤ綔鏁扮粍澶у皬瓒冲
        if self.face_fluxes.len() < flow_data.len() {
            self.face_fluxes.resize(flow_data.len(), TracerFaceFlux::default());
        }

        // 璁＄畻鎵€鏈夐潰鐨勯€氶噺
        for (i, face) in flow_data.iter().enumerate() {
            let c_left = field.concentration(face.left_cell);
            let c_right = face.right_cell
                .map(|idx| field.concentration(idx))
                .unwrap_or(c_left); // 杈圭晫闈娇鐢ㄥ乏渚у€?

            // 瀵规祦閫氶噺
            let advective = self.compute_advective_flux_upwind(
                c_left,
                c_right,
                face.h_face,
                face.un,
                face.length,
            );

            // 鎵╂暎閫氶噺
            let diffusive = if face.right_cell.is_some() {
                self.compute_diffusive_flux(
                    c_left,
                    c_right,
                    face.h_face,
                    face_distances[i],
                    face.length,
                    diffusivity,
                )
            } else {
                0.0 // 杈圭晫闈㈡棤鎵╂暎
            };

            self.face_fluxes[i] = TracerFaceFlux { advective, diffusive };

            // 绱姞鍒板崟鍏?RHS
            let flux = advective + diffusive;

            // 宸︿晶鍗曞厓锛氶€氶噺娴佸嚭涓鸿礋
            let vol_left = cell_volumes[face.left_cell];
            if vol_left > 0.0 {
                field.add_rhs(face.left_cell, -flux / vol_left);
            }

            // 鍙充晶鍗曞厓锛堝鏋滃瓨鍦級锛氶€氶噺娴佸叆涓烘
            if let Some(right_cell) = face.right_cell {
                let vol_right = cell_volumes[right_cell];
                if vol_right > 0.0 {
                    field.add_rhs(right_cell, flux / vol_right);
                }
            }
        }
    }

    /// 鏃堕棿姝ヨ繘鏇存柊
    ///
    /// 浣跨敤鏄惧紡娆ф媺鏍煎紡鏇存柊瀹堟亽閲忋€?
    ///
    /// # 鍙傛暟
    /// - `field`: 绀鸿釜鍓傚満
    /// - `dt`: 鏃堕棿姝ラ暱 [s]
    pub fn update_forward_euler(&self, field: &mut TracerField, dt: f64) {
        field.apply_euler_update(dt);
    }

    /// 搴旂敤娴撳害闄愬埗
    pub fn apply_clipping(&self, field: &mut TracerField) {
        if self.config.enable_clipping {
            field.clamp_concentration(self.config.c_min, self.config.c_max);
        }
    }

    /// 瀹屾暣鐨勫崟姝ユ洿鏂版祦绋?
    ///
    /// # 鍙傛暟
    /// - `field`: 绀鸿釜鍓傚満
    /// - `flow_data`: 闈㈡祦鍔ㄦ暟鎹?
    /// - `cell_volumes`: 鍗曞厓浣撶Н
    /// - `face_distances`: 闈㈠搴旂殑鍗曞厓涓績闂磋窛
    /// - `water_depths`: 姘存繁鏁扮粍锛堢敤浜庢洿鏂版祿搴︼級
    /// - `dt`: 鏃堕棿姝ラ暱
    pub fn step(
        &mut self,
        field: &mut TracerField,
        flow_data: &[FaceFlowData],
        cell_volumes: &[f64],
        face_distances: &[f64],
        water_depths: &[f64],
        dt: f64,
    ) {
        // 1. 璁＄畻 RHS
        self.compute_rhs(field, flow_data, cell_volumes, face_distances);

        // 2. 鏃堕棿姝ヨ繘
        self.update_forward_euler(field, dt);

        // 3. 浠庡畧鎭掗噺鏇存柊娴撳害
        field.update_concentration_from_conserved(water_depths, self.config.h_min);

        // 4. 搴旂敤闄愬埗
        self.apply_clipping(field);
    }

    /// 璁＄畻绀鸿釜鍓傜殑 CFL 闄愬埗鏃堕棿姝?
    ///
    /// # 鍙傛暟
    /// - `max_velocity`: 鏈€澶ф祦閫?[m/s]
    /// - `min_cell_size`: 鏈€灏忓崟鍏冨昂瀵?[m]
    /// - `cfl_number`: CFL 鏁帮紙榛樿 0.5锛?
    ///
    /// # 杩斿洖
    /// 寤鸿鐨勬渶澶ф椂闂存闀?[s]
    pub fn compute_dt_limit(
        &self,
        max_velocity: f64,
        min_cell_size: f64,
        cfl_number: f64,
    ) -> f64 {
        let dx = min_cell_size.max(1e-10);

        // 瀵规祦闄愬埗
        let dt_advection = if max_velocity > 1e-10 {
            cfl_number * dx / max_velocity
        } else {
            f64::MAX
        };

        // 鎵╂暎闄愬埗锛堝鏋滃惎鐢級
        let dt_diffusion = if self.config.diffusion.enabled {
            let k = self.config.diffusion.horizontal_diffusivity;
            if k > 1e-10 {
                0.5 * cfl_number * dx * dx / k
            } else {
                f64::MAX
            }
        } else {
            f64::MAX
        };

        dt_advection.min(dt_diffusion)
    }
}

impl Default for TracerTransportSolver {
    fn default() -> Self {
        Self::new(TracerTransportConfig::default())
    }
}

// ============================================================
// 澶氱ず韪墏姹傝В鍣?
// ============================================================

/// 澶氱ず韪墏杈撹繍姹傝В鍣?
///
/// 鍖呰 TracerTransportSolver锛屾敮鎸佸悓鏃跺鐞嗗涓ず韪墏銆?
pub struct MultiTracerSolver {
    /// 鍗曠ず韪墏姹傝В鍣?
    solver: TracerTransportSolver,
}

impl MultiTracerSolver {
    /// 鍒涘缓鏂扮殑澶氱ず韪墏姹傝В鍣?
    pub fn new(config: TracerTransportConfig) -> Self {
        Self {
            solver: TracerTransportSolver::new(config),
        }
    }

    /// 鏇存柊鎵€鏈夌ず韪墏
    pub fn step_all(
        &mut self,
        state: &mut TracerState,
        flow_data: &[FaceFlowData],
        cell_volumes: &[f64],
        face_distances: &[f64],
        water_depths: &[f64],
        dt: f64,
    ) {
        for (_, field) in state.iter_mut() {
            self.solver.step(field, flow_data, cell_volumes, face_distances, water_depths, dt);
        }
    }

    /// 鑾峰彇鍐呴儴姹傝В鍣?
    pub fn solver(&self) -> &TracerTransportSolver {
        &self.solver
    }

    /// 鑾峰彇鍐呴儴姹傝В鍣紙鍙彉锛?
    pub fn solver_mut(&mut self) -> &mut TracerTransportSolver {
        &mut self.solver
    }
}

// ============================================================
// 娴嬭瘯
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::state::TracerType;
    use super::super::state::TracerProperties;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_advection_scheme() {
        let scheme = TracerAdvectionScheme::FirstOrderUpwind;
        assert!(!scheme.requires_gradient());

        let scheme = TracerAdvectionScheme::TvdMinmod;
        assert!(scheme.requires_gradient());
    }

    #[test]
    fn test_diffusion_config() {
        let config = TracerDiffusionConfig::constant(50.0);
        assert!(config.enabled);
        assert!(approx_eq(config.horizontal_diffusivity, 50.0));
        assert!(!config.use_smagorinsky);

        let config = TracerDiffusionConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_upwind_flux() {
        let solver = TracerTransportSolver::default();

        // 浠庡乏鍒板彸娴佸姩
        let flux = solver.compute_advective_flux_upwind(10.0, 20.0, 1.0, 1.0, 1.0);
        assert!(approx_eq(flux, 10.0)); // 浣跨敤宸︿晶娴撳害

        // 浠庡彸鍒板乏娴佸姩
        let flux = solver.compute_advective_flux_upwind(10.0, 20.0, 1.0, -1.0, 1.0);
        assert!(approx_eq(flux, -20.0)); // 浣跨敤鍙充晶娴撳害
    }

    #[test]
    fn test_diffusive_flux() {
        let solver = TracerTransportSolver::new(TracerTransportConfig {
            diffusion: TracerDiffusionConfig::constant(10.0),
            ..Default::default()
        });

        // 娴撳害姊害锛氫粠宸?10)鍒板彸(20)锛屾墿鏁ｅ簲璇ヤ粠楂樺埌浣?
        let flux = solver.compute_diffusive_flux(10.0, 20.0, 1.0, 1.0, 1.0, 10.0);
        // F = -h * K * dC/dx = -1 * 10 * (20-10)/1 = -100
        assert!(approx_eq(flux, -100.0));
    }

    #[test]
    fn test_dt_limit() {
        let solver = TracerTransportSolver::new(TracerTransportConfig {
            diffusion: TracerDiffusionConfig::constant(10.0),
            ..Default::default()
        });

        let dt = solver.compute_dt_limit(1.0, 10.0, 0.5);
        // 瀵规祦闄愬埗: 0.5 * 10 / 1 = 5
        // 鎵╂暎闄愬埗: 0.5 * 0.5 * 100 / 10 = 2.5
        assert!(approx_eq(dt, 2.5));
    }

    #[test]
    fn test_single_step() {
        let mut solver = TracerTransportSolver::default();
        let props = TracerProperties::salinity().with_background(0.0);
        let mut field = TracerField::from_concentration(
            props,
            vec![10.0, 5.0, 0.0], // 娴撳害姊害
        );

        // 鍒濆鍖栧畧鎭掗噺
        let depths = vec![1.0, 1.0, 1.0];
        field.update_conserved_from_depth(&depths);

        // 绠€鍗曠殑涓ら潰娴佸姩鏁版嵁
        let flow_data = vec![
            FaceFlowData {
                face_id: 0,
                left_cell: 0,
                right_cell: Some(1),
                normal: DVec2::new(1.0, 0.0),
                length: 1.0,
                un: 1.0,  // 浠庡乏鍒板彸
                h_face: 1.0,
            },
            FaceFlowData {
                face_id: 1,
                left_cell: 1,
                right_cell: Some(2),
                normal: DVec2::new(1.0, 0.0),
                length: 1.0,
                un: 1.0,
                h_face: 1.0,
            },
        ];

        let volumes = vec![1.0, 1.0, 1.0];
        let distances = vec![1.0, 1.0];

        // 鎵ц涓€姝?
        solver.step(&mut field, &flow_data, &volumes, &distances, &depths, 0.1);

        // 娴撳害搴旇鍙樺寲浜?
        // 鐢变簬杩庨鏍煎紡锛屾祿搴︿細鍚戝彸浼犺緭
        assert!(field.concentration(1) > 5.0); // 浠庡崟鍏?鑾峰緱璐ㄩ噺
    }

    #[test]
    fn test_multi_tracer_solver() {
        let mut state = TracerState::new(10);
        state.add_tracer(TracerProperties::salinity()).unwrap();
        state.add_tracer(TracerProperties::temperature()).unwrap();

        let _solver = MultiTracerSolver::new(TracerTransportConfig::default());

        // 纭繚鍙互璁块棶涓や釜绀鸿釜鍓?
        assert!(state.get(TracerType::Salinity).is_some());
        assert!(state.get(TracerType::Temperature).is_some());
    }
}
