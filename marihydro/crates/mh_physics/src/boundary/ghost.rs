// crates/mh_physics/src/boundary/ghost.rs

//! 骞界伒鐘舵€佽绠楀櫒
//!
//! 鏈ā鍧楁彁渚涘熀浜庤竟鐣屾潯浠惰绠楀菇鐏靛崟鍏冪姸鎬佺殑鍔熻兘锛?
//! - GhostStateCalculator: 骞界伒鐘舵€佽绠楀櫒
//! - GhostMomentumMode: 鍔ㄩ噺闀滃儚妯″紡
//!
//! # 姒傚康璇存槑
//!
//! 骞界伒鍗曞厓鏄竴绉嶈竟鐣屽鐞嗘妧鏈細
//! 1. 鍦ㄨ竟鐣屽铏氭嫙涓€涓崟鍏冿紙骞界伒鍗曞厓锛?
//! 2. 鏍规嵁杈圭晫鏉′欢璁剧疆骞界伒鍗曞厓鐨勭姸鎬?
//! 3. 浣跨敤鍐呴儴鍗曞厓鍜屽菇鐏靛崟鍏冭繘琛岄€氶噺璁＄畻
//!
//! 杩欑鏂规硶鐨勪紭鐐癸細
//! - 缁熶竴鍐呴儴鍜岃竟鐣岀殑鏁板€兼牸寮?
//! - 鍙鐢ㄧ浉鍚岀殑閫氶噺璁＄畻鍑芥暟
//! - 瀹炵幇绠€鍗曪紝鏄撲簬骞惰鍖?
//!
//! # 杩佺Щ璇存槑
//!
//! 浠?history_src/domain/boundary/ghost.rs 杩佺Щ锛屾敼杩涳細
//! - 浣跨敤鏋氫妇鏇夸唬甯冨皵鍙傛暟
//! - 鏀寔鏇村杈圭晫绫诲瀷
//! - 涓?BoundaryManager 闆嗘垚

use glam::DVec2;

use super::types::{BoundaryKind, BoundaryParams, ExternalForcing};
use crate::state::ConservedState;

// ============================================================
// 鍔ㄩ噺闀滃儚妯″紡
// ============================================================

/// 鍔ㄩ噺闀滃儚妯″紡
///
/// 鎺у埗閫熷害鍒嗛噺濡備綍闀滃儚鍒板菇鐏靛崟鍏冦€?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GhostMomentumMode {
    /// 瀹屽叏鍙嶅皠锛氬垏鍚戜繚鎸侊紝娉曞悜鍙嶅悜
    ///
    /// 鐢ㄤ簬鏃犳粦绉诲浐澹佽竟鐣屻€?
    #[default]
    FullReflect,

    /// 鑷敱婊戠Щ锛氬垏鍚戜繚鎸侊紝娉曞悜鍙嶅悜浣嗗姩閲忓噺鍗?
    ///
    /// 鐢ㄤ簬鑷敱婊戠Щ杈圭晫銆?
    FreeSlip,

    /// 鏃犲弽灏勶細鐩存帴澶嶅埗
    ///
    /// 鐢ㄤ簬瀵圭О杈圭晫銆?
    NoReflect,

    /// 瀹屽叏鎶垫秷锛氬垏鍚戝拰娉曞悜閮藉弽鍚?
    ///
    /// 鐢ㄤ簬鏃犳粦绉昏竟鐣岋紙榛忔€ф晥鏋滐級銆?
    FullCancel,
}

impl GhostMomentumMode {
    /// 浠庤竟鐣岀被鍨嬫帹鏂姩閲忔ā寮?
    pub fn from_boundary_kind(kind: BoundaryKind) -> Self {
        match kind {
            BoundaryKind::Wall => Self::FullReflect,
            BoundaryKind::Symmetry => Self::NoReflect,
            BoundaryKind::OpenSea | BoundaryKind::Outflow => Self::NoReflect,
            BoundaryKind::RiverInflow => Self::NoReflect,
            BoundaryKind::Periodic => Self::NoReflect,
        }
    }
}

// ============================================================
// 骞界伒鐘舵€佽绠楀櫒
// ============================================================

/// 骞界伒鐘舵€佽绠楀櫒
///
/// 璐熻矗鏍规嵁杈圭晫鏉′欢璁＄畻骞界伒鍗曞厓鐨勭姸鎬併€?
///
/// # 浣跨敤鏂瑰紡
///
/// ```ignore
/// use mh_physics::boundary::{GhostStateCalculator, BoundaryKind, BoundaryParams};
/// use mh_physics::state::ConservedState;
/// use glam::DVec2;
///
/// let calculator = GhostStateCalculator::new(BoundaryParams::default());
/// let interior = ConservedState::from_primitive(1.0, 0.5, 0.0);
/// let normal = DVec2::new(1.0, 0.0);
/// let z_bed = 0.0; // 搴曞簥楂樼▼
///
/// let ghost = calculator.compute_ghost(
///     interior,
///     BoundaryKind::Wall,
///     normal,
///     None,
///     z_bed,
/// );
/// ```
pub struct GhostStateCalculator {
    params: BoundaryParams,
}

impl GhostStateCalculator {
    /// 鍒涘缓骞界伒鐘舵€佽绠楀櫒
    pub fn new(params: BoundaryParams) -> Self {
        Self { params }
    }

    /// 浠庢暟鍊煎弬鏁板垱寤?
    pub fn from_numerical_params(params: &crate::types::NumericalParams) -> Self {
        Self::new(BoundaryParams::from_numerical_params(params))
    }

    /// 璁＄畻骞界伒鍗曞厓鐘舵€?
    ///
    /// # 鍙傛暟
    /// - `interior`: 鍐呴儴鍗曞厓鐘舵€?
    /// - `kind`: 杈圭晫绫诲瀷
    /// - `normal`: 闈㈠娉曞悜閲忥紙鍗曚綅鍚戦噺锛?
    /// - `external`: 澶栭儴寮鸿揩鏁版嵁锛堢敤浜庡紑杈圭晫锛?
    /// - `z_bed`: 鍐呴儴鍗曞厓搴曞簥楂樼▼锛堢敤浜庤绠楁按浣嶏級
    ///
    /// # 杩斿洖
    /// 骞界伒鍗曞厓鐨勫畧鎭掗噺鐘舵€?
    pub fn compute_ghost(
        &self,
        interior: ConservedState,
        kind: BoundaryKind,
        normal: DVec2,
        external: Option<&ExternalForcing>,
        z_bed: f64,
    ) -> ConservedState {
        match kind {
            BoundaryKind::Wall => self.compute_wall_ghost(interior, normal),
            BoundaryKind::Symmetry => self.compute_symmetry_ghost(interior, normal),
            BoundaryKind::OpenSea => {
                self.compute_open_sea_ghost(interior, normal, external.unwrap_or(&ExternalForcing::ZERO), z_bed)
            }
            BoundaryKind::Outflow => self.compute_outflow_ghost(interior),
            BoundaryKind::RiverInflow => {
                self.compute_inflow_ghost(interior, external.unwrap_or(&ExternalForcing::ZERO))
            }
            BoundaryKind::Periodic => {
                // 鍛ㄦ湡杈圭晫闇€瑕佺壒娈婂鐞嗭紝杩欓噷杩斿洖鍐呴儴鐘舵€佷綔涓哄崰浣?
                // 瀹為檯鍛ㄦ湡杈圭晫鍦ㄧ綉鏍艰繛鎺ラ樁娈靛鐞?
                interior
            }
        }
    }

    /// 璁＄畻鍥哄杈圭晫鐨勫菇鐏电姸鎬?
    ///
    /// 瀹炵幇鏃犵┛閫忔潯浠讹細娉曞悜閫熷害鍙嶅悜銆?
    fn compute_wall_ghost(&self, interior: ConservedState, normal: DVec2) -> ConservedState {
        let h = interior.h.max(self.params.h_min);

        // 璁＄畻閫熷害
        let u = interior.hu / h;
        let v = interior.hv / h;
        let velocity = DVec2::new(u, v);

        // 鍒嗚В涓烘硶鍚戝拰鍒囧悜鍒嗛噺
        let un = velocity.dot(normal);
        let ut = velocity - normal * un;

        // 骞界伒閫熷害锛氭硶鍚戝弽杞紝鍒囧悜淇濇寔
        let ghost_velocity = ut - normal * un;

        ConservedState {
            h,
            hu: h * ghost_velocity.x,
            hv: h * ghost_velocity.y,
        }
    }

    /// 璁＄畻瀵圭О杈圭晫鐨勫菇鐏电姸鎬?
    ///
    /// 涓庡浐澹佺被浼硷紝浣嗗彲鑳芥湁涓嶅悓鐨勫姩閲忓鐞嗐€?
    fn compute_symmetry_ghost(&self, interior: ConservedState, normal: DVec2) -> ConservedState {
        // 瀵圭О杈圭晫涓庡浐澹佺被浼硷紝娉曞悜閫熷害鍙嶅悜
        self.compute_wall_ghost(interior, normal)
    }

    /// 璁＄畻寮€娴疯竟鐣岀殑骞界伒鐘舵€?
    ///
    /// 浣跨敤 Flather 杈愬皠鏉′欢銆?
    /// 
    /// Flather 鏉′欢鍩轰簬鐗瑰緛鍒嗚В锛?
    /// un* = un_ext + (c/h)(畏_int - 畏_ext)
    /// 鍏朵腑 畏 = h + z_bed 鏄按浣?
    fn compute_open_sea_ghost(
        &self,
        interior: ConservedState,
        normal: DVec2,
        external: &ExternalForcing,
        z_bed: f64,
    ) -> ConservedState {
        let h_int = interior.h.max(self.params.h_min);
        let c = self.params.wave_speed(h_int);

        // 鍐呴儴閫熷害
        let u_int = interior.hu / h_int;
        let v_int = interior.hv / h_int;
        let velocity_int = DVec2::new(u_int, v_int);

        // 娉曞悜閫熷害
        let un_int = velocity_int.dot(normal);
        let un_ext = external.velocity.dot(normal);

        // Flather 鏉′欢淇娉曞悜閫熷害
        // 姝ｇ‘浣跨敤姘翠綅 畏 = h + z_bed
        let eta_int = h_int + z_bed;
        let eta_ext = external.eta.max(self.params.h_min);
        let eta_diff = eta_int - eta_ext;
        let un_ghost = un_ext - (c / h_int) * eta_diff;

        // 鍒囧悜閫熷害淇濇寔
        let ut = velocity_int - normal * un_int;
        let ghost_velocity = ut + normal * un_ghost;

        // 骞界伒姘存繁锛氫粠澶栭儴姘翠綅鍑忓幓搴曞簥楂樼▼
        // h_ghost = max(0, eta_ext - z_bed)
        let h_ghost = (external.eta - z_bed).max(self.params.h_min);

        ConservedState {
            h: h_ghost,
            hu: h_ghost * ghost_velocity.x,
            hv: h_ghost * ghost_velocity.y,
        }
    }

    /// 璁＄畻鍑烘祦杈圭晫鐨勫菇鐏电姸鎬?
    ///
    /// 闆舵搴﹀鎺細鐩存帴澶嶅埗鍐呴儴鐘舵€併€?
    fn compute_outflow_ghost(&self, interior: ConservedState) -> ConservedState {
        interior
    }

    /// 璁＄畻鍏ユ祦杈圭晫鐨勫菇鐏电姸鎬?
    ///
    /// 浣跨敤澶栭儴寮鸿揩鐨勯€熷害鍜屾按娣便€?
    fn compute_inflow_ghost(
        &self,
        _interior: ConservedState,
        external: &ExternalForcing,
    ) -> ConservedState {
        let h = external.eta.max(self.params.h_min);
        ConservedState {
            h,
            hu: h * external.velocity.x,
            hv: h * external.velocity.y,
        }
    }

    /// 浣跨敤鎸囧畾鐨勫姩閲忔ā寮忚绠楀菇鐏电姸鎬?
    ///
    /// 鏇寸伒娲荤殑鎺ュ彛锛屽厑璁歌嚜瀹氫箟鍔ㄩ噺澶勭悊鏂瑰紡銆?
    ///
    /// # 鍙傛暟
    /// - `interior`: 鍐呴儴鍗曞厓鐘舵€?
    /// - `normal`: 闈㈠娉曞悜閲?
    /// - `mode`: 鍔ㄩ噺闀滃儚妯″紡
    ///
    /// # 杩斿洖
    /// 骞界伒鍗曞厓鐘舵€?
    pub fn compute_ghost_with_mode(
        &self,
        interior: ConservedState,
        normal: DVec2,
        mode: GhostMomentumMode,
    ) -> ConservedState {
        let h = interior.h.max(self.params.h_min);
        let u = interior.hu / h;
        let v = interior.hv / h;
        let velocity = DVec2::new(u, v);

        let un = velocity.dot(normal);
        let ut = velocity - normal * un;

        let ghost_velocity = match mode {
            GhostMomentumMode::FullReflect => ut - normal * un,
            GhostMomentumMode::FreeSlip => ut - normal * (un * 0.5),
            GhostMomentumMode::NoReflect => velocity,
            GhostMomentumMode::FullCancel => -velocity,
        };

        ConservedState {
            h,
            hu: h * ghost_velocity.x,
            hv: h * ghost_velocity.y,
        }
    }

    /// 鎵归噺璁＄畻骞界伒鐘舵€?
    ///
    /// 瀵规€ц兘鏁忔劅鐨勫満鏅紝鎵归噺澶勭悊鏇撮珮鏁堛€?
    ///
    /// # 鍙傛暟
    /// - `interiors`: 鍐呴儴鍗曞厓鐘舵€佹暟缁?
    /// - `kinds`: 杈圭晫绫诲瀷鏁扮粍
    /// - `normals`: 娉曞悜閲忔暟缁?
    /// - `externals`: 澶栭儴寮鸿揩鏁扮粍锛堝彲閫夛級
    /// - `z_beds`: 搴曞簥楂樼▼鏁扮粍
    /// - `output`: 杈撳嚭鏁扮粍
    pub fn compute_ghost_batch(
        &self,
        interiors: &[ConservedState],
        kinds: &[BoundaryKind],
        normals: &[DVec2],
        externals: Option<&[ExternalForcing]>,
        z_beds: &[f64],
        output: &mut [ConservedState],
    ) {
        debug_assert_eq!(interiors.len(), kinds.len());
        debug_assert_eq!(interiors.len(), normals.len());
        debug_assert_eq!(interiors.len(), z_beds.len());
        debug_assert_eq!(interiors.len(), output.len());

        let empty_forcing = ExternalForcing::ZERO;

        for i in 0..interiors.len() {
            let external = externals.map(|e| &e[i]).unwrap_or(&empty_forcing);
            output[i] = self.compute_ghost(interiors[i], kinds[i], normals[i], Some(external), z_beds[i]);
        }
    }

    /// 鑾峰彇鍙傛暟寮曠敤
    pub fn params(&self) -> &BoundaryParams {
        &self.params
    }
}

impl Default for GhostStateCalculator {
    fn default() -> Self {
        Self::new(BoundaryParams::default())
    }
}

// ============================================================
// 杈呭姪鍑芥暟
// ============================================================

/// 鍙嶅皠閫熷害鍚戦噺
///
/// 灏嗛€熷害鍚戦噺鍏充簬娉曞悜閲忓弽灏勩€?
///
/// # 鍙傛暟
/// - `velocity`: 鍘熷閫熷害
/// - `normal`: 鍙嶅皠闈㈡硶鍚戦噺锛堝崟浣嶅悜閲忥級
///
/// # 杩斿洖
/// 鍙嶅皠鍚庣殑閫熷害
#[inline]
pub fn reflect_velocity(velocity: DVec2, normal: DVec2) -> DVec2 {
    let un = velocity.dot(normal);
    velocity - 2.0 * un * normal
}

/// 鍒嗚В閫熷害涓烘硶鍚戝拰鍒囧悜鍒嗛噺
///
/// # 鍙傛暟
/// - `velocity`: 閫熷害鍚戦噺
/// - `normal`: 娉曞悜閲忥紙鍗曚綅鍚戦噺锛?
///
/// # 杩斿洖
/// (娉曞悜鍒嗛噺鏍囬噺, 鍒囧悜鍒嗛噺鍚戦噺)
#[inline]
pub fn decompose_velocity(velocity: DVec2, normal: DVec2) -> (f64, DVec2) {
    let un = velocity.dot(normal);
    let ut = velocity - normal * un;
    (un, ut)
}

// ============================================================
// 娴嬭瘯
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_wall_ghost_no_penetration() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::from_primitive(1.0, 1.0, 0.0);
        let normal = DVec2::new(1.0, 0.0);

        let ghost = calculator.compute_ghost(interior, BoundaryKind::Wall, normal, None, 0.0);

        // 姘存繁淇濇寔
        assert!(approx_eq(ghost.h, 1.0));
        // 娉曞悜鍔ㄩ噺鍙嶅悜
        assert!(approx_eq(ghost.hu, -1.0));
        // 鍒囧悜鍔ㄩ噺淇濇寔
        assert!(approx_eq(ghost.hv, 0.0));
    }

    #[test]
    fn test_wall_ghost_oblique() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::from_primitive(1.0, 1.0, 1.0);
        let normal = DVec2::new(1.0, 0.0);

        let ghost = calculator.compute_ghost(interior, BoundaryKind::Wall, normal, None, 0.0);

        // 娉曞悜鍙嶈浆锛屽垏鍚戜繚鎸?
        assert!(approx_eq(ghost.hu, -1.0));
        assert!(approx_eq(ghost.hv, 1.0));
    }

    #[test]
    fn test_outflow_ghost() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::from_primitive(1.5, 0.5, 0.3);
        let normal = DVec2::new(1.0, 0.0);

        let ghost = calculator.compute_ghost(interior, BoundaryKind::Outflow, normal, None, 0.0);

        // 鍑烘祦锛氬畬鍏ㄥ鍒?
        assert!(approx_eq(ghost.h, 1.5));
        assert!(approx_eq(ghost.hu, 0.75)); // 1.5 * 0.5
        assert!(approx_eq(ghost.hv, 0.45)); // 1.5 * 0.3
    }

    #[test]
    fn test_inflow_ghost() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::from_primitive(1.0, 0.0, 0.0);
        let normal = DVec2::new(-1.0, 0.0);
        let external = ExternalForcing::new(2.0, 1.0, 0.0);

        let ghost = calculator.compute_ghost(
            interior,
            BoundaryKind::RiverInflow,
            normal,
            Some(&external),
            0.0,
        );

        // 浣跨敤澶栭儴寮鸿揩
        assert!(approx_eq(ghost.h, 2.0));
        assert!(approx_eq(ghost.hu, 2.0)); // h * u = 2.0 * 1.0
        assert!(approx_eq(ghost.hv, 0.0));
    }

    #[test]
    fn test_flather_open_sea_with_z_bed() {
        // 娴嬭瘯 Flather 杈圭晫鏉′欢姝ｇ‘浣跨敤姘翠綅 畏 = h + z_bed
        let calculator = GhostStateCalculator::default();
        
        // 鍐呴儴鍗曞厓: h=1.0, z_bed=0.5, 鎵€浠?畏_int = 1.5
        let interior = ConservedState::from_primitive(1.0, 0.0, 0.0);
        let normal = DVec2::new(1.0, 0.0);
        let z_bed = 0.5;
        
        // 澶栭儴寮鸿揩: 畏_ext = 1.5 (涓庡唴閮ㄧ浉鍚?
        let external = ExternalForcing::new(1.5, 0.0, 0.0);
        
        let ghost = calculator.compute_ghost(
            interior,
            BoundaryKind::OpenSea,
            normal,
            Some(&external),
            z_bed,
        );
        
        // 褰?畏_int = 畏_ext 鏃讹紝Flather 鏉′欢搴旇缁欏嚭 un_ghost = un_ext = 0
        // 骞界伒姘存繁 h_ghost = 畏_ext - z_bed = 1.5 - 0.5 = 1.0
        assert!(approx_eq(ghost.h, 1.0));
        assert!(ghost.hu.abs() < 1e-9); // 閫熷害鎺ヨ繎闆?
    }

    #[test]
    fn test_ghost_momentum_modes() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::from_primitive(1.0, 1.0, 0.0);
        let normal = DVec2::new(1.0, 0.0);

        // FullReflect
        let ghost = calculator.compute_ghost_with_mode(interior, normal, GhostMomentumMode::FullReflect);
        assert!(approx_eq(ghost.hu, -1.0));

        // NoReflect
        let ghost = calculator.compute_ghost_with_mode(interior, normal, GhostMomentumMode::NoReflect);
        assert!(approx_eq(ghost.hu, 1.0));

        // FullCancel
        let ghost = calculator.compute_ghost_with_mode(interior, normal, GhostMomentumMode::FullCancel);
        assert!(approx_eq(ghost.hu, -1.0));
        assert!(approx_eq(ghost.hv, 0.0));
    }

    #[test]
    fn test_reflect_velocity() {
        let v = DVec2::new(1.0, 0.0);
        let n = DVec2::new(1.0, 0.0);
        let reflected = reflect_velocity(v, n);
        assert!(approx_eq(reflected.x, -1.0));
        assert!(approx_eq(reflected.y, 0.0));

        // 鏂滃悜鍏ュ皠
        let v = DVec2::new(1.0, 1.0);
        let n = DVec2::new(1.0, 0.0);
        let reflected = reflect_velocity(v, n);
        assert!(approx_eq(reflected.x, -1.0));
        assert!(approx_eq(reflected.y, 1.0));
    }

    #[test]
    fn test_decompose_velocity() {
        let v = DVec2::new(3.0, 4.0);
        let n = DVec2::new(1.0, 0.0);
        let (un, ut) = decompose_velocity(v, n);
        assert!(approx_eq(un, 3.0));
        assert!(approx_eq(ut.x, 0.0));
        assert!(approx_eq(ut.y, 4.0));
    }

    #[test]
    fn test_batch_compute() {
        let calculator = GhostStateCalculator::default();

        let interiors = vec![
            ConservedState::from_primitive(1.0, 1.0, 0.0),
            ConservedState::from_primitive(2.0, 0.0, 1.0),
        ];
        let kinds = vec![BoundaryKind::Wall, BoundaryKind::Outflow];
        let normals = vec![DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.0)];
        let z_beds = vec![0.0, 0.0];

        let mut output = vec![ConservedState::default(); 2];
        calculator.compute_ghost_batch(&interiors, &kinds, &normals, None, &z_beds, &mut output);

        // 鍥哄锛氭硶鍚戝弽杞?
        assert!(approx_eq(output[0].hu, -1.0));
        // 鍑烘祦锛氱洿鎺ュ鍒?
        assert!(approx_eq(output[1].hv, 2.0)); // h * v = 2.0 * 1.0
    }

    #[test]
    fn test_momentum_mode_from_kind() {
        assert_eq!(
            GhostMomentumMode::from_boundary_kind(BoundaryKind::Wall),
            GhostMomentumMode::FullReflect
        );
        assert_eq!(
            GhostMomentumMode::from_boundary_kind(BoundaryKind::Symmetry),
            GhostMomentumMode::NoReflect
        );
        assert_eq!(
            GhostMomentumMode::from_boundary_kind(BoundaryKind::OpenSea),
            GhostMomentumMode::NoReflect
        );
    }
}
