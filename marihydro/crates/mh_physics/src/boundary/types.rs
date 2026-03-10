// crates/mh_physics/src/boundary/types.rs

//! 杈圭晫鏉′欢绫诲瀷瀹氫箟
//!
//! 鏈ā鍧楀畾涔夋祬姘存柟绋嬫眰瑙ｆ墍闇€鐨勮竟鐣屾潯浠剁被鍨嬶紝鍖呮嫭锛?
//! - BoundaryKind: 杈圭晫绫诲瀷鏋氫妇
//! - BoundaryCondition: 杈圭晫鏉′欢閰嶇疆
//! - ExternalForcing: 澶栭儴寮鸿揩鏁版嵁
//! - BoundaryParams: 杈圭晫璁＄畻鍙傛暟
//!
//! # 杩佺Щ璇存槑
//!
//! 浠?history_src/domain/boundary/types.rs 杩佺Щ锛岄€傞厤鏂版灦鏋勶細
//! - 浣跨敤 glam::DVec2 浠ｆ浛 (f64, f64) 琛ㄧず閫熷害
//! - 浣跨敤 serde 鏀寔閰嶇疆鏂囦欢
//! - 浣跨敤 repr(u8) 鏀寔 GPU 浼犺緭

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::types::NumericalParams;

// ============================================================
// 杈圭晫绫诲瀷鏋氫妇
// ============================================================

/// 杈圭晫绫诲瀷鏋氫妇
///
/// 瀹氫箟娴呮按鏂圭▼鏀寔鐨勮竟鐣屾潯浠剁被鍨嬨€備娇鐢?`repr(u8)` 浠ヤ究浜?GPU 鏁版嵁浼犺緭銆?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum BoundaryKind {
    /// 鍥哄杈圭晫锛堟棤绌块€忥級
    ///
    /// 娉曞悜閫熷害鍙嶅皠锛屽垏鍚戦€熷害淇濇寔銆傞€傜敤浜庝笉鍙笚閫忕殑杈圭晫銆?
    #[default]
    Wall = 0,

    /// 寮€娴疯竟鐣岋紙Flather 杈愬皠杈圭晫鏉′欢锛?
    ///
    /// 浣跨敤鐗瑰緛鍏崇郴缁撳悎澶栭儴寮鸿揩鏁版嵁锛屽厑璁告尝鍔ㄨ嚜鐢变紶鍑恒€?
    OpenSea = 1,

    /// 娌虫祦鍏ユ祦
    ///
    /// 缁欏畾娴侀噺鎴栨按浣嶇殑鍏ユ祦杈圭晫鏉′欢銆?
    RiverInflow = 2,

    /// 鑷敱鍑烘祦
    ///
    /// 闆舵搴﹀鎺紝鍏佽姘存祦鑷敱娴佸嚭璁＄畻鍩熴€?
    Outflow = 3,

    /// 瀵圭О杈圭晫
    ///
    /// 涓庡浐澹佺被浼间絾鏃犳懇鎿︼紝鐢ㄤ簬妯″瀷瀵圭О绠€鍖栥€?
    Symmetry = 4,

    /// 鍛ㄦ湡杈圭晫
    ///
    /// 闇€瑕佹垚瀵硅缃紝鐢ㄤ簬妯℃嫙鍛ㄦ湡鎬ф祦鍔ㄣ€?
    Periodic = 5,
}

impl BoundaryKind {
    /// 鏄惁闇€瑕佸閮ㄥ己杩暟鎹?
    ///
    /// OpenSea 鍜?RiverInflow 绫诲瀷闇€瑕佸閮ㄦ彁渚涙按浣嶆垨娴侀噺鏁版嵁銆?
    #[inline]
    pub fn requires_forcing(&self) -> bool {
        matches!(self, Self::OpenSea | Self::RiverInflow)
    }

    /// 鏄惁涓哄浐澹佺被鍨嬶紙鍙嶅皠杈圭晫锛?
    ///
    /// Wall 鍜?Symmetry 閮戒細鍙嶅皠娉曞悜閫熷害銆?
    #[inline]
    pub fn is_solid(&self) -> bool {
        matches!(self, Self::Wall | Self::Symmetry)
    }

    /// 鏄惁涓哄紑杈圭晫绫诲瀷
    ///
    /// 鍏佽鐗╄川鍜岃兘閲忛€氳繃鐨勮竟鐣屻€?
    #[inline]
    pub fn is_open(&self) -> bool {
        matches!(self, Self::OpenSea | Self::Outflow | Self::RiverInflow)
    }

    /// 浠?u8 鍊艰浆鎹紙鐢ㄤ簬 GPU 鏁版嵁璇诲彇锛?
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Wall),
            1 => Some(Self::OpenSea),
            2 => Some(Self::RiverInflow),
            3 => Some(Self::Outflow),
            4 => Some(Self::Symmetry),
            5 => Some(Self::Periodic),
            _ => None,
        }
    }

    /// 杞崲涓?u8 鍊?
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

impl std::fmt::Display for BoundaryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Wall => "Wall",
            Self::OpenSea => "OpenSea",
            Self::RiverInflow => "RiverInflow",
            Self::Outflow => "Outflow",
            Self::Symmetry => "Symmetry",
            Self::Periodic => "Periodic",
        };
        write!(f, "{}", name)
    }
}

// ============================================================
// 杈圭晫鏉′欢閰嶇疆
// ============================================================

/// 杈圭晫鏉′欢閰嶇疆
///
/// 瀹屾暣鎻忚堪涓€涓竟鐣屾潯浠剁殑鍙傛暟锛屽寘鎷被鍨嬨€佸浐瀹氬€煎拰鍏宠仈鐨勫己杩暟鎹簮銆?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryCondition {
    /// 杈圭晫鍚嶇О锛堢敤浜庢爣璇嗗拰鏌ユ壘锛?
    pub name: String,

    /// 杈圭晫绫诲瀷
    pub kind: BoundaryKind,

    /// 鍥哄畾姘翠綅鍊?[m]
    ///
    /// 鐢ㄤ簬 OpenSea 杈圭晫鐨勬亽瀹氭按浣嶆垨 Outflow 鐨勫弬鑰冩按浣嶃€?
    pub fixed_eta: Option<f64>,

    /// 鍥哄畾娴侀噺鍊?[m鲁/s]
    ///
    /// 鐢ㄤ簬 RiverInflow 杈圭晫鐨勬亽瀹氭祦閲忋€?
    pub fixed_discharge: Option<f64>,

    /// 鍏宠仈鐨勫己杩暟鎹?Provider ID
    ///
    /// 鐢ㄤ簬鏌ユ壘鏃跺彉寮鸿揩鏁版嵁婧愩€?
    pub forcing_id: Option<usize>,

    /// 鏇煎畞绮楃硻搴︾郴鏁?
    ///
    /// 鐢ㄤ簬杈圭晫澶勭殑鎽╂摝璁＄畻銆?
    pub manning_n: Option<f64>,
}

impl BoundaryCondition {
    /// 鍒涘缓鍥哄杈圭晫鏉′欢
    ///
    /// # 绀轰緥
    /// ```
    /// use mh_physics::boundary::BoundaryCondition;
    ///
    /// let bc = BoundaryCondition::wall("north_wall");
    /// assert!(bc.kind.is_solid());
    /// ```
    pub fn wall(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: BoundaryKind::Wall,
            fixed_eta: None,
            fixed_discharge: None,
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 鍒涘缓寮€娴疯竟鐣屾潯浠?
    ///
    /// # 绀轰緥
    /// ```
    /// use mh_physics::boundary::BoundaryCondition;
    ///
    /// let bc = BoundaryCondition::open_sea("south_open");
    /// assert!(bc.kind.requires_forcing());
    /// ```
    pub fn open_sea(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: BoundaryKind::OpenSea,
            fixed_eta: None,
            fixed_discharge: None,
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 鍒涘缓娌虫祦鍏ユ祦杈圭晫鏉′欢
    ///
    /// # 鍙傛暟
    /// - `name`: 杈圭晫鍚嶇О
    /// - `discharge`: 鎭掑畾鍏ユ祦閲?[m鲁/s]
    ///
    /// # 绀轰緥
    /// ```
    /// use mh_physics::boundary::BoundaryCondition;
    ///
    /// let bc = BoundaryCondition::river_inflow("yangtze", 30000.0);
    /// assert_eq!(bc.fixed_discharge, Some(30000.0));
    /// ```
    pub fn river_inflow(name: impl Into<String>, discharge: f64) -> Self {
        Self {
            name: name.into(),
            kind: BoundaryKind::RiverInflow,
            fixed_eta: None,
            fixed_discharge: Some(discharge),
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 鍒涘缓鑷敱鍑烘祦杈圭晫鏉′欢
    pub fn outflow(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: BoundaryKind::Outflow,
            fixed_eta: None,
            fixed_discharge: None,
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 鍒涘缓瀵圭О杈圭晫鏉′欢
    pub fn symmetry(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: BoundaryKind::Symmetry,
            fixed_eta: None,
            fixed_discharge: None,
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 璁剧疆寮鸿揩鏁版嵁婧?ID
    pub fn with_forcing(mut self, forcing_id: usize) -> Self {
        self.forcing_id = Some(forcing_id);
        self
    }

    /// 璁剧疆鍥哄畾姘翠綅
    pub fn with_fixed_eta(mut self, eta: f64) -> Self {
        self.fixed_eta = Some(eta);
        self
    }

    /// 璁剧疆鍥哄畾娴侀噺
    pub fn with_fixed_discharge(mut self, discharge: f64) -> Self {
        self.fixed_discharge = Some(discharge);
        self
    }

    /// 璁剧疆鏇煎畞绮楃硻搴?
    pub fn with_manning_n(mut self, n: f64) -> Self {
        self.manning_n = Some(n);
        self
    }
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        Self::wall("default")
    }
}

// ============================================================
// 澶栭儴寮鸿揩鏁版嵁
// ============================================================

/// 澶栭儴寮鸿揩鏁版嵁
///
/// 杈圭晫澶勭殑姘翠綅鍜岄€熷害鏁版嵁锛岀敤浜?OpenSea銆丷iverInflow 绛夎竟鐣屾潯浠躲€?
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExternalForcing {
    /// 姘翠綅 [m]
    pub eta: f64,

    /// 閫熷害鍚戦噺 [m/s]
    pub velocity: DVec2,
}

impl ExternalForcing {
    /// 闆跺己杩父閲?
    pub const ZERO: Self = Self {
        eta: 0.0,
        velocity: DVec2::ZERO,
    };

    /// 鍒涘缓瀹屾暣鐨勫己杩暟鎹?
    ///
    /// # 鍙傛暟
    /// - `eta`: 姘翠綅 [m]
    /// - `u`: x 鏂瑰悜閫熷害 [m/s]
    /// - `v`: y 鏂瑰悜閫熷害 [m/s]
    #[inline]
    pub fn new(eta: f64, u: f64, v: f64) -> Self {
        Self {
            eta,
            velocity: DVec2::new(u, v),
        }
    }

    /// 鍒涘缓浠呮按浣嶇殑寮鸿揩鏁版嵁
    #[inline]
    pub fn with_eta(eta: f64) -> Self {
        Self {
            eta,
            velocity: DVec2::ZERO,
        }
    }

    /// 鍒涘缓浠呴€熷害鐨勫己杩暟鎹?
    #[inline]
    pub fn with_velocity(u: f64, v: f64) -> Self {
        Self {
            eta: 0.0,
            velocity: DVec2::new(u, v),
        }
    }

    /// 鑾峰彇 x 鏂瑰悜閫熷害
    #[inline]
    pub fn u(&self) -> f64 {
        self.velocity.x
    }

    /// 鑾峰彇 y 鏂瑰悜閫熷害
    #[inline]
    pub fn v(&self) -> f64 {
        self.velocity.y
    }

    /// 妫€鏌ユ暟鎹槸鍚︽湁鏁?
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.eta.is_finite() && self.velocity.is_finite()
    }
}

// ============================================================
// 杈圭晫璁＄畻鍙傛暟
// ============================================================

/// 杈圭晫璁＄畻鍙傛暟
///
/// 杈圭晫閫氶噺璁＄畻鎵€闇€鐨勭墿鐞嗗弬鏁板拰棰勮绠楀父閲忋€?
#[derive(Debug, Clone, Copy)]
pub struct BoundaryParams {
    /// 閲嶅姏鍔犻€熷害 [m/s虏]
    pub gravity: f64,

    /// 鏈€灏忔按娣遍槇鍊?[m]
    pub h_min: f64,

    /// sqrt(g) - 棰勮绠椾互鎻愰珮鎬ц兘
    pub sqrt_g: f64,
}

impl BoundaryParams {
    /// 鍒涘缓杈圭晫鍙傛暟
    ///
    /// # 鍙傛暟
    /// - `gravity`: 閲嶅姏鍔犻€熷害 [m/s虏]
    /// - `h_min`: 鏈€灏忔按娣遍槇鍊?[m]
    pub fn new(gravity: f64, h_min: f64) -> Self {
        Self {
            gravity,
            h_min,
            sqrt_g: gravity.sqrt(),
        }
    }

    /// 浠庢暟鍊煎弬鏁板垱寤?
    ///
    /// 浣跨敤榛樿閲嶅姏鍔犻€熷害 (9.81 m/s虏)銆?
    /// 濡傛灉闇€瑕佽嚜瀹氫箟閲嶅姏锛岃浣跨敤 `new` 鏂规硶銆?
    pub fn from_numerical_params(params: &NumericalParams) -> Self {
        Self::new(9.81, params.h_min)
    }

    /// 浠庢暟鍊煎弬鏁板拰鐗╃悊甯告暟鍒涘缓
    pub fn from_params(numerical: &NumericalParams, physics: &crate::types::PhysicalConstants) -> Self {
        Self::new(physics.g, numerical.h_min)
    }

    /// 璁＄畻鐗瑰緛閫熷害锛堟尝閫燂級
    ///
    /// c = sqrt(g * h)
    #[inline]
    pub fn wave_speed(&self, h: f64) -> f64 {
        self.sqrt_g * h.max(self.h_min).sqrt()
    }

    /// 璁＄畻闈欐按鍘嬪姏
    ///
    /// p = 0.5 * g * h虏
    #[inline]
    pub fn hydrostatic_pressure(&self, h: f64) -> f64 {
        0.5 * self.gravity * h * h
    }
}

impl Default for BoundaryParams {
    fn default() -> Self {
        Self::new(9.81, 1e-6)
    }
}

// ============================================================
// 娴嬭瘯
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_kind_properties() {
        assert!(BoundaryKind::Wall.is_solid());
        assert!(!BoundaryKind::Wall.is_open());
        assert!(!BoundaryKind::Wall.requires_forcing());

        assert!(BoundaryKind::OpenSea.is_open());
        assert!(BoundaryKind::OpenSea.requires_forcing());
        assert!(!BoundaryKind::OpenSea.is_solid());

        assert!(BoundaryKind::RiverInflow.requires_forcing());
        assert!(BoundaryKind::RiverInflow.is_open());

        assert!(BoundaryKind::Symmetry.is_solid());
    }

    #[test]
    fn test_boundary_kind_conversion() {
        for i in 0..=5 {
            let kind = BoundaryKind::from_u8(i).unwrap();
            assert_eq!(kind.as_u8(), i);
        }
        assert!(BoundaryKind::from_u8(6).is_none());
    }

    #[test]
    fn test_boundary_condition_builders() {
        let wall = BoundaryCondition::wall("north");
        assert_eq!(wall.name, "north");
        assert_eq!(wall.kind, BoundaryKind::Wall);

        let river = BoundaryCondition::river_inflow("yangtze", 30000.0);
        assert_eq!(river.kind, BoundaryKind::RiverInflow);
        assert!((river.fixed_discharge.unwrap() - 30000.0).abs() < 1e-10);

        let open = BoundaryCondition::open_sea("south")
            .with_forcing(1)
            .with_fixed_eta(0.5);
        assert_eq!(open.forcing_id, Some(1));
        assert!((open.fixed_eta.unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_external_forcing() {
        let forcing = ExternalForcing::new(1.5, 0.5, -0.3);
        assert!((forcing.eta - 1.5).abs() < 1e-10);
        assert!((forcing.u() - 0.5).abs() < 1e-10);
        assert!((forcing.v() - (-0.3)).abs() < 1e-10);
        assert!(forcing.is_valid());

        let zero = ExternalForcing::ZERO;
        assert!((zero.eta).abs() < 1e-10);
    }

    #[test]
    fn test_boundary_params() {
        let params = BoundaryParams::default();
        assert!((params.gravity - 9.81).abs() < 1e-10);
        assert!((params.sqrt_g - 9.81_f64.sqrt()).abs() < 1e-10);

        // 娉㈤€熸祴璇?
        let c = params.wave_speed(1.0);
        assert!((c - 9.81_f64.sqrt()).abs() < 1e-10);

        // 闈欐按鍘嬪姏娴嬭瘯
        let p = params.hydrostatic_pressure(1.0);
        assert!((p - 0.5 * 9.81).abs() < 1e-10);
    }
}
