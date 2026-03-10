// crates/mh_physics/src/engine/solver.rs

//! 娴呮按鏂圭▼姹傝В鍣?
//!
//! 鍩轰簬鏈夐檺浣撶Н娉曠殑闈炵粨鏋勫寲缃戞牸姹傝В鍣紝鏀寔锛?
//! - HLLC 榛庢浖姹傝В鍣?
//! - 骞叉箍澶勭悊
//! - 闈欐按閲嶆瀯
//! - 澶氱鏃堕棿绉垎鏂规
//!
//! # 杩佺Щ璇存槑
//!
//! 浠?history_src/physics/engine/solver.rs 杩佺Щ锛屼繚鎸佹牳蹇冪畻娉曚笉鍙樸€?
//! 婧愰」锛堟懇鎿︺€佺姘忓姏绛夛級灏嗗湪 sources 妯″潡杩佺Щ鍚庨泦鎴愩€?

use crate::adapter::PhysicsMesh;
use crate::engine::timestep::{TimeStepController, TimeStepControllerBuilder};
use crate::schemes::{HllcSolver, RiemannFlux, RiemannSolver};
use crate::schemes::wetting_drying::{WetState, WettingDryingHandler};
use crate::state::ShallowWaterState;
use crate::types::NumericalParams;
use crate::numerics::reconstruction::{MusclConfig, MusclReconstructor, Reconstructor};

use glam::DVec2;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ============================================================
// 姹傝В鍣ㄩ厤缃?
// ============================================================

/// 鏁板€兼牸寮忕被鍨?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumericalScheme {
    /// 涓€闃剁簿搴?
    FirstOrder,
    /// 浜岄樁 MUSCL
    #[default]
    SecondOrderMuscl,
    /// 浜岄樁 WENO
    SecondOrderWeno,
}

impl std::fmt::Display for NumericalScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstOrder => write!(f, "First Order"),
            Self::SecondOrderMuscl => write!(f, "MUSCL"),
            Self::SecondOrderWeno => write!(f, "WENO"),
        }
    }
}

/// 鍥為€€绛栫暐
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FallbackStrategy {
    /// 涓嶅洖閫€锛屽け璐ユ椂鎶ラ敊
    NoFallback,
    /// 鍥為€€鍒颁竴闃舵牸寮?
    #[default]
    FallbackToFirstOrder,
    /// 鍥為€€鍒拌緝灏忔椂闂存
    ReduceTimestep,
    /// 缁煎悎绛栫暐锛氬厛鍑忓皬鏃堕棿姝ワ紝鍐嶉檷浣庢牸寮忕簿搴?
    Progressive,
}

impl std::fmt::Display for FallbackStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFallback => write!(f, "无回退"),
            Self::FallbackToFirstOrder => write!(f, "回退一阶"),
            Self::ReduceTimestep => write!(f, "减小时间步"),
            Self::Progressive => write!(f, "渐进回退"),
        }
    }
}

/// 绋冲畾鎬ф鏌ラ€夐」
#[derive(Debug, Clone, Copy)]
pub struct StabilityOptions {
    /// 鏄惁妫€鏌?NaN/Inf
    pub check_nan: bool,
    /// 鏄惁妫€鏌ヨ礋姘存繁
    pub check_negative_depth: bool,
    /// 鏄惁妫€鏌ユ瀬澶ч€熷害
    pub check_extreme_velocity: bool,
    /// 閫熷害涓婇檺 [m/s]
    pub velocity_limit: f64,
    /// 姘存繁涓婇檺 [m]
    pub depth_limit: f64,
}

impl Default for StabilityOptions {
    fn default() -> Self {
        Self {
            check_nan: true,
            check_negative_depth: true,
            check_extreme_velocity: true,
            velocity_limit: 100.0,
            depth_limit: 1000.0,
        }
    }
}

impl StabilityOptions {
    /// 涓ユ牸妯″紡
    pub fn strict() -> Self {
        Self {
            check_nan: true,
            check_negative_depth: true,
            check_extreme_velocity: true,
            velocity_limit: 50.0,
            depth_limit: 500.0,
        }
    }

    /// 瀹芥澗妯″紡
    pub fn relaxed() -> Self {
        Self {
            check_nan: true,
            check_negative_depth: true,
            check_extreme_velocity: false,
            velocity_limit: 200.0,
            depth_limit: 2000.0,
        }
    }
}

/// 鏃堕棿绉垎鍣ㄧ被鍨?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeIntegrator {
    /// 鏄惧紡鏃堕棿绉垎
    #[default]
    Explicit,
    /// 鍗婇殣寮忔椂闂寸Н鍒嗭紙鍘嬪姏鏍℃锛?
    SemiImplicit,
}

impl std::fmt::Display for TimeIntegrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Explicit => write!(f, "显式"),
            Self::SemiImplicit => write!(f, "半隐式"),
        }
    }
}

/// 姹傝В鍣ㄩ厤缃?
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// 鏁板€煎弬鏁?
    pub params: NumericalParams,
    /// 閲嶅姏鍔犻€熷害 [m/s虏]
    pub gravity: f64,
    /// 鏄惁鍚敤闈欐按閲嶆瀯
    pub use_hydrostatic_reconstruction: bool,
    /// 骞惰鍖栭槇鍊硷紙闈㈡暟锛?
    pub parallel_threshold: usize,
    /// 鏄惁鍚敤闅愬紡鎽╂摝锛堥鐣欙級
    pub implicit_friction: bool,
    /// 鏁板€兼牸寮?
    pub scheme: NumericalScheme,
    /// 鍥為€€绛栫暐
    pub fallback: FallbackStrategy,
    /// 绋冲畾鎬ф鏌ラ€夐」
    pub stability: StabilityOptions,
    /// 鏈€澶у洖閫€娆℃暟
    pub max_fallback_attempts: u32,
    /// 鏃堕棿姝ュ噺灏忓洜瀛愶紙鍥為€€鏃朵娇鐢級
    pub timestep_reduction_factor: f64,
    /// 鏃堕棿绉垎鍣ㄧ被鍨?
    pub integrator: TimeIntegrator,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            params: NumericalParams::default(),
            gravity: 9.81,
            use_hydrostatic_reconstruction: true,
            parallel_threshold: 1000,
            implicit_friction: true,
            scheme: NumericalScheme::default(),
            fallback: FallbackStrategy::default(),
            stability: StabilityOptions::default(),
            max_fallback_attempts: 3,
            timestep_reduction_factor: 0.5,
            integrator: TimeIntegrator::default(),
        }
    }
}

impl SolverConfig {
    /// 鍒涘缓鏋勫缓鍣?
    pub fn builder() -> SolverConfigBuilder {
        SolverConfigBuilder::default()
    }

    /// 蹇€熼厤缃細鎬ц兘浼樺厛
    pub fn performance() -> Self {
        Self {
            scheme: NumericalScheme::FirstOrder,
            stability: StabilityOptions::relaxed(),
            parallel_threshold: 500,
            ..Default::default()
        }
    }

    /// 蹇€熼厤缃細绮惧害浼樺厛
    pub fn accuracy() -> Self {
        Self {
            scheme: NumericalScheme::SecondOrderMuscl,
            stability: StabilityOptions::strict(),
            fallback: FallbackStrategy::Progressive,
            ..Default::default()
        }
    }

    /// 蹇€熼厤缃細绋冲仴妯″紡
    pub fn robust() -> Self {
        Self {
            scheme: NumericalScheme::FirstOrder,
            fallback: FallbackStrategy::Progressive,
            stability: StabilityOptions::strict(),
            max_fallback_attempts: 5,
            timestep_reduction_factor: 0.25,
            ..Default::default()
        }
    }
}

/// 閰嶇疆鏋勫缓鍣?
#[derive(Default)]
pub struct SolverConfigBuilder {
    config: SolverConfig,
}

impl SolverConfigBuilder {
    pub fn params(mut self, params: NumericalParams) -> Self {
        self.config.params = params;
        self
    }

    pub fn gravity(mut self, g: f64) -> Self {
        self.config.gravity = g;
        self
    }

    pub fn use_hydrostatic_reconstruction(mut self, enable: bool) -> Self {
        self.config.use_hydrostatic_reconstruction = enable;
        self
    }

    pub fn parallel_threshold(mut self, threshold: usize) -> Self {
        self.config.parallel_threshold = threshold;
        self
    }

    pub fn implicit_friction(mut self, enable: bool) -> Self {
        self.config.implicit_friction = enable;
        self
    }

    /// 璁剧疆鏁板€兼牸寮?
    pub fn scheme(mut self, scheme: NumericalScheme) -> Self {
        self.config.scheme = scheme;
        self
    }

    /// 璁剧疆鍥為€€绛栫暐
    pub fn fallback(mut self, fallback: FallbackStrategy) -> Self {
        self.config.fallback = fallback;
        self
    }

    /// 璁剧疆绋冲畾鎬ч€夐」
    pub fn stability(mut self, stability: StabilityOptions) -> Self {
        self.config.stability = stability;
        self
    }

    /// 璁剧疆鏈€澶у洖閫€娆℃暟
    pub fn max_fallback_attempts(mut self, attempts: u32) -> Self {
        self.config.max_fallback_attempts = attempts;
        self
    }

    /// 璁剧疆鏃堕棿姝ュ噺灏忓洜瀛?
    pub fn timestep_reduction_factor(mut self, factor: f64) -> Self {
        self.config.timestep_reduction_factor = factor.clamp(0.1, 0.9);
        self
    }

    pub fn build(self) -> SolverConfig {
        self.config
    }
}

// ============================================================
// 姹傝В鍣ㄧ粺璁?
// ============================================================

/// 姹傝В鍣ㄦ杩涚粺璁?
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    /// 鏈€澶ф尝閫?[m/s]
    pub max_wave_speed: f64,
    /// 骞插崟鍏冩暟閲?
    pub dry_cells: usize,
    /// 琚檺鍒剁殑闈㈡暟閲?
    pub limited_faces: usize,
    /// 褰撳墠鏃堕棿姝ラ暱 [s]
    pub dt: f64,
    /// 鍥為€€娆℃暟
    pub fallback_count: u32,
    /// 褰撳墠浣跨敤鐨勬牸寮?
    pub current_scheme: NumericalScheme,
    /// 绋冲畾鎬х姸鎬?
    pub stability_status: StabilityStatus,
}

/// 绋冲畾鎬х姸鎬?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StabilityStatus {
    /// 绋冲畾
    #[default]
    Stable,
    /// 鎺ヨ繎涓嶇ǔ瀹?
    Marginal,
    /// 闇€瑕佸洖閫€
    NeedsFallback,
    /// 涓嶇ǔ瀹氾紙璁＄畻澶辫触锛?
    Unstable,
}

impl std::fmt::Display for StabilityStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stable => write!(f, "稳定"),
            Self::Marginal => write!(f, "临界"),
            Self::NeedsFallback => write!(f, "需要回退"),
            Self::Unstable => write!(f, "不稳定"),
        }
    }
}

impl SolverStats {
    /// 妫€鏌ユ槸鍚﹂渶瑕佸洖閫€
    pub fn needs_fallback(&self) -> bool {
        matches!(self.stability_status, StabilityStatus::NeedsFallback | StabilityStatus::Unstable)
    }

    /// 鐢熸垚璇婃柇鎽樿
    pub fn summary(&self) -> String {
        format!(
            "dt={:.4}s, wave_speed={:.2}m/s, dry={}, limited={}, status={}, fallbacks={}",
            self.dt,
            self.max_wave_speed,
            self.dry_cells,
            self.limited_faces,
            self.stability_status,
            self.fallback_count
        )
    }
}

// ============================================================
// 姹傝В鍣ㄥ伐浣滃尯
// ============================================================

/// 姹傝В鍣ㄥ伐浣滃尯
///
/// 瀛樺偍涓棿璁＄畻缁撴灉锛岄伩鍏嶉噸澶嶅垎閰?
#[derive(Debug)]
pub struct SolverWorkspace {
    /// 閫氶噺绱姞锛堣川閲忥級
    pub flux_h: Vec<f64>,
    /// 閫氶噺绱姞锛坸鍔ㄩ噺锛?
    pub flux_hu: Vec<f64>,
    /// 閫氶噺绱姞锛坹鍔ㄩ噺锛?
    pub flux_hv: Vec<f64>,
    /// 婧愰」绱姞锛坸鍔ㄩ噺锛?
    pub source_hu: Vec<f64>,
    /// 婧愰」绱姞锛坹鍔ㄩ噺锛?
    pub source_hv: Vec<f64>,
    /// 鍗曞厓閫熷害 u 鍒嗛噺锛堢敤浜庨噸鏋勶級
    pub vel_u: Vec<f64>,
    /// 鍗曞厓閫熷害 v 鍒嗛噺锛堢敤浜庨噸鏋勶級
    pub vel_v: Vec<f64>,
    /// 姘翠綅 畏 = h + z锛堢敤浜?well-balanced 閲嶆瀯锛?
    pub eta: Vec<f64>,
}

impl SolverWorkspace {
    /// 鍒涘缓宸ヤ綔鍖?
    pub fn new(n_cells: usize) -> Self {
        Self {
            flux_h: vec![0.0; n_cells],
            flux_hu: vec![0.0; n_cells],
            flux_hv: vec![0.0; n_cells],
            source_hu: vec![0.0; n_cells],
            source_hv: vec![0.0; n_cells],
            vel_u: vec![0.0; n_cells],
            vel_v: vec![0.0; n_cells],
            eta: vec![0.0; n_cells],
        }
    }

    /// 閲嶇疆閫氶噺
    pub fn reset_fluxes(&mut self) {
        self.flux_h.fill(0.0);
        self.flux_hu.fill(0.0);
        self.flux_hv.fill(0.0);
    }

    /// 閲嶇疆婧愰」
    pub fn reset_sources(&mut self) {
        self.source_hu.fill(0.0);
        self.source_hv.fill(0.0);
    }

    /// 閲嶇疆鎵€鏈?
    pub fn reset(&mut self) {
        self.reset_fluxes();
        self.reset_sources();
    }

    /// 璋冩暣澶у皬
    pub fn resize(&mut self, n_cells: usize) {
        self.flux_h.resize(n_cells, 0.0);
        self.flux_hu.resize(n_cells, 0.0);
        self.flux_hv.resize(n_cells, 0.0);
        self.source_hu.resize(n_cells, 0.0);
        self.source_hv.resize(n_cells, 0.0);
        self.vel_u.resize(n_cells, 0.0);
        self.vel_v.resize(n_cells, 0.0);
        self.eta.resize(n_cells, 0.0);
    }
}

// ============================================================
// 闈欐按閲嶆瀯
// ============================================================

/// 闈笂鐨勯潤姘撮噸鏋勭姸鎬?
#[derive(Debug, Clone, Copy)]
pub struct HydrostaticFaceState {
    /// 宸︿晶鏈夋晥姘存繁
    pub h_left: f64,
    /// 鍙充晶鏈夋晥姘存繁
    pub h_right: f64,
    /// 宸︿晶閫熷害
    pub vel_left: DVec2,
    /// 鍙充晶閫熷害
    pub vel_right: DVec2,
    /// 闈㈠楂樼▼
    pub z_face: f64,
}

/// 搴婂潯婧愰」淇锛堝垎鍒粰宸﹀彸鍗曞厓锛?
#[derive(Debug, Clone, Copy)]
pub struct BedSlopeCorrection {
    /// 宸︿晶锛坥wner锛夊崟鍏?x 鏂瑰悜婧愰」
    pub source_left_x: f64,
    /// 宸︿晶锛坥wner锛夊崟鍏?y 鏂瑰悜婧愰」
    pub source_left_y: f64,
    /// 鍙充晶锛坣eighbor锛夊崟鍏?x 鏂瑰悜婧愰」
    pub source_right_x: f64,
    /// 鍙充晶锛坣eighbor锛夊崟鍏?y 鏂瑰悜婧愰」
    pub source_right_y: f64,
}

impl BedSlopeCorrection {
    /// 闆舵簮椤?
    pub const ZERO: Self = Self {
        source_left_x: 0.0,
        source_left_y: 0.0,
        source_right_x: 0.0,
        source_right_y: 0.0,
    };
}

/// 闈欐按閲嶆瀯澶勭悊鍣?
#[derive(Debug, Clone)]
pub struct HydrostaticReconstruction {
    /// 鏁板€煎弬鏁?
    params: NumericalParams,
    /// 閲嶅姏鍔犻€熷害
    g: f64,
}

impl HydrostaticReconstruction {
    /// 鍒涘缓闈欐按閲嶆瀯澶勭悊鍣?
    pub fn new(params: &NumericalParams, g: f64) -> Self {
        Self {
            params: params.clone(),
            g,
        }
    }

    /// 绠€鍗曢潤姘撮噸鏋?
    ///
    /// 瀵归潰涓や晶鐨勬按娣辫繘琛屼慨姝ｏ紝纭繚闈欐按骞宠　
    pub fn reconstruct_face_simple(
        &self,
        h_l: f64,
        h_r: f64,
        z_l: f64,
        z_r: f64,
        vel_l: DVec2,
        vel_r: DVec2,
    ) -> HydrostaticFaceState {
        // 闈㈠楂樼▼鍙栨渶澶у€硷紙淇濆畧澶勭悊锛?
        let z_face = z_l.max(z_r);

        // 淇鍚庣殑姘存繁 = max(0, 畏 - z_face)
        // 鍏朵腑 畏 = h + z 鏄按浣?
        let eta_l = h_l + z_l;
        let eta_r = h_r + z_r;

        let h_left = (eta_l - z_face).max(0.0);
        let h_right = (eta_r - z_face).max(0.0);

        HydrostaticFaceState {
            h_left,
            h_right,
            vel_left: vel_l,
            vel_right: vel_r,
            z_face,
        }
    }

    /// 搴婂潯婧愰」淇
    ///
    /// 璁＄畻鐢变簬楂樼▼宸骇鐢熺殑鍘嬪姏姊害婧愰」
    pub fn bed_slope_correction(
        &self,
        h_l: f64,
        h_r: f64,
        z_l: f64,
        z_r: f64,
        normal: DVec2,
        length: f64,
    ) -> BedSlopeCorrection {
        // 浣跨敤闈㈠钩鍧囨按娣?
        let h_face = 0.5 * (h_l + h_r);
        
        if h_face < self.params.h_dry {
            return BedSlopeCorrection::ZERO;
        }

        // 鏃х殑搴婂潯婧愰」鏂规硶锛堝凡寮冪敤锛屼繚鐣欎緵鍙傝€冿級
        // 娉ㄦ剰锛氭眰瑙ｅ櫒涓娇鐢?compute_hydrostatic_bed_slope 鏇夸唬
        let dz = z_r - z_l;
        let factor = -self.g * h_face * dz * length;

        // 绠€鍖栧鐞嗭細宸﹀彸鍚勫垎涓€鍗?
        BedSlopeCorrection {
            source_left_x: 0.5 * factor * normal.x,
            source_left_y: 0.5 * factor * normal.y,
            source_right_x: -0.5 * factor * normal.x,
            source_right_y: -0.5 * factor * normal.y,
        }
    }
}

// ============================================================
// 涓绘眰瑙ｅ櫒
// ============================================================

/// 娴呮按鏂圭▼姹傝В鍣?
///
/// 鍩轰簬鏈夐檺浣撶Н娉曠殑闈炵粨鏋勫寲缃戞牸姹傝В鍣ㄣ€?
pub struct ShallowWaterSolver {
    /// 缃戞牸
    mesh: Arc<PhysicsMesh>,
    /// 閰嶇疆
    config: SolverConfig,
    /// 宸ヤ綔鍖?
    workspace: SolverWorkspace,
    /// 榛庢浖姹傝В鍣?
    riemann: HllcSolver,
    /// 骞叉箍澶勭悊鍣?
    wetting_drying: WettingDryingHandler,
    /// 闈欐按閲嶆瀯
    hydrostatic: HydrostaticReconstruction,
    /// 鏃堕棿姝ユ帶鍒跺櫒
    timestep_ctrl: TimeStepController,
    /// 缁熻淇℃伅
    stats: SolverStats,
    /// 姘翠綅閲嶆瀯鍣紙鐢ㄤ簬 well-balanced 鏂规硶锛?
    muscl_eta: MusclReconstructor,
    /// u 閫熷害閲嶆瀯鍣?
    muscl_u: MusclReconstructor,
    /// v 閫熷害閲嶆瀯鍣?
    muscl_v: MusclReconstructor,
}

impl ShallowWaterSolver {
    /// 鍒涘缓姹傝В鍣?
    pub fn new(mesh: Arc<PhysicsMesh>, config: SolverConfig) -> Self {
        let n_cells = mesh.n_cells();

        let timestep_ctrl = TimeStepControllerBuilder::new(config.gravity)
            .with_cfl(config.params.cfl)
            .with_dt_limits(config.params.dt_min, config.params.dt_max)
            .build();

        let muscl_config = if matches!(config.scheme, NumericalScheme::FirstOrder) {
            MusclConfig::first_order()
        } else {
            MusclConfig::default()
        };

        // 浣跨敤 muscl_eta 浠ｆ浛 muscl_h锛屽疄鐜?well-balanced 浜岄樁閲嶆瀯
        let muscl_eta = MusclReconstructor::new(muscl_config.clone(), mesh.clone());
        let muscl_u = MusclReconstructor::new(muscl_config.clone(), mesh.clone());
        let muscl_v = MusclReconstructor::new(muscl_config.clone(), mesh.clone());

        Self {
            mesh: mesh.clone(),
            config: config.clone(),
            workspace: SolverWorkspace::new(n_cells),
            riemann: HllcSolver::new(&config.params, config.gravity),
            wetting_drying: WettingDryingHandler::from_params(&config.params),
            hydrostatic: HydrostaticReconstruction::new(&config.params, config.gravity),
            timestep_ctrl,
            stats: SolverStats::default(),
            muscl_eta,
            muscl_u,
            muscl_v,
        }
    }

    /// 鎵ц涓€涓椂闂存
    ///
    /// 杩斿洖浣跨敤鐨勬椂闂存闀?
    pub fn step(&mut self, state: &mut ShallowWaterState, dt: f64) -> f64 {
        // 1. 閲嶇疆宸ヤ綔鍖?
        self.workspace.reset();

        // 2. 棰勮绠楅€熷害骞跺噯澶囦簩闃堕噸鏋?
        self.prepare_reconstruction(state);

        // 3. 璁＄畻閫氶噺锛堥泦鎴愬共婀垮鐞嗭級
        let max_wave_speed = if self.mesh.n_faces() >= self.config.parallel_threshold {
            self.compute_fluxes_parallel(state)
        } else {
            self.compute_fluxes_serial(state)
        };

        // 4. 鏇存柊鐘舵€?
        self.update_state(state, dt);

        // 5. 寮哄埗姝ｆ€у苟澶勭悊骞插尯
        let (dry_cells, _) = self.enforce_positivity(state, dt);

        // 6. 鏇存柊缁熻
        self.stats.max_wave_speed = max_wave_speed;
        self.stats.dry_cells = dry_cells;
        self.stats.dt = dt;

        dt
    }

    /// 璁＄畻鑷€傚簲鏃堕棿姝ラ暱
    pub fn compute_dt(&mut self, state: &ShallowWaterState) -> f64 {
        self.timestep_ctrl.update(state, &self.mesh, &self.config.params)
    }

    /// 鏄惁浣跨敤浜岄樁鏍煎紡
    #[inline]
    fn use_second_order(&self) -> bool {
        matches!(self.config.scheme, NumericalScheme::SecondOrderMuscl | NumericalScheme::SecondOrderWeno)
    }

    /// 鏍规嵁閰嶇疆鍚屾閲嶆瀯鍣ㄥ紑鍏冲苟璁＄畻姊害
    ///
    /// 瀵逛簬 well-balanced 浜岄樁鏍煎紡锛岄噸鏋勬按浣?畏 = h + z 鑰岄潪姘存繁 h锛?
    /// 浠ヤ繚璇?C-property锛堥潤姘村钩琛℃椂閫熷害涓洪浂锛?
    fn prepare_reconstruction(&mut self, state: &ShallowWaterState) {
        let n = state.n_cells();
        if self.workspace.vel_u.len() != n {
            self.workspace.resize(n);
        }

        // 棰勮绠楀畨鍏ㄩ€熷害鍜屾按浣?
        for i in 0..n {
            let (u, v) = self.config.params.safe_velocity_components(state.hu[i], state.hv[i], state.h[i]);
            self.workspace.vel_u[i] = u;
            self.workspace.vel_v[i] = v;
            // 璁＄畻姘翠綅 畏 = h + z锛堢敤浜?well-balanced 閲嶆瀯锛?
            self.workspace.eta[i] = state.h[i] + state.z[i];
        }

        if !self.use_second_order() {
            let cfg = MusclConfig::first_order();
            self.muscl_eta.set_config(cfg.clone());
            self.muscl_u.set_config(cfg.clone());
            self.muscl_v.set_config(cfg);
            return;
        }

        // 浜岄樁妯″紡锛氫娇鐢ㄩ粯璁ら厤缃?
        let cfg = MusclConfig::default();
        self.muscl_eta.set_config(cfg.clone());
        self.muscl_u.set_config(cfg.clone());
        self.muscl_v.set_config(cfg);

        // 瀵规按浣?畏 鑰岄潪姘存繁 h 璁＄畻姊害锛屼繚璇?C-property
        self.muscl_eta.compute_gradients(&self.workspace.eta);
        self.muscl_u.compute_gradients(&self.workspace.vel_u);
        self.muscl_v.compute_gradients(&self.workspace.vel_v);
    }

    // =========================================================================
    // 閫氶噺璁＄畻锛堜覆琛岋級
    // =========================================================================

    fn compute_fluxes_serial(&mut self, state: &ShallowWaterState) -> f64 {
        let n_faces = self.mesh.n_faces();
        let mut max_wave_speed = 0.0f64;

        for face_idx in 0..n_faces {
            let (flux, bed_src, length, owner, neighbor) = 
                self.compute_face_flux(state, face_idx);

            max_wave_speed = max_wave_speed.max(flux.max_wave_speed);

            // 绱姞鍒?owner
            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;

            self.workspace.flux_h[owner] -= fh;
            self.workspace.flux_hu[owner] -= fhu;
            self.workspace.flux_hv[owner] -= fhv;
            self.workspace.source_hu[owner] += bed_src.source_left_x;
            self.workspace.source_hv[owner] += bed_src.source_left_y;

            // 绱姞鍒?neighbor锛堝鏋滃瓨鍦級
            if let Some(neigh) = neighbor {
                self.workspace.flux_h[neigh] += fh;
                self.workspace.flux_hu[neigh] += fhu;
                self.workspace.flux_hv[neigh] += fhv;
                self.workspace.source_hu[neigh] += bed_src.source_right_x;
                self.workspace.source_hv[neigh] += bed_src.source_right_y;
            }
        }

        max_wave_speed
    }

    // =========================================================================
    // 閫氶噺璁＄畻锛堟敹闆嗗悗绱姞绛栫暐锛?
    // =========================================================================

    /// 浣跨敤"鏀堕泦鍚庣疮鍔?绛栫暐璁＄畻閫氶噺
    ///
    /// # 鎶€鏈€哄姟 (TD-5.3.2, TD-5.3.3)
    ///
    /// 褰撳墠瀹炵幇鏄吉骞惰锛?
    /// 1. 骞惰闃舵锛氬悇闈㈤€氶噺璁＄畻鏄湡姝ｅ苟琛岀殑
    /// 2. 绱姞闃舵锛氭敹闆嗘墍鏈夌粨鏋滃悗涓茶绱姞鍒板崟鍏?
    ///
    /// 瀵逛簬澶ц妯＄綉鏍硷紝涓茶绱姞浼氭垚涓烘€ц兘鐡堕銆?
    /// 鏈潵闇€瑕佸疄鐜扮潃鑹插苟琛?Colored Parallel)浠ヨ幏寰楃湡姝ｇ殑鏃犻攣绱姞銆?
    fn compute_fluxes_parallel(&mut self, state: &ShallowWaterState) -> f64 {
        let n_faces = self.mesh.n_faces();
        let max_speed_atomic = AtomicU64::new(0u64);

        // 闃舵1: 骞惰璁＄畻鎵€鏈夐潰鐨勯€氶噺 (鐪熸骞惰)
        let face_results: Vec<_> = (0..n_faces)
            .into_par_iter()
            .map(|face_idx| {
                let (flux, bed_src, length, owner, neighbor) = 
                    self.compute_face_flux(state, face_idx);

                // 鏇存柊鏈€澶ф尝閫?
                max_speed_atomic.fetch_max(flux.max_wave_speed.to_bits(), Ordering::Relaxed);

                (flux, bed_src, length, owner, neighbor)
            })
            .collect();

        // 闃舵2: 涓茶绱姞鍒板崟鍏?(鎬ц兘鐡堕)
        // TODO(TD-5.3.3): 浣跨敤鐫€鑹插苟琛屽疄鐜版棤閿佺疮鍔?
        for (flux, bed_src, length, owner, neighbor) in face_results {
            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;

            self.workspace.flux_h[owner] -= fh;
            self.workspace.flux_hu[owner] -= fhu;
            self.workspace.flux_hv[owner] -= fhv;
            self.workspace.source_hu[owner] += bed_src.source_left_x;
            self.workspace.source_hv[owner] += bed_src.source_left_y;

            if let Some(neigh) = neighbor {
                self.workspace.flux_h[neigh] += fh;
                self.workspace.flux_hu[neigh] += fhu;
                self.workspace.flux_hv[neigh] += fhv;
                self.workspace.source_hu[neigh] += bed_src.source_right_x;
                self.workspace.source_hv[neigh] += bed_src.source_right_y;
            }
        }

        f64::from_bits(max_speed_atomic.load(Ordering::Relaxed))
    }

    // =========================================================================
    // 鍗曢潰閫氶噺璁＄畻
    // =========================================================================

    fn compute_face_flux(
        &self,
        state: &ShallowWaterState,
        face_idx: usize,
    ) -> (RiemannFlux, BedSlopeCorrection, f64, usize, Option<usize>) {
        let normal = self.mesh.face_normal(face_idx);
        let length = self.mesh.face_length(face_idx);
        let owner = self.mesh.face_owner(face_idx);
        let neighbor = self.mesh.face_neighbor(face_idx);

        // 閲嶆瀯鍚庣殑宸?鍙崇姸鎬?
        // 娉ㄦ剰锛氫簩闃堕噸鏋勪娇鐢ㄦ按浣?畏 = h + z锛岀劧鍚庡弽鎺ㄦ按娣?h = 畏 - z
        // 杩欐槸 well-balanced 鏂规硶鐨勫叧閿紝淇濊瘉 C-property锛堥潤姘村钩琛℃椂 畏=const锛?
        let (h_l, vel_l, z_l, h_r, vel_r, z_r) = if self.use_second_order() {
            // 閲嶆瀯姘翠綅 畏 鑰岄潪姘存繁 h
            let eta_rec = self.muscl_eta.reconstruct_scalar(face_idx, &self.workspace.eta);
            let u_rec = self.muscl_u.reconstruct_scalar(face_idx, &self.workspace.vel_u);
            let v_rec = self.muscl_v.reconstruct_scalar(face_idx, &self.workspace.vel_v);

            if let Some(neigh) = neighbor {
                let z_owner = state.z[owner];
                let z_neigh = state.z[neigh];
                // 浠庢按浣嶅弽鎺ㄦ按娣憋細h = max(0, 畏 - z)
                let h_left = (eta_rec.left - z_owner).max(0.0);
                let h_right = (eta_rec.right - z_neigh).max(0.0);
                (
                    h_left,
                    DVec2::new(u_rec.left, v_rec.left),
                    z_owner,
                    h_right,
                    DVec2::new(u_rec.right, v_rec.right),
                    z_neigh,
                )
            } else {
                // 杈圭晫锛氬彸渚т娇鐢ㄥ弽灏勬潯浠?
                let z_owner = state.z[owner];
                let h_left = (eta_rec.left - z_owner).max(0.0);
                let vel_left = DVec2::new(u_rec.left, v_rec.left);
                let vn = vel_left.dot(normal);
                (
                    h_left,
                    vel_left,
                    z_owner,
                    h_left,
                    vel_left - 2.0 * vn * normal,
                    z_owner,
                )
            }
        } else {
            // 涓€闃讹細浣跨敤鍗曞厓涓績鍊?
            let h_l = state.h[owner];
            let z_l = state.z[owner];
            let (u_l, v_l) = self.config.params.safe_velocity_components(
                state.hu[owner], state.hv[owner], h_l,
            );
            let vel_l = DVec2::new(u_l, v_l);

            if let Some(neigh) = neighbor {
                let h = state.h[neigh];
                let (u, v) = self.config.params.safe_velocity_components(
                    state.hu[neigh], state.hv[neigh], h,
                );
                (
                    h_l,
                    vel_l,
                    z_l,
                    h,
                    DVec2::new(u, v),
                    state.z[neigh],
                )
            } else {
                // 杈圭晫鍙嶅皠
                let vn = vel_l.dot(normal);
                (
                    h_l,
                    vel_l,
                    z_l,
                    h_l,
                    vel_l - 2.0 * vn * normal,
                    z_l,
                )
            }
        };

        // 闈欐按閲嶆瀯
        // TODO(phase5): 鑰冭檻娣诲姞 reconstruct_face_muscl() 鏂规硶鏀寔楂橀樁閲嶆瀯
        let recon = if self.config.use_hydrostatic_reconstruction {
            self.hydrostatic.reconstruct_face_simple(h_l, h_r, z_l, z_r, vel_l, vel_r)
        } else {
            HydrostaticFaceState {
                h_left: h_l,
                h_right: h_r,
                vel_left: vel_l,
                vel_right: vel_r,
                z_face: 0.5 * (z_l + z_r),
            }
        };

        // 骞叉箍鐣岄潰閫氶噺闄愬埗
        // 娉ㄦ剰锛氫笉鑳界畝鍗曞湴闄愬埗閫氶噺锛屽惁鍒欎細闃绘娑︽箍杩囩▼
        // 鍙湁鍦ㄤ袱渚ч兘骞叉垨鑰呭浜庤繃娓″尯鏃舵墠闄愬埗
        let wet_l = self.wetting_drying.get_state(recon.h_left);
        let wet_r = self.wetting_drying.get_state(recon.h_right);
        let flux_limiter = match (wet_l, wet_r) {
            (WetState::Dry, WetState::Dry) => 0.0,
            // 骞叉箍鐣岄潰锛氬厑璁镐粠婀夸晶娴佸悜骞蹭晶锛屼笉闄愬埗閫氶噺
            (WetState::Dry, _) | (_, WetState::Dry) => 1.0,
            (WetState::PartiallyWet, WetState::PartiallyWet) => {
                // 涓や晶閮藉湪杩囨浮鍖烘椂骞虫粦闄愬埗
                let h_min = recon.h_left.min(recon.h_right);
                ((h_min - self.config.params.h_dry) 
                    / (self.config.params.h_wet - self.config.params.h_dry)).clamp(0.0, 1.0)
            }
            _ => 1.0,
        };

        // 姹傝В榛庢浖闂
        // TODO(phase5): 闆嗘垚 AdaptiveSolver 鏀寔鑷姩閫夋嫨鏈€浼樻眰瑙ｅ櫒
        let flux = self.riemann.solve(
            recon.h_left, recon.h_right,
            recon.vel_left, recon.vel_right,
            normal,
        ).unwrap_or(RiemannFlux::ZERO);

        // 搴旂敤骞叉箍闄愬埗
        let limited_flux = if flux_limiter < 1.0 {
            flux.scaled(flux_limiter)
        } else {
            flux
        };

        // 搴婂潯婧愰」锛堝熀浜庨潤姘撮噸鏋勭殑姘存繁宸紓锛?
        // 鍙傝€? Audusse et al. (2004) - 寮?(4.5)
        // S_bed = 0.5 * g * (h_L^2 - h_L*^2) * n  (宸︿晶璐＄尞)
        //       + 0.5 * g * (h_R^2 - h_R*^2) * n  (鍙充晶璐＄尞锛岀鍙风浉鍙?
        // 杩欑‘淇濅簡 C-property: 褰?畏 = const 鏃讹紝閫氶噺鍜屾簮椤瑰畬鍏ㄥ钩琛?
        let bed_src = self.compute_hydrostatic_bed_slope(
            h_l, h_r, recon.h_left, recon.h_right, normal, length
        );

        (limited_flux, bed_src, length, owner, neighbor)
    }

    /// 璁＄畻闈欐按閲嶆瀯鍚庣殑搴婂潯婧愰」
    ///
    /// 鍩轰簬 Audusse (2004) 鏂规硶锛屼娇鐢ㄥ師濮嬫按娣卞拰閲嶆瀯姘存繁鐨勫樊寮傝绠?
    /// 杩欎繚璇佷簡 C-property锛堥潤姘村钩琛℃椂閫熷害涓洪浂锛?
    ///
    /// 鍘熺悊锛?
    /// - 閫氶噺浣跨敤閲嶆瀯姘存繁 h* 璁＄畻锛屽帇鍔涢」鍙樹负 0.5*g*h*虏
    /// - 浣嗗疄闄呭帇鍔涘簲涓?0.5*g*h虏
    /// - 鍥犳闇€瑕佽ˉ鍋垮帇鍔涘樊: 0.5*g*(h虏 - h*虏)
    /// - 杩欎釜琛ュ伩浣滀负婧愰」鍔犲埌瀵瑰簲鍗曞厓涓?
    fn compute_hydrostatic_bed_slope(
        &self,
        h_l: f64,
        h_r: f64,
        h_l_star: f64,  // 闈欐按閲嶆瀯鍚庣殑宸︿晶姘存繁
        h_r_star: f64,  // 闈欐按閲嶆瀯鍚庣殑鍙充晶姘存繁
        normal: DVec2,
        length: f64,
    ) -> BedSlopeCorrection {
        let g = self.config.gravity;
        
        // 宸︿晶鍗曞厓鐨勫帇鍔涜ˉ鍋? 0.5 * g * (h_L虏 - h_L*虏) * L
        // 杩欐槸鍥犱负閫氶噺涓敤鐨勬槸 h_L*锛岄渶瑕佽ˉ鍋垮埌瀹為檯鍘嬪姏
        let pressure_diff_l = 0.5 * g * (h_l * h_l - h_l_star * h_l_star) * length;
        
        // 鍙充晶鍗曞厓鐨勫帇鍔涜ˉ鍋? 0.5 * g * (h_R虏 - h_R*虏) * L
        let pressure_diff_r = 0.5 * g * (h_r * h_r - h_r_star * h_r_star) * length;
        
        // 宸︿晶婧愰」娌胯礋娉曞悜锛堟硶鍚戞寚鍚戝渚э紝婧愰」琛ュ伩搴旀寚鍚戝崟鍏冨唴閮ㄦ柟鍚戯級
        // 瀹為檯涓婃牴鎹畧鎭掑緥鐨勬帹瀵硷紝宸︿晶琛ュ伩搴旀部璐熸硶鍚?
        BedSlopeCorrection {
            source_left_x: -pressure_diff_l * normal.x,
            source_left_y: -pressure_diff_l * normal.y,
            source_right_x: pressure_diff_r * normal.x,
            source_right_y: pressure_diff_r * normal.y,
        }
    }

    // =========================================================================
    // 鐘舵€佹洿鏂?
    // =========================================================================

    fn update_state(&self, state: &mut ShallowWaterState, dt: f64) {
        let n = state.n_cells();

        for i in 0..n {
            let area = self.mesh.cell_area(i).unwrap_or(1.0);
            let inv_area = 1.0 / area;

            state.h[i] += dt * inv_area * self.workspace.flux_h[i];
            state.hu[i] += dt * inv_area * (self.workspace.flux_hu[i] + self.workspace.source_hu[i]);
            state.hv[i] += dt * inv_area * (self.workspace.flux_hv[i] + self.workspace.source_hv[i]);
        }
    }

    // =========================================================================
    // 姝ｆ€т繚鎸?
    // =========================================================================

    fn enforce_positivity(&mut self, state: &mut ShallowWaterState, _dt: f64) -> (usize, usize) {
        let h_min = self.config.params.h_min;
        let h_dry = self.config.params.h_dry;
        let mut dry_count = 0;
        let mut limited_count = 0;

        for i in 0..state.n_cells() {
            if state.h[i] < h_min {
                // 璐熸按娣变慨姝?
                state.h[i] = 0.0;
                state.hu[i] = 0.0;
                state.hv[i] = 0.0;
                dry_count += 1;
            } else if state.h[i] < h_dry {
                // 骞叉箍杩囨浮鍖哄姩閲忚“鍑?
                let factor = self.wetting_drying.wet_fraction_smooth(state.h[i]);
                state.hu[i] *= factor;
                state.hv[i] *= factor;
                dry_count += 1;
                limited_count += 1;
            }
        }

        (dry_count, limited_count)
    }

    // =========================================================================
    // 璁块棶鍣?
    // =========================================================================

    /// 鑾峰彇缃戞牸
    pub fn mesh(&self) -> &PhysicsMesh {
        &self.mesh
    }

    /// 鑾峰彇閰嶇疆
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// 鑾峰彇缁熻淇℃伅
    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }

    /// 鑾峰彇鏈€澶ф尝閫?
    pub fn max_wave_speed(&self) -> f64 {
        self.stats.max_wave_speed
    }

    /// 鑾峰彇骞插崟鍏冩暟閲?
    pub fn dry_cell_count(&self) -> usize {
        self.stats.dry_cells
    }
}

// ============================================================
// 姹傝В鍣ㄦ瀯寤哄櫒
// ============================================================

/// 姹傝В鍣ㄦ瀯寤哄櫒
pub struct SolverBuilder {
    mesh: Option<Arc<PhysicsMesh>>,
    config: SolverConfig,
}

impl SolverBuilder {
    /// 鍒涘缓鏋勫缓鍣?
    pub fn new() -> Self {
        Self {
            mesh: None,
            config: SolverConfig::default(),
        }
    }

    /// 璁剧疆缃戞牸
    pub fn mesh(mut self, mesh: Arc<PhysicsMesh>) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// 璁剧疆閰嶇疆
    pub fn config(mut self, config: SolverConfig) -> Self {
        self.config = config;
        self
    }

    /// 璁剧疆鏁板€煎弬鏁?
    pub fn params(mut self, params: NumericalParams) -> Self {
        self.config.params = params;
        self
    }

    /// 璁剧疆閲嶅姏鍔犻€熷害
    pub fn gravity(mut self, g: f64) -> Self {
        self.config.gravity = g;
        self
    }

    /// 璁剧疆鏄惁浣跨敤闈欐按閲嶆瀯
    pub fn use_hydrostatic_reconstruction(mut self, enable: bool) -> Self {
        self.config.use_hydrostatic_reconstruction = enable;
        self
    }

    /// 鏋勫缓姹傝В鍣?
    pub fn build(self) -> Result<ShallowWaterSolver, &'static str> {
        let mesh = self.mesh.ok_or("mesh is required")?;
        Ok(ShallowWaterSolver::new(mesh, self.config))
    }
}

impl Default for SolverBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// 娴嬭瘯
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mh_geo::{Point2D, Point3D};
    use mh_mesh::FrozenMesh;

    /// 鍒涘缓绠€鍗曠殑 2x2 缃戞牸鐢ㄤ簬娴嬭瘯
    /// 
    /// 缃戞牸甯冨眬:
    /// ```text
    ///   +---+---+
    ///   | 2 | 3 |
    ///   +---+---+
    ///   | 0 | 1 |
    ///   +---+---+
    /// ```
    /// 
    /// 鑺傜偣 (9涓?:
    /// ```text
    ///   6---7---8
    ///   |   |   |
    ///   3---4---5
    ///   |   |   |
    ///   0---1---2
    /// ```
    /// 
    /// 鍐呴儴闈?(4鏉?: 杩炴帴鐩搁偦鍗曞厓
    /// 杈圭晫闈?(8鏉?: 澶栬竟鐣?
    fn create_simple_mesh() -> PhysicsMesh {
        // 鍗曞厓灏哄
        let dx = 1.0;
        let dy = 1.0;
        
        // 鑺傜偣鍧愭爣 (3x3 = 9涓妭鐐?
        let node_coords = vec![
            Point3D::new(0.0, 0.0, 0.0), // 0
            Point3D::new(dx, 0.0, 0.0),  // 1
            Point3D::new(2.0*dx, 0.0, 0.0), // 2
            Point3D::new(0.0, dy, 0.0),  // 3
            Point3D::new(dx, dy, 0.0),   // 4
            Point3D::new(2.0*dx, dy, 0.0), // 5
            Point3D::new(0.0, 2.0*dy, 0.0), // 6
            Point3D::new(dx, 2.0*dy, 0.0),  // 7
            Point3D::new(2.0*dx, 2.0*dy, 0.0), // 8
        ];
        
        // 鍗曞厓涓績
        let cell_center = vec![
            Point2D::new(0.5*dx, 0.5*dy), // 鍗曞厓0
            Point2D::new(1.5*dx, 0.5*dy), // 鍗曞厓1
            Point2D::new(0.5*dx, 1.5*dy), // 鍗曞厓2
            Point2D::new(1.5*dx, 1.5*dy), // 鍗曞厓3
        ];
        
        // 鍗曞厓闈㈢Н
        let cell_area = vec![dx*dy, dx*dy, dx*dy, dx*dy];
        
        // 鍗曞厓搴曞簥楂樼▼ (骞冲簳)
        let cell_z_bed = vec![0.0, 0.0, 0.0, 0.0];
        
        // 鍐呴儴闈?(4鏉?
        // 闈?: 鍗曞厓0-1 (鍨傜洿闈紝娉曞悜x姝?
        // 闈?: 鍗曞厓0-2 (姘村钩闈紝娉曞悜y姝?
        // 闈?: 鍗曞厓1-3 (姘村钩闈紝娉曞悜y姝?
        // 闈?: 鍗曞厓2-3 (鍨傜洿闈紝娉曞悜x姝?
        
        // 杈圭晫闈?(8鏉?
        // 闈?-5: 涓嬭竟鐣?(鍗曞厓0,1)
        // 闈?-7: 鍙宠竟鐣?(鍗曞厓1,3)
        // 闈?-9: 涓婅竟鐣?(鍗曞厓2,3)
        // 闈?0-11: 宸﹁竟鐣?(鍗曞厓0,2)
        
        let n_interior = 4;
        let n_boundary = 8;
        let n_faces = n_interior + n_boundary;
        
        // 闈㈡暟鎹?
        let face_center = vec![
            // 鍐呴儴闈?
            Point2D::new(dx, 0.5*dy),     // 0: 鍗曞厓0-1
            Point2D::new(0.5*dx, dy),     // 1: 鍗曞厓0-2
            Point2D::new(1.5*dx, dy),     // 2: 鍗曞厓1-3
            Point2D::new(dx, 1.5*dy),     // 3: 鍗曞厓2-3
            // 杈圭晫闈?
            Point2D::new(0.5*dx, 0.0),    // 4: 涓?鍗曞厓0
            Point2D::new(1.5*dx, 0.0),    // 5: 涓?鍗曞厓1
            Point2D::new(2.0*dx, 0.5*dy), // 6: 鍙?鍗曞厓1
            Point2D::new(2.0*dx, 1.5*dy), // 7: 鍙?鍗曞厓3
            Point2D::new(0.5*dx, 2.0*dy), // 8: 涓?鍗曞厓2
            Point2D::new(1.5*dx, 2.0*dy), // 9: 涓?鍗曞厓3
            Point2D::new(0.0, 0.5*dy),    // 10: 宸?鍗曞厓0
            Point2D::new(0.0, 1.5*dy),    // 11: 宸?鍗曞厓2
        ];
        
        let face_normal = vec![
            // 鍐呴儴闈?
            Point3D::new(1.0, 0.0, 0.0),  // 0: x姝?
            Point3D::new(0.0, 1.0, 0.0),  // 1: y姝?
            Point3D::new(0.0, 1.0, 0.0),  // 2: y姝?
            Point3D::new(1.0, 0.0, 0.0),  // 3: x姝?
            // 杈圭晫闈?(澶栧悜娉曞悜)
            Point3D::new(0.0, -1.0, 0.0), // 4: 涓?
            Point3D::new(0.0, -1.0, 0.0), // 5: 涓?
            Point3D::new(1.0, 0.0, 0.0),  // 6: 鍙?
            Point3D::new(1.0, 0.0, 0.0),  // 7: 鍙?
            Point3D::new(0.0, 1.0, 0.0),  // 8: 涓?
            Point3D::new(0.0, 1.0, 0.0),  // 9: 涓?
            Point3D::new(-1.0, 0.0, 0.0), // 10: 宸?
            Point3D::new(-1.0, 0.0, 0.0), // 11: 宸?
        ];
        
        let face_length = vec![
            dy, dx, dx, dy,              // 鍐呴儴闈?
            dx, dx, dy, dy, dx, dx, dy, dy, // 杈圭晫闈?
        ];
        
        let face_owner = vec![
            0, 0, 1, 2,                  // 鍐呴儴闈?
            0, 1, 1, 3, 2, 3, 0, 2,      // 杈圭晫闈?
        ];
        
        let face_neighbor: Vec<u32> = vec![
            1, 2, 3, 3,                  // 鍐呴儴闈?(鏈夐偦灞?
            u32::MAX, u32::MAX, u32::MAX, u32::MAX, // 杈圭晫闈?(鏃犻偦灞?
            u32::MAX, u32::MAX, u32::MAX, u32::MAX,
        ];
        
        // 闈㈤珮绋?
        let face_z_left = vec![0.0; n_faces];
        let face_z_right = vec![0.0; n_faces];
        
        // 闈㈠埌鍗曞厓涓績鐨勫悜閲?
        let face_delta_owner = vec![Point2D::new(0.0, 0.0); n_faces];
        let face_delta_neighbor = vec![Point2D::new(0.0, 0.0); n_faces];
        let face_dist_o2n = vec![dx; n_faces];
        
        // 鍗曞厓鎷撴墤 (绠€鍖栫増)
        let cell_node_offsets = vec![0, 4, 8, 12, 16];
        let cell_node_indices = vec![
            0, 1, 4, 3,  // 鍗曞厓0
            1, 2, 5, 4,  // 鍗曞厓1
            3, 4, 7, 6,  // 鍗曞厓2
            4, 5, 8, 7,  // 鍗曞厓3
        ];
        
        let cell_face_offsets = vec![0, 4, 8, 12, 16];
        let cell_face_indices = vec![
            0, 1, 4, 10,   // 鍗曞厓0
            0, 2, 5, 6,    // 鍗曞厓1
            1, 3, 8, 11,   // 鍗曞厓2
            2, 3, 7, 9,    // 鍗曞厓3
        ];
        
        let cell_neighbor_offsets = vec![0, 2, 4, 6, 8];
        let cell_neighbor_indices = vec![
            1, 2,      // 鍗曞厓0鐨勯偦灞?
            0, 3,      // 鍗曞厓1鐨勯偦灞?
            0, 3,      // 鍗曞厓2鐨勯偦灞?
            1, 2,      // 鍗曞厓3鐨勯偦灞?
        ];
        
        let frozen = FrozenMesh {
            n_nodes: 9,
            node_coords,
            n_cells: 4,
            cell_center,
            cell_area,
            cell_z_bed,
            cell_node_offsets,
            cell_node_indices,
            cell_face_offsets,
            cell_face_indices,
            cell_neighbor_offsets,
            cell_neighbor_indices,
            n_faces,
            n_interior_faces: n_interior,
            face_center,
            face_normal,
            face_length,
            face_z_left,
            face_z_right,
            face_owner,
            face_neighbor,
            face_delta_owner,
            face_delta_neighbor,
            face_dist_o2n,
            boundary_face_indices: (4..12).map(|i| i as u32).collect(),
            boundary_names: vec!["boundary".to_string()],
            face_boundary_id: (0..n_faces).map(|i| if i >= 4 { Some(0) } else { None }).collect(),
            min_cell_size: dx.min(dy),
            max_cell_size: dx.max(dy),
            // AMR 棰勫垎閰嶅瓧娈?
            cell_refinement_level: vec![0; 4],
            cell_parent: vec![0, 1, 2, 3],
            ghost_capacity: 0,
            // ID 鏄犲皠涓庢帓鍒楀瓧娈?
            cell_original_id: Vec::new(),
            face_original_id: Vec::new(),
            cell_permutation: Vec::new(),
            cell_inv_permutation: Vec::new(),
        };
        
        PhysicsMesh::from_frozen(&frozen)
    }

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert!((config.gravity - 9.81).abs() < 1e-10);
        assert!(config.use_hydrostatic_reconstruction);
        assert!(config.implicit_friction);
    }

    #[test]
    fn test_solver_config_builder() {
        let config = SolverConfig::builder()
            .gravity(10.0)
            .use_hydrostatic_reconstruction(false)
            .parallel_threshold(5000)
            .build();

        assert!((config.gravity - 10.0).abs() < 1e-10);
        assert!(!config.use_hydrostatic_reconstruction);
        assert_eq!(config.parallel_threshold, 5000);
    }

    #[test]
    fn test_solver_workspace() {
        let mut ws = SolverWorkspace::new(10);
        assert_eq!(ws.flux_h.len(), 10);

        ws.flux_h[0] = 1.0;
        ws.reset_fluxes();
        assert_eq!(ws.flux_h[0], 0.0);

        ws.resize(20);
        assert_eq!(ws.flux_h.len(), 20);
    }

    #[test]
    fn test_hydrostatic_reconstruction() {
        let params = NumericalParams::default();
        let hydro = HydrostaticReconstruction::new(&params, 9.81);

        // 骞冲簳鎯呭喌
        let recon = hydro.reconstruct_face_simple(
            1.0, 1.0, 
            0.0, 0.0, 
            DVec2::ZERO, DVec2::ZERO
        );
        assert!((recon.h_left - 1.0).abs() < 1e-10);
        assert!((recon.h_right - 1.0).abs() < 1e-10);

        // 楂樼▼宸儏鍐?
        let recon = hydro.reconstruct_face_simple(
            1.0, 0.5,
            0.0, 0.5,
            DVec2::ZERO, DVec2::ZERO
        );
        // z_face = max(0, 0.5) = 0.5
        // h_left = max(0, (1.0 + 0.0) - 0.5) = 0.5
        // h_right = max(0, (0.5 + 0.5) - 0.5) = 0.5
        assert!((recon.h_left - 0.5).abs() < 1e-10);
        assert!((recon.h_right - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bed_slope_correction() {
        // 娴嬭瘯鏃х殑 bed_slope_correction 鏂规硶锛堜繚鐣欎緵鍙傝€冿級
        let params = NumericalParams::default();
        let hydro = HydrostaticReconstruction::new(&params, 9.81);

        let correction = hydro.bed_slope_correction(
            1.0, 1.0,
            0.0, 1.0,
            DVec2::X,
            1.0,
        );

        // S = -g * h_face * dz * L * n
        // dz = 1.0, h_face = 1.0, L = 1.0, n = (1, 0)
        // S_x = -9.81 * 1.0 * 1.0 * 1.0 * 1.0 = -9.81
        // 宸﹀彸鍚勫垎涓€鍗? left = 0.5 * (-9.81) = -4.905
        assert!((correction.source_left_x - (-4.905)).abs() < 1e-10);
        assert!(correction.source_left_y.abs() < 1e-10);
        assert!((correction.source_right_x - 4.905).abs() < 1e-10);
        assert!(correction.source_right_y.abs() < 1e-10);
    }

    #[test]
    fn test_solver_builder() {
        let builder = SolverBuilder::new()
            .gravity(10.0)
            .use_hydrostatic_reconstruction(false);

        assert!((builder.config.gravity - 10.0).abs() < 1e-10);
        assert!(!builder.config.use_hydrostatic_reconstruction);

        // 娌℃湁璁剧疆 mesh锛屾瀯寤哄簲璇ュけ璐?
        assert!(builder.build().is_err());
    }

    // =========================================================================
    // TD-5.3.1: Solver鏍稿績鏂规硶娴嬭瘯
    // =========================================================================

    #[test]
    fn test_simple_mesh_creation() {
        let mesh = create_simple_mesh();
        assert_eq!(mesh.n_cells(), 4);
        assert_eq!(mesh.n_faces(), 12);
        assert_eq!(mesh.n_interior_faces(), 4);
        assert_eq!(mesh.n_boundary_faces(), 8);
    }

    #[test]
    fn test_solver_creation_with_mesh() {
        let mesh = create_simple_mesh();
        let config = SolverConfig::default();
        let solver = ShallowWaterSolver::new(Arc::new(mesh), config);

        assert_eq!(solver.mesh().n_cells(), 4);
        assert!((solver.config().gravity - 9.81).abs() < 1e-10);
    }

    #[test]
    fn test_solver_step_still_water() {
        // 娴嬭瘯闈欐按鎯呭喌: 姘存繁鍧囧寑锛屾棤閫熷害锛屾棤楂樼▼宸?
        // 棰勬湡: 鐘舵€佸簲淇濇寔涓嶅彉锛堝钩琛℃€侊級
        
        let mesh = Arc::new(create_simple_mesh());
        let config = SolverConfig::builder()
            .parallel_threshold(100) // 寮哄埗涓茶浠ヤ究璋冭瘯
            .build();
        let mut solver = ShallowWaterSolver::new(mesh.clone(), config);
        
        // 鍒涘缓闈欐按鐘舵€?
        let h0 = 1.0;
        let mut state = ShallowWaterState::new(4);
        for i in 0..4 {
            state.h[i] = h0;
            state.hu[i] = 0.0;
            state.hv[i] = 0.0;
            state.z[i] = 0.0;
        }
        
        // 璁板綍鍒濆璐ㄩ噺
        let initial_mass: f64 = state.h.iter().sum();
        
        // 鎵ц涓€姝?
        let dt = 0.001;
        solver.step(&mut state, dt);
        
        // 楠岃瘉璐ㄩ噺瀹堟亽
        let final_mass: f64 = state.h.iter().sum();
        let mass_error = (final_mass - initial_mass).abs() / initial_mass;
        assert!(mass_error < 1e-10, "璐ㄩ噺瀹堟亽璇樊: {}", mass_error);
        
        // 楠岃瘉闈欐按淇濇寔锛堥€氶噺搴旀帴杩戦浂锛?
        for i in 0..4 {
            assert!((state.h[i] - h0).abs() < 1e-10, 
                "鍗曞厓{} 姘存繁鍙樺寲杩囧ぇ: {} -> {}", i, h0, state.h[i]);
            assert!(state.hu[i].abs() < 1e-10, 
                "鍗曞厓{} 鍑虹幇闈為浂x鍔ㄩ噺: {}", i, state.hu[i]);
            assert!(state.hv[i].abs() < 1e-10,
                "鍗曞厓{} 鍑虹幇闈為浂y鍔ㄩ噺: {}", i, state.hv[i]);
        }
    }

    #[test]
    fn test_solver_step_uniform_flow() {
        // 娴嬭瘯鍧囧寑娴佹儏鍐? 姘存繁鍧囧寑锛屾湁鍧囧寑閫熷害
        // 棰勬湡: 璐ㄩ噺瀹堟亽
        
        let mesh = Arc::new(create_simple_mesh());
        let config = SolverConfig::builder()
            .parallel_threshold(100)
            .build();
        let mut solver = ShallowWaterSolver::new(mesh.clone(), config);
        
        // 鍒涘缓鍧囧寑娴佺姸鎬?
        let h0 = 1.0;
        let u0 = 0.1;  // 灏忛€熷害
        let mut state = ShallowWaterState::new(4);
        for i in 0..4 {
            state.h[i] = h0;
            state.hu[i] = h0 * u0;  // hu = h * u
            state.hv[i] = 0.0;
            state.z[i] = 0.0;
        }
        
        let initial_mass: f64 = state.h.iter().sum();
        
        // 鎵ц涓€姝?
        let dt = 0.001;
        solver.step(&mut state, dt);
        
        // 楠岃瘉璐ㄩ噺瀹堟亽锛堝厑璁歌竟鐣屾祦鍑?娴佸叆鐨勫奖鍝嶏級
        let final_mass: f64 = state.h.iter().sum();
        // 杈圭晫浣跨敤鍙嶅皠鏉′欢锛屾墍浠ヨ川閲忓簲璇ュ畧鎭?
        let mass_error = (final_mass - initial_mass).abs() / initial_mass;
        assert!(mass_error < 1e-6, "璐ㄩ噺瀹堟亽璇樊: {}", mass_error);
        
        // 楠岃瘉姘存繁淇濇寔姝ｅ€?
        for i in 0..4 {
            assert!(state.h[i] > 0.0, "鍗曞厓{} 姘存繁涓鸿礋: {}", i, state.h[i]);
        }
    }

    #[test]
    fn test_solver_step_dam_break() {
        // 娴嬭瘯婧冨潩鎯呭喌: 宸﹁竟楂樻按娣憋紝鍙宠竟浣庢按娣?
        // 棰勬湡: 姘翠粠宸﹀悜鍙虫祦鍔?
        
        let mesh = Arc::new(create_simple_mesh());
        let config = SolverConfig::builder()
            .parallel_threshold(100)
            .build();
        let mut solver = ShallowWaterSolver::new(mesh.clone(), config);
        
        // 鍒涘缓婧冨潩鍒濆鏉′欢
        // 鍗曞厓0,2 (宸︿晶) 姘存繁楂?
        // 鍗曞厓1,3 (鍙充晶) 姘存繁浣?
        let h_left = 2.0;
        let h_right = 1.0;
        let mut state = ShallowWaterState::new(4);
        state.h[0] = h_left;
        state.h[1] = h_right;
        state.h[2] = h_left;
        state.h[3] = h_right;
        for i in 0..4 {
            state.hu[i] = 0.0;
            state.hv[i] = 0.0;
            state.z[i] = 0.0;
        }
        
        let initial_mass: f64 = state.h.iter().sum();
        
        // 鎵ц澶氭
        let dt = 0.001;
        for _ in 0..10 {
            solver.step(&mut state, dt);
        }
        
        // 楠岃瘉璐ㄩ噺瀹堟亽
        let final_mass: f64 = state.h.iter().sum();
        let mass_error = (final_mass - initial_mass).abs() / initial_mass;
        assert!(mass_error < 1e-6, "璐ㄩ噺瀹堟亽璇樊: {}", mass_error);
        
        // 楠岃瘉姘存祦鏂瑰悜姝ｇ‘ (宸︿晶姘存繁搴斿噺灏戞垨鍔ㄩ噺鍚戝彸)
        // 鐢变簬婧冨潩锛屽乏渚у崟鍏冨簲璇ユ湁姝ｇ殑x鍔ㄩ噺锛堝悜鍙虫祦鍔級
        // 杩欐槸涓€涓畾鎬ф祴璇?
        let left_momentum: f64 = state.hu[0] + state.hu[2];
        let right_momentum: f64 = state.hu[1] + state.hu[3];
        
        // 鐢变簬杈圭晫鍙嶅皠锛屼笉瀹规槗鐩存帴楠岃瘉娴佸悜
        // 浣嗚嚦灏戝簲璇ヤ骇鐢熶簡鍔ㄩ噺
        let total_momentum = left_momentum.abs() + right_momentum.abs();
        assert!(total_momentum > 1e-6, "搴旇浜х敓鍔ㄩ噺锛屼絾 total_momentum = {}", total_momentum);
    }

    #[test]
    fn test_solver_compute_fluxes_serial() {
        // 鐩存帴娴嬭瘯閫氶噺璁＄畻鏂规硶
        
        let mesh = Arc::new(create_simple_mesh());
        let config = SolverConfig::builder()
            .parallel_threshold(10000) // 寮哄埗涓茶
            .scheme(NumericalScheme::FirstOrder) // 浣跨敤涓€闃堕伩鍏嶉噸鏋勪緷璧?
            .build();
        let mut solver = ShallowWaterSolver::new(mesh.clone(), config);
        
        // 鍒涘缓鏈夋按娣卞樊鐨勭姸鎬?
        let mut state = ShallowWaterState::new(4);
        state.h[0] = 2.0;
        state.h[1] = 1.0;
        state.h[2] = 2.0;
        state.h[3] = 1.0;
        for i in 0..4 {
            state.hu[i] = 0.0;
            state.hv[i] = 0.0;
            state.z[i] = 0.0;
        }
        
        // 閲嶇疆宸ヤ綔绌洪棿
        solver.workspace.reset();
        
        // 璁＄畻閫氶噺
        let max_speed = solver.compute_fluxes_serial(&state);
        
        // 楠岃瘉鏈€澶ф尝閫熷悎鐞?
        // 娉㈤€?c = sqrt(g*h)锛屽浜?h=2, c 鈮?4.43
        assert!(max_speed > 0.0, "鏈€澶ф尝閫熷簲澶т簬0, 瀹為檯: {}", max_speed);
        assert!(max_speed < 10.0, "鏈€澶ф尝閫熷簲鍚堢悊: {}", max_speed);
        
        // 楠岃瘉閫氶噺闈為浂
        let total_flux: f64 = solver.workspace.flux_h.iter().map(|x| x.abs()).sum();
        assert!(total_flux > 1e-10, "搴旇浜х敓闈為浂閫氶噺");
    }

    #[test]
    fn test_solver_compute_fluxes_parallel() {
        // 娴嬭瘯骞惰閫氶噺璁＄畻涓庝覆琛岀粨鏋滀竴鑷?
        
        let mesh = Arc::new(create_simple_mesh());
        
        // 鍒涘缓鐘舵€?
        let mut state = ShallowWaterState::new(4);
        state.h[0] = 2.0;
        state.h[1] = 1.0;
        state.h[2] = 1.5;
        state.h[3] = 1.2;
        for i in 0..4 {
            state.hu[i] = 0.0;
            state.hv[i] = 0.0;
            state.z[i] = 0.0;
        }
        
        // 涓茶璁＄畻锛堜娇鐢ㄤ竴闃堕伩鍏嶉噸鏋勪緷璧栵級
        let config_serial = SolverConfig::builder()
            .parallel_threshold(10000)
            .scheme(NumericalScheme::FirstOrder)
            .build();
        let mut solver_serial = ShallowWaterSolver::new(mesh.clone(), config_serial);
        solver_serial.workspace.reset();
        let max_speed_serial = solver_serial.compute_fluxes_serial(&state);
        let flux_h_serial = solver_serial.workspace.flux_h.clone();
        
        // 骞惰璁＄畻锛堝悓鏍蜂娇鐢ㄤ竴闃讹級
        let config_parallel = SolverConfig::builder()
            .parallel_threshold(0)
            .scheme(NumericalScheme::FirstOrder)
            .build();
        let mut solver_parallel = ShallowWaterSolver::new(mesh.clone(), config_parallel);
        solver_parallel.workspace.reset();
        let max_speed_parallel = solver_parallel.compute_fluxes_parallel(&state);
        let flux_h_parallel = solver_parallel.workspace.flux_h.clone();
        
        // 楠岃瘉缁撴灉涓€鑷?
        assert!((max_speed_serial - max_speed_parallel).abs() < 1e-10,
            "鏈€澶ф尝閫熶笉涓€鑷? serial={}, parallel={}", max_speed_serial, max_speed_parallel);
        
        for i in 0..4 {
            assert!((flux_h_serial[i] - flux_h_parallel[i]).abs() < 1e-10,
                "鍗曞厓{} 璐ㄩ噺閫氶噺涓嶄竴鑷? serial={}, parallel={}", 
                i, flux_h_serial[i], flux_h_parallel[i]);
        }
    }

    #[test]
    fn test_solver_positivity_enforcement() {
        // 娴嬭瘯姝ｆ€т繚鎸?
        
        let mesh = Arc::new(create_simple_mesh());
        let config = SolverConfig::default();
        let mut solver = ShallowWaterSolver::new(mesh.clone(), config);
        
        // 鍒涘缓鏈夎礋姘存繁鐨勭姸鎬?
        let mut state = ShallowWaterState::new(4);
        state.h[0] = -0.01;  // 璐熸按娣?
        state.h[1] = 0.001;  // 鏋佸皬姘存繁
        state.h[2] = 1.0;    // 姝ｅ父姘存繁
        state.h[3] = 0.0;    // 闆舵按娣?
        state.hu[0] = 1.0;
        state.hu[1] = 0.1;
        
        let (dry_count, _) = solver.enforce_positivity(&mut state, 0.001);
        
        // 楠岃瘉璐熸按娣辫淇
        assert!(state.h[0] >= 0.0, "璐熸按娣辨湭琚慨姝?);
        assert!(state.h[1] >= 0.0, "鏋佸皬姘存繁鍑洪棶棰?);
        
        // 楠岃瘉鍔ㄩ噺琚“鍑?
        assert_eq!(state.hu[0], 0.0, "骞插崟鍏冨姩閲忓簲涓?");
        
        // 楠岃瘉缁熻姝ｇ‘
        assert!(dry_count >= 2, "搴旇妫€娴嬪埌鑷冲皯2涓共鍗曞厓");
    }

    #[test]
    fn test_solver_stats() {
        let mesh = Arc::new(create_simple_mesh());
        let config = SolverConfig::default();
        let mut solver = ShallowWaterSolver::new(mesh.clone(), config);
        
        let mut state = ShallowWaterState::new(4);
        for i in 0..4 {
            state.h[i] = 1.0;
            state.hu[i] = 0.0;
            state.hv[i] = 0.0;
            state.z[i] = 0.0;
        }
        
        solver.step(&mut state, 0.001);
        
        let stats = solver.stats();
        assert!(stats.max_wave_speed >= 0.0);
        assert!(stats.dt > 0.0);
    }
}
