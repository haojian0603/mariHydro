// crates/mh_physics/src/engine/parallel.rs

//! 骞惰閫氶噺璁＄畻妯″潡
//!
//! 鎻愪緵澶氱骞惰绛栫暐鐢ㄤ簬鍔犻€熼€氶噺璁＄畻锛?
//! - 涓茶璁＄畻锛堝皬瑙勬ā闂锛?
//! - 鏀堕泦鍚庣疮鍔狅紙鍏堝苟琛岃绠楅€氶噺锛屽悗涓茶绱姞鍒板崟鍏冿級
//! - 鐫€鑹插苟琛岋紙浣跨敤鍥剧潃鑹插疄鐜扮湡姝ｆ棤閿佸苟琛岋紝TODO锛?
//!
//! # 杩佺Щ璇存槑
//!
//! 浠?history_src/physics/engine/parallel.rs 绠€鍖栬縼绉汇€?
//! 瀹屾暣鐨勭潃鑹插苟琛岀瓑楂樼骇鍔熻兘灏嗗湪鍚庣画鐗堟湰瀹炵幇銆?
//!
//! # 鎶€鏈€哄姟 (TD-5.3.2, TD-5.3.3)
//!
//! 褰撳墠瀹炵幇鐨?骞惰"鏄吉骞惰锛氶€氶噺璁＄畻骞惰锛屼絾绱姞闃舵涓茶銆?
//! 瀵逛簬澶ц妯＄綉鏍硷紝闇€瑕佸疄鐜扮湡姝ｇ殑鐫€鑹插苟琛屼互閬垮厤绱姞鐡堕銆?

use crate::adapter::PhysicsMesh;
use crate::engine::solver::{BedSlopeCorrection, HydrostaticFaceState, HydrostaticReconstruction};
use crate::schemes::{HllcSolver, RiemannFlux, RiemannSolver};
use crate::schemes::wetting_drying::{WetState, WettingDryingHandler};
use crate::state::ShallowWaterState;
use crate::types::NumericalParams;

use glam::DVec2;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ============================================================
// 閰嶇疆
// ============================================================

/// 骞惰绛栫暐
///
/// # 绛栫暐璇存槑
///
/// - `Sequential`: 瀹屽叏涓茶鎵ц锛岄€傜敤浜庡皬瑙勬ā闂
/// - `CollectThenAccumulate`: 鍏堝苟琛岃绠楀悇闈㈤€氶噺(鐪熸骞惰)锛?
///   鐒跺悗涓茶绱姞鍒板崟鍏?鐡堕)銆傚浜庝腑绛夎妯￠棶棰樻湁鏁堛€?
/// - `Colored`: 浣跨敤鍥剧潃鑹插疄鐜扮湡姝ｇ殑鏃犻攣骞惰绱姞
/// - `Auto`: 鏍规嵁闈㈡暟鑷姩閫夋嫨绛栫暐
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum ParallelStrategy {
    /// 涓茶鎵ц
    Sequential,
    /// 鏀堕泦鍚庣疮鍔狅細骞惰璁＄畻閫氶噺 鈫?鏀堕泦缁撴灉 鈫?涓茶绱姞
    ///
    /// 娉ㄦ剰锛氱疮鍔犻樁娈垫槸涓茶鐨勶紝瀵逛簬澶ц妯＄綉鏍煎彲鑳芥垚涓虹摱棰?
    CollectThenAccumulate,
    /// 鐫€鑹插苟琛岋細浣跨敤鍥剧潃鑹插垎缁勯潰锛屽悓涓€棰滆壊鐨勯潰鍙畨鍏ㄥ苟琛屽鐞?
    /// 
    /// 杩欐槸鎺ㄨ崘鐨勫ぇ瑙勬ā骞惰绛栫暐锛岄渶瑕侀鍏堣绠楅潰鐫€鑹?
    Colored,
    /// 鑷姩閫夋嫨锛堟牴鎹棶棰樿妯★級
    #[default]
    Auto,
}


/// 骞惰璁＄畻閰嶇疆
#[derive(Debug, Clone)]
pub struct ParallelFluxConfig {
    /// 鏁板€煎弬鏁?
    pub params: NumericalParams,
    /// 閲嶅姏鍔犻€熷害
    pub g: f64,
    /// 鏈€灏忓苟琛岄潰鏁帮紙浣庝簬姝ゅ€间娇鐢ㄤ覆琛岋級
    pub min_parallel_size: usize,
    /// 骞惰绛栫暐
    pub strategy: ParallelStrategy,
    /// 鏄惁鍚敤闈欐按閲嶆瀯
    pub use_hydrostatic_reconstruction: bool,
}

impl Default for ParallelFluxConfig {
    fn default() -> Self {
        Self {
            params: NumericalParams::default(),
            g: 9.81,
            min_parallel_size: 1000,
            strategy: ParallelStrategy::Auto,
            use_hydrostatic_reconstruction: true,
        }
    }
}

impl ParallelFluxConfig {
    /// 鍒涘缓鏋勫缓鍣?
    pub fn builder() -> ParallelFluxConfigBuilder {
        ParallelFluxConfigBuilder::default()
    }
}

/// 閰嶇疆鏋勫缓鍣?
#[derive(Default)]
pub struct ParallelFluxConfigBuilder {
    config: ParallelFluxConfig,
}

impl ParallelFluxConfigBuilder {
    pub fn params(mut self, params: NumericalParams) -> Self {
        self.config.params = params;
        self
    }

    pub fn gravity(mut self, g: f64) -> Self {
        self.config.g = g;
        self
    }

    pub fn min_parallel_size(mut self, size: usize) -> Self {
        self.config.min_parallel_size = size;
        self
    }

    pub fn strategy(mut self, strategy: ParallelStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn use_hydrostatic_reconstruction(mut self, enable: bool) -> Self {
        self.config.use_hydrostatic_reconstruction = enable;
        self
    }

    pub fn build(self) -> ParallelFluxConfig {
        self.config
    }
}

// ============================================================
// 鎬ц兘鎸囨爣
// ============================================================

/// 鎬ц兘鎸囨爣
#[derive(Debug, Clone, Default)]
pub struct FluxComputeMetrics {
    /// 鎬昏绠楁鏁?
    pub total_calls: usize,
    /// 骞惰璁＄畻娆℃暟
    pub parallel_calls: usize,
    /// 涓茶璁＄畻娆℃暟
    pub sequential_calls: usize,
    /// 鎬昏绠楁椂闂?
    pub total_duration: Duration,
    /// 澶勭悊鐨勯潰鎬绘暟
    pub total_faces: usize,
}

impl FluxComputeMetrics {
    /// 璁板綍涓€娆¤绠?
    // TODO(phase5): 娣诲姞绛栫暐閫夋嫨鐨勮缁嗘棩蹇楋紙濡?legacy 鐨?StrategySelector锛?
    pub fn record(&mut self, n_faces: usize, is_parallel: bool, duration: Duration) {
        self.total_calls += 1;
        self.total_faces += n_faces;
        self.total_duration += duration;
        if is_parallel {
            self.parallel_calls += 1;
        } else {
            self.sequential_calls += 1;
        }
    }

    /// 閲嶇疆鎸囨爣
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// 骞冲潎姣忛潰璁＄畻鏃堕棿
    pub fn avg_time_per_face(&self) -> Duration {
        if self.total_faces > 0 {
            self.total_duration / self.total_faces as u32
        } else {
            Duration::ZERO
        }
    }
}

// ============================================================
// 骞惰閫氶噺璁＄畻鍣?
// ============================================================

/// 骞惰閫氶噺璁＄畻鍣?
///
/// 灏佽閫氶噺璁＄畻鐨勫苟琛屾墽琛岄€昏緫銆?
pub struct ParallelFluxCalculator {
    config: ParallelFluxConfig,
    /// 榛庢浖姹傝В鍣?
    riemann: HllcSolver,
    /// 骞叉箍澶勭悊鍣?
    wetting_drying: WettingDryingHandler,
    /// 闈欐按閲嶆瀯
    hydrostatic: HydrostaticReconstruction,
    /// 鎬ц兘鎸囨爣
    metrics: FluxComputeMetrics,
    /// 闈㈢潃鑹诧紙鐢ㄤ簬 Colored 绛栫暐锛?
    /// 姣忎釜鍏冪礌鏄竴缁勫彲浠ュ苟琛屽鐞嗙殑闈㈢储寮?
    face_colors: Option<Vec<Vec<usize>>>,
}

impl ParallelFluxCalculator {
    /// 鍒涘缓璁＄畻鍣?
    pub fn new(config: ParallelFluxConfig) -> Self {
        Self {
            riemann: HllcSolver::new(&config.params, config.g),
            wetting_drying: WettingDryingHandler::from_params(&config.params),
            hydrostatic: HydrostaticReconstruction::new(&config.params, config.g),
            metrics: FluxComputeMetrics::default(),
            face_colors: None,
            config,
        }
    }

    /// 涓虹綉鏍艰缃潰鐫€鑹诧紙鐢ㄤ簬 Colored 绛栫暐锛?
    /// 
    /// 闈㈢潃鑹插皢闈㈠垎鎴愯嫢骞茬粍锛屽悓涓€缁勫唴鐨勯潰涓嶅叡浜崟鍏冿紝
    /// 鍥犳鍙互瀹夊叏鍦板苟琛屾洿鏂拌繖浜涢潰鍏宠仈鐨勫崟鍏冦€?
    /// 
    /// # 鍙傛暟
    /// - `mesh`: 缃戞牸
    pub fn setup_face_coloring(&mut self, mesh: &PhysicsMesh) {
        let n_faces = mesh.n_faces();
        if n_faces == 0 {
            self.face_colors = Some(Vec::new());
            return;
        }

        // 鏋勫缓闈㈢殑閭绘帴鍏崇郴
        // 涓や釜闈㈢浉閭?<=> 瀹冧滑鍏变韩涓€涓崟鍏?
        // 鍗?face_i 鍜?face_j 鐩搁偦褰撲笖浠呭綋锛?
        //   owner(face_i) == owner(face_j) 鎴?
        //   owner(face_i) == neighbor(face_j) 鎴?
        //   neighbor(face_i) == owner(face_j) 鎴?
        //   neighbor(face_i) == neighbor(face_j)
        
        use std::collections::{HashMap, HashSet};
        
        // 鏋勫缓鍗曞厓鍒伴潰鐨勬槧灏?
        let mut cell_to_faces: HashMap<usize, Vec<usize>> = HashMap::new();
        for face_idx in 0..n_faces {
            let owner = mesh.face_owner(face_idx);
            cell_to_faces.entry(owner).or_default().push(face_idx);
            if let Some(neigh) = mesh.face_neighbor(face_idx) {
                cell_to_faces.entry(neigh).or_default().push(face_idx);
            }
        }

        // 鏋勫缓闈㈢殑閭绘帴琛?
        let mut face_neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); n_faces];
        for faces in cell_to_faces.values() {
            // 鍚屼竴鍗曞厓鐨勬墍鏈夐潰浜掍负閭诲眳
            for i in 0..faces.len() {
                for j in (i + 1)..faces.len() {
                    face_neighbors[faces[i]].insert(faces[j]);
                    face_neighbors[faces[j]].insert(faces[i]);
                }
            }
        }

        // 璐績鐫€鑹?
        let mut face_color = vec![usize::MAX; n_faces];
        let mut num_colors = 0;

        // 鎸夐偦灞呮暟閲忔帓搴忥紙楂樺害鏁颁紭鍏堬級
        let mut order: Vec<usize> = (0..n_faces).collect();
        order.sort_by_key(|&f| std::cmp::Reverse(face_neighbors[f].len()));

        for &face in &order {
            // 鎵惧埌閭诲眳浣跨敤鐨勯鑹?
            let used_colors: HashSet<usize> = face_neighbors[face]
                .iter()
                .filter_map(|&n| {
                    if face_color[n] != usize::MAX {
                        Some(face_color[n])
                    } else {
                        None
                    }
                })
                .collect();

            // 鎵惧埌鏈€灏忓彲鐢ㄩ鑹?
            let mut color = 0;
            while used_colors.contains(&color) {
                color += 1;
            }

            face_color[face] = color;
            num_colors = num_colors.max(color + 1);
        }

        // 鎸夐鑹插垎缁勯潰
        let mut color_faces: Vec<Vec<usize>> = vec![Vec::new(); num_colors];
        for (face, &color) in face_color.iter().enumerate() {
            if color != usize::MAX {
                color_faces[color].push(face);
            }
        }

        self.face_colors = Some(color_faces);
    }

    /// 妫€鏌ユ槸鍚﹀凡璁剧疆闈㈢潃鑹?
    pub fn has_face_coloring(&self) -> bool {
        self.face_colors.is_some()
    }

    /// 鑾峰彇棰滆壊鏁伴噺
    pub fn num_colors(&self) -> usize {
        self.face_colors.as_ref().map(|c| c.len()).unwrap_or(0)
    }

    /// 璁＄畻閫氶噺锛堣嚜鍔ㄩ€夋嫨绛栫暐锛?
    pub fn compute_fluxes(
        &mut self,
        state: &ShallowWaterState,
        mesh: &PhysicsMesh,
        flux_h: &mut [f64],
        flux_hu: &mut [f64],
        flux_hv: &mut [f64],
        source_hu: &mut [f64],
        source_hv: &mut [f64],
    ) -> f64 {
        let n_faces = mesh.n_faces();
        let start = Instant::now();

        let (max_speed, is_parallel) = match self.config.strategy {
            ParallelStrategy::Sequential => {
                (self.compute_serial(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), false)
            }
            ParallelStrategy::CollectThenAccumulate => {
                (self.compute_parallel(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), true)
            }
            ParallelStrategy::Colored => {
                // 濡傛灉娌℃湁璁剧疆鐫€鑹诧紝鍏堣缃?
                if !self.has_face_coloring() {
                    self.setup_face_coloring(mesh);
                }
                (self.compute_colored(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), true)
            }
            ParallelStrategy::Auto => {
                if n_faces < self.config.min_parallel_size {
                    (self.compute_serial(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), false)
                } else if self.has_face_coloring() {
                    // 鏈夌潃鑹插氨鐢ㄧ潃鑹插苟琛?
                    (self.compute_colored(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), true)
                } else {
                    // 鍚﹀垯鐢ㄦ敹闆嗗悗绱姞
                    (self.compute_parallel(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), true)
                }
            }
        };

        let duration = start.elapsed();
        self.metrics.record(n_faces, is_parallel, duration);

        max_speed
    }

    /// 涓茶璁＄畻
    fn compute_serial(
        &self,
        state: &ShallowWaterState,
        mesh: &PhysicsMesh,
        flux_h: &mut [f64],
        flux_hu: &mut [f64],
        flux_hv: &mut [f64],
        source_hu: &mut [f64],
        source_hv: &mut [f64],
    ) -> f64 {
        // 閲嶇疆
        flux_h.fill(0.0);
        flux_hu.fill(0.0);
        flux_hv.fill(0.0);
        source_hu.fill(0.0);
        source_hv.fill(0.0);

        let n_faces = mesh.n_faces();
        let mut max_speed = 0.0f64;

        for face_idx in 0..n_faces {
            let (flux, bed_src, length, owner, neighbor) = 
                self.compute_face(state, mesh, face_idx);

            max_speed = max_speed.max(flux.max_wave_speed);

            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;

            flux_h[owner] -= fh;
            flux_hu[owner] -= fhu;
            flux_hv[owner] -= fhv;
            source_hu[owner] += bed_src.source_left_x;
            source_hv[owner] += bed_src.source_left_y;

            if let Some(neigh) = neighbor {
                flux_h[neigh] += fh;
                flux_hu[neigh] += fhu;
                flux_hv[neigh] += fhv;
                source_hu[neigh] += bed_src.source_right_x;
                source_hv[neigh] += bed_src.source_right_y;
            }
        }

        max_speed
    }

    /// 骞惰璁＄畻锛堝厛骞惰璁＄畻锛屽悗涓茶绱姞锛?
    fn compute_parallel(
        &self,
        state: &ShallowWaterState,
        mesh: &PhysicsMesh,
        flux_h: &mut [f64],
        flux_hu: &mut [f64],
        flux_hv: &mut [f64],
        source_hu: &mut [f64],
        source_hv: &mut [f64],
    ) -> f64 {
        let n_faces = mesh.n_faces();
        let max_speed_atomic = AtomicU64::new(0u64);

        // 骞惰璁＄畻鎵€鏈夐潰
        let face_results: Vec<_> = (0..n_faces)
            .into_par_iter()
            .map(|face_idx| {
                let (flux, bed_src, length, owner, neighbor) = 
                    self.compute_face(state, mesh, face_idx);

                max_speed_atomic.fetch_max(flux.max_wave_speed.to_bits(), Ordering::Relaxed);

                (flux, bed_src, length, owner, neighbor)
            })
            .collect();

        // 涓茶绱姞
        flux_h.fill(0.0);
        flux_hu.fill(0.0);
        flux_hv.fill(0.0);
        source_hu.fill(0.0);
        source_hv.fill(0.0);

        for (flux, bed_src, length, owner, neighbor) in face_results {
            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;

            flux_h[owner] -= fh;
            flux_hu[owner] -= fhu;
            flux_hv[owner] -= fhv;
            source_hu[owner] += bed_src.source_left_x;
            source_hv[owner] += bed_src.source_left_y;

            if let Some(neigh) = neighbor {
                flux_h[neigh] += fh;
                flux_hu[neigh] += fhu;
                flux_hv[neigh] += fhv;
                source_hu[neigh] += bed_src.source_right_x;
                source_hv[neigh] += bed_src.source_right_y;
            }
        }

        f64::from_bits(max_speed_atomic.load(Ordering::Relaxed))
    }

    /// 鐫€鑹插苟琛岃绠?
    /// 
    /// 浣跨敤棰勮绠楃殑闈㈢潃鑹诧紝鍚屼竴棰滆壊鐨勯潰鍙互骞惰璁＄畻鍜岀疮鍔?
    /// 鍥犱负瀹冧滑涓嶅叡浜崟鍏?
    fn compute_colored(
        &self,
        state: &ShallowWaterState,
        mesh: &PhysicsMesh,
        flux_h: &mut [f64],
        flux_hu: &mut [f64],
        flux_hv: &mut [f64],
        source_hu: &mut [f64],
        source_hv: &mut [f64],
    ) -> f64 {
        // 閲嶇疆
        flux_h.fill(0.0);
        flux_hu.fill(0.0);
        flux_hv.fill(0.0);
        source_hu.fill(0.0);
        source_hv.fill(0.0);

        let max_speed_atomic = AtomicU64::new(0u64);

        let color_faces = match &self.face_colors {
            Some(cf) => cf,
            None => return 0.0, // 娌℃湁鐫€鑹诧紝杩斿洖0
        };

        // 鎸夐鑹叉壒娆″鐞?
        // 鍚屼竴棰滆壊鐨勯潰涓嶅叡浜崟鍏冿紝鍙互瀹夊叏骞惰
        for faces_in_color in color_faces {
            // 骞惰璁＄畻褰撳墠棰滆壊鐨勬墍鏈夐潰
            let results: Vec<_> = faces_in_color
                .par_iter()
                .map(|&face_idx| {
                    let (flux, bed_src, length, owner, neighbor) = 
                        self.compute_face(state, mesh, face_idx);
                    
                    max_speed_atomic.fetch_max(flux.max_wave_speed.to_bits(), Ordering::Relaxed);
                    
                    (flux, bed_src, length, owner, neighbor)
                })
                .collect();

            // 绱姞褰撳墠棰滆壊鐨勭粨鏋滐紙浠嶇劧闇€瑕佷覆琛岋紝浣嗘壒娆″唴宸茬粡鏄棤閿佺殑锛?
            // 鐢变簬鍚屼竴棰滆壊鐨勯潰涓嶅叡浜崟鍏冿紝鍙互瀹夊叏绱姞
            for (flux, bed_src, length, owner, neighbor) in results {
                let fh = flux.mass * length;
                let fhu = flux.momentum_x * length;
                let fhv = flux.momentum_y * length;

                flux_h[owner] -= fh;
                flux_hu[owner] -= fhu;
                flux_hv[owner] -= fhv;
                source_hu[owner] += bed_src.source_left_x;
                source_hv[owner] += bed_src.source_left_y;

                if let Some(neigh) = neighbor {
                    flux_h[neigh] += fh;
                    flux_hu[neigh] += fhu;
                    flux_hv[neigh] += fhv;
                    source_hu[neigh] += bed_src.source_right_x;
                    source_hv[neigh] += bed_src.source_right_y;
                }
            }
        }

        f64::from_bits(max_speed_atomic.load(Ordering::Relaxed))
    }

    /// 璁＄畻鍗曚釜闈㈢殑閫氶噺
    fn compute_face(
        &self,
        state: &ShallowWaterState,
        mesh: &PhysicsMesh,
        face_idx: usize,
    ) -> (RiemannFlux, BedSlopeCorrection, f64, usize, Option<usize>) {
        let normal = mesh.face_normal(face_idx);
        let length = mesh.face_length(face_idx);
        let owner = mesh.face_owner(face_idx);
        let neighbor = mesh.face_neighbor(face_idx);

        // 宸︿晶鐘舵€?
        let h_l = state.h[owner];
        let z_l = state.z[owner];
        let (u_l, v_l) = self.config.params.safe_velocity_components(
            state.hu[owner], state.hv[owner], h_l
        );
        let vel_l = DVec2::new(u_l, v_l);

        // 鍙充晶鐘舵€?
        let (h_r, vel_r, z_r) = if let Some(neigh) = neighbor {
            let h = state.h[neigh];
            let (u, v) = self.config.params.safe_velocity_components(
                state.hu[neigh], state.hv[neigh], h
            );
            (h, DVec2::new(u, v), state.z[neigh])
        } else {
            let vn = vel_l.dot(normal);
            (h_l, vel_l - 2.0 * vn * normal, z_l)
        };

        // 闈欐按閲嶆瀯
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

        // 骞叉箍闄愬埗
        let wet_l = self.wetting_drying.get_state(recon.h_left);
        let wet_r = self.wetting_drying.get_state(recon.h_right);
        let flux_limiter = match (wet_l, wet_r) {
            (WetState::Dry, WetState::Dry) => 0.0,
            (WetState::Dry, _) | (_, WetState::Dry) => {
                let h_min = recon.h_left.min(recon.h_right);
                (h_min / self.config.params.h_wet).min(1.0)
            }
            (WetState::PartiallyWet, _) | (_, WetState::PartiallyWet) => {
                let h_min = recon.h_left.min(recon.h_right);
                ((h_min - self.config.params.h_dry)
                    / (self.config.params.h_wet - self.config.params.h_dry)).clamp(0.0, 1.0)
            }
            _ => 1.0,
        };

        // 榛庢浖閫氶噺
        let flux = self.riemann.solve(
            recon.h_left, recon.h_right,
            recon.vel_left, recon.vel_right,
            normal,
        ).unwrap_or(RiemannFlux::ZERO);

        let limited_flux = if flux_limiter < 1.0 {
            flux.scaled(flux_limiter)
        } else {
            flux
        };

        // 搴婂潯婧愰」
        let bed_src = self.hydrostatic.bed_slope_correction(h_l, h_r, z_l, z_r, normal, length);

        (limited_flux, bed_src, length, owner, neighbor)
    }

    // =========================================================================
    // 璁块棶鍣?
    // =========================================================================

    /// 鑾峰彇閰嶇疆
    pub fn config(&self) -> &ParallelFluxConfig {
        &self.config
    }

    /// 鑾峰彇鎬ц兘鎸囨爣
    pub fn metrics(&self) -> &FluxComputeMetrics {
        &self.metrics
    }

    /// 閲嶇疆鎬ц兘鎸囨爣
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }
}

// ============================================================
// 鏋勫缓鍣?
// ============================================================

/// 骞惰璁＄畻鍣ㄦ瀯寤哄櫒
pub struct ParallelFluxCalculatorBuilder {
    config: ParallelFluxConfig,
}

impl ParallelFluxCalculatorBuilder {
    pub fn new() -> Self {
        Self {
            config: ParallelFluxConfig::default(),
        }
    }

    pub fn config(mut self, config: ParallelFluxConfig) -> Self {
        self.config = config;
        self
    }

    pub fn params(mut self, params: NumericalParams) -> Self {
        self.config.params = params;
        self
    }

    pub fn gravity(mut self, g: f64) -> Self {
        self.config.g = g;
        self
    }

    pub fn strategy(mut self, strategy: ParallelStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn build(self) -> ParallelFluxCalculator {
        ParallelFluxCalculator::new(self.config)
    }
}

impl Default for ParallelFluxCalculatorBuilder {
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

    #[test]
    fn test_config_default() {
        let config = ParallelFluxConfig::default();
        assert!((config.g - 9.81).abs() < 1e-10);
        assert_eq!(config.min_parallel_size, 1000);
        assert_eq!(config.strategy, ParallelStrategy::Auto);
    }

    #[test]
    fn test_config_builder() {
        let config = ParallelFluxConfig::builder()
            .gravity(10.0)
            .min_parallel_size(500)
            .strategy(ParallelStrategy::Sequential)
            .build();

        assert!((config.g - 10.0).abs() < 1e-10);
        assert_eq!(config.min_parallel_size, 500);
        assert_eq!(config.strategy, ParallelStrategy::Sequential);
    }

    #[test]
    fn test_colored_strategy() {
        let config = ParallelFluxConfig::builder()
            .strategy(ParallelStrategy::Colored)
            .build();
        
        assert_eq!(config.strategy, ParallelStrategy::Colored);
    }

    #[test]
    fn test_metrics() {
        let mut metrics = FluxComputeMetrics::default();
        metrics.record(1000, true, Duration::from_millis(10));
        metrics.record(500, false, Duration::from_millis(5));

        assert_eq!(metrics.total_calls, 2);
        assert_eq!(metrics.parallel_calls, 1);
        assert_eq!(metrics.sequential_calls, 1);
        assert_eq!(metrics.total_faces, 1500);
    }

    #[test]
    fn test_calculator_builder() {
        let calc = ParallelFluxCalculatorBuilder::new()
            .gravity(10.0)
            .strategy(ParallelStrategy::CollectThenAccumulate)
            .build();

        assert!((calc.config().g - 10.0).abs() < 1e-10);
        assert_eq!(calc.config().strategy, ParallelStrategy::CollectThenAccumulate);
    }
}
