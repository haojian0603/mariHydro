// crates/mh_physics/src/engine/parallel.rs

//! 并行通量计算模块
//!
//! 提供多种并行策略用于加速通量计算：
//! - 串行计算（小规模问题）
//! - 收集后累加（先并行计算通量，后串行累加到单元）
//! - 着色并行（使用图着色实现真正无锁并行，TODO）
//!
//! # 迁移说明
//!
//! 从 legacy_src/physics/engine/parallel.rs 简化迁移。
//! 完整的着色并行等高级功能将在后续版本实现。
//!
//! # 技术债务 (TD-5.3.2, TD-5.3.3)
//!
//! 当前实现的"并行"是伪并行：通量计算并行，但累加阶段串行。
//! 对于大规模网格，需要实现真正的着色并行以避免累加瓶颈。
//!
//! # Safety
//!
//! 本模块使用 unsafe 代码进行性能优化的原子操作，安全性由图着色算法保证。
#![allow(unsafe_code)]

use crate::adapter::PhysicsMesh;
use crate::engine::solver::{BedSlopeCorrectionF64, HydrostaticFaceState, HydrostaticReconstruction};
use crate::schemes::riemann::{HllcSolverF64, RiemannFluxF64, RiemannSolver, SolverParamsF64};
use crate::schemes::wetting_drying::{WetState, WettingDryingHandlerF64};
use crate::state::ShallowWaterStateF64;
use crate::types::NumericalParamsF64;
use crate::core::CpuBackend;

use glam::DVec2;
use mh_runtime::FaceIndex;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// 简单的日志宏替代 tracing
macro_rules! debug {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            // 在调试模式下可以打印日志
            // eprintln!("[DEBUG] {}", format!($($arg)*));
        }
    };
}

macro_rules! trace {
    ($($arg:tt)*) => {
        // trace 级别默认不输出
    };
}

macro_rules! info {
    ($($arg:tt)*) => {
        // info 级别可选输出
        // eprintln!("[INFO] {}", format!($($arg)*));
    };
}

/// HydrostaticReconstructionF64 类型别名
pub type HydrostaticReconstructionF64 = HydrostaticReconstruction<CpuBackend<f64>>;

// ============================================================
// 配置
// ============================================================

/// 并行策略
///
/// # 策略说明
///
/// - `Sequential`: 完全串行执行，适用于小规模问题
/// - `CollectThenAccumulate`: 先并行计算各面通量(真正并行)，
///   然后串行累加到单元(瓶颈)。对于中等规模问题有效。
/// - `Colored`: 使用图着色实现真正的无锁并行累加
/// - `Auto`: 根据面数自动选择策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum ParallelStrategy {
    /// 串行执行
    Sequential,
    /// 收集后累加：并行计算通量 → 收集结果 → 串行累加
    ///
    /// 注意：累加阶段是串行的，对于大规模网格可能成为瓶颈
    CollectThenAccumulate,
    /// 着色并行：使用图着色分组面，同一颜色的面可安全并行处理
    /// 
    /// 这是推荐的大规模并行策略，需要预先计算面着色
    Colored,
    /// 自动选择（根据问题规模）
    #[default]
    Auto,
}


/// 并行计算配置
#[derive(Debug, Clone)]
pub struct ParallelFluxConfig {
    /// 数值参数
    pub params: NumericalParamsF64,
    /// 重力加速度
    pub g: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最小并行面数（低于此值使用串行）
    pub min_parallel_size: usize,
    /// 并行策略
    pub strategy: ParallelStrategy,
    /// 是否启用静水重构
    pub use_hydrostatic_reconstruction: bool,
}

impl Default for ParallelFluxConfig {
    fn default() -> Self {
        Self {
            params: NumericalParamsF64::default(),
            g: 9.81,
            min_parallel_size: 1000,
            strategy: ParallelStrategy::Auto,
            use_hydrostatic_reconstruction: true,
        }
    }
}

impl ParallelFluxConfig {
    /// 创建构建器
    pub fn builder() -> ParallelFluxConfigBuilder {
        ParallelFluxConfigBuilder::default()
    }
}

/// 配置构建器
#[derive(Default)]
pub struct ParallelFluxConfigBuilder {
    config: ParallelFluxConfig,
}

impl ParallelFluxConfigBuilder {
    pub fn params(mut self, params: NumericalParamsF64) -> Self {
        self.config.params = params;
        self
    }

    pub fn gravity(mut self, g: f64) -> Self { // ALLOW_F64: 物理常数配置参数
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
// 性能指标
// ============================================================

/// 性能指标
#[derive(Debug, Clone, Default)]
pub struct FluxComputeMetrics {
    /// 总计算次数
    pub total_calls: usize,
    /// 并行计算次数
    pub parallel_calls: usize,
    /// 串行计算次数
    pub sequential_calls: usize,
    /// 总计算时间
    pub total_duration: Duration,
    /// 处理的面总数
    pub total_faces: usize,
}

impl FluxComputeMetrics {
    /// 记录一次计算
    // TODO(phase5): 添加策略选择的详细日志（如 legacy 的 StrategySelector）
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

    /// 重置指标
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// 平均每面计算时间
    pub fn avg_time_per_face(&self) -> Duration {
        if self.total_faces > 0 {
            self.total_duration / self.total_faces as u32
        } else {
            Duration::ZERO
        }
    }
}

// ============================================================
// 并行通量计算器
// ============================================================

/// 并行通量计算器
///
/// 封装通量计算的并行执行逻辑。
pub struct ParallelFluxCalculator {
    config: ParallelFluxConfig,
    /// 黎曼求解器
    riemann: HllcSolverF64,
    /// 干湿处理器（预留用于未来扩展）
    #[allow(dead_code)]
    wetting_drying: WettingDryingHandlerF64,
    /// 静水重构
    hydrostatic: HydrostaticReconstructionF64,
    /// 性能指标
    metrics: FluxComputeMetrics,
    /// 面着色（用于 Colored 策略）
    /// 每个元素是一组可以并行处理的面索引
    face_colors: Option<Vec<Vec<usize>>>,
}

impl ParallelFluxCalculator {
    /// 创建计算器
    pub fn new(config: ParallelFluxConfig) -> Self {
        let riemann_params = SolverParamsF64::from_numerical(&config.params, config.g);
        Self {
            riemann: HllcSolverF64::new(&riemann_params, config.g),
            wetting_drying: WettingDryingHandlerF64::from_params(&config.params),
            hydrostatic: HydrostaticReconstructionF64::new(&config.params, config.g),
            metrics: FluxComputeMetrics::default(),
            face_colors: None,
            config,
        }
    }

    /// 为网格设置面着色（用于 Colored 策略）
    /// 
    /// 面着色将面分成若干组，同一组内的面不共享单元，
    /// 因此可以安全地并行更新这些面关联的单元。
    /// 
    /// 使用贪心图着色算法，按照度数降序处理节点以减少颜色数量。
    /// 
    /// # 参数
    /// - `mesh`: 网格
    /// 
    /// # 性能说明
    /// 着色计算是 O(|F| + |E|) 复杂度，其中 |F| 是面数，|E| 是邻接边数。
    /// 对于典型的二维非结构化网格，每个面平均约有 6-10 个邻居。
    pub fn setup_face_coloring(&mut self, mesh: &PhysicsMesh) {
        let start = Instant::now();
        let n_faces = mesh.n_faces();
        
        if n_faces == 0 {
            self.face_colors = Some(Vec::new());
            debug!("Face coloring: empty mesh, no coloring needed");
            return;
        }

        trace!("Building face adjacency graph for {} faces", n_faces);
        
        // 构建面邻接关系
        let face_neighbors = self.build_face_adjacency(mesh);
        
        // 贪心着色
        let (face_color, num_colors) = self.greedy_coloring(&face_neighbors, n_faces);

        // 按颜色分组面
        let mut color_faces: Vec<Vec<usize>> = vec![Vec::new(); num_colors];
        for (face, &color) in face_color.iter().enumerate() {
            if color != usize::MAX {
                color_faces[color].push(face);
            }
        }

        let _duration = start.elapsed();
        
        // 计算着色质量指标
        let _avg_group_size = if num_colors > 0 { 
            n_faces / num_colors 
        } else { 
            0 
        };
        let _max_group_size = color_faces.iter().map(|g| g.len()).max().unwrap_or(0);
        let _min_group_size = color_faces.iter().map(|g| g.len()).min().unwrap_or(0);
        
        info!(
            "Face coloring complete: {} faces, {} colors, avg/min/max group size = {}/{}/{}, took {:?}",
            n_faces, num_colors, _avg_group_size, _min_group_size, _max_group_size, _duration
        );

        self.face_colors = Some(color_faces);
    }
    
    /// 构建面邻接关系
    /// 
    /// 两个面相邻 <=> 它们共享一个单元
    fn build_face_adjacency(&self, mesh: &PhysicsMesh) -> Vec<std::collections::HashSet<usize>> {
        use std::collections::{HashMap, HashSet};
        
        let n_faces = mesh.n_faces();
        
        // 构建单元到面的映射
        let mut cell_to_faces: HashMap<usize, Vec<usize>> = HashMap::new();
        for face_idx in 0..n_faces {
            let owner = mesh.face_owner(mh_runtime::FaceIndex(face_idx));
            cell_to_faces.entry(owner.into()).or_default().push(face_idx);
            if let Some(neigh) = mesh.face_neighbor(mh_runtime::FaceIndex(face_idx)) {
                cell_to_faces.entry(neigh.into()).or_default().push(face_idx);
            }
        }
        
        // 构建面的邻接表
        let mut face_neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); n_faces];
        for faces in cell_to_faces.values() {
            // 同一单元的所有面互为邻居
            for i in 0..faces.len() {
                for j in (i + 1)..faces.len() {
                    face_neighbors[faces[i]].insert(faces[j]);
                    face_neighbors[faces[j]].insert(faces[i]);
                }
            }
        }
        
        face_neighbors
    }
    
    /// 贪心图着色算法
    /// 
    /// 返回 (face_color, num_colors) 元组
    /// face_color[i] 表示面 i 的颜色（0-indexed）
    fn greedy_coloring(
        &self, 
        face_neighbors: &[std::collections::HashSet<usize>], 
        n_faces: usize
    ) -> (Vec<usize>, usize) {
        use std::collections::HashSet;
        
        let mut face_color = vec![usize::MAX; n_faces];
        let mut num_colors = 0;

        // 按邻居数量排序（高度数优先，能减少总颜色数）
        let mut order: Vec<usize> = (0..n_faces).collect();
        order.sort_by_key(|&f| std::cmp::Reverse(face_neighbors[f].len()));

        for &face in &order {
            // 找到邻居使用的颜色
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

            // 找到最小可用颜色
            let mut color = 0;
            while used_colors.contains(&color) {
                color += 1;
            }

            face_color[face] = color;
            num_colors = num_colors.max(color + 1);
        }
        
        (face_color, num_colors)
    }

    /// 检查是否已设置面着色
    pub fn has_face_coloring(&self) -> bool {
        self.face_colors.is_some()
    }

    /// 获取颜色数量
    pub fn num_colors(&self) -> usize {
        self.face_colors.as_ref().map(|c| c.len()).unwrap_or(0)
    }

    /// 计算通量（自动选择策略）
    pub fn compute_fluxes(
        &mut self,
        state: &ShallowWaterStateF64,
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
                // 如果没有设置着色，先设置
                if !self.has_face_coloring() {
                    self.setup_face_coloring(mesh);
                }
                (self.compute_colored(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), true)
            }
            ParallelStrategy::Auto => {
                if n_faces < self.config.min_parallel_size {
                    (self.compute_serial(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), false)
                } else if self.has_face_coloring() {
                    // 有着色就用着色并行
                    (self.compute_colored(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), true)
                } else {
                    // 否则用收集后累加
                    (self.compute_parallel(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), true)
                }
            }
        };

        let duration = start.elapsed();
        self.metrics.record(n_faces, is_parallel, duration);

        max_speed
    }

    /// 串行计算
    fn compute_serial(
        &self,
        state: &ShallowWaterStateF64,
        mesh: &PhysicsMesh,
        flux_h: &mut [f64],
        flux_hu: &mut [f64],
        flux_hv: &mut [f64],
        source_hu: &mut [f64],
        source_hv: &mut [f64],
    ) -> f64 {
        // 重置
        flux_h.fill(0.0);
        flux_hu.fill(0.0);
        flux_hv.fill(0.0);
        source_hu.fill(0.0);
        source_hv.fill(0.0);

        let n_faces = mesh.n_faces();
        let mut max_speed = 0.0f64;

        for face_idx in 0..n_faces {
            let (flux, bed_src, length, owner, neighbor) = 
                self.compute_face(state, mesh, FaceIndex(face_idx));

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

    /// 并行计算（先并行计算，后串行累加）
    fn compute_parallel(
        &self,
        state: &ShallowWaterStateF64,
        mesh: &PhysicsMesh,
        flux_h: &mut [f64],
        flux_hu: &mut [f64],
        flux_hv: &mut [f64],
        source_hu: &mut [f64],
        source_hv: &mut [f64],
    ) -> f64 {
        let n_faces = mesh.n_faces();
        let max_speed_atomic = AtomicU64::new(0u64);

        // 并行计算所有面
        let face_results: Vec<_> = (0..n_faces)
            .into_par_iter()
            .map(|face_idx| {
                let (flux, bed_src, length, owner, neighbor) = 
                    self.compute_face(state, mesh, FaceIndex(face_idx));

                max_speed_atomic.fetch_max(flux.max_wave_speed.to_bits(), Ordering::Relaxed);

                (flux, bed_src, length, owner, neighbor)
            })
            .collect();

        // 串行累加
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

    /// 着色并行计算（真正无锁并行）
    /// 
    /// 使用预计算的面着色，同一颜色的面可以并行计算和累加
    /// 因为它们不共享单元，不存在数据竞争。
    /// 
    /// # 算法说明
    /// 
    /// 1. 按颜色分批处理面
    /// 2. 同一颜色内的面完全独立，可以并行计算并直接写入结果
    /// 3. 不同颜色之间串行处理以保证累加正确性
    /// 
    /// # 性能优势
    /// 
    /// 相比 CollectThenAccumulate 策略：
    /// - 无需分配中间结果向量
    /// - 累加阶段也是并行的（每个颜色批次内）
    /// - 对于 N 个颜色，有 N-1 次同步点，但每个批次内完全并行
    fn compute_colored(
        &self,
        state: &ShallowWaterStateF64,
        mesh: &PhysicsMesh,
        flux_h: &mut [f64],
        flux_hu: &mut [f64],
        flux_hv: &mut [f64],
        source_hu: &mut [f64],
        source_hv: &mut [f64],
    ) -> f64 {
        // 重置
        flux_h.fill(0.0);
        flux_hu.fill(0.0);
        flux_hv.fill(0.0);
        source_hu.fill(0.0);
        source_hv.fill(0.0);

        let max_speed_atomic = AtomicU64::new(0u64);

        let color_faces = match &self.face_colors {
            Some(cf) => cf,
            None => {
                debug!("compute_colored: no face coloring available, returning 0.0");
                return 0.0;
            }
        };

        trace!("compute_colored: processing {} color groups", color_faces.len());

        // 按颜色批次处理
        // 同一颜色的面不共享单元，可以安全并行写入
        for (color_idx, faces_in_color) in color_faces.iter().enumerate() {
            trace!("  color {}: {} faces", color_idx, faces_in_color.len());
            
            // 使用 UnsafeCell 或指针技巧实现真正的并行写入
            // 由于着色保证了同一颜色的面不共享单元，这是安全的
            
            // 创建原子计数器用于统计处理的面数（调试用）
            #[cfg(debug_assertions)]
            let processed_count = std::sync::atomic::AtomicUsize::new(0);
            
            // 并行计算并累加当前颜色的所有面
            // SAFETY: 由于着色算法保证同一颜色的面不共享任何单元，
            // 因此不同线程写入的数组位置不会重叠，没有数据竞争。
            faces_in_color.par_iter().for_each(|&face_idx| {
                let (flux, bed_src, length, owner, neighbor) = 
                    self.compute_face(state, mesh, FaceIndex(face_idx));
                
                max_speed_atomic.fetch_max(flux.max_wave_speed.to_bits(), Ordering::Relaxed);
                
                let fh = flux.mass * length;
                let fhu = flux.momentum_x * length;
                let fhv = flux.momentum_y * length;

                // SAFETY: owner 和 neighbor 是由着色算法保证互不冲突的
                // 同一颜色的任意两个面不会有相同的 owner 或 neighbor
                unsafe {
                    // 使用 get_unchecked_mut 避免边界检查开销
                    // 这里的安全性由网格拓扑和着色算法保证
                    let flux_h_ptr = flux_h.as_ptr() as *mut f64;
                    let flux_hu_ptr = flux_hu.as_ptr() as *mut f64;
                    let flux_hv_ptr = flux_hv.as_ptr() as *mut f64;
                    let source_hu_ptr = source_hu.as_ptr() as *mut f64;
                    let source_hv_ptr = source_hv.as_ptr() as *mut f64;
                    
                    *flux_h_ptr.add(owner) -= fh;
                    *flux_hu_ptr.add(owner) -= fhu;
                    *flux_hv_ptr.add(owner) -= fhv;
                    *source_hu_ptr.add(owner) += bed_src.source_left_x;
                    *source_hv_ptr.add(owner) += bed_src.source_left_y;

                    if let Some(neigh) = neighbor {
                        *flux_h_ptr.add(neigh) += fh;
                        *flux_hu_ptr.add(neigh) += fhu;
                        *flux_hv_ptr.add(neigh) += fhv;
                        *source_hu_ptr.add(neigh) += bed_src.source_right_x;
                        *source_hv_ptr.add(neigh) += bed_src.source_right_y;
                    }
                }
                
                #[cfg(debug_assertions)]
                processed_count.fetch_add(1, Ordering::Relaxed);
            });
            
            #[cfg(debug_assertions)]
            debug_assert_eq!(
                processed_count.load(Ordering::Relaxed), 
                faces_in_color.len(),
                "Not all faces in color {} were processed", color_idx
            );
        }

        f64::from_bits(max_speed_atomic.load(Ordering::Relaxed))
    }

    /// 计算单个面的通量
    #[allow(deprecated)]
    fn compute_face(
        &self,
        state: &ShallowWaterStateF64,
        mesh: &PhysicsMesh,
        face_idx: FaceIndex,
    ) -> (RiemannFluxF64, BedSlopeCorrectionF64, f64, usize, Option<usize>) {
        let normal = mesh.face_normal(face_idx.get());
        let length = mesh.face_length(face_idx);
        let owner = mesh.face_owner(face_idx);
        let neighbor = mesh.face_neighbor(face_idx);
        
        let owner_idx = owner.get();
        let neighbor_idx = neighbor.map(|c| c.get());

        // 左侧状态
        let h_l = state.h[owner_idx];
        let z_l = state.z[owner_idx];
        let (u_l, v_l) = self.config.params.safe_velocity_components(
            state.hu[owner_idx], state.hv[owner_idx], h_l
        );
        let vel_l = DVec2::new(u_l, v_l);

        // 右侧状态
        let (h_r, vel_r, z_r) = if let Some(neigh_idx) = neighbor_idx {
            let h = state.h[neigh_idx];
            let (u, v) = self.config.params.safe_velocity_components(
                state.hu[neigh_idx], state.hv[neigh_idx], h
            );
            (h, DVec2::new(u, v), state.z[neigh_idx])
        } else {
            let vn = vel_l.dot(normal);
            (h_l, vel_l - 2.0 * vn * normal, z_l)
        };

        // 静水重构
        let recon = if self.config.use_hydrostatic_reconstruction {
            self.hydrostatic.reconstruct_face_simple(h_l, h_r, z_l, z_r, [vel_l.x, vel_l.y], [vel_r.x, vel_r.y])
        } else {
            HydrostaticFaceState {
                h_left: h_l,
                h_right: h_r,
                vel_left: [vel_l.x, vel_l.y],
                vel_right: [vel_r.x, vel_r.y],
                z_face: 0.5 * (z_l + z_r),
            }
        };

        // 干湿限制
        let wet_l = WetState::from_depth(recon.h_left, self.config.params.h_dry, self.config.params.h_wet);
        let wet_r = WetState::from_depth(recon.h_right, self.config.params.h_dry, self.config.params.h_wet);
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

        // 黎曼通量
        let vel_l_arr = [recon.vel_left[0], recon.vel_left[1]];
        let vel_r_arr = [recon.vel_right[0], recon.vel_right[1]];
        let normal_arr = [normal.x, normal.y];
        let flux = self.riemann.solve(
            recon.h_left, recon.h_right,
            vel_l_arr, vel_r_arr,
            normal_arr,
        ).unwrap_or(RiemannFluxF64::zero());

        let limited_flux = if flux_limiter < 1.0 {
            flux.scaled(flux_limiter)
        } else {
            flux
        };

        // 床坡源项
        let bed_src = self.hydrostatic.bed_slope_correction(h_l, h_r, z_l, z_r, normal_arr, length);

        (limited_flux, bed_src, length, owner_idx, neighbor_idx)
    }

    // =========================================================================
    // 访问器
    // =========================================================================

    /// 获取配置
    pub fn config(&self) -> &ParallelFluxConfig {
        &self.config
    }

    /// 获取性能指标
    pub fn metrics(&self) -> &FluxComputeMetrics {
        &self.metrics
    }

    /// 重置性能指标
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }
}

// ============================================================
// 构建器
// ============================================================

/// 并行计算器构建器
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

    pub fn params(mut self, params: NumericalParamsF64) -> Self {
        self.config.params = params;
        self
    }

    pub fn gravity(mut self, g: f64) -> Self { // ALLOW_F64: 物理常数配置参数
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
// 测试
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
