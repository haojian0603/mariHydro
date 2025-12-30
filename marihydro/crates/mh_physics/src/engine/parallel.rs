// crates/mh_physics/src/engine/parallel.rs

//! 并行通量计算模块（Backend泛型化版本）
//!
//! 提供多种并行策略用于加速通量计算，支持任意Backend（CPU f32/f64, GPU）。
//!
//! # 并行策略
//! - 串行计算（小规模问题）
//! - 收集后累加（先并行计算通量，后串行累加到单元）
//! - 着色并行（使用图着色实现真正无锁并行）
//!
//! # Backend泛型化
//! 整个模块完全Backend化，所有计算数据使用`B::Buffer<B::Scalar>`存储，
//! 几何数据使用`B::Vector2D`，支持运行时精度切换。

#![allow(unsafe_code)]

use crate::adapter::PhysicsMesh;
use crate::engine::solver::{BedSlopeCorrection, HydrostaticFaceState, HydrostaticReconstruction};
use crate::schemes::riemann::{HllcSolver, RiemannFlux, RiemannSolver};
use crate::schemes::wetting_drying::{WetState, WettingDryingHandler};
use crate::state::ShallowWaterState;
use crate::types::NumericalParams;

use mh_runtime::{Backend, CellIndex, DeviceBuffer, FaceIndex, RuntimeScalar};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// 线程安全指针包装器

/// 用于并行计算的可发送原始指针包装器
/// 
/// SAFETY: 调用者必须确保：
/// 1. 指针在整个并行操作期间有效
/// 2. 不同线程不会写入同一内存位置
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);

// SAFETY: SendPtr 可以安全地在线程间发送，因为我们保证：
// 1. 着色算法确保同一颜色的面不共享单元
// 2. 不同线程写入不同的数组位置
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T: Copy> SendPtr<T> {
    /// 读取指定偏移位置的值
    #[inline]
    unsafe fn read_at(&self, offset: usize) -> T {
        self.0.add(offset).read()
    }
    
    /// 写入指定偏移位置的值
    #[inline]
    unsafe fn write_at(&self, offset: usize, value: T) {
        self.0.add(offset).write(value);
    }
}

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

// ============================================================
// 配置
// ============================================================

/// 并行策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum ParallelStrategy {
    /// 串行执行
    Sequential,
    /// 收集后累加：并行计算通量 → 收集结果 → 串行累加
    CollectThenAccumulate,
    /// 着色并行：使用图着色分组面，同一颜色的面可安全并行处理
    Colored,
    /// 自动选择（根据问题规模）
    #[default]
    Auto,
}

/// 并行计算配置（Backend泛型化）
#[derive(Debug, Clone)]
pub struct ParallelFluxConfig<S: RuntimeScalar> {
    /// 数值参数
    pub params: NumericalParams<S>,
    /// 重力加速度
    pub g: S,
    /// 最小并行面数（低于此值使用串行）
    pub min_parallel_size: usize,
    /// 并行策略
    pub strategy: ParallelStrategy,
    /// 是否启用静水重构
    pub use_hydrostatic_reconstruction: bool,
}

impl<S: RuntimeScalar> Default for ParallelFluxConfig<S> {
    fn default() -> Self {
        Self {
            params: NumericalParams::<S>::default(),
            g: S::from_config(9.81).unwrap_or(S::ZERO),
            min_parallel_size: 1000,
            strategy: ParallelStrategy::Auto,
            use_hydrostatic_reconstruction: true,
        }
    }
}

impl<S: RuntimeScalar> ParallelFluxConfig<S> {
    /// 创建构建器
    pub fn builder() -> ParallelFluxConfigBuilder<S> {
        ParallelFluxConfigBuilder::default()
    }
}

/// 配置构建器
pub struct ParallelFluxConfigBuilder<S: RuntimeScalar> {
    config: ParallelFluxConfig<S>,
}

impl<S: RuntimeScalar> Default for ParallelFluxConfigBuilder<S> {
    fn default() -> Self {
        Self {
            config: ParallelFluxConfig::default(),
        }
    }
}

impl<S: RuntimeScalar> ParallelFluxConfigBuilder<S> {
    pub fn params(mut self, params: NumericalParams<S>) -> Self {
        self.config.params = params;
        self
    }

    pub fn gravity(mut self, g: S) -> Self {
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

    pub fn build(self) -> ParallelFluxConfig<S> {
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
// 并行通量计算器（Backend泛型化版本）
// ============================================================

/// 并行通量计算器（Backend泛型化）
///
/// 封装通量计算的并行执行逻辑，支持任意Backend。
/// 
/// # 类型参数
/// 
/// - `B`: 计算后端，提供存储和计算能力
pub struct ParallelFluxCalculator<B: Backend> {
    config: ParallelFluxConfig<B::Scalar>,
    /// 黎曼求解器
    riemann: HllcSolver<B>,
    /// 干湿处理器
    wetting_drying: WettingDryingHandler<B>,
    /// 静水重构
    hydrostatic: HydrostaticReconstruction<B>,
    /// 性能指标
    metrics: FluxComputeMetrics,
    /// 面着色（用于 Colored 策略）
    /// 每个元素是一组可以并行处理的面索引
    face_colors: Option<Vec<Vec<usize>>>,
    /// 后端实例（预留用于GPU加速）
    #[allow(dead_code)]
    backend: B,
}

impl<B: Backend> ParallelFluxCalculator<B> {
    /// 创建计算器
    pub fn new(config: ParallelFluxConfig<B::Scalar>, backend: B) -> Self {
        let riemann_params = crate::schemes::riemann::SolverParams::<B::Scalar>::from_numerical(&config.params, config.g);
        Self {
            riemann: HllcSolver::<B>::new(&riemann_params, config.g),
            wetting_drying: WettingDryingHandler::<B>::from_params(&config.params),
            hydrostatic: HydrostaticReconstruction::<B>::new(&config.params, config.g),
            metrics: FluxComputeMetrics::default(),
            face_colors: None,
            config,
            backend,
        }
    }

    /// 为网格设置面着色（用于 Colored 策略）
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
        
        info!(
            "Face coloring complete: {} faces, {} colors, took {:?}",
            n_faces, num_colors, _duration
        );

        self.face_colors = Some(color_faces);
    }
    
    /// 构建面邻接关系
    fn build_face_adjacency(&self, mesh: &PhysicsMesh) -> Vec<std::collections::HashSet<usize>> {
        use std::collections::{HashMap, HashSet};
        
        let n_faces = mesh.n_faces();
        
        // 构建单元到面的映射
        let mut cell_to_faces: HashMap<usize, Vec<usize>> = HashMap::new();
        for face_idx in 0..n_faces {
            let owner = mesh.face_owner(FaceIndex(face_idx));
            cell_to_faces.entry(owner.get()).or_default().push(face_idx);
            if let Some(neigh) = mesh.face_neighbor(FaceIndex(face_idx)) {
                cell_to_faces.entry(neigh.get()).or_default().push(face_idx);
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
        state: &ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        flux_h: &mut B::Buffer<B::Scalar>,
        flux_hu: &mut B::Buffer<B::Scalar>,
        flux_hv: &mut B::Buffer<B::Scalar>,
        source_hu: &mut B::Buffer<B::Scalar>,
        source_hv: &mut B::Buffer<B::Scalar>,
    ) -> B::Scalar {
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
        state: &ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        flux_h: &mut B::Buffer<B::Scalar>,
        flux_hu: &mut B::Buffer<B::Scalar>,
        flux_hv: &mut B::Buffer<B::Scalar>,
        source_hu: &mut B::Buffer<B::Scalar>,
        source_hv: &mut B::Buffer<B::Scalar>,
    ) -> B::Scalar {
        // 重置
        flux_h.fill(B::Scalar::ZERO);
        flux_hu.fill(B::Scalar::ZERO);
        flux_hv.fill(B::Scalar::ZERO);
        source_hu.fill(B::Scalar::ZERO);
        source_hv.fill(B::Scalar::ZERO);

        let n_faces = mesh.n_faces();
        let mut max_speed = B::Scalar::ZERO;

        for face_idx in 0..n_faces {
            let (flux, bed_src, length, owner, neighbor) = 
                self.compute_face(state, mesh, FaceIndex(face_idx));

            if flux.max_wave_speed > max_speed {
                max_speed = flux.max_wave_speed;
            }

            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;

            let owner_idx = owner.get();
            flux_h[owner_idx] = flux_h[owner_idx] - fh;
            flux_hu[owner_idx] = flux_hu[owner_idx] - fhu;
            flux_hv[owner_idx] = flux_hv[owner_idx] - fhv;
            source_hu[owner_idx] = source_hu[owner_idx] + bed_src.source_left_x;
            source_hv[owner_idx] = source_hv[owner_idx] + bed_src.source_left_y;

            if let Some(neigh) = neighbor {
                let neigh_idx = neigh.get();
                flux_h[neigh_idx] = flux_h[neigh_idx] + fh;
                flux_hu[neigh_idx] = flux_hu[neigh_idx] + fhu;
                flux_hv[neigh_idx] = flux_hv[neigh_idx] + fhv;
                source_hu[neigh_idx] = source_hu[neigh_idx] + bed_src.source_right_x;
                source_hv[neigh_idx] = source_hv[neigh_idx] + bed_src.source_right_y;
            }
        }

        max_speed
    }

    /// 并行计算（先并行计算，后串行累加）
    fn compute_parallel(
        &self,
        state: &ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        flux_h: &mut B::Buffer<B::Scalar>,
        flux_hu: &mut B::Buffer<B::Scalar>,
        flux_hv: &mut B::Buffer<B::Scalar>,
        source_hu: &mut B::Buffer<B::Scalar>,
        source_hv: &mut B::Buffer<B::Scalar>,
    ) -> B::Scalar {
        let n_faces = mesh.n_faces();
        
        // 使用f64 atomic作为临时方案（后续在RuntimeScalar trait中添加Atomic关联类型）
        let max_speed_atomic = AtomicU64::new(0u64);

        // 并行计算所有面
        let face_results: Vec<_> = (0..n_faces)
            .into_par_iter()
            .map(|face_idx| {
                let (flux, bed_src, length, owner, neighbor) = 
                    self.compute_face(state, mesh, FaceIndex(face_idx));

                // 临时方案：转换为f64进行atomic操作
                let speed_f64 = flux.max_wave_speed.to_f64().unwrap_or(0.0);
                max_speed_atomic.fetch_max(speed_f64.to_bits(), Ordering::Relaxed);

                (flux, bed_src, length, owner, neighbor)
            })
            .collect();

        // 串行累加
        flux_h.fill(B::Scalar::ZERO);
        flux_hu.fill(B::Scalar::ZERO);
        flux_hv.fill(B::Scalar::ZERO);
        source_hu.fill(B::Scalar::ZERO);
        source_hv.fill(B::Scalar::ZERO);

        for (flux, bed_src, length, owner, neighbor) in face_results {
            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;

            let owner_idx = owner.get();
            flux_h[owner_idx] = flux_h[owner_idx] - fh;
            flux_hu[owner_idx] = flux_hu[owner_idx] - fhu;
            flux_hv[owner_idx] = flux_hv[owner_idx] - fhv;
            source_hu[owner_idx] = source_hu[owner_idx] + bed_src.source_left_x;
            source_hv[owner_idx] = source_hv[owner_idx] + bed_src.source_left_y;

            if let Some(neigh) = neighbor {
                let neigh_idx = neigh.get();
                flux_h[neigh_idx] = flux_h[neigh_idx] + fh;
                flux_hu[neigh_idx] = flux_hu[neigh_idx] + fhu;
                flux_hv[neigh_idx] = flux_hv[neigh_idx] + fhv;
                source_hu[neigh_idx] = source_hu[neigh_idx] + bed_src.source_right_x;
                source_hv[neigh_idx] = source_hv[neigh_idx] + bed_src.source_right_y;
            }
        }

        // 临时方案：从atomic加载并转换回B::Scalar
        let max_speed_bits = max_speed_atomic.load(Ordering::Relaxed);
        B::Scalar::from_f64(f64::from_bits(max_speed_bits)).unwrap_or(B::Scalar::ZERO)
    }

    /// 着色并行计算（真正无锁并行）
    fn compute_colored(
        &self,
        state: &ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        flux_h: &mut B::Buffer<B::Scalar>,
        flux_hu: &mut B::Buffer<B::Scalar>,
        flux_hv: &mut B::Buffer<B::Scalar>,
        source_hu: &mut B::Buffer<B::Scalar>,
        source_hv: &mut B::Buffer<B::Scalar>,
    ) -> B::Scalar {
        // 重置
        flux_h.fill(B::Scalar::ZERO);
        flux_hu.fill(B::Scalar::ZERO);
        flux_hv.fill(B::Scalar::ZERO);
        source_hu.fill(B::Scalar::ZERO);
        source_hv.fill(B::Scalar::ZERO);

        // 临时方案：使用f64 atomic
        let max_speed_atomic = AtomicU64::new(0u64);

        let color_faces = match &self.face_colors {
            Some(cf) => cf,
            None => {
                debug!("compute_colored: no face coloring available, returning 0.0");
                return B::Scalar::ZERO;
            }
        };

        trace!("compute_colored: processing {} color groups", color_faces.len());

        // 按颜色批次处理
        // 同一颜色的面不共享单元，可以安全并行写入
        for (_color_idx, faces_in_color) in color_faces.iter().enumerate() {
            trace!("  color {}: {} faces", _color_idx, faces_in_color.len());
            
            // 创建原子计数器用于统计处理的面数（调试用）
            #[cfg(debug_assertions)]
            let processed_count = std::sync::atomic::AtomicUsize::new(0);
            
            // SAFETY: 由于着色算法保证同一颜色的面不共享任何单元，
            // 因此不同线程写入的数组位置不会重叠，没有数据竞争。
            // 使用 SendPtr 包装原始指针以满足 Sync 要求。
            let flux_h_ptr = SendPtr(flux_h.as_slice_mut().as_mut_ptr());
            let flux_hu_ptr = SendPtr(flux_hu.as_slice_mut().as_mut_ptr());
            let flux_hv_ptr = SendPtr(flux_hv.as_slice_mut().as_mut_ptr());
            let source_hu_ptr = SendPtr(source_hu.as_slice_mut().as_mut_ptr());
            let source_hv_ptr = SendPtr(source_hv.as_slice_mut().as_mut_ptr());
            
            // 并行计算并累加当前颜色的所有面
            faces_in_color.par_iter().for_each(|&face_idx| {
                let (flux, bed_src, length, owner, neighbor) = 
                    self.compute_face(state, mesh, FaceIndex(face_idx));
                
                // 临时方案：转换为f64进行atomic操作
                let speed_f64 = flux.max_wave_speed.to_f64().unwrap_or(0.0);
                max_speed_atomic.fetch_max(speed_f64.to_bits(), Ordering::Relaxed);
                
                let fh = flux.mass * length;
                let fhu = flux.momentum_x * length;
                let fhv = flux.momentum_y * length;

                // 安全累加（不同线程不会冲突）
                // SAFETY: 着色保证同一颜色的面不共享单元
                unsafe {
                    let owner_idx = owner.get();
                    flux_h_ptr.write_at(owner_idx, flux_h_ptr.read_at(owner_idx) - fh);
                    flux_hu_ptr.write_at(owner_idx, flux_hu_ptr.read_at(owner_idx) - fhu);
                    flux_hv_ptr.write_at(owner_idx, flux_hv_ptr.read_at(owner_idx) - fhv);
                    source_hu_ptr.write_at(owner_idx, source_hu_ptr.read_at(owner_idx) + bed_src.source_left_x);
                    source_hv_ptr.write_at(owner_idx, source_hv_ptr.read_at(owner_idx) + bed_src.source_left_y);
                    
                    if let Some(neigh) = neighbor {
                        let neigh_idx = neigh.get();
                        flux_h_ptr.write_at(neigh_idx, flux_h_ptr.read_at(neigh_idx) + fh);
                        flux_hu_ptr.write_at(neigh_idx, flux_hu_ptr.read_at(neigh_idx) + fhu);
                        flux_hv_ptr.write_at(neigh_idx, flux_hv_ptr.read_at(neigh_idx) + fhv);
                        source_hu_ptr.write_at(neigh_idx, source_hu_ptr.read_at(neigh_idx) + bed_src.source_right_x);
                        source_hv_ptr.write_at(neigh_idx, source_hv_ptr.read_at(neigh_idx) + bed_src.source_right_y);
                    }
                }
                
                #[cfg(debug_assertions)]
                processed_count.fetch_add(1, Ordering::Relaxed);
            });
            
            #[cfg(debug_assertions)]
            debug_assert_eq!(
                processed_count.load(Ordering::Relaxed), 
                faces_in_color.len(),
                "Not all faces in color {} were processed", _color_idx
            );
        }

        // 临时方案：从atomic加载并转换回B::Scalar
        let max_speed_bits = max_speed_atomic.load(Ordering::Relaxed);
        B::Scalar::from_f64(f64::from_bits(max_speed_bits)).unwrap_or(B::Scalar::ZERO)
    }

    /// 计算单个面的通量
    fn compute_face(
        &self,
        state: &ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        face_idx: FaceIndex,
    ) -> (RiemannFlux<B::Scalar>, BedSlopeCorrection<B>, B::Scalar, CellIndex, Option<CellIndex>) {
        let normal = mesh.face_normal_generic::<B>(face_idx);
        let length_f64 = mesh.face_length(face_idx);
        let length = B::Scalar::from_f64(length_f64).unwrap_or(B::Scalar::ZERO);
        let owner = mesh.face_owner(face_idx);
        let neighbor = mesh.face_neighbor(face_idx);

        let owner_idx = owner.get();
        let neighbor_idx = neighbor.map(|c| c.get());

        // 左侧状态
        let h_l = state.h[owner_idx];
        let z_l = B::Scalar::from_f64(mesh.cell_z_bed(owner)).unwrap_or(B::Scalar::ZERO);
        let (u_l, v_l) = self.config.params.safe_velocity_components(
            state.hu[owner_idx], state.hv[owner_idx], h_l
        );
        let vel_l = B::vec2_new(u_l, v_l);

        // 右侧状态
        let (h_r, vel_r, z_r) = if let Some(neigh_idx) = neighbor_idx {
            let h = state.h[neigh_idx];
            let (u, v) = self.config.params.safe_velocity_components(
                state.hu[neigh_idx], state.hv[neigh_idx], h
            );
            (h, B::vec2_new(u, v), B::Scalar::from_f64(mesh.cell_z_bed(CellIndex(neigh_idx))).unwrap_or(B::Scalar::ZERO))
        } else {
            // 边界处理: vn = vel_l · normal, vel_r = vel_l - 2 * vn * normal
            let vn = B::vec2_dot(&vel_l, &normal);
            let two = B::Scalar::from_f64(2.0).unwrap_or(B::Scalar::TWO);
            let vel_r = B::vec2_sub(&vel_l, &B::vec2_scale(&normal, vn * two));
            (h_l, vel_r, z_l)
        };

        // 静水重构
        let recon = if self.config.use_hydrostatic_reconstruction {
            self.hydrostatic.reconstruct_face_simple(h_l, h_r, z_l, z_r, vel_l, vel_r)
        } else {
            HydrostaticFaceState {
                h_left: h_l,
                h_right: h_r,
                vel_left: vel_l,
                vel_right: vel_r,
                z_face: (z_l + z_r) * B::Scalar::from_f64(0.5).unwrap_or(B::Scalar::HALF),
            }
        };

        // 干湿限制
        let wet_l = self.wetting_drying.get_state(recon.h_left);
        let wet_r = self.wetting_drying.get_state(recon.h_right);
        let flux_limiter = match (wet_l, wet_r) {
            (WetState::Dry, WetState::Dry) => B::Scalar::ZERO,
            (WetState::Dry, _) | (_, WetState::Dry) => {
                let h_min = recon.h_left.min(recon.h_right);
                let h_wet = self.config.params.h_wet;
                if h_min > h_wet {
                    B::Scalar::ONE
                } else {
                    h_min / h_wet
                }
            }
            (WetState::PartiallyWet, _) | (_, WetState::PartiallyWet) => {
                let h_min = recon.h_left.min(recon.h_right);
                let h_dry = self.config.params.h_dry;
                let h_wet = self.config.params.h_wet;
                let fraction = (h_min - h_dry) / (h_wet - h_dry);
                let one = B::Scalar::ONE;
                let zero = B::Scalar::ZERO;
                if fraction > one {
                    one
                } else if fraction < zero {
                    zero
                } else {
                    fraction
                }
            }
            _ => B::Scalar::ONE,
        };

        // 黎曼通量
        let flux = self.riemann.solve(
            recon.h_left,
            recon.h_right,
            recon.vel_left,
            recon.vel_right,
            normal,
        ).unwrap_or_else(|_| RiemannFlux::zero());

        let limited_flux = flux.scaled(flux_limiter);

        // 床坡源项
        let bed_src = self.hydrostatic.bed_slope_correction(h_l, h_r, z_l, z_r, normal, length);

        (limited_flux, bed_src, length, owner, neighbor)
    }

    // =========================================================================
    // 访问器
    // =========================================================================

    /// 获取配置
    pub fn config(&self) -> &ParallelFluxConfig<B::Scalar> {
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

/// 并行计算器构建器（Backend泛型化）
pub struct ParallelFluxCalculatorBuilder<B: Backend> {
    config: ParallelFluxConfig<B::Scalar>,
    backend: B,
}

impl<B: Backend> ParallelFluxCalculatorBuilder<B> {
    pub fn new(backend: B) -> Self {
        Self {
            config: ParallelFluxConfig::default(),
            backend,
        }
    }

    pub fn config(mut self, config: ParallelFluxConfig<B::Scalar>) -> Self {
        self.config = config;
        self
    }

    pub fn params(mut self, params: NumericalParams<B::Scalar>) -> Self {
        self.config.params = params;
        self
    }

    pub fn gravity(mut self, g: B::Scalar) -> Self { 
        self.config.g = g;
        self
    }

    pub fn strategy(mut self, strategy: ParallelStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn build(self) -> ParallelFluxCalculator<B> {
        ParallelFluxCalculator::new(self.config, self.backend)
    }
}

impl<B: Backend> Default for ParallelFluxCalculatorBuilder<B> 
where
    B: Default,
{
    fn default() -> Self {
        Self::new(B::default())
    }
}
    
// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_config_default() {
        let config = ParallelFluxConfig::<f64>::default();
        assert!((config.g - 9.81).abs() < 1e-10);
        assert_eq!(config.min_parallel_size, 1000);
        assert_eq!(config.strategy, ParallelStrategy::Auto);
    }

    #[test]
    fn test_config_builder() {
        let config = ParallelFluxConfig::<f64>::builder()
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
        let config = ParallelFluxConfig::<f64>::builder()
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
        let backend = CpuBackend::<f64>::new();
        let calc = ParallelFluxCalculatorBuilder::new(backend)
            .gravity(10.0)
            .strategy(ParallelStrategy::CollectThenAccumulate)
            .build();

        assert!((calc.config().g - 10.0).abs() < 1e-10);
        assert_eq!(calc.config().strategy, ParallelStrategy::CollectThenAccumulate);
    }

    #[test]
    fn test_f32_backend() {
        let backend = CpuBackend::<f32>::new();
        let config = ParallelFluxConfig::<f32>::builder()
            .gravity(10.0f32)
            .build();
        
        let calc = ParallelFluxCalculator::<CpuBackend<f32>>::new(config, backend);
        assert!((calc.config().g - 10.0f32).abs() < 1e-6);
    }
}