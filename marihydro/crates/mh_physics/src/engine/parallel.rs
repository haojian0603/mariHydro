// crates/mh_physics/src/engine/parallel.rs

//! 并行通量计算器
//!
//! 提供多种并行策略加速通量计算，支持任意Backend（CPU f32/f64、GPU）。
//! 采用图着色算法实现无锁并行累加，适用于大规模网格。
//!
//! # 并行策略
//!
//! - `Sequential`: 串行计算，适用于小规模问题
//! - `CollectThenAccumulate`: 并行计算通量后串行累加
//! - `Colored`: 基于图着色的无锁并行累加
//! - `Auto`: 根据问题规模自动选择策略
//!
//! # 线程安全
//!
//! `Colored`策略使用图着色确保同一颜色的面不共享单元，并通过原子累加
//! 保证并行写入无数据竞争。

#![allow(unsafe_code)]

use crate::adapter::{CellIndex, PhysicsMesh};
use crate::engine::solver::{BedSlopeCorrection, HydrostaticFaceState, HydrostaticReconstruction};
use crate::schemes::riemann::{HllcSolver, RiemannFlux, RiemannSolver};
use crate::schemes::wetting_drying::{WetState, WettingDryingHandler};
use crate::state::ShallowWaterState;
use crate::types::NumericalParams;

use log::info;
use mh_runtime::{AtomicScalar, Backend, DeviceBuffer, FaceIndex as RuntimeFaceIndex, RuntimeScalar};
use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// 生命周期绑定的可发送裸指针封装
#[derive(Clone, Copy)]
struct SendPtr<'a, T: ?Sized> {
    ptr: *mut T,
    _marker: PhantomData<&'a mut T>,
}

// SAFETY: 生命周期由 PhantomData 绑定，使用者需确保不产生别名冲突
unsafe impl<'a, T: ?Sized + Send> Send for SendPtr<'a, T> {}
unsafe impl<'a, T: ?Sized + Send + Sync> Sync for SendPtr<'a, T> {}

impl<'a, T: ?Sized> SendPtr<'a, T> {
    fn new(ptr: *mut T) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// 从可变引用创建
    fn from_ref(r: &'a mut T) -> Self {
        Self::new(r as *mut T)
    }

    /// 取出可变引用（调用方需保证无别名写冲突）
    unsafe fn as_mut(&self) -> &'a mut T {
        &mut *self.ptr
    }
}

/// 并行策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParallelStrategy {
    /// 串行执行
    Sequential,
    /// 并行计算通量后串行累加
    CollectThenAccumulate,
    /// 基于图着色的无锁并行累加
    Colored,
    /// 自动选择最优策略
    #[default]
    Auto,
}

/// 并行计算配置
///
/// 控制并行策略、阈值和数值参数。所有数值字段使用`B::Scalar`泛型，
/// 支持运行时精度切换。
#[derive(Debug, Clone)]
pub struct ParallelFluxConfig<S: RuntimeScalar> {
    /// 数值参数
    pub params: NumericalParams<S>,
    /// 重力加速度
    pub g: S,
    /// 最小并行面数
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
            g: S::from_f64(9.81).unwrap_or(S::ZERO),
            min_parallel_size: 1000,
            strategy: ParallelStrategy::Auto,
            use_hydrostatic_reconstruction: true,
        }
    }
}

impl<S: RuntimeScalar> ParallelFluxConfig<S> {
    /// 创建配置构建器
    pub fn builder() -> ParallelFluxConfigBuilder<S> {
        ParallelFluxConfigBuilder::default()
    }
}

/// 并行计算配置构建器
#[derive(Debug)]
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
    /// 设置数值参数
    pub fn params(mut self, params: NumericalParams<S>) -> Self {
        self.config.params = params;
        self
    }

    /// 设置重力加速度
    pub fn gravity(mut self, g: S) -> Self {
        self.config.g = g;
        self
    }

    /// 设置最小并行面数
    pub fn min_parallel_size(mut self, size: usize) -> Self {
        self.config.min_parallel_size = size;
        self
    }

    /// 设置并行策略
    pub fn strategy(mut self, strategy: ParallelStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// 设置是否启用静水重构
    pub fn use_hydrostatic_reconstruction(mut self, enable: bool) -> Self {
        self.config.use_hydrostatic_reconstruction = enable;
        self
    }

    /// 构建配置
    pub fn build(self) -> ParallelFluxConfig<S> {
        self.config
    }
}

/// 性能指标
///
/// 记录并行计算的调用次数、处理面数和时间消耗，用于性能分析。
#[derive(Debug, Clone, Default)]
pub struct FluxComputeMetrics {
    /// 总计算次数
    pub total_calls: usize,
    /// 并行计算次数
    pub parallel_calls: usize,
    /// 串行计算次数
    pub sequential_calls: usize,
    /// 累计计算时间
    pub total_duration: Duration,
    /// 累计处理面数
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

fn create_atomic_buffer<S: RuntimeScalar>(len: usize) -> Vec<S::Atomic> {
    let zero = S::ZERO;
    (0..len).map(|_| S::Atomic::new(zero)).collect()
}

/// 并行通量计算器
///
/// 封装多种并行策略的通量计算逻辑，支持任意Backend。
///
/// # 类型参数
/// - `B`: 计算后端，提供存储和计算能力
pub struct ParallelFluxCalculator<B: Backend> {
    config: ParallelFluxConfig<B::Scalar>,
    riemann: HllcSolver<B>,
    wetting_drying: WettingDryingHandler<B>,
    hydrostatic: HydrostaticReconstruction<B>,
    metrics: FluxComputeMetrics,
    face_colors: Option<Vec<Vec<usize>>>,
    _backend: B,
}

impl<B: Backend> ParallelFluxCalculator<B>
where
    B::Scalar: RuntimeScalar,
{
    /// 创建计算器
    pub fn new(config: ParallelFluxConfig<B::Scalar>, backend: B) -> Self {
        let riemann_params =
            crate::schemes::riemann::SolverParams::<B::Scalar>::from_numerical(&config.params, config.g);
        Self {
            riemann: HllcSolver::<B>::new(&riemann_params, config.g),
            wetting_drying: WettingDryingHandler::<B>::from_params(&config.params),
            hydrostatic: HydrostaticReconstruction::<B>::new(&riemann_params, config.g),
            metrics: FluxComputeMetrics::default(),
            face_colors: None,
            config,
            _backend: backend,
        }
    }

    /// 为网格设置面着色
    ///
    /// 构建面邻接图并执行贪心着色算法，确保同一颜色的面不共享单元。
    /// 此方法必须在首次使用`Colored`策略前调用。
    pub fn setup_face_coloring(&mut self, mesh: &PhysicsMesh) {
        let _start = Instant::now();
        let n_faces = mesh.n_faces();

        if n_faces == 0 {
            self.face_colors = Some(Vec::new());
            return;
        }

        let face_neighbors = self.build_face_adjacency(mesh);
        let (face_color, num_colors) = self.greedy_coloring(&face_neighbors, n_faces);

        let mut color_faces = vec![Vec::new(); num_colors];
        for (face, &color) in face_color.iter().enumerate() {
            if color != usize::MAX {
                color_faces[color].push(face);
            }
        }

        let _duration = _start.elapsed();

        info!(
            "面着色完成：{} 个面，{} 种颜色，耗时 {:?}",
            n_faces, num_colors, _duration
        );

        self.face_colors = Some(color_faces);
    }

    /// 构建面邻接关系
    fn build_face_adjacency(
        &self,
        mesh: &PhysicsMesh,
    ) -> Vec<std::collections::HashSet<usize>> {
        use std::collections::{HashMap, HashSet};

        let n_faces = mesh.n_faces();
        let mut cell_to_faces: HashMap<usize, Vec<usize>> = HashMap::new();

        for face_idx in 0..n_faces {
            let owner = mesh.face_owner(RuntimeFaceIndex(face_idx));
            cell_to_faces
                .entry(owner.get())
                .or_default()
                .push(face_idx);

            if let Some(neigh) = mesh.face_neighbor(RuntimeFaceIndex(face_idx)) {
                cell_to_faces.entry(neigh.get()).or_default().push(face_idx);
            }
        }

        let mut face_neighbors = vec![HashSet::new(); n_faces];
        for faces in cell_to_faces.values() {
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
        n_faces: usize,
    ) -> (Vec<usize>, usize) {
        use std::collections::HashSet;

        let mut face_color = vec![usize::MAX; n_faces];
        let mut num_colors = 0;

        let mut order: Vec<usize> = (0..n_faces).collect();
        order.sort_by_key(|&f| std::cmp::Reverse(face_neighbors[f].len()));

        for &face in &order {
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
            ParallelStrategy::Sequential => (
                self.compute_serial(
                    state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv,
                ),
                false,
            ),
            ParallelStrategy::CollectThenAccumulate => (
                self.compute_parallel(
                    state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv,
                ),
                true,
            ),
            ParallelStrategy::Colored => {
                if !self.has_face_coloring() {
                    self.setup_face_coloring(mesh);
                }
                (
                    self.compute_colored(
                        state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv,
                    ),
                    true,
                )
            }
            ParallelStrategy::Auto => {
                if n_faces < self.config.min_parallel_size {
                    (
                        self.compute_serial(
                            state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv,
                        ),
                        false,
                    )
                } else if self.has_face_coloring() {
                    (
                        self.compute_colored(
                            state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv,
                        ),
                        true,
                    )
                } else {
                    (
                        self.compute_parallel(
                            state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv,
                        ),
                        true,
                    )
                }
            }
        };

        self.metrics.record(n_faces, is_parallel, start.elapsed());
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
        let zero = B::Scalar::ZERO;
        flux_h.fill(zero);
        flux_hu.fill(zero);
        flux_hv.fill(zero);
        source_hu.fill(zero);
        source_hv.fill(zero);

        let n_faces = mesh.n_faces();
        let mut max_wave_speed = zero;

        for face_idx in 0..n_faces {
            let (flux, bed_src, length, owner, neighbor) =
                self.compute_face(state, mesh, RuntimeFaceIndex(face_idx));

            if flux.max_wave_speed > max_wave_speed {
                max_wave_speed = flux.max_wave_speed;
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

        max_wave_speed
    }

    /// 并行计算后串行累加
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
        let zero = B::Scalar::ZERO;
        let n_faces = mesh.n_faces();
        let n_cells = mesh.n_cells();

        let max_speed_atomic = <B::Scalar as RuntimeScalar>::Atomic::new(zero);
        let flux_h_atomic = create_atomic_buffer::<B::Scalar>(n_cells);
        let flux_hu_atomic = create_atomic_buffer::<B::Scalar>(n_cells);
        let flux_hv_atomic = create_atomic_buffer::<B::Scalar>(n_cells);
        let source_hu_atomic = create_atomic_buffer::<B::Scalar>(n_cells);
        let source_hv_atomic = create_atomic_buffer::<B::Scalar>(n_cells);

        (0..n_faces)
            .into_par_iter()
            .for_each(|face_idx| {
                let (flux, bed_src, length, owner, neighbor) =
                    self.compute_face(state, mesh, RuntimeFaceIndex(face_idx));

                max_speed_atomic.fetch_max(flux.max_wave_speed, Ordering::Relaxed);

                let fh = flux.mass * length;
                let fhu = flux.momentum_x * length;
                let fhv = flux.momentum_y * length;

                let owner_idx = owner.get();
                flux_h_atomic[owner_idx].fetch_add(-fh, Ordering::Relaxed);
                flux_hu_atomic[owner_idx].fetch_add(-fhu, Ordering::Relaxed);
                flux_hv_atomic[owner_idx].fetch_add(-fhv, Ordering::Relaxed);
                source_hu_atomic[owner_idx].fetch_add(bed_src.source_left_x, Ordering::Relaxed);
                source_hv_atomic[owner_idx].fetch_add(bed_src.source_left_y, Ordering::Relaxed);

                if let Some(neigh) = neighbor {
                    let neigh_idx = neigh.get();
                    flux_h_atomic[neigh_idx].fetch_add(fh, Ordering::Relaxed);
                    flux_hu_atomic[neigh_idx].fetch_add(fhu, Ordering::Relaxed);
                    flux_hv_atomic[neigh_idx].fetch_add(fhv, Ordering::Relaxed);
                    source_hu_atomic[neigh_idx].fetch_add(bed_src.source_right_x, Ordering::Relaxed);
                    source_hv_atomic[neigh_idx].fetch_add(bed_src.source_right_y, Ordering::Relaxed);
                }
            });

        for i in 0..n_cells {
            flux_h[i] = flux_h_atomic[i].load(Ordering::Relaxed);
            flux_hu[i] = flux_hu_atomic[i].load(Ordering::Relaxed);
            flux_hv[i] = flux_hv_atomic[i].load(Ordering::Relaxed);
            source_hu[i] = source_hu_atomic[i].load(Ordering::Relaxed);
            source_hv[i] = source_hv_atomic[i].load(Ordering::Relaxed);
        }

        max_speed_atomic.load(Ordering::Relaxed)
    }

    /// 基于图着色的无锁并行累加
    ///
    /// 通过贪心着色算法将面分组，确保同组面不共享单元，实现无锁并行写入。
    /// 使用图着色提供的“无共享单元”保证进行并行累加。
    ///
    /// 优先走“可变切片”快路径以避免原子开销；若 backend 无法暴露可变切片则
    /// 回退到原子累加路径。
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
        let zero = B::Scalar::ZERO;
        let color_faces = match &self.face_colors {
            Some(cf) => cf,
            None => return zero,
        };

        // 快路径：backend 支持可变切片，结合颜色保证同色不共享单元，消除原子开销。
        if let (
            Some(flux_h_slice),
            Some(flux_hu_slice),
            Some(flux_hv_slice),
            Some(source_hu_slice),
            Some(source_hv_slice),
        ) = (
            flux_h.try_as_slice_mut(),
            flux_hu.try_as_slice_mut(),
            flux_hv.try_as_slice_mut(),
            source_hu.try_as_slice_mut(),
            source_hv.try_as_slice_mut(),
        ) {
            flux_h_slice.fill(zero);
            flux_hu_slice.fill(zero);
            flux_hv_slice.fill(zero);
            source_hu_slice.fill(zero);
            source_hv_slice.fill(zero);

            let max_speed_atomic = <B::Scalar as RuntimeScalar>::Atomic::new(zero);
            let flux_h_ptr = SendPtr::from_ref(flux_h_slice);
            let flux_hu_ptr = SendPtr::from_ref(flux_hu_slice);
            let flux_hv_ptr = SendPtr::from_ref(flux_hv_slice);
            let source_hu_ptr = SendPtr::from_ref(source_hu_slice);
            let source_hv_ptr = SendPtr::from_ref(source_hv_slice);

            for faces_in_color in color_faces {
                faces_in_color.par_iter().for_each(|&face_idx| {
                    let (flux, bed_src, length, owner, neighbor) =
                        self.compute_face(state, mesh, RuntimeFaceIndex(face_idx));

                    max_speed_atomic.fetch_max(flux.max_wave_speed, Ordering::Relaxed);

                    let fh = flux.mass * length;
                    let fhu = flux.momentum_x * length;
                    let fhv = flux.momentum_y * length;

                    // SAFETY: 着色保证同一颜色内的 faces 不共享单元，避免别名写冲突
                    let flux_h = unsafe { flux_h_ptr.as_mut() };
                    let flux_hu = unsafe { flux_hu_ptr.as_mut() };
                    let flux_hv = unsafe { flux_hv_ptr.as_mut() };
                    let source_hu = unsafe { source_hu_ptr.as_mut() };
                    let source_hv = unsafe { source_hv_ptr.as_mut() };

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
                });
            }

            return max_speed_atomic.load(Ordering::Relaxed);
        }

        // 回退：backend 不支持切片时使用原子累加路径。
        self.compute_colored_atomic(
            state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv,
        )
    }

    /// 原子累加回退路径，适用于无法获取可变切片的 backend。
    fn compute_colored_atomic(
        &self,
        state: &ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        flux_h: &mut B::Buffer<B::Scalar>,
        flux_hu: &mut B::Buffer<B::Scalar>,
        flux_hv: &mut B::Buffer<B::Scalar>,
        source_hu: &mut B::Buffer<B::Scalar>,
        source_hv: &mut B::Buffer<B::Scalar>,
    ) -> B::Scalar {
        let zero = B::Scalar::ZERO;
        let n_cells = mesh.n_cells();
        let color_faces = match &self.face_colors {
            Some(cf) => cf,
            None => return zero,
        };

        let max_speed_atomic = <B::Scalar as RuntimeScalar>::Atomic::new(zero);
        let flux_h_atomic = create_atomic_buffer::<B::Scalar>(n_cells);
        let flux_hu_atomic = create_atomic_buffer::<B::Scalar>(n_cells);
        let flux_hv_atomic = create_atomic_buffer::<B::Scalar>(n_cells);
        let source_hu_atomic = create_atomic_buffer::<B::Scalar>(n_cells);
        let source_hv_atomic = create_atomic_buffer::<B::Scalar>(n_cells);

        for faces_in_color in color_faces {
            faces_in_color.par_iter().for_each(|&face_idx| {
                let (flux, bed_src, length, owner, neighbor) =
                    self.compute_face(state, mesh, RuntimeFaceIndex(face_idx));

                max_speed_atomic.fetch_max(flux.max_wave_speed, Ordering::Relaxed);

                let fh = flux.mass * length;
                let fhu = flux.momentum_x * length;
                let fhv = flux.momentum_y * length;

                let owner_idx = owner.get();
                flux_h_atomic[owner_idx].fetch_add(-fh, Ordering::Relaxed);
                flux_hu_atomic[owner_idx].fetch_add(-fhu, Ordering::Relaxed);
                flux_hv_atomic[owner_idx].fetch_add(-fhv, Ordering::Relaxed);
                source_hu_atomic[owner_idx].fetch_add(bed_src.source_left_x, Ordering::Relaxed);
                source_hv_atomic[owner_idx].fetch_add(bed_src.source_left_y, Ordering::Relaxed);

                if let Some(neigh) = neighbor {
                    let neigh_idx = neigh.get();
                    flux_h_atomic[neigh_idx].fetch_add(fh, Ordering::Relaxed);
                    flux_hu_atomic[neigh_idx].fetch_add(fhu, Ordering::Relaxed);
                    flux_hv_atomic[neigh_idx].fetch_add(fhv, Ordering::Relaxed);
                    source_hu_atomic[neigh_idx].fetch_add(bed_src.source_right_x, Ordering::Relaxed);
                    source_hv_atomic[neigh_idx].fetch_add(bed_src.source_right_y, Ordering::Relaxed);
                }
            });
        }

        for i in 0..n_cells {
            flux_h[i] = flux_h_atomic[i].load(Ordering::Relaxed);
            flux_hu[i] = flux_hu_atomic[i].load(Ordering::Relaxed);
            flux_hv[i] = flux_hv_atomic[i].load(Ordering::Relaxed);
            source_hu[i] = source_hu_atomic[i].load(Ordering::Relaxed);
            source_hv[i] = source_hv_atomic[i].load(Ordering::Relaxed);
        }

        max_speed_atomic.load(Ordering::Relaxed)
    }

    /// 计算单个面的通量和源项
    fn compute_face(
        &self,
        state: &ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        face_idx: RuntimeFaceIndex,
    ) -> (
        RiemannFlux<B::Scalar>,
        BedSlopeCorrection<B>,
        B::Scalar,
        CellIndex,
        Option<CellIndex>,
    ) {
        let normal = mesh.face_normal_generic::<B>(face_idx)
            .expect("边界面法向量转换失败：坐标超出Backend标量范围");
        let length_f64 = mesh.face_length(face_idx);
        let length = B::Scalar::from_f64(length_f64).unwrap_or(B::Scalar::ZERO);
        let owner = mesh.face_owner(face_idx);
        let neighbor = mesh.face_neighbor(face_idx);

        let owner_idx = owner.get();
        let h_l = state.h[owner_idx];
        let z_l = state.z[owner_idx];
        let (u_l, v_l) = self.config.params.safe_velocity_components(
            state.hu[owner_idx], state.hv[owner_idx], h_l,
        );
        let vel_l = B::vec2_new(u_l, v_l);

        let (h_r, vel_r, z_r) = if let Some(neigh) = neighbor {
            let neigh_idx = neigh.get();
            let h = state.h[neigh_idx];
            let (u, v) = self.config.params.safe_velocity_components(
                state.hu[neigh_idx], state.hv[neigh_idx], h,
            );
            (h, B::vec2_new(u, v), state.z[neigh_idx])
        } else {
            let vn = B::vec2_dot(&vel_l, &normal);
            let two = B::Scalar::from_f64(2.0).unwrap_or(B::Scalar::TWO);
            let vel_r = B::vec2_sub(&vel_l, &B::vec2_scale(&normal, vn * two));
            (h_l, vel_r, z_l)
        };

        let recon_state = if self.config.use_hydrostatic_reconstruction {
            self.hydrostatic.reconstruct_face_simple(h_l, h_r, z_l, z_r, vel_l, vel_r)
        } else {
            let half = B::Scalar::from_f64(0.5).unwrap_or(B::Scalar::HALF);
            HydrostaticFaceState {
                h_left: h_l,
                h_right: h_r,
                vel_left: vel_l,
                vel_right: vel_r,
                z_face: (z_l + z_r) * half,
            }
        };

        let wet_l = self.wetting_drying.get_state(recon_state.h_left);
        let wet_r = self.wetting_drying.get_state(recon_state.h_right);
        let flux_limiter = match (wet_l, wet_r) {
            (WetState::Dry, WetState::Dry) => B::Scalar::ZERO,
            (WetState::Dry, _) | (_, WetState::Dry) => B::Scalar::ONE,
            (WetState::PartiallyWet, WetState::PartiallyWet) => {
                let h_min = recon_state.h_left.min(recon_state.h_right);
                let fraction = (h_min - self.config.params.h_dry)
                    / (self.config.params.h_wet - self.config.params.h_dry);
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

        let flux = self.riemann.solve(
            recon_state.h_left,
            recon_state.h_right,
            recon_state.vel_left,
            recon_state.vel_right,
            normal,
        ).unwrap_or_else(|_| RiemannFlux::zero());

        let bed_src = self.hydrostatic.bed_slope_correction(
            h_l, h_r, recon_state.h_left, recon_state.h_right, normal, length,
        );

        (flux.scaled(flux_limiter), bed_src, length, owner, neighbor)
    }

    /// 获取配置引用
    pub fn config(&self) -> &ParallelFluxConfig<B::Scalar> {
        &self.config
    }

    /// 获取性能指标引用
    pub fn metrics(&self) -> &FluxComputeMetrics {
        &self.metrics
    }

    /// 重置性能指标
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }
}

/// 并行计算器构建器
#[derive(Debug)]
pub struct ParallelFluxCalculatorBuilder<B: Backend> {
    config: ParallelFluxConfig<B::Scalar>,
    backend: B,
}

impl<B: Backend> ParallelFluxCalculatorBuilder<B> {
    /// 创建构建器
    pub fn new(backend: B) -> Self {
        Self {
            config: ParallelFluxConfig::default(),
            backend,
        }
    }

    /// 设置配置
    pub fn config(mut self, config: ParallelFluxConfig<B::Scalar>) -> Self {
        self.config = config;
        self
    }

    /// 设置数值参数
    pub fn params(mut self, params: NumericalParams<B::Scalar>) -> Self {
        self.config.params = params;
        self
    }

    /// 设置重力加速度
    pub fn gravity(mut self, g: B::Scalar) -> Self {
        self.config.g = g;
        self
    }

    /// 设置并行策略
    pub fn strategy(mut self, strategy: ParallelStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// 构建计算器
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
    fn test_metrics_record() {
        let mut metrics = FluxComputeMetrics::default();
        metrics.record(1000, true, Duration::from_millis(10));
        metrics.record(500, false, Duration::from_millis(5));

        assert_eq!(metrics.total_calls, 2);
        assert_eq!(metrics.parallel_calls, 1);
        assert_eq!(metrics.sequential_calls, 1);
        assert_eq!(metrics.total_faces, 1500);
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