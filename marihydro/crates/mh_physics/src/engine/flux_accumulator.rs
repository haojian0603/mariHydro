// crates/mh_physics/src/engine/flux_accumulator.rs

//! 通量累加器模块
//!
//! 提供通量累加功能，将面上计算的通量累加到单元上。
//! 支持 Backend 泛型，可适配 f32/f64/GPU 等不同计算后端。
//!
//! # 设计
//!
//! - `FluxAccumulator<B>` - 单线程累加器，适用于小规模问题
//! - `AtomicFluxAccumulator<B>` - 原子操作累加器，支持并行计算
//!
//! # 原子操作说明
//!
//! 当前 `AtomicFluxAccumulator` 使用 `AtomicU64` 作为临时方案，
//! 仅支持 f64 精度。完整的 f32/f64 统一原子操作接口将在
//! `RuntimeScalar` trait 中添加 `Atomic` 关联类型后实现。

use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar};
use crate::adapter::PhysicsMesh;
use crate::schemes::RiemannFlux;
use mh_runtime::FaceIndex;
use std::sync::atomic::{AtomicU64, Ordering};
use num_traits::FromPrimitive;

/// 单线程通量累加器
///
/// 用于小规模问题的串行通量累加。所有缓冲区通过 Backend 分配，
/// 支持运行时精度切换。
#[derive(Clone)]
pub struct FluxAccumulator<B: Backend> {
    n_cells: usize,
    /// 质量通量累加
    pub delta_h: B::Buffer<B::Scalar>,
    /// x 动量通量累加
    pub delta_hu: B::Buffer<B::Scalar>,
    /// y 动量通量累加
    pub delta_hv: B::Buffer<B::Scalar>,
    /// 床坡源项 x 分量
    pub bed_source_x: B::Buffer<B::Scalar>,
    /// 床坡源项 y 分量
    pub bed_source_y: B::Buffer<B::Scalar>,
    /// Backend 实例引用
    backend: B,
}

impl<B: Backend + Clone> FluxAccumulator<B> {
    /// 创建新的通量累加器
    ///
    /// # 参数
    /// - `backend`: 计算后端实例
    /// - `n_cells`: 单元数量
    pub fn new(backend: &B, n_cells: usize) -> Self {
        let zero = B::Scalar::ZERO;
        Self {
            n_cells,
            delta_h: backend.alloc_init(n_cells, zero),
            delta_hu: backend.alloc_init(n_cells, zero),
            delta_hv: backend.alloc_init(n_cells, zero),
            bed_source_x: backend.alloc_init(n_cells, zero),
            bed_source_y: backend.alloc_init(n_cells, zero),
            backend: backend.clone(),
        }
    }

    /// 重置所有累加值为零
    pub fn reset(&mut self) {
        let zero = B::Scalar::ZERO;
        self.delta_h.fill(zero);
        self.delta_hu.fill(zero);
        self.delta_hv.fill(zero);
        self.bed_source_x.fill(zero);
        self.bed_source_y.fill(zero);
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        if n_cells != self.n_cells {
            self.n_cells = n_cells;
            let zero = B::Scalar::ZERO;
            self.delta_h = self.backend.alloc_init(n_cells, zero);
            self.delta_hu = self.backend.alloc_init(n_cells, zero);
            self.delta_hv = self.backend.alloc_init(n_cells, zero);
            self.bed_source_x = self.backend.alloc_init(n_cells, zero);
            self.bed_source_y = self.backend.alloc_init(n_cells, zero);
        }
    }

    /// 获取单元数量
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 累加面通量到单元
    ///
    /// # 参数
    /// - `face_idx`: 面索引
    /// - `flux`: 黎曼通量
    /// - `length`: 面长度
    /// - `mesh`: 物理网格
    #[inline]
    pub fn accumulate_flux(
        &mut self,
        face_idx: usize,
        flux: &RiemannFlux<B::Scalar>,
        length: B::Scalar,
        mesh: &PhysicsMesh,
    ) {
        let face = FaceIndex::new(face_idx);
        let owner = mesh.face_owner(face);
        let neighbor = mesh.face_neighbor(face);

        let flux_h = flux.mass * length;
        let flux_hu = flux.momentum_x * length;
        let flux_hv = flux.momentum_y * length;

        // 所有者单元（通量流出为负）
        let owner_idx = owner.get();
        self.delta_h[owner_idx] = self.delta_h[owner_idx] - flux_h;
        self.delta_hu[owner_idx] = self.delta_hu[owner_idx] - flux_hu;
        self.delta_hv[owner_idx] = self.delta_hv[owner_idx] - flux_hv;

        // 邻居单元（通量流入为正）
        if let Some(neigh) = neighbor {
            let neigh_idx = neigh.get();
            self.delta_h[neigh_idx] = self.delta_h[neigh_idx] + flux_h;
            self.delta_hu[neigh_idx] = self.delta_hu[neigh_idx] + flux_hu;
            self.delta_hv[neigh_idx] = self.delta_hv[neigh_idx] + flux_hv;
        }
    }

    /// 累加床坡源项
    #[inline]
    pub fn accumulate_bed_source(&mut self, cell_idx: usize, source_x: B::Scalar, source_y: B::Scalar) {
        self.bed_source_x[cell_idx] = self.bed_source_x[cell_idx] + source_x;
        self.bed_source_y[cell_idx] = self.bed_source_y[cell_idx] + source_y;
    }

    /// 应用累加的通量到状态
    ///
    /// # 参数
    /// - `h`: 水深数组（就地修改）
    /// - `hu`: x 动量数组（就地修改）
    /// - `hv`: y 动量数组（就地修改）
    /// - `areas`: 单元面积数组
    /// - `dt`: 时间步长
    pub fn apply_to_state(
        &self,
        h: &mut B::Buffer<B::Scalar>,
        hu: &mut B::Buffer<B::Scalar>,
        hv: &mut B::Buffer<B::Scalar>,
        areas: &[B::Scalar],
        dt: B::Scalar,
    ) {
        for i in 0..self.n_cells {
            let inv_area = B::Scalar::ONE / areas[i];
            let dt_inv = dt * inv_area;
            h[i] = h[i] + dt_inv * self.delta_h[i];
            hu[i] = hu[i] + dt_inv * (self.delta_hu[i] + self.bed_source_x[i]);
            hv[i] = hv[i] + dt_inv * (self.delta_hv[i] + self.bed_source_y[i]);
        }
    }
}

/// 原子操作通量累加器
///
/// 支持并行计算的通量累加器，使用原子操作避免数据竞争。
/// 适用于大规模问题的并行计算。
pub struct AtomicFluxAccumulator<B: Backend> {
    n_cells: usize,
    delta_h: Vec<AtomicU64>,
    delta_hu: Vec<AtomicU64>,
    delta_hv: Vec<AtomicU64>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> AtomicFluxAccumulator<B> {
    /// 创建新的原子通量累加器
    pub fn new(_backend: &B, n_cells: usize) -> Self {
        Self {
            n_cells,
            delta_h: (0..n_cells).map(|_| AtomicU64::new(0)).collect(),
            delta_hu: (0..n_cells).map(|_| AtomicU64::new(0)).collect(),
            delta_hv: (0..n_cells).map(|_| AtomicU64::new(0)).collect(),
            _marker: std::marker::PhantomData,
        }
    }

    /// 重置所有累加值为零
    pub fn reset(&self) {
        for i in 0..self.n_cells {
            self.delta_h[i].store(0, Ordering::Relaxed);
            self.delta_hu[i].store(0, Ordering::Relaxed);
            self.delta_hv[i].store(0, Ordering::Relaxed);
        }
    }

    /// 获取单元数量
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 原子加法操作
    ///
    /// 使用 compare-exchange 循环实现浮点数的原子加法
    #[inline]
    fn atomic_add(atomic: &AtomicU64, val: f64) {
        let mut old = atomic.load(Ordering::Relaxed);
        loop {
            let old_f = f64::from_bits(old);
            let new_f = old_f + val;
            match atomic.compare_exchange_weak(
                old,
                new_f.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => old = x,
            }
        }
    }

    /// 累加通量到指定单元（线程安全）
    #[inline]
    pub fn accumulate(&self, cell_idx: usize, dh: f64, dhu: f64, dhv: f64) {
        Self::atomic_add(&self.delta_h[cell_idx], dh);
        Self::atomic_add(&self.delta_hu[cell_idx], dhu);
        Self::atomic_add(&self.delta_hv[cell_idx], dhv);
    }

    /// 累加面通量（线程安全）
    ///
    /// 同时更新 owner 和 neighbor 单元
    pub fn accumulate_flux(&self, owner: usize, neighbor: Option<usize>, fh: f64, fhu: f64, fhv: f64) {
        // 所有者单元（通量流出为负）
        self.accumulate(owner, -fh, -fhu, -fhv);

        // 邻居单元（通量流入为正）
        if let Some(neigh) = neighbor {
            self.accumulate(neigh, fh, fhu, fhv);
        }
    }

    /// 收集累加结果
    ///
    /// 返回 (delta_h, delta_hu, delta_hv) 的非原子副本
    pub fn collect(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let h: Vec<f64> = self.delta_h
            .iter()
            .map(|a| f64::from_bits(a.load(Ordering::Relaxed)))
            .collect();
        let hu: Vec<f64> = self.delta_hu
            .iter()
            .map(|a| f64::from_bits(a.load(Ordering::Relaxed)))
            .collect();
        let hv: Vec<f64> = self.delta_hv
            .iter()
            .map(|a| f64::from_bits(a.load(Ordering::Relaxed)))
            .collect();
        (h, hu, hv)
    }

    /// 应用累加的通量到状态
    pub fn apply_to_state(
        &self,
        h: &mut B::Buffer<B::Scalar>,
        hu: &mut B::Buffer<B::Scalar>,
        hv: &mut B::Buffer<B::Scalar>,
        areas: &[B::Scalar],
        dt: B::Scalar,
    ) {
        let (delta_h, delta_hu, delta_hv) = self.collect();
        for i in 0..self.n_cells {
            let inv_area = B::Scalar::ONE / areas[i];
            let dt_inv = dt * inv_area;
            let dh = B::Scalar::from_f64(delta_h[i]).unwrap_or(B::Scalar::ZERO);
            let dhu = B::Scalar::from_f64(delta_hu[i]).unwrap_or(B::Scalar::ZERO);
            let dhv = B::Scalar::from_f64(delta_hv[i]).unwrap_or(B::Scalar::ZERO);

            h[i] = h[i] + dt_inv * dh;
            hu[i] = hu[i] + dt_inv * dhu;
            hv[i] = hv[i] + dt_inv * dhv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    #[test]
    fn 测试单线程累加器创建() {
        let backend = CpuBackend::<f64>::new();
        let acc = FluxAccumulator::new(&backend, 10);
        assert_eq!(acc.n_cells(), 10);
        assert!(acc.delta_h.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn 测试单线程累加器重置() {
        let backend = CpuBackend::<f64>::new();
        let mut acc = FluxAccumulator::new(&backend, 5);
        acc.delta_h[0] = 1.0;
        acc.delta_hu[1] = 2.0;
        acc.delta_hv[2] = 3.0;

        acc.reset();

        assert!(acc.delta_h.iter().all(|&x| x == 0.0));
        assert!(acc.delta_hu.iter().all(|&x| x == 0.0));
        assert!(acc.delta_hv.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn 测试源项累加() {
        let backend = CpuBackend::<f64>::new();
        let mut acc = FluxAccumulator::new(&backend, 3);
        acc.accumulate_bed_source(0, 1.0, 2.0);
        acc.accumulate_bed_source(0, 0.5, 0.5);

        assert!((acc.bed_source_x[0] - 1.5).abs() < 1e-10);
        assert!((acc.bed_source_y[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn 测试状态更新() {
        let backend = CpuBackend::<f64>::new();
        let mut acc = FluxAccumulator::new(&backend, 2);
        acc.delta_h[0] = 10.0;
        acc.delta_h[1] = -10.0;
        acc.delta_hu[0] = 5.0;
        acc.delta_hv[1] = 3.0;
        acc.bed_source_x[0] = 1.0;

        let mut h = backend.alloc_init(2, 1.0);
        let mut hu = backend.alloc_init(2, 0.0);
        let mut hv = backend.alloc_init(2, 0.0);
        let areas = backend.alloc_init(2, 1.0);
        let dt = 0.1_f64;

        acc.apply_to_state(&mut h, &mut hu, &mut hv, &areas, dt);

        assert!((h[0] - 2.0).abs() < 1e-10);
        assert!((h[1] - 0.5).abs() < 1e-10);
        assert!((hu[0] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn 测试原子累加器创建() {
        let backend = CpuBackend::<f64>::new();
        let acc = AtomicFluxAccumulator::new(&backend, 10);
        assert_eq!(acc.n_cells(), 10);
    }

    #[test]
    fn 测试原子累加() {
        let backend = CpuBackend::<f64>::new();
        let acc = AtomicFluxAccumulator::new(&backend, 3);
        acc.accumulate(0, 1.0, 2.0, 3.0);
        acc.accumulate(0, 0.5, 0.5, 0.5);

        let (h, hu, hv) = acc.collect();
        assert!((h[0] - 1.5).abs() < 1e-10);
        assert!((hu[0] - 2.5).abs() < 1e-10);
        assert!((hv[0] - 3.5).abs() < 1e-10);
    }

    #[test]
    fn 测试原子面通量累加() {
        let backend = CpuBackend::<f64>::new();
        let acc = AtomicFluxAccumulator::new(&backend, 3);

        acc.accumulate_flux(0, Some(1), 1.0, 2.0, 3.0);

        let (h, _hu, _hv) = acc.collect();
        assert!((h[0] - (-1.0)).abs() < 1e-10);
        assert!((h[1] - 1.0).abs() < 1e-10);
    }
}