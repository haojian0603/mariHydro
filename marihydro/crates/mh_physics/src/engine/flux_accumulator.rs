// crates/mh_physics/src/engine/flux_accumulator.rs

//! 通量累加器模块
//!
//! 提供通量累加功能，将面上计算的通量累加到单元上。
//!
//! # 设计
//!
//! - `FluxAccumulator` - 单线程累加器，适用于小规模问题
//! - `AtomicFluxAccumulator` - 原子操作累加器，支持并行计算
//!

use crate::adapter::PhysicsMesh;
use crate::schemes::RiemannFlux;
use std::sync::atomic::{AtomicU64, Ordering};

/// 单线程通量累加器
///
/// 用于小规模问题的串行通量累加。
#[derive(Clone)]
pub struct FluxAccumulator {
    n_cells: usize,
    /// 质量通量累加
    pub delta_h: Vec<f64>,
    /// x动量通量累加
    pub delta_hu: Vec<f64>,
    /// y动量通量累加
    pub delta_hv: Vec<f64>,
    /// 床坡源项x分量
    pub bed_source_x: Vec<f64>,
    /// 床坡源项y分量
    pub bed_source_y: Vec<f64>,
}

impl FluxAccumulator {
    /// 创建新的通量累加器
    ///
    /// # 参数
    /// - `n_cells`: 单元数量
    pub fn new(n_cells: usize) -> Self {
        Self {
            n_cells,
            delta_h: vec![0.0; n_cells],
            delta_hu: vec![0.0; n_cells],
            delta_hv: vec![0.0; n_cells],
            bed_source_x: vec![0.0; n_cells],
            bed_source_y: vec![0.0; n_cells],
        }
    }

    /// 重置所有累加值为零
    pub fn reset(&mut self) {
        self.delta_h.fill(0.0);
        self.delta_hu.fill(0.0);
        self.delta_hv.fill(0.0);
        self.bed_source_x.fill(0.0);
        self.bed_source_y.fill(0.0);
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        self.n_cells = n_cells;
        self.delta_h.resize(n_cells, 0.0);
        self.delta_hu.resize(n_cells, 0.0);
        self.delta_hv.resize(n_cells, 0.0);
        self.bed_source_x.resize(n_cells, 0.0);
        self.bed_source_y.resize(n_cells, 0.0);
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
    pub fn accumulate_flux(&mut self, face_idx: usize, flux: &RiemannFlux, length: f64, mesh: &PhysicsMesh) {
        let owner = mesh.face_owner(face_idx);
        let neighbor = mesh.face_neighbor(face_idx);

        let flux_h = flux.mass * length;
        let flux_hu = flux.momentum_x * length;
        let flux_hv = flux.momentum_y * length;

        // 所有者单元（通量流出为负）
        self.delta_h[owner] -= flux_h;
        self.delta_hu[owner] -= flux_hu;
        self.delta_hv[owner] -= flux_hv;

        // 邻居单元（通量流入为正）
        if let Some(neigh) = neighbor {
            self.delta_h[neigh] += flux_h;
            self.delta_hu[neigh] += flux_hu;
            self.delta_hv[neigh] += flux_hv;
        }
    }

    /// 累加床坡源项
    #[inline]
    pub fn accumulate_bed_source(&mut self, cell_idx: usize, source_x: f64, source_y: f64) {
        self.bed_source_x[cell_idx] += source_x;
        self.bed_source_y[cell_idx] += source_y;
    }

    /// 应用累加的通量到状态
    ///
    /// # 参数
    /// - `h`: 水深数组 (就地修改)
    /// - `hu`: x动量数组 (就地修改)
    /// - `hv`: y动量数组 (就地修改)
    /// - `areas`: 单元面积数组
    /// - `dt`: 时间步长
    pub fn apply_to_state(
        &self,
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
        areas: &[f64],
        dt: f64,
    ) {
        for i in 0..self.n_cells {
            let inv_area = 1.0 / areas[i];
            h[i] += dt * inv_area * self.delta_h[i];
            hu[i] += dt * inv_area * (self.delta_hu[i] + self.bed_source_x[i]);
            hv[i] += dt * inv_area * (self.delta_hv[i] + self.bed_source_y[i]);
        }
    }
}

/// 原子操作通量累加器
///
/// 支持并行计算的通量累加器，使用原子操作避免数据竞争。
/// 适用于大规模问题的并行计算。
pub struct AtomicFluxAccumulator {
    n_cells: usize,
    delta_h: Vec<AtomicU64>,
    delta_hu: Vec<AtomicU64>,
    delta_hv: Vec<AtomicU64>,
}

impl AtomicFluxAccumulator {
    /// 创建新的原子通量累加器
    pub fn new(n_cells: usize) -> Self {
        Self {
            n_cells,
            delta_h: (0..n_cells).map(|_| AtomicU64::new(0)).collect(),
            delta_hu: (0..n_cells).map(|_| AtomicU64::new(0)).collect(),
            delta_hv: (0..n_cells).map(|_| AtomicU64::new(0)).collect(),
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
    /// 同时更新owner和neighbor单元
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
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
        areas: &[f64],
        dt: f64,
    ) {
        let (delta_h, delta_hu, delta_hv) = self.collect();
        for i in 0..self.n_cells {
            let inv_area = 1.0 / areas[i];
            h[i] += dt * inv_area * delta_h[i];
            hu[i] += dt * inv_area * delta_hu[i];
            hv[i] += dt * inv_area * delta_hv[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flux_accumulator_new() {
        let acc = FluxAccumulator::new(10);
        assert_eq!(acc.n_cells(), 10);
        assert!(acc.delta_h.iter().all(|&x| x == 0.0));
        assert!(acc.delta_hu.iter().all(|&x| x == 0.0));
        assert!(acc.delta_hv.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_flux_accumulator_reset() {
        let mut acc = FluxAccumulator::new(5);
        acc.delta_h[0] = 1.0;
        acc.delta_hu[1] = 2.0;
        acc.delta_hv[2] = 3.0;

        acc.reset();

        assert!(acc.delta_h.iter().all(|&x| x == 0.0));
        assert!(acc.delta_hu.iter().all(|&x| x == 0.0));
        assert!(acc.delta_hv.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_flux_accumulator_bed_source() {
        let mut acc = FluxAccumulator::new(3);
        acc.accumulate_bed_source(0, 1.0, 2.0);
        acc.accumulate_bed_source(0, 0.5, 0.5);

        assert!((acc.bed_source_x[0] - 1.5).abs() < 1e-10);
        assert!((acc.bed_source_y[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_flux_accumulator_apply_to_state() {
        let mut acc = FluxAccumulator::new(2);
        acc.delta_h[0] = 10.0;
        acc.delta_h[1] = -10.0;
        acc.delta_hu[0] = 5.0;
        acc.delta_hv[1] = 3.0;
        acc.bed_source_x[0] = 1.0;

        let mut h = vec![1.0, 1.0];
        let mut hu = vec![0.0, 0.0];
        let mut hv = vec![0.0, 0.0];
        let areas = vec![1.0, 2.0];
        let dt = 0.1;

        acc.apply_to_state(&mut h, &mut hu, &mut hv, &areas, dt);

        // h[0] += 0.1 * 1.0 * 10.0 = 2.0
        assert!((h[0] - 2.0).abs() < 1e-10);
        // h[1] += 0.1 * 0.5 * (-10.0) = 0.5
        assert!((h[1] - 0.5).abs() < 1e-10);
        // hu[0] += 0.1 * 1.0 * (5.0 + 1.0) = 0.6
        assert!((hu[0] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_atomic_accumulator_new() {
        let acc = AtomicFluxAccumulator::new(10);
        assert_eq!(acc.n_cells(), 10);
    }

    #[test]
    fn test_atomic_accumulator_accumulate() {
        let acc = AtomicFluxAccumulator::new(3);
        acc.accumulate(0, 1.0, 2.0, 3.0);
        acc.accumulate(0, 0.5, 0.5, 0.5);

        let (h, hu, hv) = acc.collect();
        assert!((h[0] - 1.5).abs() < 1e-10);
        assert!((hu[0] - 2.5).abs() < 1e-10);
        assert!((hv[0] - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_atomic_accumulator_reset() {
        let acc = AtomicFluxAccumulator::new(3);
        acc.accumulate(0, 1.0, 2.0, 3.0);
        acc.reset();

        let (h, hu, hv) = acc.collect();
        assert!(h.iter().all(|&x| x == 0.0));
        assert!(hu.iter().all(|&x| x == 0.0));
        assert!(hv.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_atomic_accumulator_flux() {
        let acc = AtomicFluxAccumulator::new(3);
        
        // 模拟面通量：owner=0, neighbor=1
        acc.accumulate_flux(0, Some(1), 1.0, 2.0, 3.0);

        let (h, hu, hv) = acc.collect();
        // owner得到负通量
        assert!((h[0] - (-1.0)).abs() < 1e-10);
        assert!((hu[0] - (-2.0)).abs() < 1e-10);
        assert!((hv[0] - (-3.0)).abs() < 1e-10);
        // neighbor得到正通量
        assert!((h[1] - 1.0).abs() < 1e-10);
        assert!((hu[1] - 2.0).abs() < 1e-10);
        assert!((hv[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_atomic_accumulator_boundary_flux() {
        let acc = AtomicFluxAccumulator::new(3);
        
        // 边界面：owner=0, neighbor=None
        acc.accumulate_flux(0, None, 1.0, 2.0, 3.0);

        let (h, _hu, _hv) = acc.collect();
        // 只有owner得到通量
        assert!((h[0] - (-1.0)).abs() < 1e-10);
        assert!((h[1] - 0.0).abs() < 1e-10);
        assert!((h[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_flux_accumulator_resize() {
        let mut acc = FluxAccumulator::new(5);
        acc.delta_h[0] = 1.0;

        acc.resize(10);

        assert_eq!(acc.n_cells(), 10);
        assert_eq!(acc.delta_h.len(), 10);
        // 原有数据保留
        assert!((acc.delta_h[0] - 1.0).abs() < 1e-10);
    }
}
