// src-tauri/src/marihydro/physics/engine/flux_accumulator.rs
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, FaceIndex};
use crate::marihydro::physics::schemes::InterfaceFlux;

pub struct FluxAccumulator {
    n_cells: usize,
    pub delta_h: Vec<f64>,
    pub delta_hu: Vec<f64>,
    pub delta_hv: Vec<f64>,
    pub bed_source_x: Vec<f64>,
    pub bed_source_y: Vec<f64>,
}

impl FluxAccumulator {
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

    pub fn reset(&mut self) {
        self.delta_h.fill(0.0);
        self.delta_hu.fill(0.0);
        self.delta_hv.fill(0.0);
        self.bed_source_x.fill(0.0);
        self.bed_source_y.fill(0.0);
    }

    #[inline]
    pub fn accumulate_flux<M: MeshAccess>(&mut self, face: FaceIndex, flux: &InterfaceFlux, length: f64, mesh: &M) {
        let owner = mesh.face_owner(face);
        let neighbor = mesh.face_neighbor(face);
        let flux_h = flux.mass * length;
        let flux_hu = flux.momentum_x * length;
        let flux_hv = flux.momentum_y * length;
        self.delta_h[owner.0] -= flux_h;
        self.delta_hu[owner.0] -= flux_hu;
        self.delta_hv[owner.0] -= flux_hv;
        if neighbor.is_valid() {
            self.delta_h[neighbor.0] += flux_h;
            self.delta_hu[neighbor.0] += flux_hu;
            self.delta_hv[neighbor.0] += flux_hv;
        }
    }

    #[inline]
    pub fn accumulate_bed_source(&mut self, cell: CellIndex, source_x: f64, source_y: f64) {
        self.bed_source_x[cell.0] += source_x;
        self.bed_source_y[cell.0] += source_y;
    }

    pub fn apply_to_state(&self, h: &mut [f64], hu: &mut [f64], hv: &mut [f64], areas: &[f64], dt: f64) {
        for i in 0..self.n_cells {
            let inv_area = 1.0 / areas[i];
            h[i] += dt * inv_area * self.delta_h[i];
            hu[i] += dt * inv_area * (self.delta_hu[i] + self.bed_source_x[i]);
            hv[i] += dt * inv_area * (self.delta_hv[i] + self.bed_source_y[i]);
        }
    }
}

pub struct AtomicFluxAccumulator {
    n_cells: usize,
    delta_h: Vec<std::sync::atomic::AtomicU64>,
    delta_hu: Vec<std::sync::atomic::AtomicU64>,
    delta_hv: Vec<std::sync::atomic::AtomicU64>,
}

impl AtomicFluxAccumulator {
    pub fn new(n_cells: usize) -> Self {
        Self {
            n_cells,
            delta_h: (0..n_cells).map(|_| std::sync::atomic::AtomicU64::new(0)).collect(),
            delta_hu: (0..n_cells).map(|_| std::sync::atomic::AtomicU64::new(0)).collect(),
            delta_hv: (0..n_cells).map(|_| std::sync::atomic::AtomicU64::new(0)).collect(),
        }
    }

    pub fn reset(&self) {
        for i in 0..self.n_cells {
            self.delta_h[i].store(0, std::sync::atomic::Ordering::Relaxed);
            self.delta_hu[i].store(0, std::sync::atomic::Ordering::Relaxed);
            self.delta_hv[i].store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[inline]
    fn atomic_add(atomic: &std::sync::atomic::AtomicU64, val: f64) {
        use std::sync::atomic::Ordering;
        let mut old = atomic.load(Ordering::Relaxed);
        loop {
            let old_f = f64::from_bits(old);
            let new_f = old_f + val;
            match atomic.compare_exchange_weak(old, new_f.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(x) => old = x,
            }
        }
    }

    pub fn accumulate(&self, cell_idx: usize, dh: f64, dhu: f64, dhv: f64) {
        Self::atomic_add(&self.delta_h[cell_idx], dh);
        Self::atomic_add(&self.delta_hu[cell_idx], dhu);
        Self::atomic_add(&self.delta_hv[cell_idx], dhv);
    }

    pub fn collect(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let h: Vec<f64> = self.delta_h.iter().map(|a| f64::from_bits(a.load(std::sync::atomic::Ordering::Relaxed))).collect();
        let hu: Vec<f64> = self.delta_hu.iter().map(|a| f64::from_bits(a.load(std::sync::atomic::Ordering::Relaxed))).collect();
        let hv: Vec<f64> = self.delta_hv.iter().map(|a| f64::from_bits(a.load(std::sync::atomic::Ordering::Relaxed))).collect();
        (h, hu, hv)
    }
}
