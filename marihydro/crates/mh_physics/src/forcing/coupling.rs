// crates/mh_physics/src/forcing/coupling.rs
//! 强迫场与网格耦合与时空插值

use crate::adapter::PhysicsMesh;
use crate::types::CellIndex;
use mh_runtime::{CpuBackend, Vector2D};

use super::data::ForcingField;

/// 空间插值方法
#[derive(Debug, Clone, Copy)]
pub enum SpatialInterpolation {
    /// 最近邻
    NearestNeighbor,
    /// 反距离加权
    InverseDistanceWeighting { power: f64 },
}

impl Default for SpatialInterpolation {
    fn default() -> Self {
        Self::InverseDistanceWeighting { power: 2.0 }
    }
}

/// 插值权重（预计算）
#[derive(Debug, Clone, Default)]
pub struct InterpolationWeights {
    /// 每个单元的源点索引
    pub source_indices: Vec<Vec<usize>>,
    /// 对应的权重
    pub weights: Vec<Vec<f64>>,
}

impl InterpolationWeights {
    pub fn apply(&self, source_values: &[f64]) -> Vec<f64> {
        let n_cells = self.source_indices.len();
        let mut result = vec![0.0; n_cells];

        for cell_idx in 0..n_cells {
            let indices = &self.source_indices[cell_idx];
            let weights = &self.weights[cell_idx];
            let mut value = 0.0;
            for (&idx, &w) in indices.iter().zip(weights.iter()) {
                if idx >= source_values.len() {
                    continue;
                }
                value += source_values[idx] * w;
            }
            result[cell_idx] = value;
        }

        result
    }
}

/// 强迫-网格耦合器
pub struct ForcingMeshCoupler<'m> {
    /// 目标网格
    mesh: &'m PhysicsMesh,
    /// 插值方法
    method: SpatialInterpolation,
    /// 预计算的插值权重
    weights: InterpolationWeights,
}

impl<'m> ForcingMeshCoupler<'m> {
    pub fn new(mesh: &'m PhysicsMesh, method: SpatialInterpolation) -> Self {
        Self {
            mesh,
            method,
            weights: InterpolationWeights::default(),
        }
    }

    pub fn weights(&self) -> &InterpolationWeights {
        &self.weights
    }

    pub fn into_weights(self) -> InterpolationWeights {
        self.weights
    }

    /// 预计算插值权重（针对给定源点）
    pub fn precompute_weights(&mut self, source_positions: &[(f64, f64)]) {
        if source_positions.is_empty() {
            self.weights = InterpolationWeights::default();
            return;
        }
        let n_cells = self.mesh.cell_count();
        let mut source_indices = Vec::with_capacity(n_cells);
        let mut weights = Vec::with_capacity(n_cells);

        for cell_idx in 0..n_cells {
            let center = self
                .mesh
                .cell_center_generic::<CpuBackend<f64>>(CellIndex::new(cell_idx))
                .expect("cell_center out of range");
            let centroid = (center.x(), center.y());
            let (indices, w) = self.compute_weights_for_point(centroid, source_positions);
            source_indices.push(indices);
            weights.push(w);
        }

        self.weights = InterpolationWeights { source_indices, weights };
    }

    fn compute_weights_for_point(
        &self,
        point: (f64, f64),
        sources: &[(f64, f64)],
    ) -> (Vec<usize>, Vec<f64>) {
        match self.method {
            SpatialInterpolation::NearestNeighbor => {
                let (idx, _) = self.find_nearest(point, sources);
                (vec![idx], vec![1.0])
            }
            SpatialInterpolation::InverseDistanceWeighting { power } => {
                self.compute_idw_weights(point, sources, power)
            }
        }
    }

    fn find_nearest(&self, point: (f64, f64), sources: &[(f64, f64)]) -> (usize, f64) {
        let mut min_dist = f64::MAX;
        let mut min_idx = 0;

        for (i, &src) in sources.iter().enumerate() {
            let dx = point.0 - src.0;
            let dy = point.1 - src.1;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        (min_idx, min_dist)
    }

    fn compute_idw_weights(
        &self,
        point: (f64, f64),
        sources: &[(f64, f64)],
        power: f64,
    ) -> (Vec<usize>, Vec<f64>) {
        const MIN_DIST: f64 = 1e-10;
        const MAX_SOURCES: usize = 8;

        if sources.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let p = if power <= 0.0 { 1.0 } else { power };

        let mut dists: Vec<(usize, f64)> = sources
            .iter()
            .enumerate()
            .map(|(i, &src)| {
                let dx = point.0 - src.0;
                let dy = point.1 - src.1;
                (i, (dx * dx + dy * dy).sqrt().max(MIN_DIST))
            })
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let n = dists.len().min(MAX_SOURCES);

        let indices: Vec<usize> = dists[..n].iter().map(|&(i, _)| i).collect();
        let inv_dists: Vec<f64> = dists[..n]
            .iter()
            .map(|&(_, d)| 1.0 / d.powf(p))
            .collect();
        let sum: f64 = inv_dists.iter().sum();
        let weights: Vec<f64> = if sum > 0.0 {
            inv_dists.iter().map(|w| w / sum).collect()
        } else {
            let uniform = 1.0 / n as f64;
            vec![uniform; n]
        };

        (indices, weights)
    }

    /// 将源场值映射到网格
    pub fn interpolate_to_mesh(&self, source_values: &[f64]) -> Vec<f64> {
        let n_cells = self.mesh.cell_count();
        let mut result = vec![0.0; n_cells];

        for cell_idx in 0..n_cells {
            let indices = &self.weights.source_indices[cell_idx];
            let weights = &self.weights.weights[cell_idx];

            let mut value = 0.0;
            for (&idx, &w) in indices.iter().zip(weights.iter()) {
                if idx >= source_values.len() {
                    continue;
                }
                value += source_values[idx] * w;
            }
            result[cell_idx] = value;
        }

        result
    }
}

pub fn compute_interpolation_weights(
    mesh: &PhysicsMesh,
    method: SpatialInterpolation,
    source_positions: &[(f64, f64)],
) -> InterpolationWeights {
    let mut coupler = ForcingMeshCoupler::new(mesh, method);
    coupler.precompute_weights(source_positions);
    coupler.into_weights()
}

/// 时空强迫数据管理器
pub struct SpatioTemporalForcing {
    /// 空间数据（多时刻）
    fields: Vec<ForcingField>,
    /// 时间戳
    times: Vec<f64>,
}

impl SpatioTemporalForcing {
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
            times: Vec::new(),
        }
    }

    /// 添加时刻数据
    pub fn add_field(&mut self, field: ForcingField) {
        let time = field.time;
        if !time.is_finite() {
            return;
        }
        match self.times.binary_search_by(|t| t.partial_cmp(&time).unwrap()) {
            Ok(pos) => {
                self.times[pos] = time;
                self.fields[pos] = field;
            }
            Err(pos) => {
                self.times.insert(pos, time);
                self.fields.insert(pos, field);
            }
        }
    }

    /// 时间插值获取场值（双线性 + 线性时间插值）
    pub fn interpolate_temporal(&self, time: f64, lon: f64, lat: f64) -> Option<f64> {
        if self.times.is_empty() {
            return None;
        }

        let (t0_idx, t1_idx, frac) = self.find_time_bracket(time)?;

        let v0 = self.fields[t0_idx].interpolate_bilinear(lon, lat)?;
        let v1 = self.fields[t1_idx].interpolate_bilinear(lon, lat)?;

        Some(v0 * (1.0 - frac) + v1 * frac)
    }

    fn find_time_bracket(&self, time: f64) -> Option<(usize, usize, f64)> {
        if !time.is_finite() {
            return None;
        }
        if self.times.iter().any(|t| !t.is_finite()) {
            return None;
        }
        if self.times.len() == 1 {
            return Some((0, 0, 0.0));
        }

        let first = *self.times.first()?;
        let last = *self.times.last()?;
        if time <= first {
            return Some((0, 0, 0.0));
        }
        if time >= last {
            let idx = self.times.len() - 1;
            return Some((idx, idx, 0.0));
        }

        for i in 0..self.times.len() - 1 {
            if self.times[i] <= time && time <= self.times[i + 1] {
                let denom = self.times[i + 1] - self.times[i];
                let frac = if denom.abs() < 1e-14 { 0.0 } else { (time - self.times[i]) / denom };
                return Some((i, i + 1, frac));
            }
        }

        None
    }
}

impl Default for SpatioTemporalForcing {
    fn default() -> Self {
        Self::new()
    }
}
