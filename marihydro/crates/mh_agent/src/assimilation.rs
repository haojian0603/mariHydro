use crate::{
    scalar_from_f64_or_panic, AIAgent, AiError, Assimilable, DefaultBackend, PhysicsSnapshot,
    ScalarSamples,
};
use bytemuck::Pod;
use mh_runtime::prelude::Float;
use mh_runtime::{Backend, CellIndex, RuntimeScalar, Vector2D};
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Mutex;

/// Nudging 同化配置
#[derive(Debug, Clone)]
pub struct NudgingConfig<B: Backend = DefaultBackend> {
    /// 同化率
    pub rate: B::Scalar,
    /// 最大修正量
    pub max_correction: B::Scalar,
    /// 空间平滑半径
    pub smoothing_radius: Option<B::Scalar>,
    /// 时间衰减系数
    pub temporal_decay: B::Scalar,
}

impl<B: Backend> Default for NudgingConfig<B>
where
    B::Scalar: RuntimeScalar,
{
    fn default() -> Self {
        Self {
            rate: scalar_from_f64_or_panic::<B>(0.2, "nudging.default_rate"),
            max_correction: scalar_from_f64_or_panic::<B>(0.2, "nudging.default_max_correction"),
            smoothing_radius: None,
            temporal_decay: B::Scalar::ZERO,
        }
    }
}

/// 观测数据
#[derive(Debug, Clone)]
pub struct Observation<B: Backend = DefaultBackend> {
    /// 观测值
    pub values: ScalarSamples<B>,
    /// 观测对应的网格索引
    pub cell_indices: Vec<CellIndex>,
    /// 观测不确定性
    pub uncertainty: ScalarSamples<B>,
    /// 观测时间戳
    pub time: B::Scalar,
}

impl<B: Backend> Observation<B>
where
    B::Scalar: RuntimeScalar,
{
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn validate(&self) -> Result<(), AiError> {
        if self.values.len() != self.cell_indices.len()
            || self.values.len() != self.uncertainty.len()
        {
            return Err(AiError::InvalidObservation(
                "observation data length mismatch".to_string(),
            ));
        }
        if !self.time.is_finite() {
            return Err(AiError::InvalidObservation(
                "observation time is not finite".into(),
            ));
        }
        for (&v, &u) in self.values.iter().zip(self.uncertainty.iter()) {
            if !v.is_finite() || !u.is_finite() || u < B::Scalar::ZERO {
                return Err(AiError::InvalidObservation(
                    "observation value or uncertainty is invalid".into(),
                ));
            }
        }
        Ok(())
    }

    fn validate_with_bounds(&self, n_cells: usize) -> Result<(), AiError> {
        self.validate()?;
        for idx in &self.cell_indices {
            if idx.get() >= n_cells {
                return Err(AiError::InvalidObservation(format!(
                    "observation cell index is out of bounds: {} >= {}",
                    idx.get(),
                    n_cells
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
struct NeighborGraph {
    entries: Vec<Vec<CellIndex>>,
}

impl NeighborGraph {
    fn new(entries: Vec<Vec<CellIndex>>) -> Self {
        Self { entries }
    }
}

impl Deref for NeighborGraph {
    type Target = [Vec<CellIndex>];

    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}

#[derive(Clone)]
struct NudgingState<B: Backend> {
    last_assimilation_time: B::Scalar,
    cumulative_correction: B::Scalar,
    pending_observation: Option<Observation<B>>,
    cell_centers: Option<Vec<B::Vector2D>>,
    last_snapshot_time: B::Scalar,
    neighbor_list: Option<NeighborGraph>,
    neighbor_radius: Option<B::Scalar>,
}

/// Nudging 同化器
pub struct NudgingAssimilator<B: Backend = DefaultBackend> {
    config: NudgingConfig<B>,
    state: Mutex<NudgingState<B>>,
}

impl<B: Backend> NudgingAssimilator<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: Pod,
{
    pub fn new(config: NudgingConfig<B>) -> Self {
        Self {
            config,
            state: Mutex::new(NudgingState {
                last_assimilation_time: B::Scalar::ZERO,
                cumulative_correction: B::Scalar::ZERO,
                pending_observation: None,
                cell_centers: None,
                last_snapshot_time: B::Scalar::ZERO,
                neighbor_list: None,
                neighbor_radius: None,
            }),
        }
    }

    /// 璁剧疆瑙傛祴
    pub fn set_observation(&mut self, observation: Observation<B>) -> Result<(), AiError> {
        observation.validate()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| AiError::StateAccessError("nudging state lock poisoned".into()))?;
        state.pending_observation = Some(observation);
        Ok(())
    }

    /// 执行单次同化
    pub fn assimilate(
        &mut self,
        state: &mut dyn Assimilable<B>,
        observation: &Observation<B>,
        current_time: B::Scalar,
    ) -> Result<AssimilationResult<B>, AiError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| AiError::StateAccessError("nudging state lock poisoned".into()))?;
        self.assimilate_internal(&mut guard, state, observation, current_time)
    }

    /// 计算单个观测点的修正量
    fn compute_correction(
        &self,
        simulated: B::Scalar,
        observed: B::Scalar,
        uncertainty: B::Scalar,
    ) -> B::Scalar {
        let mismatch = observed - simulated;
        let weight = B::Scalar::ONE / (B::Scalar::ONE + uncertainty.abs());
        let raw = mismatch * self.config.rate * weight;
        raw.clamp_value(-self.config.max_correction, self.config.max_correction)
    }

    /// 绌洪棿骞虫粦
    fn apply_smoothing(&self, corrections: &mut [B::Scalar], cell_centers: &[B::Vector2D]) {
        let radius = match self.config.smoothing_radius {
            Some(r) if r > B::Scalar::ZERO => r,
            _ => return,
        };
        let n = corrections.len();
        if n == 0 || cell_centers.len() != n {
            return;
        }

        let mut smoothed = vec![B::Scalar::ZERO; n];
        let radius_sq = radius * radius;
        let tiny = scalar_from_f64_or_panic::<B>(1e-6, "nudging.smoothing_tiny");

        for i in 0..n {
            let mut weighted_sum = B::Scalar::ZERO;
            let mut weight_total = B::Scalar::ZERO;
            for j in 0..n {
                let dx = cell_centers[i].x() - cell_centers[j].x();
                let dy = cell_centers[i].y() - cell_centers[j].y();
                let dist_sq = dx * dx + dy * dy;
                if dist_sq <= radius_sq {
                    let w = B::Scalar::ONE / (dist_sq.safe_sqrt() + tiny);
                    weighted_sum += w * corrections[j];
                    weight_total += w;
                }
            }
            if weight_total > B::Scalar::ZERO {
                smoothed[i] = weighted_sum / weight_total;
            }
        }

        corrections.copy_from_slice(&smoothed);
    }

    fn apply_smoothing_with_neighbors(
        &self,
        corrections: &mut [B::Scalar],
        cell_centers: &[B::Vector2D],
        neighbors: &NeighborGraph,
    ) {
        let n = corrections.len();
        if n == 0 || cell_centers.len() != n || neighbors.len() != n {
            return;
        }

        let mut smoothed = vec![B::Scalar::ZERO; n];
        let tiny = scalar_from_f64_or_panic::<B>(1e-6, "nudging.neighbor_smoothing_tiny");
        for (i, nbrs) in neighbors.iter().enumerate() {
            let mut weighted_sum = B::Scalar::ZERO;
            let mut weight_total = B::Scalar::ZERO;
            for &neighbor in nbrs {
                let j = neighbor.get();
                let dx = cell_centers[i].x() - cell_centers[j].x();
                let dy = cell_centers[i].y() - cell_centers[j].y();
                let w = B::Scalar::ONE / (dx.hypot(dy) + tiny);
                weighted_sum += w * corrections[j];
                weight_total += w;
            }
            if weight_total > B::Scalar::ZERO {
                smoothed[i] = weighted_sum / weight_total;
            }
        }

        corrections.copy_from_slice(&smoothed);
    }
}

fn build_neighbors<B: Backend>(centers: &[B::Vector2D], radius: B::Scalar) -> NeighborGraph
where
    B::Scalar: RuntimeScalar,
{
    let n = centers.len();
    if n == 0 || radius <= B::Scalar::ZERO {
        return NeighborGraph::new(vec![Vec::new(); n]);
    }

    let r2 = radius * radius;
    let cell_size = radius.max(scalar_from_f64_or_panic::<B>(
        1e-6,
        "nudging.grid_cell_size_min",
    ));
    let cell_size_f64 = cell_size.to_f64_lossy();
    let mut grid: HashMap<(i32, i32), Vec<CellIndex>> = HashMap::new();

    for (i, c) in centers.iter().enumerate() {
        let gx = (c.x().to_f64_lossy() / cell_size_f64).floor() as i32;
        let gy = (c.y().to_f64_lossy() / cell_size_f64).floor() as i32;
        grid.entry((gx, gy)).or_default().push(CellIndex::new(i));
    }

    let mut neighbors = vec![Vec::new(); n];
    for (i, c) in centers.iter().enumerate() {
        let gx = (c.x().to_f64_lossy() / cell_size_f64).floor() as i32;
        let gy = (c.y().to_f64_lossy() / cell_size_f64).floor() as i32;

        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(bucket) = grid.get(&(gx + dx, gy + dy)) {
                    for &j in bucket {
                        let j_idx = j.get();
                        let ddx = c.x() - centers[j_idx].x();
                        let ddy = c.y() - centers[j_idx].y();
                        if ddx * ddx + ddy * ddy <= r2 {
                            neighbors[i].push(j);
                        }
                    }
                }
            }
        }
    }

    NeighborGraph::new(neighbors)
}

/// 鍚屽寲缁撴灉
#[derive(Debug, Clone)]
pub struct AssimilationResult<B: Backend = DefaultBackend> {
    pub cells_modified: usize,
    pub total_correction: B::Scalar,
    pub max_correction: B::Scalar,
    pub conservation_error: B::Scalar,
}

impl<B: Backend> AIAgent<B> for NudgingAssimilator<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: Pod,
{
    fn name(&self) -> &'static str {
        "Nudging-Assimilator"
    }

    fn update(&mut self, snapshot: &PhysicsSnapshot<B>) -> Result<(), AiError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| AiError::StateAccessError("nudging state lock poisoned".into()))?;
        guard.last_snapshot_time = snapshot.time;
        guard.cell_centers = Some(snapshot.cell_centers.to_vec());
        if let Some(radius) = self.config.smoothing_radius {
            let rebuild = guard.neighbor_list.is_none()
                || guard.neighbor_radius.is_none_or(|r| {
                    (r - radius).abs()
                        > scalar_from_f64_or_panic::<B>(1e-12, "nudging.radius_change_epsilon")
                })
                || guard.cell_centers.as_ref().map(|c| c.len())
                    != guard.neighbor_list.as_ref().map(|n| n.len());
            if rebuild {
                guard.neighbor_list = Some(build_neighbors::<B>(&snapshot.cell_centers, radius));
                guard.neighbor_radius = Some(radius);
            }
        }
        Ok(())
    }

    fn apply(&self, state: &mut dyn Assimilable<B>) -> Result<(), AiError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| AiError::StateAccessError("nudging state lock poisoned".into()))?;
        let observation = guard
            .pending_observation
            .as_ref()
            .ok_or_else(|| {
                AiError::InvalidObservation("pending observation is unavailable".into())
            })?
            .clone();
        let current_time = if observation.time.is_finite() {
            observation.time
        } else {
            guard.last_snapshot_time
        };
        self.assimilate_internal(&mut guard, state, &observation, current_time)
            .map(|_| ())
    }
}

impl<B: Backend> NudgingAssimilator<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: Pod,
{
    fn assimilate_internal(
        &self,
        internal: &mut NudgingState<B>,
        state: &mut dyn Assimilable<B>,
        observation: &Observation<B>,
        current_time: B::Scalar,
    ) -> Result<AssimilationResult<B>, AiError> {
        observation.validate_with_bounds(state.n_cells())?;

        let before = state.total_water_volume();
        let depth_snapshot = state.get_depth().to_vec();
        let n_cells = state.get_depth().len();
        if n_cells == 0 {
            return Err(AiError::StateAccessError(
                "nudging state lock poisoned".into(),
            ));
        }

        let dt = (current_time - internal.last_assimilation_time).max(B::Scalar::ZERO);
        let temporal_factor = if self.config.temporal_decay > B::Scalar::ZERO {
            (-self.config.temporal_decay * dt).exp()
        } else {
            B::Scalar::ONE
        };

        let mut corrections = vec![B::Scalar::ZERO; n_cells];
        let mut max_corr = B::Scalar::ZERO;
        let mut total_corr = B::Scalar::ZERO;
        let mut cells_modified = 0usize;

        for ((idx, &obs_val), &uncertainty) in observation
            .cell_indices
            .iter()
            .zip(observation.values.iter())
            .zip(observation.uncertainty.iter())
        {
            let i = idx.get();
            let simulated = depth_snapshot[i];
            let corr = self.compute_correction(simulated, obs_val, uncertainty) * temporal_factor;
            if corr.abs() > B::Scalar::ZERO {
                corrections[i] = corr;
                max_corr = max_corr.max_value(corr.abs());
                total_corr += corr;
                cells_modified += 1;
            }
        }

        if let (Some(_radius), Some(centers)) =
            (self.config.smoothing_radius, internal.cell_centers.as_ref())
        {
            if let Some(neighbors) = internal.neighbor_list.as_ref() {
                self.apply_smoothing_with_neighbors(&mut corrections, centers, neighbors);
            } else {
                self.apply_smoothing(&mut corrections, centers);
            }
            max_corr =
                corrections.iter().fold(
                    B::Scalar::ZERO,
                    |m, &c| if c.abs() > m { c.abs() } else { m },
                );
            total_corr = corrections.iter().copied().sum();
        }

        {
            let depth = state.get_depth_mut();
            for (cell, corr) in corrections.iter().enumerate() {
                if corr.abs() <= B::Scalar::EPSILON {
                    continue;
                }
                depth[cell] = (depth[cell] + *corr).max(B::Scalar::ZERO);
            }
        }

        let after = state.total_water_volume();
        let conservation_error = after - before;

        internal.last_assimilation_time = current_time;
        internal.cumulative_correction += total_corr;
        internal.pending_observation = None;

        Ok(AssimilationResult {
            cells_modified,
            total_correction: total_corr,
            max_correction: max_corr,
            conservation_error,
        })
    }
}
