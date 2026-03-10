use crate::{AIAgent, AiError, Assimilable, PhysicsSnapshot};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Mutex;

/// Nudging 同化配置
#[derive(Debug, Clone)]
pub struct NudgingConfig {
    /// 同化率 (0.0~1.0)
    pub rate: f64,
    /// 单次最大修正幅度
    pub max_correction: f64,
    /// 空间平滑半径（None 表示不平滑）
    pub smoothing_radius: Option<f64>,
    /// 时间衰减系数
    pub temporal_decay: f64,
}

impl Default for NudgingConfig {
    fn default() -> Self {
        Self {
            rate: 0.2,
            max_correction: 0.2,
            smoothing_radius: None,
            temporal_decay: 0.0,
        }
    }
}

/// 观测数据
#[derive(Debug, Clone)]
pub struct Observation {
    pub values: Vec<f64>,
    pub cell_indices: Vec<usize>,
    pub uncertainty: Vec<f64>,
    pub time: f64,
}

impl Observation {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    fn validate(&self) -> Result<(), AiError> {
        if self.values.len() != self.cell_indices.len() || self.values.len() != self.uncertainty.len() {
            return Err(AiError::InvalidObservation("观测数组长度不一致".into()));
        }
        if self.uncertainty.iter().any(|u| !u.is_finite() || *u < 0.0) {
            return Err(AiError::InvalidObservation("观测不确定性必须为有限非负值".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct SmoothingGrid {
    cell_size: f64,
    bins: HashMap<(i64, i64), Vec<usize>>,
}

impl SmoothingGrid {
    fn build(centers: &[[f64; 2]], radius: f64) -> Self {
        let cell_size = radius.max(f64::EPSILON);
        let mut bins: HashMap<(i64, i64), Vec<usize>> = HashMap::new();

        for (idx, center) in centers.iter().enumerate() {
            let key = (
                (center[0] / cell_size).floor() as i64,
                (center[1] / cell_size).floor() as i64,
            );
            bins.entry(key).or_default().push(idx);
        }

        Self { cell_size, bins }
    }

    fn query_candidates(&self, center: [f64; 2], out: &mut Vec<usize>) {
        out.clear();
        let gx = (center[0] / self.cell_size).floor() as i64;
        let gy = (center[1] / self.cell_size).floor() as i64;

        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(indices) = self.bins.get(&(gx + dx, gy + dy)) {
                    out.extend(indices.iter().copied());
                }
            }
        }
    }
}

struct NudgingState {
    last_assimilation_time: f64,
    cumulative_correction: f64,
    pending_observation: Option<Observation>,
    cell_centers: Option<Vec<[f64; 2]>>,
    last_snapshot_time: f64,
}

/// 同化结果
#[derive(Debug, Clone)]
pub struct AssimilationResult {
    pub cells_modified: usize,
    pub total_correction: f64,
    pub max_correction: f64,
    pub conservation_error: f64,
}

/// Nudging 同化器
pub struct NudgingAssimilator<S: Assimilable> {
    config: NudgingConfig,
    state: Mutex<NudgingState>,
    _marker: PhantomData<fn() -> S>,
}

impl<S: Assimilable> NudgingAssimilator<S> {
    pub fn new(config: NudgingConfig) -> Self {
        Self {
            config,
            state: Mutex::new(NudgingState {
                last_assimilation_time: 0.0,
                cumulative_correction: 0.0,
                pending_observation: None,
                cell_centers: None,
                last_snapshot_time: 0.0,
            }),
            _marker: PhantomData,
        }
    }

    pub fn set_observation(&mut self, observation: Observation) -> Result<(), AiError> {
        observation.validate()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| AiError::StateAccessError("获取同化状态锁失败".into()))?;
        state.pending_observation = Some(observation);
        Ok(())
    }

    pub fn assimilate(
        &mut self,
        state: &mut S,
        observation: &Observation,
        current_time: f64,
    ) -> Result<AssimilationResult, AiError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| AiError::StateAccessError("获取同化状态锁失败".into()))?;
        self.assimilate_internal(&mut guard, state, observation, current_time)
    }

    fn compute_correction(&self, simulated: f64, observed: f64, uncertainty: f64) -> f64 {
        let mismatch = observed - simulated;
        let weight = 1.0 / (1.0 + uncertainty.abs());
        let raw = mismatch * self.config.rate * weight;
        raw.clamp(-self.config.max_correction, self.config.max_correction)
    }

    fn apply_smoothing(
        &self,
        corrections: &mut [f64],
        cell_centers: &[[f64; 2]],
        radius: f64,
    ) {
        if corrections.is_empty() || corrections.len() != cell_centers.len() {
            return;
        }

        let radius_sq = radius * radius;
        let grid = SmoothingGrid::build(cell_centers, radius);
        let mut smoothed = vec![0.0; corrections.len()];
        let mut candidates = Vec::new();

        for i in 0..corrections.len() {
            grid.query_candidates(cell_centers[i], &mut candidates);

            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;

            for &j in &candidates {
                let dx = cell_centers[i][0] - cell_centers[j][0];
                let dy = cell_centers[i][1] - cell_centers[j][1];
                let dist_sq = dx * dx + dy * dy;
                if dist_sq <= radius_sq {
                    let w = 1.0 / (dist_sq.sqrt() + 1e-6);
                    weighted_sum += w * corrections[j];
                    weight_total += w;
                }
            }

            if weight_total > 0.0 {
                smoothed[i] = weighted_sum / weight_total;
            }
        }

        corrections.copy_from_slice(&smoothed);
    }

    fn assimilate_internal(
        &self,
        internal: &mut NudgingState,
        state: &mut S,
        observation: &Observation,
        current_time: f64,
    ) -> Result<AssimilationResult, AiError> {
        observation.validate()?;

        let n_cells = state.n_cells();
        if n_cells == 0 {
            return Err(AiError::StateAccessError("状态为空".into()));
        }

        let depth_view = state.get_depth();
        if depth_view.len() != n_cells {
            return Err(AiError::StateAccessError("水深数组长度与单元数不一致".into()));
        }

        let dt = (current_time - internal.last_assimilation_time).max(0.0);
        let temporal_factor = if self.config.temporal_decay > 0.0 {
            (-self.config.temporal_decay * dt).exp()
        } else {
            1.0
        };

        let mut corrections = vec![0.0f64; n_cells];
        let mut max_corr: f64 = 0.0;
        let mut total_corr = 0.0;
        let mut cells_modified = 0usize;

        for ((&idx, &obs_val), &uncertainty) in observation
            .cell_indices
            .iter()
            .zip(observation.values.iter())
            .zip(observation.uncertainty.iter())
        {
            if idx >= n_cells {
                return Err(AiError::InvalidObservation(format!(
                    "观测索引超出范围: {idx} >= {n_cells}"
                )));
            }

            let simulated = depth_view[idx];
            let corr = self.compute_correction(simulated, obs_val, uncertainty) * temporal_factor;
            if corr.abs() > 0.0 {
                corrections[idx] = corr;
                max_corr = max_corr.max(corr.abs());
                total_corr += corr;
                cells_modified += 1;
            }
        }

        if let (Some(radius), Some(centers)) = (self.config.smoothing_radius, internal.cell_centers.as_ref()) {
            if radius > 0.0 {
                self.apply_smoothing(&mut corrections, centers, radius);
                max_corr = corrections.iter().fold(0.0, |m, &c| m.max(c.abs()));
                total_corr = corrections.iter().sum();
            }
        }

        let before = state.total_water_volume();

        {
            let depth_mut = state.get_depth_mut();
            for (cell, corr) in corrections.iter().enumerate() {
                if corr.abs() < f64::EPSILON {
                    continue;
                }
                depth_mut[cell] = (depth_mut[cell] + *corr).max(0.0);
            }
        }

        let after = state.total_water_volume();
        let conservation_error = after - before;

        internal.last_assimilation_time = current_time;
        internal.cumulative_correction += total_corr;

        Ok(AssimilationResult {
            cells_modified,
            total_correction: total_corr,
            max_correction: max_corr,
            conservation_error,
        })
    }
}

impl<S: Assimilable> AIAgent for NudgingAssimilator<S> {
    type State = S;

    fn name(&self) -> &'static str {
        "Nudging-Assimilator"
    }

    fn update(&mut self, snapshot: &PhysicsSnapshot) -> Result<(), AiError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| AiError::StateAccessError("获取同化状态锁失败".into()))?;
        guard.last_snapshot_time = snapshot.time;
        guard.cell_centers = Some(snapshot.cell_centers.clone());
        Ok(())
    }

    fn apply(&self, state: &mut Self::State) -> Result<(), AiError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| AiError::StateAccessError("获取同化状态锁失败".into()))?;

        let observation = guard
            .pending_observation
            .as_ref()
            .ok_or_else(|| AiError::InvalidObservation("缺少观测数据".into()))?
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
