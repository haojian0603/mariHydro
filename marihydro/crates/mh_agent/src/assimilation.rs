// crates/mh_agent/src/assimilation.rs

use crate::{AiError, AIAgent, Assimilable, PhysicsSnapshot};
use std::sync::Mutex;

/// Nudging同化配置
#[derive(Debug, Clone)]
pub struct NudgingConfig {
    /// 同化率 (0.0 - 1.0)
    pub rate: f64,
    /// 最大修正量限制
    pub max_correction: f64,
    /// 空间平滑半径
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

/// 观测数据结构
#[derive(Debug, Clone)]
pub struct Observation {
    /// 观测值
    pub values: Vec<f64>,
    /// 观测位置索引
    pub cell_indices: Vec<usize>,
    /// 观测不确定性
    pub uncertainty: Vec<f64>,
    /// 观测时间
    pub time: f64,
}

impl Observation {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    fn validate(&self) -> Result<(), AiError> {
        if self.values.len() != self.cell_indices.len()
            || self.values.len() != self.uncertainty.len()
        {
            return Err(AiError::InvalidObservation(
                "观测数据长度不一致".to_string(),
            ));
        }
        Ok(())
    }
}

struct NudgingState {
    last_assimilation_time: f64,
    cumulative_correction: f64,
    pending_observation: Option<Observation>,
    cell_centers: Option<Vec<[f64; 2]>>,
    last_snapshot_time: f64,
}

/// Nudging同化器
pub struct NudgingAssimilator {
    config: NudgingConfig,
    state: Mutex<NudgingState>,
}

impl NudgingAssimilator {
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
        }
    }

    /// 设置当前可用的观测
    pub fn set_observation(&mut self, observation: Observation) -> Result<(), AiError> {
        observation.validate()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| AiError::StateAccessError("获取同化状态锁失败".into()))?;
        state.pending_observation = Some(observation);
        Ok(())
    }

    /// 执行Nudging同化
    pub fn assimilate(
        &mut self,
        state: &mut dyn Assimilable,
        observation: &Observation,
        current_time: f64,
    ) -> Result<AssimilationResult, AiError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| AiError::StateAccessError("获取同化状态锁失败".into()))?;
        self.assimilate_internal(&mut guard, state, observation, current_time)
    }

    /// 计算单点修正量
    fn compute_correction(&self, simulated: f64, observed: f64, uncertainty: f64) -> f64 {
        let mismatch = observed - simulated;
        let weight = 1.0 / (1.0 + uncertainty.abs());
        let raw = mismatch * self.config.rate * weight;
        raw.clamp(-self.config.max_correction, self.config.max_correction)
    }

    /// 应用空间平滑
    fn apply_smoothing(&self, corrections: &mut [f64], cell_centers: &[[f64; 2]]) {
        let radius = match self.config.smoothing_radius {
            Some(r) if r > 0.0 => r,
            _ => return,
        };
        let n = corrections.len();
        if n == 0 || cell_centers.len() != n {
            return;
        }

        let mut smoothed = vec![0.0; n];
        let radius_sq = radius * radius;

        for i in 0..n {
            if corrections[i].abs() < f64::EPSILON {
                continue;
            }
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;
            for j in 0..n {
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

        for (dst, src) in corrections.iter_mut().zip(smoothed.iter()) {
            *dst = *src;
        }
    }
}

/// 同化结果
#[derive(Debug, Clone)]
pub struct AssimilationResult {
    pub cells_modified: usize,
    pub total_correction: f64,
    pub max_correction: f64,
    pub conservation_error: f64,
}

impl AIAgent for NudgingAssimilator {
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

    fn apply(&self, state: &mut dyn Assimilable) -> Result<(), AiError> {
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
    }
}

impl NudgingAssimilator {
    fn assimilate_internal(
        &self,
        internal: &mut NudgingState,
        state: &mut dyn Assimilable,
        observation: &Observation,
        current_time: f64,
    ) -> Result<AssimilationResult, AiError> {
        observation.validate()?;

        let mut depth = state.get_depth_mut();
        let n_cells = depth.len();
        if n_cells == 0 {
            return Err(AiError::StateAccessError("状态为空".into()));
        }

        let dt = (current_time - internal.last_assimilation_time).max(0.0);
        let temporal_factor = if self.config.temporal_decay > 0.0 {
            (-self.config.temporal_decay * dt).exp()
        } else {
            1.0
        };

        let mut corrections = vec![0.0f64; n_cells];
        let mut max_corr = 0.0;
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
            let simulated = depth[idx];
            let corr = self.compute_correction(simulated, obs_val, uncertainty)
                * temporal_factor;
            if corr.abs() > 0.0 {
                corrections[idx] = corr;
                max_corr = max_corr.max(corr.abs());
                total_corr += corr;
                cells_modified += 1;
            }
        }

        if let (Some(_radius), Some(centers)) = (self.config.smoothing_radius, internal.cell_centers.as_ref()) {
            self.apply_smoothing(&mut corrections, centers);
            max_corr = corrections
                .iter()
                .fold(0.0, |m, &c| if c.abs() > m { c.abs() } else { m });
            total_corr = corrections.iter().sum();
        }

        let before = state.total_water_volume();

        for (cell, corr) in corrections.iter().enumerate() {
            if corr.abs() < f64::EPSILON {
                continue;
            }
            let new_h = (depth[cell] + *corr).max(0.0);
            depth[cell] = new_h;
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
