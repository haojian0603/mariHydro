use crate::{
    scalar_from_f64_or_panic, AIAgent, AiError, Assimilable, DefaultBackend, PhysicsSnapshot,
    ScalarSamples,
};
use mh_runtime::prelude::Float;
use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct SurrogateConfig<B: Backend = DefaultBackend> {
    pub model_path: Option<String>,
    pub input_features: Vec<String>,
    pub output_features: Vec<String>,
    pub prediction_horizon: B::Scalar,
    pub estimate_uncertainty: bool,
    pub assimilation_rate: B::Scalar,
    pub learning_rate: B::Scalar,
    pub l2_reg: B::Scalar,
    pub min_std: B::Scalar,
}

#[derive(Debug, Clone)]
pub struct SurrogatePrediction<B: Backend = DefaultBackend> {
    pub values: ScalarSamples<B>,
    pub uncertainty: Option<ScalarSamples<B>>,
    pub prediction_time: B::Scalar,
    pub confidence: B::Scalar,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub count: usize,
    pub m2: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearRegressionCore {
    input_dim: usize,
    output_dim: usize,
    weights: Vec<f64>,
    bias: Vec<f64>,
}

impl LinearRegressionCore {
    fn new(output_dim: usize) -> Self {
        let output_dim = output_dim.max(1);
        Self {
            input_dim: 0,
            output_dim,
            weights: Vec::new(),
            bias: vec![0.0; output_dim],
        }
    }

    fn ensure_shape(&mut self, input_dim: usize, output_dim: usize) {
        let input_dim = input_dim.max(1);
        let output_dim = output_dim.max(1);
        let expected_weights = input_dim * output_dim;
        if self.input_dim == input_dim
            && self.output_dim == output_dim
            && self.weights.len() == expected_weights
            && self.bias.len() == output_dim
        {
            return;
        }
        self.input_dim = input_dim;
        self.output_dim = output_dim;
        self.weights = vec![0.0; expected_weights];
        self.bias = vec![0.0; output_dim];
    }

    fn validate(&self) -> Result<(), AiError> {
        if self.output_dim == 0 {
            return Err(AiError::InvalidShape {
                expected: vec![1],
                actual: vec![0],
            });
        }
        let expected_weights = self
            .input_dim
            .checked_mul(self.output_dim)
            .ok_or_else(|| AiError::InferenceFailed("surrogate weight shape overflowed".into()))?;
        if self.weights.len() != expected_weights {
            return Err(AiError::InvalidShape {
                expected: vec![expected_weights],
                actual: vec![self.weights.len()],
            });
        }
        if self.bias.len() != self.output_dim {
            return Err(AiError::InvalidShape {
                expected: vec![self.output_dim],
                actual: vec![self.bias.len()],
            });
        }
        Ok(())
    }

    fn predict(&self, features: &[f64]) -> Result<Vec<f64>, AiError> {
        self.validate()?;
        if self.input_dim == 0 || self.weights.is_empty() {
            return Err(AiError::NotReady(
                "surrogate linear core has no trained weights".into(),
            ));
        }
        if features.len() != self.input_dim {
            return Err(AiError::InvalidShape {
                expected: vec![self.input_dim],
                actual: vec![features.len()],
            });
        }
        let mut out = vec![0.0; self.output_dim];
        for (o, out_item) in out.iter_mut().enumerate() {
            let row_start = o * self.input_dim;
            let row_end = row_start + self.input_dim;
            let weighted_sum = self.weights[row_start..row_end]
                .iter()
                .zip(features.iter())
                .map(|(weight, feature)| weight * feature)
                .sum::<f64>();
            *out_item = self.bias[o] + weighted_sum;
        }
        Ok(out)
    }
}

pub struct SurrogateModel<B: Backend = DefaultBackend> {
    config: SurrogateConfig<B>,
    current_prediction: Option<SurrogatePrediction<B>>,
    input_normalization: Option<NormalizationParams>,
    output_normalization: Option<NormalizationParams>,
    last_update_time: B::Scalar,
    linear_core: LinearRegressionCore,
    error_ema: Option<f64>,
    trained: bool,
}

impl<B: Backend> SurrogateModel<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: bytemuck::Pod,
{
    pub fn new(config: SurrogateConfig<B>) -> Result<Self, AiError> {
        validate_supported_output_features(&config.output_features)?;
        let output_dim = config.output_features.len().max(1);
        let mut model = Self {
            config,
            current_prediction: None,
            input_normalization: None,
            output_normalization: None,
            last_update_time: B::Scalar::ZERO,
            linear_core: LinearRegressionCore::new(output_dim),
            error_ema: None,
            trained: false,
        };

        if let Some(path) = model.config.model_path.clone() {
            model.load_state(&path)?;
        }

        Ok(model)
    }
    pub fn predict(
        &mut self,
        snapshot: &PhysicsSnapshot<B>,
    ) -> Result<SurrogatePrediction<B>, AiError> {
        if !self.trained {
            return Err(AiError::NotReady(
                "surrogate model must be trained or loaded before prediction".into(),
            ));
        }

        let mut features = self.extract_features(snapshot)?;
        self.normalize_input(&mut features)?;

        let mut values = self.forward_linear(&features)?;
        self.denormalize_output(&mut values)?;
        let values: ScalarSamples<B> = values
            .into_iter()
            .map(|v| scalar_from_f64_or_panic::<B>(v, "surrogate.prediction_value"))
            .collect::<Vec<_>>()
            .into();

        let uncertainty = if self.config.estimate_uncertainty {
            if let Some(error_ema) = self.error_ema {
                let estimated_std = error_ema.max(self.config.min_std.to_f64_lossy().max(1e-6));
                Some(
                    vec![
                        scalar_from_f64_or_panic::<B>(
                            estimated_std,
                            "surrogate.estimated_uncertainty",
                        );
                        values.len()
                    ]
                    .into(),
                )
            } else {
                None
            }
        } else {
            None
        };

        let confidence = self
            .error_ema
            .map(|e| 1.0 / (1.0 + e))
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        let prediction = SurrogatePrediction {
            values: values.clone(),
            uncertainty: uncertainty.clone(),
            prediction_time: snapshot.time + self.config.prediction_horizon,
            confidence: scalar_from_f64_or_panic::<B>(confidence, "surrogate.confidence"),
        };

        self.current_prediction = Some(prediction.clone());
        self.last_update_time = snapshot.time;
        Ok(prediction)
    }

    fn extract_features(&self, snapshot: &PhysicsSnapshot<B>) -> Result<Vec<f64>, AiError> {
        let mut feats = Vec::new();
        let push_buffer = |buf: &B::Buffer<B::Scalar>, out: &mut Vec<f64>| {
            out.extend(buf.iter().map(|v| v.to_f64_lossy()));
        };

        if self.config.input_features.is_empty() {
            push_buffer(&snapshot.h, &mut feats);
            push_buffer(&snapshot.u, &mut feats);
            push_buffer(&snapshot.v, &mut feats);
        } else {
            for name in &self.config.input_features {
                match name.as_str() {
                    "h" => push_buffer(&snapshot.h, &mut feats),
                    "u" => push_buffer(&snapshot.u, &mut feats),
                    "v" => push_buffer(&snapshot.v, &mut feats),
                    "z" => push_buffer(&snapshot.z, &mut feats),
                    "sediment" => {
                        let sediment = snapshot.sediment.as_ref().ok_or_else(|| {
                            AiError::StateAccessError(
                                "snapshot is missing sediment requested by surrogate input_features"
                                    .into(),
                            )
                        })?;
                        push_buffer(sediment, &mut feats);
                    }
                    other => {
                        return Err(AiError::InvalidObservation(format!(
                            "unsupported surrogate input feature: {other}"
                        )));
                    }
                }
            }
        }
        if feats.is_empty() {
            return Err(AiError::InvalidShape {
                expected: vec![1],
                actual: vec![0],
            });
        }
        Ok(feats)
    }

    fn normalize_input(&self, features: &mut [f64]) -> Result<(), AiError> {
        if let Some(norm) = &self.input_normalization {
            let (mean, std, _, _) = scalar_normalization_state(norm, "input normalization")?;
            let min_std = self.config.min_std.to_f64_lossy().max(1e-6);
            let std = std.max(min_std);
            for val in features.iter_mut() {
                *val = (*val - mean) / std;
            }
        }
        Ok(())
    }

    fn normalize_output(&self, output: &mut [f64]) -> Result<(), AiError> {
        if let Some(norm) = &self.output_normalization {
            let (mean, std, _, _) = scalar_normalization_state(norm, "output normalization")?;
            let min_std = self.config.min_std.to_f64_lossy().max(1e-6);
            let std = std.max(min_std);
            for val in output.iter_mut() {
                *val = (*val - mean) / std;
            }
        }
        Ok(())
    }

    fn denormalize_output(&self, output: &mut [f64]) -> Result<(), AiError> {
        if let Some(norm) = &self.output_normalization {
            let (mean, std, _, _) = scalar_normalization_state(norm, "output normalization")?;
            for val in output.iter_mut() {
                *val = *val * std + mean;
            }
        }
        Ok(())
    }

    pub fn evaluate_prediction(
        &self,
        prediction: &SurrogatePrediction<B>,
        ground_truth: &PhysicsSnapshot<B>,
    ) -> PredictionMetrics {
        let gt: Vec<f64> = ground_truth.h.iter().map(|v| v.to_f64_lossy()).collect();
        let pred: Vec<f64> = prediction.values.iter().map(|v| v.to_f64_lossy()).collect();
        let mut rmse = 0.0;
        let mut max_err: f64 = 0.0;
        let mut corr_num = 0.0;
        let mut corr_den = 0.0;

        let n = gt.len().min(pred.len());
        for i in 0..n {
            let err = pred[i] - gt[i];
            rmse += err * err;
            max_err = max_err.max(err.abs());
            corr_num += pred[i] * gt[i];
            corr_den += gt[i] * gt[i];
        }
        rmse = if n > 0 { (rmse / n as f64).sqrt() } else { 0.0 };
        let correlation = if corr_den > 0.0 {
            corr_num / corr_den.sqrt()
        } else {
            0.0
        };

        PredictionMetrics {
            rmse,
            max_error: max_err,
            correlation,
            bias: if n > 0 {
                let mean_pred = pred.iter().take(n).sum::<f64>() / n as f64;
                let mean_gt = gt.iter().take(n).sum::<f64>() / n as f64;
                mean_pred - mean_gt
            } else {
                0.0
            },
        }
    }

    pub fn update_model(
        &mut self,
        snapshot: &PhysicsSnapshot<B>,
        target: &[B::Scalar],
    ) -> Result<(), AiError> {
        if target.is_empty() {
            return Err(AiError::InvalidShape {
                expected: vec![1],
                actual: vec![0],
            });
        }

        let features_raw = self.extract_features(snapshot)?;
        let output_dim = target.len();
        self.linear_core
            .ensure_shape(features_raw.len(), output_dim);

        self.update_normalization_input(&features_raw)?;
        let target_raw: Vec<f64> = target.iter().map(|v| v.to_f64_lossy()).collect();
        self.update_normalization_output(&target_raw)?;

        let mut features = features_raw.clone();
        let mut target_norm = target_raw.clone();
        self.normalize_input(&mut features)?;
        self.normalize_output(&mut target_norm)?;

        let pred = self.forward_linear(&features)?;
        let lr = self.config.learning_rate.to_f64_lossy().max(1e-8);
        let l2 = self.config.l2_reg.to_f64_lossy().max(0.0);

        for o in 0..self.linear_core.output_dim {
            let err = pred[o] - target_norm[o];
            self.linear_core.bias[o] -= lr * err;
            for (i, feature) in features
                .iter()
                .take(self.linear_core.input_dim)
                .copied()
                .enumerate()
            {
                let idx = o * self.linear_core.input_dim + i;
                let grad = err * feature + l2 * self.linear_core.weights[idx];
                self.linear_core.weights[idx] -= lr * grad;
            }
        }

        let rmse = if pred.is_empty() {
            0.0
        } else {
            let mut acc = 0.0;
            for (p, t) in pred.iter().zip(target_norm.iter()) {
                let e = p - t;
                acc += e * e;
            }
            (acc / pred.len() as f64).sqrt()
        };

        self.error_ema = Some(match self.error_ema {
            Some(old) => 0.9 * old + 0.1 * rmse,
            None => rmse,
        });

        self.trained = true;
        self.last_update_time = snapshot.time;

        if let Some(path) = self.config.model_path.clone() {
            self.save_state(&path)?;
        }

        Ok(())
    }

    pub fn uncertainty(&self) -> Option<&ScalarSamples<B>> {
        self.current_prediction
            .as_ref()
            .and_then(|p| p.uncertainty.as_ref())
    }

    pub fn is_applicable(&self, _snapshot: &PhysicsSnapshot<B>) -> bool {
        true
    }

    fn forward_linear(&self, features: &[f64]) -> Result<Vec<f64>, AiError> {
        self.linear_core.predict(features)
    }

    fn update_normalization_input(&mut self, values: &[f64]) -> Result<(), AiError> {
        match &mut self.input_normalization {
            Some(norm) => update_normalization(norm, values, self.config.min_std.to_f64_lossy())?,
            None => {
                self.input_normalization = Some(init_normalization(
                    values,
                    self.config.min_std.to_f64_lossy(),
                ))
            }
        }
        Ok(())
    }

    fn update_normalization_output(&mut self, values: &[f64]) -> Result<(), AiError> {
        match &mut self.output_normalization {
            Some(norm) => update_normalization(norm, values, self.config.min_std.to_f64_lossy())?,
            None => {
                self.output_normalization = Some(init_normalization(
                    values,
                    self.config.min_std.to_f64_lossy(),
                ))
            }
        }
        Ok(())
    }

    fn save_state(&self, path: &str) -> Result<(), AiError> {
        let state = SurrogateState {
            linear_core: self.linear_core.clone(),
            input_normalization: self.input_normalization.clone(),
            output_normalization: self.output_normalization.clone(),
            error_ema: self.error_ema,
        };
        let json =
            serde_json::to_string_pretty(&state).map_err(|e| AiError::Other(e.to_string()))?;
        std::fs::write(path, json).map_err(|e| AiError::Other(e.to_string()))
    }

    fn load_state(&mut self, path: &str) -> Result<(), AiError> {
        let content = std::fs::read_to_string(path).map_err(|e| AiError::Other(e.to_string()))?;
        let state: SurrogateState =
            serde_json::from_str(&content).map_err(|e| AiError::Other(e.to_string()))?;
        state.linear_core.validate()?;
        if let Some(norm) = &state.input_normalization {
            let _ = scalar_normalization_state(norm, "input normalization")?;
        }
        if let Some(norm) = &state.output_normalization {
            let _ = scalar_normalization_state(norm, "output normalization")?;
        }
        self.linear_core = state.linear_core;
        self.input_normalization = state.input_normalization;
        self.output_normalization = state.output_normalization;
        self.error_ema = state.error_ema;
        self.trained = true;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PredictionMetrics {
    pub rmse: f64,
    pub max_error: f64,
    pub correlation: f64,
    pub bias: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SurrogateState {
    linear_core: LinearRegressionCore,
    input_normalization: Option<NormalizationParams>,
    output_normalization: Option<NormalizationParams>,
    error_ema: Option<f64>,
}

fn init_normalization(values: &[f64], min_std: f64) -> NormalizationParams {
    let mean = if values.is_empty() {
        vec![0.0]
    } else {
        vec![values.iter().sum::<f64>() / values.len() as f64]
    };
    let std = vec![min_std.max(1e-6)];
    NormalizationParams {
        mean,
        std,
        count: values.len().max(1),
        m2: vec![0.0],
    }
}

fn update_normalization(
    norm: &mut NormalizationParams,
    values: &[f64],
    min_std: f64,
) -> Result<(), AiError> {
    if values.is_empty() {
        return Err(AiError::InvalidShape {
            expected: vec![1],
            actual: vec![0],
        });
    }
    let (mut mean, _, mut m2, mut count) =
        scalar_normalization_state(norm, "surrogate normalization state")?;

    for &x in values {
        count += 1;
        let delta = x - mean;
        mean += delta / count as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    let variance = if count > 1 {
        m2 / (count as f64 - 1.0)
    } else {
        0.0
    };
    let std = variance.sqrt().max(min_std);
    norm.mean = vec![mean];
    norm.std = vec![std];
    norm.count = count;
    norm.m2 = vec![m2];
    Ok(())
}

fn validate_supported_output_features(output_features: &[String]) -> Result<(), AiError> {
    if output_features.is_empty() {
        return Ok(());
    }
    if output_features.len() == 1 && output_features[0] == "h" {
        return Ok(());
    }
    Err(AiError::InvalidObservation(
        "surrogate output_features currently only support the depth field 'h'".into(),
    ))
}

fn scalar_normalization_state(
    norm: &NormalizationParams,
    label: &str,
) -> Result<(f64, f64, f64, usize), AiError> {
    if norm.count == 0 {
        return Err(AiError::InvalidObservation(format!(
            "{label} must carry at least one sample"
        )));
    }
    if norm.mean.len() != 1 || norm.std.len() != 1 || norm.m2.len() != 1 {
        return Err(AiError::InvalidShape {
            expected: vec![1, 1, 1],
            actual: vec![norm.mean.len(), norm.std.len(), norm.m2.len()],
        });
    }
    Ok((norm.mean[0], norm.std[0], norm.m2[0], norm.count))
}

impl<B: Backend> AIAgent<B> for SurrogateModel<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: bytemuck::Pod,
{
    fn name(&self) -> &'static str {
        "Linear-Surrogate"
    }

    fn update(&mut self, snapshot: &PhysicsSnapshot<B>) -> Result<(), AiError> {
        let _ = self.predict(snapshot)?;
        Ok(())
    }

    fn apply(&self, state: &mut dyn Assimilable<B>) -> Result<(), AiError> {
        if let Some(pred) = &self.current_prediction {
            let before_volume = state.total_water_volume().to_f64_lossy();
            let cell_areas = state.cell_areas().copy_to_vec();
            if pred.values.is_empty() || cell_areas.is_empty() {
                return Ok(());
            }
            if pred.values.len() != cell_areas.len() {
                return Err(AiError::InvalidShape {
                    expected: vec![cell_areas.len()],
                    actual: vec![pred.values.len()],
                });
            }
            let n = pred.values.len();

            let old_depth: Vec<B::Scalar> = {
                let depth = state.get_depth();
                depth.iter().take(n).copied().collect()
            };

            {
                let depth = state.get_depth_mut();
                let mut proposed = vec![B::Scalar::ZERO; n];
                for i in 0..n {
                    let blended = (B::Scalar::ONE - self.config.assimilation_rate) * depth[i]
                        + self.config.assimilation_rate * pred.values[i];
                    proposed[i] = blended.max(B::Scalar::ZERO);
                }
                for i in 0..n {
                    depth[i] = proposed[i];
                }
            }

            let depth_after = {
                let mut diff = before_volume - state.total_water_volume().to_f64_lossy();
                let depth = state.get_depth_mut();
                let mut iter = 0usize;
                while diff.abs() > 1e-8 && iter < 5 {
                    let mut sum_area = 0.0f64;
                    for i in 0..n {
                        if depth[i] > B::Scalar::ZERO {
                            sum_area += cell_areas[i].to_f64_lossy();
                        }
                    }
                    if sum_area <= 0.0 {
                        break;
                    }
                    let delta = diff / sum_area;
                    let mut applied = 0.0f64;
                    for i in 0..n {
                        if depth[i] <= B::Scalar::ZERO {
                            continue;
                        }
                        let area = cell_areas[i].to_f64_lossy();
                        if area <= 0.0 {
                            continue;
                        }
                        let new_h = (depth[i].to_f64_lossy() + delta).max(0.0);
                        applied += (new_h - depth[i].to_f64_lossy()) * area;
                        depth[i] =
                            scalar_from_f64_or_panic::<B>(new_h, "surrogate.depth_correction");
                    }
                    diff -= applied;
                    iter += 1;
                }
                depth.to_vec()
            };

            if let Some((u, v)) = state.get_velocity_mut() {
                let n_vel = n.min(u.len()).min(v.len());
                for i in 0..n_vel {
                    let h0 = old_depth[i].to_f64_lossy();
                    let h1 = depth_after[i].to_f64_lossy();
                    if h0 > 0.0 && h1 > 0.0 {
                        let scale = (h0 / h1).clamp(0.1, 10.0);
                        u[i] = scalar_from_f64_or_panic::<B>(
                            u[i].to_f64_lossy() * scale,
                            "surrogate.velocity_scale_u",
                        );
                        v[i] = scalar_from_f64_or_panic::<B>(
                            v[i].to_f64_lossy() * scale,
                            "surrogate.velocity_scale_v",
                        );
                    }
                }
            }

            Ok(())
        } else {
            Err(AiError::NotReady(
                "代理模型预测尚未就绪".into(),
            ))
        }
    }

    fn get_prediction(&self) -> Option<&ScalarSamples<B>> {
        self.current_prediction.as_ref().map(|p| &p.values)
    }

    fn get_uncertainty(&self) -> Option<&ScalarSamples<B>> {
        self.uncertainty()
    }
}
