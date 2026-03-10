use crate::{AIAgent, AiError, Assimilable, PhysicsSnapshot};
use std::marker::PhantomData;

/// 代理模型类型
#[derive(Debug, Clone, Copy)]
pub enum SurrogateType {
    NeuralNetwork,
    ReducedOrder,
    GaussianProcess,
    PolynomialChaos,
}

/// 代理模型配置
#[derive(Debug, Clone)]
pub struct SurrogateConfig {
    pub model_type: SurrogateType,
    pub model_path: Option<String>,
    pub input_features: Vec<String>,
    pub output_features: Vec<String>,
    pub prediction_horizon: f64,
    pub estimate_uncertainty: bool,
}

/// 预测结果
#[derive(Debug, Clone)]
pub struct SurrogatePrediction {
    pub values: Vec<f64>,
    pub uncertainty: Option<Vec<f64>>,
    pub prediction_time: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct NormalizationParams {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PredictionMetrics {
    pub rmse: f64,
    pub max_error: f64,
    pub correlation: f64,
    pub bias: f64,
}

/// 物理代理模型
pub struct SurrogateModel<S: Assimilable> {
    config: SurrogateConfig,
    current_prediction: Option<SurrogatePrediction>,
    input_normalization: Option<NormalizationParams>,
    output_normalization: Option<NormalizationParams>,
    last_update_time: f64,
    _marker: PhantomData<fn() -> S>,
}

impl<S: Assimilable> SurrogateModel<S> {
    pub fn new(config: SurrogateConfig) -> Result<Self, AiError> {
        Ok(Self {
            config,
            current_prediction: None,
            input_normalization: None,
            output_normalization: None,
            last_update_time: 0.0,
            _marker: PhantomData,
        })
    }

    /// 快速预测（简化线性代理）
    pub fn predict(&mut self, snapshot: &PhysicsSnapshot) -> Result<SurrogatePrediction, AiError> {
        let mut features = self.extract_features(snapshot);
        self.normalize_input(&mut features);

        let mean_feature = if features.is_empty() {
            0.0
        } else {
            features.iter().sum::<f64>() / features.len() as f64
        };

        let mut values = vec![mean_feature; self.config.output_features.len().max(1)];
        self.denormalize_output(&mut values);

        let uncertainty = if self.config.estimate_uncertainty {
            Some(vec![0.1; values.len()])
        } else {
            None
        };

        let prediction = SurrogatePrediction {
            values: values.clone(),
            uncertainty: uncertainty.clone(),
            prediction_time: snapshot.time + self.config.prediction_horizon,
            confidence: 0.8,
        };

        self.current_prediction = Some(prediction.clone());
        self.last_update_time = snapshot.time;
        Ok(prediction)
    }

    fn extract_features(&self, snapshot: &PhysicsSnapshot) -> Vec<f64> {
        let mut feats = Vec::new();
        if self.config.input_features.is_empty() {
            feats.extend_from_slice(&snapshot.h);
            feats.extend_from_slice(&snapshot.u);
            feats.extend_from_slice(&snapshot.v);
            return feats;
        }

        for name in &self.config.input_features {
            match name.as_str() {
                "h" => feats.extend_from_slice(&snapshot.h),
                "u" => feats.extend_from_slice(&snapshot.u),
                "v" => feats.extend_from_slice(&snapshot.v),
                "z" => feats.extend_from_slice(&snapshot.z),
                "sediment" => {
                    if let Some(s) = &snapshot.sediment {
                        feats.extend_from_slice(s);
                    }
                }
                _ => {}
            }
        }

        feats
    }

    fn normalize_input(&self, features: &mut [f64]) {
        if let Some(norm) = &self.input_normalization {
            for (i, val) in features.iter_mut().enumerate() {
                let mean = norm.mean.get(i % norm.mean.len()).copied().unwrap_or(0.0);
                let std = norm
                    .std
                    .get(i % norm.std.len())
                    .copied()
                    .unwrap_or(1.0)
                    .max(1e-6);
                *val = (*val - mean) / std;
            }
        }
    }

    fn denormalize_output(&self, output: &mut [f64]) {
        if let Some(norm) = &self.output_normalization {
            for (i, val) in output.iter_mut().enumerate() {
                let mean = norm.mean.get(i % norm.mean.len()).copied().unwrap_or(0.0);
                let std = norm.std.get(i % norm.std.len()).copied().unwrap_or(1.0);
                *val = *val * std + mean;
            }
        }
    }

    pub fn evaluate_prediction(
        &self,
        prediction: &SurrogatePrediction,
        ground_truth: &PhysicsSnapshot,
    ) -> PredictionMetrics {
        let gt = &ground_truth.h;
        let n = gt.len().min(prediction.values.len());

        if n == 0 {
            return PredictionMetrics {
                rmse: 0.0,
                max_error: 0.0,
                correlation: 0.0,
                bias: 0.0,
            };
        }

        let mut rmse = 0.0;
        let mut max_err: f64 = 0.0;
        let mut corr_num = 0.0;
        let mut corr_den = 0.0;

        for i in 0..n {
            let err = prediction.values[i] - gt[i];
            rmse += err * err;
            max_err = max_err.max(err.abs());
            corr_num += prediction.values[i] * gt[i];
            corr_den += gt[i] * gt[i];
        }

        let rmse = (rmse / n as f64).sqrt();
        let correlation = if corr_den > 0.0 {
            corr_num / corr_den.sqrt()
        } else {
            0.0
        };

        let mean_pred = prediction.values.iter().take(n).sum::<f64>() / n as f64;
        let mean_gt = gt.iter().take(n).sum::<f64>() / n as f64;

        PredictionMetrics {
            rmse,
            max_error: max_err,
            correlation,
            bias: mean_pred - mean_gt,
        }
    }

    pub fn update_model(&mut self, _snapshot: &PhysicsSnapshot, _target: &[f64]) -> Result<(), AiError> {
        Ok(())
    }

    pub fn uncertainty(&self) -> Option<&[f64]> {
        self.current_prediction
            .as_ref()
            .and_then(|p| p.uncertainty.as_ref())
            .map(|u| u.as_slice())
    }

    pub fn is_applicable(&self, _snapshot: &PhysicsSnapshot) -> bool {
        true
    }
}

impl<S: Assimilable> AIAgent for SurrogateModel<S> {
    type State = S;

    fn name(&self) -> &'static str {
        "Surrogate-Model"
    }

    fn update(&mut self, snapshot: &PhysicsSnapshot) -> Result<(), AiError> {
        let _ = self.predict(snapshot)?;
        Ok(())
    }

    fn apply(&self, state: &mut Self::State) -> Result<(), AiError> {
        if let Some(pred) = &self.current_prediction {
            let depth = state.get_depth_mut();
            for (dst, src) in depth.iter_mut().zip(pred.values.iter()) {
                *dst = src.max(0.0);
            }
            Ok(())
        } else {
            Err(AiError::NotReady("代理预测尚未生成".into()))
        }
    }

    fn get_prediction(&self) -> Option<&[f64]> {
        self.current_prediction.as_ref().map(|p| p.values.as_slice())
    }

    fn get_uncertainty(&self) -> Option<&[f64]> {
        self.uncertainty()
    }
}
