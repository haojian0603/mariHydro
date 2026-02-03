// crates/mh_agent/src/surrogate.rs

use crate::{AIAgent, AiError, Assimilable, PhysicsSnapshot};
use serde::{Deserialize, Serialize};
use serde_json;

/// 代理模型类型
#[derive(Debug, Clone, Copy)]
pub enum SurrogateType {
    /// 神经网络代理
    NeuralNetwork,
    /// 降阶模型（POD/DMD）
    ReducedOrder,
    /// 高斯过程回归
    GaussianProcess,
    /// 多项式混沌展开
    PolynomialChaos,
}

/// 代理模型配置
#[derive(Debug, Clone)]
pub struct SurrogateConfig {
    pub model_type: SurrogateType,
    pub model_path: Option<String>,
    /// 输入特征列表
    pub input_features: Vec<String>,
    /// 输出特征列表
    pub output_features: Vec<String>,
    /// 预测时间步长 [s]
    pub prediction_horizon: f64,
    /// 是否提供不确定性估计
    pub estimate_uncertainty: bool,
    /// 融合系数
    pub assimilation_rate: f64,
    /// 学习率
    pub learning_rate: f64,
    /// L2 正则
    pub l2_reg: f64,
    /// 最小标准差
    pub min_std: f64,
}

/// 代理模型预测结果
#[derive(Debug, Clone)]
pub struct SurrogatePrediction {
    /// 预测值
    pub values: Vec<f64>,
    /// 不确定性（如果可用）
    pub uncertainty: Option<Vec<f64>>,
    /// 预测时间
    pub prediction_time: f64,
    /// 模型置信度
    pub confidence: f64,
}

/// 物理代理模型
pub struct SurrogateModel {
    config: SurrogateConfig,
    /// 当前预测缓存
    current_prediction: Option<SurrogatePrediction>,
    /// 输入归一化参数
    input_normalization: Option<NormalizationParams>,
    /// 输出归一化参数
    output_normalization: Option<NormalizationParams>,
    /// 上次更新时间
    last_update_time: f64,
    /// 线性模型权重（输出维度 × 输入维度）
    weights: Option<Vec<f64>>,
    /// 偏置项
    bias: Vec<f64>,
    /// 误差指数滑动平均
    error_ema: Option<f64>,
    /// 输入维度
    input_dim: usize,
    /// 输出维度
    output_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub count: usize,
    pub m2: Vec<f64>,
}

impl SurrogateModel {
    pub fn new(config: SurrogateConfig) -> Result<Self, AiError> {
        let mut model = Self {
            input_dim: 0,
            output_dim: config.output_features.len().max(1),
            config,
            current_prediction: None,
            input_normalization: None,
            output_normalization: None,
            last_update_time: 0.0,
            weights: None,
            bias: Vec::new(),
            error_ema: None,
        };

        if let Some(path) = model.config.model_path.clone() {
            let _ = model.load_state(&path);
        }

        Ok(model)
    }
    
    /// 快速预测（替代完整物理计算）
    pub fn predict(&mut self, snapshot: &PhysicsSnapshot) -> Result<SurrogatePrediction, AiError> {
        let mut features = self.extract_features(snapshot);
        self.ensure_model_initialized(features.len());
        self.normalize_input(&mut features);

        let mut values = self.forward_linear(&features);
        self.denormalize_output(&mut values);

        let uncertainty = if self.config.estimate_uncertainty {
            Some(vec![0.1; values.len()])
        } else {
            None
        };

        let confidence = self
            .error_ema
            .map(|e| 1.0 / (1.0 + e))
            .unwrap_or(0.8)
            .clamp(0.0, 1.0);

        let prediction = SurrogatePrediction {
            values: values.clone(),
            uncertainty: uncertainty.clone(),
            prediction_time: snapshot.time + self.config.prediction_horizon,
            confidence,
        };

        self.current_prediction = Some(prediction.clone());
        self.last_update_time = snapshot.time;
        Ok(prediction)
    }
    
    /// 提取输入特征
    fn extract_features(&self, snapshot: &PhysicsSnapshot) -> Vec<f64> {
        let mut feats = Vec::new();
        if self.config.input_features.is_empty() {
            feats.extend_from_slice(&snapshot.h);
            feats.extend_from_slice(&snapshot.u);
            feats.extend_from_slice(&snapshot.v);
        } else {
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
        }
        feats
    }
    
    /// 归一化输入
    fn normalize_input(&self, features: &mut [f64]) {
        if let Some(norm) = &self.input_normalization {
            for (i, val) in features.iter_mut().enumerate() {
                let mean = norm.mean.get(i % norm.mean.len()).copied().unwrap_or(0.0);
                let std = norm.std.get(i % norm.std.len()).copied().unwrap_or(1.0).max(self.config.min_std);
                *val = (*val - mean) / std;
            }
        }
    }

    fn normalize_output(&self, output: &mut [f64]) {
        if let Some(norm) = &self.output_normalization {
            for (i, val) in output.iter_mut().enumerate() {
                let mean = norm.mean.get(i % norm.mean.len()).copied().unwrap_or(0.0);
                let std = norm.std.get(i % norm.std.len()).copied().unwrap_or(1.0).max(self.config.min_std);
                *val = (*val - mean) / std;
            }
        }
    }
    
    /// 反归一化输出
    fn denormalize_output(&self, output: &mut [f64]) {
        if let Some(norm) = &self.output_normalization {
            for (i, val) in output.iter_mut().enumerate() {
                let mean = norm.mean.get(i % norm.mean.len()).copied().unwrap_or(0.0);
                let std = norm.std.get(i % norm.std.len()).copied().unwrap_or(1.0);
                *val = *val * std + mean;
            }
        }
    }
    
    /// 评估预测质量（与完整物理对比）
    pub fn evaluate_prediction(
        &self,
        prediction: &SurrogatePrediction,
        ground_truth: &PhysicsSnapshot,
    ) -> PredictionMetrics {
        let gt = &ground_truth.h;
        let mut rmse = 0.0;
        let mut max_err = 0.0;
        let mut corr_num = 0.0;
        let mut corr_den = 0.0;

        let n = gt.len().min(prediction.values.len());
        for i in 0..n {
            let err = prediction.values[i] - gt[i];
            rmse += err * err;
            max_err = max_err.max(err.abs());
            corr_num += prediction.values[i] * gt[i];
            corr_den += gt[i] * gt[i];
        }
        rmse = if n > 0 { (rmse / n as f64).sqrt() } else { 0.0 };
        let correlation = if corr_den > 0.0 { corr_num / corr_den.sqrt() } else { 0.0 };

        PredictionMetrics {
            rmse,
            max_error: max_err,
            correlation,
            bias: if n > 0 {
                let mean_pred = prediction.values.iter().take(n).sum::<f64>() / n as f64;
                let mean_gt = gt.iter().take(n).sum::<f64>() / n as f64;
                mean_pred - mean_gt
            } else {
                0.0
            },
        }
    }
    
    /// 更新模型（在线学习）
    pub fn update_model(&mut self, snapshot: &PhysicsSnapshot, target: &[f64]) -> Result<(), AiError> {
        let features_raw = self.extract_features(snapshot);
        self.ensure_model_initialized(features_raw.len());
        let output_dim = target.len().max(1);
        if output_dim != self.output_dim {
            self.output_dim = output_dim;
            self.weights = None;
            self.bias = Vec::new();
            self.ensure_model_initialized(features_raw.len());
        }

        self.update_normalization_input(&features_raw);
        self.update_normalization_output(target);

        let mut features = features_raw.clone();
        let mut target_norm = target.to_vec();
        self.normalize_input(&mut features);
        self.normalize_output(&mut target_norm);

        let pred = self.forward_linear(&features);
        let lr = self.config.learning_rate.max(1e-8);
        let l2 = self.config.l2_reg.max(0.0);

        if let Some(weights) = &mut self.weights {
            for o in 0..self.output_dim {
                let err = pred.get(o).copied().unwrap_or(0.0) - target_norm.get(o).copied().unwrap_or(0.0);
                self.bias[o] -= lr * err;
                for i in 0..self.input_dim {
                    let idx = o * self.input_dim + i;
                    let grad = err * features.get(i).copied().unwrap_or(0.0) + l2 * weights[idx];
                    weights[idx] -= lr * grad;
                }
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

        self.last_update_time = snapshot.time;

        if let Some(path) = self.config.model_path.clone() {
            let _ = self.save_state(&path);
        }

        Ok(())
    }
    
    /// 获取预测不确定性
    pub fn uncertainty(&self) -> Option<&[f64]> {
        self.current_prediction
            .as_ref()
            .and_then(|p| p.uncertainty.as_ref())
            .map(|u| u.as_slice())
    }
    
    /// 检查模型是否适用于当前状态
    pub fn is_applicable(&self, _snapshot: &PhysicsSnapshot) -> bool {
        true
    }

    fn ensure_model_initialized(&mut self, input_dim: usize) {
        if self.weights.is_some() && self.input_dim == input_dim {
            return;
        }
        self.input_dim = input_dim.max(1);
        self.output_dim = self.output_dim.max(1);
        self.weights = Some(vec![0.0; self.input_dim * self.output_dim]);
        self.bias = vec![0.0; self.output_dim];
    }

    fn forward_linear(&self, features: &[f64]) -> Vec<f64> {
        let output_dim = self.output_dim.max(1);
        let input_dim = self.input_dim.max(1);
        let mut out = vec![0.0; output_dim];
        if let Some(weights) = &self.weights {
            for o in 0..output_dim {
                let mut sum = self.bias.get(o).copied().unwrap_or(0.0);
                for i in 0..input_dim {
                    let idx = o * input_dim + i;
                    sum += weights[idx] * features.get(i).copied().unwrap_or(0.0);
                }
                out[o] = sum;
            }
        }
        out
    }

    fn update_normalization_input(&mut self, values: &[f64]) {
        match &mut self.input_normalization {
            Some(norm) => update_normalization(norm, values, self.config.min_std),
            None => self.input_normalization = Some(init_normalization(values, self.config.min_std)),
        }
    }

    fn update_normalization_output(&mut self, values: &[f64]) {
        match &mut self.output_normalization {
            Some(norm) => update_normalization(norm, values, self.config.min_std),
            None => self.output_normalization = Some(init_normalization(values, self.config.min_std)),
        }
    }

    fn save_state(&self, path: &str) -> Result<(), AiError> {
        let state = SurrogateState {
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            weights: self.weights.clone().unwrap_or_default(),
            bias: self.bias.clone(),
            input_normalization: self.input_normalization.clone(),
            output_normalization: self.output_normalization.clone(),
            error_ema: self.error_ema,
        };
        let json = serde_json::to_string_pretty(&state).map_err(|e| AiError::Other(e.to_string()))?;
        std::fs::write(path, json).map_err(|e| AiError::Other(e.to_string()))
    }

    fn load_state(&mut self, path: &str) -> Result<(), AiError> {
        let content = std::fs::read_to_string(path).map_err(|e| AiError::Other(e.to_string()))?;
        let state: SurrogateState = serde_json::from_str(&content).map_err(|e| AiError::Other(e.to_string()))?;
        self.input_dim = state.input_dim.max(1);
        self.output_dim = state.output_dim.max(1);
        self.weights = Some(state.weights);
        self.bias = state.bias;
        self.input_normalization = state.input_normalization;
        self.output_normalization = state.output_normalization;
        self.error_ema = state.error_ema;
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SurrogateState {
    input_dim: usize,
    output_dim: usize,
    weights: Vec<f64>,
    bias: Vec<f64>,
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

fn update_normalization(norm: &mut NormalizationParams, values: &[f64], min_std: f64) {
    if values.is_empty() {
        return;
    }
    let mut mean = norm.mean.get(0).copied().unwrap_or(0.0);
    let mut m2 = norm.m2.get(0).copied().unwrap_or(0.0);
    let mut count = norm.count;

    for &x in values {
        count += 1;
        let delta = x - mean;
        mean += delta / count as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    let variance = if count > 1 { m2 / (count as f64 - 1.0) } else { 0.0 };
    let std = variance.sqrt().max(min_std);
    norm.mean = vec![mean];
    norm.std = vec![std];
    norm.count = count;
    norm.m2 = vec![m2];
}

impl AIAgent for SurrogateModel {
    fn name(&self) -> &'static str { "Surrogate-Model" }
    
    fn update(&mut self, snapshot: &PhysicsSnapshot) -> Result<(), AiError> {
        let _ = self.predict(snapshot)?;
        Ok(())
    }
    
    fn apply(&self, state: &mut dyn Assimilable) -> Result<(), AiError> {
        if let Some(pred) = &self.current_prediction {
            let depth = state.get_depth_mut();
            let n = pred.values.len().min(depth.len());
            if n == 0 {
                return Ok(());
            }

            let before_volume = state.total_water_volume();
            let old_depth: Vec<f64> = depth.iter().take(n).copied().collect();
            let mut proposed = vec![0.0f64; n];

            for i in 0..n {
                let blended = (1.0 - self.config.assimilation_rate) * depth[i]
                    + self.config.assimilation_rate * pred.values[i];
                proposed[i] = blended.max(0.0);
            }

            for i in 0..n {
                depth[i] = proposed[i];
            }

            let cell_areas = state.cell_areas();
            let mut iter = 0usize;
            let mut diff = before_volume - state.total_water_volume();
            while diff.abs() > 1e-8 && iter < 5 {
                let mut sum_area = 0.0f64;
                for i in 0..n {
                    if depth[i] > 0.0 {
                        sum_area += cell_areas.get(i).copied().unwrap_or(0.0);
                    }
                }
                if sum_area <= 0.0 {
                    break;
                }
                let delta = diff / sum_area;
                let mut applied = 0.0f64;
                for i in 0..n {
                    if depth[i] <= 0.0 {
                        continue;
                    }
                    let area = cell_areas.get(i).copied().unwrap_or(0.0);
                    if area <= 0.0 {
                        continue;
                    }
                    let new_h = (depth[i] + delta).max(0.0);
                    applied += (new_h - depth[i]) * area;
                    depth[i] = new_h;
                }
                diff -= applied;
                iter += 1;
            }

            if let Some((u, v)) = state.get_velocity_mut() {
                let n_vel = n.min(u.len()).min(v.len());
                for i in 0..n_vel {
                    let h0 = old_depth[i];
                    let h1 = depth[i];
                    if h0 > 0.0 && h1 > 0.0 {
                        let scale = (h0 / h1).clamp(0.1, 10.0);
                        u[i] *= scale;
                        v[i] *= scale;
                    }
                }
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
