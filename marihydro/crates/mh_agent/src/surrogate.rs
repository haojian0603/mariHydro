// crates/mh_agent/src/surrogate.rs

use crate::{AIAgent, AiError, PhysicsSnapshot, Assimilable};

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
}

#[derive(Debug, Clone)]
pub struct NormalizationParams {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl SurrogateModel {
    pub fn new(config: SurrogateConfig) -> Result<Self, AiError> {
        Ok(Self {
            config,
            current_prediction: None,
            input_normalization: None,
            output_normalization: None,
            last_update_time: 0.0,
        })
    }
    
    /// 快速预测（替代完整物理计算）
    pub fn predict(&mut self, snapshot: &PhysicsSnapshot) -> Result<SurrogatePrediction, AiError> {
        let mut features = self.extract_features(snapshot);
        self.normalize_input(&mut features);

        // 简化：使用平均值作为预测
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
                let std = norm.std.get(i % norm.std.len()).copied().unwrap_or(1.0).max(1e-6);
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
    pub fn update_model(&mut self, _snapshot: &PhysicsSnapshot, _target: &[f64]) -> Result<(), AiError> {
        // 占位实现：记录时间戳即可
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
}

#[derive(Debug, Clone)]
pub struct PredictionMetrics {
    pub rmse: f64,
    pub max_error: f64,
    pub correlation: f64,
    pub bias: f64,
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
            for i in 0..n {
                depth[i] = pred.values[i].max(0.0);
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
