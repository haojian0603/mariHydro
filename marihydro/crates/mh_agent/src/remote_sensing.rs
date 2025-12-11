// crates/mh_agent/src/remote_sensing.rs

use crate::{AIAgent, AiError, Assimilable, PhysicsSnapshot};

/// 传感器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorType {
    /// 光学遥感（MODIS, Landsat, Sentinel-2）
    Optical,
    /// 合成孔径雷达（Sentinel-1, RADARSAT）
    SAR,
    /// 高光谱
    Hyperspectral,
}

/// 卫星图像数据
#[derive(Debug, Clone)]
pub struct SatelliteImage {
    /// 反射率/后向散射数据
    pub data: Vec<f32>,
    /// 图像尺寸 (width, height)
    pub dimensions: (usize, usize),
    /// 地理范围 [min_x, min_y, max_x, max_y]
    pub bounds: [f64; 4],
    /// 获取时间（Unix时间戳）
    pub timestamp: f64,
    /// 传感器类型
    pub sensor: SensorType,
    /// 云覆盖率 (0.0 - 1.0)
    pub cloud_cover: f32,
    /// 空间分辨率 [m]
    pub resolution: f64,
}

/// 遥感反演配置
#[derive(Debug, Clone)]
pub struct RemoteSensingConfig {
    /// 模型路径（ONNX格式）
    pub model_path: Option<String>,
    /// 同化率
    pub assimilation_rate: f64,
    /// 最大反演浓度 [kg/m³]
    pub max_concentration: f64,
    /// 最小可信云覆盖阈值
    pub max_cloud_cover: f32,
    /// 空间插值方法
    pub interpolation: InterpolationMethod,
}

#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod {
    NearestNeighbor,
    Bilinear,
    IDW { power: f64 },
}

/// 遥感反演结果
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// 反演的浓度场
    pub concentration: Vec<f64>,
    /// 不确定性估计
    pub uncertainty: Vec<f64>,
    /// 质量标志（云遮挡、边界效应等）
    pub quality_flags: Vec<u8>,
}

/// 遥感泥沙反演代理
pub struct RemoteSensingAgent {
    config: RemoteSensingConfig,
    /// 预测结果缓存
    predicted: Vec<f64>,
    /// 不确定性缓存
    uncertainty: Vec<f64>,
    /// 上次反演时间
    last_inference_time: f64,
    /// 是否有有效预测
    has_prediction: bool,
}

impl RemoteSensingAgent {
    pub fn new(config: RemoteSensingConfig) -> Self {
        Self {
            config,
            predicted: Vec::new(),
            uncertainty: Vec::new(),
            last_inference_time: 0.0,
            has_prediction: false,
        }
    }
    
    /// 从卫星图像进行推理
    pub fn infer(&mut self, image: &SatelliteImage, target_cells: &[[f64; 2]]) -> Result<InferenceResult, AiError> {
        self.validate_image(image)?;
        let mapped = self.interpolate_to_grid(&image.data, image, target_cells);

        let mut uncertainty = vec![0.0; mapped.len()];
        for u in &mut uncertainty {
            *u = (image.cloud_cover as f64).min(1.0);
        }

        let result = InferenceResult {
            concentration: mapped.clone(),
            uncertainty: uncertainty.clone(),
            quality_flags: vec![0u8; mapped.len()],
        };

        self.predicted = mapped;
        self.uncertainty = uncertainty;
        self.has_prediction = true;
        self.last_inference_time = image.timestamp;
        Ok(result)
    }
    
    /// 获取预测浓度场
    pub fn predicted(&self) -> Option<&[f64]> {
        if self.has_prediction { Some(&self.predicted) } else { None }
    }
    
    /// 获取不确定性
    pub fn uncertainty(&self) -> Option<&[f64]> {
        if self.has_prediction { Some(&self.uncertainty) } else { None }
    }
    
    /// 检查图像质量
    fn validate_image(&self, image: &SatelliteImage) -> Result<(), AiError> {
        let (w, h) = image.dimensions;
        if w == 0 || h == 0 {
            return Err(AiError::InvalidObservation("图像尺寸无效".into()));
        }
        if image.data.len() != w * h {
            return Err(AiError::InvalidObservation("图像数据长度与尺寸不匹配".into()));
        }
        if image.cloud_cover > self.config.max_cloud_cover {
            return Err(AiError::InvalidObservation("云覆盖率超出阈值".into()));
        }
        Ok(())
    }
    
    /// 空间插值到目标网格
    fn interpolate_to_grid(
        &self,
        data: &[f32],
        image: &SatelliteImage,
        target_cells: &[[f64; 2]],
    ) -> Vec<f64> {
        let (width, height) = image.dimensions;
        let (min_x, min_y, max_x, max_y) = (image.bounds[0], image.bounds[1], image.bounds[2], image.bounds[3]);
        let dx = (max_x - min_x) / width as f64;
        let dy = (max_y - min_y) / height as f64;

        let mut result = Vec::with_capacity(target_cells.len());
        for &cell in target_cells {
            let gx = ((cell[0] - min_x) / dx).clamp(0.0, width.saturating_sub(1) as f64);
            let gy = ((cell[1] - min_y) / dy).clamp(0.0, height.saturating_sub(1) as f64);

            let value = match self.config.interpolation {
                InterpolationMethod::NearestNeighbor => {
                    let ix = gx.round() as usize;
                    let iy = gy.round() as usize;
                    let idx = iy * width + ix;
                    data.get(idx).copied().unwrap_or_default() as f64
                }
                InterpolationMethod::Bilinear => {
                    let x0 = gx.floor().max(0.0) as usize;
                    let x1 = (x0 + 1).min(width.saturating_sub(1));
                    let y0 = gy.floor().max(0.0) as usize;
                    let y1 = (y0 + 1).min(height.saturating_sub(1));
                    let tx = gx - x0 as f64;
                    let ty = gy - y0 as f64;

                    let v00 = data.get(y0 * width + x0).copied().unwrap_or_default() as f64;
                    let v10 = data.get(y0 * width + x1).copied().unwrap_or_default() as f64;
                    let v01 = data.get(y1 * width + x0).copied().unwrap_or_default() as f64;
                    let v11 = data.get(y1 * width + x1).copied().unwrap_or_default() as f64;
                    let vx0 = v00 * (1.0 - tx) + v10 * tx;
                    let vx1 = v01 * (1.0 - tx) + v11 * tx;
                    vx0 * (1.0 - ty) + vx1 * ty
                }
                InterpolationMethod::IDW { power } => {
                    let ix = gx.round().max(0.0) as usize;
                    let iy = gy.round().max(0.0) as usize;
                    let idx = iy * width + ix;
                    let base = data.get(idx).copied().unwrap_or_default() as f64;
                    let w = 1.0 / (dx.hypot(dy).max(1e-6).powf(power));
                    base * w
                }
            };
            let conc = self.empirical_inversion(value as f32, image.sensor)
                .min(self.config.max_concentration)
                .max(0.0);
            result.push(conc);
        }
        result
    }
    
    /// 经验公式反演（无模型时使用）
    fn empirical_inversion(&self, reflectance: f32, sensor: SensorType) -> f64 {
        match sensor {
            SensorType::Optical => (reflectance.max(1e-6)).ln().abs() * 10.0,
            SensorType::SAR => (reflectance as f64).abs() * 5.0,
            SensorType::Hyperspectral => (reflectance as f64).sqrt() * 8.0,
        }
    }
    
    /// 清除缓存
    pub fn clear_cache(&mut self) {
        self.predicted.clear();
        self.uncertainty.clear();
        self.has_prediction = false;
    }
}

impl AIAgent for RemoteSensingAgent {
    fn name(&self) -> &'static str { "RemoteSensing-Sediment" }
    
    fn update(&mut self, _snapshot: &PhysicsSnapshot) -> Result<(), AiError> {
        // 遥感推理依赖外部图像，update不执行耗时操作
        Ok(())
    }
    
    fn apply(&self, state: &mut dyn Assimilable) -> Result<(), AiError> {
        if !self.has_prediction {
            return Err(AiError::NotReady("遥感预测尚未生成".into()));
        }
        let prediction = &self.predicted;
        let target = state
            .get_tracer_mut("sediment")
            .ok_or_else(|| AiError::StateAccessError("无法获取泥沙示踪剂".into()))?;
        let n = prediction.len().min(target.len());
        for i in 0..n {
            let blended = (1.0 - self.config.assimilation_rate) * target[i]
                + self.config.assimilation_rate * prediction[i];
            target[i] = blended.min(self.config.max_concentration);
        }
        Ok(())
    }
    
    fn get_prediction(&self) -> Option<&[f64]> {
        self.predicted()
    }
    
    fn get_uncertainty(&self) -> Option<&[f64]> {
        self.uncertainty()
    }
}
