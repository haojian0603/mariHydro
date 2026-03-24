use crate::{AIAgent, AiError, Assimilable, DefaultBackend, PhysicsSnapshot, ScalarSamples};
use mh_runtime::{Backend, RuntimeScalar, Vector2D};
use mh_runtime::prelude::{Float, FromPrimitive};

/// 传感器类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorType {
    Optical,
    SAR,
    Hyperspectral,
}

/// 卫星图像数据。
#[derive(Debug, Clone)]
pub struct SatelliteImage {
    pub data: Vec<f32>,
    pub dimensions: (usize, usize),
    pub bounds: [f64; 4],
    pub timestamp: f64,
    pub sensor: SensorType,
    pub cloud_cover: f32,
    pub resolution: f64,
}

/// 遥感同化配置。
#[derive(Debug, Clone)]
pub struct RemoteSensingConfig<B: Backend = DefaultBackend> {
    pub model_path: Option<String>,
    pub assimilation_rate: B::Scalar,
    pub max_concentration: B::Scalar,
    pub max_cloud_cover: f32,
    pub interpolation: InterpolationMethod<B>,
}

/// 插值方式。
#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod<B: Backend = DefaultBackend> {
    NearestNeighbor,
    Bilinear,
    IDW { power: B::Scalar },
}

/// 遥感推理结果。
#[derive(Debug, Clone)]
pub struct InferenceResult<B: Backend = DefaultBackend> {
    pub concentration: ScalarSamples<B>,
    pub uncertainty: ScalarSamples<B>,
    pub quality_flags: Vec<u8>,
}

/// 遥感代理。
pub struct RemoteSensingAgent<B: Backend = DefaultBackend> {
    config: RemoteSensingConfig<B>,
    predicted: ScalarSamples<B>,
    uncertainty: ScalarSamples<B>,
    last_inference_time: f64,
    has_prediction: bool,
}

impl<B: Backend> RemoteSensingAgent<B>
where
    B::Scalar: RuntimeScalar,
{
    pub fn new(config: RemoteSensingConfig<B>) -> Self {
        Self {
            config,
            predicted: ScalarSamples::default(),
            uncertainty: ScalarSamples::default(),
            last_inference_time: 0.0,
            has_prediction: false,
        }
    }

    /// 从卫星图像反演浓度。
    pub fn infer(
        &mut self,
        image: &SatelliteImage,
        target_cells: &[B::Vector2D],
    ) -> Result<InferenceResult<B>, AiError> {
        self.validate_image(image)?;
        let mapped = self.interpolate_to_grid(&image.data, image, target_cells);

        let cloud = B::Scalar::from_f64(image.cloud_cover as f64).unwrap_or(B::Scalar::ONE);
        let mut uncertainty = vec![B::Scalar::ZERO; mapped.len()];
        for u in &mut uncertainty {
            *u = cloud.min(B::Scalar::ONE);
        }

        let result = InferenceResult {
            concentration: mapped.clone(),
            uncertainty: uncertainty.clone().into(),
            quality_flags: vec![0u8; mapped.len()],
        };

        self.predicted = mapped;
        self.uncertainty = uncertainty.into();
        self.has_prediction = true;
        self.last_inference_time = image.timestamp;
        Ok(result)
    }

    pub fn predicted(&self) -> Option<&ScalarSamples<B>> {
        if self.has_prediction {
            Some(&self.predicted)
        } else {
            None
        }
    }

    pub fn uncertainty(&self) -> Option<&ScalarSamples<B>> {
        if self.has_prediction {
            Some(&self.uncertainty)
        } else {
            None
        }
    }

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

    fn interpolate_to_grid(
        &self,
        data: &[f32],
        image: &SatelliteImage,
        target_cells: &[B::Vector2D],
    ) -> ScalarSamples<B> {
        let (width, height) = image.dimensions;
        let (min_x, min_y, max_x, max_y) =
            (image.bounds[0], image.bounds[1], image.bounds[2], image.bounds[3]);
        let dx = (max_x - min_x) / width.saturating_sub(1).max(1) as f64;
        let dy = (max_y - min_y) / height.saturating_sub(1).max(1) as f64;

        let mut result = Vec::with_capacity(target_cells.len());
        for cell in target_cells {
            let x = cell.x().to_f64_lossy();
            let y = cell.y().to_f64_lossy();
            let gx = ((x - min_x) / dx).clamp(0.0, width.saturating_sub(1) as f64);
            let gy = ((y - min_y) / dy).clamp(0.0, height.saturating_sub(1) as f64);

            let reflectance = match self.config.interpolation {
                InterpolationMethod::NearestNeighbor => {
                    let ix = (gx + 0.5).floor().clamp(0.0, width.saturating_sub(1) as f64) as usize;
                    let iy = (gy + 0.5).floor().clamp(0.0, height.saturating_sub(1) as f64) as usize;
                    let idx = iy * width + ix;
                    data.get(idx).copied().unwrap_or_default() as f64
                }
                InterpolationMethod::Bilinear => {
                    let x0 = gx.floor().max(0.0) as usize;
                    let y0 = gy.floor().max(0.0) as usize;
                    let x1 = (x0 + 1).min(width.saturating_sub(1));
                    let y1 = (y0 + 1).min(height.saturating_sub(1));
                    let tx = (gx - x0 as f64).clamp(0.0, 1.0);
                    let ty = (gy - y0 as f64).clamp(0.0, 1.0);
                    let v00 = data.get(y0 * width + x0).copied().unwrap_or_default() as f64;
                    let v10 = data.get(y0 * width + x1).copied().unwrap_or_default() as f64;
                    let v01 = data.get(y1 * width + x0).copied().unwrap_or_default() as f64;
                    let v11 = data.get(y1 * width + x1).copied().unwrap_or_default() as f64;
                    let vx0 = v00 * (1.0 - tx) + v10 * tx;
                    let vx1 = v01 * (1.0 - tx) + v11 * tx;
                    vx0 * (1.0 - ty) + vx1 * ty
                }
                InterpolationMethod::IDW { power } => {
                    let x0 = gx.floor() as usize;
                    let y0 = gy.floor() as usize;
                    let x1 = (x0 + 1).min(width.saturating_sub(1));
                    let y1 = (y0 + 1).min(height.saturating_sub(1));
                    let pts = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
                    let mut ws = 0.0;
                    let mut vs = 0.0;
                    let power = power.to_f64_lossy();
                    for (ix, iy) in pts {
                        let px = min_x + ix as f64 * dx;
                        let py = min_y + iy as f64 * dy;
                        let d = ((x - px).powi(2) + (y - py).powi(2)).sqrt().max(1e-6);
                        let wgt = 1.0 / d.powf(power);
                        let val = data.get(iy * width + ix).copied().unwrap_or_default() as f64;
                        ws += wgt;
                        vs += wgt * val;
                    }
                    if ws > 0.0 { vs / ws } else { 0.0 }
                }
            };

            let conc = self.empirical_inversion(reflectance, image.sensor);
            result.push(conc.min(self.config.max_concentration).max(B::Scalar::ZERO));
        }
        result.into()
    }

    fn empirical_inversion(&self, reflectance: f64, sensor: SensorType) -> B::Scalar {
        let value = match sensor {
            SensorType::Optical => (reflectance.max(1e-6)).ln().abs() * 10.0,
            SensorType::SAR => reflectance.abs() * 5.0,
            SensorType::Hyperspectral => reflectance.sqrt() * 8.0,
        };
        B::Scalar::from_f64(value).unwrap_or(B::Scalar::ZERO)
    }

    pub fn clear_cache(&mut self) {
        self.predicted = ScalarSamples::default();
        self.uncertainty = ScalarSamples::default();
        self.has_prediction = false;
    }
}

impl<B: Backend> AIAgent<B> for RemoteSensingAgent<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: bytemuck::Pod,
{
    fn name(&self) -> &'static str {
        "RemoteSensing-Sediment"
    }

    fn update(&mut self, _snapshot: &PhysicsSnapshot<B>) -> Result<(), AiError> {
        Ok(())
    }

    fn apply(&self, state: &mut dyn Assimilable<B>) -> Result<(), AiError> {
        if !self.has_prediction {
            return Err(AiError::NotReady("遥感反演结果尚未生成".into()));
        }

        let prediction = &self.predicted;
        let target = state
            .get_tracer_mut("sediment")
            .ok_or_else(|| AiError::StateAccessError("无法获取泥沙示踪量".into()))?;
        let n = prediction.len().min(target.len());
        for i in 0..n {
            let blended = (B::Scalar::ONE - self.config.assimilation_rate) * target[i]
                + self.config.assimilation_rate * prediction[i];
            target[i] = blended.min(self.config.max_concentration);
        }
        Ok(())
    }

    fn get_prediction(&self) -> Option<&ScalarSamples<B>> {
        self.predicted()
    }

    fn get_uncertainty(&self) -> Option<&ScalarSamples<B>> {
        self.uncertainty()
    }
}
