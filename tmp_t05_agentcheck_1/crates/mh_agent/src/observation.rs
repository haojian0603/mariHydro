use crate::{DefaultBackend, PhysicsSnapshot};
use bytemuck::Pod;
use mh_runtime::{Backend, CellIndex, RuntimeScalar};
use mh_runtime::prelude::{Float, FromPrimitive};

/// 观测算子抽象。
pub trait ObservationOperator<B: Backend = DefaultBackend>: Send + Sync
where
    B::Vector2D: Pod,
{
    /// 观测算子名称。
    fn name(&self) -> &'static str;

    /// 将物理状态映射到观测空间。
    fn observe(&self, snapshot: &PhysicsSnapshot<B>) -> Vec<B::Scalar>;

    /// 计算残差。
    fn residual(&self, snapshot: &PhysicsSnapshot<B>, observation: &[B::Scalar]) -> Vec<B::Scalar> {
        let simulated = self.observe(snapshot);
        simulated
            .iter()
            .zip(observation.iter())
            .map(|(s, o)| *o - *s)
            .collect()
    }

    /// 观测误差方差。
    fn observation_error_variance(&self) -> Option<Vec<B::Scalar>> {
        None
    }

    /// 按观测数量扩展误差方差。
    fn observation_error_variance_for(&self, n_obs: usize) -> Option<Vec<B::Scalar>> {
        self.observation_error_variance()
            .map(|v| vec![v.first().copied().unwrap_or(B::Scalar::ZERO); n_obs])
    }

    /// 线性化结果。
    fn linearize(&self, _snapshot: &PhysicsSnapshot<B>) -> Option<Vec<Vec<B::Scalar>>> {
        None
    }
}

/// 反射率观测算子。
pub struct ReflectanceOperator<B: Backend = DefaultBackend> {
    wavelength: B::Scalar,
    calibration: Vec<B::Scalar>,
    observation_std: B::Scalar,
}

impl<B: Backend> ReflectanceOperator<B>
where
    B::Scalar: RuntimeScalar,
{
    pub fn new(wavelength: f64, calibration: Vec<f64>, observation_std: f64) -> Self {
        Self {
            wavelength: B::Scalar::from_f64(wavelength).unwrap_or(B::Scalar::ZERO),
            calibration: calibration
                .into_iter()
                .map(|v| B::Scalar::from_f64(v).unwrap_or(B::Scalar::ZERO))
                .collect(),
            observation_std: B::Scalar::from_f64(observation_std).unwrap_or(B::Scalar::ZERO),
        }
    }

    /// MODIS 红波段默认参数。
    pub fn modis_red_band() -> Self {
        Self::new(645.0, vec![0.12, 0.01], 0.02)
    }

    /// Sentinel-2 B4 默认参数。
    pub fn sentinel2_b4() -> Self {
        Self::new(665.0, vec![0.09, 0.0], 0.02)
    }
}

impl<B: Backend> ObservationOperator<B> for ReflectanceOperator<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: Pod,
{
    fn name(&self) -> &'static str {
        "Reflectance"
    }

    fn observe(&self, snapshot: &PhysicsSnapshot<B>) -> Vec<B::Scalar> {
        let _ = self.wavelength;
        snapshot
            .sediment
            .as_ref()
            .map(|c| {
                c.iter()
                    .map(|&conc| {
                        let c_safe = conc.max(B::Scalar::from_f64(1e-10).unwrap_or(B::Scalar::MIN_POSITIVE));
                        let a = *self.calibration.get(0).unwrap_or(&B::Scalar::ONE);
                        let b = *self.calibration.get(1).unwrap_or(&B::Scalar::ZERO);
                        a * c_safe.ln() + b
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![B::Scalar::ZERO; snapshot.n_cells()])
    }

    fn observation_error_variance_for(&self, n_obs: usize) -> Option<Vec<B::Scalar>> {
        Some(vec![self.observation_std * self.observation_std; n_obs])
    }
}

/// SAR 极化方式。
#[derive(Debug, Clone, Copy)]
pub enum Polarization {
    VV,
    VH,
    HH,
    HV,
}

/// SAR 后向散射观测算子。
pub struct SAROperator<B: Backend = DefaultBackend> {
    incidence_angle: B::Scalar,
    polarization: Polarization,
    wind_correction: B::Scalar,
    observation_std: B::Scalar,
}

impl<B: Backend> SAROperator<B>
where
    B::Scalar: RuntimeScalar,
{
    pub fn new(incidence_angle: f64, polarization: Polarization) -> Self {
        Self {
            incidence_angle: B::Scalar::from_f64(incidence_angle).unwrap_or(B::Scalar::ZERO),
            polarization,
            wind_correction: B::Scalar::ONE,
            observation_std: B::Scalar::ONE,
        }
    }
}

impl<B: Backend> ObservationOperator<B> for SAROperator<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: Pod,
{
    fn name(&self) -> &'static str {
        "SAR-Backscatter"
    }

    fn observe(&self, snapshot: &PhysicsSnapshot<B>) -> Vec<B::Scalar> {
        let mut result = Vec::with_capacity(snapshot.n_cells());
        let tiny = B::Scalar::from_f64(1e-6).unwrap_or(B::Scalar::MIN_POSITIVE);
        for i in 0..snapshot.n_cells() {
            let speed = snapshot.u[i].hypot(snapshot.v[i]);
            let depth = snapshot.h[i].max(B::Scalar::from_f64(1e-6).unwrap_or(B::Scalar::MIN_POSITIVE));
            let incidence_factor = self.incidence_angle.to_f64_lossy().to_radians().cos().abs();
            let incidence_factor = B::Scalar::from_f64(incidence_factor).unwrap_or(B::Scalar::ONE);
            let pol_factor = match self.polarization {
                Polarization::VV | Polarization::HH => B::Scalar::ONE,
                _ => B::Scalar::from_f64(0.8).unwrap_or(B::Scalar::ONE),
            };
            let backscatter = B::Scalar::from_f64(10.0).unwrap_or(B::Scalar::ONE)
                * ((speed / depth) * incidence_factor * pol_factor * self.wind_correction + tiny).ln();
            result.push(backscatter);
        }
        result
    }

    fn observation_error_variance_for(&self, n_obs: usize) -> Option<Vec<B::Scalar>> {
        Some(vec![self.observation_std * self.observation_std; n_obs])
    }
}

/// 水位观测算子。
pub struct WaterLevelOperator<B: Backend = DefaultBackend> {
    station_indices: Vec<CellIndex>,
    observation_std: B::Scalar,
}

impl<B: Backend> WaterLevelOperator<B>
where
    B::Scalar: RuntimeScalar,
{
    pub fn new(
        station_indices: Vec<CellIndex>,
        observation_std: f64,
        n_cells: usize,
    ) -> Result<Self, crate::AiError> {
        if station_indices.iter().any(|idx| idx.get() >= n_cells) {
            return Err(crate::AiError::InvalidObservation("观测站索引超出范围".into()));
        }

        Ok(Self {
            station_indices,
            observation_std: B::Scalar::from_f64(observation_std).unwrap_or(B::Scalar::ZERO),
        })
    }
}

impl<B: Backend> ObservationOperator<B> for WaterLevelOperator<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: Pod,
{
    fn name(&self) -> &'static str {
        "WaterLevel"
    }

    fn observe(&self, snapshot: &PhysicsSnapshot<B>) -> Vec<B::Scalar> {
        self.station_indices
            .iter()
            .map(|idx| snapshot.h[idx.get()] + snapshot.z[idx.get()])
            .collect()
    }

    fn observation_error_variance(&self) -> Option<Vec<B::Scalar>> {
        Some(vec![self.observation_std * self.observation_std; self.station_indices.len()])
    }
}
