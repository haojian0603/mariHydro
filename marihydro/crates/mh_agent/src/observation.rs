use crate::{
    scalar_from_f64_or_panic, AiError, DefaultBackend, DenseScalarMatrix, PhysicsSnapshot,
    ScalarSamples,
};
use bytemuck::Pod;
use mh_runtime::prelude::Float;
use mh_runtime::{Backend, CellIndex, RuntimeScalar};

pub trait ObservationOperator<B: Backend = DefaultBackend>: Send + Sync
where
    B::Vector2D: Pod,
{
    fn name(&self) -> &'static str;

    fn observe(&self, snapshot: &PhysicsSnapshot<B>) -> ScalarSamples<B>;

    fn residual(
        &self,
        snapshot: &PhysicsSnapshot<B>,
        observation: &[B::Scalar],
    ) -> ScalarSamples<B> {
        let simulated = self.observe(snapshot);
        simulated
            .iter()
            .zip(observation.iter())
            .map(|(s, o)| *o - *s)
            .collect::<Vec<_>>()
            .into()
    }

    fn observation_error_variance(&self) -> Option<ScalarSamples<B>> {
        None
    }

    fn observation_error_variance_for(&self, n_obs: usize) -> Option<ScalarSamples<B>> {
        self.observation_error_variance()
            .map(|v| vec![v.first().copied().unwrap_or(B::Scalar::ZERO); n_obs].into())
    }

    fn linearize(&self, _snapshot: &PhysicsSnapshot<B>) -> Option<DenseScalarMatrix<B>> {
        None
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReflectanceCalibration {
    pub log_slope: f64,
    pub intercept: f64,
}

impl ReflectanceCalibration {
    pub const fn new(log_slope: f64, intercept: f64) -> Self {
        Self {
            log_slope,
            intercept,
        }
    }
}

pub struct ReflectanceOperator<B: Backend = DefaultBackend> {
    wavelength: B::Scalar,
    log_slope: B::Scalar,
    intercept: B::Scalar,
    observation_std: B::Scalar,
}

impl<B: Backend> ReflectanceOperator<B>
where
    B::Scalar: RuntimeScalar,
{
    /// 创建显式标定的反射率观测算子。
    ///
    /// 本仓库不再提供内置传感器默认系数；调用方必须提供经过场景、
    /// 波段和悬沙标定得到的对数线性系数。
    pub fn new(wavelength: f64, calibration: ReflectanceCalibration, observation_std: f64) -> Self {
        Self {
            wavelength: scalar_from_f64_or_panic::<B>(wavelength, "reflectance.wavelength"),
            log_slope: scalar_from_f64_or_panic::<B>(calibration.log_slope, "reflectance.log_slope"),
            intercept: scalar_from_f64_or_panic::<B>(calibration.intercept, "reflectance.intercept"),
            observation_std: scalar_from_f64_or_panic::<B>(
                observation_std,
                "reflectance.observation_std",
            ),
        }
    }

    pub fn from_coefficients(
        wavelength: f64,
        log_slope: f64,
        intercept: f64,
        observation_std: f64,
    ) -> Self {
        Self::new(
            wavelength,
            ReflectanceCalibration::new(log_slope, intercept),
            observation_std,
        )
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

    fn observe(&self, snapshot: &PhysicsSnapshot<B>) -> ScalarSamples<B> {
        let _ = self.wavelength;
        snapshot
            .sediment
            .as_ref()
            .map(|c| {
                c.iter()
                    .map(|&conc| {
                        let c_safe = conc.max(scalar_from_f64_or_panic::<B>(
                            1e-10,
                            "reflectance.min_concentration",
                        ));
                        self.log_slope * c_safe.ln() + self.intercept
                    })
                    .collect::<Vec<_>>()
                    .into()
            })
            .unwrap_or_else(|| ScalarSamples::from(vec![B::Scalar::ZERO; snapshot.n_cells()]))
    }

    fn observation_error_variance_for(&self, n_obs: usize) -> Option<ScalarSamples<B>> {
        Some(vec![self.observation_std * self.observation_std; n_obs].into())
    }
}

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
    ) -> Result<Self, AiError> {
        if station_indices.iter().any(|idx| idx.get() >= n_cells) {
            return Err(AiError::InvalidObservation(
                "station index out of bounds".into(),
            ));
        }

        Ok(Self {
            station_indices,
            observation_std: scalar_from_f64_or_panic::<B>(
                observation_std,
                "water_level.observation_std",
            ),
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

    fn observe(&self, snapshot: &PhysicsSnapshot<B>) -> ScalarSamples<B> {
        self.station_indices
            .iter()
            .map(|idx| snapshot.h[idx.get()] + snapshot.z[idx.get()])
            .collect::<Vec<_>>()
            .into()
    }

    fn observation_error_variance(&self) -> Option<ScalarSamples<B>> {
        Some(vec![self.observation_std * self.observation_std; self.station_indices.len()].into())
    }
}
