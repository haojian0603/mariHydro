//! AI 代理与同化接口。
//!
//! 这里集中定义 `mh_agent` 对外提供的核心类型：
//! - AI 代理错误类型
//! - 物理状态快照
//! - AI 代理与可同化状态的抽象接口

pub mod assimilation;
pub mod observation;
pub mod registry;
pub mod remote_sensing;
pub mod surrogate;

use bytemuck::Pod;
use mh_runtime::prelude::{Float, FromPrimitive};
use mh_runtime::{Backend, CpuBackend, RuntimeScalar};
use std::ops::Deref;
use thiserror::Error;

/// 默认后端。
pub type DefaultBackend = CpuBackend<f64>;

pub(crate) fn scalar_from_f64_or_panic<B: Backend>(value: f64, context: &'static str) -> B::Scalar
where
    B::Scalar: RuntimeScalar,
{
    B::Scalar::from_f64(value)
        .unwrap_or_else(|| panic!("failed to convert agent scalar for {context}: {value}"))
}

pub(crate) fn scalar_from_f32_or_panic<B: Backend>(value: f32, context: &'static str) -> B::Scalar
where
    B::Scalar: RuntimeScalar,
{
    scalar_from_f64_or_panic::<B>(value as f64, context)
}

/// AI 代理错误类型。
#[derive(Error, Debug)]
pub enum AiError {
    /// 推理失败。
    #[error("推理失败: {0}")]
    InferenceFailed(String),

    /// 守恒性被破坏。
    #[error("守恒性被破坏: 期望 {expected:.6e}, 实际 {actual:.6e}")]
    ConservationViolated { expected: f64, actual: f64 },

    /// 数据形状不一致。
    #[error("形状无效: 期望 {expected:?}, 实际 {actual:?}")]
    InvalidShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// 尚未初始化。
    #[error("尚未初始化")]
    NotInitialized,

    /// 尚未就绪。
    #[error("暂未就绪: {0}")]
    NotReady(String),

    /// 观测数据无效。
    #[error("观测无效: {0}")]
    InvalidObservation(String),

    #[error("代理模型类型不受支持: {0}")]
    UnsupportedModelType(String),

    /// 状态访问失败。
    #[error("状态访问错误: {0}")]
    StateAccessError(String),

    /// 其他错误。
    #[error("{0}")]
    Other(String),
}

/// AI 模块公用数据序列。
#[derive(Debug, Clone)]
pub struct ScalarSamples<B: Backend> {
    data: Vec<B::Scalar>,
}

impl<B: Backend> ScalarSamples<B> {
    pub fn new(data: Vec<B::Scalar>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[B::Scalar] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [B::Scalar] {
        &mut self.data
    }

    pub fn into_vec(self) -> Vec<B::Scalar> {
        self.data
    }

    pub fn iter(&self) -> std::slice::Iter<'_, B::Scalar> {
        self.data.iter()
    }
}

impl<B: Backend> Deref for ScalarSamples<B> {
    type Target = [B::Scalar];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<B: Backend> From<Vec<B::Scalar>> for ScalarSamples<B> {
    fn from(value: Vec<B::Scalar>) -> Self {
        Self::new(value)
    }
}

impl<B: Backend> Default for ScalarSamples<B> {
    fn default() -> Self {
        Self { data: Vec::new() }
    }
}

#[derive(Debug, Clone)]
pub struct DenseScalarMatrix<B: Backend> {
    rows: Vec<ScalarSamples<B>>,
    ncols: usize,
}

impl<B: Backend> DenseScalarMatrix<B> {
    pub fn empty() -> Self {
        Self {
            rows: Vec::new(),
            ncols: 0,
        }
    }

    pub fn try_new(rows: Vec<ScalarSamples<B>>) -> Result<Self, AiError> {
        let ncols = rows.first().map(ScalarSamples::len).unwrap_or(0);
        let mut actual = Vec::with_capacity(rows.len());
        let mut valid = true;
        for row in &rows {
            let len = row.len();
            actual.push(len);
            if len != ncols {
                valid = false;
            }
        }
        if !valid {
            return Err(AiError::InvalidShape {
                expected: vec![ncols],
                actual,
            });
        }
        Ok(Self { rows, ncols })
    }

    pub fn nrows(&self) -> usize {
        self.rows.len()
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn row(&self, index: usize) -> Option<&ScalarSamples<B>> {
        self.rows.get(index)
    }

    pub fn rows(&self) -> &[ScalarSamples<B>] {
        &self.rows
    }

    pub fn iter_rows(&self) -> std::slice::Iter<'_, ScalarSamples<B>> {
        self.rows.iter()
    }
}

impl<B: Backend> Default for DenseScalarMatrix<B> {
    fn default() -> Self {
        Self::empty()
    }
}

pub type VelocityBuffersMut<'a, B> = (
    &'a mut <B as Backend>::Buffer<<B as Backend>::Scalar>,
    &'a mut <B as Backend>::Buffer<<B as Backend>::Scalar>,
);

#[derive(Clone)]
pub struct PhysicsSnapshot<B: Backend = DefaultBackend>
where
    B::Vector2D: Pod,
{
    /// 水深。
    pub h: B::Buffer<B::Scalar>,
    /// x 向速度。
    pub u: B::Buffer<B::Scalar>,
    /// y 向速度。
    pub v: B::Buffer<B::Scalar>,
    /// 床面高程。
    pub z: B::Buffer<B::Scalar>,
    /// 泥沙浓度。
    pub sediment: Option<B::Buffer<B::Scalar>>,
    /// 时间。
    pub time: B::Scalar,
    /// 单元中心坐标。
    pub cell_centers: B::Buffer<B::Vector2D>,
    /// 单元面积。
    pub cell_areas: B::Buffer<B::Scalar>,
}

impl<B: Backend> PhysicsSnapshot<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: Pod + Default,
{
    /// 创建空快照。
    pub fn empty(backend: &B, n_cells: usize) -> Self {
        Self {
            h: backend.alloc_init(n_cells, B::Scalar::ZERO),
            u: backend.alloc_init(n_cells, B::Scalar::ZERO),
            v: backend.alloc_init(n_cells, B::Scalar::ZERO),
            z: backend.alloc_init(n_cells, B::Scalar::ZERO),
            sediment: None,
            time: B::Scalar::ZERO,
            cell_centers: backend.alloc_init(n_cells, B::Vector2D::default()),
            cell_areas: backend.alloc_init(n_cells, B::Scalar::ONE),
        }
    }

    /// 校验后构造快照。
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        h: B::Buffer<B::Scalar>,
        u: B::Buffer<B::Scalar>,
        v: B::Buffer<B::Scalar>,
        z: B::Buffer<B::Scalar>,
        sediment: Option<B::Buffer<B::Scalar>>,
        time: B::Scalar,
        cell_centers: B::Buffer<B::Vector2D>,
        cell_areas: B::Buffer<B::Scalar>,
    ) -> Result<Self, AiError> {
        let n = h.len();
        if u.len() != n
            || v.len() != n
            || z.len() != n
            || cell_centers.len() != n
            || cell_areas.len() != n
            || sediment.as_ref().is_some_and(|s| s.len() != n)
        {
            return Err(AiError::InvalidShape {
                expected: vec![n],
                actual: vec![],
            });
        }

        if !time.is_finite() {
            return Err(AiError::InvalidObservation(
                "snapshot time is not finite".into(),
            ));
        }

        Ok(Self {
            h,
            u,
            v,
            z,
            sediment,
            time,
            cell_centers,
            cell_areas,
        })
    }
}

impl<B: Backend> PhysicsSnapshot<B>
where
    B::Vector2D: Pod,
{
    /// 单元数量。
    pub fn n_cells(&self) -> usize {
        self.h.len()
    }

    /// 总水量。
    pub fn total_water_volume(&self) -> B::Scalar {
        self.h
            .iter()
            .zip(self.cell_areas.iter())
            .map(|(&h, &a)| h * a)
            .sum()
    }

    /// 总动量。
    pub fn total_momentum(&self) -> (B::Scalar, B::Scalar) {
        let mut mx = B::Scalar::ZERO;
        let mut my = B::Scalar::ZERO;
        for i in 0..self.n_cells() {
            let h = self.h[i];
            let area = self.cell_areas[i];
            mx += h * self.u[i] * area;
            my += h * self.v[i] * area;
        }
        (mx, my)
    }
}

/// AI 代理抽象接口。
pub trait AIAgent<B: Backend = DefaultBackend>: Send + Sync
where
    B::Vector2D: Pod,
{
    /// 代理名称。
    fn name(&self) -> &'static str;

    /// 用最新快照更新内部状态。
    fn update(&mut self, snapshot: &PhysicsSnapshot<B>) -> Result<(), AiError>;

    /// 将代理结果写回可同化状态。
    fn apply(&self, state: &mut dyn Assimilable<B>) -> Result<(), AiError>;

    /// 是否需要做守恒性检查。
    fn requires_conservation_check(&self) -> bool {
        true
    }

    /// 获取预测结果。
    fn get_prediction(&self) -> Option<&ScalarSamples<B>> {
        None
    }

    /// 获取不确定度。
    fn get_uncertainty(&self) -> Option<&ScalarSamples<B>> {
        None
    }
}

/// 可被 AI 同化的状态接口。
pub trait Assimilable<B: Backend = DefaultBackend>
where
    B::Vector2D: Pod,
{
    /// 返回后端。
    fn backend(&self) -> &B;

    /// 获取示踪量的可变缓冲区。
    fn get_tracer_mut(&mut self, name: &str) -> Option<&mut B::Buffer<B::Scalar>>;

    /// 获取速度分量的可变缓冲区。
    fn get_velocity_mut(&mut self) -> Option<VelocityBuffersMut<'_, B>>;

    /// 获取水深缓冲区。
    fn get_depth(&self) -> &B::Buffer<B::Scalar>;

    /// 获取可变水深缓冲区。
    fn get_depth_mut(&mut self) -> &mut B::Buffer<B::Scalar>;

    /// 获取床面高程缓冲区。
    fn get_bed_elevation_mut(&mut self) -> &mut B::Buffer<B::Scalar>;

    /// 单元数量。
    fn n_cells(&self) -> usize;

    /// 获取单元面积。
    fn cell_areas(&self) -> &B::Buffer<B::Scalar>;

    /// 获取单元中心。
    fn cell_centers(&self) -> &B::Buffer<B::Vector2D>;

    /// 计算总水量。
    fn total_water_volume(&self) -> B::Scalar {
        self.get_depth()
            .iter()
            .zip(self.cell_areas().iter())
            .map(|(&h, &a)| h * a)
            .sum()
    }
}

/// 重新导出常用类型。
pub use assimilation::{AssimilationResult, NudgingAssimilator, NudgingConfig, Observation};
pub use observation::{
    ObservationOperator, Polarization, ReflectanceCalibration, ReflectanceOperator, SAROperator,
    WaterLevelOperator,
};
pub use registry::AgentRegistry;
pub use remote_sensing::{
    ImageBounds, InferenceResult, InterpolationMethod, RemoteSensingAgent, RemoteSensingConfig,
    SatelliteImage, SensorType,
};
pub use surrogate::{
    PredictionMetrics, SurrogateConfig, SurrogateModel, SurrogatePrediction, SurrogateType,
};
