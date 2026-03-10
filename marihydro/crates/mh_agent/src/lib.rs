//! AI 代理层：遥感驱动的预测与同化。

pub mod registry;
pub mod assimilation;
pub mod remote_sensing;
pub mod observation;
pub mod surrogate;

use thiserror::Error;

/// AI 代理层错误类型
#[derive(Error, Debug)]
pub enum AiError {
    /// 模型推理失败
    #[error("模型推理失败: {0}")]
    InferenceFailed(String),

    /// 守恒性违反
    #[error("守恒性违反: 期望 {expected:.6e}, 实际 {actual:.6e}")]
    ConservationViolated { expected: f64, actual: f64 },

    /// 输入形状不匹配
    #[error("输入形状不匹配: 期望 {expected:?}, 实际 {actual:?}")]
    InvalidShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// 代理未初始化
    #[error("代理未初始化")]
    NotInitialized,

    /// 代理未就绪
    #[error("代理未就绪: {0}")]
    NotReady(String),

    /// 观测数据无效
    #[error("观测数据无效: {0}")]
    InvalidObservation(String),

    /// 状态访问错误
    #[error("状态访问错误: {0}")]
    StateAccessError(String),
}

/// 物理状态快照（只读）
#[derive(Debug, Clone)]
pub struct PhysicsSnapshot {
    /// 水深场 [m]
    pub h: Vec<f64>,
    /// x 方向速度 [m/s]
    pub u: Vec<f64>,
    /// y 方向速度 [m/s]
    pub v: Vec<f64>,
    /// 床面高程 [m]
    pub z: Vec<f64>,
    /// 泥沙浓度（可选）[kg/m^3]
    pub sediment: Option<Vec<f64>>,
    /// 当前模拟时间 [s]
    pub time: f64,
    /// 单元中心坐标 [(x, y), ...]
    pub cell_centers: Vec<[f64; 2]>,
    /// 单元面积 [m^2]
    pub cell_areas: Vec<f64>,
}

impl PhysicsSnapshot {
    /// 创建空快照
    pub fn empty(n_cells: usize) -> Self {
        Self {
            h: vec![0.0; n_cells],
            u: vec![0.0; n_cells],
            v: vec![0.0; n_cells],
            z: vec![0.0; n_cells],
            sediment: None,
            time: 0.0,
            cell_centers: vec![[0.0, 0.0]; n_cells],
            cell_areas: vec![1.0; n_cells],
        }
    }

    /// 单元数量
    pub fn n_cells(&self) -> usize {
        self.h.len()
    }

    /// 计算总水量
    pub fn total_water_volume(&self) -> f64 {
        self.h
            .iter()
            .zip(self.cell_areas.iter())
            .map(|(&h, &a)| h * a)
            .sum()
    }

    /// 计算总动量
    pub fn total_momentum(&self) -> (f64, f64) {
        let mut mx = 0.0;
        let mut my = 0.0;
        for i in 0..self.n_cells() {
            mx += self.h[i] * self.u[i] * self.cell_areas[i];
            my += self.h[i] * self.v[i] * self.cell_areas[i];
        }
        (mx, my)
    }
}

/// AI 代理 trait
pub trait AIAgent: Send + Sync {
    type State: Assimilable;

    /// 获取代理名称
    fn name(&self) -> &'static str;

    /// 基于当前快照更新内部预测
    fn update(&mut self, snapshot: &PhysicsSnapshot) -> Result<(), AiError>;

    /// 应用修正到可同化状态
    fn apply(&self, state: &mut Self::State) -> Result<(), AiError>;

    /// 是否需要守恒性检查
    fn requires_conservation_check(&self) -> bool {
        true
    }

    /// 获取预测结果（用于可视化）
    fn get_prediction(&self) -> Option<&[f64]> {
        None
    }

    /// 获取不确定性（可选）
    fn get_uncertainty(&self) -> Option<&[f64]> {
        None
    }
}

/// 可同化状态接口
pub trait Assimilable {
    /// 获取指定示踪剂可变切片
    fn get_tracer_mut(&mut self, name: &str) -> Option<&mut [f64]>;

    /// 获取速度场可变切片 (u, v)
    fn get_velocity_mut(&mut self) -> Option<(&mut [f64], &mut [f64])>;

    /// 获取水深可变切片
    fn get_depth_mut(&mut self) -> &mut [f64];

    /// 获取水深只读切片
    fn get_depth(&self) -> &[f64];

    /// 获取床面高程可变切片
    fn get_bed_elevation_mut(&mut self) -> &mut [f64];

    /// 获取单元数量
    fn n_cells(&self) -> usize;

    /// 获取单元面积
    fn cell_areas(&self) -> &[f64];

    /// 获取当前总水量（用于守恒校验）
    fn total_water_volume(&self) -> f64 {
        self.get_depth()
            .iter()
            .zip(self.cell_areas().iter())
            .map(|(&h, &a)| h * a)
            .sum()
    }
}

// 重导出
pub use registry::AgentRegistry;
pub use assimilation::{NudgingAssimilator, NudgingConfig, Observation, AssimilationResult};
pub use remote_sensing::{RemoteSensingAgent, RemoteSensingConfig, SatelliteImage, SensorType};
pub use observation::{ObservationOperator, ReflectanceOperator, SAROperator, WaterLevelOperator};
pub use surrogate::{SurrogateModel, SurrogateConfig, SurrogateType, SurrogatePrediction};
