//! AI 代理层 - 遥感驱动的智能预测与同化
//!
//! 该 crate 提供了 AI 增强的水动力建模能力：
//! - 遥感数据反演与融合
//! - 代理模型加速
//! - 数据同化
//!
//! # 设计原则
//!
//! 1. **非侵入**: 不修改 `mh_physics` 物理核心代码
//! 2. **异步解耦**: AI 推理不阻塞物理计算主循环
//! 3. **守恒安全**: AI 注入后自动校验质量/动量守恒
//!
//! # 模块结构
//!
//! - `registry`: AI 代理注册中心，管理多个代理的生命周期
//! - `remote_sensing`: 遥感反演代理，从卫星影像提取水位/流速
//! - `surrogate`: 代理模型，用于快速预测替代物理求解
//! - `observation`: 观测算子，将模型状态映射到观测空间
//! - `assimilation`: 数据同化算法（Nudging, EnKF 等）
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_agent::{AgentRegistry, PhysicsSnapshot, AIAgent};
//!
//! // 创建代理注册中心
//! let mut registry = AgentRegistry::new();
//!
//! // 注册遥感反演代理
//! registry.register(Box::new(RemoteSensingAgent::new(...)));
//!
//! // 物理求解循环中调用
//! let snapshot = PhysicsSnapshot::from(&state);
//! registry.update_all(&snapshot)?;
//! registry.apply_all(&mut state)?;
//! ```

pub mod registry;
pub mod assimilation;

use thiserror::Error;

/// AI 代理层错误类型
#[derive(Error, Debug)]
pub enum AiError {
    /// 模型推理失败
    #[error("模型推理失败: {0}")]
    InferenceFailed(String),
    
    /// 守恒性违反
    #[error("守恒性违反: 期望 {expected:.6e}, 实际 {actual:.6e}")]
    ConservationViolated {
        expected: f64,
        actual: f64,
    },
    
    /// 输入形状不匹配
    #[error("输入形状不匹配: 期望 {expected:?}, 实际 {actual:?}")]
    InvalidShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    
    /// 代理未初始化
    #[error("代理未初始化")]
    NotInitialized,
    
    /// 观测数据无效
    #[error("观测数据无效: {0}")]
    InvalidObservation(String),
    
    /// 物理状态访问错误
    #[error("物理状态访问错误: {0}")]
    StateAccessError(String),
}

/// 物理状态快照（只读，用于 AI 推理）
///
/// 包含当前时刻的物理状态副本，供 AI 代理进行分析和预测。
/// 快照是不可变的，确保 AI 推理不会意外修改物理状态。
#[derive(Debug, Clone)]
pub struct PhysicsSnapshot {
    /// 水深场 [m]
    pub h: Vec<f64>,
    /// x 方向速度场 [m/s]
    pub u: Vec<f64>,
    /// y 方向速度场 [m/s]
    pub v: Vec<f64>,
    /// 床面高程 [m]
    pub z: Vec<f64>,
    /// 泥沙浓度（可选）[kg/m³]
    pub sediment: Option<Vec<f64>>,
    /// 当前模拟时间 [s]
    pub time: f64,
    /// 单元中心坐标 [(x, y), ...]
    pub cell_centers: Vec<[f64; 2]>,
    /// 单元面积 [m²]
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
        self.h.iter()
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

/// AI 代理 Trait
///
/// 所有 AI 代理必须实现此 trait，提供统一的更新和应用接口。
///
/// # 生命周期
///
/// 1. `update()`: 接收物理状态快照，进行 AI 推理/预测
/// 2. `apply()`: 将 AI 结果应用到可同化状态
///
/// # 线程安全
///
/// 代理必须是 `Send + Sync`，支持多线程环境。
pub trait AIAgent: Send + Sync {
    /// 获取代理名称
    fn name(&self) -> &'static str;
    
    /// 更新代理内部状态
    ///
    /// 基于物理状态快照进行 AI 推理或预测。
    /// 此方法应该是计算密集型的，可能涉及神经网络推理。
    ///
    /// # 参数
    ///
    /// - `snapshot`: 当前物理状态的只读快照
    ///
    /// # 返回
    ///
    /// 成功返回 `Ok(())`，失败返回 `AiError`
    fn update(&mut self, snapshot: &PhysicsSnapshot) -> Result<(), AiError>;
    
    /// 应用修正到物理状态
    ///
    /// 将 AI 预测/同化结果应用到可变的物理状态。
    ///
    /// # 参数
    ///
    /// - `state`: 可同化的物理状态
    ///
    /// # 返回
    ///
    /// 成功返回 `Ok(())`，失败返回 `AiError`
    fn apply(&self, state: &mut dyn Assimilable) -> Result<(), AiError>;
    
    /// 是否需要守恒性校验
    ///
    /// 如果返回 `true`，注册中心会在 `apply()` 后检查质量/动量守恒。
    fn requires_conservation_check(&self) -> bool {
        true
    }
    
    /// 获取预测结果（用于可视化）
    fn get_prediction(&self) -> Option<&[f64]> {
        None
    }
    
    /// 获取置信度/不确定性（可选）
    fn get_uncertainty(&self) -> Option<&[f64]> {
        None
    }
}

/// 可同化状态接口
///
/// 物理状态必须实现此 trait 才能接受 AI 代理的修正。
/// 提供对各个物理场的可变访问。
pub trait Assimilable {
    /// 获取示踪剂可变引用
    fn get_tracer_mut(&mut self, name: &str) -> Option<&mut [f64]>;
    
    /// 获取速度场可变引用 (u, v)
    fn get_velocity_mut(&mut self) -> Option<(&mut [f64], &mut [f64])>;
    
    /// 获取水深可变引用
    fn get_depth_mut(&mut self) -> &mut [f64];
    
    /// 获取床面高程可变引用
    fn get_bed_elevation_mut(&mut self) -> &mut [f64];
    
    /// 获取单元数量
    fn n_cells(&self) -> usize;
    
    /// 获取单元面积
    fn cell_areas(&self) -> &[f64];
    
    /// 获取当前总水量（用于守恒校验）
    fn total_water_volume(&self) -> f64 {
        let depth = unsafe { 
            // 安全：我们只是读取
            std::slice::from_raw_parts(
                self.get_depth_mut().as_ptr(),
                self.n_cells()
            )
        };
        depth.iter()
            .zip(self.cell_areas().iter())
            .map(|(&h, &a)| h * a)
            .sum()
    }
}

// 重导出
pub use registry::AgentRegistry;
pub use assimilation::NudgingAssimilator;
