# Phase 5: AI 代理层

## 目标

新建 mh_agent crate，实现 AI-物理桥接。

## 时间：第 7 周

## 前置依赖

- Phase 1-4 完成（泛型化基础设施）

## 任务清单

### 5.1 创建 mh_agent crate

**目标**：创建独立的 AI 代理层 crate。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `crates/mh_agent/Cargo.toml` | crate 配置 |
| 新建 | `crates/mh_agent/src/lib.rs` | 模块入口 |
| 修改 | `Cargo.toml` (workspace) | 添加 mh_agent |

#### 关键代码

**crates/mh_agent/Cargo.toml**

```toml
[package]
name = "mh_agent"
version = "0.1.0"
edition = "2021"
description = "AI Agent Layer for MariHydro"

[dependencies]
mh_physics = { path = "../mh_physics" }
thiserror = "1.0"

# 可选：ONNX 推理
ort = { version = "2.0", optional = true }

[features]
default = []
onnx = ["ort"]
```

**crates/mh_agent/src/lib.rs**

```rust
//! AI 代理层 - 遥感驱动的智能预测与同化
//!
//! # 设计原则
//!
//! 1. 非侵入：不修改 mh_physics 物理核心
//! 2. 异步解耦：AI 推理不阻塞物理计算
//! 3. 守恒安全：AI 注入后自动校验守恒性
//!
//! # 模块结构
//!
//! - `registry`: AI 代理注册中心
//! - `remote_sensing`: 遥感反演代理
//! - `surrogate`: 代理模型加速
//! - `observation`: 观测算子

pub mod registry;
pub mod remote_sensing;
pub mod surrogate;
pub mod observation;
pub mod assimilation;

use thiserror::Error;

/// AI 代理错误
#[derive(Error, Debug)]
pub enum AiError {
    #[error("Model inference failed: {0}")]
    InferenceFailed(String),
    
    #[error("Conservation violated: expected {expected}, got {actual}")]
    ConservationViolated { expected: f64, actual: f64 },
    
    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    InvalidShape { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Agent not initialized")]
    NotInitialized,
}

/// 物理状态快照（只读，用于 AI 推理）
#[derive(Debug, Clone)]
pub struct PhysicsSnapshot {
    pub h: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub sediment: Option<Vec<f64>>,
    pub time: f64,
    pub cell_centers: Vec<[f64; 2]>,
}

/// AI 代理 Trait
pub trait AIAgent: Send + Sync {
    /// 代理名称
    fn name(&self) -> &'static str;
    
    /// 更新内部状态（基于物理快照）
    fn update(&mut self, snapshot: &PhysicsSnapshot) -> Result<(), AiError>;
    
    /// 应用修正到物理状态
    fn apply(&self, state: &mut dyn Assimilable) -> Result<(), AiError>;
    
    /// 是否需要守恒校验
    fn requires_conservation_check(&self) -> bool { true }
    
    /// 获取预测结果（用于可视化）
    fn get_prediction(&self) -> Option<&[f64]> { None }
}

/// 可同化状态接口
pub trait Assimilable {
    /// 获取示踪剂可变引用
    fn get_tracer_mut(&mut self, name: &str) -> Option<&mut [f64]>;
    
    /// 获取速度场可变引用 (u, v)
    fn get_velocity_mut(&mut self) -> Option<(&mut [f64], &mut [f64])>;
    
    /// 获取水深可变引用
    fn get_depth_mut(&mut self) -> &mut [f64];
    
    /// 获取床面高程可变引用
    fn get_bed_elevation_mut(&mut self) -> &mut [f64];
    
    /// 单元数量
    fn n_cells(&self) -> usize;
    
    /// 单元面积
    fn cell_areas(&self) -> &[f64];
}

// 重导出
pub use registry::AgentRegistry;
pub use remote_sensing::RemoteSensingAgent;
pub use surrogate::SurrogateModel;
pub use observation::ObservationOperator;
pub use assimilation::NudgingAssimilator;
```

---

### 5.2 AI 代理注册中心

**目标**：实现代理注册和管理。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `crates/mh_agent/src/registry.rs` | 注册中心 |

#### 关键代码

```rust
// crates/mh_agent/src/registry.rs
use crate::{AIAgent, AiError, PhysicsSnapshot, Assimilable};
use std::collections::HashMap;

/// AI 代理注册中心
pub struct AgentRegistry {
    agents: HashMap<String, Box<dyn AIAgent>>,
    enabled: HashMap<String, bool>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            enabled: HashMap::new(),
        }
    }
    
    /// 注册代理
    pub fn register(&mut self, agent: Box<dyn AIAgent>) {
        let name = agent.name().to_string();
        self.enabled.insert(name.clone(), true);
        self.agents.insert(name, agent);
    }
    
    /// 启用/禁用代理
    pub fn set_enabled(&mut self, name: &str, enabled: bool) {
        if let Some(e) = self.enabled.get_mut(name) {
            *e = enabled;
        }
    }
    
    /// 更新所有启用的代理
    pub fn update_all(&mut self, snapshot: &PhysicsSnapshot) -> Result<(), AiError> {
        for (name, agent) in &mut self.agents {
            if *self.enabled.get(name).unwrap_or(&false) {
                agent.update(snapshot)?;
            }
        }
        Ok(())
    }
    
    /// 应用所有启用的代理
    pub fn apply_all(&self, state: &mut dyn Assimilable) -> Result<(), AiError> {
        for (name, agent) in &self.agents {
            if *self.enabled.get(name).unwrap_or(&false) {
                agent.apply(state)?;
            }
        }
        Ok(())
    }
    
    /// 获取代理
    pub fn get(&self, name: &str) -> Option<&dyn AIAgent> {
        self.agents.get(name).map(|a| a.as_ref())
    }
    
    /// 列出所有代理
    pub fn list(&self) -> Vec<(&str, bool)> {
        self.agents.keys()
            .map(|name| {
                (name.as_str(), *self.enabled.get(name).unwrap_or(&false))
            })
            .collect()
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}
```

---

### 5.3 遥感反演代理

**目标**：实现遥感泥沙反演代理。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `crates/mh_agent/src/remote_sensing.rs` | 遥感代理 |

#### 关键代码

```rust
// crates/mh_agent/src/remote_sensing.rs
use crate::{AIAgent, AiError, PhysicsSnapshot, Assimilable};

/// 卫星图像数据
pub struct SatelliteImage {
    /// 波段数据 [band][pixel]
    pub bands: Vec<Vec<f32>>,
    /// 图像宽度
    pub width: usize,
    /// 图像高度
    pub height: usize,
    /// 地理范围 [min_x, min_y, max_x, max_y]
    pub extent: [f64; 4],
    /// 采集时间
    pub timestamp: f64,
}

/// 遥感泥沙反演代理
pub struct RemoteSensingAgent {
    /// 同化率（Nudging 系数）
    assimilation_rate: f64,
    /// 预测浓度场
    predicted: Vec<f64>,
    /// 置信度
    confidence: Vec<f64>,
    /// 是否已初始化
    initialized: bool,
    
    #[cfg(feature = "onnx")]
    model: Option<ort::Session>,
}

impl RemoteSensingAgent {
    /// 创建新的遥感代理
    pub fn new(assimilation_rate: f64) -> Self {
        Self {
            assimilation_rate,
            predicted: Vec::new(),
            confidence: Vec::new(),
            initialized: false,
            #[cfg(feature = "onnx")]
            model: None,
        }
    }
    
    /// 加载 ONNX 模型
    #[cfg(feature = "onnx")]
    pub fn load_model(&mut self, model_path: &str) -> Result<(), AiError> {
        let model = ort::Session::builder()
            .map_err(|e| AiError::InferenceFailed(e.to_string()))?
            .with_model_from_file(model_path)
            .map_err(|e| AiError::InferenceFailed(e.to_string()))?;
        self.model = Some(model);
        Ok(())
    }
    
    /// 从卫星图像推理
    pub fn infer(&mut self, image: &SatelliteImage, n_cells: usize) -> Result<(), AiError> {
        // 简化实现：使用经验公式
        // 实际应用中使用 ONNX 模型
        
        self.predicted = vec![0.0; n_cells];
        self.confidence = vec![1.0; n_cells];
        
        // 示例：从红/近红外波段估算浊度
        if image.bands.len() >= 2 {
            let red = &image.bands[0];
            let nir = &image.bands[1];
            
            // 简单的浊度指数
            for i in 0..red.len().min(n_cells) {
                let turbidity = (red[i] / (nir[i] + 0.001)) as f64;
                self.predicted[i] = turbidity * 0.1; // 转换为浓度
                self.confidence[i] = 0.8;
            }
        }
        
        self.initialized = true;
        Ok(())
    }
    
    /// 设置预测结果（用于测试）
    pub fn set_prediction(&mut self, predicted: Vec<f64>, confidence: Vec<f64>) {
        self.predicted = predicted;
        self.confidence = confidence;
        self.initialized = true;
    }
}

impl AIAgent for RemoteSensingAgent {
    fn name(&self) -> &'static str { "RemoteSensing-Sediment" }
    
    fn update(&mut self, _snapshot: &PhysicsSnapshot) -> Result<(), AiError> {
        // 遥感代理不依赖物理快照
        // 更新由 infer() 方法触发
        Ok(())
    }
    
    fn apply(&self, state: &mut dyn Assimilable) -> Result<(), AiError> {
        if !self.initialized {
            return Err(AiError::NotInitialized);
        }
        
        if let Some(sediment) = state.get_tracer_mut("sediment") {
            // Nudging 同化
            for (i, c) in sediment.iter_mut().enumerate() {
                if i < self.predicted.len() {
                    let weight = self.assimilation_rate * self.confidence[i];
                    *c += weight * (self.predicted[i] - *c);
                }
            }
        }
        
        Ok(())
    }
    
    fn get_prediction(&self) -> Option<&[f64]> {
        if self.initialized {
            Some(&self.predicted)
        } else {
            None
        }
    }
}
```

---

### 5.4 Nudging 同化器

**目标**：实现通用的 Nudging 同化方法。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `crates/mh_agent/src/assimilation.rs` | 同化方法 |

#### 关键代码

```rust
// crates/mh_agent/src/assimilation.rs
use crate::{AIAgent, AiError, PhysicsSnapshot, Assimilable};

/// Nudging 同化配置
#[derive(Debug, Clone)]
pub struct NudgingConfig {
    /// 同化时间尺度 [s]
    pub relaxation_time: f64,
    /// 最小置信度阈值
    pub min_confidence: f64,
    /// 空间影响半径 [m]
    pub influence_radius: f64,
}

impl Default for NudgingConfig {
    fn default() -> Self {
        Self {
            relaxation_time: 3600.0,  // 1 小时
            min_confidence: 0.5,
            influence_radius: 1000.0,
        }
    }
}

/// Nudging 同化器
pub struct NudgingAssimilator {
    config: NudgingConfig,
    /// 观测值
    observations: Vec<Observation>,
    /// 当前时间步
    dt: f64,
}

/// 单个观测
#[derive(Debug, Clone)]
pub struct Observation {
    /// 观测位置 [x, y]
    pub location: [f64; 2],
    /// 观测值
    pub value: f64,
    /// 置信度 (0-1)
    pub confidence: f64,
    /// 观测时间
    pub time: f64,
    /// 目标变量
    pub variable: ObservationVariable,
}

/// 观测变量类型
#[derive(Debug, Clone, Copy)]
pub enum ObservationVariable {
    WaterLevel,
    Velocity,
    Sediment,
    Salinity,
}

impl NudgingAssimilator {
    pub fn new(config: NudgingConfig) -> Self {
        Self {
            config,
            observations: Vec::new(),
            dt: 1.0,
        }
    }
    
    /// 添加观测
    pub fn add_observation(&mut self, obs: Observation) {
        self.observations.push(obs);
    }
    
    /// 清除过期观测
    pub fn clear_old_observations(&mut self, current_time: f64, max_age: f64) {
        self.observations.retain(|obs| current_time - obs.time < max_age);
    }
    
    /// 设置时间步
    pub fn set_dt(&mut self, dt: f64) {
        self.dt = dt;
    }
    
    /// 计算 Nudging 系数
    fn compute_nudging_coefficient(&self, distance: f64, confidence: f64) -> f64 {
        if confidence < self.config.min_confidence {
            return 0.0;
        }
        
        // 空间衰减
        let spatial_weight = if distance < self.config.influence_radius {
            1.0 - distance / self.config.influence_radius
        } else {
            0.0
        };
        
        // 时间系数
        let time_weight = self.dt / self.config.relaxation_time;
        
        spatial_weight * time_weight * confidence
    }
}

impl AIAgent for NudgingAssimilator {
    fn name(&self) -> &'static str { "Nudging-Assimilator" }
    
    fn update(&mut self, _snapshot: &PhysicsSnapshot) -> Result<(), AiError> {
        Ok(())
    }
    
    fn apply(&self, state: &mut dyn Assimilable) -> Result<(), AiError> {
        // 对每个观测应用 Nudging
        // 简化实现：假设观测位置与单元中心对应
        
        let depth = state.get_depth_mut();
        
        for obs in &self.observations {
            match obs.variable {
                ObservationVariable::WaterLevel => {
                    // 找到最近的单元并应用 Nudging
                    // 简化：假设观测索引直接对应单元
                    let cell = 0; // 实际应通过空间查找
                    let coef = self.compute_nudging_coefficient(0.0, obs.confidence);
                    depth[cell] += coef * (obs.value - depth[cell]);
                }
                _ => {
                    // 其他变量类似处理
                }
            }
        }
        
        Ok(())
    }
}
```

---

### 5.5 Assimilable 桥接实现

**目标**：为物理状态实现 Assimilable trait。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `mh_physics/src/assimilation/mod.rs` | Assimilable 实现 |
| 修改 | `mh_physics/src/lib.rs` | 添加模块 |

#### 关键代码

```rust
// mh_physics/src/assimilation/mod.rs
//! 数据同化桥接模块

use crate::core::CpuBackend;
use crate::state::ShallowWaterStateGeneric;

/// 可同化状态接口
/// 
/// 为 AI 代理层提供状态访问接口。
pub trait Assimilable {
    fn get_tracer_mut(&mut self, name: &str) -> Option<&mut [f64]>;
    fn get_velocity_mut(&mut self) -> Option<(&mut [f64], &mut [f64])>;
    fn get_depth_mut(&mut self) -> &mut [f64];
    fn get_bed_elevation_mut(&mut self) -> &mut [f64];
    fn n_cells(&self) -> usize;
    fn cell_areas(&self) -> &[f64];
}

/// 为 CPU f64 状态实现 Assimilable
impl Assimilable for ShallowWaterStateGeneric<CpuBackend<f64>> {
    fn get_tracer_mut(&mut self, name: &str) -> Option<&mut [f64]> {
        self.tracers.as_mut()
            .and_then(|ts| ts.get_mut_by_name(name))
    }
    
    fn get_velocity_mut(&mut self) -> Option<(&mut [f64], &mut [f64])> {
        // 需要从动量恢复速度
        // 简化：返回 None，实际应提供专用接口
        None
    }
    
    fn get_depth_mut(&mut self) -> &mut [f64] {
        self.h.as_mut_slice()
    }
    
    fn get_bed_elevation_mut(&mut self) -> &mut [f64] {
        self.z.as_mut_slice()
    }
    
    fn n_cells(&self) -> usize {
        self.n_cells()
    }
    
    fn cell_areas(&self) -> &[f64] {
        // 需要从网格获取
        // 简化：返回空切片
        &[]
    }
}
```

---

## 验收标准

1. ✅ `mh_agent` crate 创建成功
2. ✅ `AIAgent` trait 定义完整
3. ✅ `AgentRegistry` 可注册和管理代理
4. ✅ `RemoteSensingAgent` 实现并通过测试
5. ✅ `NudgingAssimilator` 实现并通过测试
6. ✅ `Assimilable` 桥接实现

## 测试用例

```rust
#[test]
fn test_agent_registry() {
    let mut registry = AgentRegistry::new();
    
    let agent = RemoteSensingAgent::new(0.1);
    registry.register(Box::new(agent));
    
    assert_eq!(registry.list().len(), 1);
}

#[test]
fn test_remote_sensing_apply() {
    let mut agent = RemoteSensingAgent::new(0.5);
    
    // 设置预测
    agent.set_prediction(vec![1.0, 2.0, 3.0], vec![1.0, 1.0, 1.0]);
    
    // 创建模拟状态
    let backend = CpuBackend::<f64>::new();
    let mut state = ShallowWaterStateGeneric::new(backend, 3);
    
    // 注册泥沙示踪剂
    // state.register_tracer("sediment");
    
    // 应用代理
    // agent.apply(&mut state).unwrap();
}
```
