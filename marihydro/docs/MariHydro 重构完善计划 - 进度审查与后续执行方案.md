# MariHydro 重构完善计划 - 进度审查与后续执行方案

## 一、当前状态诊断

基于KIMI审查报告和当前文件结构，我进行综合诊断：

### 1.1 文件存在性审查

| 模块 | 计划文件 | 实际状态 | 问题 |
|------|----------|----------|------|
| **核心抽象** | `core/scalar.rs` | ✅ 存在 | - |
| **核心抽象** | `core/backend.rs` | ✅ 存在 | 需确认实例方法 |
| **核心抽象** | `core/buffer.rs` | ✅ 存在 | - |
| **核心抽象** | `core/gpu.rs` | ✅ 存在 | 占位实现 |
| **核心抽象** | `core/kernel.rs` | ✅ 存在 | - |
| **核心抽象** | `core/hybrid.rs` | ❌ **缺失** | 需新建 |
| **状态** | 泛型化 | ⚠️ 不完整 | 仍硬编码f64 |
| **源项** | `sources/registry.rs` | ❌ **缺失** | 需新建 |
| **源项** | 双轨制 | ⚠️ 债务 | 需合并 |
| **泥沙** | `sediment/manager.rs` | ✅ 存在 | 需确认完整性 |
| **泥沙** | `sediment/exchange.rs` | ❌ **缺失** | 需新建 |
| **示踪剂** | `tracer/settling.rs` | ❌ **缺失** | 需新建 |
| **AI代理** | `mh_agent/assimilation.rs` | ❌ **缺失** | 🔴 编译阻塞！ |
| **AI代理** | `mh_agent/remote_sensing.rs` | ❌ **缺失** | 需新建 |
| **AI代理** | `mh_agent/observation.rs` | ❌ **缺失** | 需新建 |
| **AI代理** | `mh_agent/surrogate.rs` | ❌ **缺失** | 需新建 |
| **桥接** | `mh_physics/src/assimilation/` | ❌ **缺失** | 整个目录不存在 |

### 1.2 严重问题清单

| 优先级 | 问题 | 影响 | 所在Phase |
|--------|------|------|-----------|
| 🔴 P0 | `assimilation.rs`缺失但被引用 | **编译失败** | Phase 5 |
| 🟠 P1 | 状态未完全泛型化 | 阻塞后续泛型化 | Phase 1 |
| 🟠 P1 | `sources/registry.rs`缺失 | 源项无法统一管理 | Phase 3 |
| 🟡 P2 | 双轨制代码并存 | 维护成本高 | Phase 3 |
| 🟡 P2 | `exchange.rs`缺失 | 泥沙系统不完整 | Phase 4 |
| 🟡 P2 | `settling.rs`缺失 | 沉降隐式不可用 | Phase 3 |
| 🟢 P3 | AI代理层不完整 | AI功能缺失 | Phase 5 |
| 🟢 P3 | 桥接层缺失 | AI-物理无法交互 | Phase 5 |

---

## 二、后续执行计划

### 执行原则

1. **先修复编译阻塞**：立即创建缺失文件
2. **完善核心架构**：补全Phase 1-4的缺失部分
3. **消除技术债务**：合并双轨制代码
4. **完善扩展层**：补全AI代理和桥接
5. **不允许简化**：每个文件必须完整实现

---

## Phase R0: 紧急修复（编译阻塞）

**目标**：使项目恢复可编译状态

### R0.1 创建 `mh_agent/src/assimilation.rs`

**文件路径**：`crates/mh_agent/src/assimilation.rs`

**必须包含的内容**：

```rust
// 必须实现的结构体和trait

/// Nudging同化配置
pub struct NudgingConfig {
    /// 同化率 (0.0 - 1.0)
    pub rate: f64,
    /// 最大修正量限制
    pub max_correction: f64,
    /// 空间平滑半径
    pub smoothing_radius: Option<f64>,
    /// 时间衰减系数
    pub temporal_decay: f64,
}

/// 观测数据结构
pub struct Observation {
    /// 观测值
    pub values: Vec<f64>,
    /// 观测位置索引
    pub cell_indices: Vec<usize>,
    /// 观测不确定性
    pub uncertainty: Vec<f64>,
    /// 观测时间
    pub time: f64,
}

/// Nudging同化器
pub struct NudgingAssimilator {
    config: NudgingConfig,
    /// 上次同化时间
    last_assimilation_time: f64,
    /// 累积修正量统计
    cumulative_correction: f64,
}

impl NudgingAssimilator {
    pub fn new(config: NudgingConfig) -> Self;
    
    /// 执行Nudging同化
    pub fn assimilate(
        &mut self,
        state: &mut dyn Assimilable,
        observation: &Observation,
        current_time: f64,
    ) -> Result<AssimilationResult, AiError>;
    
    /// 计算单点修正量
    fn compute_correction(&self, simulated: f64, observed: f64, uncertainty: f64) -> f64;
    
    /// 应用空间平滑
    fn apply_smoothing(&self, corrections: &mut [f64], cell_centers: &[[f64; 2]]);
}

/// 同化结果
pub struct AssimilationResult {
    pub cells_modified: usize,
    pub total_correction: f64,
    pub max_correction: f64,
    pub conservation_error: f64,
}

impl AIAgent for NudgingAssimilator {
    fn name(&self) -> &'static str { "Nudging-Assimilator" }
    fn update(&mut self, snapshot: &PhysicsSnapshot) -> Result<(), AiError>;
    fn apply(&self, state: &mut dyn Assimilable) -> Result<(), AiError>;
}
```

**验证命令**：
```bash
cargo check -p mh_agent
```

---

## Phase R1: 核心架构补全

**目标**：完善Phase 1-4中的缺失核心组件

### R1.1 创建 `mh_physics/src/sources/registry.rs`

**文件路径**：`crates/mh_physics/src/sources/registry.rs`

**必须包含的内容**：

```rust
use crate::core::Backend;
use super::traits_generic::{SourceTermGeneric, SourceContributionGeneric, SourceContextGeneric, SourceStiffness};
use crate::state::ShallowWaterState;
use crate::engine::strategy::workspace::SolverWorkspaceGeneric;
use std::collections::HashMap;

/// 源项注册中心
pub struct SourceRegistry<B: Backend> {
    /// 已注册的源项
    sources: Vec<Box<dyn SourceTermGeneric<B>>>,
    /// 名称到索引的映射
    name_index: HashMap<String, usize>,
    /// 启用状态
    enabled: Vec<bool>,
    /// 并行计算阈值
    parallel_threshold: usize,
}

impl<B: Backend> SourceRegistry<B> {
    pub fn new() -> Self;
    
    /// 注册源项
    pub fn register<S: SourceTermGeneric<B> + 'static>(&mut self, source: S) -> usize;
    
    /// 按名称获取源项
    pub fn get(&self, name: &str) -> Option<&dyn SourceTermGeneric<B>>;
    
    /// 按名称获取可变源项
    pub fn get_mut(&mut self, name: &str) -> Option<&mut dyn SourceTermGeneric<B>>;
    
    /// 启用/禁用源项
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> bool;
    
    /// 移除源项
    pub fn unregister(&mut self, name: &str) -> bool;
    
    /// 获取所有已注册的源项名称
    pub fn list_sources(&self) -> Vec<&str>;
    
    /// 累加所有源项贡献到工作区
    pub fn accumulate_all(
        &self,
        state: &ShallowWaterState,
        workspace: &mut SolverWorkspaceGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    );
    
    /// 仅累加显式源项
    pub fn accumulate_explicit(
        &self,
        state: &ShallowWaterState,
        workspace: &mut SolverWorkspaceGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    );
    
    /// 仅累加局部隐式源项
    pub fn accumulate_locally_implicit(
        &self,
        state: &ShallowWaterState,
        workspace: &mut SolverWorkspaceGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    );
    
    /// 批量计算（并行优化）
    fn accumulate_parallel(
        &self,
        state: &ShallowWaterState,
        contributions: &mut [SourceContributionGeneric<B::Scalar>],
        ctx: &SourceContextGeneric<B::Scalar>,
    );
    
    /// 获取指定刚性类型的源项
    pub fn filter_by_stiffness(&self, stiffness: SourceStiffness) -> Vec<&dyn SourceTermGeneric<B>>;
}

impl<B: Backend> Default for SourceRegistry<B> {
    fn default() -> Self { Self::new() }
}
```

**修改文件**：`crates/mh_physics/src/sources/mod.rs`
- 添加 `pub mod registry;`
- 添加 `pub use registry::SourceRegistry;`

### R1.2 创建 `mh_physics/src/sediment/exchange.rs`

**文件路径**：`crates/mh_physics/src/sediment/exchange.rs`

**必须包含的内容**：

```rust
use crate::core::{Backend, Scalar};

/// 泥沙交换参数
#[derive(Debug, Clone)]
pub struct ExchangeParams<S: Scalar> {
    /// 临界剪切应力 [Pa]
    pub tau_critical: S,
    /// 侵蚀系数 [kg/m²/s/Pa] (Partheniades公式)
    pub erosion_rate: S,
    /// 沉降速度 [m/s]
    pub settling_velocity: S,
    /// 泥沙干密度 [kg/m³]
    pub dry_density: S,
    /// 床面孔隙率
    pub porosity: S,
}

/// 泥沙交换通量计算器
pub struct SedimentExchange<B: Backend> {
    params: ExchangeParams<B::Scalar>,
    /// 交换通量缓存 [kg/m²/s]，正值=侵蚀，负值=沉降
    flux: B::Buffer<B::Scalar>,
    /// 侵蚀通量（分离存储用于诊断）
    erosion: B::Buffer<B::Scalar>,
    /// 沉降通量
    deposition: B::Buffer<B::Scalar>,
    /// 累积交换量（用于守恒校验）
    cumulative_exchange: B::Scalar,
    backend: B,
}

impl<B: Backend> SedimentExchange<B> {
    pub fn new(backend: B, n_cells: usize, params: ExchangeParams<B::Scalar>) -> Self;
    
    /// 计算侵蚀/沉降通量
    /// 
    /// # 参数
    /// - `tau_bed`: 床面剪切应力 [Pa]
    /// - `concentration`: 近底层泥沙浓度 [kg/m³]
    /// - `depth`: 水深 [m]
    pub fn compute(
        &mut self,
        tau_bed: &B::Buffer<B::Scalar>,
        concentration: &B::Buffer<B::Scalar>,
        depth: &B::Buffer<B::Scalar>,
    );
    
    /// 获取净交换通量
    pub fn flux(&self) -> &B::Buffer<B::Scalar>;
    
    /// 获取侵蚀通量
    pub fn erosion(&self) -> &B::Buffer<B::Scalar>;
    
    /// 获取沉降通量
    pub fn deposition(&self) -> &B::Buffer<B::Scalar>;
    
    /// 应用通量更新床面质量
    /// 
    /// bed_mass[i] += flux[i] * dt * cell_area[i]
    pub fn apply_to_bed(
        &self,
        bed_mass: &mut B::Buffer<B::Scalar>,
        dt: B::Scalar,
        cell_areas: &B::Buffer<B::Scalar>,
    );
    
    /// 应用通量更新悬沙浓度
    /// 
    /// concentration[i] -= flux[i] * dt / depth[i]
    pub fn apply_to_suspended(
        &self,
        concentration: &mut B::Buffer<B::Scalar>,
        depth: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
    );
    
    /// 计算侵蚀率（Partheniades公式）
    fn compute_erosion_rate(&self, tau: B::Scalar) -> B::Scalar {
        if tau > self.params.tau_critical {
            self.params.erosion_rate * (tau - self.params.tau_critical)
        } else {
            B::Scalar::ZERO
        }
    }
    
    /// 计算沉降率
    fn compute_deposition_rate(&self, concentration: B::Scalar) -> B::Scalar {
        self.params.settling_velocity * concentration
    }
    
    /// 获取累积交换量（用于守恒校验）
    pub fn cumulative_exchange(&self) -> B::Scalar;
    
    /// 重置累积统计
    pub fn reset_statistics(&mut self);
}
```

**修改文件**：`crates/mh_physics/src/sediment/mod.rs`
- 添加 `pub mod exchange;`
- 添加 `pub use exchange::{SedimentExchange, ExchangeParams};`

### R1.3 创建 `mh_physics/src/tracer/settling.rs`

**文件路径**：`crates/mh_physics/src/tracer/settling.rs`

**必须包含的内容**：

```rust
use crate::core::{Backend, Scalar};

/// 沉降求解器配置
#[derive(Debug, Clone)]
pub struct SettlingConfig<S: Scalar> {
    /// 沉降速度 [m/s]
    pub settling_velocity: S,
    /// 是否使用隐式格式
    pub implicit: bool,
    /// 隐式求解容差（仅隐式模式）
    pub tolerance: S,
    /// 最大迭代次数（仅隐式模式）
    pub max_iterations: usize,
    /// 最小水深阈值
    pub min_depth: S,
}

impl<S: Scalar> Default for SettlingConfig<S> {
    fn default() -> Self {
        Self {
            settling_velocity: S::from_f64(0.001), // 1 mm/s
            implicit: true,
            tolerance: S::from_f64(1e-6),
            max_iterations: 10,
            min_depth: S::from_f64(0.01),
        }
    }
}

/// 沉降求解结果
#[derive(Debug, Clone)]
pub struct SettlingResult<S: Scalar> {
    /// 实际迭代次数
    pub iterations: usize,
    /// 是否收敛
    pub converged: bool,
    /// 最大相对变化
    pub max_relative_change: S,
    /// 总沉降质量
    pub total_settled_mass: S,
}

/// 隐式沉降求解器
pub struct SettlingSolver<B: Backend> {
    config: SettlingConfig<B::Scalar>,
    /// 工作数组：上一迭代浓度
    c_old: B::Buffer<B::Scalar>,
    /// 工作数组：隐式系数
    coeff: B::Buffer<B::Scalar>,
    backend: B,
}

impl<B: Backend> SettlingSolver<B> {
    pub fn new(backend: B, n_cells: usize, config: SettlingConfig<B::Scalar>) -> Self;
    
    /// 隐式求解沉降
    /// 
    /// 求解: (1 + dt * ws / h) * C^{n+1} = C^n
    /// 
    /// # 参数
    /// - `concentration`: 浓度场（输入/输出）
    /// - `depth`: 水深场
    /// - `dt`: 时间步长
    pub fn solve(
        &mut self,
        concentration: &mut B::Buffer<B::Scalar>,
        depth: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) -> SettlingResult<B::Scalar>;
    
    /// 显式沉降（仅用于小时间步）
    /// 
    /// C^{n+1} = C^n - dt * ws * C^n / h
    pub fn apply_explicit(
        &self,
        concentration: &mut B::Buffer<B::Scalar>,
        depth: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
    );
    
    /// 计算隐式系数 1 / (1 + dt * ws / h)
    fn compute_implicit_coefficient(
        &self,
        depth: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
        coeff: &mut B::Buffer<B::Scalar>,
    );
    
    /// 检查CFL稳定性条件
    pub fn check_explicit_stability(
        &self,
        depth: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) -> bool;
    
    /// 更新配置
    pub fn set_config(&mut self, config: SettlingConfig<B::Scalar>);
    
    /// 获取配置
    pub fn config(&self) -> &SettlingConfig<B::Scalar>;
}
```

**修改文件**：`crates/mh_physics/src/tracer/mod.rs`
- 添加 `pub mod settling;`
- 添加 `pub use settling::{SettlingSolver, SettlingConfig, SettlingResult};`

---

## Phase R2: 技术债务清理

**目标**：消除双轨制代码，统一接口

### R2.1 合并源项双轨制

**操作序列**：

1. **修改** `crates/mh_physics/src/sources/traits.rs`
   - 将 `traits_generic.rs` 中的泛型定义合并到此文件
   - 保留非泛型类型别名作为向后兼容层
   - 添加废弃标记引导用户使用泛型版本

   ```rust
   // 在文件顶部添加
   // =============================================================================
   // 泛型版本（推荐使用）
   // =============================================================================
   
   // ... 从traits_generic.rs合并的内容 ...
   
   // =============================================================================
   // 向后兼容别名（废弃）
   // =============================================================================
   
   #[deprecated(since = "0.4.0", note = "Use SourceTermGeneric<CpuBackend<f64>> instead")]
   pub type SourceTerm = dyn SourceTermGeneric<CpuBackend<f64>>;
   ```

2. **删除** `crates/mh_physics/src/sources/traits_generic.rs`

3. **修改** `crates/mh_physics/src/sources/friction.rs`
   - 将 `friction_generic.rs` 中的泛型实现合并到此文件
   - 保留非泛型类型别名

4. **删除** `crates/mh_physics/src/sources/friction_generic.rs`

5. **修改** `crates/mh_physics/src/sources/mod.rs`
   - 删除对 `traits_generic` 和 `friction_generic` 的引用
   - 更新导出

### R2.2 清理遗留文件

**操作序列**：

1. **修改** `crates/mh_workflow/src/job.rs`
   - 合并 `job_v2.rs` 中有价值的功能

2. **删除** `crates/mh_workflow/src/job_v2.rs`

3. **修改** `crates/mh_workflow/src/lib.rs`
   - 删除 `job_v2` 的导出

---

## Phase R3: AI代理层完善

**目标**：补全AI代理层缺失文件

### R3.1 创建 `mh_agent/src/remote_sensing.rs`

**文件路径**：`crates/mh_agent/src/remote_sensing.rs`

**必须包含的内容**：

```rust
use crate::{AIAgent, AiError, PhysicsSnapshot, Assimilable};

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
    // 注：ONNX运行时为可选依赖，通过feature gate控制
    // #[cfg(feature = "onnx")]
    // model: Option<ort::Session>,
}

impl RemoteSensingAgent {
    pub fn new(config: RemoteSensingConfig) -> Self;
    
    /// 从卫星图像进行推理
    pub fn infer(&mut self, image: &SatelliteImage, target_cells: &[[f64; 2]]) -> Result<InferenceResult, AiError>;
    
    /// 获取预测浓度场
    pub fn predicted(&self) -> Option<&[f64]>;
    
    /// 获取不确定性
    pub fn uncertainty(&self) -> Option<&[f64]>;
    
    /// 检查图像质量
    fn validate_image(&self, image: &SatelliteImage) -> Result<(), AiError>;
    
    /// 空间插值到目标网格
    fn interpolate_to_grid(
        &self,
        data: &[f32],
        image: &SatelliteImage,
        target_cells: &[[f64; 2]],
    ) -> Vec<f64>;
    
    /// 经验公式反演（无模型时使用）
    fn empirical_inversion(&self, reflectance: f32, sensor: SensorType) -> f64;
    
    /// 清除缓存
    pub fn clear_cache(&mut self);
}

impl AIAgent for RemoteSensingAgent {
    fn name(&self) -> &'static str { "RemoteSensing-Sediment" }
    
    fn update(&mut self, snapshot: &PhysicsSnapshot) -> Result<(), AiError>;
    
    fn apply(&self, state: &mut dyn Assimilable) -> Result<(), AiError>;
    
    fn get_prediction(&self) -> Option<&[f64]> {
        if self.has_prediction { Some(&self.predicted) } else { None }
    }
    
    fn get_uncertainty(&self) -> Option<&[f64]> {
        if self.has_prediction { Some(&self.uncertainty) } else { None }
    }
}
```

### R3.2 创建 `mh_agent/src/observation.rs`

**文件路径**：`crates/mh_agent/src/observation.rs`

**必须包含的内容**：

```rust
use crate::PhysicsSnapshot;

/// 观测算子trait
pub trait ObservationOperator: Send + Sync {
    /// 观测类型名称
    fn name(&self) -> &'static str;
    
    /// 模拟状态 → 观测空间
    fn observe(&self, snapshot: &PhysicsSnapshot) -> Vec<f64>;
    
    /// 计算观测-模拟残差
    fn residual(&self, snapshot: &PhysicsSnapshot, observation: &[f64]) -> Vec<f64>;
    
    /// 获取观测误差协方差（对角阵时返回方差）
    fn observation_error_variance(&self) -> Option<Vec<f64>> { None }
    
    /// 线性化观测算子（返回雅可比矩阵）
    fn linearize(&self, snapshot: &PhysicsSnapshot) -> Option<Vec<Vec<f64>>> { None }
}

/// 遥感反射率观测算子
pub struct ReflectanceOperator {
    /// 波长 [nm]
    wavelength: f64,
    /// 校准参数 [a, b, c, ...] for R = a * ln(C) + b
    calibration: Vec<f64>,
    /// 观测误差标准差
    observation_std: f64,
}

impl ReflectanceOperator {
    pub fn new(wavelength: f64, calibration: Vec<f64>, observation_std: f64) -> Self;
    
    /// 使用默认的MODIS红波段校准参数
    pub fn modis_red_band() -> Self;
    
    /// 使用默认的Sentinel-2校准参数
    pub fn sentinel2_b4() -> Self;
}

impl ObservationOperator for ReflectanceOperator {
    fn name(&self) -> &'static str { "Reflectance" }
    
    fn observe(&self, snapshot: &PhysicsSnapshot) -> Vec<f64> {
        // R = a * ln(C + epsilon) + b
        snapshot.sediment.as_ref()
            .map(|c| c.iter().map(|&conc| {
                let c_safe = conc.max(1e-10);
                self.calibration[0] * c_safe.ln() + self.calibration[1]
            }).collect())
            .unwrap_or_default()
    }
    
    fn residual(&self, snapshot: &PhysicsSnapshot, observation: &[f64]) -> Vec<f64>;
    
    fn observation_error_variance(&self) -> Option<Vec<f64>>;
}

/// SAR后向散射观测算子
pub struct SAROperator {
    /// 入射角 [degrees]
    incidence_angle: f64,
    /// 极化方式
    polarization: Polarization,
    /// 风速校正系数
    wind_correction: f64,
    /// 观测误差标准差 [dB]
    observation_std: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum Polarization {
    VV,
    VH,
    HH,
    HV,
}

impl SAROperator {
    pub fn new(incidence_angle: f64, polarization: Polarization) -> Self;
}

impl ObservationOperator for SAROperator {
    fn name(&self) -> &'static str { "SAR-Backscatter" }
    
    fn observe(&self, snapshot: &PhysicsSnapshot) -> Vec<f64>;
    fn residual(&self, snapshot: &PhysicsSnapshot, observation: &[f64]) -> Vec<f64>;
}

/// 水位观测算子（验潮站）
pub struct WaterLevelOperator {
    /// 观测站位置索引
    station_indices: Vec<usize>,
    /// 观测误差标准差 [m]
    observation_std: f64,
}

impl ObservationOperator for WaterLevelOperator {
    fn name(&self) -> &'static str { "WaterLevel" }
    
    fn observe(&self, snapshot: &PhysicsSnapshot) -> Vec<f64> {
        self.station_indices.iter()
            .map(|&i| snapshot.h.get(i).copied().unwrap_or(0.0) + snapshot.z.get(i).copied().unwrap_or(0.0))
            .collect()
    }
    
    fn residual(&self, snapshot: &PhysicsSnapshot, observation: &[f64]) -> Vec<f64>;
}
```

### R3.3 创建 `mh_agent/src/surrogate.rs`

**文件路径**：`crates/mh_agent/src/surrogate.rs`

**必须包含的内容**：

```rust
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
    pub fn new(config: SurrogateConfig) -> Result<Self, AiError>;
    
    /// 快速预测（替代完整物理计算）
    pub fn predict(&mut self, snapshot: &PhysicsSnapshot) -> Result<SurrogatePrediction, AiError>;
    
    /// 提取输入特征
    fn extract_features(&self, snapshot: &PhysicsSnapshot) -> Vec<f64>;
    
    /// 归一化输入
    fn normalize_input(&self, features: &mut [f64]);
    
    /// 反归一化输出
    fn denormalize_output(&self, output: &mut [f64]);
    
    /// 评估预测质量（与完整物理对比）
    pub fn evaluate_prediction(
        &self,
        prediction: &SurrogatePrediction,
        ground_truth: &PhysicsSnapshot,
    ) -> PredictionMetrics;
    
    /// 更新模型（在线学习）
    pub fn update_model(&mut self, snapshot: &PhysicsSnapshot, target: &[f64]) -> Result<(), AiError>;
    
    /// 获取预测不确定性
    pub fn uncertainty(&self) -> Option<&[f64]>;
    
    /// 检查模型是否适用于当前状态
    pub fn is_applicable(&self, snapshot: &PhysicsSnapshot) -> bool;
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
            // 将预测结果应用到状态
            // 这里简化处理，实际需要根据output_features映射
            Ok(())
        } else {
            Err(AiError::NotReady("No prediction available".into()))
        }
    }
    
    fn get_prediction(&self) -> Option<&[f64]> {
        self.current_prediction.as_ref().map(|p| p.values.as_slice())
    }
    
    fn get_uncertainty(&self) -> Option<&[f64]> {
        self.current_prediction.as_ref()
            .and_then(|p| p.uncertainty.as_ref())
            .map(|u| u.as_slice())
    }
}
```

### R3.4 更新 `mh_agent/src/lib.rs`

**修改文件**：`crates/mh_agent/src/lib.rs`

确保包含以下模块导出：

```rust
pub mod registry;
pub mod assimilation;
pub mod remote_sensing;
pub mod observation;
pub mod surrogate;

pub use registry::AgentRegistry;
pub use assimilation::{NudgingAssimilator, NudgingConfig, Observation, AssimilationResult};
pub use remote_sensing::{RemoteSensingAgent, RemoteSensingConfig, SatelliteImage, SensorType};
pub use observation::{ObservationOperator, ReflectanceOperator, SAROperator, WaterLevelOperator};
pub use surrogate::{SurrogateModel, SurrogateConfig, SurrogateType, SurrogatePrediction};
```

---

## Phase R4: 桥接层建设

**目标**：创建AI-物理桥接，使AI代理能访问和修改物理状态

### R4.1 创建 `mh_physics/src/assimilation/mod.rs`

**文件路径**：`crates/mh_physics/src/assimilation/mod.rs`

**必须包含的内容**：

```rust
//! 数据同化桥接层
//! 
//! 提供AI代理层与物理核心之间的接口

mod bridge;
mod conservation;

pub use bridge::{AssimilableBridge, StateSnapshot};
pub use conservation::{ConservationChecker, ConservedQuantities};

use crate::state::ShallowWaterState;
use crate::tracer::TracerType;

/// 可同化状态接口（重新定义，因为mh_agent的trait不能直接用于mh_physics）
pub trait PhysicsAssimilable {
    /// 获取示踪剂可变引用
    fn get_tracer_mut(&mut self, tracer_type: TracerType) -> Option<&mut [f64]>;
    
    /// 获取速度场可变引用 (u, v)
    fn get_velocity_mut(&mut self) -> (&mut [f64], &mut [f64]);
    
    /// 获取水深可变引用
    fn get_depth_mut(&mut self) -> &mut [f64];
    
    /// 获取床面高程可变引用
    fn get_bed_elevation_mut(&mut self) -> &mut [f64];
    
    /// 单元数量
    fn n_cells(&self) -> usize;
    
    /// 单元面积
    fn cell_areas(&self) -> &[f64];
    
    /// 单元中心坐标
    fn cell_centers(&self) -> &[[f64; 2]];
    
    /// 创建状态快照（用于AI推理）
    fn create_snapshot(&self) -> StateSnapshot;
    
    /// 计算守恒量
    fn compute_conserved(&self) -> ConservedQuantities;
    
    /// 强制守恒
    fn enforce_conservation(&mut self, reference: &ConservedQuantities, tolerance: f64);
}
```

### R4.2 创建 `mh_physics/src/assimilation/bridge.rs`

**文件路径**：`crates/mh_physics/src/assimilation/bridge.rs`

**必须包含的内容**：

```rust
use super::PhysicsAssimilable;
use crate::state::ShallowWaterState;
use crate::tracer::TracerType;
use crate::core::{Backend, CpuBackend};

/// 状态快照（与mh_agent::PhysicsSnapshot兼容）
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub h: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub z: Vec<f64>,
    pub sediment: Option<Vec<f64>>,
    pub time: f64,
    pub cell_centers: Vec<[f64; 2]>,
    pub cell_areas: Vec<f64>,
}

/// 桥接适配器
pub struct AssimilableBridge<'a> {
    state: &'a mut ShallowWaterState,
    cell_areas: Vec<f64>,
    cell_centers: Vec<[f64; 2]>,
}

impl<'a> AssimilableBridge<'a> {
    pub fn new(
        state: &'a mut ShallowWaterState,
        cell_areas: Vec<f64>,
        cell_centers: Vec<[f64; 2]>,
    ) -> Self {
        Self { state, cell_areas, cell_centers }
    }
}

impl<'a> PhysicsAssimilable for AssimilableBridge<'a> {
    fn get_tracer_mut(&mut self, tracer_type: TracerType) -> Option<&mut [f64]> {
        // 实现示踪剂访问
        // 需要根据ShallowWaterState的实际结构实现
        todo!("Implement based on actual ShallowWaterState structure")
    }
    
    fn get_velocity_mut(&mut self) -> (&mut [f64], &mut [f64]) {
        // 从动量恢复速度需要水深
        // 这里返回动量字段，调用者需要除以水深
        (self.state.hu.as_mut_slice(), self.state.hv.as_mut_slice())
    }
    
    fn get_depth_mut(&mut self) -> &mut [f64] {
        self.state.h.as_mut_slice()
    }
    
    fn get_bed_elevation_mut(&mut self) -> &mut [f64] {
        self.state.z.as_mut_slice()
    }
    
    fn n_cells(&self) -> usize {
        self.state.n_cells()
    }
    
    fn cell_areas(&self) -> &[f64] {
        &self.cell_areas
    }
    
    fn cell_centers(&self) -> &[[f64; 2]] {
        &self.cell_centers
    }
    
    fn create_snapshot(&self) -> StateSnapshot {
        let n = self.n_cells();
        let h = self.state.h.as_slice().to_vec();
        
        // 计算速度
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for i in 0..n {
            let depth = h[i].max(1e-10);
            u[i] = self.state.hu[i] / depth;
            v[i] = self.state.hv[i] / depth;
        }
        
        StateSnapshot {
            h,
            u,
            v,
            z: self.state.z.as_slice().to_vec(),
            sediment: None, // 根据实际情况提取
            time: 0.0,      // 需要从外部传入
            cell_centers: self.cell_centers.clone(),
            cell_areas: self.cell_areas.clone(),
        }
    }
    
    fn compute_conserved(&self) -> super::ConservedQuantities {
        super::ConservedQuantities::compute(self)
    }
    
    fn enforce_conservation(&mut self, reference: &super::ConservedQuantities, tolerance: f64) {
        // 实现守恒强制
        let current = self.compute_conserved();
        
        // 质量修正
        let mass_error = current.total_mass - reference.total_mass;
        if mass_error.abs() > tolerance {
            let correction = reference.total_mass / current.total_mass;
            for h in self.state.h.as_mut_slice() {
                *h *= correction;
            }
        }
        
        // 类似地处理动量和泥沙
    }
}
```

### R4.3 创建 `mh_physics/src/assimilation/conservation.rs`

**文件路径**：`crates/mh_physics/src/assimilation/conservation.rs`

**必须包含的内容**：

```rust
use super::PhysicsAssimilable;

/// 守恒量快照
#[derive(Debug, Clone)]
pub struct ConservedQuantities {
    /// 总水体质量 [kg]
    pub total_mass: f64,
    /// 总x方向动量 [kg·m/s]
    pub total_momentum_x: f64,
    /// 总y方向动量 [kg·m/s]
    pub total_momentum_y: f64,
    /// 总泥沙质量 [kg]（如果有）
    pub total_sediment: Option<f64>,
    /// 总能量 [J]（势能+动能）
    pub total_energy: f64,
}

impl ConservedQuantities {
    /// 从可同化状态计算守恒量
    pub fn compute(state: &dyn PhysicsAssimilable) -> Self {
        let n = state.n_cells();
        let areas = state.cell_areas();
        
        let h = unsafe { 
            std::slice::from_raw_parts(
                state.get_depth_mut() as *const _ as *const f64,
                n
            )
        };
        
        let mut total_mass = 0.0;
        let mut total_energy = 0.0;
        
        const RHO: f64 = 1000.0; // 水密度
        const G: f64 = 9.81;
        
        for i in 0..n {
            let volume = h[i] * areas[i];
            total_mass += RHO * volume;
            total_energy += 0.5 * RHO * G * h[i] * h[i] * areas[i]; // 势能
        }
        
        Self {
            total_mass,
            total_momentum_x: 0.0, // 需要实现
            total_momentum_y: 0.0,
            total_sediment: None,
            total_energy,
        }
    }
    
    /// 计算与参考值的相对误差
    pub fn relative_error(&self, reference: &Self) -> ConservationError {
        ConservationError {
            mass_error: (self.total_mass - reference.total_mass) / reference.total_mass.max(1e-10),
            momentum_x_error: (self.total_momentum_x - reference.total_momentum_x).abs(),
            momentum_y_error: (self.total_momentum_y - reference.total_momentum_y).abs(),
            sediment_error: match (&self.total_sediment, &reference.total_sediment) {
                (Some(s1), Some(s2)) => Some((s1 - s2) / s2.max(1e-10)),
                _ => None,
            },
            energy_error: (self.total_energy - reference.total_energy) / reference.total_energy.max(1e-10),
        }
    }
}

/// 守恒误差
#[derive(Debug, Clone)]
pub struct ConservationError {
    pub mass_error: f64,
    pub momentum_x_error: f64,
    pub momentum_y_error: f64,
    pub sediment_error: Option<f64>,
    pub energy_error: f64,
}

impl ConservationError {
    /// 检查是否在容差范围内
    pub fn within_tolerance(&self, tol: f64) -> bool {
        self.mass_error.abs() < tol
            && self.momentum_x_error.abs() < tol
            && self.momentum_y_error.abs() < tol
            && self.sediment_error.map(|e| e.abs() < tol).unwrap_or(true)
    }
}

/// 守恒校验器
pub struct ConservationChecker {
    /// 初始守恒量
    initial: ConservedQuantities,
    /// 容差
    tolerance: f64,
    /// 历史记录
    history: Vec<(f64, ConservationError)>,
}

impl ConservationChecker {
    pub fn new(initial: ConservedQuantities, tolerance: f64) -> Self {
        Self {
            initial,
            tolerance,
            history: Vec::new(),
        }
    }
    
    /// 检查当前状态的守恒性
    pub fn check(&mut self, state: &dyn PhysicsAssimilable, time: f64) -> ConservationError {
        let current = ConservedQuantities::compute(state);
        let error = current.relative_error(&self.initial);
        self.history.push((time, error.clone()));
        error
    }
    
    /// 获取最大历史误差
    pub fn max_error(&self) -> Option<&ConservationError> {
        self.history.iter()
            .max_by(|a, b| a.1.mass_error.abs().partial_cmp(&b.1.mass_error.abs()).unwrap())
            .map(|(_, e)| e)
    }
}
```

### R4.4 更新 `mh_physics/src/lib.rs`

**修改文件**：`crates/mh_physics/src/lib.rs`

添加模块导出：

```rust
pub mod assimilation;
```

---

## Phase R5: 测试补全

**目标**：创建关键测试文件

### R5.1 创建 `tests/backend_generic.rs`

**文件路径**：`crates/mh_physics/tests/backend_generic.rs`

**必须包含的测试**：

```rust
//! Backend泛型化测试
//! 验证f32/f64后端的一致性和正确性

use mh_physics::core::{Backend, CpuBackend, Scalar};

/// 测试f32/f64后端的一致性
#[test]
fn test_f32_f64_consistency() {
    let backend_f32 = CpuBackend::<f32>::default();
    let backend_f64 = CpuBackend::<f64>::default();
    
    let n = 1000;
    
    // 分配缓冲区
    let x_f32 = backend_f32.alloc(n, 1.0f32);
    let mut y_f32 = backend_f32.alloc(n, 2.0f32);
    
    let x_f64 = backend_f64.alloc(n, 1.0f64);
    let mut y_f64 = backend_f64.alloc(n, 2.0f64);
    
    // axpy: y = 0.5 * x + y
    backend_f32.axpy(0.5, &x_f32, &mut y_f32);
    backend_f64.axpy(0.5, &x_f64, &mut y_f64);
    
    // 比较结果
    for i in 0..n {
        let diff = (y_f32[i] as f64 - y_f64[i]).abs();
        assert!(diff < 1e-5, "f32/f64 inconsistency at index {}: diff = {}", i, diff);
    }
}

/// 测试dot产品精度
#[test]
fn test_dot_precision() {
    let backend = CpuBackend::<f64>::default();
    let n = 10000;
    
    let x = backend.alloc(n, 1.0);
    let y = backend.alloc(n, 1.0);
    
    let result = backend.dot(&x, &y);
    let expected = n as f64;
    
    assert!((result - expected).abs() < 1e-10, "Dot product error: {} vs {}", result, expected);
}

/// 测试reduce操作
#[test]
fn test_reduce_operations() {
    let backend = CpuBackend::<f64>::default();
    
    let mut data = backend.alloc(100, 0.0);
    for i in 0..100 {
        data[i] = i as f64;
    }
    
    let max = backend.reduce_max(&data);
    let min = backend.reduce_min(&data);
    let sum = backend.reduce_sum(&data);
    
    assert_eq!(max, 99.0);
    assert_eq!(min, 0.0);
    assert_eq!(sum, 4950.0); // 0 + 1 + ... + 99
}

/// 测试正性保持
#[test]
fn test_enforce_positivity() {
    let backend = CpuBackend::<f64>::default();
    
    let mut data = vec![-1.0, 0.0, 1.0, -0.5, 2.0];
    backend.enforce_positivity(&mut data, 0.0);
    
    assert!(data.iter().all(|&x| x >= 0.0));
}

/// 测试内存位置
#[test]
fn test_memory_location() {
    use mh_physics::core::MemoryLocation;
    
    let backend = CpuBackend::<f64>::default();
    assert_eq!(backend.memory_location(), MemoryLocation::Host);
}
```

### R5.2 创建 `tests/strategy_switching.rs`

**文件路径**：`crates/mh_physics/tests/strategy_switching.rs`

**必须包含的测试**：

```rust
//! 策略切换测试
//! 验证显式/半隐式策略的切换和状态连续性

use mh_physics::engine::strategy::{
    TimeIntegrationStrategy, ExplicitStrategy, SemiImplicitStrategyGeneric,
    ExplicitConfig, SemiImplicitConfig, StrategyKind,
};
use mh_physics::core::CpuBackend;

/// 测试策略可以被创建
#[test]
fn test_strategy_creation() {
    let _explicit = ExplicitStrategy::new(ExplicitConfig::default());
    let _semi_implicit = SemiImplicitStrategyGeneric::<CpuBackend<f64>>::new(
        SemiImplicitConfig::default()
    );
}

/// 测试策略名称
#[test]
fn test_strategy_names() {
    let explicit = ExplicitStrategy::new(ExplicitConfig::default());
    assert!(!explicit.name().is_empty());
    
    let semi_implicit = SemiImplicitStrategyGeneric::<CpuBackend<f64>>::new(
        SemiImplicitConfig::default()
    );
    assert!(!semi_implicit.name().is_empty());
}

/// 测试策略CFL支持
#[test]
fn test_cfl_support() {
    let explicit = ExplicitStrategy::new(ExplicitConfig::default());
    assert!(!explicit.supports_large_cfl());
    
    let semi_implicit = SemiImplicitStrategyGeneric::<CpuBackend<f64>>::new(
        SemiImplicitConfig::default()
    );
    assert!(semi_implicit.supports_large_cfl());
}

// 更多策略切换测试需要完整的Solver设置...
```

### R5.3 创建 `tests/sediment_coupling.rs`

**文件路径**：`crates/mh_physics/tests/sediment_coupling.rs`

**必须包含的测试**：

```rust
//! 泥沙耦合测试
//! 验证泥沙系统的质量守恒

use mh_physics::sediment::manager::SedimentManager;
use mh_physics::core::CpuBackend;

/// 测试泥沙管理器创建
#[test]
fn test_sediment_manager_creation() {
    // 根据实际SedimentManager接口实现
    // let manager = SedimentManager::<CpuBackend<f64>>::new(...);
}

/// 测试质量守恒
#[test]
fn test_mass_conservation() {
    // 创建简单场景
    // 执行多步更新
    // 验证总质量守恒
}

/// 测试侵蚀/沉降平衡
#[test]
fn test_erosion_deposition_balance() {
    // 在平衡条件下（tau = tau_critical）
    // 验证净通量为零
}
```

### R5.4 创建 `tests/ai_assimilation.rs`

**文件路径**：`crates/mh_physics/tests/ai_assimilation.rs`

**必须包含的测试**：

```rust
//! AI同化测试
//! 验证Nudging同化的正确性

use mh_physics::assimilation::{PhysicsAssimilable, ConservedQuantities};

/// 测试守恒量计算
#[test]
fn test_conserved_quantities() {
    // 创建简单状态
    // 计算守恒量
    // 验证结果正确
}

/// 测试守恒校验
#[test]
fn test_conservation_check() {
    // 创建守恒校验器
    // 模拟状态变化
    // 验证误差计算
}
```

---

## 执行优先级与依赖关系

```
Phase R0 (编译阻塞修复)
    │
    └──▶ Phase R1.1 (SourceRegistry) ──┐
         Phase R1.2 (Exchange)     ────┼──▶ Phase R2 (债务清理)
         Phase R1.3 (Settling)     ────┘        │
                                                │
Phase R3 (AI代理层)                             │
    │                                           │
    └──────────────────┬────────────────────────┘
                       │
                       ▼
                Phase R4 (桥接层)
                       │
                       ▼
                Phase R5 (测试)
```

---

## 文件操作汇总

### 新建文件（12个）

| 序号 | 文件路径 | 优先级 |
|------|----------|--------|
| 1 | `mh_agent/src/assimilation.rs` | 🔴 P0 |
| 2 | `mh_physics/src/sources/registry.rs` | 🟠 P1 |
| 3 | `mh_physics/src/sediment/exchange.rs` | 🟠 P1 |
| 4 | `mh_physics/src/tracer/settling.rs` | 🟠 P1 |
| 5 | `mh_agent/src/remote_sensing.rs` | 🟡 P2 |
| 6 | `mh_agent/src/observation.rs` | 🟡 P2 |
| 7 | `mh_agent/src/surrogate.rs` | 🟡 P2 |
| 8 | `mh_physics/src/assimilation/mod.rs` | 🟡 P2 |
| 9 | `mh_physics/src/assimilation/bridge.rs` | 🟡 P2 |
| 10 | `mh_physics/src/assimilation/conservation.rs` | 🟡 P2 |
| 11 | `mh_physics/tests/backend_generic.rs` | 🟢 P3 |
| 12 | `mh_physics/tests/strategy_switching.rs` | 🟢 P3 |

### 删除文件（3个）

| 序号 | 文件路径 | 原因 |
|------|----------|------|
| 1 | `mh_physics/src/sources/traits_generic.rs` | 合并到traits.rs |
| 2 | `mh_physics/src/sources/friction_generic.rs` | 合并到friction.rs |
| 3 | `mh_workflow/src/job_v2.rs` | 合并到job.rs |

### 修改文件（8个）

| 序号 | 文件路径 | 修改内容 |
|------|----------|----------|
| 1 | `mh_agent/src/lib.rs` | 添加新模块导出 |
| 2 | `mh_physics/src/lib.rs` | 添加assimilation模块 |
| 3 | `mh_physics/src/sources/mod.rs` | 添加registry，删除*_generic |
| 4 | `mh_physics/src/sources/traits.rs` | 合并泛型版本 |
| 5 | `mh_physics/src/sources/friction.rs` | 合并泛型版本 |
| 6 | `mh_physics/src/sediment/mod.rs` | 添加exchange模块 |
| 7 | `mh_physics/src/tracer/mod.rs` | 添加settling模块 |
| 8 | `mh_workflow/src/job.rs` | 合并job_v2功能 |

---

## 执行验证

每个Phase完成后运行：

```bash
# Phase R0 完成后
cargo check -p mh_agent

# Phase R1 完成后
cargo check -p mh_physics

# Phase R2 完成后
cargo check --all

# Phase R3 完成后
cargo check -p mh_agent

# Phase R4 完成后
cargo check -p mh_physics

# Phase R5 完成后
cargo test -p mh_physics

# 全部完成后
cargo test --all
cargo clippy --all
```

---

## 执行指令

**致执行Agent**：

请按以下顺序执行：

1. **第一步（紧急）**：创建 `mh_agent/src/assimilation.rs`，恢复编译
2. **第二步**：创建 `sources/registry.rs`、`sediment/exchange.rs`、`tracer/settling.rs`
3. **第三步**：合并并删除双轨制文件
4. **第四步**：创建AI代理层剩余文件
5. **第五步**：创建桥接层文件
6. **第六步**：创建测试文件
7. **第七步**：更新所有mod.rs和lib.rs

**每个文件必须完整实现，不允许使用`todo!()`占位符（除非有明确的feature gate说明）**。