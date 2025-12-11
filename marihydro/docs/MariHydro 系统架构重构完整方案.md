# MariHydro 系统架构重构完整方案

## 一、核心问题与分歧点分析

### 1.1 综合诊断

| 问题域 | 现状 | 严重程度 | 多个AI共识 |
|--------|------|----------|-----------|
| **Backend悬空** | trait定义完善但0%使用率 | 🔴 Critical | ✅ 一致认为必须强制渗透 |
| **Scalar双轨制** | mh_foundation与mh_physics各有定义 | 🔴 Critical | ✅ 一致认为必须合并 |
| **静态方法陷阱** | Backend方法无&self，GPU无法持有设备 | 🔴 Critical | ✅ 一致认为必须改实例方法 |
| **泥沙模块断裂** | 床变/悬沙/示踪剂/垂向无耦合 | 🟠 High | ✅ 一致认为需要SedimentManager |
| **半隐式骨架化** | Poisson求解器缺失 | 🟠 High | ✅ 一致认为需要完善PCG |
| **AI集成方式** | 接口碎片化 | 🟡 Medium | ⚠️ 分歧：实现SourceTerm vs 独立桥接 |

### 1.2 关键分歧点决策

| 分歧点 | 方案A | 方案B | **我的决策** | 理由 |
|--------|-------|-------|-------------|------|
| Backend方法 | 静态方法 | 实例方法 | **实例方法** | GPU需持有CudaDevice/Stream |
| Scalar位置 | 保留两个 | 合并到physics | **合并到physics** | 单一权威源，mh_foundation重导出 |
| AI集成 | 实现SourceTerm | 独立Assimilable | **独立Assimilable** | AI不应污染物理核心 |
| 精度控制 | 运行时混合 | 编译期Backend | **编译期Backend** | GPU kernel需编译期确定类型 |
| 结构化网格 | 立即实现 | 仅保留抽象 | **仅保留抽象** | 非优先级，长江口用非结构化 |
| 3D支持 | 完整实现 | 仅保留trait | **仅保留trait** | 2.5D足够，3D ROI低 |

---

## 二、目标架构设计

### 2.1 分层架构图

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 0: 应用层 (mh_cli, mh_desktop)                                   │
│  - 命令行工具、GUI可视化                                                 │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│  Layer 1: AI代理层 (mh_agent) [新建]                                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                    │
│  │RemoteSensing │ │ Surrogate   │ │ DataAssim.  │                     │
│  │    Agent     │ │   Model     │ │   (EnKF)    │                     │
│  └──────┬───────┘ └──────┬──────┘ └──────┬──────┘                     │
└─────────┼────────────────┼───────────────┼─────────────────────────────┘
          │ Assimilable trait              │
┌─────────▼────────────────▼───────────────▼─────────────────────────────┐
│  Layer 2: 物理引擎 (mh_physics)                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ engine/                                                          │   │
│  │ ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐      │   │
│  │ │   Solver    │──│TimeIntegration  │──│   Workspace      │      │   │
│  │ │  (调度器)   │  │  Strategy<B>    │  │   <B: Backend>   │      │   │
│  │ └─────────────┘  └────────┬────────┘  └──────────────────┘      │   │
│  │                   ┌───────┴───────┐                              │   │
│  │              ┌────▼────┐    ┌─────▼─────┐                       │   │
│  │              │Explicit │    │SemiImplicit│                      │   │
│  │              │Strategy │    │ Strategy   │                      │   │
│  │              └─────────┘    └────────────┘                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌───────────────┐ ┌────────────────┐ ┌─────────────────────────┐     │
│  │ sources/<B>   │ │ sediment/<B>   │ │ vertical/<B>            │     │
│  │ (摩擦/风/...)  │ │ (SedimentMgr)  │ │ (ProfileRestorer)       │     │
│  └───────────────┘ └────────────────┘ └─────────────────────────┘     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ Backend trait
┌────────────────────────────────▼────────────────────────────────────────┐
│  Layer 3: 核心抽象层 (mh_physics/core)                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  Backend    │  │  Scalar     │  │DeviceBuffer │  │ Dimension   │   │
│  │  (trait)    │  │  (trait)    │  │  (trait)    │  │ (D2/D3)     │   │
│  └──────┬──────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│         │                                                              │
│  ┌──────┴───────────────────────────────────────┐                     │
│  │ 实现                                          │                     │
│  │ ┌──────────────┐  ┌──────────────┐           │                     │
│  │ │CpuBackend<S> │  │CudaBackend<S>│ (未来)    │                     │
│  │ │ S: f32/f64   │  │ S: f32/f64   │           │                     │
│  │ └──────────────┘  └──────────────┘           │                     │
│  └──────────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│  Layer 4: 基础设施 (mh_foundation, mh_mesh, mh_geo, mh_io)             │
│  - 重导出core::Scalar                                                   │
│  - 网格拓扑、地理投影、IO驱动                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

1. **Backend强制渗透**：所有物理模块必须接受`<B: Backend>`泛型
2. **Scalar单一权威**：`mh_physics::core::Scalar`是唯一定义，其他crate重导出
3. **策略模式统一**：显式/半隐式作为`TimeIntegrationStrategy`的实现
4. **AI非侵入**：通过`Assimilable` trait桥接，不修改物理核心
5. **2.5D外挂**：`ProfileRestorer`在2D求解后恢复垂向剖面

---

## 三、完整改动文件结构

```text
crates/
├── mh_foundation/src/
│   ├── scalar.rs              # 【重构】删除Float/ScalarOps，重导出physics::core::Scalar
│   ├── memory.rs              # 【重构】AlignedVec标记deprecated，推荐DeviceBuffer
│   └── lib.rs                 # 【修改】更新导出
│
├── mh_physics/src/
│   ├── lib.rs                 # 【修改】更新模块结构
│   │
│   ├── core/                  # 【重构】核心抽象层
│   │   ├── mod.rs             # 【修改】模块导出
│   │   ├── scalar.rs          # 【重构】统一Scalar trait（权威定义）
│   │   ├── backend.rs         # 【重构】Backend改为实例方法
│   │   ├── buffer.rs          # 【扩展】DeviceBuffer能力增强
│   │   ├── cpu_backend.rs     # 【新建】CpuBackend<f32/f64>完整实现
│   │   ├── gpu.rs             # 【扩展】CudaBackend骨架+feature gate
│   │   └── dimension.rs       # 【保持】D2/D3 marker trait
│   │
│   ├── state.rs               # 【重构】统一为ShallowWaterState<B>，删除legacy
│   │
│   ├── mesh/                  # 【保持结构】
│   │   ├── mod.rs
│   │   ├── topology.rs        # 【扩展】MeshTopology<B>泛型化
│   │   ├── unstructured.rs    # 【重构】适配Backend
│   │   └── structured.rs      # 【保持】仅trait骨架
│   │
│   ├── engine/
│   │   ├── mod.rs             # 【修改】删除legacy solver引用
│   │   ├── solver.rs          # 【重构】变为纯调度器，持有Strategy
│   │   ├── workspace.rs       # 【重构】SolverWorkspace<B>泛型化
│   │   └── strategy/
│   │       ├── mod.rs         # 【修改】TimeIntegrationStrategy<B> trait
│   │       ├── explicit.rs    # 【重构】ExplicitStrategy<B>，持有backend实例
│   │       └── semi_implicit.rs # 【重构】完善PCG求解器集成
│   │
│   ├── numerics/
│   │   └── linear_algebra/
│   │       ├── mod.rs         # 【修改】
│   │       ├── csr.rs         # 【扩展】CSR<B>泛型化
│   │       ├── pcg.rs         # 【新建】PCG求解器完整实现
│   │       └── preconditioner.rs # 【扩展】Jacobi/ILU预条件器
│   │
│   ├── sources/
│   │   ├── mod.rs             # 【修改】
│   │   ├── traits.rs          # 【重构】SourceTerm<B>，删除requires_implicit_treatment
│   │   ├── registry.rs        # 【新建】SourceRegistry<B>统一管理
│   │   ├── friction.rs        # 【重构】泛型化
│   │   ├── coriolis.rs        # 【重构】泛型化
│   │   ├── atmosphere.rs      # 【重构】泛型化
│   │   └── turbulence/
│   │       ├── mod.rs         # 【修改】删除k_epsilon引用
│   │       └── smagorinsky.rs # 【保持】标记2D-only
│   │
│   ├── tracer/
│   │   ├── mod.rs             # 【修改】
│   │   ├── state.rs           # 【重构】TracerField<B>泛型化
│   │   ├── transport.rs       # 【扩展】支持沉降隐式求解
│   │   └── settling.rs        # 【新建】沉降隐式求解器
│   │
│   ├── sediment/
│   │   ├── mod.rs             # 【修改】
│   │   ├── manager.rs         # 【新建】SedimentManager<B>统一管理
│   │   ├── morphology.rs      # 【重构】重命名morphology_2d.rs，泛型化
│   │   ├── transport_2_5d.rs  # 【重构】集成TracerField和ProfileRestorer
│   │   └── exchange.rs        # 【新建】侵蚀/沉降交换通量
│   │
│   ├── vertical/
│   │   ├── mod.rs             # 【修改】
│   │   ├── state.rs           # 【重构】LayeredState<B>泛型化
│   │   ├── profile.rs         # 【扩展】ProfileRestorer<B>完整实现
│   │   └── sigma.rs           # 【保持】σ坐标工具
│   │
│   ├── assimilation/          # 【新建】数据同化桥接
│   │   ├── mod.rs             # Assimilable trait定义
│   │   └── bridge.rs          # ShallowWaterState实现Assimilable
│   │
│   └── boundary/              # 【保持结构】泛型化
│       ├── mod.rs
│       ├── types.rs
│       ├── manager.rs         # 【重构】BoundaryManager<B>
│       └── ghost.rs           # 【重构】泛型化
│
├── mh_agent/                  # 【新建crate】AI代理层
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs             # AIAgent trait，Registry
│       ├── registry.rs        # 【新建】AI代理注册中心
│       ├── remote_sensing.rs  # 【新建】遥感反演代理
│       ├── surrogate.rs       # 【新建】代理模型加速
│       └── observation.rs     # 【新建】观测算子
│
└── tests/
    ├── backend_generic.rs     # 【新建】Backend泛型化验证
    ├── strategy_switching.rs  # 【新建】策略切换验证
    ├── sediment_coupling.rs   # 【新建】泥沙耦合验证
    └── ai_assimilation.rs     # 【新建】AI同化验证
```

---

## 四、关键接口设计

### 4.1 统一Scalar Trait

```rust
// mh_physics/src/core/scalar.rs

use bytemuck::Pod;
use num_traits::{Float, NumAssign};

/// 统一标量类型约束 - 项目唯一权威定义
pub trait Scalar: 
    Float + Pod + NumAssign + Default + 
    Copy + Clone + Send + Sync + 'static +
    std::fmt::Debug + std::fmt::Display
{
    const ZERO: Self;
    const ONE: Self;
    const EPSILON: Self;
    const PI: Self;
    const GRAVITY: Self;  // 9.81
    
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
    fn is_finite(self) -> bool;
}

impl Scalar for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const EPSILON: f32 = 1e-6;
    const PI: f32 = std::f32::consts::PI;
    const GRAVITY: f32 = 9.81;
    
    fn from_f64(x: f64) -> f32 { x as f32 }
    fn to_f64(self) -> f64 { self as f64 }
    // ... 其他实现
}

impl Scalar for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const EPSILON: f64 = 1e-12;
    const PI: f64 = std::f64::consts::PI;
    const GRAVITY: f64 = 9.81;
    
    fn from_f64(x: f64) -> f64 { x }
    fn to_f64(self) -> f64 { self }
    // ... 其他实现
}
```

### 4.2 Backend Trait（实例方法版）

```rust
// mh_physics/src/core/backend.rs

use super::scalar::Scalar;
use super::buffer::DeviceBuffer;

/// 计算后端内存位置
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    Host,
    Device(usize),  // GPU设备ID
}

/// 计算后端抽象 - 统一CPU/GPU内存和算子接口
pub trait Backend: Clone + Send + Sync + 'static + std::fmt::Debug {
    /// 标量类型（f32或f64）
    type Scalar: Scalar;
    
    /// 设备缓冲区类型
    type Buffer<T: bytemuck::Pod + Send + Sync>: DeviceBuffer<T>;
    
    /// 后端名称
    fn name(&self) -> &'static str;
    
    /// 内存位置
    fn memory_location(&self) -> MemoryLocation;
    
    /// 分配缓冲区（实例方法）
    fn alloc<T: bytemuck::Pod + Clone + Send + Sync>(&self, len: usize, init: T) -> Self::Buffer<T>;
    
    /// 分配未初始化缓冲区
    fn alloc_uninit<T: bytemuck::Pod + Send + Sync>(&self, len: usize) -> Self::Buffer<T>;
    
    /// 同步操作
    fn synchronize(&self);
    
    // ========== BLAS Level 1 算子 ==========
    
    /// y = alpha * x + y
    fn axpy(&self, alpha: Self::Scalar, x: &Self::Buffer<Self::Scalar>, y: &mut Self::Buffer<Self::Scalar>);
    
    /// dot = x · y
    fn dot(&self, x: &Self::Buffer<Self::Scalar>, y: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    /// y = x
    fn copy(&self, src: &Self::Buffer<Self::Scalar>, dst: &mut Self::Buffer<Self::Scalar>);
    
    /// x *= alpha
    fn scale(&self, alpha: Self::Scalar, x: &mut Self::Buffer<Self::Scalar>);
    
    /// max(x)
    fn reduce_max(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    /// sum(x)
    fn reduce_sum(&self, x: &Self::Buffer<Self::Scalar>) -> Self::Scalar;
    
    // ========== 物理专用算子 ==========
    
    /// 逐元素应用函数 f(x[i])
    fn apply_elementwise<F>(&self, f: F, x: &mut Self::Buffer<Self::Scalar>)
    where F: Fn(Self::Scalar) -> Self::Scalar + Send + Sync;
    
    /// 确保正性：x[i] = max(x[i], min_val)
    fn enforce_positivity(&self, x: &mut Self::Buffer<Self::Scalar>, min_val: Self::Scalar);
}
```

### 4.3 CpuBackend实现

```rust
// mh_physics/src/core/cpu_backend.rs

use super::{Backend, MemoryLocation, Scalar};
use std::marker::PhantomData;

/// CPU后端（无状态，实例化零开销）
#[derive(Clone, Debug, Default)]
pub struct CpuBackend<S: Scalar> {
    _marker: PhantomData<S>,
}

impl<S: Scalar> CpuBackend<S> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<S: Scalar> Backend for CpuBackend<S> {
    type Scalar = S;
    type Buffer<T: bytemuck::Pod + Send + Sync> = Vec<T>;
    
    fn name(&self) -> &'static str {
        if std::mem::size_of::<S>() == 4 { "CPU-f32" } else { "CPU-f64" }
    }
    
    fn memory_location(&self) -> MemoryLocation {
        MemoryLocation::Host
    }
    
    fn alloc<T: bytemuck::Pod + Clone + Send + Sync>(&self, len: usize, init: T) -> Vec<T> {
        vec![init; len]
    }
    
    fn alloc_uninit<T: bytemuck::Pod + Send + Sync>(&self, len: usize) -> Vec<T> {
        let mut v = Vec::with_capacity(len);
        unsafe { v.set_len(len); }
        v
    }
    
    fn synchronize(&self) {
        // CPU无需同步
    }
    
    fn axpy(&self, alpha: S, x: &Vec<S>, y: &mut Vec<S>) {
        debug_assert_eq!(x.len(), y.len());
        for (xi, yi) in x.iter().zip(y.iter_mut()) {
            *yi = *yi + alpha * *xi;
        }
    }
    
    fn dot(&self, x: &Vec<S>, y: &Vec<S>) -> S {
        debug_assert_eq!(x.len(), y.len());
        x.iter().zip(y.iter()).fold(S::ZERO, |acc, (&xi, &yi)| acc + xi * yi)
    }
    
    fn copy(&self, src: &Vec<S>, dst: &mut Vec<S>) {
        dst.copy_from_slice(src);
    }
    
    fn scale(&self, alpha: S, x: &mut Vec<S>) {
        for xi in x.iter_mut() {
            *xi = *xi * alpha;
        }
    }
    
    fn reduce_max(&self, x: &Vec<S>) -> S {
        x.iter().cloned().fold(S::neg_infinity(), S::max)
    }
    
    fn reduce_sum(&self, x: &Vec<S>) -> S {
        x.iter().cloned().fold(S::ZERO, |a, b| a + b)
    }
    
    fn apply_elementwise<F>(&self, f: F, x: &mut Vec<S>)
    where F: Fn(S) -> S + Send + Sync
    {
        for xi in x.iter_mut() {
            *xi = f(*xi);
        }
    }
    
    fn enforce_positivity(&self, x: &mut Vec<S>, min_val: S) {
        for xi in x.iter_mut() {
            *xi = xi.max(min_val);
        }
    }
}
```

### 4.4 TimeIntegrationStrategy Trait

```rust
// mh_physics/src/engine/strategy/mod.rs

use crate::core::Backend;
use crate::state::ShallowWaterState;
use crate::mesh::MeshTopology;
use crate::sources::SourceRegistry;
use crate::engine::workspace::SolverWorkspace;

/// 时间积分步结果
#[derive(Debug, Clone)]
pub struct StepResult<S> {
    pub dt_used: S,
    pub max_wave_speed: S,
    pub dry_cells: usize,
    pub limited_cells: usize,
    pub converged: bool,       // 半隐式迭代是否收敛
    pub iterations: usize,     // 迭代次数（半隐式用）
}

/// 时间积分策略 Trait
pub trait TimeIntegrationStrategy<B: Backend>: Send + Sync {
    /// 策略名称
    fn name(&self) -> &'static str;
    
    /// 执行单步时间积分
    fn step(
        &mut self,
        state: &mut ShallowWaterState<B>,
        mesh: &dyn MeshTopology<B>,
        sources: &SourceRegistry<B>,
        workspace: &mut SolverWorkspace<B>,
        dt: B::Scalar,
    ) -> StepResult<B::Scalar>;
    
    /// 计算稳定时间步长
    fn compute_stable_dt(
        &self,
        state: &ShallowWaterState<B>,
        mesh: &dyn MeshTopology<B>,
        cfl: B::Scalar,
    ) -> B::Scalar;
    
    /// 是否支持大CFL数
    fn supports_large_cfl(&self) -> bool { false }
    
    /// 获取持有的Backend引用
    fn backend(&self) -> &B;
}
```

### 4.5 Assimilable Trait（AI桥接）

```rust
// mh_physics/src/assimilation/mod.rs

use crate::tracer::TracerType;

/// 可同化状态接口 - AI代理层与物理核心的桥接
pub trait Assimilable {
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
    
    /// 单元面积（用于质量计算）
    fn cell_areas(&self) -> &[f64];
}
```

### 4.6 SedimentManager（泥沙统一管理）

```rust
// mh_physics/src/sediment/manager.rs

use crate::core::Backend;
use crate::state::ShallowWaterState;
use crate::tracer::{TracerField, TracerType};
use crate::vertical::LayeredState;

/// 泥沙系统统一管理器 - 闭环质量守恒
pub struct SedimentManager<B: Backend> {
    /// 床面泥沙质量 [kg/m²]
    bed_mass: B::Buffer<B::Scalar>,
    
    /// 悬沙浓度（深度平均） [kg/m³] - 复用TracerField
    suspended: TracerField<B>,
    
    /// 垂向分层浓度（2.5D）
    layered: Option<LayeredState<B>>,
    
    /// 床面侵蚀/沉降交换通量 [kg/m²/s]
    exchange_flux: B::Buffer<B::Scalar>,
    
    /// 初始总质量（守恒校验用）
    initial_total_mass: B::Scalar,
    
    /// 守恒误差容限
    conservation_tolerance: B::Scalar,
    
    backend: B,
}

impl<B: Backend> SedimentManager<B> {
    /// 单步更新（耦合求解）
    pub fn step(
        &mut self,
        state: &ShallowWaterState<B>,
        tau_bed: &B::Buffer<B::Scalar>,  // 床面剪切应力
        dt: B::Scalar,
    ) -> Result<(), SedimentError> {
        // 1. 计算侵蚀/沉降交换
        self.compute_exchange(state, tau_bed)?;
        
        // 2. 更新悬沙（对流+扩散+沉降）
        self.update_suspended(state, dt)?;
        
        // 3. 更新床面质量
        self.update_bed_mass(dt)?;
        
        // 4. 可选：同步到垂向分层
        if let Some(ref mut layered) = self.layered {
            self.sync_to_layered(layered)?;
        }
        
        // 5. 守恒校验与自动修正
        self.enforce_conservation(state)?;
        
        Ok(())
    }
}
```

---

## 五、分阶段实施计划

### Phase 0: 清理与根基（第1周）

**目标**：删除死代码，统一Scalar定义，修复Backend静态方法问题

#### 步骤0.1：删除3D死代码
| 操作 | 文件 | 说明 |
|------|------|------|
| 删除 | `sources/turbulence/k_epsilon.rs` | 3D湍流模型，无2D支持 |
| 修改 | `sources/turbulence/mod.rs` | 删除k_epsilon引用 |
| 删除 | `sources/implicit.rs`中的`ImplicitMethod::CrankNicolson` | 未使用变体 |

**验证**：`cargo check -p mh_physics` 通过

#### 步骤0.2：统一Scalar到physics
| 操作 | 文件 | 说明 |
|------|------|------|
| 重构 | `mh_physics/src/core/scalar.rs` | 完整Scalar trait定义（见4.1） |
| 重构 | `mh_foundation/src/scalar.rs` | 删除Float/ScalarOps，改为重导出 |
| 修改 | `mh_foundation/src/lib.rs` | `pub use mh_physics::core::Scalar;` |
| 标记 | `mh_foundation/src/memory.rs` | `#[deprecated]` AlignedVec |

```rust
// mh_foundation/src/scalar.rs（重构后）
//! Scalar类型重导出
//! 
//! 权威定义位于 `mh_physics::core::scalar`
//! 此模块仅提供向后兼容重导出

pub use mh_physics::core::scalar::{Scalar};

/// 全局精度类型别名
#[cfg(feature = "f32-global")]
pub type GlobalScalar = f32;
#[cfg(not(feature = "f32-global"))]
pub type GlobalScalar = f64;
```

**验证**：全项目`cargo check`通过

#### 步骤0.3：Backend改为实例方法
| 操作 | 文件 | 说明 |
|------|------|------|
| 重构 | `core/backend.rs` | 所有方法添加`&self`（见4.2） |
| 新建 | `core/cpu_backend.rs` | CpuBackend<f32/f64>完整实现（见4.3） |
| 修改 | `core/mod.rs` | 更新导出 |

**验证**：
```rust
#[test]
fn test_cpu_backend_instance() {
    let backend = CpuBackend::<f64>::new();
    let x = backend.alloc(100, 1.0);
    let mut y = backend.alloc(100, 2.0);
    backend.axpy(0.5, &x, &mut y);
    assert!((y[0] - 2.5).abs() < 1e-10);
}
```

---

### Phase 1: 状态与网格泛型化（第2周）

**目标**：ShallowWaterState和MeshTopology全面泛型化

#### 步骤1.1：状态泛型化
| 操作 | 文件 | 说明 |
|------|------|------|
| 重构 | `state.rs` | 统一为`ShallowWaterState<B>`，删除legacy版本 |

```rust
// mh_physics/src/state.rs（关键部分）

use crate::core::Backend;
use crate::tracer::TracerState;

/// 浅水方程状态（泛型化）
pub struct ShallowWaterState<B: Backend> {
    /// 水深 [m]
    pub h: B::Buffer<B::Scalar>,
    /// x方向动量 [m²/s]
    pub hu: B::Buffer<B::Scalar>,
    /// y方向动量 [m²/s]
    pub hv: B::Buffer<B::Scalar>,
    /// 床面高程 [m]
    pub z: B::Buffer<B::Scalar>,
    
    /// 示踪剂状态（可选）
    pub tracers: Option<TracerState<B>>,
    
    /// 单元数量
    n_cells: usize,
    
    /// 持有的Backend引用
    backend: B,
}

impl<B: Backend> ShallowWaterState<B> {
    pub fn new(backend: B, n_cells: usize) -> Self {
        Self {
            h: backend.alloc(n_cells, B::Scalar::ZERO),
            hu: backend.alloc(n_cells, B::Scalar::ZERO),
            hv: backend.alloc(n_cells, B::Scalar::ZERO),
            z: backend.alloc(n_cells, B::Scalar::ZERO),
            tracers: None,
            n_cells,
            backend,
        }
    }
    
    pub fn n_cells(&self) -> usize { self.n_cells }
    pub fn backend(&self) -> &B { &self.backend }
}
```

**验证**：编译通过，现有测试适配（使用`CpuBackend<f64>`）

#### 步骤1.2：网格适配器泛型化
| 操作 | 文件 | 说明 |
|------|------|------|
| 重构 | `mesh/topology.rs` | `MeshTopology<B>`添加泛型 |
| 重构 | `mesh/unstructured.rs` | `UnstructuredMesh<B>`适配 |

```rust
// mh_physics/src/mesh/topology.rs（关键部分）

use crate::core::Backend;

/// 网格拓扑抽象
pub trait MeshTopology<B: Backend>: Send + Sync {
    fn n_cells(&self) -> usize;
    fn n_faces(&self) -> usize;
    fn n_boundary_faces(&self) -> usize;
    
    /// 几何数据（Device Buffer）
    fn cell_centers(&self) -> &B::Buffer<[B::Scalar; 2]>;
    fn cell_volumes(&self) -> &B::Buffer<B::Scalar>;
    fn face_normals(&self) -> &B::Buffer<[B::Scalar; 2]>;
    fn face_areas(&self) -> &B::Buffer<B::Scalar>;
    
    /// 拓扑数据
    fn face_owner(&self) -> &B::Buffer<u32>;
    fn face_neighbor(&self) -> &B::Buffer<i32>; // -1表示边界
}
```

---

### Phase 2: 求解器策略化（第3-4周）

**目标**：统一显式和半隐式为策略模式，完善PCG求解器

#### 步骤2.1：工作区泛型化
| 操作 | 文件 | 说明 |
|------|------|------|
| 重构 | `engine/workspace.rs` | `SolverWorkspace<B>` |

```rust
// mh_physics/src/engine/workspace.rs（关键部分）

use crate::core::Backend;

/// 求解器工作区（复用临时数组）
pub struct SolverWorkspace<B: Backend> {
    /// 面通量
    pub flux_h: B::Buffer<B::Scalar>,
    pub flux_hu: B::Buffer<B::Scalar>,
    pub flux_hv: B::Buffer<B::Scalar>,
    
    /// 单元RHS
    pub rhs_h: B::Buffer<B::Scalar>,
    pub rhs_hu: B::Buffer<B::Scalar>,
    pub rhs_hv: B::Buffer<B::Scalar>,
    
    /// 梯度
    pub grad_h: B::Buffer<[B::Scalar; 2]>,
    pub grad_z: B::Buffer<[B::Scalar; 2]>,
    
    /// 半隐式专用
    pub u_star: B::Buffer<B::Scalar>,
    pub v_star: B::Buffer<B::Scalar>,
    pub eta_prime: B::Buffer<B::Scalar>,
    
    backend: B,
}

impl<B: Backend> SolverWorkspace<B> {
    pub fn new(backend: B, n_cells: usize, n_faces: usize) -> Self {
        Self {
            flux_h: backend.alloc(n_faces, B::Scalar::ZERO),
            flux_hu: backend.alloc(n_faces, B::Scalar::ZERO),
            flux_hv: backend.alloc(n_faces, B::Scalar::ZERO),
            rhs_h: backend.alloc(n_cells, B::Scalar::ZERO),
            rhs_hu: backend.alloc(n_cells, B::Scalar::ZERO),
            rhs_hv: backend.alloc(n_cells, B::Scalar::ZERO),
            grad_h: backend.alloc(n_cells, [B::Scalar::ZERO; 2]),
            grad_z: backend.alloc(n_cells, [B::Scalar::ZERO; 2]),
            u_star: backend.alloc(n_cells, B::Scalar::ZERO),
            v_star: backend.alloc(n_cells, B::Scalar::ZERO),
            eta_prime: backend.alloc(n_cells, B::Scalar::ZERO),
            backend,
        }
    }
    
    pub fn reset(&mut self) {
        // 清零所有工作数组
        self.backend.scale(B::Scalar::ZERO, &mut self.rhs_h);
        self.backend.scale(B::Scalar::ZERO, &mut self.rhs_hu);
        self.backend.scale(B::Scalar::ZERO, &mut self.rhs_hv);
    }
}
```

#### 步骤2.2：显式策略重构
| 操作 | 文件 | 说明 |
|------|------|------|
| 重构 | `engine/strategy/explicit.rs` | `ExplicitStrategy<B>`持有backend实例 |

```rust
// mh_physics/src/engine/strategy/explicit.rs（关键部分）

use crate::core::Backend;
use super::{TimeIntegrationStrategy, StepResult};

pub struct ExplicitStrategy<B: Backend> {
    backend: B,
    config: ExplicitConfig,
    riemann_solver: RiemannSolver,
    wetting_drying: WettingDryingHandler,
}

impl<B: Backend> ExplicitStrategy<B> {
    pub fn new(backend: B, config: ExplicitConfig) -> Self {
        Self {
            backend,
            config,
            riemann_solver: RiemannSolver::new(config.riemann_type),
            wetting_drying: WettingDryingHandler::new(config.dry_tolerance),
        }
    }
}

impl<B: Backend> TimeIntegrationStrategy<B> for ExplicitStrategy<B> {
    fn name(&self) -> &'static str { "Explicit-Godunov" }
    
    fn step(
        &mut self,
        state: &mut ShallowWaterState<B>,
        mesh: &dyn MeshTopology<B>,
        sources: &SourceRegistry<B>,
        workspace: &mut SolverWorkspace<B>,
        dt: B::Scalar,
    ) -> StepResult<B::Scalar> {
        // 1. 重置工作区
        workspace.reset();
        
        // 2. 计算通量
        let max_speed = self.compute_fluxes(state, mesh, workspace);
        
        // 3. 累加源项
        sources.accumulate_all(state, workspace, dt);
        
        // 4. 更新状态
        self.update_state(state, workspace, dt);
        
        // 5. 正性保持
        let (dry, limited) = self.enforce_positivity(state);
        
        StepResult {
            dt_used: dt,
            max_wave_speed: max_speed,
            dry_cells: dry,
            limited_cells: limited,
            converged: true,
            iterations: 1,
        }
    }
    
    fn compute_stable_dt(&self, state: &ShallowWaterState<B>, mesh: &dyn MeshTopology<B>, cfl: B::Scalar) -> B::Scalar {
        // CFL条件计算
        // dt = cfl * min(dx / (|u| + sqrt(gh)))
        todo!("实现CFL计算")
    }
    
    fn backend(&self) -> &B { &self.backend }
}
```

#### 步骤2.3：PCG求解器实现
| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `numerics/linear_algebra/pcg.rs` | 预条件共轭梯度求解器 |

```rust
// mh_physics/src/numerics/linear_algebra/pcg.rs（关键部分）

use crate::core::Backend;

/// PCG求解器
pub struct PcgSolver<B: Backend> {
    /// 最大迭代次数
    max_iterations: usize,
    /// 相对容差
    tolerance: B::Scalar,
    /// 工作向量
    r: B::Buffer<B::Scalar>,   // 残差
    z: B::Buffer<B::Scalar>,   // 预条件后残差
    p: B::Buffer<B::Scalar>,   // 搜索方向
    ap: B::Buffer<B::Scalar>,  // 矩阵-向量积
    
    backend: B,
}

impl<B: Backend> PcgSolver<B> {
    pub fn new(backend: B, n: usize, max_iter: usize, tol: B::Scalar) -> Self {
        Self {
            max_iterations: max_iter,
            tolerance: tol,
            r: backend.alloc(n, B::Scalar::ZERO),
            z: backend.alloc(n, B::Scalar::ZERO),
            p: backend.alloc(n, B::Scalar::ZERO),
            ap: backend.alloc(n, B::Scalar::ZERO),
            backend,
        }
    }
    
    /// 求解 Ax = b
    pub fn solve<M, P>(
        &mut self,
        matrix: &M,           // 矩阵
        precond: &P,          // 预条件器
        b: &B::Buffer<B::Scalar>,
        x: &mut B::Buffer<B::Scalar>,
    ) -> PcgResult<B::Scalar>
    where
        M: SparseMatrix<B>,
        P: Preconditioner<B>,
    {
        let n = b.len();
        
        // r = b - Ax
        matrix.spmv(x, &mut self.r);
        self.backend.axpy(-B::Scalar::ONE, &self.r, &mut self.r);
        self.backend.axpy(B::Scalar::ONE, b, &mut self.r);
        
        let b_norm = self.backend.dot(b, b).sqrt();
        let mut r_norm = self.backend.dot(&self.r, &self.r).sqrt();
        
        if r_norm / b_norm < self.tolerance {
            return PcgResult { converged: true, iterations: 0, residual: r_norm };
        }
        
        // z = M^{-1} r
        precond.apply(&self.r, &mut self.z);
        
        // p = z
        self.backend.copy(&self.z, &mut self.p);
        
        let mut rz = self.backend.dot(&self.r, &self.z);
        
        for k in 0..self.max_iterations {
            // ap = A * p
            matrix.spmv(&self.p, &mut self.ap);
            
            // alpha = rz / (p · ap)
            let pap = self.backend.dot(&self.p, &self.ap);
            let alpha = rz / pap;
            
            // x = x + alpha * p
            self.backend.axpy(alpha, &self.p, x);
            
            // r = r - alpha * ap
            self.backend.axpy(-alpha, &self.ap, &mut self.r);
            
            r_norm = self.backend.dot(&self.r, &self.r).sqrt();
            if r_norm / b_norm < self.tolerance {
                return PcgResult { converged: true, iterations: k + 1, residual: r_norm };
            }
            
            // z = M^{-1} r
            precond.apply(&self.r, &mut self.z);
            
            let rz_new = self.backend.dot(&self.r, &self.z);
            let beta = rz_new / rz;
            rz = rz_new;
            
            // p = z + beta * p
            self.backend.scale(beta, &mut self.p);
            self.backend.axpy(B::Scalar::ONE, &self.z, &mut self.p);
        }
        
        PcgResult { converged: false, iterations: self.max_iterations, residual: r_norm }
    }
}
```

#### 步骤2.4：半隐式策略完善
| 操作 | 文件 | 说明 |
|------|------|------|
| 重构 | `engine/strategy/semi_implicit.rs` | 集成PCG求解器 |

```rust
// mh_physics/src/engine/strategy/semi_implicit.rs（关键部分）

use crate::core::Backend;
use crate::numerics::linear_algebra::{PcgSolver, CsrMatrix, JacobiPreconditioner};

pub struct SemiImplicitStrategy<B: Backend> {
    backend: B,
    config: SemiImplicitConfig,
    
    /// 压力矩阵
    pressure_matrix: CsrMatrix<B>,
    
    /// PCG求解器
    pcg_solver: PcgSolver<B>,
    
    /// Jacobi预条件器
    preconditioner: JacobiPreconditioner<B>,
}

impl<B: Backend> TimeIntegrationStrategy<B> for SemiImplicitStrategy<B> {
    fn name(&self) -> &'static str { "Semi-Implicit-Projection" }
    
    fn step(
        &mut self,
        state: &mut ShallowWaterState<B>,
        mesh: &dyn MeshTopology<B>,
        sources: &SourceRegistry<B>,
        workspace: &mut SolverWorkspace<B>,
        dt: B::Scalar,
    ) -> StepResult<B::Scalar> {
        // 1. 预测步：显式计算u*, v*
        self.compute_prediction(state, mesh, sources, workspace, dt);
        
        // 2. 组装压力Poisson矩阵
        self.assemble_pressure_matrix(state, mesh, dt);
        
        // 3. 计算RHS：∇·(H u*)
        self.compute_divergence(state, workspace);
        
        // 4. PCG求解 η'
        let pcg_result = self.pcg_solver.solve(
            &self.pressure_matrix,
            &self.preconditioner,
            &workspace.rhs_h,
            &mut workspace.eta_prime,
        );
        
        // 5. 校正步：更新u, v, h
        self.apply_correction(state, workspace, dt);
        
        StepResult {
            dt_used: dt,
            max_wave_speed: B::Scalar::ZERO, // 半隐式不需要
            dry_cells: 0,
            limited_cells: 0,
            converged: pcg_result.converged,
            iterations: pcg_result.iterations,
        }
    }
    
    fn supports_large_cfl(&self) -> bool { true }
    
    fn backend(&self) -> &B { &self.backend }
}
```

#### 步骤2.5：统一求解器调度
| 操作 | 文件 | 说明 |
|------|------|------|
| 重构 | `engine/solver.rs` | 变为纯调度器 |

```rust
// mh_physics/src/engine/solver.rs（关键部分）

use crate::core::Backend;
use super::strategy::TimeIntegrationStrategy;

/// 求解策略类型
pub enum StrategyKind {
    Explicit(ExplicitConfig),
    SemiImplicit(SemiImplicitConfig),
}

/// 浅水方程求解器（调度器）
pub struct ShallowWaterSolver<B: Backend> {
    /// 网格
    mesh: Arc<dyn MeshTopology<B>>,
    
    /// 状态
    state: ShallowWaterState<B>,
    
    /// 时间积分策略
    strategy: Box<dyn TimeIntegrationStrategy<B>>,
    
    /// 工作区
    workspace: SolverWorkspace<B>,
    
    /// 源项注册
    sources: SourceRegistry<B>,
    
    /// 边界管理
    boundary: BoundaryManager<B>,
    
    /// 配置
    config: SolverConfig,
}

impl<B: Backend> ShallowWaterSolver<B> {
    pub fn step(&mut self, dt: B::Scalar) -> StepResult<B::Scalar> {
        // 1. 边界条件准备
        self.boundary.apply(&mut self.state);
        
        // 2. 委托策略执行
        let result = self.strategy.step(
            &mut self.state,
            self.mesh.as_ref(),
            &self.sources,
            &mut self.workspace,
            dt,
        );
        
        // 3. 更新时间
        self.current_time += dt.to_f64();
        
        result
    }
    
    /// 运行时切换策略
    pub fn set_strategy(&mut self, kind: StrategyKind) {
        let backend = self.strategy.backend().clone();
        self.strategy = match kind {
            StrategyKind::Explicit(cfg) => {
                Box::new(ExplicitStrategy::new(backend, cfg))
            }
            StrategyKind::SemiImplicit(cfg) => {
                Box::new(SemiImplicitStrategy::new(backend, cfg))
            }
        };
    }
}
```

---

### Phase 3: 源项与示踪剂泛型化（第5周）

**目标**：完成源项系统和示踪剂的Backend泛型化

#### 步骤3.1：源项Trait重构
| 操作 | 文件 | 说明 |
|------|------|------|
| 重构 | `sources/traits.rs` | `SourceTerm<B>`，删除旧接口 |
| 新建 | `sources/registry.rs` | `SourceRegistry<B>`统一管理 |

```rust
// mh_physics/src/sources/traits.rs（关键部分）

use crate::core::Backend;

/// 源项刚性分类
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceStiffness {
    /// 显式处理
    Explicit,
    /// 局部隐式（如摩擦的 1/(1+dt*γ)）
    LocallyImplicit,
}

/// 源项贡献
pub struct SourceContribution<S> {
    pub s_h: S,
    pub s_hu: S,
    pub s_hv: S,
}

/// 源项 Trait
pub trait SourceTerm<B: Backend>: Send + Sync {
    fn name(&self) -> &'static str;
    fn stiffness(&self) -> SourceStiffness;
    
    /// 逐单元计算
    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContext<B::Scalar>,
    ) -> SourceContribution<B::Scalar>;
    
    /// 批量计算（可被GPU重载）
    fn compute_batch(
        &self,
        state: &ShallowWaterState<B>,
        contributions: &mut [SourceContribution<B::Scalar>],
        ctx: &SourceContext<B::Scalar>,
    ) {
        for cell in 0..state.n_cells() {
            contributions[cell] = self.compute_cell(cell, state, ctx);
        }
    }
}
```

#### 步骤3.2：示踪剂泛型化
| 操作 | 文件 | 说明 |
|------|------|------|
| 重构 | `tracer/state.rs` | `TracerField<B>`、`TracerState<B>` |
| 扩展 | `tracer/transport.rs` | 沉降隐式支持 |
| 新建 | `tracer/settling.rs` | 沉降隐式求解器 |

```rust
// mh_physics/src/tracer/state.rs（关键部分）

use crate::core::Backend;

/// 示踪剂场
pub struct TracerField<B: Backend> {
    /// 物理属性
    pub properties: TracerProperties,
    
    /// 浓度场 [单位/m³]
    concentration: B::Buffer<B::Scalar>,
    
    /// 守恒量 h*C
    conserved: B::Buffer<B::Scalar>,
    
    /// RHS
    rhs: B::Buffer<B::Scalar>,
    
    backend: B,
}

impl<B: Backend> TracerField<B> {
    pub fn new(backend: B, n_cells: usize, properties: TracerProperties) -> Self {
        Self {
            properties,
            concentration: backend.alloc(n_cells, B::Scalar::ZERO),
            conserved: backend.alloc(n_cells, B::Scalar::ZERO),
            rhs: backend.alloc(n_cells, B::Scalar::ZERO),
            backend,
        }
    }
    
    pub fn concentration(&self) -> &B::Buffer<B::Scalar> { &self.concentration }
    pub fn concentration_mut(&mut self) -> &mut B::Buffer<B::Scalar> { &mut self.concentration }
}

/// 示踪剂状态管理
pub struct TracerState<B: Backend> {
    fields: HashMap<TracerType, TracerField<B>>,
    backend: B,
}
```

---

### Phase 4: 泥沙系统耦合（第6周）

**目标**：实现SedimentManager，闭合泥沙质量守恒

#### 步骤4.1：新建SedimentManager
| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `sediment/manager.rs` | 统一管理器（见4.6） |
| 新建 | `sediment/exchange.rs` | 侵蚀/沉降交换通量 |
| 重构 | `sediment/morphology.rs` | 泛型化，接入Manager |

```rust
// mh_physics/src/sediment/exchange.rs（关键部分）

use crate::core::Backend;

/// 泥沙交换通量计算
pub struct SedimentExchange<B: Backend> {
    /// 临界剪切应力 [Pa]
    tau_critical: B::Scalar,
    
    /// 侵蚀系数 [kg/m²/s/Pa]
    erosion_rate: B::Scalar,
    
    /// 沉降速度 [m/s]
    settling_velocity: B::Scalar,
    
    backend: B,
}

impl<B: Backend> SedimentExchange<B> {
    /// 计算侵蚀/沉降通量
    pub fn compute(
        &self,
        tau_bed: &B::Buffer<B::Scalar>,
        concentration: &B::Buffer<B::Scalar>,
        depth: &B::Buffer<B::Scalar>,
        flux: &mut B::Buffer<B::Scalar>,  // 正=侵蚀，负=沉降
    ) {
        let n = tau_bed.len();
        for i in 0..n {
            let tau = tau_bed[i];
            let c = concentration[i];
            let h = depth[i];
            
            // 侵蚀（Partheniades公式）
            let erosion = if tau > self.tau_critical {
                self.erosion_rate * (tau - self.tau_critical)
            } else {
                B::Scalar::ZERO
            };
            
            // 沉降
            let deposition = self.settling_velocity * c;
            
            flux[i] = erosion - deposition;
        }
    }
}
```

#### 步骤4.2：2.5D集成ProfileRestorer
| 操作 | 文件 | 说明 |
|------|------|------|
| 扩展 | `vertical/profile.rs` | 完整ProfileRestorer实现 |
| 重构 | `sediment/transport_2_5d.rs` | 调用ProfileRestorer |

```rust
// mh_physics/src/vertical/profile.rs（关键部分）

use crate::core::Backend;

/// 垂向剖面恢复器（从2D状态恢复3D场）
pub struct ProfileRestorer<B: Backend> {
    /// σ坐标
    sigma: SigmaCoordinate,
    
    /// 床面粗糙度
    roughness: B::Buffer<B::Scalar>,
    
    /// 层数
    n_layers: usize,
    
    backend: B,
}

impl<B: Backend> ProfileRestorer<B> {
    /// 恢复垂向速度剖面（对数律）
    pub fn restore_velocity(
        &self,
        h: &B::Buffer<B::Scalar>,
        hu: &B::Buffer<B::Scalar>,
        hv: &B::Buffer<B::Scalar>,
        tau_bed: &B::Buffer<B::Scalar>,
        output: &mut LayeredState<B>,
    ) {
        let n_cells = h.len();
        
        for cell in 0..n_cells {
            let depth = h[cell];
            let u_avg = hu[cell] / depth;
            let v_avg = hv[cell] / depth;
            
            // 计算摩阻速度
            let u_star = (tau_bed[cell] / RHO_WATER).sqrt();
            let z0 = self.roughness[cell];
            
            for k in 0..self.n_layers {
                let z = self.sigma.z_at_layer(k, depth);
                
                // 对数律剖面
                let factor = (z / z0).ln() / VON_KARMAN;
                let u_k = u_avg + u_star * factor * u_avg.signum();
                let v_k = v_avg + u_star * factor * v_avg.signum();
                
                output.set_velocity(cell, k, u_k, v_k);
            }
        }
    }
    
    /// 恢复垂向浓度剖面（Rouse分布）
    pub fn restore_concentration(
        &self,
        c_avg: &B::Buffer<B::Scalar>,
        h: &B::Buffer<B::Scalar>,
        ws: B::Scalar,           // 沉降速度
        u_star: &B::Buffer<B::Scalar>,
        output: &mut LayeredState<B>,
    ) {
        let n_cells = c_avg.len();
        
        for cell in 0..n_cells {
            let depth = h[cell];
            let c0 = c_avg[cell];
            
            // Rouse数
            let rouse = ws / (VON_KARMAN * u_star[cell]);
            
            for k in 0..self.n_layers {
                let z = self.sigma.z_at_layer(k, depth);
                let z_rel = z / depth;
                
                // Rouse分布
                let c_k = c0 * ((1.0 - z_rel) / z_rel).powf(rouse);
                output.set_sediment(cell, k, c_k);
            }
        }
    }
}
```

---

### Phase 5: AI代理层（第7周）

**目标**：新建mh_agent crate，实现AI-物理桥接

#### 步骤5.1：创建mh_agent crate
| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `mh_agent/Cargo.toml` | crate配置 |
| 新建 | `mh_agent/src/lib.rs` | AIAgent trait |
| 新建 | `mh_agent/src/registry.rs` | AI代理注册中心 |

```rust
// mh_agent/src/lib.rs

//! AI代理层 - 遥感驱动的智能预测与同化
//! 
//! 设计原则：
//! 1. 非侵入：不修改mh_physics物理核心
//! 2. 异步解耦：AI推理不阻塞物理计算
//! 3. 守恒安全：AI注入后自动校验守恒性

pub mod registry;
pub mod remote_sensing;
pub mod observation;
pub mod surrogate;

use mh_physics::assimilation::Assimilable;

/// AI代理 Trait
pub trait AIAgent: Send + Sync {
    /// 代理名称
    fn name(&self) -> &'static str;
    
    /// 更新内部状态（基于物理快照）
    fn update(&mut self, snapshot: &PhysicsSnapshot) -> Result<(), AiError>;
    
    /// 应用修正到物理状态
    fn apply(&self, state: &mut dyn Assimilable) -> Result<(), AiError>;
    
    /// 是否需要守恒校验
    fn requires_conservation_check(&self) -> bool { true }
}

/// 物理状态快照（只读，用于AI推理）
pub struct PhysicsSnapshot {
    pub h: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub sediment: Option<Vec<f64>>,
    pub time: f64,
}
```

#### 步骤5.2：遥感反演代理
| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `mh_agent/src/remote_sensing.rs` | 遥感反演实现 |

```rust
// mh_agent/src/remote_sensing.rs

use crate::{AIAgent, PhysicsSnapshot, AiError};
use mh_physics::assimilation::Assimilable;
use mh_physics::tracer::TracerType;

/// 遥感泥沙反演代理
pub struct RemoteSensingAgent {
    /// ONNX模型
    model: ort::Session,
    
    /// 同化率（Nudging系数）
    assimilation_rate: f64,
    
    /// 预测浓度场
    predicted: Vec<f64>,
}

impl RemoteSensingAgent {
    pub fn new(model_path: &str, rate: f64) -> Result<Self, AiError> {
        let model = ort::Session::builder()?
            .with_model_from_file(model_path)?;
        
        Ok(Self {
            model,
            assimilation_rate: rate,
            predicted: Vec::new(),
        })
    }
    
    /// 从卫星图像推理
    pub fn infer(&mut self, image: &SatelliteImage) -> Result<(), AiError> {
        // 准备输入张量
        let input = image.to_tensor()?;
        
        // ONNX推理
        let outputs = self.model.run(ort::inputs![input]?)?;
        
        // 提取预测浓度
        self.predicted = outputs[0].extract_tensor::<f32>()?
            .view()
            .iter()
            .map(|&x| x as f64)
            .collect();
        
        Ok(())
    }
}

impl AIAgent for RemoteSensingAgent {
    fn name(&self) -> &'static str { "RemoteSensing-Sediment" }
    
    fn update(&mut self, _snapshot: &PhysicsSnapshot) -> Result<(), AiError> {
        // 遥感代理不依赖物理快照
        Ok(())
    }
    
    fn apply(&self, state: &mut dyn Assimilable) -> Result<(), AiError> {
        if let Some(sediment) = state.get_tracer_mut(TracerType::Sediment) {
            // Nudging同化
            for (i, c) in sediment.iter_mut().enumerate() {
                if i < self.predicted.len() {
                    *c += self.assimilation_rate * (self.predicted[i] - *c);
                }
            }
        }
        Ok(())
    }
}
```

#### 步骤5.3：Assimilable桥接实现
| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `mh_physics/src/assimilation/mod.rs` | Assimilable trait |
| 新建 | `mh_physics/src/assimilation/bridge.rs` | State实现 |

```rust
// mh_physics/src/assimilation/bridge.rs

use super::Assimilable;
use crate::core::CpuBackend;
use crate::state::ShallowWaterState;
use crate::tracer::TracerType;

/// 为CPU后端状态实现Assimilable
impl Assimilable for ShallowWaterState<CpuBackend<f64>> {
    fn get_tracer_mut(&mut self, tracer_type: TracerType) -> Option<&mut [f64]> {
        self.tracers.as_mut()
            .and_then(|ts| ts.get_mut(&tracer_type))
            .map(|f| f.concentration_mut().as_mut_slice())
    }
    
    fn get_velocity_mut(&mut self) -> (&mut [f64], &mut [f64]) {
        let n = self.n_cells();
        // 需要从动量恢复速度，这里简化处理
        // 实际应提供专用接口
        todo!("实现速度访问")
    }
    
    fn get_depth_mut(&mut self) -> &mut [f64] {
        self.h.as_mut_slice()
    }
    
    fn get_bed_elevation_mut(&mut self) -> &mut [f64] {
        self.z.as_mut_slice()
    }
    
    fn n_cells(&self) -> usize {
        self.n_cells
    }
    
    fn cell_areas(&self) -> &[f64] {
        // 需要从网格获取
        todo!("实现面积访问")
    }
}
```

---

### Phase 6: GPU准备（第8周）

**目标**：完成CUDA接入准备，设计HybridBackend

#### 步骤6.1：CudaBackend骨架
| 操作 | 文件 | 说明 |
|------|------|------|
| 扩展 | `core/gpu.rs` | CudaBackend定义+feature gate |

```rust
// mh_physics/src/core/gpu.rs

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};

/// CUDA后端（需要feature = "cuda"）
#[cfg(feature = "cuda")]
pub struct CudaBackend<S: Scalar> {
    device: std::sync::Arc<CudaDevice>,
    stream: CudaStream,
    _marker: std::marker::PhantomData<S>,
}

#[cfg(feature = "cuda")]
impl<S: Scalar> Backend for CudaBackend<S> {
    type Scalar = S;
    type Buffer<T: bytemuck::Pod + Send + Sync> = CudaSlice<T>;
    
    fn name(&self) -> &'static str {
        if std::mem::size_of::<S>() == 4 { "CUDA-f32" } else { "CUDA-f64" }
    }
    
    fn memory_location(&self) -> MemoryLocation {
        MemoryLocation::Device(0)
    }
    
    fn alloc<T: bytemuck::Pod + Clone + Send + Sync>(&self, len: usize, init: T) -> CudaSlice<T> {
        // 创建Host数组并上传
        let host = vec![init; len];
        self.device.htod_sync_copy(&host).unwrap()
    }
    
    fn synchronize(&self) {
        self.stream.synchronize().unwrap();
    }
    
    fn axpy(&self, alpha: S, x: &CudaSlice<S>, y: &mut CudaSlice<S>) {
        // 调用cuBLAS或自定义kernel
        todo!("Phase 7: 实现CUDA axpy kernel")
    }
    
    // ... 其他方法
}
```

#### 步骤6.2：Kernel接口规范
| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `core/kernel.rs` | Kernel trait定义 |
| 新建 | `docs/gpu_kernel_spec.md` | Kernel实现规范 |

```rust
// mh_physics/src/core/kernel.rs

/// GPU Kernel接口规范
/// 
/// 每个Kernel需实现：
/// 1. 参数校验
/// 2. Grid/Block配置
/// 3. 错误处理
pub trait Kernel {
    /// Kernel名称
    fn name(&self) -> &'static str;
    
    /// Grid配置
    fn grid_config(&self, n: usize) -> (u32, u32, u32);
    
    /// Block配置
    fn block_config(&self) -> (u32, u32, u32) {
        (256, 1, 1)  // 默认256线程/block
    }
}

// ========== 需要实现的Kernel清单 ==========
// P0: axpy, dot, scale, reduce_max, reduce_sum
// P1: flux_compute, state_update
// P2: source_batch, gradient_compute
// P3: pcg_spmv (稀疏矩阵向量积)
```

---

### Phase 7: 测试与验证（第9周）

**目标**：完成架构验证测试

#### 测试矩阵

| 测试用例 | 测试内容 | 验证标准 |
|----------|----------|----------|
| `backend_generic.rs` | f32/f64后端切换 | 结果差异 < 1e-6 |
| `strategy_switching.rs` | 显式/半隐式切换 | 状态连续性 |
| `sediment_coupling.rs` | 泥沙质量守恒 | 误差 < 1e-10 |
| `ai_assimilation.rs` | AI同化验证 | 浓度场更新符合Nudging |
| `dambreak_generic.rs` | 溃坝标准算例 | L2误差 < 1e-3 |
| `thacker_generic.rs` | Thacker解析解 | 收敛阶 ≥ 1.5 |

```rust
// tests/backend_generic.rs

#[test]
fn test_f32_f64_consistency() {
    let backend_f32 = CpuBackend::<f32>::new();
    let backend_f64 = CpuBackend::<f64>::new();
    
    let mesh = create_test_mesh();
    
    let mut solver_f32 = ShallowWaterSolver::new(
        backend_f32,
        mesh.clone(),
        ExplicitConfig::default(),
    );
    
    let mut solver_f64 = ShallowWaterSolver::new(
        backend_f64,
        mesh,
        ExplicitConfig::default(),
    );
    
    // 运行100步
    for _ in 0..100 {
        solver_f32.step(0.001);
        solver_f64.step(0.001);
    }
    
    // 比较结果
    let h_f32: Vec<f64> = solver_f32.state().h.iter().map(|&x| x as f64).collect();
    let h_f64: Vec<f64> = solver_f64.state().h.iter().cloned().collect();
    
    let max_diff = h_f32.iter().zip(h_f64.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    
    assert!(max_diff < 1e-3, "f32/f64差异过大: {}", max_diff);
}
```

---

## 六、实施时间线总结

```text
Week 1: Phase 0 - 清理与根基
├── 删除k_epsilon.rs等3D死代码
├── 统一Scalar到mh_physics::core
└── Backend改为实例方法

Week 2: Phase 1 - 状态与网格泛型化
├── ShallowWaterState<B>重构
└── MeshTopology<B>适配

Week 3-4: Phase 2 - 求解器策略化
├── SolverWorkspace<B>泛型化
├── ExplicitStrategy<B>重构
├── PCG求解器实现
├── SemiImplicitStrategy<B>完善
└── 统一Solver调度器

Week 5: Phase 3 - 源项与示踪剂泛型化
├── SourceTerm<B>重构
├── SourceRegistry<B>新建
├── TracerField<B>泛型化
└── 沉降隐式求解器

Week 6: Phase 4 - 泥沙系统耦合
├── SedimentManager<B>新建
├── 侵蚀/沉降交换通量
├── ProfileRestorer<B>完善
└── 2.5D输运集成

Week 7: Phase 5 - AI代理层
├── mh_agent crate新建
├── AIAgent trait定义
├── RemoteSensingAgent实现
└── Assimilable桥接

Week 8: Phase 6 - GPU准备
├── CudaBackend骨架
├── Kernel接口规范
└── HybridBackend设计

Week 9: Phase 7 - 测试与验证
├── 后端泛型测试
├── 策略切换测试
├── 泥沙耦合测试
└── AI同化测试
```

---

## 七、代码改动量估计

| Phase | 新建行数 | 重构行数 | 删除行数 | 净变化 |
|-------|----------|----------|----------|--------|
| Phase 0 | 200 | 300 | 400 | +100 |
| Phase 1 | 100 | 600 | 200 | +500 |
| Phase 2 | 1500 | 800 | 300 | +2000 |
| Phase 3 | 400 | 500 | 100 | +800 |
| Phase 4 | 800 | 400 | 0 | +1200 |
| Phase 5 | 600 | 100 | 0 | +700 |
| Phase 6 | 300 | 0 | 0 | +300 |
| Phase 7 | 500 | 0 | 0 | +500 |
| **合计** | **4400** | **2700** | **1000** | **+6100** |

---

## 八、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 泛型编译时间↑ | 开发效率 | 使用type alias减少泛型传播深度 |
| f32精度不足 | 数值稳定性 | 关键路径（压力Poisson）强制f64 |
| PCG不收敛 | 半隐式失败 | 预条件器 + 残差监控 + 自动回退显式 |
| AI同化破坏守恒 | 物理错误 | ConservationEnforcer强制校验 |
| GPU内存溢出 | 大网格失败 | 分块处理 + 动态内存池 |

---

**结论**：本方案系统性地解决了Backend悬空、Scalar双轨、求解器碎片化、泥沙断裂等核心问题，通过9周的分阶段实施，将项目从"实验性代码"升级为"生产级架构"。关键设计决策（实例方法Backend、独立AI层、策略模式求解器）均基于对多个AI方案的综合分析和实际需求判断。