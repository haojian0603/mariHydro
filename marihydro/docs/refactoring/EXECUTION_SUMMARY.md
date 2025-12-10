# MariHydro 重构执行总结

## 项目概述

MariHydro 是一个高性能浅水方程求解器，本次重构旨在解决以下核心问题：

1. **Backend 悬空**：trait 定义完善但实际使用率低
2. **Scalar 双轨制**：mh_foundation 与 mh_physics 各有定义
3. **静态方法陷阱**：Backend 方法无 &self，GPU 无法持有设备
4. **泥沙模块断裂**：床变/悬沙/示踪剂/垂向无耦合
5. **半隐式骨架化**：Poisson 求解器缺失

## 重构目标

将项目从"实验性代码"升级为"生产级架构"，具体目标：

- ✅ Backend 强制渗透：所有物理模块接受 `<B: Backend>` 泛型
- ✅ Scalar 单一权威：`mh_physics::core::Scalar` 是唯一定义
- ✅ 策略模式统一：显式/半隐式作为 `TimeIntegrationStrategy` 实现
- ✅ AI 非侵入：通过 `Assimilable` trait 桥接
- ✅ 2.5D 外挂：`ProfileRestorer` 在 2D 求解后恢复垂向剖面

## 实施阶段

### Phase 0: 清理与根基 (Week 1)

**目标**：删除死代码，统一 Scalar，修复 Backend 静态方法

**关键改动**：
- 删除 `k_epsilon.rs` 等 3D 死代码
- 扩展 `mh_physics::core::Scalar` 添加物理常量
- Backend trait 所有方法改为实例方法
- 新建 `cpu_backend.rs` 完整实现

**验收标准**：
- `cargo check --workspace` 通过
- Backend 方法可通过实例调用

---

### Phase 1: 状态与网格泛型化 (Week 2)

**目标**：ShallowWaterState 和 MeshTopology 全面泛型化

**关键改动**：
- 新建 `ShallowWaterStateGeneric<B>`
- 新建 `MeshTopologyGeneric<B>` trait
- 保留原有类型作为别名，保持向后兼容

**验收标准**：
- f32/f64 后端可实例化状态
- 现有测试继续通过

---

### Phase 2: 求解器策略化 (Week 3-4)

**目标**：统一显式和半隐式为策略模式，完善 PCG 求解器

**关键改动**：
- 定义 `TimeIntegrationStrategy<B>` trait
- 实现 `ExplicitStrategyGeneric<B>`
- 实现 `SemiImplicitStrategyGeneric<B>`
- 新建 `PcgSolver<B>` 预条件共轭梯度求解器
- 重构 `ShallowWaterSolver` 为纯调度器

**验收标准**：
- 策略可运行时切换
- PCG 求解器收敛测试通过

---

### Phase 3: 源项与示踪剂泛型化 (Week 5)

**目标**：完成源项系统和示踪剂的 Backend 泛型化

**关键改动**：
- 定义 `SourceTermGeneric<B>` trait
- 新建 `SourceRegistry<B>` 统一管理
- 泛型化 Manning 摩擦、科氏力
- 泛型化 `TracerFieldGeneric<B>`

**验收标准**：
- 源项可注册和累加
- 示踪剂守恒量正确更新

---

### Phase 4: 泥沙系统耦合 (Week 6)

**目标**：实现 SedimentManager，闭合泥沙质量守恒

**关键改动**：
- 新建 `SedimentManager<B>` 统一管理器
- 实现侵蚀/沉降交换通量计算
- 实现 `ProfileRestorer<B>` 垂向剖面恢复
- 质量守恒校验与自动修正

**验收标准**：
- 泥沙质量守恒误差 < 1e-10
- 侵蚀/沉降物理行为正确

---

### Phase 5: AI 代理层 (Week 7)

**目标**：新建 mh_agent crate，实现 AI-物理桥接

**关键改动**：
- 新建 `mh_agent` crate
- 定义 `AIAgent` trait
- 实现 `AgentRegistry` 注册中心
- 实现 `RemoteSensingAgent` 遥感反演
- 实现 `NudgingAssimilator` 同化器
- 定义 `Assimilable` 桥接接口

**验收标准**：
- AI 代理可注册和管理
- Nudging 同化正确应用

---

### Phase 6: GPU 准备 (Week 8)

**目标**：完成 CUDA 接入准备

**关键改动**：
- 定义 `CudaBackend<S>` 骨架
- 使用 feature gate 控制 CUDA 依赖
- 定义 Kernel 接口规范
- 设计 `HybridBackend` 混合后端

**验收标准**：
- Feature gate 正确控制
- CPU fallback 正常工作

---

### Phase 7: 测试与验证 (Week 9)

**目标**：完成架构验证测试

**测试矩阵**：

| 测试 | 内容 | 标准 |
|------|------|------|
| backend_generic | f32/f64 切换 | 差异 < 1e-6 |
| strategy_switching | 策略切换 | 状态连续 |
| sediment_coupling | 泥沙守恒 | 误差 < 1e-10 |
| dambreak_generic | 溃坝算例 | L2 < 1e-3 |
| thacker_generic | Thacker 解 | 阶 ≥ 1.5 |
| ai_assimilation | AI 同化 | Nudging 正确 |

---

## 代码改动统计

| Phase | 新建 | 重构 | 删除 | 净变化 |
|-------|------|------|------|--------|
| 0 | 200 | 300 | 400 | +100 |
| 1 | 100 | 600 | 200 | +500 |
| 2 | 1500 | 800 | 300 | +2000 |
| 3 | 400 | 500 | 100 | +800 |
| 4 | 800 | 400 | 0 | +1200 |
| 5 | 600 | 100 | 0 | +700 |
| 6 | 300 | 0 | 0 | +300 |
| 7 | 500 | 0 | 0 | +500 |
| **总计** | **4400** | **2700** | **1000** | **+6100** |

## 新增文件清单

```
crates/
├── mh_physics/src/
│   ├── core/
│   │   └── cpu_backend.rs          # CpuBackend 完整实现
│   ├── state/
│   │   └── generic.rs              # 泛型状态
│   ├── mesh/
│   │   ├── topology_generic.rs     # 泛型网格拓扑
│   │   └── unstructured_generic.rs # 泛型非结构化网格
│   ├── engine/
│   │   ├── workspace_generic.rs    # 泛型工作区
│   │   ├── solver_generic.rs       # 泛型求解器
│   │   └── strategy/
│   │       ├── traits.rs           # 策略 trait
│   │       ├── explicit_generic.rs # 泛型显式策略
│   │       └── semi_implicit_generic.rs # 泛型半隐式策略
│   ├── numerics/linear_algebra/
│   │   └── pcg.rs                  # PCG 求解器
│   ├── sources/
│   │   ├── traits_generic.rs       # 泛型源项 trait
│   │   ├── registry.rs             # 源项注册中心
│   │   ├── friction_generic.rs     # 泛型摩擦
│   │   └── coriolis_generic.rs     # 泛型科氏力
│   ├── tracer/
│   │   ├── state_generic.rs        # 泛型示踪剂状态
│   │   └── transport_generic.rs    # 泛型示踪剂输运
│   ├── sediment/
│   │   ├── manager.rs              # 泥沙管理器
│   │   └── exchange.rs             # 交换通量
│   ├── vertical/
│   │   └── profile_generic.rs      # 泛型剖面恢复器
│   └── assimilation/
│       └── mod.rs                  # Assimilable 实现
│
├── mh_agent/                       # 新 crate
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── registry.rs
│       ├── remote_sensing.rs
│       ├── surrogate.rs
│       ├── observation.rs
│       └── assimilation.rs
│
└── tests/
    ├── backend_generic.rs
    ├── strategy_switching.rs
    ├── sediment_coupling.rs
    ├── dambreak_generic.rs
    ├── thacker_generic.rs
    └── ai_assimilation.rs
```

## 关键接口变更

### Backend Trait

```rust
// 旧接口（静态方法）
pub trait Backend {
    fn axpy(alpha: Self::Scalar, x: &Self::Buffer, y: &mut Self::Buffer);
}

// 新接口（实例方法）
pub trait Backend {
    fn axpy(&self, alpha: Self::Scalar, x: &Self::Buffer, y: &mut Self::Buffer);
}
```

### 状态类型

```rust
// 旧类型
pub struct ShallowWaterState { ... }

// 新类型（泛型）
pub struct ShallowWaterStateGeneric<B: Backend> { ... }

// 类型别名（向后兼容）
pub type ShallowWaterState = ShallowWaterStateGeneric<CpuBackend<f64>>;
```

### 求解器

```rust
// 旧接口
impl ShallowWaterSolver {
    pub fn step(&mut self, dt: f64) -> StepResult;
}

// 新接口（策略模式）
impl<B: Backend> ShallowWaterSolverGeneric<B> {
    pub fn step(&mut self, dt: B::Scalar) -> StepResult<B::Scalar>;
    pub fn set_strategy(&mut self, kind: StrategyKindGeneric);
}
```

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 泛型编译时间增加 | 使用 type alias 减少泛型传播 |
| f32 精度不足 | 关键路径强制 f64 |
| PCG 不收敛 | 预条件器 + 自动回退显式 |
| AI 同化破坏守恒 | ConservationEnforcer 校验 |
| GPU 内存溢出 | 分块处理 + 动态内存池 |

## 后续工作

1. **GPU Kernel 实现**：完成 P0/P1 优先级 Kernel
2. **性能优化**：SIMD 向量化、并行化
3. **更多 AI 代理**：EnKF、代理模型
4. **3D 支持**：完整 3D 求解器（低优先级）

## 文档索引

- [README.md](README.md) - 总览
- [phase0-cleanup.md](phase0-cleanup.md) - Phase 0 详细计划
- [phase1-state-mesh.md](phase1-state-mesh.md) - Phase 1 详细计划
- [phase2-solver-strategy.md](phase2-solver-strategy.md) - Phase 2 详细计划
- [phase3-sources-tracer.md](phase3-sources-tracer.md) - Phase 3 详细计划
- [phase4-sediment.md](phase4-sediment.md) - Phase 4 详细计划
- [phase5-ai-agent.md](phase5-ai-agent.md) - Phase 5 详细计划
- [phase6-gpu-prep.md](phase6-gpu-prep.md) - Phase 6 详细计划
- [phase7-testing.md](phase7-testing.md) - Phase 7 详细计划

---

*文档生成时间: 2025-12-10*
