# MariHydro 重构进度分析报告 (Phase 0-4)

**分析日期**: 2025年12月10日  
**分析范围**: Phase 0 - Phase 4 (核心架构重构)

---

## 总体进度概览

| Phase | 目标 | 完成度 | 状态 |
|-------|------|--------|------|
| Phase 0 | 清理与根基 | 85% | 🟡 基本完成 |
| Phase 1 | 状态与网格泛型化 | 30% | 🟠 部分完成 |
| Phase 2 | 求解器策略化 | 60% | 🟡 进行中 |
| Phase 3 | 源项与示踪剂泛型化 | 40% | 🟠 部分完成 |
| Phase 4 | 泥沙系统耦合 | 20% | 🔴 待完善 |

---

## Phase 0: 清理与根基 (85% 完成)

### 0.1 删除3D死代码 ✅ 完成

**计划任务**:
- 删除 `sources/turbulence/k_epsilon.rs`
- 修改 `sources/turbulence/mod.rs` 删除k_epsilon引用
- 删除 `ImplicitMethod::CrankNicolson` 变体

**实际状态**:
- ✅ `k_epsilon.rs` 文件不存在（已删除或从未创建）
- ✅ `turbulence/mod.rs` 中无 k_epsilon 引用
- ✅ `implicit.rs` 中 `ImplicitMethod` 枚举已简化

### 0.2 统一Scalar到physics ✅ 完成

**计划任务**:
- 在 `mh_physics/src/core/scalar.rs` 定义完整 Scalar trait
- `mh_foundation/src/scalar.rs` 改为重导出或兼容层

**实际状态**:
- ✅ `mh_physics::core::Scalar` 已定义为权威版本
  - 包含物理常量: `GRAVITY`, `VON_KARMAN`, `WATER_DENSITY`, `SEAWATER_DENSITY`, `AIR_DENSITY`
  - 包含完整数学运算: `sqrt`, `abs`, `max`, `min`, `clamp`, `powf`, `exp`, `ln`, `sin`, `cos`, `atan2`, `signum`, `floor`, `ceil`
  - 支持 f32/f64 两种精度
- ✅ `mh_foundation::scalar` 保留为兼容层
  - 注释明确推荐使用 `mh_physics::core::Scalar`
  - `ScalarOps` trait 保留但功能较简单
  - 物理常量通过 `constants` 模块提供

**差异说明**:
- 计划中要求 `mh_foundation` 重导出 `mh_physics::core::Scalar`，但实际采用了独立兼容层方案
- 原因：避免循环依赖问题（mh_foundation 是基础层，不应依赖 mh_physics）
- 这是合理的架构决策

### 0.3 Backend改为实例方法 ✅ 完成

**计划任务**:
- Backend trait 所有方法添加 `&self`
- 新建 `CpuBackend<f32/f64>` 完整实现

**实际状态**:
- ✅ `Backend` trait 已改为实例方法设计
  - 所有操作通过 `&self` 调用
  - 支持 GPU 后端持有设备句柄
- ✅ `CpuBackend<f32>` 和 `CpuBackend<f64>` 分别实现
  - 包含 BLAS Level 1 算子: `axpy`, `dot`, `copy`, `scale`
  - 包含归约操作: `reduce_max`, `reduce_min`, `reduce_sum`, `norm2`
  - 包含物理专用算子: `enforce_positivity`, `elementwise_mul`, `elementwise_div_safe`
- ✅ `DefaultBackend = CpuBackend<f64>` 类型别名已定义
- ✅ 完整的单元测试覆盖

**额外完成**:
- 添加了 `reduce_min` 和 `norm2` 方法（计划中未提及）
- 添加了 `elementwise_mul` 和 `elementwise_div_safe` 方法

### Phase 0 遗留问题

1. ⚠️ `mh_foundation/src/memory.rs` 中 `AlignedVec` 未标记 `#[deprecated]`
   - 计划要求标记废弃，推荐使用 `DeviceBuffer`
   - 实际：仍在广泛使用（如 `ShallowWaterState`）

---

## Phase 1: 状态与网格泛型化 (30% 完成)

### 1.1 创建泛型状态类型 🔴 未完成

**计划任务**:
- 新建 `state/generic.rs` 实现 `ShallowWaterStateGeneric<B>`
- 修改 `state.rs` 添加类型别名

**实际状态**:
- ❌ `ShallowWaterStateGeneric<B>` 未创建
- ❌ 现有 `ShallowWaterState` 仍使用 `AlignedVec<f64>` 硬编码
- ❌ 无泛型状态类型别名

**现有实现分析**:
```rust
// 当前 state.rs 中的实现
pub struct ShallowWaterState {
    n_cells: usize,
    pub h: AlignedVec<f64>,   // 硬编码 f64
    pub hu: AlignedVec<f64>,
    pub hv: AlignedVec<f64>,
    pub z: AlignedVec<f64>,
    pub tracers: DynamicScalars,
    pub field_registry: FieldRegistry,
}
```

**需要完成**:
1. 创建 `ShallowWaterStateGeneric<B: Backend>` 结构体
2. 使用 `B::Buffer<B::Scalar>` 替代 `AlignedVec<f64>`
3. 添加类型别名保持向后兼容

### 1.2 网格拓扑泛型化 🟡 部分完成

**计划任务**:
- 新建 `mesh/topology_generic.rs` 定义 `MeshTopologyGeneric<B>`
- 新建 `mesh/unstructured_generic.rs` 实现泛型适配器

**实际状态**:
- ✅ `MeshTopology<B: Backend>` trait 已定义（在 `topology.rs` 中）
  - 包含泛型参数 `<B: Backend>`
  - 定义了 `cell_areas_buffer()` 和 `face_lengths_buffer()` 返回 `&B::Buffer<B::Scalar>`
- ❌ 但实际实现 `UnstructuredMeshAdapter` 未使用泛型
- ❌ 缺少 `UnstructuredMeshGeneric<B>` 实现

**现有实现分析**:
```rust
// topology.rs 中的 trait 定义（已泛型化）
pub trait MeshTopology<B: Backend>: Send + Sync {
    fn n_cells(&self) -> usize;
    fn n_faces(&self) -> usize;
    // ...
    fn cell_areas_buffer(&self) -> &B::Buffer<B::Scalar>;
    fn face_lengths_buffer(&self) -> &B::Buffer<B::Scalar>;
}
```

**需要完成**:
1. 实现 `UnstructuredMeshGeneric<B>` 结构体
2. 为现有 `UnstructuredMeshAdapter` 添加泛型支持

---

## Phase 2: 求解器策略化 (60% 完成)

### 2.1 工作区泛型化 ✅ 完成

**计划任务**:
- 新建 `engine/workspace_generic.rs` 实现 `SolverWorkspaceGeneric<B>`

**实际状态**:
- ✅ `SolverWorkspaceGeneric<B>` 已在 `engine/strategy/workspace.rs` 中实现
- ✅ 从 `engine/mod.rs` 导出

### 2.2 时间积分策略Trait ✅ 完成

**计划任务**:
- 定义 `TimeIntegrationStrategy<B>` trait

**实际状态**:
- ✅ `TimeIntegrationStrategy` trait 已定义
- ✅ `StepResult` 结构体已定义
- ✅ `StrategyKind` 枚举已定义
- ✅ 从 `engine/mod.rs` 导出

### 2.3 显式策略重构 ✅ 完成

**计划任务**:
- 实现 `ExplicitStrategyGeneric<B>`

**实际状态**:
- ✅ `ExplicitStrategy` 已实现（在 `engine/strategy/explicit.rs`）
- ✅ `ExplicitConfig` 配置结构已定义
- ✅ 从 `engine/mod.rs` 导出

### 2.4 PCG求解器实现 ✅ 完成

**计划任务**:
- 新建 `numerics/linear_algebra/pcg.rs` 实现 PCG 求解器

**实际状态**:
- ✅ `PcgSolver` 已实现（在 `engine/pcg.rs`）
- ✅ `PcgConfig`, `PcgResult`, `PcgWorkspace` 已定义
- ✅ `PreconditionerType` 枚举已定义
- ✅ `SparseMvp`, `DiagonalMatrix`, `CsrMatrix` 已实现
- ✅ `PoissonMatrixBuilder` 已实现
- ✅ 从 `engine/mod.rs` 导出

### 2.5 半隐式策略完善 ✅ 完成

**计划任务**:
- 实现 `SemiImplicitStrategyGeneric<B>` 集成 PCG

**实际状态**:
- ✅ `SemiImplicitStrategy` 已实现（在 `engine/semi_implicit.rs`）
- ✅ `SemiImplicitStrategyGeneric` 泛型版本已实现（在 `engine/strategy/semi_implicit.rs`）
- ✅ `SemiImplicitConfig`, `SemiImplicitStats` 已定义
- ✅ 从 `engine/mod.rs` 导出

### 2.6 统一求解器调度 🟡 部分完成

**计划任务**:
- 重构 `ShallowWaterSolver` 为纯调度器

**实际状态**:
- ✅ `ShallowWaterSolver` 存在且功能完整
- ⚠️ 但未完全重构为纯调度器模式
- ⚠️ 仍包含大量直接计算逻辑（如 `compute_fluxes_serial`, `compute_fluxes_parallel`）
- ❌ 未实现运行时策略切换 `set_strategy()` 方法

**需要完成**:
1. 将通量计算逻辑移至策略实现
2. 添加 `set_strategy()` 方法支持运行时切换
3. Solver 应仅负责调度，不包含具体算法

---

## Phase 3: 源项与示踪剂泛型化 (40% 完成)

### 3.1 源项Trait重构 ✅ 完成

**计划任务**:
- 新建 `sources/traits_generic.rs` 定义 `SourceTermGeneric<B>`
- 新建 `sources/registry.rs` 实现 `SourceRegistry<B>`

**实际状态**:
- ✅ `SourceTermGeneric<B>` trait 已定义（在 `sources/traits_generic.rs`）
- ✅ `SourceContributionGeneric<S>` 已定义
- ✅ `SourceContextGeneric<S>` 已定义
- ✅ `SourceStiffness` 枚举已定义
- ✅ `SourceRegistryGeneric<B>` 已实现
- ✅ 从 `sources/mod.rs` 导出

### 3.2 摩擦源项泛型化 ✅ 完成

**计划任务**:
- 实现 `ManningFrictionGeneric<B>`

**实际状态**:
- ✅ `ManningFrictionGeneric<B>` 已实现（在 `sources/friction_generic.rs`）
- ✅ `ManningFrictionConfigGeneric<S>` 已定义
- ✅ `ChezyFrictionGeneric<B>` 也已实现
- ✅ 从 `sources/mod.rs` 导出

### 3.3 科氏力源项泛型化 🔴 未完成

**计划任务**:
- 实现 `CoriolisSourceGeneric<B>`

**实际状态**:
- ❌ `CoriolisSourceGeneric<B>` 未实现
- ✅ 非泛型版本 `CoriolisSource` 存在

**需要完成**:
1. 创建 `coriolis_generic.rs`
2. 实现 `CoriolisSourceGeneric<B>`

### 3.4 示踪剂泛型化 🔴 未完成

**计划任务**:
- 新建 `tracer/state_generic.rs` 实现 `TracerFieldGeneric<B>`
- 新建 `tracer/transport_generic.rs`

**实际状态**:
- ❌ `TracerFieldGeneric<B>` 未实现
- ❌ `TracerStateGeneric<B>` 未实现
- ✅ 非泛型版本 `TracerField`, `TracerState` 存在

**需要完成**:
1. 创建泛型示踪剂状态
2. 创建泛型示踪剂输运

---

## Phase 4: 泥沙系统耦合 (20% 完成)

### 4.1 SedimentManager实现 🔴 未完成

**计划任务**:
- 新建 `sediment/manager.rs` 实现统一管理器

**实际状态**:
- ✅ `sediment/manager.rs` 文件存在
- ⚠️ 但实现可能不完整（需要进一步检查）
- ❌ 未确认是否实现了泛型版本 `SedimentManager<B>`

### 4.2 侵蚀/沉降交换模块 🔴 未完成

**计划任务**:
- 新建 `sediment/exchange.rs`

**实际状态**:
- ❌ `sediment/exchange.rs` 文件不存在
- ⚠️ 交换通量计算可能分散在其他文件中

### 4.3 ProfileRestorer实现 🔴 未完成

**计划任务**:
- 新建 `vertical/profile_generic.rs` 实现垂向剖面恢复器

**实际状态**:
- ✅ `vertical/profile.rs` 文件存在
- ❌ 未确认是否实现了泛型版本 `ProfileRestorer<B>`
- ❌ 未确认是否实现了 Rouse 分布恢复

---

## 关键差距分析

### 高优先级待完成项

1. **状态泛型化** (Phase 1.1)
   - 影响：阻塞所有依赖泛型状态的功能
   - 工作量：中等（约2-3天）

2. **网格适配器泛型化** (Phase 1.2)
   - 影响：阻塞泛型求解器的完整实现
   - 工作量：中等（约2天）

3. **求解器调度器重构** (Phase 2.6)
   - 影响：无法实现运行时策略切换
   - 工作量：较大（约3-4天）

4. **示踪剂泛型化** (Phase 3.4)
   - 影响：阻塞泥沙系统的泛型化
   - 工作量：中等（约2天）

### 中优先级待完成项

1. 科氏力泛型化 (Phase 3.3)
2. SedimentManager 泛型化 (Phase 4.1)
3. 侵蚀/沉降交换模块 (Phase 4.2)
4. ProfileRestorer 泛型化 (Phase 4.3)

### 低优先级/可延后项

1. AlignedVec 废弃标记
2. 其他源项泛型化（大气、植被等）

---

## 建议的下一步行动

### 短期（1-2周）

1. **完成 Phase 1**：状态和网格泛型化
   - 创建 `ShallowWaterStateGeneric<B>`
   - 创建 `UnstructuredMeshGeneric<B>`
   - 添加类型别名保持兼容

2. **完善 Phase 2**：求解器调度器
   - 重构 `ShallowWaterSolver` 为纯调度器
   - 实现 `set_strategy()` 方法

### 中期（2-4周）

3. **完成 Phase 3**：示踪剂泛型化
   - 创建 `TracerFieldGeneric<B>`
   - 创建 `TracerStateGeneric<B>`
   - 科氏力泛型化

4. **完成 Phase 4**：泥沙系统
   - 完善 `SedimentManager<B>`
   - 创建 `exchange.rs`
   - 完善 `ProfileRestorer<B>`

---

## 代码质量评估

### 优点

1. ✅ Backend trait 设计合理，实例方法支持 GPU 扩展
2. ✅ Scalar trait 包含丰富的物理常量
3. ✅ PCG 求解器实现完整
4. ✅ 策略模式框架已建立
5. ✅ 源项泛型化框架已建立

### 需改进

1. ⚠️ 状态类型未泛型化，是最大的技术债务
2. ⚠️ 求解器仍包含过多直接计算逻辑
3. ⚠️ 泛型和非泛型版本并存，增加维护成本
4. ⚠️ 部分模块缺少泛型实现

---

*报告生成时间: 2025-12-10*
