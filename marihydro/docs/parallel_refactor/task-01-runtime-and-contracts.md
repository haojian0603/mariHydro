# T01 运行时与公共契约冻结

负责人建议：最熟悉 Rust trait、泛型边界、crate 依赖关系的人

## 1. 目标

冻结整个重构阶段要依赖的公共接口，避免后续任务互相修改底座。

本任务负责：

- `Backend` / `Scalar` / `DeviceBuffer` 契约
- 公共索引与状态访问 trait
- 网格访问公共接口
- physics 对 runtime-like 能力的统一出口
- 生产占位实现的错误语义

## 2. 边界文件

主要文件：

- `crates/mh_physics/src/core/backend.rs`
- `crates/mh_physics/src/core/buffer.rs`
- `crates/mh_physics/src/core/scalar.rs`
- `crates/mh_physics/src/core/gpu.rs`
- `crates/mh_physics/src/core/mod.rs`
- `crates/mh_physics/src/types.rs`
- `crates/mh_physics/src/traits.rs`
- `crates/mh_physics/src/mesh/topology.rs`
- `crates/mh_physics/src/adapter.rs`
- `crates/mh_physics/src/lib.rs`

可联动文件：

- `crates/mh_physics/src/mesh/structured.rs`
- `crates/mh_physics/src/mesh/unstructured.rs`

禁止擅自深入修改：

- `sources/*`
- `tracer/*`
- `sediment/*`
- `vertical/*`
- `waves/*`
- `mh_agent/*`

## 3. 必须解决的问题

### 3.1 Backend/Buffer 错误语义

- 修正 `scalar_from_f64` 的静默回零/静默降级问题。
- 明确 `try_as_slice` / `try_as_slice_mut` 的语义：可访问、不可访问、需要同步，至少要统一错误处理路径。
- 明确 GPU 占位实现是否：
  - 只允许返回结构化错误；或
  - 只在 feature 下暴露；或
  - 降为明确的 CPU fallback 类型，不再假装 GPU。

### 3.2 公共 trait 冻结

- 冻结 `StateAccess` / `StateAccessMut`
- 冻结 `MeshTopology`
- 冻结公共索引导出方式
- 冻结 `PhysicsMesh` 对外能力边界

### 3.3 精度和索引原则

- 禁止公共接口继续向外暴露裸 `f64` / `usize`，除非该值明确属于配置层或 I/O 层。
- 明确哪些几何量保留 `f64`，哪些必须是 `B::Scalar`。
- 统一 `CellIndex` / `FaceIndex` / `NodeIndex` 的出口和使用规范。

### 3.4 占位与实验能力

- `core/gpu.rs` 必须从“模糊占位”变成“明确不可用”或“明确 CPU fallback”。
- `structured.rs` 虽然已有最小实现，但要校验是否满足 `MeshTopology` 的最小公共契约，不满足则补齐。

## 4. 拆解步骤

1. 审核 `core/*` 当前 public API，列出将冻结的签名。
2. 修复 `Backend` / `Buffer` 的高风险静默行为。
3. 整理 `types.rs` 与 `traits.rs` 中的公共状态接口，删除多余别名和重复出口。
4. 整理 `mesh/topology.rs` 与 `adapter.rs`，统一 mesh 公共访问出口。
5. 更新 `mh_physics/src/lib.rs` 的重导出，使其符合当前架构现实。
6. 给出“公共契约冻结说明”，供 T02-T05 消费。

## 5. 非目标

- 不负责重写引擎算法。
- 不负责改源项业务逻辑。
- 不负责 AI/同化逻辑。
- 不负责 transport 业务模块内部实现。

## 6. 输出物

- 稳定的公共 trait / type / re-export 出口
- 结构化错误替代静默 fallback
- 一份简短的接口冻结清单（可写入 PR 描述或附属 md）

## 7. 验收标准

- 不再出现“转换失败后回零继续跑”的主路径语义。
- `core/gpu.rs` 不再是假装可用的生产实现。
- `mh_physics/src/lib.rs` 的重导出与实际模块边界一致。
- 后续任务不需要再改这些文件的签名。

## 8. 预期冲突点

- 与 T02 的 `engine/*` 调用点冲突
- 与 T03 的 `SourceContext` / `StateAccess` 调用点冲突
- 与 T05 的 `AssimilableBridge` / `PhysicsMesh` 调用点冲突

处理规则：

- 本任务负责签名
- 其他任务只负责修调用点
