# T05 AI 与同化桥接重构

负责人建议：最熟悉 `mh_agent` 与 `mh_physics/assimilation` 双边接口的人

## 1. 目标

把 AI 层和 physics 层的同化接口统一起来，消除当前“两边各有一套 Assimilable/快照/守恒桥”的架构断层。

## 2. 边界文件

主要文件：

- `crates/mh_agent/src/lib.rs`
- `crates/mh_agent/src/assimilation.rs`
- `crates/mh_agent/src/registry.rs`
- `crates/mh_agent/src/observation.rs`
- `crates/mh_agent/src/remote_sensing.rs`
- `crates/mh_agent/src/surrogate.rs`
- `crates/mh_physics/src/assimilation/mod.rs`
- `crates/mh_physics/src/assimilation/bridge.rs`
- `crates/mh_physics/src/assimilation/conservation.rs`

可联动文件：

- `crates/mh_physics/src/tracer/state.rs`
- `crates/mh_physics/src/state.rs`

## 3. 必须解决的问题

### 3.1 双重同化接口

当前存在：

- `mh_agent::Assimilable`
- `mh_physics::assimilation::PhysicsAssimilable`

这两套接口不能长期并存。必须决定：

- 由 physics 提供唯一权威同化接口，然后 agent 消费；或
- 由 agent 定义接口，physics 只实现桥接

建议采用前者：物理状态接口由 physics 持有，agent 只消费。

### 3.2 快照与观测模型

- 清理 `PhysicsSnapshot`、`Observation`、`StateSnapshot` 的重复职责。
- 决定哪些字段保留 `f64`，哪些进入 generic 运行时主路径。
- 去掉裸 `Vec<f64>` + 裸 `[f64; 2]` 作为长期主模型。

### 3.3 AI 注册中心与守恒

- `AgentRegistry` 的更新、应用、守恒检查逻辑要和 physics 同化桥接一致。
- 避免重复做总量扫描、重复做守恒纠偏。

### 3.4 当前实现缺陷

- `NudgingAssimilator` 的 O(n^2) 平滑必须重新设计：
  - 至少拆成可替换的平滑策略
  - 最好降到邻域或索引驱动
- `surrogate` 的“声明能力”必须与真实实现一致
- `remote_sensing` 和 `observation` 模块需与 physics 实体语义对齐

## 4. 拆解步骤

1. 统一同化接口归属。
2. 统一快照模型、守恒检查模型、桥接模型。
3. 重构 `mh_agent` registry 与 nudging 主路径。
4. 修正 `surrogate` / `remote_sensing` / `observation` 的对外语义。
5. 加回归：
   - observation 接入
   - nudging 应用
   - 守恒校验
   - physics 状态桥接

## 5. 非目标

- 不负责改引擎主求解器
- 不负责 sources 抽象
- 不负责 transport 主实现

## 6. 验收标准

- `mh_agent` 与 `mh_physics` 之间只有一套权威同化接口。
- 不再长期依赖裸 `f64` 快照作为主路径。
- `NudgingAssimilator` 不再把 O(n^2) 平滑写死在核心流程中。
- AI 模块对外声称的能力与实现一致。

## 7. 预期冲突点

- 与 T01 在状态/网格公共接口处冲突
- 与 T04 在 tracer/sediment 同化入口处冲突

处理方式：

- T01 定义状态公共契约
- T05 负责同化接口统一与上层消费
