# T03 源项系统单轨化

负责人建议：最熟悉 friction/coriolis/inflow/wave/vegetation/turbulence 等源项的人

## 1. 目标

把 `sources` 从 legacy + generic 双轨，统一成一个单轨的 generic 源项系统，并让所有源项都通过同一套 registry、context、contribution、stiffness 语义工作。

## 2. 边界文件

主要文件：

- `crates/mh_physics/src/sources/traits.rs`
- `crates/mh_physics/src/sources/registry.rs`
- `crates/mh_physics/src/sources/mod.rs`
- `crates/mh_physics/src/sources/atmosphere.rs`
- `crates/mh_physics/src/sources/coriolis.rs`
- `crates/mh_physics/src/sources/friction.rs`
- `crates/mh_physics/src/sources/implicit.rs`
- `crates/mh_physics/src/sources/inflow.rs`
- `crates/mh_physics/src/sources/vegetation.rs`
- `crates/mh_physics/src/sources/wave_forcing.rs`
- `crates/mh_physics/src/sources/structures/*`
- `crates/mh_physics/src/sources/turbulence/*`

禁止擅自修改：

- `engine/*` 的公共签名
- `types.rs` / `traits.rs` 公共接口

## 3. 必须解决的问题

### 3.1 双轨删除

- 删除或隔离 `SourceTerm` legacy 主路径。
- 保证对外只暴露 `SourceTermGeneric<B>` 及相关 generic 类型。
- 不再保留 `SourceTermF64`、兼容壳、deprecated 别名。

### 3.2 registry 统一

- `SourceRegistry` 只保留一种主实现。
- 清理伪并行接口、死字段和“有名字但无收益”的路径。
- 统一 `compute_batch` / `accumulate` / stiffness 过滤语义。

### 3.3 所有源项接入同一语义

必须统一处理：

- 显式源项
- 局部隐式源项
- 全局隐式源项标记
- 守恒/钳制/湿干过渡帮助函数

### 3.4 Legacy 实现迁移

以下模块优先迁到 generic：

- `vegetation.rs`
- `wave_forcing.rs`
- `structures/bridge_pier.rs`
- `structures/weir.rs`

这些文件当前仍带明显 f64-only 倾向。

## 4. 拆解步骤

1. 先整理 `traits.rs`：定义唯一权威接口。
2. 改 `registry.rs`，去掉双轨兼容与伪并行。
3. 逐个迁移源项实现到 generic：
   - friction
   - coriolis
   - inflow
   - atmosphere
   - vegetation
   - wave_forcing
   - structures
   - turbulence
4. 更新 `sources/mod.rs` 与 `mh_physics/src/lib.rs` 的导出。
5. 加回归测试：每类源项至少一个 batch 路径测试。

## 5. 非目标

- 不负责改 `engine` 主求解器算法
- 不负责 `tracer/sediment/waves` 内部 transport 逻辑
- 不负责 AI 层

## 6. 验收标准

- `sources/traits.rs` 不再是双轨主设计。
- `registry.rs` 没有空壳并行入口。
- 所有源项都能通过统一 generic registry 工作。
- 对外 API 不再导出 legacy 源项接口。

## 7. 预期冲突点

- 与 T02 的 `engine` 调用点
- 与 T04 的 tracer/sediment/waves 消费源项接口路径

处理方式：

- 本任务负责源项接口和实现
- 其他任务只改调用点
