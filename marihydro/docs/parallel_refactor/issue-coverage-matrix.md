# 并行任务问题覆盖矩阵

更新时间：2026-03-10

本文件把前期审查问题映射到 6 个任务，确保无遗漏。

## 1. 一级覆盖矩阵

| 问题域 | 主要问题 | 归属任务 |
| --- | --- | --- |
| 运行时抽象 | `Backend`/`Buffer`/`Scalar` 语义不稳、静默 fallback、GPU 占位 | T01 |
| 公共状态与网格 | 索引/状态/网格接口不统一、f64/usize 裸类型外溢 | T01 |
| 引擎主干 | 伪并行、假迭代、数值路径重复、CFL/半隐式链条不统一 | T02 |
| 数值离散 | CSR/PCG/重构/限制器/Riemann 路径不一致、f64 专用实现 | T02 |
| 源项系统 | `SourceTerm` 与 `SourceTermGeneric` 双轨、registry 空壳并行、legacy 壳未删 | T03 |
| 传输族 | tracer/sediment/vertical/waves 各自维护状态和边界语义，存在未完成实现 | T04 |
| AI 同化 | `mh_agent` 裸 `f64` 快照、独立 Assimilable 体系、与 physics 桥接割裂 | T05 |
| 验证与门禁 | 测试缺口、慢测试被忽略、文档失真、审计规则缺门禁 | T06 |

## 2. 用户审查报告中的重点问题映射

### 2.1 `mh_agent`

| 问题 | 任务 |
| --- | --- |
| `PhysicsSnapshot`、`Observation`、`Assimilable` 以 `f64` 为中心 | T05 |
| `NudgingAssimilator` 平滑 O(n^2) | T05 |
| `AgentRegistry` 顺序串行与守恒检查重复遍历 | T05 |
| `surrogate` 声称多模型，实际近似线性实现 | T05 |
| 遥感/观测算子与物理态耦合松散 | T05 |

### 2.2 `mh_physics/core` 与 runtime-like 能力

| 问题 | 任务 |
| --- | --- |
| `scalar_from_f64` 静默降级 | T01 |
| GPU 占位实现仍假装可用 | T01 |
| Buffer/Backend 访问路径 CPU 假设过强 | T01 |
| 索引/公共 trait 出口不统一 | T01 |

### 2.3 `engine` 与 `numerics`

| 问题 | 任务 |
| --- | --- |
| `engine/parallel.rs` 声称并行但部分路径仍伪并行 | T02 |
| `solver_builder`/策略层简化实现与真实引擎语义偏离 | T02 |
| `pcg`、`csr`、预处理器、限制器、重构器存在精度/复杂度债务 | T02 |
| Riemann 路径中 f64 专用实现残留 | T02 |
| `mesh/structured` 与离散化链条契约仍需整理 | T01 + T02 |

### 2.4 `sources`

| 问题 | 任务 |
| --- | --- |
| `SourceTerm` 与 `SourceTermGeneric` 双轨并存 | T03 |
| `registry` 中空壳并行、RefCell scratch 模式与实际策略不一致 | T03 |
| atmosphere/coriolis/friction/inflow/wave/vegetation 等源项风格不统一 | T03 |
| 结构物、湍流源项未完全服从单轨 generic 契约 | T03 |

### 2.5 `tracer/sediment/vertical/waves`

| 问题 | 任务 |
| --- | --- |
| `tracer` 边界、扩散、沉降、transport 之间接口割裂 | T04 |
| `sediment` 中 morphology/transport/resuspension/settling 路径不统一 | T04 |
| `vertical` 层状态与 mixing/profile/velocity 耦合松散 | T04 |
| `waves` 中 bottom friction / spectral / radiation stress 精度和实现成熟度不一致 | T04 |

### 2.6 测试与文档

| 问题 | 任务 |
| --- | --- |
| 慢测试被 `ignore`，长期稳定性未进入门禁 | T06 |
| 文档与实际 crate 结构不一致 | T06 |
| 审计脚本只产出候选问题，没有和 DoD 绑定 | T06 |

## 3. 本轮抽样核实发现的现实问题映射

| 现实问题 | 任务 |
| --- | --- |
| `mh_agent/src/lib.rs` 仍是裸 `f64` snapshot + `Assimilable` | T05 |
| `mh_physics/src/assimilation/mod.rs` 还维护另一套 `PhysicsAssimilable` | T05 |
| `mh_physics/src/sources/traits.rs` legacy 轨道仍存在 | T03 |
| `mh_physics/src/sources/registry.rs` 目前仍是 `Box<dyn SourceTermGeneric<B>>` 方案 | T03 |
| `mh_physics/src/core/gpu.rs` 仍是占位实现 | T01 |
| `mh_physics/src/engine/parallel.rs` 仍保留 TODO 和半成品 colored/collect 路径 | T02 |
| `mh_physics/src/mesh/structured.rs` 已有最小实现，但仍需纳入公共契约校准 | T01 / T02 |

## 4. 覆盖原则

如果某问题同时影响多个任务，按以下规则处理：

1. “接口定义”归 T01。
2. “数值主干实现”归 T02。
3. “源项抽象与所有源项调用点”归 T03。
4. “tracer/sediment/vertical/waves 业务实现”归 T04。
5. “AI 层与 physics 同化桥”归 T05。
6. “测试、文档、门禁”归 T06。
