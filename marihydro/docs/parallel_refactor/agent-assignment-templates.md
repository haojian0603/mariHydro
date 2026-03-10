# 多 Agent 分工模板

更新时间：2026-03-10

本文件提供可直接复制给 agent 的启动模板。

使用方式：

1. 先复制“通用前置说明”
2. 再追加对应任务的专属模板
3. 如有必要，再追加你自己的额外约束

---

## A. 通用前置说明模板

```md
你现在负责 MariHydro 并行重构中的一个独立任务包。

先阅读以下文档：
1. docs/parallel_refactor/README.md
2. docs/parallel_refactor/issue-coverage-matrix.md
3. docs/parallel_refactor/coordination-protocol.md
4. 你的任务文档

工作要求：
- 你会看到完整代码问题清单，但你只能处理自己任务范围内的问题。
- 默认只改自己任务边界内的文件。
- 如果必须跨边界改共享文件，先产出“跨边界说明”，不要直接扩散修改。
- 每修复一部分就必须按固定格式产出一次“变更播报”，写清楚：
  - 改了哪些文件
  - 改了什么
  - 为什么这样改
  - 是否影响其他任务
  - 是否改了 public 接口
- 如果你发现另一种可行方案，也要在播报中写出来，供后续选择。

禁止事项：
- 不要新增第二套兼容接口。
- 不要为了省事重新引入 CpuBackend<f64> 专用主路径。
- 不要把未完成实现伪装成可用主路径。
- 不要无说明地修改共享文件的 public 签名。

你的输出必须包括：
1. 你对本任务的理解
2. 你准备怎么分阶段做
3. 第一阶段准备修改哪些文件
4. 每一阶段完成后的“变更播报”
```

---

## B. T01 模板：运行时与公共契约冻结

```md
你负责 T01：运行时与公共契约冻结。

你的任务文档：
- docs/parallel_refactor/task-01-runtime-and-contracts.md

你的核心目标：
- 冻结 Backend / Scalar / Buffer / StateAccess / MeshTopology / PhysicsMesh 的公共契约
- 修掉高风险静默 fallback
- 把 GPU 占位行为改成明确语义
- 让后续任务不再修改这些公共签名

你必须优先核对的文件：
- crates/mh_physics/src/core/backend.rs
- crates/mh_physics/src/core/buffer.rs
- crates/mh_physics/src/core/scalar.rs
- crates/mh_physics/src/core/gpu.rs
- crates/mh_physics/src/types.rs
- crates/mh_physics/src/traits.rs
- crates/mh_physics/src/mesh/topology.rs
- crates/mh_physics/src/adapter.rs
- crates/mh_physics/src/lib.rs

你需要按下面顺序推进：
1. 先梳理 public trait/struct/re-export 清单。
2. 先修错误语义，再修接口组织。
3. 最后再动导出层和文档注释。

你要重点解决的问题：
- scalar_from_f64 等静默回零/静默 fallback
- try_as_slice 系列的运行时语义
- gpu.rs 占位实现语义
- 公共索引/状态/网格接口出口不统一
- lib.rs 重导出和现实架构不一致

你每次播报时必须额外写清楚：
- 哪些 public trait 签名变了
- 哪些调用点任务需要跟进
- 哪些原有默认行为被改了
```

---

## C. T02 模板：引擎与数值主干统一

```md
你负责 T02：引擎与数值主干统一。

你的任务文档：
- docs/parallel_refactor/task-02-engine-and-numerics.md

你的核心目标：
- 统一 engine / numerics / schemes 主干
- 清理伪并行、假迭代、重复路径
- 保证显式/半隐式/线性代数/重构/限制器/Riemann 在同一套契约上工作

你的边界文件：
- crates/mh_physics/src/engine/*
- crates/mh_physics/src/numerics/*
- crates/mh_physics/src/schemes/*

你必须先确认：
- T01 是否已经冻结公共接口
- 你不会自行修改 core/types/traits 的 public 签名

你推荐的实施顺序：
1. 先画出 engine 主调用链
2. 再收敛 parallel 路径
3. 再统一 linear_algebra / discretization
4. 最后处理 reconstruction / limiter / Riemann

你要重点解决的问题：
- engine/parallel 中半成品并行策略
- solver / strategy / timestep / integrator 的重复职责
- numerics 中不同模块对精度/状态接口的割裂
- schemes 中仍不一致的 Riemann 路径

你每次播报时必须额外写清楚：
- 本轮修复落在哪条求解链上
- 是否改变了默认数值路径
- 并行路径现在到底保留哪些、删除哪些、降级哪些
```

---

## D. T03 模板：源项系统单轨化

```md
你负责 T03：源项系统单轨化。

你的任务文档：
- docs/parallel_refactor/task-03-sources-single-track.md

你的核心目标：
- 删掉 SourceTerm 与 SourceTermGeneric 双轨长期并存状态
- 统一 registry / context / contribution / stiffness 语义
- 让所有源项通过同一套 generic 接口工作

你的边界文件：
- crates/mh_physics/src/sources/*

你的实施顺序：
1. 先处理 traits.rs
2. 再处理 registry.rs
3. 再逐个迁移具体源项
4. 最后更新 mod.rs / lib.rs 导出

你要重点解决的问题：
- legacy SourceTerm 主路径仍存在
- registry 里伪并行和死代码
- vegetation / wave_forcing / structures 仍偏 f64-only
- turbulence 路径要和统一接口对齐

你每次播报时必须额外写清楚：
- 本轮是否删除了旧接口
- 哪些源项已经接入新接口
- 哪些调用点还要别的任务跟进
```

---

## E. T04 模板：传输物理族统一

```md
你负责 T04：传输物理族统一。

你的任务文档：
- docs/parallel_refactor/task-04-transport-physics-family.md

你的核心目标：
- 统一 tracer / sediment / vertical / waves 四个家族
- 去掉每个家族各自维护一套状态和边界语义的分裂状态
- 补齐或降级未完成实现

你的边界文件：
- crates/mh_physics/src/tracer/*
- crates/mh_physics/src/sediment/*
- crates/mh_physics/src/vertical/*
- crates/mh_physics/src/waves/*

推荐顺序：
1. tracer
2. sediment
3. vertical
4. waves

你要重点解决的问题：
- tracer 的 boundary / diffusion / settling / transport 不统一
- sediment 的 morphology / suspended / bed_load 语义割裂
- vertical 的层状态与 profile/velocity/mixing 耦合松散
- waves 中模块成熟度差异大，存在 f64-only 主路径和实验代码

你每次播报时必须额外写清楚：
- 本轮统一的是哪个家族
- 是补齐实现还是降级/隔离实验路径
- 是否依赖 T03 的源项接口变化
```

---

## F. T05 模板：AI 与同化桥接重构

```md
你负责 T05：AI 与同化桥接重构。

你的任务文档：
- docs/parallel_refactor/task-05-ai-and-assimilation.md

你的核心目标：
- 统一 mh_agent 与 mh_physics 的同化接口
- 去掉两边各有一套 Assimilable / Snapshot / Conservation 桥的割裂设计
- 让 AI 主路径不再长期围绕裸 f64 快照展开

你的边界文件：
- crates/mh_agent/src/*
- crates/mh_physics/src/assimilation/*

推荐顺序：
1. 先统一接口归属
2. 再统一快照模型
3. 再改 registry / nudging / conservation
4. 最后收 remote_sensing / observation / surrogate

你要重点解决的问题：
- mh_agent::Assimilable 与 mh_physics::PhysicsAssimilable 双重接口
- PhysicsSnapshot / StateSnapshot / Observation 职责重叠
- Nudging O(n^2) 平滑写死在主路径
- surrogate / remote_sensing 对外语义与实际实现不一致

你每次播报时必须额外写清楚：
- 这轮是否改了 AI 与 physics 的接口边界
- 是否需要 T04 调整 tracer/sediment 接入点
- 是否改变了守恒校验流程
```

---

## G. T06 模板：测试、文档与门禁收口

```md
你负责 T06：测试、文档与门禁收口。

你的任务文档：
- docs/parallel_refactor/task-06-tests-docs-and-gates.md

你的核心目标：
- 把前 5 个任务沉淀为测试、文档和审计门禁
- 明确什么是快速门禁，什么是慢回归，什么是基准测试
- 修正文档与现实架构不一致的问题

你的边界文件：
- crates/mh_physics/tests/*
- tests/*
- docs/*
- scripts/architecture_audit.ps1

推荐顺序：
1. 先整理现有测试矩阵
2. 再定义门禁层次
3. 再补文档
4. 最后升级审计脚本

你要重点解决的问题：
- 慢测试长期被 ignore
- 文档继续引用失效 crate 结构
- 审计脚本只是报告，没有绑定 DoD

你每次播报时必须额外写清楚：
- 新增或修复了哪些门禁
- 哪些测试依赖前面任务落地后再接入
- 哪些旧文档被判定为失真并已更新
```

---

## H. 建议你额外附给每个 agent 的一句话

```md
你会看到完整问题清单，但你不是“全仓修复 agent”，你是“单任务闭环 agent”。
先把自己的任务边界做实，再通过变更播报去协调交叉调用点。
```
