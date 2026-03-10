# 多 Agent 协作与变更播报协议

更新时间：2026-03-10

本文件用于规范多个 agent 在同一轮重构中的执行方式，减少交叉文件冲突、重复修复和方案分叉。

## 1. 使用方式

每个 agent 开工前必须阅读：

1. `docs/parallel_refactor/README.md`
2. `docs/parallel_refactor/issue-coverage-matrix.md`
3. 自己对应的 `task-0x-*.md`
4. 本文件

## 2. 执行原则

1. 一个 agent 只对一个任务包负责。
2. 默认只改自己任务边界内的文件。
3. 公共接口文件由 T01 拥有，其他任务只能消费。
4. 如果遇到必须跨边界改动的情况，要先产出“跨边界说明”，再实施。
5. 每次改动以“一个可解释的小闭环”为单位，不要攒一大批修改后再统一汇报。

## 3. 工作节奏

建议每个 agent 采用以下节奏：

1. 先做 30-60 分钟的局部梳理。
2. 确认本轮只做一个子目标。
3. 完成一个子目标后立即产出一次“变更播报”。
4. 再进入下一个子目标。

不要一次性同时推进多个共享文件上的大改。

## 4. 共享文件规则

以下文件属于共享文件，高冲突风险：

- `crates/mh_physics/src/lib.rs`
- `crates/mh_physics/src/types.rs`
- `crates/mh_physics/src/traits.rs`
- `crates/mh_physics/src/adapter.rs`
- `crates/mh_physics/src/state.rs`
- `crates/mh_physics/src/sources/traits.rs`
- `crates/mh_physics/src/sources/registry.rs`
- `crates/mh_agent/src/lib.rs`
- `crates/mh_physics/src/assimilation/mod.rs`

规则：

1. T01 可直接改公共契约文件。
2. T03 可直接改 `sources/traits.rs`、`sources/registry.rs`。
3. T05 可直接改 `mh_agent/src/lib.rs`、`mh_physics/src/assimilation/*`。
4. 其他任务若必须改共享文件，只能做“调用点修复”，不得自行扩展接口。

## 5. 每修一部分必须播报的内容

每个 agent 在完成一个阶段性修复后，必须用固定格式汇报。

### 5.1 最小播报模板

```md
## 变更播报

### 本轮目标
- 修复的子问题：

### 修改文件
- 路径1
- 路径2

### 修改内容
- 做了什么
- 为什么这样做

### 接口/行为变化
- 对外签名是否变化
- 默认行为是否变化
- 是否影响其他任务调用点

### 备选方案
- 方案A：为什么不用
- 方案B：为什么采用当前方案

### 后续影响
- 哪些 agent 需要同步更新调用点
- 哪些文档/测试需要跟进
```

### 5.2 必须额外说明的情况

出现以下情况时，播报必须单独标红说明：

1. 修改了共享文件
2. 改了 public trait 或 public struct 字段
3. 删除了兼容层
4. 改了默认数值行为
5. 改了测试基线

## 6. 方案分歧处理

如果多个 agent 在交叉代码上出现不同解法，不要各自继续推进。

处理方式：

1. 各自写出“变更播报”。
2. 明确列出：
   - 方案目标
   - 修改文件
   - 接口影响
   - 风险
3. 由总控或人工选择方案。
4. 未被选中的方案停止继续扩散。

## 7. 跨边界说明模板

如果某任务必须跨任务边界改文件，先提交如下说明：

```md
## 跨边界说明

### 所属任务
- T0X

### 需要跨改的文件
- 路径

### 原因
- 为什么本任务无法只在边界内完成

### 修改类型
- 调用点修复 / 接口扩展 / 行为修正

### 风险
- 可能影响的任务

### 建议处理方式
- 直接改
- 转给 T01
- 暂缓，等待上游接口稳定
```

## 8. 子任务粒度建议

推荐每个 agent 的子任务粒度如下：

- 一个 public trait 重构
- 一个 registry 统一
- 一个 solver 主链修复
- 一个 transport 子模块统一
- 一个边界系统收口
- 一组明确相关的测试修复

不建议：

- 一次重写整个 `engine`
- 一次重写整个 `tracer`
- 一次同时改 3 个共享文件并重构接口

## 9. 合并前自检

每个 agent 在提交前至少回答下面 8 个问题：

1. 我是否只完成了自己任务包负责的内容？
2. 我是否修改了共享文件？
3. 如果修改了共享文件，是否明确说明了接口变化？
4. 我是否删除了兼容层，而不是又加了一层兼容？
5. 我是否引入了新的 `CpuBackend<f64>` 专用路径？
6. 我是否补了最少必要测试？
7. 我是否更新了与本变更直接相关的文档/注释？
8. 我的方案是否需要其他 agent 改调用点？

## 10. 推荐的提交消息格式

```text
T0X: 简述本轮修复目标
```

示例：

- `T01: unify backend error semantics`
- `T03: remove legacy source term path`
- `T05: merge agent and physics assimilation interfaces`
