# T02 引擎与数值主干统一

负责人建议：最熟悉时间积分、线性求解、重构、限制器、Riemann 求解器的人

## 1. 目标

把 `engine`、`numerics`、`schemes` 的主干统一到单轨 generic 实现，清理伪并行、假迭代、重复路径和半成品算法。

## 2. 边界文件

主要文件：

- `crates/mh_physics/src/engine/*`
- `crates/mh_physics/src/engine/strategy/*`
- `crates/mh_physics/src/numerics/*`
- `crates/mh_physics/src/schemes/*`

可联动文件：

- `crates/mh_physics/src/mesh/structured.rs`
- `crates/mh_physics/src/mesh/unstructured.rs`

禁止擅自改签名：

- `core/*`
- `types.rs`
- `traits.rs`
- `mesh/topology.rs`

这些若需要变更，回推给 T01。

## 3. 必须解决的问题

### 3.1 并行路径收敛

- 清理 `engine/parallel.rs` 中“文档说真并行、实现仍半成品”的路径。
- 对 `CollectThenAccumulate`、`Colored`、`Auto` 给出明确选择：
  - 保留并补齐
  - 降级并删文档承诺
  - feature 隔离

### 3.2 时间积分与半隐式统一

- 校正 `explicit` / `semi_implicit` / `timestep` / `time_integrator` 的职责边界。
- 清理“假迭代”“占位迭代”“仅为框架存在但没有真实数学意义”的实现。
- 保证显式、局部隐式、全局隐式的入口语义一致。

### 3.3 线性代数和离散化主干

- 审核 CSR、PCG、预处理器、assembler、back_sub 的主路径。
- 删除与当前主求解链无关的重复入口。
- 确保压力矩阵、回代、CFL、通量累加都接到同一套运行时抽象上。

### 3.4 重构器、限制器、Riemann 统一

- `green_gauss`、`least_squares`、`muscl`、`weno`、限制器、Riemann 求解器必须在同一套精度/状态接口上运行。
- 去掉与 generic 主干不一致的 f64 专用路径。
- 删除“声称支持但仅占位”的算法说明。

## 4. 拆解步骤

1. 整理 `engine` 的主调用链：solver -> parallel -> strategy -> integrator -> numerics。
2. 确定真实支持的并行策略，删掉伪入口或补齐实现。
3. 统一 `numerics/discretization`、`linear_algebra`、`reconstruction`、`limiter` 与 `schemes` 的数据流。
4. 补测试：
   - 压力矩阵对称性
   - CFL 选择
   - 并行/串行一致性
   - 重构/限制器基本守恒与有界性

## 5. 非目标

- 不负责源项抽象
- 不负责 tracer/sediment/waves 业务模型
- 不负责 AI/同化

## 6. 关键验收

- `engine/parallel.rs` 中不存在“有入口无有效实现”的并行路径。
- `schemes` 与 `numerics` 不再保留 f64 专用主路径。
- 主求解链上的显式/半隐式路径都能解释清楚数学意义。
- 关键基准和数值测试可运行。

## 7. 预期冲突点

- 与 T01 在 `MeshTopology`、`StateAccess` 调用点冲突
- 与 T03 在 `SourceContext` / `SourceRegistry` 接口消费点冲突

冲突处理：

- 调用点由 T02 适配
- 公共签名由 T01 定义
