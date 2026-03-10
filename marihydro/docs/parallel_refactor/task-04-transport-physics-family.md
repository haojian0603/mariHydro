# T04 传输物理族统一

负责人建议：最熟悉 tracer、sediment、vertical、waves 这一组耦合问题的人

## 1. 目标

统一 `tracer`、`sediment`、`vertical`、`waves` 这四个家族的状态接口、边界接口、扩散/沉降/交换语义，消除“每个模块各管一套”的分裂状态。

## 2. 边界文件

主要文件：

- `crates/mh_physics/src/tracer/*`
- `crates/mh_physics/src/sediment/*`
- `crates/mh_physics/src/vertical/*`
- `crates/mh_physics/src/waves/*`

可联动文件：

- `crates/mh_physics/src/fields.rs`
- `crates/mh_physics/src/state.rs`

不允许擅自改动：

- `sources/traits.rs` 签名
- `engine/*` 公共接口
- `mh_agent/*`

## 3. 必须解决的问题

### 3.1 tracer 家族

- 边界条件、扩散、沉降、transport 语义必须统一。
- 去掉“边界是配置层，transport 是运行时层，但没有清晰桥接”的状态。
- 确保守恒量与浓度量转换路径唯一。

### 3.2 sediment 家族

- 统一 bed load / suspended / morphology / resuspension / settling / transport_2_5d。
- 清理“一个模块说完整输运，实际只实现沉降/交换”的半成品语义。
- 对公式实现与配置做分层：配置层可 `f64`，运行时主路径必须 generic。

### 3.3 vertical 家族

- 统一 sigma / state / profile / velocity / mixing 的状态布局。
- 解决分层存储和访问模式割裂的问题。
- 确保 vertical 相关模块对 state 的依赖是单一的。

### 3.4 waves 家族

- 整理 `bottom_friction`、`radiation_stress`、`spectral` 的成熟度差异。
- 明确哪些是生产路径，哪些是实验路径。
- 去除波浪模块中的 f64-only 主路径和未落地框架。

## 4. 拆解步骤

1. tracer：边界/扩散/沉降/transport/field/state 统一。
2. sediment：先 suspended，再 bed_load，再 morphology。
3. vertical：先 state/sigma，再 profile/velocity/mixing。
4. waves：先 radiation_stress / bottom_friction，再处理 spectral 是否保留在主路径。
5. 加族级回归测试：
   - tracer 守恒
   - sediment 质量平衡
   - vertical 剖面恢复
   - waves 诊断量一致性

## 5. 非目标

- 不负责公共 trait 签名冻结
- 不负责源项体系抽象
- 不负责 AI 层桥接

## 6. 验收标准

- 四个家族都接入同一套 runtime-like 能力与状态语义。
- 不再保留“主路径未完成但对外默认可用”的实现。
- 文档和代码对每个模块的成熟度说明一致。

## 7. 预期冲突点

- 与 T03 在源项接口消费点冲突
- 与 T02 在 transport 调用求解器路径冲突
- 与 T05 在 sediment/tracer 同化入口冲突

处理方式：

- 只消费 T01/T03 定好的接口
- 需要扩展时先登记，再协调
