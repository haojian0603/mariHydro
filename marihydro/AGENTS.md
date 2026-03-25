# MariHydro 协作规范

本文件是当前仓库的常驻协作入口。并行 agent、收敛任务和后续增量修改都以这里为准；如果规则变化，直接更新本文件，不再散落到临时提示词里。

## 1. 工作方式

- 默认直接执行，不要频繁征求用户意见。
- 只有在无法从本地代码和现有规则中推断、且错误代价明显偏高时，才允许停下来询问。
- 发现 `cargo`、`clippy`、测试或构建目录被锁时，不要终止任务；先继续做不依赖锁的代码修改、扫描和文档更新，稍后再回到门禁。
- 发现其他 agent 的正常并行改动时，不要把它当成阻塞事件；只要不是自己将要覆盖的写集，就继续推进。

## 2. 提交与推送

- 按批次提交，不要改一个文件就提交一次。
- 每批提交默认不少于 7 个文件；只有明确的单点修复且会阻塞门禁时才允许例外。
- 每一批在提交前必须跑完整门禁。
- 每一批提交后立即推送到当前工作分支。
- 提交信息优先使用英文短句，格式保持稳定，例如：`refactor: ... batch N`。

## 3. 门禁规则

- 门禁只能收紧，不能放水。
- 每一批至少执行：
  - `cargo check --workspace`
  - `cargo clippy --workspace --all-targets`
  - `cargo test --workspace`
  - `powershell -ExecutionPolicy Bypass -File scripts/check_tracked_temp_artifacts.ps1`
  - `powershell -ExecutionPolicy Bypass -File scripts/verify_architecture.ps1`
  - `powershell -ExecutionPolicy Bypass -File scripts/architecture_audit.ps1`
- 如果当前批次引入了新的架构约束，必须同步更新门禁脚本，确保后续不会回退。
- 门禁如果新增扫描项，优先先做 advisory，再视收敛成熟度升级为 blocking。

## 4. 架构收敛优先级

- 优先收敛 runtime/core/public contract，再收敛上层功能模块。
- 禁止新增静默标量回退：
  - 禁止 `from_config(...).unwrap_or(...)`
  - 禁止 `from_f64/from_f32(...).unwrap_or(...)`
  - 禁止新增 `if let Some(...) = ...::from_config(...)` 这类裸 `Option` 主链分叉
- `mh_physics` 主链路优先使用显式 conversion helper 或 backend 上下文 helper。
- `sources`、`mh_agent` 中如需暂时保留 legacy/test residue，必须限制在测试支撑层，不要继续扩散到生产主链。

## 5. Sources 与 AI 层规则

- `sources` 的目标是单一主路径；不要新增 `CpuBackend<f64>` 专用的生产实现。
- 可以保留测试后端，但要尽量集中到共享测试支撑中，不要每个文件各写一套。
- `mh_agent` 继续收敛裸 `Vec<f64>`、裸索引和裸几何数组；配置层和纯统计层可以暂时保留，但 apply/update 主链路不再扩大使用面。

## 6. 规范记录

- 需要长期保留的协作规范，统一写入本文件。
- 某一批的具体实现说明和临时结论，写进对应任务日志或 Git 提交，不要求额外向用户重复汇报。
- 如果规则和已有提示词冲突，以本文件和门禁脚本为准。

## 7. 编码与注释

- 用户侧沟通、注释和文档优先中文。
- 涉及中文文件写入时，必须显式保证 UTF-8；不要用会污染编码的 shell 重定向方式直接写中文内容。
- 如果终端显示异常，先区分“终端乱码”与“文件真坏了”，不要在未确认前重复覆盖文件。
