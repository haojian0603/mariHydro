# 配置层与运行层分离设计原理    

## 架构决策核心：为何必须保持两层独立？

### 1. **数据权威性：PhysicalConstants 是"唯一真理源"**

**权威位置**：`crates/mh_physics/src/types.rs` 中的 `PhysicalConstants` 结构体

**硬性规定**：
- **仅此一处**允许硬编码物理常量（如 `g: 9.81_f64`）
- **任何 `Config` 结构体**（如 `SolverConfig`）**禁止**重复硬编码 `9.81`
- **运行时**通过 `S::from_f64(constants.g)` 转换

**正确模式**：
```rust
// 1. 权威配置（仅此一处硬编码）
let constants = PhysicalConstants::seawater(); // g = 9.81_f64

// 2. Config 结构体引用 PhysicalConstants
pub struct SolverConfig {
    pub constants: PhysicalConstants,  // ✅ 不硬编码，只存储引用
}

// 3. 运行时转换为 S
impl<S: Scalar> Solver<S> {
    pub fn new(config: &SolverConfig) -> Self {
        Self {
            gravity: S::from_f64(config.constants.g),  // 转换边界
        }
    }
}
```

**红线**：在 `SolverConfig` 或任何其他结构体中写 `const GRAVITY: f64 = 9.81;` 是**架构违规**。

### 2. **编译期多态 vs 运行时多态**


**错误设计（过度泛型化）**：
```rust
// ❌ 配置层就绑定精度，失去灵活性
pub struct Config<S: Scalar> {
    gravity: S,  // 用户必须提前决定 f32/f64
}

// 用户被迫在配置阶段选精度
let config = Config::<f32>::new(); // 一旦选定，全程锁定
```

**正确设计（两层分离）**：
```rust
// ✅ 配置永不绑定精度
pub struct Config {
    gravity: f64,  // 永远是权威值
}

// 运行时按需转换（初始化一次）
let operator_f32 = DiffusionOperator::<f32>::new(config);
let operator_f64 = DiffusionOperator::<f64>::new(config); // 同一配置，不同精度
```

**核心优势**：
- **编译期单态化**：`f32` 和 `f64` 版本生成**独立代码**，无分支预测、虚表开销
- **零运行时开销**：`S::from_f64(config.gravity)` 在 `new()` 中**仅执行一次**，后续百万次循环直接操作 `S`

### 3. **工程解耦：配置团队 vs 算法团队**


**现实协作场景**：
- **观测员**（海洋学家）提供配置文件：`boundary.toml` 含 `gravity = 9.80665`
- **算法工程师**（我）写求解器：`<S: Scalar>` 泛型化
- **GPU 工程师**（同事）部署到 RTX 4090：实例化 `Solver::<f32>`

**若配置泛型化**：
```rust
// ❌ 观测员必须懂 Rust 泛型
[boundary]
type = "Dirichlet"
value = 35.0  # 这是什么类型？f32? f64? 取决于编译 feature？
```

**分离后**：
```rust
// 配置纯粹是数据，与实现无关
[boundary]
type = "Dirichlet"
value = 35.0  # 永远是 f64，人的读法

// 配置层方法（返回运行时结构体）
impl DiffusionConfig {
    pub fn to_operator<S: Scalar>(&self) -> DiffusionOperator<S> { ... }
}
```

### 4. **内存安全与单次转换**

**配置层 f64 → GPU f32 的两阶段转换**：
1. **磁盘 → 配置结构体**：反序列化为 `f64`（`ron::from_str` 自动完成）
2. **配置 → GPU 缓冲区**：`config.to_precision::<f32>()` → `Vec<f32>` → `bytemuck::cast_slice()` **单次转换，多次复用上传**

**若配置是 `f32`**：
```rust
// ❌ 配置从磁盘读取时就是 f32，损失了原始数据
let config_f32 = ron::from_str("gravity: 9.80665")?; // 9.80665 → 9.80665f32，精度已丢失
// 再也恢复不到精确的 9.80665f64 用于 CPU 调试
```

**分离后**：
```rust
// ✅ 配置永远是 f64，CPU 和 GPU 各取所需
let config_f64 = ron::from_str("gravity: 9.80665")?; // 精确值
let gpu_state = GpuState::from_config::<f32>(&config_f64); // GPU 用 f32
let cpu_state = CpuState::from_config::<f64>(&config_f64); // CPU 用 f64
```

### 5. **符合科学计算软件传统**

**PETSc、deal.II、Trilinos 的成熟模式**：
- **参数文件**：纯文本，`double` 精度（C++ 硬编码）
- **求解器模板**：`Solver<Number>`，`Number = float` or `double`（编译期选择）
- **零运行时切换**：不可能在运行时从 `double` 切到 `float`，避免性能灾难

**我们的设计是 Rust 版的最佳实践复现**。

### 6. **认知负荷最小化**

**维护者视角**：
- **阅读配置代码**：只看 `f64`，无需思考精度问题
- **阅读算法代码**：只看 `<S: Scalar>`，清晰的泛型边界
- **调试时**：配置值 = 观测值，直接比对 CSV 和 TOML

---

### **附录：运行时转换规范**

**转换时机**：仅在算子/策略的 `new()` 或 `from_config()` 方法中执行

**代码模板**：
```rust
impl<S: Scalar> ShallowWaterSolver<S> {
    pub fn from_config(config: &SolverConfig) -> Self {
        Self {
            gravity: S::from_f64(config.constants.g),
            rho_water: S::from_f64(config.constants.rho_water),
        }
    }
}
```

**禁止行为**：
- ❌ 在 `Scalar` trait 中定义物理常量
- ❌ 在 `Config` 结构体中硬编码 `9.81`
- ❌ 在运行时循环中使用 `config.constants.g`（类型不匹配）

---

## ✅ PR 审查清单

- [ ] 新增 Struct 命名为 `Config` 或 `Operator`？
- [ ] `Config` 结构体**无泛型参数**，字段**显式为 `f64`**
- [ ] `Operator` 字段类型为 `S`（通过构造函数转换）
- [ ] 物理常量来源：**仅允许 `PhysicalConstants`**（禁止硬编码）
- [ ] 转换语法：`<S as Scalar>::from_f64(...)`（避免歧义）
- [ ] 转换位置：仅在 `new()` / `from_config()` 中（循环内禁止）
- [ ] 所有 `Config` 结构体引用 `PhysicalConstants`？