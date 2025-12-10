# Phase 3: 源项与示踪剂泛型化

## 目标

完成源项系统和示踪剂的 Backend 泛型化。

## 时间：第 5 周

## 前置依赖

- Phase 1 完成（泛型状态）
- Phase 2 完成（求解器策略化）

## 任务清单

### 3.1 源项 Trait 重构

**目标**：创建 `SourceTermGeneric<B>` trait，支持泛型后端。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `sources/traits_generic.rs` | 泛型源项 trait |
| 新建 | `sources/registry.rs` | 源项注册中心 |
| 修改 | `sources/mod.rs` | 更新导出 |

#### 关键代码

```rust
// sources/traits_generic.rs
use crate::core::Backend;
use crate::state::ShallowWaterStateGeneric;

/// 源项刚性分类
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceStiffness {
    /// 显式处理
    Explicit,
    /// 局部隐式（如摩擦的 1/(1+dt*γ)）
    LocallyImplicit,
}

/// 源项贡献
#[derive(Debug, Clone, Copy, Default)]
pub struct SourceContributionGeneric<S> {
    pub s_h: S,
    pub s_hu: S,
    pub s_hv: S,
}

/// 源项上下文
pub struct SourceContextGeneric<S> {
    pub dt: S,
    pub gravity: S,
    pub time: f64,
}

/// 泛型源项 Trait
pub trait SourceTermGeneric<B: Backend>: Send + Sync {
    /// 源项名称
    fn name(&self) -> &'static str;
    
    /// 刚性分类
    fn stiffness(&self) -> SourceStiffness;
    
    /// 逐单元计算
    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterStateGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar>;
    
    /// 批量计算（可被 GPU 重载）
    fn compute_batch(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        contributions: &mut [SourceContributionGeneric<B::Scalar>],
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        for cell in 0..state.n_cells() {
            contributions[cell] = self.compute_cell(cell, state, ctx);
        }
    }
}
```

```rust
// sources/registry.rs
use crate::core::Backend;
use super::traits_generic::SourceTermGeneric;

/// 源项注册中心
pub struct SourceRegistry<B: Backend> {
    sources: Vec<Box<dyn SourceTermGeneric<B>>>,
}

impl<B: Backend> SourceRegistry<B> {
    pub fn new() -> Self {
        Self { sources: Vec::new() }
    }
    
    /// 注册源项
    pub fn register(&mut self, source: Box<dyn SourceTermGeneric<B>>) {
        self.sources.push(source);
    }
    
    /// 累加所有源项
    pub fn accumulate_all(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        rhs: &mut RhsBuffersGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        for source in &self.sources {
            let mut contributions = vec![
                SourceContributionGeneric::default(); 
                state.n_cells()
            ];
            source.compute_batch(state, &mut contributions, ctx);
            
            for (i, c) in contributions.iter().enumerate() {
                rhs.dh_dt.as_mut_slice()[i] = 
                    rhs.dh_dt.as_slice()[i] + c.s_h;
                rhs.dhu_dt.as_mut_slice()[i] = 
                    rhs.dhu_dt.as_slice()[i] + c.s_hu;
                rhs.dhv_dt.as_mut_slice()[i] = 
                    rhs.dhv_dt.as_slice()[i] + c.s_hv;
            }
        }
    }
    
    /// 源项数量
    pub fn len(&self) -> usize {
        self.sources.len()
    }
}
```

---

### 3.2 摩擦源项泛型化

**目标**：实现 `ManningFrictionGeneric<B>`。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `sources/friction_generic.rs` | 泛型摩擦源项 |

#### 关键代码

```rust
// sources/friction_generic.rs
use crate::core::{Backend, Scalar};
use super::traits_generic::*;

/// Manning 摩擦配置
#[derive(Debug, Clone)]
pub struct ManningFrictionConfigGeneric<S> {
    pub gravity: S,
    pub manning_n: Vec<S>,  // 每个单元的 Manning 系数
    pub min_depth: S,
}

/// 泛型 Manning 摩擦
pub struct ManningFrictionGeneric<B: Backend> {
    config: ManningFrictionConfigGeneric<B::Scalar>,
    backend: B,
}

impl<B: Backend> ManningFrictionGeneric<B> {
    pub fn new(backend: B, config: ManningFrictionConfigGeneric<B::Scalar>) -> Self {
        Self { config, backend }
    }
    
    pub fn uniform(backend: B, n_cells: usize, manning_n: B::Scalar) -> Self {
        Self::new(backend, ManningFrictionConfigGeneric {
            gravity: B::Scalar::GRAVITY,
            manning_n: vec![manning_n; n_cells],
            min_depth: B::Scalar::from_f64(1e-6),
        })
    }
}

impl<B: Backend> SourceTermGeneric<B> for ManningFrictionGeneric<B> {
    fn name(&self) -> &'static str { "Manning-Friction" }
    
    fn stiffness(&self) -> SourceStiffness { SourceStiffness::LocallyImplicit }
    
    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterStateGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        let h = state.h.as_slice()[cell];
        let hu = state.hu.as_slice()[cell];
        let hv = state.hv.as_slice()[cell];
        
        if h < self.config.min_depth {
            return SourceContributionGeneric::default();
        }
        
        let n = self.config.manning_n[cell];
        let g = self.config.gravity;
        
        // 速度
        let u = hu / h;
        let v = hv / h;
        let speed = (u * u + v * v).sqrt();
        
        // Manning 摩擦系数
        // τ = ρ g n² |u| u / h^(1/3)
        let h_pow = h.powf(B::Scalar::from_f64(1.0 / 3.0));
        let cf = g * n * n / h_pow;
        
        // 隐式处理因子
        let gamma = cf * speed / h;
        let factor = B::Scalar::ONE / (B::Scalar::ONE + ctx.dt * gamma);
        
        SourceContributionGeneric {
            s_h: B::Scalar::ZERO,
            s_hu: -cf * speed * u * factor,
            s_hv: -cf * speed * v * factor,
        }
    }
}
```

---

### 3.3 科氏力源项泛型化

**目标**：实现 `CoriolisSourceGeneric<B>`。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `sources/coriolis_generic.rs` | 泛型科氏力源项 |

#### 关键代码

```rust
// sources/coriolis_generic.rs
use crate::core::{Backend, Scalar};
use super::traits_generic::*;

/// 地球自转角速度 (rad/s)
pub const EARTH_OMEGA: f64 = 7.2921e-5;

/// 泛型科氏力源项
pub struct CoriolisSourceGeneric<B: Backend> {
    /// 科氏参数 f = 2Ω sin(φ)
    f: B::Buffer<B::Scalar>,
    backend: B,
}

impl<B: Backend> CoriolisSourceGeneric<B> {
    /// 从纬度创建（均匀 f）
    pub fn from_latitude(backend: B, n_cells: usize, latitude_deg: f64) -> Self {
        let f_val = B::Scalar::from_f64(
            2.0 * EARTH_OMEGA * (latitude_deg.to_radians()).sin()
        );
        let f = backend.alloc_init(n_cells, f_val);
        Self { f, backend }
    }
    
    /// 从 f 参数数组创建（变化 f）
    pub fn from_f_array(backend: B, f_values: Vec<B::Scalar>) -> Self {
        let mut f = backend.alloc::<B::Scalar>(f_values.len());
        f.as_mut_slice().copy_from_slice(&f_values);
        Self { f, backend }
    }
}

impl<B: Backend> SourceTermGeneric<B> for CoriolisSourceGeneric<B> {
    fn name(&self) -> &'static str { "Coriolis" }
    
    fn stiffness(&self) -> SourceStiffness { SourceStiffness::Explicit }
    
    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterStateGeneric<B>,
        _ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        let hu = state.hu.as_slice()[cell];
        let hv = state.hv.as_slice()[cell];
        let f = self.f.as_slice()[cell];
        
        // 科氏力: f × (hu, hv) = (f*hv, -f*hu)
        SourceContributionGeneric {
            s_h: B::Scalar::ZERO,
            s_hu: f * hv,
            s_hv: -f * hu,
        }
    }
}
```

---

### 3.4 示踪剂泛型化

**目标**：创建 `TracerFieldGeneric<B>` 和 `TracerStateGeneric<B>`。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `tracer/state_generic.rs` | 泛型示踪剂状态 |
| 新建 | `tracer/transport_generic.rs` | 泛型示踪剂输运 |
| 修改 | `tracer/mod.rs` | 更新导出 |

#### 关键代码

```rust
// tracer/state_generic.rs
use crate::core::Backend;
use std::collections::HashMap;

/// 示踪剂类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TracerType {
    Sediment,
    Salinity,
    Temperature,
    Custom(u32),
}

/// 示踪剂属性
#[derive(Debug, Clone)]
pub struct TracerPropertiesGeneric<S> {
    pub name: String,
    pub diffusivity: S,
    pub settling_velocity: Option<S>,
    pub decay_rate: Option<S>,
}

/// 泛型示踪剂场
pub struct TracerFieldGeneric<B: Backend> {
    pub properties: TracerPropertiesGeneric<B::Scalar>,
    concentration: B::Buffer<B::Scalar>,
    conserved: B::Buffer<B::Scalar>,  // h * C
    rhs: B::Buffer<B::Scalar>,
    backend: B,
}

impl<B: Backend> TracerFieldGeneric<B> {
    pub fn new(
        backend: B, 
        n_cells: usize, 
        properties: TracerPropertiesGeneric<B::Scalar>
    ) -> Self {
        Self {
            properties,
            concentration: backend.alloc_init(n_cells, B::Scalar::ZERO),
            conserved: backend.alloc_init(n_cells, B::Scalar::ZERO),
            rhs: backend.alloc_init(n_cells, B::Scalar::ZERO),
            backend,
        }
    }
    
    pub fn concentration(&self) -> &B::Buffer<B::Scalar> { &self.concentration }
    pub fn concentration_mut(&mut self) -> &mut B::Buffer<B::Scalar> { &mut self.concentration }
    
    /// 从水深更新守恒量
    pub fn update_conserved(&mut self, h: &B::Buffer<B::Scalar>) {
        for i in 0..self.concentration.as_slice().len() {
            self.conserved.as_mut_slice()[i] = 
                h.as_slice()[i] * self.concentration.as_slice()[i];
        }
    }
    
    /// 从守恒量恢复浓度
    pub fn recover_concentration(&mut self, h: &B::Buffer<B::Scalar>, min_h: B::Scalar) {
        for i in 0..self.concentration.as_slice().len() {
            let h_val = h.as_slice()[i];
            if h_val > min_h {
                self.concentration.as_mut_slice()[i] = 
                    self.conserved.as_slice()[i] / h_val;
            } else {
                self.concentration.as_mut_slice()[i] = B::Scalar::ZERO;
            }
        }
    }
}

/// 泛型示踪剂状态管理
pub struct TracerStateGeneric<B: Backend> {
    fields: HashMap<TracerType, TracerFieldGeneric<B>>,
    backend: B,
}

impl<B: Backend> TracerStateGeneric<B> {
    pub fn new(backend: B) -> Self {
        Self {
            fields: HashMap::new(),
            backend,
        }
    }
    
    pub fn register(
        &mut self, 
        tracer_type: TracerType, 
        n_cells: usize,
        properties: TracerPropertiesGeneric<B::Scalar>
    ) {
        let field = TracerFieldGeneric::new(self.backend.clone(), n_cells, properties);
        self.fields.insert(tracer_type, field);
    }
    
    pub fn get(&self, tracer_type: &TracerType) -> Option<&TracerFieldGeneric<B>> {
        self.fields.get(tracer_type)
    }
    
    pub fn get_mut(&mut self, tracer_type: &TracerType) -> Option<&mut TracerFieldGeneric<B>> {
        self.fields.get_mut(tracer_type)
    }
}
```

---

## 验收标准

1. ✅ `SourceTermGeneric<B>` trait 定义完整
2. ✅ `SourceRegistry<B>` 可注册和累加源项
3. ✅ Manning 摩擦泛型化并通过测试
4. ✅ 科氏力泛型化并通过测试
5. ✅ 示踪剂状态泛型化并通过测试
6. ✅ 所有现有测试通过

## 测试用例

```rust
#[test]
fn test_source_registry() {
    let backend = CpuBackend::<f64>::new();
    let mut registry = SourceRegistry::new();
    
    // 注册摩擦
    let friction = ManningFrictionGeneric::uniform(backend.clone(), 100, 0.025);
    registry.register(Box::new(friction));
    
    // 注册科氏力
    let coriolis = CoriolisSourceGeneric::from_latitude(backend.clone(), 100, 30.0);
    registry.register(Box::new(coriolis));
    
    assert_eq!(registry.len(), 2);
}

#[test]
fn test_tracer_conservation() {
    let backend = CpuBackend::<f64>::new();
    let n_cells = 10;
    
    let props = TracerPropertiesGeneric {
        name: "sediment".to_string(),
        diffusivity: 0.1,
        settling_velocity: Some(0.001),
        decay_rate: None,
    };
    
    let mut field = TracerFieldGeneric::new(backend.clone(), n_cells, props);
    
    // 设置初始浓度
    field.concentration_mut().as_mut_slice().fill(1.0);
    
    // 模拟水深
    let h = backend.alloc_init(n_cells, 2.0);
    
    // 更新守恒量
    field.update_conserved(&h);
    
    // 验证守恒量
    assert!((field.conserved.as_slice()[0] - 2.0).abs() < 1e-10);
}
```
