# Phase 1: 状态与网格泛型化

## 目标

将 ShallowWaterState 和 MeshTopology 全面泛型化，支持 Backend 抽象。

## 时间：第 2 周

## 前置依赖

- Phase 0 完成（Backend 实例方法）

## 任务清单

### 1.1 创建泛型状态类型

**目标**：创建 `ShallowWaterStateGeneric<B>` 泛型状态，保留原有 `ShallowWaterState` 作为类型别名。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `state/generic.rs` | 泛型状态实现 |
| 修改 | `state.rs` | 添加类型别名，保持向后兼容 |
| 修改 | `lib.rs` | 更新导出 |

#### 具体改动

**mh_physics/src/state/generic.rs（新建）**

```rust
// marihydro/crates/mh_physics/src/state/generic.rs
//! 泛型浅水方程状态

use crate::core::{Backend, Scalar};
use crate::fields::FieldRegistry;

/// 泛型浅水方程状态
/// 
/// # 类型参数
/// 
/// - `B`: 计算后端（CpuBackend<f32>, CpuBackend<f64>, CudaBackend<f32>等）
pub struct ShallowWaterStateGeneric<B: Backend> {
    /// 单元数量
    n_cells: usize,
    
    /// 水深 [m]
    pub h: B::Buffer<B::Scalar>,
    /// x 方向动量 [m²/s]
    pub hu: B::Buffer<B::Scalar>,
    /// y 方向动量 [m²/s]
    pub hv: B::Buffer<B::Scalar>,
    /// 底床高程 [m]
    pub z: B::Buffer<B::Scalar>,
    
    /// 示踪剂状态（可选）
    pub tracers: Option<TracerStateGeneric<B>>,
    
    /// 字段注册表
    pub field_registry: FieldRegistry,
    
    /// 持有的 Backend 引用
    backend: B,
}

impl<B: Backend> ShallowWaterStateGeneric<B> {
    /// 创建新状态
    pub fn new(backend: B, n_cells: usize) -> Self {
        Self {
            n_cells,
            h: backend.alloc_init(n_cells, B::Scalar::ZERO),
            hu: backend.alloc_init(n_cells, B::Scalar::ZERO),
            hv: backend.alloc_init(n_cells, B::Scalar::ZERO),
            z: backend.alloc_init(n_cells, B::Scalar::ZERO),
            tracers: None,
            field_registry: FieldRegistry::shallow_water(),
            backend,
        }
    }
    
    /// 从初始水位和底床创建（冷启动）
    pub fn cold_start(backend: B, initial_eta: B::Scalar, z_bed: &[B::Scalar]) -> Self {
        let n_cells = z_bed.len();
        
        let mut state = Self::new(backend.clone(), n_cells);
        
        // 计算初始水深
        for i in 0..n_cells {
            let h_val = (initial_eta - z_bed[i]).max(B::Scalar::ZERO);
            state.h.as_mut_slice()[i] = h_val;
            state.z.as_mut_slice()[i] = z_bed[i];
        }
        
        state
    }
    
    /// 单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
    
    /// 获取 Backend 引用
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 克隆结构（不复制数据）
    pub fn clone_structure(&self) -> Self {
        Self {
            n_cells: self.n_cells,
            h: self.backend.alloc_init(self.n_cells, B::Scalar::ZERO),
            hu: self.backend.alloc_init(self.n_cells, B::Scalar::ZERO),
            hv: self.backend.alloc_init(self.n_cells, B::Scalar::ZERO),
            z: self.backend.alloc_init(self.n_cells, B::Scalar::ZERO),
            tracers: self.tracers.as_ref().map(|t| t.clone_structure()),
            field_registry: self.field_registry.clone(),
            backend: self.backend.clone(),
        }
    }
    
    /// 从另一个状态复制数据
    pub fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.n_cells, other.n_cells);
        self.backend.copy(&other.h, &mut self.h);
        self.backend.copy(&other.hu, &mut self.hu);
        self.backend.copy(&other.hv, &mut self.hv);
        // z 通常不变，不复制
    }
    
    /// 重置为零
    pub fn reset(&mut self) {
        self.backend.fill(&mut self.h, B::Scalar::ZERO);
        self.backend.fill(&mut self.hu, B::Scalar::ZERO);
        self.backend.fill(&mut self.hv, B::Scalar::ZERO);
    }
    
    /// 强制正性约束
    pub fn enforce_positivity(&mut self) {
        self.backend.enforce_positivity(&mut self.h, B::Scalar::ZERO);
    }
    
    /// 添加缩放的 RHS: self += scale * rhs
    pub fn add_scaled_rhs(&mut self, rhs: &RhsBuffersGeneric<B>, scale: B::Scalar) {
        self.backend.axpy(scale, &rhs.dh_dt, &mut self.h);
        self.backend.axpy(scale, &rhs.dhu_dt, &mut self.hu);
        self.backend.axpy(scale, &rhs.dhv_dt, &mut self.hv);
    }
    
    /// 线性组合: self = a*A + b*B
    pub fn linear_combine(&mut self, a: B::Scalar, state_a: &Self, b: B::Scalar, state_b: &Self) {
        // self = a * state_a
        self.backend.copy(&state_a.h, &mut self.h);
        self.backend.scale(a, &mut self.h);
        self.backend.copy(&state_a.hu, &mut self.hu);
        self.backend.scale(a, &mut self.hu);
        self.backend.copy(&state_a.hv, &mut self.hv);
        self.backend.scale(a, &mut self.hv);
        
        // self += b * state_b
        self.backend.axpy(b, &state_b.h, &mut self.h);
        self.backend.axpy(b, &state_b.hu, &mut self.hu);
        self.backend.axpy(b, &state_b.hv, &mut self.hv);
    }
    
    /// 计算总质量
    pub fn total_mass(&self, cell_areas: &B::Buffer<B::Scalar>) -> B::Scalar {
        // 需要逐元素乘法，这里简化处理
        let h_slice = self.h.as_slice();
        let area_slice = cell_areas.as_slice();
        h_slice.iter().zip(area_slice.iter())
            .fold(B::Scalar::ZERO, |acc, (&h, &a)| acc + h * a)
    }
}

/// 泛型 RHS 缓冲区
pub struct RhsBuffersGeneric<B: Backend> {
    /// 水深变化率 [m/s]
    pub dh_dt: B::Buffer<B::Scalar>,
    /// x 动量变化率 [m²/s²]
    pub dhu_dt: B::Buffer<B::Scalar>,
    /// y 动量变化率 [m²/s²]
    pub dhv_dt: B::Buffer<B::Scalar>,
    
    backend: B,
}

impl<B: Backend> RhsBuffersGeneric<B> {
    /// 创建新的 RHS 缓冲区
    pub fn new(backend: B, n_cells: usize) -> Self {
        Self {
            dh_dt: backend.alloc_init(n_cells, B::Scalar::ZERO),
            dhu_dt: backend.alloc_init(n_cells, B::Scalar::ZERO),
            dhv_dt: backend.alloc_init(n_cells, B::Scalar::ZERO),
            backend,
        }
    }
    
    /// 重置为零
    pub fn reset(&mut self) {
        self.backend.fill(&mut self.dh_dt, B::Scalar::ZERO);
        self.backend.fill(&mut self.dhu_dt, B::Scalar::ZERO);
        self.backend.fill(&mut self.dhv_dt, B::Scalar::ZERO);
    }
}

/// 泛型示踪剂状态（占位）
pub struct TracerStateGeneric<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> TracerStateGeneric<B> {
    pub fn clone_structure(&self) -> Self {
        Self { _marker: std::marker::PhantomData }
    }
}
```

**mh_physics/src/state.rs（添加类型别名）**

```rust
// 在文件末尾添加

// ============================================================
// 泛型状态（新 API）
// ============================================================

mod generic;
pub use generic::{ShallowWaterStateGeneric, RhsBuffersGeneric, TracerStateGeneric};

use crate::core::{CpuBackend, DefaultBackend};

/// 默认状态类型别名（f64 CPU）
pub type ShallowWaterStateDefault = ShallowWaterStateGeneric<DefaultBackend>;

/// f32 CPU 状态类型别名
pub type ShallowWaterStateF32 = ShallowWaterStateGeneric<CpuBackend<f32>>;

/// f64 CPU 状态类型别名
pub type ShallowWaterStateF64 = ShallowWaterStateGeneric<CpuBackend<f64>>;
```

#### 验证

```rust
#[test]
fn test_generic_state_f64() {
    use crate::core::CpuBackend;
    
    let backend = CpuBackend::<f64>::new();
    let state = ShallowWaterStateGeneric::new(backend, 100);
    
    assert_eq!(state.n_cells(), 100);
    assert_eq!(state.h.as_slice().len(), 100);
}

#[test]
fn test_generic_state_f32() {
    use crate::core::CpuBackend;
    
    let backend = CpuBackend::<f32>::new();
    let state = ShallowWaterStateGeneric::new(backend, 100);
    
    assert_eq!(state.n_cells(), 100);
}
```

---

### 1.2 网格拓扑泛型化

**目标**：创建 `MeshTopologyGeneric<B>` trait，支持泛型后端。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `mesh/topology_generic.rs` | 泛型网格拓扑 trait |
| 修改 | `mesh/mod.rs` | 更新导出 |
| 新建 | `mesh/unstructured_generic.rs` | 泛型非结构化网格适配器 |

#### 具体改动

**mh_physics/src/mesh/topology_generic.rs（新建）**

```rust
// marihydro/crates/mh_physics/src/mesh/topology_generic.rs
//! 泛型网格拓扑抽象

use crate::core::Backend;

/// 泛型网格拓扑 trait
/// 
/// 提供网格几何和拓扑信息的统一接口。
pub trait MeshTopologyGeneric<B: Backend>: Send + Sync {
    /// 单元数量
    fn n_cells(&self) -> usize;
    
    /// 面数量
    fn n_faces(&self) -> usize;
    
    /// 边界面数量
    fn n_boundary_faces(&self) -> usize;
    
    /// 内部面数量
    #[inline]
    fn n_internal_faces(&self) -> usize {
        self.n_faces() - self.n_boundary_faces()
    }
    
    // ========== 几何数据 ==========
    
    /// 单元中心坐标 [x, y]
    fn cell_centers(&self) -> &B::Buffer<[B::Scalar; 2]>;
    
    /// 单元面积/体积
    fn cell_volumes(&self) -> &B::Buffer<B::Scalar>;
    
    /// 面法向量 [nx, ny]（指向 neighbor）
    fn face_normals(&self) -> &B::Buffer<[B::Scalar; 2]>;
    
    /// 面长度/面积
    fn face_areas(&self) -> &B::Buffer<B::Scalar>;
    
    /// 面中心坐标 [x, y]
    fn face_centers(&self) -> &B::Buffer<[B::Scalar; 2]>;
    
    // ========== 拓扑数据 ==========
    
    /// 面的 owner 单元索引
    fn face_owner(&self) -> &B::Buffer<u32>;
    
    /// 面的 neighbor 单元索引（-1 表示边界）
    fn face_neighbor(&self) -> &B::Buffer<i32>;
    
    // ========== 辅助方法 ==========
    
    /// 获取面的两侧单元
    #[inline]
    fn face_cells(&self, face: usize) -> (usize, Option<usize>) {
        let owner = self.face_owner().as_slice()[face] as usize;
        let neighbor = self.face_neighbor().as_slice()[face];
        if neighbor >= 0 {
            (owner, Some(neighbor as usize))
        } else {
            (owner, None)
        }
    }
    
    /// 判断是否为边界面
    #[inline]
    fn is_boundary_face(&self, face: usize) -> bool {
        self.face_neighbor().as_slice()[face] < 0
    }
}

/// 网格类型标记
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshKindGeneric {
    /// 非结构化三角形/多边形网格
    Unstructured,
    /// 结构化矩形网格
    Structured,
    /// 混合网格
    Hybrid,
}
```

**mh_physics/src/mesh/unstructured_generic.rs（新建）**

```rust
// marihydro/crates/mh_physics/src/mesh/unstructured_generic.rs
//! 泛型非结构化网格适配器

use crate::core::{Backend, Scalar};
use super::topology_generic::MeshTopologyGeneric;

/// 泛型非结构化网格适配器
/// 
/// 从 mh_mesh 的网格数据创建物理求解器所需的拓扑结构。
pub struct UnstructuredMeshGeneric<B: Backend> {
    n_cells: usize,
    n_faces: usize,
    n_boundary_faces: usize,
    
    cell_centers: B::Buffer<[B::Scalar; 2]>,
    cell_volumes: B::Buffer<B::Scalar>,
    face_normals: B::Buffer<[B::Scalar; 2]>,
    face_areas: B::Buffer<B::Scalar>,
    face_centers: B::Buffer<[B::Scalar; 2]>,
    face_owner: B::Buffer<u32>,
    face_neighbor: B::Buffer<i32>,
    
    backend: B,
}

impl<B: Backend> UnstructuredMeshGeneric<B> {
    /// 从原始数据创建
    pub fn from_raw(
        backend: B,
        n_cells: usize,
        n_faces: usize,
        n_boundary_faces: usize,
        cell_centers: Vec<[B::Scalar; 2]>,
        cell_volumes: Vec<B::Scalar>,
        face_normals: Vec<[B::Scalar; 2]>,
        face_areas: Vec<B::Scalar>,
        face_centers: Vec<[B::Scalar; 2]>,
        face_owner: Vec<u32>,
        face_neighbor: Vec<i32>,
    ) -> Self {
        // 将数据复制到 Backend Buffer
        let mut cell_centers_buf = backend.alloc::<[B::Scalar; 2]>(n_cells);
        cell_centers_buf.as_mut_slice().copy_from_slice(&cell_centers);
        
        let mut cell_volumes_buf = backend.alloc::<B::Scalar>(n_cells);
        cell_volumes_buf.as_mut_slice().copy_from_slice(&cell_volumes);
        
        let mut face_normals_buf = backend.alloc::<[B::Scalar; 2]>(n_faces);
        face_normals_buf.as_mut_slice().copy_from_slice(&face_normals);
        
        let mut face_areas_buf = backend.alloc::<B::Scalar>(n_faces);
        face_areas_buf.as_mut_slice().copy_from_slice(&face_areas);
        
        let mut face_centers_buf = backend.alloc::<[B::Scalar; 2]>(n_faces);
        face_centers_buf.as_mut_slice().copy_from_slice(&face_centers);
        
        let mut face_owner_buf = backend.alloc::<u32>(n_faces);
        face_owner_buf.as_mut_slice().copy_from_slice(&face_owner);
        
        let mut face_neighbor_buf = backend.alloc::<i32>(n_faces);
        face_neighbor_buf.as_mut_slice().copy_from_slice(&face_neighbor);
        
        Self {
            n_cells,
            n_faces,
            n_boundary_faces,
            cell_centers: cell_centers_buf,
            cell_volumes: cell_volumes_buf,
            face_normals: face_normals_buf,
            face_areas: face_areas_buf,
            face_centers: face_centers_buf,
            face_owner: face_owner_buf,
            face_neighbor: face_neighbor_buf,
            backend,
        }
    }
    
    /// 获取 Backend 引用
    pub fn backend(&self) -> &B {
        &self.backend
    }
}

impl<B: Backend> MeshTopologyGeneric<B> for UnstructuredMeshGeneric<B> {
    fn n_cells(&self) -> usize { self.n_cells }
    fn n_faces(&self) -> usize { self.n_faces }
    fn n_boundary_faces(&self) -> usize { self.n_boundary_faces }
    
    fn cell_centers(&self) -> &B::Buffer<[B::Scalar; 2]> { &self.cell_centers }
    fn cell_volumes(&self) -> &B::Buffer<B::Scalar> { &self.cell_volumes }
    fn face_normals(&self) -> &B::Buffer<[B::Scalar; 2]> { &self.face_normals }
    fn face_areas(&self) -> &B::Buffer<B::Scalar> { &self.face_areas }
    fn face_centers(&self) -> &B::Buffer<[B::Scalar; 2]> { &self.face_centers }
    fn face_owner(&self) -> &B::Buffer<u32> { &self.face_owner }
    fn face_neighbor(&self) -> &B::Buffer<i32> { &self.face_neighbor }
}
```

**mh_physics/src/mesh/mod.rs（更新）**

```rust
// 添加泛型模块
pub mod topology_generic;
pub mod unstructured_generic;

// 重导出
pub use topology_generic::{MeshTopologyGeneric, MeshKindGeneric};
pub use unstructured_generic::UnstructuredMeshGeneric;
```

---

### 1.3 适配现有代码

**目标**：确保现有代码继续工作，通过类型别名保持向后兼容。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `lib.rs` | 添加泛型类型导出 |
| 修改 | 各使用 State 的模块 | 逐步迁移到泛型版本 |

#### 具体改动

**mh_physics/src/lib.rs（更新导出）**

```rust
// 在现有导出后添加

// 泛型状态类型
pub use state::{
    ShallowWaterStateGeneric, RhsBuffersGeneric,
    ShallowWaterStateDefault, ShallowWaterStateF32, ShallowWaterStateF64,
};

// 泛型网格类型
pub use mesh::{
    MeshTopologyGeneric, MeshKindGeneric, UnstructuredMeshGeneric,
};
```

---

## 验收标准

1. ✅ `ShallowWaterStateGeneric<B>` 可用 f32/f64 后端实例化
2. ✅ `MeshTopologyGeneric<B>` trait 定义完整
3. ✅ 现有 `ShallowWaterState` 继续工作（类型别名）
4. ✅ 所有现有测试通过
5. ✅ 新增泛型测试通过

## 测试用例

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::CpuBackend;

    #[test]
    fn test_state_f32_f64_interop() {
        let backend_f32 = CpuBackend::<f32>::new();
        let backend_f64 = CpuBackend::<f64>::new();
        
        let state_f32 = ShallowWaterStateGeneric::new(backend_f32, 100);
        let state_f64 = ShallowWaterStateGeneric::new(backend_f64, 100);
        
        assert_eq!(state_f32.n_cells(), state_f64.n_cells());
    }

    #[test]
    fn test_state_cold_start() {
        let backend = CpuBackend::<f64>::new();
        let z_bed = vec![0.0, 0.1, 0.2, 0.3];
        let state = ShallowWaterStateGeneric::cold_start(backend, 1.0, &z_bed);
        
        assert_eq!(state.n_cells(), 4);
        assert!((state.h.as_slice()[0] - 1.0).abs() < 1e-10);
        assert!((state.h.as_slice()[3] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_state_operations() {
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterStateGeneric::new(backend.clone(), 10);
        
        // 设置初始值
        state.h.as_mut_slice().fill(1.0);
        state.hu.as_mut_slice().fill(0.5);
        
        // 测试复制
        let mut state2 = state.clone_structure();
        state2.copy_from(&state);
        
        assert!((state2.h.as_slice()[0] - 1.0).abs() < 1e-10);
        
        // 测试正性约束
        state.h.as_mut_slice()[0] = -0.1;
        state.enforce_positivity();
        assert!(state.h.as_slice()[0] >= 0.0);
    }
}
```

## 依赖关系

```
Phase 1 完成后:
├── Phase 2 可以开始（求解器策略化依赖泛型状态）
├── Phase 3 可以开始（源项泛型化依赖泛型状态）
└── Phase 4-7 等待前置 Phase
```

## 迁移指南

### 从旧 API 迁移到新 API

```rust
// 旧代码
let state = ShallowWaterState::new(100);

// 新代码（显式后端）
let backend = CpuBackend::<f64>::new();
let state = ShallowWaterStateGeneric::new(backend, 100);

// 或使用类型别名
let state = ShallowWaterStateDefault::new(CpuBackend::new(), 100);
```

### 泛型函数签名

```rust
// 旧签名
fn process_state(state: &ShallowWaterState) { ... }

// 新签名（泛型）
fn process_state<B: Backend>(state: &ShallowWaterStateGeneric<B>) { ... }

// 或约束到特定后端
fn process_state(state: &ShallowWaterStateDefault) { ... }
```
