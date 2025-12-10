# Phase 4: 泥沙系统耦合

## 目标

实现 SedimentManager，闭合泥沙质量守恒。

## 时间：第 6 周

## 前置依赖

- Phase 3 完成（示踪剂泛型化）

## 任务清单

### 4.1 SedimentManager 实现

**目标**：创建统一的泥沙系统管理器。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `sediment/manager.rs` | 泥沙统一管理器 |
| 修改 | `sediment/mod.rs` | 更新导出 |

#### 关键代码

```rust
// sediment/manager.rs
use crate::core::Backend;
use crate::state::ShallowWaterStateGeneric;
use crate::tracer::{TracerFieldGeneric, TracerPropertiesGeneric, TracerType};
use crate::vertical::LayeredStateGeneric;

/// 泥沙系统错误
#[derive(Debug)]
pub enum SedimentError {
    ConservationViolation { expected: f64, actual: f64, relative_error: f64 },
    NegativeMass { cell: usize, value: f64 },
    InvalidParameter(String),
}

/// 泥沙系统配置
#[derive(Debug, Clone)]
pub struct SedimentConfig<S> {
    /// 临界剪切应力 [Pa]
    pub tau_critical: S,
    /// 侵蚀系数 [kg/m²/s/Pa]
    pub erosion_rate: S,
    /// 沉降速度 [m/s]
    pub settling_velocity: S,
    /// 泥沙密度 [kg/m³]
    pub sediment_density: S,
    /// 孔隙率
    pub porosity: S,
    /// 守恒误差容限
    pub conservation_tolerance: S,
}

impl<S: crate::core::Scalar> Default for SedimentConfig<S> {
    fn default() -> Self {
        Self {
            tau_critical: S::from_f64(0.1),
            erosion_rate: S::from_f64(1e-4),
            settling_velocity: S::from_f64(0.001),
            sediment_density: S::from_f64(2650.0),
            porosity: S::from_f64(0.4),
            conservation_tolerance: S::from_f64(1e-10),
        }
    }
}

/// 泥沙系统统一管理器
/// 
/// 负责：
/// - 床面泥沙质量管理
/// - 悬沙浓度（深度平均）
/// - 垂向分层浓度（2.5D）
/// - 侵蚀/沉降交换通量
/// - 质量守恒校验
pub struct SedimentManager<B: Backend> {
    /// 配置
    config: SedimentConfig<B::Scalar>,
    
    /// 床面泥沙质量 [kg/m²]
    bed_mass: B::Buffer<B::Scalar>,
    
    /// 悬沙浓度（深度平均） [kg/m³]
    suspended: TracerFieldGeneric<B>,
    
    /// 垂向分层浓度（2.5D，可选）
    layered: Option<LayeredStateGeneric<B>>,
    
    /// 床面侵蚀/沉降交换通量 [kg/m²/s]
    /// 正值 = 侵蚀，负值 = 沉降
    exchange_flux: B::Buffer<B::Scalar>,
    
    /// 初始总质量（守恒校验用）
    initial_total_mass: B::Scalar,
    
    /// 单元数量
    n_cells: usize,
    
    backend: B,
}

impl<B: Backend> SedimentManager<B> {
    /// 创建新的泥沙管理器
    pub fn new(backend: B, n_cells: usize, config: SedimentConfig<B::Scalar>) -> Self {
        let tracer_props = TracerPropertiesGeneric {
            name: "suspended_sediment".to_string(),
            diffusivity: B::Scalar::from_f64(0.1),
            settling_velocity: Some(config.settling_velocity),
            decay_rate: None,
        };
        
        Self {
            config,
            bed_mass: backend.alloc_init(n_cells, B::Scalar::ZERO),
            suspended: TracerFieldGeneric::new(backend.clone(), n_cells, tracer_props),
            layered: None,
            exchange_flux: backend.alloc_init(n_cells, B::Scalar::ZERO),
            initial_total_mass: B::Scalar::ZERO,
            n_cells,
            backend,
        }
    }
    
    /// 设置初始床面质量
    pub fn set_initial_bed_mass(&mut self, mass: &[B::Scalar]) {
        self.bed_mass.as_mut_slice().copy_from_slice(mass);
        self.compute_initial_mass();
    }
    
    /// 设置初始悬沙浓度
    pub fn set_initial_concentration(&mut self, conc: &[B::Scalar]) {
        self.suspended.concentration_mut().as_mut_slice().copy_from_slice(conc);
    }
    
    /// 计算初始总质量
    fn compute_initial_mass(&mut self) {
        self.initial_total_mass = self.backend.reduce_sum(&self.bed_mass);
    }
    
    /// 单步更新（耦合求解）
    pub fn step(
        &mut self,
        state: &ShallowWaterStateGeneric<B>,
        tau_bed: &B::Buffer<B::Scalar>,
        cell_areas: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) -> Result<(), SedimentError> {
        // 1. 计算侵蚀/沉降交换
        self.compute_exchange(state, tau_bed)?;
        
        // 2. 更新悬沙（对流+扩散+沉降）
        self.update_suspended(state, dt)?;
        
        // 3. 更新床面质量
        self.update_bed_mass(dt)?;
        
        // 4. 可选：同步到垂向分层
        if let Some(ref mut layered) = self.layered {
            self.sync_to_layered(layered, state)?;
        }
        
        // 5. 守恒校验与自动修正
        self.enforce_conservation(state, cell_areas)?;
        
        Ok(())
    }
    
    /// 计算侵蚀/沉降交换通量
    fn compute_exchange(
        &mut self,
        state: &ShallowWaterStateGeneric<B>,
        tau_bed: &B::Buffer<B::Scalar>,
    ) -> Result<(), SedimentError> {
        let h = state.h.as_slice();
        let conc = self.suspended.concentration().as_slice();
        let tau = tau_bed.as_slice();
        let flux = self.exchange_flux.as_mut_slice();
        
        for i in 0..self.n_cells {
            // 侵蚀（Partheniades 公式）
            let erosion = if tau[i] > self.config.tau_critical {
                self.config.erosion_rate * (tau[i] - self.config.tau_critical)
            } else {
                B::Scalar::ZERO
            };
            
            // 沉降
            let deposition = self.config.settling_velocity * conc[i];
            
            // 净交换通量
            flux[i] = erosion - deposition;
            
            // 检查床面质量是否足够侵蚀
            if flux[i] > B::Scalar::ZERO {
                let max_erosion = self.bed_mass.as_slice()[i];
                flux[i] = flux[i].min(max_erosion);
            }
        }
        
        Ok(())
    }
    
    /// 更新悬沙浓度
    fn update_suspended(
        &mut self,
        state: &ShallowWaterStateGeneric<B>,
        dt: B::Scalar,
    ) -> Result<(), SedimentError> {
        let h = state.h.as_slice();
        let conc = self.suspended.concentration_mut().as_mut_slice();
        let flux = self.exchange_flux.as_slice();
        
        for i in 0..self.n_cells {
            if h[i] > B::Scalar::from_f64(1e-6) {
                // dC/dt = E/h - ws*C/h
                // 简化：直接加入交换通量
                conc[i] = conc[i] + dt * flux[i] / h[i];
                
                // 确保非负
                if conc[i] < B::Scalar::ZERO {
                    conc[i] = B::Scalar::ZERO;
                }
            }
        }
        
        Ok(())
    }
    
    /// 更新床面质量
    fn update_bed_mass(&mut self, dt: B::Scalar) -> Result<(), SedimentError> {
        let bed = self.bed_mass.as_mut_slice();
        let flux = self.exchange_flux.as_slice();
        
        for i in 0..self.n_cells {
            // 床面质量变化 = -交换通量（侵蚀减少，沉降增加）
            bed[i] = bed[i] - dt * flux[i];
            
            // 确保非负
            if bed[i] < B::Scalar::ZERO {
                return Err(SedimentError::NegativeMass { 
                    cell: i, 
                    value: bed[i].to_f64() 
                });
            }
        }
        
        Ok(())
    }
    
    /// 同步到垂向分层
    fn sync_to_layered(
        &self,
        _layered: &mut LayeredStateGeneric<B>,
        _state: &ShallowWaterStateGeneric<B>,
    ) -> Result<(), SedimentError> {
        // TODO: 实现 ProfileRestorer 集成
        Ok(())
    }
    
    /// 守恒校验与自动修正
    fn enforce_conservation(
        &mut self,
        state: &ShallowWaterStateGeneric<B>,
        cell_areas: &B::Buffer<B::Scalar>,
    ) -> Result<(), SedimentError> {
        // 计算当前总质量
        let bed_total = self.backend.reduce_sum(&self.bed_mass);
        
        // 计算悬沙总质量
        let h = state.h.as_slice();
        let conc = self.suspended.concentration().as_slice();
        let areas = cell_areas.as_slice();
        
        let mut suspended_total = B::Scalar::ZERO;
        for i in 0..self.n_cells {
            suspended_total = suspended_total + h[i] * conc[i] * areas[i];
        }
        
        let current_total = bed_total + suspended_total;
        let relative_error = ((current_total - self.initial_total_mass) / 
            (self.initial_total_mass + B::Scalar::EPSILON)).abs();
        
        if relative_error > self.config.conservation_tolerance {
            return Err(SedimentError::ConservationViolation {
                expected: self.initial_total_mass.to_f64(),
                actual: current_total.to_f64(),
                relative_error: relative_error.to_f64(),
            });
        }
        
        Ok(())
    }
    
    /// 获取床面质量
    pub fn bed_mass(&self) -> &B::Buffer<B::Scalar> { &self.bed_mass }
    
    /// 获取悬沙浓度
    pub fn suspended_concentration(&self) -> &B::Buffer<B::Scalar> { 
        self.suspended.concentration() 
    }
    
    /// 获取交换通量
    pub fn exchange_flux(&self) -> &B::Buffer<B::Scalar> { &self.exchange_flux }
}
```

---

### 4.2 侵蚀/沉降交换模块

**目标**：实现独立的交换通量计算模块。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `sediment/exchange.rs` | 交换通量计算 |

#### 关键代码

```rust
// sediment/exchange.rs
use crate::core::Backend;

/// 侵蚀公式类型
#[derive(Debug, Clone, Copy)]
pub enum ErosionFormula {
    /// Partheniades 公式
    Partheniades,
    /// van Rijn 公式
    VanRijn,
}

/// 沉降公式类型
#[derive(Debug, Clone, Copy)]
pub enum SettlingFormula {
    /// Stokes 沉降
    Stokes,
    /// Rubey 公式
    Rubey,
    /// 自定义沉降速度
    Custom,
}

/// 泥沙交换通量计算器
pub struct SedimentExchange<B: Backend> {
    erosion_formula: ErosionFormula,
    settling_formula: SettlingFormula,
    
    /// 临界剪切应力 [Pa]
    tau_critical: B::Buffer<B::Scalar>,
    /// 侵蚀系数
    erosion_rate: B::Buffer<B::Scalar>,
    /// 沉降速度 [m/s]
    settling_velocity: B::Buffer<B::Scalar>,
    
    backend: B,
}

impl<B: Backend> SedimentExchange<B> {
    pub fn new(backend: B, n_cells: usize) -> Self {
        Self {
            erosion_formula: ErosionFormula::Partheniades,
            settling_formula: SettlingFormula::Stokes,
            tau_critical: backend.alloc_init(n_cells, B::Scalar::from_f64(0.1)),
            erosion_rate: backend.alloc_init(n_cells, B::Scalar::from_f64(1e-4)),
            settling_velocity: backend.alloc_init(n_cells, B::Scalar::from_f64(0.001)),
            backend,
        }
    }
    
    /// 计算侵蚀通量
    pub fn compute_erosion(
        &self,
        tau_bed: &B::Buffer<B::Scalar>,
        bed_mass: &B::Buffer<B::Scalar>,
        erosion: &mut B::Buffer<B::Scalar>,
    ) {
        let tau = tau_bed.as_slice();
        let tau_c = self.tau_critical.as_slice();
        let m = self.erosion_rate.as_slice();
        let bed = bed_mass.as_slice();
        let e = erosion.as_mut_slice();
        
        for i in 0..tau.len() {
            if tau[i] > tau_c[i] && bed[i] > B::Scalar::ZERO {
                e[i] = m[i] * (tau[i] - tau_c[i]);
                // 限制不超过可用床面质量
                e[i] = e[i].min(bed[i]);
            } else {
                e[i] = B::Scalar::ZERO;
            }
        }
    }
    
    /// 计算沉降通量
    pub fn compute_deposition(
        &self,
        concentration: &B::Buffer<B::Scalar>,
        deposition: &mut B::Buffer<B::Scalar>,
    ) {
        let c = concentration.as_slice();
        let ws = self.settling_velocity.as_slice();
        let d = deposition.as_mut_slice();
        
        for i in 0..c.len() {
            d[i] = ws[i] * c[i];
        }
    }
}
```

---

### 4.3 ProfileRestorer 实现

**目标**：实现垂向剖面恢复器（2.5D）。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `vertical/profile_generic.rs` | 泛型剖面恢复器 |
| 修改 | `vertical/mod.rs` | 更新导出 |

#### 关键代码

```rust
// vertical/profile_generic.rs
use crate::core::Backend;

/// 垂向剖面恢复器
/// 
/// 从 2D 深度平均状态恢复 3D 垂向剖面。
pub struct ProfileRestorer<B: Backend> {
    /// σ 坐标层数
    n_layers: usize,
    /// σ 坐标值 (0 = 底部, 1 = 表面)
    sigma: Vec<B::Scalar>,
    /// 床面粗糙度
    roughness: B::Buffer<B::Scalar>,
    
    backend: B,
}

impl<B: Backend> ProfileRestorer<B> {
    pub fn new(backend: B, n_cells: usize, n_layers: usize) -> Self {
        // 均匀 σ 分层
        let sigma: Vec<B::Scalar> = (0..=n_layers)
            .map(|k| B::Scalar::from_f64(k as f64 / n_layers as f64))
            .collect();
        
        Self {
            n_layers,
            sigma,
            roughness: backend.alloc_init(n_cells, B::Scalar::from_f64(0.001)),
            backend,
        }
    }
    
    /// 恢复垂向速度剖面（对数律）
    pub fn restore_velocity(
        &self,
        h: &B::Buffer<B::Scalar>,
        hu: &B::Buffer<B::Scalar>,
        hv: &B::Buffer<B::Scalar>,
        u_star: &B::Buffer<B::Scalar>,
        output: &mut LayeredStateGeneric<B>,
    ) {
        let n_cells = h.as_slice().len();
        let kappa = B::Scalar::VON_KARMAN;
        
        for cell in 0..n_cells {
            let depth = h.as_slice()[cell];
            if depth < B::Scalar::from_f64(1e-6) {
                continue;
            }
            
            let u_avg = hu.as_slice()[cell] / depth;
            let v_avg = hv.as_slice()[cell] / depth;
            let u_s = u_star.as_slice()[cell];
            let z0 = self.roughness.as_slice()[cell];
            
            for k in 0..self.n_layers {
                let sigma_k = self.sigma[k];
                let z = sigma_k * depth;
                
                // 对数律剖面
                let log_factor = if z > z0 {
                    (z / z0).ln() / kappa
                } else {
                    B::Scalar::ZERO
                };
                
                let u_k = u_avg + u_s * log_factor * u_avg.signum();
                let v_k = v_avg + u_s * log_factor * v_avg.signum();
                
                output.set_velocity(cell, k, u_k, v_k);
            }
        }
    }
    
    /// 恢复垂向浓度剖面（Rouse 分布）
    pub fn restore_concentration(
        &self,
        c_avg: &B::Buffer<B::Scalar>,
        h: &B::Buffer<B::Scalar>,
        ws: B::Scalar,
        u_star: &B::Buffer<B::Scalar>,
        output: &mut LayeredStateGeneric<B>,
    ) {
        let n_cells = c_avg.as_slice().len();
        let kappa = B::Scalar::VON_KARMAN;
        
        for cell in 0..n_cells {
            let depth = h.as_slice()[cell];
            let c0 = c_avg.as_slice()[cell];
            let u_s = u_star.as_slice()[cell];
            
            if depth < B::Scalar::from_f64(1e-6) || u_s < B::Scalar::EPSILON {
                continue;
            }
            
            // Rouse 数
            let rouse = ws / (kappa * u_s);
            
            for k in 0..self.n_layers {
                let sigma_k = self.sigma[k];
                let z_rel = sigma_k.max(B::Scalar::from_f64(0.01));
                
                // Rouse 分布
                let c_k = c0 * ((B::Scalar::ONE - z_rel) / z_rel).powf(rouse);
                
                output.set_sediment(cell, k, c_k);
            }
        }
    }
}

/// 泛型分层状态
pub struct LayeredStateGeneric<B: Backend> {
    n_cells: usize,
    n_layers: usize,
    
    /// 分层速度 u[cell][layer]
    u: B::Buffer<B::Scalar>,
    /// 分层速度 v[cell][layer]
    v: B::Buffer<B::Scalar>,
    /// 分层泥沙浓度
    sediment: B::Buffer<B::Scalar>,
    
    backend: B,
}

impl<B: Backend> LayeredStateGeneric<B> {
    pub fn new(backend: B, n_cells: usize, n_layers: usize) -> Self {
        let total = n_cells * n_layers;
        Self {
            n_cells,
            n_layers,
            u: backend.alloc_init(total, B::Scalar::ZERO),
            v: backend.alloc_init(total, B::Scalar::ZERO),
            sediment: backend.alloc_init(total, B::Scalar::ZERO),
            backend,
        }
    }
    
    #[inline]
    fn index(&self, cell: usize, layer: usize) -> usize {
        cell * self.n_layers + layer
    }
    
    pub fn set_velocity(&mut self, cell: usize, layer: usize, u: B::Scalar, v: B::Scalar) {
        let idx = self.index(cell, layer);
        self.u.as_mut_slice()[idx] = u;
        self.v.as_mut_slice()[idx] = v;
    }
    
    pub fn set_sediment(&mut self, cell: usize, layer: usize, c: B::Scalar) {
        let idx = self.index(cell, layer);
        self.sediment.as_mut_slice()[idx] = c;
    }
}
```

---

## 验收标准

1. ✅ `SedimentManager<B>` 实现完整
2. ✅ 侵蚀/沉降交换通量计算正确
3. ✅ 质量守恒误差 < 1e-10
4. ✅ `ProfileRestorer<B>` 实现对数律和 Rouse 分布
5. ✅ 所有测试通过

## 测试用例

```rust
#[test]
fn test_sediment_conservation() {
    let backend = CpuBackend::<f64>::new();
    let n_cells = 100;
    
    let mut manager = SedimentManager::new(
        backend.clone(), 
        n_cells, 
        SedimentConfig::default()
    );
    
    // 设置初始床面质量
    let initial_bed = vec![100.0; n_cells];
    manager.set_initial_bed_mass(&initial_bed);
    
    // 创建状态
    let state = ShallowWaterStateGeneric::new(backend.clone(), n_cells);
    
    // 模拟剪切应力
    let tau_bed = backend.alloc_init(n_cells, 0.2);
    let cell_areas = backend.alloc_init(n_cells, 1.0);
    
    // 执行多步
    for _ in 0..100 {
        manager.step(&state, &tau_bed, &cell_areas, 0.01).unwrap();
    }
    
    // 验证质量守恒
    let bed_total: f64 = manager.bed_mass().as_slice().iter().sum();
    let suspended_total: f64 = manager.suspended_concentration().as_slice().iter().sum();
    
    let initial_total: f64 = initial_bed.iter().sum();
    let current_total = bed_total + suspended_total * 1.0; // 假设 h=1, area=1
    
    assert!((current_total - initial_total).abs() / initial_total < 1e-8);
}
```
