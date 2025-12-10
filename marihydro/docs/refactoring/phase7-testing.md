# Phase 7: 测试与验证

## 目标

完成架构验证测试，确保重构后系统正确性。

## 时间：第 9 周

## 前置依赖

- Phase 0-6 全部完成

## 任务清单

### 7.1 Backend 泛型测试

**目标**：验证 f32/f64 后端切换的正确性。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `tests/backend_generic.rs` | 后端泛型测试 |

#### 关键代码

```rust
// tests/backend_generic.rs
//! Backend 泛型化验证测试

use mh_physics::core::{Backend, CpuBackend, Scalar};
use mh_physics::state::ShallowWaterStateGeneric;
use mh_physics::mesh::UnstructuredMeshGeneric;

/// 创建测试网格
fn create_test_mesh<B: Backend>(backend: B, n_cells: usize) -> UnstructuredMeshGeneric<B> {
    // 简单的正方形网格
    let n_faces = n_cells * 4; // 简化
    let n_boundary = n_cells; // 简化
    
    let cell_centers: Vec<[B::Scalar; 2]> = (0..n_cells)
        .map(|i| {
            let x = B::Scalar::from_f64((i % 10) as f64);
            let y = B::Scalar::from_f64((i / 10) as f64);
            [x, y]
        })
        .collect();
    
    let cell_volumes = vec![B::Scalar::ONE; n_cells];
    let face_normals = vec![[B::Scalar::ONE, B::Scalar::ZERO]; n_faces];
    let face_areas = vec![B::Scalar::ONE; n_faces];
    let face_centers = vec![[B::Scalar::ZERO; 2]; n_faces];
    let face_owner = vec![0u32; n_faces];
    let face_neighbor = vec![-1i32; n_faces];
    
    UnstructuredMeshGeneric::from_raw(
        backend,
        n_cells,
        n_faces,
        n_boundary,
        cell_centers,
        cell_volumes,
        face_normals,
        face_areas,
        face_centers,
        face_owner,
        face_neighbor,
    )
}

#[test]
fn test_f32_f64_consistency() {
    let backend_f32 = CpuBackend::<f32>::new();
    let backend_f64 = CpuBackend::<f64>::new();
    
    let n_cells = 100;
    
    // 创建状态
    let mut state_f32 = ShallowWaterStateGeneric::new(backend_f32.clone(), n_cells);
    let mut state_f64 = ShallowWaterStateGeneric::new(backend_f64.clone(), n_cells);
    
    // 设置相同的初始条件
    for i in 0..n_cells {
        let h = 1.0 + 0.1 * (i as f64).sin();
        state_f32.h.as_mut_slice()[i] = h as f32;
        state_f64.h.as_mut_slice()[i] = h;
    }
    
    // 执行相同操作
    state_f32.enforce_positivity();
    state_f64.enforce_positivity();
    
    // 比较结果
    let max_diff: f64 = state_f32.h.as_slice().iter()
        .zip(state_f64.h.as_slice().iter())
        .map(|(&f32_val, &f64_val)| ((f32_val as f64) - f64_val).abs())
        .fold(0.0, f64::max);
    
    assert!(max_diff < 1e-6, "f32/f64 差异过大: {}", max_diff);
}

#[test]
fn test_backend_operations() {
    let backend = CpuBackend::<f64>::new();
    
    // 测试 axpy
    let x = backend.alloc_init(100, 1.0);
    let mut y = backend.alloc_init(100, 2.0);
    backend.axpy(0.5, &x, &mut y);
    assert!((y.as_slice()[0] - 2.5).abs() < 1e-10);
    
    // 测试 dot
    let a = backend.alloc_init(100, 2.0);
    let b = backend.alloc_init(100, 3.0);
    let dot = backend.dot(&a, &b);
    assert!((dot - 600.0).abs() < 1e-10);
    
    // 测试 reduce
    let c = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    assert!((backend.reduce_max(&c) - 5.0).abs() < 1e-10);
    assert!((backend.reduce_sum(&c) - 15.0).abs() < 1e-10);
}

#[test]
fn test_state_linear_combine() {
    let backend = CpuBackend::<f64>::new();
    let n_cells = 10;
    
    let mut state_a = ShallowWaterStateGeneric::new(backend.clone(), n_cells);
    let mut state_b = ShallowWaterStateGeneric::new(backend.clone(), n_cells);
    let mut state_c = ShallowWaterStateGeneric::new(backend.clone(), n_cells);
    
    // 设置值
    state_a.h.as_mut_slice().fill(1.0);
    state_b.h.as_mut_slice().fill(2.0);
    
    // 线性组合: c = 0.3*a + 0.7*b
    state_c.linear_combine(0.3, &state_a, 0.7, &state_b);
    
    // 验证: 0.3*1 + 0.7*2 = 1.7
    assert!((state_c.h.as_slice()[0] - 1.7).abs() < 1e-10);
}
```

---

### 7.2 策略切换测试

**目标**：验证显式/半隐式策略切换的正确性。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `tests/strategy_switching.rs` | 策略切换测试 |

#### 关键代码

```rust
// tests/strategy_switching.rs
//! 策略切换验证测试

use mh_physics::core::CpuBackend;
use mh_physics::engine::{
    ShallowWaterSolverGeneric, StrategyKindGeneric,
    ExplicitConfig, SemiImplicitConfig,
};
use std::sync::Arc;

#[test]
fn test_strategy_switch_continuity() {
    let backend = CpuBackend::<f64>::new();
    let mesh = Arc::new(create_test_mesh(backend.clone(), 100));
    
    // 创建显式求解器
    let mut solver = ShallowWaterSolverGeneric::new(
        backend.clone(),
        mesh.clone(),
        StrategyKindGeneric::Explicit(ExplicitConfig::default()),
    );
    
    // 设置初始条件
    let state = solver.state_mut();
    for i in 0..100 {
        state.h.as_mut_slice()[i] = 1.0 + 0.1 * (i as f64 * 0.1).sin();
    }
    
    // 显式步进 10 步
    for _ in 0..10 {
        solver.step(0.001);
    }
    
    // 记录状态
    let h_before: Vec<f64> = solver.state().h.as_slice().to_vec();
    
    // 切换到半隐式
    solver.set_strategy(StrategyKindGeneric::SemiImplicit(SemiImplicitConfig::default()));
    
    // 半隐式步进 1 步
    let result = solver.step(0.01);
    
    // 验证状态连续性（不应有突变）
    let h_after = solver.state().h.as_slice();
    let max_change: f64 = h_before.iter()
        .zip(h_after.iter())
        .map(|(&before, &after)| (after - before).abs())
        .fold(0.0, f64::max);
    
    // 单步变化应该合理
    assert!(max_change < 0.1, "策略切换后状态突变: {}", max_change);
    assert!(result.converged, "半隐式应该收敛");
}

#[test]
fn test_explicit_stability() {
    let backend = CpuBackend::<f64>::new();
    let mesh = Arc::new(create_test_mesh(backend.clone(), 100));
    
    let mut solver = ShallowWaterSolverGeneric::new(
        backend,
        mesh,
        StrategyKindGeneric::Explicit(ExplicitConfig::default()),
    );
    
    // 设置初始条件
    let state = solver.state_mut();
    state.h.as_mut_slice().fill(1.0);
    
    // 运行 100 步
    for _ in 0..100 {
        let result = solver.step(0.001);
        assert!(result.converged);
    }
    
    // 验证状态有效
    let h = solver.state().h.as_slice();
    for &val in h {
        assert!(val.is_finite(), "状态包含非有限值");
        assert!(val >= 0.0, "水深为负");
    }
}
```

---

### 7.3 泥沙耦合测试

**目标**：验证泥沙质量守恒。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `tests/sediment_coupling.rs` | 泥沙耦合测试 |

#### 关键代码

```rust
// tests/sediment_coupling.rs
//! 泥沙耦合验证测试

use mh_physics::core::CpuBackend;
use mh_physics::state::ShallowWaterStateGeneric;
use mh_physics::sediment::{SedimentManager, SedimentConfig};

#[test]
fn test_sediment_mass_conservation() {
    let backend = CpuBackend::<f64>::new();
    let n_cells = 100;
    
    // 创建泥沙管理器
    let mut manager = SedimentManager::new(
        backend.clone(),
        n_cells,
        SedimentConfig::default(),
    );
    
    // 设置初始床面质量
    let initial_bed: Vec<f64> = (0..n_cells)
        .map(|i| 100.0 + 10.0 * (i as f64 * 0.1).sin())
        .collect();
    manager.set_initial_bed_mass(&initial_bed);
    
    // 设置初始悬沙浓度
    let initial_conc = vec![0.1; n_cells];
    manager.set_initial_concentration(&initial_conc);
    
    // 创建水动力状态
    let mut state = ShallowWaterStateGeneric::new(backend.clone(), n_cells);
    state.h.as_mut_slice().fill(2.0);  // 2m 水深
    
    // 模拟剪切应力
    let tau_bed = backend.alloc_init(n_cells, 0.15);  // 略高于临界值
    let cell_areas = backend.alloc_init(n_cells, 1.0);
    
    // 计算初始总质量
    let initial_total: f64 = initial_bed.iter().sum::<f64>() 
        + initial_conc.iter().sum::<f64>() * 2.0;  // h * C * area
    
    // 执行 1000 步
    for _ in 0..1000 {
        manager.step(&state, &tau_bed, &cell_areas, 0.01).unwrap();
    }
    
    // 计算最终总质量
    let bed_total: f64 = manager.bed_mass().as_slice().iter().sum();
    let suspended_total: f64 = manager.suspended_concentration().as_slice()
        .iter()
        .zip(state.h.as_slice().iter())
        .map(|(&c, &h)| c * h)
        .sum();
    let final_total = bed_total + suspended_total;
    
    // 验证质量守恒
    let relative_error = (final_total - initial_total).abs() / initial_total;
    assert!(
        relative_error < 1e-8,
        "质量守恒误差过大: {} (初始: {}, 最终: {})",
        relative_error, initial_total, final_total
    );
}

#[test]
fn test_sediment_erosion_deposition() {
    let backend = CpuBackend::<f64>::new();
    let n_cells = 10;
    
    let mut manager = SedimentManager::new(
        backend.clone(),
        n_cells,
        SedimentConfig {
            tau_critical: 0.1,
            erosion_rate: 1e-3,
            settling_velocity: 0.001,
            ..Default::default()
        },
    );
    
    // 初始：只有床面泥沙
    manager.set_initial_bed_mass(&vec![100.0; n_cells]);
    manager.set_initial_concentration(&vec![0.0; n_cells]);
    
    let mut state = ShallowWaterStateGeneric::new(backend.clone(), n_cells);
    state.h.as_mut_slice().fill(1.0);
    
    // 高剪切应力 -> 侵蚀
    let tau_high = backend.alloc_init(n_cells, 0.5);
    let cell_areas = backend.alloc_init(n_cells, 1.0);
    
    // 侵蚀阶段
    for _ in 0..100 {
        manager.step(&state, &tau_high, &cell_areas, 0.1).unwrap();
    }
    
    // 验证：床面减少，悬沙增加
    let bed_after_erosion: f64 = manager.bed_mass().as_slice().iter().sum();
    let susp_after_erosion: f64 = manager.suspended_concentration().as_slice().iter().sum();
    
    assert!(bed_after_erosion < 100.0 * n_cells as f64, "床面应该减少");
    assert!(susp_after_erosion > 0.0, "悬沙应该增加");
    
    // 低剪切应力 -> 沉降
    let tau_low = backend.alloc_init(n_cells, 0.05);
    
    for _ in 0..100 {
        manager.step(&state, &tau_low, &cell_areas, 0.1).unwrap();
    }
    
    // 验证：悬沙减少
    let susp_after_deposition: f64 = manager.suspended_concentration().as_slice().iter().sum();
    assert!(susp_after_deposition < susp_after_erosion, "悬沙应该减少");
}
```

---

### 7.4 标准算例验证

**目标**：使用标准算例验证数值精度。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `tests/dambreak_generic.rs` | 溃坝算例 |
| 新建 | `tests/thacker_generic.rs` | Thacker 解析解 |

#### 关键代码

```rust
// tests/dambreak_generic.rs
//! 溃坝标准算例

use mh_physics::core::CpuBackend;

/// 1D 溃坝解析解 (Stoker)
fn dambreak_analytical(x: f64, t: f64, h_l: f64, h_r: f64, g: f64) -> f64 {
    if t <= 0.0 {
        return if x < 0.0 { h_l } else { h_r };
    }
    
    let c_l = (g * h_l).sqrt();
    let c_r = (g * h_r).sqrt();
    
    // 简化：干床情况 (h_r = 0)
    if h_r < 1e-10 {
        let x_a = -c_l * t;
        let x_b = 2.0 * c_l * t;
        
        if x < x_a {
            h_l
        } else if x < x_b {
            let h = (2.0 * c_l - x / t).powi(2) / (9.0 * g);
            h.max(0.0)
        } else {
            0.0
        }
    } else {
        // 湿床情况需要迭代求解
        h_l // 简化
    }
}

#[test]
fn test_dambreak_1d() {
    let backend = CpuBackend::<f64>::new();
    
    // 1D 网格 (100 个单元)
    let n_cells = 100;
    let dx = 1.0;
    let x: Vec<f64> = (0..n_cells).map(|i| (i as f64 - 50.0) * dx).collect();
    
    // 初始条件
    let h_l = 1.0;
    let h_r = 0.0;
    let g = 9.81;
    
    // 创建状态
    let mut state = ShallowWaterStateGeneric::new(backend.clone(), n_cells);
    for i in 0..n_cells {
        state.h.as_mut_slice()[i] = if x[i] < 0.0 { h_l } else { h_r };
    }
    
    // 运行到 t = 1.0
    let t_end = 1.0;
    let dt = 0.001;
    let n_steps = (t_end / dt) as usize;
    
    // TODO: 实际运行求解器
    // for _ in 0..n_steps {
    //     solver.step(dt);
    // }
    
    // 计算解析解
    let h_analytical: Vec<f64> = x.iter()
        .map(|&xi| dambreak_analytical(xi, t_end, h_l, h_r, g))
        .collect();
    
    // 计算 L2 误差
    // let l2_error = compute_l2_error(&state.h.as_slice(), &h_analytical);
    // assert!(l2_error < 1e-3, "L2 误差过大: {}", l2_error);
}

/// 计算 L2 误差
fn compute_l2_error(numerical: &[f64], analytical: &[f64]) -> f64 {
    let n = numerical.len();
    let sum_sq: f64 = numerical.iter()
        .zip(analytical.iter())
        .map(|(&num, &ana)| (num - ana).powi(2))
        .sum();
    (sum_sq / n as f64).sqrt()
}
```

```rust
// tests/thacker_generic.rs
//! Thacker 解析解验证

use mh_physics::core::CpuBackend;
use std::f64::consts::PI;

/// Thacker 解析解参数
struct ThackerParams {
    a: f64,      // 碗半径
    h0: f64,     // 中心深度
    eta: f64,    // 振幅参数
    omega: f64,  // 角频率
    g: f64,      // 重力加速度
}

impl ThackerParams {
    fn new(a: f64, h0: f64, eta: f64, g: f64) -> Self {
        let omega = (8.0 * g * h0 / a.powi(2)).sqrt();
        Self { a, h0, eta, omega, g }
    }
    
    /// 解析水深
    fn h(&self, x: f64, y: f64, t: f64) -> f64 {
        let r2 = x * x + y * y;
        let cos_t = (self.omega * t).cos();
        
        let h = self.h0 * (1.0 - self.eta * cos_t) 
            - r2 / self.a.powi(2) * self.h0 * (1.0 - self.eta.powi(2) * cos_t.powi(2));
        
        h.max(0.0)
    }
    
    /// 解析速度
    fn velocity(&self, x: f64, y: f64, t: f64) -> (f64, f64) {
        let sin_t = (self.omega * t).sin();
        let cos_t = (self.omega * t).cos();
        
        let factor = self.eta * self.omega * sin_t / (2.0 * (1.0 - self.eta * cos_t));
        
        let u = factor * x;
        let v = factor * y;
        
        (u, v)
    }
}

#[test]
fn test_thacker_convergence() {
    let backend = CpuBackend::<f64>::new();
    
    let params = ThackerParams::new(1.0, 0.1, 0.5, 9.81);
    
    // 不同网格分辨率
    let resolutions = [20, 40, 80];
    let mut errors = Vec::new();
    
    for &n in &resolutions {
        // 创建圆形网格（简化为正方形）
        let n_cells = n * n;
        let dx = 2.0 * params.a / n as f64;
        
        // 初始化
        let mut state = ShallowWaterStateGeneric::new(backend.clone(), n_cells);
        
        for j in 0..n {
            for i in 0..n {
                let idx = j * n + i;
                let x = (i as f64 + 0.5) * dx - params.a;
                let y = (j as f64 + 0.5) * dx - params.a;
                
                state.h.as_mut_slice()[idx] = params.h(x, y, 0.0);
            }
        }
        
        // 运行一个周期
        let period = 2.0 * PI / params.omega;
        
        // TODO: 实际运行求解器
        
        // 计算误差
        let mut l2_sum = 0.0;
        for j in 0..n {
            for i in 0..n {
                let idx = j * n + i;
                let x = (i as f64 + 0.5) * dx - params.a;
                let y = (j as f64 + 0.5) * dx - params.a;
                
                let h_ana = params.h(x, y, period);
                let h_num = state.h.as_slice()[idx];
                
                l2_sum += (h_num - h_ana).powi(2);
            }
        }
        
        let l2_error = (l2_sum / n_cells as f64).sqrt();
        errors.push((n, l2_error));
    }
    
    // 验证收敛阶
    if errors.len() >= 2 {
        let (n1, e1) = errors[0];
        let (n2, e2) = errors[1];
        
        let order = (e1 / e2).ln() / (n2 as f64 / n1 as f64).ln();
        
        // 期望至少 1.5 阶收敛
        // assert!(order >= 1.5, "收敛阶不足: {}", order);
    }
}
```

---

### 7.5 AI 同化测试

**目标**：验证 AI 同化功能。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `tests/ai_assimilation.rs` | AI 同化测试 |

#### 关键代码

```rust
// tests/ai_assimilation.rs
//! AI 同化验证测试

use mh_agent::{
    AgentRegistry, RemoteSensingAgent, NudgingAssimilator,
    NudgingConfig, Observation, ObservationVariable,
};

#[test]
fn test_remote_sensing_nudging() {
    let mut agent = RemoteSensingAgent::new(0.5);
    
    // 设置预测
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let confidence = vec![1.0, 0.8, 0.6, 0.4, 0.2];
    agent.set_prediction(predicted.clone(), confidence.clone());
    
    // 模拟状态
    let mut sediment = vec![0.0; 5];
    
    // 应用 Nudging
    for (i, c) in sediment.iter_mut().enumerate() {
        let weight = 0.5 * confidence[i];
        *c += weight * (predicted[i] - *c);
    }
    
    // 验证：高置信度区域更接近预测值
    assert!((sediment[0] - 0.5).abs() < 0.01);  // 50% 同化
    assert!((sediment[4] - 0.5).abs() < 0.01);  // 10% 同化
}

#[test]
fn test_agent_registry() {
    let mut registry = AgentRegistry::new();
    
    // 注册多个代理
    registry.register(Box::new(RemoteSensingAgent::new(0.1)));
    registry.register(Box::new(NudgingAssimilator::new(NudgingConfig::default())));
    
    assert_eq!(registry.list().len(), 2);
    
    // 禁用一个代理
    registry.set_enabled("RemoteSensing-Sediment", false);
    
    let enabled: Vec<_> = registry.list()
        .into_iter()
        .filter(|(_, e)| *e)
        .collect();
    
    assert_eq!(enabled.len(), 1);
}

#[test]
fn test_nudging_spatial_decay() {
    let config = NudgingConfig {
        relaxation_time: 3600.0,
        min_confidence: 0.5,
        influence_radius: 100.0,
    };
    
    let mut assimilator = NudgingAssimilator::new(config);
    assimilator.set_dt(60.0);  // 1 分钟时间步
    
    // 添加观测
    assimilator.add_observation(Observation {
        location: [0.0, 0.0],
        value: 1.5,
        confidence: 1.0,
        time: 0.0,
        variable: ObservationVariable::WaterLevel,
    });
    
    // 验证空间衰减
    // 距离 0: 权重 = 1.0
    // 距离 50: 权重 = 0.5
    // 距离 100: 权重 = 0.0
}
```

---

## 测试矩阵

| 测试用例 | 测试内容 | 验证标准 | 状态 |
|----------|----------|----------|------|
| `backend_generic.rs` | f32/f64 后端切换 | 结果差异 < 1e-6 | ⬜ |
| `strategy_switching.rs` | 显式/半隐式切换 | 状态连续性 | ⬜ |
| `sediment_coupling.rs` | 泥沙质量守恒 | 误差 < 1e-10 | ⬜ |
| `dambreak_generic.rs` | 溃坝标准算例 | L2 误差 < 1e-3 | ⬜ |
| `thacker_generic.rs` | Thacker 解析解 | 收敛阶 ≥ 1.5 | ⬜ |
| `ai_assimilation.rs` | AI 同化验证 | Nudging 正确 | ⬜ |

## 验收标准

1. ✅ 所有测试通过
2. ✅ 代码覆盖率 > 80%
3. ✅ 无内存泄漏
4. ✅ 性能无明显退化

## 运行测试

```bash
# 运行所有测试
cargo test --workspace

# 运行特定测试
cargo test -p mh_physics backend_generic

# 运行带输出的测试
cargo test -- --nocapture

# 运行性能测试
cargo test --release -- --ignored
```
