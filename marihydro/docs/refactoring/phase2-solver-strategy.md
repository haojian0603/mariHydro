# Phase 2: 求解器策略化

## 目标

统一显式和半隐式为策略模式，完善 PCG 求解器。

## 时间：第 3-4 周

## 前置依赖

- Phase 0 完成（Backend 实例方法）
- Phase 1 完成（泛型状态）

## 任务清单

### 2.1 工作区泛型化

**目标**：创建 `SolverWorkspaceGeneric<B>` 泛型工作区。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `engine/workspace_generic.rs` | 泛型工作区实现 |
| 修改 | `engine/mod.rs` | 更新导出 |

#### 关键代码

```rust
// engine/workspace_generic.rs
use crate::core::Backend;

/// 泛型求解器工作区
pub struct SolverWorkspaceGeneric<B: Backend> {
    // 面通量
    pub flux_h: B::Buffer<B::Scalar>,
    pub flux_hu: B::Buffer<B::Scalar>,
    pub flux_hv: B::Buffer<B::Scalar>,
    
    // 单元 RHS
    pub rhs_h: B::Buffer<B::Scalar>,
    pub rhs_hu: B::Buffer<B::Scalar>,
    pub rhs_hv: B::Buffer<B::Scalar>,
    
    // 梯度
    pub grad_h: B::Buffer<[B::Scalar; 2]>,
    pub grad_z: B::Buffer<[B::Scalar; 2]>,
    
    // 半隐式专用
    pub u_star: B::Buffer<B::Scalar>,
    pub v_star: B::Buffer<B::Scalar>,
    pub eta_prime: B::Buffer<B::Scalar>,
    
    backend: B,
}

impl<B: Backend> SolverWorkspaceGeneric<B> {
    pub fn new(backend: B, n_cells: usize, n_faces: usize) -> Self {
        Self {
            flux_h: backend.alloc_init(n_faces, B::Scalar::ZERO),
            flux_hu: backend.alloc_init(n_faces, B::Scalar::ZERO),
            flux_hv: backend.alloc_init(n_faces, B::Scalar::ZERO),
            rhs_h: backend.alloc_init(n_cells, B::Scalar::ZERO),
            rhs_hu: backend.alloc_init(n_cells, B::Scalar::ZERO),
            rhs_hv: backend.alloc_init(n_cells, B::Scalar::ZERO),
            grad_h: backend.alloc_init(n_cells, [B::Scalar::ZERO; 2]),
            grad_z: backend.alloc_init(n_cells, [B::Scalar::ZERO; 2]),
            u_star: backend.alloc_init(n_cells, B::Scalar::ZERO),
            v_star: backend.alloc_init(n_cells, B::Scalar::ZERO),
            eta_prime: backend.alloc_init(n_cells, B::Scalar::ZERO),
            backend,
        }
    }
    
    pub fn reset(&mut self) {
        self.backend.fill(&mut self.rhs_h, B::Scalar::ZERO);
        self.backend.fill(&mut self.rhs_hu, B::Scalar::ZERO);
        self.backend.fill(&mut self.rhs_hv, B::Scalar::ZERO);
    }
}
```


---

### 2.2 时间积分策略 Trait

**目标**：定义 `TimeIntegrationStrategy<B>` trait，统一显式和半隐式接口。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 重构 | `engine/strategy/mod.rs` | 泛型策略 trait |
| 新建 | `engine/strategy/traits.rs` | 策略接口定义 |

#### 关键代码

```rust
// engine/strategy/traits.rs
use crate::core::Backend;
use crate::state::ShallowWaterStateGeneric;
use crate::mesh::MeshTopologyGeneric;

/// 时间积分步结果
#[derive(Debug, Clone)]
pub struct StepResult<S> {
    pub dt_used: S,
    pub max_wave_speed: S,
    pub dry_cells: usize,
    pub limited_cells: usize,
    pub converged: bool,
    pub iterations: usize,
}

/// 时间积分策略 Trait
pub trait TimeIntegrationStrategy<B: Backend>: Send + Sync {
    /// 策略名称
    fn name(&self) -> &'static str;
    
    /// 执行单步时间积分
    fn step(
        &mut self,
        state: &mut ShallowWaterStateGeneric<B>,
        mesh: &dyn MeshTopologyGeneric<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        dt: B::Scalar,
    ) -> StepResult<B::Scalar>;
    
    /// 计算稳定时间步长
    fn compute_stable_dt(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        mesh: &dyn MeshTopologyGeneric<B>,
        cfl: B::Scalar,
    ) -> B::Scalar;
    
    /// 是否支持大 CFL 数
    fn supports_large_cfl(&self) -> bool { false }
    
    /// 获取 Backend 引用
    fn backend(&self) -> &B;
}

/// 策略类型枚举
pub enum StrategyKindGeneric {
    Explicit(ExplicitConfig),
    SemiImplicit(SemiImplicitConfig),
}
```

---

### 2.3 显式策略重构

**目标**：实现 `ExplicitStrategyGeneric<B>`。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `engine/strategy/explicit_generic.rs` | 泛型显式策略 |

#### 关键代码

```rust
// engine/strategy/explicit_generic.rs
use crate::core::Backend;
use super::traits::{TimeIntegrationStrategy, StepResult};

/// 显式策略配置
#[derive(Debug, Clone)]
pub struct ExplicitConfig {
    pub riemann_type: RiemannSolverType,
    pub dry_tolerance: f64,
    pub use_muscl: bool,
}

impl Default for ExplicitConfig {
    fn default() -> Self {
        Self {
            riemann_type: RiemannSolverType::Hllc,
            dry_tolerance: 1e-6,
            use_muscl: true,
        }
    }
}

/// 泛型显式策略
pub struct ExplicitStrategyGeneric<B: Backend> {
    backend: B,
    config: ExplicitConfig,
}

impl<B: Backend> ExplicitStrategyGeneric<B> {
    pub fn new(backend: B, config: ExplicitConfig) -> Self {
        Self { backend, config }
    }
}

impl<B: Backend> TimeIntegrationStrategy<B> for ExplicitStrategyGeneric<B> {
    fn name(&self) -> &'static str { "Explicit-Godunov" }
    
    fn step(
        &mut self,
        state: &mut ShallowWaterStateGeneric<B>,
        mesh: &dyn MeshTopologyGeneric<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        
dt: B::Scalar,
    ) -> StepResult<B::Scalar> {
        // 1. 重置工作区
        workspace.reset();
        
        // 2. 计算通量（Riemann 求解器）
        let max_speed = self.compute_fluxes(state, mesh, workspace);
        
        // 3. 累加源项
        // sources.accumulate_all(state, workspace, dt);
        
        // 4. 更新状态
        self.update_state(state, workspace, dt);
        
        // 5. 正性保持
        state.enforce_positivity();
        
        StepResult {
            dt_used: dt,
            max_wave_speed: max_speed,
            dry_cells: 0,
            limited_cells: 0,
            converged: true,
            iterations: 1,
        }
    }
    
    fn compute_stable_dt(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        mesh: &dyn MeshTopologyGeneric<B>,
        cfl: B::Scalar,
    ) -> B::Scalar {
        // CFL 条件: dt = cfl * min(dx / (|u| + sqrt(gh)))
        // 简化实现
        B::Scalar::from_f64(0.001)
    }
    
    fn backend(&self) -> &B { &self.backend }
}
```

---

### 2.4 PCG 求解器实现

**目标**：实现预条件共轭梯度求解器。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `numerics/linear_algebra/pcg.rs` | PCG 求解器 |
| 新建 | `numerics/linear_algebra/csr_generic.rs` | 泛型 CSR 矩阵 |
| 修改 | `numerics/linear_algebra/mod.rs` | 更新导出 |

#### 关键代码

```rust
// numerics/linear_algebra/pcg.rs
use crate::core::Backend;

/// PCG 求解结果
#[derive(Debug, Clone)]
pub struct PcgResult<S> {
    pub converged: bool,
    pub iterations: usize,
    pub residual: S,
}

/// PCG 求解器
pub struct PcgSolver<B: Backend> {
    max_iterations: usize,
    tolerance: B::Scalar,
    
    // 工作向量
    r: B::Buffer<B::Scalar>,   // 残差
    z: B::Buffer<B::Scalar>,   // 预条件后残差
    p: B::Buffer<B::Scalar>,   // 搜索方向
    ap: B::Buffer<B::Scalar>,  // 矩阵-向量积
    
    backend: B,
}

impl<B: Backend> PcgSolver<B> {
    pub fn new(backend: B, n: usize, max_iter: usize, tol: B::Scalar) -> Self {
        Self {
            max_iterations: max_iter,
            tolerance: tol,
            r: backend.alloc_init(n, B::Scalar::ZERO),
            z: backend.alloc_init(n, B::Scalar::ZERO),
            p: backend.alloc_init(n, B::Scalar::ZERO),
            ap: backend.alloc_init(n, B::Scalar::ZERO),
            backend,
        }
    }
    
    /// 求解 Ax = b
    pub fn solve<M, P>(
        &mut self,
        matrix: &M,
        precond: &P,
        b: &B::Buffer<B::Scalar>,
        x: &mut B::Buffer<B::Scalar>,
    ) -> PcgResult<B::Scalar>
    where
        M: SparseMatrix<B>,
        P: Preconditioner<B>,
    {
        // r = b - Ax
        matrix.spmv(x, &mut self.r);
        // r = b - r
        for i in 0..b.as_slice().len() {
            self.r.as_mut_slice()[i] = b.as_slice()[i] - self.r.as_slice()[i];
        }
        
        let b_norm = self.backend.dot(b, b).sqrt();
        let mut r_norm = self.backend.dot(&self.r, &self.r).sqrt();
        
        if r_norm / b_norm < self.tolerance {
            return PcgResult { converged: true, iterations: 0, residual: r_norm };
        }
        
        // z = M^{-1} r
        precond.apply(&self.r, &mut self.z);
        
        // p = z
        self.backend.copy(&self.z, &mut self.p);
        
        let mut rz = self.backend.dot(&self.r, &self.z);
        
        for k in 0..self.max_iterations {
            // ap = A * p
            matrix.spmv(&self.p, &mut self.ap);
            
            // alpha = rz / (p · ap)
            let pap = self.backend.dot(&self.p, &self.ap);
            let alpha = rz / pap;
            
            // x = x + alpha * p
            self.backend.axpy(alpha, &self.p, x);
            
            // r = r - alpha * ap
            self.backend.axpy(-alpha, &self.ap, &mut self.r);
            
            r_norm = self.backend.dot(&self.r, &self.r).sqrt();
            
            if r_norm / b_norm < self.tolerance {
                return PcgResult { converged: true, iterations: k + 1, residual: r_norm };
            }
            
            // z = M^{-1} r
            precond.apply(&self.r, &mut self.z);
            
            let rz_new = self.backend.dot(&self.r, &self.z);
            let beta = rz_new / rz;
            rz = rz_new;
            
            // p = z + beta * p
            self.backend.scale(beta, &mut self.p);
            self.backend.axpy(B::Scalar::ONE, &self.z, &mut self.p);
        }
        
        PcgResult { converged: false, iterations: self.max_iterations, residual: r_norm }
    }
}

/// 稀疏矩阵 trait
pub trait SparseMatrix<B: Backend> {
    fn spmv(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>);
}

/// 预条件器 trait
pub trait Preconditioner<B: Backend> {
    fn apply(&self, r: &B::Buffer<B::Scalar>, z: &mut B::Buffer<B::Scalar>);
}

/// Jacobi 预条件器
pub struct JacobiPreconditioner<B: Backend> {
    diag_inv: B::Buffer<B::Scalar>,
    backend: B,
}

impl<B: Backend> JacobiPreconditioner<B> {
    pub fn new(backend: B, diag: &B::Buffer<B::Scalar>) -> Self {
        let n = diag.as_slice().len();
        let mut diag_inv = backend.alloc_init(n, B::Scalar::ZERO);
        for i in 0..n {
            let d = diag.as_slice()[i];
            diag_inv.as_mut_slice()[i] = if d.abs() > B::Scalar::EPSILON {
                B::Scalar::ONE / d
            } else {
                B::Scalar::ONE
            };
        }
        Self { diag_inv, backend }
    }
}

impl<B: Backend> Preconditioner<B> for JacobiPreconditioner<B> {
    fn apply(&self, r: &B::Buffer<B::Scalar>, z: &mut B::Buffer<B::Scalar>) {
        for i in 0..r.as_slice().len() {
            z.as_mut_slice()[i] = r.as_slice()[i] * self.diag_inv.as_slice()[i];
        }
    }
}
```

---

### 2.5 半隐式策略完善

**目标**：实现 `SemiImplicitStrategyGeneric<B>`，集成 PCG 求解器。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `engine/strategy/semi_implicit_generic.rs` | 泛型半隐式策略 |

#### 关键代码

```rust
// engine/strategy/semi_implicit_generic.rs
use crate::core::Backend;
use crate::numerics::linear_algebra::{PcgSolver, JacobiPreconditioner, CsrMatrix};

/// 半隐式策略配置
#[derive(Debug, Clone)]
pub struct SemiImplicitConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub theta: f64,  // 隐式权重 (0.5 = Crank-Nicolson, 1.0 = 全隐式)
}

impl Default for SemiImplicitConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-8,
            theta: 0.5,
        }
    }
}

/// 泛型半隐式策略
pub struct SemiImplicitStrategyGeneric<B: Backend> {
    backend: B,
    config: SemiImplicitConfig,
    pressure_matrix: Option<CsrMatrix<B>>,
    pcg_solver: PcgSolver<B>,
    preconditioner: Option<JacobiPreconditioner<B>>,
}

impl<B: Backend> SemiImplicitStrategyGeneric<B> {
    pub fn new(backend: B, n_cells: usize, config: SemiImplicitConfig) -> Self {
        let pcg_solver = PcgSolver::new(
            backend.clone(),
            n_cells,
            config.max_iterations,
            B::Scalar::from_f64(config.tolerance),
        );
        
        Self {
            backend,
            config,
            pressure_matrix: None,
            pcg_solver,
            preconditioner: None,
        }
    }
}

impl<B: Backend> TimeIntegrationStrategy<B> for SemiImplicitStrategyGeneric<B> {
    fn name(&self) -> &'static str { "Semi-Implicit-Projection" }
    
    fn step(
        &mut self,
        state: &mut ShallowWaterStateGeneric<B>,
        mesh: &dyn MeshTopologyGeneric<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        dt: B::Scalar,
    ) -> StepResult<B::Scalar> {
        // 1. 预测步：显式计算 u*, v*
        self.compute_prediction(state, mesh, workspace, dt);
        
        // 2. 组装压力 Poisson 矩阵
        self.assemble_pressure_matrix(state, mesh, dt);
        
        // 3. 计算 RHS：∇·(H u*)
        self.compute_divergence(state, workspace);
        
        // 4. PCG 求解 η'
        let pcg_result = if let (Some(matrix), Some(precond)) = 
            (&self.pressure_matrix, &self.preconditioner) 
        {
            self.pcg_solver.solve(
                matrix,
                precond,
                &workspace.rhs_h,
                &mut workspace.eta_prime,
            )
        } else {
            PcgResult { converged: false, iterations: 0, residual: B::Scalar::MAX }
        };
        
        // 5. 校正步：更新 u, v, h
        self.apply_correction(state, workspace, dt);
        
        StepResult {
            dt_used: dt,
            max_wave_speed: B::Scalar::ZERO,
            dry_cells: 0,
            limited_cells: 0,
            converged: pcg_result.converged,
            iterations: pcg_result.iterations,
        }
    }
    
    fn compute_stable_dt(
        &self,
        _state: &ShallowWaterStateGeneric<B>,
        _mesh: &dyn MeshTopologyGeneric<B>,
        _cfl: B::Scalar,
    ) -> B::Scalar {
        // 半隐式允许更大时间步
        B::Scalar::from_f64(0.1)
    }
    
    fn supports_large_cfl(&self) -> bool { true }
    
    fn backend(&self) -> &B { &self.backend }
}
```

---

### 2.6 统一求解器调度

**目标**：重构 `ShallowWaterSolver` 为纯调度器。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `engine/solver_generic.rs` | 泛型求解器调度器 |

#### 关键代码

```rust
// engine/solver_generic.rs
use crate::core::Backend;
use std::sync::Arc;

/// 泛型浅水方程求解器
pub struct ShallowWaterSolverGeneric<B: Backend> {
    mesh: Arc<dyn MeshTopologyGeneric<B>>,
    state: ShallowWaterStateGeneric<B>,
    strategy: Box<dyn TimeIntegrationStrategy<B>>,
    workspace: SolverWorkspaceGeneric<B>,
    current_time: f64,
}

impl<B: Backend> ShallowWaterSolverGeneric<B> {
    pub fn new(
        backend: B,
        mesh: Arc<dyn MeshTopologyGeneric<B>>,
        strategy_kind: StrategyKindGeneric,
    ) -> Self {
        let n_cells = mesh.n_cells();
        let n_faces = mesh.n_faces();
        
        let state = ShallowWaterStateGeneric::new(backend.clone(), n_cells);
        let workspace = SolverWorkspaceGeneric::new(backend.clone(), n_cells, n_faces);
        
        let strategy: Box<dyn TimeIntegrationStrategy<B>> = match strategy_kind {
            StrategyKindGeneric::Explicit(cfg) => {
                Box::new(ExplicitStrategyGeneric::new(backend, cfg))
            }
            StrategyKindGeneric::SemiImplicit(cfg) => {
                Box::new(SemiImplicitStrategyGeneric::new(backend, n_cells, cfg))
            }
        };
        
        Self {
            mesh,
            state,
            strategy,
            workspace,
            current_time: 0.0,
        }
    }
    
    /// 执行单步
    pub fn step(&mut self, dt: B::Scalar) -> StepResult<B::Scalar> {
        let result = self.strategy.step(
            &mut self.state,
            self.mesh.as_ref(),
            &mut self.workspace,
            dt,
        );
        
        self.current_time += dt.to_f64();
        result
    }
    
    /// 运行时切换策略
    pub fn set_strategy(&mut self, kind: StrategyKindGeneric) {
        let backend = self.strategy.backend().clone();
        let n_cells = self.mesh.n_cells();
        
        self.strategy = match kind {
            StrategyKindGeneric::Explicit(cfg) => {
                Box::new(ExplicitStrategyGeneric::new(backend, cfg))
            }
            StrategyKindGeneric::SemiImplicit(cfg) => {
                Box::new(SemiImplicitStrategyGeneric::new(backend, n_cells, cfg))
            }
        };
    }
    
    /// 获取状态引用
    pub fn state(&self) -> &ShallowWaterStateGeneric<B> { &self.state }
    
    /// 获取可变状态引用
    pub fn state_mut(&mut self) -> &mut ShallowWaterStateGeneric<B> { &mut self.state }
    
    /// 当前时间
    pub fn current_time(&self) -> f64 { self.current_time }
}
```

---

## 验收标准

1. ✅ `TimeIntegrationStrategy<B>` trait 定义完整
2. ✅ `ExplicitStrategyGeneric<B>` 实现并通过测试
3. ✅ `SemiImplicitStrategyGeneric<B>` 实现并通过测试
4. ✅ PCG 求解器收敛测试通过
5. ✅ 策略可运行时切换
6. ✅ 所有现有测试通过

## 测试用例

```rust
#[test]
fn test_strategy_switching() {
    let backend = CpuBackend::<f64>::new();
    let mesh = create_test_mesh(backend.clone());
    
    let mut solver = ShallowWaterSolverGeneric::new(
        backend,
        Arc::new(mesh),
        StrategyKindGeneric::Explicit(ExplicitConfig::default()),
    );
    
    // 显式步进
    let result1 = solver.step(0.001);
    assert!(result1.converged);
    
    // 切换到半隐式
    solver.set_strategy(StrategyKindGeneric::SemiImplicit(SemiImplicitConfig::default()));
    
    // 半隐式步进
    let result2 = solver.step(0.01);
    assert!(result2.converged);
}

#[test]
fn test_pcg_convergence() {
    let backend = CpuBackend::<f64>::new();
    let n = 100;
    
    // 创建简单对角矩阵测试
    let mut solver = PcgSolver::new(backend.clone(), n, 100, 1e-10);
    
    // ... 测试 PCG 收敛性
}
```
