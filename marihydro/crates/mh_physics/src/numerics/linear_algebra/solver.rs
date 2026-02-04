// crates/mh_physics/src/numerics/linear_algebra/solver.rs
//! 迭代线性求解器
//!
//! 提供用于求解稀疏线性系统 Ax = b 的迭代方法：
//! 支持泛型标量类型 `S: RuntimeScalar`（f32 或 f64）。
//!
//! # 求解器类型
//!
//! - [`ConjugateGradient`]: 共轭梯度法（CG）
//! - [`PcgSolver`]: 预条件共轭梯度法（PCG）
//! - [`BiCgStabSolver`]: 双共轭梯度稳定法（BiCGStab）
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::numerics::linear_algebra::{
//!     CsrMatrix, PcgSolver, JacobiPreconditioner, SolverConfig,
//! };
//! use mh_runtime::CpuBackend;
//!
//! let backend = CpuBackend::<f64>::new();
//! let matrix: CsrMatrix<f64> = /* ... */;
//! let mut b = backend.alloc(3);
//! b.copy_from_slice(&[1.0, 2.0, 3.0]);
//! let mut x = backend.alloc_init(3, 0.0);
//!
//! let precond = JacobiPreconditioner::from_matrix(&backend, &matrix).unwrap();
//! let config = SolverConfig::new(1e-8, 100);
//! let mut solver = PcgSolver::new(backend.clone(), config);
//!
//! let result = solver.solve(&matrix, &b, &mut x, &precond);
//! println!("Converged in {} iterations", result.iterations);
//! ```

use super::csr::CsrMatrix;
use super::preconditioner::Preconditioner;
use crate::core::kernel::spmv_kernel;
use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar};
use num_traits::Float;
use serde::{Deserialize, Serialize};

// ============================================================================
// 配置层 (Layer 4) - 允许使用 f64
// ============================================================================

/// 求解器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// 相对收敛容差
    pub rtol: f64, // ALLOW_F64: Layer 4 配置参数
    /// 绝对收敛容差
    pub atol: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最大迭代次数
    pub max_iter: usize,
    /// 是否打印迭代信息
    pub verbose: bool,
    /// 停滞判定阈值（相对残差变化率）
    pub stagnation_tol: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-14,
            max_iter: 1000,
            verbose: false,
            stagnation_tol: 1e-12,
        }
    }
}

impl SolverConfig {
    /// 创建求解器配置
    // ALLOW_F64: Layer 4 配置参数构造方法
    pub fn new(rtol: f64, max_iter: usize) -> Self {
        Self {
            rtol,
            max_iter,
            ..Default::default()
        }
    }

    /// 设置绝对容差
    // ALLOW_F64: Layer 4 配置参数设置方法
    pub fn with_atol(mut self, atol: f64) -> Self {
        self.atol = atol;
        self
    }

    /// 启用详细输出
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

/// 求解器状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverStatus {
    /// 收敛
    Converged,
    /// 达到最大迭代次数
    MaxIterationsReached,
    /// 发散
    Diverged,
    /// 停滞
    Stagnated,
    /// 外部提前停止
    Stopped,
}

/// 求解器结果
#[derive(Debug, Clone)]
pub struct SolverResult<S: RuntimeScalar> {
    /// 求解状态
    pub status: SolverStatus,
    /// 迭代次数
    pub iterations: usize,
    /// 最终残差范数
    pub residual_norm: S,
    /// 初始残差范数
    pub initial_residual_norm: S,
    /// 相对残差
    pub relative_residual: S,
}

/// 迭代过程观察者（早停钩子）
pub trait IterationObserver<S: RuntimeScalar>: Send + Sync {
    /// 返回 true 则提前停止
    fn should_stop(&mut self, iter: usize, residual_norm: S, relative_residual: S) -> bool;
}

impl<S: RuntimeScalar> SolverResult<S> {
    /// 是否成功收敛
    pub fn is_converged(&self) -> bool {
        self.status == SolverStatus::Converged
    }

    /// 返回是否收敛（别名）
    pub fn converged(&self) -> bool {
        self.is_converged()
    }
}

/// CG 求解器工作区
///
/// 预分配的工作向量，避免 solve 内部频繁分配
#[derive(Debug, Clone)]
pub struct CgWorkspace<B: Backend> {
    /// 残差向量
    pub r: B::Buffer<B::Scalar>,
    /// 搜索方向
    pub p: B::Buffer<B::Scalar>,
    /// A*p
    pub ap: B::Buffer<B::Scalar>,
    /// 预条件后的残差
    pub z: B::Buffer<B::Scalar>,
}

impl<B: Backend> CgWorkspace<B> {
    /// 创建新的工作区
    pub fn new(backend: &B, n: usize) -> Self {
        let mut r = backend.alloc(n);
        let mut p = backend.alloc(n);
        let mut ap = backend.alloc(n);
        let mut z = backend.alloc(n);
        r.fill(B::Scalar::ZERO);
        p.fill(B::Scalar::ZERO);
        ap.fill(B::Scalar::ZERO);
        z.fill(B::Scalar::ZERO);
        Self { r, p, ap, z }
    }

    /// 调整工作区大小并清零
    ///
    /// 无论大小是否变化，均清零防止历史数据污染。
    pub fn resize(&mut self, backend: &B, n: usize) {
        if self.r.len() != n {
            self.r = backend.alloc(n);
            self.p = backend.alloc(n);
            self.ap = backend.alloc(n);
            self.z = backend.alloc(n);
        }
        self.clear();
    }

    /// 清零工作区
    pub fn clear(&mut self) {
        self.r.fill(B::Scalar::ZERO);
        self.p.fill(B::Scalar::ZERO);
        self.ap.fill(B::Scalar::ZERO);
        self.z.fill(B::Scalar::ZERO);
    }
}

/// BiCGStab 求解器工作区
#[derive(Debug, Clone)]
pub struct BiCgStabWorkspace<B: Backend> {
    /// 残差向量
    pub r: B::Buffer<B::Scalar>,
    /// 影子残差，必须保持不变
    pub r0: B::Buffer<B::Scalar>,
    /// 搜索方向
    pub p: B::Buffer<B::Scalar>,
    /// A*p_hat
    pub v: B::Buffer<B::Scalar>,
    /// 中间残差
    pub s: B::Buffer<B::Scalar>,
    /// A*s_hat
    pub t: B::Buffer<B::Scalar>,
    /// 预条件后的向量
    pub p_hat: B::Buffer<B::Scalar>,
    /// 预条件后的向量
    pub s_hat: B::Buffer<B::Scalar>,
}

impl<B: Backend> BiCgStabWorkspace<B> {
    /// 创建新的工作区
    pub fn new(backend: &B, n: usize) -> Self {
        let mut r = backend.alloc(n);
        let mut r0 = backend.alloc(n);
        let mut p = backend.alloc(n);
        let mut v = backend.alloc(n);
        let mut s = backend.alloc(n);
        let mut t = backend.alloc(n);
        let mut p_hat = backend.alloc(n);
        let mut s_hat = backend.alloc(n);
        r.fill(B::Scalar::ZERO);
        r0.fill(B::Scalar::ZERO);
        p.fill(B::Scalar::ZERO);
        v.fill(B::Scalar::ZERO);
        s.fill(B::Scalar::ZERO);
        t.fill(B::Scalar::ZERO);
        p_hat.fill(B::Scalar::ZERO);
        s_hat.fill(B::Scalar::ZERO);
        Self { r, r0, p, v, s, t, p_hat, s_hat }
    }

    /// 调整工作区大小并清零
    ///
    /// 无论大小是否变化，均清零防止历史数据污染。
    pub fn resize(&mut self, backend: &B, n: usize) {
        if self.r.len() != n {
            self.r = backend.alloc(n);
            self.r0 = backend.alloc(n);
            self.p = backend.alloc(n);
            self.v = backend.alloc(n);
            self.s = backend.alloc(n);
            self.t = backend.alloc(n);
            self.p_hat = backend.alloc(n);
            self.s_hat = backend.alloc(n);
        }
        self.clear();
    }

    /// 清零工作区
    pub fn clear(&mut self) {
        self.r.fill(B::Scalar::ZERO);
        self.r0.fill(B::Scalar::ZERO);
        self.p.fill(B::Scalar::ZERO);
        self.v.fill(B::Scalar::ZERO);
        self.s.fill(B::Scalar::ZERO);
        self.t.fill(B::Scalar::ZERO);
        self.p_hat.fill(B::Scalar::ZERO);
        self.s_hat.fill(B::Scalar::ZERO);
    }
}

/// 迭代求解器 trait
pub trait IterativeSolver<B: Backend> {
    /// 求解线性系统 Ax = b
    ///
    /// # 参数
    ///
    /// - `matrix`: 系数矩阵 A
    /// - `b`: 右端项向量
    /// - `x`: 解向量（输入初始猜测，输出解）
    /// - `precond`: 预条件器
    ///
    /// # 返回
    ///
    /// 求解结果
    fn solve<P: Preconditioner<B>>(
        &mut self,
        matrix: &CsrMatrix<B::Scalar>,
        b: &B::Buffer<B::Scalar>,
        x: &mut B::Buffer<B::Scalar>,
        precond: &P,
    ) -> SolverResult<B::Scalar>;

    /// 获取求解器名称
    fn name(&self) -> &'static str;
}

/// 共轭梯度法求解器
///
/// 适用于对称正定矩阵
pub struct ConjugateGradient<B: Backend> {
    config: SolverConfig,
    backend: B,
    // 工作向量
    r: B::Buffer<B::Scalar>,
    p: B::Buffer<B::Scalar>,
    ap: B::Buffer<B::Scalar>,
    observer: Option<Box<dyn IterationObserver<B::Scalar>>>,
}

impl<B: Backend> ConjugateGradient<B> {
    /// 创建共轭梯度求解器
    pub fn new(backend: B, config: SolverConfig) -> Self {
        let r = backend.alloc(0);
        let p = backend.alloc(0);
        let ap = backend.alloc(0);
        Self {
            config,
            backend,
            r,
            p,
            ap,
            observer: None,
        }
    }

    /// 设置迭代观察者
    pub fn set_observer(&mut self, observer: Option<Box<dyn IterationObserver<B::Scalar>>>) {
        self.observer = observer;
    }

    /// 通过 builder 风格设置迭代观察者
    pub fn with_observer(mut self, observer: Box<dyn IterationObserver<B::Scalar>>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// 确保工作向量大小正确
    fn ensure_workspace(&mut self, n: usize) {
        if self.r.len() != n {
            self.r = self.backend.alloc(n);
            self.p = self.backend.alloc(n);
            self.ap = self.backend.alloc(n);
        }
        self.r.fill(B::Scalar::ZERO);
        self.p.fill(B::Scalar::ZERO);
        self.ap.fill(B::Scalar::ZERO);
    }
}

impl<B> IterativeSolver<B> for ConjugateGradient<B>
where
    B: Backend,
{
    fn solve<P: Preconditioner<B>>(
        &mut self,
        matrix: &CsrMatrix<B::Scalar>,
        b: &B::Buffer<B::Scalar>,
        x: &mut B::Buffer<B::Scalar>,
        _precond: &P,
    ) -> SolverResult<B::Scalar> {
        let n = b.len();
        self.ensure_workspace(n);
        let rtol = self.backend.scalar_from_f64(self.config.rtol);
        let atol = self.backend.scalar_from_f64(self.config.atol);
        let breakdown_tol = self.backend.scalar_from_f64(1e-30);
        let stag_tol = self.backend.scalar_from_f64(self.config.stagnation_tol);

        // r = b - A*x
        spmv_kernel(matrix, x.as_slice(), self.r.as_slice_mut());
        for i in 0..n {
            self.r[i] = b[i] - self.r[i];
        }

        let initial_norm = self.backend.norm2(&self.r);
        let b_norm = self.backend.norm2(b);
        let use_absolute = b_norm <= atol;
        let effective_tol = if use_absolute {
            atol
        } else {
            atol.max(rtol * b_norm)
        };

        if initial_norm < effective_tol {
            return SolverResult {
                status: SolverStatus::Converged,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: B::Scalar::ZERO,
            };
        }

        // p = r
        self.backend.copy(&self.r, &mut self.p);

        let mut rr = self.backend.dot(&self.r, &self.r);
        let mut prev_res = initial_norm;

        for iter in 0..self.config.max_iter {
            // ap = A * p
            spmv_kernel(matrix, self.p.as_slice(), self.ap.as_slice_mut());

            // alpha = r'r / p'Ap
            let pap = self.backend.dot(&self.p, &self.ap);
            if pap.abs() < breakdown_tol {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: self.backend.norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: if use_absolute {
                        self.backend.norm2(&self.r)
                    } else {
                        self.backend.norm2(&self.r) / b_norm
                    },
                };
            }

            let alpha = rr / pap;

            // x = x + alpha * p
            self.backend.axpy(alpha, &self.p, x);

            // r = r - alpha * ap
            self.backend.axpy(-alpha, &self.ap, &mut self.r);

            let res_norm = self.backend.norm2(&self.r);
            let rel_res = if use_absolute { res_norm } else { res_norm / b_norm };

            if self.config.verbose {
                log::trace!("CG iter {}: residual = {:?}", iter + 1, res_norm);
            }

            // 检查收敛
            if res_norm < effective_tol || (!use_absolute && rel_res < rtol) {
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            if (prev_res - res_norm).abs() <= stag_tol * prev_res {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            if let Some(observer) = self.observer.as_mut() {
                if observer.should_stop(iter + 1, res_norm, rel_res) {
                    return SolverResult {
                        status: SolverStatus::Stopped,
                        iterations: iter + 1,
                        residual_norm: res_norm,
                        initial_residual_norm: initial_norm,
                        relative_residual: rel_res,
                    };
                }
            }

            prev_res = res_norm;

            // beta = r'r_new / r'r_old
            let rr_new = self.backend.dot(&self.r, &self.r);
            let beta = rr_new / rr;
            rr = rr_new;

            // p = r + beta * p
            for i in 0..n {
                self.p[i] = self.r[i] + beta * self.p[i];
            }
        }

        SolverResult {
            status: SolverStatus::MaxIterationsReached,
            iterations: self.config.max_iter,
            residual_norm: self.backend.norm2(&self.r),
            initial_residual_norm: initial_norm,
            relative_residual: if use_absolute {
                self.backend.norm2(&self.r)
            } else {
                self.backend.norm2(&self.r) / b_norm
            },
        }
    }

    fn name(&self) -> &'static str {
        "CG"
    }
}

/// 预条件共轭梯度法求解器
///
/// 适用于对称正定矩阵，使用预条件器加速收敛
pub struct PcgSolver<B: Backend> {
    config: SolverConfig,
    backend: B,
    // 工作向量
    r: B::Buffer<B::Scalar>,
    z: B::Buffer<B::Scalar>,
    p: B::Buffer<B::Scalar>,
    ap: B::Buffer<B::Scalar>,
    observer: Option<Box<dyn IterationObserver<B::Scalar>>>,
}

impl<B: Backend> PcgSolver<B> {
    /// 创建 PCG 求解器
    pub fn new(backend: B, config: SolverConfig) -> Self {
        let r = backend.alloc(0);
        let z = backend.alloc(0);
        let p = backend.alloc(0);
        let ap = backend.alloc(0);
        Self {
            config,
            backend,
            r,
            z,
            p,
            ap,
            observer: None,
        }
    }

    /// 设置迭代观察者
    pub fn set_observer(&mut self, observer: Option<Box<dyn IterationObserver<B::Scalar>>>) {
        self.observer = observer;
    }

    /// 通过 builder 风格设置迭代观察者
    pub fn with_observer(mut self, observer: Box<dyn IterationObserver<B::Scalar>>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// 确保工作向量大小正确
    fn ensure_workspace(&mut self, n: usize) {
        if self.r.len() != n {
            self.r = self.backend.alloc(n);
            self.z = self.backend.alloc(n);
            self.p = self.backend.alloc(n);
            self.ap = self.backend.alloc(n);
        }
        self.r.fill(B::Scalar::ZERO);
        self.z.fill(B::Scalar::ZERO);
        self.p.fill(B::Scalar::ZERO);
        self.ap.fill(B::Scalar::ZERO);
    }

    /// 使用外部工作区求解（避免内部分配）
    ///
    /// # 参数
    ///
    /// - `matrix`: 系数矩阵
    /// - `b`: 右端项
    /// - `x`: 解向量
    /// - `precond`: 预条件器
    /// - `ws`: 外部工作区
    pub fn solve_with_workspace<P: Preconditioner<B>>(
        &mut self,
        matrix: &CsrMatrix<B::Scalar>,
        b: &B::Buffer<B::Scalar>,
        x: &mut B::Buffer<B::Scalar>,
        precond: &P,
        ws: &mut CgWorkspace<B>,
    ) -> SolverResult<B::Scalar> {
        let n = b.len();
        ws.resize(&self.backend, n);
        let rtol = self.backend.scalar_from_f64(self.config.rtol);
        let atol = self.backend.scalar_from_f64(self.config.atol);
        let breakdown_tol = self.backend.scalar_from_f64(1e-30);
        let stag_tol = self.backend.scalar_from_f64(self.config.stagnation_tol);

        // r = b - A*x
        spmv_kernel(matrix, x.as_slice(), ws.r.as_slice_mut());
        for i in 0..n {
            ws.r[i] = b[i] - ws.r[i];
        }

        let initial_norm = self.backend.norm2(&ws.r);
        let b_norm = self.backend.norm2(b);

        // 鲁棒的收敛判据：处理 b_norm ≈ 0 的情况
        let use_absolute = b_norm <= atol;
        let effective_tol = if use_absolute {
            atol
        } else {
            atol.max(rtol * b_norm)
        };

        if initial_norm < effective_tol {
            return SolverResult {
                status: SolverStatus::Converged,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: B::Scalar::ZERO,
            };
        }

        // z = M^{-1} * r
        if let Err(_err) = precond.apply(&ws.r, &mut ws.z) {
            return SolverResult {
                status: SolverStatus::Stopped,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: if use_absolute { initial_norm } else { initial_norm / b_norm },
            };
        }

        // p = z
        self.backend.copy(&ws.z, &mut ws.p);

        let mut rz = self.backend.dot(&ws.r, &ws.z);
        let mut prev_res = initial_norm;

        for iter in 0..self.config.max_iter {
            // ap = A * p
            spmv_kernel(matrix, ws.p.as_slice(), ws.ap.as_slice_mut());

            // alpha = r'z / p'Ap
            let pap = self.backend.dot(&ws.p, &ws.ap);
            if pap.abs() < breakdown_tol {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: self.backend.norm2(&ws.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: if use_absolute {
                        self.backend.norm2(&ws.r)
                    } else {
                        self.backend.norm2(&ws.r) / b_norm
                    },
                };
            }

            let alpha = rz / pap;

            // x = x + alpha * p
            self.backend.axpy(alpha, &ws.p, x);

            // r = r - alpha * ap
            self.backend.axpy(-alpha, &ws.ap, &mut ws.r);

            let res_norm = self.backend.norm2(&ws.r);
            let rel_res = if use_absolute { res_norm } else { res_norm / b_norm };

            if self.config.verbose {
                log::trace!("PCG iter {}: residual = {:?}", iter + 1, res_norm);
            }

            // 检查收敛
            if res_norm < effective_tol {
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            if (prev_res - res_norm).abs() <= stag_tol * prev_res {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            if let Some(observer) = self.observer.as_mut() {
                if observer.should_stop(iter + 1, res_norm, rel_res) {
                    return SolverResult {
                        status: SolverStatus::Stopped,
                        iterations: iter + 1,
                        residual_norm: res_norm,
                        initial_residual_norm: initial_norm,
                        relative_residual: rel_res,
                    };
                }
            }

            prev_res = res_norm;

            // z = M^{-1} * r
            if let Err(_err) = precond.apply(&ws.r, &mut ws.z) {
                return SolverResult {
                    status: SolverStatus::Stopped,
                    iterations: iter + 1,
                    residual_norm: self.backend.norm2(&ws.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: if use_absolute {
                        self.backend.norm2(&ws.r)
                    } else {
                        self.backend.norm2(&ws.r) / b_norm
                    },
                };
            }

            // beta = r'z_new / r'z_old
            let rz_new = self.backend.dot(&ws.r, &ws.z);
            let beta = rz_new / rz;
            rz = rz_new;

            // p = z + beta * p
            for i in 0..n {
                ws.p[i] = ws.z[i] + beta * ws.p[i];
            }
        }

        SolverResult {
            status: SolverStatus::MaxIterationsReached,
            iterations: self.config.max_iter,
            residual_norm: self.backend.norm2(&ws.r),
            initial_residual_norm: initial_norm,
            relative_residual: if use_absolute {
                self.backend.norm2(&ws.r)
            } else {
                self.backend.norm2(&ws.r) / b_norm
            },
        }
    }
}

impl<B> IterativeSolver<B> for PcgSolver<B>
where
    B: Backend,
{
    fn solve<P: Preconditioner<B>>(
        &mut self,
        matrix: &CsrMatrix<B::Scalar>,
        b: &B::Buffer<B::Scalar>,
        x: &mut B::Buffer<B::Scalar>,
        precond: &P,
    ) -> SolverResult<B::Scalar> {
        let n = b.len();
        self.ensure_workspace(n);
        let rtol = self.backend.scalar_from_f64(self.config.rtol);
        let atol = self.backend.scalar_from_f64(self.config.atol);
        let breakdown_tol = self.backend.scalar_from_f64(1e-30);
        let stag_tol = self.backend.scalar_from_f64(self.config.stagnation_tol);

        // r = b - A*x
        spmv_kernel(matrix, x.as_slice(), self.r.as_slice_mut());
        for i in 0..n {
            self.r[i] = b[i] - self.r[i];
        }

        let initial_norm = self.backend.norm2(&self.r);
        let b_norm = self.backend.norm2(b);
        let use_absolute = b_norm <= atol;
        let effective_tol = if use_absolute {
            atol
        } else {
            atol.max(rtol * b_norm)
        };

        if initial_norm < effective_tol {
            return SolverResult {
                status: SolverStatus::Converged,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: B::Scalar::ZERO,
            };
        }

        // z = M^{-1} * r
        if let Err(_err) = precond.apply(&self.r, &mut self.z) {
            return SolverResult {
                status: SolverStatus::Stopped,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: if use_absolute { initial_norm } else { initial_norm / b_norm },
            };
        }

        // p = z
        self.backend.copy(&self.z, &mut self.p);

        let mut rz = self.backend.dot(&self.r, &self.z);
        let mut prev_res = initial_norm;

        for iter in 0..self.config.max_iter {
            // ap = A * p
            spmv_kernel(matrix, self.p.as_slice(), self.ap.as_slice_mut());

            // alpha = r'z / p'Ap
            let pap = self.backend.dot(&self.p, &self.ap);
            if pap.abs() < breakdown_tol {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: self.backend.norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: if use_absolute {
                        self.backend.norm2(&self.r)
                    } else {
                        self.backend.norm2(&self.r) / b_norm
                    },
                };
            }

            let alpha = rz / pap;

            // x = x + alpha * p
            self.backend.axpy(alpha, &self.p, x);

            // r = r - alpha * ap
            self.backend.axpy(-alpha, &self.ap, &mut self.r);

            let res_norm = self.backend.norm2(&self.r);
            let rel_res = if use_absolute { res_norm } else { res_norm / b_norm };

            if self.config.verbose {
                log::trace!("PCG iter {}: residual = {:?}", iter + 1, res_norm);
            }

            // 检查收敛
            if res_norm < effective_tol || (!use_absolute && rel_res < rtol) {
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            if (prev_res - res_norm).abs() <= stag_tol * prev_res {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            if let Some(observer) = self.observer.as_mut() {
                if observer.should_stop(iter + 1, res_norm, rel_res) {
                    return SolverResult {
                        status: SolverStatus::Stopped,
                        iterations: iter + 1,
                        residual_norm: res_norm,
                        initial_residual_norm: initial_norm,
                        relative_residual: rel_res,
                    };
                }
            }

            prev_res = res_norm;

            // z = M^{-1} * r
            if let Err(_err) = precond.apply(&self.r, &mut self.z) {
                return SolverResult {
                    status: SolverStatus::Stopped,
                    iterations: iter + 1,
                    residual_norm: self.backend.norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: if use_absolute {
                        self.backend.norm2(&self.r)
                    } else {
                        self.backend.norm2(&self.r) / b_norm
                    },
                };
            }

            // beta = r'z_new / r'z_old
            let rz_new = self.backend.dot(&self.r, &self.z);
            let beta = rz_new / rz;
            rz = rz_new;

            // p = z + beta * p
            for i in 0..n {
                self.p[i] = self.z[i] + beta * self.p[i];
            }
        }

        SolverResult {
            status: SolverStatus::MaxIterationsReached,
            iterations: self.config.max_iter,
            residual_norm: self.backend.norm2(&self.r),
            initial_residual_norm: initial_norm,
            relative_residual: if use_absolute {
                self.backend.norm2(&self.r)
            } else {
                self.backend.norm2(&self.r) / b_norm
            },
        }
    }

    fn name(&self) -> &'static str {
        "PCG"
    }
}

/// 双共轭梯度稳定法求解器
///
/// 适用于非对称矩阵
pub struct BiCgStabSolver<B: Backend> {
    config: SolverConfig,
    backend: B,
    // 工作向量
    r: B::Buffer<B::Scalar>,
    r0: B::Buffer<B::Scalar>,
    p: B::Buffer<B::Scalar>,
    v: B::Buffer<B::Scalar>,
    s: B::Buffer<B::Scalar>,
    t: B::Buffer<B::Scalar>,
    z: B::Buffer<B::Scalar>,
    observer: Option<Box<dyn IterationObserver<B::Scalar>>>,
}

impl<B: Backend> BiCgStabSolver<B> {
    /// 创建 BiCGStab 求解器
    pub fn new(backend: B, config: SolverConfig) -> Self {
        let r = backend.alloc(0);
        let r0 = backend.alloc(0);
        let p = backend.alloc(0);
        let v = backend.alloc(0);
        let s = backend.alloc(0);
        let t = backend.alloc(0);
        let z = backend.alloc(0);
        Self {
            config,
            backend,
            r,
            r0,
            p,
            v,
            s,
            t,
            z,
            observer: None,
        }
    }

    /// 设置迭代观察者
    pub fn set_observer(&mut self, observer: Option<Box<dyn IterationObserver<B::Scalar>>>) {
        self.observer = observer;
    }

    /// 通过 builder 风格设置迭代观察者
    pub fn with_observer(mut self, observer: Box<dyn IterationObserver<B::Scalar>>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// 确保工作向量大小正确
    fn ensure_workspace(&mut self, n: usize) {
        if self.r.len() != n {
            self.r = self.backend.alloc(n);
            self.r0 = self.backend.alloc(n);
            self.p = self.backend.alloc(n);
            self.v = self.backend.alloc(n);
            self.s = self.backend.alloc(n);
            self.t = self.backend.alloc(n);
            self.z = self.backend.alloc(n);
        }
        self.r.fill(B::Scalar::ZERO);
        self.r0.fill(B::Scalar::ZERO);
        self.p.fill(B::Scalar::ZERO);
        self.v.fill(B::Scalar::ZERO);
        self.s.fill(B::Scalar::ZERO);
        self.t.fill(B::Scalar::ZERO);
        self.z.fill(B::Scalar::ZERO);
    }
}

impl<B> IterativeSolver<B> for BiCgStabSolver<B>
where
    B: Backend,
{
    fn solve<P: Preconditioner<B>>(
        &mut self,
        matrix: &CsrMatrix<B::Scalar>,
        b: &B::Buffer<B::Scalar>,
        x: &mut B::Buffer<B::Scalar>,
        precond: &P,
    ) -> SolverResult<B::Scalar> {
        let n = b.len();
        self.ensure_workspace(n);
        let rtol = self.backend.scalar_from_f64(self.config.rtol);
        let atol = self.backend.scalar_from_f64(self.config.atol);
        let breakdown_tol = self.backend.scalar_from_f64(1e-30);
        let stag_tol = self.backend.scalar_from_f64(self.config.stagnation_tol);
        let div_factor = self.backend.scalar_from_f64(1e6);

        // r = b - A*x
        spmv_kernel(matrix, x.as_slice(), self.r.as_slice_mut());
        for i in 0..n {
            self.r[i] = b[i] - self.r[i];
        }

        let initial_norm = self.backend.norm2(&self.r);
        let b_norm = self.backend.norm2(b);
        let use_absolute = b_norm <= atol;
        let effective_tol = if use_absolute {
            atol
        } else {
            atol.max(rtol * b_norm)
        };

        if initial_norm < effective_tol {
            return SolverResult {
                status: SolverStatus::Converged,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: B::Scalar::ZERO,
            };
        }

        // r0 = r (shadow residual) - 固定为初始残差，在迭代中保持不变
        self.backend.copy(&self.r, &mut self.r0);

        // 标准 BiCGStab: rho_old 用于计算 beta
        let mut rho_old = B::Scalar::ONE;
        let mut alpha = B::Scalar::ONE;
        let mut omega = B::Scalar::ONE;

        self.v.fill(B::Scalar::ZERO);
        self.p.fill(B::Scalar::ZERO);
        let mut prev_res = initial_norm;

        for iter in 0..self.config.max_iter {
            // 计算 rho = (r0, r)
            let rho = self.backend.dot(&self.r0, &self.r);

            // 检查 rho breakdown
            if rho.abs() < breakdown_tol {
                if iter == 0 {
                    // 初始残差与影子残差正交，已经收敛
                    return SolverResult {
                        status: SolverStatus::Converged,
                        iterations: 0,
                        residual_norm: initial_norm,
                        initial_residual_norm: initial_norm,
                        relative_residual: B::Scalar::ZERO,
                    };
                }
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: self.backend.norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: if use_absolute {
                        self.backend.norm2(&self.r)
                    } else {
                        self.backend.norm2(&self.r) / b_norm
                    },
                };
            }

            // 计算 beta（第一次迭代时 beta = 0，因为 rho_old = 1, omega = 1）
            let beta = if iter == 0 {
                // 首次迭代: p = r
                B::Scalar::ZERO
            } else {
                // 标准公式: beta = (rho / rho_old) * (alpha / omega)
                (rho / rho_old) * (alpha / omega)
            };

            // 保存 rho 供下次迭代使用
            rho_old = rho;

            // p = r + beta * (p - omega * v)
            for i in 0..n {
                self.p[i] = self.r[i] + beta * (self.p[i] - omega * self.v[i]);
            }

            // z = M^{-1} * p
            if let Err(_err) = precond.apply(&self.p, &mut self.z) {
                return SolverResult {
                    status: SolverStatus::Stopped,
                    iterations: iter,
                    residual_norm: self.backend.norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: if use_absolute {
                        self.backend.norm2(&self.r)
                    } else {
                        self.backend.norm2(&self.r) / b_norm
                    },
                };
            }

            // v = A * z
            spmv_kernel(matrix, self.z.as_slice(), self.v.as_slice_mut());

            // alpha = rho / (r0, v)
            let r0v = self.backend.dot(&self.r0, &self.v);
            if r0v.abs() < breakdown_tol {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: self.backend.norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: if use_absolute {
                        self.backend.norm2(&self.r)
                    } else {
                        self.backend.norm2(&self.r) / b_norm
                    },
                };
            }
            alpha = rho / r0v;

            // s = r - alpha * v
            for i in 0..n {
                self.s[i] = self.r[i] - alpha * self.v[i];
            }

            // 检查 s 的范数
            let s_norm = self.backend.norm2(&self.s);
            if s_norm < atol {
                // x = x + alpha * z
                self.backend.axpy(alpha, &self.z, x);
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: s_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: if use_absolute { s_norm } else { s_norm / b_norm },
                };
            }

            // z = M^{-1} * s
            if let Err(_err) = precond.apply(&self.s, &mut self.z) {
                return SolverResult {
                    status: SolverStatus::Stopped,
                    iterations: iter,
                    residual_norm: self.backend.norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: if use_absolute {
                        self.backend.norm2(&self.r)
                    } else {
                        self.backend.norm2(&self.r) / b_norm
                    },
                };
            }

            // t = A * z
            spmv_kernel(matrix, self.z.as_slice(), self.t.as_slice_mut());

            // omega = (t, s) / (t, t)
            let tt = self.backend.dot(&self.t, &self.t);
            if tt.abs() < breakdown_tol {
                omega = B::Scalar::ONE;
            } else {
                omega = self.backend.dot(&self.t, &self.s) / tt;
            }

            // 检查 omega breakdown（omega 过小会导致算法不稳定）
            if omega.abs() < breakdown_tol {
                // 只更新 x 的 alpha 部分后返回
                if let Err(_err) = precond.apply(&self.p, &mut self.z) {
                    return SolverResult {
                        status: SolverStatus::Stopped,
                        iterations: iter + 1,
                        residual_norm: self.backend.norm2(&self.s),
                        initial_residual_norm: initial_norm,
                        relative_residual: if use_absolute {
                            self.backend.norm2(&self.s)
                        } else {
                            self.backend.norm2(&self.s) / b_norm
                        },
                    };
                }
                self.backend.axpy(alpha, &self.z, x);
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter + 1,
                    residual_norm: self.backend.norm2(&self.s),
                    initial_residual_norm: initial_norm,
                    relative_residual: if use_absolute {
                        self.backend.norm2(&self.s)
                    } else {
                        self.backend.norm2(&self.s) / b_norm
                    },
                };
            }

            // x = x + alpha * (M^{-1} p) + omega * (M^{-1} s)
            if let Err(_err) = precond.apply(&self.p, &mut self.z) {
                return SolverResult {
                    status: SolverStatus::Stopped,
                    iterations: iter + 1,
                    residual_norm: self.backend.norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: if use_absolute {
                        self.backend.norm2(&self.r)
                    } else {
                        self.backend.norm2(&self.r) / b_norm
                    },
                };
            }
            self.backend.axpy(alpha, &self.z, x);
            if let Err(_err) = precond.apply(&self.s, &mut self.z) {
                return SolverResult {
                    status: SolverStatus::Stopped,
                    iterations: iter + 1,
                    residual_norm: self.backend.norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: if use_absolute {
                        self.backend.norm2(&self.r)
                    } else {
                        self.backend.norm2(&self.r) / b_norm
                    },
                };
            }
            self.backend.axpy(omega, &self.z, x);

            // r = s - omega * t
            for i in 0..n {
                self.r[i] = self.s[i] - omega * self.t[i];
            }

            let res_norm = self.backend.norm2(&self.r);
            let rel_res = if use_absolute { res_norm } else { res_norm / b_norm };

            if self.config.verbose {
                log::trace!("BiCGStab iter {}: residual = {:?}", iter + 1, res_norm);
            }

            // 检查收敛
            if res_norm < effective_tol || (!use_absolute && rel_res < rtol) {
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            if (prev_res - res_norm).abs() <= stag_tol * prev_res {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            if let Some(observer) = self.observer.as_mut() {
                if observer.should_stop(iter + 1, res_norm, rel_res) {
                    return SolverResult {
                        status: SolverStatus::Stopped,
                        iterations: iter + 1,
                        residual_norm: res_norm,
                        initial_residual_norm: initial_norm,
                        relative_residual: rel_res,
                    };
                }
            }

            prev_res = res_norm;

            // 检查发散
            if res_norm > initial_norm * div_factor {
                return SolverResult {
                    status: SolverStatus::Diverged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }
        }

        SolverResult {
            status: SolverStatus::MaxIterationsReached,
            iterations: self.config.max_iter,
            residual_norm: self.backend.norm2(&self.r),
            initial_residual_norm: initial_norm,
            relative_residual: if use_absolute {
                self.backend.norm2(&self.r)
            } else {
                self.backend.norm2(&self.r) / b_norm
            },
        }
    }

    fn name(&self) -> &'static str {
        "BiCGStab"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::linear_algebra::csr::CsrBuilder;
    use crate::numerics::linear_algebra::preconditioner::{
        IdentityPreconditioner, JacobiPreconditioner,
    };
    use mh_runtime::CpuBackend;

    fn create_spd_matrix(n: usize) -> CsrMatrix<f64> {
        // 创建三对角对称正定矩阵
        let mut builder = CsrBuilder::<f64>::new_square(n);
        for i in 0..n {
            builder.set(i, i, 4.0);
            if i > 0 {
                builder.set(i, i - 1, -1.0);
            }
            if i < n - 1 {
                builder.set(i, i + 1, -1.0);
            }
        }
        builder.build()
    }

    #[test]
    fn test_cg_simple() {
        let backend = CpuBackend::<f64>::new();
        let matrix = create_spd_matrix(10);
        let mut b = backend.alloc(10);
        b.fill(1.0);
        let mut x = backend.alloc_init(10, 0.0);

        let config = SolverConfig::new(1e-10, 100);
        let mut solver = ConjugateGradient::new(backend.clone(), config);
        let precond: IdentityPreconditioner<CpuBackend<f64>> = IdentityPreconditioner::new(backend.clone());

        let result = solver.solve(&matrix, &b, &mut x, &precond);

        assert!(result.is_converged());
        assert!(result.relative_residual < backend.scalar_from_f64(1e-8));
    }

    #[test]
    fn test_pcg_simple() {
        let backend = CpuBackend::<f64>::new();
        let matrix = create_spd_matrix(10);
        let mut b = backend.alloc(10);
        b.fill(1.0);
        let mut x = backend.alloc_init(10, 0.0);

        let config = SolverConfig::new(1e-10, 100);
        let mut solver = PcgSolver::new(backend.clone(), config);
        let precond = JacobiPreconditioner::<CpuBackend<f64>>::from_matrix(&backend, &matrix).unwrap();

        let result = solver.solve(&matrix, &b, &mut x, &precond);

        assert!(result.is_converged());
        assert!(result.relative_residual < backend.scalar_from_f64(1e-8));
    }

    #[test]
    fn test_pcg_faster_than_cg() {
        let backend = CpuBackend::<f64>::new();
        let matrix = create_spd_matrix(50);
        let mut b = backend.alloc(50);
        b.fill(1.0);

        // CG
        let mut x_cg = backend.alloc_init(50, 0.0);
        let config = SolverConfig::new(1e-10, 200);
        let mut cg_solver = ConjugateGradient::new(backend.clone(), config.clone());
        let ident: IdentityPreconditioner<CpuBackend<f64>> = IdentityPreconditioner::new(backend.clone());
        let cg_result = cg_solver.solve(&matrix, &b, &mut x_cg, &ident);

        // PCG
        let mut x_pcg = backend.alloc_init(50, 0.0);
        let mut pcg_solver = PcgSolver::new(backend.clone(), config);
        let precond = JacobiPreconditioner::<CpuBackend<f64>>::from_matrix(&backend, &matrix).unwrap();
        let pcg_result = pcg_solver.solve(&matrix, &b, &mut x_pcg, &precond);

        // PCG 应该更快收敛
        assert!(pcg_result.is_converged());
        assert!(cg_result.is_converged());
        assert!(pcg_result.iterations <= cg_result.iterations);
    }

    #[test]
    fn test_bicgstab_simple() {
        let backend = CpuBackend::<f64>::new();
        let matrix = create_spd_matrix(10);
        let mut b = backend.alloc(10);
        b.fill(1.0);
        let mut x = backend.alloc_init(10, 0.0);

        let config = SolverConfig::new(1e-10, 100);
        let mut solver = BiCgStabSolver::new(backend.clone(), config);
        let precond = JacobiPreconditioner::<CpuBackend<f64>>::from_matrix(&backend, &matrix).unwrap();

        let result = solver.solve(&matrix, &b, &mut x, &precond);

        assert!(result.is_converged());
        assert!(result.relative_residual < backend.scalar_from_f64(1e-8));
    }

    #[test]
    fn test_already_converged() {
        let backend = CpuBackend::<f64>::new();
        let matrix = create_spd_matrix(3);
        // b = A * x_exact
        let x_exact = vec![0.25, 0.25, 0.25];
        let mut x_exact_buf = backend.alloc(3);
        x_exact_buf.copy_from_slice(&x_exact);
        let mut b = backend.alloc(3);
        spmv_kernel(&matrix, x_exact_buf.as_slice(), b.as_slice_mut());

        let mut x = x_exact_buf.clone();

        let config = SolverConfig::new(1e-10, 100);
        let mut solver = PcgSolver::new(backend.clone(), config);
        let precond: IdentityPreconditioner<CpuBackend<f64>> = IdentityPreconditioner::new(backend.clone());

        let result = solver.solve(&matrix, &b, &mut x, &precond);

        assert!(result.is_converged());
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_solver_result() {
        let result = SolverResult::<f64> {
            status: SolverStatus::Converged,
            iterations: 10,
            residual_norm: 1e-12,
            initial_residual_norm: 1.0,
            relative_residual: 1e-12,
        };

        assert!(result.is_converged());
    }
}