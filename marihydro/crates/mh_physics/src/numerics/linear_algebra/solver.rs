// crates/mh_physics/src/numerics/linear_algebra/solver.rs

//! 迭代线性求解器
//!
//! 提供用于求解稀疏线性系统 Ax = b 的迭代方法：
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
//!
//! let matrix: CsrMatrix = /* ... */;
//! let b = vec![1.0, 2.0, 3.0];
//! let mut x = vec![0.0; 3];
//!
//! let precond = JacobiPreconditioner::from_matrix(&matrix);
//! let config = SolverConfig::new(1e-8, 100);
//! let mut solver = PcgSolver::new(config);
//!
//! let result = solver.solve(&matrix, &b, &mut x, &precond);
//! println!("Converged in {} iterations", result.iterations);
//! ```

use super::csr::CsrMatrix;
use super::preconditioner::Preconditioner;
use super::vector_ops::{axpy, copy, dot, norm2};
use mh_foundation::Scalar;
use serde::{Deserialize, Serialize};

/// 求解器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// 相对收敛容差
    pub rtol: Scalar,
    /// 绝对收敛容差
    pub atol: Scalar,
    /// 最大迭代次数
    pub max_iter: usize,
    /// 是否打印迭代信息
    pub verbose: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-14,
            max_iter: 1000,
            verbose: false,
        }
    }
}

impl SolverConfig {
    /// 创建求解器配置
    pub fn new(rtol: Scalar, max_iter: usize) -> Self {
        Self {
            rtol,
            max_iter,
            ..Default::default()
        }
    }

    /// 设置绝对容差
    pub fn with_atol(mut self, atol: Scalar) -> Self {
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
}

/// 求解器结果
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// 求解状态
    pub status: SolverStatus,
    /// 迭代次数
    pub iterations: usize,
    /// 最终残差范数
    pub residual_norm: Scalar,
    /// 初始残差范数
    pub initial_residual_norm: Scalar,
    /// 相对残差
    pub relative_residual: Scalar,
}

impl SolverResult {
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
#[derive(Debug, Clone, Default)]
pub struct CgWorkspace {
    /// 残差向量
    pub r: Vec<Scalar>,
    /// 搜索方向
    pub p: Vec<Scalar>,
    /// A*p
    pub ap: Vec<Scalar>,
    /// 预条件后的残差
    pub z: Vec<Scalar>,
}

impl CgWorkspace {
    /// 创建新的工作区
    pub fn new(n: usize) -> Self {
        Self {
            r: vec![0.0; n],
            p: vec![0.0; n],
            ap: vec![0.0; n],
            z: vec![0.0; n],
        }
    }

    /// 调整工作区大小
    pub fn resize(&mut self, n: usize) {
        if self.r.len() != n {
            self.r.resize(n, 0.0);
            self.p.resize(n, 0.0);
            self.ap.resize(n, 0.0);
            self.z.resize(n, 0.0);
        }
    }
}

/// BiCGStab 求解器工作区
#[derive(Debug, Clone, Default)]
pub struct BiCgStabWorkspace {
    /// 残差向量
    pub r: Vec<Scalar>,
    /// 影子残差，必须保持不变
    pub r0: Vec<Scalar>,
    /// 搜索方向
    pub p: Vec<Scalar>,
    /// A*p_hat
    pub v: Vec<Scalar>,
    /// 中间残差
    pub s: Vec<Scalar>,
    /// A*s_hat
    pub t: Vec<Scalar>,
    /// 预条件后的向量
    pub p_hat: Vec<Scalar>,
    /// 预条件后的向量
    pub s_hat: Vec<Scalar>,
}

impl BiCgStabWorkspace {
    /// 创建新的工作区
    pub fn new(n: usize) -> Self {
        Self {
            r: vec![0.0; n],
            r0: vec![0.0; n],
            p: vec![0.0; n],
            v: vec![0.0; n],
            s: vec![0.0; n],
            t: vec![0.0; n],
            p_hat: vec![0.0; n],
            s_hat: vec![0.0; n],
        }
    }

    /// 调整工作区大小
    pub fn resize(&mut self, n: usize) {
        if self.r.len() != n {
            self.r.resize(n, 0.0);
            self.r0.resize(n, 0.0);
            self.p.resize(n, 0.0);
            self.v.resize(n, 0.0);
            self.s.resize(n, 0.0);
            self.t.resize(n, 0.0);
            self.p_hat.resize(n, 0.0);
            self.s_hat.resize(n, 0.0);
        }
    }
}

/// 迭代求解器 trait
pub trait IterativeSolver {
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
    fn solve<P: Preconditioner>(
        &mut self,
        matrix: &CsrMatrix,
        b: &[Scalar],
        x: &mut [Scalar],
        precond: &P,
    ) -> SolverResult;

    /// 获取求解器名称
    fn name(&self) -> &'static str;
}

/// 共轭梯度法求解器
///
/// 适用于对称正定矩阵
pub struct ConjugateGradient {
    config: SolverConfig,
    // 工作向量
    r: Vec<Scalar>,
    p: Vec<Scalar>,
    ap: Vec<Scalar>,
}

impl ConjugateGradient {
    /// 创建共轭梯度求解器
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            r: Vec::new(),
            p: Vec::new(),
            ap: Vec::new(),
        }
    }

    /// 确保工作向量大小正确
    fn ensure_workspace(&mut self, n: usize) {
        if self.r.len() != n {
            self.r = vec![0.0; n];
            self.p = vec![0.0; n];
            self.ap = vec![0.0; n];
        }
    }
}

impl IterativeSolver for ConjugateGradient {
    fn solve<P: Preconditioner>(
        &mut self,
        matrix: &CsrMatrix,
        b: &[Scalar],
        x: &mut [Scalar],
        _precond: &P,
    ) -> SolverResult {
        let n = b.len();
        self.ensure_workspace(n);

        // r = b - A*x
        matrix.mul_vec(x, &mut self.r);
        for i in 0..n {
            self.r[i] = b[i] - self.r[i];
        }

        let initial_norm = norm2(&self.r);
        if initial_norm < self.config.atol {
            return SolverResult {
                status: SolverStatus::Converged,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: 0.0,
            };
        }

        // p = r
        copy(&self.r, &mut self.p);

        let mut rr = dot(&self.r, &self.r);

        for iter in 0..self.config.max_iter {
            // ap = A * p
            matrix.mul_vec(&self.p, &mut self.ap);

            // alpha = r'r / p'Ap
            let pap = dot(&self.p, &self.ap);
            if pap.abs() < 1e-30 {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: norm2(&self.r) / initial_norm,
                };
            }

            let alpha = rr / pap;

            // x = x + alpha * p
            axpy(alpha, &self.p, x);

            // r = r - alpha * ap
            axpy(-alpha, &self.ap, &mut self.r);

            let res_norm = norm2(&self.r);
            let rel_res = res_norm / initial_norm;

            if self.config.verbose {
                log::trace!("CG iter {}: residual = {:.6e}", iter + 1, res_norm);
            }

            // 检查收敛
            if res_norm < self.config.atol || rel_res < self.config.rtol {
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            // beta = r'r_new / r'r_old
            let rr_new = dot(&self.r, &self.r);
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
            residual_norm: norm2(&self.r),
            initial_residual_norm: initial_norm,
            relative_residual: norm2(&self.r) / initial_norm,
        }
    }

    fn name(&self) -> &'static str {
        "CG"
    }
}

/// 预条件共轭梯度法求解器
///
/// 适用于对称正定矩阵，使用预条件器加速收敛
pub struct PcgSolver {
    config: SolverConfig,
    // 工作向量
    r: Vec<Scalar>,
    z: Vec<Scalar>,
    p: Vec<Scalar>,
    ap: Vec<Scalar>,
}

impl PcgSolver {
    /// 创建 PCG 求解器
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            r: Vec::new(),
            z: Vec::new(),
            p: Vec::new(),
            ap: Vec::new(),
        }
    }

    /// 确保工作向量大小正确
    fn ensure_workspace(&mut self, n: usize) {
        if self.r.len() != n {
            self.r = vec![0.0; n];
            self.z = vec![0.0; n];
            self.p = vec![0.0; n];
            self.ap = vec![0.0; n];
        }
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
    pub fn solve_with_workspace<P: Preconditioner>(
        &self,
        matrix: &CsrMatrix,
        b: &[Scalar],
        x: &mut [Scalar],
        precond: &P,
        ws: &mut CgWorkspace,
    ) -> SolverResult {
        let n = b.len();
        ws.resize(n);

        // r = b - A*x
        matrix.mul_vec(x, &mut ws.r);
        for i in 0..n {
            ws.r[i] = b[i] - ws.r[i];
        }

        let initial_norm = norm2(&ws.r);
        let b_norm = norm2(b);

        // 鲁棒的收敛判据：处理 b_norm ≈ 0 的情况
        let effective_tol = if b_norm < f64::MIN_POSITIVE {
            self.config.atol
        } else {
            self.config.atol.max(self.config.rtol * b_norm)
        };

        if initial_norm < effective_tol {
            return SolverResult {
                status: SolverStatus::Converged,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: 0.0,
            };
        }

        // z = M^{-1} * r
        precond.apply(&ws.r, &mut ws.z);

        // p = z
        copy(&ws.z, &mut ws.p);

        let mut rz = dot(&ws.r, &ws.z);

        for iter in 0..self.config.max_iter {
            // ap = A * p
            matrix.mul_vec(&ws.p, &mut ws.ap);

            // alpha = r'z / p'Ap
            let pap = dot(&ws.p, &ws.ap);
            if pap.abs() < 1e-30 {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: norm2(&ws.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: if initial_norm > 0.0 {
                        norm2(&ws.r) / initial_norm
                    } else {
                        0.0
                    },
                };
            }

            let alpha = rz / pap;

            // x = x + alpha * p
            axpy(alpha, &ws.p, x);

            // r = r - alpha * ap
            axpy(-alpha, &ws.ap, &mut ws.r);

            let res_norm = norm2(&ws.r);

            if self.config.verbose {
                log::trace!("PCG iter {}: residual = {:.6e}", iter + 1, res_norm);
            }

            // 检查收敛
            if res_norm < effective_tol {
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: if initial_norm > 0.0 {
                        res_norm / initial_norm
                    } else {
                        0.0
                    },
                };
            }

            // z = M^{-1} * r
            precond.apply(&ws.r, &mut ws.z);

            // beta = r'z_new / r'z_old
            let rz_new = dot(&ws.r, &ws.z);
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
            residual_norm: norm2(&ws.r),
            initial_residual_norm: initial_norm,
            relative_residual: if initial_norm > 0.0 {
                norm2(&ws.r) / initial_norm
            } else {
                0.0
            },
        }
    }
}

impl IterativeSolver for PcgSolver {
    fn solve<P: Preconditioner>(
        &mut self,
        matrix: &CsrMatrix,
        b: &[Scalar],
        x: &mut [Scalar],
        precond: &P,
    ) -> SolverResult {
        let n = b.len();
        self.ensure_workspace(n);

        // r = b - A*x
        matrix.mul_vec(x, &mut self.r);
        for i in 0..n {
            self.r[i] = b[i] - self.r[i];
        }

        let initial_norm = norm2(&self.r);
        if initial_norm < self.config.atol {
            return SolverResult {
                status: SolverStatus::Converged,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: 0.0,
            };
        }

        // z = M^{-1} * r
        precond.apply(&self.r, &mut self.z);

        // p = z
        copy(&self.z, &mut self.p);

        let mut rz = dot(&self.r, &self.z);

        for iter in 0..self.config.max_iter {
            // ap = A * p
            matrix.mul_vec(&self.p, &mut self.ap);

            // alpha = r'z / p'Ap
            let pap = dot(&self.p, &self.ap);
            if pap.abs() < 1e-30 {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: norm2(&self.r) / initial_norm,
                };
            }

            let alpha = rz / pap;

            // x = x + alpha * p
            axpy(alpha, &self.p, x);

            // r = r - alpha * ap
            axpy(-alpha, &self.ap, &mut self.r);

            let res_norm = norm2(&self.r);
            let rel_res = res_norm / initial_norm;

            if self.config.verbose {
                log::trace!("PCG iter {}: residual = {:.6e}", iter + 1, res_norm);
            }

            // 检查收敛
            if res_norm < self.config.atol || rel_res < self.config.rtol {
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            // z = M^{-1} * r
            precond.apply(&self.r, &mut self.z);

            // beta = r'z_new / r'z_old
            let rz_new = dot(&self.r, &self.z);
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
            residual_norm: norm2(&self.r),
            initial_residual_norm: initial_norm,
            relative_residual: norm2(&self.r) / initial_norm,
        }
    }

    fn name(&self) -> &'static str {
        "PCG"
    }
}

/// 双共轭梯度稳定法求解器
///
/// 适用于非对称矩阵
pub struct BiCgStabSolver {
    config: SolverConfig,
    // 工作向量
    r: Vec<Scalar>,
    r0: Vec<Scalar>,
    p: Vec<Scalar>,
    v: Vec<Scalar>,
    s: Vec<Scalar>,
    t: Vec<Scalar>,
    z: Vec<Scalar>,
}

impl BiCgStabSolver {
    /// 创建 BiCGStab 求解器
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            r: Vec::new(),
            r0: Vec::new(),
            p: Vec::new(),
            v: Vec::new(),
            s: Vec::new(),
            t: Vec::new(),
            z: Vec::new(),
        }
    }

    /// 确保工作向量大小正确
    fn ensure_workspace(&mut self, n: usize) {
        if self.r.len() != n {
            self.r = vec![0.0; n];
            self.r0 = vec![0.0; n];
            self.p = vec![0.0; n];
            self.v = vec![0.0; n];
            self.s = vec![0.0; n];
            self.t = vec![0.0; n];
            self.z = vec![0.0; n];
        }
    }
}

impl IterativeSolver for BiCgStabSolver {
    fn solve<P: Preconditioner>(
        &mut self,
        matrix: &CsrMatrix,
        b: &[Scalar],
        x: &mut [Scalar],
        precond: &P,
    ) -> SolverResult {
        let n = b.len();
        self.ensure_workspace(n);

        // r = b - A*x
        matrix.mul_vec(x, &mut self.r);
        for i in 0..n {
            self.r[i] = b[i] - self.r[i];
        }

        let initial_norm = norm2(&self.r);
        if initial_norm < self.config.atol {
            return SolverResult {
                status: SolverStatus::Converged,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: 0.0,
            };
        }

        // r0 = r (shadow residual) - 固定为初始残差，在迭代中保持不变
        copy(&self.r, &mut self.r0);

        // 修复：rho 初始化为 (r0, r) 而非 1.0
        let mut rho = dot(&self.r0, &self.r);
        if rho.abs() < 1e-30 {
            return SolverResult {
                status: SolverStatus::Converged,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: 0.0,
            };
        }
        let mut alpha = 1.0;
        let mut omega = 1.0;

        self.v.fill(0.0);
        self.p.fill(0.0);

        for iter in 0..self.config.max_iter {
            // 计算 beta（第一次迭代时使用特殊处理）
            let beta = if iter == 0 {
                0.0
            } else {
                let rho_new = dot(&self.r0, &self.r);
                if rho_new.abs() < 1e-30 {
                    return SolverResult {
                        status: SolverStatus::Stagnated,
                        iterations: iter,
                        residual_norm: norm2(&self.r),
                        initial_residual_norm: initial_norm,
                        relative_residual: norm2(&self.r) / initial_norm,
                    };
                }
                let b = (rho_new / rho) * (alpha / omega);
                rho = rho_new;
                b
            };

            // p = r + beta * (p - omega * v)
            for i in 0..n {
                self.p[i] = self.r[i] + beta * (self.p[i] - omega * self.v[i]);
            }

            // z = M^{-1} * p
            precond.apply(&self.p, &mut self.z);

            // v = A * z
            matrix.mul_vec(&self.z, &mut self.v);

            // alpha = rho / (r0, v)
            let r0v = dot(&self.r0, &self.v);
            if r0v.abs() < 1e-30 {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: norm2(&self.r) / initial_norm,
                };
            }
            alpha = rho / r0v;

            // s = r - alpha * v
            for i in 0..n {
                self.s[i] = self.r[i] - alpha * self.v[i];
            }

            // 检查 s 的范数
            let s_norm = norm2(&self.s);
            if s_norm < self.config.atol {
                // x = x + alpha * z
                axpy(alpha, &self.z, x);
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: s_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: s_norm / initial_norm,
                };
            }

            // z = M^{-1} * s
            precond.apply(&self.s, &mut self.z);

            // t = A * z
            matrix.mul_vec(&self.z, &mut self.t);

            // omega = (t, s) / (t, t)
            let tt = dot(&self.t, &self.t);
            if tt.abs() < 1e-30 {
                omega = 1.0;
            } else {
                omega = dot(&self.t, &self.s) / tt;
            }

            // x = x + alpha * (M^{-1} p) + omega * (M^{-1} s)
            precond.apply(&self.p, &mut self.z);
            axpy(alpha, &self.z, x);
            precond.apply(&self.s, &mut self.z);
            axpy(omega, &self.z, x);

            // r = s - omega * t
            for i in 0..n {
                self.r[i] = self.s[i] - omega * self.t[i];
            }

            let res_norm = norm2(&self.r);
            let rel_res = res_norm / initial_norm;

            if self.config.verbose {
                log::trace!("BiCGStab iter {}: residual = {:.6e}", iter + 1, res_norm);
            }

            // 检查收敛
            if res_norm < self.config.atol || rel_res < self.config.rtol {
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            // 检查发散
            if res_norm > initial_norm * 1e6 {
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
            residual_norm: norm2(&self.r),
            initial_residual_norm: initial_norm,
            relative_residual: norm2(&self.r) / initial_norm,
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

    fn create_spd_matrix(n: usize) -> CsrMatrix {
        // 创建三对角对称正定矩阵
        let mut builder = CsrBuilder::new_square(n);
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
        let matrix = create_spd_matrix(10);
        let b = vec![1.0; 10];
        let mut x = vec![0.0; 10];

        let config = SolverConfig::new(1e-10, 100);
        let mut solver = ConjugateGradient::new(config);
        let precond = IdentityPreconditioner::new();

        let result = solver.solve(&matrix, &b, &mut x, &precond);

        assert!(result.is_converged());
        assert!(result.relative_residual < 1e-8);
    }

    #[test]
    fn test_pcg_simple() {
        let matrix = create_spd_matrix(10);
        let b = vec![1.0; 10];
        let mut x = vec![0.0; 10];

        let config = SolverConfig::new(1e-10, 100);
        let mut solver = PcgSolver::new(config);
        let precond = JacobiPreconditioner::from_matrix(&matrix);

        let result = solver.solve(&matrix, &b, &mut x, &precond);

        assert!(result.is_converged());
        assert!(result.relative_residual < 1e-8);
    }

    #[test]
    fn test_pcg_faster_than_cg() {
        let matrix = create_spd_matrix(50);
        let b = vec![1.0; 50];

        // CG
        let mut x_cg = vec![0.0; 50];
        let config = SolverConfig::new(1e-10, 200);
        let mut cg_solver = ConjugateGradient::new(config.clone());
        let ident = IdentityPreconditioner::new();
        let cg_result = cg_solver.solve(&matrix, &b, &mut x_cg, &ident);

        // PCG
        let mut x_pcg = vec![0.0; 50];
        let mut pcg_solver = PcgSolver::new(config);
        let precond = JacobiPreconditioner::from_matrix(&matrix);
        let pcg_result = pcg_solver.solve(&matrix, &b, &mut x_pcg, &precond);

        // PCG 应该更快收敛
        assert!(pcg_result.is_converged());
        assert!(cg_result.is_converged());
        assert!(pcg_result.iterations <= cg_result.iterations);
    }

    #[test]
    fn test_bicgstab_simple() {
        let matrix = create_spd_matrix(10);
        let b = vec![1.0; 10];
        let mut x = vec![0.0; 10];

        let config = SolverConfig::new(1e-10, 100);
        let mut solver = BiCgStabSolver::new(config);
        let precond = JacobiPreconditioner::from_matrix(&matrix);

        let result = solver.solve(&matrix, &b, &mut x, &precond);

        assert!(result.is_converged());
        assert!(result.relative_residual < 1e-8);
    }

    #[test]
    fn test_already_converged() {
        let matrix = create_spd_matrix(3);
        // b = A * x_exact
        let x_exact = vec![0.25, 0.25, 0.25];
        let mut b = vec![0.0; 3];
        matrix.mul_vec(&x_exact, &mut b);

        let mut x = x_exact.clone();

        let config = SolverConfig::new(1e-10, 100);
        let mut solver = PcgSolver::new(config);
        let precond = IdentityPreconditioner::new();

        let result = solver.solve(&matrix, &b, &mut x, &precond);

        assert!(result.is_converged());
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_solver_result() {
        let result = SolverResult {
            status: SolverStatus::Converged,
            iterations: 10,
            residual_norm: 1e-12,
            initial_residual_norm: 1.0,
            relative_residual: 1e-12,
        };

        assert!(result.is_converged());
    }
}
