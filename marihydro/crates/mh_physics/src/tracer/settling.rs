// crates/mh_physics/src/tracer/settling.rs

use crate::core::{Backend, DeviceBuffer, Scalar};
use num_traits::Float;

/// 沉降求解器配置
#[derive(Debug, Clone)]
pub struct SettlingConfig<S: Scalar> {
    /// 沉降速度 [m/s]
    pub settling_velocity: S,
    /// 是否使用隐式格式
    pub implicit: bool,
    /// 隐式求解容差（仅隐式模式）
    pub tolerance: S,
    /// 最大迭代次数（仅隐式模式）
    pub max_iterations: usize,
    /// 最小水深阈值
    pub min_depth: S,
}

impl<S: Scalar> Default for SettlingConfig<S> {
    fn default() -> Self {
        Self {
            settling_velocity: <S as Scalar>::from_f64(0.001),
            implicit: true,
            tolerance: <S as Scalar>::from_f64(1e-6),
            max_iterations: 10,
            min_depth: <S as Scalar>::from_f64(0.01),
        }
    }
}

/// 沉降求解结果
#[derive(Debug, Clone)]
pub struct SettlingResult<S: Scalar> {
    /// 实际迭代次数
    pub iterations: usize,
    /// 是否收敛
    pub converged: bool,
    /// 最大相对变化
    pub max_relative_change: S,
    /// 总沉降质量
    pub total_settled_mass: S,
}

/// 隐式沉降求解器
pub struct SettlingSolver<B: Backend> {
    config: SettlingConfig<B::Scalar>,
    /// 工作数组：上一迭代浓度
    c_old: B::Buffer<B::Scalar>,
    /// 工作数组：隐式系数
    coeff: B::Buffer<B::Scalar>,
    #[allow(dead_code)]
    backend: B,
}

impl<B: Backend> SettlingSolver<B> {
    pub fn new(backend: B, n_cells: usize, config: SettlingConfig<B::Scalar>) -> Self {
        let zero = B::Scalar::ZERO;
        Self {
            config,
            c_old: backend.alloc_init(n_cells, zero),
            coeff: backend.alloc_init(n_cells, zero),
            backend,
        }
    }
    
    /// 隐式求解沉降
    /// 
    /// 求解: (1 + dt * ws / h) * C^{n+1} = C^n
    /// 
    /// # 参数
    /// - `concentration`: 浓度场（输入/输出）
    /// - `depth`: 水深场
    /// - `dt`: 时间步长
    pub fn solve(
        &mut self,
        concentration: &mut B::Buffer<B::Scalar>,
        depth: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) -> SettlingResult<B::Scalar> {
        let n = concentration.len().min(depth.len());
        let mut result = SettlingResult {
            iterations: 0,
            converged: false,
            max_relative_change: B::Scalar::ZERO,
            total_settled_mass: B::Scalar::ZERO,
        };

        if dt <= B::Scalar::ZERO {
            return result;
        }

        self.c_old.copy_from_slice(&concentration.copy_to_vec());
        
        // 计算隐式系数
        let n_coeff = depth.len().min(self.coeff.len());
        if let (Some(depth_slice), Some(coeff_slice)) = (
            depth.as_slice(),
            self.coeff.as_slice_mut(),
        ) {
            for i in 0..n_coeff {
                let h = Float::max(depth_slice[i], self.config.min_depth);
                coeff_slice[i] = B::Scalar::ONE / (B::Scalar::ONE + dt * self.config.settling_velocity / h);
            }
        } else {
            let depth_host = depth.copy_to_vec();
            let mut coeff_host = self.coeff.copy_to_vec();
            for i in 0..n_coeff {
                let h = Float::max(depth_host[i], self.config.min_depth);
                coeff_host[i] = B::Scalar::ONE / (B::Scalar::ONE + dt * self.config.settling_velocity / h);
            }
            self.coeff.copy_from_slice(&coeff_host[..n_coeff]);
        }

        for iter in 0..self.config.max_iterations {
            result.iterations = iter + 1;
            let (mut max_rel, mut settled) = (B::Scalar::ZERO, B::Scalar::ZERO);

            if let (Some(c_old), Some(coeff), Some(c_new), Some(h_slice)) = (
                self.c_old.as_slice(),
                self.coeff.as_slice(),
                concentration.as_slice_mut(),
                depth.as_slice(),
            ) {
                for i in 0..n {
                    let h = Float::max(h_slice[i], self.config.min_depth);
                    let updated = Float::max(c_old[i] * coeff[i], B::Scalar::ZERO);
                    let denom = Float::max(Float::abs(c_new[i]), <B::Scalar as Scalar>::from_f64(1e-12));
                    let rel = Float::abs(updated - c_new[i]) / denom;
                    max_rel = Float::max(max_rel, rel);
                    settled = settled + Float::max(c_old[i] - updated, B::Scalar::ZERO) * h;
                    c_new[i] = updated;
                }
            } else {
                let c_old_host = self.c_old.copy_to_vec();
                let coeff_host = self.coeff.copy_to_vec();
                let mut c_new_host = concentration.copy_to_vec();
                let depth_host = depth.copy_to_vec();
                for i in 0..n {
                    let h = Float::max(depth_host[i], self.config.min_depth);
                    let updated = Float::max(c_old_host[i] * coeff_host[i], B::Scalar::ZERO);
                    let denom = Float::max(Float::abs(c_new_host[i]), <B::Scalar as Scalar>::from_f64(1e-12));
                    let rel = Float::abs(updated - c_new_host[i]) / denom;
                    max_rel = Float::max(max_rel, rel);
                    settled = settled + Float::max(c_old_host[i] - updated, B::Scalar::ZERO) * h;
                    c_new_host[i] = updated;
                }
                concentration.copy_from_slice(&c_new_host);
            }

            result.max_relative_change = max_rel;
            result.total_settled_mass = settled;
            if max_rel.to_f64() < self.config.tolerance.to_f64() {
                result.converged = true;
                break;
            }

            self.c_old.copy_from_slice(&concentration.copy_to_vec());
        }

        result
    }
    
    /// 显式沉降（仅用于小时间步）
    /// 
    /// C^{n+1} = C^n - dt * ws * C^n / h
    pub fn apply_explicit(
        &self,
        concentration: &mut B::Buffer<B::Scalar>,
        depth: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) {
        let n = concentration.len().min(depth.len());
        if let (Some(conc), Some(depth_slice)) = (
            concentration.as_slice_mut(),
            depth.as_slice(),
        ) {
            for i in 0..n {
                let h = Float::max(depth_slice[i], self.config.min_depth);
                let c_val = conc[i];
                conc[i] = Float::max(c_val - self.config.settling_velocity * dt * c_val / h, B::Scalar::ZERO);
            }
        } else {
            let mut conc_host = concentration.copy_to_vec();
            let depth_host = depth.copy_to_vec();
            for i in 0..n {
                let h = Float::max(depth_host[i], self.config.min_depth);
                let c_val = conc_host[i];
                conc_host[i] = Float::max(c_val - self.config.settling_velocity * dt * c_val / h, B::Scalar::ZERO);
            }
            concentration.copy_from_slice(&conc_host[..n]);
        }
    }
    
    /// 计算隐式系数 1 / (1 + dt * ws / h)
    #[allow(dead_code)]
    fn compute_implicit_coefficient(
        &self,
        depth: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
        coeff: &mut B::Buffer<B::Scalar>,
    ) {
        let n = depth.len().min(coeff.len());
        if let (Some(depth_slice), Some(coeff_slice)) = (
            depth.as_slice(),
            coeff.as_slice_mut(),
        ) {
            for i in 0..n {
                let h = Float::max(depth_slice[i], self.config.min_depth);
                coeff_slice[i] = B::Scalar::ONE / (B::Scalar::ONE + dt * self.config.settling_velocity / h);
            }
        } else {
            let depth_host = depth.copy_to_vec();
            let mut coeff_host = coeff.copy_to_vec();
            for i in 0..n {
                let h = Float::max(depth_host[i], self.config.min_depth);
                coeff_host[i] = B::Scalar::ONE / (B::Scalar::ONE + dt * self.config.settling_velocity / h);
            }
            coeff.copy_from_slice(&coeff_host[..n]);
        }
    }
    
    /// 检查CFL稳定性条件
    pub fn check_explicit_stability(
        &self,
        depth: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) -> bool {
        if let Some(depth_slice) = depth.as_slice() {
            for &h in depth_slice {
                if h <= self.config.min_depth {
                    continue;
                }
                let cfl = dt * self.config.settling_velocity / h;
                if cfl > B::Scalar::ONE {
                    return false;
                }
            }
            true
        } else {
            let depth_host = depth.copy_to_vec();
            for h in depth_host {
                if h <= self.config.min_depth {
                    continue;
                }
                let cfl = dt * self.config.settling_velocity / h;
                if cfl > B::Scalar::ONE {
                    return false;
                }
            }
            true
        }
    }
    
    /// 更新配置
    pub fn set_config(&mut self, config: SettlingConfig<B::Scalar>) {
        self.config = config;
    }
    
    /// 获取配置
    pub fn config(&self) -> &SettlingConfig<B::Scalar> {
        &self.config
    }
}
