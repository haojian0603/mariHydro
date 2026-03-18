// crates/mh_physics/src/assimilation/conservation.rs

use super::PhysicsAssimilable;
use crate::tracer::TracerType;
use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar};
use bytemuck::Pod;
use num_traits::Float;

/// 守恒量快照（Backend 泛型）
#[derive(Debug, Clone)]
pub struct ConservedQuantities<B: Backend> {
    /// 总水体质量 [kg]
    pub total_mass: B::Scalar,
    /// 总 x 方向动量 [kg·m/s]
    pub total_momentum_x: B::Scalar,
    /// 总 y 方向动量 [kg·m/s]
    pub total_momentum_y: B::Scalar,
    /// 总泥沙质量 [kg]（如果有）
    pub total_sediment: Option<B::Scalar>,
    /// 总能量 [J]
    pub total_energy: B::Scalar,
}

impl<B: Backend> ConservedQuantities<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: Pod,
{
    /// 从可同化状态计算守恒量
    pub fn compute(state: &mut dyn PhysicsAssimilable<B>) -> Self {
        let n = state.n_cells();

        // 复制数据以避免借用冲突
        let areas: Vec<B::Scalar> = state.cell_areas().copy_to_vec();
        let h_vec: Vec<B::Scalar> = {
            let depth = state.get_depth_mut();
            depth.copy_to_vec()
        };
        let (hu_vec, hv_vec): (Vec<B::Scalar>, Vec<B::Scalar>) = {
            let (hu_buf, hv_buf) = state.get_momentum_mut();
            (hu_buf.copy_to_vec(), hv_buf.copy_to_vec())
        };

        let backend = state.backend();

        let mut total_mass = B::Scalar::ZERO;
        let mut total_energy = B::Scalar::ZERO;
        let mut momentum_x = B::Scalar::ZERO;
        let mut momentum_y = B::Scalar::ZERO;

        let rho = backend.scalar_from_f64(1000.0);
        let g = backend.scalar_from_f64(9.81);
        let half = backend.scalar_from_f64(0.5);

        for i in 0..n {
            let area = areas.get(i).copied().unwrap_or(B::Scalar::ONE);
            let depth = h_vec.get(i).copied().unwrap_or(B::Scalar::ZERO).max(B::Scalar::ZERO);
            let hu_val = hu_vec.get(i).copied().unwrap_or(B::Scalar::ZERO);
            let hv_val = hv_vec.get(i).copied().unwrap_or(B::Scalar::ZERO);

            let volume = depth * area;
            let mass = rho * volume;
            total_mass = total_mass + mass;

            // 动量 = 质量 * 速度 = rho * h * area * (hu/h)
            let u = if depth > B::Scalar::ZERO { hu_val / depth } else { B::Scalar::ZERO };
            let v = if depth > B::Scalar::ZERO { hv_val / depth } else { B::Scalar::ZERO };
            momentum_x = momentum_x + mass * u;
            momentum_y = momentum_y + mass * v;

            if depth > B::Scalar::ZERO {
                let vel_sq = u * u + v * v;
                let kinetic = half * rho * volume * vel_sq;
                let potential = half * rho * g * depth * depth * area;
                total_energy = total_energy + kinetic + potential;
            }
        }

        let total_sediment = state
            .get_tracer_mut(TracerType::Sediment)
            .map(|c| {
                let mut sum = B::Scalar::ZERO;
                let c_vec = c.copy_to_vec();
                for (i, &conc) in c_vec.iter().enumerate() {
                    let area = areas.get(i).copied().unwrap_or(B::Scalar::ONE);
                    let depth = h_vec.get(i).copied().unwrap_or(B::Scalar::ZERO);
                    sum = sum + conc * depth * area;
                }
                sum
            });

        Self {
            total_mass,
            total_momentum_x: momentum_x,
            total_momentum_y: momentum_y,
            total_sediment,
            total_energy,
        }
    }

    /// 计算与参考值的相对误差
    pub fn relative_error(&self, backend: &B, reference: &Self) -> ConservationError<B> {
        let eps = backend.scalar_from_f64(1e-10);
        ConservationError {
            mass_error: (self.total_mass - reference.total_mass) / reference.total_mass.max(eps),
            momentum_x_error: (self.total_momentum_x - reference.total_momentum_x)
                / reference.total_momentum_x.max(eps),
            momentum_y_error: (self.total_momentum_y - reference.total_momentum_y)
                / reference.total_momentum_y.max(eps),
            sediment_error: match (&self.total_sediment, &reference.total_sediment) {
                (Some(s1), Some(s2)) => Some((*s1 - *s2) / (*s2).max(eps)),
                _ => None,
            },
            energy_error: (self.total_energy - reference.total_energy) / reference.total_energy.max(eps),
        }
    }
}

/// 守恒误差（Backend 泛型）
#[derive(Debug, Clone)]
pub struct ConservationError<B: Backend> {
    pub mass_error: B::Scalar,
    pub momentum_x_error: B::Scalar,
    pub momentum_y_error: B::Scalar,
    pub sediment_error: Option<B::Scalar>,
    pub energy_error: B::Scalar,
}

impl<B: Backend> ConservationError<B>
where
    B::Scalar: RuntimeScalar,
{
    /// 检查是否在容差范围内
    pub fn within_tolerance(&self, tol: B::Scalar) -> bool {
        self.mass_error.abs() < tol
            && self.momentum_x_error.abs() < tol
            && self.momentum_y_error.abs() < tol
            && self.sediment_error.map(|e| e.abs() < tol).unwrap_or(true)
    }
}

/// 守恒校验器
pub struct ConservationChecker<B: Backend> {
    /// 初始守恒量
    initial: ConservedQuantities<B>,
    /// 容差
    #[allow(dead_code)]
    tolerance: B::Scalar,
    /// 历史记录 (time, error)
    history: Vec<(B::Scalar, ConservationError<B>)>,
}

impl<B: Backend> ConservationChecker<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: Pod,
{
    pub fn new(initial: ConservedQuantities<B>, tolerance: B::Scalar) -> Self {
        Self {
            initial,
            tolerance,
            history: Vec::new(),
        }
    }
    
    /// 检查当前状态的守恒性
    pub fn check(&mut self, state: &mut dyn PhysicsAssimilable<B>, time: B::Scalar) -> ConservationError<B> {
        let current = ConservedQuantities::compute(state);
        let backend = state.backend();
        let error = current.relative_error(backend, &self.initial);
        self.history.push((time, error.clone()));
        error
    }
    
    /// 获取最大历史质量误差
    pub fn max_error(&self) -> Option<&ConservationError<B>> {
        self.history
            .iter()
            .max_by(|a, b| a.1.mass_error.abs().partial_cmp(&b.1.mass_error.abs()).unwrap())
            .map(|(_, e)| e)
    }
}

// ============================================================================
// 能量守恒检查
// ============================================================================

/// 能量守恒检查结果
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum EnergyCheckResult<S> {
    /// 能量守恒在容差范围内
    Conserved {
        /// 能量变化量
        change: S,
        /// 相对变化
        relative_change: S,
    },
    /// 能量耗散（物理上允许）
    Dissipated {
        /// 耗散量
        dissipation: S,
        /// 相对耗散率
        relative_rate: S,
    },
    /// 能量增加（非物理，错误）
    Increased {
        /// 增加量
        increase: S,
        /// 相对增加率
        relative_rate: S,
    },
}

impl<S: RuntimeScalar> EnergyCheckResult<S> {
    /// 检查是否为物理合理状态
    #[allow(dead_code)]
    pub fn is_physical(&self) -> bool {
        matches!(self, Self::Conserved { .. } | Self::Dissipated { .. })
    }
    
    /// 检查是否违反能量守恒
    #[allow(dead_code)]
    pub fn is_violated(&self) -> bool {
        matches!(self, Self::Increased { .. })
    }
}

/// 检查能量守恒
///
/// # 参数
///
/// - `before`: 变化前的守恒量
/// - `after`: 变化后的守恒量
/// - `tolerance`: 相对容差（用于判断是否守恒）
///
/// # 返回
///
/// 能量检查结果
///
/// # 说明
///
/// - 能量守恒：变化在容差范围内
/// - 能量耗散：能量减少（物理上允许，如摩擦耗散）
/// - 能量增加：能量增加（非物理，表明数值方法有问题）
#[allow(dead_code)]
pub fn check_energy_conservation<B: Backend>(
    before: &ConservedQuantities<B>,
    after: &ConservedQuantities<B>,
    tolerance: B::Scalar,
) -> EnergyCheckResult<B::Scalar>
{
    let energy_before = before.total_energy;
    let energy_after = after.total_energy;
    let change = energy_after - energy_before;
    
    // 避免除零
    let reference_energy = energy_before
        .abs()
        .max(B::Scalar::from_config(1e-10).unwrap_or(B::Scalar::MIN_POSITIVE));
    let relative_change = change / reference_energy;
    
    if relative_change.abs() < tolerance {
        EnergyCheckResult::Conserved {
            change,
            relative_change,
        }
    } else if relative_change < B::Scalar::ZERO {
        // 能量减少 = 耗散
        EnergyCheckResult::Dissipated {
            dissipation: -change,
            relative_rate: -relative_change,
        }
    } else {
        // 能量增加 = 非物理
        EnergyCheckResult::Increased {
            increase: change,
            relative_rate: relative_change,
        }
    }
}

/// 验证能量守恒，违反时返回错误
///
/// # 参数
///
/// - `before`: 变化前的守恒量
/// - `after`: 变化后的守恒量
/// - `tolerance`: 相对容差
///
/// # 返回
///
/// - `Ok(())`: 能量守恒或耗散
/// - `Err(PhysicsError)`: 能量非物理增加
#[allow(dead_code)]
pub fn verify_energy_conservation<B: Backend>(
    before: &ConservedQuantities<B>,
    after: &ConservedQuantities<B>,
    tolerance: B::Scalar,
) -> crate::error::PhysicsResult<()>
where
    B::Scalar: RuntimeScalar,
{
    match check_energy_conservation(before, after, tolerance) {
        EnergyCheckResult::Conserved { .. } | EnergyCheckResult::Dissipated { .. } => Ok(()),
        EnergyCheckResult::Increased { increase: _, relative_rate } => {
            Err(crate::error::PhysicsError::EnergyIncreased {
                before: before.total_energy.to_f64_lossy(),
                after: after.total_energy.to_f64_lossy(),
                relative_increase: relative_rate.to_f64_lossy(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    type CQ = ConservedQuantities<CpuBackend<f64>>;

    fn make_quantities(energy: f64) -> CQ {
        CQ {
            total_mass: 1000.0,
            total_momentum_x: 0.0,
            total_momentum_y: 0.0,
            total_sediment: None,
            total_energy: energy,
        }
    }

    #[test]
    fn test_energy_conserved() {
        let before = make_quantities(1000.0);
        let after = make_quantities(1000.001);
        
        let result = check_energy_conservation::<CpuBackend<f64>>(&before, &after, 0.01);
        assert!(matches!(result, EnergyCheckResult::Conserved { .. }));
        assert!(result.is_physical());
    }

    #[test]
    fn test_energy_dissipated() {
        let before = make_quantities(1000.0);
        let after = make_quantities(900.0);
        
        let result = check_energy_conservation::<CpuBackend<f64>>(&before, &after, 0.01);
        assert!(matches!(result, EnergyCheckResult::Dissipated { .. }));
        assert!(result.is_physical());
    }

    #[test]
    fn test_energy_increased() {
        let before = make_quantities(1000.0);
        let after = make_quantities(1100.0);
        
        let result = check_energy_conservation::<CpuBackend<f64>>(&before, &after, 0.01);
        assert!(matches!(result, EnergyCheckResult::Increased { .. }));
        assert!(!result.is_physical());
        assert!(result.is_violated());
    }

    #[test]
    fn test_verify_energy_conservation() {
        let before = make_quantities(1000.0);
        let after_ok = make_quantities(950.0);
        let after_bad = make_quantities(1100.0);
        
        assert!(verify_energy_conservation::<CpuBackend<f64>>(&before, &after_ok, 0.01).is_ok());
        assert!(verify_energy_conservation::<CpuBackend<f64>>(&before, &after_bad, 0.01).is_err());
    }
}
