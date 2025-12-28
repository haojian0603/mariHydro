// crates/mh_physics/src/assimilation/conservation.rs

use super::PhysicsAssimilable;
use crate::tracer::TracerType;

/// 守恒量快照
#[derive(Debug, Clone)]
pub struct ConservedQuantities {
    /// 总水体质量 [kg]
    pub total_mass: f64,
    /// 总x方向动量 [kg·m/s]
    pub total_momentum_x: f64,
    /// 总y方向动量 [kg·m/s]
    pub total_momentum_y: f64,
    /// 总泥沙质量 [kg]（如果有）
    pub total_sediment: Option<f64>,
    /// 总能量 [J]（势能+动能）
    pub total_energy: f64,
}

impl ConservedQuantities {
    /// 从可同化状态计算守恒量
    pub fn compute(state: &mut dyn PhysicsAssimilable) -> Self {
        let n = state.n_cells();
        
        // 先复制数据以避免借用冲突
        let areas: Vec<f64> = state.cell_areas().to_vec();
        let h_vec: Vec<f64> = state.get_depth_mut().to_vec();
        let (hu_slice, hv_slice) = state.get_velocity_mut();
        let hu_vec: Vec<f64> = hu_slice.to_vec();
        let hv_vec: Vec<f64> = hv_slice.to_vec();

        let mut total_mass = 0.0;
        let mut total_energy = 0.0;
        let mut momentum_x = 0.0;
        let mut momentum_y = 0.0;

        const RHO: f64 = 1000.0;
        const G: f64 = 9.81;

        for i in 0..n {
            let area = areas.get(i).copied().unwrap_or(1.0);
            let depth = h_vec.get(i).copied().unwrap_or(0.0).max(0.0);
            let hu_val = hu_vec.get(i).copied().unwrap_or(0.0);
            let hv_val = hv_vec.get(i).copied().unwrap_or(0.0);

            let volume = depth * area;
            total_mass += RHO * volume;
            momentum_x += hu_val * area;
            momentum_y += hv_val * area;

            if depth > 0.0 {
                let kinetic = 0.5 * RHO * (hu_val * hu_val + hv_val * hv_val) / depth;
                let potential = 0.5 * RHO * G * depth * depth;
                total_energy += (kinetic + potential) * area;
            }
        }

        let total_sediment = state
            .get_tracer_mut(TracerType::Sediment)
            .map(|c| {
                let mut sum = 0.0;
                for (i, &conc) in c.iter().enumerate() {
                    let area = areas.get(i).copied().unwrap_or(1.0);
                    let depth = h_vec.get(i).copied().unwrap_or(0.0);
                    sum += conc * depth * area;
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
    pub fn relative_error(&self, reference: &Self) -> ConservationError {
        ConservationError {
            mass_error: (self.total_mass - reference.total_mass) / reference.total_mass.max(1e-10),
            momentum_x_error: (self.total_momentum_x - reference.total_momentum_x)
                / reference.total_momentum_x.max(1e-10),
            momentum_y_error: (self.total_momentum_y - reference.total_momentum_y)
                / reference.total_momentum_y.max(1e-10),
            sediment_error: match (&self.total_sediment, &reference.total_sediment) {
                (Some(s1), Some(s2)) => Some((s1 - s2) / s2.max(1e-10)),
                _ => None,
            },
            energy_error: (self.total_energy - reference.total_energy) / reference.total_energy.max(1e-10),
        }
    }
}

/// 守恒误差
#[derive(Debug, Clone)]
pub struct ConservationError {
    pub mass_error: f64,
    pub momentum_x_error: f64,
    pub momentum_y_error: f64,
    pub sediment_error: Option<f64>,
    pub energy_error: f64,
}

impl ConservationError {
    /// 检查是否在容差范围内
    pub fn within_tolerance(&self, tol: f64) -> bool {
        self.mass_error.abs() < tol
            && self.momentum_x_error.abs() < tol
            && self.momentum_y_error.abs() < tol
            && self.sediment_error.map(|e| e.abs() < tol).unwrap_or(true)
    }
}

/// 守恒校验器
pub struct ConservationChecker {
    /// 初始守恒量
    initial: ConservedQuantities,
    /// 容差
    #[allow(dead_code)]
    tolerance: f64,
    /// 历史记录
    history: Vec<(f64, ConservationError)>,
}

impl ConservationChecker {
    pub fn new(initial: ConservedQuantities, tolerance: f64) -> Self {
        Self {
            initial,
            tolerance,
            history: Vec::new(),
        }
    }
    
    /// 检查当前状态的守恒性
    pub fn check(&mut self, state: &mut dyn PhysicsAssimilable, time: f64) -> ConservationError {
        let current = ConservedQuantities::compute(state);
        let error = current.relative_error(&self.initial);
        self.history.push((time, error.clone()));
        error
    }
    
    /// 获取最大历史误差
    pub fn max_error(&self) -> Option<&ConservationError> {
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
#[derive(Debug, Clone)]
pub enum EnergyCheckResult {
    /// 能量守恒在容差范围内
    Conserved {
        /// 能量变化量
        change: f64,
        /// 相对变化
        relative_change: f64,
    },
    /// 能量耗散（物理上允许）
    Dissipated {
        /// 耗散量
        dissipation: f64,
        /// 相对耗散率
        relative_rate: f64,
    },
    /// 能量增加（非物理，错误）
    Increased {
        /// 增加量
        increase: f64,
        /// 相对增加率
        relative_rate: f64,
    },
}

impl EnergyCheckResult {
    /// 检查是否为物理合理状态
    pub fn is_physical(&self) -> bool {
        matches!(self, Self::Conserved { .. } | Self::Dissipated { .. })
    }
    
    /// 检查是否违反能量守恒
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
pub fn check_energy_conservation(
    before: &ConservedQuantities,
    after: &ConservedQuantities,
    tolerance: f64,
) -> EnergyCheckResult {
    let energy_before = before.total_energy;
    let energy_after = after.total_energy;
    let change = energy_after - energy_before;
    
    // 避免除零
    let reference_energy = energy_before.abs().max(1e-10);
    let relative_change = change / reference_energy;
    
    if relative_change.abs() < tolerance {
        EnergyCheckResult::Conserved {
            change,
            relative_change,
        }
    } else if relative_change < 0.0 {
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
pub fn verify_energy_conservation(
    before: &ConservedQuantities,
    after: &ConservedQuantities,
    tolerance: f64,
) -> crate::error::PhysicsResult<()> {
    match check_energy_conservation(before, after, tolerance) {
        EnergyCheckResult::Conserved { .. } | EnergyCheckResult::Dissipated { .. } => Ok(()),
        EnergyCheckResult::Increased { increase, relative_rate } => {
            Err(crate::error::PhysicsError::EnergyIncreased {
                before: before.total_energy,
                after: after.total_energy,
                relative_increase: relative_rate,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_quantities(energy: f64) -> ConservedQuantities {
        ConservedQuantities {
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
        
        let result = check_energy_conservation(&before, &after, 0.01);
        assert!(matches!(result, EnergyCheckResult::Conserved { .. }));
        assert!(result.is_physical());
    }

    #[test]
    fn test_energy_dissipated() {
        let before = make_quantities(1000.0);
        let after = make_quantities(900.0);
        
        let result = check_energy_conservation(&before, &after, 0.01);
        assert!(matches!(result, EnergyCheckResult::Dissipated { .. }));
        assert!(result.is_physical());
    }

    #[test]
    fn test_energy_increased() {
        let before = make_quantities(1000.0);
        let after = make_quantities(1100.0);
        
        let result = check_energy_conservation(&before, &after, 0.01);
        assert!(matches!(result, EnergyCheckResult::Increased { .. }));
        assert!(!result.is_physical());
        assert!(result.is_violated());
    }

    #[test]
    fn test_verify_energy_conservation() {
        let before = make_quantities(1000.0);
        let after_ok = make_quantities(950.0);
        let after_bad = make_quantities(1100.0);
        
        assert!(verify_energy_conservation(&before, &after_ok, 0.01).is_ok());
        assert!(verify_energy_conservation(&before, &after_bad, 0.01).is_err());
    }
}
