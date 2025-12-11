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
