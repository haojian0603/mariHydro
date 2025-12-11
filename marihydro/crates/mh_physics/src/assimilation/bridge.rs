// crates/mh_physics/src/assimilation/bridge.rs

use super::PhysicsAssimilable;
use crate::state::ShallowWaterState;
use crate::tracer::TracerType;

/// 状态快照（与mh_agent::PhysicsSnapshot兼容）
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub h: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub z: Vec<f64>,
    pub sediment: Option<Vec<f64>>,
    pub time: f64,
    pub cell_centers: Vec<[f64; 2]>,
    pub cell_areas: Vec<f64>,
}

/// 桥接适配器
pub struct AssimilableBridge<'a> {
    state: &'a mut ShallowWaterState,
    cell_areas: Vec<f64>,
    cell_centers: Vec<[f64; 2]>,
    time: f64,
}

impl<'a> AssimilableBridge<'a> {
    pub fn new(
        state: &'a mut ShallowWaterState,
        cell_areas: Vec<f64>,
        cell_centers: Vec<[f64; 2]>,
    ) -> Self {
        Self { state, cell_areas, cell_centers, time: 0.0 }
    }

    /// 设置当前时间
    pub fn with_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }
}

impl<'a> PhysicsAssimilable for AssimilableBridge<'a> {
    fn get_tracer_mut(&mut self, tracer_type: TracerType) -> Option<&mut [f64]> {
        let name = tracer_type.name();
        self.state.tracers.get_mut_by_name(name)
    }
    
    fn get_velocity_mut(&mut self) -> (&mut [f64], &mut [f64]) {
        (self.state.hu.as_mut_slice(), self.state.hv.as_mut_slice())
    }
    
    fn get_depth_mut(&mut self) -> &mut [f64] {
        self.state.h.as_mut_slice()
    }
    
    fn get_bed_elevation_mut(&mut self) -> &mut [f64] {
        self.state.z.as_mut_slice()
    }
    
    fn n_cells(&self) -> usize {
        self.state.n_cells()
    }
    
    fn cell_areas(&self) -> &[f64] {
        &self.cell_areas
    }
    
    fn cell_centers(&self) -> &[[f64; 2]] {
        &self.cell_centers
    }
    
    fn create_snapshot(&self) -> StateSnapshot {
        let n = self.n_cells();
        let h = self.state.h.as_slice().to_vec();

        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for i in 0..n {
            let depth = h[i].max(1e-10);
            u[i] = self.state.hu[i] / depth;
            v[i] = self.state.hv[i] / depth;
        }

        let sediment = self
            .state
            .tracers
            .get_by_name(TracerType::Sediment.name())
            .map(|s| s.to_vec());

        StateSnapshot {
            h,
            u,
            v,
            z: self.state.z.as_slice().to_vec(),
            sediment,
            time: self.time,
            cell_centers: self.cell_centers.clone(),
            cell_areas: self.cell_areas.clone(),
        }
    }
    
    fn compute_conserved(&mut self) -> super::ConservedQuantities {
        super::ConservedQuantities::compute(self)
    }
    
    fn enforce_conservation(&mut self, reference: &super::ConservedQuantities, tolerance: f64) {
        let current = self.compute_conserved();

        // 质量修正
        let mass_error = current.total_mass - reference.total_mass;
        if mass_error.abs() > tolerance {
            let correction = reference.total_mass / current.total_mass.max(1e-12);
            for h in self.state.h.as_mut_slice() {
                *h *= correction;
            }
        }

        // 动量修正（按比例缩放）
        let (u_slice, v_slice) = self.get_velocity_mut();
        if current.total_momentum_x.abs() > tolerance {
            let scale = reference.total_momentum_x / current.total_momentum_x.max(1e-12);
            for u in u_slice.iter_mut() {
                *u *= scale;
            }
        }
        if current.total_momentum_y.abs() > tolerance {
            let scale = reference.total_momentum_y / current.total_momentum_y.max(1e-12);
            for v in v_slice.iter_mut() {
                *v *= scale;
            }
        }
    }
}
