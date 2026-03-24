// crates/mh_physics/src/assimilation/bridge.rs

use super::{ConservationConstraints, PhysicsAssimilable};
use crate::state::ShallowWaterState;
use crate::tracer::TracerType;
use bytemuck::Pod;
use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar};
use num_traits::Float;

/// 状态快照（Backend 感知，SoA 打包）
#[derive(Debug, Clone)]
pub struct StateSnapshot<B: Backend>
where
    B::Vector2D: Pod,
{
    /// 连续数据块：h|hu|hv|z
    pub buffer: B::Buffer<B::Scalar>,
    /// 偏移 [h, hu, hv, z]
    pub offsets: [usize; 4],
    /// 并行安全的时间戳
    pub time: B::Scalar,
    /// 单元中心
    pub cell_centers: B::Buffer<B::Vector2D>,
    /// 单元面积
    pub cell_areas: B::Buffer<B::Scalar>,
    /// 可选泥沙场
    pub sediment: Option<B::Buffer<B::Scalar>>,
}

/// 桥接适配器（Backend-first）
pub struct AssimilableBridge<'a, B: Backend>
where
    B::Vector2D: Pod,
{
    state: &'a mut ShallowWaterState<B>,
    backend: &'a B,
    cell_areas: B::Buffer<B::Scalar>,
    cell_centers: B::Buffer<B::Vector2D>,
    time: B::Scalar,
}

impl<'a, B: Backend> AssimilableBridge<'a, B>
where
    B::Vector2D: Pod + Default,
    B::Scalar: RuntimeScalar,
{
    pub fn new(
        backend: &'a B,
        state: &'a mut ShallowWaterState<B>,
        cell_areas: &B::Buffer<B::Scalar>,
        cell_centers: &B::Buffer<B::Vector2D>,
    ) -> Self {
        let areas_buf = backend.clone_buffer(cell_areas);
        let centers_buf = backend.clone_buffer(cell_centers);

        Self {
            state,
            backend,
            cell_areas: areas_buf,
            cell_centers: centers_buf,
            time: B::Scalar::ZERO,
        }
    }

    /// 设置当前时间
    pub fn with_time(mut self, time: B::Scalar) -> Self {
        self.time = time;
        self
    }
}

impl<'a, B: Backend> PhysicsAssimilable<B> for AssimilableBridge<'a, B>
where
    B::Vector2D: Pod + Default,
    B::Scalar: RuntimeScalar,
{
    fn backend(&self) -> &B {
        self.backend
    }

    fn get_tracer_mut(&mut self, tracer_type: TracerType) -> Option<&mut B::Buffer<B::Scalar>> {
        let name = tracer_type.name();
        self.state.tracers.get_buffer_mut_by_name(name)
    }

    fn get_momentum_mut(&mut self) -> (&mut B::Buffer<B::Scalar>, &mut B::Buffer<B::Scalar>) {
        (&mut self.state.hu, &mut self.state.hv)
    }

    fn get_depth_mut(&mut self) -> &mut B::Buffer<B::Scalar> {
        &mut self.state.h
    }

    fn get_bed_elevation_mut(&mut self) -> &mut B::Buffer<B::Scalar> {
        &mut self.state.z
    }

    fn n_cells(&self) -> usize {
        self.state.n_cells()
    }

    fn cell_areas(&self) -> &B::Buffer<B::Scalar> {
        &self.cell_areas
    }

    fn cell_centers(&self) -> &B::Buffer<B::Vector2D> {
        &self.cell_centers
    }

    fn create_snapshot(&self) -> StateSnapshot<B> {
        let n = self.n_cells();
        let mut buffer = self.backend.alloc(n * 4);
        let h_vec = self.state.h.copy_to_vec();
        let hu_vec = self.state.hu.copy_to_vec();
        let hv_vec = self.state.hv.copy_to_vec();
        let z_vec = self.state.z.copy_to_vec();
        self.backend
            .copy_interleaved(&[&h_vec, &hu_vec, &hv_vec, &z_vec], &mut buffer);

        let sediment = self
            .state
            .tracers
            .get_buffer_by_name(TracerType::Sediment.name())
            .map(|s| {
                let mut buf = self.backend.alloc(s.len());
                self.backend.copy(s, &mut buf);
                buf
            });

        StateSnapshot {
            buffer,
            offsets: [0, n, n * 2, n * 3],
            time: self.time,
            cell_centers: self.backend.clone_buffer(&self.cell_centers),
            cell_areas: self.backend.clone_buffer(&self.cell_areas),
            sediment,
        }
    }

    fn compute_conserved(&mut self) -> super::ConservedQuantities<B> {
        super::ConservedQuantities::compute(self)
    }

    fn enforce_conservation(
        &mut self,
        reference: &super::ConservedQuantities<B>,
        tolerance: B::Scalar,
    ) {
        let current = self.compute_conserved();

        // 质量修正
        let mass_error = current.total_mass - reference.total_mass;
        if mass_error.abs() > tolerance {
            let correction = reference.total_mass
                / current
                    .total_mass
                    .max(B::Scalar::from_config(1e-12).unwrap_or(B::Scalar::MIN_POSITIVE));
            let h_slice = self.state.h.try_as_slice_mut().unwrap_or(&mut []);
            for h in h_slice {
                *h = *h * correction;
            }
        }

        // 动量修正（按比例缩放）
        let (u_buf, v_buf) = self.get_momentum_mut();
        if current.total_momentum_x.abs() > tolerance {
            let scale = reference.total_momentum_x
                / current
                    .total_momentum_x
                    .max(B::Scalar::from_config(1e-12).unwrap_or(B::Scalar::MIN_POSITIVE));
            let u_slice = u_buf.try_as_slice_mut().unwrap_or(&mut []);
            for u in u_slice {
                *u = *u * scale;
            }
        }
        if current.total_momentum_y.abs() > tolerance {
            let scale = reference.total_momentum_y
                / current
                    .total_momentum_y
                    .max(B::Scalar::from_config(1e-12).unwrap_or(B::Scalar::MIN_POSITIVE));
            let v_slice = v_buf.try_as_slice_mut().unwrap_or(&mut []);
            for v in v_slice {
                *v = *v * scale;
            }
        }
    }

    fn enforce_conservation_constrained(
        &mut self,
        reference: &super::ConservedQuantities<B>,
        tolerance: B::Scalar,
        constraints: &ConservationConstraints<B>,
    ) {
        let n = self.n_cells();
        let areas = self.cell_areas().copy_to_vec();

        let mut active_mask = vec![true; n];
        for idx in &constraints.dry_cells {
            let idx = idx.get();
            if idx < n {
                active_mask[idx] = false;
            }
        }
        for idx in &constraints.boundary_cells {
            let idx = idx.get();
            if idx < n {
                active_mask[idx] = false;
            }
        }

        let mut total_active_area = B::Scalar::ZERO;
        for (i, &area) in areas.iter().enumerate() {
            if active_mask[i] {
                total_active_area = total_active_area + area;
            }
        }

        let min_active_area = self
            .backend
            .config_scalar(1e-12, "AssimilationBridge.min_active_area");
        if total_active_area <= min_active_area {
            return;
        }

        let current = self.compute_conserved();

        // 质量修正（分布式）
        let mass_error = current.total_mass - reference.total_mass;
        if mass_error.abs() > tolerance {
            let rho = B::Scalar::from_config(1000.0).unwrap_or(B::Scalar::ZERO);
            let h_correction_per_area = mass_error / (rho * total_active_area);
            let min_h = constraints.min_depth;

            let h_buf = self.get_depth_mut();
            let h_slice = h_buf.try_as_slice_mut().unwrap_or(&mut []);
            for i in 0..n {
                if active_mask[i] {
                    let new_h = h_slice[i] - h_correction_per_area;
                    h_slice[i] = new_h.max(min_h);
                }
            }
        }

        // 动量修正（按水深加权分布）
        let h_snapshot: Vec<B::Scalar> = self.get_depth_mut().copy_to_vec();
        let (hu_buf, hv_buf) = self.get_momentum_mut();
        let hu_slice = hu_buf.try_as_slice_mut().unwrap_or(&mut []);
        let hv_slice = hv_buf.try_as_slice_mut().unwrap_or(&mut []);

        let momentum_x_error = current.total_momentum_x - reference.total_momentum_x;
        let momentum_y_error = current.total_momentum_y - reference.total_momentum_y;

        let mut total_weighted = B::Scalar::ZERO;
        for i in 0..n {
            if active_mask[i] {
                total_weighted = total_weighted + h_snapshot[i] * areas[i];
            }
        }

        let min_weight = B::Scalar::from_config(1e-12).unwrap_or(B::Scalar::MIN_POSITIVE);
        let min_depth = constraints.min_depth;
        let max_vel = constraints.max_velocity;
        if total_weighted > min_weight {
            for i in 0..n {
                if active_mask[i] {
                    let weight = h_snapshot[i] * areas[i] / total_weighted;
                    hu_slice[i] = hu_slice[i] - momentum_x_error * weight;
                    hv_slice[i] = hv_slice[i] - momentum_y_error * weight;

                    let h = h_snapshot[i];
                    if h > min_depth {
                        let u = hu_slice[i] / h;
                        let v = hv_slice[i] / h;
                        let speed = (u * u + v * v).sqrt();
                        if speed > max_vel {
                            let scale = max_vel / speed;
                            hu_slice[i] = hu_slice[i] * scale;
                            hv_slice[i] = hv_slice[i] * scale;
                        }
                    }
                }
            }
        }
    }
}
