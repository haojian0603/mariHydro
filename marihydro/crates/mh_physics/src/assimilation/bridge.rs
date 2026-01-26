// crates/mh_physics/src/assimilation/bridge.rs

use super::{ConservationConstraints, PhysicsAssimilable};
use crate::state::ShallowWaterState;
use crate::tracer::TracerType;
use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar};
use num_traits::{Float, FromPrimitive};
use bytemuck::Pod;

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
        cell_areas: &[B::Scalar],
        cell_centers: &[B::Vector2D],
    ) -> Self {
        let mut areas_buf = backend.alloc(cell_areas.len());
        areas_buf.clone_from_slice(cell_areas);
        let mut centers_buf = backend.alloc(cell_centers.len());
        centers_buf.clone_from_slice(cell_centers);

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
    fn get_tracer_mut(&mut self, tracer_type: TracerType) -> Option<&mut [B::Scalar]> {
        let name = tracer_type.name();
        self.state.tracers.get_mut_by_name(name)
    }

    fn get_momentum_mut(&mut self) -> (&mut [B::Scalar], &mut [B::Scalar]) {
        (self.state.hu.as_slice_mut(), self.state.hv.as_slice_mut())
    }

    fn get_depth_mut(&mut self) -> &mut [B::Scalar] {
        self.state.h.as_slice_mut()
    }

    fn get_bed_elevation_mut(&mut self) -> &mut [B::Scalar] {
        self.state.z.as_slice_mut()
    }

    fn n_cells(&self) -> usize {
        self.state.n_cells()
    }

    fn cell_areas(&self) -> &[B::Scalar] {
        self.cell_areas.as_slice()
    }

    fn cell_centers(&self) -> &[B::Vector2D] {
        self.cell_centers.as_slice()
    }

    fn create_snapshot(&self) -> StateSnapshot<B> {
        let n = self.n_cells();
        let mut buffer = self.backend.alloc(n * 4);
        self.backend.copy_interleaved(
            &[
                self.state.h.as_slice(),
                self.state.hu.as_slice(),
                self.state.hv.as_slice(),
                self.state.z.as_slice(),
            ],
            &mut buffer,
        );

        let sediment = self
            .state
            .tracers
            .get_by_name(TracerType::Sediment.name())
            .map(|s| {
                let mut buf = self.backend.alloc(s.len());
                buf.clone_from_slice(s);
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

    fn enforce_conservation(&mut self, reference: &super::ConservedQuantities<B>, tolerance: B::Scalar) {
        let current = self.compute_conserved();

        // 质量修正
        let mass_error = current.total_mass - reference.total_mass;
        if mass_error.abs() > tolerance {
            let correction = reference.total_mass
                / current.total_mass.max(B::Scalar::from_f64(1e-12).unwrap_or(B::Scalar::ZERO));
            for h in self.state.h.as_slice_mut() {
                *h *= correction;
            }
        }

        // 动量修正（按比例缩放）
        let (u_slice, v_slice) = self.get_momentum_mut();
        if current.total_momentum_x.abs() > tolerance {
            let scale = reference.total_momentum_x
                / current.total_momentum_x.max(B::Scalar::from_f64(1e-12).unwrap_or(B::Scalar::ZERO));
            for u in u_slice.iter_mut() {
                *u *= scale;
            }
        }
        if current.total_momentum_y.abs() > tolerance {
            let scale = reference.total_momentum_y
                / current.total_momentum_y.max(B::Scalar::from_f64(1e-12).unwrap_or(B::Scalar::ZERO));
            for v in v_slice.iter_mut() {
                *v *= scale;
            }
        }
    }

    fn enforce_conservation_constrained(
        &mut self,
        reference: &super::ConservedQuantities<B>,
        tolerance: B::Scalar,
        constraints: &ConservationConstraints,
    ) {
        let n = self.n_cells();
        let areas = self.cell_areas().to_vec();

        let mut active_mask = vec![true; n];
        for &idx in &constraints.dry_cells {
            if idx < n {
                active_mask[idx] = false;
            }
        }
        for &idx in &constraints.boundary_cells {
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

        let min_active_area = B::Scalar::from_f64(1e-12).unwrap_or(B::Scalar::ZERO);
        if total_active_area <= min_active_area {
            return;
        }

        let current = self.compute_conserved();

        // 质量修正（分布式）
        let mass_error = current.total_mass - reference.total_mass;
        if mass_error.abs() > tolerance {
            let rho = B::Scalar::from_f64(1000.0).unwrap_or(B::Scalar::ONE);
            let h_correction_per_area = mass_error / (rho * total_active_area);
            let min_h = B::Scalar::from_f64(constraints.min_depth).unwrap_or(B::Scalar::ZERO);

            let h_slice = self.get_depth_mut();
            for i in 0..n {
                if active_mask[i] {
                    let new_h = h_slice[i] - h_correction_per_area;
                    h_slice[i] = new_h.max(min_h);
                }
            }
        }

        // 动量修正（按水深加权分布）
        let h_snapshot: Vec<B::Scalar> = self.get_depth_mut().to_vec();
        let (hu_slice, hv_slice) = self.get_momentum_mut();

        let momentum_x_error = current.total_momentum_x - reference.total_momentum_x;
        let momentum_y_error = current.total_momentum_y - reference.total_momentum_y;

        let mut total_weighted = B::Scalar::ZERO;
        for i in 0..n {
            if active_mask[i] {
                total_weighted = total_weighted + h_snapshot[i] * areas[i];
            }
        }

        let min_weight = B::Scalar::from_f64(1e-12).unwrap_or(B::Scalar::ZERO);
        if total_weighted > min_weight {
            let min_depth = B::Scalar::from_f64(constraints.min_depth).unwrap_or(B::Scalar::ZERO);
            let max_vel = B::Scalar::from_f64(constraints.max_velocity).unwrap_or(B::Scalar::ZERO);

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
