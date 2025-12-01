// src-tauri/src/marihydro/physics/sources/turbulence.rs
// 优化版本：使用 Workspace 复用缓冲区，消除临时 Vec 分配
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::source::{SourceContribution, SourceContext, SourceTerm};
use crate::marihydro::core::traits::state::StateAccess;
use crate::marihydro::core::types::{CellIndex, FaceIndex, NumericalParams};
use crate::marihydro::core::Workspace;
use glam::DVec2;

pub struct SmagorinskyTurbulence {
    cs: f64,
    nu_min: f64,
    nu_max: f64,
}

impl SmagorinskyTurbulence {
    pub fn new(cs: f64) -> Self {
        Self { cs, nu_min: 1e-6, nu_max: 1000.0 }
    }

    pub fn with_limits(mut self, nu_min: f64, nu_max: f64) -> Self {
        self.nu_min = nu_min;
        self.nu_max = nu_max;
        self
    }

    /// 计算涡粘系数并写入 workspace.nu_t
    pub fn compute_eddy_viscosity_into<M: MeshAccess, S: StateAccess>(
        &self, mesh: &M, state: &S, params: &NumericalParams, workspace: &mut Workspace,
    ) -> MhResult<()> {
        let n = mesh.n_cells();
        // 计算速度场
        for i in 0..n {
            let h = state.h(CellIndex(i));
            workspace.velocities[i] = if params.is_dry(h) { DVec2::ZERO }
            else { DVec2::new(state.hu(CellIndex(i)) / h, state.hv(CellIndex(i)) / h) };
        }
        // 计算速度梯度
        self.compute_velocity_gradient_into(mesh, workspace);
        // 计算涡粘系数
        for i in 0..n {
            let area = mesh.cell_area(CellIndex(i));
            if area < 1e-14 { workspace.nu_t[i] = self.nu_min; continue; }
            let delta = area.sqrt();
            let s11 = workspace.du_dx[i];
            let s22 = workspace.dv_dy[i];
            let s12 = 0.5 * (workspace.du_dy[i] + workspace.dv_dx[i]);
            let s_mag = (2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12)).sqrt();
            let nu = (self.cs * delta).powi(2) * s_mag;
            workspace.nu_t[i] = nu.clamp(self.nu_min, self.nu_max);
        }
        Ok(())
    }

    /// 使用 Green-Gauss 方法计算速度梯度，写入 workspace
    fn compute_velocity_gradient_into<M: MeshAccess>(&self, mesh: &M, workspace: &mut Workspace) {
        workspace.du_dx.fill(0.0);
        workspace.du_dy.fill(0.0);
        workspace.dv_dx.fill(0.0);
        workspace.dv_dy.fill(0.0);
        for i in 0..mesh.n_cells() {
            let cell = CellIndex(i);
            let area = mesh.cell_area(cell);
            if area < 1e-14 { continue; }
            let mut gu = DVec2::ZERO;
            let mut gv = DVec2::ZERO;
            for &face in mesh.cell_faces(cell) {
                let owner = mesh.face_owner(face);
                let neighbor = mesh.face_neighbor(face);
                let normal = mesh.face_normal(face);
                let length = mesh.face_length(face);
                let sign = if i == owner.0 { 1.0 } else { -1.0 };
                let ds = normal * length * sign;
                let vel_face = if !neighbor.is_valid() { workspace.velocities[i] }
                else {
                    let o = if i == owner.0 { neighbor.0 } else { owner.0 };
                    (workspace.velocities[i] + workspace.velocities[o]) * 0.5
                };
                gu += ds * vel_face.x;
                gv += ds * vel_face.y;
            }
            workspace.du_dx[i] = gu.x / area;
            workspace.du_dy[i] = gu.y / area;
            workspace.dv_dx[i] = gv.x / area;
            workspace.dv_dy[i] = gv.y / area;
        }
    }

    /// 计算扩散通量
    fn compute_diffusion_flux<M: MeshAccess, S: StateAccess>(
        &self, mesh: &M, state: &S, workspace: &Workspace, params: &NumericalParams,
        acc_hu: &mut [f64], acc_hv: &mut [f64],
    ) {
        for face_idx in 0..mesh.n_faces() {
            let face = FaceIndex(face_idx);
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            if !neighbor.is_valid() { continue; }
            let h_o = state.h(owner);
            let h_n = state.h(neighbor);
            if params.is_dry(h_o) && params.is_dry(h_n) { continue; }
            let u_o = if params.is_dry(h_o) { 0.0 } else { state.hu(owner) / h_o };
            let v_o = if params.is_dry(h_o) { 0.0 } else { state.hv(owner) / h_o };
            let u_n = if params.is_dry(h_n) { 0.0 } else { state.hu(neighbor) / h_n };
            let v_n = if params.is_dry(h_n) { 0.0 } else { state.hv(neighbor) / h_n };
            let dist = (mesh.cell_centroid(neighbor) - mesh.cell_centroid(owner)).length();
            if dist < 1e-14 { continue; }
            // 调和平均涡粘系数（保证正定性，物理上更合理）
            let nu_o = workspace.nu_t[owner.0];
            let nu_n = workspace.nu_t[neighbor.0];
            let nu_face = if nu_o + nu_n > 1e-14 {
                2.0 * nu_o * nu_n / (nu_o + nu_n)
            } else {
                0.0
            };
            let h_face = 0.5 * (h_o + h_n);
            let length = mesh.face_length(face);
            let flux_u = nu_face * h_face * (u_n - u_o) / dist * length;
            let flux_v = nu_face * h_face * (v_n - v_o) / dist * length;
            acc_hu[owner.0] += flux_u;
            acc_hv[owner.0] += flux_v;
            acc_hu[neighbor.0] -= flux_u;
            acc_hv[neighbor.0] -= flux_v;
        }
    }
}

impl SourceTerm for SmagorinskyTurbulence {
    fn name(&self) -> &'static str { "SmagorinskyTurbulence" }

    fn compute_cell<M: MeshAccess, S: StateAccess>(
        &self, _cell_idx: usize, _mesh: &M, _state: &S, _ctx: &SourceContext,
    ) -> SourceContribution {
        SourceContribution::ZERO
    }

    fn compute_all<M: MeshAccess, S: StateAccess>(
        &self, mesh: &M, state: &S, ctx: &SourceContext,
        _output_h: &mut [f64], output_hu: &mut [f64], output_hv: &mut [f64],
    ) -> MhResult<()> {
        // 使用 unsafe 从不可变引用获取可变引用（workspace 在 ctx 生命周期内是独占的）
        let workspace = unsafe { &mut *(ctx.workspace as *const Workspace as *mut Workspace) };
        self.compute_eddy_viscosity_into(mesh, state, ctx.params, workspace)?;
        self.compute_diffusion_flux(mesh, state, workspace, ctx.params, output_hu, output_hv);
        Ok(())
    }
}

/// 计算涡度场，写入 output
pub fn compute_vorticity_into<M: MeshAccess, S: StateAccess>(
    mesh: &M, state: &S, params: &NumericalParams, workspace: &mut Workspace, output: &mut [f64],
) {
    let n = mesh.n_cells();
    for i in 0..n {
        let h = state.h(CellIndex(i));
        workspace.velocities[i] = if params.is_dry(h) { DVec2::ZERO }
        else { DVec2::new(state.hu(CellIndex(i)) / h, state.hv(CellIndex(i)) / h) };
    }
    for i in 0..n {
        let cell = CellIndex(i);
        let area = mesh.cell_area(cell);
        if area < 1e-14 { output[i] = 0.0; continue; }
        let mut dv_dx = 0.0;
        let mut du_dy = 0.0;
        for &face in mesh.cell_faces(cell) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            let normal = mesh.face_normal(face);
            let length = mesh.face_length(face);
            let sign = if i == owner.0 { 1.0 } else { -1.0 };
            let vel = if !neighbor.is_valid() { workspace.velocities[i] }
            else {
                let o = if i == owner.0 { neighbor.0 } else { owner.0 };
                (workspace.velocities[i] + workspace.velocities[o]) * 0.5
            };
            dv_dx += sign * vel.y * normal.x * length;
            du_dy += sign * vel.x * normal.y * length;
        }
        output[i] = dv_dx / area - du_dy / area;
    }
}

/// 兼容旧接口：计算涡度（返回 Vec）
pub fn compute_vorticity<M: MeshAccess, S: StateAccess>(mesh: &M, state: &S, params: &NumericalParams) -> Vec<f64> {
    let n = mesh.n_cells();
    let mut workspace = Workspace::new(n, 0);
    let mut output = vec![0.0; n];
    compute_vorticity_into(mesh, state, params, &mut workspace, &mut output);
    output
}
