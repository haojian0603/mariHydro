// src-tauri/src/marihydro/physics/sources/turbulence.rs
//
// 注意: SourceTerm trait 已被枚举化重构，计算逻辑已移至
// core/traits/source.rs 中的 SourceTermKind::Smagorinsky
// 此文件保留辅助函数和公共API

// 从核心模块导入配置类型
pub use crate::marihydro::core::traits::source::SmagorinskyConfig;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::state::StateAccess;
use crate::marihydro::core::types::{CellIndex, NumericalParams};
use crate::marihydro::core::Workspace;
use glam::DVec2;

/// Smagorinsky 湍流源项包装器（便捷构造）
pub struct SmagorinskyTurbulence;

impl SmagorinskyTurbulence {
    /// 创建 Smagorinsky 配置
    pub fn new(cs: f64) -> SmagorinskyConfig {
        SmagorinskyConfig::new(cs)
    }

    /// 创建默认配置 (cs=0.1)
    pub fn default_config() -> SmagorinskyConfig {
        SmagorinskyConfig::new(0.1)
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
