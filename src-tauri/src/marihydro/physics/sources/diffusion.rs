// src-tauri/src/marihydro/physics/sources/diffusion.rs

use rayon::prelude::*;

use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;
use crate::marihydro::infra::error::{MhError, MhResult};

/// 显式扩散求解器（非结构化网格 Face-based 方法）
pub fn apply_diffusion_explicit(
    field: &[f64],
    field_out: &mut [f64],
    mesh: &UnstructuredMesh,
    nu: f64,
    dt: f64,
) -> MhResult<()> {
    validate_diffusion_params(nu, dt)?;

    if field.len() != mesh.n_cells || field_out.len() != mesh.n_cells {
        return Err(MhError::InvalidInput(format!(
            "场数组尺寸不匹配: 期望 {}, 实际 in={}, out={}",
            mesh.n_cells,
            field.len(),
            field_out.len()
        )));
    }

    let mut flux_sum: Vec<f64> = vec![0.0; mesh.n_cells];

    for face_idx in mesh.interior_faces() {
        let owner = mesh.face_owner[face_idx];
        let neighbor = mesh.face_neighbor[face_idx];

        let d = mesh.face_dist_o2n[face_idx];
        if d < 1e-12 {
            continue;
        }

        let length = mesh.face_length[face_idx];
        let phi_o = field[owner];
        let phi_n = field[neighbor];

        let flux = -nu * (phi_n - phi_o) / d * length;

        flux_sum[owner] += flux;
        flux_sum[neighbor] -= flux;
    }

    field_out
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, phi_out)| {
            let area = mesh.cell_area[i];
            if area > 1e-12 {
                *phi_out = field[i] + dt * flux_sum[i] / area;
            } else {
                *phi_out = field[i];
            }
        });

    Ok(())
}

pub fn apply_diffusion_inplace(
    field: &mut [f64],
    mesh: &UnstructuredMesh,
    nu: f64,
    dt: f64,
) -> MhResult<()> {
    let mut temp = vec![0.0; field.len()];
    apply_diffusion_explicit(field, &mut temp, mesh, nu, dt)?;
    field.copy_from_slice(&temp);
    Ok(())
}

pub fn apply_diffusion_substeps(
    field: &mut [f64],
    mesh: &UnstructuredMesh,
    nu: f64,
    dt: f64,
    n_substeps: usize,
) -> MhResult<()> {
    if n_substeps == 0 {
        return Ok(());
    }

    let sub_dt = dt / n_substeps as f64;
    let mut buffer = vec![0.0; field.len()];

    for step in 0..n_substeps {
        if step % 2 == 0 {
            apply_diffusion_explicit(field, &mut buffer, mesh, nu, sub_dt)?;
        } else {
            apply_diffusion_explicit(&buffer, field, mesh, nu, sub_dt)?;
        }
    }

    if n_substeps % 2 == 1 {
        field.copy_from_slice(&buffer);
    }

    Ok(())
}

fn validate_diffusion_params(nu: f64, dt: f64) -> MhResult<()> {
    if nu < 0.0 || nu > 1000.0 {
        return Err(MhError::InvalidInput(format!(
            "扩散系数异常: ν = {:.3} m²/s",
            nu
        )));
    }

    if dt <= 0.0 || dt > 3600.0 {
        return Err(MhError::InvalidInput(format!(
            "时间步长异常: dt = {:.3} s",
            dt
        )));
    }

    Ok(())
}

pub fn estimate_stable_dt(mesh: &UnstructuredMesh, nu: f64) -> f64 {
    if nu < 1e-12 {
        return f64::MAX;
    }

    let min_dist = mesh
        .face_dist_o2n
        .iter()
        .filter(|&&d| d > 1e-12)
        .cloned()
        .fold(f64::MAX, f64::min);

    if min_dist >= f64::MAX {
        return 1.0;
    }

    0.4 * min_dist * min_dist / nu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_params() {
        assert!(validate_diffusion_params(1.0, 1.0).is_ok());
        assert!(validate_diffusion_params(-1.0, 1.0).is_err());
        assert!(validate_diffusion_params(1.0, -1.0).is_err());
        assert!(validate_diffusion_params(1.0, 5000.0).is_err());
    }

    #[test]
    fn test_stable_dt_zero_viscosity() {
        use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;
        let mesh = UnstructuredMesh::new();
        let dt = estimate_stable_dt(&mesh, 0.0);
        assert_eq!(dt, f64::MAX);
    }
}
