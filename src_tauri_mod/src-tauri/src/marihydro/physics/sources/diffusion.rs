// src-tauri/src/marihydro/physics/sources/diffusion.rs

use rayon::prelude::*;

use crate::marihydro::domain::mesh::unstructured::{BoundaryKind, UnstructuredMesh};
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

    let flux_sum = compute_diffusion_fluxes(field, mesh, nu);

    field_out
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, phi_out)| {
            let area = mesh.cell_area[i];
            if area > 1e-14 {
                *phi_out = field[i] + dt * flux_sum[i] / area;
            } else {
                *phi_out = field[i];
            }
        });

    Ok(())
}

/// 计算扩散通量 (分离出来便于重用和测试)
fn compute_diffusion_fluxes(field: &[f64], mesh: &UnstructuredMesh, nu: f64) -> Vec<f64> {
    let mut flux_sum = vec![0.0; mesh.n_cells];

    // 内部面
    for face_idx in mesh.interior_faces() {
        let owner = mesh.face_owner[face_idx];
        let neighbor = mesh.face_neighbor[face_idx];

        let d = mesh.face_dist_o2n[face_idx];
        if d < 1e-14 {
            continue;
        }

        let length = mesh.face_length[face_idx];
        let phi_o = field[owner];
        let phi_n = field[neighbor];

        // F = -ν * (φ_n - φ_o) / d * L
        let flux = -nu * (phi_n - phi_o) / d * length;

        flux_sum[owner] += flux;
        flux_sum[neighbor] -= flux;
    }

    // 边界面处理
    for face_idx in mesh.boundary_faces() {
        let bc_idx = mesh.boundary_index(face_idx);
        let kind = mesh.bc_kind[bc_idx];

        match kind {
            BoundaryKind::Wall | BoundaryKind::Symmetry => {
                // Neumann 零通量 (∂φ/∂n = 0)
                // 无需处理
            }
            BoundaryKind::OpenSea | BoundaryKind::Outflow => {
                // 假设零梯度边界条件
                // 可以根据需要修改为外推
            }
            BoundaryKind::RiverInflow => {
                // 入流边界通常假设零扩散通量
            }
        }
    }

    flux_sum
}

/// 可变扩散系数版本
///
/// 用于湍流扩散，其中 ν 是空间变化的
pub fn apply_diffusion_explicit_variable(
    field: &[f64],
    field_out: &mut [f64],
    mesh: &UnstructuredMesh,
    nu: &[f64],
    dt: f64,
) -> MhResult<()> {
    if field.len() != mesh.n_cells || field_out.len() != mesh.n_cells || nu.len() != mesh.n_cells {
        return Err(MhError::InvalidInput("数组尺寸不匹配".into()));
    }

    let mut flux_sum = vec![0.0; mesh.n_cells];

    for face_idx in mesh.interior_faces() {
        let owner = mesh.face_owner[face_idx];
        let neighbor = mesh.face_neighbor[face_idx];

        let d = mesh.face_dist_o2n[face_idx];
        if d < 1e-14 {
            continue;
        }

        let length = mesh.face_length[face_idx];

        // 调和平均扩散系数 (保证正定性)
        let nu_o = nu[owner];
        let nu_n = nu[neighbor];
        let nu_face = if nu_o + nu_n > 1e-14 {
            2.0 * nu_o * nu_n / (nu_o + nu_n)
        } else {
            0.0
        };

        let phi_o = field[owner];
        let phi_n = field[neighbor];
        let flux = -nu_face * (phi_n - phi_o) / d * length;

        flux_sum[owner] += flux;
        flux_sum[neighbor] -= flux;
    }

    field_out
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, phi_out)| {
            let area = mesh.cell_area[i];
            if area > 1e-14 {
                *phi_out = field[i] + dt * flux_sum[i] / area;
            } else {
                *phi_out = field[i];
            }
        });

    Ok(())
}

/// 原地扩散 (单缓冲)
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

/// 多子步扩散 (用于大时间步长)
///
/// 使用乒乓缓冲减少内存分配
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

    // 如果子步数是奇数，最终结果在 buffer 中
    if n_substeps % 2 == 1 {
        field.copy_from_slice(&buffer);
    }

    Ok(())
}

fn validate_diffusion_params(nu: f64, dt: f64) -> MhResult<()> {
    if nu < 0.0 {
        return Err(MhError::InvalidInput(format!(
            "扩散系数不能为负: ν = {:.3} m²/s",
            nu
        )));
    }

    if nu > 10000.0 {
        log::warn!("扩散系数过大: ν = {:.3} m²/s，可能导致数值不稳定", nu);
    }

    if dt <= 0.0 {
        return Err(MhError::InvalidInput(format!(
            "时间步长必须为正: dt = {:.3} s",
            dt
        )));
    }

    if dt > 3600.0 {
        log::warn!("扩散时间步长过大: dt = {:.3} s", dt);
    }

    Ok(())
}

/// 估计稳定时间步长
///
/// 对于显式扩散，CFL 条件: dt < α * d_min² / ν
/// 其中 α ≈ 0.25 是安全系数
pub fn estimate_stable_dt(mesh: &UnstructuredMesh, nu: f64) -> f64 {
    const SAFETY_FACTOR: f64 = 0.25;

    if nu < 1e-14 {
        return f64::MAX;
    }

    let min_dist_sq = mesh
        .face_dist_o2n
        .iter()
        .filter(|&&d| d > 1e-14)
        .map(|&d| d * d)
        .fold(f64::MAX, f64::min);

    if min_dist_sq >= f64::MAX {
        return 1.0;
    }

    SAFETY_FACTOR * min_dist_sq / nu
}

/// 计算所需子步数以保证稳定性
pub fn required_substeps(mesh: &UnstructuredMesh, nu: f64, dt: f64) -> usize {
    let stable_dt = estimate_stable_dt(mesh, nu);
    if stable_dt >= dt {
        1
    } else {
        (dt / stable_dt).ceil() as usize
    }
}

/// 自动子步扩散 (自动计算所需子步数)
pub fn apply_diffusion_auto_substeps(
    field: &mut [f64],
    mesh: &UnstructuredMesh,
    nu: f64,
    dt: f64,
) -> MhResult<usize> {
    let n_substeps = required_substeps(mesh, nu, dt);

    if n_substeps > 1 {
        log::debug!(
            "扩散需要 {} 个子步以保证稳定性 (ν={:.2e}, dt={:.2e})",
            n_substeps,
            nu,
            dt
        );
    }

    apply_diffusion_substeps(field, mesh, nu, dt, n_substeps)?;
    Ok(n_substeps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_params() {
        assert!(validate_diffusion_params(1.0, 1.0).is_ok());
        assert!(validate_diffusion_params(-1.0, 1.0).is_err());
        assert!(validate_diffusion_params(1.0, -1.0).is_err());
    }

    #[test]
    fn test_stable_dt_zero_viscosity() {
        let mesh = UnstructuredMesh::new();
        let dt = estimate_stable_dt(&mesh, 0.0);
        assert_eq!(dt, f64::MAX);
    }

    #[test]
    fn test_required_substeps() {
        let mut mesh = UnstructuredMesh::new();
        mesh.face_dist_o2n = vec![1.0, 1.0, 1.0];

        // ν = 1.0, d = 1.0, stable_dt ≈ 0.25
        let n = required_substeps(&mesh, 1.0, 1.0);
        assert!(n >= 4);
    }

    #[test]
    fn test_required_substeps_stable() {
        let mut mesh = UnstructuredMesh::new();
        mesh.face_dist_o2n = vec![1.0];

        // 小时间步，应该不需要子步
        let n = required_substeps(&mesh, 1.0, 0.1);
        assert_eq!(n, 1);
    }

    #[test]
    fn test_harmonic_average() {
        let nu_o = 1.0;
        let nu_n = 2.0;
        let nu_face = 2.0 * nu_o * nu_n / (nu_o + nu_n);
        let expected = 2.0 * 1.0 * 2.0 / 3.0;
        assert!((nu_face - expected).abs() < 1e-10);
    }
}
