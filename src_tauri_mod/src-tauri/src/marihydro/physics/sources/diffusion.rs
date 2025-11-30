// src-tauri/src/marihydro/physics/sources/diffusion.rs
use rayon::prelude::*;
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::traits::mesh::MeshAccess;

pub fn apply_diffusion_explicit<M: MeshAccess>(
    field: &[f64], field_out: &mut [f64], mesh: &M, nu: f64, dt: f64,
) -> MhResult<()> {
    validate_diffusion_params(nu, dt)?;
    let n = mesh.n_cells();
    if field.len() != n || field_out.len() != n {
        return Err(MhError::InvalidInput(format!(
            "Field array size mismatch: expected {}, in={}, out={}",
            n, field.len(), field_out.len()
        )));
    }
    let flux_sum = compute_diffusion_fluxes(field, mesh, nu);
    field_out.par_iter_mut().enumerate().for_each(|(i, out)| {
        let area = mesh.cell_area(i);
        *out = if area > 1e-14 { field[i] + dt * flux_sum[i] / area } else { field[i] };
    });
    Ok(())
}

fn compute_diffusion_fluxes<M: MeshAccess>(field: &[f64], mesh: &M, nu: f64) -> Vec<f64> {
    let mut flux_sum = vec![0.0; mesh.n_cells()];
    for face_idx in 0..mesh.n_faces() {
        if mesh.is_boundary_face(face_idx) { continue; }
        let (owner, neighbor) = mesh.face_cells(face_idx);
        if neighbor.is_none() { continue; }
        let neighbor = neighbor.unwrap();
        let d = mesh.face_distance(face_idx);
        if d < 1e-14 { continue; }
        let length = mesh.face_length(face_idx);
        let flux = -nu * (field[neighbor] - field[owner]) / d * length;
        flux_sum[owner] += flux;
        flux_sum[neighbor] -= flux;
    }
    flux_sum
}

pub fn apply_diffusion_explicit_variable<M: MeshAccess>(
    field: &[f64], field_out: &mut [f64], mesh: &M, nu: &[f64], dt: f64,
) -> MhResult<()> {
    let n = mesh.n_cells();
    if field.len() != n || field_out.len() != n || nu.len() != n {
        return Err(MhError::InvalidInput("Array size mismatch".into()));
    }
    let mut flux_sum = vec![0.0; n];
    for face_idx in 0..mesh.n_faces() {
        if mesh.is_boundary_face(face_idx) { continue; }
        let (owner, neighbor_opt) = mesh.face_cells(face_idx);
        if neighbor_opt.is_none() { continue; }
        let neighbor = neighbor_opt.unwrap();
        let d = mesh.face_distance(face_idx);
        if d < 1e-14 { continue; }
        let length = mesh.face_length(face_idx);
        let nu_o = nu[owner];
        let nu_n = nu[neighbor];
        let nu_face = if nu_o + nu_n > 1e-14 { 2.0 * nu_o * nu_n / (nu_o + nu_n) } else { 0.0 };
        let flux = -nu_face * (field[neighbor] - field[owner]) / d * length;
        flux_sum[owner] += flux;
        flux_sum[neighbor] -= flux;
    }
    field_out.par_iter_mut().enumerate().for_each(|(i, out)| {
        let area = mesh.cell_area(i);
        *out = if area > 1e-14 { field[i] + dt * flux_sum[i] / area } else { field[i] };
    });
    Ok(())
}

pub fn apply_diffusion_inplace<M: MeshAccess>(
    field: &mut [f64], mesh: &M, nu: f64, dt: f64,
) -> MhResult<()> {
    let mut temp = vec![0.0; field.len()];
    apply_diffusion_explicit(field, &mut temp, mesh, nu, dt)?;
    field.copy_from_slice(&temp);
    Ok(())
}

pub fn apply_diffusion_substeps<M: MeshAccess>(
    field: &mut [f64], mesh: &M, nu: f64, dt: f64, n_substeps: usize,
) -> MhResult<()> {
    if n_substeps == 0 { return Ok(()); }
    let sub_dt = dt / n_substeps as f64;
    let mut buffer = vec![0.0; field.len()];
    for step in 0..n_substeps {
        if step % 2 == 0 {
            apply_diffusion_explicit(field, &mut buffer, mesh, nu, sub_dt)?;
        } else {
            apply_diffusion_explicit(&buffer, field, mesh, nu, sub_dt)?;
        }
    }
    if n_substeps % 2 == 1 { field.copy_from_slice(&buffer); }
    Ok(())
}

fn validate_diffusion_params(nu: f64, dt: f64) -> MhResult<()> {
    if nu < 0.0 {
        return Err(MhError::InvalidInput(format!("Diffusion coefficient cannot be negative: ν={:.3}", nu)));
    }
    if dt <= 0.0 {
        return Err(MhError::InvalidInput(format!("Time step must be positive: dt={:.3}", dt)));
    }
    Ok(())
}
