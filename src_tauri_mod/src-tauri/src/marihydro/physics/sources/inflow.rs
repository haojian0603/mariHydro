// src-tauri/src/marihydro/physics/sources/inflow.rs
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::types::CellIndex;

#[derive(Debug, Clone)]
pub struct ActiveRiverSource {
    pub cell_id: CellIndex,
    pub flow_rate: f64,
    pub name: String,
}

impl ActiveRiverSource {
    pub fn new(cell_id: CellIndex, flow_rate: f64, name: String) -> Self {
        Self { cell_id, flow_rate, name }
    }
}

pub fn apply_river_inflow(
    h: &mut [f64], cell_areas: &[f64], rivers: &[ActiveRiverSource], dt: f64,
) -> MhResult<usize> {
    if dt <= 0.0 || dt > 3600.0 {
        return Err(MhError::invalid_input(format!("Time step abnormal: dt={:.3}s", dt)));
    }
    if rivers.is_empty() { return Ok(0); }
    let mut applied_count = 0;
    for river in rivers {
        let idx = river.cell_id.get();
        if idx >= h.len() { continue; }
        let area = cell_areas[idx];
        if area <= 0.0 { continue; }
        let delta_h = river.flow_rate * dt / area;
        let cell = &mut h[idx];
        if *cell + delta_h < 0.0 { *cell = 0.0; }
        else { *cell += delta_h; }
        applied_count += 1;
    }
    Ok(applied_count)
}

pub fn apply_river_inflow_with_momentum(
    h: &mut [f64], hu: &mut [f64], hv: &mut [f64],
    cell_areas: &[f64], rivers: &[ActiveRiverSource],
    river_velocities: &[(f64, f64)], dt: f64,
) -> MhResult<usize> {
    if dt <= 0.0 || dt > 3600.0 {
        return Err(MhError::invalid_input(format!("Time step abnormal: dt={:.3}s", dt)));
    }
    if rivers.is_empty() { return Ok(0); }
    if rivers.len() != river_velocities.len() {
        return Err(MhError::invalid_input("River count != velocity count"));
    }
    let mut applied_count = 0;
    for (river, &(u_river, v_river)) in rivers.iter().zip(river_velocities) {
        let idx = river.cell_id.get();
        if idx >= h.len() { continue; }
        let area = cell_areas[idx];
        if area <= 0.0 { continue; }
        let h_old = h[idx];
        let u_old = if h_old > 1e-6 { hu[idx] / h_old } else { 0.0 };
        let v_old = if h_old > 1e-6 { hv[idx] / h_old } else { 0.0 };
        let delta_h = river.flow_rate * dt / area;
        let h_new = h_old + delta_h;
        if h_new < 0.0 {
            h[idx] = 0.0; hu[idx] = 0.0; hv[idx] = 0.0;
            continue;
        }
        h[idx] = h_new;
        hu[idx] = h_old * u_old + delta_h * u_river;
        hv[idx] = h_old * v_old + delta_h * v_river;
        applied_count += 1;
    }
    Ok(applied_count)
}

#[derive(Debug, Clone)]
pub struct PointSource {
    pub cell_id: CellIndex,
    pub mass_rate: f64,
    pub name: String,
}

impl PointSource {
    pub fn new(cell_id: CellIndex, mass_rate: f64, name: String) -> Self {
        Self { cell_id, mass_rate, name }
    }
}

pub fn apply_point_sources(
    h: &mut [f64], cell_areas: &[f64], sources: &[PointSource], dt: f64,
) -> MhResult<usize> {
    if dt <= 0.0 { return Err(MhError::invalid_input("dt must be positive")); }
    let mut count = 0;
    for src in sources {
        let idx = src.cell_id.get();
        if idx >= h.len() { continue; }
        let area = cell_areas[idx];
        if area <= 0.0 { continue; }
        h[idx] += src.mass_rate * dt / area;
        if h[idx] < 0.0 { h[idx] = 0.0; }
        count += 1;
    }
    Ok(count)
}
