// src-tauri/src/marihydro/physics/sources/inflow.rs

use crate::marihydro::forcing::context::ActiveRiverSource;
use crate::marihydro::infra::error::{MhError, MhResult};

pub fn apply_river_inflow(
    h: &mut [f64],
    cell_areas: &[f64],
    rivers: &[ActiveRiverSource],
    dt: f64,
) -> MhResult<usize> {
    if dt <= 0.0 || dt > 3600.0 {
        return Err(MhError::InvalidInput(format!(
            "时间步长异常: dt = {:.3} s",
            dt
        )));
    }
    if rivers.is_empty() {
        return Ok(0);
    }
    let mut applied_count = 0;
    for river in rivers {
        let idx = river.cell_id.idx();
        if idx >= h.len() {
            log::warn!(
                "河流源 '{}' 索引越界: idx={}, len={}",
                river.name,
                idx,
                h.len()
            );
            continue;
        }
        let area = cell_areas[idx];
        if area <= 0.0 {
            log::warn!("河流源 '{}' 单元面积无效: {:.6}", river.name, area);
            continue;
        }
        let delta_h = river.flow_rate * dt / area;
        let cell = &mut h[idx];
        if *cell + delta_h < 0.0 {
            log::warn!(
                "河流 '{}' 抽水过量: h={:.3} m, delta={:.3} m",
                river.name,
                cell,
                delta_h
            );
            *cell = 0.0;
        } else {
            *cell += delta_h;
        }
        applied_count += 1;
    }
    if applied_count < rivers.len() {
        log::warn!("部分河流源未应用: {}/{}", applied_count, rivers.len());
    }
    Ok(applied_count)
}

pub fn apply_river_inflow_with_momentum(
    h: &mut [f64],
    hu: &mut [f64],
    hv: &mut [f64],
    cell_areas: &[f64],
    rivers: &[ActiveRiverSource],
    river_velocities: &[(f64, f64)],
    dt: f64,
) -> MhResult<usize> {
    if dt <= 0.0 || dt > 3600.0 {
        return Err(MhError::InvalidInput(format!(
            "时间步长异常: dt = {:.3} s",
            dt
        )));
    }
    if rivers.is_empty() {
        return Ok(0);
    }
    if rivers.len() != river_velocities.len() {
        return Err(MhError::InvalidInput("河流源数量与速度数量不匹配".into()));
    }
    let mut applied_count = 0;
    for (river, &(u_river, v_river)) in rivers.iter().zip(river_velocities) {
        let idx = river.cell_id.idx();
        if idx >= h.len() {
            log::warn!(
                "河流源 '{}' 索引越界: idx={}, len={}",
                river.name,
                idx,
                h.len()
            );
            continue;
        }
        let area = cell_areas[idx];
        if area <= 0.0 {
            continue;
        }
        let h_old = h[idx];
        let u_old = if h_old > 1e-6 { hu[idx] / h_old } else { 0.0 };
        let v_old = if h_old > 1e-6 { hv[idx] / h_old } else { 0.0 };
        let delta_h = river.flow_rate * dt / area;
        let h_new = h_old + delta_h;
        if h_new < 0.0 {
            log::warn!(
                "河流 '{}' 抽水导致负水深: h_old={:.3}, delta={:.3}",
                river.name,
                h_old,
                delta_h
            );
            h[idx] = 0.0;
            hu[idx] = 0.0;
            hv[idx] = 0.0;
            continue;
        }
        let hu_old = h_old * u_old;
        let hv_old = h_old * v_old;
        let hu_new = hu_old + delta_h * u_river;
        let hv_new = hv_old + delta_h * v_river;
        h[idx] = h_new;
        hu[idx] = hu_new;
        hv[idx] = hv_new;
        applied_count += 1;
    }
    Ok(applied_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marihydro::domain::mesh::indices::CellId;

    fn make_river(idx: usize, flow_rate: f64, name: &str) -> ActiveRiverSource {
        ActiveRiverSource {
            cell_id: CellId::new(idx),
            flow_rate,
            name: name.to_string(),
        }
    }

    #[test]
    fn test_basic_inflow() {
        let mut h = vec![0.0; 100];
        let areas = vec![100.0; 100];
        let rivers = vec![
            make_river(11, 10.0, "River1"),
            make_river(22, -5.0, "River2"),
        ];
        let count = apply_river_inflow(&mut h, &areas, &rivers, 10.0).unwrap();
        assert_eq!(count, 2);
        assert!((h[11] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_negative_depth_protection() {
        let mut h = vec![1.0; 100];
        let areas = vec![100.0; 100];
        let rivers = vec![make_river(11, -50.0, "Pump")];
        apply_river_inflow(&mut h, &areas, &rivers, 10.0).unwrap();
        assert_eq!(h[11], 0.0);
    }

    #[test]
    fn test_empty_rivers() {
        let mut h = vec![0.0; 100];
        let areas = vec![100.0; 100];
        let count = apply_river_inflow(&mut h, &areas, &[], 1.0).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_invalid_dt() {
        let mut h = vec![0.0; 100];
        let areas = vec![100.0; 100];
        let rivers = vec![make_river(11, 1.0, "R1")];
        assert!(apply_river_inflow(&mut h, &areas, &rivers, -1.0).is_err());
        assert!(apply_river_inflow(&mut h, &areas, &rivers, 5000.0).is_err());
    }

    #[test]
    fn test_out_of_bounds() {
        let mut h = vec![0.0; 100];
        let areas = vec![100.0; 100];
        let rivers = vec![make_river(1000, 1.0, "OutOfBounds")];
        let count = apply_river_inflow(&mut h, &areas, &rivers, 1.0).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_momentum_inflow() {
        let mut h = vec![1.0; 100];
        let mut hu = vec![0.5; 100];
        let mut hv = vec![0.0; 100];
        let areas = vec![100.0; 100];
        let rivers = vec![make_river(11, 100.0, "R1")];
        let velocities = vec![(2.0, 1.0)];
        let count = apply_river_inflow_with_momentum(
            &mut h,
            &mut hu,
            &mut hv,
            &areas,
            &rivers,
            &velocities,
            1.0,
        )
        .unwrap();
        assert_eq!(count, 1);
        assert!((h[11] - 2.0).abs() < 1e-10);
        assert!((hu[11] - 2.5).abs() < 1e-10);
        assert!((hv[11] - 1.0).abs() < 1e-10);
    }
}
