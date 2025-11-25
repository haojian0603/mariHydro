//! 河流入流源项

use ndarray::ArrayViewMut2;

use crate::marihydro::forcing::context::RiverSource;
use crate::marihydro::infra::error::MhResult;

pub fn apply_river_inflow(
    h: &mut ArrayViewMut2<f64>,
    rivers: &[RiverSource],
    dt: f64,
) -> MhResult<usize> {
    if dt <= 0.0 || dt > 3600.0 {
        return Err(crate::marihydro::infra::error::MhError::InvalidInput(
            format!("时间步长异常: dt = {:.3} s", dt),
        ));
    }

    if rivers.is_empty() {
        return Ok(0);
    }

    let h_slice = h.as_slice_memory_order_mut().ok_or_else(|| {
        crate::marihydro::infra::error::MhError::Runtime("State数组内存非连续".into())
    })?;

    let mut applied_count = 0;

    for river in rivers {
        if river.idx_1d >= h_slice.len() {
            log::warn!(
                "河流源索引越界: idx={}, len={}",
                river.idx_1d,
                h_slice.len()
            );
            continue;
        }

        let cell = &mut h_slice[river.idx_1d];
        let delta_h = river.flow_rate * dt;

        if *cell + delta_h < 0.0 {
            log::warn!("河流抽水过量: h={:.3} m, delta={:.3} m", cell, delta_h);
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

#[cfg(feature = "river_momentum")]
pub fn apply_river_inflow_with_momentum(
    h: &mut ArrayViewMut2<f64>,
    u: &mut ArrayViewMut2<f64>,
    v: &mut ArrayViewMut2<f64>,
    rivers: &[RiverSource],
    dt: f64,
) -> MhResult<usize> {
    if dt <= 0.0 || dt > 3600.0 {
        return Err(crate::marihydro::infra::error::MhError::InvalidInput(
            format!("时间步长异常: dt = {:.3} s", dt),
        ));
    }

    if rivers.is_empty() {
        return Ok(0);
    }

    let h_slice = h.as_slice_memory_order_mut().ok_or_else(|| {
        crate::marihydro::infra::error::MhError::Runtime("h数组内存非连续".into())
    })?;
    let u_slice = u.as_slice_memory_order_mut().ok_or_else(|| {
        crate::marihydro::infra::error::MhError::Runtime("u数组内存非连续".into())
    })?;
    let v_slice = v.as_slice_memory_order_mut().ok_or_else(|| {
        crate::marihydro::infra::error::MhError::Runtime("v数组内存非连续".into())
    })?;

    let mut applied_count = 0;

    for river in rivers {
        let idx = river.idx_1d;

        if idx >= h_slice.len() {
            log::warn!("河流源索引越界: idx={}, len={}", idx, h_slice.len());
            continue;
        }

        let h_old = h_slice[idx];
        let u_old = u_slice[idx];
        let v_old = v_slice[idx];

        let delta_h = river.flow_rate * dt;
        let h_new = h_old + delta_h;

        if h_new < 0.0 {
            log::warn!(
                "河流抽水导致负水深: h_old={:.3}, delta={:.3}",
                h_old,
                delta_h
            );
            h_slice[idx] = 0.0;
            u_slice[idx] = 0.0;
            v_slice[idx] = 0.0;
            continue;
        }

        let u_river = river.velocity_u;
        let v_river = river.velocity_v;

        let hu_old = h_old * u_old;
        let hv_old = h_old * v_old;

        let hu_new = hu_old + delta_h * u_river;
        let hv_new = hv_old + delta_h * v_river;

        h_slice[idx] = h_new;
        u_slice[idx] = hu_new / h_new;
        v_slice[idx] = hv_new / h_new;

        applied_count += 1;
    }

    Ok(applied_count)
}

#[cfg(debug_assertions)]
pub fn validate_river_indices(rivers: &[RiverSource]) -> Result<(), String> {
    use std::collections::HashSet;

    let mut seen = HashSet::new();

    for (i, river) in rivers.iter().enumerate() {
        if !seen.insert(river.idx_1d) {
            return Err(format!("河流源索引重复: idx={} 位置 {}", river.idx_1d, i));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_basic_inflow() {
        let mut h = Array2::zeros((10, 10));
        let rivers = vec![RiverSource::new(11, 1.0), RiverSource::new(22, -0.5)];

        let count = apply_river_inflow(&mut h.view_mut(), &rivers, 10.0).unwrap();

        assert_eq!(count, 2);
        assert_eq!(h[[1, 1]], 10.0);
    }

    #[test]
    fn test_negative_depth_protection() {
        let mut h = Array2::from_elem((10, 10), 1.0);
        let rivers = vec![RiverSource::new(11, -0.5)];

        apply_river_inflow(&mut h.view_mut(), &rivers, 10.0).unwrap();
        assert_eq!(h[[1, 1]], 0.0);
    }

    #[test]
    fn test_empty_rivers() {
        let mut h = Array2::zeros((10, 10));
        let count = apply_river_inflow(&mut h.view_mut(), &[], 1.0).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_invalid_dt() {
        let mut h = Array2::zeros((10, 10));
        let rivers = vec![RiverSource::new(11, 1.0)];

        assert!(apply_river_inflow(&mut h.view_mut(), &rivers, -1.0).is_err());
        assert!(apply_river_inflow(&mut h.view_mut(), &rivers, 5000.0).is_err());
    }

    #[test]
    fn test_out_of_bounds() {
        let mut h = Array2::zeros((10, 10));
        let rivers = vec![RiverSource::new(1000, 1.0)];

        let count = apply_river_inflow(&mut h.view_mut(), &rivers, 1.0).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_validate_unique() {
        let rivers = vec![RiverSource::new(10, 1.0), RiverSource::new(20, 2.0)];
        assert!(validate_river_indices(&rivers).is_ok());
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_validate_duplicate() {
        let rivers = vec![RiverSource::new(10, 1.0), RiverSource::new(10, 3.0)];
        assert!(validate_river_indices(&rivers).is_err());
    }

    #[test]
    #[cfg(feature = "river_momentum")]
    fn test_momentum_inflow() {
        let mut h = Array2::from_elem((10, 10), 1.0);
        let mut u = Array2::from_elem((10, 10), 0.5);
        let mut v = Array2::from_elem((10, 10), 0.0);

        let rivers = vec![RiverSource::with_velocity(11, 1.0, 2.0, 1.0)];

        let count = apply_river_inflow_with_momentum(
            &mut h.view_mut(),
            &mut u.view_mut(),
            &mut v.view_mut(),
            &rivers,
            1.0,
        )
        .unwrap();

        assert_eq!(count, 1);
        assert_eq!(h[[1, 1]], 2.0);

        let expected_u = (0.5 * 1.0 + 1.0 * 2.0) / 2.0;
        assert!((u[[1, 1]] - expected_u).abs() < 1e-10);
    }
}
