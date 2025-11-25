// src-tauri/src/marihydro/forcing/river.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::marihydro::domain::mesh::Mesh;
use crate::marihydro::infra::error::{MhError, MhResult};

use super::context::RiverSource;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiverTimeSeries {
    pub times: Vec<f64>,
    pub flow_rates: Vec<f64>,

    #[cfg(feature = "river_momentum")]
    pub velocities_u: Option<Vec<f64>>,

    #[cfg(feature = "river_momentum")]
    pub velocities_v: Option<Vec<f64>>,
}

impl RiverTimeSeries {
    pub fn constant(flow_rate: f64) -> Self {
        Self {
            times: vec![0.0],
            flow_rates: vec![flow_rate],
            #[cfg(feature = "river_momentum")]
            velocities_u: None,
            #[cfg(feature = "river_momentum")]
            velocities_v: None,
        }
    }

    #[cfg(feature = "river_momentum")]
    pub fn constant_with_velocity(flow_rate: f64, velocity_u: f64, velocity_v: f64) -> Self {
        Self {
            times: vec![0.0],
            flow_rates: vec![flow_rate],
            velocities_u: Some(vec![velocity_u]),
            velocities_v: Some(vec![velocity_v]),
        }
    }

    pub fn new(times: Vec<f64>, flow_rates: Vec<f64>) -> MhResult<Self> {
        if times.len() != flow_rates.len() {
            return Err(MhError::InvalidInput("时间序列长度不匹配".into()));
        }

        if times.is_empty() {
            return Err(MhError::InvalidInput("时间序列不能为空".into()));
        }

        for i in 0..times.len() - 1 {
            if times[i] >= times[i + 1] {
                return Err(MhError::InvalidInput("时间序列必须单调递增".into()));
            }
        }

        Ok(Self {
            times,
            flow_rates,
            #[cfg(feature = "river_momentum")]
            velocities_u: None,
            #[cfg(feature = "river_momentum")]
            velocities_v: None,
        })
    }

    pub fn interpolate_flow(&self, t: f64) -> f64 {
        if self.times.len() == 1 {
            return self.flow_rates[0];
        }

        if t <= self.times[0] {
            return self.flow_rates[0];
        }
        if t >= *self.times.last().unwrap() {
            return *self.flow_rates.last().unwrap();
        }

        for i in 0..self.times.len() - 1 {
            if t >= self.times[i] && t < self.times[i + 1] {
                let t0 = self.times[i];
                let t1 = self.times[i + 1];
                let q0 = self.flow_rates[i];
                let q1 = self.flow_rates[i + 1];

                let alpha = (t - t0) / (t1 - t0);
                return q0 + alpha * (q1 - q0);
            }
        }

        *self.flow_rates.last().unwrap()
    }

    #[cfg(feature = "river_momentum")]
    pub fn interpolate_velocity(&self, t: f64) -> Option<(f64, f64)> {
        let u_vec = self.velocities_u.as_ref()?;
        let v_vec = self.velocities_v.as_ref()?;

        if self.times.len() == 1 {
            return Some((u_vec[0], v_vec[0]));
        }

        if t <= self.times[0] {
            return Some((u_vec[0], v_vec[0]));
        }
        if t >= *self.times.last().unwrap() {
            return Some((*u_vec.last().unwrap(), *v_vec.last().unwrap()));
        }

        for i in 0..self.times.len() - 1 {
            if t >= self.times[i] && t < self.times[i + 1] {
                let t0 = self.times[i];
                let t1 = self.times[i + 1];
                let alpha = (t - t0) / (t1 - t0);

                let u = u_vec[i] + alpha * (u_vec[i + 1] - u_vec[i]);
                let v = v_vec[i] + alpha * (v_vec[i + 1] - v_vec[i]);

                return Some((u, v));
            }
        }

        Some((*u_vec.last().unwrap(), *v_vec.last().unwrap()))
    }
}

pub struct RiverManager {
    sources: Vec<RiverSource>,
    time_series: HashMap<usize, RiverTimeSeries>,
    current_time: f64,
}

impl RiverManager {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            time_series: HashMap::new(),
            current_time: 0.0,
        }
    }

    pub fn add_constant_source(&mut self, idx_1d: usize, flow_rate: f64) {
        self.sources.push(RiverSource::new(idx_1d, flow_rate));
        self.time_series
            .insert(idx_1d, RiverTimeSeries::constant(flow_rate));
    }

    #[cfg(feature = "river_momentum")]
    pub fn add_constant_source_with_velocity(
        &mut self,
        idx_1d: usize,
        flow_rate: f64,
        velocity_u: f64,
        velocity_v: f64,
    ) {
        self.sources.push(RiverSource::with_velocity(
            idx_1d, flow_rate, velocity_u, velocity_v,
        ));
        self.time_series.insert(
            idx_1d,
            RiverTimeSeries::constant_with_velocity(flow_rate, velocity_u, velocity_v),
        );
    }

    pub fn add_time_series(&mut self, idx_1d: usize, time_series: RiverTimeSeries) -> MhResult<()> {
        if time_series.times.len() != time_series.flow_rates.len() {
            return Err(MhError::InvalidInput("时间序列长度不匹配".into()));
        }

        for i in 0..time_series.times.len() - 1 {
            if time_series.times[i] >= time_series.times[i + 1] {
                return Err(MhError::InvalidInput("时间序列必须单调递增".into()));
            }
        }

        #[cfg(feature = "river_momentum")]
        {
            if let Some(ref u_vec) = time_series.velocities_u {
                if u_vec.len() != time_series.times.len() {
                    return Err(MhError::InvalidInput("速度时间序列长度不匹配".into()));
                }
            }
            if let Some(ref v_vec) = time_series.velocities_v {
                if v_vec.len() != time_series.times.len() {
                    return Err(MhError::InvalidInput("速度时间序列长度不匹配".into()));
                }
            }
        }

        let initial_flow = time_series.flow_rates[0];

        #[cfg(feature = "river_momentum")]
        {
            if let (Some(u_vec), Some(v_vec)) =
                (&time_series.velocities_u, &time_series.velocities_v)
            {
                self.sources.push(RiverSource::with_velocity(
                    idx_1d,
                    initial_flow,
                    u_vec[0],
                    v_vec[0],
                ));
            } else {
                self.sources.push(RiverSource::new(idx_1d, initial_flow));
            }
        }

        #[cfg(not(feature = "river_momentum"))]
        {
            self.sources.push(RiverSource::new(idx_1d, initial_flow));
        }

        self.time_series.insert(idx_1d, time_series);

        Ok(())
    }

    pub fn update(&mut self, current_time: f64) {
        self.current_time = current_time;

        for source in &mut self.sources {
            if let Some(ts) = self.time_series.get(&source.idx_1d) {
                source.flow_rate = ts.interpolate_flow(current_time);

                #[cfg(feature = "river_momentum")]
                {
                    if let Some((u, v)) = ts.interpolate_velocity(current_time) {
                        source.velocity_u = u;
                        source.velocity_v = v;
                    }
                }
            }
        }
    }

    pub fn get_sources(&self) -> &[RiverSource] {
        &self.sources
    }

    pub fn validate(&self, mesh: &Mesh) -> MhResult<()> {
        use std::collections::HashSet;

        let mut seen = HashSet::new();
        let total_cells = mesh.total_size().0 * mesh.total_size().1;

        for source in &self.sources {
            if source.idx_1d >= total_cells {
                return Err(MhError::InvalidInput(format!(
                    "河流源索引越界: {} >= {}",
                    source.idx_1d, total_cells
                )));
            }

            if !seen.insert(source.idx_1d) {
                return Err(MhError::InvalidInput(format!(
                    "河流源索引重复: {}",
                    source.idx_1d
                )));
            }

            const MAX_FLOW_RATE: f64 = 100.0;
            if source.flow_rate.abs() > MAX_FLOW_RATE {
                log::warn!(
                    "河流源流量异常: {:.2} m/s at idx {}",
                    source.flow_rate,
                    source.idx_1d
                );
            }
        }

        Ok(())
    }

    pub fn clear(&mut self) {
        self.sources.clear();
        self.time_series.clear();
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    pub fn remove_source(&mut self, idx_1d: usize) -> bool {
        if let Some(pos) = self.sources.iter().position(|s| s.idx_1d == idx_1d) {
            self.sources.remove(pos);
            self.time_series.remove(&idx_1d);
            true
        } else {
            false
        }
    }
}

impl Default for RiverManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_series_constant() {
        let ts = RiverTimeSeries::constant(10.0);
        assert_eq!(ts.interpolate_flow(0.0), 10.0);
        assert_eq!(ts.interpolate_flow(100.0), 10.0);
    }

    #[test]
    fn test_time_series_new_validation() {
        let result = RiverTimeSeries::new(vec![0.0, 10.0, 5.0], vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());

        let result = RiverTimeSeries::new(vec![0.0, 10.0], vec![1.0]);
        assert!(result.is_err());

        let result = RiverTimeSeries::new(vec![0.0, 10.0], vec![1.0, 2.0]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_time_series_interpolation() {
        let ts = RiverTimeSeries::new(vec![0.0, 10.0, 20.0], vec![0.0, 10.0, 5.0]).unwrap();

        assert_eq!(ts.interpolate_flow(0.0), 0.0);
        assert_eq!(ts.interpolate_flow(10.0), 10.0);
        assert_eq!(ts.interpolate_flow(5.0), 5.0);
        assert_eq!(ts.interpolate_flow(15.0), 7.5);
        assert_eq!(ts.interpolate_flow(100.0), 5.0);
        assert_eq!(ts.interpolate_flow(-10.0), 0.0);
    }

    #[test]
    fn test_river_manager_add_constant() {
        let mut mgr = RiverManager::new();
        mgr.add_constant_source(100, 10.0);

        assert_eq!(mgr.len(), 1);
        assert_eq!(mgr.get_sources()[0].flow_rate, 10.0);
        assert_eq!(mgr.get_sources()[0].idx_1d, 100);
    }

    #[test]
    fn test_river_manager_update() {
        let mut mgr = RiverManager::new();

        let ts = RiverTimeSeries::new(vec![0.0, 10.0], vec![5.0, 15.0]).unwrap();

        mgr.add_time_series(100, ts).unwrap();

        mgr.update(0.0);
        assert_eq!(mgr.get_sources()[0].flow_rate, 5.0);

        mgr.update(5.0);
        assert_eq!(mgr.get_sources()[0].flow_rate, 10.0);

        mgr.update(10.0);
        assert_eq!(mgr.get_sources()[0].flow_rate, 15.0);

        assert_eq!(mgr.current_time(), 10.0);
    }

    #[test]
    fn test_river_manager_clear() {
        let mut mgr = RiverManager::new();
        mgr.add_constant_source(100, 10.0);
        mgr.add_constant_source(200, 20.0);

        assert_eq!(mgr.len(), 2);

        mgr.clear();
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_river_manager_remove() {
        let mut mgr = RiverManager::new();
        mgr.add_constant_source(100, 10.0);
        mgr.add_constant_source(200, 20.0);

        assert!(mgr.remove_source(100));
        assert_eq!(mgr.len(), 1);

        assert!(!mgr.remove_source(999));
        assert_eq!(mgr.len(), 1);
    }

    #[test]
    fn test_time_series_validation_in_manager() {
        let mut mgr = RiverManager::new();

        let invalid_ts = RiverTimeSeries {
            times: vec![0.0, 10.0],
            flow_rates: vec![5.0],
            #[cfg(feature = "river_momentum")]
            velocities_u: None,
            #[cfg(feature = "river_momentum")]
            velocities_v: None,
        };

        assert!(mgr.add_time_series(100, invalid_ts).is_err());
    }

    #[test]
    #[cfg(feature = "river_momentum")]
    fn test_constant_with_velocity() {
        let ts = RiverTimeSeries::constant_with_velocity(10.0, 2.0, 1.5);

        assert_eq!(ts.interpolate_flow(0.0), 10.0);

        let (u, v) = ts.interpolate_velocity(0.0).unwrap();
        assert_eq!(u, 2.0);
        assert_eq!(v, 1.5);
    }

    #[test]
    #[cfg(feature = "river_momentum")]
    fn test_velocity_interpolation() {
        let mut ts = RiverTimeSeries::new(vec![0.0, 10.0], vec![5.0, 15.0]).unwrap();

        ts.velocities_u = Some(vec![0.0, 2.0]);
        ts.velocities_v = Some(vec![0.0, 4.0]);

        let (u, v) = ts.interpolate_velocity(5.0).unwrap();
        assert_eq!(u, 1.0);
        assert_eq!(v, 2.0);
    }

    #[test]
    #[cfg(feature = "river_momentum")]
    fn test_manager_with_velocity() {
        let mut mgr = RiverManager::new();
        mgr.add_constant_source_with_velocity(100, 10.0, 2.0, 1.5);

        assert_eq!(mgr.get_sources()[0].velocity_u, 2.0);
        assert_eq!(mgr.get_sources()[0].velocity_v, 1.5);
    }
}
