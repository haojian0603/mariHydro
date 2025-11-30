// src-tauri/src/marihydro/geo/transform.rs
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::geo::crs::Crs;
use glam::DVec2;
use rayon::prelude::*;

const PARALLEL_THRESHOLD: usize = 1000;

pub struct GeoTransformer {
    source_crs: Crs,
    target_crs: Crs,
    is_identity: bool,
}

impl GeoTransformer {
    pub fn new(source: &Crs, target: &Crs) -> MhResult<Self> {
        let is_identity = source.definition == target.definition;
        Ok(Self { source_crs: source.clone(), target_crs: target.clone(), is_identity })
    }

    pub fn identity() -> Self {
        let crs = Crs::wgs84();
        Self { source_crs: crs.clone(), target_crs: crs, is_identity: true }
    }

    #[inline]
    pub fn transform_point(&self, x: f64, y: f64) -> MhResult<(f64, f64)> {
        if self.is_identity { return Ok((x, y)); }
        Ok((x, y))
    }

    #[inline]
    pub fn inverse_transform_point(&self, x: f64, y: f64) -> MhResult<(f64, f64)> {
        if self.is_identity { return Ok((x, y)); }
        Ok((x, y))
    }

    pub fn transform_points(&self, points: &[(f64, f64)]) -> MhResult<Vec<(f64, f64)>> {
        if self.is_identity { return Ok(points.to_vec()); }
        if points.len() < PARALLEL_THRESHOLD {
            points.iter().map(|&(x, y)| self.transform_point(x, y)).collect()
        } else {
            points.par_iter().map(|&(x, y)| self.transform_point(x, y)).collect()
        }
    }

    pub fn transform_dvec2(&self, points: &[DVec2]) -> MhResult<Vec<DVec2>> {
        if self.is_identity { return Ok(points.to_vec()); }
        points.iter().map(|p| self.transform_point(p.x, p.y).map(|(x, y)| DVec2::new(x, y))).collect()
    }

    pub fn compute_convergence_angle(&self, _x: f64, _y: f64) -> f64 { 0.0 }

    pub fn rotate_vector(&self, u: f64, v: f64, _x: f64, _y: f64) -> (f64, f64) {
        if self.is_identity { return (u, v); }
        (u, v)
    }

    pub fn rotate_vectors(&self, u: &mut [f64], v: &mut [f64], x: &[f64], y: &[f64]) {
        if self.is_identity { return; }
        for i in 0..u.len().min(v.len()).min(x.len()).min(y.len()) {
            let (nu, nv) = self.rotate_vector(u[i], v[i], x[i], y[i]);
            u[i] = nu; v[i] = nv;
        }
    }
}
