//! 湍流闭合模型

use ndarray::{Array2, Axis};
use rayon::prelude::*;

use crate::marihydro::domain::mesh::Mesh;
use crate::marihydro::infra::error::MhResult;

#[derive(Debug, Clone, Copy)]
pub struct SmagorinskyModel {
    pub cs: f64,
}

impl Default for SmagorinskyModel {
    fn default() -> Self {
        Self { cs: 0.15 }
    }
}

impl SmagorinskyModel {
    pub fn new(cs: f64) -> Self {
        Self { cs }
    }

    pub fn compute_eddy_viscosity(
        &self,
        u: &Array2<f64>,
        v: &Array2<f64>,
        mesh: &Mesh,
    ) -> MhResult<Array2<f64>> {
        let (dx, dy) = mesh.transform.resolution();
        let delta = (dx * dy).sqrt();
        let cs_delta_sq = (self.cs * delta).powi(2);

        let (total_ny, total_nx) = mesh.total_size();

        if u.dim() != (total_ny, total_nx) || v.dim() != (total_ny, total_nx) {
            return Err(crate::marihydro::infra::error::MhError::InvalidInput(
                "速度场尺寸与网格不匹配".into(),
            ));
        }

        let mut nu_t = Array2::zeros((total_ny, total_nx));

        let inv_2dx = 0.5 / dx;
        let inv_2dy = 0.5 / dy;

        let u_slice = u.as_slice_memory_order().ok_or_else(|| {
            crate::marihydro::infra::error::MhError::InternalError("u数组内存非连续".into())
        })?;
        let v_slice = v.as_slice_memory_order().ok_or_else(|| {
            crate::marihydro::infra::error::MhError::InternalError("v数组内存非连续".into())
        })?;

        let stride = total_nx;

        nu_t.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(j, mut row)| {
                if j == 0 || j == total_ny - 1 {
                    return;
                }
                for i in 1..total_nx - 1 {
                    let idx = j * stride + i;
                    unsafe {
                        let u_e = *u_slice.get_unchecked(idx + 1);
                        let u_w = *u_slice.get_unchecked(idx - 1);
                        let u_n = *u_slice.get_unchecked(idx + stride);
                        let u_s = *u_slice.get_unchecked(idx - stride);

                        let v_e = *v_slice.get_unchecked(idx + 1);
                        let v_w = *v_slice.get_unchecked(idx - 1);
                        let v_n = *v_slice.get_unchecked(idx + stride);
                        let v_s = *v_slice.get_unchecked(idx - stride);

                        let dudx = (u_e - u_w) * inv_2dx;
                        let dudy = (u_n - u_s) * inv_2dy;
                        let dvdx = (v_e - v_w) * inv_2dx;
                        let dvdy = (v_n - v_s) * inv_2dy;

                        let s_xx = dudx;
                        let s_yy = dvdy;
                        let s_xy = 0.5 * (dudy + dvdx);

                        let s_mag_sq = s_xx.powi(2) + s_yy.powi(2) + 2.0 * s_xy.powi(2);
                        let s_mag = (2.0 * s_mag_sq).sqrt();

                        row[i] = cs_delta_sq * s_mag;
                    }
                }
            });

        Ok(nu_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cs_default_range() {
        let model = SmagorinskyModel::default();
        assert!(model.cs > 0.0 && model.cs < 0.5);
    }

    #[test]
    fn test_custom_cs() {
        let model = SmagorinskyModel::new(0.12);
        assert_eq!(model.cs, 0.12);
    }
}
