// src-tauri/src/marihydro/physics/numerics/limiter/venkatakrishnan.rs

//! Venkatakrishnan 限制器（更平滑）
//!
//! 参数 k 控制限制器的行为：
//! - 较小的 k (0.1-0.3): 更保守，适合激波/强梯度区域
//! - 中等的 k (0.3-0.5): 平衡选择，适合一般情况
//! - 较大的 k (0.5-1.0): 更精确，适合平滑流动
//!
//! 参考值:
//! - 水力激波、溃坝: k ≈ 0.2
//! - 一般河流流动: k ≈ 0.3-0.5
//! - 平静水面、潮汐: k ≈ 0.5-0.7

use super::{GradientLimiter, ScalarGradientStorage};
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::CellIndex;
use glam::DVec2;
use rayon::prelude::*;

pub struct VenkatakrishnanLimiter {
    k: f64,
}

impl Default for VenkatakrishnanLimiter {
    fn default() -> Self {
        Self { k: 0.3 }
    }
}

impl VenkatakrishnanLimiter {
    /// 创建自定义参数的限制器
    ///
    /// # 参数
    /// - `k`: 限制器参数，推荐范围 [0.1, 1.0]
    pub fn new(k: f64) -> Self {
        Self { k: k.clamp(0.01, 2.0) }
    }

    /// 保守模式：适用于激波、溃坝等强梯度流动
    ///
    /// k = 0.2，产生较多耗散但更稳定
    pub fn conservative() -> Self {
        Self { k: 0.2 }
    }

    /// 标准模式：适用于一般河流流动
    ///
    /// k = 0.3，平衡稳定性和精度
    pub fn standard() -> Self {
        Self::default()
    }

    /// 精确模式：适用于平滑流动，如潮汐、缓流
    ///
    /// k = 0.5，较少耗散，更高精度
    pub fn accurate() -> Self {
        Self { k: 0.5 }
    }

    /// 高精度模式：适用于非常平滑的流动
    ///
    /// k = 0.7，最少耗散，需要确保流场平滑
    pub fn high_accuracy() -> Self {
        Self { k: 0.7 }
    }

    /// 根据弗劳德数自适应选择参数
    ///
    /// - Fr > 1 (超临界流): 使用保守参数
    /// - Fr ≈ 1 (临界流): 使用标准参数
    /// - Fr < 1 (亚临界流): 使用精确参数
    pub fn adaptive(froude_number: f64) -> Self {
        let k = if froude_number > 1.5 {
            0.2  // 强超临界流，保守
        } else if froude_number > 1.0 {
            0.25 // 弱超临界流
        } else if froude_number > 0.5 {
            0.35 // 中等亚临界流
        } else {
            0.5  // 平缓流动，精确
        };
        Self { k }
    }

    /// 获取当前参数值
    pub fn k(&self) -> f64 {
        self.k
    }

    #[inline]
    fn venkat_fn(delta_max: f64, delta: f64, eps_sq: f64) -> f64 {
        if delta.abs() < 1e-30 {
            return 1.0;
        }
        let dm2 = delta_max * delta_max;
        let d2 = delta * delta;
        let num = dm2 + eps_sq + 2.0 * delta_max * delta;
        let den = dm2 + 2.0 * d2 + delta_max * delta + eps_sq;
        if den.abs() < 1e-30 {
            1.0
        } else {
            (num / den).clamp(0.0, 1.0)
        }
    }

    fn compute_cell<M: MeshAccess>(
        &self,
        i: usize,
        field: &[f64],
        grad: &ScalarGradientStorage,
        mesh: &M,
    ) -> f64 {
        let cell = CellIndex(i);
        let phi_c = field[i];
        let center = mesh.cell_centroid(cell);
        let g = grad.get(i);
        let h = mesh.cell_area(cell).sqrt();
        let eps_sq = (self.k * h).powi(3);

        let (mut phi_max, mut phi_min) = (phi_c, phi_c);
        for &nb in mesh.cell_neighbors(cell) {
            if nb.is_valid() {
                let pn = field[nb.0];
                phi_max = phi_max.max(pn);
                phi_min = phi_min.min(pn);
            }
        }
        let (dmax, dmin) = (phi_max - phi_c, phi_min - phi_c);

        let mut alpha: f64 = 1.0;
        for &face in mesh.cell_faces(cell) {
            let r = mesh.face_centroid(face) - center;
            let delta = g.dot(r);
            if delta > 1e-14 {
                alpha = alpha.min(Self::venkat_fn(dmax, delta, eps_sq));
            } else if delta < -1e-14 {
                alpha = alpha.min(Self::venkat_fn(-dmin, -delta, eps_sq));
            }
        }
        alpha
    }
}

impl GradientLimiter for VenkatakrishnanLimiter {
    fn limit<M: MeshAccess>(
        &self,
        field: &[f64],
        gradient: &mut ScalarGradientStorage,
        mesh: &M,
    ) -> MhResult<()> {
        let mut lim = vec![1.0; mesh.n_cells()];
        self.compute_limiters(field, gradient, mesh, &mut lim)?;
        gradient.apply_limiter(&lim);
        Ok(())
    }

    fn compute_limiters<M: MeshAccess>(
        &self,
        field: &[f64],
        gradient: &ScalarGradientStorage,
        mesh: &M,
        output: &mut [f64],
    ) -> MhResult<()> {
        for i in 0..mesh.n_cells() {
            output[i] = self.compute_cell(i, field, gradient, mesh);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Venkatakrishnan"
    }
}

pub mod optimized {
    use super::*;
    use crate::marihydro::domain::mesh::UnstructuredMesh;

    pub fn limit_parallel(
        field: &[f64],
        gradient: &mut ScalarGradientStorage,
        mesh: &UnstructuredMesh,
        k: f64,
    ) -> MhResult<()> {
        let lim: Vec<f64> = (0..mesh.n_cells())
            .into_par_iter()
            .map(|i| {
                let cell = CellIndex(i);
                let phi_c = field[i];
                let center = mesh.cell_centroid(cell);
                let g = gradient.get(i);
                let h = mesh.cell_area(cell).sqrt();
                let eps_sq = (k * h).powi(3);

                let (mut pmax, mut pmin) = (phi_c, phi_c);
                for &nb in mesh.cell_neighbors(cell) {
                    if nb.is_valid() {
                        let pn = field[nb.0];
                        pmax = pmax.max(pn);
                        pmin = pmin.min(pn);
                    }
                }
                let (dmax, dmin) = (pmax - phi_c, pmin - phi_c);

                let mut a: f64 = 1.0;
                for &face in mesh.cell_faces(cell) {
                    let r = mesh.face_centroid(face) - center;
                    let d = g.dot(r);
                    if d > 1e-14 {
                        a = a.min(VenkatakrishnanLimiter::venkat_fn(dmax, d, eps_sq));
                    } else if d < -1e-14 {
                        a = a.min(VenkatakrishnanLimiter::venkat_fn(-dmin, -d, eps_sq));
                    }
                }
                a
            })
            .collect();
        gradient.apply_limiter(&lim);
        Ok(())
    }
}
