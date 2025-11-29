// src-tauri/src/marihydro/physics/numerics/gradient.rs

use glam::DVec2;
use rayon::prelude::*;

use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;

/// 标量场梯度存储
#[derive(Debug, Clone)]
pub struct ScalarGradient {
    pub grad_x: Vec<f64>,
    pub grad_y: Vec<f64>,
}

impl ScalarGradient {
    pub fn new(n_cells: usize) -> Self {
        Self {
            grad_x: vec![0.0; n_cells],
            grad_y: vec![0.0; n_cells],
        }
    }

    pub fn reset(&mut self) {
        self.grad_x.iter_mut().for_each(|x| *x = 0.0);
        self.grad_y.iter_mut().for_each(|y| *y = 0.0);
    }

    #[inline]
    pub fn get(&self, idx: usize) -> DVec2 {
        DVec2::new(self.grad_x[idx], self.grad_y[idx])
    }

    #[inline]
    pub fn set(&mut self, idx: usize, grad: DVec2) {
        self.grad_x[idx] = grad.x;
        self.grad_y[idx] = grad.y;
    }

    pub fn len(&self) -> usize {
        self.grad_x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.grad_x.is_empty()
    }
}

/// 向量场梯度存储 (用于速度梯度张量)
#[derive(Debug, Clone)]
pub struct VectorGradient {
    pub dudx: Vec<f64>,
    pub dudy: Vec<f64>,
    pub dvdx: Vec<f64>,
    pub dvdy: Vec<f64>,
}

impl VectorGradient {
    pub fn new(n_cells: usize) -> Self {
        Self {
            dudx: vec![0.0; n_cells],
            dudy: vec![0.0; n_cells],
            dvdx: vec![0.0; n_cells],
            dvdy: vec![0.0; n_cells],
        }
    }

    pub fn reset(&mut self) {
        self.dudx.iter_mut().for_each(|x| *x = 0.0);
        self.dudy.iter_mut().for_each(|x| *x = 0.0);
        self.dvdx.iter_mut().for_each(|x| *x = 0.0);
        self.dvdy.iter_mut().for_each(|x| *x = 0.0);
    }

    #[inline]
    pub fn grad_u(&self, idx: usize) -> DVec2 {
        DVec2::new(self.dudx[idx], self.dudy[idx])
    }

    #[inline]
    pub fn grad_v(&self, idx: usize) -> DVec2 {
        DVec2::new(self.dvdx[idx], self.dvdy[idx])
    }

    /// 计算应变率张量的模
    /// |S| = √(2 * S_ij * S_ij)
    #[inline]
    pub fn strain_rate_magnitude(&self, idx: usize) -> f64 {
        let s_xx = self.dudx[idx];
        let s_yy = self.dvdy[idx];
        let s_xy = 0.5 * (self.dudy[idx] + self.dvdx[idx]);
        let s_mag_sq = s_xx.powi(2) + s_yy.powi(2) + 2.0 * s_xy.powi(2);
        (2.0 * s_mag_sq).sqrt()
    }

    /// 计算涡量 ω = ∂v/∂x - ∂u/∂y
    #[inline]
    pub fn vorticity(&self, idx: usize) -> f64 {
        self.dvdx[idx] - self.dudy[idx]
    }

    pub fn len(&self) -> usize {
        self.dudx.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dudx.is_empty()
    }
}

/// Green-Gauss 标量梯度重构
///
/// ∇φ ≈ (1/V) ∮ φ_f * n_f dA
pub fn compute_gradient_green_gauss(field: &[f64], mesh: &UnstructuredMesh) -> Vec<DVec2> {
    debug_assert_eq!(field.len(), mesh.n_cells);

    let n_cells = mesh.n_cells;
    let mut grad = vec![DVec2::ZERO; n_cells];

    for face_idx in 0..mesh.n_faces {
        let owner = mesh.face_owner[face_idx];
        let neighbor = mesh.face_neighbor[face_idx];

        let normal = mesh.face_normal[face_idx];
        let length = mesh.face_length[face_idx];
        let ds = normal * length;

        let phi_face = if neighbor != usize::MAX {
            0.5 * (field[owner] + field[neighbor])
        } else {
            field[owner]
        };

        grad[owner] += ds * phi_face;

        if neighbor != usize::MAX {
            grad[neighbor] -= ds * phi_face;
        }
    }

    for i in 0..n_cells {
        let area = mesh.cell_area[i];
        if area > 1e-14 {
            grad[i] /= area;
        } else {
            grad[i] = DVec2::ZERO;
        }
    }

    grad
}

/// Green-Gauss 标量梯度重构 (写入预分配结构)
pub fn compute_gradient_green_gauss_into(
    field: &[f64],
    mesh: &UnstructuredMesh,
    output: &mut ScalarGradient,
) {
    debug_assert_eq!(field.len(), mesh.n_cells);
    debug_assert_eq!(output.len(), mesh.n_cells);

    output.reset();

    let mut grad_x = std::mem::take(&mut output.grad_x);
    let mut grad_y = std::mem::take(&mut output.grad_y);

    for face_idx in 0..mesh.n_faces {
        let owner = mesh.face_owner[face_idx];
        let neighbor = mesh.face_neighbor[face_idx];

        let normal = mesh.face_normal[face_idx];
        let length = mesh.face_length[face_idx];

        let phi_face = if neighbor != usize::MAX {
            0.5 * (field[owner] + field[neighbor])
        } else {
            field[owner]
        };

        let flux_x = normal.x * length * phi_face;
        let flux_y = normal.y * length * phi_face;

        grad_x[owner] += flux_x;
        grad_y[owner] += flux_y;

        if neighbor != usize::MAX {
            grad_x[neighbor] -= flux_x;
            grad_y[neighbor] -= flux_y;
        }
    }

    for i in 0..mesh.n_cells {
        let area = mesh.cell_area[i];
        if area > 1e-14 {
            let inv_area = 1.0 / area;
            grad_x[i] *= inv_area;
            grad_y[i] *= inv_area;
        } else {
            grad_x[i] = 0.0;
            grad_y[i] = 0.0;
        }
    }

    output.grad_x = grad_x;
    output.grad_y = grad_y;
}

/// Green-Gauss 向量场梯度重构
pub fn compute_vector_gradient_green_gauss(
    velocities: &[DVec2],
    mesh: &UnstructuredMesh,
) -> VectorGradient {
    debug_assert_eq!(velocities.len(), mesh.n_cells);

    let n_cells = mesh.n_cells;
    let mut grad = VectorGradient::new(n_cells);

    for face_idx in 0..mesh.n_faces {
        let owner = mesh.face_owner[face_idx];
        let neighbor = mesh.face_neighbor[face_idx];

        let normal = mesh.face_normal[face_idx];
        let length = mesh.face_length[face_idx];

        let vel_face = if neighbor != usize::MAX {
            (velocities[owner] + velocities[neighbor]) * 0.5
        } else {
            velocities[owner]
        };

        let flux_ux = normal.x * length * vel_face.x;
        let flux_uy = normal.y * length * vel_face.x;
        let flux_vx = normal.x * length * vel_face.y;
        let flux_vy = normal.y * length * vel_face.y;

        grad.dudx[owner] += flux_ux;
        grad.dudy[owner] += flux_uy;
        grad.dvdx[owner] += flux_vx;
        grad.dvdy[owner] += flux_vy;

        if neighbor != usize::MAX {
            grad.dudx[neighbor] -= flux_ux;
            grad.dudy[neighbor] -= flux_uy;
            grad.dvdx[neighbor] -= flux_vx;
            grad.dvdy[neighbor] -= flux_vy;
        }
    }

    for i in 0..n_cells {
        let area = mesh.cell_area[i];
        if area > 1e-14 {
            let inv_area = 1.0 / area;
            grad.dudx[i] *= inv_area;
            grad.dudy[i] *= inv_area;
            grad.dvdx[i] *= inv_area;
            grad.dvdy[i] *= inv_area;
        } else {
            grad.dudx[i] = 0.0;
            grad.dudy[i] = 0.0;
            grad.dvdx[i] = 0.0;
            grad.dvdy[i] = 0.0;
        }
    }

    grad
}

/// 最小二乘梯度重构 (对非正交网格更精确)
pub fn compute_gradient_least_squares(field: &[f64], mesh: &UnstructuredMesh) -> Vec<DVec2> {
    debug_assert_eq!(field.len(), mesh.n_cells);

    let n_cells = mesh.n_cells;

    (0..n_cells)
        .into_par_iter()
        .map(|cell_idx| {
            let cell_faces = &mesh.cell_faces[cell_idx];
            let phi_c = field[cell_idx];
            let center_c = mesh.cell_center[cell_idx];

            let mut a11 = 0.0;
            let mut a12 = 0.0;
            let mut a22 = 0.0;
            let mut b1 = 0.0;
            let mut b2 = 0.0;

            for (local_idx, &face_id) in cell_faces.faces.iter().enumerate() {
                let face_idx = face_id.idx();
                let is_owner = cell_faces.is_owner(local_idx);

                let neighbor = if is_owner {
                    mesh.face_neighbor[face_idx]
                } else {
                    mesh.face_owner[face_idx]
                };

                if neighbor == usize::MAX {
                    continue;
                }

                let center_n = mesh.cell_center[neighbor];
                let r = center_n - center_c;
                let d_sq = r.length_squared();

                if d_sq < 1e-20 {
                    continue;
                }

                let w = 1.0 / d_sq.sqrt();
                let w_sq = w * w;

                let phi_n = field[neighbor];
                let dphi = phi_n - phi_c;

                a11 += w_sq * r.x * r.x;
                a12 += w_sq * r.x * r.y;
                a22 += w_sq * r.y * r.y;
                b1 += w_sq * r.x * dphi;
                b2 += w_sq * r.y * dphi;
            }

            let det = a11 * a22 - a12 * a12;

            if det.abs() < 1e-14 {
                return DVec2::ZERO;
            }

            let inv_det = 1.0 / det;
            DVec2::new(
                (a22 * b1 - a12 * b2) * inv_det,
                (-a12 * b1 + a11 * b2) * inv_det,
            )
        })
        .collect()
}

/// Barth-Jespersen 梯度限制器
///
/// 限制梯度以保证面重构值不超过邻居极值
pub fn limit_gradient_barth_jespersen(field: &[f64], grad: &mut [DVec2], mesh: &UnstructuredMesh) {
    debug_assert_eq!(field.len(), mesh.n_cells);
    debug_assert_eq!(grad.len(), mesh.n_cells);

    for cell_idx in 0..mesh.n_cells {
        let phi_c = field[cell_idx];
        let center_c = mesh.cell_center[cell_idx];
        let cell_faces = &mesh.cell_faces[cell_idx];

        let mut phi_min = phi_c;
        let mut phi_max = phi_c;

        for (local_idx, &face_id) in cell_faces.faces.iter().enumerate() {
            let face_idx = face_id.idx();
            let is_owner = cell_faces.is_owner(local_idx);

            let neighbor = if is_owner {
                mesh.face_neighbor[face_idx]
            } else {
                mesh.face_owner[face_idx]
            };

            if neighbor != usize::MAX {
                let phi_n = field[neighbor];
                phi_min = phi_min.min(phi_n);
                phi_max = phi_max.max(phi_n);
            }
        }

        let mut alpha = 1.0;

        for &face_id in &cell_faces.faces {
            let face_idx = face_id.idx();
            let face_center = mesh.face_center[face_idx];
            let r = face_center - center_c;

            let phi_face = phi_c + grad[cell_idx].dot(r);
            let delta = phi_face - phi_c;

            if delta.abs() < 1e-14 {
                continue;
            }

            let alpha_face = if delta > 0.0 {
                ((phi_max - phi_c) / delta).min(1.0)
            } else {
                ((phi_min - phi_c) / delta).min(1.0)
            };

            alpha = alpha.min(alpha_face.max(0.0));
        }

        grad[cell_idx] *= alpha;
    }
}

/// Venkatakrishnan 梯度限制器 (更光滑)
pub fn limit_gradient_venkatakrishnan(
    field: &[f64],
    grad: &mut [DVec2],
    mesh: &UnstructuredMesh,
    k: f64,
) {
    debug_assert_eq!(field.len(), mesh.n_cells);
    debug_assert_eq!(grad.len(), mesh.n_cells);

    for cell_idx in 0..mesh.n_cells {
        let phi_c = field[cell_idx];
        let center_c = mesh.cell_center[cell_idx];
        let cell_faces = &mesh.cell_faces[cell_idx];

        let mut phi_min = phi_c;
        let mut phi_max = phi_c;

        for (local_idx, &face_id) in cell_faces.faces.iter().enumerate() {
            let face_idx = face_id.idx();
            let is_owner = cell_faces.is_owner(local_idx);

            let neighbor = if is_owner {
                mesh.face_neighbor[face_idx]
            } else {
                mesh.face_owner[face_idx]
            };

            if neighbor != usize::MAX {
                let phi_n = field[neighbor];
                phi_min = phi_min.min(phi_n);
                phi_max = phi_max.max(phi_n);
            }
        }

        let eps_sq = (k * mesh.cell_area[cell_idx].sqrt()).powi(3);

        let mut alpha = 1.0;

        for &face_id in &cell_faces.faces {
            let face_idx = face_id.idx();
            let face_center = mesh.face_center[face_idx];
            let r = face_center - center_c;

            let delta = grad[cell_idx].dot(r);

            if delta.abs() < 1e-14 {
                continue;
            }

            let delta_max = if delta > 0.0 {
                phi_max - phi_c
            } else {
                phi_min - phi_c
            };

            let dm = delta_max;
            let dp = delta;

            let alpha_face = if dp.abs() < 1e-14 {
                1.0
            } else {
                let r_sq = dm / dp;
                let num = (r_sq * r_sq + 2.0 * r_sq) * dp * dp + eps_sq;
                let den = (r_sq * r_sq + r_sq + 2.0) * dp * dp + eps_sq;
                (num / den).min(1.0).max(0.0)
            };

            alpha = alpha.min(alpha_face);
        }

        grad[cell_idx] *= alpha;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_gradient_creation() {
        let grad = ScalarGradient::new(100);
        assert_eq!(grad.grad_x.len(), 100);
        assert_eq!(grad.grad_y.len(), 100);
    }

    #[test]
    fn test_scalar_gradient_reset() {
        let mut grad = ScalarGradient::new(10);
        grad.grad_x[0] = 1.0;
        grad.grad_y[0] = 2.0;
        grad.reset();
        assert_eq!(grad.grad_x[0], 0.0);
        assert_eq!(grad.grad_y[0], 0.0);
    }

    #[test]
    fn test_vector_gradient_creation() {
        let grad = VectorGradient::new(50);
        assert_eq!(grad.dudx.len(), 50);
        assert_eq!(grad.dvdy.len(), 50);
    }

    #[test]
    fn test_strain_rate_magnitude() {
        let mut grad = VectorGradient::new(1);
        grad.dudx[0] = 1.0;
        grad.dvdy[0] = -1.0;
        grad.dudy[0] = 0.0;
        grad.dvdx[0] = 0.0;

        let s_mag = grad.strain_rate_magnitude(0);
        let expected = (2.0 * (1.0 + 1.0)).sqrt();
        assert!((s_mag - expected).abs() < 1e-10);
    }

    #[test]
    fn test_vorticity() {
        let mut grad = VectorGradient::new(1);
        grad.dvdx[0] = 2.0;
        grad.dudy[0] = 1.0;

        let omega = grad.vorticity(0);
        assert!((omega - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_constant_field_zero_gradient() {
        let mut mesh = UnstructuredMesh::new();
        mesh.n_cells = 4;
        mesh.n_faces = 0;
        mesh.cell_area = vec![1.0; 4];
        mesh.cell_center = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(1.0, 0.0),
            DVec2::new(0.0, 1.0),
            DVec2::new(1.0, 1.0),
        ];
        mesh.cell_faces = vec![Default::default(); 4];
        mesh.face_owner = vec![];
        mesh.face_neighbor = vec![];
        mesh.face_normal = vec![];
        mesh.face_length = vec![];

        let field = vec![1.0; 4];
        let grad = compute_gradient_green_gauss(&field, &mesh);

        for g in grad {
            assert!(g.length() < 1e-10);
        }
    }
}
