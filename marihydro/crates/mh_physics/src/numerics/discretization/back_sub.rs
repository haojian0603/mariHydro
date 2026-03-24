// crates/mh_physics/src/numerics/discretization/back_sub.rs

//! 回代更新模块
//!
//! 在隐式求解后，需要将校正量回代到物理量：
//! - 水深校正：h = h* + η'
//! - 速度校正：u = u* - dt * g * ∇η'
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::numerics::discretization::{DepthCorrector, VelocityCorrector};
//!
//! // 求解压力校正方程后
//! let eta_prime = solver.solve(&matrix, &rhs, &mut x)?;
//!
//! // 校正水深
//! let mut depth_corrector = DepthCorrector::new(n_cells);
//! depth_corrector.correct(&mut state.h, &eta_prime);
//!
//! // 校正速度
//! let mut vel_corrector = VelocityCorrector::new(&topo);
//! vel_corrector.correct(&mut state.u, &mut state.v, &eta_prime, &mesh, dt, g);
//! ```

use crate::adapter::PhysicsMesh;
use super::topology::CellFaceTopology;
use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar};
use num_traits::Float;

/// 水深校正器
///
/// 根据压力校正量更新水深：h = h* + η'
pub struct DepthCorrector<B: Backend> {
    /// 单元数量
    n_cells: usize,
    /// 最小水深
    h_min: B::Scalar,
    /// 是否保证非负
    ensure_positive: bool,
    /// 后端实例
    backend: B,
}

impl<B> DepthCorrector<B>
where
    B: Backend,
    B::Scalar: RuntimeScalar,
{
    /// 创建水深校正器
    pub fn new(n_cells: usize, backend: B) -> Self {
        Self {
            n_cells,
            h_min: B::Scalar::ZERO,
            ensure_positive: true,
            backend,
        }
    }

    /// 设置最小水深
    pub fn with_h_min(mut self, h_min: B::Scalar) -> Self {
        self.h_min = h_min;
        self
    }

    /// 禁用非负约束
    pub fn allow_negative(mut self) -> Self {
        self.ensure_positive = false;
        self
    }

    /// 校正水深
    ///
    /// # 参数
    ///
    /// - `h`: 水深场（将被修改）
    /// - `eta_prime`: 水位校正量
    pub fn correct(&self, h: &mut B::Buffer<B::Scalar>, eta_prime: &B::Buffer<B::Scalar>) {
        debug_assert_eq!(h.len(), self.n_cells);
        debug_assert_eq!(eta_prime.len(), self.n_cells);

        for i in 0..self.n_cells {
            h[i] += eta_prime[i];
            if self.ensure_positive && h[i] < self.h_min {
                h[i] = self.h_min;
            }
        }
    }

    /// 校正水深并返回统计
    pub fn correct_with_stats(
        &self,
        h: &mut B::Buffer<B::Scalar>,
        eta_prime: &B::Buffer<B::Scalar>,
    ) -> CorrectionStats<B::Scalar> {
        debug_assert_eq!(h.len(), self.n_cells);
        debug_assert_eq!(eta_prime.len(), self.n_cells);

        let mut stats: CorrectionStats<B::Scalar> = CorrectionStats::default();

        for i in 0..self.n_cells {
            let correction = eta_prime[i];
            stats.max_correction = stats.max_correction.max(correction.abs());
            stats.sum_correction += correction;

            h[i] += correction;

            if self.ensure_positive && h[i] < self.h_min {
                h[i] = self.h_min;
                stats.clipped_count += 1;
            }
        }

        stats.mean_correction = stats.sum_correction
            / self.backend.config_scalar(self.n_cells as f64, "back_sub.n_cells");
        stats
    }

    /// 切片版本，兼容旧路径
    pub fn correct_slice(&self, h: &mut [B::Scalar], eta_prime: &[B::Scalar]) {
        debug_assert_eq!(h.len(), self.n_cells);
        debug_assert_eq!(eta_prime.len(), self.n_cells);

        for i in 0..self.n_cells {
            h[i] += eta_prime[i];
            if self.ensure_positive && h[i] < self.h_min {
                h[i] = self.h_min;
            }
        }
    }
}

/// 速度校正器
///
/// 根据压力梯度校正速度：
/// $$\vec{u}^{n+1} = \vec{u}^* - \Delta t \cdot g \cdot \nabla \eta'$$
pub struct VelocityCorrector<B: Backend> {
    /// 单元数量
    n_cells: usize,
    /// 梯度 x 分量
    grad_x: B::Buffer<B::Scalar>,
    /// 梯度 y 分量
    grad_y: B::Buffer<B::Scalar>,
    /// 最小水深
    h_min: B::Scalar,
    /// 后端实例
    backend: B,
}

impl<B> VelocityCorrector<B>
where
    B: Backend,
    B::Scalar: RuntimeScalar,
{
    /// 创建速度校正器
    pub fn new(topo: &CellFaceTopology, backend: B) -> Self {
        let n_cells = topo.n_cells();
        Self {
            n_cells,
            grad_x: backend.alloc(n_cells),
            grad_y: backend.alloc(n_cells),
            h_min: backend.config_scalar(1e-4, "back_sub.h_min"),
            backend,
        }
    }

    /// 设置最小水深
    pub fn with_h_min(mut self, h_min: B::Scalar) -> Self {
        self.h_min = h_min;
        self
    }

    /// 计算梯度
    fn compute_gradient(
        &mut self,
        topo: &CellFaceTopology,
        mesh: &PhysicsMesh,
        eta_prime: &[B::Scalar],
    ) {
        // 清零
        self.grad_x.fill(B::Scalar::ZERO);
        self.grad_y.fill(B::Scalar::ZERO);

        // 使用 Green-Gauss 方法计算梯度
        for &face_idx in topo.interior_faces() {
            let face = topo.face(face_idx);
            let owner = face.owner;
            let neighbor = face.neighbor.expect("interior face");

            // 面值（算术平均）
            let eta_f = B::Scalar::HALF * (eta_prime[owner] + eta_prime[neighbor]);

            // 面通量
            let flux_x = eta_f
                * self.backend.config_scalar(face.normal.0, "back_sub.normal.x")
                * self.backend.config_scalar(face.length, "back_sub.face.length");
            let flux_y = eta_f
                * self.backend.config_scalar(face.normal.1, "back_sub.normal.y")
                * self.backend.config_scalar(face.length, "back_sub.face.length");

            // 累加到梯度
            let area_o = self.backend.config_scalar(mesh.cell_area_unchecked(mh_runtime::CellIndex(owner)), "back_sub.area_owner");
            let area_n = self.backend.config_scalar(mesh.cell_area_unchecked(mh_runtime::CellIndex(neighbor)), "back_sub.area_neighbor");

            self.grad_x[owner] += flux_x / area_o;
            self.grad_y[owner] += flux_y / area_o;

            self.grad_x[neighbor] -= flux_x / area_n;
            self.grad_y[neighbor] -= flux_y / area_n;
        }

        // 边界面贡献（假设零梯度）
        for &face_idx in topo.boundary_faces() {
            let face = topo.face(face_idx);
            let owner = face.owner;

            let eta_f = eta_prime[owner]; // 零梯度外推

            let flux_x = eta_f
                * self.backend.config_scalar(face.normal.0, "back_sub.boundary_normal.x")
                * self.backend.config_scalar(face.length, "back_sub.boundary_face.length");
            let flux_y = eta_f
                * self.backend.config_scalar(face.normal.1, "back_sub.boundary_normal.y")
                * self.backend.config_scalar(face.length, "back_sub.boundary_face.length");

            let area_o = self.backend.config_scalar(mesh.cell_area_unchecked(mh_runtime::CellIndex(owner)), "back_sub.boundary_area_owner");

            self.grad_x[owner] += flux_x / area_o;
            self.grad_y[owner] += flux_y / area_o;
        }
    }

    /// 校正速度
    ///
    /// # 参数
    ///
    /// - `u`: x 方向速度（将被修改）
    /// - `v`: y 方向速度（将被修改）
    /// - `h`: 水深（用于干湿判断）
    /// - `eta_prime`: 水位校正量
    /// - `topo`: 拓扑信息
    /// - `mesh`: 物理网格
    /// - `dt`: 时间步长
    /// - `g`: 重力加速度
    pub fn correct(
        &mut self,
        u: &mut B::Buffer<B::Scalar>,
        v: &mut B::Buffer<B::Scalar>,
        h: &B::Buffer<B::Scalar>,
        eta_prime: &B::Buffer<B::Scalar>,
        topo: &CellFaceTopology,
        mesh: &PhysicsMesh,
        dt: B::Scalar,
        g: B::Scalar,
    ) {
        // 计算 η' 的梯度
        self.compute_gradient(topo, mesh, eta_prime.as_slice());

        // 校正速度
        let coef = -dt * g;
        for i in 0..self.n_cells {
            if h[i] > self.h_min {
                u[i] += coef * self.grad_x[i];
                v[i] += coef * self.grad_y[i];
            } else {
                // 干单元：速度设为零
                u[i] = B::Scalar::ZERO;
                v[i] = B::Scalar::ZERO;
            }
        }
    }

    /// 获取梯度 x 分量
    pub fn gradient_x(&self) -> &[B::Scalar] {
        self.grad_x.as_slice()
    }

    /// 获取梯度 y 分量
    pub fn gradient_y(&self) -> &[B::Scalar] {
        self.grad_y.as_slice()
    }
}

/// 校正统计
#[derive(Debug, Clone, Default)]
pub struct CorrectionStats<S: RuntimeScalar> {
    /// 最大校正量绝对值
    pub max_correction: S,
    /// 平均校正量
    pub mean_correction: S,
    /// 校正量之和
    pub sum_correction: S,
    /// 被裁剪的单元数
    pub clipped_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_depth_corrector() {
        let backend = CpuBackend::<f64>::new();
        let corrector = DepthCorrector::new(3, backend);

        let mut h = vec![1.0, 2.0, 0.5];
        let eta_prime = vec![0.1, -0.3, -0.6];

        corrector.correct_slice(&mut h, &eta_prime);

        assert!((h[0] - 1.1).abs() < 1e-14);
        assert!((h[1] - 1.7).abs() < 1e-14);
        assert!(h[2] >= 0.0); // 非负约束
    }

    #[test]
    fn test_depth_corrector_with_stats() {
        let backend = CpuBackend::<f64>::new();
        let corrector = DepthCorrector::new(3, backend.clone());

        let mut h = backend.alloc(3);
        h.copy_from_slice(&[1.0, 2.0, 0.1]);
        let mut eta_prime = backend.alloc(3);
        eta_prime.copy_from_slice(&[0.1, -0.1, -0.2]);

        let stats = corrector.correct_with_stats(&mut h, &eta_prime);

        assert!((stats.max_correction - 0.2).abs() < 1e-14);
        assert_eq!(stats.clipped_count, 1);
    }

    #[test]
    fn test_depth_corrector_allow_negative() {
        let backend = CpuBackend::<f64>::new();
        let corrector = DepthCorrector::new(2, backend).allow_negative();

        let mut h = vec![0.5, 0.3];
        let eta_prime = vec![-0.6, -0.1];

        corrector.correct_slice(&mut h, &eta_prime);

        assert!((h[0] - (-0.1)).abs() < 1e-14); // 允许负值
        assert!((h[1] - 0.2).abs() < 1e-14);
    }
}
