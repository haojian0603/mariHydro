//! 垂向速度计算
//!
//! 从水平连续方程计算垂向速度 w：
//! ∂w/∂σ = -h × ∇·u - σ × (∂h/∂t + ∇·(h u))
//!
//! 简化形式（忽略时变项）：
//! w(σ) = -∫[σ,-1] h × ∇·u dσ

use super::sigma::SigmaCoordinate;
use crate::state::ShallowWaterState;
use mh_runtime::{Backend, RuntimeScalar};

/// 垂向速度计算器
pub struct VerticalVelocity<B: Backend> {
    /// σ坐标
    sigma: SigmaCoordinate,
    /// 单元数量
    n_cells: usize,
    /// 垂向速度场 [m/s]（n_cells × n_layers+1）
    /// w[cell][k] = 层界面 k 的垂向速度
    w: Vec<B::Buffer<B::Scalar>>,
    /// 后端实例
    backend: B,
}

impl<B: Backend> VerticalVelocity<B> {
    /// 创建新的垂向速度计算器
    pub fn new_with_backend(backend: B, n_cells: usize, sigma: SigmaCoordinate) -> Self {
        let n_interfaces = sigma.n_layers() + 1;
        let w = (0..n_interfaces)
            .map(|_| backend.alloc(n_cells))
            .collect();
        
        Self {
            sigma,
            n_cells,
            w,
            backend,
        }
    }

    /// 从水平散度场计算垂向速度
    ///
    /// # 参数
    /// - `div_hu`: 水平动量散度场 ∇·(hu) [m/s]
    /// - `state`: 浅水状态
    ///
    /// # 算法
    /// 从底部积分：w(σ=0) = 0, w(k) = w(k+1) - ∫ h×∇·u dσ
    pub fn compute_from_divergence(
        &mut self,
        div_hu: &B::Buffer<B::Scalar>,
        state: &ShallowWaterState<B>,
    ) {
        let n_layers = self.sigma.n_layers();
        let eps = self
            .backend
            .config_scalar(1e-6, "VerticalVelocity.compute_from_divergence.eps");
        let zero = <B::Scalar as RuntimeScalar>::ZERO;

        for cell in 0..self.n_cells.min(div_hu.len()) {
            let h = state.h[cell];
            if h < eps {
                // 干单元：w = 0
                for k in 0..=n_layers {
                    self.w[k][cell] = zero;
                }
                continue;
            }

            // 底部边界条件：w = 0 (无穿透)
            self.w[n_layers][cell] = zero;

            // 从底部向上积分
            // w(k) = w(k+1) - div_hu * Δσ
            for k in (0..n_layers).rev() {
                let d_sigma = self.backend.config_scalar(
                    self.sigma.layer_thickness_sigma(k),
                    "VerticalVelocity.compute_from_divergence.layer_thickness",
                );
                self.w[k][cell] = self.w[k + 1][cell] - div_hu[cell] * d_sigma;
            }

            // 水面边界条件检验（应该接近 ∂η/∂t）
            // 这里简化为 w(σ=0) 自由演化
        }
    }

    /// 从分层水平速度计算垂向速度
    ///
    /// 更精确的版本，使用每层的实际速度
    pub fn compute_from_layered_velocity(
        &mut self,
        u_layers: &[&B::Buffer<B::Scalar>],
        v_layers: &[&B::Buffer<B::Scalar>],
        du_dx: &[&B::Buffer<B::Scalar>],
        dv_dy: &[&B::Buffer<B::Scalar>],
        state: &ShallowWaterState<B>,
    ) {
        let n_layers = self.sigma.n_layers();
        let eps = self
            .backend
            .config_scalar(1e-6, "VerticalVelocity.compute_from_layered_velocity.eps");
        let zero = <B::Scalar as RuntimeScalar>::ZERO;

        for cell in 0..self.n_cells {
            let h = state.h[cell];
            if h < eps {
                for k in 0..=n_layers {
                    self.w[k][cell] = zero;
                }
                continue;
            }

            // 底部边界：w = 0
            self.w[n_layers][cell] = zero;

            // 逐层积分
            for k in (0..n_layers).rev() {
                let d_sigma = self.backend.config_scalar(
                    self.sigma.layer_thickness_sigma(k),
                    "VerticalVelocity.compute_from_layered_velocity.layer_thickness",
                );
                let layer_h = d_sigma * h;

                // 水平散度（若提供）
                let div_uv = if k < du_dx.len() && k < dv_dy.len() {
                    du_dx[k][cell] + dv_dy[k][cell]
                } else {
                    zero
                };

                // 连续方程：∂w/∂z = -∇·u → w(z_top) = w(z_bot) - ∇·u × Δz
                self.w[k][cell] = self.w[k + 1][cell] - div_uv * layer_h;
            }
        }

        // 抑制未使用变量警告
        let _ = (u_layers, v_layers);
    }

    /// 获取特定层界面的垂向速度
    pub fn w_at_interface(&self, k: usize) -> &B::Buffer<B::Scalar> {
        &self.w[k]
    }

    /// 获取特定单元、层界面的垂向速度
    pub fn get(&self, cell: usize, k: usize) -> B::Scalar {
        self.w[k][cell]
    }

    /// 层数
    pub fn n_layers(&self) -> usize {
        self.sigma.n_layers()
    }

    /// 单元数
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// σ坐标引用
    pub fn sigma(&self) -> &SigmaCoordinate {
        &self.sigma
    }

    /// 获取后端引用
    pub fn backend(&self) -> &B {
        &self.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::{Backend, CpuBackend, DeviceBuffer};

    fn create_test_state(
        backend: CpuBackend<f64>,
        n_cells: usize,
        h: f64,
    ) -> ShallowWaterState<CpuBackend<f64>> {
        let mut state = ShallowWaterState::new_with_backend(backend.clone(), n_cells);
        let h_val = backend.config_scalar(h, "VerticalVelocity.tests.create_test_state.h");
        for i in 0..n_cells {
            state.h[i] = h_val;
        }
        state
    }

    #[test]
    fn test_vertical_velocity_creation() {
        let sigma = SigmaCoordinate::uniform(5);
        let backend = CpuBackend::<f64>::new();
        let vv = VerticalVelocity::new_with_backend(backend, 10, sigma);

        assert_eq!(vv.n_cells(), 10);
        assert_eq!(vv.n_layers(), 5);
    }

    #[test]
    fn test_zero_divergence() {
        let sigma = SigmaCoordinate::uniform(5);
        let backend = CpuBackend::<f64>::new();
        let mut vv = VerticalVelocity::new_with_backend(backend.clone(), 10, sigma);
        let state = create_test_state(backend.clone(), 10, 2.0);
        let mut div_hu = backend.alloc(10);
        div_hu.fill(backend.config_scalar(
            0.0,
            "VerticalVelocity.tests.zero_divergence.value",
        ));

        vv.compute_from_divergence(&div_hu, &state);

        // 零散度 → 零垂向速度
        for k in 0..=5 {
            for cell in 0..10 {
                assert!((vv.get(cell, k)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_constant_divergence() {
        let sigma = SigmaCoordinate::uniform(5);
        let backend = CpuBackend::<f64>::new();
        let mut vv = VerticalVelocity::new_with_backend(backend.clone(), 10, sigma);
        let state = create_test_state(backend.clone(), 10, 2.0);
        let mut div_hu = backend.alloc(10);
        div_hu.fill(backend.config_scalar(
            0.1,
            "VerticalVelocity.tests.constant_divergence.value",
        ));

        vv.compute_from_divergence(&div_hu, &state);

        // 底部 w = 0
        for cell in 0..10 {
            assert!((vv.get(cell, 5)).abs() < 1e-10);
        }

        // 水面 w 应该为负（水面下降）
        for cell in 0..10 {
            assert!(vv.get(cell, 0) < 0.0);
        }
    }

    #[test]
    fn test_dry_cell() {
        let sigma = SigmaCoordinate::uniform(5);
        let backend = CpuBackend::<f64>::new();
        let mut vv = VerticalVelocity::new_with_backend(backend.clone(), 10, sigma);
        let state = create_test_state(backend.clone(), 10, 1e-8); // 干单元
        let mut div_hu = backend.alloc(10);
        div_hu.fill(backend.config_scalar(
            1.0,
            "VerticalVelocity.tests.dry_cell.value",
        ));

        vv.compute_from_divergence(&div_hu, &state);

        // 干单元 w = 0
        for k in 0..=5 {
            for cell in 0..10 {
                assert!((vv.get(cell, k)).abs() < 1e-10);
            }
        }
    }
}
