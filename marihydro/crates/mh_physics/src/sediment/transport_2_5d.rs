//! marihydro\crates\mh_physics\src\sediment\transport_2_5d.rs
//! 2.5D 泥沙输运
//!
//! 基于垂向剖面恢复的泥沙输运计算。
//! 
//! 这是一个独立的扩展模块，不侵入2D核心。

use crate::core::Backend;
use crate::state::ShallowWaterStateGeneric;
use crate::vertical::profile::{ProfileMethod, ProfileRestorer, VerticalProfile};
use mh_runtime::RuntimeScalar;
use num_traits::FromPrimitive;

/// 2.5D 泥沙输运求解器
pub struct Transport2_5D<B: Backend> {
    /// 剖面恢复器
    profile_restorer: ProfileRestorer<B>,
    /// 垂向剖面缓存
    profile: VerticalProfile<B>,
    /// 泥沙浓度 [n_cells * n_layers]
    concentration: B::Buffer<B::Scalar>,
    /// 沉降速度
    settling_velocity: B::Scalar,
    /// 扩散系数
    diffusion_coeff: B::Scalar,
    /// 层数
    n_layers: usize,
    /// 后端
    _backend: B,
}

impl<B> Transport2_5D<B>
where
    B: Backend + Clone,
    B::Scalar: RuntimeScalar + FromPrimitive,
{
    /// 创建求解器（显式指定后端）
    pub fn new_with_backend(
        backend: B,
        n_cells: usize,
        n_layers: usize,
        settling_velocity: B::Scalar,
        diffusion_coeff: B::Scalar,
    ) -> Self {
        let total = n_cells * n_layers;

        Self {
            profile_restorer: ProfileRestorer::new_with_backend(backend.clone(), n_cells, n_layers, ProfileMethod::Logarithmic),
            profile: VerticalProfile::new_with_backend(backend.clone(), n_cells, n_layers),
            concentration: backend.alloc_init(total, B::Scalar::ZERO),
            settling_velocity,
            diffusion_coeff,
            n_layers,
            _backend: backend,
        }
    }

    /// 执行一步输运计算
    pub fn step(
        &mut self,
        state: &ShallowWaterStateGeneric<B>,
        dt: B::Scalar,
    ) {
        // 1. 恢复垂向剖面（占位，可替换为具体恢复逻辑）
        let _ = (&self.profile_restorer, &self.profile);

        // 2. 垂向输运（扩散+沉降）
        self.compute_vertical_transport(state, dt);
    }

    /// 垂向输运（扩散+沉降）
    fn compute_vertical_transport(
        &mut self,
        state: &ShallowWaterStateGeneric<B>,
        dt: B::Scalar,
    ) {
        let h: &B::Buffer<B::Scalar> = &state.h;
        let n_cells = state.n_cells();
        let n_layers = self.n_layers;

        let ws = self.settling_velocity;
        let kv = self.diffusion_coeff;

        for cell in 0..n_cells {
            let h_cell = h[cell];

            if h_cell <= B::Scalar::from_f64(1e-6).unwrap_or(B::Scalar::ZERO) {
                continue;
            }

            let dz = h_cell / B::Scalar::from_f64(n_layers as f64).unwrap_or(B::Scalar::ONE);

            // 显式欧拉
            for k in 1..n_layers {
                let idx = cell * n_layers + k;
                let idx_above = idx - 1;

                // 扩散项
                let c_k = self.concentration[idx];
                let c_above = self.concentration[idx_above];
                let diffusion = kv * (c_above - c_k) / (dz * dz);

                // 沉降项
                let settling = -ws * c_k / dz;

                self.concentration[idx] += dt * (diffusion + settling);

                // 保证非负
                if self.concentration[idx] < B::Scalar::ZERO {
                    self.concentration[idx] = B::Scalar::ZERO;
                }
            }
        }
    }

    /// 获取深度平均浓度
    pub fn depth_averaged_concentration(&self, cell: usize) -> B::Scalar {
        let n_layers = self.n_layers;
        let mut sum = B::Scalar::ZERO;

        for k in 0..n_layers {
            let idx = cell * n_layers + k;
            sum += self.concentration[idx];
        }

        sum / B::Scalar::from_f64(n_layers as f64).unwrap_or(B::Scalar::ONE)
    }

    /// 设置浓度
    pub fn set_concentration(&mut self, cell: usize, layer: usize, value: B::Scalar) {
        let idx = cell * self.n_layers + layer;
        self.concentration[idx] = value;
    }

    /// 获取浓度
    pub fn get_concentration(&self, cell: usize, layer: usize) -> B::Scalar {
        let idx = cell * self.n_layers + layer;
        self.concentration[idx]
    }

    /// 获取垂向剖面引用
    pub fn profile(&self) -> &VerticalProfile<B> {
        &self.profile
    }
}
