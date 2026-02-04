//! marihydro\crates\mh_physics\src\sediment\transport_2_5d.rs
//! 2.5D 泥沙输运
//!
//! 基于垂向剖面恢复的泥沙输运计算。
//! 
//! 这是一个独立的扩展模块，不侵入2D核心。

use crate::core::Backend;
use crate::state::ShallowWaterState;
use crate::vertical::profile::{ProfileMethod, ProfileRestorer, VerticalProfile};
use mh_runtime::RuntimeScalar;
use num_traits::Float;

/// 2.5D 泥沙输运求解器
pub struct Transport2_5D<B: Backend> {
    /// 剖面恢复器
    #[allow(dead_code)]
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
    /// 配置
    config: Transport2_5DConfig<B::Scalar>,
    /// 后端
    backend: B,
}

/// 垂向浓度剖面类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcentrationProfileMethod {
    /// 均匀分布
    Uniform,
    /// 指数剖面（由沉降/扩散平衡推导）
    Exponential,
}

/// 2.5D 输运配置
#[derive(Debug, Clone, Copy)]
pub struct Transport2_5DConfig<S: RuntimeScalar> {
    /// 是否进行垂向剖面恢复
    pub recover_profile: bool,
    /// 剖面方法
    pub profile_method: ConcentrationProfileMethod,
    /// 最小水深阈值
    pub min_depth: S,
    /// 最小扩散系数
    pub min_diffusivity: S,
}

impl<S: RuntimeScalar> Transport2_5DConfig<S> {
    pub fn with_backend_defaults<B: Backend<Scalar = S>>(backend: &B) -> Self {
        Self {
            recover_profile: true,
            profile_method: ConcentrationProfileMethod::Exponential,
            min_depth: backend.scalar_from_f64(1e-6),
            min_diffusivity: backend.scalar_from_f64(1e-6),
        }
    }
}

impl<B> Transport2_5D<B>
where
    B: Backend + Clone,
    B::Scalar: RuntimeScalar + Float,
{
    /// 创建求解器（显式指定后端）
    pub fn new_with_backend(
        backend: B,
        n_cells: usize,
        n_layers: usize,
        settling_velocity: B::Scalar,
        diffusion_coeff: B::Scalar,
    ) -> Self {
        let config = Transport2_5DConfig::with_backend_defaults(&backend);
        Self::new_with_config(
            backend,
            n_cells,
            n_layers,
            settling_velocity,
            diffusion_coeff,
            config,
        )
    }

    /// 使用自定义配置创建
    pub fn new_with_config(
        backend: B,
        n_cells: usize,
        n_layers: usize,
        settling_velocity: B::Scalar,
        diffusion_coeff: B::Scalar,
        config: Transport2_5DConfig<B::Scalar>,
    ) -> Self {
        assert!(n_cells > 0 && n_layers > 0, "Transport2_5D: n_cells/n_layers 必须为正");
        assert!(settling_velocity.is_finite(), "Transport2_5D: settling_velocity 必须为有限值");
        assert!(diffusion_coeff.is_finite() && diffusion_coeff >= B::Scalar::ZERO, "Transport2_5D: diffusion_coeff 必须为非负有限值");
        let total = n_cells * n_layers;

        Self {
            profile_restorer: ProfileRestorer::new_with_backend(backend.clone(), n_cells, n_layers, ProfileMethod::Logarithmic),
            profile: VerticalProfile::new_with_backend(backend.clone(), n_cells, n_layers),
            concentration: backend.alloc_init(total, B::Scalar::ZERO),
            settling_velocity,
            diffusion_coeff,
            n_layers,
            config,
            backend,
        }
    }

    /// 获取配置
    pub fn config(&self) -> &Transport2_5DConfig<B::Scalar> {
        &self.config
    }

    /// 设置配置
    pub fn set_config(&mut self, config: Transport2_5DConfig<B::Scalar>) {
        self.config = config;
    }

    /// 执行一步输运计算
    pub fn step(
        &mut self,
        state: &ShallowWaterState<B>,
        dt: B::Scalar,
    ) {
        if !dt.is_finite() || dt <= B::Scalar::ZERO {
            return;
        }
        // 1. 恢复垂向剖面
        if self.config.recover_profile {
            self.recover_concentration_profile(state);
        }

        // 2. 垂向输运（扩散+沉降）
        self.compute_vertical_transport(state, dt);
    }

    /// 恢复垂向浓度剖面（基于深度平均浓度）
    fn recover_concentration_profile(&mut self, state: &ShallowWaterState<B>) {
        let h: &B::Buffer<B::Scalar> = &state.h;
        let n_cells = state.n_cells();
        let n_layers = self.n_layers;
        let ws = self.settling_velocity.abs();
        let min_depth = self.config.min_depth;
        let min_diff = self.config.min_diffusivity;
        let eps = self.backend.scalar_from_f64(1e-12);

        for cell in 0..n_cells {
            let h_cell = h[cell];
            if h_cell <= min_depth {
                for k in 0..n_layers {
                    let idx = cell * n_layers + k;
                    self.concentration[idx] = B::Scalar::ZERO;
                }
                continue;
            }

            let c_avg = self.depth_averaged_concentration(cell);
            if c_avg <= B::Scalar::ZERO {
                for k in 0..n_layers {
                    let idx = cell * n_layers + k;
                    self.concentration[idx] = B::Scalar::ZERO;
                }
                continue;
            }

            let dz = h_cell / self.backend.scalar_from_f64(n_layers as f64);
            let kv = self.diffusion_coeff.max(min_diff);

            let mut weights = vec![B::Scalar::ZERO; n_layers];
            let mut sum_w = B::Scalar::ZERO;

            for k in 0..n_layers {
                let z = dz * (self.backend.scalar_from_f64(k as f64 + 0.5));
                let w = match self.config.profile_method {
                    ConcentrationProfileMethod::Uniform => B::Scalar::ONE,
                    ConcentrationProfileMethod::Exponential => {
                        if kv <= eps {
                            B::Scalar::ONE
                        } else {
                            let exponent = -(ws * z).safe_div(kv, B::Scalar::ZERO);
                            exponent.exp()
                        }
                    }
                };
                weights[k] = w;
                sum_w = sum_w + w;
            }

            let scale = if sum_w > eps {
                c_avg * self.backend.scalar_from_f64(n_layers as f64) / sum_w
            } else {
                c_avg
            };

            for k in 0..n_layers {
                let idx = cell * n_layers + k;
                let value = weights[k] * scale;
                self.concentration[idx] = if value > B::Scalar::ZERO { value } else { B::Scalar::ZERO };
            }
        }
    }

    /// 垂向输运（扩散+沉降）
    fn compute_vertical_transport(
        &mut self,
        state: &ShallowWaterState<B>,
        dt: B::Scalar,
    ) {
        let h: &B::Buffer<B::Scalar> = &state.h;
        let n_cells = state.n_cells();
        let n_layers = self.n_layers;

        let ws = self.settling_velocity;
        let kv = self.diffusion_coeff.max(self.config.min_diffusivity);
        let min_depth = self.config.min_depth;

        for cell in 0..n_cells {
            let h_cell = h[cell];

            if h_cell <= min_depth {
                continue;
            }

            let dz = h_cell / self.backend.scalar_from_f64(n_layers as f64);

            let mut flux = vec![B::Scalar::ZERO; n_layers + 1];

            // 内部界面通量（k=1..n_layers-1）
            for k in 1..n_layers {
                let idx_up = cell * n_layers + (k - 1);
                let idx_dn = cell * n_layers + k;
                let c_up = self.concentration[idx_up];
                let c_dn = self.concentration[idx_dn];

                let diff = kv * (c_dn - c_up) / dz;
                let c_face = if ws >= B::Scalar::ZERO { c_up } else { c_dn };
                flux[k] = -diff - ws * c_face;
            }

            // 底部沉降通量
            flux[0] = -ws * self.concentration[cell * n_layers];
            // 顶部零通量
            flux[n_layers] = B::Scalar::ZERO;

            for k in 0..n_layers {
                let idx = cell * n_layers + k;
                let tendency = (flux[k] - flux[k + 1]) / dz;
                self.concentration[idx] = self.concentration[idx] + dt * tendency;
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

        sum / self.backend.scalar_from_f64(n_layers as f64)
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
