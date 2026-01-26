//! 垂向剖面恢复器
//!
//! 从2D深度平均状态恢复垂向速度剖面。

use crate::core::{Backend, DeviceBuffer, CpuBackend};
use mh_runtime::RuntimeScalar as Scalar;
use crate::state::ShallowWaterStateGeneric;
use crate::vertical::sigma::SigmaCoordinate;

/// 垂向剖面
#[derive(Debug, Clone)]
pub struct VerticalProfile<B: Backend> {
    /// 单元数量
    n_cells: usize,
    /// 层数
    n_layers: usize,
    /// 各层 u 速度 [n_cells * n_layers]
    pub u_layers: B::Buffer<B::Scalar>,
    /// 各层 v 速度 [n_cells * n_layers]
    pub v_layers: B::Buffer<B::Scalar>,
    /// 各层高度 [n_cells * n_layers]
    pub z_layers: B::Buffer<B::Scalar>,
    /// 后端实例
    backend: B,
}

impl<B: Backend> VerticalProfile<B> {
    /// 使用后端创建垂向剖面
    pub fn new_with_backend(backend: B, n_cells: usize, n_layers: usize) -> Self {
        let total = n_cells * n_layers;
        Self {
            n_cells,
            n_layers,
            u_layers: backend.alloc(total),
            v_layers: backend.alloc(total),
            z_layers: backend.alloc(total),
            backend,
        }
    }
    
    /// 获取索引
    #[inline]
    pub fn index(&self, cell: usize, layer: usize) -> usize {
        cell * self.n_layers + layer
    }
    
    /// 单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
    
    /// 层数
    #[inline]
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
    
    /// 获取后端引用
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
}

/// CPU f64 后端的便捷方法
impl VerticalProfile<CpuBackend<f64>> {
    /// 使用默认后端创建
    pub fn new(n_cells: usize, n_layers: usize) -> Self {
        Self::new_with_backend(CpuBackend::<f64>::new(), n_cells, n_layers)
    }
}

/// 剖面恢复方法
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileMethod {
    /// 对数律剖面
    Logarithmic,
    /// 抛物线剖面
    Parabolic,
    /// 均匀剖面
    Uniform,
}

/// 浓度剖面恢复方法
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcentrationProfileMethod {
    /// 均匀分布
    Uniform,
    /// 指数剖面（沉降-扩散平衡）
    Exponential,
}

/// 垂向浓度剖面恢复工具
pub struct ConcentrationProfile;

impl ConcentrationProfile {
    /// 恢复垂向浓度剖面（返回每层中心浓度）
    pub fn recover<S: Scalar>(
        c_avg: S,
        h: S,
        n_layers: usize,
        settling_velocity: S,
        diffusivity: S,
        method: ConcentrationProfileMethod,
    ) -> Vec<S> {
        if n_layers == 0 {
            return Vec::new();
        }
        if h <= S::ZERO || c_avg <= S::ZERO {
            return vec![S::ZERO; n_layers];
        }

        let dz = h / S::from_usize(n_layers).unwrap_or(S::ONE);
        let kv = diffusivity.max(S::from_f64(1e-12).unwrap_or(S::ZERO));
        let ws = settling_velocity.abs();

        let mut weights = vec![S::ZERO; n_layers];
        let mut sum_w = S::ZERO;

        for k in 0..n_layers {
            let z = dz * S::from_f64(k as f64 + 0.5).unwrap_or(S::ZERO);
            let w = match method {
                ConcentrationProfileMethod::Uniform => S::ONE,
                ConcentrationProfileMethod::Exponential => {
                    let exponent = -(ws * z).safe_div(kv, S::ZERO);
                    exponent.exp()
                }
            };
            weights[k] = w;
            sum_w = sum_w + w;
        }

        let scale = if sum_w > S::ZERO {
            c_avg * S::from_usize(n_layers).unwrap_or(S::ONE) / sum_w
        } else {
            c_avg
        };

        for w in &mut weights {
            *w = (*w * scale).max(S::ZERO);
        }

        weights
    }
}

/// von Karman 常数
const VON_KARMAN: f64 = 0.41;

/// 垂向剖面恢复器
#[allow(dead_code)]
pub struct ProfileRestorer<B: Backend> {
    /// σ坐标
    sigma: SigmaCoordinate,
    /// 糙率 [n_cells]
    roughness: B::Buffer<B::Scalar>,
    /// 层数
    n_layers: usize,
    /// 恢复方法
    method: ProfileMethod,
    /// von Karman 常数
    von_karman: B::Scalar,
    /// 后端实例
    backend: B,
}

impl<B: Backend> ProfileRestorer<B> {
    /// 使用后端创建恢复器
    pub fn new_with_backend(backend: B, n_cells: usize, n_layers: usize, method: ProfileMethod) -> Self {
        Self {
            sigma: SigmaCoordinate::uniform(n_layers),
            roughness: backend.alloc_init(n_cells, <B::Scalar as Scalar>::from_config(0.01).unwrap_or(B::Scalar::ZERO)), // 默认糙率
            n_layers,
            method,
            von_karman: <B::Scalar as Scalar>::from_config(VON_KARMAN).unwrap_or(B::Scalar::ZERO),
            backend,
        }
    }
    
    /// 设置糙率（仅 CPU 后端有效）
    pub fn set_roughness(&mut self, cell: usize, z0: B::Scalar) {
        if let Some(slice) = self.roughness.try_as_slice_mut() {
            slice[cell] = z0;
        }
    }
    
    /// 恢复方法
    #[inline]
    pub fn method(&self) -> ProfileMethod {
        self.method
    }
    
    /// 层数
    #[inline]
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
    
    /// 获取后端引用
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    // 从2D状态恢复垂向剖面（通用后端，按层均匀分配）
    pub fn restore(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        output: &mut VerticalProfile<B>,
    ) {
        // 尽量通过切片访问以兼容 CPU/GPU，失败则直接返回
        let h = match state.h.try_as_slice() {
            Some(s) => s,
            None => return,
        };
        let hu = match state.hu.try_as_slice() {
            Some(s) => s,
            None => return,
        };
        let hv = match state.hv.try_as_slice() {
            Some(s) => s,
            None => return,
        };
        let z = match state.z.try_as_slice() {
            Some(s) => s,
            None => return,
        };

        let roughness = match self.roughness.try_as_slice() {
            Some(s) => s,
            None => return,
        };

        let u_out = match output.u_layers.try_as_slice_mut() {
            Some(s) => s,
            None => return,
        };
        let v_out = match output.v_layers.try_as_slice_mut() {
            Some(s) => s,
            None => return,
        };
        let z_out = match output.z_layers.try_as_slice_mut() {
            Some(s) => s,
            None => return,
        };

        let n_cells = state.n_cells();
        let n_layers = self.n_layers;
        let sigma_levels = self.sigma.sigma_centers();

        for cell in 0..n_cells {
            let h_cell = h[cell];
            let z_bed = z[cell];

            if h_cell < <B::Scalar as Scalar>::from_config(1e-6).unwrap_or(B::Scalar::ZERO) {
                for k in 0..n_layers {
                    let idx = cell * n_layers + k;
                    u_out[idx] = B::Scalar::ZERO;
                    v_out[idx] = B::Scalar::ZERO;
                    z_out[idx] = z_bed;
                }
                continue;
            }

            let u_avg = hu[cell] / h_cell;
            let v_avg = hv[cell] / h_cell;
            let z0 = roughness[cell];

            for k in 0..n_layers {
                let idx = cell * n_layers + k;
                let sigma = sigma_levels[k];
                let sigma_s = <B::Scalar as Scalar>::from_config(sigma).unwrap_or(B::Scalar::ZERO);
                let one = B::Scalar::ONE;
                let z_layer = z_bed + h_cell * (one + sigma_s);
                z_out[idx] = z_layer;

                let factor = match self.method {
                    ProfileMethod::Uniform => one,
                    ProfileMethod::Parabolic => {
                        let sigma_sq = sigma_s * sigma_s;
                        let c = <B::Scalar as Scalar>::from_config(1.5).unwrap_or(one + one / (one + one));
                        c * (one - sigma_sq)
                    }
                    ProfileMethod::Logarithmic => {
                        let z_rel = h_cell * (one + sigma_s);
                        if z_rel > z0 {
                            let ratio = z_rel.safe_div(z0, one);
                            let top = ratio.safe_ln();
                            let denom = h_cell.safe_div(z0, one).safe_ln();
                            let val = top.safe_div(denom, one);
                            val.clamp_value(B::Scalar::ZERO, <B::Scalar as Scalar>::from_config(2.0).unwrap_or(one + one))
                        } else {
                            B::Scalar::ZERO
                        }
                    }
                };

                u_out[idx] = u_avg * factor;
                v_out[idx] = v_avg * factor;
            }
        }
    }
}

impl ProfileRestorer<CpuBackend<f64>> {
    /// 使用默认后端创建
    pub fn new(n_cells: usize, n_layers: usize, method: ProfileMethod) -> Self {
        Self::new_with_backend(CpuBackend::<f64>::new(), n_cells, n_layers, method)
    }
}
