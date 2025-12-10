//! 垂向剖面恢复器
//!
//! 从2D深度平均状态恢复垂向速度剖面。

use crate::core::{Backend, Scalar, DeviceBuffer, CpuBackend};
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
            roughness: backend.alloc_init(n_cells, B::Scalar::from_f64(0.01)), // 默认糙率
            n_layers,
            method,
            von_karman: B::Scalar::from_f64(VON_KARMAN),
            backend,
        }
    }
    
    /// 设置糙率（仅 CPU 后端有效）
    pub fn set_roughness(&mut self, cell: usize, z0: B::Scalar) {
        if let Some(slice) = self.roughness.as_slice_mut() {
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
}

impl ProfileRestorer<CpuBackend<f64>> {
    /// 使用默认后端创建
    pub fn new(n_cells: usize, n_layers: usize, method: ProfileMethod) -> Self {
        Self::new_with_backend(CpuBackend::<f64>::new(), n_cells, n_layers, method)
    }
    
    /// 从2D状态恢复垂向剖面
    pub fn restore(
        &self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        output: &mut VerticalProfile<CpuBackend<f64>>,
    ) {
        let h: &[f64] = &state.h;
        let hu: &[f64] = &state.hu;
        let hv: &[f64] = &state.hv;
        let z: &[f64] = &state.z;
        let roughness: &[f64] = &self.roughness;
        
        let u_out: &mut [f64] = &mut output.u_layers;
        let v_out: &mut [f64] = &mut output.v_layers;
        let z_out: &mut [f64] = &mut output.z_layers;
        
        let n_cells = state.n_cells();
        let n_layers = self.n_layers;
        let sigma_levels = self.sigma.sigma_centers();
        
        for cell in 0..n_cells {
            let h_cell = h[cell];
            let z_bed = z[cell];
            
            if h_cell < 1e-6 {
                // 干单元
                for k in 0..n_layers {
                    let idx = cell * n_layers + k;
                    u_out[idx] = 0.0;
                    v_out[idx] = 0.0;
                    z_out[idx] = z_bed;
                }
                continue;
            }
            
            let u_avg = hu[cell] / h_cell;
            let v_avg = hv[cell] / h_cell;
            let _speed_avg = (u_avg * u_avg + v_avg * v_avg).sqrt();
            let z0 = roughness[cell];
            
            for k in 0..n_layers {
                let idx = cell * n_layers + k;
                let sigma = sigma_levels[k];
                let z_layer = z_bed + h_cell * (1.0 + sigma); // σ: 0 at surface, -1 at bottom
                z_out[idx] = z_layer;
                
                let factor = match self.method {
                    ProfileMethod::Uniform => 1.0,
                    ProfileMethod::Parabolic => {
                        // u(σ) = 1.5 * u_avg * (1 - σ²)
                        1.5 * (1.0 - sigma * sigma)
                    }
                    ProfileMethod::Logarithmic => {
                        // 对数律剖面
                        let z_rel = h_cell * (1.0 + sigma);
                        if z_rel > z0 {
                            let log_factor = (z_rel / z0).ln() / (h_cell / z0).ln();
                            log_factor.max(0.0).min(2.0)
                        } else {
                            0.0
                        }
                    }
                };
                
                u_out[idx] = u_avg * factor;
                v_out[idx] = v_avg * factor;
            }
        }
    }
}
