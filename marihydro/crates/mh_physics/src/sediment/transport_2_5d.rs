//! marihydro\crates\mh_physics\src\sediment\transport_2_5d.rs
//! 2.5D 泥沙输运
//!
//! 基于垂向剖面恢复的泥沙输运计算。
//! 
//! 这是一个独立的扩展模块，不侵入2D核心。

use crate::core::{Backend, CpuBackend};
use crate::state::ShallowWaterStateGeneric;
use crate::vertical::profile::{ProfileRestorer, VerticalProfile, ProfileMethod};

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
}

impl Transport2_5D<CpuBackend<f64>> {
    /// 创建求解器
    pub fn new(
        n_cells: usize,
        n_layers: usize,
        settling_velocity: f64,
        diffusion_coeff: f64,
    ) -> Self {
        let total = n_cells * n_layers;
        
        Self {
            profile_restorer: ProfileRestorer::new(n_cells, n_layers, ProfileMethod::Logarithmic),
            profile: VerticalProfile::new(n_cells, n_layers),
            concentration: vec![0.0; total],
            settling_velocity,
            diffusion_coeff,
            n_layers,
        }
    }
    
    /// 执行一步输运计算
    pub fn step(
        &mut self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        dt: f64,
    ) {
        // 1. 恢复垂向剖面
        self.profile_restorer.restore(state, &mut self.profile);
        
        // 2. 垂向输运（扩散+沉降）
        self.compute_vertical_transport(state, dt);
    }
    
    /// 垂向输运（扩散+沉降）
    fn compute_vertical_transport(
        &mut self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        dt: f64,
    ) {
        let h: &[f64] = &state.h;
        let n_cells = state.n_cells();
        let n_layers = self.n_layers;
        
        let ws = self.settling_velocity;
        let kv = self.diffusion_coeff;
        
        for cell in 0..n_cells {
            let h_cell = h[cell];
            
            if h_cell < 1e-6 {
                continue;
            }
            
            let dz = h_cell / (n_layers as f64);
            
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
                if self.concentration[idx] < 0.0 {
                    self.concentration[idx] = 0.0;
                }
            }
        }
    }
    
    /// 获取深度平均浓度
    pub fn depth_averaged_concentration(&self, cell: usize) -> f64 {
        let n_layers = self.n_layers;
        let mut sum = 0.0;
        
        for k in 0..n_layers {
            let idx = cell * n_layers + k;
            sum += self.concentration[idx];
        }
        
        sum / (n_layers as f64)
    }
    
    /// 设置浓度
    pub fn set_concentration(&mut self, cell: usize, layer: usize, value: f64) {
        let idx = cell * self.n_layers + layer;
        self.concentration[idx] = value;
    }
    
    /// 获取浓度
    pub fn get_concentration(&self, cell: usize, layer: usize) -> f64 {
        let idx = cell * self.n_layers + layer;
        self.concentration[idx]
    }
    
    /// 获取垂向剖面引用
    pub fn profile(&self) -> &VerticalProfile<CpuBackend<f64>> {
        &self.profile
    }
}
