// crates/mh_physics/src/sediment/exchange.rs

use crate::core::{Backend, DeviceBuffer};
use mh_runtime::RuntimeScalar as Scalar;

/// 泥沙交换参数
#[derive(Debug, Clone)]
pub struct ExchangeParams<S: Scalar> {
    /// 临界剪切应力 [Pa]
    pub tau_critical: S,
    /// 侵蚀系数 [kg/m²/s/Pa] (Partheniades公式)
    pub erosion_rate: S,
    /// 沉降速度 [m/s]
    pub settling_velocity: S,
    /// 泥沙干密度 [kg/m³]
    pub dry_density: S,
    /// 床面孔隙率
    pub porosity: S,
}

/// 泥沙交换通量计算器
pub struct SedimentExchange<B: Backend> {
    params: ExchangeParams<B::Scalar>,
    /// 交换通量缓存 [kg/m²/s]，正值=侵蚀，负值=沉降
    flux: B::Buffer<B::Scalar>,
    /// 侵蚀通量（分离存储用于诊断）
    erosion: B::Buffer<B::Scalar>,
    /// 沉降通量
    deposition: B::Buffer<B::Scalar>,
    /// 累积交换量（用于守恒校验）
    cumulative_exchange: B::Scalar,
    backend: B,
}

impl<B: Backend> SedimentExchange<B> {
    pub fn new(backend: B, n_cells: usize, params: ExchangeParams<B::Scalar>) -> Self {
        let zero = B::Scalar::ZERO;
        Self {
            params,
            flux: backend.alloc_init(n_cells, zero),
            erosion: backend.alloc_init(n_cells, zero),
            deposition: backend.alloc_init(n_cells, zero),
            cumulative_exchange: zero,
            backend,
        }
    }
    
    /// 计算侵蚀/沉降通量
    /// 
    /// # 参数
    /// - `tau_bed`: 床面剪切应力 [Pa]
    /// - `concentration`: 近底层泥沙浓度 [kg/m³]
    /// - `depth`: 水深 [m]
    pub fn compute(
        &mut self,
        tau_bed: &B::Buffer<B::Scalar>,
        concentration: &B::Buffer<B::Scalar>,
        depth: &B::Buffer<B::Scalar>,
    ) {
        let n = tau_bed.len().min(concentration.len()).min(depth.len());
        let zero = B::Scalar::ZERO;

        // 提取参数以避免借用冲突
        let tau_critical = self.params.tau_critical;
        let erosion_rate = self.params.erosion_rate;
        let settling_velocity = self.params.settling_velocity;
        
        let tau_slice = tau_bed.as_slice();
        let conc_slice = concentration.as_slice();
        let depth_slice = depth.as_slice();
        let flux_slice = self.flux.as_slice_mut();
        let ero_slice = self.erosion.as_slice_mut();
        let dep_slice = self.deposition.as_slice_mut();
        
        for i in 0..n {
            if depth_slice[i] <= zero {
                flux_slice[i] = zero;
                ero_slice[i] = zero;
                dep_slice[i] = zero;
                continue;
            }
            // 内联计算侵蚀率（Partheniades公式）
            let e_rate = if tau_slice[i] > tau_critical {
                erosion_rate * (tau_slice[i] - tau_critical)
            } else {
                zero
            };
            // 内联计算沉降率
            let d_rate = settling_velocity * conc_slice[i];
            ero_slice[i] = e_rate;
            dep_slice[i] = d_rate;
            flux_slice[i] = e_rate - d_rate;
        }

        let slice = self.flux.as_slice();
        let added = slice
            .iter()
            .take(n)
            .fold(0.0, |acc, &v| acc + v.to_f64());
        self.cumulative_exchange += <B::Scalar as Scalar>::from_config(added).unwrap_or(B::Scalar::ZERO);
    }
    
    /// 获取净交换通量
    pub fn flux(&self) -> &B::Buffer<B::Scalar> {
        &self.flux
    }
    
    /// 获取侵蚀通量
    pub fn erosion(&self) -> &B::Buffer<B::Scalar> {
        &self.erosion
    }
    
    /// 获取沉降通量
    pub fn deposition(&self) -> &B::Buffer<B::Scalar> {
        &self.deposition
    }
    
    /// 应用通量更新床面质量
    /// 
    /// bed_mass[i] += flux[i] * dt * cell_area[i]
    pub fn apply_to_bed(
        &self,
        bed_mass: &mut B::Buffer<B::Scalar>,
        dt: B::Scalar,
        cell_areas: &B::Buffer<B::Scalar>,
    ) {
        let n = bed_mass.len().min(self.flux.len()).min(cell_areas.len());
        let bed_slice = bed_mass.as_slice_mut();
        let flux_slice = self.flux.as_slice();
        let area_slice = cell_areas.as_slice();
        
        for i in 0..n {
            bed_slice[i] += flux_slice[i] * dt * area_slice[i];
        }
    }
    
    /// 应用通量更新悬沙浓度
    /// 
    /// concentration[i] -= flux[i] * dt / depth[i]
    pub fn apply_to_suspended(
        &self,
        concentration: &mut B::Buffer<B::Scalar>,
        depth: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) {
        let n = concentration.len().min(depth.len()).min(self.flux.len());
        let zero = B::Scalar::ZERO;
        let conc_slice = concentration.as_slice_mut();
        let depth_slice = depth.as_slice();
        let flux_slice = self.flux.as_slice();
        
        for i in 0..n {
            let h = depth_slice[i];
            if h <= zero {
                continue;
            }
            conc_slice[i] -= flux_slice[i] * dt / h;
            if conc_slice[i] < zero {
                conc_slice[i] = zero;
            }
        }
    }
    
    /// 计算侵蚀率（Partheniades公式）
    #[allow(dead_code)]
    fn compute_erosion_rate(&self, tau: B::Scalar) -> B::Scalar {
        if tau > self.params.tau_critical {
            self.params.erosion_rate * (tau - self.params.tau_critical)
        } else {
            B::Scalar::ZERO
        }
    }
    
    /// 计算沉降率
    #[allow(dead_code)]
    fn compute_deposition_rate(&self, concentration: B::Scalar) -> B::Scalar {
        self.params.settling_velocity * concentration
    }
    
    /// 获取累积交换量（用于守恒校验）
    pub fn cumulative_exchange(&self) -> B::Scalar {
        self.cumulative_exchange
    }
    
    /// 重置累积统计
    pub fn reset_statistics(&mut self) {
        self.cumulative_exchange = B::Scalar::ZERO;
        self.backend.enforce_positivity(&mut self.erosion, B::Scalar::ZERO);
        self.backend.enforce_positivity(&mut self.deposition, B::Scalar::ZERO);
        self.backend.enforce_positivity(&mut self.flux, B::Scalar::ZERO);
    }
}
