// crates/mh_physics/src/sources/wave_source.rs

//! 波浪辐射应力源项
//!
//! 将波浪模块的辐射应力计算集成为源项接口。
//!
//! # 物理背景
//!
//! 波浪辐射应力是波浪对平均流动的动量传递，表现为：
//!
//! ```text
//! S_xx = E(n(cos²θ + 1) - 0.5)
//! S_yy = E(n(sin²θ + 1) - 0.5)
//! S_xy = E × n × sinθ × cosθ
//! ```
//!
//! 动量方程源项为辐射应力梯度：
//!
//! ```text
//! F_x = -1/(ρh) × (∂S_xx/∂x + ∂S_xy/∂y)
//! F_y = -1/(ρh) × (∂S_xy/∂x + ∂S_yy/∂y)
//! ```

use crate::sources::traits::{
    SourceContributionGeneric, SourceContextGeneric, SourceStiffness, SourceTermGeneric,
};
use crate::state::ShallowWaterState;
use crate::waves::radiation_stress::{RadiationStressCalculatorGeneric, RadiationStressTensorGeneric, WaveFieldGeneric, WaveFieldError};
use crate::core::CpuBackend;
use mh_runtime::DeviceBuffer;

/// 波浪辐射应力源项
///
/// 从波场计算辐射应力梯度，作为动量源项
pub struct WaveRadiationSource {
    /// 辐射应力计算器（预留用于未来扩展）
    #[allow(dead_code)]
    calculator: RadiationStressCalculatorGeneric<CpuBackend<f64>>,
    /// 波场数据
    wave_field: WaveFieldGeneric<CpuBackend<f64>>,
    /// 辐射应力场
    stress: Vec<RadiationStressTensorGeneric<f64>>,
    /// 辐射应力梯度 (∂S_xx/∂x + ∂S_xy/∂y, ∂S_xy/∂x + ∂S_yy/∂y)
    stress_gradient: Vec<(f64, f64)>,
    /// 水密度 [kg/m³]
    rho_water: f64,
    /// 是否启用
    enabled: bool,
    /// 是否已计算梯度
    gradient_computed: bool,
}

impl WaveRadiationSource {
    /// 创建新的波浪辐射应力源项
    pub fn new(n_cells: usize) -> Self {
        let backend = CpuBackend::<f64>::new();
        Self {
            calculator: RadiationStressCalculatorGeneric::<CpuBackend<f64>>::new(backend.clone(), n_cells),
            wave_field: WaveFieldGeneric::<CpuBackend<f64>>::new(backend, n_cells),
            stress: vec![RadiationStressTensorGeneric::<f64>::default(); n_cells],
            stress_gradient: vec![(0.0, 0.0); n_cells],
            rho_water: 1025.0,
            enabled: true,
            gradient_computed: false,
        }
    }

    /// 设置波场数据
    pub fn set_wave_field(&mut self, wave_field: WaveFieldGeneric<CpuBackend<f64>>) {
        self.wave_field = wave_field;
        self.gradient_computed = false;
    }

    /// 更新波场参数（单一均匀波浪）
    pub fn set_uniform_waves(&mut self, height: f64, period: f64, direction: f64, depth: &[f64]) -> Result<(), WaveFieldError> {
        let n_cells = self.wave_field.len();
        let height_buf = self.wave_field.height.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("height buffer not accessible".to_string())
        })?;
        let period_buf = self.wave_field.period.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("period buffer not accessible".to_string())
        })?;
        let direction_buf = self.wave_field.direction.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("direction buffer not accessible".to_string())
        })?;
        let wavenumber = self.wave_field.wavenumber.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("wavenumber buffer not accessible".to_string())
        })?;
        let group_factor = self.wave_field.group_factor.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("group_factor buffer not accessible".to_string())
        })?;
        let wavelength = self.wave_field.wavelength.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("wavelength buffer not accessible".to_string())
        })?;
        let energy = self.wave_field.energy.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("energy buffer not accessible".to_string())
        })?;

        let backend = CpuBackend::<f64>::new();
        for i in 0..n_cells {
            height_buf[i] = height;
            period_buf[i] = period;
            direction_buf[i] = direction;
            
            // 计算波数和群速度因子
            let h = depth.get(i).copied().unwrap_or(10.0);
            let omega = 2.0 * std::f64::consts::PI / period.max(1e-6);
            let (k, n) = crate::waves::radiation_stress::compute_wavenumber_and_n(&backend, omega, h);
            wavenumber[i] = k;
            group_factor[i] = n;
            wavelength[i] = 2.0 * std::f64::consts::PI / k;
            
            // 能量密度
            let rho_g = self.rho_water * 9.81;
            energy[i] = rho_g * height * height / 8.0;
        }
        self.gradient_computed = false;
        Ok(())
    }

    /// 计算辐射应力场
    pub fn compute_stress(&mut self) {
        let n_cells = self.wave_field.len();
        let energy = match self.wave_field.energy.try_as_slice() {
            Some(v) => v,
            None => return,
        };
        let group_factor = match self.wave_field.group_factor.try_as_slice() {
            Some(v) => v,
            None => return,
        };
        let direction = match self.wave_field.direction.try_as_slice() {
            Some(v) => v,
            None => return,
        };
        for i in 0..n_cells {
            // 使用 RadiationStressTensor::compute 直接计算每个单元的辐射应力
            self.stress[i] = RadiationStressTensorGeneric::<f64>::compute(
                energy[i],
                group_factor[i],
                direction[i],
            );
        }
    }

    /// 计算辐射应力梯度（需要网格信息）
    ///
    /// 简化版本：使用相邻单元差分估计梯度
    /// 完整版本需要网格拓扑信息
    ///
    /// # 参数
    /// - `cell_centers`: 单元中心坐标 [(x, y), ...]
    /// - `neighbors`: 每个单元的邻居列表
    /// - `face_normals`: 面法向量
    pub fn compute_gradient_simple(&mut self, _cell_sizes: &[f64]) {
        // Explicit placeholder: without mesh topology we do not attempt an
        // approximate gradient reconstruction.
        for grad in &mut self.stress_gradient {
            *grad = (0.0, 0.0);
        }

        self.enabled = false;
        self.gradient_computed = false;
    }

    /// 璁剧疆棰勮绠楃殑姊害锛堜粠澶栭儴璁＄畻鍚庝紶鍏ワ級
    pub fn set_gradient(&mut self, gradient: &[(f64, f64)]) {
        let n = self.stress_gradient.len().min(gradient.len());
        self.stress_gradient[..n].copy_from_slice(&gradient[..n]);
        self.enabled = true;
        self.gradient_computed = true;
    }

    /// 鑾峰彇杈愬皠搴斿姏鍦?
    pub fn stress(&self) -> &[RadiationStressTensorGeneric<f64>] {
        &self.stress
    }

    /// 鑾峰彇娉㈠満
    pub fn wave_field(&self) -> &WaveFieldGeneric<CpuBackend<f64>> {
        &self.wave_field
    }
}

impl SourceTermGeneric<CpuBackend<f64>> for WaveRadiationSource {
    fn name(&self) -> &'static str {
        "WaveRadiation"
    }

    fn stiffness(&self) -> SourceStiffness {
        SourceStiffness::Explicit
    }

    fn is_enabled(&self) -> bool {
        self.enabled && self.gradient_computed
    }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<CpuBackend<f64>>,
        ctx: &SourceContextGeneric<f64>,
    ) -> SourceContributionGeneric<f64> {
        if !self.is_enabled() {
            return SourceContributionGeneric::default();
        }

        let h = state.h[cell];
        if ctx.is_dry(h) {
            return SourceContributionGeneric::default();
        }

        let (grad_x, grad_y) = self.stress_gradient.get(cell).copied().unwrap_or((0.0, 0.0));
        let fx = -grad_x / (self.rho_water * h);
        let fy = -grad_y / (self.rho_water * h);

        SourceContributionGeneric::momentum(fx, fy)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<CpuBackend<f64>>,
        _rhs_h: &mut Vec<f64>,
        rhs_hu: &mut Vec<f64>,
        rhs_hv: &mut Vec<f64>,
        ctx: &SourceContextGeneric<f64>,
    ) {
        if !self.is_enabled() {
            return;
        }

        let n = state.n_cells().min(self.stress.len());
        if rhs_hu.len() < n {
            rhs_hu.resize(n, 0.0);
        }
        if rhs_hv.len() < n {
            rhs_hv.resize(n, 0.0);
        }

        for cell in 0..n {
            let contrib = SourceTermGeneric::compute_cell(self, cell, state, ctx);
            rhs_hu[cell] += contrib.s_hu;
            rhs_hv[cell] += contrib.s_hv;
        }
    }
}
pub struct WaveRadiationSourceGeneric {
    /// 预计算的动量源项 [m/s²]
    momentum_source: Vec<(f64, f64)>,
    /// 是否启用
    enabled: bool,
}

impl WaveRadiationSourceGeneric {
    /// 创建新的泛型波浪源项
    pub fn new(n_cells: usize) -> Self {
        Self {
            momentum_source: vec![(0.0, 0.0); n_cells],
            enabled: false, // 默认禁用，直到设置源项
        }
    }

    /// 设置动量源项（从外部计算后设置）
    pub fn set_momentum_source(&mut self, source: &[(f64, f64)]) {
        let n = self.momentum_source.len().min(source.len());
        self.momentum_source[..n].copy_from_slice(&source[..n]);
        self.enabled = true;
    }

    /// 启用/禁用
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl SourceTermGeneric<CpuBackend<f64>> for WaveRadiationSourceGeneric {
    fn name(&self) -> &'static str {
        "WaveRadiation"
    }

    fn stiffness(&self) -> SourceStiffness {
        SourceStiffness::Explicit
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<CpuBackend<f64>>,
        ctx: &SourceContextGeneric<f64>,
    ) -> SourceContributionGeneric<f64> {
        let h = state.h[cell];
        if ctx.is_dry(h) {
            return SourceContributionGeneric::default();
        }

        let (fx, fy) = self.momentum_source.get(cell).copied().unwrap_or((0.0, 0.0));
        SourceContributionGeneric::momentum(fx, fy)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<CpuBackend<f64>>,
        _rhs_h: &mut Vec<f64>,
        rhs_hu: &mut Vec<f64>,
        rhs_hv: &mut Vec<f64>,
        ctx: &SourceContextGeneric<f64>,
    ) {
        if !self.enabled {
            return;
        }

        for cell in 0..state.n_cells() {
            let h = state.h[cell];
            if !ctx.is_dry(h) {
                let (fx, fy) = self.momentum_source.get(cell).copied().unwrap_or((0.0, 0.0));
                rhs_hu[cell] += fx;
                rhs_hv[cell] += fy;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wave_radiation_source_creation() {
        let source = WaveRadiationSource::new(100);
        assert_eq!(source.name(), "WaveRadiation");
        assert!(!source.is_enabled()); // 未计算梯度前禁用
    }

    #[test]
    fn test_generic_wave_source() {
        let mut source = WaveRadiationSourceGeneric::new(10);
        assert!(!source.is_enabled());
        
        source.set_momentum_source(&[(0.1, 0.2); 10]);
        assert!(source.is_enabled());
    }
}
