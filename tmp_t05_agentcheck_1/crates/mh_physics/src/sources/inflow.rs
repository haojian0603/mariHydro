// crates/mh_physics/src/sources/inflow.rs

//! 入流源项
//!
//! 实现各种入流和出流的源项处理，包括：
//! - 河流入流
//! - 降雨
//! - 蒸发
//! - 渗透
//! - 点源/汇
//!
//! # 入流/出流模型
//!
//! 入流作为质量源项添加到连续性方程：
//! ```text
//! ∂h/∂t + ∇·(hu) = S_h
//! ```
//!
//! 其中 S_h 是单位面积的水深变化率 [m/s]。
//!
//! 对于带动量的入流（如河流），还需要添加动量源项：
//! ```text
//! ∂(hu)/∂t + ∇·F = S_hu
//! ```

use super::traits::{
    SourceContributionGeneric, SourceContextGeneric, SourceStiffness, SourceTermGeneric,
};
use crate::state::ShallowWaterState;
use mh_runtime::CpuBackend;

// 注意：CpuBackend 已在上方导入

/// 入流类型
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum InflowType {
    /// 无入流
    #[default]
    None,
    /// 恒定流量 [m³/s]
    // ALLOW_F64: Layer 4 配置参数
    ConstantDischarge(f64),
    /// 恒定流速入流 [m/s, 方向]
    ConstantVelocity {
        /// 入流流速 [m/s]
        velocity: f64, // ALLOW_F64: Layer 4 配置参数
        /// 入流方向 [弧度]
        direction: f64, // ALLOW_F64: Layer 4 配置参数
    },
    /// 均匀面源（如降雨）[m/s]
    // ALLOW_F64: Layer 4 配置参数
    UniformFlux(f64),
    /// 时变流量（需要外部更新）
    TimeVarying,
}


impl InflowType {
    /// 创建恒定流量入流
    // ALLOW_F64: 物理参数
    pub fn constant_discharge(q: f64) -> Self {
        Self::ConstantDischarge(q)
    }

    /// 创建恒定流速入流
    // ALLOW_F64: 物理参数
    pub fn constant_velocity(velocity: f64, direction_deg: f64) -> Self {
        Self::ConstantVelocity {
            velocity: velocity.abs(),
            direction: direction_deg.to_radians(),
        }
    }

    /// 创建降雨入流
    // ALLOW_F64: 物理参数
    pub fn rainfall(intensity_mm_hr: f64) -> Self {
        // 转换 mm/hr 到 m/s
        let flux = intensity_mm_hr / (1000.0 * 3600.0);
        Self::UniformFlux(flux)
    }

    /// 创建蒸发出流
    // ALLOW_F64: 物理参数
    pub fn evaporation(rate_mm_hr: f64) -> Self {
        // 负值表示出流
        let flux = -rate_mm_hr / (1000.0 * 3600.0);
        Self::UniformFlux(flux)
    }
}

/// 入流边界配置
#[derive(Debug, Clone)]
pub struct InflowConfig {
    /// 是否启用
    pub enabled: bool,
    /// 入流类型（每个单元）
    pub inflow_type: Vec<InflowType>,
    /// 当前流量值 [m³/s]（用于时变入流）
    pub current_discharge: Vec<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 入流方向 [弧度]
    pub inflow_direction: Vec<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 单元面积 [m²]
    pub cell_area: Vec<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 最小水深
    pub h_min: f64, // ALLOW_F64: Layer 4 配置参数
}

impl InflowConfig {
    /// 创建新配置
    pub fn new(n_cells: usize) -> Self {
        Self {
            enabled: true,
            inflow_type: vec![InflowType::None; n_cells],
            current_discharge: vec![0.0; n_cells],
            inflow_direction: vec![0.0; n_cells],
            cell_area: vec![1.0; n_cells],
            h_min: 1e-4,
        }
    }

    /// 设置单元入流类型
    pub fn set_inflow(&mut self, cell: usize, inflow: InflowType) {
        if cell < self.inflow_type.len() {
            self.inflow_type[cell] = inflow;
        }
    }

    /// 设置单元面积
    // ALLOW_F64: 物理参数
    pub fn set_cell_area(&mut self, cell: usize, area: f64) {
        if cell < self.cell_area.len() {
            self.cell_area[cell] = area.max(1e-6);
        }
    }

    /// 批量设置单元面积
    pub fn set_cell_areas(&mut self, areas: &[f64]) {
        let n = self.cell_area.len().min(areas.len());
        for i in 0..n {
            self.cell_area[i] = areas[i].max(1e-6);
        }
    }

    /// 更新时变流量
    // ALLOW_F64: 物理参数
    pub fn update_discharge(&mut self, cell: usize, discharge: f64) {
        if cell < self.current_discharge.len() {
            self.current_discharge[cell] = discharge;
        }
    }

    /// 批量更新时变流量
    pub fn update_discharges(&mut self, discharges: &[f64]) {
        let n = self.current_discharge.len().min(discharges.len());
        self.current_discharge[..n].copy_from_slice(&discharges[..n]);
    }

    /// 设置均匀降雨
    // ALLOW_F64: 物理参数
    pub fn with_uniform_rainfall(mut self, intensity_mm_hr: f64) -> Self {
        let inflow = InflowType::rainfall(intensity_mm_hr);
        self.inflow_type.fill(inflow);
        self
    }

    /// 设置点源入流
    // ALLOW_F64: 物理参数
    pub fn add_point_source(&mut self, cell: usize, discharge: f64, direction_deg: f64) {
        if cell < self.inflow_type.len() {
            self.inflow_type[cell] = InflowType::constant_discharge(discharge);
            self.inflow_direction[cell] = direction_deg.to_radians();
        }
    }

    /// 设置河流入流边界
    // ALLOW_F64: 物理参数
    pub fn add_river_inflow(&mut self, cells: &[usize], total_discharge: f64, direction_deg: f64) {
        if cells.is_empty() {
            return;
        }

        // 均分流量到各单元
        // ALLOW_F64: 源项计算
        let q_per_cell = total_discharge / cells.len() as f64;
        let dir_rad = direction_deg.to_radians();

        for &cell in cells {
            if cell < self.inflow_type.len() {
                self.inflow_type[cell] = InflowType::constant_discharge(q_per_cell);
                self.inflow_direction[cell] = dir_rad;
            }
        }
    }
}

impl SourceTermGeneric<CpuBackend<f64>> for InflowConfig {
    fn name(&self) -> &'static str { "Inflow" }

    fn stiffness(&self) -> SourceStiffness { SourceStiffness::Explicit }

    fn is_enabled(&self) -> bool { self.enabled }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<CpuBackend<f64>>,
        _ctx: &SourceContextGeneric<f64>,
    ) -> SourceContributionGeneric<f64> {
        let inflow = self.inflow_type.get(cell).copied().unwrap_or(InflowType::None);

        match inflow {
            InflowType::None => SourceContributionGeneric::default(),
            InflowType::ConstantDischarge(q) => {
                let area = self.cell_area.get(cell).copied().unwrap_or(1.0);
                let direction = self.inflow_direction.get(cell).copied().unwrap_or(0.0);

                if !q.is_finite() || !area.is_finite() || area <= 0.0 || !direction.is_finite() {
                    return SourceContributionGeneric::default();
                }

                let s_h = q / area;
                let h = state.h[cell].max(self.h_min);
                let v_in = q / (area * h).max(1e-10);
                let s_hu = s_h * v_in * direction.cos();
                let s_hv = s_h * v_in * direction.sin();
                SourceContributionGeneric::new(s_h, s_hu, s_hv)
            }
            InflowType::ConstantVelocity { velocity, direction } => {
                if !velocity.is_finite() || !direction.is_finite() {
                    return SourceContributionGeneric::default();
                }
                let h = state.h[cell];
                if h < self.h_min {
                    return SourceContributionGeneric::default();
                }
                SourceContributionGeneric::momentum(velocity * direction.cos(), velocity * direction.sin())
            }
            InflowType::UniformFlux(flux) => {
                if flux.is_finite() {
                    SourceContributionGeneric::mass(flux)
                } else {
                    SourceContributionGeneric::default()
                }
            }
            InflowType::TimeVarying => {
                let q = self.current_discharge.get(cell).copied().unwrap_or(0.0);
                let area = self.cell_area.get(cell).copied().unwrap_or(1.0);
                let direction = self.inflow_direction.get(cell).copied().unwrap_or(0.0);

                if !q.is_finite() || !area.is_finite() || area <= 0.0 || !direction.is_finite() {
                    return SourceContributionGeneric::default();
                }

                let s_h = q / area;
                let h = state.h[cell].max(self.h_min);
                let v_in = q / (area * h).max(1e-10);
                let s_hu = s_h * v_in * direction.cos();
                let s_hv = s_h * v_in * direction.sin();
                SourceContributionGeneric::new(s_h, s_hu, s_hv)
            }
        }
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<CpuBackend<f64>>,
        rhs_h: &mut Vec<f64>,
        rhs_hu: &mut Vec<f64>,
        rhs_hv: &mut Vec<f64>,
        ctx: &SourceContextGeneric<f64>,
    ) {
        if !self.enabled {
            return;
        }

        let n = state.n_cells();
        if rhs_h.len() < n { rhs_h.resize(n, 0.0); }
        if rhs_hu.len() < n { rhs_hu.resize(n, 0.0); }
        if rhs_hv.len() < n { rhs_hv.resize(n, 0.0); }

        for cell in 0..n {
            let contrib = self.compute_cell(cell, state, ctx);
            rhs_h[cell] += contrib.s_h;
            rhs_hu[cell] += contrib.s_hu;
            rhs_hv[cell] += contrib.s_hv;
        }
    }
}
pub struct RainfallConfig {
    /// 是否启用
    pub enabled: bool,
    /// 降雨强度 [m/s]（每个单元）
    pub intensity: Vec<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 是否考虑渗透
    pub with_infiltration: bool,
    /// 渗透率 [m/s]（每个单元）
    pub infiltration_rate: Vec<f64>, // ALLOW_F64: Layer 4 配置参数
}

impl RainfallConfig {
    /// 创建新配置
    pub fn new(n_cells: usize) -> Self {
        Self {
            enabled: true,
            intensity: vec![0.0; n_cells],
            with_infiltration: false,
            infiltration_rate: vec![0.0; n_cells],
        }
    }

    /// 设置均匀降雨强度 [mm/hr]
    // ALLOW_F64: 物理参数
    pub fn with_uniform_intensity(mut self, intensity_mm_hr: f64) -> Self {
        let intensity_ms = intensity_mm_hr / (1000.0 * 3600.0);
        self.intensity.fill(intensity_ms);
        self
    }

    /// 设置单元降雨强度 [mm/hr]
    // ALLOW_F64: 物理参数
    pub fn set_intensity(&mut self, cell: usize, intensity_mm_hr: f64) {
        if cell < self.intensity.len() {
            self.intensity[cell] = intensity_mm_hr / (1000.0 * 3600.0);
        }
    }

    /// 启用渗透
    // ALLOW_F64: 物理参数
    pub fn with_infiltration(mut self, rate_mm_hr: f64) -> Self {
        self.with_infiltration = true;
        let rate_ms = rate_mm_hr / (1000.0 * 3600.0);
        self.infiltration_rate.fill(rate_ms);
        self
    }

    /// 计算净降雨强度 [m/s]
    // ALLOW_F64: 源项计算
    pub fn net_intensity(&self, cell: usize) -> f64 {
        let rain = self.intensity.get(cell).copied().unwrap_or(0.0);
        if !rain.is_finite() {
            return 0.0;
        }
        if self.with_infiltration {
            let infil = self.infiltration_rate.get(cell).copied().unwrap_or(0.0);
            if !infil.is_finite() {
                return rain.max(0.0);
            }
            (rain - infil).max(0.0)
        } else {
            rain
        }
    }
}

impl SourceTermGeneric<CpuBackend<f64>> for RainfallConfig {
    fn name(&self) -> &'static str { "Rainfall" }

    fn stiffness(&self) -> SourceStiffness { SourceStiffness::Explicit }

    fn is_enabled(&self) -> bool { self.enabled }

    fn compute_cell(
        &self,
        cell: usize,
        _state: &ShallowWaterState<CpuBackend<f64>>,
        _ctx: &SourceContextGeneric<f64>,
    ) -> SourceContributionGeneric<f64> {
        SourceContributionGeneric::mass(self.net_intensity(cell))
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<CpuBackend<f64>>,
        rhs_h: &mut Vec<f64>,
        _rhs_hu: &mut Vec<f64>,
        _rhs_hv: &mut Vec<f64>,
        ctx: &SourceContextGeneric<f64>,
    ) {
        if !self.enabled {
            return;
        }

        let n = state.n_cells();
        if rhs_h.len() < n { rhs_h.resize(n, 0.0); }
        for cell in 0..n {
            let contrib = self.compute_cell(cell, state, ctx);
            rhs_h[cell] += contrib.s_h;
        }
    }
}
pub struct EvaporationConfig {
    /// 是否启用
    pub enabled: bool,
    /// 蒸发率 [m/s]（每个单元）
    pub rate: Vec<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 最小水深（低于此不蒸发）
    pub h_min: f64, // ALLOW_F64: Layer 4 配置参数
}

impl EvaporationConfig {
    /// 创建新配置
    pub fn new(n_cells: usize) -> Self {
        Self {
            enabled: true,
            rate: vec![0.0; n_cells],
            h_min: 1e-6,
        }
    }

    /// 设置均匀蒸发率 [mm/hr]
    // ALLOW_F64: 物理参数
    pub fn with_uniform_rate(mut self, rate_mm_hr: f64) -> Self {
        let rate_ms = rate_mm_hr / (1000.0 * 3600.0);
        self.rate.fill(rate_ms);
        self
    }

    /// 设置单元蒸发率 [mm/hr]
    // ALLOW_F64: 物理参数
    pub fn set_rate(&mut self, cell: usize, rate_mm_hr: f64) {
        if cell < self.rate.len() {
            self.rate[cell] = rate_mm_hr / (1000.0 * 3600.0);
        }
    }
}

impl SourceTermGeneric<CpuBackend<f64>> for EvaporationConfig {
    fn name(&self) -> &'static str { "Evaporation" }

    fn stiffness(&self) -> SourceStiffness { SourceStiffness::Explicit }

    fn is_enabled(&self) -> bool { self.enabled }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<CpuBackend<f64>>,
        ctx: &SourceContextGeneric<f64>,
    ) -> SourceContributionGeneric<f64> {
        let h = state.h[cell];
        if h < self.h_min || ctx.is_dry(h) {
            return SourceContributionGeneric::default();
        }

        let rate = self.rate.get(cell).copied().unwrap_or(0.0);
        if !rate.is_finite() {
            return SourceContributionGeneric::default();
        }

        SourceContributionGeneric::mass(-rate)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<CpuBackend<f64>>,
        rhs_h: &mut Vec<f64>,
        _rhs_hu: &mut Vec<f64>,
        _rhs_hv: &mut Vec<f64>,
        ctx: &SourceContextGeneric<f64>,
    ) {
        if !self.enabled {
            return;
        }

        let n = state.n_cells();
        if rhs_h.len() < n { rhs_h.resize(n, 0.0); }
        for cell in 0..n {
            let contrib = self.compute_cell(cell, state, ctx);
            rhs_h[cell] += contrib.s_h;
        }
    }
}
pub struct InflowSource;

impl InflowSource {
    /// 创建新配置
    pub fn new(n_cells: usize) -> InflowConfig {
        InflowConfig::new(n_cells)
    }
}

/// 降雨源便捷构造器
pub struct RainfallSource;

impl RainfallSource {
    /// 创建新配置
    pub fn new(n_cells: usize) -> RainfallConfig {
        RainfallConfig::new(n_cells)
    }

    /// 创建均匀降雨配置
    // ALLOW_F64: 物理参数
    pub fn uniform(n_cells: usize, intensity_mm_hr: f64) -> RainfallConfig {
        RainfallConfig::new(n_cells).with_uniform_intensity(intensity_mm_hr)
    }
}

/// 蒸发源便捷构造器
pub struct EvaporationSource;

impl EvaporationSource {
    /// 创建新配置
    pub fn new(n_cells: usize) -> EvaporationConfig {
        EvaporationConfig::new(n_cells)
    }

    /// 创建均匀蒸发配置
    // ALLOW_F64: 物理参数
    pub fn uniform(n_cells: usize, rate_mm_hr: f64) -> EvaporationConfig {
        EvaporationConfig::new(n_cells).with_uniform_rate(rate_mm_hr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    fn create_test_state(n_cells: usize, h: f64) -> ShallowWaterState<CpuBackend<f64>> {
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(backend, n_cells);
        for i in 0..n_cells {
            state.h[i] = h;
            state.z[i] = 0.0;
        }
        state
    }

    #[test]
    fn test_inflow_type_none() {
        let inflow = InflowType::None;
        assert_eq!(inflow, InflowType::default());
    }

    #[test]
    fn test_inflow_type_discharge() {
        let inflow = InflowType::constant_discharge(10.0);
        assert!(matches!(inflow, InflowType::ConstantDischarge(q) if (q - 10.0).abs() < 1e-10));
    }

    #[test]
    fn test_inflow_type_rainfall() {
        let inflow = InflowType::rainfall(36.0); // 36 mm/hr
        assert!(matches!(inflow, InflowType::UniformFlux(flux) if (flux - 1e-5).abs() < 1e-10));
    }

    #[test]
    fn test_inflow_type_evaporation() {
        let inflow = InflowType::evaporation(3.6); // 3.6 mm/hr
        assert!(matches!(inflow, InflowType::UniformFlux(flux) if (flux - (-1e-6)).abs() < 1e-12));
    }

    #[test]
    fn test_inflow_config_creation() {
        let config = InflowConfig::new(10);
        assert!(config.enabled);
        assert_eq!(config.inflow_type.len(), 10);
    }

    #[test]
    fn test_inflow_config_point_source() {
        let mut config = InflowConfig::new(10);
        config.add_point_source(0, 5.0, 45.0);

        assert!(matches!(
            config.inflow_type[0],
            InflowType::ConstantDischarge(q) if (q - 5.0).abs() < 1e-10
        ));
    }

    #[test]
    fn test_inflow_source_uniform_flux() {
        let config = InflowConfig::new(10)
            .with_uniform_rainfall(36.0);

        let state = create_test_state(10, 1.0);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 1.0);

        let contrib = config.compute_cell(0, &state, &ctx);

        assert!((contrib.s_h - 1e-5).abs() < 1e-10);
        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_inflow_source_discharge() {
        let mut config = InflowConfig::new(10);
        config.set_cell_area(0, 100.0); // 100 m²
        config.add_point_source(0, 1.0, 0.0); // 1 m³/s, 东向

        let state = create_test_state(10, 1.0);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 1.0);

        let contrib = config.compute_cell(0, &state, &ctx);

        // s_h = Q/A = 1/100 = 0.01 m/s
        assert!((contrib.s_h - 0.01).abs() < 1e-10);
        // s_hu > 0 (东向)
        assert!(contrib.s_hu > 0.0);
    }

    #[test]
    fn test_rainfall_config() {
        let config = RainfallConfig::new(10)
            .with_uniform_intensity(36.0);

        assert!((config.intensity[0] - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_rainfall_with_infiltration() {
        let config = RainfallConfig::new(10)
            .with_uniform_intensity(36.0)   // 36 mm/hr = 1e-5 m/s
            .with_infiltration(18.0);       // 18 mm/hr = 5e-6 m/s

        let net = config.net_intensity(0);
        // net = 1e-5 - 5e-6 = 5e-6 m/s
        assert!((net - 5e-6).abs() < 1e-11);
    }

    #[test]
    fn test_rainfall_source_term() {
        let config = RainfallConfig::new(10)
            .with_uniform_intensity(36.0);

        let state = create_test_state(10, 1.0);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 1.0);

        let contrib = config.compute_cell(0, &state, &ctx);

        assert!((contrib.s_h - 1e-5).abs() < 1e-10);
        assert_eq!(config.name(), "Rainfall");
    }

    #[test]
    fn test_evaporation_config() {
        let config = EvaporationConfig::new(10)
            .with_uniform_rate(3.6);

        assert!((config.rate[0] - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn test_evaporation_source_term() {
        let config = EvaporationConfig::new(10)
            .with_uniform_rate(3.6);

        let state = create_test_state(10, 1.0);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 1.0);

        let contrib = config.compute_cell(0, &state, &ctx);

        // 蒸发为负值
        assert!((contrib.s_h - (-1e-6)).abs() < 1e-12);
        assert_eq!(config.name(), "Evaporation");
    }

    #[test]
    fn test_evaporation_dry_cell() {
        let config = EvaporationConfig::new(10)
            .with_uniform_rate(3.6);

        let state = create_test_state(10, 1e-8);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 1.0);

        let contrib = config.compute_cell(0, &state, &ctx);

        // 干单元不蒸发
        assert_eq!(contrib.s_h, 0.0);
    }

    #[test]
    fn test_source_term_traits() {
        let inflow = InflowConfig::new(10);
        assert_eq!(inflow.name(), "Inflow");
        assert_eq!(inflow.stiffness(), SourceStiffness::Explicit);

        let rainfall = RainfallConfig::new(10);
        assert_eq!(rainfall.name(), "Rainfall");
        assert_eq!(rainfall.stiffness(), SourceStiffness::Explicit);

        let evap = EvaporationConfig::new(10);
        assert_eq!(evap.name(), "Evaporation");
        assert_eq!(evap.stiffness(), SourceStiffness::Explicit);
    }
}
