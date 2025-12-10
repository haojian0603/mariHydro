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

use super::traits::{SourceContribution, SourceContext, SourceTerm};
use crate::state::ShallowWaterState;

/// 入流类型
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum InflowType {
    /// 无入流
    #[default]
    None,
    /// 恒定流量 [m³/s]
    ConstantDischarge(f64),
    /// 恒定流速入流 [m/s, 方向]
    ConstantVelocity {
        /// 入流流速 [m/s]
        velocity: f64,
        /// 入流方向 [弧度]
        direction: f64,
    },
    /// 均匀面源（如降雨）[m/s]
    UniformFlux(f64),
    /// 时变流量（需要外部更新）
    TimeVarying,
}


impl InflowType {
    /// 创建恒定流量入流
    pub fn constant_discharge(q: f64) -> Self {
        Self::ConstantDischarge(q)
    }

    /// 创建恒定流速入流
    pub fn constant_velocity(velocity: f64, direction_deg: f64) -> Self {
        Self::ConstantVelocity {
            velocity: velocity.abs(),
            direction: direction_deg.to_radians(),
        }
    }

    /// 创建降雨入流
    pub fn rainfall(intensity_mm_hr: f64) -> Self {
        // 转换 mm/hr 到 m/s
        let flux = intensity_mm_hr / (1000.0 * 3600.0);
        Self::UniformFlux(flux)
    }

    /// 创建蒸发出流
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
    pub current_discharge: Vec<f64>,
    /// 入流方向 [弧度]
    pub inflow_direction: Vec<f64>,
    /// 单元面积 [m²]
    pub cell_area: Vec<f64>,
    /// 最小水深
    pub h_min: f64,
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
    pub fn with_uniform_rainfall(mut self, intensity_mm_hr: f64) -> Self {
        let inflow = InflowType::rainfall(intensity_mm_hr);
        self.inflow_type.fill(inflow);
        self
    }

    /// 设置点源入流
    pub fn add_point_source(&mut self, cell: usize, discharge: f64, direction_deg: f64) {
        if cell < self.inflow_type.len() {
            self.inflow_type[cell] = InflowType::constant_discharge(discharge);
            self.inflow_direction[cell] = direction_deg.to_radians();
        }
    }

    /// 设置河流入流边界
    pub fn add_river_inflow(&mut self, cells: &[usize], total_discharge: f64, direction_deg: f64) {
        if cells.is_empty() {
            return;
        }

        // 均分流量到各单元
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

impl SourceTerm for InflowConfig {
    fn name(&self) -> &'static str {
        "Inflow"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        _ctx: &SourceContext,
    ) -> SourceContribution {
        let inflow = self.inflow_type.get(cell).copied().unwrap_or(InflowType::None);

        match inflow {
            InflowType::None => SourceContribution::ZERO,

            InflowType::ConstantDischarge(q) => {
                let area = self.cell_area.get(cell).copied().unwrap_or(1.0);
                let direction = self.inflow_direction.get(cell).copied().unwrap_or(0.0);

                // 水深变化率 = Q / A [m/s]
                let s_h = q / area;

                // 动量源项 = Q * v / A = Q² / (A * h) for 入流
                // 简化：假设入流速度与方向一致
                let h = state.h[cell].max(self.h_min);
                let v_in = q / (area * h).max(1e-10);

                let s_hu = s_h * v_in * direction.cos();
                let s_hv = s_h * v_in * direction.sin();

                SourceContribution::new(s_h, s_hu, s_hv)
            }

            InflowType::ConstantVelocity { velocity, direction } => {
                let h = state.h[cell];
                if h < self.h_min {
                    return SourceContribution::ZERO;
                }

                // 对于速度入流，不改变水深，只改变动量
                let s_hu = velocity * direction.cos();
                let s_hv = velocity * direction.sin();

                SourceContribution::momentum(s_hu, s_hv)
            }

            InflowType::UniformFlux(flux) => {
                // 均匀面源只影响水深
                SourceContribution::mass(flux)
            }

            InflowType::TimeVarying => {
                let q = self.current_discharge.get(cell).copied().unwrap_or(0.0);
                let area = self.cell_area.get(cell).copied().unwrap_or(1.0);
                let direction = self.inflow_direction.get(cell).copied().unwrap_or(0.0);

                let s_h = q / area;

                let h = state.h[cell].max(self.h_min);
                let v_in = q / (area * h).max(1e-10);

                let s_hu = s_h * v_in * direction.cos();
                let s_hv = s_h * v_in * direction.sin();

                SourceContribution::new(s_h, s_hu, s_hv)
            }
        }
    }

    fn is_explicit(&self) -> bool {
        true
    }
}

/// 降雨配置
#[derive(Debug, Clone)]
pub struct RainfallConfig {
    /// 是否启用
    pub enabled: bool,
    /// 降雨强度 [m/s]（每个单元）
    pub intensity: Vec<f64>,
    /// 是否考虑渗透
    pub with_infiltration: bool,
    /// 渗透率 [m/s]（每个单元）
    pub infiltration_rate: Vec<f64>,
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
    pub fn with_uniform_intensity(mut self, intensity_mm_hr: f64) -> Self {
        let intensity_ms = intensity_mm_hr / (1000.0 * 3600.0);
        self.intensity.fill(intensity_ms);
        self
    }

    /// 设置单元降雨强度 [mm/hr]
    pub fn set_intensity(&mut self, cell: usize, intensity_mm_hr: f64) {
        if cell < self.intensity.len() {
            self.intensity[cell] = intensity_mm_hr / (1000.0 * 3600.0);
        }
    }

    /// 启用渗透
    pub fn with_infiltration(mut self, rate_mm_hr: f64) -> Self {
        self.with_infiltration = true;
        let rate_ms = rate_mm_hr / (1000.0 * 3600.0);
        self.infiltration_rate.fill(rate_ms);
        self
    }

    /// 计算净降雨强度 [m/s]
    pub fn net_intensity(&self, cell: usize) -> f64 {
        let rain = self.intensity.get(cell).copied().unwrap_or(0.0);
        if self.with_infiltration {
            let infil = self.infiltration_rate.get(cell).copied().unwrap_or(0.0);
            (rain - infil).max(0.0)
        } else {
            rain
        }
    }
}

impl SourceTerm for RainfallConfig {
    fn name(&self) -> &'static str {
        "Rainfall"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        _state: &ShallowWaterState,
        cell: usize,
        _ctx: &SourceContext,
    ) -> SourceContribution {
        let net = self.net_intensity(cell);
        SourceContribution::mass(net)
    }

    fn is_explicit(&self) -> bool {
        true
    }
}

/// 蒸发配置
#[derive(Debug, Clone)]
pub struct EvaporationConfig {
    /// 是否启用
    pub enabled: bool,
    /// 蒸发率 [m/s]（每个单元）
    pub rate: Vec<f64>,
    /// 最小水深（低于此不蒸发）
    pub h_min: f64,
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
    pub fn with_uniform_rate(mut self, rate_mm_hr: f64) -> Self {
        let rate_ms = rate_mm_hr / (1000.0 * 3600.0);
        self.rate.fill(rate_ms);
        self
    }

    /// 设置单元蒸发率 [mm/hr]
    pub fn set_rate(&mut self, cell: usize, rate_mm_hr: f64) {
        if cell < self.rate.len() {
            self.rate[cell] = rate_mm_hr / (1000.0 * 3600.0);
        }
    }
}

impl SourceTerm for EvaporationConfig {
    fn name(&self) -> &'static str {
        "Evaporation"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h[cell];
        if h < self.h_min || ctx.is_dry(h) {
            return SourceContribution::ZERO;
        }

        let rate = self.rate.get(cell).copied().unwrap_or(0.0);
        // 蒸发为负值（水深减少）
        SourceContribution::mass(-rate)
    }

    fn is_explicit(&self) -> bool {
        true
    }
}

/// 入流源便捷构造器
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
    pub fn uniform(n_cells: usize, rate_mm_hr: f64) -> EvaporationConfig {
        EvaporationConfig::new(n_cells).with_uniform_rate(rate_mm_hr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NumericalParams;

    fn create_test_state(n_cells: usize, h: f64) -> ShallowWaterState {
        let mut state = ShallowWaterState::new(n_cells);
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
        match inflow {
            InflowType::ConstantDischarge(q) => {
                assert!((q - 10.0).abs() < 1e-10);
            }
            _ => panic!("Expected ConstantDischarge"),
        }
    }

    #[test]
    fn test_inflow_type_rainfall() {
        let inflow = InflowType::rainfall(36.0); // 36 mm/hr
        match inflow {
            InflowType::UniformFlux(flux) => {
                // 36 mm/hr = 36 / (1000 * 3600) m/s = 1e-5 m/s
                assert!((flux - 1e-5).abs() < 1e-10);
            }
            _ => panic!("Expected UniformFlux"),
        }
    }

    #[test]
    fn test_inflow_type_evaporation() {
        let inflow = InflowType::evaporation(3.6); // 3.6 mm/hr
        match inflow {
            InflowType::UniformFlux(flux) => {
                // -3.6 mm/hr = -1e-6 m/s
                assert!((flux - (-1e-6)).abs() < 1e-12);
            }
            _ => panic!("Expected UniformFlux"),
        }
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

        match config.inflow_type[0] {
            InflowType::ConstantDischarge(q) => {
                assert!((q - 5.0).abs() < 1e-10);
            }
            _ => panic!("Expected ConstantDischarge"),
        }
    }

    #[test]
    fn test_inflow_source_uniform_flux() {
        let config = InflowConfig::new(10)
            .with_uniform_rainfall(36.0);

        let state = create_test_state(10, 1.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

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
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

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
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

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
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

        // 蒸发为负值
        assert!((contrib.s_h - (-1e-6)).abs() < 1e-12);
        assert_eq!(config.name(), "Evaporation");
    }

    #[test]
    fn test_evaporation_dry_cell() {
        let config = EvaporationConfig::new(10)
            .with_uniform_rate(3.6);

        let state = create_test_state(10, 1e-8);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

        // 干单元不蒸发
        assert_eq!(contrib.s_h, 0.0);
    }

    #[test]
    fn test_source_term_traits() {
        let inflow = InflowConfig::new(10);
        assert_eq!(inflow.name(), "Inflow");
        assert!(inflow.is_explicit());

        let rainfall = RainfallConfig::new(10);
        assert_eq!(rainfall.name(), "Rainfall");
        assert!(rainfall.is_explicit());

        let evap = EvaporationConfig::new(10);
        assert_eq!(evap.name(), "Evaporation");
        assert!(evap.is_explicit());
    }
}
