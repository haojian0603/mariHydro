// src-tauri/src/marihydro/domain/state/tracer.rs
//! 标量示踪剂状态存储
//! 支持多种示踪剂类型和属性

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// 示踪剂类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TracerType {
    /// 保守示踪剂（无衰减、无沉降）
    Passive,
    /// 一阶衰减（如细菌、放射性物质）
    Decaying,
    /// 沉降颗粒（如悬浮泥沙）
    Settling,
    /// 反应性示踪剂（参与化学反应）
    Reactive,
    /// 温度场
    Temperature,
    /// 盐度场
    Salinity,
}

impl Default for TracerType {
    fn default() -> Self {
        Self::Passive
    }
}

/// 示踪剂物理属性
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerProperties {
    /// 分子扩散系数 [m²/s]
    pub diffusivity: f64,
    /// 沉降速度 [m/s]（正值向下）
    pub settling_velocity: f64,
    /// 一阶衰减率 [1/s]
    pub decay_rate: f64,
    /// 示踪剂类型
    pub tracer_type: TracerType,
    /// 最小浓度（用于数值稳定性）
    pub min_concentration: f64,
    /// 最大浓度（用于限制器）
    pub max_concentration: f64,
}

impl Default for TracerProperties {
    fn default() -> Self {
        Self {
            diffusivity: 1e-6,         // 分子扩散
            settling_velocity: 0.0,
            decay_rate: 0.0,
            tracer_type: TracerType::Passive,
            min_concentration: 0.0,
            max_concentration: f64::MAX,
        }
    }
}

impl TracerProperties {
    /// 创建温度场属性
    pub fn temperature() -> Self {
        Self {
            diffusivity: 1.4e-7,  // 水的热扩散率
            tracer_type: TracerType::Temperature,
            ..Default::default()
        }
    }

    /// 创建盐度场属性
    pub fn salinity() -> Self {
        Self {
            diffusivity: 1.5e-9,  // 盐的分子扩散
            tracer_type: TracerType::Salinity,
            min_concentration: 0.0,
            max_concentration: 42.0,  // PSU
            ..Default::default()
        }
    }

    /// 创建悬浮泥沙属性
    pub fn suspended_sediment(d50_mm: f64) -> Self {
        // 使用 Stokes 沉降公式估算沉降速度
        // ws = (ρs - ρw) * g * d² / (18 * μ)
        let d = d50_mm * 1e-3;  // mm -> m
        let rho_s = 2650.0;     // 石英密度 kg/m³
        let rho_w = 1025.0;     // 海水密度
        let g = 9.81;
        let mu = 1e-3;          // 动力粘度 Pa·s
        
        let ws = (rho_s - rho_w) * g * d * d / (18.0 * mu);
        
        Self {
            diffusivity: 1e-6,
            settling_velocity: ws.clamp(0.0, 0.1),  // 限制最大沉降速度
            tracer_type: TracerType::Settling,
            min_concentration: 0.0,
            ..Default::default()
        }
    }

    /// 创建细菌/大肠杆菌属性
    pub fn bacteria(t90_hours: f64) -> Self {
        // T90 = ln(10) / k
        let decay_rate = std::f64::consts::LN_10 / (t90_hours * 3600.0);
        
        Self {
            diffusivity: 1e-9,
            decay_rate,
            tracer_type: TracerType::Decaying,
            min_concentration: 0.0,
            ..Default::default()
        }
    }
}

/// 点源/面源定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerSource {
    /// 源所在单元索引
    pub cell_indices: Vec<usize>,
    /// 源强度 [单位/s]（正值为释放）
    pub rate: f64,
    /// 开始时间 [s]
    pub start_time: f64,
    /// 结束时间 [s]
    pub end_time: f64,
    /// 是否为质量源（true）还是浓度源（false）
    pub is_mass_source: bool,
}

impl TracerSource {
    /// 创建持续点源
    pub fn continuous(cell_idx: usize, rate: f64) -> Self {
        Self {
            cell_indices: vec![cell_idx],
            rate,
            start_time: 0.0,
            end_time: f64::MAX,
            is_mass_source: true,
        }
    }

    /// 创建脉冲源
    pub fn pulse(cell_idx: usize, total_mass: f64, duration: f64, start_time: f64) -> Self {
        Self {
            cell_indices: vec![cell_idx],
            rate: total_mass / duration,
            start_time,
            end_time: start_time + duration,
            is_mass_source: true,
        }
    }

    /// 检查源在给定时间是否活跃
    #[inline]
    pub fn is_active(&self, time: f64) -> bool {
        time >= self.start_time && time < self.end_time
    }
}

/// 单个示踪剂场
#[derive(Debug, Clone)]
pub struct TracerField {
    /// 示踪剂名称
    pub name: String,
    /// 浓度场 [单位/m³ 或 无量纲]
    pub concentration: Vec<f64>,
    /// 物理属性
    pub properties: TracerProperties,
    /// 源项列表
    pub sources: Vec<TracerSource>,
    /// 是否启用
    pub enabled: bool,
}

impl TracerField {
    /// 创建新的示踪剂场
    pub fn new(name: &str, n_cells: usize, properties: TracerProperties) -> Self {
        Self {
            name: name.to_string(),
            concentration: vec![0.0; n_cells],
            properties,
            sources: Vec::new(),
            enabled: true,
        }
    }

    /// 创建温度场
    pub fn temperature(n_cells: usize, initial_temp: f64) -> Self {
        let mut field = Self::new("temperature", n_cells, TracerProperties::temperature());
        field.concentration.fill(initial_temp);
        field
    }

    /// 创建盐度场
    pub fn salinity(n_cells: usize, initial_salinity: f64) -> Self {
        let mut field = Self::new("salinity", n_cells, TracerProperties::salinity());
        field.concentration.fill(initial_salinity);
        field
    }

    /// 添加源项
    pub fn add_source(&mut self, source: TracerSource) {
        self.sources.push(source);
    }

    /// 应用源项（单时间步）
    pub fn apply_sources(&mut self, time: f64, dt: f64, cell_volumes: &[f64]) {
        for source in &self.sources {
            if !source.is_active(time) {
                continue;
            }

            for &cell_idx in &source.cell_indices {
                if cell_idx >= self.concentration.len() {
                    continue;
                }

                let vol = cell_volumes.get(cell_idx).copied().unwrap_or(1.0);
                if vol < 1e-14 {
                    continue;
                }

                if source.is_mass_source {
                    // 质量源：dC/dt = S / V
                    self.concentration[cell_idx] += source.rate * dt / vol;
                } else {
                    // 浓度源：直接设置
                    self.concentration[cell_idx] = source.rate;
                }
            }
        }
    }

    /// 应用一阶衰减
    pub fn apply_decay(&mut self, dt: f64) {
        if self.properties.decay_rate <= 0.0 {
            return;
        }

        let factor = (-self.properties.decay_rate * dt).exp();
        self.concentration.par_iter_mut().for_each(|c| {
            *c *= factor;
        });
    }

    /// 应用浓度限制器
    pub fn apply_limiter(&mut self) {
        let min_c = self.properties.min_concentration;
        let max_c = self.properties.max_concentration;
        
        self.concentration.par_iter_mut().for_each(|c| {
            *c = c.clamp(min_c, max_c);
        });
    }

    /// 计算总质量
    pub fn total_mass(&self, cell_volumes: &[f64]) -> f64 {
        self.concentration
            .iter()
            .zip(cell_volumes.iter())
            .map(|(&c, &v)| c * v)
            .sum()
    }

    /// 计算平均浓度
    pub fn mean_concentration(&self, cell_volumes: &[f64]) -> f64 {
        let total_vol: f64 = cell_volumes.iter().sum();
        if total_vol < 1e-14 {
            return 0.0;
        }
        self.total_mass(cell_volumes) / total_vol
    }

    /// 重置浓度场
    pub fn reset(&mut self, value: f64) {
        self.concentration.fill(value);
    }

    /// 从另一场复制
    pub fn copy_from(&mut self, other: &TracerField) {
        self.concentration.copy_from_slice(&other.concentration);
    }
}

/// 多示踪剂管理器
#[derive(Debug, Clone, Default)]
pub struct TracerManager {
    /// 示踪剂场列表
    pub fields: Vec<TracerField>,
}

impl TracerManager {
    /// 创建空管理器
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// 添加示踪剂场
    pub fn add_tracer(&mut self, field: TracerField) -> usize {
        let idx = self.fields.len();
        self.fields.push(field);
        idx
    }

    /// 按名称获取示踪剂
    pub fn get_by_name(&self, name: &str) -> Option<&TracerField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// 按名称获取可变引用
    pub fn get_by_name_mut(&mut self, name: &str) -> Option<&mut TracerField> {
        self.fields.iter_mut().find(|f| f.name == name)
    }

    /// 获取示踪剂数量
    pub fn count(&self) -> usize {
        self.fields.len()
    }

    /// 获取所有启用的示踪剂名称
    pub fn enabled_names(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|f| f.enabled)
            .map(|f| f.name.as_str())
            .collect()
    }

    /// 应用所有源项
    pub fn apply_all_sources(&mut self, time: f64, dt: f64, cell_volumes: &[f64]) {
        for field in &mut self.fields {
            if field.enabled {
                field.apply_sources(time, dt, cell_volumes);
            }
        }
    }

    /// 应用所有衰减
    pub fn apply_all_decay(&mut self, dt: f64) {
        for field in &mut self.fields {
            if field.enabled {
                field.apply_decay(dt);
            }
        }
    }

    /// 应用所有限制器
    pub fn apply_all_limiters(&mut self) {
        for field in &mut self.fields {
            if field.enabled {
                field.apply_limiter();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracer_properties() {
        let props = TracerProperties::suspended_sediment(0.1);
        assert!(props.settling_velocity > 0.0);
        assert!(props.settling_velocity < 0.01);
    }

    #[test]
    fn test_bacteria_decay() {
        let props = TracerProperties::bacteria(24.0); // T90 = 24小时
        // k = ln(10) / T90
        let expected_k = std::f64::consts::LN_10 / (24.0 * 3600.0);
        assert!((props.decay_rate - expected_k).abs() < 1e-12);
    }

    #[test]
    fn test_source_activity() {
        let source = TracerSource::pulse(0, 1000.0, 100.0, 500.0);
        assert!(!source.is_active(0.0));
        assert!(source.is_active(500.0));
        assert!(source.is_active(550.0));
        assert!(!source.is_active(600.0));
    }

    #[test]
    fn test_mass_conservation() {
        let mut field = TracerField::new("test", 10, TracerProperties::default());
        let volumes = vec![100.0; 10];
        
        // 初始质量为0
        assert!((field.total_mass(&volumes)).abs() < 1e-14);
        
        // 设置均匀浓度
        field.reset(1.0);
        let mass = field.total_mass(&volumes);
        assert!((mass - 1000.0).abs() < 1e-10);
    }
}
