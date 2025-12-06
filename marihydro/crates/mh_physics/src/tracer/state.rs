// crates/mh_physics/src/tracer/state.rs

//! 示踪剂状态模块
//!
//! 本模块定义示踪剂相关的状态类型：
//! - TracerType: 示踪剂类型枚举
//! - TracerProperties: 示踪剂物理属性
//! - TracerField: 单个示踪剂的场数据
//! - TracerState: 多示踪剂集合状态
//!
//! # 概念说明
//!
//! 示踪剂（Tracer）是指随水流运移的物质，包括：
//! - 被动示踪剂：盐度、温度等（不影响水动力）
//! - 主动示踪剂：泥沙等（可能影响水密度和流动）
//!
//! # 迁移说明
//!
//! 从 legacy_src/tracer/tracer.rs 迁移，改进：
//! - 使用枚举类型替代字符串标识
//! - 支持 serde 序列化
//! - 与新架构的 StateAccess trait 集成

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================
// 示踪剂类型
// ============================================================

/// 示踪剂类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TracerType {
    /// 盐度 [PSU 或 kg/m³]
    Salinity,

    /// 温度 [°C 或 K]
    Temperature,

    /// 悬浮泥沙 [kg/m³]
    Sediment,

    /// 污染物 [任意浓度单位]
    Pollutant,

    /// 溶解氧 [mg/L]
    DissolvedOxygen,

    /// 叶绿素 [μg/L]
    Chlorophyll,

    /// 自定义示踪剂
    Custom(u16),
}

impl TracerType {
    /// 获取类型的字符串标识
    pub fn name(&self) -> &'static str {
        match self {
            Self::Salinity => "salinity",
            Self::Temperature => "temperature",
            Self::Sediment => "sediment",
            Self::Pollutant => "pollutant",
            Self::DissolvedOxygen => "dissolved_oxygen",
            Self::Chlorophyll => "chlorophyll",
            Self::Custom(_) => "custom",
        }
    }

    /// 是否为被动示踪剂
    ///
    /// 被动示踪剂不影响水动力方程。
    pub fn is_passive(&self) -> bool {
        match self {
            Self::Sediment => false, // 泥沙可能影响密度
            _ => true,
        }
    }

    /// 是否需要额外的源汇项
    pub fn has_source_terms(&self) -> bool {
        match self {
            Self::DissolvedOxygen | Self::Chlorophyll => true, // 生化反应
            Self::Sediment => true, // 沉降/再悬浮
            _ => false,
        }
    }
}

impl Default for TracerType {
    fn default() -> Self {
        Self::Salinity
    }
}

// ============================================================
// 示踪剂属性
// ============================================================

/// 示踪剂物理属性
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerProperties {
    /// 示踪剂类型
    pub tracer_type: TracerType,

    /// 示踪剂名称（用于显示）
    pub name: String,

    /// 单位
    pub unit: String,

    /// 分子扩散系数 [m²/s]
    pub molecular_diffusivity: f64,

    /// 背景浓度（用于边界和初始化）
    pub background_value: f64,

    /// 衰减系数 [1/s]
    ///
    /// 用于简单的一阶衰减模型：dC/dt = -k * C
    pub decay_rate: f64,

    /// 沉降速度 [m/s]
    ///
    /// 仅适用于泥沙等可沉降物质，正值表示向下沉降。
    pub settling_velocity: f64,

    /// 是否启用
    pub enabled: bool,
}

impl TracerProperties {
    /// 创建默认盐度示踪剂
    pub fn salinity() -> Self {
        Self {
            tracer_type: TracerType::Salinity,
            name: "Salinity".to_string(),
            unit: "PSU".to_string(),
            molecular_diffusivity: 1.5e-9,
            background_value: 35.0,
            decay_rate: 0.0,
            settling_velocity: 0.0,
            enabled: true,
        }
    }

    /// 创建默认温度示踪剂
    pub fn temperature() -> Self {
        Self {
            tracer_type: TracerType::Temperature,
            name: "Temperature".to_string(),
            unit: "°C".to_string(),
            molecular_diffusivity: 1.4e-7,
            background_value: 20.0,
            decay_rate: 0.0,
            settling_velocity: 0.0,
            enabled: true,
        }
    }

    /// 创建默认泥沙示踪剂
    pub fn sediment() -> Self {
        Self {
            tracer_type: TracerType::Sediment,
            name: "Suspended Sediment".to_string(),
            unit: "kg/m³".to_string(),
            molecular_diffusivity: 0.0, // 主要靠湍流扩散
            background_value: 0.0,
            decay_rate: 0.0,
            settling_velocity: 1e-4, // 0.1 mm/s
            enabled: true,
        }
    }

    /// 创建自定义示踪剂
    pub fn custom(id: u16, name: &str, unit: &str) -> Self {
        Self {
            tracer_type: TracerType::Custom(id),
            name: name.to_string(),
            unit: unit.to_string(),
            molecular_diffusivity: 1e-9,
            background_value: 0.0,
            decay_rate: 0.0,
            settling_velocity: 0.0,
            enabled: true,
        }
    }

    /// 使用 Builder 模式设置分子扩散系数
    pub fn with_diffusivity(mut self, diffusivity: f64) -> Self {
        self.molecular_diffusivity = diffusivity;
        self
    }

    /// 使用 Builder 模式设置背景值
    pub fn with_background(mut self, value: f64) -> Self {
        self.background_value = value;
        self
    }

    /// 使用 Builder 模式设置衰减率
    pub fn with_decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = rate;
        self
    }

    /// 使用 Builder 模式设置沉降速度
    pub fn with_settling_velocity(mut self, velocity: f64) -> Self {
        self.settling_velocity = velocity;
        self
    }
}

impl Default for TracerProperties {
    fn default() -> Self {
        Self::salinity()
    }
}

// ============================================================
// 示踪剂场
// ============================================================

/// 单个示踪剂的场数据
///
/// 存储示踪剂在所有计算单元上的浓度值。
#[derive(Debug, Clone)]
pub struct TracerField {
    /// 示踪剂属性
    properties: TracerProperties,

    /// 浓度场 [单位取决于示踪剂类型]
    ///
    /// 索引与计算单元对应。
    concentration: Vec<f64>,

    /// 守恒量场 (h * C)
    ///
    /// 用于有限体积法计算。
    conserved: Vec<f64>,

    /// 右手项累加器 (dC/dt)
    rhs: Vec<f64>,
}

impl TracerField {
    /// 创建新的示踪剂场
    ///
    /// # 参数
    /// - `properties`: 示踪剂属性
    /// - `n_cells`: 计算单元数量
    pub fn new(properties: TracerProperties, n_cells: usize) -> Self {
        let background = properties.background_value;
        Self {
            properties,
            concentration: vec![background; n_cells],
            conserved: vec![0.0; n_cells], // 需要与水深配合初始化
            rhs: vec![0.0; n_cells],
        }
    }

    /// 从浓度数组创建
    pub fn from_concentration(properties: TracerProperties, concentration: Vec<f64>) -> Self {
        let n = concentration.len();
        Self {
            properties,
            concentration,
            conserved: vec![0.0; n],
            rhs: vec![0.0; n],
        }
    }

    /// 获取示踪剂属性
    pub fn properties(&self) -> &TracerProperties {
        &self.properties
    }

    /// 获取示踪剂类型
    pub fn tracer_type(&self) -> TracerType {
        self.properties.tracer_type
    }

    /// 获取单元数量
    pub fn len(&self) -> usize {
        self.concentration.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.concentration.is_empty()
    }

    /// 获取单元浓度（只读）
    pub fn concentration(&self, cell_idx: usize) -> f64 {
        self.concentration[cell_idx]
    }

    /// 获取浓度场切片
    pub fn concentration_slice(&self) -> &[f64] {
        &self.concentration
    }

    /// 获取浓度场可变切片
    pub fn concentration_slice_mut(&mut self) -> &mut [f64] {
        &mut self.concentration
    }

    /// 获取守恒量（h * C）
    pub fn conserved(&self, cell_idx: usize) -> f64 {
        self.conserved[cell_idx]
    }

    /// 获取守恒量场切片
    pub fn conserved_slice(&self) -> &[f64] {
        &self.conserved
    }

    /// 获取守恒量场可变切片
    pub fn conserved_slice_mut(&mut self) -> &mut [f64] {
        &mut self.conserved
    }

    /// 设置单元浓度
    pub fn set_concentration(&mut self, cell_idx: usize, value: f64) {
        self.concentration[cell_idx] = value;
    }

    /// 设置守恒量
    pub fn set_conserved(&mut self, cell_idx: usize, value: f64) {
        self.conserved[cell_idx] = value;
    }

    /// 从水深更新守恒量
    ///
    /// 用于初始化或重置：conserved = h * concentration
    pub fn update_conserved_from_depth(&mut self, water_depths: &[f64]) {
        debug_assert_eq!(water_depths.len(), self.concentration.len());
        for i in 0..self.concentration.len() {
            self.conserved[i] = water_depths[i] * self.concentration[i];
        }
    }

    /// 从守恒量更新浓度
    ///
    /// 用于时间步进后：concentration = conserved / h
    pub fn update_concentration_from_conserved(&mut self, water_depths: &[f64], h_min: f64) {
        debug_assert_eq!(water_depths.len(), self.concentration.len());
        for i in 0..self.concentration.len() {
            let h = water_depths[i].max(h_min);
            self.concentration[i] = self.conserved[i] / h;
        }
    }

    /// 获取 RHS 切片（用于时间积分）
    pub fn rhs_slice(&self) -> &[f64] {
        &self.rhs
    }

    /// 获取 RHS 可变切片
    pub fn rhs_slice_mut(&mut self) -> &mut [f64] {
        &mut self.rhs
    }

    /// 清零 RHS
    pub fn clear_rhs(&mut self) {
        self.rhs.fill(0.0);
    }

    /// 累加 RHS
    pub fn add_rhs(&mut self, cell_idx: usize, value: f64) {
        self.rhs[cell_idx] += value;
    }

    /// 使用显式欧拉格式更新守恒量
    ///
    /// conserved += dt * rhs
    ///
    /// # 参数
    /// - `dt`: 时间步长 [s]
    pub fn apply_euler_update(&mut self, dt: f64) {
        for i in 0..self.conserved.len() {
            self.conserved[i] += dt * self.rhs[i];
        }
    }

    /// 应用衰减（一阶衰减模型）
    ///
    /// # 参数
    /// - `dt`: 时间步长 [s]
    pub fn apply_decay(&mut self, dt: f64) {
        let k = self.properties.decay_rate;
        if k > 0.0 {
            let factor = (-k * dt).exp();
            for c in &mut self.concentration {
                *c *= factor;
            }
            for hc in &mut self.conserved {
                *hc *= factor;
            }
        }
    }

    /// 计算场统计量
    pub fn statistics(&self) -> TracerFieldStats {
        if self.concentration.is_empty() {
            return TracerFieldStats::default();
        }

        let mut min = f64::MAX;
        let mut max = f64::MIN;
        let mut sum = 0.0;

        for &c in &self.concentration {
            min = min.min(c);
            max = max.max(c);
            sum += c;
        }

        TracerFieldStats {
            min,
            max,
            mean: sum / self.concentration.len() as f64,
        }
    }

    /// 限制浓度在物理范围内
    ///
    /// # 参数
    /// - `c_min`: 最小浓度（通常为 0）
    /// - `c_max`: 最大浓度（可选）
    pub fn clamp_concentration(&mut self, c_min: f64, c_max: Option<f64>) {
        for c in &mut self.concentration {
            *c = c.max(c_min);
            if let Some(max_val) = c_max {
                *c = c.min(max_val);
            }
        }
    }
}

/// 示踪剂场统计量
#[derive(Debug, Clone, Copy, Default)]
pub struct TracerFieldStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

// ============================================================
// 多示踪剂状态
// ============================================================

/// 多示踪剂集合状态
///
/// 管理多个示踪剂的场数据。
#[derive(Debug, Clone)]
pub struct TracerState {
    /// 示踪剂场集合（按类型索引）
    fields: HashMap<TracerType, TracerField>,

    /// 类型列表（保持添加顺序）
    types: Vec<TracerType>,

    /// 计算单元数量
    n_cells: usize,
}

impl TracerState {
    /// 创建新的多示踪剂状态
    pub fn new(n_cells: usize) -> Self {
        Self {
            fields: HashMap::new(),
            types: Vec::new(),
            n_cells,
        }
    }

    /// 添加示踪剂
    ///
    /// # 参数
    /// - `properties`: 示踪剂属性
    ///
    /// # 返回
    /// 如果类型已存在则返回错误
    pub fn add_tracer(&mut self, properties: TracerProperties) -> Result<(), TracerError> {
        let tracer_type = properties.tracer_type;
        if self.fields.contains_key(&tracer_type) {
            return Err(TracerError::DuplicateType(tracer_type));
        }

        let field = TracerField::new(properties, self.n_cells);
        self.fields.insert(tracer_type, field);
        self.types.push(tracer_type);
        Ok(())
    }

    /// 获取示踪剂场
    pub fn get(&self, tracer_type: TracerType) -> Option<&TracerField> {
        self.fields.get(&tracer_type)
    }

    /// 获取示踪剂场（可变）
    pub fn get_mut(&mut self, tracer_type: TracerType) -> Option<&mut TracerField> {
        self.fields.get_mut(&tracer_type)
    }

    /// 检查是否包含指定类型
    pub fn contains(&self, tracer_type: TracerType) -> bool {
        self.fields.contains_key(&tracer_type)
    }

    /// 获取示踪剂数量
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// 获取所有示踪剂类型
    pub fn types(&self) -> &[TracerType] {
        &self.types
    }

    /// 遍历所有场
    pub fn iter(&self) -> impl Iterator<Item = (&TracerType, &TracerField)> {
        self.fields.iter()
    }

    /// 遍历所有场（可变）
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&TracerType, &mut TracerField)> {
        self.fields.iter_mut()
    }

    /// 从水深更新所有守恒量
    pub fn update_conserved_from_depth(&mut self, water_depths: &[f64]) {
        for field in self.fields.values_mut() {
            field.update_conserved_from_depth(water_depths);
        }
    }

    /// 从守恒量更新所有浓度
    pub fn update_concentration_from_conserved(&mut self, water_depths: &[f64], h_min: f64) {
        for field in self.fields.values_mut() {
            field.update_concentration_from_conserved(water_depths, h_min);
        }
    }

    /// 清零所有 RHS
    pub fn clear_all_rhs(&mut self) {
        for field in self.fields.values_mut() {
            field.clear_rhs();
        }
    }

    /// 应用衰减到所有示踪剂
    pub fn apply_all_decay(&mut self, dt: f64) {
        for field in self.fields.values_mut() {
            field.apply_decay(dt);
        }
    }
}

// ============================================================
// 错误类型
// ============================================================

/// 示踪剂模块错误
#[derive(Debug, Error)]
pub enum TracerError {
    /// 重复的示踪剂类型
    #[error("示踪剂类型 {0:?} 已存在")]
    DuplicateType(TracerType),

    /// 示踪剂未找到
    #[error("示踪剂类型 {0:?} 未找到")]
    NotFound(TracerType),

    /// 数组大小不匹配
    #[error("数组大小不匹配: 期望 {expected}, 实际 {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    /// 无效的浓度值
    #[error("无效的浓度值: {0}")]
    InvalidValue(f64),
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_tracer_type() {
        let t = TracerType::Salinity;
        assert_eq!(t.name(), "salinity");
        assert!(t.is_passive());
        assert!(!t.has_source_terms());

        let t = TracerType::Sediment;
        assert!(!t.is_passive());
        assert!(t.has_source_terms());
    }

    #[test]
    fn test_tracer_properties() {
        let props = TracerProperties::salinity()
            .with_background(30.0)
            .with_diffusivity(2e-9);

        assert_eq!(props.tracer_type, TracerType::Salinity);
        assert!(approx_eq(props.background_value, 30.0));
        assert!(approx_eq(props.molecular_diffusivity, 2e-9));
    }

    #[test]
    fn test_tracer_field_creation() {
        let props = TracerProperties::salinity();
        let field = TracerField::new(props, 100);

        assert_eq!(field.len(), 100);
        assert_eq!(field.tracer_type(), TracerType::Salinity);
        assert!(approx_eq(field.concentration(0), 35.0)); // 背景值
    }

    #[test]
    fn test_tracer_field_conserved() {
        let props = TracerProperties::salinity().with_background(10.0);
        let mut field = TracerField::new(props, 3);

        // 假设水深
        let depths = vec![1.0, 2.0, 3.0];
        field.update_conserved_from_depth(&depths);

        assert!(approx_eq(field.conserved(0), 10.0)); // 1.0 * 10
        assert!(approx_eq(field.conserved(1), 20.0)); // 2.0 * 10
        assert!(approx_eq(field.conserved(2), 30.0)); // 3.0 * 10
    }

    #[test]
    fn test_tracer_field_decay() {
        let props = TracerProperties::salinity()
            .with_background(100.0)
            .with_decay_rate(0.1);
        let mut field = TracerField::new(props, 1);

        field.apply_decay(1.0);
        // 应该是 100 * exp(-0.1) ≈ 90.48
        assert!((field.concentration(0) - 90.48373903808578).abs() < 1e-6);
    }

    #[test]
    fn test_tracer_field_statistics() {
        let props = TracerProperties::salinity();
        let mut field = TracerField::from_concentration(props, vec![10.0, 20.0, 30.0]);

        let stats = field.statistics();
        assert!(approx_eq(stats.min, 10.0));
        assert!(approx_eq(stats.max, 30.0));
        assert!(approx_eq(stats.mean, 20.0));

        // 测试限制
        field.clamp_concentration(15.0, Some(25.0));
        assert!(approx_eq(field.concentration(0), 15.0));
        assert!(approx_eq(field.concentration(1), 20.0));
        assert!(approx_eq(field.concentration(2), 25.0));
    }

    #[test]
    fn test_tracer_state() {
        let mut state = TracerState::new(100);

        state.add_tracer(TracerProperties::salinity()).unwrap();
        state.add_tracer(TracerProperties::temperature()).unwrap();

        assert_eq!(state.len(), 2);
        assert!(state.contains(TracerType::Salinity));
        assert!(state.contains(TracerType::Temperature));
        assert!(!state.contains(TracerType::Sediment));
    }

    #[test]
    fn test_duplicate_tracer_error() {
        let mut state = TracerState::new(100);

        state.add_tracer(TracerProperties::salinity()).unwrap();
        let result = state.add_tracer(TracerProperties::salinity());

        assert!(result.is_err());
        if let Err(TracerError::DuplicateType(t)) = result {
            assert_eq!(t, TracerType::Salinity);
        } else {
            panic!("Expected DuplicateType error");
        }
    }

    #[test]
    fn test_tracer_state_update() {
        let mut state = TracerState::new(3);
        state.add_tracer(TracerProperties::salinity().with_background(10.0)).unwrap();

        let depths = vec![1.0, 2.0, 3.0];
        state.update_conserved_from_depth(&depths);

        let field = state.get(TracerType::Salinity).unwrap();
        assert!(approx_eq(field.conserved(1), 20.0));
    }
}
