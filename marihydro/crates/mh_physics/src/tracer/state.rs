// crates/mh_physics/src/tracer/state.rs

//! 示踪剂状态模块
//!
//! 本模块定义示踪剂相关的状态类型：
//! - TracerType: 示踪剂类型枚举
//! - TracerProperties: 示踪剂物理属性
//! - TracerField: 单个示踪剂的场数据（泛型版本）
//! - TracerState: 多示踪剂集合状态
//!
//! # 设计原则
//!
//! 1. **单轨泛型**: 所有接口基于 Backend trait，无 Legacy f64 双轨实现
//! 2. **Backend 抽象**: 使用 `B::Buffer<B::Scalar>` 存储数据，支持 CPU/GPU
//! 3. **强类型索引**: 使用 `CellIndex` 而非裸 usize
//!
//! # 概念说明
//!
//! 示踪剂（Tracer）是指随水流运移的物质，包括：
//! - 被动示踪剂：盐度、温度等（不影响水动力）
//! - 主动示踪剂：泥沙等（可能影响水密度和流动）

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use num_traits::Float;

use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar as Scalar};

// ============================================================
// 示踪剂类型
// ============================================================

/// 示踪剂类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum TracerType {
    /// 盐度 [PSU 或 kg/m³]
    #[default]
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


// ============================================================
// 示踪剂属性（泛型化）
// ============================================================

/// 示踪剂物理属性
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerProperties<S: Scalar> {
    /// 示踪剂类型
    pub tracer_type: TracerType,

    /// 示踪剂名称（用于显示）
    pub name: String,

    /// 单位
    pub unit: String,

    /// 分子扩散系数 [m²/s]
    pub molecular_diffusivity: S,

    /// 背景浓度（用于边界和初始化）
    pub background_value: S,

    /// 衰减系数 [1/s]
    ///
    /// 用于简单的一阶衰减模型：dC/dt = -k * C
    pub decay_rate: S,

    /// 沉降速度 [m/s]
    ///
    /// 仅适用于泥沙等可沉降物质，正值表示向下沉降。
    pub settling_velocity: S,

    /// 是否启用
    pub enabled: bool,
}

impl<S: Scalar> TracerProperties<S> {
    /// 创建默认盐度示踪剂
    pub fn salinity<B: Backend<Scalar = S>>(backend: &B) -> Self {
        Self {
            tracer_type: TracerType::Salinity,
            name: "Salinity".to_string(),
            unit: "PSU".to_string(),
            molecular_diffusivity: backend.scalar_from_f64(1.5e-9),
            background_value: backend.scalar_from_f64(35.0),
            decay_rate: S::ZERO,
            settling_velocity: S::ZERO,
            enabled: true,
        }
    }

    /// 创建默认温度示踪剂
    pub fn temperature<B: Backend<Scalar = S>>(backend: &B) -> Self {
        Self {
            tracer_type: TracerType::Temperature,
            name: "Temperature".to_string(),
            unit: "°C".to_string(),
            molecular_diffusivity: backend.scalar_from_f64(1.4e-7),
            background_value: backend.scalar_from_f64(20.0),
            decay_rate: S::ZERO,
            settling_velocity: S::ZERO,
            enabled: true,
        }
    }

    /// 创建默认泥沙示踪剂
    pub fn sediment<B: Backend<Scalar = S>>(backend: &B) -> Self {
        Self {
            tracer_type: TracerType::Sediment,
            name: "Suspended Sediment".to_string(),
            unit: "kg/m³".to_string(),
            molecular_diffusivity: S::ZERO, // 主要靠湍流扩散
            background_value: S::ZERO,
            decay_rate: S::ZERO,
            settling_velocity: backend.scalar_from_f64(1e-4), // 0.1 mm/s
            enabled: true,
        }
    }

    /// 创建自定义示踪剂
    pub fn custom<B: Backend<Scalar = S>>(backend: &B, id: u16, name: &str, unit: &str) -> Self {
        Self {
            tracer_type: TracerType::Custom(id),
            name: name.to_string(),
            unit: unit.to_string(),
            molecular_diffusivity: backend.scalar_from_f64(1e-9),
            background_value: S::ZERO,
            decay_rate: S::ZERO,
            settling_velocity: S::ZERO,
            enabled: true,
        }
    }

    /// 使用 Builder 模式设置分子扩散系数
    pub fn with_diffusivity(mut self, diffusivity: S) -> Self {
        self.molecular_diffusivity = diffusivity;
        self
    }

    /// 使用 Builder 模式设置背景值
    pub fn with_background(mut self, value: S) -> Self {
        self.background_value = value;
        self
    }

    /// 使用 Builder 模式设置衰减率
    pub fn with_decay_rate(mut self, rate: S) -> Self {
        self.decay_rate = rate;
        self
    }

    /// 使用 Builder 模式设置沉降速度
    pub fn with_settling_velocity(mut self, velocity: S) -> Self {
        self.settling_velocity = velocity;
        self
    }
}


// ============================================================
// 示踪剂场（Backend 泛型化）
// ============================================================

/// 示踪剂场统计量
#[derive(Debug, Clone, Copy, Default)]
pub struct TracerFieldStats<S: Scalar> {
    pub min: S,
    pub max: S,
    pub mean: S,
}

impl<S: Scalar> Default for TracerFieldStats<S> {
    fn default() -> Self {
        Self {
            min: S::ZERO,
            max: S::ZERO,
            mean: S::ZERO,
        }
    }
}

/// 单个示踪剂的场数据
///
/// 使用 Backend trait 抽象存储，支持 CPU/GPU 后端。
///
/// # 类型参数
///
/// - `B`: 计算后端类型，必须实现 `Backend` trait
#[derive(Clone)]
pub struct TracerField<B: Backend> {
    /// 示踪剂属性
    properties: TracerProperties<B::Scalar>,
    /// 浓度场 [单位取决于示踪剂类型]
    concentration: B::Buffer<B::Scalar>,
    /// 守恒量场 (h * C)
    conserved: B::Buffer<B::Scalar>,
    /// 右手项累加器 (dC/dt)
    rhs: B::Buffer<B::Scalar>,
    /// 单元数量
    n_cells: usize,
    /// 后端实例
    backend: B,
}

impl<B: Backend> std::fmt::Debug for TracerField<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracerField")
            .field("properties", &self.properties)
            .field("n_cells", &self.n_cells)
            .finish()
    }
}

impl<B: Backend> TracerField<B> {
    /// 使用后端实例创建新的示踪剂场
    pub fn new_with_backend(backend: B, properties: TracerProperties<B::Scalar>, n_cells: usize) -> Self {
        let background = properties.background_value;
        let mut concentration = backend.alloc(n_cells);
        concentration.fill(background);
        let mut conserved = backend.alloc(n_cells);
        conserved.fill(B::Scalar::ZERO);
        let mut rhs = backend.alloc(n_cells);
        rhs.fill(B::Scalar::ZERO);
        
        Self {
            properties,
            concentration,
            conserved,
            rhs,
            n_cells,
            backend,
        }
    }
    
    /// 获取示踪剂属性
    pub fn properties(&self) -> &TracerProperties<B::Scalar> {
        &self.properties
    }
    
    /// 获取示踪剂类型
    pub fn tracer_type(&self) -> TracerType {
        self.properties.tracer_type
    }
    
    /// 获取单元数量
    pub fn len(&self) -> usize {
        self.n_cells
    }
    
    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.n_cells == 0
    }
    
    /// 获取后端引用
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 获取浓度场引用
    pub fn concentration(&self) -> &B::Buffer<B::Scalar> {
        &self.concentration
    }
    
    /// 获取浓度场可变引用
    pub fn concentration_mut(&mut self) -> &mut B::Buffer<B::Scalar> {
        &mut self.concentration
    }
    
    /// 获取守恒量场引用
    pub fn conserved(&self) -> &B::Buffer<B::Scalar> {
        &self.conserved
    }
    
    /// 获取守恒量场可变引用
    pub fn conserved_mut(&mut self) -> &mut B::Buffer<B::Scalar> {
        &mut self.conserved
    }
    
    /// 获取 RHS 引用
    pub fn rhs(&self) -> &B::Buffer<B::Scalar> {
        &self.rhs
    }
    
    /// 获取 RHS 可变引用
    pub fn rhs_mut(&mut self) -> &mut B::Buffer<B::Scalar> {
        &mut self.rhs
    }
    
    /// 清零 RHS
    pub fn clear_rhs(&mut self) {
        self.rhs.fill(B::Scalar::ZERO);
    }
    
    /// 重置为背景值
    pub fn reset(&mut self) {
        let background = self.properties.background_value;
        self.concentration.fill(background);
        self.conserved.fill(B::Scalar::ZERO);
        self.rhs.fill(B::Scalar::ZERO);
    }
    
    // ========== 通用访问方法（所有 Backend 可用） ==========
    
    /// 获取浓度切片（需要 CPU 可访问）
    #[inline]
    pub fn concentration_slice(&self) -> Result<&[B::Scalar], TracerError> {
        self.concentration
            .try_as_slice()
            .ok_or(TracerError::BackendAccess { field: "concentration" })
    }
    
    /// 获取浓度可变切片（需要 CPU 可访问）
    #[inline]
    pub fn concentration_slice_mut(&mut self) -> Result<&mut [B::Scalar], TracerError> {
        self.concentration
            .try_as_slice_mut()
            .ok_or(TracerError::BackendAccess { field: "concentration" })
    }
    
    /// 获取守恒量切片（需要 CPU 可访问）
    #[inline]
    pub fn conserved_slice(&self) -> Result<&[B::Scalar], TracerError> {
        self.conserved
            .try_as_slice()
            .ok_or(TracerError::BackendAccess { field: "conserved" })
    }
    
    /// 获取守恒量可变切片（需要 CPU 可访问）
    #[inline]
    pub fn conserved_slice_mut(&mut self) -> Result<&mut [B::Scalar], TracerError> {
        self.conserved
            .try_as_slice_mut()
            .ok_or(TracerError::BackendAccess { field: "conserved" })
    }
    
    /// 获取 RHS 切片（需要 CPU 可访问）
    #[inline]
    pub fn rhs_slice(&self) -> Result<&[B::Scalar], TracerError> {
        self.rhs
            .try_as_slice()
            .ok_or(TracerError::BackendAccess { field: "rhs" })
    }
    
    /// 获取 RHS 可变切片（需要 CPU 可访问）
    #[inline]
    pub fn rhs_slice_mut(&mut self) -> Result<&mut [B::Scalar], TracerError> {
        self.rhs
            .try_as_slice_mut()
            .ok_or(TracerError::BackendAccess { field: "rhs" })
    }
    
    /// 累加值到 RHS（需要 CPU 可访问）
    #[inline]
    pub fn add_rhs(&mut self, cell_idx: usize, value: B::Scalar) -> Result<(), TracerError> {
        let rhs = self.rhs_slice_mut()?;
        rhs[cell_idx] = rhs[cell_idx] + value;
        Ok(())
    }
    
    /// 从守恒量更新浓度（需要 CPU 可访问）
    pub fn update_concentration_from_conserved(
        &mut self,
        water_depths: &B::Buffer<B::Scalar>,
        h_min: B::Scalar,
    ) -> Result<(), TracerError> {
        if water_depths.len() != self.n_cells {
            return Err(TracerError::SizeMismatch {
                expected: self.n_cells,
                actual: water_depths.len(),
            });
        }
        let depths = water_depths
            .try_as_slice()
            .ok_or(TracerError::BackendAccess { field: "water_depths" })?;
        let concentration = self.concentration_slice_mut()?;
        let conserved = self.conserved_slice()?;
        for i in 0..self.n_cells {
            let h = if depths[i] > h_min { depths[i] } else { h_min };
            concentration[i] = conserved[i] / h;
        }
        Ok(())
    }
    
    /// 使用显式欧拉格式更新守恒量（需要 CPU 可访问）
    pub fn apply_euler_update(&mut self, dt: B::Scalar) -> Result<(), TracerError> {
        let conserved = self.conserved_slice_mut()?;
        let rhs = self.rhs_slice()?;
        for i in 0..self.n_cells {
            conserved[i] = conserved[i] + dt * rhs[i];
        }
        Ok(())
    }
    
    /// 限制浓度在物理范围内（需要 CPU 可访问）
    pub fn clamp_concentration(&mut self, c_min: B::Scalar, c_max: Option<B::Scalar>) -> Result<(), TracerError> {
        let concentration = self.concentration_slice_mut()?;
        for c in concentration.iter_mut() {
            if *c < c_min {
                *c = c_min;
            }
            if let Some(max_val) = c_max {
                if *c > max_val {
                    *c = max_val;
                }
            }
        }
        Ok(())
    }

    /// 从水深更新守恒量（需要 CPU 可访问）
    pub fn update_conserved_from_depth(
        &mut self,
        water_depths: &B::Buffer<B::Scalar>,
    ) -> Result<(), TracerError> {
        if water_depths.len() != self.n_cells {
            return Err(TracerError::SizeMismatch {
                expected: self.n_cells,
                actual: water_depths.len(),
            });
        }
        let depths = water_depths
            .try_as_slice()
            .ok_or(TracerError::BackendAccess { field: "water_depths" })?;
        let concentration = self.concentration_slice()?;
        let conserved = self.conserved_slice_mut()?;
        for i in 0..self.n_cells {
            conserved[i] = depths[i] * concentration[i];
        }
        Ok(())
    }

    /// 应用衰减
    pub fn apply_decay(&mut self, dt: B::Scalar) -> Result<(), TracerError> {
        let k = self.properties.decay_rate;
        if k > B::Scalar::ZERO {
            let factor = (-k * dt).exp();
            let concentration = self.concentration_slice_mut()?;
            for c in concentration.iter_mut() {
                *c *= factor;
            }
            let conserved = self.conserved_slice_mut()?;
            for hc in conserved.iter_mut() {
                *hc *= factor;
            }
        }
        Ok(())
    }

    /// 计算场统计量（使用 Backend 归约）
    pub fn statistics(&self) -> TracerFieldStats<B::Scalar> {
        if self.n_cells == 0 {
            return TracerFieldStats::default();
        }
        let min = self.backend.reduce_min(&self.concentration);
        let max = self.backend.reduce_max(&self.concentration);
        let sum = self.backend.reduce_sum(&self.concentration);
        TracerFieldStats {
            min,
            max,
            mean: sum / self.backend.scalar_from_f64(self.n_cells as f64),
        }
    }
}

// ============================================================
// 多示踪剂状态
// ============================================================

/// 多示踪剂集合状态
///
/// 管理多个示踪剂的场数据。
#[derive(Clone)]
pub struct TracerState<B: Backend> {
    /// 示踪剂场集合（按类型索引）
    fields: HashMap<TracerType, TracerField<B>>,

    /// 类型列表（保持添加顺序）
    types: Vec<TracerType>,

    /// 计算单元数量
    n_cells: usize,
    
    /// 后端实例
    backend: B,
}

impl<B: Backend> std::fmt::Debug for TracerState<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracerState")
            .field("n_tracers", &self.types.len())
            .field("types", &self.types)
            .field("n_cells", &self.n_cells)
            .finish()
    }
}

impl<B: Backend> TracerState<B> {
    /// 创建新的多示踪剂状态
    pub fn new_with_backend(backend: B, n_cells: usize) -> Self {
        Self {
            fields: HashMap::new(),
            types: Vec::new(),
            n_cells,
            backend,
        }
    }

    /// 添加示踪剂
    ///
    /// # 参数
    /// - `properties`: 示踪剂属性
    ///
    /// # 返回
    /// 如果类型已存在则返回错误
    pub fn add_tracer(&mut self, properties: TracerProperties<B::Scalar>) -> Result<(), TracerError> {
        let tracer_type = properties.tracer_type;
        if self.fields.contains_key(&tracer_type) {
            return Err(TracerError::DuplicateType(tracer_type));
        }

        if let TracerType::Custom(id) = tracer_type {
            if self.types.iter().any(|t| matches!(t, TracerType::Custom(existing) if *existing == id)) {
                return Err(TracerError::DuplicateCustomId(id));
            }
        }

        // 尺寸一致性保护
        if self.n_cells == 0 {
            return Err(TracerError::SizeMismatch { expected: 0, actual: 0 });
        }

        let field = TracerField::new_with_backend(self.backend.clone(), properties, self.n_cells);
        self.fields.insert(tracer_type, field);
        self.types.push(tracer_type);
        Ok(())
    }

    /// 校验示踪剂状态
    pub fn validate(&self) -> Result<(), TracerError> {
        for (tracer_type, field) in &self.fields {
            if field.len() != self.n_cells {
                return Err(TracerError::SizeMismatch {
                    expected: self.n_cells,
                    actual: field.len(),
                });
            }

            if let Some(conc) = field.concentration().try_as_slice() {
                if conc.iter().any(|v| !v.is_finite()) {
                    return Err(TracerError::NonFiniteValue { tracer: *tracer_type });
                }
            }

            if let Some(conserved) = field.conserved().try_as_slice() {
                if conserved.iter().any(|v| !v.is_finite()) {
                    return Err(TracerError::NonFiniteValue { tracer: *tracer_type });
                }
            }

            if let Some(rhs) = field.rhs().try_as_slice() {
                if rhs.iter().any(|v| !v.is_finite()) {
                    return Err(TracerError::NonFiniteValue { tracer: *tracer_type });
                }
            }
        }
        Ok(())
    }

    /// 获取示踪剂场
    pub fn get(&self, tracer_type: TracerType) -> Option<&TracerField<B>> {
        self.fields.get(&tracer_type)
    }

    /// 获取示踪剂场（可变）
    pub fn get_mut(&mut self, tracer_type: TracerType) -> Option<&mut TracerField<B>> {
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
    pub fn iter(&self) -> impl Iterator<Item = (&TracerType, &TracerField<B>)> {
        self.fields.iter()
    }

    /// 遍历所有场（可变）
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&TracerType, &mut TracerField<B>)> {
        self.fields.iter_mut()
    }

    /// 清零所有 RHS
    pub fn clear_all_rhs(&mut self) {
        for field in self.fields.values_mut() {
            field.clear_rhs();
        }
    }
}

impl<B: Backend> TracerState<B> {
    /// 从水深更新所有守恒量
    pub fn update_conserved_from_depth(
        &mut self,
        water_depths: &B::Buffer<B::Scalar>,
    ) -> Result<(), TracerError> {
        for field in self.fields.values_mut() {
            field.update_conserved_from_depth(water_depths)?;
        }
        Ok(())
    }

    /// 从守恒量更新所有浓度
    pub fn update_concentration_from_conserved(
        &mut self,
        water_depths: &B::Buffer<B::Scalar>,
        h_min: B::Scalar,
    ) -> Result<(), TracerError> {
        for field in self.fields.values_mut() {
            field.update_concentration_from_conserved(water_depths, h_min)?;
        }
        Ok(())
    }

    /// 应用衰减到所有示踪剂
    pub fn apply_all_decay(&mut self, dt: B::Scalar) -> Result<(), TracerError> {
        for field in self.fields.values_mut() {
            field.apply_decay(dt)?;
        }
        Ok(())
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

    /// 重复的自定义示踪剂 ID
    #[error("自定义示踪剂 ID {0} 已存在")]
    DuplicateCustomId(u16),

    /// 示踪剂未找到
    #[error("示踪剂类型 {0:?} 未找到")]
    NotFound(TracerType),

    /// 数组大小不匹配
    #[error("数组大小不匹配: 期望 {expected}, 实际 {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    /// 无效的浓度值
    #[error("无效的浓度值: {0}")]
    InvalidValue(f64),

    /// 非有限值
    #[error("示踪剂 {tracer:?} 包含非有限值")]
    NonFiniteValue { tracer: TracerType },
    /// 后端缓冲区不可访问
    #[error("后端缓冲区不可访问: {field}")]
    BackendAccess { field: &'static str },
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

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
        let backend = CpuBackend::<f64>::new();
        let props: TracerProperties<f64> = TracerProperties::salinity(&backend)
            .with_background(30.0)
            .with_diffusivity(2e-9);

        assert_eq!(props.tracer_type, TracerType::Salinity);
        assert!(approx_eq(props.background_value, 30.0));
        assert!(approx_eq(props.molecular_diffusivity, 2e-9));
    }

    #[test]
    fn test_tracer_field_creation() {
        let backend = CpuBackend::<f64>::new();
        let props: TracerProperties<f64> = TracerProperties::salinity(&backend);
        let field = TracerField::<CpuBackend<f64>>::new_with_backend(backend, props, 100);

        assert_eq!(field.len(), 100);
        assert_eq!(field.tracer_type(), TracerType::Salinity);
        assert!(approx_eq(field.concentration_slice().unwrap()[0], 35.0)); // 背景值
    }

    #[test]
    fn test_tracer_field_conserved() {
        let backend = CpuBackend::<f64>::new();
        let props: TracerProperties<f64> = TracerProperties::salinity(&backend).with_background(10.0);
        let mut field = TracerField::<CpuBackend<f64>>::new_with_backend(backend.clone(), props, 3);

        // 假设水深
        let mut depths = backend.alloc(3);
        depths.copy_from_slice(&[1.0, 2.0, 3.0]);
        field.update_conserved_from_depth(&depths).unwrap();

        let conserved = field.conserved_slice().unwrap();
        assert!(approx_eq(conserved[0], 10.0)); // 1.0 * 10
        assert!(approx_eq(conserved[1], 20.0)); // 2.0 * 10
        assert!(approx_eq(conserved[2], 30.0)); // 3.0 * 10
    }

    #[test]
    fn test_tracer_field_decay() {
        let backend = CpuBackend::<f64>::new();
        let props: TracerProperties<f64> = TracerProperties::salinity(&backend)
            .with_background(100.0)
            .with_decay_rate(0.1);
        let mut field = TracerField::<CpuBackend<f64>>::new_with_backend(backend, props, 1);

        field.apply_decay(1.0).unwrap();
        // 精确指数衰减解：C = C0 * exp(-k * dt)
        let expected = 100.0 * (-0.1_f64).exp();
        assert!((field.concentration_slice().unwrap()[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn test_tracer_field_statistics() {
        let backend = CpuBackend::<f64>::new();
        let props: TracerProperties<f64> = TracerProperties::salinity(&backend);
        let mut field = TracerField::<CpuBackend<f64>>::new_with_backend(backend, props, 3);
        
        let concentration = field.concentration_slice_mut().unwrap();
        concentration[0] = 10.0;
        concentration[1] = 20.0;
        concentration[2] = 30.0;

        let stats = field.statistics();
        assert!(approx_eq(stats.min, 10.0));
        assert!(approx_eq(stats.max, 30.0));
        assert!(approx_eq(stats.mean, 20.0));

        // 测试限制
        field.clamp_concentration(15.0, Some(25.0)).unwrap();
        let concentration = field.concentration_slice().unwrap();
        assert!(approx_eq(concentration[0], 15.0));
        assert!(approx_eq(concentration[1], 20.0));
        assert!(approx_eq(concentration[2], 25.0));
    }

    #[test]
    fn test_tracer_state() {
        let backend = CpuBackend::<f64>::new();
        let mut state = TracerState::<CpuBackend<f64>>::new_with_backend(backend.clone(), 100);

        state.add_tracer(TracerProperties::salinity(&backend)).unwrap();
        state.add_tracer(TracerProperties::temperature(&backend)).unwrap();

        assert_eq!(state.len(), 2);
        assert!(state.contains(TracerType::Salinity));
        assert!(state.contains(TracerType::Temperature));
        assert!(!state.contains(TracerType::Sediment));
    }

    #[test]
    fn test_duplicate_tracer_error() {
        let backend = CpuBackend::<f64>::new();
        let mut state = TracerState::<CpuBackend<f64>>::new_with_backend(backend.clone(), 100);

        state.add_tracer(TracerProperties::salinity(&backend)).unwrap();
        let result = state.add_tracer(TracerProperties::salinity(&backend));

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(TracerError::DuplicateType(TracerType::Salinity))
        ));
    }

    #[test]
    fn test_tracer_state_update() {
        let backend = CpuBackend::<f64>::new();
        let mut state = TracerState::<CpuBackend<f64>>::new_with_backend(backend.clone(), 3);
        state
            .add_tracer(TracerProperties::salinity(&backend).with_background(10.0))
            .unwrap();

        let mut depths = backend.alloc(3);
        depths.copy_from_slice(&[1.0, 2.0, 3.0]);
        state.update_conserved_from_depth(&depths).unwrap();

        let field = state.get(TracerType::Salinity).unwrap();
        assert!(approx_eq(field.conserved_slice().unwrap()[1], 20.0));
    }
}
