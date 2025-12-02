// src-tauri/src/marihydro/core/types/field_types.rs

//! 场类型定义
//!
//! 提供标量场和向量场的抽象，用于存储和操作物理量分布。

use glam::DVec2;
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

/// 标量场
///
/// 存储定义在网格单元上的标量量（如水深、压力等）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarField {
    values: Vec<f64>,
    name: String,
    unit: Option<String>,
}

impl ScalarField {
    /// 创建新的标量场
    pub fn new(name: impl Into<String>, size: usize) -> Self {
        Self {
            values: vec![0.0; size],
            name: name.into(),
            unit: None,
        }
    }

    /// 从现有数据创建
    pub fn from_values(name: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            values,
            name: name.into(),
            unit: None,
        }
    }

    /// 创建常量场
    pub fn constant(name: impl Into<String>, size: usize, value: f64) -> Self {
        Self {
            values: vec![value; size],
            name: name.into(),
            unit: None,
        }
    }

    /// 设置物理单位
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// 获取场名称
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 获取物理单位
    pub fn unit(&self) -> Option<&str> {
        self.unit.as_deref()
    }

    /// 场大小
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// 获取值的不可变引用
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// 获取值的可变引用
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// 填充常量值
    pub fn fill(&mut self, value: f64) {
        self.values.fill(value);
    }

    /// 计算统计信息
    pub fn stats(&self) -> FieldStats {
        FieldStats::compute(&self.values)
    }

    /// 应用函数到所有值
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(f64) -> f64,
    {
        for v in &mut self.values {
            *v = f(*v);
        }
    }

    /// 限制值范围
    pub fn clamp(&mut self, min: f64, max: f64) {
        for v in &mut self.values {
            *v = v.clamp(min, max);
        }
    }

    /// 强制非负
    pub fn enforce_non_negative(&mut self) {
        for v in &mut self.values {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    }

    /// 与另一个场相加
    pub fn add_field(&mut self, other: &ScalarField) {
        debug_assert_eq!(self.len(), other.len());
        for (a, b) in self.values.iter_mut().zip(other.values.iter()) {
            *a += *b;
        }
    }

    /// 标量乘法
    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.values {
            *v *= factor;
        }
    }

    /// 轴向加法 (self += factor * other)
    pub fn axpy(&mut self, factor: f64, other: &ScalarField) {
        debug_assert_eq!(self.len(), other.len());
        for (a, b) in self.values.iter_mut().zip(other.values.iter()) {
            *a += factor * *b;
        }
    }

    /// 检查是否包含 NaN 或无穷大
    pub fn has_invalid_values(&self) -> bool {
        self.values.iter().any(|v| !v.is_finite())
    }

    /// 找出无效值的索引
    pub fn invalid_indices(&self) -> Vec<usize> {
        self.values
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_finite())
            .map(|(i, _)| i)
            .collect()
    }

    /// 复制到另一个场
    pub fn copy_to(&self, target: &mut ScalarField) {
        debug_assert_eq!(self.len(), target.len());
        target.values.copy_from_slice(&self.values);
    }

    /// 交换两个场的内容
    pub fn swap(&mut self, other: &mut ScalarField) {
        debug_assert_eq!(self.len(), other.len());
        std::mem::swap(&mut self.values, &mut other.values);
    }
}

impl Index<usize> for ScalarField {
    type Output = f64;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for ScalarField {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

/// 向量场
///
/// 存储定义在网格单元上的向量量（如速度、动量等）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorField {
    /// x 分量
    u: Vec<f64>,
    /// y 分量
    v: Vec<f64>,
    /// 场名称
    name: String,
    /// 物理单位
    unit: Option<String>,
}

impl VectorField {
    /// 创建新的向量场
    pub fn new(name: impl Into<String>, size: usize) -> Self {
        Self {
            u: vec![0.0; size],
            v: vec![0.0; size],
            name: name.into(),
            unit: None,
        }
    }

    /// 从分量创建
    pub fn from_components(name: impl Into<String>, u: Vec<f64>, v: Vec<f64>) -> Self {
        debug_assert_eq!(u.len(), v.len());
        Self {
            u,
            v,
            name: name.into(),
            unit: None,
        }
    }

    /// 创建常量场
    pub fn constant(name: impl Into<String>, size: usize, u: f64, v: f64) -> Self {
        Self {
            u: vec![u; size],
            v: vec![v; size],
            name: name.into(),
            unit: None,
        }
    }

    /// 设置物理单位
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// 获取场名称
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 获取物理单位
    pub fn unit(&self) -> Option<&str> {
        self.unit.as_deref()
    }

    /// 场大小
    pub fn len(&self) -> usize {
        self.u.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.u.is_empty()
    }

    /// 获取 x 分量
    pub fn u(&self) -> &[f64] {
        &self.u
    }

    /// 获取 y 分量
    pub fn v(&self) -> &[f64] {
        &self.v
    }

    /// 获取 x 分量可变引用
    pub fn u_mut(&mut self) -> &mut [f64] {
        &mut self.u
    }

    /// 获取 y 分量可变引用
    pub fn v_mut(&mut self) -> &mut [f64] {
        &mut self.v
    }

    /// 获取单个向量
    #[inline]
    pub fn get(&self, index: usize) -> DVec2 {
        DVec2::new(self.u[index], self.v[index])
    }

    /// 设置单个向量
    #[inline]
    pub fn set(&mut self, index: usize, vec: DVec2) {
        self.u[index] = vec.x;
        self.v[index] = vec.y;
    }

    /// 填充常量值
    pub fn fill(&mut self, u: f64, v: f64) {
        self.u.fill(u);
        self.v.fill(v);
    }

    /// 填充零
    pub fn fill_zero(&mut self) {
        self.fill(0.0, 0.0);
    }

    /// 计算速度大小场
    pub fn magnitude(&self) -> ScalarField {
        let values: Vec<f64> = self
            .u
            .iter()
            .zip(self.v.iter())
            .map(|(u, v)| (u * u + v * v).sqrt())
            .collect();
        ScalarField::from_values(format!("{}_magnitude", self.name), values)
    }

    /// 计算速度大小（单点）
    #[inline]
    pub fn speed(&self, index: usize) -> f64 {
        (self.u[index] * self.u[index] + self.v[index] * self.v[index]).sqrt()
    }

    /// 计算速度平方（单点）
    #[inline]
    pub fn speed_squared(&self, index: usize) -> f64 {
        self.u[index] * self.u[index] + self.v[index] * self.v[index]
    }

    /// 计算统计信息
    pub fn stats(&self) -> VectorFieldStats {
        VectorFieldStats {
            u_stats: FieldStats::compute(&self.u),
            v_stats: FieldStats::compute(&self.v),
            max_magnitude: self
                .u
                .iter()
                .zip(self.v.iter())
                .map(|(u, v)| (u * u + v * v).sqrt())
                .fold(0.0_f64, f64::max),
        }
    }

    /// 标量乘法
    pub fn scale(&mut self, factor: f64) {
        for u in &mut self.u {
            *u *= factor;
        }
        for v in &mut self.v {
            *v *= factor;
        }
    }

    /// 限制最大速度
    pub fn clamp_magnitude(&mut self, max_speed: f64) {
        for i in 0..self.len() {
            let speed = self.speed(i);
            if speed > max_speed && speed > 1e-14 {
                let factor = max_speed / speed;
                self.u[i] *= factor;
                self.v[i] *= factor;
            }
        }
    }

    /// 与另一个场相加
    pub fn add_field(&mut self, other: &VectorField) {
        debug_assert_eq!(self.len(), other.len());
        for (a, b) in self.u.iter_mut().zip(other.u.iter()) {
            *a += *b;
        }
        for (a, b) in self.v.iter_mut().zip(other.v.iter()) {
            *a += *b;
        }
    }

    /// 检查是否包含无效值
    pub fn has_invalid_values(&self) -> bool {
        self.u.iter().any(|v| !v.is_finite()) || self.v.iter().any(|v| !v.is_finite())
    }

    /// 迭代所有向量
    pub fn iter(&self) -> impl Iterator<Item = DVec2> + '_ {
        self.u
            .iter()
            .zip(self.v.iter())
            .map(|(&u, &v)| DVec2::new(u, v))
    }
}

/// 标量场统计信息
#[derive(Debug, Clone, Copy, Default)]
pub struct FieldStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub sum: f64,
    pub count: usize,
    pub nan_count: usize,
    pub inf_count: usize,
}

impl FieldStats {
    /// 计算统计信息
    pub fn compute(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut valid_count = 0usize;
        let mut nan_count = 0usize;
        let mut inf_count = 0usize;

        for &v in values {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            } else {
                min = min.min(v);
                max = max.max(v);
                sum += v;
                valid_count += 1;
            }
        }

        let mean = if valid_count > 0 {
            sum / valid_count as f64
        } else {
            f64::NAN
        };

        Self {
            min: if valid_count > 0 { min } else { f64::NAN },
            max: if valid_count > 0 { max } else { f64::NAN },
            mean,
            sum,
            count: values.len(),
            nan_count,
            inf_count,
        }
    }

    /// 是否有无效值
    pub fn has_invalid(&self) -> bool {
        self.nan_count > 0 || self.inf_count > 0
    }

    /// 有效值数量
    pub fn valid_count(&self) -> usize {
        self.count - self.nan_count - self.inf_count
    }
}

/// 向量场统计信息
#[derive(Debug, Clone, Copy, Default)]
pub struct VectorFieldStats {
    pub u_stats: FieldStats,
    pub v_stats: FieldStats,
    pub max_magnitude: f64,
}

impl VectorFieldStats {
    /// 是否有无效值
    pub fn has_invalid(&self) -> bool {
        self.u_stats.has_invalid() || self.v_stats.has_invalid()
    }
}

/// 场掩膜（用于标记有效/无效单元）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMask {
    mask: Vec<bool>,
}

impl FieldMask {
    /// 创建全部有效的掩膜
    pub fn all_valid(size: usize) -> Self {
        Self {
            mask: vec![true; size],
        }
    }

    /// 创建全部无效的掩膜
    pub fn all_invalid(size: usize) -> Self {
        Self {
            mask: vec![false; size],
        }
    }

    /// 从布尔数组创建
    pub fn from_vec(mask: Vec<bool>) -> Self {
        Self { mask }
    }

    /// 从条件创建
    pub fn from_condition<F>(size: usize, condition: F) -> Self
    where
        F: Fn(usize) -> bool,
    {
        Self {
            mask: (0..size).map(condition).collect(),
        }
    }

    /// 从标量场创建（基于阈值）
    pub fn from_field_threshold(field: &ScalarField, threshold: f64) -> Self {
        Self {
            mask: field.values().iter().map(|&v| v >= threshold).collect(),
        }
    }

    /// 检查索引是否有效
    #[inline]
    pub fn is_valid(&self, index: usize) -> bool {
        self.mask[index]
    }

    /// 设置有效性
    #[inline]
    pub fn set_valid(&mut self, index: usize, valid: bool) {
        self.mask[index] = valid;
    }

    /// 获取掩膜数组
    pub fn as_slice(&self) -> &[bool] {
        &self.mask
    }

    /// 有效单元数量
    pub fn valid_count(&self) -> usize {
        self.mask.iter().filter(|&&v| v).count()
    }

    /// 无效单元数量
    pub fn invalid_count(&self) -> usize {
        self.mask.len() - self.valid_count()
    }

    /// 总数量
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// 有效索引迭代器
    pub fn valid_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.mask
            .iter()
            .enumerate()
            .filter(|(_, &v)| v)
            .map(|(i, _)| i)
    }

    /// 与运算
    pub fn and(&self, other: &FieldMask) -> Self {
        debug_assert_eq!(self.len(), other.len());
        Self {
            mask: self
                .mask
                .iter()
                .zip(other.mask.iter())
                .map(|(&a, &b)| a && b)
                .collect(),
        }
    }

    /// 或运算
    pub fn or(&self, other: &FieldMask) -> Self {
        debug_assert_eq!(self.len(), other.len());
        Self {
            mask: self
                .mask
                .iter()
                .zip(other.mask.iter())
                .map(|(&a, &b)| a || b)
                .collect(),
        }
    }

    /// 取反
    pub fn not(&self) -> Self {
        Self {
            mask: self.mask.iter().map(|&v| !v).collect(),
        }
    }
}

impl Index<usize> for FieldMask {
    type Output = bool;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.mask[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_field_creation() {
        let field = ScalarField::new("test", 10);
        assert_eq!(field.len(), 10);
        assert_eq!(field.name(), "test");
        assert!(field.values().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_scalar_field_constant() {
        let field = ScalarField::constant("const", 5, 42.0);
        assert!(field.values().iter().all(|&v| (v - 42.0).abs() < 1e-10));
    }

    #[test]
    fn test_scalar_field_stats() {
        let field = ScalarField::from_values("test", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = field.stats();

        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_scalar_field_operations() {
        let mut field = ScalarField::from_values("test", vec![1.0, 2.0, 3.0]);

        field.scale(2.0);
        assert!((field[0] - 2.0).abs() < 1e-10);
        assert!((field[1] - 4.0).abs() < 1e-10);

        field.clamp(0.0, 5.0);
        assert!((field[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_field_axpy() {
        let mut a = ScalarField::from_values("a", vec![1.0, 2.0, 3.0]);
        let b = ScalarField::from_values("b", vec![1.0, 1.0, 1.0]);

        a.axpy(2.0, &b);

        assert!((a[0] - 3.0).abs() < 1e-10);
        assert!((a[1] - 4.0).abs() < 1e-10);
        assert!((a[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_field_creation() {
        let field = VectorField::new("velocity", 10);
        assert_eq!(field.len(), 10);
        assert!(field.u().iter().all(|&v| v == 0.0));
        assert!(field.v().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_vector_field_magnitude() {
        let field = VectorField::from_components("vel", vec![3.0, 0.0], vec![4.0, 1.0]);

        assert!((field.speed(0) - 5.0).abs() < 1e-10);
        assert!((field.speed(1) - 1.0).abs() < 1e-10);

        let mag = field.magnitude();
        assert!((mag[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_field_clamp() {
        let mut field = VectorField::from_components("vel", vec![30.0], vec![40.0]);

        field.clamp_magnitude(10.0);

        let speed = field.speed(0);
        assert!((speed - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_field_mask() {
        let mask = FieldMask::from_vec(vec![true, false, true, false, true]);

        assert!(mask.is_valid(0));
        assert!(!mask.is_valid(1));
        assert_eq!(mask.valid_count(), 3);
        assert_eq!(mask.invalid_count(), 2);

        let indices: Vec<_> = mask.valid_indices().collect();
        assert_eq!(indices, vec![0, 2, 4]);
    }

    #[test]
    fn test_field_mask_operations() {
        let a = FieldMask::from_vec(vec![true, true, false, false]);
        let b = FieldMask::from_vec(vec![true, false, true, false]);

        let and_result = a.and(&b);
        assert!(and_result.is_valid(0));
        assert!(!and_result.is_valid(1));
        assert!(!and_result.is_valid(2));

        let or_result = a.or(&b);
        assert!(or_result.is_valid(0));
        assert!(or_result.is_valid(1));
        assert!(or_result.is_valid(2));
        assert!(!or_result.is_valid(3));
    }

    #[test]
    fn test_field_stats_with_invalid() {
        let values = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0];
        let stats = FieldStats::compute(&values);

        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.inf_count, 1);
        assert_eq!(stats.valid_count(), 3);
        assert!(stats.has_invalid());
    }
}
