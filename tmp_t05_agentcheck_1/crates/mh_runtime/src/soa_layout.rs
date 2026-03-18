// crates/mh_runtime/src/soa_layout.rs

//! SoA (Structure of Arrays) 内存布局描述符
//!
//! 提供编译期偏移计算和类型安全的字段访问。
//!
//! # 设计目标
//!
//! 1. **单块分配**：所有字段连续存储在同一缓冲区
//! 2. **编译期偏移**：使用 const fn 计算字段偏移
//! 3. **类型安全视图**：提供字段切片的安全访问
//! 4. **GPU 友好**：直接传输整个缓冲区，偏移传给 kernel
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_runtime::soa_layout::{SoaLayout, FieldDescriptor};
//!
//! // 定义布局
//! let layout = SoaLayout::builder()
//!     .add_field::<f64>("h", n_cells)
//!     .add_field::<f64>("hu", n_cells)
//!     .add_field::<f64>("hv", n_cells)
//!     .add_field::<f64>("z", n_cells)
//!     .build();
//!
//! // 分配缓冲区
//! let mut buffer = vec![0u8; layout.total_bytes()];
//!
//! // 获取字段视图
//! let h: &mut [f64] = layout.get_field_mut("h", &mut buffer);
//! ```

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::marker::PhantomData;
use crate::error::{RuntimeError, RuntimeResult};

/// 对齐要求（字节）
pub const DEFAULT_ALIGNMENT: usize = 64;

/// 字段描述符
#[derive(Debug, Clone, Copy)]
pub struct FieldDescriptor {
    /// 字段名称（调试用）
    pub name_hash: u64,
    /// 元素数量
    pub count: usize,
    /// 元素大小（字节）
    pub element_size: usize,
    /// 字节偏移
    pub byte_offset: usize,
    /// 对齐要求
    pub alignment: usize,
}

impl FieldDescriptor {
    /// 计算字段的总字节数
    #[inline]
    pub const fn byte_size(&self) -> usize {
        self.count * self.element_size
    }

    /// 获取对齐后的字节偏移
    #[inline]
    pub const fn aligned_offset(current: usize, alignment: usize) -> usize {
        let remainder = current % alignment;
        if remainder == 0 {
            current
        } else {
            current + alignment - remainder
        }
    }
}

/// SoA 布局描述符
#[derive(Debug, Clone)]
pub struct SoaLayout {
    /// 字段描述符列表
    fields: Vec<FieldDescriptor>,
    /// 字段名称到索引的映射
    field_indices: HashMap<String, usize>,
    /// 总字节数（对齐后）
    total_bytes: usize,
    /// 元素数量（所有字段相同）
    element_count: usize,
}

impl SoaLayout {
    /// 创建布局构建器
    pub fn builder() -> SoaLayoutBuilder {
        SoaLayoutBuilder::new()
    }

    /// 获取字段数量
    #[inline]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// 获取元素数量
    #[inline]
    pub fn element_count(&self) -> usize {
        self.element_count
    }

    /// 获取总字节数
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// 获取字段描述符
    pub fn get_field_descriptor(&self, name: &str) -> Option<&FieldDescriptor> {
        self.field_indices.get(name).map(|&idx| &self.fields[idx])
    }

    /// 获取字段偏移（按名称）
    pub fn field_offset(&self, name: &str) -> Option<usize> {
        self.get_field_descriptor(name).map(|f| f.byte_offset)
    }

    /// 获取字段切片（只读）
    ///
    /// # Safety
    ///
    /// 调用者需确保 buffer 的生命周期和类型正确
    pub unsafe fn get_field_slice<'a, T>(&self, name: &str, buffer: &'a [u8]) -> Option<&'a [T]> {
        let desc = self.get_field_descriptor(name)?;
        debug_assert_eq!(std::mem::size_of::<T>(), desc.element_size);
        debug_assert!(buffer.len() >= desc.byte_offset + desc.byte_size());

        let ptr = buffer.as_ptr().add(desc.byte_offset) as *const T;
        Some(std::slice::from_raw_parts(ptr, desc.count))
    }

    /// 获取字段切片（可变）
    ///
    /// # Safety
    ///
    /// 调用者需确保 buffer 的生命周期和类型正确，且无其他借用冲突
    pub unsafe fn get_field_slice_mut<'a, T>(
        &self,
        name: &str,
        buffer: &'a mut [u8],
    ) -> Option<&'a mut [T]> {
        let desc = self.get_field_descriptor(name)?;
        debug_assert_eq!(std::mem::size_of::<T>(), desc.element_size);
        debug_assert!(buffer.len() >= desc.byte_offset + desc.byte_size());

        let ptr = buffer.as_mut_ptr().add(desc.byte_offset) as *mut T;
        Some(std::slice::from_raw_parts_mut(ptr, desc.count))
    }

    /// 获取字段切片（带边界与类型检查）
    pub fn get_field<'a, T>(&self, name: &str, buffer: &'a [u8]) -> RuntimeResult<&'a [T]> {
        let desc = self
            .get_field_descriptor(name)
            .ok_or_else(|| RuntimeError::validation("field not found"))?;
        if desc.element_size != std::mem::size_of::<T>() {
            return Err(RuntimeError::validation("field type size mismatch"));
        }
        if desc.byte_offset + desc.byte_size() > buffer.len() {
            return Err(RuntimeError::buffer("buffer too small"));
        }
        let slice = unsafe { self.get_field_slice(name, buffer) }
            .ok_or_else(|| RuntimeError::validation("field slice not available"))?;
        Ok(slice)
    }

    /// 获取字段可变切片（带边界与类型检查）
    pub fn get_field_mut<'a, T>(
        &self,
        name: &str,
        buffer: &'a mut [u8],
    ) -> RuntimeResult<&'a mut [T]> {
        let desc = self
            .get_field_descriptor(name)
            .ok_or_else(|| RuntimeError::validation("field not found"))?;
        if desc.element_size != std::mem::size_of::<T>() {
            return Err(RuntimeError::validation("field type size mismatch"));
        }
        if desc.byte_offset + desc.byte_size() > buffer.len() {
            return Err(RuntimeError::buffer("buffer too small"));
        }
        let slice = unsafe { self.get_field_slice_mut(name, buffer) }
            .ok_or_else(|| RuntimeError::validation("field slice not available"))?;
        Ok(slice)
    }

    /// 获取所有字段的偏移数组（用于 GPU kernel）
    pub fn all_offsets(&self) -> Vec<usize> {
        self.fields.iter().map(|f| f.byte_offset).collect()
    }

    /// 迭代所有字段描述符
    pub fn iter_fields(&self) -> impl Iterator<Item = &FieldDescriptor> {
        self.fields.iter()
    }
}

/// SoA 布局构建器
#[derive(Debug)]
pub struct SoaLayoutBuilder {
    fields: Vec<(String, usize, usize, usize)>, // (name, count, element_size, alignment)
    alignment: usize,
}

impl SoaLayoutBuilder {
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
            alignment: DEFAULT_ALIGNMENT,
        }
    }

    /// 设置全局对齐
    pub fn alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }

    /// 添加字段
    pub fn add_field<T: Sized>(mut self, name: &str, count: usize) -> Self {
        let element_size = std::mem::size_of::<T>();
        let field_align = std::mem::align_of::<T>().max(self.alignment);
        self.fields.push((name.to_string(), count, element_size, field_align));
        self
    }

    /// 添加 f64 字段
    pub fn add_f64_field(self, name: &str, count: usize) -> Self {
        self.add_field::<f64>(name, count)
    }

    /// 添加 f32 字段
    pub fn add_f32_field(self, name: &str, count: usize) -> Self {
        self.add_field::<f32>(name, count)
    }

    /// 构建布局
    pub fn build(self) -> SoaLayout {
        self.build_checked().unwrap_or_else(|e| {
            panic!("SoaLayout::build failed: {e}")
        })
    }

    /// 构建 SoA 布局（返回错误而非吞掉）
    pub fn build_checked(self) -> RuntimeResult<SoaLayout> {
        let mut descriptors = Vec::with_capacity(self.fields.len());
        let mut field_indices = HashMap::new();
        let mut current_offset = 0;
        let element_count = self.fields.first().map(|(_, c, _, _)| *c).unwrap_or(0);

        for (idx, (name, count, element_size, alignment)) in self.fields.into_iter().enumerate() {
            if count != element_count {
                return Err(RuntimeError::size_mismatch(name.clone(), element_count, count));
            }
            if field_indices.contains_key(&name) {
                return Err(RuntimeError::validation("duplicate field name"));
            }
            // 对齐当前偏移
            let aligned_offset = FieldDescriptor::aligned_offset(current_offset, alignment);

            let desc = FieldDescriptor {
                name_hash: fxhash(&name),
                count,
                element_size,
                byte_offset: aligned_offset,
                alignment,
            };

            field_indices.insert(name, idx);
            current_offset = aligned_offset + desc.byte_size();
            descriptors.push(desc);
        }

        // 最终对齐
        let total_bytes = FieldDescriptor::aligned_offset(current_offset, self.alignment);

        Ok(SoaLayout {
            fields: descriptors,
            field_indices,
            total_bytes,
            element_count,
        })
    }
}

impl Default for SoaLayoutBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// 简单的哈希函数（用于字段名称）
fn fxhash(s: &str) -> u64 {
    let mut hash: u64 = 0;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(0x517cc1b727220a95).wrapping_add(byte as u64);
    }
    hash
}

// ============================================================================
// 浅水方程状态布局（预定义）
// ============================================================================

/// 浅水方程状态偏移（编译期计算）
#[derive(Debug, Clone, Copy)]
pub struct ShallowWaterOffsets {
    /// 水深 h 偏移
    pub h: usize,
    /// x 动量 hu 偏移
    pub hu: usize,
    /// y 动量 hv 偏移
    pub hv: usize,
    /// 底床高程 z 偏移
    pub z: usize,
    /// 示踪剂起始偏移
    pub tracers: usize,
    /// 总元素数（不含示踪剂）
    pub base_fields: usize,
    /// 总字节数
    pub total_bytes: usize,
    /// 单元数量
    pub n_cells: usize,
}

impl ShallowWaterOffsets {
    /// 创建浅水方程状态偏移（编译期友好）
    #[inline]
    pub const fn new(n_cells: usize, n_tracers: usize, element_size: usize) -> Self {
        let h = 0;
        let hu = n_cells;
        let hv = n_cells * 2;
        let z = n_cells * 3;
        let tracers = n_cells * 4;
        let base_fields = 4;
        let total_elements = n_cells * (4 + n_tracers);
        let total_bytes = total_elements * element_size;

        Self {
            h,
            hu,
            hv,
            z,
            tracers,
            base_fields,
            total_bytes,
            n_cells,
        }
    }

    /// 创建 f64 版本偏移
    #[inline]
    pub const fn new_f64(n_cells: usize, n_tracers: usize) -> Self {
        Self::new(n_cells, n_tracers, 8)
    }

    /// 创建 f32 版本偏移
    #[inline]
    pub const fn new_f32(n_cells: usize, n_tracers: usize) -> Self {
        Self::new(n_cells, n_tracers, 4)
    }

    /// 获取示踪剂 k 的偏移
    #[inline]
    pub const fn tracer_offset(&self, k: usize) -> usize {
        self.tracers + k * self.n_cells
    }

    /// 总字段数（含示踪剂）
    #[inline]
    pub const fn total_fields(&self, n_tracers: usize) -> usize {
        self.base_fields + n_tracers
    }
}

// ============================================================================
// 类型安全的字段视图
// ============================================================================

/// 类型安全的字段视图（不可变）
#[derive(Debug)]
pub struct FieldView<'a, T> {
    data: &'a [T],
    offset: usize,
    _marker: PhantomData<T>,
}

impl<'a, T> FieldView<'a, T> {
    /// 创建字段视图
    pub fn new(data: &'a [T], offset: usize, len: usize) -> Self {
        Self {
            data: &data[offset..offset + len],
            offset,
            _marker: PhantomData,
        }
    }

    /// 获取切片
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// 获取偏移
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// 长度
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a, T> std::ops::Deref for FieldView<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

/// 类型安全的字段视图（可变）
#[derive(Debug)]
pub struct FieldViewMut<'a, T> {
    data: &'a mut [T],
    offset: usize,
    _marker: PhantomData<T>,
}

impl<'a, T> FieldViewMut<'a, T> {
    /// 创建可变字段视图
    pub fn new(data: &'a mut [T], offset: usize, len: usize) -> Self {
        Self {
            data: &mut data[offset..offset + len],
            offset,
            _marker: PhantomData,
        }
    }

    /// 获取切片
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// 获取可变切片
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    /// 获取偏移
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// 长度
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a, T> std::ops::Deref for FieldViewMut<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, T> std::ops::DerefMut for FieldViewMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_layout_builder() {
        let layout = SoaLayout::builder()
            .add_f64_field("h", 100)
            .add_f64_field("hu", 100)
            .add_f64_field("hv", 100)
            .add_f64_field("z", 100)
            .build();

        assert_eq!(layout.field_count(), 4);
        assert_eq!(layout.element_count(), 100);
        assert!(layout.total_bytes() >= 100 * 8 * 4);
    }

    #[test]
    fn test_field_offsets() {
        let layout = SoaLayout::builder()
            .add_f64_field("h", 100)
            .add_f64_field("hu", 100)
            .build();

        let h_offset = layout.field_offset("h").unwrap();
        let hu_offset = layout.field_offset("hu").unwrap();

        // h 应该在偏移 0
        assert_eq!(h_offset, 0);
        // hu 应该在 h 之后（100 * 8 = 800 字节，对齐到 64）
        assert!(hu_offset >= 800);
    }

    #[test]
    fn test_shallow_water_offsets() {
        let offsets = ShallowWaterOffsets::new_f64(1000, 2);

        assert_eq!(offsets.h, 0);
        assert_eq!(offsets.hu, 1000);
        assert_eq!(offsets.hv, 2000);
        assert_eq!(offsets.z, 3000);
        assert_eq!(offsets.tracers, 4000);
        assert_eq!(offsets.tracer_offset(0), 4000);
        assert_eq!(offsets.tracer_offset(1), 5000);
    }

    #[test]
    fn test_field_view() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let view = FieldView::new(&data, 2, 4);

        assert_eq!(view.len(), 4);
        assert_eq!(view[0], 3.0);
        assert_eq!(view[3], 6.0);
    }

    #[test]
    fn test_field_view_mut() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        {
            let mut view = FieldViewMut::new(&mut data, 2, 4);
            view[0] = 10.0;
            view[1] = 20.0;
        }

        assert_eq!(data[2], 10.0);
        assert_eq!(data[3], 20.0);
    }
}
