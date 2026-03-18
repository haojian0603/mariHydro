//! 字段描述系统
//!
//! 提供 MHB 二进制格式的字段元数据定义。
//!
//! # 设计说明
//!
//! 每个字段记录：
//! - 数据类型（f32/f64/u32/Point2D 等）
//! - 存储位置（文件偏移）
//! - 压缩方式（无/LZ4/Zstd）
//! - 校验和（可选）

use serde::{Deserialize, Serialize};

/// 数据类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum DataType {
    /// 32位浮点
    F32 = 1,
    /// 64位浮点
    F64 = 2,
    /// 32位无符号整数
    U32 = 3,
    /// 64位无符号整数
    U64 = 4,
    /// 8位无符号整数
    U8 = 5,
    /// 2D点 (2 x f64)
    Point2D = 10,
    /// 3D点 (3 x f64)
    Point3D = 11,
    /// 单元索引
    CellIdx = 20,
    /// 面索引
    FaceIdx = 21,
    /// 节点索引
    NodeIdx = 22,
}

impl DataType {
    /// 返回单个元素的字节数
    pub fn element_size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F64 => 8,
            DataType::U32 => 4,
            DataType::U64 => 8,
            DataType::U8 => 1,
            DataType::Point2D => 16,  // 2 x f64
            DataType::Point3D => 24,  // 3 x f64
            DataType::CellIdx => 4,
            DataType::FaceIdx => 4,
            DataType::NodeIdx => 4,
        }
    }
}

/// 压缩方式枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum Compression {
    /// 无压缩
    #[default]
    None = 0,
    /// LZ4 快速压缩（未来支持）
    Lz4 = 1,
    /// Zstd 压缩（未来支持）
    Zstd = 2,
}

/// 字段描述符
///
/// 描述 MHB 文件中单个字段的元数据。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDescriptor {
    /// 字段名称
    pub name: String,
    /// 数据类型
    pub dtype: DataType,
    /// 元素数量
    pub count: u64,
    /// 文件中的偏移位置
    pub offset: u64,
    /// 原始数据大小（字节）
    pub size_raw: u64,
    /// 存储大小（压缩后）
    pub size_stored: u64,
    /// 压缩方式
    pub compression: Compression,
    /// CRC32 校验和（可选）
    pub checksum: Option<u32>,
}

impl FieldDescriptor {
    /// 创建新的字段描述符
    pub fn new(name: impl Into<String>, dtype: DataType, count: u64) -> Self {
        let size = count * dtype.element_size() as u64;
        Self {
            name: name.into(),
            dtype,
            count,
            offset: 0,
            size_raw: size,
            size_stored: size,
            compression: Compression::None,
            checksum: None,
        }
    }

    /// 设置偏移位置
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    /// 是否已压缩
    pub fn is_compressed(&self) -> bool {
        self.compression != Compression::None
    }
}

/// 字段索引
///
/// 存储 MHB 文件中所有字段的描述符。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FieldIndex {
    /// 所有字段描述符
    pub fields: Vec<FieldDescriptor>,
}

impl FieldIndex {
    /// 创建新的空索引
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// 添加字段
    pub fn add(&mut self, field: FieldDescriptor) {
        self.fields.push(field);
    }

    /// 通过名称查找字段
    pub fn find(&self, name: &str) -> Option<&FieldDescriptor> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// 获取字段数量
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// 计算总存储大小
    pub fn total_size(&self) -> u64 {
        self.fields.iter().map(|f| f.size_stored).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_size() {
        assert_eq!(DataType::F32.element_size(), 4);
        assert_eq!(DataType::F64.element_size(), 8);
        assert_eq!(DataType::Point2D.element_size(), 16);
        assert_eq!(DataType::Point3D.element_size(), 24);
    }

    #[test]
    fn test_field_descriptor() {
        let field = FieldDescriptor::new("cell_area", DataType::F64, 100);
        assert_eq!(field.name, "cell_area");
        assert_eq!(field.dtype, DataType::F64);
        assert_eq!(field.count, 100);
        assert_eq!(field.size_raw, 800); // 100 * 8
        assert!(!field.is_compressed());
    }

    #[test]
    fn test_field_index() {
        let mut index = FieldIndex::new();
        index.add(FieldDescriptor::new("field1", DataType::F64, 10));
        index.add(FieldDescriptor::new("field2", DataType::U32, 20));

        assert_eq!(index.len(), 2);
        assert!(index.find("field1").is_some());
        assert!(index.find("nonexistent").is_none());
    }
}
