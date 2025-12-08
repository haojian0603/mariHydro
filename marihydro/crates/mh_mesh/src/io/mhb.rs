// marihydro\crates\mh_mesh\src/io/mhb.rs

//! MHB 二进制格式
//!
//! 自定义的高性能网格二进制格式，支持字段级随机访问和压缩预留。
//!
//! # 格式结构
//!
//! ```text
//! +----------------+
//! | Magic (4B)     |  "MHB1"
//! | Version (4B)   |  格式版本号
//! | Flags (4B)     |  标志位
//! | Index Offset   |  字段索引偏移 (8B)
//! +----------------+
//! | Data Chunk 1   |
//! | Data Chunk 2   |
//! | ...            |
//! +----------------+
//! | Field Index    |  JSON 序列化的 FieldIndex
//! +----------------+
//! ```

use super::fields::{FieldDescriptor, FieldIndex, DataType};
#[allow(unused_imports)]
use super::fields::Compression;
use std::io::{Read, Write, Seek, SeekFrom, Result, Error, ErrorKind};

/// MHB 文件魔数
pub const MHB_MAGIC: &[u8; 4] = b"MHB1";

/// MHB 当前版本
pub const MHB_VERSION: u32 = 1;

/// MHB 文件头
#[derive(Debug, Clone, Default)]
pub struct MhbHeader {
    /// 版本号
    pub version: u32,
    /// 标志位
    pub flags: u32,
    /// 字段索引在文件中的偏移
    pub index_offset: u64,
}

impl MhbHeader {
    /// 头部大小（字节）
    pub const SIZE: usize = 4 + 4 + 4 + 8; // magic + version + flags + index_offset

    /// 创建新头部
    pub fn new() -> Self {
        Self {
            version: MHB_VERSION,
            flags: 0,
            index_offset: 0,
        }
    }

    /// 写入头部
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(MHB_MAGIC)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.flags.to_le_bytes())?;
        writer.write_all(&self.index_offset.to_le_bytes())?;
        Ok(())
    }

    /// 读取头部
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MHB_MAGIC {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid MHB magic"));
        }

        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4)?;
        let flags = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf8)?;
        let index_offset = u64::from_le_bytes(buf8);

        Ok(Self { version, flags, index_offset })
    }
}

/// MHB 写入器
///
/// 支持流式写入大型网格文件。
pub struct MhbWriter<W: Write + Seek> {
    writer: W,
    index: FieldIndex,
    current_offset: u64,
}

impl<W: Write + Seek> MhbWriter<W> {
    /// 创建新的写入器
    pub fn new(mut writer: W) -> Result<Self> {
        // 写入占位头部
        let header = MhbHeader::new();
        header.write(&mut writer)?;

        Ok(Self {
            writer,
            index: FieldIndex::new(),
            current_offset: MhbHeader::SIZE as u64,
        })
    }

    /// 写入 f64 数组字段
    pub fn write_f64_field(&mut self, name: &str, data: &[f64]) -> Result<()> {
        let mut desc = FieldDescriptor::new(name, DataType::F64, data.len() as u64);
        desc.offset = self.current_offset;

        // 写入数据
        for &v in data {
            self.writer.write_all(&v.to_le_bytes())?;
        }

        self.current_offset += desc.size_raw;
        self.index.add(desc);
        Ok(())
    }

    /// 写入 u32 数组字段
    pub fn write_u32_field(&mut self, name: &str, data: &[u32]) -> Result<()> {
        let mut desc = FieldDescriptor::new(name, DataType::U32, data.len() as u64);
        desc.offset = self.current_offset;

        for &v in data {
            self.writer.write_all(&v.to_le_bytes())?;
        }

        self.current_offset += desc.size_raw;
        self.index.add(desc);
        Ok(())
    }

    /// 完成写入并关闭文件
    pub fn finish(mut self) -> Result<W> {
        // 记录索引偏移
        let index_offset = self.current_offset;

        // 写入字段索引（JSON 格式）
        let index_json = serde_json::to_vec(&self.index)
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        self.writer.write_all(&index_json)?;

        // 回写头部
        self.writer.seek(SeekFrom::Start(0))?;
        let header = MhbHeader {
            version: MHB_VERSION,
            flags: 0,
            index_offset,
        };
        header.write(&mut self.writer)?;

        Ok(self.writer)
    }
}

/// MHB 读取器
pub struct MhbReader<R: Read + Seek> {
    reader: R,
    #[allow(dead_code)]
    header: MhbHeader,
    index: FieldIndex,
}

impl<R: Read + Seek> MhbReader<R> {
    /// 打开 MHB 文件
    pub fn open(mut reader: R) -> Result<Self> {
        // 读取头部
        let header = MhbHeader::read(&mut reader)?;

        // 定位到索引
        reader.seek(SeekFrom::Start(header.index_offset))?;

        // 读取索引
        let mut index_data = Vec::new();
        reader.read_to_end(&mut index_data)?;
        let index: FieldIndex = serde_json::from_slice(&index_data)
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        Ok(Self { reader, header, index })
    }

    /// 获取字段索引
    pub fn index(&self) -> &FieldIndex {
        &self.index
    }

    /// 读取 f64 字段
    pub fn read_f64_field(&mut self, name: &str) -> Result<Vec<f64>> {
        let desc = self.index.find(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("Field not found: {}", name)))?;

        if desc.dtype != DataType::F64 {
            return Err(Error::new(ErrorKind::InvalidData, "Field is not F64 type"));
        }

        self.reader.seek(SeekFrom::Start(desc.offset))?;

        let mut data = Vec::with_capacity(desc.count as usize);
        let mut buf = [0u8; 8];
        for _ in 0..desc.count {
            self.reader.read_exact(&mut buf)?;
            data.push(f64::from_le_bytes(buf));
        }

        Ok(data)
    }

    /// 读取 u32 字段
    pub fn read_u32_field(&mut self, name: &str) -> Result<Vec<u32>> {
        let desc = self.index.find(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("Field not found: {}", name)))?;

        if desc.dtype != DataType::U32 {
            return Err(Error::new(ErrorKind::InvalidData, "Field is not U32 type"));
        }

        self.reader.seek(SeekFrom::Start(desc.offset))?;

        let mut data = Vec::with_capacity(desc.count as usize);
        let mut buf = [0u8; 4];
        for _ in 0..desc.count {
            self.reader.read_exact(&mut buf)?;
            data.push(u32::from_le_bytes(buf));
        }

        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_header_roundtrip() {
        let header = MhbHeader {
            version: 1,
            flags: 0,
            index_offset: 1234,
        };

        let mut buf = Vec::new();
        header.write(&mut buf).unwrap();

        let mut cursor = Cursor::new(buf);
        let header2 = MhbHeader::read(&mut cursor).unwrap();

        assert_eq!(header2.version, 1);
        assert_eq!(header2.index_offset, 1234);
    }

    #[test]
    fn test_write_read_roundtrip() {
        let data_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data_u32 = vec![10u32, 20, 30];

        // 写入
        let buf = Cursor::new(Vec::new());
        let mut writer = MhbWriter::new(buf).unwrap();
        writer.write_f64_field("values", &data_f64).unwrap();
        writer.write_u32_field("indices", &data_u32).unwrap();
        let buf = writer.finish().unwrap();

        // 读取
        let mut reader = MhbReader::open(Cursor::new(buf.into_inner())).unwrap();
        
        assert_eq!(reader.index().len(), 2);
        
        let read_f64 = reader.read_f64_field("values").unwrap();
        assert_eq!(read_f64, data_f64);
        
        let read_u32 = reader.read_u32_field("indices").unwrap();
        assert_eq!(read_u32, data_u32);
    }
}
