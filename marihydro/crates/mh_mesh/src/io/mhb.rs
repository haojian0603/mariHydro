// marihydro\crates\mh_mesh\src\io\mhb.rs

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

use super::fields::{DataType, FieldDescriptor, FieldIndex};
#[allow(unused_imports)]
use super::fields::Compression;
use std::io::{Error, ErrorKind, Read, Result, Seek, SeekFrom, Write};
use std::path::Path;
use std::fs::File;
use crate::{FrozenMesh, FrozenMeshGeneric};  // FIX: Import from crate root
use mh_geo::{Point2D, Point3D};
use mh_runtime::{Backend, RuntimeScalar};
use serde_json;

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
        // 写入临时头部（索引偏移稍后回填）
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

    /// 写入 f32 数组字段
    pub fn write_f32_field(&mut self, name: &str, data: &[f32]) -> Result<()> {
        let mut desc = FieldDescriptor::new(name, DataType::F32, data.len() as u64);
        desc.offset = self.current_offset;

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

    /// 写入 u64 数组字段
    pub fn write_u64_field(&mut self, name: &str, data: &[u64]) -> Result<()> {
        let mut desc = FieldDescriptor::new(name, DataType::U64, data.len() as u64);
        desc.offset = self.current_offset;

        for &v in data {
            self.writer.write_all(&v.to_le_bytes())?;
        }

        self.current_offset += desc.size_raw;
        self.index.add(desc);
        Ok(())
    }

    /// 写入 u8 数组字段
    pub fn write_u8_field(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let mut desc = FieldDescriptor::new(name, DataType::U8, data.len() as u64);
        desc.offset = self.current_offset;

        self.writer.write_all(data)?;

        self.current_offset += desc.size_raw;
        self.index.add(desc);
        Ok(())
    }

    /// 写入 Point2D 数组字段
    pub fn write_point2d_field(&mut self, name: &str, data: &[Point2D]) -> Result<()> {
        let mut desc = FieldDescriptor::new(name, DataType::Point2D, data.len() as u64);
        desc.offset = self.current_offset;

        for p in data {
            self.writer.write_all(&p.x.to_le_bytes())?;
            self.writer.write_all(&p.y.to_le_bytes())?;
        }

        self.current_offset += desc.size_raw;
        self.index.add(desc);
        Ok(())
    }

    /// 写入 Point3D 数组字段
    pub fn write_point3d_field(&mut self, name: &str, data: &[Point3D]) -> Result<()> {
        let mut desc = FieldDescriptor::new(name, DataType::Point3D, data.len() as u64);
        desc.offset = self.current_offset;

        for p in data {
            self.writer.write_all(&p.x.to_le_bytes())?;
            self.writer.write_all(&p.y.to_le_bytes())?;
            self.writer.write_all(&p.z.to_le_bytes())?;
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

    /// 字段是否存在
    pub fn has_field(&self, name: &str) -> bool {
        self.index.find(name).is_some()
    }

    /// 获取字段数据类型
    pub fn get_field_dtype(&self, name: &str) -> Result<DataType> {
        let desc = self
            .index
            .find(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("Field not found: {}", name)))?;
        Ok(desc.dtype)
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

    /// 读取 f32 字段
    pub fn read_f32_field(&mut self, name: &str) -> Result<Vec<f32>> {
        let desc = self.index.find(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("Field not found: {}", name)))?;

        if desc.dtype != DataType::F32 {
            return Err(Error::new(ErrorKind::InvalidData, "Field is not F32 type"));
        }

        self.reader.seek(SeekFrom::Start(desc.offset))?;

        let mut data = Vec::with_capacity(desc.count as usize);
        let mut buf = [0u8; 4];
        for _ in 0..desc.count {
            self.reader.read_exact(&mut buf)?;
            data.push(f32::from_le_bytes(buf));
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

    /// 读取 u64 字段
    pub fn read_u64_field(&mut self, name: &str) -> Result<Vec<u64>> {
        let desc = self.index.find(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("Field not found: {}", name)))?;

        if desc.dtype != DataType::U64 {
            return Err(Error::new(ErrorKind::InvalidData, "Field is not U64 type"));
        }

        self.reader.seek(SeekFrom::Start(desc.offset))?;

        let mut data = Vec::with_capacity(desc.count as usize);
        let mut buf = [0u8; 8];
        for _ in 0..desc.count {
            self.reader.read_exact(&mut buf)?;
            data.push(u64::from_le_bytes(buf));
        }

        Ok(data)
    }

    /// 读取 u8 字段
    pub fn read_u8_field(&mut self, name: &str) -> Result<Vec<u8>> {
        let desc = self.index.find(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("Field not found: {}", name)))?;

        if desc.dtype != DataType::U8 {
            return Err(Error::new(ErrorKind::InvalidData, "Field is not U8 type"));
        }

        self.reader.seek(SeekFrom::Start(desc.offset))?;

        let mut data = vec![0u8; desc.count as usize];
        self.reader.read_exact(&mut data)?;
        Ok(data)
    }

    /// 读取 Point2D 字段
    pub fn read_point2d_field(&mut self, name: &str) -> Result<Vec<Point2D>> {
        let desc = self.index.find(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("Field not found: {}", name)))?;

        if desc.dtype == DataType::F64 {
            let data = self.read_f64_field(name)?;
            return Ok(data
                .chunks(2)
                .map(|chunk| Point2D::new(chunk[0], chunk[1]))
                .collect());
        }
        if desc.dtype != DataType::Point2D {
            return Err(Error::new(ErrorKind::InvalidData, "Field is not Point2D type"));
        }

        self.reader.seek(SeekFrom::Start(desc.offset))?;

        let mut data = Vec::with_capacity(desc.count as usize);
        let mut buf = [0u8; 8];
        for _ in 0..desc.count {
            self.reader.read_exact(&mut buf)?;
            let x = f64::from_le_bytes(buf);
            self.reader.read_exact(&mut buf)?;
            let y = f64::from_le_bytes(buf);
            data.push(Point2D::new(x, y));
        }

        Ok(data)
    }

    /// 读取 Point3D 字段
    pub fn read_point3d_field(&mut self, name: &str) -> Result<Vec<Point3D>> {
        let desc = self.index.find(name)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, format!("Field not found: {}", name)))?;

        if desc.dtype == DataType::F64 {
            let data = self.read_f64_field(name)?;
            return Ok(data
                .chunks(3)
                .map(|chunk| Point3D::new(chunk[0], chunk[1], chunk[2]))
                .collect());
        }
        if desc.dtype != DataType::Point3D {
            return Err(Error::new(ErrorKind::InvalidData, "Field is not Point3D type"));
        }

        self.reader.seek(SeekFrom::Start(desc.offset))?;

        let mut data = Vec::with_capacity(desc.count as usize);
        let mut buf = [0u8; 8];
        for _ in 0..desc.count {
            self.reader.read_exact(&mut buf)?;
            let x = f64::from_le_bytes(buf);
            self.reader.read_exact(&mut buf)?;
            let y = f64::from_le_bytes(buf);
            self.reader.read_exact(&mut buf)?;
            let z = f64::from_le_bytes(buf);
            data.push(Point3D::new(x, y, z));
        }

        Ok(data)
    }
}

/// 便捷函数：保存 MHB 文件
pub fn save_mhb<B: Backend>(path: &Path, mesh: &FrozenMeshGeneric<B>) -> Result<()> {
    let file = File::create(path)?;
    let writer = MhbWriter::new(file)?;
    write_mesh_fields(writer, mesh)?;
    Ok(())
}

/// 便捷函数：加载 MHB 文件（返回 f64 精度 FrozenMesh）
pub fn load_mhb(path: &Path) -> Result<FrozenMesh> {
    let file = File::open(path)?;
    let mut reader = MhbReader::open(file)?;

    let n_nodes = read_count(&mut reader, "n_nodes")?;
    let n_cells = read_count(&mut reader, "n_cells")?;
    let n_faces = read_count(&mut reader, "n_faces")?;
    let n_interior_faces = if reader.has_field("n_interior_faces") {
        read_count(&mut reader, "n_interior_faces")?
    } else {
        0
    };

    let node_coords = reader.read_point3d_field("node_coords")?;
    let cell_center = reader.read_point2d_field("cell_center")?;
    let cell_area = read_scalar_field_f64(&mut reader, "cell_area")?;
    let cell_z_bed = read_scalar_field_f64(&mut reader, "cell_z_bed")?;
    let cell_node_offsets = read_offsets_as_usize(&mut reader, "cell_node_offsets")?;
    let cell_node_indices = reader.read_u32_field("cell_node_indices")?;
    let cell_face_offsets = read_offsets_as_usize(&mut reader, "cell_face_offsets")?;
    let cell_face_indices = reader.read_u32_field("cell_face_indices")?;
    let cell_neighbor_offsets = read_offsets_as_usize(&mut reader, "cell_neighbor_offsets")?;
    let cell_neighbor_indices = reader.read_u32_field("cell_neighbor_indices")?;
    let face_center = reader.read_point2d_field("face_center")?;
    let face_normal = reader.read_point3d_field("face_normal")?;
    let face_length = read_scalar_field_f64(&mut reader, "face_length")?;
    let face_z_left = read_scalar_field_f64(&mut reader, "face_z_left")?;
    let face_z_right = read_scalar_field_f64(&mut reader, "face_z_right")?;
    let face_owner = reader.read_u32_field("face_owner")?;
    let face_neighbor = reader.read_u32_field("face_neighbor")?;
    let face_delta_owner = reader.read_point2d_field("face_delta_owner")?;
    let face_delta_neighbor = reader.read_point2d_field("face_delta_neighbor")?;
    let face_dist_o2n = read_scalar_field_f64(&mut reader, "face_dist_o2n")?;

    let boundary_face_indices = read_optional_u32(&mut reader, "boundary_face_indices")?
        .unwrap_or_else(Vec::new);
    let face_boundary_id_raw = read_optional_u32(&mut reader, "face_boundary_id")?
        .unwrap_or_else(|| vec![u32::MAX; n_faces]);
    let boundary_names = read_boundary_names(&mut reader)?;

    let cell_refinement_level = read_optional_u8(&mut reader, "cell_refinement_level")?
        .unwrap_or_else(|| vec![0u8; n_cells]);
    let cell_parent = read_optional_u32(&mut reader, "cell_parent")?
        .unwrap_or_else(|| vec![u32::MAX; n_cells]);
    let ghost_capacity = read_optional_u32(&mut reader, "ghost_capacity")?
        .unwrap_or_else(|| vec![0u32; n_cells]);
    let cell_original_id = read_optional_u32(&mut reader, "cell_original_id")?
        .unwrap_or_else(|| (0..n_cells as u32).collect());
    let face_original_id = read_optional_u32(&mut reader, "face_original_id")?
        .unwrap_or_else(|| (0..n_faces as u32).collect());
    let cell_permutation = read_optional_u32(&mut reader, "cell_permutation")?
        .unwrap_or_else(|| (0..n_cells as u32).collect());
    let cell_inv_permutation = read_optional_u32(&mut reader, "cell_inv_permutation")?
        .unwrap_or_else(|| (0..n_cells as u32).collect());

    let face_boundary_id = face_boundary_id_raw
        .into_iter()
        .map(|v| if v == u32::MAX { None } else { Some(v) })
        .collect::<Vec<_>>();

    let (min_cell_size, max_cell_size) = compute_cell_size_stats(&cell_area);

    let ghost_capacity_value = ghost_capacity.iter().copied().max().unwrap_or(0) as usize;

    let mut mesh = FrozenMesh::empty_with_cells(n_cells);
    mesh.n_nodes = n_nodes;
    mesh.node_coords = node_coords;
    mesh.n_cells = n_cells;
    mesh.cell_center = cell_center;
    mesh.cell_area = cell_area;
    mesh.cell_z_bed = cell_z_bed;
    mesh.cell_node_offsets = cell_node_offsets;
    mesh.cell_node_indices = cell_node_indices;
    mesh.cell_face_offsets = cell_face_offsets;
    mesh.cell_face_indices = cell_face_indices;
    mesh.cell_neighbor_offsets = cell_neighbor_offsets;
    mesh.cell_neighbor_indices = cell_neighbor_indices;
    mesh.n_faces = n_faces;
    mesh.n_interior_faces = n_interior_faces;
    mesh.face_center = face_center;
    mesh.face_normal = face_normal;
    mesh.face_length = face_length;
    mesh.face_z_left = face_z_left;
    mesh.face_z_right = face_z_right;
    mesh.face_owner = face_owner;
    mesh.face_neighbor = face_neighbor;
    mesh.face_delta_owner = face_delta_owner;
    mesh.face_delta_neighbor = face_delta_neighbor;
    mesh.face_dist_o2n = face_dist_o2n;
    mesh.boundary_face_indices = boundary_face_indices;
    mesh.boundary_names = boundary_names;
    mesh.face_boundary_id = face_boundary_id;
    mesh.min_cell_size = min_cell_size;
    mesh.max_cell_size = max_cell_size;
    mesh.cell_refinement_level = cell_refinement_level;
    mesh.cell_parent = cell_parent;
    mesh.ghost_capacity = ghost_capacity_value;
    mesh.cell_original_id = cell_original_id;
    mesh.face_original_id = face_original_id;
    mesh.cell_permutation = cell_permutation;
    mesh.cell_inv_permutation = cell_inv_permutation;

    mesh.validate()
        .map_err(|e| Error::new(ErrorKind::InvalidData, format!("冻结网格校验失败: {}", e)))?;

    Ok(mesh)
}

fn write_mesh_fields<B: Backend>(mut writer: MhbWriter<File>, mesh: &FrozenMeshGeneric<B>) -> Result<()> {
    writer.write_u64_field("n_nodes", &[mesh.n_nodes as u64])?;
    writer.write_u64_field("n_cells", &[mesh.n_cells as u64])?;
    writer.write_u64_field("n_faces", &[mesh.n_faces as u64])?;
    writer.write_u64_field("n_interior_faces", &[mesh.n_interior_faces as u64])?;

    writer.write_point3d_field("node_coords", &mesh.node_coords)?;
    writer.write_point2d_field("cell_center", &mesh.cell_center)?;
    write_scalar_field(&mut writer, "cell_area", &mesh.cell_area)?;
    write_scalar_field(&mut writer, "cell_z_bed", &mesh.cell_z_bed)?;

    let cell_node_offsets: Vec<u64> = mesh.cell_node_offsets.iter().map(|&v| v as u64).collect();
    let cell_face_offsets: Vec<u64> = mesh.cell_face_offsets.iter().map(|&v| v as u64).collect();
    let cell_neighbor_offsets: Vec<u64> = mesh.cell_neighbor_offsets.iter().map(|&v| v as u64).collect();
    writer.write_u64_field("cell_node_offsets", &cell_node_offsets)?;
    writer.write_u32_field("cell_node_indices", &mesh.cell_node_indices)?;
    writer.write_u64_field("cell_face_offsets", &cell_face_offsets)?;
    writer.write_u32_field("cell_face_indices", &mesh.cell_face_indices)?;
    writer.write_u64_field("cell_neighbor_offsets", &cell_neighbor_offsets)?;
    writer.write_u32_field("cell_neighbor_indices", &mesh.cell_neighbor_indices)?;

    writer.write_point2d_field("face_center", &mesh.face_center)?;
    writer.write_point3d_field("face_normal", &mesh.face_normal)?;
    write_scalar_field(&mut writer, "face_length", &mesh.face_length)?;
    write_scalar_field(&mut writer, "face_z_left", &mesh.face_z_left)?;
    write_scalar_field(&mut writer, "face_z_right", &mesh.face_z_right)?;
    writer.write_u32_field("face_owner", &mesh.face_owner)?;
    writer.write_u32_field("face_neighbor", &mesh.face_neighbor)?;
    writer.write_point2d_field("face_delta_owner", &mesh.face_delta_owner)?;
    writer.write_point2d_field("face_delta_neighbor", &mesh.face_delta_neighbor)?;
    write_scalar_field(&mut writer, "face_dist_o2n", &mesh.face_dist_o2n)?;

    writer.write_u32_field("boundary_face_indices", &mesh.boundary_face_indices)?;
    let face_boundary_id: Vec<u32> = mesh
        .face_boundary_id
        .iter()
        .map(|v| v.unwrap_or(u32::MAX))
        .collect();
    writer.write_u32_field("face_boundary_id", &face_boundary_id)?;

    let boundary_names = serde_json::to_vec(&mesh.boundary_names)
        .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
    writer.write_u8_field("boundary_names", &boundary_names)?;

    writer.write_u8_field("cell_refinement_level", &mesh.cell_refinement_level)?;
    writer.write_u32_field("cell_parent", &mesh.cell_parent)?;
    writer.write_u32_field("ghost_capacity", &[mesh.ghost_capacity as u32])?;
    writer.write_u32_field("cell_original_id", &mesh.cell_original_id)?;
    writer.write_u32_field("face_original_id", &mesh.face_original_id)?;
    writer.write_u32_field("cell_permutation", &mesh.cell_permutation)?;
    writer.write_u32_field("cell_inv_permutation", &mesh.cell_inv_permutation)?;

    writer.finish()?;
    Ok(())
}

fn write_scalar_field<S: RuntimeScalar>(
    writer: &mut MhbWriter<File>,
    name: &str,
    data: &[S],
) -> Result<()> {
    if std::mem::size_of::<S>() == 4 {
        let buf: Vec<f32> = data.iter().map(|v| v.to_f32().unwrap_or(0.0)).collect();
        writer.write_f32_field(name, &buf)
    } else {
        let buf: Vec<f64> = data.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
        writer.write_f64_field(name, &buf)
    }
}

fn read_count(reader: &mut MhbReader<File>, name: &str) -> Result<usize> {
    match reader.get_field_dtype(name) {
        Ok(DataType::U64) => {
            let values = reader.read_u64_field(name)?;
            Ok(values.first().copied().unwrap_or(0) as usize)
        }
        Ok(DataType::U32) => {
            let values = reader.read_u32_field(name)?;
            Ok(values.first().copied().unwrap_or(0) as usize)
        }
        Ok(dtype) => Err(Error::new(
            ErrorKind::InvalidData,
            format!("Field {} has incompatible dtype {:?}", name, dtype),
        )),
        Err(e) => Err(e),
    }
}

fn read_offsets_as_usize(reader: &mut MhbReader<File>, name: &str) -> Result<Vec<usize>> {
    match reader.get_field_dtype(name) {
        Ok(DataType::U64) => Ok(reader
            .read_u64_field(name)?
            .into_iter()
            .map(|v| v as usize)
            .collect()),
        Ok(DataType::U32) => Ok(reader
            .read_u32_field(name)?
            .into_iter()
            .map(|v| v as usize)
            .collect()),
        Ok(dtype) => Err(Error::new(
            ErrorKind::InvalidData,
            format!("Field {} has incompatible dtype {:?}", name, dtype),
        )),
        Err(e) => Err(e),
    }
}

fn read_scalar_field_f64(reader: &mut MhbReader<File>, name: &str) -> Result<Vec<f64>> {
    match reader.get_field_dtype(name)? {
        DataType::F32 => Ok(reader.read_f32_field(name)?.into_iter().map(|v| v as f64).collect()),
        DataType::F64 => reader.read_f64_field(name),
        dtype => Err(Error::new(
            ErrorKind::InvalidData,
            format!("Field {} has incompatible dtype {:?}", name, dtype),
        )),
    }
}

fn read_optional_u32(reader: &mut MhbReader<File>, name: &str) -> Result<Option<Vec<u32>>> {
    if reader.has_field(name) {
        Ok(Some(reader.read_u32_field(name)?))
    } else {
        Ok(None)
    }
}

fn read_optional_u8(reader: &mut MhbReader<File>, name: &str) -> Result<Option<Vec<u8>>> {
    if reader.has_field(name) {
        Ok(Some(reader.read_u8_field(name)?))
    } else {
        Ok(None)
    }
}

fn read_boundary_names(reader: &mut MhbReader<File>) -> Result<Vec<String>> {
    if reader.has_field("boundary_names") {
        let bytes = reader.read_u8_field("boundary_names")?;
        let names: Vec<String> = serde_json::from_slice(&bytes)
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        Ok(names)
    } else {
        Ok(Vec::new())
    }
}

fn compute_cell_size_stats(cell_area: &[f64]) -> (f64, f64) {
    if cell_area.is_empty() {
        return (0.0, 0.0);
    }
    let mut min_area = f64::INFINITY;
    let mut max_area: f64 = 0.0;
    for &area in cell_area {
        if area > 0.0 {
            min_area = min_area.min(area);
            max_area = max_area.max(area);
        }
    }
    let min_size = if min_area.is_finite() { min_area.sqrt() } else { 0.0 };
    let max_size = max_area.sqrt();
    (min_size, max_size)
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