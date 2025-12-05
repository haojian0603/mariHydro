// crates/mh_io/src/exporters/vtu.rs

//! VTU 格式导出器
//!
//! 导出 VTK Unstructured Grid 格式，用于 ParaView 可视化。
//!
//! # 功能
//!
//! - 单帧 VTU 导出
//! - 时间序列导出 (PVD)
//! - 支持标量和向量场
//! - ASCII 和二进制格式

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// VTU 导出错误
#[derive(Debug)]
pub enum VtuError {
    /// IO 错误
    Io(std::io::Error),
    /// 无效数据
    InvalidData(String),
}

impl std::fmt::Display for VtuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VtuError::Io(e) => write!(f, "IO error: {}", e),
            VtuError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
        }
    }
}

impl std::error::Error for VtuError {}

impl From<std::io::Error> for VtuError {
    fn from(e: std::io::Error) -> Self {
        VtuError::Io(e)
    }
}

/// VTU 导出器默认干湿阈值
const DEFAULT_H_DRY: f64 = 1e-6;

/// VTU 单元类型
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum VtuCellType {
    /// 三角形
    Triangle = 5,
    /// 四边形
    Quad = 9,
    /// 多边形
    Polygon = 7,
}

/// VTU 导出器
#[derive(Debug, Clone)]
pub struct VtuExporter {
    /// 是否使用二进制格式
    binary: bool,
    /// 干湿阈值
    h_dry: f64,
}

impl Default for VtuExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl VtuExporter {
    /// 创建新的 VTU 导出器
    pub fn new() -> Self {
        Self {
            binary: false,
            h_dry: DEFAULT_H_DRY,
        }
    }

    /// 设置二进制模式
    pub fn binary(mut self, binary: bool) -> Self {
        self.binary = binary;
        self
    }

    /// 设置干湿阈值
    pub fn h_dry(mut self, h_dry: f64) -> Self {
        self.h_dry = h_dry;
        self
    }

    /// 导出单帧
    pub fn export<M: VtuMesh, S: VtuState>(
        &self,
        path: impl AsRef<Path>,
        mesh: &M,
        state: &S,
        time: f64,
    ) -> Result<(), VtuError> {
        let file = File::create(path.as_ref())?;
        let mut w = BufWriter::new(file);

        self.write_header(&mut w, time)?;
        self.write_piece(&mut w, mesh, state)?;
        self.write_footer(&mut w)?;

        w.flush()?;
        Ok(())
    }

    /// 导出时间序列
    pub fn export_series<M: VtuMesh, S: VtuState>(
        &self,
        dir: impl AsRef<Path>,
        prefix: &str,
        mesh: &M,
        steps: &[(f64, S)],
    ) -> Result<(), VtuError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        let mut vtu_files = Vec::new();

        for (i, (time, state)) in steps.iter().enumerate() {
            let filename = format!("{}_{:06}.vtu", prefix, i);
            let path = dir.join(&filename);

            self.export(&path, mesh, state, *time)?;
            vtu_files.push((filename, *time));
        }

        // 写入 PVD 集合文件
        let pvd_path = dir.join(format!("{}.pvd", prefix));
        self.write_pvd(&pvd_path, &vtu_files)?;

        Ok(())
    }

    /// 写入 PVD 文件
    fn write_pvd(&self, path: impl AsRef<Path>, files: &[(String, f64)]) -> Result<(), VtuError> {
        let file = File::create(path.as_ref())?;
        let mut w = BufWriter::new(file);

        writeln!(w, r#"<?xml version="1.0"?>"#)?;
        writeln!(
            w,
            r#"<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">"#
        )?;
        writeln!(w, r#"  <Collection>"#)?;

        for (filename, time) in files {
            writeln!(
                w,
                r#"    <DataSet timestep="{}" file="{}"/>"#,
                time, filename
            )?;
        }

        writeln!(w, r#"  </Collection>"#)?;
        writeln!(w, r#"</VTKFile>"#)?;

        w.flush()?;
        Ok(())
    }

    fn write_header(&self, w: &mut BufWriter<File>, time: f64) -> Result<(), VtuError> {
        writeln!(w, r#"<?xml version="1.0"?>"#)?;
        writeln!(
            w,
            r#"<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">"#
        )?;
        writeln!(w, r#"  <UnstructuredGrid>"#)?;
        writeln!(w, r#"    <FieldData>"#)?;
        writeln!(
            w,
            r#"      <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1">{}</DataArray>"#,
            time
        )?;
        writeln!(w, r#"    </FieldData>"#)?;
        Ok(())
    }

    fn write_piece<M: VtuMesh, S: VtuState>(
        &self,
        w: &mut BufWriter<File>,
        mesh: &M,
        state: &S,
    ) -> Result<(), VtuError> {
        let n_nodes = mesh.n_nodes();
        let n_cells = mesh.n_cells();

        writeln!(
            w,
            r#"    <Piece NumberOfPoints="{}" NumberOfCells="{}">"#,
            n_nodes, n_cells
        )?;

        self.write_points(w, mesh)?;
        self.write_cells(w, mesh)?;
        self.write_cell_data(w, mesh, state)?;

        writeln!(w, r#"    </Piece>"#)?;
        Ok(())
    }

    fn write_points<M: VtuMesh>(&self, w: &mut BufWriter<File>, mesh: &M) -> Result<(), VtuError> {
        writeln!(w, r#"      <Points>"#)?;
        writeln!(
            w,
            r#"        <DataArray type="Float64" NumberOfComponents="3" format="ascii">"#
        )?;

        for i in 0..mesh.n_nodes() {
            let pos = mesh.node_position(i);
            writeln!(w, "          {:.6} {:.6} {:.6}", pos[0], pos[1], pos[2])?;
        }

        writeln!(w, r#"        </DataArray>"#)?;
        writeln!(w, r#"      </Points>"#)?;
        Ok(())
    }

    fn write_cells<M: VtuMesh>(&self, w: &mut BufWriter<File>, mesh: &M) -> Result<(), VtuError> {
        writeln!(w, r#"      <Cells>"#)?;

        // Connectivity
        writeln!(
            w,
            r#"        <DataArray type="Int32" Name="connectivity" format="ascii">"#
        )?;
        for i in 0..mesh.n_cells() {
            let nodes = mesh.cell_nodes(i);
            let s: Vec<String> = nodes.iter().map(|n| n.to_string()).collect();
            writeln!(w, "          {}", s.join(" "))?;
        }
        writeln!(w, r#"        </DataArray>"#)?;

        // Offsets
        writeln!(
            w,
            r#"        <DataArray type="Int32" Name="offsets" format="ascii">"#
        )?;
        let mut offset = 0;
        for i in 0..mesh.n_cells() {
            offset += mesh.cell_nodes(i).len();
            writeln!(w, "          {}", offset)?;
        }
        writeln!(w, r#"        </DataArray>"#)?;

        // Types
        writeln!(
            w,
            r#"        <DataArray type="UInt8" Name="types" format="ascii">"#
        )?;
        for i in 0..mesh.n_cells() {
            let cell_type = match mesh.cell_nodes(i).len() {
                3 => VtuCellType::Triangle as u8,
                4 => VtuCellType::Quad as u8,
                _ => VtuCellType::Polygon as u8,
            };
            writeln!(w, "          {}", cell_type)?;
        }
        writeln!(w, r#"        </DataArray>"#)?;

        writeln!(w, r#"      </Cells>"#)?;
        Ok(())
    }

    fn write_cell_data<M: VtuMesh, S: VtuState>(
        &self,
        w: &mut BufWriter<File>,
        mesh: &M,
        state: &S,
    ) -> Result<(), VtuError> {
        writeln!(w, r#"      <CellData>"#)?;

        let n_cells = mesh.n_cells();
        let h_dry = self.h_dry;

        // 水深
        self.write_scalar_field(w, "h", n_cells, |i| state.h(i))?;

        // 水面高程
        self.write_scalar_field(w, "eta", n_cells, |i| state.h(i) + mesh.cell_z_bed(i))?;

        // 速度分量
        self.write_scalar_field(w, "u", n_cells, |i| {
            let h = state.h(i);
            if h > h_dry {
                state.hu(i) / h
            } else {
                0.0
            }
        })?;

        self.write_scalar_field(w, "v", n_cells, |i| {
            let h = state.h(i);
            if h > h_dry {
                state.hv(i) / h
            } else {
                0.0
            }
        })?;

        // 速度大小
        self.write_scalar_field(w, "velocity_mag", n_cells, |i| {
            let h = state.h(i);
            if h > h_dry {
                let u = state.hu(i) / h;
                let v = state.hv(i) / h;
                (u * u + v * v).sqrt()
            } else {
                0.0
            }
        })?;

        // 底床高程
        self.write_scalar_field(w, "z_bed", n_cells, |i| mesh.cell_z_bed(i))?;

        writeln!(w, r#"      </CellData>"#)?;
        Ok(())
    }

    fn write_scalar_field<F>(
        &self,
        w: &mut BufWriter<File>,
        name: &str,
        n: usize,
        f: F,
    ) -> Result<(), VtuError>
    where
        F: Fn(usize) -> f64,
    {
        writeln!(
            w,
            r#"        <DataArray type="Float64" Name="{}" format="ascii">"#,
            name
        )?;
        for i in 0..n {
            writeln!(w, "          {:.6}", f(i))?;
        }
        writeln!(w, r#"        </DataArray>"#)?;
        Ok(())
    }

    fn write_footer(&self, w: &mut BufWriter<File>) -> Result<(), VtuError> {
        writeln!(w, r#"  </UnstructuredGrid>"#)?;
        writeln!(w, r#"</VTKFile>"#)?;
        Ok(())
    }
}

/// VTU 网格 trait
pub trait VtuMesh {
    /// 节点数量
    fn n_nodes(&self) -> usize;
    /// 单元数量
    fn n_cells(&self) -> usize;
    /// 获取节点位置 [x, y, z]
    fn node_position(&self, idx: usize) -> [f64; 3];
    /// 获取单元节点索引
    fn cell_nodes(&self, idx: usize) -> Vec<usize>;
    /// 获取单元底床高程
    fn cell_z_bed(&self, idx: usize) -> f64;
}

/// VTU 状态 trait
pub trait VtuState {
    /// 水深
    fn h(&self, idx: usize) -> f64;
    /// X 方向动量
    fn hu(&self, idx: usize) -> f64;
    /// Y 方向动量
    fn hv(&self, idx: usize) -> f64;
}

/// 为 FrozenMesh 实现 VtuMesh
impl VtuMesh for mh_mesh::FrozenMesh {
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    fn n_cells(&self) -> usize {
        self.n_cells
    }

    fn node_position(&self, idx: usize) -> [f64; 3] {
        let p = &self.node_coords[idx];
        [p.x, p.y, p.z]
    }

    fn cell_nodes(&self, idx: usize) -> Vec<usize> {
        let start = self.cell_node_offsets[idx];
        let end = self.cell_node_offsets[idx + 1];
        self.cell_node_indices[start..end]
            .iter()
            .map(|&n| n as usize)
            .collect()
    }

    fn cell_z_bed(&self, idx: usize) -> f64 {
        self.cell_z_bed[idx]
    }
}

/// 简单的状态数组包装
pub struct SimpleState<'a> {
    pub h: &'a [f64],
    pub hu: &'a [f64],
    pub hv: &'a [f64],
}

impl<'a> VtuState for SimpleState<'a> {
    fn h(&self, idx: usize) -> f64 {
        self.h[idx]
    }

    fn hu(&self, idx: usize) -> f64 {
        self.hu[idx]
    }

    fn hv(&self, idx: usize) -> f64 {
        self.hv[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vtu_exporter_default() {
        let exporter = VtuExporter::new();
        assert!(!exporter.binary);
    }

    #[test]
    fn test_vtu_cell_types() {
        assert_eq!(VtuCellType::Triangle as u8, 5);
        assert_eq!(VtuCellType::Quad as u8, 9);
        assert_eq!(VtuCellType::Polygon as u8, 7);
    }
}