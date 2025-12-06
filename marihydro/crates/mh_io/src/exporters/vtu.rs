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

// ============================================================
// VTU 导出配置
// ============================================================

/// VTU 导出配置
#[derive(Debug, Clone)]
pub struct VtuExportConfig {
    /// 是否使用二进制格式
    pub binary: bool,
    /// 干湿阈值
    pub h_dry: f64,
    /// 是否导出速度场
    pub export_velocity: bool,
    /// 是否导出底床高程
    pub export_bed: bool,
    /// 是否导出水位
    pub export_eta: bool,
    /// 是否导出弗劳德数
    pub export_froude: bool,
    /// 附加标量场名称
    pub extra_scalars: Vec<String>,
    /// 精度（小数位数）
    pub precision: usize,
}

impl Default for VtuExportConfig {
    fn default() -> Self {
        Self {
            binary: false,
            h_dry: 1e-6,
            export_velocity: true,
            export_bed: true,
            export_eta: true,
            export_froude: false,
            extra_scalars: Vec::new(),
            precision: 6,
        }
    }
}

impl VtuExportConfig {
    /// 创建完整导出配置（所有字段）
    pub fn full() -> Self {
        Self {
            export_froude: true,
            ..Default::default()
        }
    }

    /// 创建精简导出配置（仅基本字段）
    pub fn minimal() -> Self {
        Self {
            export_velocity: false,
            export_froude: false,
            ..Default::default()
        }
    }
}

// ============================================================
// VTU Trait 定义
// ============================================================

/// VTU 网格 trait
///
/// 提供 VTU 导出所需的网格信息。
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
    /// 获取单元面积（可选，用于计算弗劳德数）
    fn cell_area(&self, _idx: usize) -> f64 {
        1.0
    }
}

/// VTU 状态 trait
///
/// 提供 VTU 导出所需的状态信息。
pub trait VtuState {
    /// 单元数量
    fn n_cells(&self) -> usize {
        0
    }
    /// 水深 [m]
    fn h(&self, idx: usize) -> f64;
    /// X 方向动量 [m²/s]
    fn hu(&self, idx: usize) -> f64;
    /// Y 方向动量 [m²/s]
    fn hv(&self, idx: usize) -> f64;
    /// 获取附加标量（可选）
    fn scalar(&self, _name: &str, _idx: usize) -> Option<f64> {
        None
    }
    /// 可用的标量场名称
    fn available_scalars(&self) -> Vec<String> {
        Vec::new()
    }
}

/// VTU 状态扩展（提供便捷计算方法）
pub trait VtuStateExt: VtuState {
    /// 计算速度 u
    fn velocity_u(&self, idx: usize, h_dry: f64) -> f64 {
        let h = self.h(idx);
        if h > h_dry {
            self.hu(idx) / h
        } else {
            0.0
        }
    }

    /// 计算速度 v
    fn velocity_v(&self, idx: usize, h_dry: f64) -> f64 {
        let h = self.h(idx);
        if h > h_dry {
            self.hv(idx) / h
        } else {
            0.0
        }
    }

    /// 计算速度大小
    fn velocity_magnitude(&self, idx: usize, h_dry: f64) -> f64 {
        let u = self.velocity_u(idx, h_dry);
        let v = self.velocity_v(idx, h_dry);
        (u * u + v * v).sqrt()
    }

    /// 计算弗劳德数
    fn froude_number(&self, idx: usize, h_dry: f64, gravity: f64) -> f64 {
        let h = self.h(idx);
        if h > h_dry {
            let vel = self.velocity_magnitude(idx, h_dry);
            let c = (gravity * h).sqrt();
            vel / c
        } else {
            0.0
        }
    }
}

// 自动实现
impl<T: VtuState + ?Sized> VtuStateExt for T {}

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

    fn cell_area(&self, idx: usize) -> f64 {
        self.cell_area[idx]
    }
}

/// 简单的状态数组包装
pub struct SimpleState<'a> {
    pub h: &'a [f64],
    pub hu: &'a [f64],
    pub hv: &'a [f64],
}

impl<'a> SimpleState<'a> {
    /// 创建新的简单状态
    pub fn new(h: &'a [f64], hu: &'a [f64], hv: &'a [f64]) -> Self {
        Self { h, hu, hv }
    }
}

impl<'a> VtuState for SimpleState<'a> {
    fn n_cells(&self) -> usize {
        self.h.len()
    }

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

/// 带附加标量的状态
pub struct StateWithScalars<'a> {
    /// 基础状态
    pub base: SimpleState<'a>,
    /// 标量场 (名称, 数据)
    pub scalars: Vec<(&'a str, &'a [f64])>,
}

impl<'a> StateWithScalars<'a> {
    /// 创建新状态
    pub fn new(h: &'a [f64], hu: &'a [f64], hv: &'a [f64]) -> Self {
        Self {
            base: SimpleState::new(h, hu, hv),
            scalars: Vec::new(),
        }
    }

    /// 添加标量场
    pub fn with_scalar(mut self, name: &'a str, data: &'a [f64]) -> Self {
        self.scalars.push((name, data));
        self
    }
}

impl<'a> VtuState for StateWithScalars<'a> {
    fn n_cells(&self) -> usize {
        self.base.h.len()
    }

    fn h(&self, idx: usize) -> f64 {
        self.base.h(idx)
    }

    fn hu(&self, idx: usize) -> f64 {
        self.base.hu(idx)
    }

    fn hv(&self, idx: usize) -> f64 {
        self.base.hv(idx)
    }

    fn scalar(&self, name: &str, idx: usize) -> Option<f64> {
        self.scalars
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, data)| data[idx])
    }

    fn available_scalars(&self) -> Vec<String> {
        self.scalars.iter().map(|(n, _)| n.to_string()).collect()
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

    #[test]
    fn test_vtu_export_config() {
        let config = VtuExportConfig::default();
        assert!(config.export_velocity);
        assert!(!config.export_froude);

        let full = VtuExportConfig::full();
        assert!(full.export_froude);

        let minimal = VtuExportConfig::minimal();
        assert!(!minimal.export_velocity);
    }

    #[test]
    fn test_simple_state() {
        let h = vec![1.0, 2.0, 3.0];
        let hu = vec![0.1, 0.2, 0.3];
        let hv = vec![0.0, 0.0, 0.0];

        let state = SimpleState::new(&h, &hu, &hv);
        assert_eq!(state.n_cells(), 3);
        assert!((state.h(0) - 1.0).abs() < 1e-10);

        // 测试扩展方法
        let u = state.velocity_u(0, 1e-6);
        assert!((u - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_state_with_scalars() {
        let h = vec![1.0, 2.0];
        let hu = vec![0.0; 2];
        let hv = vec![0.0; 2];
        let temp = vec![20.0, 21.0];

        let state = StateWithScalars::new(&h, &hu, &hv).with_scalar("temperature", &temp);

        assert_eq!(state.available_scalars(), vec!["temperature"]);
        assert!((state.scalar("temperature", 0).unwrap() - 20.0).abs() < 1e-10);
        assert!(state.scalar("salinity", 0).is_none());
    }

    #[test]
    fn test_froude_number() {
        let h = vec![1.0, 0.0001];
        let hu = vec![3.13, 0.0]; // 大约 Fr = 1 对于 h=1, g=9.81
        let hv = vec![0.0, 0.0];

        let state = SimpleState::new(&h, &hu, &hv);
        let fr = state.froude_number(0, 1e-6, 9.81);
        assert!((fr - 1.0).abs() < 0.01);

        // 干单元应该返回 0
        let fr_dry = state.froude_number(1, 0.001, 9.81);
        assert!((fr_dry - 0.0).abs() < 1e-10);
    }
}
