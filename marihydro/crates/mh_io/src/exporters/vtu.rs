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

use base64::engine::general_purpose::STANDARD as BASE64_STD;
use base64::Engine;

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
        let mut config = VtuExportConfig::default();
        config.binary = self.binary;
        config.h_dry = self.h_dry;
        self.export_with_config(path, mesh, state, time, &config)
    }

    /// 导出单帧（带配置）
    pub fn export_with_config<M: VtuMesh, S: VtuState>(
        &self,
        path: impl AsRef<Path>,
        mesh: &M,
        state: &S,
        time: f64,
        config: &VtuExportConfig,
    ) -> Result<(), VtuError> {
        if !time.is_finite() {
            return Err(VtuError::InvalidData(format!("time 无效: {}", time)));
        }
        let n_cells = mesh.n_cells();
        let state_cells = state.n_cells();
        if state_cells != 0 && state_cells != n_cells {
            return Err(VtuError::InvalidData(format!(
                "状态单元数不一致: mesh={}, state={}",
                n_cells, state_cells
            )));
        }
        if config.precision == 0 || config.precision > 16 {
            return Err(VtuError::InvalidData(format!(
                "precision 无效: {} (应在 1..=16)",
                config.precision
            )));
        }
        if !config.h_dry.is_finite() || config.h_dry < 0.0 {
            return Err(VtuError::InvalidData(format!(
                "h_dry 无效: {}",
                config.h_dry
            )));
        }
        if !config.extra_scalars.is_empty() {
            let available: std::collections::HashSet<_> =
                state.available_scalars().into_iter().collect();
            for name in &config.extra_scalars {
                if !available.contains(name) {
                    return Err(VtuError::InvalidData(format!(
                        "请求的标量场不存在: {}",
                        name
                    )));
                }
            }
        }

        let file = File::create(path.as_ref())?;
        let mut w = BufWriter::new(file);

        self.write_header_with_config(&mut w, time, config)?;
        self.write_piece_with_config(&mut w, mesh, state, config)?;
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
        let mut config = VtuExportConfig::default();
        config.binary = self.binary;
        config.h_dry = self.h_dry;
        self.export_series_with_config(dir, prefix, mesh, steps, &config)
    }

    /// 导出时间序列（带配置）
    pub fn export_series_with_config<M: VtuMesh, S: VtuState>(
        &self,
        dir: impl AsRef<Path>,
        prefix: &str,
        mesh: &M,
        steps: &[(f64, S)],
        config: &VtuExportConfig,
    ) -> Result<(), VtuError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        let mut vtu_files = Vec::new();

        for (i, (time, state)) in steps.iter().enumerate() {
            let filename = format!("{}_{:06}.vtu", prefix, i);
            let path = dir.join(&filename);

            self.export_with_config(&path, mesh, state, *time, config)?;
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
        let mut config = VtuExportConfig::default();
        config.binary = self.binary;
        self.write_header_with_config(w, time, &config)
    }

    fn write_header_with_config(
        &self,
        w: &mut BufWriter<File>,
        time: f64,
        config: &VtuExportConfig,
    ) -> Result<(), VtuError> {
        writeln!(w, r#"<?xml version="1.0"?>"#)?;
        if config.binary {
            writeln!(
                w,
                r#"<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">"#
            )?;
        } else {
            writeln!(
                w,
                r#"<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">"#
            )?;
        }
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
        let config = VtuExportConfig::default();
        self.write_piece_with_config(w, mesh, state, &config)
    }

    fn write_piece_with_config<M: VtuMesh, S: VtuState>(
        &self,
        w: &mut BufWriter<File>,
        mesh: &M,
        state: &S,
        config: &VtuExportConfig,
    ) -> Result<(), VtuError> {
        let n_nodes = mesh.n_nodes();
        let n_cells = mesh.n_cells();

        writeln!(
            w,
            r#"    <Piece NumberOfPoints="{}" NumberOfCells="{}">"#,
            n_nodes, n_cells
        )?;

        self.write_points_with_config(w, mesh, config)?;
        self.write_cells_with_config(w, mesh, config)?;
        self.write_cell_data_with_config(w, mesh, state, config)?;

        writeln!(w, r#"    </Piece>"#)?;
        Ok(())
    }

    fn write_points<M: VtuMesh>(&self, w: &mut BufWriter<File>, mesh: &M) -> Result<(), VtuError> {
        let config = VtuExportConfig::default();
        self.write_points_with_config(w, mesh, &config)
    }

    fn write_points_with_config<M: VtuMesh>(
        &self,
        w: &mut BufWriter<File>,
        mesh: &M,
        config: &VtuExportConfig,
    ) -> Result<(), VtuError> {
        writeln!(w, r#"      <Points>"#)?;
        if config.binary {
            let mut payload = Vec::with_capacity(mesh.n_nodes() * 3 * 8);
            for i in 0..mesh.n_nodes() {
                let pos = mesh.node_position(i);
                push_f64_bytes(&mut payload, pos[0]);
                push_f64_bytes(&mut payload, pos[1]);
                push_f64_bytes(&mut payload, pos[2]);
            }
            let encoded = encode_binary_payload(&payload);
            writeln!(
                w,
                r#"        <DataArray type="Float64" NumberOfComponents="3" format="binary">"#
            )?;
            writeln!(w, "          {}", encoded)?;
            writeln!(w, r#"        </DataArray>"#)?;
        } else {
            let precision = config.precision;
            writeln!(
                w,
                r#"        <DataArray type="Float64" NumberOfComponents="3" format="ascii">"#
            )?;

            for i in 0..mesh.n_nodes() {
                let pos = mesh.node_position(i);
                writeln!(
                    w,
                    "          {:.*} {:.*} {:.*}",
                    precision,
                    pos[0],
                    precision,
                    pos[1],
                    precision,
                    pos[2]
                )?;
            }

            writeln!(w, r#"        </DataArray>"#)?;
        }
        writeln!(w, r#"      </Points>"#)?;
        Ok(())
    }

    fn write_cells<M: VtuMesh>(&self, w: &mut BufWriter<File>, mesh: &M) -> Result<(), VtuError> {
        let config = VtuExportConfig::default();
        self.write_cells_with_config(w, mesh, &config)
    }

    fn write_cells_with_config<M: VtuMesh>(
        &self,
        w: &mut BufWriter<File>,
        mesh: &M,
        config: &VtuExportConfig,
    ) -> Result<(), VtuError> {
        writeln!(w, r#"      <Cells>"#)?;
        let n_nodes = mesh.n_nodes();

        // Connectivity
        if config.binary {
            let mut payload = Vec::new();
            for i in 0..mesh.n_cells() {
                let nodes = mesh.cell_nodes(i);
                for &n in &nodes {
                    if n >= n_nodes {
                        return Err(VtuError::InvalidData(format!(
                            "单元节点索引越界: node={}, n_nodes={} (cell={})",
                            n, n_nodes, i
                        )));
                    }
                    let v = i32::try_from(n).map_err(|_| {
                        VtuError::InvalidData(format!("节点索引超出 Int32 范围: {}", n))
                    })?;
                    push_i32_bytes(&mut payload, v);
                }
            }
            let encoded = encode_binary_payload(&payload);
            writeln!(
                w,
                r#"        <DataArray type="Int32" Name="connectivity" format="binary">"#
            )?;
            writeln!(w, "          {}", encoded)?;
            writeln!(w, r#"        </DataArray>"#)?;
        } else {
            writeln!(
                w,
                r#"        <DataArray type="Int32" Name="connectivity" format="ascii">"#
            )?;
            for i in 0..mesh.n_cells() {
                let nodes = mesh.cell_nodes(i);
                for &n in &nodes {
                    if n >= n_nodes {
                        return Err(VtuError::InvalidData(format!(
                            "单元节点索引越界: node={}, n_nodes={} (cell={})",
                            n, n_nodes, i
                        )));
                    }
                }
                let s: Vec<String> = nodes.iter().map(|n| n.to_string()).collect();
                writeln!(w, "          {}", s.join(" "))?;
            }
            writeln!(w, r#"        </DataArray>"#)?;
        }

        // Offsets
        if config.binary {
            let mut payload = Vec::with_capacity(mesh.n_cells() * 4);
            let mut offset: usize = 0;
            for i in 0..mesh.n_cells() {
                offset = offset.saturating_add(mesh.cell_nodes(i).len());
                let v = i32::try_from(offset).map_err(|_| {
                    VtuError::InvalidData(format!("offset 超出 Int32 范围: {}", offset))
                })?;
                push_i32_bytes(&mut payload, v);
            }
            let encoded = encode_binary_payload(&payload);
            writeln!(
                w,
                r#"        <DataArray type="Int32" Name="offsets" format="binary">"#
            )?;
            writeln!(w, "          {}", encoded)?;
            writeln!(w, r#"        </DataArray>"#)?;
        } else {
            writeln!(
                w,
                r#"        <DataArray type="Int32" Name="offsets" format="ascii">"#
            )?;
            let mut offset = 0;
            for i in 0..mesh.n_cells() {
                offset += mesh.cell_nodes(i).len();
                if offset > i32::MAX as usize {
                    return Err(VtuError::InvalidData(format!(
                        "offset 超出 Int32 范围: {}",
                        offset
                    )));
                }
                writeln!(w, "          {}", offset)?;
            }
            writeln!(w, r#"        </DataArray>"#)?;
        }

        // Types
        if config.binary {
            let mut payload = Vec::with_capacity(mesh.n_cells());
            for i in 0..mesh.n_cells() {
                let cell_type = match mesh.cell_nodes(i).len() {
                    3 => VtuCellType::Triangle as u8,
                    4 => VtuCellType::Quad as u8,
                    _ => VtuCellType::Polygon as u8,
                };
                payload.push(cell_type);
            }
            let encoded = encode_binary_payload(&payload);
            writeln!(
                w,
                r#"        <DataArray type="UInt8" Name="types" format="binary">"#
            )?;
            writeln!(w, "          {}", encoded)?;
            writeln!(w, r#"        </DataArray>"#)?;
        } else {
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
        }

        writeln!(w, r#"      </Cells>"#)?;
        Ok(())
    }

    fn write_cell_data<M: VtuMesh, S: VtuState>(
        &self,
        w: &mut BufWriter<File>,
        mesh: &M,
        state: &S,
    ) -> Result<(), VtuError> {
        let config = VtuExportConfig::default();
        self.write_cell_data_with_config(w, mesh, state, &config)
    }

    fn write_cell_data_with_config<M: VtuMesh, S: VtuState>(
        &self,
        w: &mut BufWriter<File>,
        mesh: &M,
        state: &S,
        config: &VtuExportConfig,
    ) -> Result<(), VtuError> {
        writeln!(w, r#"      <CellData>"#)?;

        let n_cells = mesh.n_cells();
        let h_dry = config.h_dry;
        let precision = config.precision;

        for i in 0..n_cells {
            let h = state.h(i);
            if !h.is_finite() || h < 0.0 {
                return Err(VtuError::InvalidData(format!(
                    "水深无效: h[{}]={}",
                    i, h
                )));
            }
        }

        // 水深
        self.write_scalar_field(w, "h", n_cells, precision, config.binary, |i| state.h(i))?;

        // 水面高程
        if config.export_eta {
            self.write_scalar_field(
                w,
                "eta",
                n_cells,
                precision,
                config.binary,
                |i| state.h(i) + mesh.cell_z_bed(i),
            )?;
        }

        // 速度分量
        if config.export_velocity {
            self.write_scalar_field(w, "u", n_cells, precision, config.binary, |i| {
                let h = state.h(i);
                if h > h_dry {
                    state.hu(i) / h
                } else {
                    0.0
                }
            })?;

            self.write_scalar_field(w, "v", n_cells, precision, config.binary, |i| {
                let h = state.h(i);
                if h > h_dry {
                    state.hv(i) / h
                } else {
                    0.0
                }
            })?;

            self.write_scalar_field(w, "velocity_mag", n_cells, precision, config.binary, |i| {
                let h = state.h(i);
                if h > h_dry {
                    let u = state.hu(i) / h;
                    let v = state.hv(i) / h;
                    (u * u + v * v).sqrt()
                } else {
                    0.0
                }
            })?;
        }

        // 底床高程
        if config.export_bed {
            self.write_scalar_field(w, "z_bed", n_cells, precision, config.binary, |i| {
                mesh.cell_z_bed(i)
            })?;
        }

        // 弗劳德数
        if config.export_froude {
            const GRAVITY: f64 = 9.81;
            self.write_scalar_field(w, "froude", n_cells, precision, config.binary, |i| {
                let h = state.h(i);
                if h > h_dry {
                    let u = state.hu(i) / h;
                    let v = state.hv(i) / h;
                    let vel = (u * u + v * v).sqrt();
                    let c = (GRAVITY * h).sqrt();
                    vel / c
                } else {
                    0.0
                }
            })?;
        }

        // 附加标量
        if !config.extra_scalars.is_empty() {
            for name in &config.extra_scalars {
                let name = name.as_str();
                let mut values = Vec::with_capacity(n_cells);
                for i in 0..n_cells {
                    match state.scalar(name, i) {
                        Some(v) => values.push(v),
                        None => {
                            return Err(VtuError::InvalidData(format!(
                                "标量场缺失: {} (cell={})",
                                name, i
                            )));
                        }
                    }
                }
                self.write_scalar_field_values(w, name, &values, precision, config.binary)?;
            }
        }

        writeln!(w, r#"      </CellData>"#)?;
        Ok(())
    }

    fn write_scalar_field<F>(
        &self,
        w: &mut BufWriter<File>,
        name: &str,
        n: usize,
        precision: usize,
        binary: bool,
        f: F,
    ) -> Result<(), VtuError>
    where
        F: Fn(usize) -> f64,
    {
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            values.push(f(i));
        }
        self.write_scalar_field_values(w, name, &values, precision, binary)
    }

    fn write_scalar_field_values(
        &self,
        w: &mut BufWriter<File>,
        name: &str,
        values: &[f64],
        precision: usize,
        binary: bool,
    ) -> Result<(), VtuError> {
        for (i, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(VtuError::InvalidData(format!(
                    "标量场 {} 包含非有限值: index={}, value={}",
                    name, i, value
                )));
            }
        }
        if binary {
            let mut payload = Vec::with_capacity(values.len() * 8);
            for &value in values {
                push_f64_bytes(&mut payload, value);
            }
            let encoded = encode_binary_payload(&payload);
            writeln!(
                w,
                r#"        <DataArray type="Float64" Name="{}" format="binary">"#,
                name
            )?;
            writeln!(w, "          {}", encoded)?;
            writeln!(w, r#"        </DataArray>"#)?;
        } else {
            writeln!(
                w,
                r#"        <DataArray type="Float64" Name="{}" format="ascii">"#,
                name
            )?;
            for &value in values {
                writeln!(w, "          {:.*}", precision, value)?;
            }
            writeln!(w, r#"        </DataArray>"#)?;
        }
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

// ============================================================
// 二进制编码辅助
// ============================================================

fn encode_binary_payload(payload: &[u8]) -> String {
    let mut buffer = Vec::with_capacity(4 + payload.len());
    buffer.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buffer.extend_from_slice(payload);
    BASE64_STD.encode(buffer)
}

#[inline]
fn push_f64_bytes(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[inline]
fn push_i32_bytes(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
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
            .and_then(|(_, data)| data.get(idx).copied())
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
