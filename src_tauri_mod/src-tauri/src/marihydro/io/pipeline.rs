// File: src-tauri/src/marihydro/io/pipeline.rs
//! 异步IO管道
//! 
//! 提供异步文件写入功能，避免阻塞计算线程。
//! 支持二进制VTU格式输出以提高性能。

use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::state::StateAccess;
use crate::marihydro::core::types::CellIndex;
use std::collections::VecDeque;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

/// 输出请求
pub enum OutputRequest {
    /// 写入VTU文件（ASCII格式）
    WriteVtuAscii {
        path: PathBuf,
        mesh_data: MeshSnapshot,
        state_data: StateSnapshot,
        time: f64,
    },
    /// 写入VTU文件（二进制格式）
    WriteVtuBinary {
        path: PathBuf,
        mesh_data: MeshSnapshot,
        state_data: StateSnapshot,
        time: f64,
    },
    /// 写入检查点
    WriteCheckpoint {
        path: PathBuf,
        state_data: StateSnapshot,
        time: f64,
        step: usize,
    },
    /// 停止管道
    Shutdown,
}

/// 网格快照（用于异步传输）
#[derive(Clone)]
pub struct MeshSnapshot {
    pub n_nodes: usize,
    pub n_cells: usize,
    pub node_positions: Vec<(f64, f64)>,
    pub cell_nodes: Vec<Vec<usize>>,
    pub cell_areas: Vec<f64>,
    pub bed_elevations: Vec<f64>,
}

impl MeshSnapshot {
    /// 从网格创建快照
    pub fn from_mesh<M: MeshAccess>(mesh: &M) -> Self {
        let n_nodes = mesh.n_nodes();
        let n_cells = mesh.n_cells();
        
        let node_positions: Vec<_> = (0..n_nodes)
            .map(|i| {
                let pos = mesh.node_position(i);
                (pos.x, pos.y)
            })
            .collect();
        
        let cell_nodes: Vec<_> = (0..n_cells)
            .map(|i| {
                mesh.cell_nodes(CellIndex(i))
                    .iter()
                    .map(|n| n.0)
                    .collect()
            })
            .collect();
        
        let cell_areas: Vec<_> = (0..n_cells)
            .map(|i| mesh.cell_area(CellIndex(i)))
            .collect();
        
        let bed_elevations: Vec<_> = (0..n_cells)
            .map(|i| mesh.cell_bed_elevation(CellIndex(i)))
            .collect();
        
        Self {
            n_nodes,
            n_cells,
            node_positions,
            cell_nodes,
            cell_areas,
            bed_elevations,
        }
    }
}

/// 状态快照（用于异步传输）
#[derive(Clone)]
pub struct StateSnapshot {
    pub h: Vec<f64>,
    pub hu: Vec<f64>,
    pub hv: Vec<f64>,
}

impl StateSnapshot {
    /// 从状态创建快照
    pub fn from_state<S: StateAccess>(state: &S) -> Self {
        let n = state.n_cells();
        Self {
            h: (0..n).map(|i| state.h(i)).collect(),
            hu: (0..n).map(|i| state.hu(i)).collect(),
            hv: (0..n).map(|i| state.hv(i)).collect(),
        }
    }
}

/// 异步IO管道
pub struct IoPipeline {
    sender: Sender<OutputRequest>,
    worker: Option<JoinHandle<()>>,
    pending_count: Arc<Mutex<usize>>,
}

impl IoPipeline {
    /// 创建新的IO管道
    pub fn new() -> Self {
        let (sender, receiver) = channel();
        let pending_count = Arc::new(Mutex::new(0));
        let pending_clone = pending_count.clone();
        
        let worker = thread::spawn(move || {
            Self::worker_loop(receiver, pending_clone);
        });
        
        Self {
            sender,
            worker: Some(worker),
            pending_count,
        }
    }
    
    /// 提交输出请求
    pub fn submit(&self, request: OutputRequest) -> MhResult<()> {
        {
            let mut count = self.pending_count.lock().unwrap();
            *count += 1;
        }
        
        self.sender
            .send(request)
            .map_err(|e| MhError::Io(format!("Failed to submit IO request: {}", e)))
    }
    
    /// 获取待处理请求数
    pub fn pending_count(&self) -> usize {
        *self.pending_count.lock().unwrap()
    }
    
    /// 等待所有请求完成
    pub fn flush(&self) {
        while self.pending_count() > 0 {
            thread::sleep(std::time::Duration::from_millis(10));
        }
    }
    
    /// 工作线程循环
    fn worker_loop(receiver: Receiver<OutputRequest>, pending_count: Arc<Mutex<usize>>) {
        while let Ok(request) = receiver.recv() {
            match request {
                OutputRequest::WriteVtuAscii { path, mesh_data, state_data, time } => {
                    if let Err(e) = Self::write_vtu_ascii(&path, &mesh_data, &state_data, time) {
                        eprintln!("VTU write error: {}", e);
                    }
                }
                OutputRequest::WriteVtuBinary { path, mesh_data, state_data, time } => {
                    if let Err(e) = Self::write_vtu_binary(&path, &mesh_data, &state_data, time) {
                        eprintln!("VTU binary write error: {}", e);
                    }
                }
                OutputRequest::WriteCheckpoint { path, state_data, time, step } => {
                    if let Err(e) = Self::write_checkpoint(&path, &state_data, time, step) {
                        eprintln!("Checkpoint write error: {}", e);
                    }
                }
                OutputRequest::Shutdown => {
                    break;
                }
            }
            
            {
                let mut count = pending_count.lock().unwrap();
                *count = count.saturating_sub(1);
            }
        }
    }
    
    /// 写入ASCII VTU
    fn write_vtu_ascii(
        path: &Path,
        mesh: &MeshSnapshot,
        state: &StateSnapshot,
        time: f64,
    ) -> MhResult<()> {
        use std::fs::File;
        
        let file = File::create(path)
            .map_err(|e| MhError::Io(format!("Cannot create {}: {}", path.display(), e)))?;
        let mut w = BufWriter::new(file);
        
        // Header
        writeln!(w, r#"<?xml version="1.0"?>"#)?;
        writeln!(w, r#"<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">"#)?;
        writeln!(w, r#"  <UnstructuredGrid>"#)?;
        writeln!(w, r#"    <FieldData>"#)?;
        writeln!(w, r#"      <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1">{}</DataArray>"#, time)?;
        writeln!(w, r#"    </FieldData>"#)?;
        
        // Piece
        writeln!(w, r#"    <Piece NumberOfPoints="{}" NumberOfCells="{}">"#, mesh.n_nodes, mesh.n_cells)?;
        
        // Points
        writeln!(w, r#"      <Points>"#)?;
        writeln!(w, r#"        <DataArray type="Float64" NumberOfComponents="3" format="ascii">"#)?;
        for (x, y) in &mesh.node_positions {
            writeln!(w, "          {:.6} {:.6} 0.0", x, y)?;
        }
        writeln!(w, r#"        </DataArray>"#)?;
        writeln!(w, r#"      </Points>"#)?;
        
        // Cells
        writeln!(w, r#"      <Cells>"#)?;
        writeln!(w, r#"        <DataArray type="Int32" Name="connectivity" format="ascii">"#)?;
        for nodes in &mesh.cell_nodes {
            let s: Vec<String> = nodes.iter().map(|n| n.to_string()).collect();
            writeln!(w, "          {}", s.join(" "))?;
        }
        writeln!(w, r#"        </DataArray>"#)?;
        
        writeln!(w, r#"        <DataArray type="Int32" Name="offsets" format="ascii">"#)?;
        let mut offset = 0;
        for nodes in &mesh.cell_nodes {
            offset += nodes.len();
            writeln!(w, "          {}", offset)?;
        }
        writeln!(w, r#"        </DataArray>"#)?;
        
        writeln!(w, r#"        <DataArray type="UInt8" Name="types" format="ascii">"#)?;
        for nodes in &mesh.cell_nodes {
            let t = match nodes.len() {
                3 => 5,  // VTK_TRIANGLE
                4 => 9,  // VTK_QUAD
                _ => 7,  // VTK_POLYGON
            };
            writeln!(w, "          {}", t)?;
        }
        writeln!(w, r#"        </DataArray>"#)?;
        writeln!(w, r#"      </Cells>"#)?;
        
        // Cell data
        writeln!(w, r#"      <CellData>"#)?;
        
        // h
        writeln!(w, r#"        <DataArray type="Float64" Name="h" format="ascii">"#)?;
        for &h in &state.h {
            writeln!(w, "          {:.6}", h)?;
        }
        writeln!(w, r#"        </DataArray>"#)?;
        
        // eta
        writeln!(w, r#"        <DataArray type="Float64" Name="eta" format="ascii">"#)?;
        for (h, z) in state.h.iter().zip(mesh.bed_elevations.iter()) {
            writeln!(w, "          {:.6}", h + z)?;
        }
        writeln!(w, r#"        </DataArray>"#)?;
        
        // u, v
        let h_dry = 1e-6;
        writeln!(w, r#"        <DataArray type="Float64" Name="u" format="ascii">"#)?;
        for (&h, &hu) in state.h.iter().zip(state.hu.iter()) {
            let u = if h > h_dry { hu / h } else { 0.0 };
            writeln!(w, "          {:.6}", u)?;
        }
        writeln!(w, r#"        </DataArray>"#)?;
        
        writeln!(w, r#"        <DataArray type="Float64" Name="v" format="ascii">"#)?;
        for (&h, &hv) in state.h.iter().zip(state.hv.iter()) {
            let v = if h > h_dry { hv / h } else { 0.0 };
            writeln!(w, "          {:.6}", v)?;
        }
        writeln!(w, r#"        </DataArray>"#)?;
        
        // velocity magnitude
        writeln!(w, r#"        <DataArray type="Float64" Name="velocity_mag" format="ascii">"#)?;
        for i in 0..mesh.n_cells {
            let h = state.h[i];
            let mag = if h > h_dry {
                let u = state.hu[i] / h;
                let v = state.hv[i] / h;
                (u * u + v * v).sqrt()
            } else {
                0.0
            };
            writeln!(w, "          {:.6}", mag)?;
        }
        writeln!(w, r#"        </DataArray>"#)?;
        
        writeln!(w, r#"      </CellData>"#)?;
        writeln!(w, r#"    </Piece>"#)?;
        writeln!(w, r#"  </UnstructuredGrid>"#)?;
        writeln!(w, r#"</VTKFile>"#)?;
        
        w.flush()?;
        Ok(())
    }
    
    /// 写入二进制VTU（AppendedData格式）
    fn write_vtu_binary(
        path: &Path,
        mesh: &MeshSnapshot,
        state: &StateSnapshot,
        time: f64,
    ) -> MhResult<()> {
        use std::fs::File;
        
        let file = File::create(path)
            .map_err(|e| MhError::Io(format!("Cannot create {}: {}", path.display(), e)))?;
        let mut w = BufWriter::new(file);
        
        // 收集所有二进制数据块
        let mut appended_data: Vec<u8> = Vec::new();
        let mut offsets: Vec<usize> = Vec::new();
        
        // Points data
        let points_offset = appended_data.len();
        offsets.push(points_offset);
        let points_bytes = mesh.n_nodes * 3 * 8; // 3 components, f64
        appended_data.extend_from_slice(&(points_bytes as u32).to_le_bytes());
        for &(x, y) in &mesh.node_positions {
            appended_data.extend_from_slice(&x.to_le_bytes());
            appended_data.extend_from_slice(&y.to_le_bytes());
            appended_data.extend_from_slice(&0.0f64.to_le_bytes());
        }
        
        // Connectivity
        let conn_offset = appended_data.len();
        offsets.push(conn_offset);
        let conn_count: usize = mesh.cell_nodes.iter().map(|c| c.len()).sum();
        appended_data.extend_from_slice(&((conn_count * 4) as u32).to_le_bytes());
        for nodes in &mesh.cell_nodes {
            for &n in nodes {
                appended_data.extend_from_slice(&(n as i32).to_le_bytes());
            }
        }
        
        // Offsets
        let off_offset = appended_data.len();
        offsets.push(off_offset);
        appended_data.extend_from_slice(&((mesh.n_cells * 4) as u32).to_le_bytes());
        let mut cumulative = 0i32;
        for nodes in &mesh.cell_nodes {
            cumulative += nodes.len() as i32;
            appended_data.extend_from_slice(&cumulative.to_le_bytes());
        }
        
        // Types
        let types_offset = appended_data.len();
        offsets.push(types_offset);
        appended_data.extend_from_slice(&(mesh.n_cells as u32).to_le_bytes());
        for nodes in &mesh.cell_nodes {
            let t: u8 = match nodes.len() { 3 => 5, 4 => 9, _ => 7 };
            appended_data.push(t);
        }
        
        // Helper to add f64 array
        let mut add_f64_array = |data: &[f64]| -> usize {
            let offset = appended_data.len();
            appended_data.extend_from_slice(&((data.len() * 8) as u32).to_le_bytes());
            for &v in data {
                appended_data.extend_from_slice(&v.to_le_bytes());
            }
            offset
        };
        
        // Cell data arrays
        let h_offset = add_f64_array(&state.h);
        
        let eta: Vec<f64> = state.h.iter().zip(mesh.bed_elevations.iter())
            .map(|(&h, &z)| h + z).collect();
        let eta_offset = add_f64_array(&eta);
        
        let h_dry = 1e-6;
        let u: Vec<f64> = state.h.iter().zip(state.hu.iter())
            .map(|(&h, &hu)| if h > h_dry { hu / h } else { 0.0 }).collect();
        let u_offset = add_f64_array(&u);
        
        let v: Vec<f64> = state.h.iter().zip(state.hv.iter())
            .map(|(&h, &hv)| if h > h_dry { hv / h } else { 0.0 }).collect();
        let v_offset = add_f64_array(&v);
        
        let mag: Vec<f64> = (0..mesh.n_cells).map(|i| {
            let h = state.h[i];
            if h > h_dry {
                let u = state.hu[i] / h;
                let v = state.hv[i] / h;
                (u * u + v * v).sqrt()
            } else { 0.0 }
        }).collect();
        let mag_offset = add_f64_array(&mag);
        
        // Write XML header
        writeln!(w, r#"<?xml version="1.0"?>"#)?;
        writeln!(w, r#"<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">"#)?;
        writeln!(w, r#"  <UnstructuredGrid>"#)?;
        writeln!(w, r#"    <FieldData>"#)?;
        writeln!(w, r#"      <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1" format="ascii">{}</DataArray>"#, time)?;
        writeln!(w, r#"    </FieldData>"#)?;
        writeln!(w, r#"    <Piece NumberOfPoints="{}" NumberOfCells="{}">"#, mesh.n_nodes, mesh.n_cells)?;
        writeln!(w, r#"      <Points>"#)?;
        writeln!(w, r#"        <DataArray type="Float64" NumberOfComponents="3" format="appended" offset="{}"/>"#, points_offset)?;
        writeln!(w, r#"      </Points>"#)?;
        writeln!(w, r#"      <Cells>"#)?;
        writeln!(w, r#"        <DataArray type="Int32" Name="connectivity" format="appended" offset="{}"/>"#, conn_offset)?;
        writeln!(w, r#"        <DataArray type="Int32" Name="offsets" format="appended" offset="{}"/>"#, off_offset)?;
        writeln!(w, r#"        <DataArray type="UInt8" Name="types" format="appended" offset="{}"/>"#, types_offset)?;
        writeln!(w, r#"      </Cells>"#)?;
        writeln!(w, r#"      <CellData>"#)?;
        writeln!(w, r#"        <DataArray type="Float64" Name="h" format="appended" offset="{}"/>"#, h_offset)?;
        writeln!(w, r#"        <DataArray type="Float64" Name="eta" format="appended" offset="{}"/>"#, eta_offset)?;
        writeln!(w, r#"        <DataArray type="Float64" Name="u" format="appended" offset="{}"/>"#, u_offset)?;
        writeln!(w, r#"        <DataArray type="Float64" Name="v" format="appended" offset="{}"/>"#, v_offset)?;
        writeln!(w, r#"        <DataArray type="Float64" Name="velocity_mag" format="appended" offset="{}"/>"#, mag_offset)?;
        writeln!(w, r#"      </CellData>"#)?;
        writeln!(w, r#"    </Piece>"#)?;
        writeln!(w, r#"  </UnstructuredGrid>"#)?;
        writeln!(w, r#"  <AppendedData encoding="raw">"#)?;
        write!(w, "_")?;
        w.write_all(&appended_data)?;
        writeln!(w)?;
        writeln!(w, r#"  </AppendedData>"#)?;
        writeln!(w, r#"</VTKFile>"#)?;
        
        w.flush()?;
        Ok(())
    }
    
    /// 写入检查点
    fn write_checkpoint(
        path: &Path,
        state: &StateSnapshot,
        time: f64,
        step: usize,
    ) -> MhResult<()> {
        use std::fs::File;
        
        let file = File::create(path)
            .map_err(|e| MhError::Io(format!("Cannot create checkpoint: {}", e)))?;
        let mut w = BufWriter::new(file);
        
        // 简单二进制格式：header + data
        // Header: magic(4) + version(4) + n_cells(8) + time(8) + step(8)
        w.write_all(b"MHCK")?;  // Magic
        w.write_all(&1u32.to_le_bytes())?;  // Version
        w.write_all(&(state.h.len() as u64).to_le_bytes())?;
        w.write_all(&time.to_le_bytes())?;
        w.write_all(&(step as u64).to_le_bytes())?;
        
        // Data
        for &h in &state.h {
            w.write_all(&h.to_le_bytes())?;
        }
        for &hu in &state.hu {
            w.write_all(&hu.to_le_bytes())?;
        }
        for &hv in &state.hv {
            w.write_all(&hv.to_le_bytes())?;
        }
        
        w.flush()?;
        Ok(())
    }
}

impl Drop for IoPipeline {
    fn drop(&mut self) {
        let _ = self.sender.send(OutputRequest::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Default for IoPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// 为 std::io::Error 实现转换
impl From<std::io::Error> for MhError {
    fn from(e: std::io::Error) -> Self {
        MhError::Io(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pipeline_creation() {
        let pipeline = IoPipeline::new();
        assert_eq!(pipeline.pending_count(), 0);
    }
    
    #[test]
    fn test_state_snapshot() {
        let snapshot = StateSnapshot {
            h: vec![1.0, 2.0, 3.0],
            hu: vec![0.1, 0.2, 0.3],
            hv: vec![0.0, 0.0, 0.0],
        };
        assert_eq!(snapshot.h.len(), 3);
    }
}
