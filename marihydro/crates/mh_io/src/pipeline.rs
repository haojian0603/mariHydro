// crates/mh_io/src/pipeline.rs

//! 异步 IO 管道
//!
//! 提供异步文件写入功能，避免阻塞计算线程。
//!
//! # 设计说明
//!
//! IO 管道使用独立的工作线程处理文件写入请求，主线程只需提交请求即可继续计算。
//! 支持背压控制，当待处理请求过多时会阻塞提交。
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use mh_io::pipeline::{IoPipeline, OutputRequest};
//!
//! let pipeline = IoPipeline::new();
//!
//! // 提交 VTU 写入请求
//! pipeline.write_vtu_ascii("output.vtu", mesh_snap, state_snap, 0.0)?;
//!
//! // 等待所有请求完成
//! pipeline.flush()?;
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::snapshot::{MeshSnapshot, StateSnapshot};

// ============================================================
// 错误类型（内部实现细节）
// ============================================================

/// IO 管道内部错误（不对外暴露）
#[derive(Debug)]
pub(crate) enum PipelineError {
    /// IO 错误
    Io(std::io::Error),
    /// 管道已关闭
    PipelineClosed,
    /// 序列化错误
    Serialization(String),
    /// 队列已满（背压）
    QueueFull,
    /// 超时
    Timeout,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Io(e) => write!(f, "IO 错误: {}", e),
            PipelineError::PipelineClosed => write!(f, "管道已关闭"),
            PipelineError::Serialization(msg) => write!(f, "序列化错误: {}", msg),
            PipelineError::QueueFull => write!(f, "队列已满"),
            PipelineError::Timeout => write!(f, "操作超时"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<std::io::Error> for PipelineError {
    fn from(e: std::io::Error) -> Self {
        PipelineError::Io(e)
    }
}

/// IO 管道内部操作结果
type PipelineResult<T> = Result<T, PipelineError>;

// ============================================================
// 输出请求
// ============================================================

/// 输出请求类型
pub enum OutputRequest {
    /// 写入 VTU 文件（ASCII 格式）
    WriteVtuAscii {
        path: PathBuf,
        mesh_data: MeshSnapshot,
        state_data: StateSnapshot,
        time: f64,
    },
    /// 写入 VTU 文件（二进制格式）
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
        mesh_snapshot: Option<MeshSnapshot>,
        time: f64,
        step: usize,
    },
    /// 写入 PVD 集合文件
    WritePvd {
        path: PathBuf,
        entries: Vec<PvdEntry>,
    },
    /// 写入原始数据
    WriteRaw {
        path: PathBuf,
        data: Vec<u8>,
    },
    /// 刷新所有待处理请求
    Flush,
    /// 关闭管道
    Shutdown,
}

/// PVD 条目
#[derive(Debug, Clone)]
pub struct PvdEntry {
    /// 时间值
    pub time: f64,
    /// VTU 文件相对路径
    pub file_path: String,
}

impl PvdEntry {
    /// 创建新条目
    pub fn new(time: f64, file_path: impl Into<String>) -> Self {
        Self {
            time,
            file_path: file_path.into(),
        }
    }
}

// ============================================================
// 管道统计
// ============================================================

/// 管道统计信息
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// 总请求数
    pub total_requests: u64,
    /// 已完成请求数
    pub completed_requests: u64,
    /// 失败请求数
    pub failed_requests: u64,
    /// 总写入字节数
    pub total_bytes_written: u64,
    /// 平均写入时间（毫秒）
    pub average_write_time_ms: f64,
    /// 最大队列长度
    pub max_queue_length: usize,
    /// 当前队列长度
    pub current_queue_length: usize,
}

// ============================================================
// IO 管道
// ============================================================

/// 异步 IO 管道配置
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// 最大待处理请求数（0 表示无限制）
    pub max_pending: usize,
    /// 工作线程名称
    pub thread_name: String,
    /// 写入超时（毫秒）
    pub write_timeout_ms: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_pending: 100,
            thread_name: "mh-io-worker".to_string(),
            write_timeout_ms: 30000,
        }
    }
}

/// 异步 IO 管道
///
/// 管理后台文件写入，避免阻塞主计算线程。
pub struct IoPipeline {
    /// 请求发送端
    sender: Sender<OutputRequest>,
    /// 工作线程句柄
    worker: Option<JoinHandle<()>>,
    /// 待处理请求计数
    pending_count: Arc<AtomicUsize>,
    /// 统计信息
    stats: Arc<Mutex<PipelineStats>>,
    /// 关闭标志
    shutdown_flag: Arc<AtomicBool>,
    /// 配置
    config: PipelineConfig,
}

impl IoPipeline {
    /// 创建新的 IO 管道
    pub fn new() -> Self {
        Self::with_config(PipelineConfig::default())
    }

    /// 使用指定配置创建管道
    pub fn with_config(config: PipelineConfig) -> Self {
        let (sender, receiver) = channel();
        let pending_count = Arc::new(AtomicUsize::new(0));
        let pending_clone = pending_count.clone();
        let stats = Arc::new(Mutex::new(PipelineStats::default()));
        let stats_clone = stats.clone();
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown_flag.clone();
        let thread_name = config.thread_name.clone();

        let worker = thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                Self::worker_loop(receiver, pending_clone, stats_clone, shutdown_clone);
            })
            .expect("无法创建 IO 工作线程");

        Self {
            sender,
            worker: Some(worker),
            pending_count,
            stats,
            shutdown_flag,
            config,
        }
    }

    /// 创建带容量限制的管道
    pub fn with_capacity(max_pending: usize) -> Self {
        Self::with_config(PipelineConfig {
            max_pending,
            ..Default::default()
        })
    }

    /// 提交输出请求（公共 API）
    pub fn submit(&self, request: OutputRequest) -> crate::error::IoResult<()> {
        // 检查是否已关闭
        if self.shutdown_flag.load(Ordering::SeqCst) {
            return Err(crate::error::IoError::PipelineFailed {
                stage: "submit".to_string(),
                message: "管道已关闭".to_string(),
            });
        }

        // 检查队列容量（背压控制）
        if self.config.max_pending > 0 {
            let current = self.pending_count.load(Ordering::SeqCst);
            if current >= self.config.max_pending {
                return Err(crate::error::IoError::PipelineFailed {
                    stage: "submit".to_string(),
                    message: "队列已满".to_string(),
                });
            }
        }

        // 更新计数
        self.pending_count.fetch_add(1, Ordering::SeqCst);
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_requests += 1;
            let current = self.pending_count.load(Ordering::SeqCst);
            stats.current_queue_length = current;
            if current > stats.max_queue_length {
                stats.max_queue_length = current;
            }
        }

        // 发送请求（内部使用 PipelineError）
        self.sender
            .send(request)
            .map_err(|_| crate::error::IoError::PipelineFailed {
                stage: "submit".to_string(),
                message: "管道已关闭".to_string(),
            })
    }

    /// 提交 VTU ASCII 写入（公共 API）
    pub fn write_vtu_ascii(
        &self,
        path: impl Into<PathBuf>,
        mesh: MeshSnapshot,
        state: StateSnapshot,
        time: f64,
    ) -> crate::error::IoResult<()> {
        self.submit(OutputRequest::WriteVtuAscii {
            path: path.into(),
            mesh_data: mesh,
            state_data: state,
            time,
        })
    }

    /// 提交 VTU 二进制写入（公共 API）
    pub fn write_vtu_binary(
        &self,
        path: impl Into<PathBuf>,
        mesh: MeshSnapshot,
        state: StateSnapshot,
        time: f64,
    ) -> crate::error::IoResult<()> {
        self.submit(OutputRequest::WriteVtuBinary {
            path: path.into(),
            mesh_data: mesh,
            state_data: state,
            time,
        })
    }

    /// 提交检查点写入（公共 API）
    pub fn write_checkpoint(
        &self,
        path: impl Into<PathBuf>,
        state: StateSnapshot,
        mesh: Option<MeshSnapshot>,
        time: f64,
        step: usize,
    ) -> crate::error::IoResult<()> {
        self.submit(OutputRequest::WriteCheckpoint {
            path: path.into(),
            state_data: state,
            mesh_snapshot: mesh,
            time,
            step,
        })
    }

    /// 提交 PVD 写入（公共 API）
    pub fn write_pvd(&self, path: impl Into<PathBuf>, entries: Vec<PvdEntry>) -> crate::error::IoResult<()> {
        self.submit(OutputRequest::WritePvd {
            path: path.into(),
            entries,
        })
    }

    /// 刷新待处理请求（公共 API）
    pub fn flush(&self) -> crate::error::IoResult<()> {
        self.submit(OutputRequest::Flush)
    }

    /// 获取待处理请求数
    pub fn pending_count(&self) -> usize {
        self.pending_count.load(Ordering::SeqCst)
    }

    /// 等待所有请求完成
    pub fn wait_for_completion(&self, timeout: Duration) -> bool {
        let start = Instant::now();
        while self.pending_count() > 0 {
            if start.elapsed() > timeout {
                return false;
            }
            thread::sleep(Duration::from_millis(10));
        }
        true
    }

    /// 获取统计信息
    pub fn stats(&self) -> PipelineStats {
        self.stats.lock().unwrap().clone()
    }

    /// 显式关闭管道
    pub fn shutdown(&mut self) {
        self.shutdown_flag.store(true, Ordering::SeqCst);
        let _ = self.sender.send(OutputRequest::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }

    /// 工作线程主循环（内部使用 PipelineResult）
    fn worker_loop(
        receiver: Receiver<OutputRequest>,
        pending_count: Arc<AtomicUsize>,
        stats: Arc<Mutex<PipelineStats>>,
        _shutdown_flag: Arc<AtomicBool>,
    ) {
        while let Ok(request) = receiver.recv() {
            // 检查关闭请求
            if matches!(request, OutputRequest::Shutdown) {
                break;
            }

            // 处理 Flush 请求
            if matches!(request, OutputRequest::Flush) {
                pending_count.fetch_sub(1, Ordering::SeqCst);
                continue;
            }

            // 处理请求
            let start = Instant::now();
            let result = Self::process_request(&request);
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            // 更新统计
            {
                let mut s = stats.lock().unwrap();
                if result.is_ok() {
                    s.completed_requests += 1;
                } else {
                    s.failed_requests += 1;
                    // 记录错误日志
                    if let Err(e) = &result {
                        eprintln!("[mh_io::pipeline] 写入失败: {}", e);
                    }
                }
                // 更新平均写入时间
                let total = s.completed_requests + s.failed_requests;
                s.average_write_time_ms =
                    (s.average_write_time_ms * (total - 1) as f64 + elapsed_ms) / total as f64;
            }

            // 减少待处理计数
            pending_count.fetch_sub(1, Ordering::SeqCst);
            {
                let mut s = stats.lock().unwrap();
                s.current_queue_length = pending_count.load(Ordering::SeqCst);
            }
        }
    }

    /// 处理单个请求（内部使用 PipelineResult）
    fn process_request(request: &OutputRequest) -> PipelineResult<()> {
        match request {
            OutputRequest::WriteVtuAscii {
                path,
                mesh_data,
                state_data,
                time,
            } => Self::write_vtu_ascii_impl(path, mesh_data, state_data, *time),
            OutputRequest::WriteVtuBinary {
                path,
                mesh_data,
                state_data,
                time,
            } => Self::write_vtu_binary_impl(path, mesh_data, state_data, *time),
            OutputRequest::WriteCheckpoint {
                path,
                state_data,
                mesh_snapshot,
                time,
                step,
            } => Self::write_checkpoint_impl(path, state_data, mesh_snapshot.as_ref(), *time, *step),
            OutputRequest::WritePvd { path, entries } => Self::write_pvd_impl(path, entries),
            OutputRequest::WriteRaw { path, data } => {
                let mut file = File::create(path)?;
                file.write_all(data)?;
                Ok(())
            }
            OutputRequest::Flush | OutputRequest::Shutdown => Ok(()),
        }
    }

    /// VTU ASCII 写入实现（内部使用 PipelineResult）
    fn write_vtu_ascii_impl(
        path: &Path,
        mesh: &MeshSnapshot,
        state: &StateSnapshot,
        time: f64,
    ) -> PipelineResult<()> {
        // 创建目录（如果不存在）
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // VTU XML 头
        writeln!(writer, r#"<?xml version="1.0"?>"#)?;
        writeln!(
            writer,
            r#"<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">"#
        )?;
        writeln!(writer, r#"  <UnstructuredGrid>"#)?;
        writeln!(
            writer,
            r#"    <Piece NumberOfPoints="{}" NumberOfCells="{}">"#,
            mesh.n_nodes, mesh.n_cells
        )?;

        // 时间字段数据
        writeln!(writer, r#"      <FieldData>"#)?;
        writeln!(
            writer,
            r#"        <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1" format="ascii">"#
        )?;
        writeln!(writer, "          {}", time)?;
        writeln!(writer, r#"        </DataArray>"#)?;
        writeln!(writer, r#"      </FieldData>"#)?;

        // 点坐标
        writeln!(writer, r#"      <Points>"#)?;
        writeln!(
            writer,
            r#"        <DataArray type="Float64" NumberOfComponents="3" format="ascii">"#
        )?;
        for (x, y) in &mesh.node_positions {
            writeln!(writer, "          {} {} 0.0", x, y)?;
        }
        writeln!(writer, r#"        </DataArray>"#)?;
        writeln!(writer, r#"      </Points>"#)?;

        // 单元连接
        writeln!(writer, r#"      <Cells>"#)?;
        writeln!(
            writer,
            r#"        <DataArray type="Int32" Name="connectivity" format="ascii">"#
        )?;
        for nodes in &mesh.cell_nodes {
            let nodes_str: String = nodes
                .iter()
                .map(|&n| n.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            writeln!(writer, "          {}", nodes_str)?;
        }
        writeln!(writer, r#"        </DataArray>"#)?;

        // offsets
        writeln!(
            writer,
            r#"        <DataArray type="Int32" Name="offsets" format="ascii">"#
        )?;
        let mut offset = 0;
        for nodes in &mesh.cell_nodes {
            offset += nodes.len();
            write!(writer, "{} ", offset)?;
        }
        writeln!(writer)?;
        writeln!(writer, r#"        </DataArray>"#)?;

        // types
        writeln!(
            writer,
            r#"        <DataArray type="UInt8" Name="types" format="ascii">"#
        )?;
        for nodes in &mesh.cell_nodes {
            let vtk_type = match nodes.len() {
                3 => 5,  // VTK_TRIANGLE
                4 => 9,  // VTK_QUAD
                _ => 7,  // VTK_POLYGON
            };
            write!(writer, "{} ", vtk_type)?;
        }
        writeln!(writer)?;
        writeln!(writer, r#"        </DataArray>"#)?;
        writeln!(writer, r#"      </Cells>"#)?;

        // 单元数据
        writeln!(writer, r#"      <CellData Scalars="h">"#)?;

        // 水深
        writeln!(
            writer,
            r#"        <DataArray type="Float64" Name="h" format="ascii">"#
        )?;
        for h in &state.h {
            write!(writer, "{} ", h)?;
        }
        writeln!(writer)?;
        writeln!(writer, r#"        </DataArray>"#)?;

        // 水位
        writeln!(
            writer,
            r#"        <DataArray type="Float64" Name="eta" format="ascii">"#
        )?;
        for (h, z) in state.h.iter().zip(mesh.bed_elevations.iter()) {
            write!(writer, "{} ", h + z)?;
        }
        writeln!(writer)?;
        writeln!(writer, r#"        </DataArray>"#)?;

        // 速度
        writeln!(
            writer,
            r#"        <DataArray type="Float64" Name="velocity" NumberOfComponents="2" format="ascii">"#
        )?;
        for i in 0..state.h.len() {
            let (u, v) = if state.h[i] > 1e-6 {
                (state.hu[i] / state.h[i], state.hv[i] / state.h[i])
            } else {
                (0.0, 0.0)
            };
            write!(writer, "{} {} ", u, v)?;
        }
        writeln!(writer)?;
        writeln!(writer, r#"        </DataArray>"#)?;

        // 床面高程
        writeln!(
            writer,
            r#"        <DataArray type="Float64" Name="bed_elevation" format="ascii">"#
        )?;
        for z in &mesh.bed_elevations {
            write!(writer, "{} ", z)?;
        }
        writeln!(writer)?;
        writeln!(writer, r#"        </DataArray>"#)?;

        // 标量场
        if let (Some(scalars), Some(names)) = (&state.scalars, &state.scalar_names) {
            for (scalar, name) in scalars.iter().zip(names.iter()) {
                writeln!(
                    writer,
                    r#"        <DataArray type="Float64" Name="{}" format="ascii">"#,
                    name
                )?;
                for v in scalar {
                    write!(writer, "{} ", v)?;
                }
                writeln!(writer)?;
                writeln!(writer, r#"        </DataArray>"#)?;
            }
        }

        writeln!(writer, r#"      </CellData>"#)?;
        writeln!(writer, r#"    </Piece>"#)?;
        writeln!(writer, r#"  </UnstructuredGrid>"#)?;
        writeln!(writer, r#"</VTKFile>"#)?;

        writer.flush()?;
        Ok(())
    }

    /// VTU 二进制写入实现（内部使用 PipelineResult）
    fn write_vtu_binary_impl(
        path: &Path,
        mesh: &MeshSnapshot,
        state: &StateSnapshot,
        time: f64,
    ) -> PipelineResult<()> {
        // TODO: 实现完整的二进制 VTU 输出（带 base64 编码）
        // 当前回退到 ASCII 格式
        Self::write_vtu_ascii_impl(path, mesh, state, time)
    }

    /// 检查点写入实现（内部使用 PipelineResult）
    fn write_checkpoint_impl(
        path: &Path,
        state: &StateSnapshot,
        mesh: Option<&MeshSnapshot>,
        time: f64,
        step: usize,
    ) -> PipelineResult<()> {
        // 创建目录
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // 魔数
        writer.write_all(b"MHCK")?;

        // 版本
        writer.write_all(&2u32.to_le_bytes())?;

        // 时间和步数
        writer.write_all(&time.to_le_bytes())?;
        writer.write_all(&(step as u64).to_le_bytes())?;

        // 单元数
        let n_cells = state.h.len();
        writer.write_all(&(n_cells as u64).to_le_bytes())?;

        // 状态数据
        for &h in &state.h {
            writer.write_all(&h.to_le_bytes())?;
        }
        for &hu in &state.hu {
            writer.write_all(&hu.to_le_bytes())?;
        }
        for &hv in &state.hv {
            writer.write_all(&hv.to_le_bytes())?;
        }

        // 底床高程（如果有）
        let has_z = state.z.is_some() as u8;
        writer.write_all(&[has_z])?;
        if let Some(z) = &state.z {
            for &val in z {
                writer.write_all(&val.to_le_bytes())?;
            }
        }

        // 网格哈希（用于验证）
        let mesh_hash: u64 = mesh.map_or(0, |m| m.n_cells as u64 ^ m.n_nodes as u64);
        writer.write_all(&mesh_hash.to_le_bytes())?;

        writer.flush()?;
        Ok(())
    }

    /// PVD 写入实现（内部使用 PipelineResult）
    fn write_pvd_impl(path: &Path, entries: &[PvdEntry]) -> PipelineResult<()> {
        // 创建目录
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, r#"<?xml version="1.0"?>"#)?;
        writeln!(writer, r#"<VTKFile type="Collection" version="0.1">"#)?;
        writeln!(writer, r#"  <Collection>"#)?;

        for entry in entries {
            writeln!(
                writer,
                r#"    <DataSet timestep="{}" file="{}"/>"#,
                entry.time, entry.file_path
            )?;
        }

        writeln!(writer, r#"  </Collection>"#)?;
        writeln!(writer, r#"</VTKFile>"#)?;

        writer.flush()?;
        Ok(())
    }
}

impl Default for IoPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for IoPipeline {
    fn drop(&mut self) {
        // 发送关闭请求
        self.shutdown_flag.store(true, Ordering::SeqCst);
        let _ = self.sender.send(OutputRequest::Shutdown);

        // 等待工作线程结束
        if let Some(worker) = self.worker.take() {
            // 给工作线程一些时间完成
            let _ = worker.join();
        }
    }
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = IoPipeline::new();
        assert_eq!(pipeline.pending_count(), 0);
    }

    #[test]
    fn test_pipeline_shutdown() {
        let mut pipeline = IoPipeline::new();
        pipeline.shutdown();
        // 应该正常关闭
    }

    #[test]
    fn test_pipeline_stats() {
        let pipeline = IoPipeline::new();
        let stats = pipeline.stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.completed_requests, 0);
    }

    #[test]
    fn test_pvd_entry() {
        let entry = PvdEntry::new(10.5, "output_0010.vtu");
        assert!((entry.time - 10.5).abs() < 1e-10);
        assert_eq!(entry.file_path, "output_0010.vtu");
    }

    #[test]
    fn test_pipeline_capacity() {
        let pipeline = IoPipeline::with_capacity(5);
        assert_eq!(pipeline.config.max_pending, 5);
    }
}