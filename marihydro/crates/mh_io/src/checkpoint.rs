// crates/mh_io/src/checkpoint.rs

//! 检查点保存/恢复系统
//!
//! 支持模拟中断后续算，提供二进制格式的状态保存与恢复。
//!
//! # 文件格式 (v2)
//!
//! ```text
//! [魔数: 4 bytes] "MHCK"
//! [版本: u32]
//! [时间: f64]
//! [步数: u64]
//! [配置哈希: u64]
//! [创建时间: u64]
//! [单元数: u64]
//! [h 数据: n_cells * f64]
//! [hu 数据: n_cells * f64]
//! [hv 数据: n_cells * f64]
//! [有底床标志: u8]
//! [z 数据: n_cells * f64] (可选)
//! [网格哈希: u64]
//! [CRC32: u32] (v2 新增)
//! ```
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use mh_io::checkpoint::Checkpoint;
//! use mh_io::snapshot::StateSnapshot;
//! use mh_runtime::CpuBackend;
//!
//! // 保存检查点
//! let state = StateSnapshot::<CpuBackend<f64>>::from_state_data(h, hu, hv);
//! let checkpoint = Checkpoint::new(100.0, 1000, state);
//! checkpoint.save(Path::new("checkpoint.mhck"))?;
//!
//! // 加载检查点
//! let loaded = Checkpoint::load(Path::new("checkpoint.mhck"))?;
//! println!("恢复到时间: {}", loaded.time);
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::snapshot::{MeshSnapshot, StateSnapshot};

// ============================================================
// 错误类型
// ============================================================

/// 检查点错误
#[derive(Debug)]
pub enum CheckpointError {
    /// IO 错误
    Io(std::io::Error),
    /// IO 错误（带文件路径）
    IoWithPath {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    /// 格式错误
    Format(String),
    /// 版本不兼容
    Version { file: u32, current: u32 },
    /// 网格不匹配
    MeshMismatch { expected: usize, found: usize },
    /// 校验和错误
    Checksum { expected: u32, found: u32 },
    /// 配置哈希不匹配
    ConfigMismatch { expected: u64, found: u64 },
    /// 网格哈希不匹配
    MeshHashMismatch { expected: u64, found: u64 },
    /// 数据损坏
    Corrupted(String),
}

impl CheckpointError {
    /// 创建带路径信息的IO错误
    pub fn io_with_path<P: AsRef<Path>>(path: P, source: std::io::Error) -> Self {
        CheckpointError::IoWithPath {
            path: path.as_ref().to_path_buf(),
            source,
        }
    }
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Io(e) => write!(f, "IO 错误: {}", e),
            CheckpointError::IoWithPath { path, source } => {
                write!(f, "IO 错误 [{path}]: {source}", path = path.display())
            }
            CheckpointError::Format(msg) => write!(f, "格式错误: {}", msg),
            CheckpointError::Version { file, current } => {
                write!(f, "版本不兼容: 文件版本 {}, 当前版本 {}", file, current)
            }
            CheckpointError::MeshMismatch { expected, found } => {
                write!(f, "网格不匹配: 期望 {} 单元, 文件 {} 单元", expected, found)
            }
            CheckpointError::Checksum { expected, found } => {
                write!(f, "校验和错误: 期望 {:08x}, 实际 {:08x}", expected, found)
            }
            CheckpointError::ConfigMismatch { expected, found } => {
                write!(f, "配置哈希不匹配: 期望 {}, 实际 {}", expected, found)
            }
            CheckpointError::MeshHashMismatch { expected, found } => {
                write!(f, "网格哈希不匹配: 期望 {}, 实际 {}", expected, found)
            }
            CheckpointError::Corrupted(msg) => write!(f, "数据损坏: {}", msg),
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<std::io::Error> for CheckpointError {
    fn from(e: std::io::Error) -> Self {
        CheckpointError::Io(e)
    }
}

/// 检查点操作结果
pub type CheckpointResult<T> = Result<T, CheckpointError>;

// ============================================================
// 常量
// ============================================================

/// 检查点文件格式版本
const CHECKPOINT_VERSION: u32 = 2;

/// 检查点魔数
const CHECKPOINT_MAGIC: &[u8; 4] = b"MHCK";

/// 最大支持的文件版本
const MAX_SUPPORTED_VERSION: u32 = 2;

// ============================================================
// 检查点数据
// ============================================================

/// 检查点头部信息
#[derive(Debug, Clone)]
pub struct CheckpointHeader {
    /// 版本号
    pub version: u32,
    /// 模拟时间 [s]
    pub time: f64,
    /// 时间步数
    pub step: usize,
    /// 配置摘要哈希（用于验证）
    pub config_hash: Option<u64>,
    /// 创建时间戳
    pub created_at: u64,
    /// 单元数
    pub n_cells: usize,
    /// 网格哈希
    pub mesh_hash: u64,
}

/// 检查点数据
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// 版本号
    pub version: u32,
    /// 模拟时间 [s]
    pub time: f64,
    /// 时间步数
    pub step: usize,
    /// 状态数据
    pub state: StateSnapshot<f64>,
    /// 配置摘要哈希（用于验证）
    pub config_hash: Option<u64>,
    /// 创建时间戳
    pub created_at: u64,
    /// 网格哈希（用于兼容性检查）
    pub mesh_hash: u64,
}

/// 检查点加载校验选项
#[derive(Debug, Clone, Copy)]
pub struct CheckpointLoadOptions {
    /// 期望的配置哈希
    pub expected_config_hash: Option<u64>,
    /// 期望的网格哈希
    pub expected_mesh_hash: Option<u64>,
    /// 严格模式（缺失哈希也视为不匹配）
    pub strict: bool,
}

impl Default for CheckpointLoadOptions {
    fn default() -> Self {
        Self {
            expected_config_hash: None,
            expected_mesh_hash: None,
            strict: false,
        }
    }
}

impl Checkpoint {
    /// 创建新检查点
    pub fn new(time: f64, step: usize, state: StateSnapshot<f64>) -> Self {
        Self {
            version: CHECKPOINT_VERSION,
            time,
            step,
            state,
            config_hash: None,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            mesh_hash: 0,
        }
    }

    /// 设置配置哈希
    pub fn with_config_hash(mut self, hash: u64) -> Self {
        self.config_hash = Some(hash);
        self
    }

    /// 设置网格哈希
    pub fn with_mesh_hash(mut self, hash: u64) -> Self {
        self.mesh_hash = hash;
        self
    }

    /// 从网格快照计算哈希
    pub fn with_mesh_snapshot(mut self, mesh: &MeshSnapshot<f64>) -> Self {
        self.mesh_hash = mesh.compute_hash();
        self
    }

    /// 保存到文件（二进制格式）
    pub fn save(&self, path: &Path) -> CheckpointResult<()> {
        if let Err(err) = self.state.validate() {
            return Err(CheckpointError::Corrupted(err));
        }
        if !self.time.is_finite() {
            return Err(CheckpointError::Format("时间无效".into()));
        }
        // 创建目录
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CheckpointError::io_with_path(parent, e))?;
        }

        // 使用临时文件写入，成功后重命名（原子操作）
        let temp_path = path.with_extension("mhck.tmp");

        {
            let file = File::create(&temp_path)
                .map_err(|e| CheckpointError::io_with_path(&temp_path, e))?;
            let mut writer = BufWriter::new(file);
            let mut hasher = crc32fast::Hasher::new();

            macro_rules! write_crc {
                ($bytes:expr) => {{
                    writer.write_all($bytes)?;
                    hasher.update($bytes);
                }}
            }

            write_crc!(CHECKPOINT_MAGIC);
            write_crc!(&self.version.to_le_bytes());
            write_crc!(&self.time.to_le_bytes());
            write_crc!(&(self.step as u64).to_le_bytes());

            let hash = self.config_hash.unwrap_or(0);
            write_crc!(&hash.to_le_bytes());
            write_crc!(&self.created_at.to_le_bytes());

            let n_cells = self.state.n_cells();
            write_crc!(&(n_cells as u64).to_le_bytes());

            for &h in &self.state.h {
                write_crc!(&h.to_le_bytes());
            }
            for &hu in &self.state.hu {
                write_crc!(&hu.to_le_bytes());
            }
            for &hv in &self.state.hv {
                write_crc!(&hv.to_le_bytes());
            }

            let has_z = self.state.z.is_some() as u8;
            write_crc!(&[has_z]);
            if let Some(z) = &self.state.z {
                for &val in z {
                    write_crc!(&val.to_le_bytes());
                }
            }

            write_crc!(&self.mesh_hash.to_le_bytes());

            let crc = hasher.finalize();
            writer.write_all(&crc.to_le_bytes())?;

            writer.flush().map_err(|e| CheckpointError::io_with_path(&temp_path, e))?;
            writer
                .get_ref()
                .sync_all()
                .map_err(|e| CheckpointError::io_with_path(&temp_path, e))?;
        }

        // 原子重命名
        if path.exists() {
            std::fs::remove_file(path)
                .map_err(|e| CheckpointError::io_with_path(path, e))?;
        }
        std::fs::rename(&temp_path, path)
            .map_err(|e| CheckpointError::io_with_path(path, e))?;

        Ok(())
    }

    /// 从文件加载
    pub fn load(path: &Path) -> CheckpointResult<Self> {
        Self::load_with_options(path, CheckpointLoadOptions::default())
    }

    /// 从文件加载并校验网格一致性
    pub fn load_with_mesh(path: &Path, mesh: &MeshSnapshot<f64>, strict: bool) -> CheckpointResult<Self> {
        let options = CheckpointLoadOptions {
            expected_mesh_hash: Some(mesh.compute_hash()),
            strict,
            ..Default::default()
        };
        Self::load_with_options(path, options)
    }

    /// 从文件加载（带兼容性校验）
    pub fn load_with_options(path: &Path, options: CheckpointLoadOptions) -> CheckpointResult<Self> {
        let file = File::open(path).map_err(|e| CheckpointError::io_with_path(path, e))?;
        let file_len = file
            .metadata()
            .map_err(|e| CheckpointError::io_with_path(path, e))?
            .len() as usize;
        if file_len < 12 {
            return Err(CheckpointError::Format("文件太小".into()));
        }

        let mut reader = BufReader::new(file);
        let data_len = file_len - 4;
        let mut hasher = crc32fast::Hasher::new();
        // 使用Cell避免借用冲突
        let bytes_read = std::cell::Cell::new(0usize);

        let mut read_exact_crc = |buf: &mut [u8]| -> CheckpointResult<()> {
            reader
                .read_exact(buf)
                .map_err(|e| CheckpointError::io_with_path(path, e))?;
            hasher.update(buf);
            bytes_read.set(bytes_read.get().saturating_add(buf.len()));
            Ok(())
        };

        let mut magic = [0u8; 4];
        read_exact_crc(&mut magic)?;
        if &magic != CHECKPOINT_MAGIC {
            return Err(CheckpointError::Format("无效的检查点文件格式".into()));
        }

        let mut buf4 = [0u8; 4];
        read_exact_crc(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version > MAX_SUPPORTED_VERSION {
            return Err(CheckpointError::Version {
                file: version,
                current: CHECKPOINT_VERSION,
            });
        }

        let mut buf8 = [0u8; 8];
        read_exact_crc(&mut buf8)?;
        let time = f64::from_le_bytes(buf8);
        if !time.is_finite() {
            return Err(CheckpointError::Format("时间无效".into()));
        }

        read_exact_crc(&mut buf8)?;
        let step = u64::from_le_bytes(buf8) as usize;

        read_exact_crc(&mut buf8)?;
        let config_hash = u64::from_le_bytes(buf8);

        read_exact_crc(&mut buf8)?;
        let created_at = u64::from_le_bytes(buf8);

        read_exact_crc(&mut buf8)?;
        let n_cells = u64::from_le_bytes(buf8) as usize;
        let min_payload = n_cells
            .checked_mul(8 * 3)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| CheckpointError::Format("n_cells 溢出".into()))?;
        if bytes_read.get() + min_payload > data_len {
            return Err(CheckpointError::Format("文件太小".into()));
        }

        let mut h = Vec::with_capacity(n_cells);
        let mut hu = Vec::with_capacity(n_cells);
        let mut hv = Vec::with_capacity(n_cells);

        for _ in 0..n_cells {
            read_exact_crc(&mut buf8)?;
            h.push(f64::from_le_bytes(buf8));
        }
        for _ in 0..n_cells {
            read_exact_crc(&mut buf8)?;
            hu.push(f64::from_le_bytes(buf8));
        }
        for _ in 0..n_cells {
            read_exact_crc(&mut buf8)?;
            hv.push(f64::from_le_bytes(buf8));
        }

        let mut has_z_buf = [0u8; 1];
        read_exact_crc(&mut has_z_buf)?;
        let has_z = has_z_buf[0] != 0;

        let z = if has_z {
            let mut z_vec = Vec::with_capacity(n_cells);
            for _ in 0..n_cells {
                read_exact_crc(&mut buf8)?;
                z_vec.push(f64::from_le_bytes(buf8));
            }
            Some(z_vec)
        } else {
            None
        };

        let mut mesh_hash = 0u64;
        if bytes_read.get() + 8 <= data_len {
            read_exact_crc(&mut buf8)?;
            mesh_hash = u64::from_le_bytes(buf8);
        } else if version >= 2 {
            return Err(CheckpointError::Format("缺少网格哈希".into()));
        }

        if bytes_read.get() < data_len {
            let mut remaining = data_len - bytes_read.get();
            let mut buffer = [0u8; 4096];
            while remaining > 0 {
                let to_read = remaining.min(buffer.len());
                reader
                    .read_exact(&mut buffer[..to_read])
                    .map_err(|e| CheckpointError::io_with_path(path, e))?;
                hasher.update(&buffer[..to_read]);
                bytes_read.set(bytes_read.get().saturating_add(to_read));
                remaining -= to_read;
            }
        }

        let mut crc_buf = [0u8; 4];
        reader
            .read_exact(&mut crc_buf)
            .map_err(|e| CheckpointError::io_with_path(path, e))?;
        let stored_crc = u32::from_le_bytes(crc_buf);
        let computed_crc = hasher.finalize();
        if version >= 2 {
            if stored_crc != computed_crc {
                return Err(CheckpointError::Checksum {
                    expected: stored_crc,
                    found: computed_crc,
                });
            }
        } else if stored_crc != computed_crc && stored_crc != 0 {
            return Err(CheckpointError::Checksum {
                expected: stored_crc,
                found: computed_crc,
            });
        }

        let mut state = StateSnapshot::<f64>::from_state_data(h, hu, hv);
        if let Some(z_data) = z {
            state = state.with_bed(z_data);
        }

        if let Err(err) = state.validate() {
            return Err(CheckpointError::Corrupted(err));
        }

        let checkpoint = Self {
            version,
            time,
            step,
            state,
            config_hash: if config_hash != 0 {
                Some(config_hash)
            } else {
                None
            },
            created_at,
            mesh_hash,
        };

        checkpoint.verify_compatibility(options)?;

        Ok(checkpoint)
    }

    /// 校验检查点与当前配置/网格的一致性
    pub fn verify_compatibility(&self, options: CheckpointLoadOptions) -> CheckpointResult<()> {
        if let Some(expected) = options.expected_config_hash {
            match self.config_hash {
                Some(found) if found != expected => {
                    return Err(CheckpointError::ConfigMismatch { expected, found });
                }
                None if options.strict => {
                    return Err(CheckpointError::ConfigMismatch { expected, found: 0 });
                }
                _ => {}
            }
        }

        if let Some(expected) = options.expected_mesh_hash {
            if self.mesh_hash != expected && (options.strict || self.mesh_hash != 0) {
                return Err(CheckpointError::MeshHashMismatch {
                    expected,
                    found: self.mesh_hash,
                });
            }
        }

        Ok(())
    }

    /// 仅读取头部信息（不加载状态数据）
    pub fn read_header(path: &Path) -> CheckpointResult<CheckpointHeader> {
        let file = File::open(path).map_err(|e| CheckpointError::io_with_path(path, e))?;
        let mut reader = BufReader::new(file);

        // 魔数
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| CheckpointError::io_with_path(path, e))?;
        if &magic != CHECKPOINT_MAGIC {
            return Err(CheckpointError::Format("无效的检查点文件格式".into()));
        }

        // 版本
        let mut buf = [0u8; 8];
        reader
            .read_exact(&mut buf[..4])
            .map_err(|e| CheckpointError::io_with_path(path, e))?;
        let version = u32::from_le_bytes(buf[..4].try_into().unwrap());
        if version > MAX_SUPPORTED_VERSION {
            return Err(CheckpointError::Version {
                file: version,
                current: CHECKPOINT_VERSION,
            });
        }

        // 时间
        reader
            .read_exact(&mut buf)
            .map_err(|e| CheckpointError::io_with_path(path, e))?;
        let time = f64::from_le_bytes(buf);
        if !time.is_finite() {
            return Err(CheckpointError::Format("时间无效".into()));
        }

        // 步数
        reader
            .read_exact(&mut buf)
            .map_err(|e| CheckpointError::io_with_path(path, e))?;
        let step = u64::from_le_bytes(buf) as usize;

        // 配置哈希
        reader
            .read_exact(&mut buf)
            .map_err(|e| CheckpointError::io_with_path(path, e))?;
        let config_hash = u64::from_le_bytes(buf);

        // 创建时间
        reader
            .read_exact(&mut buf)
            .map_err(|e| CheckpointError::io_with_path(path, e))?;
        let created_at = u64::from_le_bytes(buf);

        // 单元数
        reader
            .read_exact(&mut buf)
            .map_err(|e| CheckpointError::io_with_path(path, e))?;
        let n_cells = u64::from_le_bytes(buf) as usize;

        Ok(CheckpointHeader {
            version,
            time,
            step,
            config_hash: if config_hash != 0 {
                Some(config_hash)
            } else {
                None
            },
            created_at,
            n_cells,
            mesh_hash: 0, // 需要读取完整文件才能获取
        })
    }

    /// 计算 CRC32 校验和（统一使用 crc32fast）
    #[allow(dead_code)]
    fn compute_crc32(data: &[u8]) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(data);
        hasher.finalize()
    }
}

#[allow(dead_code)]
fn read_u32(data: &[u8], offset: &mut usize) -> CheckpointResult<u32> {
    if *offset + 4 > data.len() {
        return Err(CheckpointError::Corrupted("unexpected EOF".into()));
    }
    let v = u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    Ok(v)
}

#[allow(dead_code)]
fn read_u64(data: &[u8], offset: &mut usize) -> CheckpointResult<u64> {
    if *offset + 8 > data.len() {
        return Err(CheckpointError::Corrupted("unexpected EOF".into()));
    }
    let v = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    Ok(v)
}

#[allow(dead_code)]
fn read_f64(data: &[u8], offset: &mut usize) -> CheckpointResult<f64> {
    read_u64(data, offset).map(f64::from_bits)
}


// ============================================================
// 检查点管理器
// ============================================================

/// 检查点管理器
///
/// 管理多个检查点文件，支持自动清理旧检查点。
pub struct CheckpointManager {
    /// 检查点目录
    directory: std::path::PathBuf,
    /// 最大保留数量
    max_checkpoints: usize,
    /// 文件名前缀
    prefix: String,
}

impl CheckpointManager {
    /// 创建新的管理器
    pub fn new(directory: impl Into<std::path::PathBuf>, max_checkpoints: usize) -> Self {
        Self {
            directory: directory.into(),
            max_checkpoints,
            prefix: "checkpoint".to_string(),
        }
    }

    /// 设置文件名前缀
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// 保存检查点
    pub fn save(&self, checkpoint: &Checkpoint) -> CheckpointResult<std::path::PathBuf> {
        std::fs::create_dir_all(&self.directory)
            .map_err(|e| CheckpointError::io_with_path(&self.directory, e))?;

        // 生成文件名
        let filename = format!("{}_{:08}.mhck", self.prefix, checkpoint.step);
        let path = self.directory.join(&filename);

        // 保存
        checkpoint.save(&path)?;

        // 清理旧检查点
        self.cleanup()?;

        Ok(path)
    }

    /// 加载最新的检查点
    pub fn load_latest(&self) -> CheckpointResult<Option<Checkpoint>> {
        let entries = self.list_checkpoints()?;
        if entries.is_empty() {
            return Ok(None);
        }

        // 按步数排序，取最新
        let latest = entries.into_iter().max_by_key(|(_, header)| header.step);
        if let Some((path, _)) = latest {
            Ok(Some(Checkpoint::load(&path)?))
        } else {
            Ok(None)
        }
    }

    /// 列出所有检查点
    pub fn list_checkpoints(
        &self,
    ) -> CheckpointResult<Vec<(std::path::PathBuf, CheckpointHeader)>> {
        let mut results = Vec::new();

        if !self.directory.exists() {
            return Ok(results);
        }

        let read_dir = std::fs::read_dir(&self.directory)
            .map_err(|e| CheckpointError::io_with_path(&self.directory, e))?;
        for entry in read_dir {
            let entry = entry.map_err(|e| CheckpointError::io_with_path(&self.directory, e))?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "mhck") {
                if let Ok(header) = Checkpoint::read_header(&path) {
                    results.push((path, header));
                }
            }
        }

        Ok(results)
    }

    /// 清理旧检查点
    fn cleanup(&self) -> CheckpointResult<()> {
        let mut entries = self.list_checkpoints()?;

        if entries.len() <= self.max_checkpoints {
            return Ok(());
        }

        // 按步数排序
        entries.sort_by_key(|(_, header)| header.step);

        // 删除最旧的
        let to_remove = entries.len() - self.max_checkpoints;
        for (path, _) in entries.into_iter().take(to_remove) {
            std::fs::remove_file(&path)
                .map_err(|e| CheckpointError::io_with_path(&path, e))?;
        }

        Ok(())
    }
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_state() -> StateSnapshot<f64> {
        StateSnapshot::<f64>::from_state_data(
            vec![1.0, 2.0, 3.0],
            vec![0.1, 0.2, 0.3],
            vec![0.0, 0.0, 0.0],
        )
    }

    #[test]
    fn test_checkpoint_creation() {
        let state = create_test_state();
        let checkpoint = Checkpoint::new(10.5, 100, state);

        assert!((checkpoint.time - 10.5).abs() < 1e-10);
        assert_eq!(checkpoint.step, 100);
        assert_eq!(checkpoint.state.n_cells(), 3);
    }

    #[test]
    fn test_checkpoint_save_load() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_checkpoint.mhck");

        let state = create_test_state();
        let checkpoint = Checkpoint::new(10.5, 100, state).with_config_hash(12345);
        checkpoint.save(&path).unwrap();

        let loaded = Checkpoint::load(&path).unwrap();

        assert!((loaded.time - 10.5).abs() < 1e-10);
        assert_eq!(loaded.step, 100);
        assert_eq!(loaded.state.n_cells(), 3);
        assert!((loaded.state.h[0] - 1.0).abs() < 1e-10);
        assert_eq!(loaded.config_hash, Some(12345));

        // 清理
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_checkpoint_with_bed() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_checkpoint_bed.mhck");

        let state = StateSnapshot::<f64>::from_state_data(
            vec![1.0, 2.0, 3.0],
            vec![0.0; 3],
            vec![0.0; 3],
        )
        .with_bed(vec![0.5, 1.0, 1.5]);

        let checkpoint = Checkpoint::new(5.0, 50, state);
        checkpoint.save(&path).unwrap();

        let loaded = Checkpoint::load(&path).unwrap();

        assert!(loaded.state.z.is_some());
        let z = loaded.state.z.as_ref().unwrap();
        assert!((z[0] - 0.5).abs() < 1e-10);

        // 清理
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_checkpoint_verify_hash() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_checkpoint_verify.mhck");

        let state = create_test_state();
        let checkpoint = Checkpoint::new(1.0, 1, state)
            .with_config_hash(111)
            .with_mesh_hash(222);
        checkpoint.save(&path).unwrap();

        let options = CheckpointLoadOptions {
            expected_config_hash: Some(111),
            expected_mesh_hash: Some(222),
            strict: true,
        };
        let loaded = Checkpoint::load_with_options(&path, options).unwrap();
        assert_eq!(loaded.config_hash, Some(111));
        assert_eq!(loaded.mesh_hash, 222);

        let bad_options = CheckpointLoadOptions {
            expected_config_hash: Some(999),
            expected_mesh_hash: Some(222),
            strict: true,
        };
        let err = Checkpoint::load_with_options(&path, bad_options).unwrap_err();
        assert!(matches!(err, CheckpointError::ConfigMismatch { .. }));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_header() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_checkpoint_header.mhck");

        let state = create_test_state();
        let checkpoint = Checkpoint::new(25.0, 250, state);
        checkpoint.save(&path).unwrap();

        let header = Checkpoint::read_header(&path).unwrap();

        assert!((header.time - 25.0).abs() < 1e-10);
        assert_eq!(header.step, 250);
        assert_eq!(header.n_cells, 3);

        // 清理
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_crc32() {
        let data = b"Hello, World!";
        let crc = Checkpoint::compute_crc32(data);
        // CRC32 of "Hello, World!" should be a specific value
        assert!(crc != 0);

        // 验证相同数据产生相同 CRC
        let crc2 = Checkpoint::compute_crc32(data);
        assert_eq!(crc, crc2);
    }
}
