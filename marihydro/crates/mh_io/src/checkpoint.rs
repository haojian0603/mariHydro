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
//!
//! // 保存检查点
//! let state = StateSnapshot::from_state_data(h, hu, hv);
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
    /// 格式错误
    Format(String),
    /// 版本不兼容
    Version { file: u32, current: u32 },
    /// 网格不匹配
    MeshMismatch { expected: usize, found: usize },
    /// 校验和错误
    Checksum { expected: u32, found: u32 },
    /// 数据损坏
    Corrupted(String),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Io(e) => write!(f, "IO 错误: {}", e),
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
    pub state: StateSnapshot,
    /// 配置摘要哈希（用于验证）
    pub config_hash: Option<u64>,
    /// 创建时间戳
    pub created_at: u64,
    /// 网格哈希（用于兼容性检查）
    pub mesh_hash: u64,
}

impl Checkpoint {
    /// 创建新检查点
    pub fn new(time: f64, step: usize, state: StateSnapshot) -> Self {
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
    pub fn with_mesh_snapshot(mut self, mesh: &MeshSnapshot) -> Self {
        // 简单哈希：组合节点数和单元数
        self.mesh_hash = (mesh.n_nodes as u64) << 32 | (mesh.n_cells as u64);
        self
    }

    /// 保存到文件（二进制格式）
    pub fn save(&self, path: &Path) -> CheckpointResult<()> {
        // 创建目录
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // 使用临时文件写入，成功后重命名（原子操作）
        let temp_path = path.with_extension("mhck.tmp");
        
        {
            let file = File::create(&temp_path)?;
            let mut writer = BufWriter::new(file);

            // 收集所有数据用于计算 CRC
            let mut data = Vec::new();

            // 魔数
            data.extend_from_slice(CHECKPOINT_MAGIC);

            // 版本
            data.extend_from_slice(&self.version.to_le_bytes());

            // 时间和步数
            data.extend_from_slice(&self.time.to_le_bytes());
            data.extend_from_slice(&(self.step as u64).to_le_bytes());

            // 配置哈希
            let hash = self.config_hash.unwrap_or(0);
            data.extend_from_slice(&hash.to_le_bytes());

            // 创建时间
            data.extend_from_slice(&self.created_at.to_le_bytes());

            // 单元数
            let n_cells = self.state.n_cells();
            data.extend_from_slice(&(n_cells as u64).to_le_bytes());

            // 状态数据
            for &h in &self.state.h {
                data.extend_from_slice(&h.to_le_bytes());
            }
            for &hu in &self.state.hu {
                data.extend_from_slice(&hu.to_le_bytes());
            }
            for &hv in &self.state.hv {
                data.extend_from_slice(&hv.to_le_bytes());
            }

            // 底床高程
            let has_z = self.state.z.is_some() as u8;
            data.push(has_z);
            if let Some(z) = &self.state.z {
                for &val in z {
                    data.extend_from_slice(&val.to_le_bytes());
                }
            }

            // 网格哈希
            data.extend_from_slice(&self.mesh_hash.to_le_bytes());

            // 写入数据
            writer.write_all(&data)?;

            // 计算并写入 CRC32
            let crc = Self::compute_crc32(&data);
            writer.write_all(&crc.to_le_bytes())?;

            writer.flush()?;
        }

        // 原子重命名
        std::fs::rename(&temp_path, path)?;

        Ok(())
    }

    /// 从文件加载
    pub fn load(path: &Path) -> CheckpointResult<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // 读取全部数据
        let mut all_data = Vec::new();
        reader.read_to_end(&mut all_data)?;

        // 至少需要魔数 + 版本 + CRC
        if all_data.len() < 12 {
            return Err(CheckpointError::Format("文件太小".into()));
        }

        // 分离 CRC
        let crc_offset = all_data.len() - 4;
        let data = &all_data[..crc_offset];
        let stored_crc = u32::from_le_bytes([
            all_data[crc_offset],
            all_data[crc_offset + 1],
            all_data[crc_offset + 2],
            all_data[crc_offset + 3],
        ]);

        // 验证 CRC
        let computed_crc = Self::compute_crc32(data);
        if stored_crc != computed_crc && stored_crc != 0 {
            // stored_crc == 0 表示旧版本文件，跳过校验
            return Err(CheckpointError::Checksum {
                expected: stored_crc,
                found: computed_crc,
            });
        }

        // 解析数据
        let mut offset = 0;

        // 魔数
        if &data[offset..offset + 4] != CHECKPOINT_MAGIC {
            return Err(CheckpointError::Format("无效的检查点文件格式".into()));
        }
        offset += 4;

        // 版本
        let version = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;

        if version > MAX_SUPPORTED_VERSION {
            return Err(CheckpointError::Version {
                file: version,
                current: CHECKPOINT_VERSION,
            });
        }

        // 时间
        let time = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        // 步数
        let step = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        // 配置哈希
        let config_hash = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        // 创建时间
        let created_at = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        // 单元数
        let n_cells = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        // 状态数据
        let mut h = Vec::with_capacity(n_cells);
        let mut hu = Vec::with_capacity(n_cells);
        let mut hv = Vec::with_capacity(n_cells);

        for _ in 0..n_cells {
            h.push(f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()));
            offset += 8;
        }
        for _ in 0..n_cells {
            hu.push(f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()));
            offset += 8;
        }
        for _ in 0..n_cells {
            hv.push(f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()));
            offset += 8;
        }

        // 底床高程
        let has_z = data[offset] != 0;
        offset += 1;

        let z = if has_z {
            let mut z_vec = Vec::with_capacity(n_cells);
            for _ in 0..n_cells {
                z_vec.push(f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()));
                offset += 8;
            }
            Some(z_vec)
        } else {
            None
        };

        // 网格哈希
        let mesh_hash = if offset + 8 <= data.len() {
            u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
        } else {
            0
        };

        let mut state = StateSnapshot::from_state_data(h, hu, hv);
        if let Some(z_data) = z {
            state = state.with_bed(z_data);
        }

        Ok(Self {
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
        })
    }

    /// 仅读取头部信息（不加载状态数据）
    pub fn read_header(path: &Path) -> CheckpointResult<CheckpointHeader> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // 魔数
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != CHECKPOINT_MAGIC {
            return Err(CheckpointError::Format("无效的检查点文件格式".into()));
        }

        // 版本
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf[..4])?;
        let version = u32::from_le_bytes(buf[..4].try_into().unwrap());

        // 时间
        reader.read_exact(&mut buf)?;
        let time = f64::from_le_bytes(buf);

        // 步数
        reader.read_exact(&mut buf)?;
        let step = u64::from_le_bytes(buf) as usize;

        // 配置哈希
        reader.read_exact(&mut buf)?;
        let config_hash = u64::from_le_bytes(buf);

        // 创建时间
        reader.read_exact(&mut buf)?;
        let created_at = u64::from_le_bytes(buf);

        // 单元数
        reader.read_exact(&mut buf)?;
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

    /// 验证网格兼容性
    pub fn verify_mesh_compatibility(&self, expected_cells: usize) -> CheckpointResult<()> {
        let found = self.state.n_cells();
        if found != expected_cells {
            return Err(CheckpointError::MeshMismatch {
                expected: expected_cells,
                found,
            });
        }
        Ok(())
    }

    /// 验证配置兼容性
    pub fn verify_config_compatibility(&self, expected_hash: u64) -> bool {
        match self.config_hash {
            Some(hash) => hash == expected_hash,
            None => true, // 无哈希时跳过验证
        }
    }

    /// 计算 CRC32 校验和
    fn compute_crc32(data: &[u8]) -> u32 {
        // 简化的 CRC32 实现（IEEE 多项式）
        let mut crc = 0xFFFFFFFFu32;
        for &byte in data {
            let index = ((crc ^ byte as u32) & 0xFF) as usize;
            crc = CRC32_TABLE[index] ^ (crc >> 8);
        }
        !crc
    }
}

/// 生成 CRC32 查找表（编译期计算）
const fn generate_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = 0xEDB88320 ^ (crc >> 1);
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// CRC32 查找表（IEEE 多项式，编译期生成）
const CRC32_TABLE: [u32; 256] = generate_crc32_table();

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
        std::fs::create_dir_all(&self.directory)?;

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

        for entry in std::fs::read_dir(&self.directory)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map_or(false, |ext| ext == "mhck") {
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
            let _ = std::fs::remove_file(path);
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
    

    fn create_test_state() -> StateSnapshot {
        StateSnapshot::from_state_data(
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

        let state = StateSnapshot::from_state_data(
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
    fn test_mesh_compatibility() {
        let state = create_test_state();
        let checkpoint = Checkpoint::new(0.0, 0, state);

        assert!(checkpoint.verify_mesh_compatibility(3).is_ok());
        assert!(checkpoint.verify_mesh_compatibility(5).is_err());
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
