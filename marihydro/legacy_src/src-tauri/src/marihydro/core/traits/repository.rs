// src-tauri/src/marihydro/core/traits/repository.rs

//! 持久化抽象接口
//!
//! 定义数据持久化的统一抽象，遵循 Repository 模式。

use crate::marihydro::core::error::MhResult;
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;

/// 通用存储库接口
pub trait Repository: Send + Sync {
    /// 存储库名称（用于日志）
    fn name(&self) -> &str;

    /// 是否已初始化
    fn is_initialized(&self) -> bool;

    /// 关闭连接
    fn close(&mut self) -> MhResult<()>;
}

/// 模拟数据存储库接口
///
/// 专门用于水动力模拟数据的持久化
pub trait SimulationRepository: Repository {
    /// 保存模拟元数据
    fn save_metadata(&self, key: &str, value: &str) -> MhResult<()>;

    /// 读取模拟元数据
    fn load_metadata(&self, key: &str) -> MhResult<Option<String>>;

    /// 保存时间步数据
    fn save_timestep(&self, time: f64, h: &[f64], hu: &[f64], hv: &[f64]) -> MhResult<()>;

    /// 读取时间步数据
    fn load_timestep(&self, time: f64) -> MhResult<Option<TimestepData>>;

    /// 列出所有保存的时间点
    fn list_timesteps(&self) -> MhResult<Vec<f64>>;

    /// 保存网格数据
    fn save_mesh_data(
        &self,
        nodes: &[[f64; 2]],
        cells: &[Vec<usize>],
        cell_centers: &[[f64; 2]],
        cell_areas: &[f64],
    ) -> MhResult<()>;

    /// 加载网格数据
    fn load_mesh_data(&self) -> MhResult<Option<MeshData>>;

    /// 保存底床高程
    fn save_bathymetry(&self, z: &[f64]) -> MhResult<()>;

    /// 加载底床高程
    fn load_bathymetry(&self) -> MhResult<Option<Vec<f64>>>;

    /// 保存检查点（用于断点续算）
    fn save_checkpoint(
        &self,
        time: f64,
        step: usize,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
    ) -> MhResult<()>;

    /// 加载最新检查点
    fn load_latest_checkpoint(&self) -> MhResult<Option<CheckpointData>>;
}

/// 时间步数据
#[derive(Debug, Clone)]
pub struct TimestepData {
    pub time: f64,
    pub h: Vec<f64>,
    pub hu: Vec<f64>,
    pub hv: Vec<f64>,
}

/// 网格数据
#[derive(Debug, Clone)]
pub struct MeshData {
    pub nodes: Vec<[f64; 2]>,
    pub cells: Vec<Vec<usize>>,
    pub cell_centers: Vec<[f64; 2]>,
    pub cell_areas: Vec<f64>,
}

/// 检查点数据
#[derive(Debug, Clone)]
pub struct CheckpointData {
    pub time: f64,
    pub step: usize,
    pub h: Vec<f64>,
    pub hu: Vec<f64>,
    pub hv: Vec<f64>,
}

/// 键值存储接口（简化版本）
pub trait KeyValueStore: Send + Sync {
    /// 保存值
    fn set<T: Serialize>(&self, key: &str, value: &T) -> MhResult<()>;

    /// 读取值
    fn get<T: DeserializeOwned>(&self, key: &str) -> MhResult<Option<T>>;

    /// 删除键
    fn delete(&self, key: &str) -> MhResult<()>;

    /// 检查键是否存在
    fn exists(&self, key: &str) -> MhResult<bool>;

    /// 列出所有键
    fn keys(&self) -> MhResult<Vec<String>>;
}

/// 文件存储接口
pub trait FileStore: Send + Sync {
    /// 写入文件
    fn write(&self, path: &Path, data: &[u8]) -> MhResult<()>;

    /// 读取文件
    fn read(&self, path: &Path) -> MhResult<Vec<u8>>;

    /// 检查文件是否存在
    fn exists(&self, path: &Path) -> bool;

    /// 删除文件
    fn delete(&self, path: &Path) -> MhResult<()>;

    /// 列出目录内容
    fn list(&self, dir: &Path) -> MhResult<Vec<std::path::PathBuf>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestep_data() {
        let data = TimestepData {
            time: 100.0,
            h: vec![1.0, 2.0, 3.0],
            hu: vec![0.1, 0.2, 0.3],
            hv: vec![0.0, 0.0, 0.0],
        };

        assert_eq!(data.h.len(), 3);
        assert!((data.time - 100.0).abs() < 1e-10);
    }
}
