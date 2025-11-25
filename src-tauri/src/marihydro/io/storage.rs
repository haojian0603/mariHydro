// src-tauri/src/marihydro/io/storage.rs

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::time::TimeManager;

/// 模拟状态快照
/// 包含恢复模拟所需的最少信息集合
/// 使用 #[serde] 支持序列化
#[derive(Serialize, Deserialize, Debug)]
pub struct Snapshot {
    /// 版本标记 (用于防止旧版本软件读取新版本存档)
    pub version: String,

    /// 保存时的物理时间
    pub time_state: TimeManager,

    /// --- 物理场数据 (Conservative Variables) ---
    /// 水深 h
    pub h: Array2<f64>,
    /// x方向单宽流量 hu
    pub hu: Array2<f64>,
    /// y方向单宽流量 hv
    pub hv: Array2<f64>,
    /// 泥沙浓度/质量 hc
    pub hc: Array2<f64>,
    // 注意：网格几何信息 (Mesh) 通常不保存在快照中，
    // 而是重新加载原始配置生成，以减小存档体积。
    // 除非支持“动态网格自适应”，否则网格是静态的。
}

impl Snapshot {
    /// 将快照保存到文件 (二进制格式 .mhs)
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> MhResult<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        // 使用 bincode 进行二进制序列化
        bincode::serialize_into(writer, self)
            .map_err(|e| MhError::Serialization(format!("快照保存失败: {}", e)))?;

        Ok(())
    }

    /// 从文件加载快照
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> MhResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let snapshot: Self = bincode::deserialize_from(reader)
            .map_err(|e| MhError::Serialization(format!("快照读取失败或版本不兼容: {}", e)))?;

        Ok(snapshot)
    }
}
