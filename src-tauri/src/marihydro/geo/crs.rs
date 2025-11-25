// src-tauri/src/marihydro/geo/crs.rs

use crate::marihydro::infra::error::{MhError, MhResult};
use proj::Proj;
use serde::{Deserialize, Serialize};

/// 用户指定的 CRS 获取策略
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "mode", content = "value")] // 优化 JSON 格式
pub enum CrsStrategy {
    /// 手动指定 WKT 或 EPSG 代码 (例如 "EPSG:32651")
    Manual(String),

    /// 动态从第一个加载的地理文件 (GeoTIFF/NetCDF) 中读取
    /// 这是 GIS 软件最常用的行为
    FromFirstFile,

    /// 强制使用 WGS84 (经纬度)，通常不推荐用于水动力计算，除非是大尺度球面模型
    ForceWGS84,
}

/// 运行时确定的坐标参考系统
/// 这是 SimulationContext 中持有的对象
#[derive(Debug, Clone)]
pub struct ResolvedCrs {
    /// 标准化后的定义字符串 (WKT)
    pub wkt: String,
    /// 对应的 Proj 实例 (用于验证和转换)
    /// 不需要序列化，每次启动时重新构建
    _proj_instance: Proj,
}

impl ResolvedCrs {
    /// 尝试构建并验证 CRS
    pub fn new(definition: &str) -> MhResult<Self> {
        let proj = Proj::new(definition)
            .map_err(|e| MhError::Projection(format!("无效的坐标定义 '{}': {}", definition, e)))?;

        Ok(Self {
            wkt: definition.to_string(),
            _proj_instance: proj,
        })
    }

    /// 检查单位是否为米 (Meter)
    /// 水动力计算强烈建议使用投影坐标系 (米)，而非经纬度 (度)
    pub fn is_metric(&self) -> bool {
        // 这里使用简化的字符串判断，严格做法是调用 proj_get_units (需要 FFI)
        // 只要不是 4326 或 LongLat，通常都是投影坐标
        !self.wkt.contains("EPSG:4326")
            && !self.wkt.contains("GEOGCS")
            && !self.wkt.to_lowercase().contains("degree")
    }
}
