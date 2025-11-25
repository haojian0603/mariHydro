pub mod dto;
pub mod gdal_inspector;
pub mod nc_inspector;

use self::dto::DatasetMetadata;
use crate::marihydro::infra::error::MhResult;

/// 文件探查器接口
/// 用于 UI "打开文件" 后立即调用的轻量级扫描
pub trait FileInspector {
    /// 扫描文件头，不读取大块数据
    fn inspect(&self, path: &str) -> MhResult<DatasetMetadata>;
    /// 检查是否支持该扩展名
    fn supports(&self, ext: &str) -> bool;
}
