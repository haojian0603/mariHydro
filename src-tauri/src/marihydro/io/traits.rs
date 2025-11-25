// src-tauri/src/marihydro/io/traits.rs

use super::types::{GeoGridData, GeoTransform, RasterMetadata};
use crate::marihydro::infra::error::MhResult;

/// 读取请求参数
/// 用于告诉 Driver：“我不只想要读取，我还想要你帮我处理成什么样”
#[derive(Debug, Clone)]
pub struct RasterRequest {
    /// 目标坐标系 (WKT)
    /// 如果提供，Driver 必须执行 Reprojection (Warp)
    pub target_crs_wkt: Option<String>,

    /// 目标尺寸 (nx, ny)
    /// 如果提供，Driver 必须执行 Resampling
    pub target_size: Option<(usize, usize)>,

    /// 目标地理范围 (min_x, min_y, max_x, max_y)
    /// 如果提供，Driver 必须执行 Clipping
    pub target_bounds: Option<(f64, f64, f64, f64)>,
}

impl Default for RasterRequest {
    fn default() -> Self {
        Self {
            target_crs_wkt: None,
            target_size: None,
            target_bounds: None,
        }
    }
}

/// 栅格数据驱动接口
/// 任何文件格式读取器 (GeoTIFF, ASC, NC) 都必须实现此接口
pub trait RasterDriver {
    /// 仅读取元数据 (快速，不加载大数组)
    /// 用于 CRS 自动探测和界面显示文件信息
    fn read_metadata(&self, path: &str) -> MhResult<RasterMetadata>;

    /// 读取完整数据
    /// 支持按需重采样和重投影
    fn read_data(&self, path: &str, request: Option<RasterRequest>) -> MhResult<GeoGridData>;

    /// 检查驱动是否支持该文件扩展名
    fn supports_extension(&self, ext: &str) -> bool;
}

/// 数据导出接口
/// 用于保存模拟结果 (VTK, GeoTIFF)
pub trait ResultExporter {
    fn export(
        &self,
        path: &str,
        data: &GeoGridData,
        // 可能还需要传入 Mesh 拓扑信息，视具体格式而定
        // 这里暂时保持简单，针对结构化网格
    ) -> MhResult<()>;
}
