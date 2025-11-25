use crate::marihydro::infra::manifest::DataFormat;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// 变量信息摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableInfo {
    /// 变量名 (文件内部名称，如 "u10")
    pub name: String,

    /// 维度形状 (e.g., [Time: 24, Lat: 100, Lon: 100])
    /// 这里用字符串描述维度名，方便展示
    pub dimensions: Vec<String>,

    /// 数据类型 (f32, f64, i32...)
    pub dtype: String,

    /// [关键] CF-Convention 标准名 (e.g., "eastward_wind")
    /// 前端可以利用这个字段自动匹配 "wind_u"
    pub standard_name: Option<String>,

    /// [关键] 单位 (e.g., "m s-1")
    /// 前端利用这个判断是否需要 scale_factor
    pub units: Option<String>,
}

/// 数据集元数据摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// 识别出的格式
    pub format: DataFormat,

    /// 包含的所有变量列表
    pub variables: Vec<VariableInfo>,

    /// 空间参考 (WKT)
    pub crs_wkt: Option<String>,

    /// 时间覆盖范围 (Start, End)
    /// 仅针对有时变数据的文件 (NetCDF)
    pub time_coverage: Option<(DateTime<Utc>, DateTime<Utc>)>,

    /// 空间覆盖范围 (min_x, min_y, max_x, max_y)
    pub geo_bounds: Option<[f64; 4]>,
}
