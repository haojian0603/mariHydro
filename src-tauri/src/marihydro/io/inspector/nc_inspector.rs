use super::dto::{DatasetMetadata, VariableInfo};
use super::FileInspector;
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::manifest::DataFormat;
use crate::marihydro::io::drivers::nc_adapter::time::{calculate_utc_time, parse_cf_time_units};
use netcdf;

pub struct NcInspector;

impl FileInspector for NcInspector {
    fn supports(&self, ext: &str) -> bool {
        matches!(ext.to_lowercase().as_str(), "nc" | "cdf" | "netcdf")
    }

    fn inspect(&self, path: &str) -> MhResult<DatasetMetadata> {
        // 使用 netcdf crate 打开文件 (只读模式)
        let file = netcdf::open(path).map_err(MhError::NetCdf)?;

        // 1. 扫描变量
        let mut variables = Vec::new();
        let mut time_coverage = None;

        for var in file.variables() {
            // 提取属性 (Attributes)
            let standard_name = var
                .attribute("standard_name")
                .and_then(|a| a.value().as_str().map(|s| s.to_string()));

            let units = var
                .attribute("units")
                .and_then(|a| a.value().as_str().map(|s| s.to_string()));

            // 提取维度信息
            let dims: Vec<String> = var
                .dimensions()
                .iter()
                .map(|d| format!("{}:{}", d.name(), d.len()))
                .collect();

            // 记录变量信息
            variables.push(VariableInfo {
                name: var.name().to_string(),
                dimensions: dims,
                dtype: "float".to_string(), // 简化处理，实际可映射 TypeId
                standard_name,
                units: units.clone(), // Clone for later use check
            });

            // 2. 尝试解析时间覆盖范围 (启发式: 变量名叫 time 或 standard_name 是 time)
            let is_time_var = var.name() == "time"
                || var
                    .attribute("standard_name")
                    .map(|a| a.value().to_string())
                    .unwrap_or_default()
                    == "time";

            if is_time_var && time_coverage.is_none() {
                if let Some(unit_str) = units {
                    // 尝试读取首尾时间值
                    let len = var.len();
                    if len > 0 {
                        // 读取第0个和第N-1个值
                        // 注意：这里可能会有 IO 开销，但相对于整个文件很小
                        let first_val = var.value::<f64>(Some(&[0]));
                        let last_val = var.value::<f64>(Some(&[len - 1]));

                        if let (Ok(v1), Ok(v2)) = (first_val, last_val) {
                            if let Ok((base, mult)) = parse_cf_time_units(&unit_str) {
                                let t_start = calculate_utc_time(v1, base, mult);
                                let t_end = calculate_utc_time(v2, base, mult);
                                time_coverage = Some((t_start, t_end));
                            }
                        }
                    }
                }
            }
        }

        Ok(DatasetMetadata {
            format: DataFormat::NetCDF,
            variables,
            crs_wkt: None, // NC 文件通常很难直接获取 WKT，除非有 Grid Mapping 变量
            time_coverage,
            geo_bounds: None, // 需要读取 lat/lon 变量的最值，这里暂略
        })
    }
}
