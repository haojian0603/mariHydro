// src-tauri/src/marihydro/forcing/providers/wind/factory.rs
//! 风场读取器工厂
//! 自动检测格式并创建对应的 WindProvider

use super::csv_reader::{CsvWindConfig, CsvWindReader};
use super::excel_reader::{ExcelWindConfig, ExcelWindReader};
#[cfg(feature = "grib")]
use super::grib::GribWindProvider;
#[cfg(feature = "netcdf")]
use super::netcdf::NetCdfWindProvider;
use super::provider::UniformWindProvider;
use super::text_reader::{TextWindConfig, TextWindReader};
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::forcing::manager::WindProvider;
use chrono::{DateTime, Utc};
use std::path::Path;

/// 风场数据格式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindFormat {
    /// 自动检测
    Auto,
    /// 文本格式（空格/制表符分隔）
    Text,
    /// CSV 格式
    Csv,
    /// Excel 格式 (.xlsx, .xls)
    Excel,
    /// NetCDF 格式
    NetCdf,
    /// GRIB 格式
    Grib,
    /// 常量均匀风场
    Uniform,
}

/// 风场读取器配置
#[derive(Debug, Clone)]
pub struct WindReaderConfig {
    /// 网格单元数
    pub n_cells: usize,
    /// 模拟起始时间
    pub start_time: DateTime<Utc>,
    /// 格式特定配置
    pub format_config: FormatConfig,
}

/// 格式特定配置
#[derive(Debug, Clone)]
pub enum FormatConfig {
    /// 无额外配置
    None,
    /// 文本格式配置
    Text(TextWindConfig),
    /// CSV 格式配置
    Csv(CsvWindConfig),
    /// Excel 格式配置
    Excel(ExcelWindConfig),
    /// 均匀风场（u, v）
    Uniform(f64, f64),
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self::None
    }
}

/// 风场读取器工厂
pub struct WindReaderFactory;

impl WindReaderFactory {
    /// 自动检测格式并创建读取器
    pub fn create(
        path: &Path,
        config: WindReaderConfig,
    ) -> MhResult<Box<dyn WindProvider>> {
        let format = Self::detect_format(path)?;
        Self::create_with_format(path, format, config)
    }

    /// 使用指定格式创建读取器
    pub fn create_with_format(
        path: &Path,
        format: WindFormat,
        config: WindReaderConfig,
    ) -> MhResult<Box<dyn WindProvider>> {
        match format {
            WindFormat::Auto => Self::create(path, config),
            
            WindFormat::Text => {
                let text_config = match config.format_config {
                    FormatConfig::Text(c) => c,
                    _ => TextWindConfig::default(),
                };
                Ok(Box::new(TextWindReader::open_with_config(
                    path,
                    config.n_cells,
                    config.start_time,
                    text_config,
                )?))
            }
            
            WindFormat::Csv => {
                let csv_config = match config.format_config {
                    FormatConfig::Csv(c) => c,
                    _ => CsvWindConfig::default(),
                };
                Ok(Box::new(CsvWindReader::open_with_config(
                    path,
                    config.n_cells,
                    config.start_time,
                    csv_config,
                )?))
            }
            
            WindFormat::Excel => {
                let excel_config = match config.format_config {
                    FormatConfig::Excel(c) => c,
                    _ => ExcelWindConfig::default(),
                };
                Ok(Box::new(ExcelWindReader::open_with_config(
                    path,
                    config.n_cells,
                    config.start_time,
                    excel_config,
                )?))
            }
            
            WindFormat::NetCdf => {
                #[cfg(feature = "netcdf")]
                {
                    Ok(Box::new(NetCdfWindProvider::open(path, config.n_cells)?))
                }
                #[cfg(not(feature = "netcdf"))]
                {
                    Err(MhError::Config(
                        "NetCDF support not enabled. Compile with feature 'netcdf'".into(),
                    ))
                }
            }
            
            WindFormat::Grib => {
                #[cfg(feature = "grib")]
                {
                    Ok(Box::new(GribWindProvider::open(path, config.n_cells)?))
                }
                #[cfg(not(feature = "grib"))]
                {
                    Err(MhError::Config(
                        "GRIB support not enabled. Compile with feature 'grib'".into(),
                    ))
                }
            }
            
            WindFormat::Uniform => {
                let (u, v) = match config.format_config {
                    FormatConfig::Uniform(u, v) => (u, v),
                    _ => (0.0, 0.0),
                };
                Ok(Box::new(UniformWindProvider::new(u, v)))
            }
        }
    }

    /// 根据文件扩展名检测格式
    pub fn detect_format(path: &Path) -> MhResult<WindFormat> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        match ext.as_str() {
            "txt" | "dat" | "asc" => Ok(WindFormat::Text),
            "csv" | "tsv" => Ok(WindFormat::Csv),
            "xlsx" | "xls" | "xlsm" => Ok(WindFormat::Excel),
            "nc" | "nc4" | "netcdf" => Ok(WindFormat::NetCdf),
            "grib" | "grib2" | "grb" | "grb2" => Ok(WindFormat::Grib),
            _ => {
                // 尝试从文件内容推断
                Self::detect_format_from_content(path)
            }
        }
    }

    /// 从文件内容推断格式
    fn detect_format_from_content(path: &Path) -> MhResult<WindFormat> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path).map_err(|e| MhError::Io(e.to_string()))?;
        let mut reader = BufReader::new(file);
        let mut first_line = String::new();
        reader
            .read_line(&mut first_line)
            .map_err(|e| MhError::Io(e.to_string()))?;

        // 检查文件魔数
        let bytes = first_line.as_bytes();
        
        // NetCDF: 以 "CDF" 或 0x89 'HDF' 开头
        if bytes.len() >= 4 {
            if &bytes[0..3] == b"CDF" || &bytes[1..4] == b"HDF" {
                return Ok(WindFormat::NetCdf);
            }
            // GRIB: 以 "GRIB" 开头
            if &bytes[0..4] == b"GRIB" {
                return Ok(WindFormat::Grib);
            }
            // Excel XLSX: ZIP 格式 (PK)
            if &bytes[0..2] == b"PK" {
                return Ok(WindFormat::Excel);
            }
        }

        // 检查是否为 CSV（包含逗号）
        if first_line.contains(',') {
            return Ok(WindFormat::Csv);
        }

        // 默认为文本格式
        Ok(WindFormat::Text)
    }

    /// 快捷方法：从路径创建带默认配置的读取器
    pub fn from_path(
        path: &Path,
        n_cells: usize,
        start_time: DateTime<Utc>,
    ) -> MhResult<Box<dyn WindProvider>> {
        Self::create(
            path,
            WindReaderConfig {
                n_cells,
                start_time,
                format_config: FormatConfig::None,
            },
        )
    }

    /// 快捷方法：创建均匀风场
    pub fn uniform(u: f64, v: f64) -> Box<dyn WindProvider> {
        Box::new(UniformWindProvider::new(u, v))
    }

    /// 获取支持的格式列表
    pub fn supported_formats() -> Vec<WindFormat> {
        let mut formats = vec![
            WindFormat::Text,
            WindFormat::Csv,
            WindFormat::Excel,
            WindFormat::Uniform,
        ];

        #[cfg(feature = "netcdf")]
        formats.push(WindFormat::NetCdf);

        #[cfg(feature = "grib")]
        formats.push(WindFormat::Grib);

        formats
    }

    /// 获取格式对应的文件扩展名
    pub fn format_extensions(format: WindFormat) -> Vec<&'static str> {
        match format {
            WindFormat::Auto => vec![],
            WindFormat::Text => vec!["txt", "dat", "asc"],
            WindFormat::Csv => vec!["csv", "tsv"],
            WindFormat::Excel => vec!["xlsx", "xls", "xlsm"],
            WindFormat::NetCdf => vec!["nc", "nc4", "netcdf"],
            WindFormat::Grib => vec!["grib", "grib2", "grb", "grb2"],
            WindFormat::Uniform => vec![],
        }
    }
}

/// WindReaderConfig 构建器
impl WindReaderConfig {
    pub fn new(n_cells: usize, start_time: DateTime<Utc>) -> Self {
        Self {
            n_cells,
            start_time,
            format_config: FormatConfig::None,
        }
    }

    pub fn with_text_config(mut self, config: TextWindConfig) -> Self {
        self.format_config = FormatConfig::Text(config);
        self
    }

    pub fn with_csv_config(mut self, config: CsvWindConfig) -> Self {
        self.format_config = FormatConfig::Csv(config);
        self
    }

    pub fn with_excel_config(mut self, config: ExcelWindConfig) -> Self {
        self.format_config = FormatConfig::Excel(config);
        self
    }

    pub fn with_uniform(mut self, u: f64, v: f64) -> Self {
        self.format_config = FormatConfig::Uniform(u, v);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(
            WindReaderFactory::detect_format(Path::new("wind.csv")).unwrap(),
            WindFormat::Csv
        );
        assert_eq!(
            WindReaderFactory::detect_format(Path::new("wind.txt")).unwrap(),
            WindFormat::Text
        );
        assert_eq!(
            WindReaderFactory::detect_format(Path::new("wind.xlsx")).unwrap(),
            WindFormat::Excel
        );
        assert_eq!(
            WindReaderFactory::detect_format(Path::new("wind.nc")).unwrap(),
            WindFormat::NetCdf
        );
        assert_eq!(
            WindReaderFactory::detect_format(Path::new("wind.grib2")).unwrap(),
            WindFormat::Grib
        );
    }

    #[test]
    fn test_uniform_factory() {
        let provider = WindReaderFactory::uniform(5.0, 3.0);
        let mut u = vec![0.0; 10];
        let mut v = vec![0.0; 10];
        provider.get_wind_at(Utc::now(), &mut u, &mut v).unwrap();
        assert!((u[0] - 5.0).abs() < 1e-10);
        assert!((v[0] - 3.0).abs() < 1e-10);
    }
}
