// crates/mh_io/src/drivers/netcdf/driver.rs

//! NetCDF 驱动实现

use super::error::NetCdfError;
use std::path::Path;
use std::process::Command;

/// 维度信息
#[derive(Debug, Clone)]
pub struct Dimension {
    /// 名称
    pub name: String,
    /// 长度
    pub len: usize,
    /// 是否无限
    pub is_unlimited: bool,
}

/// 变量信息
#[derive(Debug, Clone)]
pub struct VariableInfo {
    /// 名称
    pub name: String,
    /// 维度名称列表
    pub dimensions: Vec<String>,
    /// 数据类型
    pub dtype: String,
    /// 标准名称 (CF 约定)
    pub standard_name: Option<String>,
    /// 长名称
    pub long_name: Option<String>,
    /// 单位
    pub units: Option<String>,
}

/// 变量数据
#[derive(Debug, Clone)]
pub struct Variable {
    /// 数据
    pub data: Vec<f64>,
    /// 维度大小
    pub dims: Vec<usize>,
}

impl Variable {
    /// 计算线性索引
    fn linear_index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.dims.len() {
            return None;
        }

        let mut idx = 0;
        let mut stride = 1;
        for (i, &dim_size) in self.dims.iter().enumerate().rev() {
            if indices[i] >= dim_size {
                return None;
            }
            idx += indices[i] * stride;
            stride *= dim_size;
        }
        Some(idx)
    }

    /// 获取指定索引的值
    pub fn get(&self, indices: &[usize]) -> Option<f64> {
        let idx = self.linear_index(indices)?;
        Some(self.data[idx])
    }

    /// 获取总元素数
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// NetCDF 驱动
#[cfg(feature = "netcdf")]
pub struct NetCdfDriver {
    file: netcdf::File,
}

#[cfg(feature = "netcdf")]
impl NetCdfDriver {
    /// 打开 NetCDF 文件
    pub fn open(path: impl AsRef<Path>) -> Result<Self, NetCdfError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(NetCdfError::FileNotFound(path.display().to_string()));
        }

        let file = netcdf::open(path)?;
        Ok(Self { file })
    }

    /// 获取所有维度
    pub fn dimensions(&self) -> Result<Vec<Dimension>, NetCdfError> {
        let dims: Vec<_> = self
            .file
            .dimensions()
            .map(|d| Dimension {
                name: d.name().to_string(),
                len: d.len(),
                is_unlimited: d.is_unlimited(),
            })
            .collect();
        Ok(dims)
    }

    /// 获取维度
    pub fn dimension(&self, name: &str) -> Result<Dimension, NetCdfError> {
        let d = self
            .file
            .dimension(name)
            .ok_or_else(|| NetCdfError::DimensionNotFound(name.to_string()))?;
        Ok(Dimension {
            name: d.name().to_string(),
            len: d.len(),
            is_unlimited: d.is_unlimited(),
        })
    }

    /// 获取所有变量信息
    pub fn variables(&self) -> Result<Vec<VariableInfo>, NetCdfError> {
        let vars: Vec<_> = self
            .file
            .variables()
            .map(|v| {
                let dims: Vec<String> = v.dimensions().iter().map(|d| d.name().to_string()).collect();
                
                VariableInfo {
                    name: v.name().to_string(),
                    dimensions: dims,
                    dtype: format!("{:?}", v.vartype()),
                    standard_name: v.attribute("standard_name").and_then(|a| a.value().ok()).and_then(|v| match v {
                        netcdf::AttrValue::Str(s) => Some(s.to_string()),
                        _ => None,
                    }),
                    long_name: v.attribute("long_name").and_then(|a| a.value().ok()).and_then(|v| match v {
                        netcdf::AttrValue::Str(s) => Some(s.to_string()),
                        _ => None,
                    }),
                    units: v.attribute("units").and_then(|a| a.value().ok()).and_then(|v| match v {
                        netcdf::AttrValue::Str(s) => Some(s.to_string()),
                        _ => None,
                    }),
                }
            })
            .collect();
        Ok(vars)
    }

    /// 读取变量
    pub fn read_variable(&self, name: &str) -> Result<Variable, NetCdfError> {
        let var = self
            .file
            .variable(name)
            .ok_or_else(|| NetCdfError::VariableNotFound(name.to_string()))?;

        let dims: Vec<usize> = var.dimensions().iter().map(|d| d.len()).collect();
        let data: Vec<f64> = var.values::<f64, _>(..)
            .map_err(|e| NetCdfError::ReadFailed(e.to_string()))?;

        Ok(Variable { data, dims })
    }

    /// 读取变量的一个时间切片
    pub fn read_variable_slice(
        &self,
        name: &str,
        time_idx: usize,
    ) -> Result<Variable, NetCdfError> {
        let var = self
            .file
            .variable(name)
            .ok_or_else(|| NetCdfError::VariableNotFound(name.to_string()))?;

        let dims: Vec<usize> = var.dimensions().iter().map(|d| d.len()).collect();
        
        if dims.is_empty() {
            return Err(NetCdfError::ReadFailed("Variable has no dimensions".to_string()));
        }

        if time_idx >= dims[0] {
            return Err(NetCdfError::ReadFailed("time index out of range".to_string()));
        }

        // 假设第一个维度是时间
        let slice_dims: Vec<usize> = dims[1..].to_vec();
        
        // 构建索引范围
        let extents: Vec<_> = std::iter::once(time_idx..time_idx + 1)
            .chain(dims[1..].iter().map(|&d| 0..d))
            .collect();
        
        let data: Vec<f64> = var.values::<f64, _>(extents.as_slice())
            .map_err(|e| NetCdfError::ReadFailed(e.to_string()))?;

        Ok(Variable { data, dims: slice_dims })
    }

    /// 获取全局属性
    pub fn global_attribute(&self, name: &str) -> Result<String, NetCdfError> {
        let attr = self
            .file
            .attribute(name)
            .ok_or_else(|| NetCdfError::AttributeNotFound(name.to_string()))?;
        
        match attr.value()? {
            netcdf::AttrValue::Str(s) => Ok(s.to_string()),
            other => Ok(format!("{:?}", other)),
        }
    }
}

/// 无 NetCDF 支持时的 CLI 驱动实现
#[cfg(not(feature = "netcdf"))]
pub struct NetCdfDriver {
    path: std::path::PathBuf,
    header: CliHeader,
}

#[cfg(not(feature = "netcdf"))]
#[derive(Debug, Clone, Default)]
struct CliHeader {
    dimensions: Vec<Dimension>,
    variables: Vec<VariableInfo>,
    global_attrs: Vec<(String, String)>,
}

#[cfg(not(feature = "netcdf"))]
impl NetCdfDriver {
    /// 打开 NetCDF 文件 (CLI 驱动)
    pub fn open(path: impl AsRef<Path>) -> Result<Self, NetCdfError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(NetCdfError::FileNotFound(path.display().to_string()));
        }

        let header = cli_read_header(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            header,
        })
    }

    /// 获取所有维度
    pub fn dimensions(&self) -> Result<Vec<Dimension>, NetCdfError> {
        Ok(self.header.dimensions.clone())
    }

    /// 获取维度
    pub fn dimension(&self, name: &str) -> Result<Dimension, NetCdfError> {
        self.header
            .dimensions
            .iter()
            .find(|d| d.name == name)
            .cloned()
            .ok_or_else(|| NetCdfError::DimensionNotFound(name.to_string()))
    }

    /// 获取所有变量信息
    pub fn variables(&self) -> Result<Vec<VariableInfo>, NetCdfError> {
        Ok(self.header.variables.clone())
    }

    /// 读取变量
    pub fn read_variable(&self, name: &str) -> Result<Variable, NetCdfError> {
        let var_info = self
            .header
            .variables
            .iter()
            .find(|v| v.name == name)
            .cloned()
            .ok_or_else(|| NetCdfError::VariableNotFound(name.to_string()))?;

        let dims = var_info
            .dimensions
            .iter()
            .map(|d| {
                self.dimension(d)
                    .map(|dim| dim.len)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let data = cli_read_variable_data(&self.path, name)?;

        Ok(Variable { data, dims })
    }

    /// 读取变量的一个时间切片
    pub fn read_variable_slice(
        &self,
        name: &str,
        time_idx: usize,
    ) -> Result<Variable, NetCdfError> {
        let var = self.read_variable(name)?;
        if var.dims.is_empty() {
            return Err(NetCdfError::ReadFailed("Variable has no dimensions".to_string()));
        }

        let time_len = var.dims[0];
        if time_idx >= time_len {
            return Err(NetCdfError::ReadFailed("time index out of range".to_string()));
        }

        let slice_dims = var.dims[1..].to_vec();
        let slice_size: usize = slice_dims.iter().product();
        let offset = time_idx * slice_size;
        let end = offset + slice_size;
        if end > var.data.len() {
            return Err(NetCdfError::ReadFailed("slice out of range".to_string()));
        }

        Ok(Variable {
            data: var.data[offset..end].to_vec(),
            dims: slice_dims,
        })
    }

    /// 获取全局属性
    pub fn global_attribute(&self, name: &str) -> Result<String, NetCdfError> {
        self.header
            .global_attrs
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| NetCdfError::AttributeNotFound(name.to_string()))
    }
}

#[cfg(not(feature = "netcdf"))]
fn cli_read_header(path: &Path) -> Result<CliHeader, NetCdfError> {
    let output = Command::new(ncdump_bin())
        .arg("-h")
        .arg(path)
        .output()
        .map_err(|_| NetCdfError::NotAvailable)?;

    if !output.status.success() {
        return Err(NetCdfError::OpenFailed(String::from_utf8_lossy(&output.stderr).to_string()));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    parse_ncdump_header(&text)
}

#[cfg(not(feature = "netcdf"))]
fn parse_ncdump_header(text: &str) -> Result<CliHeader, NetCdfError> {
    let mut header = CliHeader::default();
    let mut in_dimensions = false;
    let mut in_variables = false;
    for line in text.lines() {
        let raw = line.trim();
        if raw.starts_with("dimensions:") {
            in_dimensions = true;
            in_variables = false;
            continue;
        }
        if raw.starts_with("variables:") {
            in_dimensions = false;
            in_variables = true;
            continue;
        }
        if raw.starts_with("data:") {
            break;
        }

        if in_dimensions {
            if raw.is_empty() { continue; }
            if let Some(eq) = raw.find('=') {
                let name = raw[..eq].trim().to_string();
                let rhs = raw[eq + 1..].trim();
                let (len, unlimited) = if rhs.starts_with("UNLIMITED") {
                    let current = rhs.split('(')
                        .nth(1)
                        .and_then(|s| s.split_whitespace().next())
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(0);
                    (current, true)
                } else {
                    let len = rhs.split(';').next().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0);
                    (len, false)
                };
                header.dimensions.push(Dimension { name, len, is_unlimited: unlimited });
            }
            continue;
        }

        if in_variables {
            if raw.is_empty() { continue; }

            if raw.contains('(') && raw.ends_with(';') && !raw.contains(":") {
                // 变量声明行：dtype name(dim, dim, ...)
                let cleaned = raw.trim_end_matches(';').trim();
                if let Some(space) = cleaned.find(' ') {
                    let dtype = cleaned[..space].trim().to_string();
                    let rest = cleaned[space + 1..].trim();
                    if let Some(lparen) = rest.find('(') {
                        let name = rest[..lparen].trim().to_string();
                        let dims_str = rest[lparen + 1..].trim_end_matches(')');
                        let dims = dims_str
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect::<Vec<_>>();
                        header.variables.push(VariableInfo {
                            name,
                            dimensions: dims,
                            dtype,
                            standard_name: None,
                            long_name: None,
                            units: None,
                        });
                    }
                }
            } else if raw.starts_with(":") {
                // 全局属性
                if let Some(eq) = raw.find('=') {
                    let key = raw[1..eq].trim().to_string();
                    let val = raw[eq + 1..].trim().trim_end_matches(';').trim();
                    header.global_attrs.push((key, trim_quotes(val)));
                }
            } else if raw.contains(":") {
                // 变量属性
                let mut parts = raw.splitn(2, ':');
                let var = parts.next().unwrap_or("").trim();
                let rest = parts.next().unwrap_or("").trim();
                if let Some(eq) = rest.find('=') {
                    let key = rest[..eq].trim();
                    let val = rest[eq + 1..].trim().trim_end_matches(';');
                    if let Some(info) = header.variables.iter_mut().find(|v| v.name == var) {
                        match key {
                            "standard_name" => info.standard_name = Some(trim_quotes(val)),
                            "long_name" => info.long_name = Some(trim_quotes(val)),
                            "units" => info.units = Some(trim_quotes(val)),
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    Ok(header)
}

#[cfg(not(feature = "netcdf"))]
fn cli_read_variable_data(path: &Path, name: &str) -> Result<Vec<f64>, NetCdfError> {
    let output = Command::new(ncdump_bin())
        .arg("-v")
        .arg(name)
        .arg(path)
        .output()
        .map_err(|_| NetCdfError::NotAvailable)?;

    if !output.status.success() {
        return Err(NetCdfError::ReadFailed(String::from_utf8_lossy(&output.stderr).to_string()));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    Ok(parse_numeric_values(&text))
}

#[cfg(not(feature = "netcdf"))]
fn parse_numeric_values(text: &str) -> Vec<f64> {
    let mut values = Vec::new();
    let mut in_data = false;
    for line in text.lines() {
        let raw = line.trim();
        if raw.starts_with("data:") {
            in_data = true;
            continue;
        }
        if !in_data {
            continue;
        }
        let cleaned = raw.replace(',', " ");
        for token in cleaned.split_whitespace() {
            let lower = token.to_ascii_lowercase();
            let parsed = match lower.as_str() {
                "nan" => Some(f64::NAN),
                "inf" | "infinity" => Some(f64::INFINITY),
                "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
                _ => token.parse::<f64>().ok(),
            };
            if let Some(v) = parsed {
                values.push(v);
            }
        }
    }
    values
}

#[cfg(not(feature = "netcdf"))]
fn trim_quotes(val: &str) -> String {
    val.trim_matches('"').to_string()
}

#[cfg(not(feature = "netcdf"))]
fn ncdump_bin() -> String {
    std::env::var("NCDUMP_BIN").unwrap_or_else(|_| "ncdump".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_linear_index() {
        let var = Variable {
            data: vec![0.0; 24],
            dims: vec![2, 3, 4],
        };

        // 测试索引计算
        assert_eq!(var.linear_index(&[0, 0, 0]), Some(0));
        assert_eq!(var.linear_index(&[0, 0, 1]), Some(1));
        assert_eq!(var.linear_index(&[0, 1, 0]), Some(4));
        assert_eq!(var.linear_index(&[1, 0, 0]), Some(12));
    }

    #[test]
    fn test_variable_get() {
        let var = Variable {
            data: (0..24).map(|i| i as f64).collect(),
            dims: vec![2, 3, 4],
        };

        assert_eq!(var.get(&[0, 0, 0]), Some(0.0));
        assert_eq!(var.get(&[1, 2, 3]), Some(23.0));
        assert_eq!(var.get(&[2, 0, 0]), None); // 越界
    }
}
