// crates/mh_io/src/drivers/netcdf/driver.rs

//! NetCDF 驱动实现
//!
//! IO_SOURCE: NetCDF 经典数据模型，以及 `ncdump -h` / `ncdump -v <var>` 文本输出格式约定。
//! IO_SCOPE: 原生后端负责真实 NetCDF 变量读取；CLI 回退路径仅接受结构完整、数值 token 可完整解释的 `ncdump` 输出。遇到缺失 `data:` 段、坏 token 或维度长度非法时直接报错，不做部分解析。

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
    fn linear_index(&self, indices: &[usize]) -> Result<usize, NetCdfError> {
        if indices.len() != self.dims.len() {
            return Err(NetCdfError::InvalidIndices {
                indices: indices.to_vec(),
                dims: self.dims.clone(),
            });
        }

        let mut idx = 0;
        let mut stride = 1;
        for (i, &dim_size) in self.dims.iter().enumerate().rev() {
            if indices[i] >= dim_size {
                return Err(NetCdfError::InvalidIndices {
                    indices: indices.to_vec(),
                    dims: self.dims.clone(),
                });
            }
            idx += indices[i] * stride;
            stride *= dim_size;
        }
        Ok(idx)
    }

    /// 获取指定索引的值
    pub fn get(&self, indices: &[usize]) -> Result<f64, NetCdfError> {
        let idx = self.linear_index(indices)?;
        Ok(self.data[idx])
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

    /// 获取单个变量信息
    pub fn variable_info(&self, name: &str) -> Result<VariableInfo, NetCdfError> {
        self.variables()?
            .into_iter()
            .find(|info| info.name == name)
            .ok_or_else(|| NetCdfError::VariableNotFound(name.to_string()))
    }

    /// 检查变量是否存在
    pub fn has_variable(&self, name: &str) -> bool {
        self.variable_info(name).is_ok()
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
                let dims: Vec<String> = v
                    .dimensions()
                    .iter()
                    .map(|d| d.name().to_string())
                    .collect();

                VariableInfo {
                    name: v.name().to_string(),
                    dimensions: dims,
                    dtype: format!("{:?}", v.vartype()),
                    standard_name: v
                        .attribute("standard_name")
                        .and_then(|a| a.value().ok())
                        .and_then(|v| match v {
                            netcdf::AttrValue::Str(s) => Some(s.to_string()),
                            _ => None,
                        }),
                    long_name: v
                        .attribute("long_name")
                        .and_then(|a| a.value().ok())
                        .and_then(|v| match v {
                            netcdf::AttrValue::Str(s) => Some(s.to_string()),
                            _ => None,
                        }),
                    units: v.attribute("units").and_then(|a| a.value().ok()).and_then(
                        |v| match v {
                            netcdf::AttrValue::Str(s) => Some(s.to_string()),
                            _ => None,
                        },
                    ),
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
        let data: Vec<f64> = var
            .values::<f64, _>(..)
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
            return Err(NetCdfError::ReadFailed(
                "Variable has no dimensions".to_string(),
            ));
        }

        if time_idx >= dims[0] {
            return Err(NetCdfError::ReadFailed(
                "time index out of range".to_string(),
            ));
        }

        // 假设第一个维度是时间
        let slice_dims: Vec<usize> = dims[1..].to_vec();

        // 构建索引范围
        let extents: Vec<_> = std::iter::once(time_idx..time_idx + 1)
            .chain(dims[1..].iter().map(|&d| 0..d))
            .collect();

        let data: Vec<f64> = var
            .values::<f64, _>(extents.as_slice())
            .map_err(|e| NetCdfError::ReadFailed(e.to_string()))?;

        Ok(Variable {
            data,
            dims: slice_dims,
        })
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
#[derive(Debug, Clone)]
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

    /// 获取单个变量信息
    pub fn variable_info(&self, name: &str) -> Result<VariableInfo, NetCdfError> {
        self.header
            .variables
            .iter()
            .find(|info| info.name == name)
            .cloned()
            .ok_or_else(|| NetCdfError::VariableNotFound(name.to_string()))
    }

    /// 检查变量是否存在
    pub fn has_variable(&self, name: &str) -> bool {
        self.variable_info(name).is_ok()
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
            .map(|d| self.dimension(d).map(|dim| dim.len))
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
            return Err(NetCdfError::ReadFailed(
                "Variable has no dimensions".to_string(),
            ));
        }

        let time_len = var.dims[0];
        if time_idx >= time_len {
            return Err(NetCdfError::ReadFailed(
                "time index out of range".to_string(),
            ));
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
fn format_cli_failure(tool: &str, stage: &str, exit_code: Option<i32>, stderr: &[u8]) -> String {
    let status = match exit_code {
        Some(code) => format!("退出码 {code}"),
        None => "进程被终止".to_string(),
    };
    let stderr = String::from_utf8_lossy(stderr).trim().to_string();
    if stderr.is_empty() {
        format!("{tool} 在{stage}阶段失败（{status}，stderr 为空）")
    } else {
        format!("{tool} 在{stage}阶段失败（{status}）：{stderr}")
    }
}

#[cfg(not(feature = "netcdf"))]
fn cli_read_header(path: &Path) -> Result<CliHeader, NetCdfError> {
    let tool = ncdump_bin();
    let output = Command::new(&tool)
        .arg("-h")
        .arg(path)
        .output()
        .map_err(|error| NetCdfError::NotAvailable {
            tool: tool.clone(),
            detail: error.to_string(),
        })?;

    if !output.status.success() {
        return Err(NetCdfError::OpenFailed(format_cli_failure(
            &tool,
            "读取头部",
            output.status.code(),
            &output.stderr,
        )));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    parse_ncdump_header(&text)
}

#[cfg(not(feature = "netcdf"))]
fn parse_ncdump_header(text: &str) -> Result<CliHeader, NetCdfError> {
    let mut dimensions = Vec::new();
    let mut variables = Vec::new();
    let mut global_attrs = Vec::new();
    let mut in_dimensions = false;
    let mut in_variables = false;
    let mut saw_variables_section = false;
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
            saw_variables_section = true;
            continue;
        }
        if raw.starts_with("data:") {
            break;
        }

        if in_dimensions {
            if raw.is_empty() {
                continue;
            }
            if let Some(eq) = raw.find('=') {
                let name = raw[..eq].trim().to_string();
                let rhs = raw[eq + 1..].trim();
                let (len, unlimited) = if rhs.starts_with("UNLIMITED") {
                    let current_raw = rhs
                        .split('(')
                        .nth(1)
                        .and_then(|s| s.split_whitespace().next())
                        .ok_or_else(|| {
                            NetCdfError::ReadFailed(format!(
                                "ncdump 维度声明缺少 UNLIMITED 当前长度: {rhs}"
                            ))
                        })?;
                    let current = current_raw.parse::<usize>().map_err(|_| {
                        NetCdfError::ReadFailed(format!(
                            "ncdump UNLIMITED 当前长度无法解析为整数: {current_raw}"
                        ))
                    })?;
                    (current, true)
                } else {
                    let len_raw = rhs
                        .split(';')
                        .next()
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            NetCdfError::ReadFailed(format!("ncdump 维度声明缺少长度: {rhs}"))
                        })?;
                    let len = len_raw.parse::<usize>().map_err(|_| {
                        NetCdfError::ReadFailed(format!("ncdump 维度长度无法解析为整数: {len_raw}"))
                    })?;
                    (len, false)
                };
                dimensions.push(Dimension {
                    name,
                    len,
                    is_unlimited: unlimited,
                });
            }
            continue;
        }

        if in_variables {
            if raw.is_empty() {
                continue;
            }

            if raw.contains('(') && raw.ends_with(';') && !raw.contains(":") {
                let cleaned = raw.trim_end_matches(';').trim();
                let space = cleaned.find(' ').ok_or_else(|| {
                    NetCdfError::ReadFailed(format!("ncdump 变量声明缺少类型与名称分隔: {cleaned}"))
                })?;
                let dtype = cleaned[..space].trim().to_string();
                let rest = cleaned[space + 1..].trim();
                let lparen = rest.find('(').ok_or_else(|| {
                    NetCdfError::ReadFailed(format!("ncdump 变量声明缺少维度列表起始符号: {rest}"))
                })?;
                let name = rest[..lparen].trim().to_string();
                if name.is_empty() {
                    return Err(NetCdfError::ReadFailed(format!(
                        "ncdump 变量声明缺少变量名: {cleaned}"
                    )));
                }
                let dims_str = rest[lparen + 1..].trim_end_matches(')');
                let dims = dims_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>();
                variables.push(VariableInfo {
                    name,
                    dimensions: dims,
                    dtype,
                    standard_name: None,
                    long_name: None,
                    units: None,
                });
            } else if raw.starts_with(":") {
                let eq = raw.find('=').ok_or_else(|| {
                    NetCdfError::ReadFailed(format!("ncdump 全局属性缺少赋值符号: {raw}"))
                })?;
                let key = raw[1..eq].trim().to_string();
                if key.is_empty() {
                    return Err(NetCdfError::ReadFailed(format!(
                        "ncdump 全局属性缺少键名: {raw}"
                    )));
                }
                let val = raw[eq + 1..].trim().trim_end_matches(';').trim();
                global_attrs.push((key, trim_quotes(val)));
            } else if raw.contains(":") {
                let (var, rest) = raw.split_once(':').ok_or_else(|| {
                    NetCdfError::ReadFailed(format!("ncdump 变量属性缺少变量名前缀: {raw}"))
                })?;
                let var = var.trim();
                if var.is_empty() {
                    return Err(NetCdfError::ReadFailed(format!(
                        "ncdump 变量属性缺少变量名: {raw}"
                    )));
                }
                let rest = rest.trim();
                let eq = rest.find('=').ok_or_else(|| {
                    NetCdfError::ReadFailed(format!("ncdump 变量属性缺少赋值符号: {raw}"))
                })?;
                let key = rest[..eq].trim();
                if key.is_empty() {
                    return Err(NetCdfError::ReadFailed(format!(
                        "ncdump 变量属性缺少属性名: {raw}"
                    )));
                }
                let val = rest[eq + 1..].trim().trim_end_matches(';');
                if let Some(info) = variables.iter_mut().find(|v| v.name == var) {
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

    if !saw_variables_section {
        return Err(NetCdfError::ReadFailed(
            "ncdump 头缺少 variables: 段".to_string(),
        ));
    }
    if variables.is_empty() {
        return Err(NetCdfError::ReadFailed(
            "ncdump 头未声明任何变量".to_string(),
        ));
    }

    Ok(CliHeader {
        dimensions,
        variables,
        global_attrs,
    })
}

#[cfg(not(feature = "netcdf"))]
fn cli_read_variable_data(path: &Path, name: &str) -> Result<Vec<f64>, NetCdfError> {
    let tool = ncdump_bin();
    let output = Command::new(&tool)
        .arg("-v")
        .arg(name)
        .arg(path)
        .output()
        .map_err(|error| NetCdfError::NotAvailable {
            tool: tool.clone(),
            detail: error.to_string(),
        })?;

    if !output.status.success() {
        return Err(NetCdfError::ReadFailed(format_cli_failure(
            &tool,
            format!("读取变量 {name}").as_str(),
            output.status.code(),
            &output.stderr,
        )));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    parse_numeric_values(&text)
}

#[cfg(not(feature = "netcdf"))]
fn parse_numeric_values(text: &str) -> Result<Vec<f64>, NetCdfError> {
    let mut values = Vec::new();
    let mut in_data = false;
    let mut saw_data_section = false;
    for line in text.lines() {
        let raw = line.trim();
        if raw.starts_with("data:") {
            in_data = true;
            saw_data_section = true;
            continue;
        }
        if !in_data {
            continue;
        }
        if raw == "}" {
            break;
        }

        let payload = if let Some((_, rhs)) = raw.split_once('=') {
            rhs.trim()
        } else {
            raw
        };
        let cleaned = payload
            .trim_end_matches(';')
            .replace(',', " ")
            .trim()
            .to_string();
        if cleaned.is_empty() {
            continue;
        }
        for token in cleaned.split_whitespace() {
            let lower = token.to_ascii_lowercase();
            let value = match lower.as_str() {
                "nan" => f64::NAN,
                "inf" | "infinity" => f64::INFINITY,
                "-inf" | "-infinity" => f64::NEG_INFINITY,
                _ => token.parse::<f64>().map_err(|_| {
                    NetCdfError::ReadFailed(format!("ncdump 数值载荷无法解析: {token}"))
                })?,
            };
            values.push(value);
        }
    }

    if !saw_data_section {
        return Err(NetCdfError::ReadFailed(
            "ncdump 输出缺少 data 段".to_string(),
        ));
    }

    Ok(values)
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
        assert_eq!(var.linear_index(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(var.linear_index(&[0, 0, 1]).unwrap(), 1);
        assert_eq!(var.linear_index(&[0, 1, 0]).unwrap(), 4);
        assert_eq!(var.linear_index(&[1, 0, 0]).unwrap(), 12);
    }

    #[test]
    fn test_variable_get() {
        let var = Variable {
            data: (0..24).map(|i| i as f64).collect(),
            dims: vec![2, 3, 4],
        };

        assert_eq!(var.get(&[0, 0, 0]).unwrap(), 0.0);
        assert_eq!(var.get(&[1, 2, 3]).unwrap(), 23.0);
        let err = var.get(&[2, 0, 0]).expect_err("越界索引必须失败");
        assert!(matches!(err, NetCdfError::InvalidIndices { .. }));
    }

    #[cfg(not(feature = "netcdf"))]
    #[test]
    fn test_parse_ncdump_header_rejects_invalid_dimension_length() {
        let text = r#"
netcdf sample {
dimensions:
    lon = abc ;
variables:
    float h(lon);
data:
}
"#;
        assert!(parse_ncdump_header(text).is_err());
    }

    #[cfg(not(feature = "netcdf"))]
    #[test]
    fn test_parse_ncdump_header_rejects_malformed_variable_declaration() {
        let text = r#"
netcdf sample {
dimensions:
    lon = 3 ;
variables:
    float(lon);
data:
}
"#;
        assert!(parse_ncdump_header(text).is_err());
    }

    #[cfg(not(feature = "netcdf"))]
    #[test]
    fn test_parse_ncdump_header_rejects_malformed_variable_attribute() {
        let text = r#"
netcdf sample {
dimensions:
    lon = 3 ;
variables:
    float h(lon);
    h: units "m";
data:
}
"#;
        assert!(parse_ncdump_header(text).is_err());
    }

    #[cfg(not(feature = "netcdf"))]
    #[test]
    fn test_parse_ncdump_header_rejects_malformed_global_attribute() {
        let text = r#"
netcdf sample {
dimensions:
    lon = 3 ;
variables:
    float h(lon);
    :title "demo";
data:
}
"#;
        assert!(parse_ncdump_header(text).is_err());
    }

    #[cfg(not(feature = "netcdf"))]
    #[test]
    fn test_parse_ncdump_header_rejects_missing_variables_section() {
        let text = r#"
netcdf sample {
dimensions:
    lon = 3 ;
data:
}
"#;
        assert!(parse_ncdump_header(text).is_err());
    }

    #[cfg(not(feature = "netcdf"))]
    #[test]
    fn test_parse_ncdump_header_rejects_empty_variables_section() {
        let text = r#"
netcdf sample {
dimensions:
    lon = 3 ;
variables:

data:
}
"#;
        assert!(parse_ncdump_header(text).is_err());
    }

    #[cfg(not(feature = "netcdf"))]
    #[test]
    fn test_format_cli_failure_preserves_tool_and_stage() {
        let message = format_cli_failure("ncdump-custom", "读取头部", Some(3), b"missing variable");
        assert!(message.contains("ncdump-custom"));
        assert!(message.contains("读取头部"));
        assert!(message.contains("退出码 3"));
        assert!(message.contains("missing variable"));
    }

    #[cfg(not(feature = "netcdf"))]
    #[test]
    fn test_parse_numeric_values_supports_assignment_lines() {
        let text = r#"
netcdf sample {
data:
    h =
        1, 2, nan, -inf ;
}
"#;
        let values = parse_numeric_values(text).unwrap();
        assert_eq!(values.len(), 4);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 2.0);
        assert!(values[2].is_nan());
        assert_eq!(values[3], f64::NEG_INFINITY);
    }

    #[cfg(not(feature = "netcdf"))]
    #[test]
    fn test_parse_numeric_values_rejects_invalid_payload_token() {
        let text = r#"
netcdf sample {
data:
    h =
        1, bad, 3 ;
}
"#;
        assert!(parse_numeric_values(text).is_err());
    }

    #[cfg(not(feature = "netcdf"))]
    #[test]
    fn test_parse_numeric_values_rejects_missing_data_section() {
        let text = r#"
netcdf sample {
variables:
    float h(lon);
}
"#;
        assert!(parse_numeric_values(text).is_err());
    }
}
