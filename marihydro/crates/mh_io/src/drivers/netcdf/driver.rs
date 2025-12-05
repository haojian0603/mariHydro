// crates/mh_io/src/drivers/netcdf/driver.rs

//! NetCDF 驱动实现

use super::error::NetCdfError;
use std::path::Path;

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

/// 无 NetCDF 支持时的占位实现
#[cfg(not(feature = "netcdf"))]
pub struct NetCdfDriver;

#[cfg(not(feature = "netcdf"))]
impl NetCdfDriver {
    /// 打开 NetCDF 文件 (无 NetCDF 支持)
    pub fn open(_path: impl AsRef<Path>) -> Result<Self, NetCdfError> {
        Err(NetCdfError::NotAvailable)
    }
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
