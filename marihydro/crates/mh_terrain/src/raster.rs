// marihydro\crates\mh_terrain\src/raster.rs

//! 栅格数据管理
//!
//! 提供栅格地形数据的存储和访问。

use mh_foundation::error::MhResult;

/// 栅格数据
#[derive(Debug, Clone)]
pub struct RasterData {
    /// 数据
    pub data: Vec<f64>,
    /// 宽度
    pub width: usize,
    /// 高度
    pub height: usize,
    /// 无数据值
    pub nodata: f64,
}

impl RasterData {
    /// 创建新的栅格数据
    pub fn new(width: usize, height: usize, nodata: f64) -> Self {
        Self {
            data: vec![nodata; width * height],
            width,
            height,
            nodata,
        }
    }

    /// 从数据创建
    pub fn from_data(data: Vec<f64>, width: usize, height: usize, nodata: f64) -> MhResult<Self> {
        if data.len() != width * height {
            return Err(mh_foundation::error::MhError::size_mismatch(
                "raster data",
                width * height,
                data.len(),
            ));
        }
        Ok(Self { data, width, height, nodata })
    }

    /// 获取像素值
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Option<f64> {
        if x < self.width && y < self.height {
            Some(self.data[y * self.width + x])
        } else {
            None
        }
    }

    /// 设置像素值
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: f64) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = value;
        }
    }

    /// 判断是否为无数据
    #[inline]
    pub fn is_nodata(&self, value: f64) -> bool {
        value.is_nan() || (self.nodata.is_finite() && (value - self.nodata).abs() < 1e-10)
    }
}
