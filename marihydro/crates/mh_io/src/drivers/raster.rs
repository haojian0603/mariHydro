// crates/mh_io/src/drivers/raster.rs

//! 栅格驱动抽象接口


/// 栅格驱动 trait
pub trait RasterDriver {
    /// 获取栅格元数据
    fn metadata(&self) -> &super::gdal::RasterMetadata;
    
    /// 读取波段数据
    fn read_band(&self, band: usize) -> Result<super::gdal::RasterBand, super::gdal::GdalError>;
}
