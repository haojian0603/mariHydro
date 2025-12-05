// crates/mh_io/src/drivers/netcdf/mod.rs

//! NetCDF 驱动模块
//!
//! 提供读取 NetCDF 气象数据的功能。
//!
//! # 功能
//!
//! - 读取维度信息
//! - 读取变量数据
//! - 支持 CF 约定
//! - 时间序列支持
//!
//! # 依赖
//!
//! 需要启用 `netcdf` feature 并安装 NetCDF 库。

mod driver;
mod error;

pub use driver::*;
pub use error::*;
