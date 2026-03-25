// crates/mh_io/src/drivers/netcdf/mod.rs

//! NetCDF 驱动模块
//!
//! 提供读取 NetCDF 数值栅格与元数据的通用能力。
//!
//! IO_SOURCE: NetCDF 经典数据模型与 CF 元数据约定；运行时优先使用原生 `netcdf` 后端，未启用 feature 时退回到 `ncdump` CLI 只读路径。
//! IO_SCOPE: 当前模块只接受可完整解释的维度、变量、数值载荷和 CF 时间字符串。任一字段、token 或时间分量无法解释时直接报错，不做部分解析成功。
//!
//! # 功能
//!
//! - 读取维度信息
//! - 读取变量数据
//! - 支持 CF 约定
//! - 时间序列支持
//! - CF 时间格式解析
//!
//! # 依赖
//!
//! 需要启用 `netcdf` feature 并安装 NetCDF 库。

mod driver;
mod error;
pub mod time;

pub use driver::*;
pub use error::*;
pub use time::{CfCalendar, CfTimeError, CfTimeUnits, DateTime, TimeUnit};
