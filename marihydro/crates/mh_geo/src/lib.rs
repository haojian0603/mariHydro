// marihydro\crates\mh_geo\src/lib.rs
//! MariHydro 地理空间处理模块
//!
//! 提供坐标参考系统 (CRS)、投影转换、空间索引等功能。
//!
//! # 模块
//!
//! - `crs`: 坐标参考系统定义和解析
//! - `geometry`: 几何类型 (Point2D, Point3D)
//! - `projection`: 投影转换 (UTM, Web Mercator, 高斯-克吕格)
//! - `transform`: 坐标转换器和仿射变换
//! - `spatial_index`: 基于 R-tree 的空间索引
//!
//! # 示例
//!
//! ```
//! use mh_geo::prelude::*;
//!
//! // 创建 WGS84 CRS
//! let wgs84 = Crs::wgs84();
//! assert!(wgs84.is_geographic());
//!
//! // WGS84 -> UTM 投影转换
//! let proj = Projection::from_epsg(4326, 32650).unwrap();
//! let (x, y) = proj.forward(116.0, 40.0).unwrap();
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod crs;
pub mod geometry;
pub mod projection;
pub mod spatial_index;
pub mod transform;

/// 预导入模块
pub mod prelude {
    pub use crate::crs::{Crs, CrsDefinition, CrsStrategy};
    pub use crate::geometry::{Point2D, Point3D};
    pub use crate::projection::{Projection, ProjectionType};
    pub use crate::spatial_index::{BoundingBox, SpatialIndex};
    pub use crate::transform::{AffineTransform, GeoTransformer};
}

// 重导出常用类型
pub use crs::{Crs, CrsDefinition, CrsStrategy};
pub use geometry::{Point2D, Point3D};
pub use projection::{Projection, ProjectionType};
pub use spatial_index::BoundingBox;
pub use spatial_index::SpatialIndex;
pub use transform::{AffineTransform, GeoTransformer};
