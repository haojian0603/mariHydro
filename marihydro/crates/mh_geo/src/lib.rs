//! MariHydro 地理空间处理模块
//!
//! 提供坐标参考系统 (CRS)、投影转换、空间索引等功能。
//!
//! # 模块
//!
//! - `ellipsoid`: 椭球体定义 (WGS84, CGCS2000, GRS80 等)
//! - `crs`: 坐标参考系统定义和解析
//! - `geometry`: 几何类型 (Point2D, Point3D) 和地理距离计算
//! - `projection`: 高精度投影转换 (UTM, Web Mercator, 高斯-克吕格)
//! - `transform`: 坐标转换器和仿射变换
//! - `spatial_index`: 基于 R-tree 的空间索引
//!
//! # 精度特点
//!
//! - UTM/高斯-克吕格投影使用 Karney (2011) 算法，精度达亚毫米级
//! - 支持多种椭球体参数（WGS84、CGCS2000 等）
//! - 纯 Rust 实现，零外部 C 依赖
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
//! // WGS84 -> UTM 投影转换（高精度）
//! let proj = Projection::from_epsg(4326, 32650).unwrap();
//! let (x, y) = proj.forward(116.0, 40.0).unwrap();
//!
//! // 地理距离计算
//! let beijing = Point2D::from_lonlat(116.4, 39.9);
//! let shanghai = Point2D::from_lonlat(121.5, 31.2);
//! let distance_km = beijing.geodesic_distance_to(&shanghai) / 1000.0;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod crs;
pub mod ellipsoid;
pub mod geometry;
pub mod projection;
pub mod spatial_index;
pub mod transform;

/// 预导入模块
pub mod prelude {
    pub use crate::crs::{Crs, CrsDefinition, CrsStrategy};
    pub use crate::ellipsoid::Ellipsoid;
    pub use crate::geometry::{Point2D, Point3D, EARTH_MEAN_RADIUS};
    pub use crate::projection::{
        FastProjection, MapProjection, Projection, ProjectionType, TransverseMercatorParams,
    };
    pub use crate::spatial_index::{BoundingBox, SpatialIndex};
    pub use crate::transform::{AffineTransform, GeoTransformer};
}

// 重导出常用类型
pub use crs::{Crs, CrsDefinition, CrsStrategy};
pub use ellipsoid::Ellipsoid;
pub use geometry::{Point2D, Point3D};
pub use projection::{Projection, ProjectionType};
pub use spatial_index::{BoundingBox, SpatialIndex};
pub use transform::{AffineTransform, GeoTransformer};