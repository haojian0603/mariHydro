// marihydro\crates\mh_mesh\src/algorithms/mod.rs

//! 网格算法模块
//!
//! 提供网格细化、光顺、三角化、图着色等算法。

pub mod boundary;
pub mod coloring;
pub mod quality_metrics;
pub mod refine;
pub mod smooth;
pub mod triangulate;
pub mod validation;

// 重导出常用类型
pub use boundary::{BoundaryExtractor, BoundaryLoop};
pub use coloring::{ColoringResult, GreedyColoring};
pub use quality_metrics::{CellQuality, MeshQualityStats, QualityEvaluator};
pub use refine::{RefineConfig, Refiner};
pub use smooth::{SmoothConfig, SmoothMethod, Smoother};
pub use triangulate::{TriangulateConfig, TriangulateError, Triangulator};
pub use validation::{MeshValidator, ValidationConfig, ValidationResult};
