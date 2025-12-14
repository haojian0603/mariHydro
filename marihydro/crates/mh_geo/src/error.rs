// marihydro\crates\mh_geo\src\error.rs
//! 地理空间处理错误类型
//!
//! 包含投影转换、坐标系统、几何计算相关的错误。
//! 所有错误可转换为 `mh_foundation::MhError` 向上传播。
//! 
//! # 错误分类
//!
//! - **配置错误**：EPSG代码不支持、CRS定义无效
//! - **验证错误**：坐标越界、UTM带号无效
//! - **计算错误**：投影转换失败、迭代不收敛
//! - **几何错误**：空间索引操作失败
//! - **基础错误**：IO错误（来自Foundation）

use thiserror::Error;
use mh_foundation::MhError;

/// Geo 模块结果类型
pub type GeoResult<T> = Result<T, GeoError>;

/// 地理空间处理错误
#[derive(Error, Debug)]
pub enum GeoError {
    /// 不支持的 EPSG 代码
    #[error("不支持的 EPSG 代码: {code}")]
    UnsupportedEpsg {
        /// 请求的 EPSG 代码
        code: u32,
        /// 支持的代码范围说明
        supported: &'static str,
    },

    /// 坐标超出有效范围
    #[error("{coord_type} 超出范围: {value:.6} (允许范围: {min} 到 {max})")]
    CoordinateOutOfRange {
        /// 坐标类型（如"纬度"、"经度"、"UTM带号"）
        coord_type: &'static str,
        /// 实际值
        value: f64,
        /// 最小允许值
        min: f64,
        /// 最大允许值
        max: f64,
    },

    /// UTM 带号无效
    #[error("无效的 UTM 带号: {zone} (允许范围: 1-60)")]
    InvalidUtmZone {
        /// 无效的带号
        zone: u8,
    },

    /// 高斯-克吕格带号无效
    #[error("无效的高斯-克吕格带号: {zone} (允许范围: {min_zone}-{max_zone})")]
    InvalidGaussKrugerZone {
        /// 无效的带号
        zone: u8,
        /// 最小允许带号
        min_zone: u8,
        /// 最大允许带号
        max_zone: u8,
    },

    /// 投影转换失败
    #[error("投影转换失败: {operation}")]
    ProjectionFailed {
        /// 操作类型（如"正向投影"、"逆向投影"）
        operation: &'static str,
        /// 错误详情
        message: String,
    },

    /// CRS 定义解析失败
    #[error("CRS 定义解析失败: {definition}")]
    CrsParseFailed {
        /// 失败的定义字符串
        definition: String,
        /// 失败原因
        reason: String,
    },

    /// 几何计算失败
    #[error("{operation} 计算失败: {message}")]
    GeometryComputationFailed {
        /// 计算类型（如"Vincenty距离"、"方位角"）
        operation: &'static str,
        /// 失败原因
        message: String,
    },

    /// Vincenty 迭代不收敛
    #[error("Vincenty 公式迭代不收敛")]
    VincentyNotConverged,

    /// 空间索引操作失败
    #[error("空间索引错误: {operation} - {message}")]
    SpatialIndexError {
        /// 操作类型
        operation: &'static str,
        /// 失败原因
        message: String,
    },

    /// 仿射变换矩阵奇异（不可逆）
    #[error("仿射变换矩阵奇异（行列式接近零）")]
    SingularTransform,

    /// 收敛角计算失败
    #[error("收敛角计算失败: {message}")]
    ConvergenceAngleError {
        /// 失败原因
        message: String,
    },

    /// 基础 IO 错误（向下聚合）
    #[error("基础层错误: {0}")]
    Foundation(#[from] MhError),
}

// ============================================================================
// 转换实现
// ============================================================================

impl From<GeoError> for MhError {
    fn from(err: GeoError) -> Self {
        match err {
            GeoError::UnsupportedEpsg { code, supported } => {
                MhError::invalid_input(format!(
                    "不支持的EPSG代码 {code}。支持的代码: {supported}"
                ))
            }
            GeoError::CoordinateOutOfRange { coord_type, value, min, max } => {
                MhError::invalid_input(format!(
                    "{coord_type} 超出范围: {value:.6} (允许范围: {min} 到 {max})"
                ))
            }
            GeoError::InvalidUtmZone { zone } => {
                MhError::invalid_input(format!("无效的UTM带号 {zone} (允许范围: 1-60)"))
            }
            GeoError::InvalidGaussKrugerZone { zone, min_zone, max_zone } => {
                MhError::invalid_input(format!(
                    "无效的高斯-克吕格带号 {zone} (允许范围: {min_zone}-{max_zone})"
                ))
            }
            GeoError::ProjectionFailed { operation, message } => {
                MhError::internal(format!("投影转换失败 [{operation}]: {message}"))
            }
            GeoError::CrsParseFailed { definition, reason } => {
                MhError::invalid_input(format!("CRS解析失败 [{}]: {}", definition, reason))
            }
            GeoError::GeometryComputationFailed { operation, message } => {
                MhError::internal(format!("{operation} 计算失败: {message}"))
            }
            GeoError::VincentyNotConverged => {
                MhError::internal("Vincenty公式迭代不收敛".to_string())
            }
            GeoError::SpatialIndexError { operation, message } => {
                MhError::internal(format!("空间索引操作失败 [{operation}]: {message}"))
            }
            GeoError::SingularTransform => {
                MhError::invalid_input("仿射变换矩阵奇异（行列式接近零），无法求逆".to_string())
            }
            GeoError::ConvergenceAngleError { message } => {
                MhError::internal(format!("收敛角计算失败: {message}"))
            }
            GeoError::Io(err) => err,
        }
    }
}

// ============================================================================
// 便捷构造函数
// ============================================================================

impl GeoError {
    /// 创建不支持的 EPSG 错误
    #[inline]
    pub fn unsupported_epsg(code: u32, supported: &'static str) -> Self {
        Self::UnsupportedEpsg { code, supported }
    }

    /// 创建坐标越界错误
    #[inline]
    pub fn coordinate_out_of_range(
        coord_type: &'static str,
        value: f64,
        min: f64,
        max: f64,
    ) -> Self {
        Self::CoordinateOutOfRange {
            coord_type,
            value,
            min,
            max,
        }
    }

    /// 创建无效的 UTM 带号错误
    #[inline]
    pub fn invalid_utm_zone(zone: u8) -> Self {
        Self::InvalidUtmZone { zone }
    }

    /// 创建无效的高斯-克吕格带号错误
    #[inline]
    pub fn invalid_gauss_kruger_zone(zone: u8, min_zone: u8, max_zone: u8) -> Self {
        Self::InvalidGaussKrugerZone {
            zone,
            min_zone,
            max_zone,
        }
    }

    /// 创建投影转换失败错误
    #[inline]
    pub fn projection_failed(operation: &'static str, message: impl Into<String>) -> Self {
        Self::ProjectionFailed {
            operation,
            message: message.into(),
        }
    }

    /// 创建 CRS 解析失败错误
    #[inline]
    pub fn crs_parse_failed(definition: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::CrsParseFailed {
            definition: definition.into(),
            reason: reason.into(),
        }
    }

    /// 创建几何计算失败错误
    #[inline]
    pub fn geometry_computation_failed(
        operation: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self::GeometryComputationFailed {
            operation,
            message: message.into(),
        }
    }

    /// 创建 Vincenty 不收敛错误
    #[inline]
    pub fn vincenty_not_converged() -> Self {
        Self::VincentyNotConverged
    }

    /// 创建空间索引错误
    #[inline]
    pub fn spatial_index_error(operation: &'static str, message: impl Into<String>) -> Self {
        Self::SpatialIndexError {
            operation,
            message: message.into(),
        }
    }

    /// 创建奇异变换错误
    #[inline]
    pub fn singular_transform() -> Self {
        Self::SingularTransform
    }

    /// 创建收敛角计算错误
    #[inline]
    pub fn convergence_angle_error(message: impl Into<String>) -> Self {
        Self::ConvergenceAngleError {
            message: message.into(),
        }
    }

    /// 检查条件，不满足则返回错误
    ///
    /// # 示例
    ///
    /// ```
    /// # use mh_geo::error::{GeoError, GeoResult};
    /// fn validate_latitude(lat: f64) -> GeoResult<()> {
    ///     mh_geo::ensure!((-90.0..=90.0).contains(&lat),
    ///         GeoError::coordinate_out_of_range("纬度", lat, -90.0, 90.0)
    ///     );
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn ensure(cond: bool, err: Self) -> Result<(), Self> {
        if cond {
            Ok(())
        } else {
            Err(err)
        }
    }

    /// 验证 EPSG 代码是否在范围内
    #[inline]
    pub fn check_epsg(code: u32, min: u32, max: u32) -> Result<(), Self> {
        if code < min || code > max {
            Err(Self::unsupported_epsg(code, "请检查文档获取支持的代码范围"))
        } else {
            Ok(())
        }
    }

    /// 验证 UTM 带号
    #[inline]
    pub fn check_utm_zone(zone: u8) -> Result<(), Self> {
        if !(1..=60).contains(&zone) {
            Err(Self::invalid_utm_zone(zone))
        } else {
            Ok(())
        }
    }

    /// 验证高斯-克吕格带号
    #[inline]
    pub fn check_gauss_kruger_zone(zone: u8, min: u8, max: u8) -> Result<(), Self> {
        if !(min..=max).contains(&zone) {
            Err(Self::invalid_gauss_kruger_zone(zone, min, max))
        } else {
            Ok(())
        }
    }

    /// 验证坐标范围
    #[inline]
    pub fn check_coordinate(
        coord_type: &'static str,
        value: f64,
        min: f64,
        max: f64,
    ) -> Result<(), Self> {
        if !(min..=max).contains(&value) {
            Err(Self::coordinate_out_of_range(coord_type, value, min, max))
        } else {
            Ok(())
        }
    }
}

// ============================================================================
// 测试
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsupported_epsg_error() {
        let err = GeoError::unsupported_epsg(99999, "EPSG:4326, EPSG:3857");
        match &err {
            GeoError::UnsupportedEpsg { code, supported } => {
                assert_eq!(*code, 99999);
                assert_eq!(*supported, "EPSG:4326, EPSG:3857");
            }
            _ => panic!("错误的错误类型"),
        }
        let msg = format!("{}", err);
        assert!(msg.contains("99999"));
        assert!(msg.contains("EPSG:4326, EPSG:3857"));
    }

    #[test]
    fn test_coordinate_out_of_range_error() {
        let err = GeoError::coordinate_out_of_range("纬度", 95.5, -90.0, 90.0);
        match &err {
            GeoError::CoordinateOutOfRange { coord_type, value, min, max } => {
                assert_eq!(*coord_type, "纬度");
                assert_eq!(*value, 95.5);
                assert_eq!(*min, -90.0);
                assert_eq!(*max, 90.0);
            }
            _ => panic!("错误的错误类型"),
        }
        let msg = format!("{}", err);
        assert!(msg.contains("纬度"));
        assert!(msg.contains("95.5"));
    }

    #[test]
    fn test_invalid_utm_zone_error() {
        let err = GeoError::invalid_utm_zone(0);
        match &err {
            GeoError::InvalidUtmZone { zone } => {
                assert_eq!(*zone, 0);
            }
            _ => panic!("错误的错误类型"),
        }
        let msg = format!("{}", err);
        assert!(msg.contains("0"));
        assert!(msg.contains("1-60"));
    }

    #[test]
    fn test_invalid_gauss_kruger_zone_error() {
        let err = GeoError::invalid_gauss_kruger_zone(100, 1, 23);
        match &err {
            GeoError::InvalidGaussKrugerZone { zone, min_zone, max_zone } => {
                assert_eq!(*zone, 100);
                assert_eq!(*min_zone, 1);
                assert_eq!(*max_zone, 23);
            }
            _ => panic!("错误的错误类型"),
        }
        let msg = format!("{}", err);
        assert!(msg.contains("100"));
        assert!(msg.contains("1-23"));
    }

    #[test]
    fn test_projection_failed_error() {
        let err = GeoError::projection_failed("正向投影", "参数无效");
        match &err {
            GeoError::ProjectionFailed { operation, message } => {
                assert_eq!(*operation, "正向投影");
                assert_eq!(message, "参数无效");
            }
            _ => panic!("错误的错误类型"),
        }
        let msg = format!("{}", err);
        assert!(msg.contains("正向投影"));
        assert!(msg.contains("参数无效"));
    }

    #[test]
    fn test_crs_parse_failed_error() {
        let err = GeoError::crs_parse_failed("EPSG:4326", "未知格式");
        match &err {
            GeoError::CrsParseFailed { definition, reason } => {
                assert_eq!(definition, "EPSG:4326");
                assert_eq!(reason, "未知格式");
            }
            _ => panic!("错误的错误类型"),
        }
        let msg = format!("{}", err);
        assert!(msg.contains("EPSG:4326"));
        assert!(msg.contains("未知格式"));
    }

    #[test]
    fn test_geometry_computation_failed_error() {
        let err = GeoError::geometry_computation_failed("Vincenty距离", "反余弦参数超出[-1,1]范围");
        match &err {
            GeoError::GeometryComputationFailed { operation, message } => {
                assert_eq!(*operation, "Vincenty距离");
                assert_eq!(message, "反余弦参数超出[-1,1]范围");
            }
            _ => panic!("错误的错误类型"),
        }
        let msg = format!("{}", err);
        assert!(msg.contains("Vincenty距离"));
        assert!(msg.contains("反余弦参数"));
    }

    #[test]
    fn test_vincenty_not_converged_error() {
        let err = GeoError::vincenty_not_converged();
        match err {
            GeoError::VincentyNotConverged => {},
            _ => panic!("错误的错误类型"),
        }
        let msg = format!("{}", err);
        assert!(msg.contains("Vincenty"));
        assert!(msg.contains("不收敛"));
    }

    #[test]
    fn test_spatial_index_error() {
        let err = GeoError::spatial_index_error("最近邻搜索", "R树为空");
        match &err {
            GeoError::SpatialIndexError { operation, message } => {
                assert_eq!(*operation, "最近邻搜索");
                assert_eq!(message, "R树为空");
            }
            _ => panic!("错误的错误类型"),
        }
        let msg = format!("{}", err);
        assert!(msg.contains("最近邻搜索"));
    }

    #[test]
    fn test_singular_transform_error() {
        let err = GeoError::singular_transform();
        match err {
            GeoError::SingularTransform => {},
            _ => panic!("错误的错误类型"),
        }
        let msg = format!("{}", err);
        assert!(msg.contains("奇异"));
        assert!(msg.contains("行列式"));
    }

    #[test]
    fn test_convergence_angle_error() {
        let err = GeoError::convergence_angle_error("子午线计算溢出");
        match &err {
            GeoError::ConvergenceAngleError { message } => {
                assert_eq!(message, "子午线计算溢出");
            }
            _ => panic!("错误的错误类型"),
        }
        let msg = format!("{}", err);
        assert!(msg.contains("收敛角"));
        assert!(msg.contains("子午线计算溢出"));
    }

    #[test]
    fn test_ensure_success() {
        let result = GeoError::ensure(true, GeoError::vincenty_not_converged());
        assert!(result.is_ok());
    }

    #[test]
    fn test_ensure_failure() {
        let result = GeoError::ensure(false, GeoError::invalid_utm_zone(99));
        assert!(result.is_err());
        match result.unwrap_err() {
            GeoError::InvalidUtmZone { zone } => assert_eq!(zone, 99),
            _ => panic!("错误的错误类型"),
        }
    }

    #[test]
    fn test_check_epsg_success() {
        let result = GeoError::check_epsg(4326, 4000, 5000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_epsg_failure() {
        let result = GeoError::check_epsg(3000, 4000, 5000);
        assert!(result.is_err());
        match result.unwrap_err() {
            GeoError::UnsupportedEpsg { code, .. } => assert_eq!(code, 3000),
            _ => panic!("错误的错误类型"),
        }
    }

    #[test]
    fn test_check_utm_zone_success() {
        let result = GeoError::check_utm_zone(30);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_utm_zone_boundary() {
        assert!(GeoError::check_utm_zone(1).is_ok());
        assert!(GeoError::check_utm_zone(60).is_ok());
        assert!(GeoError::check_utm_zone(0).is_err());
        assert!(GeoError::check_utm_zone(61).is_err());
    }

    #[test]
    fn test_check_gauss_kruger_zone() {
        assert!(GeoError::check_gauss_kruger_zone(10, 1, 23).is_ok());
        assert!(GeoError::check_gauss_kruger_zone(0, 1, 23).is_err());
        assert!(GeoError::check_gauss_kruger_zone(24, 1, 23).is_err());
    }

    #[test]
    fn test_check_coordinate() {
        assert!(GeoError::check_coordinate("经度", 120.0, -180.0, 180.0).is_ok());
        assert!(GeoError::check_coordinate("经度", 200.0, -180.0, 180.0).is_err());
    }

    // ============================================================================
    // 转换为 MhError 的测试
    // ============================================================================

    #[test]
    fn test_geo_error_to_mh_error_unsupported_epsg() {
        let geo_err = GeoError::unsupported_epsg(99999, "EPSG:4326");
        let mh_err: MhError = geo_err.into();
        
        match mh_err {
            MhError::InvalidInput(msg) => {
                assert!(msg.contains("99999"));
                assert!(msg.contains("EPSG:4326"));
            }
            _ => panic!("错误的MhError类型"),
        }
    }

    #[test]
    fn test_geo_error_to_mh_error_coordinate_out_of_range() {
        let geo_err = GeoError::coordinate_out_of_range("纬度", 95.5, -90.0, 90.0);
        let mh_err: MhError = geo_err.into();
        
        match mh_err {
            MhError::InvalidInput(msg) => {
                assert!(msg.contains("纬度"));
                assert!(msg.contains("95.5"));
            }
            _ => panic!("错误的MhError类型"),
        }
    }

    #[test]
    fn test_geo_error_to_mh_error_projection_failed() {
        let geo_err = GeoError::projection_failed("逆向投影", "迭代发散");
        let mh_err: MhError = geo_err.into();
        
        match mh_err {
            MhError::Internal(msg) => {
                assert!(msg.contains("逆向投影"));
                assert!(msg.contains("迭代发散"));
            }
            _ => panic!("错误的MhError类型"),
        }
    }

    #[test]
    fn test_geo_error_to_mh_error_internal_variants() {
        // 测试所有映射到 Internal 的变体
        let variants = vec![
            GeoError::vincenty_not_converged(),
            GeoError::spatial_index_error("查询", "失败"),
            GeoError::geometry_computation_failed("计算", "错误"),
            GeoError::convergence_angle_error("失败"),
        ];
        
        for geo_err in variants {
            let mh_err: MhError = geo_err.into();
            match mh_err {
                MhError::Internal(_) => {},
                _ => panic!("应转换为Internal类型"),
            }
        }
    }

    #[test]
    fn test_geo_error_to_mh_error_invalid_input_variants() {
        // 测试所有映射到 InvalidInput 的变体
        let variants = vec![
            GeoError::unsupported_epsg(99999, "test"),
            GeoError::coordinate_out_of_range("test", 1.0, 0.0, 2.0),
            GeoError::invalid_utm_zone(99),
            GeoError::invalid_gauss_kruger_zone(99, 1, 23),
            GeoError::singular_transform(),
        ];
        
        for geo_err in variants {
            let mh_err: MhError = geo_err.into();
            match mh_err {
                MhError::InvalidInput(_) => {},
                _ => panic!("应转换为InvalidInput类型"),
            }
        }
    }
}