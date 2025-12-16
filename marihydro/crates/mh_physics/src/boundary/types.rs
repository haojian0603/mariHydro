// crates/mh_physics/src/boundary/types.rs

//! 边界条件类型定义
//!
//! 本模块定义浅水方程求解所需的边界条件类型，包括：
//! - BoundaryKind: 边界类型枚举
//! - BoundaryCondition: 边界条件配置
//! - ExternalForcing: 外部强迫数据
//! - BoundaryParams: 边界计算参数
//!
//! # 迁移说明
//!
//! 从 legacy_src/domain/boundary/types.rs 迁移，适配新架构：
//! - 使用 glam::DVec2 代替 (f64, f64) 表示速度
//! - 使用 serde 支持配置文件
//! - 使用 repr(u8) 支持 GPU 传输

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::types::NumericalParamsF64;

// ============================================================
// 边界类型枚举
// ============================================================

/// 边界类型枚举
///
/// 定义浅水方程支持的边界条件类型。使用 `repr(u8)` 以便于 GPU 数据传输。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum BoundaryKind {
    /// 固壁边界（无穿透）
    ///
    /// 法向速度反射，切向速度保持。适用于不可渗透的边界。
    #[default]
    Wall = 0,

    /// 开海边界（Flather 辐射边界条件）
    ///
    /// 使用特征关系结合外部强迫数据，允许波动自由传出。
    OpenSea = 1,

    /// 河流入流
    ///
    /// 给定流量或水位的入流边界条件。
    RiverInflow = 2,

    /// 自由出流
    ///
    /// 零梯度外推，允许水流自由流出计算域。
    Outflow = 3,

    /// 对称边界
    ///
    /// 与固壁类似但无摩擦，用于模型对称简化。
    Symmetry = 4,

    /// 周期边界
    ///
    /// 需要成对设置，用于模拟周期性流动。
    Periodic = 5,
}

impl BoundaryKind {
    /// 是否需要外部强迫数据
    ///
    /// OpenSea 和 RiverInflow 类型需要外部提供水位或流量数据。
    #[inline]
    pub fn requires_forcing(&self) -> bool {
        matches!(self, Self::OpenSea | Self::RiverInflow)
    }

    /// 是否为固壁类型（反射边界）
    ///
    /// Wall 和 Symmetry 都会反射法向速度。
    #[inline]
    pub fn is_solid(&self) -> bool {
        matches!(self, Self::Wall | Self::Symmetry)
    }

    /// 是否为开边界类型
    ///
    /// 允许物质和能量通过的边界。
    #[inline]
    pub fn is_open(&self) -> bool {
        matches!(self, Self::OpenSea | Self::Outflow | Self::RiverInflow)
    }

    /// 从 u8 值转换（用于 GPU 数据读取）
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Wall),
            1 => Some(Self::OpenSea),
            2 => Some(Self::RiverInflow),
            3 => Some(Self::Outflow),
            4 => Some(Self::Symmetry),
            5 => Some(Self::Periodic),
            _ => None,
        }
    }

    /// 转换为 u8 值
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

impl std::fmt::Display for BoundaryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Wall => "Wall",
            Self::OpenSea => "OpenSea",
            Self::RiverInflow => "RiverInflow",
            Self::Outflow => "Outflow",
            Self::Symmetry => "Symmetry",
            Self::Periodic => "Periodic",
        };
        write!(f, "{}", name)
    }
}

// ============================================================
// 边界条件配置
// ============================================================

/// 边界条件配置
///
/// 完整描述一个边界条件的参数，包括类型、固定值和关联的强迫数据源。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryCondition {
    /// 边界名称（用于标识和查找）
    pub name: String,

    /// 边界类型
    pub kind: BoundaryKind,

    /// 固定水位值 [m]
    ///
    /// 用于 OpenSea 边界的恒定水位或 Outflow 的参考水位。
    pub fixed_eta: Option<f64>,

    /// 固定流量值 [m³/s]
    ///
    /// 用于 RiverInflow 边界的恒定流量。
    pub fixed_discharge: Option<f64>,

    /// 关联的强迫数据 Provider ID
    ///
    /// 用于查找时变强迫数据源。
    pub forcing_id: Option<usize>,

    /// 曼宁粗糙度系数
    ///
    /// 用于边界处的摩擦计算。
    pub manning_n: Option<f64>,
}

impl BoundaryCondition {
    /// 创建固壁边界条件
    ///
    /// # 示例
    /// ```
    /// use mh_physics::boundary::BoundaryCondition;
    ///
    /// let bc = BoundaryCondition::wall("north_wall");
    /// assert!(bc.kind.is_solid());
    /// ```
    pub fn wall(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: BoundaryKind::Wall,
            fixed_eta: None,
            fixed_discharge: None,
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 创建开海边界条件
    ///
    /// # 示例
    /// ```
    /// use mh_physics::boundary::BoundaryCondition;
    ///
    /// let bc = BoundaryCondition::open_sea("south_open");
    /// assert!(bc.kind.requires_forcing());
    /// ```
    pub fn open_sea(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: BoundaryKind::OpenSea,
            fixed_eta: None,
            fixed_discharge: None,
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 创建河流入流边界条件
    ///
    /// # 参数
    /// - `name`: 边界名称
    /// - `discharge`: 恒定入流量 [m³/s]
    ///
    /// # 示例
    /// ```
    /// use mh_physics::boundary::BoundaryCondition;
    ///
    /// let bc = BoundaryCondition::river_inflow("yangtze", 30000.0);
    /// assert_eq!(bc.fixed_discharge, Some(30000.0));
    /// ```
    // ALLOW_F64: Layer 4 配置 API
    pub fn river_inflow(name: impl Into<String>, discharge: f64) -> Self {
        Self {
            name: name.into(),
            kind: BoundaryKind::RiverInflow,
            fixed_eta: None,
            fixed_discharge: Some(discharge),
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 创建自由出流边界条件
    pub fn outflow(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: BoundaryKind::Outflow,
            fixed_eta: None,
            fixed_discharge: None,
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 创建对称边界条件
    pub fn symmetry(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: BoundaryKind::Symmetry,
            fixed_eta: None,
            fixed_discharge: None,
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 设置强迫数据源 ID
    pub fn with_forcing(mut self, forcing_id: usize) -> Self {
        self.forcing_id = Some(forcing_id);
        self
    }

    /// 设置固定水位
    // ALLOW_F64: Layer 4 配置 API
    pub fn with_fixed_eta(mut self, eta: f64) -> Self {
        self.fixed_eta = Some(eta);
        self
    }

    /// 设置固定流量
    // ALLOW_F64: Layer 4 配置 API
    pub fn with_fixed_discharge(mut self, discharge: f64) -> Self {
        self.fixed_discharge = Some(discharge);
        self
    }

    /// 设置曼宁粗糙度
    // ALLOW_F64: Layer 4 配置 API
    pub fn with_manning_n(mut self, n: f64) -> Self {
        self.manning_n = Some(n);
        self
    }
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        Self::wall("default")
    }
}

// ============================================================
// 外部强迫数据
// ============================================================

/// 外部强迫数据
///
/// 边界处的水位和速度数据，用于 OpenSea、RiverInflow 等边界条件。
// ALLOW_F64: 边界强迫数据与 DVec2 配合使用
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExternalForcing {
    /// 水位 [m]
    pub eta: f64, // ALLOW_F64: 边界数据结构

    /// 速度向量 [m/s]
    pub velocity: DVec2,
}

impl ExternalForcing {
    /// 零强迫常量
    pub const ZERO: Self = Self {
        eta: 0.0,
        velocity: DVec2::ZERO,
    };

    /// 创建完整的强迫数据
    ///
    /// # 参数
    /// - `eta`: 水位 [m]
    /// - `u`: x 方向速度 [m/s]
    /// - `v`: y 方向速度 [m/s]
    #[inline]
    pub fn new(eta: f64, u: f64, v: f64) -> Self { // ALLOW_F64: 边界数据与 DVec2 配合
        Self {
            eta,
            velocity: DVec2::new(u, v),
        }
    }

    /// 创建仅水位的强迫数据
    #[inline]
    pub fn with_eta(eta: f64) -> Self { // ALLOW_F64: 边界数据与 DVec2 配合
        Self {
            eta,
            velocity: DVec2::ZERO,
        }
    }

    /// 创建仅速度的强迫数据
    #[inline]
    pub fn with_velocity(u: f64, v: f64) -> Self { // ALLOW_F64: 边界数据与 DVec2 配合
        Self {
            eta: 0.0,
            velocity: DVec2::new(u, v),
        }
    }

    /// 获取 x 方向速度
    #[inline]
    pub fn u(&self) -> f64 {
        self.velocity.x
    }

    /// 获取 y 方向速度
    #[inline]
    pub fn v(&self) -> f64 {
        self.velocity.y
    }

    /// 检查数据是否有效
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.eta.is_finite() && self.velocity.is_finite()
    }
}

// ============================================================
// 边界计算参数
// ============================================================

/// 边界计算参数
///
/// 边界通量计算所需的物理参数和预计算常量。
// ALLOW_F64: Layer 4 边界计算参数配置
#[derive(Debug, Clone, Copy)]
pub struct BoundaryParams {
    /// 重力加速度 [m/s²]
    pub gravity: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最小水深阈值 [m]
    pub h_min: f64, // ALLOW_F64: Layer 4 配置参数
    /// sqrt(g) - 预计算以提高性能
    pub sqrt_g: f64, // ALLOW_F64: Layer 4 配置参数
}

impl BoundaryParams {
    /// 创建边界参数
    ///
    /// # 参数
    /// - `gravity`: 重力加速度 [m/s²]
    /// - `h_min`: 最小水深阈值 [m]
    pub fn new(gravity: f64, h_min: f64) -> Self { // ALLOW_F64: Layer 4 配置 API
        Self {
            gravity,
            h_min,
            sqrt_g: gravity.sqrt(),
        }
    }

    /// 从数值参数创建
    ///
    /// 使用默认重力加速度 (9.81 m/s²)。
    /// 如果需要自定义重力，请使用 `new` 方法。
    pub fn from_numerical_params(params: &NumericalParamsF64) -> Self {
        Self::new(9.81, params.h_min)
    }

    /// 从数值参数和物理常数创建
    pub fn from_params(numerical: &NumericalParamsF64, physics: &crate::types::PhysicalConstants) -> Self {
        Self::new(physics.g, numerical.h_min)
    }

    /// 计算特征速度（波速）
    ///
    /// c = sqrt(g * h)
    #[inline]
    pub fn wave_speed(&self, h: f64) -> f64 { // ALLOW_F64: 与 BoundaryParams 配合使用
        self.sqrt_g * h.max(self.h_min).sqrt()
    }

    /// 计算静水压力
    ///
    /// p = 0.5 * g * h²
    #[inline]
    pub fn hydrostatic_pressure(&self, h: f64) -> f64 { // ALLOW_F64: 与 BoundaryParams 配合使用
        0.5 * self.gravity * h * h
    }
}

impl Default for BoundaryParams {
    fn default() -> Self {
        Self::new(9.81, 1e-6)
    }
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_kind_properties() {
        assert!(BoundaryKind::Wall.is_solid());
        assert!(!BoundaryKind::Wall.is_open());
        assert!(!BoundaryKind::Wall.requires_forcing());

        assert!(BoundaryKind::OpenSea.is_open());
        assert!(BoundaryKind::OpenSea.requires_forcing());
        assert!(!BoundaryKind::OpenSea.is_solid());

        assert!(BoundaryKind::RiverInflow.requires_forcing());
        assert!(BoundaryKind::RiverInflow.is_open());

        assert!(BoundaryKind::Symmetry.is_solid());
    }

    #[test]
    fn test_boundary_kind_conversion() {
        for i in 0..=5 {
            let kind = BoundaryKind::from_u8(i).unwrap();
            assert_eq!(kind.as_u8(), i);
        }
        assert!(BoundaryKind::from_u8(6).is_none());
    }

    #[test]
    fn test_boundary_condition_builders() {
        let wall = BoundaryCondition::wall("north");
        assert_eq!(wall.name, "north");
        assert_eq!(wall.kind, BoundaryKind::Wall);

        let river = BoundaryCondition::river_inflow("yangtze", 30000.0);
        assert_eq!(river.kind, BoundaryKind::RiverInflow);
        assert!((river.fixed_discharge.unwrap() - 30000.0).abs() < 1e-10);

        let open = BoundaryCondition::open_sea("south")
            .with_forcing(1)
            .with_fixed_eta(0.5);
        assert_eq!(open.forcing_id, Some(1));
        assert!((open.fixed_eta.unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_external_forcing() {
        let forcing = ExternalForcing::new(1.5, 0.5, -0.3);
        assert!((forcing.eta - 1.5).abs() < 1e-10);
        assert!((forcing.u() - 0.5).abs() < 1e-10);
        assert!((forcing.v() - (-0.3)).abs() < 1e-10);
        assert!(forcing.is_valid());

        let zero = ExternalForcing::ZERO;
        assert!((zero.eta).abs() < 1e-10);
    }

    #[test]
    fn test_boundary_params() {
        let params = BoundaryParams::default();
        assert!((params.gravity - 9.81).abs() < 1e-10);
        assert!((params.sqrt_g - 9.81_f64.sqrt()).abs() < 1e-10);

        // 波速测试
        let c = params.wave_speed(1.0);
        assert!((c - 9.81_f64.sqrt()).abs() < 1e-10);

        // 静水压力测试
        let p = params.hydrostatic_pressure(1.0);
        assert!((p - 0.5 * 9.81).abs() < 1e-10);
    }
}
