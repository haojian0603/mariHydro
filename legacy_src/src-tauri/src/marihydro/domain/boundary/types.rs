// src-tauri/src/marihydro/domain/boundary/types.rs

//! 边界条件类型定义

use serde::{Deserialize, Serialize};

/// 边界类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum BoundaryKind {
    /// 固壁边界（无穿透）
    Wall = 0,
    /// 开海边界（Flather）
    OpenSea = 1,
    /// 河流入流
    RiverInflow = 2,
    /// 自由出流
    Outflow = 3,
    /// 对称边界
    Symmetry = 4,
    /// 周期边界
    Periodic = 5,
}

impl Default for BoundaryKind {
    fn default() -> Self {
        Self::Wall
    }
}

impl BoundaryKind {
    /// 是否需要外部强迫数据
    pub fn requires_forcing(&self) -> bool {
        matches!(self, Self::OpenSea | Self::RiverInflow)
    }

    /// 是否为固壁类型
    pub fn is_solid(&self) -> bool {
        matches!(self, Self::Wall | Self::Symmetry)
    }

    /// 是否为开边界类型
    pub fn is_open(&self) -> bool {
        matches!(self, Self::OpenSea | Self::Outflow)
    }
}

/// 边界条件参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryCondition {
    /// 边界名称
    pub name: String,
    /// 边界类型
    pub kind: BoundaryKind,
    /// 固定水位值（用于某些边界类型）
    pub fixed_eta: Option<f64>,
    /// 固定流量值（用于河流入流）
    pub fixed_discharge: Option<f64>,
    /// 关联的强迫数据源ID
    pub forcing_id: Option<usize>,
    /// 曼宁粗糙度（用于摩擦边界）
    pub manning_n: Option<f64>,
}

impl BoundaryCondition {
    /// 创建固壁边界
    pub fn wall(name: &str) -> Self {
        Self {
            name: name.to_string(),
            kind: BoundaryKind::Wall,
            fixed_eta: None,
            fixed_discharge: None,
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 创建开海边界
    pub fn open_sea(name: &str) -> Self {
        Self {
            name: name.to_string(),
            kind: BoundaryKind::OpenSea,
            fixed_eta: None,
            fixed_discharge: None,
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 创建河流入流边界
    pub fn river_inflow(name: &str, discharge: f64) -> Self {
        Self {
            name: name.to_string(),
            kind: BoundaryKind::RiverInflow,
            fixed_eta: None,
            fixed_discharge: Some(discharge),
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 创建自由出流边界
    pub fn outflow(name: &str) -> Self {
        Self {
            name: name.to_string(),
            kind: BoundaryKind::Outflow,
            fixed_eta: None,
            fixed_discharge: None,
            forcing_id: None,
            manning_n: None,
        }
    }

    /// 设置强迫数据源
    pub fn with_forcing(mut self, forcing_id: usize) -> Self {
        self.forcing_id = Some(forcing_id);
        self
    }

    /// 设置固定水位
    pub fn with_fixed_eta(mut self, eta: f64) -> Self {
        self.fixed_eta = Some(eta);
        self
    }
}

/// 外部强迫数据
#[derive(Debug, Clone, Copy, Default)]
pub struct ExternalForcing {
    /// 水位 [m]
    pub eta: f64,
    /// x 方向速度 [m/s]
    pub u: f64,
    /// y 方向速度 [m/s]
    pub v: f64,
}

impl ExternalForcing {
    /// 零强迫
    pub const ZERO: Self = Self {
        eta: 0.0,
        u: 0.0,
        v: 0.0,
    };

    /// 创建水位强迫
    pub fn with_eta(eta: f64) -> Self {
        Self {
            eta,
            u: 0.0,
            v: 0.0,
        }
    }

    /// 创建速度强迫
    pub fn with_velocity(u: f64, v: f64) -> Self {
        Self { eta: 0.0, u, v }
    }

    /// 创建完整强迫
    pub fn new(eta: f64, u: f64, v: f64) -> Self {
        Self { eta, u, v }
    }
}

/// 边界计算参数
#[derive(Debug, Clone, Copy)]
pub struct BoundaryParams {
    /// 重力加速度
    pub gravity: f64,
    /// 最小水深
    pub h_min: f64,
    /// sqrt(g)
    pub sqrt_g: f64,
}

impl BoundaryParams {
    pub fn new(gravity: f64, h_min: f64) -> Self {
        Self {
            gravity,
            h_min,
            sqrt_g: gravity.sqrt(),
        }
    }
}

impl Default for BoundaryParams {
    fn default() -> Self {
        Self::new(9.81, 1e-6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_kind() {
        assert!(BoundaryKind::Wall.is_solid());
        assert!(!BoundaryKind::Wall.is_open());
        assert!(BoundaryKind::OpenSea.requires_forcing());
    }

    #[test]
    fn test_boundary_condition() {
        let bc = BoundaryCondition::wall("north");
        assert_eq!(bc.kind, BoundaryKind::Wall);
        assert!(bc.fixed_eta.is_none());

        let bc = BoundaryCondition::river_inflow("yangtze", 30000.0);
        assert_eq!(bc.kind, BoundaryKind::RiverInflow);
        assert!((bc.fixed_discharge.unwrap() - 30000.0).abs() < 1e-10);
    }
}
