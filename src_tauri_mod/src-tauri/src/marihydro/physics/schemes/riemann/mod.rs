// src-tauri/src/marihydro/physics/schemes/riemann/mod.rs

//! 黎曼求解器子系统
//!
//! 提供统一的黎曼求解器接口和多种实现。
//!
//! # 功能组件
//!
//! - [`traits`]: 统一的 `RiemannSolver` trait
//! - [`hllc`]: HLLC 近似黎曼求解器
//! - [`rusanov`]: Rusanov（局部Lax-Friedrichs）求解器
//! - [`adaptive`]: 自适应求解器选择器
//!
//! # 物理背景
//!
//! 黎曼问题是初值为分段常数的双曲守恒律问题。在有限体积法中，
//! 需要求解界面处左右状态形成的黎曼问题来计算数值通量。
//!
//! ## 支持的求解器
//!
//! 1. **HLLC** - 高精度求解器，包含接触波信息
//! 2. **Rusanov** - 稳健的一阶求解器，适合激波
//! 3. **Roe** (计划中) - 精确线性化求解器
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use crate::marihydro::physics::schemes::riemann::{
//!     RiemannSolver, HllcSolver, RusanovSolver, AdaptiveSolver
//! };
//!
//! // 创建求解器
//! let solver = HllcSolver::new(params, 9.81);
//!
//! // 求解界面通量
//! let flux = solver.solve(
//!     h_left, h_right,
//!     vel_left, vel_right,
//!     normal
//! )?;
//! ```
//!
//! # 关联问题
//!
//! - P5-005: HLLC 实现改进
//! - P5-006: 多求解器支持
//! - P1-009: 数值通量稳定性
//! - P1-010: 熵修正

pub mod adaptive;
pub mod hllc;
pub mod rusanov;
pub mod traits;

// 重导出常用类型
pub use adaptive::{AdaptiveSolver, SolverSelector};
pub use hllc::HllcSolver;
pub use rusanov::RusanovSolver;
pub use traits::{RiemannFlux, RiemannSolver, SolverCapabilities};

use glam::DVec2;

/// 界面状态
#[derive(Debug, Clone, Copy)]
pub struct InterfaceState {
    /// 左侧水深
    pub h_left: f64,
    /// 右侧水深
    pub h_right: f64,
    /// 左侧速度
    pub vel_left: DVec2,
    /// 右侧速度
    pub vel_right: DVec2,
    /// 界面法向量
    pub normal: DVec2,
    /// 左侧床面高程（可选）
    pub z_left: Option<f64>,
    /// 右侧床面高程（可选）
    pub z_right: Option<f64>,
}

impl InterfaceState {
    /// 创建基本状态
    pub fn new(h_left: f64, h_right: f64, vel_left: DVec2, vel_right: DVec2, normal: DVec2) -> Self {
        Self {
            h_left,
            h_right,
            vel_left,
            vel_right,
            normal,
            z_left: None,
            z_right: None,
        }
    }

    /// 添加床面高程
    pub fn with_bed(mut self, z_left: f64, z_right: f64) -> Self {
        self.z_left = Some(z_left);
        self.z_right = Some(z_right);
        self
    }

    /// 计算左侧法向速度
    #[inline]
    pub fn un_left(&self) -> f64 {
        self.vel_left.dot(self.normal)
    }

    /// 计算右侧法向速度
    #[inline]
    pub fn un_right(&self) -> f64 {
        self.vel_right.dot(self.normal)
    }

    /// 计算切向量
    #[inline]
    pub fn tangent(&self) -> DVec2 {
        DVec2::new(-self.normal.y, self.normal.x)
    }

    /// 计算左侧切向速度
    #[inline]
    pub fn ut_left(&self) -> f64 {
        self.vel_left.dot(self.tangent())
    }

    /// 计算右侧切向速度
    #[inline]
    pub fn ut_right(&self) -> f64 {
        self.vel_right.dot(self.tangent())
    }

    /// 两侧是否都干燥
    #[inline]
    pub fn is_dry(&self, h_threshold: f64) -> bool {
        self.h_left <= h_threshold && self.h_right <= h_threshold
    }
}

/// 求解器类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SolverType {
    /// HLLC 求解器
    #[default]
    Hllc,
    /// Rusanov 求解器
    Rusanov,
    /// Roe 求解器（计划中）
    Roe,
    /// 自适应选择
    Adaptive,
}

impl std::fmt::Display for SolverType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverType::Hllc => write!(f, "HLLC"),
            SolverType::Rusanov => write!(f, "Rusanov"),
            SolverType::Roe => write!(f, "Roe"),
            SolverType::Adaptive => write!(f, "Adaptive"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interface_state() {
        let state = InterfaceState::new(1.0, 0.5, DVec2::new(1.0, 0.0), DVec2::ZERO, DVec2::X);

        assert!((state.un_left() - 1.0).abs() < 1e-10);
        assert!((state.un_right()).abs() < 1e-10);
        assert!(!state.is_dry(1e-4));
    }
}
