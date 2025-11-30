// src-tauri/src/marihydro/core/traits/gradient.rs

//! 梯度计算接口
//!
//! 定义标量场和向量场梯度计算的统一抽象。

use super::mesh::MeshAccess;
use crate::marihydro::core::error::MhResult;
use glam::DVec2;

/// 标量梯度结果
#[derive(Debug, Clone, Copy, Default)]
pub struct ScalarGradient {
    /// x方向梯度
    pub dx: f64,
    /// y方向梯度
    pub dy: f64,
}

impl ScalarGradient {
    pub const ZERO: Self = Self { dx: 0.0, dy: 0.0 };

    pub fn new(dx: f64, dy: f64) -> Self {
        Self { dx, dy }
    }

    /// 转换为 DVec2
    pub fn as_vec(&self) -> DVec2 {
        DVec2::new(self.dx, self.dy)
    }

    /// 梯度大小
    pub fn magnitude(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// 方向
    pub fn direction(&self) -> f64 {
        self.dy.atan2(self.dx)
    }
}

/// 向量梯度结果（2x2 张量）
#[derive(Debug, Clone, Copy, Default)]
pub struct VectorGradient {
    /// du/dx
    pub du_dx: f64,
    /// du/dy
    pub du_dy: f64,
    /// dv/dx
    pub dv_dx: f64,
    /// dv/dy
    pub dv_dy: f64,
}

impl VectorGradient {
    pub const ZERO: Self = Self {
        du_dx: 0.0,
        du_dy: 0.0,
        dv_dx: 0.0,
        dv_dy: 0.0,
    };

    /// 散度 div(u) = du/dx + dv/dy
    pub fn divergence(&self) -> f64 {
        self.du_dx + self.dv_dy
    }

    /// 涡度 curl(u) = dv/dx - du/dy
    pub fn vorticity(&self) -> f64 {
        self.dv_dx - self.du_dy
    }

    /// 应变率张量迹（2D）
    pub fn strain_rate_magnitude(&self) -> f64 {
        let s11 = self.du_dx;
        let s22 = self.dv_dy;
        let s12 = 0.5 * (self.du_dy + self.dv_dx);

        (2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12)).sqrt()
    }
}

/// 梯度计算器接口
///
/// # 实现要求
///
/// 1. 必须处理边界单元（单侧梯度）
/// 2. 应支持病态矩阵回退（如最小二乘法回退到 Green-Gauss）
/// 3. 结果应填充到预分配的输出数组
pub trait GradientComputer: Send + Sync {
    /// 计算器名称
    fn name(&self) -> &'static str;

    /// 计算标量场梯度
    ///
    /// # 参数
    ///
    /// - `mesh`: 网格访问接口
    /// - `field`: 标量场值（长度 = n_cells）
    /// - `grad_x`: 输出 x 梯度（长度 = n_cells）
    /// - `grad_y`: 输出 y 梯度（长度 = n_cells）
    fn compute_scalar_gradient<M: MeshAccess>(
        &self,
        mesh: &M,
        field: &[f64],
        grad_x: &mut [f64],
        grad_y: &mut [f64],
    ) -> MhResult<()>;

    /// 计算向量场梯度
    ///
    /// # 参数
    ///
    /// - `mesh`: 网格访问接口
    /// - `u`: x 分量场（长度 = n_cells）
    /// - `v`: y 分量场（长度 = n_cells）
    /// - 输出：四个梯度分量数组
    fn compute_vector_gradient<M: MeshAccess>(
        &self,
        mesh: &M,
        u: &[f64],
        v: &[f64],
        du_dx: &mut [f64],
        du_dy: &mut [f64],
        dv_dx: &mut [f64],
        dv_dy: &mut [f64],
    ) -> MhResult<()>;

    /// 计算单个单元的标量梯度
    fn compute_cell_gradient<M: MeshAccess>(
        &self,
        mesh: &M,
        field: &[f64],
        cell_idx: usize,
    ) -> ScalarGradient;
}

/// 梯度计算方法类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GradientMethod {
    /// Green-Gauss 梯度（基于面积分）
    #[default]
    GreenGauss,
    /// 最小二乘梯度
    LeastSquares,
    /// 加权最小二乘
    WeightedLeastSquares,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_gradient() {
        let grad = ScalarGradient::new(3.0, 4.0);
        assert!((grad.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_gradient() {
        // 简单剪切流: u = y, v = 0
        // du/dy = 1, 其他为0
        let grad = VectorGradient {
            du_dx: 0.0,
            du_dy: 1.0,
            dv_dx: 0.0,
            dv_dy: 0.0,
        };

        assert!((grad.divergence() - 0.0).abs() < 1e-10);
        assert!((grad.vorticity() - (-1.0)).abs() < 1e-10);
    }
}
