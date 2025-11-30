// src-tauri/src/marihydro/core/traits/limiter.rs

//! 梯度限制器接口
//!
//! 定义用于抑制数值振荡的梯度限制器抽象。

use super::mesh::MeshAccess;
use crate::marihydro::core::error::MhResult;

/// 梯度限制器接口
///
/// # 物理意义
///
/// 梯度限制器用于保证高阶重构的单调性，防止在间断处产生虚假振荡。
/// 限制器系数 φ ∈ [0, 1]：
/// - φ = 0：完全一阶精度（无梯度）
/// - φ = 1：完全二阶精度（不限制）
pub trait GradientLimiter: Send + Sync {
    /// 限制器名称
    fn name(&self) -> &'static str;

    /// 计算所有单元的限制器系数
    ///
    /// # 参数
    ///
    /// - `mesh`: 网格访问接口
    /// - `field`: 标量场值
    /// - `grad_x`, `grad_y`: 未限制的梯度
    /// - `phi`: 输出限制器系数（长度 = n_cells）
    fn compute_limiter<M: MeshAccess>(
        &self,
        mesh: &M,
        field: &[f64],
        grad_x: &[f64],
        grad_y: &[f64],
        phi: &mut [f64],
    ) -> MhResult<()>;

    /// 应用限制器到梯度（原位修改）
    fn apply_limiter(&self, phi: &[f64], grad_x: &mut [f64], grad_y: &mut [f64]) {
        for i in 0..phi.len() {
            grad_x[i] *= phi[i];
            grad_y[i] *= phi[i];
        }
    }

    /// 计算并应用限制器（组合操作）
    fn compute_and_apply<M: MeshAccess>(
        &self,
        mesh: &M,
        field: &[f64],
        grad_x: &mut [f64],
        grad_y: &mut [f64],
        phi_buffer: &mut [f64],
    ) -> MhResult<()> {
        self.compute_limiter(mesh, field, grad_x, grad_y, phi_buffer)?;
        self.apply_limiter(phi_buffer, grad_x, grad_y);
        Ok(())
    }
}

/// 限制器类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LimiterType {
    /// 无限制（纯二阶）
    None,
    /// Barth-Jespersen 限制器（严格单调）
    BarthJespersen,
    /// Venkatakrishnan 限制器（平滑）
    #[default]
    Venkatakrishnan,
    /// MLP-u2 限制器
    MLP,
}

/// 无限制器（总是返回 1.0）
pub struct NoLimiter;

impl GradientLimiter for NoLimiter {
    fn name(&self) -> &'static str {
        "None"
    }

    fn compute_limiter<M: MeshAccess>(
        &self,
        mesh: &M,
        _field: &[f64],
        _grad_x: &[f64],
        _grad_y: &[f64],
        phi: &mut [f64],
    ) -> MhResult<()> {
        phi.fill(1.0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_limiter() {
        let limiter = NoLimiter;
        assert_eq!(limiter.name(), "None");

        let mut phi = vec![0.0; 5];
        let mut grad_x = vec![1.0; 5];
        let mut grad_y = vec![2.0; 5];

        limiter.apply_limiter(&vec![1.0; 5], &mut grad_x, &mut grad_y);

        assert!((grad_x[0] - 1.0).abs() < 1e-10);
        assert!((grad_y[0] - 2.0).abs() < 1e-10);
    }
}
