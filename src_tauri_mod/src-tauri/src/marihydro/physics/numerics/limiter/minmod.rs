// src-tauri/src/marihydro/physics/numerics/limiter/minmod.rs

//! Minmod 限制器
//!
//! 经典的一阶限制器，通过取最小模来避免新极值的产生。
//! 是最保守的限制器之一，数值耗散较大但稳定性好。

use super::super::gradient::ScalarGradientStorage;
use super::traits::{CellLimitResult, Limiter, LimiterCapabilities, LimiterConfig, LimiterContext};
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::CellIndex;
use glam::DVec2;

/// Minmod 限制器
#[derive(Debug, Clone, Copy)]
pub struct MinmodLimiter {
    config: LimiterConfig,
}

impl Default for MinmodLimiter {
    fn default() -> Self {
        Self {
            config: LimiterConfig::default(),
        }
    }
}

impl MinmodLimiter {
    /// 创建新的限制器
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置配置
    pub fn with_config(mut self, config: LimiterConfig) -> Self {
        self.config = config;
        self
    }

    /// Minmod 函数
    ///
    /// 返回两个数中模最小的，如果符号不同则返回 0
    #[inline]
    fn minmod(a: f64, b: f64) -> f64 {
        if a * b <= 0.0 {
            0.0
        } else if a.abs() < b.abs() {
            a
        } else {
            b
        }
    }

    /// 三参数 minmod
    ///
    /// 返回三个数中模最小的，如果任意两个符号不同则返回 0
    /// 保留供 TVD 重构等场景使用
    #[allow(dead_code)]
    #[inline]
    fn minmod3(a: f64, b: f64, c: f64) -> f64 {
        Self::minmod(a, Self::minmod(b, c))
    }
}

impl Limiter for MinmodLimiter {
    fn name(&self) -> &'static str {
        "Minmod"
    }

    fn capabilities(&self) -> LimiterCapabilities {
        LimiterCapabilities {
            parallel: true,
            smooth: false, // Minmod 不平滑
            monotone: true,
            order: 1,
        }
    }

    fn limit_cell<M: MeshAccess>(
        &self,
        cell: CellIndex,
        ctx: &LimiterContext,
        mesh: &M,
    ) -> CellLimitResult {
        let i = cell.0;
        let phi_c = ctx.field[i];
        let center = mesh.cell_centroid(cell);
        let grad = ctx.gradient.get(i);

        // 获取邻居极值
        let (mut phi_max, mut phi_min) = (phi_c, phi_c);
        for &nb in mesh.cell_neighbors(cell) {
            if nb.is_valid() {
                let phi_n = ctx.field[nb.0];
                phi_max = phi_max.max(phi_n);
                phi_min = phi_min.min(phi_n);
            }
        }

        // 计算各面的限制因子
        let mut alpha: f64 = 1.0;
        for &face in mesh.cell_faces(cell) {
            let fc = mesh.face_centroid(face);
            let r = fc - center;
            let delta = grad.dot(r);

            if delta.abs() > ctx.config.epsilon {
                let ratio = if delta > 0.0 {
                    // 重构值增加，使用 (phi_max - phi_c)
                    let delta_max = phi_max - phi_c;
                    if delta_max.abs() < ctx.config.epsilon {
                        0.0
                    } else {
                        (delta_max / delta).clamp(0.0, 1.0)
                    }
                } else {
                    // 重构值减少，使用 (phi_min - phi_c)
                    let delta_min = phi_min - phi_c;
                    if delta_min.abs() < ctx.config.epsilon {
                        0.0
                    } else {
                        (delta_min / delta).clamp(0.0, 1.0)
                    }
                };
                alpha = alpha.min(ratio);
            }
        }

        CellLimitResult::new(alpha)
    }
}

/// 超级蜂（Superbee）限制器
///
/// 比 Minmod 更激进，保留更多梯度信息
#[derive(Debug, Clone, Copy)]
pub struct SuperbeeLimiter {
    config: LimiterConfig,
}

impl Default for SuperbeeLimiter {
    fn default() -> Self {
        Self {
            config: LimiterConfig::default(),
        }
    }
}

impl SuperbeeLimiter {
    /// 创建新的限制器
    pub fn new() -> Self {
        Self::default()
    }

    /// Superbee 限制器函数
    #[inline]
    fn superbee(r: f64) -> f64 {
        if r <= 0.0 {
            0.0
        } else if r <= 0.5 {
            2.0 * r
        } else if r <= 1.0 {
            1.0
        } else if r <= 2.0 {
            r
        } else {
            2.0
        }
    }
}

impl Limiter for SuperbeeLimiter {
    fn name(&self) -> &'static str {
        "Superbee"
    }

    fn capabilities(&self) -> LimiterCapabilities {
        LimiterCapabilities {
            parallel: true,
            smooth: false,
            monotone: true,
            order: 2,
        }
    }

    fn limit_cell<M: MeshAccess>(
        &self,
        cell: CellIndex,
        ctx: &LimiterContext,
        mesh: &M,
    ) -> CellLimitResult {
        let i = cell.0;
        let phi_c = ctx.field[i];
        let center = mesh.cell_centroid(cell);
        let grad = ctx.gradient.get(i);

        // 获取邻居极值
        let (mut phi_max, mut phi_min) = (phi_c, phi_c);
        for &nb in mesh.cell_neighbors(cell) {
            if nb.is_valid() {
                let phi_n = ctx.field[nb.0];
                phi_max = phi_max.max(phi_n);
                phi_min = phi_min.min(phi_n);
            }
        }

        // 计算各面的限制因子
        let mut alpha: f64 = 1.0;
        for &face in mesh.cell_faces(cell) {
            let fc = mesh.face_centroid(face);
            let r = fc - center;
            let delta = grad.dot(r);

            if delta.abs() > ctx.config.epsilon {
                let delta_bound = if delta > 0.0 {
                    phi_max - phi_c
                } else {
                    phi_min - phi_c
                };

                if delta_bound.abs() > ctx.config.epsilon {
                    let r_ratio = delta_bound / delta;
                    let phi = Self::superbee(r_ratio);
                    alpha = alpha.min(phi.clamp(0.0, 2.0) / 2.0); // 归一化到 [0, 1]
                } else {
                    alpha = 0.0;
                }
            }
        }

        CellLimitResult::new(alpha)
    }
}

/// Van Leer 限制器
///
/// 平滑的限制器，常用于高阶格式
#[derive(Debug, Clone, Copy)]
pub struct VanLeerLimiter {
    config: LimiterConfig,
}

impl Default for VanLeerLimiter {
    fn default() -> Self {
        Self {
            config: LimiterConfig::default(),
        }
    }
}

impl VanLeerLimiter {
    /// 创建新的限制器
    pub fn new() -> Self {
        Self::default()
    }

    /// Van Leer 限制器函数
    #[inline]
    fn van_leer(r: f64) -> f64 {
        if r <= 0.0 {
            0.0
        } else {
            (r + r.abs()) / (1.0 + r.abs())
        }
    }
}

impl Limiter for VanLeerLimiter {
    fn name(&self) -> &'static str {
        "Van Leer"
    }

    fn capabilities(&self) -> LimiterCapabilities {
        LimiterCapabilities {
            parallel: true,
            smooth: true,
            monotone: true,
            order: 2,
        }
    }

    fn limit_cell<M: MeshAccess>(
        &self,
        cell: CellIndex,
        ctx: &LimiterContext,
        mesh: &M,
    ) -> CellLimitResult {
        let i = cell.0;
        let phi_c = ctx.field[i];
        let center = mesh.cell_centroid(cell);
        let grad = ctx.gradient.get(i);

        // 获取邻居极值
        let (mut phi_max, mut phi_min) = (phi_c, phi_c);
        for &nb in mesh.cell_neighbors(cell) {
            if nb.is_valid() {
                let phi_n = ctx.field[nb.0];
                phi_max = phi_max.max(phi_n);
                phi_min = phi_min.min(phi_n);
            }
        }

        // 计算各面的限制因子
        let mut alpha: f64 = 1.0;
        for &face in mesh.cell_faces(cell) {
            let fc = mesh.face_centroid(face);
            let r = fc - center;
            let delta = grad.dot(r);

            if delta.abs() > ctx.config.epsilon {
                let delta_bound = if delta > 0.0 {
                    phi_max - phi_c
                } else {
                    phi_min - phi_c
                };

                if delta_bound.abs() > ctx.config.epsilon {
                    let r_ratio = delta_bound / delta;
                    let phi = Self::van_leer(r_ratio);
                    alpha = alpha.min(phi.clamp(0.0, 2.0) / 2.0);
                } else {
                    alpha = 0.0;
                }
            }
        }

        CellLimitResult::new(alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmod() {
        // 同号
        assert!((MinmodLimiter::minmod(1.0, 2.0) - 1.0).abs() < 1e-10);
        assert!((MinmodLimiter::minmod(-1.0, -2.0) - (-1.0)).abs() < 1e-10);

        // 异号
        assert_eq!(MinmodLimiter::minmod(1.0, -2.0), 0.0);
    }

    #[test]
    fn test_superbee_function() {
        assert_eq!(SuperbeeLimiter::superbee(-1.0), 0.0);
        assert!((SuperbeeLimiter::superbee(0.25) - 0.5).abs() < 1e-10);
        assert!((SuperbeeLimiter::superbee(0.75) - 1.0).abs() < 1e-10);
        assert!((SuperbeeLimiter::superbee(1.5) - 1.5).abs() < 1e-10);
        assert!((SuperbeeLimiter::superbee(3.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_van_leer_function() {
        assert_eq!(VanLeerLimiter::van_leer(-1.0), 0.0);
        assert!(VanLeerLimiter::van_leer(1.0) > 0.0);
        assert!(VanLeerLimiter::van_leer(2.0) > VanLeerLimiter::van_leer(1.0));
    }
}
