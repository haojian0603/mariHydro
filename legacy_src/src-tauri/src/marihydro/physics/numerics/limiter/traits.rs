// src-tauri/src/marihydro/physics/numerics/limiter/traits.rs

//! 梯度限制器统一接口
//!
//! 定义所有梯度限制器必须实现的 trait。

use super::super::gradient::ScalarGradientStorage;
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::CellIndex;
use glam::DVec2;

/// 限制器能力标志
#[derive(Debug, Clone, Copy, Default)]
pub struct LimiterCapabilities {
    /// 是否支持并行计算
    pub parallel: bool,
    /// 是否平滑（避免收敛问题）
    pub smooth: bool,
    /// 是否严格保单调
    pub monotone: bool,
    /// 理论精度阶数
    pub order: u8,
}

impl LimiterCapabilities {
    /// 基本能力（不平滑，严格单调）
    pub fn basic() -> Self {
        Self {
            parallel: false,
            smooth: false,
            monotone: true,
            order: 1,
        }
    }

    /// 平滑能力（用于稳态求解）
    pub fn smooth() -> Self {
        Self {
            parallel: true,
            smooth: true,
            monotone: false,
            order: 2,
        }
    }
}

/// 限制器配置
#[derive(Debug, Clone, Copy)]
pub struct LimiterConfig {
    /// 是否启用并行
    pub parallel: bool,
    /// 并行化阈值（单元数）
    pub parallel_threshold: usize,
    /// 数值稳定化小量
    pub epsilon: f64,
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            parallel_threshold: 1000,
            epsilon: 1e-14,
        }
    }
}

/// 限制器上下文（传递给限制器的额外信息）
#[derive(Debug, Clone)]
pub struct LimiterContext<'a> {
    /// 标量场值
    pub field: &'a [f64],
    /// 梯度存储
    pub gradient: &'a ScalarGradientStorage,
    /// 每个单元的特征尺寸（可选）
    pub cell_sizes: Option<&'a [f64]>,
    /// 配置
    pub config: LimiterConfig,
}

impl<'a> LimiterContext<'a> {
    /// 创建基本上下文
    pub fn new(field: &'a [f64], gradient: &'a ScalarGradientStorage) -> Self {
        Self {
            field,
            gradient,
            cell_sizes: None,
            config: LimiterConfig::default(),
        }
    }

    /// 添加单元尺寸
    pub fn with_cell_sizes(mut self, sizes: &'a [f64]) -> Self {
        self.cell_sizes = Some(sizes);
        self
    }

    /// 设置配置
    pub fn with_config(mut self, config: LimiterConfig) -> Self {
        self.config = config;
        self
    }
}

/// 单元限制结果
#[derive(Debug, Clone, Copy)]
pub struct CellLimitResult {
    /// 限制因子 [0, 1]
    pub factor: f64,
    /// 是否被限制
    pub was_limited: bool,
}

impl CellLimitResult {
    /// 未限制（因子 = 1）
    pub const UNLIMITED: Self = Self {
        factor: 1.0,
        was_limited: false,
    };

    /// 完全限制（因子 = 0）
    pub const FULLY_LIMITED: Self = Self {
        factor: 0.0,
        was_limited: true,
    };

    /// 创建限制结果
    pub fn new(factor: f64) -> Self {
        Self {
            factor: factor.clamp(0.0, 1.0),
            was_limited: factor < 1.0 - 1e-10,
        }
    }
}

/// 梯度限制器 trait
///
/// 所有梯度限制器都必须实现此 trait。
pub trait Limiter: Send + Sync {
    /// 限制器名称
    fn name(&self) -> &'static str;

    /// 限制器能力
    fn capabilities(&self) -> LimiterCapabilities;

    /// 计算单个单元的限制因子
    fn limit_cell<M: MeshAccess>(
        &self,
        cell: CellIndex,
        ctx: &LimiterContext,
        mesh: &M,
    ) -> CellLimitResult;

    /// 批量计算限制因子
    fn compute_limiters<M: MeshAccess>(
        &self,
        ctx: &LimiterContext,
        mesh: &M,
        output: &mut [f64],
    ) -> MhResult<()> {
        for i in 0..mesh.n_cells() {
            let result = self.limit_cell(CellIndex(i), ctx, mesh);
            output[i] = result.factor;
        }
        Ok(())
    }

    /// 应用限制器到梯度（原地修改）
    fn apply<M: MeshAccess>(
        &self,
        field: &[f64],
        gradient: &mut ScalarGradientStorage,
        mesh: &M,
    ) -> MhResult<()> {
        let ctx = LimiterContext::new(field, gradient);
        let mut limiters = vec![1.0; mesh.n_cells()];
        self.compute_limiters(&ctx, mesh, &mut limiters)?;
        gradient.apply_limiter(&limiters);
        Ok(())
    }

    /// 获取限制统计信息
    fn statistics<M: MeshAccess>(
        &self,
        ctx: &LimiterContext,
        mesh: &M,
    ) -> LimiterStatistics {
        let n = mesh.n_cells();
        let mut min_factor = 1.0f64;
        let mut max_factor = 0.0f64;
        let mut sum_factor = 0.0;
        let mut limited_count = 0usize;

        for i in 0..n {
            let result = self.limit_cell(CellIndex(i), ctx, mesh);
            min_factor = min_factor.min(result.factor);
            max_factor = max_factor.max(result.factor);
            sum_factor += result.factor;
            if result.was_limited {
                limited_count += 1;
            }
        }

        LimiterStatistics {
            total_cells: n,
            limited_cells: limited_count,
            min_factor,
            max_factor,
            avg_factor: if n > 0 { sum_factor / n as f64 } else { 1.0 },
        }
    }
}

/// 限制器统计信息
#[derive(Debug, Clone, Copy)]
pub struct LimiterStatistics {
    /// 总单元数
    pub total_cells: usize,
    /// 被限制的单元数
    pub limited_cells: usize,
    /// 最小限制因子
    pub min_factor: f64,
    /// 最大限制因子
    pub max_factor: f64,
    /// 平均限制因子
    pub avg_factor: f64,
}

impl LimiterStatistics {
    /// 被限制的单元比例
    pub fn limited_ratio(&self) -> f64 {
        if self.total_cells > 0 {
            self.limited_cells as f64 / self.total_cells as f64
        } else {
            0.0
        }
    }
}

/// 无限制器（始终返回 1.0）
pub struct NoLimiter;

impl Limiter for NoLimiter {
    fn name(&self) -> &'static str {
        "None"
    }

    fn capabilities(&self) -> LimiterCapabilities {
        LimiterCapabilities {
            parallel: true,
            smooth: true,
            monotone: false,
            order: 0,
        }
    }

    fn limit_cell<M: MeshAccess>(
        &self,
        _cell: CellIndex,
        _ctx: &LimiterContext,
        _mesh: &M,
    ) -> CellLimitResult {
        CellLimitResult::UNLIMITED
    }

    fn compute_limiters<M: MeshAccess>(
        &self,
        _ctx: &LimiterContext,
        mesh: &M,
        output: &mut [f64],
    ) -> MhResult<()> {
        output[..mesh.n_cells()].fill(1.0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_limit_result() {
        let r = CellLimitResult::new(0.5);
        assert!(r.was_limited);
        assert!((r.factor - 0.5).abs() < 1e-10);

        let r = CellLimitResult::new(1.0);
        assert!(!r.was_limited);

        let r = CellLimitResult::new(1.5); // 超出范围
        assert!((r.factor - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_limiter_statistics() {
        let stats = LimiterStatistics {
            total_cells: 100,
            limited_cells: 25,
            min_factor: 0.1,
            max_factor: 1.0,
            avg_factor: 0.8,
        };
        assert!((stats.limited_ratio() - 0.25).abs() < 1e-10);
    }
}
