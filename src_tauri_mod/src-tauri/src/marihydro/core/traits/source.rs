// src-tauri/src/marihydro/core/traits/source.rs

//! 源项接口
//!
//! 定义各类源项（摩擦、科氏力、大气压、扩散等）的统一抽象。

use super::mesh::MeshAccess;
use super::state::StateAccess;
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::types::NumericalParams;
use crate::marihydro::core::Workspace;

/// 源项贡献
#[derive(Debug, Clone, Copy, Default)]
pub struct SourceContribution {
    /// 质量源 [m/s]
    pub s_h: f64,
    /// x动量源 [m²/s²]
    pub s_hu: f64,
    /// y动量源 [m²/s²]
    pub s_hv: f64,
}

impl SourceContribution {
    pub const ZERO: Self = Self {
        s_h: 0.0,
        s_hu: 0.0,
        s_hv: 0.0,
    };

    pub fn add(&self, other: &Self) -> Self {
        Self {
            s_h: self.s_h + other.s_h,
            s_hu: self.s_hu + other.s_hu,
            s_hv: self.s_hv + other.s_hv,
        }
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self {
            s_h: self.s_h * factor,
            s_hu: self.s_hu * factor,
            s_hv: self.s_hv * factor,
        }
    }
}

/// 源项计算上下文
///
/// 包含源项计算所需的所有外部信息
pub struct SourceContext<'a> {
    /// 当前模拟时间 [s]
    pub time: f64,
    /// 时间步长 [s]
    pub dt: f64,
    /// 数值参数
    pub params: &'a NumericalParams,
    /// 工作区缓冲区
    pub workspace: &'a Workspace,
}

/// 源项计算接口
///
/// # 设计原则
///
/// 1. 每个源项独立实现，可组合使用
/// 2. 支持批量计算以优化性能
/// 3. 源项可选择性启用/禁用
pub trait SourceTerm: Send + Sync {
    /// 源项名称
    fn name(&self) -> &'static str;

    /// 计算单个单元的源项
    fn compute_cell<M: MeshAccess, S: StateAccess>(
        &self,
        cell_idx: usize,
        mesh: &M,
        state: &S,
        ctx: &SourceContext,
    ) -> SourceContribution;

    /// 批量计算所有单元的源项（累加到输出缓冲区）
    fn compute_all<M: MeshAccess, S: StateAccess>(
        &self,
        mesh: &M,
        state: &S,
        ctx: &SourceContext,
        output_h: &mut [f64],
        output_hu: &mut [f64],
        output_hv: &mut [f64],
    ) -> MhResult<()> {
        let n_cells = mesh.n_cells();

        if output_h.len() != n_cells || output_hu.len() != n_cells || output_hv.len() != n_cells {
            return Err(crate::marihydro::core::MhError::size_mismatch(
                "source output arrays",
                n_cells,
                output_h.len(),
            ));
        }

        for i in 0..n_cells {
            let contrib = self.compute_cell(i, mesh, state, ctx);
            output_h[i] += contrib.s_h;
            output_hu[i] += contrib.s_hu;
            output_hv[i] += contrib.s_hv;
        }

        Ok(())
    }

    /// 是否启用
    fn is_enabled(&self) -> bool {
        true
    }

    /// 源项是否显式（需要 CFL 限制）
    fn is_explicit(&self) -> bool {
        true
    }

    /// 是否需要隐式处理（如摩擦项）
    fn requires_implicit_treatment(&self) -> bool {
        false
    }
}

/// 源项聚合器
///
/// 管理多个源项并统一计算
pub struct SourceTermAggregator {
    sources: Vec<Box<dyn SourceTerm>>,
}

impl SourceTermAggregator {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    pub fn add(&mut self, source: Box<dyn SourceTerm>) {
        self.sources.push(source);
    }

    pub fn compute_all<M: MeshAccess, S: StateAccess>(
        &self,
        mesh: &M,
        state: &S,
        ctx: &SourceContext,
        output_h: &mut [f64],
        output_hu: &mut [f64],
        output_hv: &mut [f64],
    ) -> MhResult<()> {
        // 清零输出
        output_h.fill(0.0);
        output_hu.fill(0.0);
        output_hv.fill(0.0);

        // 累加所有启用的源项
        for source in &self.sources {
            if source.is_enabled() {
                source.compute_all(mesh, state, ctx, output_h, output_hu, output_hv)?;
            }
        }

        Ok(())
    }

    pub fn enabled_names(&self) -> Vec<&'static str> {
        self.sources
            .iter()
            .filter(|s| s.is_enabled())
            .map(|s| s.name())
            .collect()
    }
}

impl Default for SourceTermAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_contribution() {
        let s1 = SourceContribution {
            s_h: 1.0,
            s_hu: 2.0,
            s_hv: 3.0,
        };
        let s2 = SourceContribution {
            s_h: 0.5,
            s_hu: 1.0,
            s_hv: 1.5,
        };

        let sum = s1.add(&s2);
        assert!((sum.s_h - 1.5).abs() < 1e-10);
        assert!((sum.s_hu - 3.0).abs() < 1e-10);
    }
}
