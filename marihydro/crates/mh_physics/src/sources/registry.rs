// crates/mh_physics/src/sources/registry.rs

use crate::core::{Backend, DeviceBuffer};
use crate::engine::strategy::workspace::SolverWorkspaceGeneric;
use crate::state::ShallowWaterStateGeneric;
use super::traits::{
    SourceContextGeneric, SourceContributionGeneric, SourceStiffness, SourceTermGeneric,
};
use std::cell::RefCell;
use std::collections::HashMap;

/// 源项注册中心
pub struct SourceRegistry<B: Backend> {
    /// 已注册的源项
    sources: Vec<Box<dyn SourceTermGeneric<B>>>,
    /// 名称到索引的映射
    name_index: HashMap<String, usize>,
    /// 启用状态
    enabled: Vec<bool>,
    /// 并行计算阈值
    #[allow(dead_code)]
    parallel_threshold: usize,
    /// 贡献缓存
    contributions: RefCell<Vec<SourceContributionGeneric<B::Scalar>>>,
}

impl<B: Backend> SourceRegistry<B> {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            name_index: HashMap::new(),
            enabled: Vec::new(),
            parallel_threshold: 512,
            contributions: RefCell::new(Vec::new()),
        }
    }
    
    /// 注册源项
    pub fn register<S: SourceTermGeneric<B> + 'static>(&mut self, source: S) -> usize {
        let name = source.name().to_string();
        let idx = self.sources.len();
        self.sources.push(Box::new(source));
        self.name_index.insert(name, idx);
        self.enabled.push(true);
        idx
    }
    
    /// 按名称获取源项
    pub fn get(&self, name: &str) -> Option<&dyn SourceTermGeneric<B>> {
        self.name_index
            .get(name)
            .and_then(|&idx| self.sources.get(idx))
            .map(|s| s.as_ref())
    }
    
    /// 按名称获取可变源项
    pub fn get_mut(&mut self, name: &str) -> Option<&mut dyn SourceTermGeneric<B>> {
        let idx = *self.name_index.get(name)?;
        Some(self.sources.get_mut(idx)?.as_mut())
    }
    
    /// 启用/禁用源项
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> bool {
        if let Some(&idx) = self.name_index.get(name) {
            if let Some(flag) = self.enabled.get_mut(idx) {
                *flag = enabled;
                return true;
            }
        }
        false
    }
    
    /// 移除源项
    pub fn unregister(&mut self, name: &str) -> bool {
        if let Some(idx) = self.name_index.remove(name) {
            self.sources.swap_remove(idx);
            self.enabled.swap_remove(idx);
            // 重建索引
            self.name_index.clear();
            for (i, s) in self.sources.iter().enumerate() {
                self.name_index.insert(s.name().to_string(), i);
            }
            return true;
        }
        false
    }
    
    /// 获取所有已注册的源项名称
    pub fn list_sources(&self) -> Vec<&str> {
        self.sources.iter().map(|s| s.name()).collect()
    }
    
    /// 累加所有源项贡献到工作区
    pub fn accumulate_all(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        self.accumulate_with_filter(state, workspace, ctx, None);
    }
    
    /// 仅累加显式源项
    pub fn accumulate_explicit(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        self.accumulate_with_filter(state, workspace, ctx, Some(SourceStiffness::Explicit));
    }
    
    /// 仅累加局部隐式源项
    pub fn accumulate_locally_implicit(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        self.accumulate_with_filter(
            state,
            workspace,
            ctx,
            Some(SourceStiffness::LocallyImplicit),
        );
    }
    
    /// 批量计算（并行优化）
    #[allow(dead_code)]
    fn accumulate_parallel(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        contributions: &mut [SourceContributionGeneric<B::Scalar>],
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        for source in &self.sources {
            if !self.is_enabled(source.name()) {
                continue;
            }
            source.compute_batch(state, contributions, ctx);
        }
    }
    
    /// 获取指定刚性类型的源项
    pub fn filter_by_stiffness(
        &self,
        stiffness: SourceStiffness,
    ) -> Vec<&dyn SourceTermGeneric<B>> {
        self.sources
            .iter()
            .filter(|s| s.stiffness() == stiffness)
            .map(|s| s.as_ref())
            .collect()
    }

    fn is_enabled(&self, name: &str) -> bool {
        if let Some(&idx) = self.name_index.get(name) {
            return *self.enabled.get(idx).unwrap_or(&true);
        }
        true
    }

    fn ensure_scratch(&self, n_cells: usize) -> std::cell::RefMut<'_, Vec<SourceContributionGeneric<B::Scalar>>> {
        let mut scratch = self.contributions.borrow_mut();
        if scratch.len() < n_cells {
            scratch.resize(n_cells, SourceContributionGeneric::default());
        }
        scratch
    }

    fn accumulate_with_filter(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
        stiffness_filter: Option<SourceStiffness>,
    ) {
        let n = state.n_cells().min(workspace.n_cells());

        let mut scratch = self.ensure_scratch(n);

        for source in &self.sources {
            if !self.is_enabled(source.name()) {
                continue;
            }
            if let Some(filter) = stiffness_filter {
                if source.stiffness() != filter {
                    continue;
                }
            }

            for c in scratch.iter_mut().take(n) {
                *c = SourceContributionGeneric::default();
            }

            source.compute_batch(state, &mut scratch[..n], ctx);

            // 直接累加到 workspace 的缓冲区
            let h_dst = workspace.flux_h.as_slice_mut();
            let hu_dst = workspace.source_hu.as_slice_mut();
            let hv_dst = workspace.source_hv.as_slice_mut();
            
            for i in 0..n {
                h_dst[i] += scratch[i].s_h;
                hu_dst[i] += scratch[i].s_hu;
                hv_dst[i] += scratch[i].s_hv;
            }
        }
    }
}

impl<B: Backend> Default for SourceRegistry<B> {
    fn default() -> Self { Self::new() }
}
