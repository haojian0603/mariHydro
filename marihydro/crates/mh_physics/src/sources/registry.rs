// crates/mh_physics/src/sources/registry.rs
//
// 泛型主链源项注册中心。
// legacy CPU/f64 桥接接口不在这里扩展；新的收敛工作都应落在泛型主链。

use crate::core::Backend;
use crate::engine::strategy::workspace::SolverWorkspaceGeneric;
use crate::state::ShallowWaterState;
use super::traits::{SourceContributionGeneric, SourceContextGeneric, SourceStiffness, SourceTermGeneric};
use mh_runtime::DeviceBuffer;
use std::cell::RefCell;
use std::collections::HashMap;

/// Source registry with generic source terms.
pub struct SourceRegistry<B: Backend, S: SourceTermGeneric<B>> {
    sources: Vec<S>,
    name_index: HashMap<String, usize>,
    enabled: Vec<bool>,
    contributions: RefCell<Vec<SourceContributionGeneric<B::Scalar>>>,
}

impl<B: Backend, S: SourceTermGeneric<B>> SourceRegistry<B, S> {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            name_index: HashMap::new(),
            enabled: Vec::new(),
            contributions: RefCell::new(Vec::new()),
        }
    }

    pub fn register(&mut self, source: S) -> usize {
        let name = source.name().to_string();
        let idx = self.sources.len();
        self.sources.push(source);
        self.name_index.insert(name, idx);
        self.enabled.push(true);
        idx
    }

    pub fn get(&self, name: &str) -> Option<&S> {
        self.name_index.get(name).and_then(|&idx| self.sources.get(idx))
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut S> {
        let idx = *self.name_index.get(name)?;
        self.sources.get_mut(idx)
    }

    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> bool {
        if let Some(&idx) = self.name_index.get(name) {
            if let Some(flag) = self.enabled.get_mut(idx) {
                *flag = enabled;
                return true;
            }
        }
        false
    }

    pub fn unregister(&mut self, name: &str) -> bool {
        let Some(idx) = self.name_index.remove(name) else {
            return false;
        };

        self.sources.swap_remove(idx);
        self.enabled.swap_remove(idx);
        if idx < self.sources.len() {
            self.name_index.insert(self.sources[idx].name().to_string(), idx);
        }
        true
    }

    pub fn list_sources(&self) -> Vec<&str> {
        self.sources.iter().map(|s| s.name()).collect()
    }

    pub fn accumulate_all(
        &self,
        state: &ShallowWaterState<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        self.accumulate_with_filter(state, workspace, ctx, None);
    }

    pub fn accumulate_explicit(
        &self,
        state: &ShallowWaterState<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        self.accumulate_with_filter(state, workspace, ctx, Some(SourceStiffness::Explicit));
    }

    pub fn accumulate_locally_implicit(
        &self,
        state: &ShallowWaterState<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        self.accumulate_with_filter(state, workspace, ctx, Some(SourceStiffness::LocallyImplicit));
    }

    pub fn filter_by_stiffness(&self, stiffness: SourceStiffness) -> Vec<&S> {
        self.sources.iter().filter(|s| s.stiffness() == stiffness).collect()
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
        state: &ShallowWaterState<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
        stiffness_filter: Option<SourceStiffness>,
    ) {
        if self.sources.is_empty() {
            return;
        }
        let n = state.n_cells().min(workspace.n_cells());
        let mut scratch = self.ensure_scratch(n);

        for (idx, source) in self.sources.iter().enumerate() {
            if !self.enabled.get(idx).copied().unwrap_or(false) {
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

            let h_dst = workspace.flux_h.as_slice_mut();
            let hu_dst = workspace.source_hu.as_slice_mut();
            let hv_dst = workspace.source_hv.as_slice_mut();
            let n = n.min(h_dst.len()).min(hu_dst.len()).min(hv_dst.len());

            for i in 0..n {
                h_dst[i] += scratch[i].s_h;
                hu_dst[i] += scratch[i].s_hu;
                hv_dst[i] += scratch[i].s_hv;
            }
        }
    }
}

impl<B: Backend, S: SourceTermGeneric<B>> Default for SourceRegistry<B, S> {
    fn default() -> Self {
        Self::new()
    }
}
