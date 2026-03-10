// crates/mh_physics/src/sources/registry.rs

use crate::core::{Backend, DeviceBuffer};
use crate::engine::strategy::workspace::SolverWorkspaceGeneric;
use crate::state::ShallowWaterStateGeneric;
use super::traits::{
    SourceContextGeneric, SourceContributionGeneric, SourceStiffness, SourceTermGeneric,
};
use std::cell::RefCell;
use std::collections::HashMap;

/// 婧愰」娉ㄥ唽涓績
pub struct SourceRegistry<B: Backend> {
    /// 宸叉敞鍐岀殑婧愰」
    sources: Vec<Box<dyn SourceTermGeneric<B>>>,
    /// 鍚嶇О鍒扮储寮曠殑鏄犲皠
    name_index: HashMap<String, usize>,
    /// 鍚敤鐘舵€?
    enabled: Vec<bool>,
    /// 璐＄尞缂撳瓨
    contributions: RefCell<Vec<SourceContributionGeneric<B::Scalar>>>,
}

impl<B: Backend> SourceRegistry<B> {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            name_index: HashMap::new(),
            enabled: Vec::new(),
            contributions: RefCell::new(Vec::new()),
        }
    }
    
    /// 娉ㄥ唽婧愰」
    pub fn register<S: SourceTermGeneric<B> + 'static>(&mut self, source: S) -> usize {
        let name = source.name().to_string();
        let idx = self.sources.len();
        self.sources.push(Box::new(source));
        self.name_index.insert(name, idx);
        self.enabled.push(true);
        idx
    }
    
    /// 鎸夊悕绉拌幏鍙栨簮椤?
    pub fn get(&self, name: &str) -> Option<&dyn SourceTermGeneric<B>> {
        self.name_index
            .get(name)
            .and_then(|&idx| self.sources.get(idx))
            .map(|s| s.as_ref())
    }
    
    /// 鎸夊悕绉拌幏鍙栧彲鍙樻簮椤?
    pub fn get_mut(&mut self, name: &str) -> Option<&mut dyn SourceTermGeneric<B>> {
        let idx = *self.name_index.get(name)?;
        Some(self.sources.get_mut(idx)?.as_mut())
    }
    
    /// 鍚敤/绂佺敤婧愰」
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> bool {
        if let Some(&idx) = self.name_index.get(name) {
            if let Some(flag) = self.enabled.get_mut(idx) {
                *flag = enabled;
                return true;
            }
        }
        false
    }
    
    /// 绉婚櫎婧愰」
    pub fn unregister(&mut self, name: &str) -> bool {
        if let Some(idx) = self.name_index.remove(name) {
            self.sources.swap_remove(idx);
            self.enabled.swap_remove(idx);
            // 閲嶅缓绱㈠紩
            self.name_index.clear();
            for (i, s) in self.sources.iter().enumerate() {
                self.name_index.insert(s.name().to_string(), i);
            }
            return true;
        }
        false
    }
    
    /// 鑾峰彇鎵€鏈夊凡娉ㄥ唽鐨勬簮椤瑰悕绉?
    pub fn list_sources(&self) -> Vec<&str> {
        self.sources.iter().map(|s| s.name()).collect()
    }
    
    /// 绱姞鎵€鏈夋簮椤硅础鐚埌宸ヤ綔鍖?
    pub fn accumulate_all(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        self.accumulate_with_filter(state, workspace, ctx, None);
    }
    
    /// 浠呯疮鍔犳樉寮忔簮椤?
    pub fn accumulate_explicit(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        self.accumulate_with_filter(state, workspace, ctx, Some(SourceStiffness::Explicit));
    }
    
    /// 浠呯疮鍔犲眬閮ㄩ殣寮忔簮椤?
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
    
    
    /// 鑾峰彇鎸囧畾鍒氭€х被鍨嬬殑婧愰」
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

            // 鐩存帴绱姞鍒?workspace 鐨勭紦鍐插尯
            if let (Some(h_dst), Some(hu_dst), Some(hv_dst)) = (
                workspace.flux_h.as_slice_mut(),
                workspace.source_hu.as_slice_mut(),
                workspace.source_hv.as_slice_mut(),
            ) {
                for i in 0..n {
                    h_dst[i] += scratch[i].s_h;
                    hu_dst[i] += scratch[i].s_hu;
                    hv_dst[i] += scratch[i].s_hv;
                }
            } else {
                // 鍥為€€璺緞锛氫娇鐢?copy_to_vec/copy_from_slice
                let mut h_host = workspace.flux_h.copy_to_vec();
                let mut hu_host = workspace.source_hu.copy_to_vec();
                let mut hv_host = workspace.source_hv.copy_to_vec();
                for i in 0..n {
                    h_host[i] += scratch[i].s_h;
                    hu_host[i] += scratch[i].s_hu;
                    hv_host[i] += scratch[i].s_hv;
                }
                workspace.flux_h.copy_from_slice(&h_host);
                workspace.source_hu.copy_from_slice(&hu_host);
                workspace.source_hv.copy_from_slice(&hv_host);
            }
        }
    }
}

impl<B: Backend> Default for SourceRegistry<B> {
    fn default() -> Self { Self::new() }
}
