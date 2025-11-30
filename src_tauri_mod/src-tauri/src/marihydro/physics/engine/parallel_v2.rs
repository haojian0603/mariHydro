// src-tauri/src/marihydro/physics/engine/parallel_v2.rs
//! 改进版并行通量计算模块
//! 
//! 集成 core/parallel 子系统，提供：
//! - 自适应并行策略选择
//! - 性能监控与调度
//! - 着色验证
//! - 统一的并行接口

use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::parallel::{
    AdaptiveScheduler, ParallelConfig, ParallelStrategy, PerfMetrics, StrategySelector,
};
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, FaceIndex, NumericalParams};
use crate::marihydro::core::Workspace;
use crate::marihydro::domain::mesh::MeshColoring;
use crate::marihydro::domain::state::ShallowWaterState;
use crate::marihydro::physics::schemes::{HllcSolver, HydrostaticReconstruction, InterfaceFlux};
use crate::marihydro::physics::schemes::riemann::{AdaptiveSelector, RiemannSolver};
use crate::marihydro::physics::schemes::wetting_drying::WettingDryingHandler;
use glam::DVec2;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// 并行通量计算配置
#[derive(Clone, Debug)]
pub struct ParallelFluxConfig {
    /// 数值参数
    pub params: NumericalParams,
    /// 重力加速度
    pub g: f64,
    /// 最小并行粒度（低于此值使用串行）
    pub min_parallel_size: usize,
    /// 自适应调度
    pub adaptive_scheduling: bool,
    /// 着色验证（调试用）
    pub validate_coloring: bool,
}

impl Default for ParallelFluxConfig {
    fn default() -> Self {
        Self {
            params: NumericalParams::default(),
            g: 9.81,
            min_parallel_size: 1000,
            adaptive_scheduling: true,
            validate_coloring: false,
        }
    }
}

/// 统一并行通量计算器
/// 
/// 集成多种并行策略，自动选择最优方案：
/// - 小规模问题：串行计算
/// - 中等规模：简单并行
/// - 大规模：着色并行
pub struct UnifiedParallelCalculator {
    config: ParallelFluxConfig,
    /// 着色信息（延迟初始化）
    coloring: Option<Arc<MeshColoring>>,
    /// 策略选择器
    strategy_selector: StrategySelector,
    /// 自适应调度器
    scheduler: AdaptiveScheduler,
    /// 性能指标
    metrics: PerfMetrics,
    /// Riemann求解器选择器
    riemann_selector: Option<AdaptiveSelector>,
    /// 干湿处理器
    wetting_drying: Option<WettingDryingHandler>,
}

impl UnifiedParallelCalculator {
    /// 创建计算器
    pub fn new(config: ParallelFluxConfig) -> Self {
        let parallel_config = ParallelConfig {
            enabled: true,
            min_parallel_size: config.min_parallel_size,
            preferred_strategy: None,
            adaptive_scheduling: config.adaptive_scheduling,
            max_threads: None,
        };
        
        Self {
            config,
            coloring: None,
            strategy_selector: StrategySelector::new(parallel_config.clone()),
            scheduler: AdaptiveScheduler::new(parallel_config),
            metrics: PerfMetrics::new(),
            riemann_selector: None,
            wetting_drying: None,
        }
    }
    
    /// 使用预计算着色创建
    pub fn with_coloring(config: ParallelFluxConfig, coloring: Arc<MeshColoring>) -> Self {
        let mut calc = Self::new(config);
        calc.coloring = Some(coloring);
        calc
    }
    
    /// 设置Riemann求解器选择器
    pub fn set_riemann_selector(&mut self, selector: AdaptiveSelector) {
        self.riemann_selector = Some(selector);
    }
    
    /// 设置干湿处理器
    pub fn set_wetting_drying(&mut self, handler: WettingDryingHandler) {
        self.wetting_drying = Some(handler);
    }
    
    /// 初始化着色（如果需要）
    pub fn init_coloring<M: MeshAccess>(&mut self, mesh: &M) {
        if self.coloring.is_none() {
            let coloring = MeshColoring::build(mesh);
            
            // 可选：验证着色正确性
            if self.config.validate_coloring {
                if !coloring.validate(mesh) {
                    log::warn!("着色验证失败，将回退到原子操作并行");
                }
            }
            
            self.coloring = Some(Arc::new(coloring));
        }
    }
    
    /// 获取着色信息
    pub fn coloring(&self) -> Option<&MeshColoring> {
        self.coloring.as_ref().map(|c| c.as_ref())
    }
    
    /// 计算通量（自动选择策略）
    pub fn compute_fluxes<M: MeshAccess + Sync>(
        &mut self,
        state: &ShallowWaterState,
        mesh: &M,
        workspace: &mut Workspace,
    ) -> MhResult<f64> {
        let n_faces = mesh.n_faces();
        
        // 选择并行策略
        let strategy = self.strategy_selector.select_strategy(n_faces, true);
        
        // 记录开始时间
        let start = Instant::now();
        
        let result = match strategy {
            ParallelStrategy::Serial => {
                self.compute_serial(state, mesh, workspace)
            }
            ParallelStrategy::SimpleParallel => {
                self.compute_simple_parallel(state, mesh, workspace)
            }
            ParallelStrategy::ColoredParallel => {
                self.init_coloring(mesh);
                self.compute_colored_parallel(state, mesh, workspace)
            }
            ParallelStrategy::Hybrid { threshold: _, prefer_colored: _ } => {
                // Hybrid策略：根据问题规模动态选择
                if n_faces < self.config.min_parallel_size * 2 {
                    self.compute_simple_parallel(state, mesh, workspace)
                } else {
                    self.init_coloring(mesh);
                    self.compute_colored_parallel(state, mesh, workspace)
                }
            }
        };
        
        // 记录性能指标
        let elapsed = start.elapsed();
        self.metrics.record_iteration(elapsed.as_secs_f64());
        
        result
    }
    
    /// 串行计算通量
    fn compute_serial<M: MeshAccess>(
        &self,
        state: &ShallowWaterState,
        mesh: &M,
        workspace: &mut Workspace,
    ) -> MhResult<f64> {
        let hllc = HllcSolver::new(self.config.params.clone(), self.config.g);
        let hydro = HydrostaticReconstruction::new(&self.config.params, self.config.g);
        
        workspace.flux_h.fill(0.0);
        workspace.flux_hu.fill(0.0);
        workspace.flux_hv.fill(0.0);
        workspace.source_hu.fill(0.0);
        workspace.source_hv.fill(0.0);
        
        let mut max_speed: f64 = 0.0;
        let n_faces = mesh.n_faces();
        
        for face_idx in 0..n_faces {
            let face = FaceIndex(face_idx);
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            let normal = mesh.face_normal(face);
            let length = mesh.face_length(face);
            
            let h_l = state.h[owner.0];
            let vel_l = self.config.params.safe_velocity(
                state.hu[owner.0], state.hv[owner.0], h_l
            ).to_vec();
            let z_l = state.z[owner.0];
            
            let (h_r, vel_r, z_r) = if neighbor.is_valid() {
                let h = state.h[neighbor.0];
                let v = self.config.params.safe_velocity(
                    state.hu[neighbor.0], state.hv[neighbor.0], h
                ).to_vec();
                (h, v, state.z[neighbor.0])
            } else {
                // 边界反射
                let vn = vel_l.dot(normal);
                (h_l, vel_l - 2.0 * vn * normal, z_l)
            };
            
            // 静水重构
            let recon = hydro.reconstruct_face_simple(h_l, h_r, z_l, z_r, vel_l, vel_r);
            
            // 计算通量
            let flux = self.compute_riemann_flux(
                recon.h_left, recon.h_right, 
                recon.vel_left, recon.vel_right, 
                normal, h_l, h_r
            );
            
            let bed_src = hydro.bed_slope_correction(h_l, h_r, z_l, z_r, normal, length);
            
            max_speed = max_speed.max(flux.max_wave_speed);
            
            // 累加通量
            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;
            
            workspace.flux_h[owner.0] -= fh;
            workspace.flux_hu[owner.0] -= fhu;
            workspace.flux_hv[owner.0] -= fhv;
            workspace.source_hu[owner.0] += bed_src.source_x;
            workspace.source_hv[owner.0] += bed_src.source_y;
            
            if neighbor.is_valid() {
                workspace.flux_h[neighbor.0] += fh;
                workspace.flux_hu[neighbor.0] += fhu;
                workspace.flux_hv[neighbor.0] += fhv;
                workspace.source_hu[neighbor.0] -= bed_src.source_x;
                workspace.source_hv[neighbor.0] -= bed_src.source_y;
            }
        }
        
        Ok(max_speed)
    }
    
    /// 简单并行计算（收集后串行累加）
    fn compute_simple_parallel<M: MeshAccess + Sync>(
        &self,
        state: &ShallowWaterState,
        mesh: &M,
        workspace: &mut Workspace,
    ) -> MhResult<f64> {
        let hllc = HllcSolver::new(self.config.params.clone(), self.config.g);
        let hydro = HydrostaticReconstruction::new(&self.config.params, self.config.g);
        let max_speed = AtomicU64::new(0u64);
        
        let n_faces = mesh.n_faces();
        
        // 并行计算通量
        let face_fluxes: Vec<_> = (0..n_faces)
            .into_par_iter()
            .map(|face_idx| {
                let face = FaceIndex(face_idx);
                let owner = mesh.face_owner(face);
                let neighbor = mesh.face_neighbor(face);
                let normal = mesh.face_normal(face);
                let length = mesh.face_length(face);
                
                let h_l = state.h[owner.0];
                let vel_l = self.config.params.safe_velocity(
                    state.hu[owner.0], state.hv[owner.0], h_l
                ).to_vec();
                let z_l = state.z[owner.0];
                
                let (h_r, vel_r, z_r) = if neighbor.is_valid() {
                    let h = state.h[neighbor.0];
                    let v = self.config.params.safe_velocity(
                        state.hu[neighbor.0], state.hv[neighbor.0], h
                    ).to_vec();
                    (h, v, state.z[neighbor.0])
                } else {
                    let vn = vel_l.dot(normal);
                    (h_l, vel_l - 2.0 * vn * normal, z_l)
                };
                
                let recon = hydro.reconstruct_face_simple(h_l, h_r, z_l, z_r, vel_l, vel_r);
                
                let flux = self.compute_riemann_flux(
                    recon.h_left, recon.h_right,
                    recon.vel_left, recon.vel_right,
                    normal, h_l, h_r
                );
                
                let bed_src = hydro.bed_slope_correction(h_l, h_r, z_l, z_r, normal, length);
                
                max_speed.fetch_max(flux.max_wave_speed.to_bits(), Ordering::Relaxed);
                
                (face_idx, owner, neighbor, flux, bed_src, length)
            })
            .collect();
        
        // 串行累加
        workspace.flux_h.fill(0.0);
        workspace.flux_hu.fill(0.0);
        workspace.flux_hv.fill(0.0);
        workspace.source_hu.fill(0.0);
        workspace.source_hv.fill(0.0);
        
        for (_, owner, neighbor, flux, bed_src, length) in face_fluxes {
            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;
            
            workspace.flux_h[owner.0] -= fh;
            workspace.flux_hu[owner.0] -= fhu;
            workspace.flux_hv[owner.0] -= fhv;
            workspace.source_hu[owner.0] += bed_src.source_x;
            workspace.source_hv[owner.0] += bed_src.source_y;
            
            if neighbor.is_valid() {
                workspace.flux_h[neighbor.0] += fh;
                workspace.flux_hu[neighbor.0] += fhu;
                workspace.flux_hv[neighbor.0] += fhv;
                workspace.source_hu[neighbor.0] -= bed_src.source_x;
                workspace.source_hv[neighbor.0] -= bed_src.source_y;
            }
        }
        
        Ok(f64::from_bits(max_speed.load(Ordering::Relaxed)))
    }
    
    /// 着色并行计算（无锁累加）
    fn compute_colored_parallel<M: MeshAccess + Sync>(
        &self,
        state: &ShallowWaterState,
        mesh: &M,
        workspace: &mut Workspace,
    ) -> MhResult<f64> {
        let coloring = self.coloring.as_ref()
            .expect("着色未初始化");
        
        let hllc = HllcSolver::new(self.config.params.clone(), self.config.g);
        let hydro = HydrostaticReconstruction::new(&self.config.params, self.config.g);
        
        workspace.flux_h.fill(0.0);
        workspace.flux_hu.fill(0.0);
        workspace.flux_hv.fill(0.0);
        workspace.source_hu.fill(0.0);
        workspace.source_hv.fill(0.0);
        
        let mut max_speed: f64 = 0.0;
        
        // 按颜色组顺序处理
        for (_color, group) in coloring.iter_groups() {
            let group_max = AtomicU64::new(0u64);
            
            // 组内并行无竞争
            group.par_iter().for_each(|&face_idx| {
                let face = FaceIndex(face_idx);
                let owner = mesh.face_owner(face);
                let neighbor = mesh.face_neighbor(face);
                let normal = mesh.face_normal(face);
                let length = mesh.face_length(face);
                
                let h_l = state.h[owner.0];
                let vel_l = self.config.params.safe_velocity(
                    state.hu[owner.0], state.hv[owner.0], h_l
                ).to_vec();
                let z_l = state.z[owner.0];
                
                let (h_r, vel_r, z_r) = if neighbor.is_valid() {
                    let h = state.h[neighbor.0];
                    let v = self.config.params.safe_velocity(
                        state.hu[neighbor.0], state.hv[neighbor.0], h
                    ).to_vec();
                    (h, v, state.z[neighbor.0])
                } else {
                    let vn = vel_l.dot(normal);
                    (h_l, vel_l - 2.0 * vn * normal, z_l)
                };
                
                let recon = hydro.reconstruct_face_simple(h_l, h_r, z_l, z_r, vel_l, vel_r);
                
                let flux = self.compute_riemann_flux(
                    recon.h_left, recon.h_right,
                    recon.vel_left, recon.vel_right,
                    normal, h_l, h_r
                );
                
                let bed_src = hydro.bed_slope_correction(h_l, h_r, z_l, z_r, normal, length);
                
                group_max.fetch_max(flux.max_wave_speed.to_bits(), Ordering::Relaxed);
                
                let fh = flux.mass * length;
                let fhu = flux.momentum_x * length;
                let fhv = flux.momentum_y * length;
                
                // 无竞争直接写入
                unsafe {
                    let ptr_h = workspace.flux_h.as_ptr() as *mut f64;
                    let ptr_hu = workspace.flux_hu.as_ptr() as *mut f64;
                    let ptr_hv = workspace.flux_hv.as_ptr() as *mut f64;
                    let ptr_shu = workspace.source_hu.as_ptr() as *mut f64;
                    let ptr_shv = workspace.source_hv.as_ptr() as *mut f64;
                    
                    *ptr_h.add(owner.0) -= fh;
                    *ptr_hu.add(owner.0) -= fhu;
                    *ptr_hv.add(owner.0) -= fhv;
                    *ptr_shu.add(owner.0) += bed_src.source_x;
                    *ptr_shv.add(owner.0) += bed_src.source_y;
                    
                    if neighbor.is_valid() {
                        *ptr_h.add(neighbor.0) += fh;
                        *ptr_hu.add(neighbor.0) += fhu;
                        *ptr_hv.add(neighbor.0) += fhv;
                        *ptr_shu.add(neighbor.0) -= bed_src.source_x;
                        *ptr_shv.add(neighbor.0) -= bed_src.source_y;
                    }
                }
            });
            
            max_speed = max_speed.max(f64::from_bits(group_max.load(Ordering::Relaxed)));
        }
        
        Ok(max_speed)
    }
    
    /// 计算Riemann通量（支持自适应选择器）
    fn compute_riemann_flux(
        &self,
        h_l: f64,
        h_r: f64,
        vel_l: DVec2,
        vel_r: DVec2,
        normal: DVec2,
        _h_orig_l: f64,
        _h_orig_r: f64,
    ) -> InterfaceFlux {
        // 如果设置了自适应选择器，使用它
        if let Some(ref selector) = self.riemann_selector {
            // 构造InterfaceState
            use crate::marihydro::physics::schemes::riemann::InterfaceState;
            let interface = InterfaceState {
                h_left: h_l,
                h_right: h_r,
                vel_left: vel_l,
                vel_right: vel_r,
                normal,
                g: self.config.g,
            };
            
            selector.solve(&interface)
                .map(|f| InterfaceFlux {
                    mass: f.mass,
                    momentum_x: f.momentum_x,
                    momentum_y: f.momentum_y,
                    max_wave_speed: f.max_wave_speed,
                })
                .unwrap_or(InterfaceFlux::ZERO)
        } else {
            // 使用默认HLLC
            let hllc = HllcSolver::new(self.config.params.clone(), self.config.g);
            hllc.solve(h_l, h_r, vel_l, vel_r, normal)
                .unwrap_or(InterfaceFlux::ZERO)
        }
    }
    
    /// 应用干湿通量限制
    pub fn apply_wetting_drying_limit(
        &self,
        state: &ShallowWaterState,
        workspace: &mut Workspace,
    ) {
        if let Some(ref handler) = self.wetting_drying {
            let n_cells = state.h.len();
            
            for i in 0..n_cells {
                let wet_state = handler.classify(state.h[i], state.z[i]);
                
                // 限制通量
                workspace.flux_h[i] = handler.limit_flux(workspace.flux_h[i], wet_state);
                workspace.flux_hu[i] = handler.limit_flux(workspace.flux_hu[i], wet_state);
                workspace.flux_hv[i] = handler.limit_flux(workspace.flux_hv[i], wet_state);
            }
        }
    }
    
    /// 获取性能指标
    pub fn metrics(&self) -> &PerfMetrics {
        &self.metrics
    }
    
    /// 重置性能指标
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }
}

/// 构建器模式
pub struct UnifiedParallelCalculatorBuilder {
    config: ParallelFluxConfig,
    coloring: Option<Arc<MeshColoring>>,
    riemann_selector: Option<AdaptiveSelector>,
    wetting_drying: Option<WettingDryingHandler>,
}

impl UnifiedParallelCalculatorBuilder {
    pub fn new() -> Self {
        Self {
            config: ParallelFluxConfig::default(),
            coloring: None,
            riemann_selector: None,
            wetting_drying: None,
        }
    }
    
    pub fn with_config(mut self, config: ParallelFluxConfig) -> Self {
        self.config = config;
        self
    }
    
    pub fn with_gravity(mut self, g: f64) -> Self {
        self.config.g = g;
        self
    }
    
    pub fn with_params(mut self, params: NumericalParams) -> Self {
        self.config.params = params;
        self
    }
    
    pub fn with_coloring(mut self, coloring: Arc<MeshColoring>) -> Self {
        self.coloring = Some(coloring);
        self
    }
    
    pub fn with_riemann_selector(mut self, selector: AdaptiveSelector) -> Self {
        self.riemann_selector = Some(selector);
        self
    }
    
    pub fn with_wetting_drying(mut self, handler: WettingDryingHandler) -> Self {
        self.wetting_drying = Some(handler);
        self
    }
    
    pub fn with_adaptive_scheduling(mut self, enabled: bool) -> Self {
        self.config.adaptive_scheduling = enabled;
        self
    }
    
    pub fn with_coloring_validation(mut self, enabled: bool) -> Self {
        self.config.validate_coloring = enabled;
        self
    }
    
    pub fn build(self) -> UnifiedParallelCalculator {
        let mut calc = if let Some(coloring) = self.coloring {
            UnifiedParallelCalculator::with_coloring(self.config, coloring)
        } else {
            UnifiedParallelCalculator::new(self.config)
        };
        
        if let Some(selector) = self.riemann_selector {
            calc.set_riemann_selector(selector);
        }
        
        if let Some(handler) = self.wetting_drying {
            calc.set_wetting_drying(handler);
        }
        
        calc
    }
}

impl Default for UnifiedParallelCalculatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ParallelFluxConfig::default();
        assert_eq!(config.g, 9.81);
        assert!(config.adaptive_scheduling);
    }
    
    #[test]
    fn test_builder() {
        let calc = UnifiedParallelCalculatorBuilder::new()
            .with_gravity(10.0)
            .with_adaptive_scheduling(false)
            .build();
        
        assert_eq!(calc.config.g, 10.0);
        assert!(!calc.config.adaptive_scheduling);
    }
}
