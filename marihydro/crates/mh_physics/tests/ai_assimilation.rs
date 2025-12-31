// crates/mh_physics/tests/ai_assimilation.rs
//! AI同化测试
//!
//! 验证Nudging同化的正确性

use mh_physics::assimilation::{PhysicsAssimilable, ConservedQuantities, ConservationChecker, AssimilableBridge};
use mh_physics::state::ShallowWaterState;
use mh_runtime::CpuBackend;
use std::sync::LazyLock;

/// 全局Backend实例
static BACKEND: LazyLock<CpuBackend<f64>> = LazyLock::new(|| CpuBackend::<f64>::new());

/// 测试辅助模块
#[cfg(test)]
mod test_harness {
    use super::*;

    pub fn test_backend() -> CpuBackend<f64> {
        *BACKEND
    }

    pub fn create_test_state(n_cells: usize) -> ShallowWaterState<CpuBackend<f64>> {
        ShallowWaterState::new_with_backend(test_backend(), n_cells)
    }

    /// 内存泄漏警告 - 仅限测试使用
    pub fn leak_state(state: ShallowWaterState<CpuBackend<f64>>) -> &'static mut ShallowWaterState<CpuBackend<f64>> {
        eprintln!("警告: 测试场景内存泄漏 - 仅限短期测试");
        Box::leak(Box::new(state))
    }
}

use test_harness::{create_test_state, leak_state};

fn build_bridge() -> AssimilableBridge<'static> {
    let mut state = create_test_state(2);
    
    // 初始化水体数据
    state.h[0] = 1.0;
    state.h[1] = 2.0;
    state.hu[0] = 0.5;
    state.hu[1] = 0.5;
    state.hv[0] = 0.0;
    state.hv[1] = 0.0;
    state.z[0] = 0.0;
    state.z[1] = 0.0;
    
    let areas = vec![1.0, 1.0];
    let centers = vec![[0.0, 0.0], [1.0, 0.0]];
    
    let state_ref = leak_state(state);
    AssimilableBridge::new(state_ref, areas, centers)
}

#[test]
fn test_conserved_quantities() {
    let mut bridge = build_bridge();
    let conserved = ConservedQuantities::compute(&mut bridge as &mut dyn PhysicsAssimilable);
    let expected_mass = 1000.0 * (1.0 + 2.0);
    assert!((conserved.total_mass - expected_mass).abs() < 1e-6);
}

#[test]
fn test_conservation_check() {
    let mut bridge = build_bridge();
    let reference = bridge.compute_conserved();
    let mut checker = ConservationChecker::new(reference.clone(), 1e-6);

    // 轻微扰动
    bridge.get_depth_mut()[0] += 0.1;
    let error = checker.check(&mut bridge as &mut dyn PhysicsAssimilable, 0.0);
    assert!(error.mass_error.abs() > 0.0);
    assert!(checker.max_error().is_some());
}