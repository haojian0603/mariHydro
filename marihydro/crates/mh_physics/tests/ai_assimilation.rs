// crates/mh_physics/tests/ai_assimilation.rs

//! AI同化测试（Backend架构强制版）
//!
//! 验证Nudging同化的正确性

use mh_physics::assimilation::{PhysicsAssimilable, ConservedQuantities, ConservationChecker, AssimilableBridge};
use mh_physics::state::ShallowWaterState;
use mh_runtime::CpuBackend;

// ============================================
// 🔥 强制测试辅助模块（所有测试必须复用）
// ============================================

#[cfg(test)]
mod test_harness {
    use super::*;

    /// Backend 单例（测试生命周期内仅创建一次）
    pub fn test_backend() -> CpuBackend<f64> {
        CpuBackend::<f64>::new()
    }

    /// 统一状态创建入口（禁止直接调用 new）
    pub fn create_test_state(n_cells: usize) -> ShallowWaterState<CpuBackend<f64>> {
        let backend = test_backend();
        ShallowWaterState::new_with_backend(backend, n_cells)
    }

    /// Box泄漏辅助（带内存泄漏警告）
    pub fn leak_state(state: ShallowWaterState<CpuBackend<f64>>) -> &'static mut ShallowWaterState<CpuBackend<f64>> {
        eprintln!("⚠️  测试场景内存泄漏 - 仅限短期测试");
        Box::leak(Box::new(state))
    }
}

// ============================================
// 测试用例（强制使用测试辅助）
// ============================================

use test_harness::{create_test_state, leak_state};

fn build_bridge() -> AssimilableBridge<'static> {
    // 🔥 强制使用辅助函数
    let mut state = create_test_state(2);
    
    // 数据初始化
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
    
    // 🔥 使用带警告的泄漏辅助
    let state_ref = leak_state(state);
    AssimilableBridge::new(state_ref, areas, centers)
}

/// 测试守恒量计算
#[test]
fn test_conserved_quantities() {
    let mut bridge = build_bridge();
    let conserved = ConservedQuantities::compute(&mut bridge as &mut dyn PhysicsAssimilable);
    let expected_mass = 1000.0 * (1.0 + 2.0);
    assert!((conserved.total_mass - expected_mass).abs() < 1e-6);
}

/// 测试守恒校验
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
