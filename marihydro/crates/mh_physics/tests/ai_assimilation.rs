// crates/mh_physics/tests/ai_assimilation.rs

//! AI同化测试
//! 验证Nudging同化的正确性

use mh_physics::assimilation::{PhysicsAssimilable, ConservedQuantities, ConservationChecker, AssimilableBridge};
use mh_physics::state::ShallowWaterState;

fn build_bridge() -> AssimilableBridge<'static> {
    let mut state = Box::new(ShallowWaterState::new(2));
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
    // Box leak to extend lifetime for test scope
    let state_ref: &'static mut ShallowWaterState = Box::leak(state);
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
