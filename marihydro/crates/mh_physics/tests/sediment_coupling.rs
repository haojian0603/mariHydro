// crates/mh_physics/tests/sediment_coupling.rs

//! 泥沙耦合测试
//! 验证泥沙系统的质量守恒

use mh_physics::sediment::{SedimentManagerGeneric, SedimentConfigGeneric};
use mh_physics::core::CpuBackend;
use mh_physics::state::ShallowWaterStateGeneric;

/// 测试泥沙管理器创建
#[test]
fn test_sediment_manager_creation() {
    let manager = SedimentManagerGeneric::<CpuBackend<f64>>::new(
        8,
        SedimentConfigGeneric::default(),
    );
    assert_eq!(manager.state().n_cells(), 8);
}

/// 测试质量守恒（零通量场景）
#[test]
fn test_mass_conservation() {
    let config = SedimentConfigGeneric::default();
    let mut manager = SedimentManagerGeneric::<CpuBackend<f64>>::new(4, config);
    let mut state = ShallowWaterStateGeneric::<CpuBackend<f64>>::new(4);
    for h in state.h.iter_mut() { *h = 1.0; }

    // 初始床面质量
    manager.set_initial_bed_mass(&[1.0, 1.0, 1.0, 1.0]);
    manager.set_initial_concentration(&[0.0, 0.0, 0.0, 0.0]);

    let cell_areas = vec![1.0; 4];
    let before: f64 = manager.state().bed_mass.iter().sum();
    let _ = manager.step(&state, &cell_areas, 1.0);
    let after: f64 = manager.state().bed_mass.iter().sum();
    assert!((after - before).abs() < 1e-6);
}

/// 测试侵蚀/沉降平衡（零浓度）
#[test]
fn test_erosion_deposition_balance() {
    let mut manager = SedimentManagerGeneric::<CpuBackend<f64>>::new(
        2,
        SedimentConfigGeneric::default(),
    );
    let mut state = ShallowWaterStateGeneric::<CpuBackend<f64>>::new(2);
    for h in state.h.iter_mut() { *h = 1.0; }
    manager.set_initial_concentration(&[0.0, 0.0]);

    manager.compute_exchange_flux(&state);
    let flux = manager.exchange_flux();
    assert!(flux.iter().all(|&f| f == 0.0));
}
