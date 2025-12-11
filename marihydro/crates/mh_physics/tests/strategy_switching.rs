// crates/mh_physics/tests/strategy_switching.rs

//! 策略切换测试
//! 验证显式/半隐式策略的切换和状态连续性

use mh_physics::engine::strategy::{
    TimeIntegrationStrategy, ExplicitStrategy, SemiImplicitStrategyGeneric,
    ExplicitConfig, SemiImplicitConfig,
};
use mh_physics::core::CpuBackend;

/// 测试策略可以被创建
#[test]
#[allow(deprecated)]
fn test_strategy_creation() {
    let _explicit: ExplicitStrategy<CpuBackend<f64>> = ExplicitStrategy::new(ExplicitConfig::default());
    let _semi_implicit = SemiImplicitStrategyGeneric::<CpuBackend<f64>>::new(
        100, // n_cells
        SemiImplicitConfig::default()
    );
}

/// 测试策略名称
#[test]
#[allow(deprecated)]
fn test_strategy_names() {
    let explicit: ExplicitStrategy<CpuBackend<f64>> = ExplicitStrategy::new(ExplicitConfig::default());
    assert!(!explicit.name().is_empty());
    
    let semi_implicit = SemiImplicitStrategyGeneric::<CpuBackend<f64>>::new(
        100, // n_cells
        SemiImplicitConfig::default()
    );
    assert!(!semi_implicit.name().is_empty());
}

/// 测试策略CFL支持
#[test]
#[allow(deprecated)]
fn test_cfl_support() {
    let explicit: ExplicitStrategy<CpuBackend<f64>> = ExplicitStrategy::new(ExplicitConfig::default());
    assert!(!explicit.supports_large_cfl());
    
    let semi_implicit = SemiImplicitStrategyGeneric::<CpuBackend<f64>>::new(
        100, // n_cells
        SemiImplicitConfig::default()
    );
    assert!(semi_implicit.supports_large_cfl());
}

// 更多策略切换测试需要完整的Solver设置...
