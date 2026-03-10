// crates/mh_physics/tests/strategy_switching.rs

//! 绛栫暐鍒囨崲娴嬭瘯
//! 楠岃瘉鏄惧紡/鍗婇殣寮忕瓥鐣ョ殑鍒囨崲鍜岀姸鎬佽繛缁€?

use mh_physics::engine::strategy::{
    TimeIntegrationStrategy, ExplicitStrategy, SemiImplicitStrategyGeneric,
    ExplicitConfig, SemiImplicitConfig,
};
use mh_physics::core::CpuBackend;

/// 娴嬭瘯绛栫暐鍙互琚垱寤?
#[test]
fn test_strategy_creation() {
    let _explicit: ExplicitStrategy<CpuBackend<f64>> = ExplicitStrategy::new_with_backend(
        CpuBackend::<f64>::new(),
        ExplicitConfig::default(),
    );
    let _semi_implicit = SemiImplicitStrategyGeneric::<CpuBackend<f64>>::new_with_backend(
        CpuBackend::<f64>::new(),
        100, // n_cells
        SemiImplicitConfig::default(),
    );
}

/// 娴嬭瘯绛栫暐鍚嶇О
#[test]
fn test_strategy_names() {
    let explicit: ExplicitStrategy<CpuBackend<f64>> = ExplicitStrategy::new_with_backend(
        CpuBackend::<f64>::new(),
        ExplicitConfig::default(),
    );
    assert!(!explicit.name().is_empty());
    
    let semi_implicit = SemiImplicitStrategyGeneric::<CpuBackend<f64>>::new_with_backend(
        CpuBackend::<f64>::new(),
        100, // n_cells
        SemiImplicitConfig::default(),
    );
    assert!(!semi_implicit.name().is_empty());
}

/// 娴嬭瘯绛栫暐CFL鏀寔
#[test]
fn test_cfl_support() {
    let explicit: ExplicitStrategy<CpuBackend<f64>> = ExplicitStrategy::new_with_backend(
        CpuBackend::<f64>::new(),
        ExplicitConfig::default(),
    );
    assert!(!explicit.supports_large_cfl());
    
    let semi_implicit = SemiImplicitStrategyGeneric::<CpuBackend<f64>>::new_with_backend(
        CpuBackend::<f64>::new(),
        100, // n_cells
        SemiImplicitConfig::default(),
    );
    assert!(semi_implicit.supports_large_cfl());
}

// 鏇村绛栫暐鍒囨崲娴嬭瘯闇€瑕佸畬鏁寸殑Solver璁剧疆...
