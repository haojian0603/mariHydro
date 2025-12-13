// crates/mh_physics/tests/smoke_test.rs

//! 快速冒烟测试
//!
//! 验证核心组件可以正确初始化和基本运行。
//! 这些测试应该快速完成（<1秒），用于 CI 快速反馈。

use mh_core::Scalar;

// ============================================================
// Plan 1: 边界数据驱动测试
// ============================================================

#[test]
fn test_timeseries_basic() {
    use mh_physics::forcing::timeseries::TimeSeries;

    let ts = TimeSeries::from_points(vec![(0.0, 10.0), (1.0, 20.0), (2.0, 15.0)]);

    // 插值
    let v = ts.interpolate(0.5);
    assert!((v - 15.0).abs() < 1e-10);

    // 外推（默认 Clamp）
    let v_before = ts.interpolate(-1.0);
    assert!((v_before - 10.0).abs() < 1e-10);

    let v_after = ts.interpolate(10.0);
    assert!((v_after - 15.0).abs() < 1e-10);
}

#[test]
fn test_timeseries_extrapolation_modes() {
    use mh_physics::forcing::timeseries::{ExtrapolationMode, TimeSeries};

    let ts = TimeSeries::from_points(vec![(0.0, 0.0), (10.0, 10.0)])
        .with_extrapolation(ExtrapolationMode::Linear);

    let v = ts.interpolate(15.0);
    assert!((v - 15.0).abs() < 1e-10);
}

#[test]
fn test_spatial_timeseries() {
    use glam::DVec2;
    use mh_physics::forcing::spatial::SpatialTimeSeries;
    use mh_physics::forcing::timeseries::TimeSeries;

    let ts1 = TimeSeries::from_points(vec![(0.0, 10.0), (1.0, 20.0)]);
    let ts2 = TimeSeries::from_points(vec![(0.0, 30.0), (1.0, 40.0)]);

    let mut spatial = SpatialTimeSeries::new(vec![
        (DVec2::new(0.0, 0.0), ts1),
    ]);
    spatial.add_station(DVec2::new(10.0, 0.0), ts2);

    // 在两站点中间插值
    let v = spatial.get_value_at(DVec2::new(5.0, 0.0), 0.5);
    // IDW 权重相等时取平均
    let expected = (15.0 + 35.0) / 2.0;
    assert!((v - expected).abs() < 1e-10);
}

// ============================================================
// Plan 2: 泥沙输运测试
// ============================================================

#[test]
fn test_transport_formula_mpm() {
    use mh_physics::sediment::formulas::{MeyerPeterMullerFormula, TransportFormula};
    use mh_physics::sediment::properties::SedimentProperties;
    use mh_physics::types::PhysicalConstants;

    let formula = MeyerPeterMullerFormula::new();
    let sediment = SedimentProperties::from_d50_mm(1.0);  // 1mm 粒径
    let physics = PhysicalConstants::seawater();

    // 无剪切应力时输沙率为零
    let qb: f64 = formula.compute_from_shear_stress(0.0, &sediment, &physics);
    assert!(qb.abs() < 1e-20);

    // 有剪切应力时输沙率为正
    let qb: f64 = formula.compute_from_shear_stress(10.0, &sediment, &physics);
    assert!(qb >= 0.0);
}

#[test]
fn test_morphology_config() {
    use mh_physics::sediment::morphology::MorphologyConfig;

    let config = MorphologyConfig::default();
    assert!((config.porosity - 0.4).abs() < 1e-10);
    assert!(config.avalanche_enabled);

    let coarse = MorphologyConfig::coarse_sediment();
    let fine = MorphologyConfig::fine_sediment();
    assert!(coarse.porosity < fine.porosity);
}

// ============================================================
// Plan 3: 示踪剂输运测试
// ============================================================

#[test]
fn test_tracer_boundary_manager() {
    use mh_physics::tracer::boundary::{TracerBoundaryCondition, TracerBoundaryManager, TracerBoundaryType};

    let mut manager = TracerBoundaryManager::new(10);

    manager.set_boundary(0, TracerBoundaryCondition::dirichlet(35.0));
    manager.set_boundary(9, TracerBoundaryCondition::neumann(-0.01));

    let bc0 = manager.get(0, 0.0);
    assert_eq!(bc0.bc_type, TracerBoundaryType::Dirichlet);
    assert!((bc0.value - 35.0_f64).abs() < 1e-10);

    let bc5 = manager.get(5, 0.0);
    assert_eq!(bc5.bc_type, TracerBoundaryType::ZeroGradient);
}

#[test]
fn test_diffusion_coefficient() {
    use mh_physics::tracer::diffusion::DiffusionCoefficient;

    let const_coef: DiffusionCoefficient<f64> = DiffusionCoefficient::Constant(10.0);
    assert!((const_coef.effective_at(0, None) - 10.0).abs() < 1e-10);

    let turb_coef: DiffusionCoefficient<f64> = DiffusionCoefficient::Turbulent { molecular: 1.0, schmidt_number: 0.7 };
    let eff = turb_coef.effective_at(0, Some(7.0));
    assert!((eff - 11.0).abs() < 1e-10);
}

// ============================================================
// Plan 4: 稀疏线性代数测试
// ============================================================

#[test]
fn test_csr_matrix_identity() {
    use mh_physics::numerics::CsrMatrix;

    let mat = CsrMatrix::<f64>::identity(5);
    assert_eq!(mat.n_rows(), 5);
    assert_eq!(mat.n_cols(), 5);
    assert_eq!(mat.nnz(), 5);

    for i in 0..5 {
        assert!((mat.get(i, i) - 1.0).abs() < 1e-14);
    }
}

#[test]
fn test_csr_mul_vec() {
    use mh_physics::numerics::CsrBuilder;

    let mut builder = CsrBuilder::<f64>::new_square(3);
    builder.set(0, 0, 2.0);
    builder.set(0, 1, -1.0);
    builder.set(1, 0, -1.0);
    builder.set(1, 1, 2.0);
    builder.set(1, 2, -1.0);
    builder.set(2, 1, -1.0);
    builder.set(2, 2, 2.0);

    let mat = builder.build();
    let x = vec![1.0, 2.0, 3.0];
    let mut y = vec![0.0; 3];

    mat.mul_vec(&x, &mut y);

    assert!((y[0] - 0.0_f64).abs() < 1e-14);
    assert!((y[1] - 0.0_f64).abs() < 1e-14);
    assert!((y[2] - 4.0_f64).abs() < 1e-14);
}

#[test]
fn test_vector_ops() {
    use mh_physics::numerics::{axpy, dot, norm2};

    let x: Vec<f64> = vec![3.0, 4.0];
    let mut y: Vec<f64> = vec![1.0, 2.0];

    assert!((norm2(&x) - 5.0).abs() < 1e-14);

    let d = dot(&x, &y);
    assert!((d - 11.0).abs() < 1e-14);

    axpy(2.0, &x, &mut y);
    assert!((y[0] - 7.0).abs() < 1e-14);
    assert!((y[1] - 10.0).abs() < 1e-14);
}

#[test]
fn test_pcg_solver_simple() {
    use mh_physics::numerics::{CsrBuilder, IterativeSolver, JacobiPreconditioner, PcgSolver, SolverConfig};

    // 简单对称正定矩阵
    let mut builder = CsrBuilder::<f64>::new_square(3);
    builder.set(0, 0, 4.0);
    builder.set(0, 1, -1.0);
    builder.set(1, 0, -1.0);
    builder.set(1, 1, 4.0);
    builder.set(1, 2, -1.0);
    builder.set(2, 1, -1.0);
    builder.set(2, 2, 4.0);

    let mat = builder.build();
    let b = vec![1.0, 2.0, 3.0];
    let mut x = vec![0.0; 3];

    let config = SolverConfig::new(1e-10, 100);
    let mut solver = PcgSolver::new(config);
    let precond = JacobiPreconditioner::from_matrix(&mat);

    let result = solver.solve(&mat, &b, &mut x, &precond);

    assert!(result.is_converged());
    assert!(result.relative_residual < 1e-8);
}

// ============================================================
// Plan 5: 离散化测试
// ============================================================

#[test]
fn test_depth_corrector() {
    use mh_physics::numerics::DepthCorrector;

    let corrector = DepthCorrector::new(3);

    let mut h = vec![1.0, 2.0, 0.5];
    let eta_prime = vec![0.1, -0.3, -0.6];

    corrector.correct(&mut h, &eta_prime);

    assert!((h[0] - 1.1).abs() < 1e-14);
    assert!((h[1] - 1.7).abs() < 1e-14);
    assert!(h[2] >= 0.0);
}

// ============================================================
// Plan 6: 半隐式策略测试
// ============================================================

#[test]
fn test_semi_implicit_config() {
    use mh_physics::engine::SemiImplicitConfig;

    let config = SemiImplicitConfig::default();
    assert!((config.constants.g - 9.81).abs() < 1e-10);
    assert!((config.theta - 0.5).abs() < 1e-10);

    let conservative = SemiImplicitConfig::conservative();
    let fast = SemiImplicitConfig::fast();

    assert!(conservative.solver_rtol < fast.solver_rtol);
}

// ============================================================
// 综合测试
// ============================================================

#[test]
fn test_types_compile() {
    use mh_physics::types::{ConstantBoundaryProvider, NumericalParams};

    let params = NumericalParams::default();
    assert!(params.h_min > 0.0);

    let provider = ConstantBoundaryProvider::new(1.5);
    let v: f64 = *provider.value();
    assert!((v - 1.5).abs() < 1e-10);
}
