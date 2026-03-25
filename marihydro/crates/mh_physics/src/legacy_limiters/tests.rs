use super::*;

#[test]
fn test_minmod() {
    assert_eq!(minmod(1.0, 2.0), 1.0);
    assert_eq!(minmod(-1.0, -2.0), -1.0);
    assert_eq!(minmod(1.0, -2.0), 0.0);
    assert_eq!(minmod(-1.0, 2.0), 0.0);
}

#[test]
fn test_limiter_functions() {
    // r = 1 时，所有限制器应返回 1
    assert!((limiter_minmod(1.0) - 1.0).abs() < 1e-10);
    assert!((limiter_superbee(1.0) - 1.0).abs() < 0.01); // superbee 返回 min(2, 1) = 1
    assert!((limiter_van_leer(1.0) - 1.0).abs() < 1e-10);
    assert!((limiter_van_albada(1.0) - 1.0).abs() < 1e-10);

    // r < 0 时，应返回 0
    assert_eq!(limiter_minmod(-0.5), 0.0);
    assert_eq!(limiter_van_albada(-0.5), 0.0);
}

#[test]
fn test_venkatakrishnan() {
    let limiter = VenkatakrishnanLimiter::new(5.0, 1.0);

    // 无梯度
    let phi = limiter.compute(0.0, 0.0, 0.0);
    assert_eq!(phi, 1.0);

    // 正常情况
    let phi = limiter.compute(-1.0, 1.0, 0.5);
    assert!((0.0..=1.0).contains(&phi));
}

#[test]
fn test_barth_jespersen() {
    // 不超过极值
    let phi = BarthJespersenLimiter::compute(1.0, 0.0, 2.0, 1.5);
    assert_eq!(phi, 1.0);

    // 超过极值
    let phi = BarthJespersenLimiter::compute(1.0, 0.0, 2.0, 3.0);
    assert!((phi - 0.5).abs() < 1e-10);
}

#[test]
fn test_muscl_reconstruct() {
    let config = LegacyMusclConfig {
        limiter: LegacyLimiterType::Minmod,
        kappa: 0.0, // Fromm
        ..Default::default()
    };
    let reconstructor = LegacyMusclReconstructor::new(config);

    // 线性场
    let (q_l, q_r) = reconstructor.reconstruct(0.0, 1.0, 2.0);

    // 应该产生二阶精度的重构
    assert!(q_l > 1.0);
    assert!(q_r < 2.0);
}

#[test]
fn test_muscl_constant_field() {
    let config = LegacyMusclConfig::default();
    let reconstructor = LegacyMusclReconstructor::new(config);

    // 常数场
    let (q_l, q_r) = reconstructor.reconstruct(1.0, 1.0, 1.0);

    assert!((q_l - 1.0).abs() < 1e-10);
    assert!((q_r - 1.0).abs() < 1e-10);
}

#[test]
fn test_venkatakrishnan_with_context() {
    let ctx = LimiterContext {
        q_center: 1.0,
        q_min: 0.5,
        q_max: 1.5,
        q_face: 1.2,
        length_scale: 1.0,
        venkat_k: 5.0,
    };
    let phi = apply_limiter_with_context(LegacyLimiterType::Venkatakrishnan, 0.0, Some(&ctx));
    assert!((0.0..=1.0).contains(&phi));
}

#[test]
fn test_barth_jespersen_with_context() {
    let ctx = LimiterContext {
        q_center: 1.0,
        q_min: 0.5,
        q_max: 1.5,
        q_face: 1.3,
        length_scale: 1.0,
        venkat_k: 5.0,
    };
    let phi = apply_limiter_with_context(LegacyLimiterType::BarthJespersen, 0.0, Some(&ctx));
    assert!((0.0..=1.0).contains(&phi));
}

#[test]
fn test_limiter_symmetry() {
    // Van Leer 应该满足 φ(r) = r * φ(1/r)
    for r in [0.5, 1.0, 2.0, 4.0] {
        let phi_r = limiter_van_leer(r);
        let phi_inv = limiter_van_leer(1.0 / r);
        assert!((phi_r - r * phi_inv).abs() < 1e-10);
    }
}
