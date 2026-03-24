use mh_agent::{
    ObservationOperator, PhysicsSnapshot, Polarization, ReflectanceCalibration,
    ReflectanceOperator, SAROperator, WaterLevelOperator,
};
use mh_runtime::{CellIndex, CpuBackend};

fn snapshot_with_sediment() -> PhysicsSnapshot<CpuBackend<f64>> {
    PhysicsSnapshot::try_new(
        vec![1.0, 2.0],
        vec![0.5, 0.25],
        vec![0.0, 0.1],
        vec![0.2, 0.3],
        Some(vec![10.0, 20.0]),
        0.0,
        vec![[0.0, 0.0], [1.0, 0.0]],
        vec![1.0, 1.0],
    )
    .expect("snapshot should be valid")
}

#[test]
fn reflectance_operator_uses_explicit_calibration() {
    let snapshot = snapshot_with_sediment();
    let op = ReflectanceOperator::<CpuBackend<f64>>::new(
        645.0,
        ReflectanceCalibration::new(1.0, 0.5),
        0.2,
    );

    let observed = op.observe(&snapshot);

    assert_eq!(observed.len(), 2);
    assert!((observed[0] - (10.0f64.ln() + 0.5)).abs() < 1e-12);
    assert!((observed[1] - (20.0f64.ln() + 0.5)).abs() < 1e-12);
}

#[test]
fn water_level_operator_validates_indices() {
    match WaterLevelOperator::<CpuBackend<f64>>::new(vec![CellIndex::new(2)], 0.1, 2) {
        Err(err) => assert!(err.to_string().contains("station index out of bounds")),
        Ok(_) => panic!("out-of-bounds station should fail"),
    }
}

#[test]
fn sar_operator_reports_one_variance_per_observation() {
    let snapshot = snapshot_with_sediment();
    let op = SAROperator::<CpuBackend<f64>>::new(35.0, Polarization::VV);

    let observed = op.observe(&snapshot);
    let variance = op
        .observation_error_variance_for(observed.len())
        .expect("variance should be available");

    assert_eq!(variance.len(), observed.len());
    assert!(variance.iter().all(|value| *value > 0.0));
}
