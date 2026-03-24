use mh_agent::{AiError, ScalarSamples, SurrogateConfig, SurrogateModel, SurrogateType};
use mh_runtime::CpuBackend;

#[test]
fn scalar_samples_wrap_slice_access() {
    let samples: ScalarSamples<CpuBackend<f64>> = vec![1.0, 2.0, 3.0].into();

    assert_eq!(samples.len(), 3);
    assert!(!samples.is_empty());
    assert_eq!(samples.as_slice()[1], 2.0);
    assert_eq!(samples.iter().copied().sum::<f64>(), 6.0);
}

#[test]
fn surrogate_rejects_unimplemented_model_types() {
    let config = SurrogateConfig::<CpuBackend<f64>> {
        model_type: SurrogateType::GaussianProcess,
        model_path: None,
        input_features: Vec::new(),
        output_features: vec!["h".into()],
        prediction_horizon: 0.0,
        estimate_uncertainty: true,
        assimilation_rate: 0.5,
        learning_rate: 0.01,
        l2_reg: 0.0,
        min_std: 1e-6,
    };

    match SurrogateModel::new(config) {
        Err(AiError::UnsupportedModelType(_)) => {}
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("expected UnsupportedModelType"),
    }
}
