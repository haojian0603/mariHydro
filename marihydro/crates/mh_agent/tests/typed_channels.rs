use mh_agent::{
    AIAgent, AiError, Assimilable, DenseScalarMatrix, PhysicsSnapshot, ScalarSamples,
    SurrogateConfig, SurrogateModel,
};
use mh_runtime::{Backend, CpuBackend};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn scalar_samples_wrap_slice_access() {
    let samples: ScalarSamples<CpuBackend<f64>> = vec![1.0, 2.0, 3.0].into();

    assert_eq!(samples.len(), 3);
    assert!(!samples.is_empty());
    assert_eq!(samples.as_slice()[1], 2.0);
    assert_eq!(samples.iter().copied().sum::<f64>(), 6.0);
}

#[test]
fn dense_scalar_matrix_preserves_rectangular_shape() {
    let matrix = DenseScalarMatrix::<CpuBackend<f64>>::try_new(vec![
        vec![1.0, 2.0].into(),
        vec![3.0, 4.0].into(),
    ])
    .expect("matrix should be rectangular");

    assert_eq!(matrix.nrows(), 2);
    assert_eq!(matrix.ncols(), 2);
    assert_eq!(matrix.row(1).unwrap().as_slice()[0], 3.0);
}

#[test]
fn dense_scalar_matrix_rejects_jagged_rows() {
    let err = DenseScalarMatrix::<CpuBackend<f64>>::try_new(vec![
        vec![1.0, 2.0].into(),
        vec![3.0].into(),
    ])
    .expect_err("jagged rows should be rejected");

    match err {
        AiError::InvalidShape { expected, actual } => {
            assert_eq!(expected, vec![2]);
            assert_eq!(actual, vec![2, 1]);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn surrogate_model_constructs_without_fake_type_selector() {
    let config = SurrogateConfig::<CpuBackend<f64>> {
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

    SurrogateModel::new(config).expect("surrogate should expose only the shipped linear model");
}

#[test]
fn surrogate_model_rejects_unsupported_output_feature() {
    let config = SurrogateConfig::<CpuBackend<f64>> {
        model_path: None,
        input_features: Vec::new(),
        output_features: vec!["u".into()],
        prediction_horizon: 0.0,
        estimate_uncertainty: true,
        assimilation_rate: 0.5,
        learning_rate: 0.01,
        l2_reg: 0.0,
        min_std: 1e-6,
    };

    let err = match SurrogateModel::new(config) {
        Ok(_) => panic!("unsupported output surface must be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, AiError::InvalidObservation(_)));
}

#[test]
fn surrogate_model_predict_requires_trained_state() {
    let backend = CpuBackend::<f64>::new();
    let config = SurrogateConfig::<CpuBackend<f64>> {
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
    let mut model = SurrogateModel::new(config).expect("surrogate should construct");
    let snapshot = PhysicsSnapshot::empty(&backend, 1);

    let err = model
        .predict(&snapshot)
        .expect_err("untrained surrogate must not synthesize predictions");
    assert!(matches!(err, AiError::NotReady(_)));
}

#[test]
fn surrogate_model_rejects_invalid_saved_state() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be monotonic enough for temp file naming")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("mh_agent_invalid_surrogate_{unique}.json"));
    let invalid_state = r#"{
  "linear_core": {
    "input_dim": 1,
    "output_dim": 1,
    "weights": [1.0],
    "bias": [0.0]
  },
  "input_normalization": {
    "mean": [],
    "std": [1.0],
    "count": 1,
    "m2": [0.0]
  },
  "output_normalization": null,
  "error_ema": null
}"#;
    fs::write(&path, invalid_state).expect("test state file should be writable");

    let config = SurrogateConfig::<CpuBackend<f64>> {
        model_path: Some(path.to_string_lossy().into_owned()),
        input_features: Vec::new(),
        output_features: vec!["h".into()],
        prediction_horizon: 0.0,
        estimate_uncertainty: false,
        assimilation_rate: 0.5,
        learning_rate: 0.01,
        l2_reg: 0.0,
        min_std: 1e-6,
    };

    let err = match SurrogateModel::new(config) {
        Ok(_) => panic!("malformed saved surrogate state must fail"),
        Err(err) => err,
    };
    assert!(matches!(err, AiError::InvalidShape { .. }));
    let _ = fs::remove_file(path);
}

#[test]
fn surrogate_apply_rejects_partial_field_prediction() {
    let backend = CpuBackend::<f64>::new();
    let config = SurrogateConfig::<CpuBackend<f64>> {
        model_path: None,
        input_features: Vec::new(),
        output_features: vec!["h".into()],
        prediction_horizon: 0.0,
        estimate_uncertainty: false,
        assimilation_rate: 0.5,
        learning_rate: 0.01,
        l2_reg: 0.0,
        min_std: 1e-6,
    };

    let mut model = SurrogateModel::new(config).expect("surrogate should construct");
    let snapshot = PhysicsSnapshot::empty(&backend, 1);
    model
        .update_model(&snapshot, &[1.0])
        .expect("single-cell training target should be accepted");
    let _ = model
        .predict(&snapshot)
        .expect("trained surrogate should produce a field");

    let mut state = DummyAssimilableState::new(backend, 2);
    let err = model
        .apply(&mut state)
        .expect_err("partial prediction fields must not be applied to larger states");
    assert!(matches!(err, AiError::InvalidShape { .. }));
}

struct DummyAssimilableState {
    backend: CpuBackend<f64>,
    depth: Vec<f64>,
    velocity_u: Vec<f64>,
    velocity_v: Vec<f64>,
    bed: Vec<f64>,
    cell_areas: Vec<f64>,
    cell_centers: Vec<[f64; 2]>,
}

impl DummyAssimilableState {
    fn new(backend: CpuBackend<f64>, n_cells: usize) -> Self {
        Self {
            backend,
            depth: backend.alloc_init(n_cells, 1.0),
            velocity_u: backend.alloc_init(n_cells, 0.0),
            velocity_v: backend.alloc_init(n_cells, 0.0),
            bed: backend.alloc_init(n_cells, 0.0),
            cell_areas: backend.alloc_init(n_cells, 1.0),
            cell_centers: backend.alloc_init(n_cells, [0.0, 0.0]),
        }
    }
}

impl Assimilable<CpuBackend<f64>> for DummyAssimilableState {
    fn backend(&self) -> &CpuBackend<f64> {
        &self.backend
    }

    fn get_tracer_mut(&mut self, _name: &str) -> Option<&mut Vec<f64>> {
        None
    }

    fn get_velocity_mut(&mut self) -> Option<(&mut Vec<f64>, &mut Vec<f64>)> {
        Some((&mut self.velocity_u, &mut self.velocity_v))
    }

    fn get_depth(&self) -> &Vec<f64> {
        &self.depth
    }

    fn get_depth_mut(&mut self) -> &mut Vec<f64> {
        &mut self.depth
    }

    fn get_bed_elevation_mut(&mut self) -> &mut Vec<f64> {
        &mut self.bed
    }

    fn n_cells(&self) -> usize {
        self.depth.len()
    }

    fn cell_areas(&self) -> &Vec<f64> {
        &self.cell_areas
    }

    fn cell_centers(&self) -> &Vec<[f64; 2]> {
        &self.cell_centers
    }
}
