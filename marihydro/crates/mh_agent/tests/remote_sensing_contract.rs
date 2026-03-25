use mh_agent::{
    AIAgent, AiError, Assimilable, ImageBounds, InterpolationMethod, RemoteSensingAgent,
    RemoteSensingConfig, SatelliteImage, SensorType, InversionModel,
};
use mh_runtime::{Backend, CpuBackend};

struct DummyState {
    backend: CpuBackend<f64>,
    depth: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
    z: Vec<f64>,
    sediment: Vec<f64>,
    cell_areas: Vec<f64>,
    cell_centers: Vec<[f64; 2]>,
}

impl DummyState {
    fn new(sediment: Vec<f64>) -> Self {
        let n = sediment.len();
        Self {
            backend: CpuBackend::new(),
            depth: vec![1.0; n],
            u: vec![0.0; n],
            v: vec![0.0; n],
            z: vec![0.0; n],
            sediment,
            cell_areas: vec![1.0; n],
            cell_centers: vec![[0.0, 0.0]; n],
        }
    }
}

impl Assimilable<CpuBackend<f64>> for DummyState {
    fn backend(&self) -> &CpuBackend<f64> {
        &self.backend
    }

    fn get_tracer_mut(
        &mut self,
        name: &str,
    ) -> Option<&mut <CpuBackend<f64> as Backend>::Buffer<f64>> {
        match name {
            "sediment" => Some(&mut self.sediment),
            _ => None,
        }
    }

    fn get_velocity_mut(
        &mut self,
    ) -> Option<(
        &mut <CpuBackend<f64> as Backend>::Buffer<f64>,
        &mut <CpuBackend<f64> as Backend>::Buffer<f64>,
    )> {
        Some((&mut self.u, &mut self.v))
    }

    fn get_depth(&self) -> &<CpuBackend<f64> as Backend>::Buffer<f64> {
        &self.depth
    }

    fn get_depth_mut(&mut self) -> &mut <CpuBackend<f64> as Backend>::Buffer<f64> {
        &mut self.depth
    }

    fn get_bed_elevation_mut(&mut self) -> &mut <CpuBackend<f64> as Backend>::Buffer<f64> {
        &mut self.z
    }

    fn n_cells(&self) -> usize {
        self.depth.len()
    }

    fn cell_areas(&self) -> &<CpuBackend<f64> as Backend>::Buffer<f64> {
        &self.cell_areas
    }

    fn cell_centers(&self) -> &<CpuBackend<f64> as Backend>::Buffer<[f64; 2]> {
        &self.cell_centers
    }
}

fn make_config() -> RemoteSensingConfig<CpuBackend<f64>> {
    RemoteSensingConfig {
        assimilation_rate: 0.5,
        max_concentration: 100.0,
        max_cloud_cover: 0.5,
        interpolation: InterpolationMethod::NearestNeighbor,
        inversion: InversionModel::linear(10.0, 0.0),
    }
}

#[test]
fn remote_sensing_rejects_invalid_bounds() {
    let mut agent = RemoteSensingAgent::<CpuBackend<f64>>::new(make_config());
    let image = SatelliteImage {
        data: vec![0.2, 0.2, 0.2, 0.2],
        dimensions: (2, 2),
        bounds: ImageBounds::new(0.0, 0.0, 0.0, 1.0),
        timestamp: 1.0,
        sensor: SensorType::Optical,
        cloud_cover: 0.1,
        resolution: 30.0,
    };

    let err = agent
        .infer(&image, &[[0.0, 0.0]])
        .expect_err("invalid bounds should be rejected");

    match err {
        AiError::InvalidObservation(message) => assert!(message.contains("invalid image bounds")),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn remote_sensing_apply_blends_prediction_into_sediment() {
    let mut agent = RemoteSensingAgent::<CpuBackend<f64>>::new(make_config());
    let image = SatelliteImage {
        data: vec![0.2, 0.2, 0.2, 0.2],
        dimensions: (2, 2),
        bounds: ImageBounds::new(0.0, 0.0, 1.0, 1.0),
        timestamp: 1.0,
        sensor: SensorType::Optical,
        cloud_cover: 0.1,
        resolution: 30.0,
    };
    let target_cells = [[0.0, 0.0]];
    let inference = agent
        .infer(&image, &target_cells)
        .expect("inference should succeed");

    let predicted = inference.concentration.as_slice()[0];
    let mut state = DummyState::new(vec![4.0]);

    agent.apply(&mut state).expect("apply should succeed");

    let expected = 0.5 * 4.0 + 0.5 * predicted;
    assert!((state.sediment[0] - expected).abs() < 1e-12);
}
