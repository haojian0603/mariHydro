// src-tauri/src/marihydro/physics/engine/solver.rs
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::source::SourceTerm;
use crate::marihydro::core::types::NumericalParams;
use crate::marihydro::core::Workspace;
use crate::marihydro::domain::mesh::UnstructuredMesh;
use crate::marihydro::domain::state::ShallowWaterState;
use crate::marihydro::physics::schemes::HllcSolver;
use super::parallel::ParallelFluxCalculator;
use super::timestep::TimeStepController;
use std::sync::Arc;

pub struct UnstructuredSolver {
    mesh: Arc<UnstructuredMesh>,
    params: NumericalParams,
    g: f64,
    workspace: Workspace,
    flux_calculator: ParallelFluxCalculator,
    timestep_ctrl: TimeStepController,
    source_terms: Vec<Box<dyn SourceTerm>>,
    max_wave_speed: f64,
}

impl UnstructuredSolver {
    pub fn new(mesh: Arc<UnstructuredMesh>, params: NumericalParams, g: f64) -> Self {
        let n_cells = mesh.n_cells();
        let n_faces = mesh.n_faces();
        Self {
            mesh: mesh.clone(),
            params: params.clone(),
            g,
            workspace: Workspace::new(n_cells, n_faces),
            flux_calculator: ParallelFluxCalculator::new(g, params.clone()),
            timestep_ctrl: TimeStepController::new(g, &params),
            source_terms: Vec::new(),
            max_wave_speed: 0.0,
        }
    }

    pub fn add_source_term(&mut self, source: Box<dyn SourceTerm>) {
        self.source_terms.push(source);
    }

    pub fn step(&mut self, state: &mut ShallowWaterState, dt: f64) -> MhResult<f64> {
        if state.n_cells() != self.mesh.n_cells() {
            return Err(crate::marihydro::core::MhError::size_mismatch("state", self.mesh.n_cells(), state.n_cells()));
        }
        self.workspace.reset_fluxes();
        self.workspace.reset_sources();
        self.max_wave_speed = self.flux_calculator.compute_fluxes(state, self.mesh.as_ref(), &mut self.workspace)?;
        self.apply_source_terms(state, dt)?;
        self.update_state(state, dt);
        self.enforce_positivity(state);
        Ok(self.max_wave_speed)
    }

    fn apply_source_terms(&mut self, state: &ShallowWaterState, dt: f64) -> MhResult<()> {
        let ctx = crate::marihydro::core::traits::source::SourceContext {
            time: 0.0,
            dt,
            params: &self.params,
            workspace: &self.workspace,
        };
        for source in &self.source_terms {
            source.compute_all(
                self.mesh.as_ref(),
                state,
                &ctx,
                &mut self.workspace.source_h,
                &mut self.workspace.source_hu,
                &mut self.workspace.source_hv,
            )?;
        }
        Ok(())
    }

    fn update_state(&self, state: &mut ShallowWaterState, dt: f64) {
        let n = state.n_cells();
        for i in 0..n {
            let area = self.mesh.cell_area(crate::marihydro::core::types::CellIndex(i));
            let inv_area = 1.0 / area;
            state.h[i] += dt * inv_area * self.workspace.flux_h[i];
            state.hu[i] += dt * inv_area * (self.workspace.flux_hu[i] + self.workspace.source_hu[i]);
            state.hv[i] += dt * inv_area * (self.workspace.flux_hv[i] + self.workspace.source_hv[i]);
        }
    }

    fn enforce_positivity(&self, state: &mut ShallowWaterState) {
        let h_min = self.params.h_min;
        for i in 0..state.n_cells() {
            if state.h[i] < h_min {
                state.h[i] = 0.0;
                state.hu[i] = 0.0;
                state.hv[i] = 0.0;
            }
        }
    }

    pub fn compute_dt(&mut self, state: &ShallowWaterState) -> f64 {
        self.timestep_ctrl.update(state, self.mesh.as_ref(), &self.params)
    }

    pub fn max_wave_speed(&self) -> f64 { self.max_wave_speed }
    pub fn mesh(&self) -> &UnstructuredMesh { &self.mesh }
    pub fn params(&self) -> &NumericalParams { &self.params }
}

pub struct SolverBuilder {
    mesh: Option<Arc<UnstructuredMesh>>,
    params: NumericalParams,
    g: f64,
    sources: Vec<Box<dyn SourceTerm>>,
}

impl SolverBuilder {
    pub fn new() -> Self {
        Self { mesh: None, params: NumericalParams::default(), g: 9.81, sources: Vec::new() }
    }
    pub fn mesh(mut self, m: Arc<UnstructuredMesh>) -> Self { self.mesh = Some(m); self }
    pub fn params(mut self, p: NumericalParams) -> Self { self.params = p; self }
    pub fn gravity(mut self, g: f64) -> Self { self.g = g; self }
    pub fn add_source(mut self, s: Box<dyn SourceTerm>) -> Self { self.sources.push(s); self }
    pub fn build(self) -> MhResult<UnstructuredSolver> {
        let mesh = self.mesh.ok_or_else(|| crate::marihydro::core::MhError::missing_config("mesh"))?;
        let mut solver = UnstructuredSolver::new(mesh, self.params, self.g);
        for s in self.sources { solver.add_source_term(s); }
        Ok(solver)
    }
}

impl Default for SolverBuilder {
    fn default() -> Self { Self::new() }
}
