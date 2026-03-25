use crate::sources::traits::{
    SourceContextGeneric, SourceContributionGeneric, SourceStiffness, SourceTermGeneric,
};
use crate::state::ShallowWaterState;
use crate::waves::radiation_stress::{
    RadiationStressCalculatorGeneric, RadiationStressTensorGeneric, WaveFieldError,
    WaveFieldGeneric, WaveParametersGeneric,
};
use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar};
use std::marker::PhantomData;

pub struct WaveRadiationSource<B: Backend> {
    #[allow(dead_code)]
    calculator: RadiationStressCalculatorGeneric<B>,
    wave_field: WaveFieldGeneric<B>,
    stress: Vec<RadiationStressTensorGeneric<B::Scalar>>,
    stress_gradient: Vec<(B::Scalar, B::Scalar)>,
    rho_water: B::Scalar,
    enabled: bool,
    gradient_computed: bool,
}

impl<B: Backend> WaveRadiationSource<B> {
    pub fn new(backend: B, n_cells: usize) -> Self {
        Self {
            calculator: RadiationStressCalculatorGeneric::new(backend.clone(), n_cells),
            wave_field: WaveFieldGeneric::new(backend.clone(), n_cells),
            stress: vec![RadiationStressTensorGeneric::default(); n_cells],
            stress_gradient: vec![(B::Scalar::ZERO, B::Scalar::ZERO); n_cells],
            rho_water: backend.config_scalar(1025.0, "WaveRadiationSource.rho_water"),
            enabled: true,
            gradient_computed: false,
        }
    }

    pub fn set_wave_field(&mut self, wave_field: WaveFieldGeneric<B>) {
        let backend = wave_field.backend().clone();
        let n_cells = wave_field.len();
        self.wave_field = wave_field;
        self.calculator = RadiationStressCalculatorGeneric::new(backend, n_cells);
        self.stress
            .resize(n_cells, RadiationStressTensorGeneric::default());
        self.stress_gradient
            .resize(n_cells, (B::Scalar::ZERO, B::Scalar::ZERO));
        self.gradient_computed = false;
    }

    pub fn set_uniform_waves(
        &mut self,
        height: B::Scalar,
        period: B::Scalar,
        direction: B::Scalar,
        depth: &[B::Scalar],
    ) -> Result<(), WaveFieldError> {
        let params = WaveParametersGeneric::new(height, period, direction);
        self.wave_field.set_uniform(&params);

        let backend = self.wave_field.backend().clone();
        let n_cells = self.wave_field.len();
        let default_depth = backend.config_scalar(10.0, "WaveRadiationSource.default_depth");
        let mut depth_values = vec![default_depth; n_cells];
        let n = n_cells.min(depth.len());
        depth_values[..n].copy_from_slice(&depth[..n]);

        let mut depth_buffer = backend.alloc_init(n_cells, default_depth);
        depth_buffer.copy_from_slice(&depth_values);
        self.wave_field.update_dispersion(&depth_buffer)?;
        self.gradient_computed = false;
        Ok(())
    }

    pub fn compute_stress(&mut self) -> Result<(), WaveFieldError> {
        self.calculator.compute_stress(&self.wave_field)?;
        let (sxx, syy, sxy) = self.calculator.stress_components();
        let sxx = sxx.try_as_slice().ok_or_else(|| {
            WaveFieldError::BackendAccess("wave stress sxx buffer not accessible".to_string())
        })?;
        let syy = syy.try_as_slice().ok_or_else(|| {
            WaveFieldError::BackendAccess("wave stress syy buffer not accessible".to_string())
        })?;
        let sxy = sxy.try_as_slice().ok_or_else(|| {
            WaveFieldError::BackendAccess("wave stress sxy buffer not accessible".to_string())
        })?;

        for i in 0..self.stress.len() {
            self.stress[i] = RadiationStressTensorGeneric {
                sxx: sxx[i],
                syy: syy[i],
                sxy: sxy[i],
            };
        }
        Ok(())
    }

    pub fn compute_gradient_simple(&mut self, _cell_sizes: &[B::Scalar]) {
        for grad in &mut self.stress_gradient {
            *grad = (B::Scalar::ZERO, B::Scalar::ZERO);
        }

        self.enabled = false;
        self.gradient_computed = false;
    }

    pub fn set_gradient(&mut self, gradient: &[(B::Scalar, B::Scalar)]) {
        let n = self.stress_gradient.len().min(gradient.len());
        self.stress_gradient[..n].copy_from_slice(&gradient[..n]);
        self.enabled = true;
        self.gradient_computed = true;
    }

    pub fn stress(&self) -> &[RadiationStressTensorGeneric<B::Scalar>] {
        &self.stress
    }

    pub fn wave_field(&self) -> &WaveFieldGeneric<B> {
        &self.wave_field
    }
}

impl<B: Backend> SourceTermGeneric<B> for WaveRadiationSource<B> {
    fn name(&self) -> &'static str {
        "WaveRadiation"
    }

    fn stiffness(&self) -> SourceStiffness {
        SourceStiffness::Explicit
    }

    fn is_enabled(&self) -> bool {
        self.enabled && self.gradient_computed
    }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        if !(self.enabled && self.gradient_computed) {
            return SourceContributionGeneric::default();
        }

        let h = state.h[cell];
        if ctx.is_dry(h) {
            return SourceContributionGeneric::default();
        }

        let (grad_x, grad_y) = self
            .stress_gradient
            .get(cell)
            .copied()
            .unwrap_or((B::Scalar::ZERO, B::Scalar::ZERO));
        let fx = -grad_x / (self.rho_water * h);
        let fy = -grad_y / (self.rho_water * h);

        SourceContributionGeneric::momentum(fx, fy)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<B>,
        _rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !(self.enabled && self.gradient_computed) {
            return;
        }

        let n = state
            .n_cells()
            .min(self.stress.len())
            .min(rhs_hu.len())
            .min(rhs_hv.len());

        for cell in 0..n {
            let contrib = SourceTermGeneric::compute_cell(self, cell, state, ctx);
            rhs_hu[cell] += contrib.s_hu;
            rhs_hv[cell] += contrib.s_hv;
        }
    }
}

pub struct WaveRadiationSourceGeneric<B: Backend> {
    momentum_source: Vec<(B::Scalar, B::Scalar)>,
    enabled: bool,
    _marker: PhantomData<B>,
}

impl<B: Backend> WaveRadiationSourceGeneric<B> {
    pub fn new(n_cells: usize) -> Self {
        Self {
            momentum_source: vec![(B::Scalar::ZERO, B::Scalar::ZERO); n_cells],
            enabled: false,
            _marker: PhantomData,
        }
    }

    pub fn set_momentum_source(&mut self, source: &[(B::Scalar, B::Scalar)]) {
        let n = self.momentum_source.len().min(source.len());
        self.momentum_source[..n].copy_from_slice(&source[..n]);
        self.enabled = true;
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl<B: Backend> SourceTermGeneric<B> for WaveRadiationSourceGeneric<B> {
    fn name(&self) -> &'static str {
        "WaveRadiation"
    }

    fn stiffness(&self) -> SourceStiffness {
        SourceStiffness::Explicit
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        let h = state.h[cell];
        if ctx.is_dry(h) {
            return SourceContributionGeneric::default();
        }

        let (fx, fy) = self
            .momentum_source
            .get(cell)
            .copied()
            .unwrap_or((B::Scalar::ZERO, B::Scalar::ZERO));
        SourceContributionGeneric::momentum(fx, fy)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<B>,
        _rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !self.enabled {
            return;
        }

        let n = state.n_cells().min(rhs_hu.len()).min(rhs_hv.len());
        for cell in 0..n {
            let h = state.h[cell];
            if !ctx.is_dry(h) {
                let (fx, fy) = self
                    .momentum_source
                    .get(cell)
                    .copied()
                    .unwrap_or((B::Scalar::ZERO, B::Scalar::ZERO));
                rhs_hu[cell] += fx;
                rhs_hv[cell] += fy;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::traits::test_support::{
        assert_source_enabled,
        assert_source_metadata,
        test_backend,
        TestBackend,
    };

    #[test]
    fn test_wave_radiation_source_creation() {
        let source = WaveRadiationSource::new(test_backend(), 100);
        assert_source_metadata(&source, "WaveRadiation", SourceStiffness::Explicit);
        assert_source_enabled(&source, false);
    }

    #[test]
    fn test_generic_wave_source() {
        let mut source = WaveRadiationSourceGeneric::<TestBackend>::new(10);
        assert_source_enabled(&source, false);

        source.set_momentum_source(&[(0.1, 0.2); 10]);
        assert_source_enabled(&source, true);
    }
}
