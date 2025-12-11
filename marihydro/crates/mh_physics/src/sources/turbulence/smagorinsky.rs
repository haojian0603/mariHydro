// crates/mh_physics/src/sources/turbulence/smagorinsky.rs

//! Smagorinsky äºšæ ¼å­å°ºåº¦æ¹æµæ¨¡åž‹
//!
//! å®žçŽ° 2D æµ…æ°´æ–¹ç¨‹çš„æ°´å¹³æ¹æµé—­åˆï¼Œä¸»è¦ç”¨äºŽæ°´å¹³æ¶¡ç²˜æ€§è®¡ç®—ã€‚
//!
//! # Smagorinsky æ¨¡åž‹
//!
//! Smagorinsky (1963) æ¨¡åž‹å‡è®¾äºšæ ¼å­æ¹æµç²˜æ€§ï¼š
//! ```text
//! Î½_t = (C_s * Î”)Â² * |S|
//! ```
//!
//! å…¶ä¸­ï¼š
//! - C_s æ˜¯ Smagorinsky å¸¸æ•°ï¼ˆé€šå¸¸ 0.1-0.2ï¼‰
//! - Î” æ˜¯ç½‘æ ¼å°ºåº¦ (âˆšA)
//! - |S| æ˜¯åº”å˜çŽ‡å¼ é‡çš„æ¨¡
//!
//! # åº”å˜çŽ‡å¼ é‡
//!
//! å¯¹äºŽäºŒç»´æµåŠ¨ï¼š
//! ```text
//! |S| = âˆš(2*(âˆ‚u/âˆ‚x)Â² + 2*(âˆ‚v/âˆ‚y)Â² + (âˆ‚u/âˆ‚y + âˆ‚v/âˆ‚x)Â²)
//! ```
//!
//! # ç‰©ç†é€‚ç”¨æ€§
//!
//! **é‡è¦è­¦å‘Š**ï¼šSmagorinsky æ¨¡åž‹åŽŸæœ¬è®¾è®¡ç”¨äºŽ 3D LESã€‚
//! åœ¨ 2D æµ…æ°´æ–¹ç¨‹ä¸­ï¼Œå…¶ç‰©ç†æ„ä¹‰æœ‰é™ï¼Œå› ä¸ºï¼š
//!
//! 1. æ·±åº¦å¹³å‡æ¶ˆé™¤äº†åž‚å‘æ¹æµç»“æž„
//! 2. åº•éƒ¨æ‘©æ“¦é€šå¸¸æ˜¯ä¸»å¯¼è€—æ•£æœºåˆ¶
//! 3. 2D æ¹æµåŠ¨åŠ›å­¦ä¸Ž 3D æœ¬è´¨ä¸åŒ
//!
//! æŽ¨èç”¨æ³•ï¼š
//! - ä½¿ç”¨ `TurbulenceModel::None` æˆ– `TurbulenceModel::Disabled`
//! - å¦‚éœ€æ°´å¹³æ‰©æ•£ï¼Œä½¿ç”¨ `TurbulenceModel::ConstantViscosity(0.1~10.0)`

use super::traits::{TurbulenceClosure, VelocityGradient};
use crate::adapter::PhysicsMesh;
use crate::sources::traits::{SourceContribution, SourceContext, SourceTerm};
use crate::state::ShallowWaterState;

/// Smagorinsky å¸¸æ•°çš„é»˜è®¤å€¼
pub const DEFAULT_SMAGORINSKY_CONSTANT: f64 = 0.15;

/// æœ€å°æ¶¡ç²˜æ€§ç³»æ•° [mÂ²/s]
pub const MIN_EDDY_VISCOSITY: f64 = 1e-6;

/// æœ€å¤§æ¶¡ç²˜æ€§ç³»æ•° [mÂ²/s]
pub const MAX_EDDY_VISCOSITY: f64 = 1e3;

/// æ¹æµæ¨¡åž‹ç±»åž‹
/// 
/// **é‡è¦è­¦å‘Š**ï¼šæµ…æ°´æ–¹ç¨‹æ˜¯æ·±åº¦å¹³å‡æ–¹ç¨‹ï¼Œç›´æŽ¥æ·»åŠ  3D æ¹æµæ‰©æ•£é¡¹
/// åœ¨ç‰©ç†ä¸Šæ˜¯ä¸æ°å½“çš„ã€‚æ·±åº¦å¹³å‡åŽçš„æ¹æµæ•ˆåº”é€šå¸¸é€šè¿‡ä»¥ä¸‹æ–¹å¼å¤„ç†ï¼š
/// 
/// 1. **åº•éƒ¨æ‘©æ“¦**ï¼šå·²åŒ…å«åœ¨æ‘©æ“¦æ¨¡å—ä¸­ï¼Œå ä¸»å¯¼ä½œç”¨
/// 2. **æ°´å¹³æ‰©æ•£**ï¼šä½¿ç”¨é€‚å½“çš„æ°´å¹³æ¶¡ç²˜æ€§ï¼ˆä¸æ˜¯ Smagorinsky çš„ 3D å…¬å¼ï¼‰
/// 3. **è‰²æ•£é¡¹**ï¼šå¦‚ Boussinesq æ–¹ç¨‹çš„è‰²æ•£ä¿®æ­£
/// 
/// å¦‚æžœç¡®å®žéœ€è¦æ°´å¹³æ‰©æ•£ï¼Œå»ºè®®ä½¿ç”¨ `ConstantViscosity` é…åˆ
/// è¾ƒå°çš„æ¶¡ç²˜æ€§å€¼ï¼ˆ0.1-10 mÂ²/sï¼‰ï¼Œæˆ–ä½¿ç”¨ `Disabled` æ¨¡å¼ã€‚
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum TurbulenceModel {
    /// æ— æ¹æµï¼ˆæŽ¨èç”¨äºŽæµ…æ°´æ–¹ç¨‹ï¼‰
    #[default]
    None,
    /// æ˜¾å¼ç¦ç”¨ï¼ˆå¸¦è­¦å‘Šï¼‰
    /// 
    /// ä½¿ç”¨æ­¤æ¨¡å¼æ—¶ï¼Œä»£ç ä¼šè¾“å‡ºä¸€æ¬¡è­¦å‘Šæ—¥å¿—ï¼Œ
    /// æé†’ç”¨æˆ·æµ…æ°´æ–¹ç¨‹ä¸åº”ä½¿ç”¨ 3D æ¹æµæ¨¡åž‹
    Disabled,
    /// å¸¸æ•°æ¶¡ç²˜æ€§ï¼ˆä»…ç”¨äºŽæ°´å¹³æ‰©æ•£ï¼Œå»ºè®®å€¼ 0.1-10 mÂ²/sï¼‰
    ConstantViscosity(f64),
}


impl TurbulenceModel {
    /// åˆ›å»ºç¦ç”¨æ¨¡å¼ï¼ˆæŽ¨èï¼‰
    pub fn disabled() -> Self {
        Self::Disabled
    }

    /// åˆ›å»ºå¸¸æ•°æ¶¡ç²˜æ€§æ¨¡åž‹ï¼ˆä»…ç”¨äºŽæ°´å¹³æ‰©æ•£ï¼‰
    /// 
    /// # å‚æ•°
    /// - `nu`: æ¶¡ç²˜æ€§ç³»æ•° [mÂ²/s]ï¼Œå»ºè®®èŒƒå›´ 0.1-10
    pub fn constant(nu: f64) -> Self {
        Self::ConstantViscosity(nu.max(MIN_EDDY_VISCOSITY).min(MAX_EDDY_VISCOSITY))
    }

    /// æ£€æŸ¥æ¨¡åž‹æ˜¯å¦å®žé™…å¯ç”¨
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::None | Self::Disabled)
    }
}

/// Smagorinsky æ¹æµæ±‚è§£å™¨
///
/// 2D æ°´å¹³æ¶¡ç²˜æ€§è®¡ç®—å™¨ã€‚
#[derive(Debug, Clone)]
pub struct SmagorinskySolver {
    /// æ¨¡åž‹é…ç½®
    pub model: TurbulenceModel,
    /// ç½‘æ ¼å°ºåº¦ [m]ï¼ˆæ¯ä¸ªå•å…ƒï¼‰
    pub grid_scale: Vec<f64>,
    /// è®¡ç®—å¾—åˆ°çš„æ¶¡ç²˜æ€§ [mÂ²/s]ï¼ˆæ¯ä¸ªå•å…ƒï¼‰
    pub eddy_viscosity: Vec<f64>,
    /// é€Ÿåº¦æ¢¯åº¦ï¼ˆæ¯ä¸ªå•å…ƒï¼‰
    pub velocity_gradient: Vec<VelocityGradient>,
    /// æœ€å°æ°´æ·±
    pub h_min: f64,
}

impl SmagorinskySolver {
    /// åˆ›å»ºæ–°çš„æ±‚è§£å™¨
    pub fn new(n_cells: usize, model: TurbulenceModel) -> Self {
        Self {
            model,
            grid_scale: vec![10.0; n_cells], // é»˜è®¤ç½‘æ ¼å°ºåº¦
            eddy_viscosity: vec![0.0; n_cells],
            velocity_gradient: vec![VelocityGradient::default(); n_cells],
            h_min: 1e-4,
        }
    }

    /// ä»Žç½‘æ ¼åˆå§‹åŒ–
    pub fn from_mesh(mesh: &PhysicsMesh, model: TurbulenceModel) -> Self {
        let n_cells = mesh.n_cells();
        let mut solver = Self::new(n_cells, model);

        // è®¡ç®—ç½‘æ ¼å°ºåº¦ï¼ˆä½¿ç”¨å•å…ƒé¢ç§¯çš„å¹³æ–¹æ ¹ï¼‰
        for i in 0..n_cells {
            if let Some(area) = mesh.cell_area(i) {
                solver.grid_scale[i] = area.sqrt();
            }
        }

        solver
    }

    /// è®¾ç½®ç½‘æ ¼å°ºåº¦
    pub fn set_grid_scale(&mut self, i: usize, scale: f64) {
        if i < self.grid_scale.len() {
            self.grid_scale[i] = scale.max(1e-3);
        }
    }

    /// è®¾ç½®é€Ÿåº¦æ¢¯åº¦ï¼ˆå¤–éƒ¨è®¡ç®—ï¼‰
    pub fn set_velocity_gradient(&mut self, i: usize, grad: VelocityGradient) {
        if i < self.velocity_gradient.len() {
            self.velocity_gradient[i] = grad;
        }
    }

    /// æ‰¹é‡è®¾ç½®é€Ÿåº¦æ¢¯åº¦
    pub fn set_velocity_gradients(&mut self, gradients: &[VelocityGradient]) {
        let n = self.velocity_gradient.len().min(gradients.len());
        self.velocity_gradient[..n].copy_from_slice(&gradients[..n]);
    }

    /// ä½¿ç”¨ç®€å•å·®åˆ†ä¼°ç®—é€Ÿåº¦æ¢¯åº¦ï¼ˆé€‚ç”¨äºŽç»“æž„åŒ–ç½‘æ ¼ï¼‰
    ///
    /// å¯¹äºŽéžç»“æž„åŒ–ç½‘æ ¼ï¼Œåº”ä½¿ç”¨å¤–éƒ¨æ¢¯åº¦æ±‚è§£å™¨
    pub fn estimate_gradient_from_neighbors(
        &mut self,
        state: &ShallowWaterState,
        mesh: &PhysicsMesh,
    ) {
        let n_cells = self.velocity_gradient.len().min(state.h.len()).min(mesh.n_cells());

        for i in 0..n_cells {
            let h = state.h[i];
            if h < self.h_min {
                self.velocity_gradient[i] = VelocityGradient::default();
                continue;
            }

            let u = state.hu[i] / h;
            let v = state.hv[i] / h;

            // ç®€å•çš„æœ€è¿‘é‚»æ¢¯åº¦ä¼°è®¡
            let mut du_dx = 0.0;
            let mut du_dy = 0.0;
            let mut dv_dx = 0.0;
            let mut dv_dy = 0.0;
            let mut weight_sum = 0.0;

            for face_id in mesh.cell_faces(i) {
                // ä½¿ç”¨ face_neighbor èŽ·å–é‚»å±…
                if let Some(neighbor) = mesh.face_neighbor(face_id) {
                    if neighbor == i {
                        continue;
                    }
                    let h_n = state.h[neighbor];
                    if h_n < self.h_min {
                        continue;
                    }

                    let u_n = state.hu[neighbor] / h_n;
                    let v_n = state.hv[neighbor] / h_n;

                    let normal = mesh.face_normal(face_id);
                    let dist = self.grid_scale[i];

                    if dist > 1e-10 {
                        let weight = 1.0 / dist;
                        du_dx += (u_n - u) * normal.x * weight;
                        du_dy += (u_n - u) * normal.y * weight;
                        dv_dx += (v_n - v) * normal.x * weight;
                        dv_dy += (v_n - v) * normal.y * weight;
                        weight_sum += weight;
                    }
                }
            }

            if weight_sum > 1e-10 {
                self.velocity_gradient[i] = VelocityGradient::new(
                    du_dx / weight_sum,
                    du_dy / weight_sum,
                    dv_dx / weight_sum,
                    dv_dy / weight_sum,
                );
            } else {
                self.velocity_gradient[i] = VelocityGradient::default();
            }
        }
    }

    /// æ›´æ–°æ¶¡ç²˜æ€§ç³»æ•°
    pub fn update_eddy_viscosity(&mut self) {
        match &self.model {
            TurbulenceModel::None | TurbulenceModel::Disabled => {
                self.eddy_viscosity.fill(0.0);
            }
            TurbulenceModel::ConstantViscosity(nu) => {
                self.eddy_viscosity.fill(*nu);
            }
        }
    }

    /// èŽ·å–å•å…ƒæ¶¡ç²˜æ€§
    pub fn get_eddy_viscosity(&self, cell: usize) -> f64 {
        self.eddy_viscosity.get(cell).copied().unwrap_or(0.0)
    }

    /// è®¡ç®—æ¹æµæ‰©æ•£é€šé‡
    ///
    /// è¿”å›ž (Fx, Fy) åŠ¨é‡æ‰©æ•£é€šé‡
    pub fn compute_diffusion_flux(&self, cell: usize, state: &ShallowWaterState) -> (f64, f64) {
        let h = state.h[cell];
        if h < self.h_min {
            return (0.0, 0.0);
        }

        let nu = self.get_eddy_viscosity(cell);
        let grad = &self.velocity_gradient[cell];

        let fx = nu * h * grad.du_dx;
        let fy = nu * h * grad.dv_dy;

        (fx, fy)
    }
}

// å®žçŽ° TurbulenceClosure trait
impl TurbulenceClosure for SmagorinskySolver {
    fn name(&self) -> &'static str {
        "Smagorinsky"
    }
    
    fn is_3d(&self) -> bool {
        false // Smagorinsky é€‚ç”¨äºŽ 2D
    }
    
    fn eddy_viscosity(&self) -> &[f64] {
        &self.eddy_viscosity
    }
    
    fn update(&mut self, velocity_gradients: &[VelocityGradient], cell_sizes: &[f64]) {
        self.set_velocity_gradients(velocity_gradients);
        let n = self.grid_scale.len().min(cell_sizes.len());
        self.grid_scale[..n].copy_from_slice(&cell_sizes[..n]);
        self.update_eddy_viscosity();
    }
    
    fn is_enabled(&self) -> bool {
        self.model.is_active()
    }
}

/// æ¹æµæºé¡¹é…ç½®
#[derive(Debug, Clone)]
pub struct TurbulenceConfig {
    /// æ˜¯å¦å¯ç”¨
    pub enabled: bool,
    /// æ¹æµæ¨¡åž‹
    pub model: TurbulenceModel,
    /// æ¶¡ç²˜æ€§ [mÂ²/s]ï¼ˆé¢„è®¡ç®—æˆ–å¸¸æ•°ï¼‰
    pub eddy_viscosity: Vec<f64>,
    /// é€Ÿåº¦æ¢¯åº¦ï¼ˆå¤–éƒ¨æä¾›ï¼‰
    pub velocity_gradient: Vec<VelocityGradient>,
    /// æœ€å°æ°´æ·±
    pub h_min: f64,
}

impl TurbulenceConfig {
    /// åˆ›å»ºæ–°é…ç½®
    pub fn new(n_cells: usize, model: TurbulenceModel) -> Self {
        Self {
            enabled: true,
            model,
            eddy_viscosity: vec![0.0; n_cells],
            velocity_gradient: vec![VelocityGradient::default(); n_cells],
            h_min: 1e-4,
        }
    }

    /// åˆ›å»ºå¸¸æ•°æ¶¡ç²˜æ€§é…ç½®
    pub fn constant(n_cells: usize, nu: f64) -> Self {
        let mut config = Self::new(n_cells, TurbulenceModel::constant(nu));
        config.eddy_viscosity.fill(nu);
        config
    }

    /// è®¾ç½®æ¶¡ç²˜æ€§
    pub fn set_eddy_viscosity(&mut self, cell: usize, nu: f64) {
        if cell < self.eddy_viscosity.len() {
            self.eddy_viscosity[cell] = nu.max(0.0);
        }
    }

    /// æ‰¹é‡è®¾ç½®æ¶¡ç²˜æ€§
    pub fn set_eddy_viscosity_field(&mut self, nu: &[f64]) {
        let n = self.eddy_viscosity.len().min(nu.len());
        self.eddy_viscosity[..n].copy_from_slice(&nu[..n]);
    }
}

impl SourceTerm for TurbulenceConfig {
    fn name(&self) -> &'static str {
        "Turbulence"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h[cell];

        // å¹²å•å…ƒä¸è®¡ç®—
        if h < self.h_min || ctx.is_dry(h) {
            return SourceContribution::ZERO;
        }

        let nu = self.eddy_viscosity.get(cell).copied().unwrap_or(0.0);
        if nu < MIN_EDDY_VISCOSITY {
            return SourceContribution::ZERO;
        }

        let grad = self.velocity_gradient.get(cell).copied().unwrap_or_default();

        // ç²˜æ€§åº”åŠ›æºé¡¹ï¼ˆç®€åŒ–å½¢å¼ï¼‰
        let s11 = 2.0 * grad.du_dx;
        let s22 = 2.0 * grad.dv_dy;
        let s12 = grad.du_dy + grad.dv_dx;
        
        let char_length = h.max(0.1);
        
        let s_hu = nu * h * (s11 + s12) / char_length;
        let s_hv = nu * h * (s12 + s22) / char_length;
        
        // é™åˆ¶æºé¡¹å¤§å°
        let max_source = nu * h * 10.0;
        let s_hu_clamped = s_hu.clamp(-max_source, max_source);
        let s_hv_clamped = s_hv.clamp(-max_source, max_source);

        SourceContribution::momentum(s_hu_clamped, s_hv_clamped)
    }

    fn is_explicit(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NumericalParams;

    fn create_test_state(n_cells: usize, h: f64, u: f64, v: f64) -> ShallowWaterState {
        let mut state = ShallowWaterState::new(n_cells);
        for i in 0..n_cells {
            state.h[i] = h;
            state.hu[i] = h * u;
            state.hv[i] = h * v;
            state.z[i] = 0.0;
        }
        state
    }

    #[test]
    fn test_turbulence_model_default() {
        let model = TurbulenceModel::default();
        assert_eq!(model, TurbulenceModel::None);
        assert!(!model.is_active());
    }

    #[test]
    fn test_turbulence_model_disabled() {
        let model = TurbulenceModel::disabled();
        assert_eq!(model, TurbulenceModel::Disabled);
        assert!(!model.is_active());
    }

    #[test]
    fn test_turbulence_model_constant() {
        let model = TurbulenceModel::constant(0.01);
        match model {
            TurbulenceModel::ConstantViscosity(nu) => {
                assert!((nu - 0.01).abs() < 1e-10);
            }
            _ => panic!("Expected ConstantViscosity model"),
        }
    }

    #[test]
    fn test_smagorinsky_solver_creation() {
        let solver = SmagorinskySolver::new(10, TurbulenceModel::default());
        assert_eq!(solver.grid_scale.len(), 10);
        assert_eq!(solver.eddy_viscosity.len(), 10);
    }

    #[test]
    fn test_smagorinsky_solver_constant_viscosity() {
        let mut solver = SmagorinskySolver::new(10, TurbulenceModel::constant(0.1));
        solver.update_eddy_viscosity();

        for i in 0..10 {
            assert!((solver.eddy_viscosity[i] - 0.1).abs() < 1e-10);
        }
    }


    #[test]
    fn test_turbulence_config_creation() {
        let config = TurbulenceConfig::new(10, TurbulenceModel::default());
        assert!(config.enabled);
        assert_eq!(config.eddy_viscosity.len(), 10);
    }

    #[test]
    fn test_turbulence_config_constant() {
        let config = TurbulenceConfig::constant(10, 0.05);
        assert!((config.eddy_viscosity[0] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_turbulence_source_term() {
        let mut config = TurbulenceConfig::constant(10, 0.1);
        config.velocity_gradient[0] = VelocityGradient::new(1.0, 0.0, 0.0, 1.0);

        let state = create_test_state(10, 2.0, 1.0, 0.5);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

        assert_eq!(contrib.s_h, 0.0);
        assert!(contrib.s_hu > 0.0);
        assert!(contrib.s_hv > 0.0);
    }

    #[test]
    fn test_turbulence_dry_cell() {
        let config = TurbulenceConfig::constant(10, 0.1);

        let state = create_test_state(10, 1e-7, 0.0, 0.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_source_term_trait() {
        let config = TurbulenceConfig::constant(10, 0.15);
        assert_eq!(config.name(), "Turbulence");
        assert!(config.is_explicit());
    }
    
    #[test]
    fn test_turbulence_closure_trait() {
        let mut solver = SmagorinskySolver::new(10, TurbulenceModel::constant(0.5));
        assert_eq!(solver.name(), "Smagorinsky");
        assert!(!solver.is_3d());
        
        let grads = vec![VelocityGradient::default(); 10];
        let sizes = vec![10.0; 10];
        solver.update(&grads, &sizes);
        
        assert!((solver.eddy_viscosity[0] - 0.5).abs() < 1e-10);
    }
}
