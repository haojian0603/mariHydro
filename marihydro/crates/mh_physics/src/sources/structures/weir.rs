//! å °æµæ¨¡åž‹
//!
//! å®žçŽ°å„ç§å °æµå…¬å¼ï¼š
//! - å®½é¡¶å °
//! - é”ç¼˜å °
//! - å®žç”¨å °
//!
//! # å °æµå…¬å¼
//!
//! ## è‡ªç”±å‡ºæµ
//! ```text
//! Q = Cd Ã— B Ã— H^1.5 Ã— âˆš(2g)
//! ```
//!
//! ## æ·¹æ²¡å‡ºæµ
//! ```text
//! Q = Cd Ã— B Ã— H^1.5 Ã— âˆš(2g) Ã— S
//! ```
//! å…¶ä¸­ S ä¸ºæ·¹æ²¡ä¿®æ­£ç³»æ•°

use crate::sources::traits::{SourceContribution, SourceContext, SourceTerm};
use crate::state::ShallowWaterState;
use mh_foundation::AlignedVec;
use serde::{Deserialize, Serialize};

/// é‡åŠ›åŠ é€Ÿåº¦
const G: f64 = 9.81;

/// å °ç±»åž‹
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[derive(Default)]
pub enum WeirType {
    /// å®½é¡¶å °ï¼ˆCd â‰ˆ 0.34-0.36ï¼‰
    #[default]
    BroadCrested,
    /// é”ç¼˜å °ï¼ˆCd â‰ˆ 0.42ï¼‰
    SharpCrested,
    /// å®žç”¨å °ï¼ˆCd â‰ˆ 0.40-0.48ï¼‰
    Practical,
    /// è‡ªå®šä¹‰æµé‡ç³»æ•°
    Custom { cd: f64 },
}


impl WeirType {
    /// èŽ·å–æµé‡ç³»æ•°
    pub fn discharge_coefficient(&self) -> f64 {
        match self {
            Self::BroadCrested => 0.35,
            Self::SharpCrested => 0.42,
            Self::Practical => 0.44,
            Self::Custom { cd } => *cd,
        }
    }
}

/// å °æµé…ç½®
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeirConfig {
    /// æ˜¯å¦å¯ç”¨
    pub enabled: bool,
    /// å °ç±»åž‹
    pub weir_type: WeirType,
    /// æ°´å¯†åº¦ [kg/mÂ³]
    pub rho_water: f64,
    /// æœ€å°æ°´å¤´ [m]
    pub h_min: f64,
}

impl Default for WeirConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            weir_type: WeirType::BroadCrested,
            rho_water: 1000.0,
            h_min: 0.001,
        }
    }
}

/// å °æµæºé¡¹
pub struct WeirFlow {
    /// é…ç½®
    config: WeirConfig,
    /// å•å…ƒæ•°
    n_cells: usize,
    /// å °é¡¶é«˜ç¨‹åœº [m]
    pub crest_elevation: AlignedVec<f64>,
    /// å °å®½åº¦åœº [m]ï¼ˆé€šå¸¸ç­‰äºŽå•å…ƒå®½åº¦ï¼‰
    pub weir_width: AlignedVec<f64>,
    /// æµé‡ç³»æ•°åœºï¼ˆè¦†ç›–é»˜è®¤å€¼ï¼‰
    pub cd_field: AlignedVec<f64>,
    /// å °æ³•å‘ï¼ˆæŒ‡å‘ä¸‹æ¸¸ï¼‰x åˆ†é‡
    pub normal_x: AlignedVec<f64>,
    /// å °æ³•å‘ y åˆ†é‡
    pub normal_y: AlignedVec<f64>,
    /// å•å…ƒé¢ç§¯ [mÂ²]
    pub cell_area: AlignedVec<f64>,
    /// è®¡ç®—å¾—åˆ°çš„è¿‡å °æµé‡ [mÂ³/s]
    discharge: AlignedVec<f64>,
}

impl WeirFlow {
    /// åˆ›å»ºæ–°çš„å °æµæºé¡¹
    pub fn new(n_cells: usize, config: WeirConfig) -> Self {
        let cd_default = config.weir_type.discharge_coefficient();
        Self {
            config,
            n_cells,
            crest_elevation: AlignedVec::from_vec(vec![f64::INFINITY; n_cells]), // é»˜è®¤æ— å °
            weir_width: AlignedVec::zeros(n_cells),
            cd_field: AlignedVec::from_vec(vec![cd_default; n_cells]),
            normal_x: AlignedVec::from_vec(vec![1.0; n_cells]), // é»˜è®¤ x æ–¹å‘
            normal_y: AlignedVec::zeros(n_cells),
            cell_area: AlignedVec::from_vec(vec![1.0; n_cells]), // é»˜è®¤å•ä½é¢ç§¯
            discharge: AlignedVec::zeros(n_cells),
        }
    }

    /// ä½¿ç”¨é»˜è®¤é…ç½®åˆ›å»º
    pub fn with_defaults(n_cells: usize) -> Self {
        Self::new(n_cells, WeirConfig::default())
    }

    /// è®¾ç½®å °å‚æ•°
    ///
    /// # å‚æ•°
    /// - `cell`: å•å…ƒç´¢å¼•
    /// - `crest`: å °é¡¶é«˜ç¨‹ [m]
    /// - `width`: å °å®½ [m]
    /// - `cd`: æµé‡ç³»æ•°ï¼ˆNone ä½¿ç”¨é»˜è®¤å€¼ï¼‰
    pub fn set_weir(
        &mut self,
        cell: usize,
        crest: f64,
        width: f64,
        cd: Option<f64>,
        normal: (f64, f64),
    ) {
        if cell < self.n_cells {
            self.crest_elevation[cell] = crest;
            self.weir_width[cell] = width;
            if let Some(c) = cd {
                self.cd_field[cell] = c;
            }
            // å½’ä¸€åŒ–æ³•å‘
            let mag = (normal.0 * normal.0 + normal.1 * normal.1).sqrt();
            if mag > 1e-10 {
                self.normal_x[cell] = normal.0 / mag;
                self.normal_y[cell] = normal.1 / mag;
            }
        }
    }

    /// è®¡ç®—è¿‡å °æµé‡
    ///
    /// # è¿”å›ž
    /// æµé‡ [mÂ³/s]ï¼Œæ­£å€¼è¡¨ç¤ºæµå‘æ³•å‘æ­£æ–¹å‘
    pub fn compute_discharge(&self, cell: usize, water_level: f64) -> f64 {
        let crest = self.crest_elevation[cell];
        if crest.is_infinite() {
            return 0.0; // æ— å °
        }

        let head = water_level - crest;
        if head < self.config.h_min {
            return 0.0; // æ— è¿‡å °æµé‡
        }

        let cd = self.cd_field[cell];
        let width = self.weir_width[cell];

        // è‡ªç”±å‡ºæµï¼šQ = Cd Ã— B Ã— H^1.5 Ã— âˆš(2g)
        

        cd * width * head.powf(1.5) * (2.0 * G).sqrt()
    }

    /// è®¡ç®—æ·¹æ²¡å‡ºæµ
    ///
    /// # å‚æ•°
    /// - `h_upstream`: ä¸Šæ¸¸æ°´å¤´ [m]
    /// - `h_downstream`: ä¸‹æ¸¸æ°´å¤´ [m]ï¼ˆç›¸å¯¹äºŽå °é¡¶ï¼‰
    pub fn compute_discharge_submerged(
        &self,
        cell: usize,
        h_upstream: f64,
        h_downstream: f64,
    ) -> f64 {
        if h_upstream < self.config.h_min {
            return 0.0;
        }

        let q_free = self.compute_discharge_from_head(cell, h_upstream);

        // Villemonte æ·¹æ²¡ä¿®æ­£
        // S = (1 - (h2/h1)^1.5)^0.385
        let ratio = (h_downstream / h_upstream).max(0.0).min(1.0);
        let submergence = (1.0 - ratio.powf(1.5)).powf(0.385);

        q_free * submergence
    }

    /// ä»Žæ°´å¤´è®¡ç®—æµé‡
    fn compute_discharge_from_head(&self, cell: usize, head: f64) -> f64 {
        let cd = self.cd_field[cell];
        let width = self.weir_width[cell];

        cd * width * head.powf(1.5) * (2.0 * G).sqrt()
    }

    /// èŽ·å–è®¡ç®—çš„æµé‡åœº
    pub fn discharge(&self) -> &[f64] {
        &self.discharge
    }

    /// è®¾ç½®å•å…ƒé¢ç§¯
    pub fn set_cell_areas(&mut self, areas: &[f64]) {
        let n = self.n_cells.min(areas.len());
        self.cell_area[..n].copy_from_slice(&areas[..n]);
    }
}

impl SourceTerm for WeirFlow {
    fn name(&self) -> &'static str {
        "WeirFlow"
    }

    fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    fn compute_cell(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        _ctx: &SourceContext,
    ) -> SourceContribution {
        let crest = self.crest_elevation[cell];
        if crest.is_infinite() {
            return SourceContribution::ZERO;
        }

        let h = state.h[cell];
        let z = state.z[cell];
        let water_level = h + z;

        let q = self.compute_discharge(cell, water_level);
        if q.abs() < 1e-10 {
            return SourceContribution::ZERO;
        }

        // èŽ·å–å•å…ƒé¢ç§¯
        let area = self.cell_area[cell].max(1e-10);

        // è´¨é‡æºé¡¹ï¼šs_h = -Q/Aï¼ˆè´Ÿå€¼è¡¨ç¤ºå‡ºæµï¼‰
        let s_h = -q / area;

        // åŠ¨é‡æºé¡¹ï¼šå‡è®¾è¿‡å °æµé€Ÿæ²¿æ³•å‘æ–¹å‘
        // è¿‡å °æµé€Ÿä¼°è®¡ï¼šv_weir = Q / (B Ã— H_head)
        let head = (water_level - crest).max(self.config.h_min);
        let width = self.weir_width[cell].max(1e-10);
        let v_weir = q / (width * head);

        // åŠ¨é‡æŸå¤±æ²¿æ³•å‘æ–¹å‘
        let nx = self.normal_x[cell];
        let ny = self.normal_y[cell];

        // s_hu = s_h Ã— v_weir Ã— nxï¼ˆå‡ºæµå¸¦èµ°åŠ¨é‡ï¼‰
        let s_hu = s_h * v_weir * nx;
        let s_hv = s_h * v_weir * ny;

        SourceContribution::new(s_h, s_hu, s_hv)
    }

    fn is_explicit(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn create_test_state(n_cells: usize, h: f64, z: f64) -> ShallowWaterState {
        let mut state = ShallowWaterState::new(n_cells);
        for i in 0..n_cells {
            state.h[i] = h;
            state.z[i] = z;
        }
        state
    }

    #[test]
    fn test_weir_type_cd() {
        assert!((WeirType::BroadCrested.discharge_coefficient() - 0.35).abs() < 1e-10);
        assert!((WeirType::SharpCrested.discharge_coefficient() - 0.42).abs() < 1e-10);
    }

    #[test]
    fn test_weir_creation() {
        let weir = WeirFlow::with_defaults(10);
        assert_eq!(weir.n_cells, 10);
    }

    #[test]
    fn test_no_weir() {
        let weir = WeirFlow::with_defaults(10);
        let q = weir.compute_discharge(0, 5.0);
        assert!((q).abs() < 1e-10); // æ— å °
    }

    #[test]
    fn test_with_weir() {
        let mut weir = WeirFlow::with_defaults(10);
        weir.set_weir(0, 2.0, 10.0, None, (1.0, 0.0)); // å °é¡¶2mï¼Œå®½10m

        let q = weir.compute_discharge(0, 3.0); // æ°´ä½3mï¼Œæ°´å¤´1m
        
        // Q = 0.35 Ã— 10 Ã— 1^1.5 Ã— âˆš(2Ã—9.81) â‰ˆ 15.5 mÂ³/s
        assert!(q > 10.0);
        assert!(q < 20.0);
    }

    #[test]
    fn test_submerged_flow() {
        let mut weir = WeirFlow::with_defaults(10);
        weir.set_weir(0, 2.0, 10.0, None, (1.0, 0.0));

        let q_free = weir.compute_discharge_from_head(0, 1.0);
        let q_submerged = weir.compute_discharge_submerged(0, 1.0, 0.5);

        // æ·¹æ²¡æµé‡ < è‡ªç”±æµé‡
        assert!(q_submerged < q_free);
        assert!(q_submerged > 0.0);
    }
}
