//! 堰流模型
//!
//! 实现各种堰流公式：
//! - 宽顶堰
//! - 锐缘堰
//! - 实用堰
//!
//! # 堰流公式
//!
//! ## 自由出流
//! ```text
//! Q = Cd × B × H^1.5 × √(2g)
//! ```
//!
//! ## 淹没出流
//! ```text
//! Q = Cd × B × H^1.5 × √(2g) × S
//! ```
//! 其中 S 为淹没修正系数

use crate::sources::traits::{
    SourceContextGeneric, SourceContributionGeneric, SourceStiffness, SourceTermGeneric,
};
use crate::state::ShallowWaterState;
use crate::types::PhysicalConstants;
use mh_foundation::error::MhResult;
use mh_foundation::AlignedVec;
use mh_runtime::{Backend, RuntimeScalar};
use num_traits::Float;
use serde::{Deserialize, Serialize};

/// 堰类型
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum WeirType {
    /// 宽顶堰（Cd ≈ 0.34-0.36）
    #[default]
    BroadCrested,
    /// 锐缘堰（Cd ≈ 0.42）
    SharpCrested,
    /// 实用堰（Cd ≈ 0.40-0.48）
    Practical,
    /// 自定义流量系数
    // ALLOW_F64: Layer 4 配置参数
    Custom { cd: f64 },
}

impl WeirType {
    /// 获取流量系数
    // ALLOW_F64: 源项计算
    pub fn discharge_coefficient(&self) -> f64 {
        match self {
            Self::BroadCrested => 0.35,
            Self::SharpCrested => 0.42,
            Self::Practical => 0.44,
            Self::Custom { cd } => *cd,
        }
    }
}

/// 堰流配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeirConfig {
    /// 是否启用
    pub enabled: bool,
    /// 堰类型
    pub weir_type: WeirType,
    /// 物理常量（唯一真理源）
    pub constants: PhysicalConstants,
    /// 最小水头 [m]
    pub h_min: f64, // ALLOW_F64: Layer 4 配置参数
}

impl Default for WeirConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            weir_type: WeirType::BroadCrested,
            constants: PhysicalConstants::seawater(),
            h_min: 0.001,
        }
    }
}

/// 堰流源项
pub struct WeirFlow {
    /// 配置
    config: WeirConfig,
    /// 物理常数缓存
    constants: PhysicalConstants,
    /// 单元数
    n_cells: usize,
    /// 堰顶高程场 [m]
    pub crest_elevation: AlignedVec<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 堰宽度场 [m]（通常等于单元宽度）
    pub weir_width: AlignedVec<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 流量系数场（覆盖默认值）
    pub cd_field: AlignedVec<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 堰法向（指向下游）x 分量
    pub normal_x: AlignedVec<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 堰法向 y 分量
    pub normal_y: AlignedVec<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 单元面积 [m²]
    pub cell_area: AlignedVec<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 计算得到的过堰流量 [m³/s]
    discharge: AlignedVec<f64>, // ALLOW_F64: 源项计算
}

impl WeirFlow {
    const NORMAL_EPS: f64 = 1e-10;

    /// 创建新的堰流源项
    pub fn new(n_cells: usize, config: WeirConfig) -> MhResult<Self> {
        let cd_default = config.weir_type.discharge_coefficient();
        Ok(Self {
            config: config.clone(),
            constants: config.constants,
            n_cells,
            crest_elevation: AlignedVec::from_vec(vec![f64::INFINITY; n_cells])?, // 默认无堰
            weir_width: AlignedVec::zeros(n_cells),
            cd_field: AlignedVec::from_vec(vec![cd_default; n_cells])?,
            normal_x: AlignedVec::from_vec(vec![1.0; n_cells])?, // 默认 x 方向
            normal_y: AlignedVec::zeros(n_cells),
            cell_area: AlignedVec::from_vec(vec![1.0; n_cells])?, // 默认单位面积
            discharge: AlignedVec::zeros(n_cells),
        })
    }

    /// 使用默认配置创建
    pub fn with_defaults(n_cells: usize) -> MhResult<Self> {
        Self::new(n_cells, WeirConfig::default())
    }

    /// 设置堰参数
    ///
    /// # 参数
    /// - `cell`: 单元索引
    /// - `crest`: 堰顶高程 [m]
    /// - `width`: 堰宽 [m]
    /// - `cd`: 流量系数（None 使用默认值）
    pub fn set_weir(
        &mut self,
        cell: usize,
        crest: f64,         // ALLOW_F64: 物理参数
        width: f64,         // ALLOW_F64: 物理参数
        cd: Option<f64>,    // ALLOW_F64: 物理参数
        normal: (f64, f64), // ALLOW_F64: 几何参数
    ) {
        if cell < self.n_cells {
            self.crest_elevation[cell] = crest;
            self.weir_width[cell] = width.max(0.0);
            if let Some(c) = cd {
                self.cd_field[cell] = c.max(0.0);
            }
            // 归一化法向
            let mag = (normal.0 * normal.0 + normal.1 * normal.1).sqrt();
            if mag > Self::NORMAL_EPS {
                self.normal_x[cell] = normal.0 / mag;
                self.normal_y[cell] = normal.1 / mag;
            }
        }
    }

    /// 计算过堰流量
    ///
    /// # 返回
    /// 流量 [m³/s]，正值表示流向法向正方向
    // ALLOW_F64: 源项计算
    pub fn compute_discharge(&self, cell: usize, water_level: f64) -> f64 {
        let crest = self.crest_elevation[cell];
        if crest.is_infinite() {
            return 0.0; // 无堰
        }

        let head = water_level - crest;
        if head < self.config.h_min {
            return 0.0; // 无过堰流量
        }

        let cd = self.cd_field[cell];
        let width = self.weir_width[cell];

        // 自由出流：Q = Cd × B × H^1.5 × √(2g)

        cd * width * head.powf(1.5) * (2.0 * self.constants.g).sqrt()
    }

    /// 计算淹没出流
    ///
    /// # 参数
    /// - `h_upstream`: 上游水头 [m]
    /// - `h_downstream`: 下游水头 [m]（相对于堰顶）
    pub fn compute_discharge_submerged(
        &self,
        cell: usize,
        h_upstream: f64,   // ALLOW_F64: 源项计算
        h_downstream: f64, // ALLOW_F64: 源项计算
    ) -> f64 {
        // ALLOW_F64: 源项计算
        if h_upstream < self.config.h_min {
            return 0.0;
        }

        let q_free = self.compute_discharge_from_head(cell, h_upstream);

        // Villemonte 淹没修正
        // S = (1 - (h2/h1)^1.5)^0.385
        let ratio = (h_downstream / h_upstream).max(0.0).min(1.0);
        let submergence = (1.0 - ratio.powf(1.5)).powf(0.385);

        q_free * submergence
    }

    /// 从水头计算流量
    // ALLOW_F64: 源项计算
    fn compute_discharge_from_head(&self, cell: usize, head: f64) -> f64 {
        let cd = self.cd_field[cell];
        let width = self.weir_width[cell];

        cd * width * head.powf(1.5) * (2.0 * self.constants.g).sqrt()
    }

    /// 获取计算的流量场
    pub fn discharge(&self) -> &[f64] {
        &self.discharge
    }

    /// 设置单元面积
    pub fn set_cell_areas(&mut self, areas: &[f64]) {
        let n = self.n_cells.min(areas.len());
        for (dst, src) in self.cell_area[..n].iter_mut().zip(&areas[..n]) {
            *dst = src.max(0.0);
        }
    }

    fn compute_discharge_generic<B: Backend>(
        &self,
        backend: &B,
        cell: usize,
        water_level: B::Scalar,
    ) -> B::Scalar {
        if !self.crest_elevation[cell].is_finite() {
            return B::Scalar::ZERO;
        }
        let crest = backend.config_scalar(self.crest_elevation[cell], "WeirFlow.crest_elevation");

        let head = water_level - crest;
        let h_min = backend.config_scalar(self.config.h_min, "WeirFlow.h_min");
        if head < h_min {
            return B::Scalar::ZERO;
        }

        let cd = backend.config_scalar(self.cd_field[cell], "WeirFlow.cd");
        let width = backend.config_scalar(self.weir_width[cell], "WeirFlow.width");
        let exponent = backend.config_scalar(1.5, "WeirFlow.head_exponent");
        let gravity = backend.config_scalar(self.constants.g, "WeirFlow.gravity");
        let two = backend.config_scalar(2.0, "WeirFlow.gravity_factor");

        cd * width * head.safe_powf(exponent) * (two * gravity).safe_sqrt()
    }
}

impl<B: Backend> SourceTermGeneric<B> for WeirFlow {
    fn name(&self) -> &'static str {
        "WeirFlow"
    }
    fn stiffness(&self) -> SourceStiffness {
        SourceStiffness::Explicit
    }
    fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        if !self.crest_elevation[cell].is_finite() {
            return SourceContributionGeneric::default();
        }
        let crest = state
            .backend()
            .config_scalar(self.crest_elevation[cell], "WeirFlow.crest_elevation");
        let h = state.h[cell];
        let z = state.z[cell];
        let water_level = h + z;
        let q = self.compute_discharge_generic(state.backend(), cell, water_level);
        let zero_cutoff = state.backend().config_scalar(1e-10, "WeirFlow.zero_cutoff");
        if q.abs() < zero_cutoff || ctx.is_dry(h) {
            return SourceContributionGeneric::default();
        }
        let area_raw = state
            .backend()
            .config_scalar(self.cell_area[cell], "WeirFlow.cell_area");
        let area = if area_raw < zero_cutoff {
            zero_cutoff
        } else {
            area_raw
        };
        let s_h = -q / area;
        let h_min = state
            .backend()
            .config_scalar(self.config.h_min, "WeirFlow.h_min");
        let head_raw = water_level - crest;
        let head = if head_raw < h_min { h_min } else { head_raw };
        let width_raw = state
            .backend()
            .config_scalar(self.weir_width[cell], "WeirFlow.width");
        let width = if width_raw < zero_cutoff {
            zero_cutoff
        } else {
            width_raw
        };
        let v_weir = q / (width * head);
        let nx = state
            .backend()
            .config_scalar(self.normal_x[cell], "WeirFlow.normal_x");
        let ny = state
            .backend()
            .config_scalar(self.normal_y[cell], "WeirFlow.normal_y");
        SourceContributionGeneric::new(s_h, s_h * v_weir * nx, s_h * v_weir * ny)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<B>,
        rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !self.config.enabled {
            return;
        }
        let n = state
            .n_cells()
            .min(self.n_cells)
            .min(rhs_h.len())
            .min(rhs_hu.len())
            .min(rhs_hv.len());
        for cell in 0..n {
            let contrib = SourceTermGeneric::compute_cell(self, cell, state, ctx);
            rhs_h[cell] += contrib.s_h;
            rhs_hu[cell] += contrib.s_hu;
            rhs_hv[cell] += contrib.s_hv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weir_type_cd() {
        assert!((WeirType::BroadCrested.discharge_coefficient() - 0.35).abs() < 1e-10);
        assert!((WeirType::SharpCrested.discharge_coefficient() - 0.42).abs() < 1e-10);
    }

    #[test]
    fn test_weir_creation() {
        let weir = WeirFlow::with_defaults(10).unwrap();
        assert_eq!(weir.n_cells, 10);
    }

    #[test]
    fn test_no_weir() {
        let weir = WeirFlow::with_defaults(10).unwrap();
        let q = weir.compute_discharge(0, 5.0);
        assert!((q).abs() < 1e-10); // 无堰
    }

    #[test]
    fn test_with_weir() {
        let mut weir = WeirFlow::with_defaults(10).unwrap();
        weir.set_weir(0, 2.0, 10.0, None, (1.0, 0.0)); // 堰顶2m，宽10m

        let q = weir.compute_discharge(0, 3.0); // 水位3m，水头1m

        // Q = 0.35 × 10 × 1^1.5 × √(2×9.81) ≈ 15.5 m³/s
        assert!(q > 10.0);
        assert!(q < 20.0);
    }

    #[test]
    fn test_submerged_flow() {
        let mut weir = WeirFlow::with_defaults(10).unwrap();
        weir.set_weir(0, 2.0, 10.0, None, (1.0, 0.0));

        let q_free = weir.compute_discharge_from_head(0, 1.0);
        let q_submerged = weir.compute_discharge_submerged(0, 1.0, 0.5);

        // 淹没流量 < 自由流量
        assert!(q_submerged < q_free);
        assert!(q_submerged > 0.0);
    }
}
