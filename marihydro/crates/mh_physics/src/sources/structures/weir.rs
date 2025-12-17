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

use crate::sources::traits::{SourceContribution, SourceContext, SourceTerm};
use crate::state::{ShallowWaterState, ShallowWaterStateF64};
use crate::types::PhysicalConstants;
use mh_foundation::AlignedVec;
use serde::{Deserialize, Serialize};

/// 堰类型
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[derive(Default)]
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
    /// 创建新的堰流源项
    pub fn new(n_cells: usize, config: WeirConfig) -> Self {
        let cd_default = config.weir_type.discharge_coefficient();
        Self {
            config: config.clone(),
            constants: config.constants,
            n_cells,
            crest_elevation: AlignedVec::from_vec(vec![f64::INFINITY; n_cells]), // 默认无堰
            weir_width: AlignedVec::zeros(n_cells),
            cd_field: AlignedVec::from_vec(vec![cd_default; n_cells]),
            normal_x: AlignedVec::from_vec(vec![1.0; n_cells]), // 默认 x 方向
            normal_y: AlignedVec::zeros(n_cells),
            cell_area: AlignedVec::from_vec(vec![1.0; n_cells]), // 默认单位面积
            discharge: AlignedVec::zeros(n_cells),
        }
    }

    /// 使用默认配置创建
    pub fn with_defaults(n_cells: usize) -> Self {
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
        crest: f64, // ALLOW_F64: 物理参数
        width: f64, // ALLOW_F64: 物理参数
        cd: Option<f64>, // ALLOW_F64: 物理参数
        normal: (f64, f64), // ALLOW_F64: 几何参数
    ) {
        if cell < self.n_cells {
            self.crest_elevation[cell] = crest;
            self.weir_width[cell] = width;
            if let Some(c) = cd {
                self.cd_field[cell] = c;
            }
            // 归一化法向
            let mag = (normal.0 * normal.0 + normal.1 * normal.1).sqrt();
            if mag > 1e-10 {
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
        h_upstream: f64, // ALLOW_F64: 源项计算
        h_downstream: f64, // ALLOW_F64: 源项计算
    ) -> f64 { // ALLOW_F64: 源项计算
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
        state: &ShallowWaterStateF64,
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

        // 获取单元面积
        let area = self.cell_area[cell].max(1e-10);

        // 质量源项：s_h = -Q/A（负值表示出流）
        let s_h = -q / area;

        // 动量源项：假设过堰流速沿法向方向
        // 过堰流速估计：v_weir = Q / (B × H_head)
        let head = (water_level - crest).max(self.config.h_min);
        let width = self.weir_width[cell].max(1e-10);
        let v_weir = q / (width * head);

        // 动量损失沿法向方向
        let nx = self.normal_x[cell];
        let ny = self.normal_y[cell];

        // s_hu = s_h × v_weir × nx（出流带走动量）
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
    fn create_test_state(n_cells: usize, h: f64, z: f64) -> ShallowWaterStateF64 {
        let mut state = ShallowWaterStateF64::new(n_cells);
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
        assert!((q).abs() < 1e-10); // 无堰
    }

    #[test]
    fn test_with_weir() {
        let mut weir = WeirFlow::with_defaults(10);
        weir.set_weir(0, 2.0, 10.0, None, (1.0, 0.0)); // 堰顶2m，宽10m

        let q = weir.compute_discharge(0, 3.0); // 水位3m，水头1m
        
        // Q = 0.35 × 10 × 1^1.5 × √(2×9.81) ≈ 15.5 m³/s
        assert!(q > 10.0);
        assert!(q < 20.0);
    }

    #[test]
    fn test_submerged_flow() {
        let mut weir = WeirFlow::with_defaults(10);
        weir.set_weir(0, 2.0, 10.0, None, (1.0, 0.0));

        let q_free = weir.compute_discharge_from_head(0, 1.0);
        let q_submerged = weir.compute_discharge_submerged(0, 1.0, 0.5);

        // 淹没流量 < 自由流量
        assert!(q_submerged < q_free);
        assert!(q_submerged > 0.0);
    }
}
