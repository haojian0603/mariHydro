//! 桥墩亚网格拖曳力
//!
//! 实现桥墩对水流的阻力效应：
//! - 拖曳力 F = 0.5 × ρ × Cd × A_block × |u| × u
//! - 阻塞修正
//!
//! # 物理背景
//!
//! 当网格尺寸大于桥墩直径时，无法直接解析桥墩边界。
//! 通过亚网格参数化方法将桥墩效应作为动量源项添加。

use crate::sources::traits::{SourceContribution, SourceContext, SourceTerm};
use crate::state::{ShallowWaterState, ShallowWaterStateF64};
use crate::types::PhysicalConstants;
use mh_foundation::AlignedVec;
use serde::{Deserialize, Serialize};

/// 桥墩拖曳力配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgePierConfig {
    /// 是否启用
    pub enabled: bool,
    /// 默认拖曳系数
    pub default_cd: f64, // ALLOW_F64: Layer 4 配置参数
    /// 物理常量（唯一真理源）
    pub constants: PhysicalConstants,
    /// 最小水深 [m]
    pub h_min: f64, // ALLOW_F64: Layer 4 配置参数
}

impl Default for BridgePierConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_cd: 1.2, // 圆柱体典型值
            constants: PhysicalConstants::seawater(),
            h_min: 0.01,
        }
    }
}

/// 桥墩亚网格拖曳力源项
pub struct BridgePierDrag {
    /// 配置
    config: BridgePierConfig,
    /// 物理常量缓存
    constants: PhysicalConstants,
    /// 阻塞率场 (0~1)：桥墩占单元面积的比例
    pub blockage: AlignedVec<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 拖曳系数场
    pub drag_coeff: AlignedVec<f64>, // ALLOW_F64: Layer 4 配置参数
}

impl BridgePierDrag {
    /// 创建新的桥墩源项
    pub fn new(n_cells: usize, config: BridgePierConfig) -> Self {
        Self {
            config: config.clone(),
            constants: config.constants,
            blockage: AlignedVec::zeros(n_cells),
            drag_coeff: AlignedVec::from_vec(vec![config.default_cd; n_cells]),
        }
    }

    /// 使用默认配置创建
    pub fn with_defaults(n_cells: usize) -> Self {
        Self::new(n_cells, BridgePierConfig::default())
    }

    /// 设置单元的桥墩参数
    ///
    /// # 参数
    /// - `cell`: 单元索引
    /// - `blockage`: 阻塞率 (0~1)
    /// - `cd`: 拖曳系数（None 使用默认值）
    // ALLOW_F64: 物理参数
    pub fn set_pier(&mut self, cell: usize, blockage: f64, cd: Option<f64>) {
        if cell < self.blockage.len() {
            self.blockage[cell] = blockage.clamp(0.0, 1.0);
            if let Some(c) = cd {
                self.drag_coeff[cell] = c.max(0.0);
            }
        }
    }

    /// 从桥墩几何参数计算阻塞率
    ///
    /// # 参数
    /// - `cell`: 单元索引
    /// - `pier_diameter`: 桥墩直径 [m]
    /// - `cell_width`: 单元宽度 [m]
    // ALLOW_F64: 物理参数
    pub fn set_from_geometry(&mut self, cell: usize, pier_diameter: f64, cell_width: f64) {
        if cell < self.blockage.len() {
            let blockage = (pier_diameter / cell_width).clamp(0.0, 0.9);
            self.blockage[cell] = blockage;
        }
    }

    /// 批量设置阻塞率
    pub fn set_blockage_field(&mut self, blockage: &[f64]) {
        let n = self.blockage.len().min(blockage.len());
        self.blockage[..n].copy_from_slice(&blockage[..n]);
    }

    /// 计算单元的拖曳力 [N/m²]
    // ALLOW_F64: 源项计算
    fn compute_drag_force(&self, cell: usize, h: f64, u: f64, v: f64) -> (f64, f64) {
        let ab = self.blockage[cell];
        if ab < 1e-10 {
            return (0.0, 0.0);
        }

        let cd = self.drag_coeff[cell];
        let speed = (u * u + v * v).sqrt();

        if speed < 1e-10 {
            return (0.0, 0.0);
        }

        // F = 0.5 × ρ × Cd × A_block × |u| × u / h
        // 除以 h 是因为动量方程是单位体积的
        let factor = 0.5 * self.constants.rho_water * cd * ab * speed / h.max(0.01);

        (-factor * u, -factor * v)
    }
}

impl SourceTerm for BridgePierDrag {
    fn name(&self) -> &'static str {
        "BridgePierDrag"
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
        let h = state.h[cell];
        if h < self.config.h_min {
            return SourceContribution::ZERO;
        }

        let u = state.hu[cell] / h;
        let v = state.hv[cell] / h;

        let (f_x, f_y) = self.compute_drag_force(cell, h, u, v);

        SourceContribution::momentum(f_x, f_y)
    }

    fn is_explicit(&self) -> bool {
        true // 显式处理
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
        }
        state
    }

    #[test]
    fn test_pier_creation() {
        let pier = BridgePierDrag::with_defaults(10);
        assert_eq!(pier.blockage.len(), 10);
    }

    #[test]
    fn test_zero_blockage() {
        let pier = BridgePierDrag::with_defaults(10);
        let state = create_test_state(10, 2.0, 1.0, 0.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = pier.compute_cell(&state, 0, &ctx);
        
        // 无桥墩 → 无阻力
        assert!((contrib.s_hu).abs() < 1e-10);
    }

    #[test]
    fn test_with_blockage() {
        let mut pier = BridgePierDrag::with_defaults(10);
        pier.set_pier(0, 0.2, None); // 20% 阻塞

        let state = create_test_state(10, 2.0, 1.0, 0.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = pier.compute_cell(&state, 0, &ctx);
        
        // 有桥墩 → 负 x 动量源
        assert!(contrib.s_hu < 0.0);
    }

    #[test]
    fn test_geometry_setup() {
        let mut pier = BridgePierDrag::with_defaults(10);
        pier.set_from_geometry(0, 2.0, 10.0); // 2m 墩径，10m 单元宽度

        assert!((pier.blockage[0] - 0.2).abs() < 1e-10);
    }
}

