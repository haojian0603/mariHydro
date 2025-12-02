// src-tauri/src/marihydro/core/traits/state.rs

//! 状态访问接口
//!
//! 定义浅水方程守恒变量的访问抽象。

use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::types::{CellIndex, NumericalParams, SafeVelocity};
use glam::DVec2;

/// 守恒状态（浅水方程）
#[derive(Debug, Clone, Copy, Default)]
pub struct ConservedState {
    /// 水深 [m]
    pub h: f64,
    /// x方向动量 [m²/s]
    pub hu: f64,
    /// y方向动量 [m²/s]
    pub hv: f64,
}

impl ConservedState {
    /// 创建新状态
    pub const fn new(h: f64, hu: f64, hv: f64) -> Self {
        Self { h, hu, hv }
    }

    /// 零状态
    pub const ZERO: Self = Self {
        h: 0.0,
        hu: 0.0,
        hv: 0.0,
    };

    /// 计算速度
    pub fn velocity(&self, params: &NumericalParams) -> SafeVelocity {
        params.safe_velocity(self.hu, self.hv, self.h)
    }

    /// 计算动能
    pub fn kinetic_energy(&self, params: &NumericalParams) -> f64 {
        let vel = self.velocity(params);
        0.5 * self.h * vel.speed_squared()
    }

    /// 是否为干单元
    pub fn is_dry(&self, params: &NumericalParams) -> bool {
        params.is_dry(self.h)
    }

    /// 状态加法
    pub fn add(&self, other: &Self) -> Self {
        Self {
            h: self.h + other.h,
            hu: self.hu + other.hu,
            hv: self.hv + other.hv,
        }
    }

    /// 状态标量乘法
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            h: self.h * factor,
            hu: self.hu * factor,
            hv: self.hv * factor,
        }
    }
}

/// 状态只读访问接口
pub trait StateAccess: Send + Sync {
    /// 单元数量
    fn n_cells(&self) -> usize;

    /// 获取单元状态
    fn get(&self, cell: CellIndex) -> ConservedState;

    /// 获取水深
    fn h(&self, cell: CellIndex) -> f64;

    /// 获取x动量
    fn hu(&self, cell: CellIndex) -> f64;

    /// 获取y动量
    fn hv(&self, cell: CellIndex) -> f64;

    /// 获取底床高程
    fn z(&self, cell: CellIndex) -> f64;

    /// 获取水位（水深 + 底床）
    fn eta(&self, cell: CellIndex) -> f64 {
        self.h(cell) + self.z(cell)
    }

    /// 计算速度
    fn velocity(&self, cell: CellIndex, params: &NumericalParams) -> SafeVelocity {
        params.safe_velocity(self.hu(cell), self.hv(cell), self.h(cell))
    }

    /// 判断是否为干单元
    fn is_dry(&self, cell: CellIndex, params: &NumericalParams) -> bool {
        params.is_dry(self.h(cell))
    }

    // ===== 批量访问 =====

    /// 水深数组引用
    fn h_slice(&self) -> &[f64];

    /// x动量数组引用
    fn hu_slice(&self) -> &[f64];

    /// y动量数组引用
    fn hv_slice(&self) -> &[f64];

    /// 底床高程数组引用
    fn z_slice(&self) -> &[f64];
}

/// 状态可变访问接口
pub trait StateAccessMut: StateAccess {
    /// 设置单元状态
    fn set(&mut self, cell: CellIndex, state: ConservedState);

    /// 设置水深
    fn set_h(&mut self, cell: CellIndex, value: f64);

    /// 设置x动量
    fn set_hu(&mut self, cell: CellIndex, value: f64);

    /// 设置y动量
    fn set_hv(&mut self, cell: CellIndex, value: f64);

    /// 设置底床高程
    fn set_z(&mut self, cell: CellIndex, value: f64);

    // ===== 批量可变访问 =====

    /// 水深数组可变引用
    fn h_slice_mut(&mut self) -> &mut [f64];

    /// x动量数组可变引用
    fn hu_slice_mut(&mut self) -> &mut [f64];

    /// y动量数组可变引用
    fn hv_slice_mut(&mut self) -> &mut [f64];

    /// 底床高程数组可变引用
    fn z_slice_mut(&mut self) -> &mut [f64];

    /// 应用通量更新
    fn apply_flux_update(
        &mut self,
        dt: f64,
        areas: &[f64],
        flux_h: &[f64],
        flux_hu: &[f64],
        flux_hv: &[f64],
    ) {
        let n = self.n_cells();
        for i in 0..n {
            let inv_area = 1.0 / areas[i];
            let cell = CellIndex(i);
            self.set_h(cell, self.h(cell) + dt * flux_h[i] * inv_area);
            self.set_hu(cell, self.hu(cell) + dt * flux_hu[i] * inv_area);
            self.set_hv(cell, self.hv(cell) + dt * flux_hv[i] * inv_area);
        }
    }

    /// 应用源项更新
    fn apply_source_update(
        &mut self,
        dt: f64,
        source_h: &[f64],
        source_hu: &[f64],
        source_hv: &[f64],
    ) {
        let n = self.n_cells();
        for i in 0..n {
            let cell = CellIndex(i);
            self.set_h(cell, self.h(cell) + dt * source_h[i]);
            self.set_hu(cell, self.hu(cell) + dt * source_hu[i]);
            self.set_hv(cell, self.hv(cell) + dt * source_hv[i]);
        }
    }

    /// 强制非负水深
    fn enforce_non_negative_depth(&mut self, h_min: f64) {
        let h = self.h_slice_mut();
        for value in h.iter_mut() {
            if *value < h_min {
                *value = 0.0;
            }
        }
    }
}

/// 状态视图（借用分离，用于同时读写不同字段）
pub struct StateView<'a> {
    pub h: &'a [f64],
    pub hu: &'a [f64],
    pub hv: &'a [f64],
    pub z: &'a [f64],
}

impl<'a> StateView<'a> {
    pub fn get(&self, i: usize) -> ConservedState {
        ConservedState {
            h: self.h[i],
            hu: self.hu[i],
            hv: self.hv[i],
        }
    }

    pub fn n_cells(&self) -> usize {
        self.h.len()
    }
}

/// 可变状态视图
pub struct StateViewMut<'a> {
    pub h: &'a mut [f64],
    pub hu: &'a mut [f64],
    pub hv: &'a mut [f64],
}

impl<'a> StateViewMut<'a> {
    pub fn set(&mut self, i: usize, state: ConservedState) {
        self.h[i] = state.h;
        self.hu[i] = state.hu;
        self.hv[i] = state.hv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conserved_state() {
        let state = ConservedState::new(1.0, 2.0, 3.0);
        assert_eq!(state.h, 1.0);
        assert_eq!(state.hu, 2.0);
        assert_eq!(state.hv, 3.0);
    }

    #[test]
    fn test_state_operations() {
        let s1 = ConservedState::new(1.0, 2.0, 3.0);
        let s2 = ConservedState::new(0.5, 1.0, 1.5);

        let sum = s1.add(&s2);
        assert!((sum.h - 1.5).abs() < 1e-10);
        assert!((sum.hu - 3.0).abs() < 1e-10);

        let scaled = s1.scale(2.0);
        assert!((scaled.h - 2.0).abs() < 1e-10);
        assert!((scaled.hu - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_velocity_calculation() {
        let params = NumericalParams::default();
        let state = ConservedState::new(2.0, 4.0, 6.0);
        let vel = state.velocity(&params);

        assert!((vel.u - 2.0).abs() < 1e-10);
        assert!((vel.v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dry_state() {
        let params = NumericalParams::default();
        let dry_state = ConservedState::new(1e-8, 0.0, 0.0);
        assert!(dry_state.is_dry(&params));

        let wet_state = ConservedState::new(1.0, 0.0, 0.0);
        assert!(!wet_state.is_dry(&params));
    }
}
