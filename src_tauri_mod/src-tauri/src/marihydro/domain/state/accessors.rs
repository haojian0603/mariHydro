// src-tauri/src/marihydro/domain/state/accessors.rs

//! 类型安全状态访问器

use super::shallow_water::ShallowWaterState;
use crate::marihydro::core::traits::state::ConservedState;
use crate::marihydro::core::types::{CellIndex, NumericalParams, SafeVelocity};

/// 状态访问器扩展
pub trait StateAccessors {
    /// 使用 CellIndex 获取守恒状态
    fn get_state(&self, cell: CellIndex) -> ConservedState;

    /// 使用 CellIndex 获取速度
    fn get_velocity(&self, cell: CellIndex, params: &NumericalParams) -> SafeVelocity;

    /// 使用 CellIndex 获取水位
    fn get_water_level(&self, cell: CellIndex) -> f64;

    /// 使用 CellIndex 判断是否为干
    fn is_cell_dry(&self, cell: CellIndex, params: &NumericalParams) -> bool;

    /// 获取最大水深
    fn max_depth(&self) -> f64;

    /// 获取最大速度
    fn max_speed(&self, params: &NumericalParams) -> f64;

    /// 获取湿单元数量
    fn wet_cell_count(&self, params: &NumericalParams) -> usize;
}

impl StateAccessors for ShallowWaterState {
    fn get_state(&self, cell: CellIndex) -> ConservedState {
        ConservedState::new(self.h[cell.0], self.hu[cell.0], self.hv[cell.0])
    }

    fn get_velocity(&self, cell: CellIndex, params: &NumericalParams) -> SafeVelocity {
        self.velocity(cell.0, params)
    }

    fn get_water_level(&self, cell: CellIndex) -> f64 {
        self.water_level(cell.0)
    }

    fn is_cell_dry(&self, cell: CellIndex, params: &NumericalParams) -> bool {
        params.is_dry(self.h[cell.0])
    }

    fn max_depth(&self) -> f64 {
        self.h.iter().cloned().fold(0.0_f64, f64::max)
    }

    fn max_speed(&self, params: &NumericalParams) -> f64 {
        let mut max_speed = 0.0_f64;

        for i in 0..self.n_cells() {
            if !params.is_dry(self.h[i]) {
                let vel = self.velocity(i, params);
                max_speed = max_speed.max(vel.speed());
            }
        }

        max_speed
    }

    fn wet_cell_count(&self, params: &NumericalParams) -> usize {
        self.h.iter().filter(|&&h| !params.is_dry(h)).count()
    }
}

/// 批量状态操作
pub trait StateBatchOps {
    /// 复制状态到目标
    fn copy_to(&self, target: &mut Self);

    /// 线性组合: self = a * self + b * other
    fn linear_combine(&mut self, a: f64, other: &Self, b: f64);

    /// 计算与另一状态的最大差异
    fn max_difference(&self, other: &Self) -> f64;
}

impl StateBatchOps for ShallowWaterState {
    fn copy_to(&self, target: &mut Self) {
        debug_assert_eq!(self.n_cells(), target.n_cells());
        target.h.copy_from_slice(&self.h);
        target.hu.copy_from_slice(&self.hu);
        target.hv.copy_from_slice(&self.hv);
    }

    fn linear_combine(&mut self, a: f64, other: &Self, b: f64) {
        debug_assert_eq!(self.n_cells(), other.n_cells());

        for i in 0..self.n_cells() {
            self.h[i] = a * self.h[i] + b * other.h[i];
            self.hu[i] = a * self.hu[i] + b * other.hu[i];
            self.hv[i] = a * self.hv[i] + b * other.hv[i];
        }
    }

    fn max_difference(&self, other: &Self) -> f64 {
        debug_assert_eq!(self.n_cells(), other.n_cells());

        let mut max_diff = 0.0_f64;

        for i in 0..self.n_cells() {
            max_diff = max_diff.max((self.h[i] - other.h[i]).abs());
            max_diff = max_diff.max((self.hu[i] - other.hu[i]).abs());
            max_diff = max_diff.max((self.hv[i] - other.hv[i]).abs());
        }

        max_diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accessors() {
        let mut state = ShallowWaterState::new(10);
        state.h[0] = 5.0;
        state.hu[0] = 10.0;
        state.hv[0] = 15.0;
        state.z[0] = -2.0;

        let params = NumericalParams::default();

        let s = state.get_state(CellIndex(0));
        assert!((s.h - 5.0).abs() < 1e-10);

        let vel = state.get_velocity(CellIndex(0), &params);
        assert!((vel.u - 2.0).abs() < 1e-10);

        assert!((state.get_water_level(CellIndex(0)) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_ops() {
        let mut state1 = ShallowWaterState::new(3);
        state1.h = vec![1.0, 2.0, 3.0];

        let mut state2 = ShallowWaterState::new(3);
        state2.h = vec![2.0, 4.0, 6.0];

        state1.linear_combine(0.5, &state2, 0.5);

        assert!((state1.h[0] - 1.5).abs() < 1e-10);
        assert!((state1.h[1] - 3.0).abs() < 1e-10);
    }
}
