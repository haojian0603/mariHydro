// marihydro/crates/mh_physics/src/traits.rs

//! 状态访问抽象接口
//!
//! 定义浅水方程守恒变量的访问抽象，支持不同状态存储实现的互换。
//!
//! # 设计说明
//!
//! 本模块从 `legacy_src/core/traits/state.rs` 迁移而来，提供统一的状态访问抽象。
//! 使用 `usize` 作为索引类型，与其他模块保持一致。
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use mh_physics::traits::{StateAccess, StateAccessMut};
//! use mh_physics::state::ShallowWaterState;
//!
//! fn compute_total_volume<S: StateAccess>(state: &S, areas: &[f64]) -> f64 {
//!     (0..state.n_cells())
//!         .map(|i| state.h(i) * areas[i])
//!         .sum()
//! }
//! ```

use crate::state::ConservedState;
use crate::types::{NumericalParams, SafeVelocity};

// ============================================================
// 状态访问 Trait
// ============================================================

/// 状态只读访问接口
///
/// 提供对浅水方程守恒变量的统一只读访问。
/// 实现此 trait 的类型应保证线程安全（Send + Sync）。
pub trait StateAccess: Send + Sync {
    /// 单元数量
    fn n_cells(&self) -> usize;

    /// 获取单元的守恒状态
    fn get(&self, cell: usize) -> ConservedState;

    /// 获取水深 [m]
    fn h(&self, cell: usize) -> f64;

    /// 获取 x 方向动量 [m²/s]
    fn hu(&self, cell: usize) -> f64;

    /// 获取 y 方向动量 [m²/s]
    fn hv(&self, cell: usize) -> f64;

    /// 获取底床高程 [m]
    fn z(&self, cell: usize) -> f64;

    /// 获取水位（水深 + 底床）[m]
    #[inline]
    fn eta(&self, cell: usize) -> f64 {
        self.h(cell) + self.z(cell)
    }

    /// 计算速度
    fn velocity(&self, cell: usize, params: &NumericalParams) -> SafeVelocity {
        params.safe_velocity(self.hu(cell), self.hv(cell), self.h(cell))
    }

    /// 判断是否为干单元
    fn is_dry(&self, cell: usize, params: &NumericalParams) -> bool {
        params.is_dry(self.h(cell))
    }

    /// 判断是否为湿单元
    #[inline]
    fn is_wet(&self, cell: usize, params: &NumericalParams) -> bool {
        !self.is_dry(cell, params)
    }

    // ===== 批量访问（切片引用）=====

    /// 水深数组引用
    fn h_slice(&self) -> &[f64];

    /// x 动量数组引用
    fn hu_slice(&self) -> &[f64];

    /// y 动量数组引用
    fn hv_slice(&self) -> &[f64];

    /// 底床高程数组引用
    fn z_slice(&self) -> &[f64];
}

/// 状态可变访问接口
///
/// 提供对浅水方程守恒变量的统一可变访问。
pub trait StateAccessMut: StateAccess {
    /// 设置单元的守恒状态
    fn set(&mut self, cell: usize, state: ConservedState);

    /// 设置水深 [m]
    fn set_h(&mut self, cell: usize, value: f64);

    /// 设置 x 方向动量 [m²/s]
    fn set_hu(&mut self, cell: usize, value: f64);

    /// 设置 y 方向动量 [m²/s]
    fn set_hv(&mut self, cell: usize, value: f64);

    /// 设置底床高程 [m]
    fn set_z(&mut self, cell: usize, value: f64);

    // ===== 批量可变访问 =====

    /// 水深数组可变引用
    fn h_slice_mut(&mut self) -> &mut [f64];

    /// x 动量数组可变引用
    fn hu_slice_mut(&mut self) -> &mut [f64];

    /// y 动量数组可变引用
    fn hv_slice_mut(&mut self) -> &mut [f64];

    /// 底床高程数组可变引用
    fn z_slice_mut(&mut self) -> &mut [f64];

    // ===== 批量操作 =====

    /// 应用通量更新
    ///
    /// # 参数
    /// - `dt`: 时间步长
    /// - `areas`: 单元面积
    /// - `flux_h`, `flux_hu`, `flux_hv`: 各单元累积通量
    fn apply_flux_update(
        &mut self,
        dt: f64,
        areas: &[f64],
        flux_h: &[f64],
        flux_hu: &[f64],
        flux_hv: &[f64],
    ) {
        let n = self.n_cells();
        debug_assert_eq!(areas.len(), n);
        debug_assert_eq!(flux_h.len(), n);
        debug_assert_eq!(flux_hu.len(), n);
        debug_assert_eq!(flux_hv.len(), n);

        for i in 0..n {
            let inv_area = 1.0 / areas[i];
            let h_new = self.h(i) + dt * flux_h[i] * inv_area;
            let hu_new = self.hu(i) + dt * flux_hu[i] * inv_area;
            let hv_new = self.hv(i) + dt * flux_hv[i] * inv_area;
            self.set_h(i, h_new);
            self.set_hu(i, hu_new);
            self.set_hv(i, hv_new);
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
            self.set_h(i, self.h(i) + dt * source_h[i]);
            self.set_hu(i, self.hu(i) + dt * source_hu[i]);
            self.set_hv(i, self.hv(i) + dt * source_hv[i]);
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

    /// 从另一个状态复制
    ///
    /// # 错误
    /// 如果单元数量不匹配则返回错误
    fn copy_from<S: StateAccess>(&mut self, other: &S) -> Result<(), &'static str> {
        if self.n_cells() != other.n_cells() {
            return Err("单元数量不匹配");
        }
        for i in 0..self.n_cells() {
            self.set(i, other.get(i));
            self.set_z(i, other.z(i));
        }
        Ok(())
    }
}

// ============================================================
// 辅助类型
// ============================================================

/// 状态视图（借用分离，用于同时读写不同字段）
pub struct StateView<'a> {
    /// 水深切片
    pub h: &'a [f64],
    /// x 动量切片
    pub hu: &'a [f64],
    /// y 动量切片
    pub hv: &'a [f64],
    /// 底床高程切片
    pub z: &'a [f64],
}

impl<'a> StateView<'a> {
    /// 创建状态视图
    pub fn new(h: &'a [f64], hu: &'a [f64], hv: &'a [f64], z: &'a [f64]) -> Self {
        Self { h, hu, hv, z }
    }

    /// 单元数量
    pub fn n_cells(&self) -> usize {
        self.h.len()
    }

    /// 获取守恒状态
    pub fn get(&self, cell: usize) -> ConservedState {
        ConservedState::new(self.h[cell], self.hu[cell], self.hv[cell])
    }

    /// 获取水位
    pub fn eta(&self, cell: usize) -> f64 {
        self.h[cell] + self.z[cell]
    }
}

/// 可变状态视图
pub struct StateViewMut<'a> {
    /// 水深切片
    pub h: &'a mut [f64],
    /// x 动量切片
    pub hu: &'a mut [f64],
    /// y 动量切片
    pub hv: &'a mut [f64],
    /// 底床高程切片
    pub z: &'a mut [f64],
}

impl<'a> StateViewMut<'a> {
    /// 创建可变状态视图
    pub fn new(
        h: &'a mut [f64],
        hu: &'a mut [f64],
        hv: &'a mut [f64],
        z: &'a mut [f64],
    ) -> Self {
        Self { h, hu, hv, z }
    }

    /// 单元数量
    pub fn n_cells(&self) -> usize {
        self.h.len()
    }

    /// 设置守恒状态
    pub fn set(&mut self, cell: usize, state: ConservedState) {
        self.h[cell] = state.h;
        self.hu[cell] = state.hu;
        self.hv[cell] = state.hv;
    }
}

// ============================================================
// StateAccess 扩展方法
// ============================================================

/// 状态访问扩展方法
///
/// 提供基于 StateAccess 的便捷方法，无需单独实现。
pub trait StateAccessExt: StateAccess {
    /// 计算总水量（体积）
    ///
    /// # 参数
    ///
    /// - `areas`: 单元面积数组
    fn total_volume(&self, areas: &[f64]) -> f64 {
        self.h_slice()
            .iter()
            .zip(areas.iter())
            .map(|(h, a)| h * a)
            .sum()
    }

    /// 计算湿单元数量
    fn wet_cell_count(&self, h_threshold: f64) -> usize {
        self.h_slice().iter().filter(|&&h| h > h_threshold).count()
    }

    /// 计算干单元数量
    fn dry_cell_count(&self, h_threshold: f64) -> usize {
        self.n_cells() - self.wet_cell_count(h_threshold)
    }

    /// 获取最大水深
    fn max_depth(&self) -> f64 {
        self.h_slice()
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// 获取最小水深
    fn min_depth(&self) -> f64 {
        self.h_slice()
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    }

    /// 获取平均水深
    fn mean_depth(&self) -> f64 {
        if self.n_cells() == 0 {
            return 0.0;
        }
        self.h_slice().iter().sum::<f64>() / self.n_cells() as f64
    }

    /// 检查是否包含 NaN 或 Inf
    fn has_invalid_values(&self) -> bool {
        self.h_slice().iter().any(|&v| !v.is_finite())
            || self.hu_slice().iter().any(|&v| !v.is_finite())
            || self.hv_slice().iter().any(|&v| !v.is_finite())
    }

    /// 获取无效值的单元索引列表
    fn invalid_cell_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        for i in 0..self.n_cells() {
            if !self.h(i).is_finite() || !self.hu(i).is_finite() || !self.hv(i).is_finite() {
                indices.push(i);
            }
        }
        indices
    }

    /// 计算最大速度（用于 CFL 约束）
    fn max_velocity_magnitude(&self, params: &NumericalParams) -> f64 {
        let mut max_v = 0.0;
        for i in 0..self.n_cells() {
            if self.h(i) > params.h_dry {
                let vel = self.velocity(i, params);
                let mag = (vel.u * vel.u + vel.v * vel.v).sqrt();
                if mag > max_v {
                    max_v = mag;
                }
            }
        }
        max_v
    }

    /// 获取状态统计信息
    fn statistics(&self) -> StateStatistics {
        StateStatistics {
            n_cells: self.n_cells(),
            h_min: self.min_depth(),
            h_max: self.max_depth(),
            h_mean: self.mean_depth(),
        }
    }
}

/// 状态统计信息
#[derive(Debug, Clone, Default)]
pub struct StateStatistics {
    /// 单元数
    pub n_cells: usize,
    /// 最小水深
    pub h_min: f64,
    /// 最大水深
    pub h_max: f64,
    /// 平均水深
    pub h_mean: f64,
}

// 为所有实现 StateAccess 的类型自动实现 StateAccessExt
impl<T: StateAccess + ?Sized> StateAccessExt for T {}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ShallowWaterState;

    #[test]
    fn test_state_access_basic() {
        let state = ShallowWaterState::new(10);
        
        // 通过 trait 访问
        fn check_state<S: StateAccess>(s: &S) -> usize {
            s.n_cells()
        }
        
        assert_eq!(check_state(&state), 10);
    }

    #[test]
    fn test_state_access_mut() {
        let mut state = ShallowWaterState::new(10);
        
        // 通过 trait 修改
        fn modify_state<S: StateAccessMut>(s: &mut S) {
            s.set_h(0, 1.5);
        }
        
        modify_state(&mut state);
        assert!((state.h(0) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_state_view() {
        let h = vec![1.0, 2.0, 3.0];
        let hu = vec![0.1, 0.2, 0.3];
        let hv = vec![0.0, 0.0, 0.0];
        let z = vec![0.0, 1.0, 2.0];
        
        let view = StateView::new(&h, &hu, &hv, &z);
        
        assert_eq!(view.n_cells(), 3);
        assert!((view.eta(1) - 3.0).abs() < 1e-10); // h=2.0 + z=1.0
    }

    #[test]
    fn test_state_access_ext() {
        let mut state = ShallowWaterState::new(5);
        state.set_h(0, 1.0);
        state.set_h(1, 2.0);
        state.set_h(2, 3.0);
        state.set_h(3, 0.001);
        state.set_h(4, 0.0);

        // 测试扩展方法
        assert_eq!(state.wet_cell_count(0.01), 3);
        assert_eq!(state.dry_cell_count(0.01), 2);
        assert!((state.max_depth() - 3.0).abs() < 1e-10);
        assert!((state.min_depth() - 0.0).abs() < 1e-10);
        assert!(!state.has_invalid_values());
    }
}
