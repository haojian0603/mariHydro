// marihydro/crates/mh_physics/src/traits.rs

//! 状态访问抽象接口
//!
//! 定义浅水方程守恒变量的访问抽象，支持不同状态存储实现的互换。
//!
//! 
//! **注意**: 本模块的 trait 现在已升级为泛型接口，支持任意标量类型。
//! 新代码应使用泛型化的 `ShallowWaterState<B>` 和 `ConservedState<S>`。
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use mh_physics::traits::{StateAccess, StateAccessMut};
//! use mh_physics::state::ShallowWaterState;
//! use mh_runtime::CpuBackend;
//!
//! fn compute_total_volume<S: StateAccess>(state: &S, areas: &[S::Scalar]) -> S::Scalar {
//!     (0..state.n_cells())
//!         .map(|i| state.h(i) * areas[i])
//!         .sum()
//! }
//! ```

use crate::state::ConservedState;
use crate::types::{NumericalParams, SafeVelocity};
use mh_runtime::RuntimeScalar;
use num_traits::Float;

/// 泛型版本的 ConservedState
pub type ConservedStateGeneric<S> = ConservedState<S>;
/// 泛型版本的 NumericalParams
pub type NumericalParamsGeneric<S> = NumericalParams<S>;
/// 泛型版本的 SafeVelocity
pub type SafeVelocityGeneric<S> = SafeVelocity<S>;

// ============================================================
// 状态访问 Trait (泛型版本)
// ============================================================

/// 状态只读访问接口（泛型版本）
///
/// 提供对浅水方程守恒变量的统一只读访问。
/// 实现此 trait 的类型应保证线程安全（Send + Sync）。
pub trait StateAccess: Send + Sync {
    /// 标量类型（运行时决定，如 f32 或 f64）
    type Scalar: RuntimeScalar + Float + std::fmt::Debug;

    /// 单元数量
    fn n_cells(&self) -> usize;

    /// 获取单元的守恒状态
    fn get(&self, cell: usize) -> ConservedState<Self::Scalar>;

    /// 获取水深 [m]
    fn h(&self, cell: usize) -> Self::Scalar;

    /// 获取 x 方向动量 [m²/s]
    fn hu(&self, cell: usize) -> Self::Scalar;

    /// 获取 y 方向动量 [m²/s]
    fn hv(&self, cell: usize) -> Self::Scalar;

    /// 获取底床高程 [m]
    fn z(&self, cell: usize) -> Self::Scalar;

    /// 获取水位（水深 + 底床）[m]
    #[inline]
    fn eta(&self, cell: usize) -> Self::Scalar {
        self.h(cell) + self.z(cell)
    }

    /// 计算速度
    #[inline]
    fn velocity(&self, cell: usize, params: &NumericalParams<Self::Scalar>) -> SafeVelocity<Self::Scalar> {
        params.safe_velocity(self.hu(cell), self.hv(cell), self.h(cell))
    }

    /// 判断是否为干单元
    #[inline]
    fn is_dry(&self, cell: usize, params: &NumericalParams<Self::Scalar>) -> bool {
        params.is_dry(self.h(cell))
    }

    /// 判断是否为湿单元
    #[inline]
    fn is_wet(&self, cell: usize, params: &NumericalParams<Self::Scalar>) -> bool {
        !self.is_dry(cell, params)
    }

    // ===== 批量访问（切片引用）=====

    /// 水深数组引用
    fn h_slice(&self) -> &[Self::Scalar];

    /// x 动量数组引用
    fn hu_slice(&self) -> &[Self::Scalar];

    /// y 动量数组引用
    fn hv_slice(&self) -> &[Self::Scalar];

    /// 底床高程数组引用
    fn z_slice(&self) -> &[Self::Scalar];
}

/// 状态可变访问接口（泛型版本）
///
/// 提供对浅水方程守恒变量的统一可变访问。
pub trait StateAccessMut: StateAccess {
    /// 设置单元的守恒状态
    fn set(&mut self, cell: usize, state: ConservedState<Self::Scalar>);

    /// 设置水深 [m]
    fn set_h(&mut self, cell: usize, value: Self::Scalar);

    /// 设置 x 方向动量 [m²/s]
    fn set_hu(&mut self, cell: usize, value: Self::Scalar);

    /// 设置 y 方向动量 [m²/s]
    fn set_hv(&mut self, cell: usize, value: Self::Scalar);

    /// 设置底床高程 [m]
    fn set_z(&mut self, cell: usize, value: Self::Scalar);

    // ===== 批量可变访问 =====

    /// 水深数组可变引用
    fn h_slice_mut(&mut self) -> &mut [Self::Scalar];

    /// x 动量数组可变引用
    fn hu_slice_mut(&mut self) -> &mut [Self::Scalar];

    /// y 动量数组可变引用
    fn hv_slice_mut(&mut self) -> &mut [Self::Scalar];

    /// 底床高程数组可变引用
    fn z_slice_mut(&mut self) -> &mut [Self::Scalar];

    // ===== 批量操作 =====

    /// 应用通量更新
    ///
    /// # 参数
    /// - `dt`: 时间步长
    /// - `areas`: 单元面积
    /// - `flux_h`, `flux_hu`, `flux_hv`: 各单元累积通量
    fn apply_flux_update(
        &mut self,
        dt: Self::Scalar,
        areas: &[Self::Scalar],
        flux_h: &[Self::Scalar],
        flux_hu: &[Self::Scalar],
        flux_hv: &[Self::Scalar],
    );

    /// 应用源项更新
    fn apply_source_update(
        &mut self,
        dt: Self::Scalar,
        source_h: &[Self::Scalar],
        source_hu: &[Self::Scalar],
        source_hv: &[Self::Scalar],
    );

    /// 强制非负水深
    fn enforce_non_negative_depth(&mut self, h_min: Self::Scalar);

    /// 从另一个状态复制
    ///
    /// # 错误
    /// 如果单元数量不匹配则返回错误
    fn copy_from<S2: StateAccess<Scalar = Self::Scalar>>(&mut self, other: &S2) -> Result<(), &'static str>;
}

// ============================================================
// 辅助类型
// ============================================================

/// 状态视图（借用分离，用于同时读写不同字段）
pub struct StateView<'a, S> {
    /// 水深切片
    pub h: &'a [S],
    /// x 动量切片
    pub hu: &'a [S],
    /// y 动量切片
    pub hv: &'a [S],
    /// 底床高程切片
    pub z: &'a [S],
}

impl<'a, S> StateView<'a, S>
where
    S: RuntimeScalar + Float,
{
    /// 创建状态视图
    pub fn new(h: &'a [S], hu: &'a [S], hv: &'a [S], z: &'a [S]) -> Self {
        Self { h, hu, hv, z }
    }

    /// 单元数量
    pub fn n_cells(&self) -> usize {
        self.h.len()
    }

    /// 获取守恒状态
    pub fn get(&self, cell: usize) -> ConservedState<S> {
        ConservedState::new(self.h[cell], self.hu[cell], self.hv[cell])
    }

    /// 获取水位
    pub fn eta(&self, cell: usize) -> S {
        self.h[cell] + self.z[cell]
    }
}

/// 可变状态视图
pub struct StateViewMut<'a, S> {
    /// 水深切片
    pub h: &'a mut [S],
    /// x 动量切片
    pub hu: &'a mut [S],
    /// y 动量切片
    pub hv: &'a mut [S],
    /// 底床高程切片
    pub z: &'a mut [S],
}

impl<'a, S> StateViewMut<'a, S>
where
    S: RuntimeScalar + Float,
{
    /// 创建可变状态视图
    pub fn new(
        h: &'a mut [S],
        hu: &'a mut [S],
        hv: &'a mut [S],
        z: &'a mut [S],
    ) -> Self {
        Self { h, hu, hv, z }
    }

    /// 单元数量
    pub fn n_cells(&self) -> usize {
        self.h.len()
    }

    /// 设置守恒状态
    pub fn set(&mut self, cell: usize, state: ConservedState<S>) {
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
    fn total_volume(&self, areas: &[Self::Scalar]) -> Self::Scalar {
        self.h_slice()
            .iter()
            .zip(areas.iter())
            .map(|(h, a)| *h * *a)
            .sum()
    }

    /// 计算湿单元数量
    fn wet_cell_count(&self, h_threshold: Self::Scalar) -> usize {
        self.h_slice().iter().filter(|&&h| h > h_threshold).count()
    }

    /// 计算干单元数量
    fn dry_cell_count(&self, h_threshold: Self::Scalar) -> usize {
        self.n_cells() - self.wet_cell_count(h_threshold)
    }

    /// 获取最大水深
    fn max_depth(&self) -> Self::Scalar {
        self.h_slice()
            .iter()
            .cloned()
            .fold(Self::Scalar::neg_infinity(), Self::Scalar::max)
    }

    /// 获取最小水深
    fn min_depth(&self) -> Self::Scalar {
        self.h_slice()
            .iter()
            .cloned()
            .fold(Self::Scalar::infinity(), Self::Scalar::min)
    }

    /// 获取平均水深
    fn mean_depth(&self) -> Self::Scalar {
        if self.n_cells() == 0 {
            return Self::Scalar::ZERO;
        }
        let sum: Self::Scalar = self.h_slice().iter().sum();
        sum / Self::Scalar::from_usize(self.n_cells()).unwrap_or(Self::Scalar::ONE)
    }

    /// 检查是否包含 NaN 或 Inf
    fn has_invalid_values(&self) -> bool {
        use num_traits::Float;
        self.h_slice().iter().any(|&v| !v.is_finite())
            || self.hu_slice().iter().any(|&v| !v.is_finite())
            || self.hv_slice().iter().any(|&v| !v.is_finite())
    }

    /// 获取无效值的单元索引列表
    fn invalid_cell_indices(&self) -> Vec<usize> {
        use num_traits::Float;
        let mut indices = Vec::new();
        for i in 0..self.n_cells() {
            if !self.h(i).is_finite() || !self.hu(i).is_finite() || !self.hv(i).is_finite() {
                indices.push(i);
            }
        }
        indices
    }

    /// 计算最大速度（用于 CFL 约束）
    fn max_velocity_magnitude(&self, params: &NumericalParams<Self::Scalar>) -> Self::Scalar {
        use num_traits::Float;
        let mut max_v = Self::Scalar::ZERO;
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
    fn statistics(&self) -> StateStatistics<Self::Scalar> {
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
pub struct StateStatistics<S> {
    /// 单元数
    pub n_cells: usize,
    /// 最小水深
    pub h_min: S,
    /// 最大水深
    pub h_max: S,
    /// 平均水深
    pub h_mean: S,
}

// 为所有实现 StateAccess 的类型自动实现 StateAccessExt
impl<T: StateAccess + ?Sized> StateAccessExt for T {}

// ============================================================
// 向后兼容类型别名（Legacy f64 版本）
// ============================================================

/// f64 版本的 ConservedState（向后兼容）
pub type ConservedStateF64 = ConservedState<f64>;
/// f64 版本的 NumericalParams（向后兼容）
pub type NumericalParamsF64 = NumericalParams<f64>;
/// f64 版本的 SafeVelocity（向后兼容）
pub type SafeVelocityF64 = SafeVelocity<f64>;

// 旧版 StateAccess trait 别名（如果外部代码依赖）
pub use crate::StateAccess as StateAccessLegacy;
pub use crate::StateAccessMut as StateAccessMutLegacy;

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ShallowWaterState;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_state_access_basic() {
        let backend = CpuBackend::<f64>::new();
        let state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(backend, 10);
        
        // 通过 trait 访问
        fn check_state<S: StateAccess<Scalar = f64>>(s: &S) -> usize {
            s.n_cells()
        }
        
        assert_eq!(check_state(&state), 10);
    }

    #[test]
    fn test_state_access_mut() {
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(backend, 10);
        
        // 通过 trait 修改
        fn modify_state<S: StateAccessMut<Scalar = f64>>(s: &mut S) {
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
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(backend, 5);
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
