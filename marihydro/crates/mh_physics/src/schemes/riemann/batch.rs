// crates/mh_physics/src/schemes/riemann/batch.rs

//! 批量 Riemann 求解器接口
//!
//! 本模块提供批量求解多个 Riemann 问题的优化接口，支持：
//! - 向量化计算（SIMD）
//! - 缓存友好的内存访问
//! - 并行计算
//!
//! # 设计理念
//!
//! 逐面计算 Riemann 问题虽然简单，但存在以下问题：
//! 1. 函数调用开销（每个面一次调用）
//! 2. 内存访问不连续（随机访问状态数组）
//! 3. 难以向量化（单次计算无法利用 SIMD）
//!
//! 批量接口将多个 Riemann 问题打包处理，可以：
//! 1. 减少函数调用开销（一次调用处理多个问题）
//! 2. 预取数据（批量读取连续内存）
//! 3. 启用向量化（编译器可自动向量化循环）
//! 4. 并行计算（批次内可并行）
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::schemes::riemann::{BatchRiemannSolver, HllcSolverF64};
//!
//! let solver = HllcSolverF64::default();
//! let left_states = vec![...];
//! let right_states = vec![...];
//! let normals = vec![...];
//! let mut fluxes = vec![RiemannFluxF64::zero(); n_faces];
//!
//! solver.solve_batch(&left_states, &right_states, &normals, &mut fluxes);
//! ```

use super::{RiemannFlux, RiemannSolver, RiemannError};
use mh_runtime::RuntimeScalar;
use rayon::prelude::*;

// ============================================================
// 批量状态结构
// ============================================================

/// 批量单元状态
///
/// 使用 SoA（Structure of Arrays）布局以优化向量化和缓存性能。
#[derive(Debug, Clone)]
pub struct BatchCellStates<S> {
    /// 水深数组 [m]
    pub h: Vec<S>,
    /// x 方向速度数组 [m/s]
    pub u: Vec<S>,
    /// y 方向速度数组 [m/s]
    pub v: Vec<S>,
}

impl<S: RuntimeScalar> BatchCellStates<S> {
    /// 创建空批量状态
    pub fn new() -> Self {
        Self {
            h: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
        }
    }

    /// 创建指定容量的批量状态
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            h: Vec::with_capacity(capacity),
            u: Vec::with_capacity(capacity),
            v: Vec::with_capacity(capacity),
        }
    }

    /// 从 AoS 格式转换
    pub fn from_aos<T: Into<(S, S, S)>>(states: impl IntoIterator<Item = T>) -> Self {
        let mut result = Self::new();
        for state in states {
            let (h, u, v) = state.into();
            result.h.push(h);
            result.u.push(u);
            result.v.push(v);
        }
        result
    }

    /// 状态数量
    #[inline]
    pub fn len(&self) -> usize {
        self.h.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.h.is_empty()
    }

    /// 添加状态
    #[inline]
    pub fn push(&mut self, h: S, u: S, v: S) {
        self.h.push(h);
        self.u.push(u);
        self.v.push(v);
    }

    /// 清空状态
    pub fn clear(&mut self) {
        self.h.clear();
        self.u.clear();
        self.v.clear();
    }

    /// 获取单个状态
    #[inline]
    pub fn get(&self, idx: usize) -> Option<(S, S, S)> {
        if idx < self.len() {
            Some((self.h[idx], self.u[idx], self.v[idx]))
        } else {
            None
        }
    }

    /// 获取单个状态（不检查边界）
    ///
    /// # Safety
    /// 调用者必须确保 idx < len()
    #[inline]
    pub unsafe fn get_unchecked(&self, idx: usize) -> (S, S, S) {
        (
            *self.h.get_unchecked(idx),
            *self.u.get_unchecked(idx),
            *self.v.get_unchecked(idx),
        )
    }
}

impl<S: RuntimeScalar> Default for BatchCellStates<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// 批量法向量
#[derive(Debug, Clone)]
pub struct BatchNormals<S> {
    /// x 分量
    pub nx: Vec<S>,
    /// y 分量
    pub ny: Vec<S>,
}

impl<S: RuntimeScalar> BatchNormals<S> {
    /// 创建空批量
    pub fn new() -> Self {
        Self {
            nx: Vec::new(),
            ny: Vec::new(),
        }
    }

    /// 创建指定容量的批量
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nx: Vec::with_capacity(capacity),
            ny: Vec::with_capacity(capacity),
        }
    }

    /// 法向量数量
    #[inline]
    pub fn len(&self) -> usize {
        self.nx.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nx.is_empty()
    }

    /// 添加法向量
    #[inline]
    pub fn push(&mut self, nx: S, ny: S) {
        self.nx.push(nx);
        self.ny.push(ny);
    }

    /// 获取单个法向量
    #[inline]
    pub fn get(&self, idx: usize) -> Option<[S; 2]> {
        if idx < self.len() {
            Some([self.nx[idx], self.ny[idx]])
        } else {
            None
        }
    }

    /// 获取单个法向量（不检查边界）
    ///
    /// # Safety
    /// 调用者必须确保 idx < len()
    #[inline]
    pub unsafe fn get_unchecked(&self, idx: usize) -> [S; 2] {
        [*self.nx.get_unchecked(idx), *self.ny.get_unchecked(idx)]
    }
}

impl<S: RuntimeScalar> Default for BatchNormals<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// 批量通量结果
#[derive(Debug, Clone)]
pub struct BatchFluxes<S> {
    /// 质量通量 [m²/s]
    pub mass: Vec<S>,
    /// x 方向动量通量 [m³/s²]
    pub momentum_x: Vec<S>,
    /// y 方向动量通量 [m³/s²]
    pub momentum_y: Vec<S>,
    /// 最大波速 [m/s]
    pub max_wave_speed: Vec<S>,
}

impl<S: RuntimeScalar> BatchFluxes<S> {
    /// 创建指定大小的零初始化通量数组
    pub fn zeros(size: usize) -> Self {
        Self {
            mass: vec![S::zero(); size],
            momentum_x: vec![S::zero(); size],
            momentum_y: vec![S::zero(); size],
            max_wave_speed: vec![S::zero(); size],
        }
    }

    /// 通量数量
    #[inline]
    pub fn len(&self) -> usize {
        self.mass.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mass.is_empty()
    }

    /// 获取单个通量
    #[inline]
    pub fn get(&self, idx: usize) -> Option<RiemannFlux<S>> {
        if idx < self.len() {
            Some(RiemannFlux::new(
                self.mass[idx],
                self.momentum_x[idx],
                self.momentum_y[idx],
                self.max_wave_speed[idx],
            ))
        } else {
            None
        }
    }

    /// 设置单个通量
    #[inline]
    pub fn set(&mut self, idx: usize, flux: RiemannFlux<S>) {
        if idx < self.len() {
            self.mass[idx] = flux.mass;
            self.momentum_x[idx] = flux.momentum_x;
            self.momentum_y[idx] = flux.momentum_y;
            self.max_wave_speed[idx] = flux.max_wave_speed;
        }
    }

    /// 获取全局最大波速
    pub fn global_max_wave_speed(&self) -> S {
        self.max_wave_speed
            .iter()
            .copied()
            .fold(S::zero(), |acc, s| if s > acc { s } else { acc })
    }
}

// ============================================================
// 批量求解器 Trait
// ============================================================

/// 批量 Riemann 求解器 Trait
///
/// 扩展 `RiemannSolver`，提供批量求解接口。
/// 默认实现使用循环调用单次求解，但实现者可以覆盖以提供优化版本。
pub trait BatchRiemannSolver: RiemannSolver {
    /// 批量求解 Riemann 问题
    ///
    /// # 参数
    /// - `left_states`: 左侧状态（SoA 格式）
    /// - `right_states`: 右侧状态（SoA 格式）
    /// - `normals`: 法向量（SoA 格式）
    /// - `fluxes`: 输出通量（预分配）
    ///
    /// # 返回
    /// 成功返回 Ok(())，失败返回错误
    fn solve_batch(
        &self,
        left_states: &BatchCellStates<Self::Scalar>,
        right_states: &BatchCellStates<Self::Scalar>,
        normals: &BatchNormals<Self::Scalar>,
        fluxes: &mut BatchFluxes<Self::Scalar>,
    ) -> Result<(), RiemannError> {
        let n = left_states.len();
        debug_assert_eq!(right_states.len(), n);
        debug_assert_eq!(normals.len(), n);
        debug_assert_eq!(fluxes.len(), n);

        // 默认实现：循环调用单次求解
        for i in 0..n {
            let (h_l, u_l, v_l) = left_states.get(i).unwrap();
            let (h_r, u_r, v_r) = right_states.get(i).unwrap();
            let normal = normals.get(i).unwrap();

            let flux = self.solve(h_l, h_r, [u_l, v_l], [u_r, v_r], normal)?;
            fluxes.set(i, flux);
        }

        Ok(())
    }

    /// 并行批量求解 Riemann 问题
    ///
    /// 使用 rayon 并行化求解过程。适用于大规模问题。
    ///
    /// # 参数
    /// - `left_states`: 左侧状态（SoA 格式）
    /// - `right_states`: 右侧状态（SoA 格式）
    /// - `normals`: 法向量（SoA 格式）
    /// - `min_parallel_size`: 最小并行大小（低于此值使用串行）
    ///
    /// # 返回
    /// 计算结果通量数组，或错误
    fn solve_batch_parallel(
        &self,
        left_states: &BatchCellStates<Self::Scalar>,
        right_states: &BatchCellStates<Self::Scalar>,
        normals: &BatchNormals<Self::Scalar>,
        min_parallel_size: usize,
    ) -> Result<BatchFluxes<Self::Scalar>, RiemannError>
    where
        Self::Scalar: Send + Sync,
    {
        let n = left_states.len();
        debug_assert_eq!(right_states.len(), n);
        debug_assert_eq!(normals.len(), n);

        if n < min_parallel_size {
            // 串行求解
            let mut fluxes = BatchFluxes::zeros(n);
            self.solve_batch(left_states, right_states, normals, &mut fluxes)?;
            return Ok(fluxes);
        }

        // 并行求解
        let results: Vec<Result<RiemannFlux<Self::Scalar>, RiemannError>> = (0..n)
            .into_par_iter()
            .map(|i| {
                // SAFETY: i 在有效范围内
                let (h_l, u_l, v_l) = unsafe { left_states.get_unchecked(i) };
                let (h_r, u_r, v_r) = unsafe { right_states.get_unchecked(i) };
                let normal = unsafe { normals.get_unchecked(i) };

                self.solve(h_l, h_r, [u_l, v_l], [u_r, v_r], normal)
            })
            .collect();

        // 收集结果
        let mut fluxes = BatchFluxes::zeros(n);
        for (i, result) in results.into_iter().enumerate() {
            fluxes.set(i, result?);
        }

        Ok(fluxes)
    }

    /// 求解并返回最大波速
    ///
    /// 便捷方法，同时返回通量和全局最大波速（用于 CFL 条件）
    fn solve_batch_with_max_speed(
        &self,
        left_states: &BatchCellStates<Self::Scalar>,
        right_states: &BatchCellStates<Self::Scalar>,
        normals: &BatchNormals<Self::Scalar>,
        fluxes: &mut BatchFluxes<Self::Scalar>,
    ) -> Result<Self::Scalar, RiemannError> {
        self.solve_batch(left_states, right_states, normals, fluxes)?;
        Ok(fluxes.global_max_wave_speed())
    }
}

// ============================================================
// 为现有求解器实现 BatchRiemannSolver
// ============================================================

// 所有实现 RiemannSolver 的类型自动获得 BatchRiemannSolver
impl<T: RiemannSolver> BatchRiemannSolver for T {}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_cell_states() {
        let mut states = BatchCellStates::<f64>::new();
        states.push(1.0, 2.0, 3.0);
        states.push(4.0, 5.0, 6.0);

        assert_eq!(states.len(), 2);
        assert_eq!(states.get(0), Some((1.0, 2.0, 3.0)));
        assert_eq!(states.get(1), Some((4.0, 5.0, 6.0)));
        assert_eq!(states.get(2), None);
    }

    #[test]
    fn test_batch_normals() {
        let mut normals = BatchNormals::<f64>::new();
        normals.push(1.0, 0.0);
        normals.push(0.0, 1.0);

        assert_eq!(normals.len(), 2);
        assert_eq!(normals.get(0), Some([1.0, 0.0]));
        assert_eq!(normals.get(1), Some([0.0, 1.0]));
    }

    #[test]
    fn test_batch_fluxes() {
        let mut fluxes = BatchFluxes::<f64>::zeros(3);
        assert_eq!(fluxes.len(), 3);

        fluxes.set(0, RiemannFlux::new(1.0, 2.0, 3.0, 4.0));
        fluxes.set(2, RiemannFlux::new(10.0, 20.0, 30.0, 40.0));

        assert_eq!(fluxes.global_max_wave_speed(), 40.0);
        assert_eq!(fluxes.get(0).unwrap().mass, 1.0);
        assert_eq!(fluxes.get(1).unwrap().mass, 0.0); // 未设置，保持零
    }

    #[test]
    fn test_batch_cell_states_from_aos() {
        let aos_data = vec![(1.0_f64, 2.0, 3.0), (4.0, 5.0, 6.0)];
        let states = BatchCellStates::from_aos(aos_data);

        assert_eq!(states.len(), 2);
        assert_eq!(states.h, vec![1.0, 4.0]);
        assert_eq!(states.u, vec![2.0, 5.0]);
        assert_eq!(states.v, vec![3.0, 6.0]);
    }
}
