// src-tauri/src/marihydro/domain/state/shallow_water.rs

//! 浅水方程状态（SoA 布局）

use glam::DVec2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::traits::state::{ConservedState, StateAccess, StateAccessMut};
use crate::marihydro::core::types::{CellIndex, NumericalParams, SafeVelocity};

/// 浅水方程守恒状态（SoA 布局）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShallowWaterState {
    n_cells: usize,

    /// 水深 [m]
    pub h: Vec<f64>,
    /// x 方向动量 [m²/s]
    pub hu: Vec<f64>,
    /// y 方向动量 [m²/s]
    pub hv: Vec<f64>,
    /// 底床高程 [m]
    pub z: Vec<f64>,

    /// 可选的标量浓度场
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hc: Option<Vec<f64>>,
}

impl ShallowWaterState {
    /// 创建新状态
    pub fn new(n_cells: usize) -> Self {
        Self {
            n_cells,
            h: vec![0.0; n_cells],
            hu: vec![0.0; n_cells],
            hv: vec![0.0; n_cells],
            z: vec![0.0; n_cells],
            hc: None,
        }
    }

    /// 创建带标量的状态
    pub fn with_scalar(n_cells: usize) -> Self {
        Self {
            n_cells,
            h: vec![0.0; n_cells],
            hu: vec![0.0; n_cells],
            hv: vec![0.0; n_cells],
            z: vec![0.0; n_cells],
            hc: Some(vec![0.0; n_cells]),
        }
    }

    /// 从初始水位和底床创建（冷启动）
    pub fn cold_start(initial_eta: f64, z_bed: &[f64]) -> MhResult<Self> {
        let n_cells = z_bed.len();

        let h: Vec<f64> = z_bed.iter().map(|&z| (initial_eta - z).max(0.0)).collect();

        Ok(Self {
            n_cells,
            h,
            hu: vec![0.0; n_cells],
            hv: vec![0.0; n_cells],
            z: z_bed.to_vec(),
            hc: None,
        })
    }

    /// 克隆结构（不复制数据）
    pub fn clone_structure(&self) -> Self {
        Self {
            n_cells: self.n_cells,
            h: vec![0.0; self.n_cells],
            hu: vec![0.0; self.n_cells],
            hv: vec![0.0; self.n_cells],
            z: self.z.clone(),
            hc: self.hc.as_ref().map(|_| vec![0.0; self.n_cells]),
        }
    }

    /// 单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 获取原始变量 (h, u, v)
    #[inline]
    pub fn primitive(&self, idx: usize, params: &NumericalParams) -> (f64, f64, f64) {
        let h = self.h[idx];
        let vel = params.safe_velocity(self.hu[idx], self.hv[idx], h);
        (h, vel.u, vel.v)
    }

    /// 获取速度
    #[inline]
    pub fn velocity(&self, idx: usize, params: &NumericalParams) -> SafeVelocity {
        params.safe_velocity(self.hu[idx], self.hv[idx], self.h[idx])
    }

    /// 获取速度（使用阈值）
    #[inline]
    pub fn velocity_with_eps(&self, idx: usize, eps: f64) -> DVec2 {
        let h = self.h[idx];
        if h > eps {
            DVec2::new(self.hu[idx] / h, self.hv[idx] / h)
        } else {
            DVec2::ZERO
        }
    }

    /// 获取水位
    #[inline]
    pub fn water_level(&self, idx: usize) -> f64 {
        self.h[idx] + self.z[idx]
    }

    /// 设置守恒变量
    #[inline]
    pub fn set(&mut self, idx: usize, h: f64, hu: f64, hv: f64) {
        self.h[idx] = h;
        self.hu[idx] = hu;
        self.hv[idx] = hv;
    }

    /// 从原始变量设置
    #[inline]
    pub fn set_from_primitive(&mut self, idx: usize, h: f64, u: f64, v: f64) {
        self.h[idx] = h;
        self.hu[idx] = h * u;
        self.hv[idx] = h * v;
    }

    /// 重置为零
    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.hu.fill(0.0);
        self.hv.fill(0.0);
        if let Some(ref mut hc) = self.hc {
            hc.fill(0.0);
        }
    }

    /// 计算总质量
    pub fn total_mass(&self, cell_areas: &[f64]) -> f64 {
        self.h.iter().zip(cell_areas).map(|(h, a)| h * a).sum()
    }

    /// 计算总动量
    pub fn total_momentum(&self, cell_areas: &[f64]) -> DVec2 {
        let hux: f64 = self.hu.iter().zip(cell_areas).map(|(hu, a)| hu * a).sum();
        let hvx: f64 = self.hv.iter().zip(cell_areas).map(|(hv, a)| hv * a).sum();
        DVec2::new(hux, hvx)
    }

    /// 验证状态有效性
    pub fn validate(&self, time: f64, params: &NumericalParams) -> MhResult<()> {
        for idx in 0..self.n_cells {
            // 检查 NaN/Inf
            if !self.h[idx].is_finite() {
                return Err(MhError::numerical_instability(
                    format!("水深异常 (NaN/Inf) 在单元 {}", idx),
                    time,
                ));
            }

            if !self.hu[idx].is_finite() || !self.hv[idx].is_finite() {
                return Err(MhError::numerical_instability(
                    format!("动量异常 (NaN/Inf) 在单元 {}", idx),
                    time,
                ));
            }

            // 检查负水深
            if self.h[idx] < 0.0 {
                return Err(MhError::numerical_instability(
                    format!("水深为负 {:.6} m 在单元 {}", self.h[idx], idx),
                    time,
                ));
            }

            // 检查速度
            if !params.is_dry(self.h[idx]) {
                let vel = self.velocity(idx, params);
                if params.is_velocity_excessive(vel.speed()) {
                    return Err(MhError::numerical_instability(
                        format!("流速过大 {:.2} m/s 在单元 {}", vel.speed(), idx),
                        time,
                    ));
                }
            }
        }

        Ok(())
    }

    // ===== 时间积分器所需方法 =====

    /// 从另一个状态复制数据
    pub fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.n_cells(), other.n_cells());
        self.h.copy_from_slice(&other.h);
        self.hu.copy_from_slice(&other.hu);
        self.hv.copy_from_slice(&other.hv);
        if let (Some(ref mut dst), Some(ref src)) = (&mut self.hc, &other.hc) {
            dst.copy_from_slice(src);
        }
    }

    /// 添加缩放的 RHS: self += scale * rhs
    pub fn add_scaled_rhs(&mut self, rhs: &crate::marihydro::physics::engine::time_integrator::RhsBuffers, scale: f64) {
        self.h.par_iter_mut()
            .zip(rhs.dh_dt.par_iter())
            .for_each(|(h, &dh)| *h += scale * dh);
        
        self.hu.par_iter_mut()
            .zip(rhs.dhu_dt.par_iter())
            .for_each(|(hu, &dhu)| *hu += scale * dhu);
        
        self.hv.par_iter_mut()
            .zip(rhs.dhv_dt.par_iter())
            .for_each(|(hv, &dhv)| *hv += scale * dhv);
        
        // 注：tracer_rhs 暂不处理，需要时再扩展
    }

    /// 二元线性组合: self = a*A + b*B
    pub fn linear_combine(&mut self, a: f64, state_a: &Self, b: f64, state_b: &Self) {
        debug_assert_eq!(self.n_cells(), state_a.n_cells());
        debug_assert_eq!(self.n_cells(), state_b.n_cells());
        
        self.h.par_iter_mut()
            .zip(state_a.h.par_iter())
            .zip(state_b.h.par_iter())
            .for_each(|((dst, &va), &vb)| *dst = a * va + b * vb);
        
        self.hu.par_iter_mut()
            .zip(state_a.hu.par_iter())
            .zip(state_b.hu.par_iter())
            .for_each(|((dst, &va), &vb)| *dst = a * va + b * vb);
        
        self.hv.par_iter_mut()
            .zip(state_a.hv.par_iter())
            .zip(state_b.hv.par_iter())
            .for_each(|((dst, &va), &vb)| *dst = a * va + b * vb);
        
        if let (Some(ref mut dst), Some(ref src_a), Some(ref src_b)) = 
            (&mut self.hc, &state_a.hc, &state_b.hc) 
        {
            dst.par_iter_mut()
                .zip(src_a.par_iter())
                .zip(src_b.par_iter())
                .for_each(|((d, &va), &vb)| *d = a * va + b * vb);
        }
    }

    /// 自线性组合: self = a * self + b * other
    /// 避免同时借用 self 作为可变和不可变引用
    pub fn axpy(&mut self, a: f64, b: f64, other: &Self) {
        debug_assert_eq!(self.n_cells(), other.n_cells());
        
        self.h.par_iter_mut()
            .zip(other.h.par_iter())
            .for_each(|(dst, &vb)| *dst = a * *dst + b * vb);
        
        self.hu.par_iter_mut()
            .zip(other.hu.par_iter())
            .for_each(|(dst, &vb)| *dst = a * *dst + b * vb);
        
        self.hv.par_iter_mut()
            .zip(other.hv.par_iter())
            .for_each(|(dst, &vb)| *dst = a * *dst + b * vb);
        
        if let (Some(ref mut dst), Some(ref src_b)) = (&mut self.hc, &other.hc) {
            dst.par_iter_mut()
                .zip(src_b.par_iter())
                .for_each(|(d, &vb)| *d = a * *d + b * vb);
        }
    }

    /// 强制正性约束
    pub fn enforce_positivity(&mut self) {
        self.h.par_iter_mut().for_each(|h| {
            if *h < 0.0 { *h = 0.0; }
        });
        
        if let Some(ref mut hc) = self.hc {
            hc.par_iter_mut().for_each(|c| {
                if *c < 0.0 { *c = 0.0; }
            });
        }
    }
}

// ===== 实现 StateAccess trait =====

impl StateAccess for ShallowWaterState {
    fn n_cells(&self) -> usize {
        self.n_cells
    }

    fn get(&self, cell: CellIndex) -> ConservedState {
        ConservedState::new(self.h[cell.0], self.hu[cell.0], self.hv[cell.0])
    }

    fn h(&self, cell: CellIndex) -> f64 {
        self.h[cell.0]
    }

    fn hu(&self, cell: CellIndex) -> f64 {
        self.hu[cell.0]
    }

    fn hv(&self, cell: CellIndex) -> f64 {
        self.hv[cell.0]
    }

    fn z(&self, cell: CellIndex) -> f64 {
        self.z[cell.0]
    }

    fn h_slice(&self) -> &[f64] {
        &self.h
    }

    fn hu_slice(&self) -> &[f64] {
        &self.hu
    }

    fn hv_slice(&self) -> &[f64] {
        &self.hv
    }

    fn z_slice(&self) -> &[f64] {
        &self.z
    }
}

impl StateAccessMut for ShallowWaterState {
    fn set(&mut self, cell: CellIndex, state: ConservedState) {
        self.h[cell.0] = state.h;
        self.hu[cell.0] = state.hu;
        self.hv[cell.0] = state.hv;
    }

    fn set_h(&mut self, cell: CellIndex, value: f64) {
        self.h[cell.0] = value;
    }

    fn set_hu(&mut self, cell: CellIndex, value: f64) {
        self.hu[cell.0] = value;
    }

    fn set_hv(&mut self, cell: CellIndex, value: f64) {
        self.hv[cell.0] = value;
    }

    fn set_z(&mut self, cell: CellIndex, value: f64) {
        self.z[cell.0] = value;
    }

    fn h_slice_mut(&mut self) -> &mut [f64] {
        &mut self.h
    }

    fn hu_slice_mut(&mut self) -> &mut [f64] {
        &mut self.hu
    }

    fn hv_slice_mut(&mut self) -> &mut [f64] {
        &mut self.hv
    }

    fn z_slice_mut(&mut self) -> &mut [f64] {
        &mut self.z
    }
}

/// 梯度状态
#[derive(Debug, Clone)]
pub struct GradientState {
    pub grad_h: Vec<DVec2>,
    pub grad_hu: Vec<DVec2>,
    pub grad_hv: Vec<DVec2>,
}

impl GradientState {
    pub fn new(n_cells: usize) -> Self {
        Self {
            grad_h: vec![DVec2::ZERO; n_cells],
            grad_hu: vec![DVec2::ZERO; n_cells],
            grad_hv: vec![DVec2::ZERO; n_cells],
        }
    }

    pub fn reset(&mut self) {
        self.grad_h.fill(DVec2::ZERO);
        self.grad_hu.fill(DVec2::ZERO);
        self.grad_hv.fill(DVec2::ZERO);
    }
}

/// 通量
#[derive(Debug, Clone, Copy, Default)]
pub struct Flux {
    pub mass: f64,
    pub mom_x: f64,
    pub mom_y: f64,
}

impl Flux {
    #[inline]
    pub const fn new(mass: f64, mom_x: f64, mom_y: f64) -> Self {
        Self { mass, mom_x, mom_y }
    }

    #[inline]
    pub fn scale(self, factor: f64) -> Self {
        Self {
            mass: self.mass * factor,
            mom_x: self.mom_x * factor,
            mom_y: self.mom_y * factor,
        }
    }

    #[inline]
    pub fn magnitude(&self) -> f64 {
        (self.mass * self.mass + self.mom_x * self.mom_x + self.mom_y * self.mom_y).sqrt()
    }
}

impl std::ops::Add for Flux {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            mass: self.mass + rhs.mass,
            mom_x: self.mom_x + rhs.mom_x,
            mom_y: self.mom_y + rhs.mom_y,
        }
    }
}

impl std::ops::AddAssign for Flux {
    fn add_assign(&mut self, rhs: Self) {
        self.mass += rhs.mass;
        self.mom_x += rhs.mom_x;
        self.mom_y += rhs.mom_y;
    }
}

impl std::ops::Sub for Flux {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            mass: self.mass - rhs.mass,
            mom_x: self.mom_x - rhs.mom_x,
            mom_y: self.mom_y - rhs.mom_y,
        }
    }
}

impl std::ops::SubAssign for Flux {
    fn sub_assign(&mut self, rhs: Self) {
        self.mass -= rhs.mass;
        self.mom_x -= rhs.mom_x;
        self.mom_y -= rhs.mom_y;
    }
}

impl std::ops::Neg for Flux {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            mass: -self.mass,
            mom_x: -self.mom_x,
            mom_y: -self.mom_y,
        }
    }
}

impl std::ops::Mul<f64> for Flux {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        self.scale(rhs)
    }
}

impl std::ops::Mul<Flux> for f64 {
    type Output = Flux;
    fn mul(self, rhs: Flux) -> Flux {
        rhs.scale(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = ShallowWaterState::new(100);
        assert_eq!(state.n_cells(), 100);
        assert_eq!(state.h.len(), 100);
    }

    #[test]
    fn test_cold_start() {
        let z_bed = vec![-10.0, -5.0, 0.0, 5.0];
        let state = ShallowWaterState::cold_start(0.0, &z_bed).unwrap();

        assert_eq!(state.h[0], 10.0);
        assert_eq!(state.h[1], 5.0);
        assert_eq!(state.h[2], 0.0);
        assert_eq!(state.h[3], 0.0);
    }

    #[test]
    fn test_velocity_calculation() {
        let mut state = ShallowWaterState::new(1);
        state.h[0] = 2.0;
        state.hu[0] = 4.0;
        state.hv[0] = 6.0;

        let params = NumericalParams::default();
        let vel = state.velocity(0, &params);

        assert!((vel.u - 2.0).abs() < 1e-10);
        assert!((vel.v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_flux_operations() {
        let f1 = Flux::new(1.0, 2.0, 3.0);
        let f2 = Flux::new(0.5, 1.0, 1.5);

        let sum = f1 + f2;
        assert_eq!(sum.mass, 1.5);

        let scaled = f1 * 2.0;
        assert_eq!(scaled.mass, 2.0);
    }
}
