// src-tauri/src/marihydro/domain/state/view.rs

//! 状态视图（用于借用分离）

use crate::marihydro::core::traits::state::ConservedState;
use crate::marihydro::core::types::{NumericalParams, SafeVelocity};
use glam::DVec2;

/// 不可变状态视图
///
/// 允许同时借用多个字段进行只读访问
pub struct StateView<'a> {
    pub h: &'a [f64],
    pub hu: &'a [f64],
    pub hv: &'a [f64],
    pub z: &'a [f64],
}

impl<'a> StateView<'a> {
    /// 创建状态视图
    pub fn new(h: &'a [f64], hu: &'a [f64], hv: &'a [f64], z: &'a [f64]) -> Self {
        debug_assert_eq!(h.len(), hu.len());
        debug_assert_eq!(h.len(), hv.len());
        debug_assert_eq!(h.len(), z.len());
        Self { h, hu, hv, z }
    }

    /// 单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.h.len()
    }

    /// 获取守恒状态
    #[inline]
    pub fn get(&self, idx: usize) -> ConservedState {
        ConservedState::new(self.h[idx], self.hu[idx], self.hv[idx])
    }

    /// 获取水位
    #[inline]
    pub fn water_level(&self, idx: usize) -> f64 {
        self.h[idx] + self.z[idx]
    }

    /// 获取速度
    #[inline]
    pub fn velocity(&self, idx: usize, params: &NumericalParams) -> SafeVelocity {
        params.safe_velocity(self.hu[idx], self.hv[idx], self.h[idx])
    }

    /// 获取原始变量
    #[inline]
    pub fn primitive(&self, idx: usize, params: &NumericalParams) -> (f64, f64, f64) {
        let vel = self.velocity(idx, params);
        (self.h[idx], vel.u, vel.v)
    }

    /// 判断是否为干单元
    #[inline]
    pub fn is_dry(&self, idx: usize, params: &NumericalParams) -> bool {
        params.is_dry(self.h[idx])
    }
}

/// 可变状态视图
///
/// 允许同时借用多个字段进行读写访问
pub struct StateViewMut<'a> {
    pub h: &'a mut [f64],
    pub hu: &'a mut [f64],
    pub hv: &'a mut [f64],
}

impl<'a> StateViewMut<'a> {
    /// 创建可变状态视图
    pub fn new(h: &'a mut [f64], hu: &'a mut [f64], hv: &'a mut [f64]) -> Self {
        debug_assert_eq!(h.len(), hu.len());
        debug_assert_eq!(h.len(), hv.len());
        Self { h, hu, hv }
    }

    /// 单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.h.len()
    }

    /// 设置守恒状态
    #[inline]
    pub fn set(&mut self, idx: usize, state: ConservedState) {
        self.h[idx] = state.h;
        self.hu[idx] = state.hu;
        self.hv[idx] = state.hv;
    }

    /// 从原始变量设置
    #[inline]
    pub fn set_from_primitive(&mut self, idx: usize, h: f64, u: f64, v: f64) {
        self.h[idx] = h;
        self.hu[idx] = h * u;
        self.hv[idx] = h * v;
    }

    /// 应用增量更新
    #[inline]
    pub fn apply_delta(&mut self, idx: usize, dh: f64, dhu: f64, dhv: f64) {
        self.h[idx] += dh;
        self.hu[idx] += dhu;
        self.hv[idx] += dhv;
    }

    /// 批量应用通量更新
    pub fn apply_flux_update(
        &mut self,
        dt: f64,
        areas: &[f64],
        flux_h: &[f64],
        flux_hu: &[f64],
        flux_hv: &[f64],
    ) {
        for i in 0..self.n_cells() {
            let inv_area = 1.0 / areas[i];
            self.h[i] += dt * flux_h[i] * inv_area;
            self.hu[i] += dt * flux_hu[i] * inv_area;
            self.hv[i] += dt * flux_hv[i] * inv_area;
        }
    }

    /// 强制非负水深
    pub fn enforce_non_negative(&mut self, h_min: f64) {
        for h in self.h.iter_mut() {
            if *h < h_min {
                *h = 0.0;
            }
        }
    }
}

/// 双缓冲状态（用于时间步进）
pub struct DoubleBufferState {
    current: usize,
    states: [Vec<f64>; 2],
}

impl DoubleBufferState {
    /// 创建双缓冲
    pub fn new(size: usize) -> Self {
        Self {
            current: 0,
            states: [vec![0.0; size], vec![0.0; size]],
        }
    }

    /// 获取当前状态
    pub fn current(&self) -> &[f64] {
        &self.states[self.current]
    }

    /// 获取下一状态（可变）
    pub fn next_mut(&mut self) -> &mut [f64] {
        &mut self.states[1 - self.current]
    }

    /// 交换缓冲区
    pub fn swap(&mut self) {
        self.current = 1 - self.current;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_view() {
        let h = vec![1.0, 2.0, 3.0];
        let hu = vec![0.1, 0.2, 0.3];
        let hv = vec![0.0, 0.0, 0.0];
        let z = vec![-1.0, -2.0, -3.0];

        let view = StateView::new(&h, &hu, &hv, &z);

        assert_eq!(view.n_cells(), 3);
        assert!((view.water_level(0) - 0.0).abs() < 1e-10);
        assert!((view.water_level(1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_state_view_mut() {
        let mut h = vec![1.0, 2.0];
        let mut hu = vec![0.0, 0.0];
        let mut hv = vec![0.0, 0.0];

        {
            let mut view = StateViewMut::new(&mut h, &mut hu, &mut hv);
            view.set_from_primitive(0, 3.0, 1.0, 0.5);
        }

        assert!((h[0] - 3.0).abs() < 1e-10);
        assert!((hu[0] - 3.0).abs() < 1e-10);
        assert!((hv[0] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_double_buffer() {
        let mut buffer = DoubleBufferState::new(10);

        buffer.next_mut()[0] = 42.0;
        buffer.swap();

        assert!((buffer.current()[0] - 42.0).abs() < 1e-10);
    }
}
