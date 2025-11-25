// src-tauri/src/marihydro/domain/state.rs

use ndarray::{s, Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::physics::schemes::PrimitiveVars;

/// 物理状态 (带Ghost层)
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct State {
    pub nx: usize,
    pub ny: usize,
    pub ng: usize,

    pub h: Array2<f64>,
    pub u: Array2<f64>,
    pub v: Array2<f64>,
    pub c: Array2<f64>,

    #[serde(skip)]
    zb: Arc<Array2<f64>>,
}

/// 安全可变切片
pub struct StateSlicesMut<'a> {
    pub h: &'a mut [f64],
    pub u: &'a mut [f64],
    pub v: &'a mut [f64],
    pub c: &'a mut [f64],
}

/// 安全只读切片
pub struct StateSlices<'a> {
    pub h: &'a [f64],
    pub u: &'a [f64],
    pub v: &'a [f64],
    pub c: &'a [f64],
}

impl State {
    /// 冷启动
    pub fn cold_start(
        nx: usize,
        ny: usize,
        ng: usize,
        initial_eta: f64,
        zb: Arc<Array2<f64>>,
    ) -> MhResult<Self> {
        let total_nx = nx + 2 * ng;
        let total_ny = ny + 2 * ng;

        if zb.dim() != (total_ny, total_nx) {
            return Err(MhError::InvalidMesh {
                message: format!(
                    "地形尺寸{:?}与状态({},{})不匹配",
                    zb.dim(),
                    total_ny,
                    total_nx
                ),
            });
        }

        let mut h = Array2::zeros((total_ny, total_nx));
        for j in 0..total_ny {
            for i in 0..total_nx {
                h[[j, i]] = (initial_eta - zb[[j, i]]).max(0.0);
            }
        }

        let state = Self {
            nx,
            ny,
            ng,
            h,
            u: Array2::zeros((total_ny, total_nx)),
            v: Array2::zeros((total_ny, total_nx)),
            c: Array2::zeros((total_ny, total_nx)),
            zb,
        };

        if !state.is_standard_layout() {
            return Err(MhError::InternalError(
                "State数组必须是C-Contiguous布局".into(),
            ));
        }

        Ok(state)
    }

    /// 克隆结构 (用于双缓冲)
    pub fn clone_structure(&self) -> Self {
        Self {
            nx: self.nx,
            ny: self.ny,
            ng: self.ng,
            h: Array2::zeros(self.h.dim()),
            u: Array2::zeros(self.u.dim()),
            v: Array2::zeros(self.v.dim()),
            c: Array2::zeros(self.c.dim()),
            zb: Arc::clone(&self.zb),
        }
    }

    /// 获取可变切片 (消除UB)
    pub fn as_slices_mut(&mut self) -> MhResult<StateSlicesMut<'_>> {
        Ok(StateSlicesMut {
            h: self
                .h
                .as_slice_mut()
                .ok_or_else(|| MhError::InternalError("h非标准布局".into()))?,
            u: self
                .u
                .as_slice_mut()
                .ok_or_else(|| MhError::InternalError("u非标准布局".into()))?,
            v: self
                .v
                .as_slice_mut()
                .ok_or_else(|| MhError::InternalError("v非标准布局".into()))?,
            c: self
                .c
                .as_slice_mut()
                .ok_or_else(|| MhError::InternalError("c非标准布局".into()))?,
        })
    }

    /// 获取只读切片
    pub fn as_slices(&self) -> MhResult<StateSlices<'_>> {
        Ok(StateSlices {
            h: self
                .h
                .as_slice()
                .ok_or_else(|| MhError::InternalError("h非标准布局".into()))?,
            u: self
                .u
                .as_slice()
                .ok_or_else(|| MhError::InternalError("u非标准布局".into()))?,
            v: self
                .v
                .as_slice()
                .ok_or_else(|| MhError::InternalError("v非标准布局".into()))?,
            c: self
                .c
                .as_slice()
                .ok_or_else(|| MhError::InternalError("c非标准布局".into()))?,
        })
    }

    /// 获取原始变量 (含地形)
    #[inline(always)]
    pub fn get_primitive(&self, idx: usize) -> PrimitiveVars {
        debug_assert!(idx < self.h.len(), "索引越界: {}", idx);
        unsafe { self.get_primitive_unchecked(idx) }
    }

    #[inline(always)]
    pub unsafe fn get_primitive_unchecked(&self, idx: usize) -> PrimitiveVars {
        let h = *self
            .h
            .as_slice_memory_order()
            .unwrap_unchecked()
            .get_unchecked(idx);
        let u = *self
            .u
            .as_slice_memory_order()
            .unwrap_unchecked()
            .get_unchecked(idx);
        let v = *self
            .v
            .as_slice_memory_order()
            .unwrap_unchecked()
            .get_unchecked(idx);
        let c = *self
            .c
            .as_slice_memory_order()
            .unwrap_unchecked()
            .get_unchecked(idx);
        let z = *self
            .zb
            .as_slice_memory_order()
            .unwrap_unchecked()
            .get_unchecked(idx);

        PrimitiveVars {
            h,
            u,
            v,
            c,
            z,
            eta: h + z,
        }
    }

    /// Ghost单元拷贝
    #[inline]
    pub unsafe fn copy_cell_unchecked(&mut self, dst: usize, src: usize) {
        let slices = self.as_slices_mut().unwrap_unchecked();
        *slices.h.get_unchecked_mut(dst) = *slices.h.get_unchecked(src);
        *slices.u.get_unchecked_mut(dst) = *slices.u.get_unchecked(src);
        *slices.v.get_unchecked_mut(dst) = *slices.v.get_unchecked(src);
        *slices.c.get_unchecked_mut(dst) = *slices.c.get_unchecked(src);
    }

    #[cfg(debug_assertions)]
    pub fn copy_cell(&mut self, dst: usize, src: usize) {
        assert!(dst < self.h.len() && src < self.h.len());
        unsafe { self.copy_cell_unchecked(dst, src) }
    }

    #[cfg(not(debug_assertions))]
    #[inline(always)]
    pub fn copy_cell(&mut self, dst: usize, src: usize) {
        unsafe { self.copy_cell_unchecked(dst, src) }
    }

    /// 网格拓扑
    #[inline(always)]
    pub fn total_shape(&self) -> (usize, usize) {
        (self.ny + 2 * self.ng, self.nx + 2 * self.ng)
    }

    #[inline(always)]
    pub fn physical_shape(&self) -> (usize, usize) {
        (self.ny, self.nx)
    }

    #[inline(always)]
    pub fn total_cells(&self) -> usize {
        let (ny, nx) = self.total_shape();
        ny * nx
    }

    #[inline(always)]
    pub fn flat_index(&self, j: usize, i: usize) -> usize {
        let (_, nx_total) = self.total_shape();
        j * nx_total + i
    }

    /// 物理区域视图
    pub fn physical_h(&self) -> ArrayView2<f64> {
        let ng = self.ng;
        self.h.slice(s![ng..ng + self.ny, ng..ng + self.nx])
    }

    pub fn physical_u(&self) -> ArrayView2<f64> {
        let ng = self.ng;
        self.u.slice(s![ng..ng + self.ny, ng..ng + self.nx])
    }

    pub fn physical_v(&self) -> ArrayView2<f64> {
        let ng = self.ng;
        self.v.slice(s![ng..ng + self.ny, ng..ng + self.nx])
    }

    #[inline]
    pub fn is_standard_layout(&self) -> bool {
        self.h.is_standard_layout()
            && self.u.is_standard_layout()
            && self.v.is_standard_layout()
            && self.c.is_standard_layout()
    }

    /// 数值健康检查
    pub fn validate(&self, time: f64) -> MhResult<()> {
        const MAX_DEPTH: f64 = 15_000.0;
        const MAX_VELOCITY: f64 = 100.0;

        let h = self.physical_h();
        let u = self.physical_u();
        let v = self.physical_v();

        for (idx, ((&val_h, &val_u), &val_v)) in h.iter().zip(u.iter()).zip(v.iter()).enumerate() {
            if val_h.is_nan() || val_u.is_nan() || val_v.is_nan() {
                let j = idx / self.nx;
                let i = idx % self.nx;
                return Err(MhError::NumericalInstability {
                    message: format!("检测到NaN at ({},{})", j, i),
                    time,
                    location: Some((j, i)),
                });
            }

            if val_h.is_infinite() || val_u.is_infinite() || val_v.is_infinite() {
                let j = idx / self.nx;
                let i = idx % self.nx;
                return Err(MhError::NumericalInstability {
                    message: format!("检测到Inf at ({},{})", j, i),
                    time,
                    location: Some((j, i)),
                });
            }

            if val_h > MAX_DEPTH {
                let j = idx / self.nx;
                let i = idx % self.nx;
                return Err(MhError::NumericalInstability {
                    message: format!("水深异常: {:.2}m at ({},{})", val_h, j, i),
                    time,
                    location: Some((j, i)),
                });
            }

            if val_u.abs() > MAX_VELOCITY || val_v.abs() > MAX_VELOCITY {
                let j = idx / self.nx;
                let i = idx % self.nx;
                return Err(MhError::NumericalInstability {
                    message: format!("流速异常: ({:.2},{:.2}) m/s at ({},{})", val_u, val_v, j, i),
                    time,
                    location: Some((j, i)),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arc_zb_zero_copy() {
        let zb = Arc::new(Array2::zeros((6, 6)));
        let state1 = State::cold_start(4, 4, 1, 0.0, Arc::clone(&zb)).unwrap();
        let state2 = state1.clone_structure();

        assert_eq!(Arc::strong_count(&zb), 3);
    }

    #[test]
    fn test_get_primitive_with_zb() {
        let zb = Arc::new(Array2::from_elem((6, 6), 5.0));
        let state = State::cold_start(4, 4, 1, 10.0, zb).unwrap();

        let idx = state.flat_index(2, 2);
        let prim = state.get_primitive(idx);

        assert_eq!(prim.z, 5.0);
        assert_eq!(prim.h, 5.0);
        assert_eq!(prim.eta, 10.0);
    }

    #[test]
    fn test_slices_mut_no_ub() {
        let zb = Arc::new(Array2::zeros((6, 6)));
        let mut state = State::cold_start(4, 4, 1, 0.0, zb).unwrap();

        let slices = state.as_slices_mut().unwrap();
        slices.h[0] = 1.0;
        slices.u[0] = 2.0;

        assert_eq!(state.h.as_slice().unwrap()[0], 1.0);
        assert_eq!(state.u.as_slice().unwrap()[0], 2.0);
    }
}
