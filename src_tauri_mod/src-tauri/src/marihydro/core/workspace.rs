// src-tauri/src/marihydro/core/workspace.rs

//! 预分配缓冲区池（已废弃）
//!
//! ⚠️ **此模块已废弃**，请使用 `core::memory` 模块替代。
//!
//! 迁移指南：
//! ```rust
//! // 旧代码
//! use crate::marihydro::core::workspace::{Workspace, BufferPool};
//!
//! // 新代码
//! use crate::marihydro::core::memory::{Workspace, BufferPool, WorkspaceBuilder};
//! ```
//!
//! 新的 `memory` 模块提供：
//! - 统一的内存管理子系统
//! - `WorkspaceBuilder` 构建器模式
//! - 增强的 `prepare_for_step()` 方法用于时间步间缓冲区管理
//! - 与 `BufferPool` 更好的集成

#![deprecated(
    since = "0.3.0",
    note = "请使用 core::memory 模块替代。此模块将在 0.4.0 版本移除。"
)]

use glam::DVec2;
use parking_lot::Mutex;

// ============================================================
// 缓冲区池（线程安全）
// ============================================================

/// 通用缓冲区池
pub struct BufferPool<T: Clone + Default + Send> {
    buffers: Mutex<Vec<Vec<T>>>,
    element_count: usize,
    max_pool_size: usize,
}

impl<T: Clone + Default + Send> BufferPool<T> {
    pub fn new(element_count: usize) -> Self {
        Self {
            buffers: Mutex::new(Vec::with_capacity(8)),
            element_count,
            max_pool_size: 16,
        }
    }

    /// 获取缓冲区（从池中取或新建）
    pub fn acquire(&self) -> PooledBuffer<T> {
        let buffer = self
            .buffers
            .lock()
            .pop()
            .unwrap_or_else(|| vec![T::default(); self.element_count]);

        PooledBuffer {
            buffer: Some(buffer),
            pool: self,
        }
    }

    /// 归还缓冲区
    fn release(&self, mut buffer: Vec<T>) {
        let mut pool = self.buffers.lock();
        if pool.len() < self.max_pool_size {
            // 清零并归还
            buffer.iter_mut().for_each(|x| *x = T::default());
            pool.push(buffer);
        }
        // 否则让buffer被drop
    }
}

/// 自动归还的缓冲区
pub struct PooledBuffer<'a, T: Clone + Default + Send> {
    buffer: Option<Vec<T>>,
    pool: &'a BufferPool<T>,
}

impl<T: Clone + Default + Send> std::ops::Deref for PooledBuffer<'_, T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        self.buffer.as_ref().unwrap()
    }
}

impl<T: Clone + Default + Send> std::ops::DerefMut for PooledBuffer<'_, T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        self.buffer.as_mut().unwrap()
    }
}

impl<T: Clone + Default + Send> Drop for PooledBuffer<'_, T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}

// ============================================================
// 主工作区
// ============================================================

/// 预分配的工作区缓冲区
///
/// # 使用模式
///
/// ```rust
/// let mut workspace = Workspace::new(mesh.n_cells(), mesh.n_faces());
///
/// // 使用固定缓冲区
/// workspace.reset_fluxes();
/// for i in 0..n_cells {
///     workspace.flux_h[i] += contribution;
/// }
///
/// // 使用池化临时缓冲区
/// let mut temp = workspace.temp_scalar();
/// // temp 在作用域结束时自动归还
/// ```
pub struct Workspace {
    n_cells: usize,
    n_faces: usize,

    // ===== 固定缓冲区（整个模拟期间复用）=====
    /// 速度场
    pub velocities: Vec<DVec2>,

    /// 标量场梯度
    pub grad_x: Vec<f64>,
    pub grad_y: Vec<f64>,

    /// 向量场梯度（2x2张量）
    pub du_dx: Vec<f64>,
    pub du_dy: Vec<f64>,
    pub dv_dx: Vec<f64>,
    pub dv_dy: Vec<f64>,

    /// 通量累加缓冲区
    pub flux_h: Vec<f64>,
    pub flux_hu: Vec<f64>,
    pub flux_hv: Vec<f64>,

    /// 涡粘系数
    pub nu_t: Vec<f64>,

    /// 源项累加缓冲区
    pub source_h: Vec<f64>,
    pub source_hu: Vec<f64>,
    pub source_hv: Vec<f64>,

    /// 左右状态重构
    pub h_left: Vec<f64>,
    pub h_right: Vec<f64>,
    pub vel_left: Vec<DVec2>,
    pub vel_right: Vec<DVec2>,

    // ===== 临时缓冲区池 =====
    scalar_pool: BufferPool<f64>,
    vector_pool: BufferPool<DVec2>,
}

impl Workspace {
    /// 创建工作区
    pub fn new(n_cells: usize, n_faces: usize) -> Self {
        Self {
            n_cells,
            n_faces,

            // 固定缓冲区
            velocities: vec![DVec2::ZERO; n_cells],
            grad_x: vec![0.0; n_cells],
            grad_y: vec![0.0; n_cells],
            du_dx: vec![0.0; n_cells],
            du_dy: vec![0.0; n_cells],
            dv_dx: vec![0.0; n_cells],
            dv_dy: vec![0.0; n_cells],
            flux_h: vec![0.0; n_cells],
            flux_hu: vec![0.0; n_cells],
            flux_hv: vec![0.0; n_cells],
            nu_t: vec![0.0; n_cells],
            source_h: vec![0.0; n_cells],
            source_hu: vec![0.0; n_cells],
            source_hv: vec![0.0; n_cells],
            h_left: vec![0.0; n_faces],
            h_right: vec![0.0; n_faces],
            vel_left: vec![DVec2::ZERO; n_faces],
            vel_right: vec![DVec2::ZERO; n_faces],

            // 缓冲区池
            scalar_pool: BufferPool::new(n_cells),
            vector_pool: BufferPool::new(n_cells),
        }
    }

    /// 获取单元数
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 获取面数
    pub fn n_faces(&self) -> usize {
        self.n_faces
    }

    /// 获取临时标量缓冲区
    pub fn temp_scalar(&self) -> PooledBuffer<f64> {
        self.scalar_pool.acquire()
    }

    /// 获取临时向量缓冲区
    pub fn temp_vector(&self) -> PooledBuffer<DVec2> {
        self.vector_pool.acquire()
    }

    /// 重置通量缓冲区
    pub fn reset_fluxes(&mut self) {
        self.flux_h.fill(0.0);
        self.flux_hu.fill(0.0);
        self.flux_hv.fill(0.0);
    }

    /// 重置源项缓冲区
    pub fn reset_sources(&mut self) {
        self.source_h.fill(0.0);
        self.source_hu.fill(0.0);
        self.source_hv.fill(0.0);
    }

    /// 重置梯度缓冲区
    pub fn reset_gradients(&mut self) {
        self.grad_x.fill(0.0);
        self.grad_y.fill(0.0);
        self.du_dx.fill(0.0);
        self.du_dy.fill(0.0);
        self.dv_dx.fill(0.0);
        self.dv_dy.fill(0.0);
    }

    /// 调整大小（网格变化时调用）
    pub fn resize_if_needed(&mut self, n_cells: usize, n_faces: usize) {
        if self.n_cells != n_cells || self.n_faces != n_faces {
            *self = Self::new(n_cells, n_faces);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool() {
        let pool = BufferPool::<f64>::new(100);

        {
            let mut buf1 = pool.acquire();
            assert_eq!(buf1.len(), 100);
            buf1[0] = 42.0;
        } // buf1归还

        {
            let buf2 = pool.acquire();
            // 应该被清零
            assert_eq!(buf2[0], 0.0);
        }
    }

    #[test]
    fn test_workspace_creation() {
        let ws = Workspace::new(10, 15);
        assert_eq!(ws.n_cells(), 10);
        assert_eq!(ws.n_faces(), 15);
        assert_eq!(ws.flux_h.len(), 10);
        assert_eq!(ws.h_left.len(), 15);
    }

    #[test]
    fn test_workspace_reset() {
        let mut ws = Workspace::new(5, 5);
        ws.flux_h[0] = 10.0;
        ws.source_hu[1] = 20.0;
        ws.grad_x[2] = 30.0;

        ws.reset_fluxes();
        assert_eq!(ws.flux_h[0], 0.0);
        assert_eq!(ws.source_hu[1], 20.0); // 未被重置

        ws.reset_sources();
        assert_eq!(ws.source_hu[1], 0.0);
        assert_eq!(ws.grad_x[2], 30.0); // 未被重置

        ws.reset_gradients();
        assert_eq!(ws.grad_x[2], 0.0);
    }
}
