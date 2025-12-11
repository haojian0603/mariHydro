//! 求解器工作区
//!
//! 提供通量、残差、临时变量的统一管理。

use crate::core::{Backend, DeviceBuffer, Scalar};

/// 泛型求解器工作区
///
/// 持有 Backend 实例以支持后续缓冲区操作。
#[derive(Debug)]
pub struct SolverWorkspaceGeneric<B: Backend> {
    /// 单元数量
    n_cells: usize,
    /// 面数量
    n_faces: usize,
    
    // ========== 通量累加 ==========
    /// 质量通量累加
    pub flux_h: B::Buffer<B::Scalar>,
    /// x动量通量累加
    pub flux_hu: B::Buffer<B::Scalar>,
    /// y动量通量累加
    pub flux_hv: B::Buffer<B::Scalar>,
    
    // ========== 源项累加 ==========
    /// x动量源项
    pub source_hu: B::Buffer<B::Scalar>,
    /// y动量源项
    pub source_hv: B::Buffer<B::Scalar>,
    
    // ========== 重构辅助 ==========
    /// 单元速度 u
    pub vel_u: B::Buffer<B::Scalar>,
    /// 单元速度 v
    pub vel_v: B::Buffer<B::Scalar>,
    /// 水位 η = h + z
    pub eta: B::Buffer<B::Scalar>,
    
    // ========== 面通量（并行计算用）==========
    /// 面质量通量
    pub face_flux_h: B::Buffer<B::Scalar>,
    /// 面x动量通量
    pub face_flux_hu: B::Buffer<B::Scalar>,
    /// 面y动量通量
    pub face_flux_hv: B::Buffer<B::Scalar>,
    /// 面最大波速
    pub face_wave_speed: B::Buffer<B::Scalar>,
    
    /// 后端实例
    backend: B,
}

impl<B: Backend> SolverWorkspaceGeneric<B> {
    /// 使用后端实例创建工作区
    pub fn new_with_backend(backend: B, n_cells: usize, n_faces: usize) -> Self {
        Self {
            n_cells,
            n_faces,
            flux_h: backend.alloc(n_cells),
            flux_hu: backend.alloc(n_cells),
            flux_hv: backend.alloc(n_cells),
            source_hu: backend.alloc(n_cells),
            source_hv: backend.alloc(n_cells),
            vel_u: backend.alloc(n_cells),
            vel_v: backend.alloc(n_cells),
            eta: backend.alloc(n_cells),
            face_flux_h: backend.alloc(n_faces),
            face_flux_hu: backend.alloc(n_faces),
            face_flux_hv: backend.alloc(n_faces),
            face_wave_speed: backend.alloc(n_faces),
            backend,
        }
    }
    
    /// 获取后端引用
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 重置通量
    pub fn reset_fluxes(&mut self) {
        let zero = <B::Scalar as Scalar>::from_f64(0.0);
        self.flux_h.fill(zero);
        self.flux_hu.fill(zero);
        self.flux_hv.fill(zero);
    }
    
    /// 重置源项
    pub fn reset_sources(&mut self) {
        let zero = <B::Scalar as Scalar>::from_f64(0.0);
        self.source_hu.fill(zero);
        self.source_hv.fill(zero);
    }
    
    /// 重置所有
    pub fn reset(&mut self) {
        self.reset_fluxes();
        self.reset_sources();
    }
    
    /// 单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
    
    /// 面数量
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.n_faces
    }
}
