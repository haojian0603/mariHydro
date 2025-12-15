// crates/mh_physics/src/engine/strategy/workspace.rs

//! 求解器工作区 - Backend泛型实现
//! 
//! 统一管理通量、源项、临时变量的存储与生命周期。
//! 所有缓冲区通过Backend泛型化，支持f32/f64/GPU后端。
//! 
//! # 内存布局
//! 
//! 采用SoA(Structure of Arrays)布局优化缓存访问：
//! 
//! ```text
//! flux_h:    [Δh₀, Δh₁, Δh₂, ...]  (质量通量)
//! flux_hu:   [Δhu₀, Δhu₁, Δhu₂, ...] (x动量通量)
//! flux_hv:   [Δhv₀, Δhv₁, Δhv₂, ...] (y动量通量)
//! vel_u:     [u₀, u₁, u₂, ...]     (单元速度x分量)
//! vel_v:     [v₀, v₁, v₂, ...]     (单元速度y分量)
//! eta:       [η₀, η₁, η₂, ...]     (水位 = h + z)
//! ```
//! 
//! # 泛型设计
//! 
//! - 所有数值字段使用`B::Scalar`类型
//! - 所有缓冲区使用`B::Buffer<B::Scalar>`类型
//! - 通过Backend实例创建和填充缓冲区
//! 
//! # 示例
//! 
//! ```rust
//! use mh_runtime::CpuBackend;
//! use mh_physics::engine::SolverWorkspaceGeneric;
//! 
//! // 创建f32精度工作区
//! let backend_f32 = CpuBackend::<f32>::new();
//! let workspace_f32 = SolverWorkspaceGeneric::new_with_backend(backend_f32, 1000, 2000);
//! 
//! // 创建f64精度工作区
//! let backend_f64 = CpuBackend::<f64>::new();
//! let workspace_f64 = SolverWorkspaceGeneric::new_with_backend(backend_f64, 1000, 2000);
//! ```

use mh_runtime::{Backend, RuntimeScalar};

/// 泛型求解器工作区
/// 
/// 持有Backend实例以支持缓冲区操作和几何计算。
/// 所有字段通过泛型参数`B`确定精度。
#[derive(Debug)]
pub struct SolverWorkspaceGeneric<B: Backend> {
    /// 单元数量
    n_cells: usize,
    /// 面数量
    n_faces: usize,
    
    // ========== 通量累加区 ==========
    /// 质量通量累加 (Δh per unit time)
    pub flux_h: B::Buffer<B::Scalar>,
    /// x动量通量累加 (Δhu per unit time)
    pub flux_hu: B::Buffer<B::Scalar>,
    /// y动量通量累加 (Δhv per unit time)
    pub flux_hv: B::Buffer<B::Scalar>,
    
    // ========== 源项累加区 ==========
    /// x动量源项 (床坡、摩擦等)
    pub source_hu: B::Buffer<B::Scalar>,
    /// y动量源项
    pub source_hv: B::Buffer<B::Scalar>,
    
    // ========== 重构辅助区 ==========
    /// 单元速度u分量 (用于梯度计算)
    pub vel_u: B::Buffer<B::Scalar>,
    /// 单元速度v分量
    pub vel_v: B::Buffer<B::Scalar>,
    /// 水位η = h + zb (用于C-property保持)
    pub eta: B::Buffer<B::Scalar>,
    
    // ========== 面通量缓存区 (并行计算用) ==========
    /// 面质量通量 (临时存储)
    pub face_flux_h: B::Buffer<B::Scalar>,
    /// 面x动量通量
    pub face_flux_hu: B::Buffer<B::Scalar>,
    /// 面y动量通量
    pub face_flux_hv: B::Buffer<B::Scalar>,
    /// 面最大波速 (用于CFL计算)
    pub face_wave_speed: B::Buffer<B::Scalar>,
    
    /// Backend实例 (用于缓冲区操作)
    backend: B,
}

impl<B: Backend> SolverWorkspaceGeneric<B> {
    /// 使用Backend实例创建工作区
    /// 
    /// # 参数
    /// - `backend`: 计算后端实例
    /// - `n_cells`: 单元数量
    /// - `n_faces`: 面数量
    /// 
    /// # 内存分配
    /// 一次性分配所有缓冲区，避免运行时重复分配。
    #[inline]
    pub fn new_with_backend(backend: B, n_cells: usize, n_faces: usize) -> Self {
        // 预分配所有缓冲区
        let flux_h = backend.alloc(n_cells);
        let flux_hu = backend.alloc(n_cells);
        let flux_hv = backend.alloc(n_cells);
        let source_hu = backend.alloc(n_cells);
        let source_hv = backend.alloc(n_cells);
        let vel_u = backend.alloc(n_cells);
        let vel_v = backend.alloc(n_cells);
        let eta = backend.alloc(n_cells);
        
        let face_flux_h = backend.alloc(n_faces);
        let face_flux_hu = backend.alloc(n_faces);
        let face_flux_hv = backend.alloc(n_faces);
        let face_wave_speed = backend.alloc(n_faces);
        
        Self {
            n_cells,
            n_faces,
            flux_h,
            flux_hu,
            flux_hv,
            source_hu,
            source_hv,
            vel_u,
            vel_v,
            eta,
            face_flux_h,
            face_flux_hu,
            face_flux_hv,
            face_wave_speed,
            backend,
        }
    }
    
    /// 获取Backend引用
    /// 
    /// # 用途
    /// 用于几何计算和标量转换。
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 重置通量累加区为零
    /// 
    /// # 实现细节
    /// 使用Backend的fill方法，避免硬编码0.0，支持GPU后端。
    pub fn reset_fluxes(&mut self) {
        let zero = <B::Scalar as RuntimeScalar>::ZERO;
        self.backend.fill(&mut self.flux_h, zero);
        self.backend.fill(&mut self.flux_hu, zero);
        self.backend.fill(&mut self.flux_hv, zero);
    }
    
    /// 重置源项累加区为零
    pub fn reset_sources(&mut self) {
        let zero = <B::Scalar as RuntimeScalar>::ZERO;
        self.backend.fill(&mut self.source_hu, zero);
        self.backend.fill(&mut self.source_hv, zero);
    }
    
    /// 重置所有工作区缓冲区
    /// 
    /// # 性能
    /// O(n_cells + n_faces)时间复杂度，适用于每步计算。
    #[inline]
    pub fn reset(&mut self) {
        self.reset_fluxes();
        self.reset_sources();
        
        let zero = <B::Scalar as RuntimeScalar>::ZERO;
        self.backend.fill(&mut self.vel_u, zero);
        self.backend.fill(&mut self.vel_v, zero);
        self.backend.fill(&mut self.eta, zero);
        self.backend.fill(&mut self.face_flux_h, zero);
        self.backend.fill(&mut self.face_flux_hu, zero);
        self.backend.fill(&mut self.face_flux_hv, zero);
        self.backend.fill(&mut self.face_wave_speed, zero);
    }
    
    /// 获取单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
    
    /// 获取面数量
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.n_faces
    }
}

// ============================================================================
// 测试模块
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_workspace_creation_f32() {
        let backend = CpuBackend::<f32>::new();
        let ws = SolverWorkspaceGeneric::new_with_backend(backend, 100, 200);
        
        assert_eq!(ws.n_cells(), 100);
        assert_eq!(ws.n_faces(), 200);
    }

    #[test]
    fn test_workspace_creation_f64() {
        let backend = CpuBackend::<f64>::new();
        let ws = SolverWorkspaceGeneric::new_with_backend(backend, 100, 200);
        
        assert_eq!(ws.n_cells(), 100);
        assert_eq!(ws.n_faces(), 200);
    }

    #[test]
    fn test_workspace_reset_fluxes() {
        let backend = CpuBackend::<f64>::new();
        let mut ws = SolverWorkspaceGeneric::new_with_backend(backend, 10, 5);
        
        // 手动设置一些值
        ws.flux_h[0] = 1.5;
        assert_eq!(ws.flux_h[0], 1.5);
        
        ws.reset_fluxes();
        assert_eq!(ws.flux_h[0], 0.0);
    }

    #[test]
    fn test_workspace_reset_all() {
        let backend = CpuBackend::<f64>::new();
        let mut ws = SolverWorkspaceGeneric::new_with_backend(backend, 5, 3);
        
        // 设置一些值
        ws.vel_u[0] = 1.0;
        ws.face_flux_h[0] = 2.0;
        
        ws.reset();
        
        assert_eq!(ws.vel_u[0], 0.0);
        assert_eq!(ws.face_flux_h[0], 0.0);
    }

    #[test]
    fn test_workspace_buffer_types() {
        let backend_f32 = CpuBackend::<f32>::new();
        let ws_f32 = SolverWorkspaceGeneric::new_with_backend(backend_f32, 10, 5);
        
        // 验证f32缓冲区
        assert_eq!(std::mem::size_of_val(&ws_f32.flux_h[0]), 4);
        
        let backend_f64 = CpuBackend::<f64>::new();
        let ws_f64 = SolverWorkspaceGeneric::new_with_backend(backend_f64, 10, 5);
        
        // 验证f64缓冲区
        assert_eq!(std::mem::size_of_val(&ws_f64.flux_h[0]), 8);
    }
}