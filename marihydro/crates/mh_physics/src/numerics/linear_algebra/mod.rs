// =============================================================================
// mh_physics/src/numerics/linear_algebra/mod.rs
// =============================================================================
//! 稀疏线性代数模块
//!
//! 提供隐式求解所需的稀疏矩阵、预条件器和迭代求解器，
//! 支持 CPU/GPU 异构计算和零成本抽象。
//!
//! # 架构设计
//!
//! 本模块采用**Backend感知**设计：
//!
//! - **Layer 1**: 基础类型（`CsrMatrix`, `AlignedVec64`）- 无泛型
//! - **Layer 2**: Backend Trait（`Preconditioner<B>`）- 关联类型抽象
//! - **Layer 3**: 具体实现（`JacobiPreconditioner<B>`）- 全泛型
//!
//! # Backend 集成
//!
//! 所有计算组件接受 `B: Backend` 参数，支持：
//! - `CpuBackend<f32>`: 单精度CPU计算
//! - `CpuBackend<f64>`: 双精度CPU计算
//! - **未来**: `GpuBackend<f32>` 等
//!
//! 关键约束：
//! - 所有内存分配使用 `AlignedVec<64>`，确保SIMD对齐
//! - 计算内核必须支持 `no_std` 友好（`Pod + Clone` 约束）
//! - 错误处理使用 `Result<_, PreconditionerError>`，强制检查
//!

// 子模块
pub mod csr;
pub mod preconditioner;
pub mod solver;
pub mod vector_ops;
// pub use crate::builder::dyn_solver::{SolverStats, DynSolver}; // 错误：路径不存在且 SolverStats 已定义

pub use csr::{
    CsrBuilder, CsrMatrix, CsrPattern, CsrBuilderF64 as CsrBuilderLegacy,
};

pub use vector_ops::{
    axpy, axpy_inplace, copy, copy_bounded, dot, fill, norm2, norm_inf,
    linear_combination, add, sub, hadamard, hadamard_div,
    relative_residual, add_scaled, scale, xpay,
};

pub use preconditioner::{
    // Trait
    Preconditioner,
    ScalarPreconditioner,
    // 错误类型
    PreconditionerError,
    // 性能统计
    PreconditionerStats, PreconditionerStatsSnapshot,
    // 参数结构体
    SsorParams,
    // 预条件器实现
    IdentityPreconditioner, JacobiPreconditioner, SsorPreconditioner, Ilu0Preconditioner,
    // 类型别名（Backend 特化）
    JacobiPreconditionerF64, JacobiPreconditionerF32,
    SsorPreconditionerF64, SsorPreconditionerF32,
    Ilu0PreconditionerF64, Ilu0PreconditionerF32,
};

// 迭代求解器
pub use solver::{
    // Trait
    IterativeSolver,
    // 求解器实现
    ConjugateGradient, BiCgStabSolver, PcgSolver,
    // 工作空间
    CgWorkspace, BiCgStabWorkspace,
    // 配置与结果
    SolverConfig, SolverResult, SolverStatus,
};

// 依赖导入（必须在此导入以避免循环）
use mh_foundation::AlignedVec;
use mh_runtime::RuntimeScalar;

// ============================================================================
// 字节对齐工具
// ============================================================================

/// 64 字节对齐的向量类型别名 (使用 mh_foundation 的 AlignedVec)
pub type AlignedVec64<T> = AlignedVec<T>;

/// 创建对齐向量的工厂函数
#[inline]
pub fn aligned_vec<S: RuntimeScalar>(n: usize) -> AlignedVec64<S> {
    AlignedVec::zeros(n)
}

/// 从 slice 创建对齐向量
#[inline]
pub fn aligned_vec_from_slice<S: RuntimeScalar + Clone>(slice: &[S]) -> AlignedVec64<S> {
    let mut vec = AlignedVec::zeros(slice.len());
    vec.copy_from_slice(slice);
    vec
}

// ============================================================================
// SIMD 能力检测（编译时）
// ============================================================================

/// 运行时 SIMD 支持检测
#[inline]
pub fn has_simd_support() -> bool {
    // 编译时检测（最高效）
    cfg_if::cfg_if! {
        if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
            true
        } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
            true
        } else {
            false
        }
    }
}

/// 获取最优对齐字节数
#[inline]
pub fn optimal_alignment() -> usize {
    if has_simd_support() {
        64  // AVX-512 缓存行对齐
    } else {
        32  // 通用对齐
    }
}

// ============================================================================
// 性能剖析工具
// ============================================================================

/// 计算内存带宽效率（GB/s）
///
/// 用于评估 BLAS 操作的内存效率
#[cfg(feature = "profiling")]
pub fn memory_bandwidth(bytes: usize, nanoseconds: u64) -> f64 {
    let seconds = nanoseconds as f64 / 1e9;
    (bytes as f64 / 1e9) / seconds
}


#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_module_exports() {
        // 验证所有类型可访问
        let _builder: CsrBuilder<f64> = CsrBuilder::new(3, 3);
        let backend = CpuBackend::<f64>::new();
        let _precond: JacobiPreconditionerF64 = JacobiPreconditionerF64::new(&backend);
        let config = SolverConfig::new(1e-8, 100); // 先创建 config
        let _solver: ConjugateGradient<f64> = ConjugateGradient::new(config); // 再传入 config
    }

    #[test]
    fn test_alignment() {
        let v: AlignedVec64<f64> = aligned_vec(100);
        assert_eq!(v.as_ptr() as usize % 64, 0, "向量必须64字节对齐");
        
        let v2 = aligned_vec_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v2.len(), 3);
        assert_eq!(v2.as_ptr() as usize % 64, 0, "向量必须64字节对齐");
    }

    #[test]
    fn test_simd_detection() {
        let has_simd = has_simd_support();
        println!("SIMD 支持: {}", has_simd);
        assert!(optimal_alignment() >= 32, "最小对齐应为32字节");
        
        // 在支持AVX2的机器上应该返回true
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        assert!(has_simd, "AVX2支持应被检测到");
    }

    #[test]
    fn test_optimal_alignment() {
        let align = optimal_alignment();
        assert!(align == 32 || align == 64, "对齐必须是32或64字节");
    }

    #[test]
    fn test_aligned_vec_zeroed() {
        let v: AlignedVec64<f64> = aligned_vec(10);
        assert!(v.iter().all(|&x| x == 0.0), "零初始化失败");
    }

    #[test]
    fn test_aligned_vec_from_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = aligned_vec_from_slice(&data);
        assert_eq!(&*v, &data[..]);
    }
}