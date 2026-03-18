// crates/mh_runtime/src/simd/mod.rs

//! SIMD 向量化加速模块
//!
//! 提供多版本 SIMD 内核：
//! - AVX-512 (x86_64)
//! - AVX2 (x86_64)
//! - 标量回退
//!
//! # 设计原则
//!
//! 1. **编译期检测**：使用 `cfg(target_feature)` 选择最优路径
//! 2. **运行时回退**：CPU 不支持时自动降级
//! 3. **对齐保证**：所有 SIMD 操作使用对齐加载/存储
//! 4. **尾部处理**：使用掩码处理非对齐尾部
//!
//! # 性能目标
//!
//! - AVX-512: 1e6 cells < 1ms
//! - AVX2: 1e6 cells < 2.5ms
//! - 标量: 1e6 cells < 10ms

// 子模块
mod kernels;
mod aligned;
mod traits;

pub use kernels::*;
pub use aligned::*;
pub use traits::*;

/// SIMD 能力检测结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdCapability {
    /// AVX-512F
    Avx512,
    /// AVX2 + FMA
    Avx2,
    /// SSE4.2
    Sse42,
    /// 标量回退
    Scalar,
}

impl SimdCapability {
    /// 检测当前 CPU 的 SIMD 能力
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        if is_x86_feature_detected!("avx512f") {
            Self::Avx512
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            Self::Avx2
        } else if is_x86_feature_detected!("sse4.2") {
            Self::Sse42
        } else {
            Self::Scalar
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn detect() -> Self {
        Self::Scalar
    }

    /// 每次迭代处理的 f64 元素数量
    pub fn f64_lanes(&self) -> usize {
        match self {
            Self::Avx512 => 8,
            Self::Avx2 => 4,
            Self::Sse42 => 2,
            Self::Scalar => 1,
        }
    }

    /// 每次迭代处理的 f32 元素数量
    pub fn f32_lanes(&self) -> usize {
        self.f64_lanes() * 2
    }

    /// 获取描述字符串
    pub fn name(&self) -> &'static str {
        match self {
            Self::Avx512 => "AVX-512",
            Self::Avx2 => "AVX2+FMA",
            Self::Sse42 => "SSE4.2",
            Self::Scalar => "Scalar",
        }
    }
}

/// 全局 SIMD 能力（惰性初始化）
static SIMD_CAPABILITY: std::sync::OnceLock<SimdCapability> = std::sync::OnceLock::new();

/// 获取当前 SIMD 能力
pub fn simd_capability() -> SimdCapability {
    *SIMD_CAPABILITY.get_or_init(SimdCapability::detect)
}
