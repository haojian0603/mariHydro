//! 维度标记
//!
//! 提供编译期维度区分，用于类型安全的 2D/3D 代码。

use std::fmt::Debug;

/// 维度标记 trait
pub trait Dimension: Debug + Clone + Copy + Send + Sync + 'static {
    /// 维度数
    const NDIM: usize;
    
    /// 维度名称
    fn name() -> &'static str;
}

/// 2D 维度标记
#[derive(Debug, Clone, Copy, Default)]
pub struct D2;

impl Dimension for D2 {
    const NDIM: usize = 2;
    
    fn name() -> &'static str { "2D" }
}

/// 3D 维度标记（预留，当前不实现具体算法）
#[derive(Debug, Clone, Copy, Default)]
pub struct D3;

impl Dimension for D3 {
    const NDIM: usize = 3;
    
    fn name() -> &'static str { "3D" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_2d() {
        assert_eq!(D2::NDIM, 2);
        assert_eq!(D2::name(), "2D");
    }

    #[test]
    fn test_dimension_3d() {
        assert_eq!(D3::NDIM, 3);
        assert_eq!(D3::name(), "3D");
    }
}
