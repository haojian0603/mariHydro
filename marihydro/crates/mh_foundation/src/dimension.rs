//! 编译期维度系统
//!
//! 通过类型参数区分 2D/3D，实现零成本抽象。
//!
//! # 用法
//!
//! ```
//! use mh_foundation::dimension::{Dimension, D2, D3};
//!
//! fn compute<D: Dimension>() {
//!     if D::IS_3D {
//!         println!("3D computation with {} layers", D::N_LAYERS);
//!     } else {
//!         println!("2D computation");
//!     }
//! }
//! ```
//!
//! # 设计说明
//!
//! - D2: 二维模拟使用
//! - D3<N>: 三维模拟，编译期指定层数
//! - D3Dynamic: 运行时动态层数（用于配置驱动场景）



/// 维度 trait
/// 
/// 所有维度标记类型必须实现此 trait。
pub trait Dimension: 'static + Copy + Clone + Default + Send + Sync + std::fmt::Debug {
    /// 垂直层数（2D=1, 3D=N）
    const N_LAYERS: usize;
    
    /// 是否为 3D
    const IS_3D: bool = Self::N_LAYERS > 1;
    
    /// 维度名称（用于日志和调试）
    fn name() -> &'static str;
    
    /// 获取运行时层数（用于 D3Dynamic）
    fn n_layers(&self) -> usize {
        Self::N_LAYERS
    }
}

/// 2D 维度标记
/// 
/// 用于二维浅水方程模拟。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct D2;

impl Dimension for D2 {
    const N_LAYERS: usize = 1;
    
    fn name() -> &'static str { "2D" }
}

/// 3D 维度标记（编译期指定层数）
/// 
/// 用于三维分层模拟，层数在编译期确定。
/// 
/// # 示例
/// 
/// ```
/// use mh_foundation::dimension::{Dimension, D3};
/// 
/// // 10 层的 3D 模拟
/// type D3_10 = D3<10>;
/// assert_eq!(D3_10::N_LAYERS, 10);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct D3<const LAYERS: usize>;

impl<const LAYERS: usize> Dimension for D3<LAYERS> {
    const N_LAYERS: usize = LAYERS;
    
    fn name() -> &'static str { "3D" }
}

/// 运行时可变层数的 3D（用于配置驱动场景）
/// 
/// 当层数需要从配置文件读取时使用此类型。
/// 注意：这会有轻微的运行时开销。
#[derive(Debug, Clone, Copy)]
pub struct D3Dynamic {
    /// 层数
    pub n_layers: usize,
}

impl Default for D3Dynamic {
    fn default() -> Self { 
        Self { n_layers: 1 } 
    }
}

impl Dimension for D3Dynamic {
    /// 编译期无法确定，使用 1 作为占位
    const N_LAYERS: usize = 1;
    
    fn name() -> &'static str { "3D-Dynamic" }
    
    fn n_layers(&self) -> usize {
        self.n_layers
    }
}

/// 维度相关的大小计算
pub trait DimensionExt: Dimension {
    /// 给定单元数，计算总存储大小
    #[inline]
    fn storage_size(n_cells: usize) -> usize {
        n_cells * Self::N_LAYERS.max(1)
    }
    
    /// 3D 索引转换：(cell, layer) -> flat index
    #[inline]
    fn flat_index(cell: usize, layer: usize) -> usize {
        cell * Self::N_LAYERS.max(1) + layer
    }
    
    /// flat index -> (cell, layer)
    #[inline]
    fn unflat_index(flat: usize) -> (usize, usize) {
        let n_layers = Self::N_LAYERS.max(1);
        (flat / n_layers, flat % n_layers)
    }
}

// 为所有 Dimension 实现 DimensionExt
impl<D: Dimension> DimensionExt for D {}

/// 常用类型别名
/// 3D 5 层模式
pub type D3_5 = D3<5>;
/// 3D 10 层模式
pub type D3_10 = D3<10>;
/// 3D 20 层模式
pub type D3_20 = D3<20>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_d2_dimension() {
        assert_eq!(D2::N_LAYERS, 1);
        assert!(!D2::IS_3D);
        assert_eq!(D2::name(), "2D");
    }
    
    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_d3_dimension() {
        assert_eq!(D3::<5>::N_LAYERS, 5);
        assert!(D3::<5>::IS_3D);
        assert_eq!(D3::<5>::name(), "3D");
    }
    
    #[test]
    fn test_storage_size() {
        assert_eq!(D2::storage_size(100), 100);
        assert_eq!(D3::<5>::storage_size(100), 500);
    }
    
    #[test]
    fn test_flat_index() {
        // 2D: flat_index == cell
        assert_eq!(D2::flat_index(10, 0), 10);
        
        // 3D with 5 layers
        assert_eq!(D3::<5>::flat_index(10, 3), 53); // 10*5 + 3
        
        // Unflat
        assert_eq!(D3::<5>::unflat_index(53), (10, 3));
    }
    
    #[test]
    fn test_d3_dynamic() {
        let dim = D3Dynamic { n_layers: 10 };
        assert_eq!(dim.n_layers(), 10);
    }
}
