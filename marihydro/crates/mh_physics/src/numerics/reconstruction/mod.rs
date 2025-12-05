//! # MUSCL 重构模块
//!
//! 实现 MUSCL (Monotonic Upstream-centered Scheme for Conservation Laws) 重构方案，
//! 将一阶单元中心值扩展到二阶精度的面值重构。
//!
//! ## 重构过程
//!
//! 1. 计算单元梯度 (Green-Gauss 或 Least-Squares)
//! 2. 应用限制器 (Venkatakrishnan, Barth-Jespersen, Minmod)
//! 3. 在面上进行线性外推
//!
//! ## 使用方式
//!
//! ```ignore
//! use mh_physics::numerics::reconstruction::{MusclReconstructor, MusclConfig};
//!
//! let config = MusclConfig::default();
//! let reconstructor = MusclReconstructor::new(config, mesh);
//!
//! // 计算梯度
//! reconstructor.compute_gradients(&cell_values);
//!
//! // 获取面值
//! let (left, right) = reconstructor.reconstruct_face(face_id);
//! ```
//!
//! ## 参考文献
//!
//! van Leer, B. (1979). "Towards the ultimate conservative difference 
//! scheme. V. A second-order sequel to Godunov's method". 
//! Journal of Computational Physics.

mod traits;
mod muscl;
mod config;

pub use traits::{Reconstructor, ReconstructedState};
pub use muscl::MusclReconstructor;
pub use config::{MusclConfig, GradientType};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_exports() {
        // 验证模块导出正确
        let config = MusclConfig::default();
        assert!(config.second_order);
    }
}
