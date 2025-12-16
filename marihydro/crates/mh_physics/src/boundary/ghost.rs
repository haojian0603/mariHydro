// crates/mh_physics/src/boundary/ghost.rs

//! 幽灵状态计算器
//!
//! 本模块提供基于边界条件计算幽灵单元状态的功能：
//! - GhostStateCalculator: 幽灵状态计算器
//! - GhostMomentumMode: 动量镜像模式
//!
//! # 概念说明
//!
//! 幽灵单元是一种边界处理技术：
//! 1. 在边界外虚拟一个单元（幽灵单元）
//! 2. 根据边界条件设置幽灵单元的状态
//! 3. 使用内部单元和幽灵单元进行通量计算
//!
//! 这种方法的优点：
//! - 统一内部和边界的数值格式
//! - 可复用相同的通量计算函数
//! - 实现简单，易于并行化
//!
//! # 迁移说明
//!
//! 从 legacy_src/domain/boundary/ghost.rs 迁移，改进：
//! - 使用枚举替代布尔参数
//! - 支持更多边界类型
//! - 与 BoundaryManager 集成

use glam::DVec2;

use super::types::{BoundaryKind, BoundaryParams, ExternalForcing};
use crate::state::ConservedState;

// ============================================================
// 动量镜像模式
// ============================================================

/// 动量镜像模式
///
/// 控制速度分量如何镜像到幽灵单元。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GhostMomentumMode {
    /// 完全反射：切向保持，法向反向
    ///
    /// 用于无滑移固壁边界。
    #[default]
    FullReflect,

    /// 自由滑移：切向保持，法向反向但动量减半
    ///
    /// 用于自由滑移边界。
    FreeSlip,

    /// 无反射：直接复制
    ///
    /// 用于对称边界。
    NoReflect,

    /// 完全抵消：切向和法向都反向
    ///
    /// 用于无滑移边界（黏性效果）。
    FullCancel,
}

impl GhostMomentumMode {
    /// 从边界类型推断动量模式
    pub fn from_boundary_kind(kind: BoundaryKind) -> Self {
        match kind {
            BoundaryKind::Wall => Self::FullReflect,
            BoundaryKind::Symmetry => Self::NoReflect,
            BoundaryKind::OpenSea | BoundaryKind::Outflow => Self::NoReflect,
            BoundaryKind::RiverInflow => Self::NoReflect,
            BoundaryKind::Periodic => Self::NoReflect,
        }
    }
}

// ============================================================
// 幽灵状态计算器
// ============================================================

/// 幽灵状态计算器
///
/// 负责根据边界条件计算幽灵单元的状态。
///
/// # 使用方式
///
/// ```ignore
/// use mh_physics::boundary::{GhostStateCalculator, BoundaryKind, BoundaryParams};
/// use mh_physics::state::ConservedState;
/// use glam::DVec2;
///
/// let calculator = GhostStateCalculator::new(BoundaryParams::default());
/// let interior = ConservedState<S>::from_primitive(1.0, 0.5, 0.0);
/// let normal = DVec2::new(1.0, 0.0);
/// let z_bed = 0.0; // 底床高程
///
/// let ghost = calculator.compute_ghost(
///     interior,
///     BoundaryKind::Wall,
///     normal,
///     None,
///     z_bed,
/// );
/// ```
pub struct GhostStateCalculator {
    params: BoundaryParams,
}

impl GhostStateCalculator {
    /// 创建幽灵状态计算器
    pub fn new(params: BoundaryParams) -> Self {
        Self { params }
    }

    /// 从数值参数创建
    pub fn from_numerical_params(params: &crate::types::NumericalParams) -> Self {
        Self::new(BoundaryParams::from_numerical_params(params))
    }

    /// 计算幽灵单元状态
    ///
    /// # 参数
    /// - `interior`: 内部单元状态
    /// - `kind`: 边界类型
    /// - `normal`: 面外法向量（单位向量）
    /// - `external`: 外部强迫数据（用于开边界）
    /// - `z_bed`: 内部单元底床高程（用于计算水位）
    ///
    /// # 返回
    /// 幽灵单元的守恒量状态
    // ALLOW_F64: 与 ConservedState 和 DVec2 配合使用
    pub fn compute_ghost(
        &self,
        interior: ConservedState,
        kind: BoundaryKind,
        normal: DVec2,
        external: Option<&ExternalForcing>,
        z_bed: f64, // ALLOW_F64: 与 ConservedState 和 DVec2 配合
    ) -> ConservedState {
        match kind {
            BoundaryKind::Wall => self.compute_wall_ghost(interior, normal),
            BoundaryKind::Symmetry => self.compute_symmetry_ghost(interior, normal),
            BoundaryKind::OpenSea => {
                self.compute_open_sea_ghost(interior, normal, external.unwrap_or(&ExternalForcing::ZERO), z_bed)
            }
            BoundaryKind::Outflow => self.compute_outflow_ghost(interior),
            BoundaryKind::RiverInflow => {
                self.compute_inflow_ghost(interior, external.unwrap_or(&ExternalForcing::ZERO))
            }
            BoundaryKind::Periodic => {
                // 周期边界需要特殊处理，这里返回内部状态作为占位
                // 实际周期边界在网格连接阶段处理
                interior
            }
        }
    }

    /// 计算固壁边界的幽灵状态
    ///
    /// 实现无穿透条件：法向速度反向。
    fn compute_wall_ghost(&self, interior: ConservedState, normal: DVec2) -> ConservedState {
        let h = interior.h.max(self.params.h_min);

        // 计算速度
        let u = interior.hu / h;
        let v = interior.hv / h;
        let velocity = DVec2::new(u, v);

        // 分解为法向和切向分量
        let un = velocity.dot(normal);
        let ut = velocity - normal * un;

        // 幽灵速度：法向反转，切向保持
        let ghost_velocity = ut - normal * un;

        ConservedState {
            h,
            hu: h * ghost_velocity.x,
            hv: h * ghost_velocity.y,
        }
    }

    /// 计算对称边界的幽灵状态
    ///
    /// 与固壁类似，但可能有不同的动量处理。
    fn compute_symmetry_ghost(&self, interior: ConservedState, normal: DVec2) -> ConservedState {
        // 对称边界与固壁类似，法向速度反向
        self.compute_wall_ghost(interior, normal)
    }

    /// 计算开海边界的幽灵状态
    ///
    /// 使用 Flather 辐射条件。
    /// 
    /// Flather 条件基于特征分解：
    /// un* = un_ext + (c/h)(η_int - η_ext)
    /// 其中 η = h + z_bed 是水位
    // ALLOW_F64: 与 ConservedState、ExternalForcing 和 DVec2 配合使用
    fn compute_open_sea_ghost(
        &self,
        interior: ConservedState,
        normal: DVec2,
        external: &ExternalForcing,
        z_bed: f64, // ALLOW_F64: 与 ConservedState 和 DVec2 配合
    ) -> ConservedState {
        let h_int = interior.h.max(self.params.h_min);
        let c = self.params.wave_speed(h_int);

        // 内部速度
        let u_int = interior.hu / h_int;
        let v_int = interior.hv / h_int;
        let velocity_int = DVec2::new(u_int, v_int);

        // 法向速度
        let un_int = velocity_int.dot(normal);
        let un_ext = external.velocity.dot(normal);

        // Flather 条件修正法向速度
        // 正确使用水位 η = h + z_bed
        let eta_int = h_int + z_bed;
        let eta_ext = external.eta.max(self.params.h_min);
        let eta_diff = eta_int - eta_ext;
        let un_ghost = un_ext - (c / h_int) * eta_diff;

        // 切向速度保持
        let ut = velocity_int - normal * un_int;
        let ghost_velocity = ut + normal * un_ghost;

        // 幽灵水深：从外部水位减去底床高程
        // h_ghost = max(0, eta_ext - z_bed)
        let h_ghost = (external.eta - z_bed).max(self.params.h_min);

        ConservedState {
            h: h_ghost,
            hu: h_ghost * ghost_velocity.x,
            hv: h_ghost * ghost_velocity.y,
        }
    }

    /// 计算出流边界的幽灵状态
    ///
    /// 零梯度外推：直接复制内部状态。
    fn compute_outflow_ghost(&self, interior: ConservedState) -> ConservedState {
        interior
    }

    /// 计算入流边界的幽灵状态
    ///
    /// 使用外部强迫的速度和水深。
    fn compute_inflow_ghost(
        &self,
        _interior: ConservedState,
        external: &ExternalForcing,
    ) -> ConservedState {
        let h = external.eta.max(self.params.h_min);
        ConservedState {
            h,
            hu: h * external.velocity.x,
            hv: h * external.velocity.y,
        }
    }

    /// 使用指定的动量模式计算幽灵状态
    ///
    /// 更灵活的接口，允许自定义动量处理方式。
    ///
    /// # 参数
    /// - `interior`: 内部单元状态
    /// - `normal`: 面外法向量
    /// - `mode`: 动量镜像模式
    ///
    /// # 返回
    /// 幽灵单元状态
    pub fn compute_ghost_with_mode(
        &self,
        interior: ConservedState,
        normal: DVec2,
        mode: GhostMomentumMode,
    ) -> ConservedState {
        let h = interior.h.max(self.params.h_min);
        let u = interior.hu / h;
        let v = interior.hv / h;
        let velocity = DVec2::new(u, v);

        let un = velocity.dot(normal);
        let ut = velocity - normal * un;

        let ghost_velocity = match mode {
            GhostMomentumMode::FullReflect => ut - normal * un,
            GhostMomentumMode::FreeSlip => ut - normal * (un * 0.5),
            GhostMomentumMode::NoReflect => velocity,
            GhostMomentumMode::FullCancel => -velocity,
        };

        ConservedState {
            h,
            hu: h * ghost_velocity.x,
            hv: h * ghost_velocity.y,
        }
    }

    /// 批量计算幽灵状态
    ///
    /// 对性能敏感的场景，批量处理更高效。
    ///
    /// # 参数
    /// - `interiors`: 内部单元状态数组
    /// - `kinds`: 边界类型数组
    /// - `normals`: 法向量数组
    /// - `externals`: 外部强迫数组（可选）
    /// - `z_beds`: 底床高程数组
    /// - `output`: 输出数组
    pub fn compute_ghost_batch(
        &self,
        interiors: &[ConservedState],
        kinds: &[BoundaryKind],
        normals: &[DVec2],
        externals: Option<&[ExternalForcing]>,
        z_beds: &[f64],
        output: &mut [ConservedState],
    ) {
        debug_assert_eq!(interiors.len(), kinds.len());
        debug_assert_eq!(interiors.len(), normals.len());
        debug_assert_eq!(interiors.len(), z_beds.len());
        debug_assert_eq!(interiors.len(), output.len());

        let empty_forcing = ExternalForcing::ZERO;

        for i in 0..interiors.len() {
            let external = externals.map(|e| &e[i]).unwrap_or(&empty_forcing);
            output[i] = self.compute_ghost(interiors[i], kinds[i], normals[i], Some(external), z_beds[i]);
        }
    }

    /// 获取参数引用
    pub fn params(&self) -> &BoundaryParams {
        &self.params
    }
}

impl Default for GhostStateCalculator {
    fn default() -> Self {
        Self::new(BoundaryParams::default())
    }
}

// ============================================================
// 辅助函数
// ============================================================

/// 反射速度向量
///
/// 将速度向量关于法向量反射。
///
/// # 参数
/// - `velocity`: 原始速度
/// - `normal`: 反射面法向量（单位向量）
///
/// # 返回
/// 反射后的速度
#[inline]
pub fn reflect_velocity(velocity: DVec2, normal: DVec2) -> DVec2 {
    let un = velocity.dot(normal);
    velocity - 2.0 * un * normal
}

/// 分解速度为法向和切向分量
///
/// # 参数
/// - `velocity`: 速度向量
/// - `normal`: 法向量（单位向量）
///
/// # 返回
/// (法向分量标量, 切向分量向量)
#[inline]
pub fn decompose_velocity(velocity: DVec2, normal: DVec2) -> (f64, DVec2) {
    let un = velocity.dot(normal);
    let ut = velocity - normal * un;
    (un, ut)
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_wall_ghost_no_penetration() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::from_primitive(1.0, 1.0, 0.0);
        let normal = DVec2::new(1.0, 0.0);

        let ghost = calculator.compute_ghost(interior, BoundaryKind::Wall, normal, None, 0.0);

        // 水深保持
        assert!(approx_eq(ghost.h, 1.0));
        // 法向动量反向
        assert!(approx_eq(ghost.hu, -1.0));
        // 切向动量保持
        assert!(approx_eq(ghost.hv, 0.0));
    }

    #[test]
    fn test_wall_ghost_oblique() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::from_primitive(1.0, 1.0, 1.0);
        let normal = DVec2::new(1.0, 0.0);

        let ghost = calculator.compute_ghost(interior, BoundaryKind::Wall, normal, None, 0.0);

        // 法向反转，切向保持
        assert!(approx_eq(ghost.hu, -1.0));
        assert!(approx_eq(ghost.hv, 1.0));
    }

    #[test]
    fn test_outflow_ghost() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::from_primitive(1.5, 0.5, 0.3);
        let normal = DVec2::new(1.0, 0.0);

        let ghost = calculator.compute_ghost(interior, BoundaryKind::Outflow, normal, None, 0.0);

        // 出流：完全复制
        assert!(approx_eq(ghost.h, 1.5));
        assert!(approx_eq(ghost.hu, 0.75)); // 1.5 * 0.5
        assert!(approx_eq(ghost.hv, 0.45)); // 1.5 * 0.3
    }

    #[test]
    fn test_inflow_ghost() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::from_primitive(1.0, 0.0, 0.0);
        let normal = DVec2::new(-1.0, 0.0);
        let external = ExternalForcing::new(2.0, 1.0, 0.0);

        let ghost = calculator.compute_ghost(
            interior,
            BoundaryKind::RiverInflow,
            normal,
            Some(&external),
            0.0,
        );

        // 使用外部强迫
        assert!(approx_eq(ghost.h, 2.0));
        assert!(approx_eq(ghost.hu, 2.0)); // h * u = 2.0 * 1.0
        assert!(approx_eq(ghost.hv, 0.0));
    }

    #[test]
    fn test_flather_open_sea_with_z_bed() {
        // 测试 Flather 边界条件正确使用水位 η = h + z_bed
        let calculator = GhostStateCalculator::default();
        
        // 内部单元: h=1.0, z_bed=0.5, 所以 η_int = 1.5
        let interior = ConservedState::from_primitive(1.0, 0.0, 0.0);
        let normal = DVec2::new(1.0, 0.0);
        let z_bed = 0.5;
        
        // 外部强迫: η_ext = 1.5 (与内部相同)
        let external = ExternalForcing::new(1.5, 0.0, 0.0);
        
        let ghost = calculator.compute_ghost(
            interior,
            BoundaryKind::OpenSea,
            normal,
            Some(&external),
            z_bed,
        );
        
        // 当 η_int = η_ext 时，Flather 条件应该给出 un_ghost = un_ext = 0
        // 幽灵水深 h_ghost = η_ext - z_bed = 1.5 - 0.5 = 1.0
        assert!(approx_eq(ghost.h, 1.0));
        assert!(ghost.hu.abs() < 1e-9); // 速度接近零
    }

    #[test]
    fn test_ghost_momentum_modes() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::from_primitive(1.0, 1.0, 0.0);
        let normal = DVec2::new(1.0, 0.0);

        // FullReflect
        let ghost = calculator.compute_ghost_with_mode(interior, normal, GhostMomentumMode::FullReflect);
        assert!(approx_eq(ghost.hu, -1.0));

        // NoReflect
        let ghost = calculator.compute_ghost_with_mode(interior, normal, GhostMomentumMode::NoReflect);
        assert!(approx_eq(ghost.hu, 1.0));

        // FullCancel
        let ghost = calculator.compute_ghost_with_mode(interior, normal, GhostMomentumMode::FullCancel);
        assert!(approx_eq(ghost.hu, -1.0));
        assert!(approx_eq(ghost.hv, 0.0));
    }

    #[test]
    fn test_reflect_velocity() {
        let v = DVec2::new(1.0, 0.0);
        let n = DVec2::new(1.0, 0.0);
        let reflected = reflect_velocity(v, n);
        assert!(approx_eq(reflected.x, -1.0));
        assert!(approx_eq(reflected.y, 0.0));

        // 斜向入射
        let v = DVec2::new(1.0, 1.0);
        let n = DVec2::new(1.0, 0.0);
        let reflected = reflect_velocity(v, n);
        assert!(approx_eq(reflected.x, -1.0));
        assert!(approx_eq(reflected.y, 1.0));
    }

    #[test]
    fn test_decompose_velocity() {
        let v = DVec2::new(3.0, 4.0);
        let n = DVec2::new(1.0, 0.0);
        let (un, ut) = decompose_velocity(v, n);
        assert!(approx_eq(un, 3.0));
        assert!(approx_eq(ut.x, 0.0));
        assert!(approx_eq(ut.y, 4.0));
    }

    #[test]
    fn test_batch_compute() {
        let calculator = GhostStateCalculator::default();

        let interiors = vec![
            ConservedState::from_primitive(1.0, 1.0, 0.0),
            ConservedState::from_primitive(2.0, 0.0, 1.0),
        ];
        let kinds = vec![BoundaryKind::Wall, BoundaryKind::Outflow];
        let normals = vec![DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.0)];
        let z_beds = vec![0.0, 0.0];

        let mut output = vec![ConservedState::default(); 2];
        calculator.compute_ghost_batch(&interiors, &kinds, &normals, None, &z_beds, &mut output);

        // 固壁：法向反转
        assert!(approx_eq(output[0].hu, -1.0));
        // 出流：直接复制
        assert!(approx_eq(output[1].hv, 2.0)); // h * v = 2.0 * 1.0
    }

    #[test]
    fn test_momentum_mode_from_kind() {
        assert_eq!(
            GhostMomentumMode::from_boundary_kind(BoundaryKind::Wall),
            GhostMomentumMode::FullReflect
        );
        assert_eq!(
            GhostMomentumMode::from_boundary_kind(BoundaryKind::Symmetry),
            GhostMomentumMode::NoReflect
        );
        assert_eq!(
            GhostMomentumMode::from_boundary_kind(BoundaryKind::OpenSea),
            GhostMomentumMode::NoReflect
        );
    }
}
