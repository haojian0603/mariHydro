// crates/mh_physics/src/boundary/ghost.rs

//! 幽灵状态计算器
//!
//! 本模块提供基于边界条件计算幽灵单元状态的功能：
//! - GhostStateCalculator: 幽灵状态计算器
//! - GhostMomentumMode: 动量镜像模式


use super::types::{BoundaryKind, BoundaryParams, ExternalForcing};
use crate::state::ConservedState;
use crate::types::NumericalParams;

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
///
/// let calculator = GhostStateCalculator::new(BoundaryParams::default());
/// let interior = ConservedState::<f64>::from_primitive(1.0, 0.5, 0.0);
/// let normal = (1.0, 0.0);  // 使用元组而非 DVec2
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
    pub fn from_numerical_params(params: &NumericalParams<f64>) -> Self {
        Self::new(BoundaryParams::from_numerical_params(params))
    }

    /// 计算幽灵单元状态
    ///
    /// # 参数
    /// - `interior`: 内部单元状态
    /// - `kind`: 边界类型
    /// - `normal`: 面外法向量（单位向量）(nx, ny)
    /// - `external`: 外部强迫数据（用于开边界）
    /// - `z_bed`: 内部单元底床高程（用于计算水位）
    ///
    /// # 返回
    /// 幽灵单元的守恒量状态
    // ALLOW_F64: 与 ConservedState 和速度元组配合使用
    pub fn compute_ghost(
        &self,
        interior: ConservedState<f64>,
        kind: BoundaryKind,
        normal: (f64, f64),
        external: Option<&ExternalForcing>,
        z_bed: f64, 
    ) -> ConservedState::<f64> {
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
    fn compute_wall_ghost(&self, interior: ConservedState<f64>, normal: (f64, f64)) -> ConservedState::<f64> {
        let h = interior.h.max(self.params.h_min);

        // 计算速度
        let u = interior.hu / h;
        let v = interior.hv / h;

        // 分解为法向和切向分量 (dot product)
        let un = u * normal.0 + v * normal.1;
        // 切向分量: ut = velocity - normal * un
        let ut_x = u - normal.0 * un;
        let ut_y = v - normal.1 * un;

        // 幽灵速度：法向反转，切向保持
        let ghost_u = ut_x - normal.0 * un;
        let ghost_v = ut_y - normal.1 * un;

        ConservedState::<f64> {
            h,
            hu: h * ghost_u,
            hv: h * ghost_v,
        }
    }

    /// 计算对称边界的幽灵状态
    ///
    /// 与固壁类似，但可能有不同的动量处理。
    fn compute_symmetry_ghost(&self, interior: ConservedState<f64>, normal: (f64, f64)) -> ConservedState::<f64> {
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
    fn compute_open_sea_ghost(
        &self,
        interior: ConservedState<f64>,
        normal: (f64, f64),
        external: &ExternalForcing,
        z_bed: f64,
    ) -> ConservedState::<f64> {
        let h_int = interior.h.max(self.params.h_min);
        let c = self.params.wave_speed(h_int);

        // 内部速度
        let u_int = interior.hu / h_int;
        let v_int = interior.hv / h_int;

        // 法向速度 (dot product)
        let un_int = u_int * normal.0 + v_int * normal.1;
        let un_ext = external.velocity.0 * normal.0 + external.velocity.1 * normal.1;

        // Flather 条件修正法向速度
        // 正确使用水位 η = h + z_bed
        let eta_int = h_int + z_bed;
        let eta_ext = external.eta.max(self.params.h_min);
        let eta_diff = eta_int - eta_ext;
        let un_ghost = un_ext - (c / h_int) * eta_diff;

        // 切向速度保持: ut = velocity - normal * un
        let ut_x = u_int - normal.0 * un_int;
        let ut_y = v_int - normal.1 * un_int;
        // ghost_velocity = ut + normal * un_ghost
        let ghost_u = ut_x + normal.0 * un_ghost;
        let ghost_v = ut_y + normal.1 * un_ghost;

        // 幽灵水深：从外部水位减去底床高程
        // h_ghost = max(0, eta_ext - z_bed)
        let h_ghost = (external.eta - z_bed).max(self.params.h_min);

        ConservedState::<f64> {
            h: h_ghost,
            hu: h_ghost * ghost_u,
            hv: h_ghost * ghost_v,
        }
    }

    /// 计算出流边界的幽灵状态
    ///
    /// 零梯度外推：直接复制内部状态。
    fn compute_outflow_ghost(&self, interior: ConservedState<f64>) -> ConservedState::<f64> {
        interior
    }

    /// 计算入流边界的幽灵状态
    ///
    /// 使用外部强迫的速度和水深。
    fn compute_inflow_ghost(
        &self,
        _interior: ConservedState<f64>,
        external: &ExternalForcing,
    ) -> ConservedState::<f64> {
        let h = external.eta.max(self.params.h_min);
        ConservedState::<f64> {
            h,
            hu: h * external.velocity.0,
            hv: h * external.velocity.1,
        }
    }

    /// 使用指定的动量模式计算幽灵状态
    ///
    /// 更灵活的接口，允许自定义动量处理方式。
    ///
    /// # 参数
    /// - `interior`: 内部单元状态
    /// - `normal`: 面外法向量 (nx, ny)
    /// - `mode`: 动量镜像模式
    ///
    /// # 返回
    /// 幽灵单元状态
    pub fn compute_ghost_with_mode(
        &self,
        interior: ConservedState<f64>,
        normal: (f64, f64),
        mode: GhostMomentumMode,
    ) -> ConservedState::<f64> {
        let h = interior.h.max(self.params.h_min);
        let u = interior.hu / h;
        let v = interior.hv / h;

        // dot product
        let un = u * normal.0 + v * normal.1;
        // ut = velocity - normal * un
        let ut_x = u - normal.0 * un;
        let ut_y = v - normal.1 * un;

        let (ghost_u, ghost_v) = match mode {
            GhostMomentumMode::FullReflect => {
                // ut - normal * un
                (ut_x - normal.0 * un, ut_y - normal.1 * un)
            }
            GhostMomentumMode::FreeSlip => {
                // ut - normal * (un * 0.5)
                (ut_x - normal.0 * (un * 0.5), ut_y - normal.1 * (un * 0.5))
            }
            GhostMomentumMode::NoReflect => (u, v),
            GhostMomentumMode::FullCancel => (-u, -v),
        };

        ConservedState::<f64> {
            h,
            hu: h * ghost_u,
            hv: h * ghost_v,
        }
    }

    /// 批量计算幽灵状态
    ///
    /// 对性能敏感的场景，批量处理更高效。
    ///
    /// # 参数
    /// - `interiors`: 内部单元状态数组
    /// - `kinds`: 边界类型数组
    /// - `normals`: 法向量数组 (nx, ny)
    /// - `externals`: 外部强迫数组（可选）
    /// - `z_beds`: 底床高程数组
    /// - `output`: 输出数组
    pub fn compute_ghost_batch(
        &self,
        interiors: &[ConservedState<f64>],
        kinds: &[BoundaryKind],
        normals: &[(f64, f64)],
        externals: Option<&[ExternalForcing]>,
        z_beds: &[f64],
        output: &mut [ConservedState<f64>],
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
/// - `velocity`: 原始速度 (u, v)
/// - `normal`: 反射面法向量（单位向量）(nx, ny)
///
/// # 返回
/// 反射后的速度 (u, v)
#[inline]
pub fn reflect_velocity(velocity: (f64, f64), normal: (f64, f64)) -> (f64, f64) {
    // dot product
    let un = velocity.0 * normal.0 + velocity.1 * normal.1;
    // velocity - 2.0 * un * normal
    (velocity.0 - 2.0 * un * normal.0, velocity.1 - 2.0 * un * normal.1)
}

/// 分解速度为法向和切向分量
///
/// # 参数
/// - `velocity`: 速度向量 (u, v)
/// - `normal`: 法向量（单位向量）(nx, ny)
///
/// # 返回
/// (法向分量标量, 切向分量向量 (ut_x, ut_y))
#[inline]
pub fn decompose_velocity(velocity: (f64, f64), normal: (f64, f64)) -> (f64, (f64, f64)) {
    // dot product
    let un = velocity.0 * normal.0 + velocity.1 * normal.1;
    // ut = velocity - normal * un
    let ut = (velocity.0 - normal.0 * un, velocity.1 - normal.1 * un);
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
        let interior = ConservedState::<f64>::from_primitive(1.0, 1.0, 0.0);
        let normal = (1.0, 0.0);

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
        let interior = ConservedState::<f64>::from_primitive(1.0, 1.0, 1.0);
        let normal = (1.0, 0.0);

        let ghost = calculator.compute_ghost(interior, BoundaryKind::Wall, normal, None, 0.0);

        // 法向反转，切向保持
        assert!(approx_eq(ghost.hu, -1.0));
        assert!(approx_eq(ghost.hv, 1.0));
    }

    #[test]
    fn test_outflow_ghost() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::<f64>::from_primitive(1.5, 0.5, 0.3);
        let normal = (1.0, 0.0);

        let ghost = calculator.compute_ghost(interior, BoundaryKind::Outflow, normal, None, 0.0);

        // 出流：完全复制
        assert!(approx_eq(ghost.h, 1.5));
        assert!(approx_eq(ghost.hu, 0.75)); // 1.5 * 0.5
        assert!(approx_eq(ghost.hv, 0.45)); // 1.5 * 0.3
    }

    #[test]
    fn test_inflow_ghost() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::<f64>::from_primitive(1.0, 0.0, 0.0);
        let normal = (-1.0, 0.0);
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
        let interior = ConservedState::<f64>::from_primitive(1.0, 0.0, 0.0);
        let normal = (1.0, 0.0);
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
        let interior = ConservedState::<f64>::from_primitive(1.0, 1.0, 0.0);
        let normal = (1.0, 0.0);

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
        let v = (1.0, 0.0);
        let n = (1.0, 0.0);
        let reflected = reflect_velocity(v, n);
        assert!(approx_eq(reflected.0, -1.0));
        assert!(approx_eq(reflected.1, 0.0));

        // 斜向入射
        let v = (1.0, 1.0);
        let n = (1.0, 0.0);
        let reflected = reflect_velocity(v, n);
        assert!(approx_eq(reflected.0, -1.0));
        assert!(approx_eq(reflected.1, 1.0));
    }

    #[test]
    fn test_decompose_velocity() {
        let v = (3.0, 4.0);
        let n = (1.0, 0.0);
        let (un, ut) = decompose_velocity(v, n);
        assert!(approx_eq(un, 3.0));
        assert!(approx_eq(ut.0, 0.0));
        assert!(approx_eq(ut.1, 4.0));
    }

    #[test]
    fn test_batch_compute() {
        let calculator = GhostStateCalculator::default();

        let interiors = vec![
            ConservedState::<f64>::from_primitive(1.0, 1.0, 0.0),
            ConservedState::<f64>::from_primitive(2.0, 0.0, 1.0),
        ];
        let kinds = vec![BoundaryKind::Wall, BoundaryKind::Outflow];
        let normals = vec![(1.0, 0.0), (0.0, 1.0)];
        let z_beds = vec![0.0, 0.0];

        let mut output = vec![ConservedState::<f64>::default(); 2];
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
