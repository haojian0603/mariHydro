// crates/mh_physics/src/boundary/ghost.rs

//! 幽灵状态计算器
//!
//! 本模块提供基于边界条件计算幽灵单元状态的功能：
//! - GhostStateCalculator: 幽灵状态计算器
//! - GhostMomentumMode: 动量镜像模式


use super::types::{BoundaryKind, BoundaryParams, ExternalForcing};
use crate::state::ConservedState;
use crate::types::NumericalParams;
use mh_runtime::{Backend, RuntimeScalar, Vector2D};
use num_traits::Float;

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
/// type B = mh_runtime::CpuBackend<f64>;
/// let calculator = GhostStateCalculator::new(BoundaryParams::default());
/// let interior = ConservedState::<f64>::from_primitive(1.0, 0.5, 0.0);
/// let normal = B::vec2_new(1.0, 0.0);  // 使用 backend 向量类型
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

    /// 计算幽灵单元状态（Backend 泛型）
    pub fn compute_ghost<B: Backend>(
        &self,
        interior: ConservedState<B::Scalar>,
        kind: BoundaryKind,
        normal: B::Vector2D,
        external: Option<&ExternalForcing>,
        z_bed: B::Scalar,
    ) -> ConservedState<B::Scalar>
    where
        B::Scalar: RuntimeScalar,
    {
        match kind {
            BoundaryKind::Wall => self.compute_wall_ghost::<B>(interior, normal),
            BoundaryKind::Symmetry => self.compute_symmetry_ghost::<B>(interior, normal),
            BoundaryKind::OpenSea => self.compute_open_sea_ghost::<B>(
                interior,
                normal,
                external.unwrap_or(&ExternalForcing::ZERO),
                z_bed,
            ),
            BoundaryKind::Outflow => self.compute_outflow_ghost::<B>(interior),
            BoundaryKind::RiverInflow => {
                self.compute_inflow_ghost::<B>(interior, external.unwrap_or(&ExternalForcing::ZERO))
            }
            BoundaryKind::Periodic => interior,
        }
    }

    /// 计算固壁边界的幽灵状态（无穿透，法向反射）
    fn compute_wall_ghost<B: Backend>(
        &self,
        interior: ConservedState<B::Scalar>,
        normal: B::Vector2D,
    ) -> ConservedState<B::Scalar>
    where
        B::Scalar: RuntimeScalar,
    {
        let h_min = B::Scalar::from_config(self.params.h_min).unwrap_or(B::Scalar::ZERO);
        let h = interior.h.max(h_min);

        let u = if h > B::Scalar::ZERO { interior.hu / h } else { B::Scalar::ZERO };
        let v = if h > B::Scalar::ZERO { interior.hv / h } else { B::Scalar::ZERO };

        let nx = normal.x();
        let ny = normal.y();
        let un = u * nx + v * ny;
        let ut_x = u - nx * un;
        let ut_y = v - ny * un;

        let ghost_u = ut_x - nx * un;
        let ghost_v = ut_y - ny * un;

        ConservedState {
            h,
            hu: h * ghost_u,
            hv: h * ghost_v,
        }
    }

    /// 对称边界：与固壁同处理
    fn compute_symmetry_ghost<B: Backend>(
        &self,
        interior: ConservedState<B::Scalar>,
        normal: B::Vector2D,
    ) -> ConservedState<B::Scalar>
    where
        B::Scalar: RuntimeScalar,
    {
        self.compute_wall_ghost::<B>(interior, normal)
    }

    /// 开海边界（Flather 辐射，含干湿处理）
    fn compute_open_sea_ghost<B: Backend>(
        &self,
        interior: ConservedState<B::Scalar>,
        normal: B::Vector2D,
        external: &ExternalForcing,
        z_bed: B::Scalar,
    ) -> ConservedState<B::Scalar>
    where
        B::Scalar: RuntimeScalar,
    {
        let h_min = B::Scalar::from_config(self.params.h_min).unwrap_or(B::Scalar::ZERO);
        let g = B::Scalar::from_config(self.params.gravity).unwrap_or(B::Scalar::ONE);

        let h_int = interior.h.max(h_min);
        if h_int <= h_min {
            return ConservedState { h: h_min, hu: B::Scalar::ZERO, hv: B::Scalar::ZERO };
        }

        let u_int = interior.hu / h_int;
        let v_int = interior.hv / h_int;
        let nx = normal.x();
        let ny = normal.y();
        let un_int = u_int * nx + v_int * ny;

        let u_ext = B::Scalar::from_config(external.velocity.0).unwrap_or(B::Scalar::ZERO);
        let v_ext = B::Scalar::from_config(external.velocity.1).unwrap_or(B::Scalar::ZERO);
        let un_ext = u_ext * nx + v_ext * ny;

        let eta_ext = B::Scalar::from_config(external.eta).unwrap_or(B::Scalar::ZERO);
        let h_ext = (eta_ext - z_bed).max(h_min);
        let c_int = (g * h_int).sqrt();
        let _c_ext = (g * h_ext).sqrt();

        let lambda_out = un_int - c_int;
        let un_ghost = if lambda_out > B::Scalar::ZERO {
            // 超临界/流出，纯辐射
            un_int - (c_int / h_int) * (h_int - h_ext)
        } else {
            // 亚临界：Flather 条件，融合外部速度与水位
            un_ext + (c_int / h_int) * (h_int - h_ext)
        };

        let ut_int = -u_int * ny + v_int * nx; // 切向分量标量
        let t = B::vec2_new(-ny, nx);
        let vel_ghost = B::vec2_add(
            &B::vec2_scale(&normal, un_ghost),
            &B::vec2_scale(&t, ut_int),
        );

        let ghost_u = vel_ghost.x();
        let ghost_v = vel_ghost.y();
        ConservedState {
            h: h_ext,
            hu: h_ext * ghost_u,
            hv: h_ext * ghost_v,
        }
    }

    /// 出流：零梯度外推
    fn compute_outflow_ghost<B: Backend>(
        &self,
        interior: ConservedState<B::Scalar>,
    ) -> ConservedState<B::Scalar> {
        interior
    }

    /// 入流：使用外部水位/速度
    fn compute_inflow_ghost<B: Backend>(
        &self,
        _interior: ConservedState<B::Scalar>,
        external: &ExternalForcing,
    ) -> ConservedState<B::Scalar>
    where
        B::Scalar: RuntimeScalar,
    {
        let h_min = B::Scalar::from_config(self.params.h_min).unwrap_or(B::Scalar::ZERO);
        let h = B::Scalar::from_config(external.eta).unwrap_or(B::Scalar::ZERO).max(h_min);
        let u = B::Scalar::from_config(external.velocity.0).unwrap_or(B::Scalar::ZERO);
        let v = B::Scalar::from_config(external.velocity.1).unwrap_or(B::Scalar::ZERO);
        ConservedState { h, hu: h * u, hv: h * v }
    }

    /// 使用指定的动量模式计算幽灵状态
    pub fn compute_ghost_with_mode<B: Backend>(
        &self,
        interior: ConservedState<B::Scalar>,
        normal: B::Vector2D,
        mode: GhostMomentumMode,
    ) -> ConservedState<B::Scalar>
    where
        B::Scalar: RuntimeScalar,
    {
        let h_min = B::Scalar::from_config(self.params.h_min).unwrap_or(B::Scalar::ZERO);
        let h = interior.h.max(h_min);
        let u = if h > B::Scalar::ZERO { interior.hu / h } else { B::Scalar::ZERO };
        let v = if h > B::Scalar::ZERO { interior.hv / h } else { B::Scalar::ZERO };

        let nx = normal.x();
        let ny = normal.y();
        let un = u * nx + v * ny;
        let ut_x = u - nx * un;
        let ut_y = v - ny * un;

        let (ghost_u, ghost_v) = match mode {
            GhostMomentumMode::FullReflect => (ut_x - nx * un, ut_y - ny * un),
            GhostMomentumMode::FreeSlip => {
                let half = B::Scalar::from_config(0.5).unwrap_or(B::Scalar::HALF);
                (ut_x - nx * (un * half), ut_y - ny * (un * half))
            }
            GhostMomentumMode::NoReflect => (u, v),
            GhostMomentumMode::FullCancel => (-u, -v),
        };

        ConservedState { h, hu: h * ghost_u, hv: h * ghost_v }
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
    pub fn compute_ghost_batch<B: Backend>(
        &self,
        interiors: &[ConservedState<B::Scalar>],
        kinds: &[BoundaryKind],
        normals: &[B::Vector2D],
        externals: Option<&[ExternalForcing]>,
        z_beds: &[B::Scalar],
        output: &mut [ConservedState<B::Scalar>],
    ) where
        B::Scalar: RuntimeScalar,
    {
        debug_assert_eq!(interiors.len(), kinds.len());
        debug_assert_eq!(interiors.len(), normals.len());
        debug_assert_eq!(interiors.len(), z_beds.len());
        debug_assert_eq!(interiors.len(), output.len());

        let empty_forcing = ExternalForcing::ZERO;

        for i in 0..interiors.len() {
            let external = externals.map(|e| &e[i]).unwrap_or(&empty_forcing);
            output[i] = self.compute_ghost::<B>(interiors[i], kinds[i], normals[i], Some(external), z_beds[i]);
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
pub fn reflect_velocity<B: Backend>(velocity: (B::Scalar, B::Scalar), normal: B::Vector2D) -> (B::Scalar, B::Scalar)
where
    B::Scalar: RuntimeScalar,
{
    let nx = normal.x();
    let ny = normal.y();
    let two = B::Scalar::from_config(2.0).unwrap_or(B::Scalar::ONE + B::Scalar::ONE);
    let un = velocity.0 * nx + velocity.1 * ny;
    (velocity.0 - two * un * nx, velocity.1 - two * un * ny)
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
pub fn decompose_velocity<B: Backend>(velocity: (B::Scalar, B::Scalar), normal: B::Vector2D) -> (B::Scalar, (B::Scalar, B::Scalar))
where
    B::Scalar: RuntimeScalar,
{
    let nx = normal.x();
    let ny = normal.y();
    let un = velocity.0 * nx + velocity.1 * ny;
    let ut = (velocity.0 - nx * un, velocity.1 - ny * un);
    (un, ut)
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    type B = CpuBackend<f64>;

    #[test]
    fn test_wall_ghost_no_penetration() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::<f64>::from_primitive(1.0, 1.0, 0.0);
        let normal = B::vec2_new(1.0, 0.0);

        let ghost = calculator.compute_ghost::<B>(interior, BoundaryKind::Wall, normal, None, 0.0);

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
        let normal = B::vec2_new(1.0, 0.0);

        let ghost = calculator.compute_ghost::<B>(interior, BoundaryKind::Wall, normal, None, 0.0);

        // 法向反转，切向保持
        assert!(approx_eq(ghost.hu, -1.0));
        assert!(approx_eq(ghost.hv, 1.0));
    }

    #[test]
    fn test_outflow_ghost() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::<f64>::from_primitive(1.5, 0.5, 0.3);
        let normal = B::vec2_new(1.0, 0.0);

        let ghost = calculator.compute_ghost::<B>(interior, BoundaryKind::Outflow, normal, None, 0.0);

        // 出流：完全复制
        assert!(approx_eq(ghost.h, 1.5));
        assert!(approx_eq(ghost.hu, 0.75)); // 1.5 * 0.5
        assert!(approx_eq(ghost.hv, 0.45)); // 1.5 * 0.3
    }

    #[test]
    fn test_inflow_ghost() {
        let calculator = GhostStateCalculator::default();
        let interior = ConservedState::<f64>::from_primitive(1.0, 0.0, 0.0);
        let normal = B::vec2_new(-1.0, 0.0);
        let external = ExternalForcing::new(2.0, 1.0, 0.0);

        let ghost = calculator.compute_ghost::<B>(
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
        let normal = B::vec2_new(1.0, 0.0);
        let z_bed = 0.5;
        
        // 外部强迫: η_ext = 1.5 (与内部相同)
        let external = ExternalForcing::new(1.5, 0.0, 0.0);
        
        let ghost = calculator.compute_ghost::<B>(
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
        let normal = B::vec2_new(1.0, 0.0);

        // FullReflect
        let ghost = calculator.compute_ghost_with_mode::<B>(interior, normal, GhostMomentumMode::FullReflect);
        assert!(approx_eq(ghost.hu, -1.0));

        // NoReflect
        let ghost = calculator.compute_ghost_with_mode::<B>(interior, normal, GhostMomentumMode::NoReflect);
        assert!(approx_eq(ghost.hu, 1.0));

        // FullCancel
        let ghost = calculator.compute_ghost_with_mode::<B>(interior, normal, GhostMomentumMode::FullCancel);
        assert!(approx_eq(ghost.hu, -1.0));
        assert!(approx_eq(ghost.hv, 0.0));
    }

    #[test]
    fn test_reflect_velocity() {
        let v = (1.0, 0.0);
        let n = B::vec2_new(1.0, 0.0);
        let reflected = reflect_velocity::<B>(v, n);
        assert!(approx_eq(reflected.0, -1.0));
        assert!(approx_eq(reflected.1, 0.0));

        // 斜向入射
        let v = (1.0, 1.0);
        let n = B::vec2_new(1.0, 0.0);
        let reflected = reflect_velocity::<B>(v, n);
        assert!(approx_eq(reflected.0, -1.0));
        assert!(approx_eq(reflected.1, 1.0));
    }

    #[test]
    fn test_decompose_velocity() {
        let v = (3.0, 4.0);
        let n = B::vec2_new(1.0, 0.0);
        let (un, ut) = decompose_velocity::<B>(v, n);
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
        let normals = vec![B::vec2_new(1.0, 0.0), B::vec2_new(0.0, 1.0)];
        let z_beds = vec![0.0, 0.0];

        let mut output = vec![ConservedState::<f64>::default(); 2];
        calculator.compute_ghost_batch::<B>(&interiors, &kinds, &normals, None, &z_beds, &mut output);

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
