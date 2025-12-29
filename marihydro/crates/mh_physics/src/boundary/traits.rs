// crates/mh_physics/src/boundary/traits.rs

//! 边界条件 Trait 定义
//!
//! 本模块提供了基于 trait 的边界条件抽象，支持：
//! - 静态分发的内置边界类型
//! - 动态分发的用户自定义边界类型
//! - 类型擦除的边界条件容器
//! - 边界条件注册表
//!
//! # 设计理念
//!
//! 使用 trait 而非 enum 表示边界条件的优势：
//! 1. **可扩展性**: 用户可以实现自定义边界类型而无需修改库代码
//! 2. **多态性**: 支持运行时动态选择边界类型
//! 3. **关注点分离**: 每种边界类型的逻辑独立封装
//! 4. **测试友好**: 易于为边界条件创建 mock
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::boundary::traits::*;
//!
//! // 使用内置边界类型
//! let reflective = Reflective;
//! let ghost = reflective.apply(&interior_state, normal, time);
//!
//! // 使用动态边界类型
//! let boundary: DynBoundaryCondition<f64> = Box::new(Reflective);
//! let ghost = boundary.apply(&interior_state, normal, time);
//! ```

use mh_runtime::RuntimeScalar;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================
// 核心类型
// ============================================================

/// 单元状态
///
/// 表示浅水方程中一个单元的守恒量状态。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellState<S> {
    /// 水深 [m]
    pub h: S,
    /// x 方向速度 [m/s]
    pub u: S,
    /// y 方向速度 [m/s]
    pub v: S,
}

impl<S: RuntimeScalar> CellState<S> {
    /// 创建新的单元状态
    #[inline]
    pub fn new(h: S, u: S, v: S) -> Self {
        Self { h, u, v }
    }

    /// 创建静止状态（零速度）
    #[inline]
    pub fn still(h: S) -> Self {
        Self {
            h,
            u: S::zero(),
            v: S::zero(),
        }
    }

    /// 创建干单元状态
    #[inline]
    pub fn dry() -> Self {
        Self {
            h: S::zero(),
            u: S::zero(),
            v: S::zero(),
        }
    }

    /// 计算法向速度分量
    #[inline]
    pub fn normal_velocity(&self, normal: [S; 2]) -> S {
        self.u * normal[0] + self.v * normal[1]
    }

    /// 计算切向速度分量
    #[inline]
    pub fn tangent_velocity(&self, normal: [S; 2]) -> S {
        // 切向 = (-ny, nx)
        -self.u * normal[1] + self.v * normal[0]
    }
}

impl<S: RuntimeScalar> Default for CellState<S> {
    fn default() -> Self {
        Self::dry()
    }
}

// ============================================================
// 边界条件 Trait
// ============================================================

/// 边界条件核心 trait
///
/// 定义边界条件必须实现的接口。边界条件负责根据内部单元状态
/// 和边界法向量计算虚拟单元（ghost cell）的状态。
///
/// # 实现要求
///
/// - `apply` 方法必须是纯函数（无副作用）
/// - 实现必须是 `Send + Sync` 以支持并行计算
/// - `is_time_dependent` 应准确反映边界是否依赖时间
///
/// # 示例实现
///
/// ```ignore
/// use mh_physics::boundary::traits::*;
/// use mh_runtime::RuntimeScalar;
///
/// struct MyCustomBoundary<S> {
///     factor: S,
/// }
///
/// impl<S: RuntimeScalar> BoundaryConditionTrait<S> for MyCustomBoundary<S> {
///     fn name(&self) -> &'static str {
///         "MyCustom"
///     }
///
///     fn apply(&self, interior: &CellState<S>, _normal: [S; 2], _time: S) -> CellState<S> {
///         CellState {
///             h: interior.h * self.factor,
///             u: interior.u,
///             v: interior.v,
///         }
///     }
/// }
/// ```
pub trait BoundaryConditionTrait<S: RuntimeScalar>: Send + Sync {
    /// 边界类型名称
    ///
    /// 返回边界类型的唯一标识符，用于日志记录和调试。
    fn name(&self) -> &'static str;

    /// 应用边界条件，返回虚拟单元状态
    ///
    /// # 参数
    /// - `interior`: 内部单元的状态
    /// - `normal`: 边界面的外法向量 `[nx, ny]`（指向计算域外部）
    /// - `time`: 当前模拟时间 [s]
    ///
    /// # 返回
    /// 虚拟单元的状态，用于 Riemann 求解器计算边界通量
    fn apply(&self, interior: &CellState<S>, normal: [S; 2], time: S) -> CellState<S>;

    /// 是否需要时间更新
    ///
    /// 如果边界条件依赖于时间（如潮汐边界），返回 `true`。
    /// 优化器可以利用此信息缓存时间无关边界的结果。
    fn is_time_dependent(&self) -> bool {
        false
    }

    /// 边界条件描述（用于调试）
    fn description(&self) -> String {
        self.name().to_string()
    }
}

// ============================================================
// 内置边界类型实现
// ============================================================

/// 反射边界（固壁）
///
/// 法向速度反转，切向速度保持不变。
/// 适用于不可穿透的固体边界。
///
/// # 物理意义
///
/// - 质量通量为零（无穿透）
/// - 动量法向分量反射
/// - 能量守恒
#[derive(Debug, Clone, Copy, Default)]
pub struct Reflective;

impl<S: RuntimeScalar> BoundaryConditionTrait<S> for Reflective {
    fn name(&self) -> &'static str {
        "Reflective"
    }

    fn apply(&self, interior: &CellState<S>, normal: [S; 2], _time: S) -> CellState<S> {
        // 计算法向速度: u_n = u*nx + v*ny
        let un = interior.u * normal[0] + interior.v * normal[1];
        // 反射: u_ghost = u - 2*u_n*n
        CellState {
            h: interior.h,
            u: interior.u - S::from_f64(2.0).unwrap_or(S::one() + S::one()) * un * normal[0],
            v: interior.v - S::from_f64(2.0).unwrap_or(S::one() + S::one()) * un * normal[1],
        }
    }
}

/// 透射边界（自由出流）
///
/// 直接复制内部状态，允许波动自由传出。
/// 适用于开放边界，但可能产生非物理反射。
///
/// # 注意
///
/// 简单透射边界在亚临界流动中可能不稳定，
/// 对于精确模拟建议使用 Flather 辐射边界。
#[derive(Debug, Clone, Copy, Default)]
pub struct Transmissive;

impl<S: RuntimeScalar> BoundaryConditionTrait<S> for Transmissive {
    fn name(&self) -> &'static str {
        "Transmissive"
    }

    fn apply(&self, interior: &CellState<S>, _normal: [S; 2], _time: S) -> CellState<S> {
        *interior
    }
}

/// 无滑移壁面边界
///
/// 法向和切向速度都设为零，适用于粘性流动中的固壁。
#[derive(Debug, Clone, Copy, Default)]
pub struct NoSlipWall;

impl<S: RuntimeScalar> BoundaryConditionTrait<S> for NoSlipWall {
    fn name(&self) -> &'static str {
        "NoSlipWall"
    }

    fn apply(&self, interior: &CellState<S>, _normal: [S; 2], _time: S) -> CellState<S> {
        CellState {
            h: interior.h,
            u: S::zero(),
            v: S::zero(),
        }
    }
}

/// 滑移壁面边界
///
/// 法向速度反转，切向速度保持。
/// 与反射边界等价，但语义更清晰。
#[derive(Debug, Clone, Copy, Default)]
pub struct SlipWall;

impl<S: RuntimeScalar> BoundaryConditionTrait<S> for SlipWall {
    fn name(&self) -> &'static str {
        "SlipWall"
    }

    fn apply(&self, interior: &CellState<S>, normal: [S; 2], _time: S) -> CellState<S> {
        let un = interior.u * normal[0] + interior.v * normal[1];
        let two = S::from_f64(2.0).unwrap_or(S::one() + S::one());
        CellState {
            h: interior.h,
            u: interior.u - two * un * normal[0],
            v: interior.v - two * un * normal[1],
        }
    }
}

/// 入流边界
///
/// 指定固定的水深和速度。
/// 适用于河流入口等已知流量的边界。
#[derive(Debug, Clone, Copy)]
pub struct Inflow<S> {
    /// 指定水深 [m]
    pub h: S,
    /// 指定 x 方向速度 [m/s]
    pub u: S,
    /// 指定 y 方向速度 [m/s]
    pub v: S,
}

impl<S: RuntimeScalar> Inflow<S> {
    /// 创建新的入流边界
    pub fn new(h: S, u: S, v: S) -> Self {
        Self { h, u, v }
    }

    /// 从流量和断面宽度创建入流边界
    ///
    /// # 参数
    /// - `h`: 水深 [m]
    /// - `discharge`: 流量 [m³/s]
    /// - `width`: 断面宽度 [m]
    /// - `normal`: 边界法向量
    pub fn from_discharge(h: S, discharge: S, width: S, normal: [S; 2]) -> Self {
        // Q = h * width * u_n => u_n = Q / (h * width)
        let h_safe = if h > S::epsilon() { h } else { S::epsilon() };
        let width_safe = if width > S::epsilon() { width } else { S::epsilon() };
        let u_n = discharge / (h_safe * width_safe);
        // 速度指向内部（与法向相反）
        Self {
            h,
            u: -u_n * normal[0],
            v: -u_n * normal[1],
        }
    }
}

impl<S: RuntimeScalar> BoundaryConditionTrait<S> for Inflow<S> {
    fn name(&self) -> &'static str {
        "Inflow"
    }

    fn apply(&self, _interior: &CellState<S>, _normal: [S; 2], _time: S) -> CellState<S> {
        CellState {
            h: self.h,
            u: self.u,
            v: self.v,
        }
    }
}

/// 固定水位出流边界
///
/// 指定水位，速度从内部外推。
/// 适用于下游水位已知的情况。
#[derive(Debug, Clone, Copy)]
pub struct FixedLevelOutflow<S> {
    /// 指定水位 [m]
    pub level: S,
    /// 床面高程 [m]
    pub bed_level: S,
}

impl<S: RuntimeScalar> FixedLevelOutflow<S> {
    /// 创建新的固定水位出流边界
    pub fn new(level: S, bed_level: S) -> Self {
        Self { level, bed_level }
    }
}

impl<S: RuntimeScalar> BoundaryConditionTrait<S> for FixedLevelOutflow<S> {
    fn name(&self) -> &'static str {
        "FixedLevelOutflow"
    }

    fn apply(&self, interior: &CellState<S>, _normal: [S; 2], _time: S) -> CellState<S> {
        // 水深 = 水位 - 床面高程
        let h = (self.level - self.bed_level).max(S::zero());
        CellState {
            h,
            u: interior.u,
            v: interior.v,
        }
    }
}

/// 潮汐边界
///
/// 水位随时间正弦变化，模拟潮汐效应。
#[derive(Debug, Clone, Copy)]
pub struct TidalLevel<S> {
    /// 基准水位 [m]
    pub base_level: S,
    /// 潮差振幅 [m]
    pub amplitude: S,
    /// 潮汐周期 [s]
    pub period: S,
    /// 初始相位 [rad]
    pub phase: S,
    /// 床面高程 [m]
    pub bed_level: S,
}

impl<S: RuntimeScalar> TidalLevel<S> {
    /// 创建新的潮汐边界
    ///
    /// # 参数
    /// - `base_level`: 平均水位 [m]
    /// - `amplitude`: 潮差振幅 [m]
    /// - `period`: 潮汐周期 [s]（半日潮约 44712 秒）
    /// - `bed_level`: 床面高程 [m]
    pub fn new(base_level: S, amplitude: S, period: S, bed_level: S) -> Self {
        Self {
            base_level,
            amplitude,
            period,
            phase: S::zero(),
            bed_level,
        }
    }

    /// 设置初始相位
    pub fn with_phase(mut self, phase: S) -> Self {
        self.phase = phase;
        self
    }
}

impl<S: RuntimeScalar> BoundaryConditionTrait<S> for TidalLevel<S> {
    fn name(&self) -> &'static str {
        "TidalLevel"
    }

    fn apply(&self, interior: &CellState<S>, _normal: [S; 2], time: S) -> CellState<S> {
        // η(t) = base + amplitude * sin(2π*t/T + φ)
        // 使用 std::f64::consts::PI，对于 f32 和 f64 转换始终成功
        let pi = S::from_f64(std::f64::consts::PI).unwrap_or(S::one());
        let two = S::from_f64(2.0).unwrap_or(S::one() + S::one());
        let period_safe = if self.period > S::epsilon() { self.period } else { S::one() };
        
        let omega = two * pi / period_safe;
        let eta = self.base_level + self.amplitude * (omega * time + self.phase).sin();
        let h = (eta - self.bed_level).max(S::zero());

        CellState {
            h,
            u: interior.u,
            v: interior.v,
        }
    }

    fn is_time_dependent(&self) -> bool {
        true
    }
}

/// Flather 辐射边界条件
///
/// 结合水位指定和辐射条件，常用于潮汐模拟的开边界。
/// 允许内部产生的波动自由传出，同时施加外部强迫水位。
///
/// # 公式
///
/// ```text
/// u_n = u_n_int + √(g/h) * (η_int - η_ext)
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FlatherBoundary<S> {
    /// 外部水位 [m]
    pub external_level: S,
    /// 重力加速度 [m/s²]
    pub gravity: S,
    /// 床面高程 [m]
    pub bed_level: S,
}

impl<S: RuntimeScalar> FlatherBoundary<S> {
    /// 创建新的 Flather 边界
    pub fn new(external_level: S, gravity: S, bed_level: S) -> Self {
        Self {
            external_level,
            gravity,
            bed_level,
        }
    }
}

impl<S: RuntimeScalar> BoundaryConditionTrait<S> for FlatherBoundary<S> {
    fn name(&self) -> &'static str {
        "FlatherBoundary"
    }

    fn apply(&self, interior: &CellState<S>, normal: [S; 2], _time: S) -> CellState<S> {
        // 内部水位
        let eta_int = interior.h + self.bed_level;
        // 外部水位
        let eta_ext = self.external_level;
        // 水深
        let h = (self.external_level - self.bed_level).max(S::epsilon());
        // 波速
        let c = (self.gravity * h).sqrt();
        // Flather 校正
        let un_int = interior.u * normal[0] + interior.v * normal[1];
        let un_ext = un_int + c * (eta_int - eta_ext) / h;

        // 计算速度分量
        let ut = -interior.u * normal[1] + interior.v * normal[0]; // 切向速度
        let u = -un_ext * normal[0] + ut * (-normal[1]);
        let v = -un_ext * normal[1] + ut * normal[0];

        CellState {
            h,
            u,
            v,
        }
    }
}

// ============================================================
// 类型擦除与注册表
// ============================================================

/// 类型擦除的边界条件
///
/// 用于存储任意类型的边界条件，支持动态分发。
pub type DynBoundaryCondition<S> = Arc<dyn BoundaryConditionTrait<S>>;

/// 边界条件工厂函数类型
pub type BoundaryFactory<S> = Box<
    dyn Fn(&serde_json::Value) -> Result<DynBoundaryCondition<S>, BoundaryParseError> + Send + Sync,
>;

/// 边界条件解析错误
#[derive(Debug, thiserror::Error)]
pub enum BoundaryParseError {
    /// 未知的边界类型
    #[error("Unknown boundary type: {0}")]
    UnknownType(String),

    /// 缺少必要参数
    #[error("Missing required parameter: {0}")]
    MissingParameter(String),

    /// 参数类型错误
    #[error("Invalid parameter type for {0}: expected {1}")]
    InvalidParameterType(String, String),

    /// 参数值无效
    #[error("Invalid parameter value for {0}: {1}")]
    InvalidParameterValue(String, String),

    /// JSON 解析错误
    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// 边界条件注册表
///
/// 管理边界条件类型的注册和实例化。
/// 支持从 JSON 配置创建边界条件实例。
pub struct BoundaryRegistry<S: RuntimeScalar> {
    factories: HashMap<String, BoundaryFactory<S>>,
}

impl<S: RuntimeScalar + 'static> BoundaryRegistry<S> {
    /// 创建新的注册表（包含内置类型）
    pub fn new() -> Self {
        let mut registry = Self {
            factories: HashMap::new(),
        };

        // 注册内置类型
        registry.register_builtin();

        registry
    }

    /// 创建空注册表
    pub fn empty() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// 注册内置边界类型
    fn register_builtin(&mut self) {
        // Reflective
        self.register("reflective", |_config| Ok(Arc::new(Reflective)));

        // Transmissive
        self.register("transmissive", |_config| Ok(Arc::new(Transmissive)));

        // NoSlipWall
        self.register("no_slip_wall", |_config| Ok(Arc::new(NoSlipWall)));

        // SlipWall
        self.register("slip_wall", |_config| Ok(Arc::new(SlipWall)));

        // Inflow
        self.register("inflow", |config| {
            let h = config
                .get("h")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| BoundaryParseError::MissingParameter("h".to_string()))?;
            let u = config.get("u").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let v = config.get("v").and_then(|v| v.as_f64()).unwrap_or(0.0);

            Ok(Arc::new(Inflow::new(
                S::from_f64(h).unwrap_or(S::one()),
                S::from_f64(u).unwrap_or(S::zero()),
                S::from_f64(v).unwrap_or(S::zero()),
            )))
        });

        // FixedLevelOutflow
        self.register("fixed_level_outflow", |config| {
            let level = config
                .get("level")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| BoundaryParseError::MissingParameter("level".to_string()))?;
            let bed_level = config.get("bed_level").and_then(|v| v.as_f64()).unwrap_or(0.0);

            Ok(Arc::new(FixedLevelOutflow::new(
                S::from_f64(level).unwrap_or(S::one()),
                S::from_f64(bed_level).unwrap_or(S::zero()),
            )))
        });

        // TidalLevel
        self.register("tidal", |config| {
            let base_level = config
                .get("base_level")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| BoundaryParseError::MissingParameter("base_level".to_string()))?;
            let amplitude = config
                .get("amplitude")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| BoundaryParseError::MissingParameter("amplitude".to_string()))?;
            let period = config
                .get("period")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| BoundaryParseError::MissingParameter("period".to_string()))?;
            let bed_level = config.get("bed_level").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let phase = config.get("phase").and_then(|v| v.as_f64()).unwrap_or(0.0);

            Ok(Arc::new(
                TidalLevel::new(
                    S::from_f64(base_level).unwrap_or(S::one()),
                    S::from_f64(amplitude).unwrap_or(S::one()),
                    S::from_f64(period).unwrap_or(S::one()),
                    S::from_f64(bed_level).unwrap_or(S::zero()),
                )
                .with_phase(S::from_f64(phase).unwrap_or(S::zero())),
            ))
        });

        // FlatherBoundary
        self.register("flather", |config| {
            let external_level = config
                .get("external_level")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| BoundaryParseError::MissingParameter("external_level".to_string()))?;
            let gravity = config.get("gravity").and_then(|v| v.as_f64()).unwrap_or(9.81);
            let bed_level = config.get("bed_level").and_then(|v| v.as_f64()).unwrap_or(0.0);

            Ok(Arc::new(FlatherBoundary::new(
                S::from_f64(external_level).unwrap_or(S::one()),
                S::from_f64(gravity).unwrap_or(S::from_f64(9.81).unwrap_or(S::one())),
                S::from_f64(bed_level).unwrap_or(S::zero()),
            )))
        });
    }

    /// 注册自定义边界类型
    ///
    /// # 参数
    /// - `name`: 边界类型名称（小写，用于 JSON 配置匹配）
    /// - `factory`: 工厂函数，从 JSON 配置创建边界条件实例
    pub fn register<F>(&mut self, name: &str, factory: F)
    where
        F: Fn(&serde_json::Value) -> Result<DynBoundaryCondition<S>, BoundaryParseError>
            + Send
            + Sync
            + 'static,
    {
        self.factories.insert(name.to_lowercase(), Box::new(factory));
    }

    /// 从配置创建边界条件实例
    ///
    /// # 参数
    /// - `name`: 边界类型名称
    /// - `config`: JSON 配置对象
    ///
    /// # 返回
    /// 边界条件实例，或解析错误
    pub fn create(
        &self,
        name: &str,
        config: &serde_json::Value,
    ) -> Result<DynBoundaryCondition<S>, BoundaryParseError> {
        let factory = self
            .factories
            .get(&name.to_lowercase())
            .ok_or_else(|| BoundaryParseError::UnknownType(name.to_string()))?;
        factory(config)
    }

    /// 列出所有已注册的边界类型
    pub fn registered_types(&self) -> Vec<&str> {
        self.factories.keys().map(|s| s.as_str()).collect()
    }
}

impl<S: RuntimeScalar + 'static> Default for BoundaryRegistry<S> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflective_boundary() {
        let bc = Reflective;
        let interior = CellState::new(1.0, 1.0, 0.0);
        let normal = [1.0, 0.0]; // x 方向法向

        let ghost = bc.apply(&interior, normal, 0.0);

        assert_eq!(ghost.h, 1.0);
        assert!((ghost.u - (-1.0_f64)).abs() < 1e-10); // u 反向
        assert!((ghost.v - 0.0_f64).abs() < 1e-10); // v 不变
    }

    #[test]
    fn test_transmissive_boundary() {
        let bc = Transmissive;
        let interior = CellState::new(1.0, 2.0, 3.0);
        let normal = [1.0, 0.0];

        let ghost = bc.apply(&interior, normal, 0.0);

        assert_eq!(ghost.h, interior.h);
        assert_eq!(ghost.u, interior.u);
        assert_eq!(ghost.v, interior.v);
    }

    #[test]
    fn test_inflow_boundary() {
        let bc = Inflow::new(2.0, 1.0, 0.5);
        let interior = CellState::new(1.0, 0.0, 0.0);
        let normal = [1.0, 0.0];

        let ghost = bc.apply(&interior, normal, 0.0);

        assert_eq!(ghost.h, 2.0);
        assert_eq!(ghost.u, 1.0);
        assert_eq!(ghost.v, 0.5);
    }

    #[test]
    fn test_tidal_boundary() {
        let bc = TidalLevel::new(1.0, 0.5, 1.0, 0.0).with_phase(0.0);
        let interior = CellState::new(1.0, 0.0, 0.0);
        let normal = [1.0, 0.0];

        // t=0: η = base + amplitude * sin(0) = 1.0
        let ghost = bc.apply(&interior, normal, 0.0);
        assert!((ghost.h - 1.0_f64).abs() < 1e-10);

        // t=T/4: η = base + amplitude * sin(π/2) = 1.0 + 0.5 = 1.5
        let ghost = bc.apply(&interior, normal, 0.25);
        assert!((ghost.h - 1.5_f64).abs() < 1e-10);

        assert!(bc.is_time_dependent());
    }

    #[test]
    fn test_no_slip_wall() {
        let bc = NoSlipWall;
        let interior = CellState::new(1.0, 2.0, 3.0);
        let normal = [1.0, 0.0];

        let ghost = bc.apply(&interior, normal, 0.0);

        assert_eq!(ghost.h, 1.0);
        assert_eq!(ghost.u, 0.0);
        assert_eq!(ghost.v, 0.0);
    }

    #[test]
    fn test_registry() {
        let registry: BoundaryRegistry<f64> = BoundaryRegistry::new();

        // 测试反射边界
        let bc = registry
            .create("reflective", &serde_json::json!({}))
            .unwrap();
        assert_eq!(bc.name(), "Reflective");

        // 测试入流边界
        let bc = registry
            .create("inflow", &serde_json::json!({"h": 2.0, "u": 1.0, "v": 0.5}))
            .unwrap();
        assert_eq!(bc.name(), "Inflow");

        // 测试未知类型
        let result = registry.create("unknown", &serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_cell_state_operations() {
        let state = CellState::new(1.0_f64, 3.0, 4.0);
        let normal = [0.6, 0.8]; // 单位向量

        let vn = state.normal_velocity(normal);
        assert!((vn - 5.0).abs() < 1e-10); // 3*0.6 + 4*0.8 = 1.8 + 3.2 = 5.0

        let vt = state.tangent_velocity(normal);
        assert!((vt - 0.0).abs() < 1e-10); // -3*0.8 + 4*0.6 = -2.4 + 2.4 = 0.0
    }
}
