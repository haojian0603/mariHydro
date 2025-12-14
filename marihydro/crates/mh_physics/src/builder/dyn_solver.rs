// crates/mh_physics/src/builder/dyn_solver.rs

//! 动态求解器trait
//!
//! 提供运行时多态的求解器接口，使App层可以无泛型地使用求解器。
//!
//! # 迁移说明
//!
//! 新代码应使用 `mh_config::DynSolver`，本模块保留用于向后兼容。

use mh_config::Precision;
use std::fmt;

// 从 mh_config 重导出规范接口（供新代码使用）
pub use mh_config::dyn_solver::DynSolver as ConfigDynSolver;

/// 动态求解器trait（运行时多态）
///
/// 所有具体求解器（如 `ShallowWaterSolver<CpuBackend<f32>>`）都自动实现此trait。
/// App层通过 `Box<dyn DynSolver>` 使用求解器，无需关心底层精度。
///
/// # 迁移指南
///
/// 新代码建议使用 `mh_config::DynSolver`：
/// ```ignore
/// use mh_config::{DynSolver, SolverConfig};
/// ```
///
/// # 示例
///
/// ```ignore
/// let solver = SolverBuilder::new(config).build()?;
/// while solver.time() < end_time {
///     let result = solver.step(dt);
///     println!("t={:.2}, dt={:.4}", solver.time(), result.dt_actual);
/// }
/// ```
pub trait DynSolver: Send + Sync {
    /// 执行时间步进
    ///
    /// # 参数
    /// - `dt`: 建议时间步长（秒）
    ///
    /// # 返回
    /// 实际使用的时间步结果
    fn step(&mut self, dt: f64) -> DynStepResult;

    /// 获取当前模拟时间（秒）
    fn time(&self) -> f64;

    /// 获取当前时间步数
    fn step_count(&self) -> usize;

    /// 获取使用的精度
    fn precision(&self) -> Precision;

    /// 导出当前状态（统一为f64）
    fn export_state(&self) -> DynState;

    /// 获取求解器统计信息
    fn stats(&self) -> SolverStats;

    /// 获取网格单元数量
    fn n_cells(&self) -> usize;

    /// 获取网格面数量
    fn n_faces(&self) -> usize;

    /// 检查求解器是否处于健康状态
    fn is_healthy(&self) -> bool;

    /// 获取求解器名称
    fn name(&self) -> &'static str;
}

/// 动态时间步结果
#[derive(Debug, Clone)]
pub struct DynStepResult {
    /// 实际使用的时间步长
    pub dt_actual: f64,
    /// 最大CFL数
    pub max_cfl: f64,
    /// 质量误差（相对值）
    pub mass_error: f64,
    /// 是否成功完成
    pub success: bool,
    /// 错误消息（如果失败）
    pub error_message: Option<String>,
}

impl Default for DynStepResult {
    fn default() -> Self {
        Self {
            dt_actual: 0.0,
            max_cfl: 0.0,
            mass_error: 0.0,
            success: true,
            error_message: None,
        }
    }
}

impl DynStepResult {
    /// 创建成功结果
    pub fn success(dt_actual: f64, max_cfl: f64, mass_error: f64) -> Self {
        Self {
            dt_actual,
            max_cfl,
            mass_error,
            success: true,
            error_message: None,
        }
    }

    /// 创建失败结果
    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            success: false,
            error_message: Some(message.into()),
            ..Default::default()
        }
    }
}

/// 统一状态导出格式（使用f64）
#[derive(Debug, Clone)]
pub struct DynState {
    /// 水深 [m]
    pub h: Vec<f64>,
    /// x方向流速 [m/s]
    pub u: Vec<f64>,
    /// y方向流速 [m/s]
    pub v: Vec<f64>,
    /// 底高程 [m]
    pub z: Vec<f64>,
    /// 当前时间 [s]
    pub time: f64,
    /// 单元数量
    pub n_cells: usize,
}

impl DynState {
    /// 创建空状态
    pub fn empty() -> Self {
        Self {
            h: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            z: Vec::new(),
            time: 0.0,
            n_cells: 0,
        }
    }

    /// 获取水位 (eta = h + z)
    pub fn water_level(&self) -> Vec<f64> {
        self.h.iter().zip(&self.z).map(|(h, z)| h + z).collect()
    }

    /// 获取流速大小
    pub fn velocity_magnitude(&self) -> Vec<f64> {
        self.u.iter()
            .zip(&self.v)
            .map(|(u, v)| (u * u + v * v).sqrt())
            .collect()
    }

    /// 获取单元的Froude数
    pub fn froude_number(&self, g: f64) -> Vec<f64> {
        self.h.iter()
            .zip(self.velocity_magnitude().iter())
            .map(|(h, vel)| {
                if *h > 1e-6 {
                    vel / (g * h).sqrt()
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// 求解器统计信息
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    /// 总时间步数
    pub total_steps: usize,
    /// 总计算时间（秒）
    pub total_compute_time: f64,
    /// 平均每步时间（秒）
    pub avg_step_time: f64,
    /// 最大CFL数
    pub max_cfl_ever: f64,
    /// 累计质量误差
    pub cumulative_mass_error: f64,
    /// 是否发生过数值问题
    pub had_numerical_issues: bool,
}

impl fmt::Display for SolverStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Steps: {}, Time: {:.2}s, Avg: {:.3}ms/step, MaxCFL: {:.3}",
            self.total_steps,
            self.total_compute_time,
            self.avg_step_time * 1000.0,
            self.max_cfl_ever
        )
    }
}
