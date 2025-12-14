// crates/mh_config/src/dyn_solver.rs

//! DynSolver - 运行时多态求解器接口
//!
//! 定义不含泛型的求解器接口，用于在应用层进行运行时多态调用。

use std::any::Any;
use crate::precision::Precision;

/// 网格信息（运行时通用）
#[derive(Debug, Clone, Default)]
pub struct GridInfo {
    /// 单元数
    pub n_cells: usize,
    /// 面数
    pub n_faces: usize,
    /// 节点数
    pub n_nodes: usize,
    /// 边界框 [min_x, min_y, max_x, max_y]
    pub bounds: [f64; 4],
}

/// 性能指标快照
#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    /// 已完成时间步数
    pub time_steps: u64,
    /// 通量计算次数
    pub flux_evals: u64,
    /// 总计算时间 [s]
    pub total_time_sec: f64,
    /// 平均每步耗时 [ms]
    pub avg_step_time_ms: f64,
}

/// 求解器错误
#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    /// CFL 条件违反
    #[error("CFL 条件违反")]
    CflViolation,
    
    /// 数值发散
    #[error("数值发散: {message}")]
    Divergence { 
        /// 错误详情
        message: String 
    },
    
    /// 无效索引
    #[error("无效索引: {index_type}({idx})")]
    InvalidIndex { 
        /// 索引类型
        index_type: &'static str, 
        /// 索引值
        idx: usize 
    },
    
    /// 配置错误
    #[error("配置错误: {0}")]
    Config(String),
    
    /// 网格错误
    #[error("网格错误: {0}")]
    Mesh(String),
    
    /// 内部错误
    #[error("内部错误: {0}")]
    Internal(String),
}

/// 运行时求解器接口（无泛型）
///
/// 所有具体求解器（如 `ShallowWaterSolver<B>`）都应实现此 trait，
/// 以支持在应用层进行类型擦除的多态调用。
///
/// # 示例
///
/// ```ignore
/// use mh_config::{SolverConfig, DynSolver};
///
/// fn run_simulation(solver: &mut dyn DynSolver, target_time: f64) {
///     while solver.time() < target_time {
///         let dt = 0.01;
///         solver.step(dt).expect("step failed");
///     }
///     println!("Final time: {}", solver.time());
/// }
/// ```
pub trait DynSolver: Send + Sync {
    /// 执行一个时间步
    ///
    /// # 参数
    /// - `dt`: 时间步长 [s]
    ///
    /// # 返回
    /// - `Ok(())`: 成功
    /// - `Err(SolverError)`: 失败
    fn step(&mut self, dt: f64) -> Result<(), SolverError>;
    
    /// 当前模拟时间 [s]
    fn time(&self) -> f64;
    
    /// 当前步数
    fn step_count(&self) -> u64;
    
    /// 导出状态（类型擦除）
    ///
    /// 返回类型擦除的状态，需要通过 `downcast` 获取具体类型。
    fn export_state(&self) -> Box<dyn Any + Send>;
    
    /// 获取精度信息
    fn precision(&self) -> Precision;
    
    /// 获取网格信息
    fn grid_info(&self) -> GridInfo;
    
    /// 获取性能指标
    fn metrics(&self) -> MetricsSnapshot;
    
    /// 重置求解器到初始状态
    fn reset(&mut self) -> Result<(), SolverError>;
    
    /// 获取当前最大 CFL 数
    fn current_cfl(&self) -> f64 {
        1.0 // 默认实现
    }
    
    /// 计算推荐时间步长
    fn recommended_dt(&self) -> f64 {
        0.001 // 默认实现
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_info_default() {
        let info = GridInfo::default();
        assert_eq!(info.n_cells, 0);
        assert_eq!(info.bounds, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_metrics_snapshot_default() {
        let metrics = MetricsSnapshot::default();
        assert_eq!(metrics.time_steps, 0);
        assert_eq!(metrics.flux_evals, 0);
    }
}
