// crates/mh_physics/src/builder/solver_builder.rs

//! 求解器构建器
//!
//! 实现从无泛型配置到泛型引擎的桥梁。

use super::config::{ConfigError, SolverConfig};
use super::dyn_solver::{DynSolver, DynState, DynStepResult, SolverStats};
use mh_core::{Precision, Scalar, Tolerance};
use std::time::Instant;

/// 构建错误
#[derive(Debug)]
pub enum BuildError {
    /// 配置验证失败
    ConfigError(ConfigError),
    /// 缺少网格
    MissingMesh,
    /// 网格无效
    InvalidMesh(String),
    /// 初始条件错误
    InitialConditionError(String),
    /// 其他错误
    Other(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::ConfigError(e) => write!(f, "配置错误: {}", e),
            BuildError::MissingMesh => write!(f, "缺少网格"),
            BuildError::InvalidMesh(msg) => write!(f, "无效网格: {}", msg),
            BuildError::InitialConditionError(msg) => write!(f, "初始条件错误: {}", msg),
            BuildError::Other(msg) => write!(f, "构建错误: {}", msg),
        }
    }
}

impl std::error::Error for BuildError {}

impl From<ConfigError> for BuildError {
    fn from(e: ConfigError) -> Self {
        BuildError::ConfigError(e)
    }
}

/// 求解器构建器
///
/// 提供从无泛型配置构建具体泛型求解器的能力。
///
/// # 示例
///
/// ```ignore
/// use mh_physics::builder::{SolverBuilder, SolverConfig};
///
/// let config = SolverConfig::default();
/// let solver = SolverBuilder::new(config)
///     .with_mesh(mesh)
///     .build()?;
///
/// println!("使用精度: {:?}", solver.precision());
/// ```
pub struct SolverBuilder {
    config: SolverConfig,
    n_cells: usize,
    initial_h: Option<Vec<f64>>,
    initial_u: Option<Vec<f64>>,
    initial_v: Option<Vec<f64>>,
    bathymetry: Option<Vec<f64>>,
}

impl SolverBuilder {
    /// 创建新的构建器
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            n_cells: 0,
            initial_h: None,
            initial_u: None,
            initial_v: None,
            bathymetry: None,
        }
    }

    /// 设置单元数量（简化网格接口）
    pub fn with_cells(mut self, n_cells: usize) -> Self {
        self.n_cells = n_cells;
        self
    }

    /// 设置初始水深
    pub fn with_initial_depth(mut self, h: Vec<f64>) -> Self {
        self.initial_h = Some(h);
        self
    }

    /// 设置初始速度
    pub fn with_initial_velocity(mut self, u: Vec<f64>, v: Vec<f64>) -> Self {
        self.initial_u = Some(u);
        self.initial_v = Some(v);
        self
    }

    /// 设置底部高程
    pub fn with_bathymetry(mut self, z: Vec<f64>) -> Self {
        self.bathymetry = Some(z);
        self
    }

    /// 设置静水初始条件
    pub fn with_still_water(mut self, depth: f64, n_cells: usize) -> Self {
        self.n_cells = n_cells;
        self.initial_h = Some(vec![depth; n_cells]);
        self.initial_u = Some(vec![0.0; n_cells]);
        self.initial_v = Some(vec![0.0; n_cells]);
        self.bathymetry = Some(vec![0.0; n_cells]);
        self
    }

    /// 构建求解器
    ///
    /// 根据配置中的精度选择实例化对应的泛型求解器。
    pub fn build(mut self) -> Result<Box<dyn DynSolver>, BuildError> {
        // 验证配置
        self.config.validate()?;
        
        // 根据精度调整容差
        self.config.adjust_for_precision();

        // 验证单元数量
        if self.n_cells == 0 {
            return Err(BuildError::MissingMesh);
        }

        // 确保有初始条件
        let n = self.n_cells;
        if self.initial_h.is_none() {
            self.initial_h = Some(vec![0.0; n]);
        }
        if self.initial_u.is_none() {
            self.initial_u = Some(vec![0.0; n]);
        }
        if self.initial_v.is_none() {
            self.initial_v = Some(vec![0.0; n]);
        }
        if self.bathymetry.is_none() {
            self.bathymetry = Some(vec![0.0; n]);
        }

        // 验证数组长度
        if self.initial_h.as_ref().unwrap().len() != n
            || self.initial_u.as_ref().unwrap().len() != n
            || self.initial_v.as_ref().unwrap().len() != n
            || self.bathymetry.as_ref().unwrap().len() != n
        {
            return Err(BuildError::InvalidMesh(
                "数组长度与单元数不匹配".to_string(),
            ));
        }

        // 根据精度分发
        match self.config.precision {
            Precision::F32 => self.build_f32(),
            Precision::F64 => self.build_f64(),
        }
    }

    fn build_f32(self) -> Result<Box<dyn DynSolver>, BuildError> {
        let solver = SimpleSolver::<f32>::new(
            self.config.clone(),
            self.n_cells,
            self.initial_h.unwrap(),
            self.initial_u.unwrap(),
            self.initial_v.unwrap(),
            self.bathymetry.unwrap(),
        );
        Ok(Box::new(solver))
    }

    fn build_f64(self) -> Result<Box<dyn DynSolver>, BuildError> {
        let solver = SimpleSolver::<f64>::new(
            self.config.clone(),
            self.n_cells,
            self.initial_h.unwrap(),
            self.initial_u.unwrap(),
            self.initial_v.unwrap(),
            self.bathymetry.unwrap(),
        );
        Ok(Box::new(solver))
    }
}

/// 简化求解器实现（用于演示Builder模式）
///
/// 这是一个简化的求解器实现，展示了泛型精度系统的工作方式。
/// 实际的 ShallowWaterSolver 会更复杂，包含完整的数值格式。
struct SimpleSolver<S: Scalar> {
    config: SolverConfig,
    n_cells: usize,
    h: Vec<S>,
    u: Vec<S>,
    v: Vec<S>,
    z: Vec<S>,
    time: S,
    step_count: usize,
    tolerance: Tolerance<S>,
    stats: SolverStats,
    start_time: Instant,
}

impl<S: Scalar + Default> SimpleSolver<S>
where
    Tolerance<S>: Default,
{
    fn new(
        config: SolverConfig,
        n_cells: usize,
        h: Vec<f64>,
        u: Vec<f64>,
        v: Vec<f64>,
        z: Vec<f64>,
    ) -> Self {
        Self {
            config,
            n_cells,
            h: h.into_iter().map(|x| S::from_f64_lossless(x)).collect(),
            u: u.into_iter().map(|x| S::from_f64_lossless(x)).collect(),
            v: v.into_iter().map(|x| S::from_f64_lossless(x)).collect(),
            z: z.into_iter().map(|x| S::from_f64_lossless(x)).collect(),
            time: S::ZERO,
            step_count: 0,
            tolerance: Tolerance::<S>::default(),
            stats: SolverStats::default(),
            start_time: Instant::now(),
        }
    }

    /// 执行一个时间步（简化版本）
    fn step_internal(&mut self, dt: S) -> (S, S, S) {
        let g = S::from_f64_lossless(self.config.gravity);
        let dt_actual = dt;

        // 简化的更新逻辑（实际求解器会使用完整的有限体积法）
        let mut max_cfl = S::ZERO;
        
        for i in 0..self.n_cells {
            let h = self.h[i];
            if h > self.tolerance.h_dry {
                // 计算波速
                let c = (g * h).sqrt();
                let vel = (self.u[i] * self.u[i] + self.v[i] * self.v[i]).sqrt();
                let cfl = (vel + c) * dt / S::from_f64_lossless(1.0); // 假设 dx = 1
                if cfl > max_cfl {
                    max_cfl = cfl;
                }
            }
        }

        self.time = self.time + dt_actual;
        self.step_count += 1;

        (dt_actual, max_cfl, S::ZERO)
    }
}

impl<S: Scalar + Default> DynSolver for SimpleSolver<S>
where
    Tolerance<S>: Default,
{
    fn step(&mut self, dt: f64) -> DynStepResult {
        let dt_s = Scalar::from_f64_lossless(dt);
        let (dt_actual, max_cfl, mass_error) = self.step_internal(dt_s);
        
        // 更新统计
        self.stats.total_steps = self.step_count;
        self.stats.total_compute_time = self.start_time.elapsed().as_secs_f64();
        if self.step_count > 0 {
            self.stats.avg_step_time = self.stats.total_compute_time / self.step_count as f64;
        }
        let cfl_f64 = max_cfl.to_f64();
        if cfl_f64 > self.stats.max_cfl_ever {
            self.stats.max_cfl_ever = cfl_f64;
        }

        DynStepResult::success(dt_actual.to_f64(), cfl_f64, mass_error.to_f64())
    }

    fn time(&self) -> f64 {
        self.time.to_f64()
    }

    fn step_count(&self) -> usize {
        self.step_count
    }

    fn precision(&self) -> Precision {
        if std::any::TypeId::of::<S>() == std::any::TypeId::of::<f32>() {
            Precision::F32
        } else {
            Precision::F64
        }
    }

    fn export_state(&self) -> DynState {
        DynState {
            h: self.h.iter().map(|x| Scalar::to_f64(*x)).collect(),
            u: self.u.iter().map(|x| Scalar::to_f64(*x)).collect(),
            v: self.v.iter().map(|x| Scalar::to_f64(*x)).collect(),
            z: self.z.iter().map(|x| Scalar::to_f64(*x)).collect(),
            time: self.time.to_f64(),
            n_cells: self.n_cells,
        }
    }

    fn stats(&self) -> SolverStats {
        self.stats.clone()
    }

    fn n_cells(&self) -> usize {
        self.n_cells
    }

    fn n_faces(&self) -> usize {
        0 // 简化实现
    }

    fn is_healthy(&self) -> bool {
        // 检查是否有NaN或Inf
        self.h.iter().all(|x| x.is_safe())
            && self.u.iter().all(|x| x.is_safe())
            && self.v.iter().all(|x| x.is_safe())
    }

    fn name(&self) -> &'static str {
        "SimpleSolver"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_f32() {
        let config = SolverConfig {
            precision: Precision::F32,
            ..Default::default()
        };
        
        let solver = SolverBuilder::new(config)
            .with_still_water(1.0, 100)
            .build()
            .unwrap();

        assert_eq!(solver.precision(), Precision::F32);
        assert_eq!(solver.n_cells(), 100);
    }

    #[test]
    fn test_builder_f64() {
        let config = SolverConfig {
            precision: Precision::F64,
            ..Default::default()
        };
        
        let solver = SolverBuilder::new(config)
            .with_still_water(1.0, 100)
            .build()
            .unwrap();

        assert_eq!(solver.precision(), Precision::F64);
    }

    #[test]
    fn test_solver_step() {
        let config = SolverConfig::default();
        let mut solver = SolverBuilder::new(config)
            .with_still_water(1.0, 10)
            .build()
            .unwrap();

        let result = solver.step(0.01);
        assert!(result.success);
        assert!(solver.time() > 0.0);
        assert_eq!(solver.step_count(), 1);
    }

    #[test]
    fn test_export_state() {
        let config = SolverConfig::default();
        let solver = SolverBuilder::new(config)
            .with_still_water(2.5, 5)
            .build()
            .unwrap();

        let state = solver.export_state();
        assert_eq!(state.n_cells, 5);
        assert_eq!(state.h.len(), 5);
        assert!((state.h[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_missing_mesh() {
        let config = SolverConfig::default();
        let result = SolverBuilder::new(config).build();
        assert!(result.is_err());
    }
}
