// crates/mh_physics/src/conservation.rs

//! 守恒量监测与质量平衡检查
//!
//! 提供工业级守恒性验证，支持：
//! - 质量守恒监测
//! - 动量守恒检查
//! - 能量收支分析
//! - 边界通量积分
//! - 源汇项平衡
//!
//! # 设计原则
//!
//! 1. **精确积分**：使用高精度求积
//! 2. **误差追踪**：记录累计误差
//! 3. **诊断输出**：支持详细报告
//! 4. **工业标准**：比肩 Delft3D 守恒性

use std::collections::HashMap;

// ============================================================================
// 守恒量类型
// ============================================================================

/// 守恒量类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConservationType {
    /// 质量（水体积）
    Mass,
    /// X方向动量
    MomentumX,
    /// Y方向动量
    MomentumY,
    /// 总能量
    Energy,
    /// 势能
    PotentialEnergy,
    /// 动能
    KineticEnergy,
    /// 恩斯特罗菲（涡度平方积分）
    Enstrophy,
    /// 被动示踪剂
    Tracer(usize),
}

impl std::fmt::Display for ConservationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mass => write!(f, "质量"),
            Self::MomentumX => write!(f, "X动量"),
            Self::MomentumY => write!(f, "Y动量"),
            Self::Energy => write!(f, "总能量"),
            Self::PotentialEnergy => write!(f, "势能"),
            Self::KineticEnergy => write!(f, "动能"),
            Self::Enstrophy => write!(f, "恩斯特罗菲"),
            Self::Tracer(id) => write!(f, "示踪剂{}", id),
        }
    }
}

// ============================================================================
// 守恒量记录
// ============================================================================

/// 单时刻守恒量快照
#[derive(Debug, Clone)]
pub struct ConservationSnapshot {
    /// 时间戳
    pub time: f64,
    /// 各守恒量的当前值
    pub quantities: HashMap<ConservationType, f64>,
    /// 边界通量（进入为正）
    pub boundary_fluxes: HashMap<String, HashMap<ConservationType, f64>>,
    /// 源汇项贡献
    pub source_terms: HashMap<String, HashMap<ConservationType, f64>>,
}

impl ConservationSnapshot {
    /// 创建空快照
    pub fn new(time: f64) -> Self {
        Self {
            time,
            quantities: HashMap::new(),
            boundary_fluxes: HashMap::new(),
            source_terms: HashMap::new(),
        }
    }

    /// 获取守恒量
    pub fn get(&self, qty: ConservationType) -> f64 {
        *self.quantities.get(&qty).unwrap_or(&0.0)
    }

    /// 设置守恒量
    pub fn set(&mut self, qty: ConservationType, value: f64) {
        self.quantities.insert(qty, value);
    }

    /// 添加边界通量
    pub fn add_boundary_flux(&mut self, boundary: &str, qty: ConservationType, flux: f64) {
        self.boundary_fluxes
            .entry(boundary.to_string())
            .or_default()
            .insert(qty, flux);
    }

    /// 添加源项贡献
    pub fn add_source_term(&mut self, source: &str, qty: ConservationType, value: f64) {
        self.source_terms
            .entry(source.to_string())
            .or_default()
            .insert(qty, value);
    }
    /// 带回调的守恒监测器（静态分发）

// ============================================================================
// 守恒性监测器
// ============================================================================

/// 守恒性监测配置
#[derive(Debug, Clone)]
pub struct ConservationConfig {
    /// 是否启用质量守恒检查
    pub check_mass: bool,
    /// 是否启用动量守恒检查
    pub check_momentum: bool,
    /// 是否启用能量守恒检查
    pub check_energy: bool,
    /// 相对误差警告阈值
    pub warning_threshold: f64,
    /// 相对误差错误阈值
    pub error_threshold: f64,
    /// 检查间隔（步数）
    pub check_interval: usize,
    /// 轻量检查间隔（步数）
    pub lightweight_interval: usize,
    /// 是否输出详细报告
    pub verbose: bool,
}

impl Default for ConservationConfig {
    fn default() -> Self {
        Self {
            check_mass: true,
            check_momentum: true,
            check_energy: false, // 能量检查开销较大
            warning_threshold: 1e-6,
            error_threshold: 1e-3,
            check_interval: 100,
            lightweight_interval: 1,
            verbose: false,
        }
    }
}

/// 守恒性监测器
///
/// 追踪模拟过程中的守恒量变化
pub struct ConservationMonitor {
    /// 配置
    config: ConservationConfig,
    /// 初始快照
    initial: Option<ConservationSnapshot>,
    /// 上一时刻快照
    previous: Option<ConservationSnapshot>,
    /// 当前快照
    current: Option<ConservationSnapshot>,
    cumulative_boundary_flux: HashMap<ConservationType, f64>,
    cumulative_sources: HashMap<ConservationType, f64>,
    /// 历史误差记录
    error_history: Vec<ConservationError>,
    /// 步数计数
    step_count: usize,
}

impl ConservationMonitor {
    /// 创建监测器
    pub fn new(config: ConservationConfig) -> Self {
        Self {
            config,
            initial: None,
            previous: None,
            current: None,
            cumulative_boundary_flux: HashMap::new(),
            cumulative_sources: HashMap::new(),
            error_history: Vec::new(),
            step_count: 0,
        }
    }

    /// 设置初始状态
    pub fn set_initial(&mut self, snapshot: ConservationSnapshot) {
        self.initial = Some(snapshot.clone());
        self.current = Some(snapshot);
    }

    /// 更新当前状态
    pub fn update(&mut self, snapshot: ConservationSnapshot, dt: f64) -> ConservationResult {
        self.step_count += 1;
        
        // 保存上一状态
        self.previous = self.current.take();
        
        // 累积边界通量
        for fluxes in snapshot.boundary_fluxes.values() {
            for (qty, flux) in fluxes {
                *self.cumulative_boundary_flux.entry(*qty).or_insert(0.0) += flux * dt;
            }
        }
        
        // 累积源项
        for sources in snapshot.source_terms.values() {
            for (qty, value) in sources {
                *self.cumulative_sources.entry(*qty).or_insert(0.0) += value * dt;
            }
        }
        
        self.current = Some(snapshot);

        let mut result = ConservationResult::Ok;

        if self.config.lightweight_interval > 0
            && self.step_count.is_multiple_of(self.config.lightweight_interval)
        {
            result = Self::merge_results(result, self.check_lightweight());
        }

        // 检查守恒性（完整）
        if self.config.check_interval > 0
            && self.step_count.is_multiple_of(self.config.check_interval)
        {
            result = Self::merge_results(result, self.check_conservation());
        }

        result
    }

    /// 轻量级守恒检查（每步可用）
    fn check_lightweight(&mut self) -> ConservationResult {
        let Some(initial) = &self.initial else {
            return ConservationResult::NotInitialized;
        };

        let Some(current) = &self.current else {
            return ConservationResult::NotInitialized;
        };

        let mut errors = Vec::new();

        if self.config.check_mass {
            if let Some(e) = self.compute_error(ConservationType::Mass, initial, current) {
                errors.push(e);
            }
        }

        if errors.is_empty() {
            ConservationResult::Ok
        } else if errors.iter().any(|e| e.relative_error > self.config.error_threshold) {
            ConservationResult::Error(errors)
        } else {
            ConservationResult::Warning(errors)
        }
    }

    /// 检查守恒性
    fn check_conservation(&mut self) -> ConservationResult {
        let Some(initial) = &self.initial else {
            return ConservationResult::NotInitialized;
        };
        
        let Some(current) = &self.current else {
            return ConservationResult::NotInitialized;
        };

        let mut errors = Vec::new();

        // 质量守恒检查
        if self.config.check_mass {
            let error = self.compute_error(ConservationType::Mass, initial, current);
            if let Some(e) = error {
                errors.push(e);
            }
        }

        // 动量守恒检查
        if self.config.check_momentum {
            if let Some(e) = self.compute_error(ConservationType::MomentumX, initial, current) {
                errors.push(e);
            }
            if let Some(e) = self.compute_error(ConservationType::MomentumY, initial, current) {
                errors.push(e);
            }
        }

        // 能量守恒检查
        if self.config.check_energy {
            if let Some(e) = self.compute_error(ConservationType::Energy, initial, current) {
                errors.push(e);
            }
        }

        // 记录历史
        self.error_history.extend(errors.iter().cloned());

        // 评估结果
        let max_error = errors.iter()
            .map(|e| e.relative_error.abs())
            .fold(0.0f64, f64::max);

        if max_error > self.config.error_threshold {
            ConservationResult::Error(errors)
        } else if max_error > self.config.warning_threshold {
            ConservationResult::Warning(errors)
        } else {
            ConservationResult::Ok
        }
    }

    /// 计算守恒误差
    fn compute_error(
        &self,
        qty: ConservationType,
        initial: &ConservationSnapshot,
        current: &ConservationSnapshot,
    ) -> Option<ConservationError> {
        let initial_value = initial.get(qty);
        let current_value = current.get(qty);
        
        // 边界和源的贡献
        let boundary_contrib = *self.cumulative_boundary_flux.get(&qty).unwrap_or(&0.0);
        let source_contrib = *self.cumulative_sources.get(&qty).unwrap_or(&0.0);
        
        // 预期值 = 初始 + 边界流入 + 源项
        let expected = initial_value + boundary_contrib + source_contrib;
        
        // 绝对误差
        let absolute_error = current_value - expected;
        
        // 相对误差（避免除零）
        let scale = initial_value.abs().max(expected.abs()).max(1e-10);
        let relative_error = absolute_error / scale;

        if relative_error.abs() > self.config.warning_threshold {
            Some(ConservationError {
                time: current.time,
                quantity: qty,
                initial_value,
                current_value,
                expected_value: expected,
                absolute_error,
                relative_error,
                boundary_contribution: boundary_contrib,
                source_contribution: source_contrib,
            })
        } else {
            None
        }
    }

    /// 获取当前守恒量
    pub fn current_quantities(&self) -> Option<&HashMap<ConservationType, f64>> {
        self.current.as_ref().map(|s| &s.quantities)
    }

    /// 获取累计边界通量
    pub fn cumulative_boundary_flux(&self) -> &HashMap<ConservationType, f64> {
        &self.cumulative_boundary_flux
    }

    /// 获取累计源项
    pub fn cumulative_sources(&self) -> &HashMap<ConservationType, f64> {
        &self.cumulative_sources
    }

    /// 生成守恒性报告
    pub fn generate_report(&self) -> ConservationReport {
        let initial = self.initial.as_ref();
        let current = self.current.as_ref();
        
        ConservationReport {
            step_count: self.step_count,
            initial_time: initial.map(|s| s.time).unwrap_or(0.0),
            current_time: current.map(|s| s.time).unwrap_or(0.0),
            initial_mass: initial.map(|s| s.get(ConservationType::Mass)).unwrap_or(0.0),
            current_mass: current.map(|s| s.get(ConservationType::Mass)).unwrap_or(0.0),
            cumulative_inflow: *self.cumulative_boundary_flux.get(&ConservationType::Mass).unwrap_or(&0.0),
            cumulative_sources: *self.cumulative_sources.get(&ConservationType::Mass).unwrap_or(&0.0),
            max_relative_error: self.error_history.iter()
                .map(|e| e.relative_error.abs())
                .fold(0.0f64, f64::max),
            error_count: self.error_history.iter()
                .filter(|e| e.relative_error.abs() > self.config.error_threshold)
                .count(),
            warning_count: self.error_history.iter()
                .filter(|e| e.relative_error.abs() > self.config.warning_threshold)
                .count(),
        }
    }

    /// 重置累计量（例如在输出后）
    pub fn reset_cumulative(&mut self) {
        self.cumulative_boundary_flux.clear();
        self.cumulative_sources.clear();
    }

    fn merge_results(a: ConservationResult, b: ConservationResult) -> ConservationResult {
        match (a, b) {
            (ConservationResult::Error(mut ea), ConservationResult::Error(eb)) => {
                ea.extend(eb);
                ConservationResult::Error(ea)
            }
            (ConservationResult::Error(ea), _) => ConservationResult::Error(ea),
            (_, ConservationResult::Error(eb)) => ConservationResult::Error(eb),
            (ConservationResult::Warning(mut wa), ConservationResult::Warning(wb)) => {
                wa.extend(wb);
                ConservationResult::Warning(wa)
            }
            (ConservationResult::Warning(wa), _) => ConservationResult::Warning(wa),
            (_, ConservationResult::Warning(wb)) => ConservationResult::Warning(wb),
            (ConservationResult::NotInitialized, other) => other,
            (other, ConservationResult::NotInitialized) => other,
            _ => ConservationResult::Ok,
        }
    }
}

// ============================================================================
// 守恒误差
// ============================================================================

/// 守恒误差记录
#[derive(Debug, Clone)]
pub struct ConservationError {
    /// 时间
    pub time: f64,
    /// 守恒量类型
    pub quantity: ConservationType,
    /// 初始值
    pub initial_value: f64,
    /// 当前值
    pub current_value: f64,
    /// 预期值
    pub expected_value: f64,
    /// 绝对误差
    pub absolute_error: f64,
    /// 相对误差
    pub relative_error: f64,
    /// 边界贡献
    pub boundary_contribution: f64,
    /// 源项贡献
    pub source_contribution: f64,
}

impl std::fmt::Display for ConservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[t={:.2}s] {} 守恒误差: {:.2e} (相对: {:.2e}%)",
            self.time,
            self.quantity,
            self.absolute_error,
            self.relative_error * 100.0
        )
    }
}

/// 守恒检查结果
#[derive(Debug)]
pub enum ConservationResult {
    /// 正常
    Ok,
    /// 未初始化
    NotInitialized,
    /// 警告级别违反
    Warning(Vec<ConservationError>),
    /// 错误级别违反
    Error(Vec<ConservationError>),
}

impl ConservationResult {
    /// 是否正常
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }

    /// 是否有错误
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

// ============================================================================
// 回调与求解器集成
// ============================================================================

/// 守恒性检查结果回调
pub trait ConservationCallback: Send + Sync {
    /// 守恒性警告触发
    fn on_warning(&self, error: &ConservationError);

    /// 守恒性错误触发（可能需要回退）
    fn on_error(&self, error: &ConservationError) -> ConservationAction;

    /// 每步后的诊断输出
    fn on_step_complete(&self, snapshot: &ConservationSnapshot);
}

/// 守恒性违反后的动作
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConservationAction {
    /// 继续计算
    Continue,
    /// 减小时间步重算
    ReduceTimestep,
    /// 回退到上一时刻
    Rollback,
    /// 停止计算
    Abort,
}

/// 默认回调实现（日志记录）
pub struct LoggingConservationCallback {
    log_level: LogLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Quiet,
    Normal,
    Verbose,
}

impl Default for LoggingConservationCallback {
    fn default() -> Self {
        Self { log_level: LogLevel::Normal }
    }
}

impl ConservationCallback for LoggingConservationCallback {
    fn on_warning(&self, error: &ConservationError) {
        if !matches!(self.log_level, LogLevel::Quiet) {
            eprintln!("[WARN] {}", error);
        }
    }

    fn on_error(&self, error: &ConservationError) -> ConservationAction {
        eprintln!(
            "[ERROR] Conservation violation: {} 相对误差 {:.2e}",
            error.quantity, error.relative_error
        );
        ConservationAction::ReduceTimestep
    }

    fn on_step_complete(&self, snapshot: &ConservationSnapshot) {
        if matches!(self.log_level, LogLevel::Verbose) {
            eprintln!("[INFO] Conservation snapshot at t={:.4}", snapshot.time);
        }
    }
}

/// 带回调的守恒监测器（静态分发）
pub struct IntegratedConservationMonitor<C: ConservationCallback> {
    inner: ConservationMonitor,
    callbacks: Vec<C>,
}

impl<C: ConservationCallback> IntegratedConservationMonitor<C> {
    pub fn new(config: ConservationConfig) -> Self {
        Self {
            inner: ConservationMonitor::new(config),
            callbacks: Vec::new(),
        }
    }

    pub fn add_callback(&mut self, callback: C) {
        self.callbacks.push(callback);
    }

    /// 更新并触发回调
    pub fn update_with_callbacks(
        &mut self,
        snapshot: ConservationSnapshot,
        dt: f64,
    ) -> ConservationAction {
        let result = self.inner.update(snapshot.clone(), dt);

        for cb in &self.callbacks {
            cb.on_step_complete(&snapshot);
        }

        match result {
            ConservationResult::Ok | ConservationResult::NotInitialized => ConservationAction::Continue,
            ConservationResult::Warning(errors) => {
                for err in &errors {
                    for cb in &self.callbacks {
                        cb.on_warning(err);
                    }
                }
                ConservationAction::Continue
            }
            ConservationResult::Error(errors) => {
                let mut action = ConservationAction::Continue;
                for err in &errors {
                    for cb in &self.callbacks {
                        let cb_action = cb.on_error(err);
                        action = pick_stronger_action(action, cb_action);
                    }
                }
                action
            }
        }
    }
}

fn pick_stronger_action(a: ConservationAction, b: ConservationAction) -> ConservationAction {
    use ConservationAction::*;
    match (a, b) {
        (Abort, _) | (_, Abort) => Abort,
        (Rollback, _) | (_, Rollback) => Rollback,
        (ReduceTimestep, _) | (_, ReduceTimestep) => ReduceTimestep,
        _ => Continue,
    }
}

/// 求解器集成 trait
pub trait ConservationAwareSolver<C: ConservationCallback> {
    /// 设置守恒监测器
    fn set_conservation_monitor(&mut self, monitor: IntegratedConservationMonitor<C>);

    /// 获取当前状态快照
    fn create_conservation_snapshot(&self, time: f64) -> ConservationSnapshot;

    /// 执行时间步后的守恒检查
    fn check_conservation_post_step(&mut self, dt: f64) -> ConservationAction;
}

// ============================================================================
// 守恒报告
// ============================================================================

/// 守恒性报告
#[derive(Debug, Clone)]
pub struct ConservationReport {
    /// 步数
    pub step_count: usize,
    /// 初始时间
    pub initial_time: f64,
    /// 当前时间
    pub current_time: f64,
    /// 初始质量
    pub initial_mass: f64,
    /// 当前质量
    pub current_mass: f64,
    /// 累计流入
    pub cumulative_inflow: f64,
    /// 累计源项
    pub cumulative_sources: f64,
    /// 最大相对误差
    pub max_relative_error: f64,
    /// 错误次数
    pub error_count: usize,
    /// 警告次数
    pub warning_count: usize,
}

/// 质量平衡闭合状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosureStatus {
    /// 正常
    Ok,
    /// 期望变化近似为零
    ExpectedZero,
    /// 初始量过小
    InitialZero,
}

/// 质量平衡闭合结果
#[derive(Debug, Clone, Copy)]
pub struct ClosureResult {
    pub value: f64,
    pub status: ClosureStatus,
}

impl ConservationReport {
    /// 质量变化
    pub fn mass_change(&self) -> f64 {
        self.current_mass - self.initial_mass
    }

    /// 质量平衡闭合
    pub fn mass_balance_closure(&self) -> f64 {
        self.mass_balance_closure_result().value
    }

    /// 质量平衡闭合（带状态）
    pub fn mass_balance_closure_result(&self) -> ClosureResult {
        let expected_change = self.cumulative_inflow + self.cumulative_sources;
        let actual_change = self.mass_change();

        if expected_change.abs() > 1e-10 {
            ClosureResult {
                value: (actual_change - expected_change) / expected_change,
                status: ClosureStatus::Ok,
            }
        } else if actual_change.abs() > 1e-10 {
            ClosureResult {
                value: f64::INFINITY,
                status: ClosureStatus::ExpectedZero,
            }
        } else {
            ClosureResult {
                value: 0.0,
                status: ClosureStatus::InitialZero,
            }
        }
    }
}

impl std::fmt::Display for ConservationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== 守恒性报告 ===")?;
        writeln!(f, "模拟时间: {:.2}s - {:.2}s ({} 步)", 
            self.initial_time, self.current_time, self.step_count)?;
        writeln!(f, "初始质量: {:.6e} m³", self.initial_mass)?;
        writeln!(f, "当前质量: {:.6e} m³", self.current_mass)?;
        writeln!(f, "质量变化: {:.6e} m³ ({:+.4}%)", 
            self.mass_change(),
            if self.initial_mass.abs() > 1e-10 {
                100.0 * self.mass_change() / self.initial_mass
            } else { 0.0 }
        )?;
        writeln!(f, "累计流入: {:.6e} m³", self.cumulative_inflow)?;
        writeln!(f, "累计源项: {:.6e} m³", self.cumulative_sources)?;
        let closure = self.mass_balance_closure_result();
        writeln!(f, "质量平衡闭合: {:.2e} ({:?})", closure.value, closure.status)?;
        writeln!(f, "最大相对误差: {:.2e}", self.max_relative_error)?;
        writeln!(f, "警告/错误: {}/{}", self.warning_count, self.error_count)?;
        Ok(())
    }
}

// ============================================================================
// 积分计算辅助
// ============================================================================

/// 计算质量（体积）
pub fn compute_mass(h: &[f64], areas: &[f64]) -> f64 {
    h.iter().zip(areas.iter())
        .map(|(&h_val, &area)| h_val.max(0.0) * area)
        .sum()
}

/// 计算动量
pub fn compute_momentum(hu: &[f64], areas: &[f64]) -> f64 {
    hu.iter().zip(areas.iter())
        .map(|(hu_val, area)| hu_val * area)
        .sum()
}

/// 计算动能
pub fn compute_kinetic_energy(h: &[f64], hu: &[f64], hv: &[f64], areas: &[f64]) -> f64 {
    let mut ke = 0.0;
    for i in 0..h.len() {
        let h_val = h[i];
        if h_val > 1e-6 {
            let u = hu[i] / h_val;
            let v = hv[i] / h_val;
            let speed_sq = u * u + v * v;
            ke += 0.5 * h_val * speed_sq * areas[i];
        }
    }
    ke
}

/// 计算势能
pub fn compute_potential_energy(h: &[f64], z: &[f64], areas: &[f64], g: f64) -> f64 {
    let mut pe = 0.0;
    for i in 0..h.len() {
        let _eta = h[i] + z[i]; // 水面高程
        let z_c = z[i] + h[i] / 2.0; // 质心高程
        pe += g * h[i] * z_c * areas[i];
    }
    pe
}

/// 计算恩斯特罗菲（2D）
pub fn compute_enstrophy(
    vorticity: &[f64],
    h: &[f64],
    areas: &[f64],
) -> f64 {
    vorticity.iter().zip(h.iter()).zip(areas.iter())
        .map(|((&omega, &h_val), &area)| {
            if h_val > 1e-6 {
                omega * omega * h_val * area
            } else {
                0.0
            }
        })
        .sum()
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservation_snapshot() {
        let mut snapshot = ConservationSnapshot::new(0.0);
        snapshot.set(ConservationType::Mass, 1000.0);
        snapshot.add_boundary_flux("inlet", ConservationType::Mass, 10.0);
        
        assert_eq!(snapshot.get(ConservationType::Mass), 1000.0);
        assert!(snapshot.boundary_fluxes.contains_key("inlet"));
    }

    #[test]
    fn test_conservation_monitor() {
        let config = ConservationConfig::default();
        let mut monitor = ConservationMonitor::new(config);
        
        // 设置初始状态
        let mut initial = ConservationSnapshot::new(0.0);
        initial.set(ConservationType::Mass, 1000.0);
        monitor.set_initial(initial);
        
        // 更新状态（无变化）
        let mut current = ConservationSnapshot::new(1.0);
        current.set(ConservationType::Mass, 1000.0);
        let result = monitor.update(current, 1.0);
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_compute_mass() {
        let h = vec![1.0, 2.0, 0.5];
        let areas = vec![10.0, 10.0, 10.0];
        let mass = compute_mass(&h, &areas);
        assert!((mass - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_kinetic_energy() {
        let h = vec![1.0, 1.0];
        let hu = vec![1.0, 0.0];
        let hv = vec![0.0, 1.0];
        let areas = vec![1.0, 1.0];
        
        let ke = compute_kinetic_energy(&h, &hu, &hv, &areas);
        // KE = 0.5 * h * (u² + v²) * area = 0.5 * 1 * 1 * 1 + 0.5 * 1 * 1 * 1 = 1.0
        assert!((ke - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_conservation_report() {
        let report = ConservationReport {
            step_count: 1000,
            initial_time: 0.0,
            current_time: 100.0,
            initial_mass: 1000.0,
            current_mass: 1010.0,
            cumulative_inflow: 10.0,
            cumulative_sources: 0.0,
            max_relative_error: 1e-8,
            error_count: 0,
            warning_count: 0,
        };
        
        assert!((report.mass_change() - 10.0).abs() < 1e-10);
        assert!((report.mass_balance_closure()).abs() < 1e-10);
    }
}
