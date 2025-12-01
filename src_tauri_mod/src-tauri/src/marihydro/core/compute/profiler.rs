//! GPU性能监控
//!
//! 提供GPU计算性能监控、统计和诊断功能

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// 性能计时器
#[derive(Debug)]
pub struct GpuTimer {
    /// 开始时间
    start: Option<Instant>,
    /// 累积时间
    accumulated: Duration,
    /// 调用次数
    count: u64,
    /// 历史记录（最近N次）
    history: VecDeque<Duration>,
    /// 历史记录容量
    history_capacity: usize,
}

impl GpuTimer {
    /// 创建新计时器
    pub fn new(history_capacity: usize) -> Self {
        Self {
            start: None,
            accumulated: Duration::ZERO,
            count: 0,
            history: VecDeque::with_capacity(history_capacity),
            history_capacity,
        }
    }
    
    /// 开始计时
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }
    
    /// 停止计时
    pub fn stop(&mut self) {
        if let Some(start) = self.start.take() {
            let elapsed = start.elapsed();
            self.accumulated += elapsed;
            self.count += 1;
            
            if self.history.len() >= self.history_capacity {
                self.history.pop_front();
            }
            self.history.push_back(elapsed);
        }
    }
    
    /// 获取累积时间
    pub fn total(&self) -> Duration {
        self.accumulated
    }
    
    /// 获取平均时间
    pub fn average(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.accumulated / self.count as u32
        }
    }
    
    /// 获取调用次数
    pub fn count(&self) -> u64 {
        self.count
    }
    
    /// 获取最近一次时间
    pub fn last(&self) -> Option<Duration> {
        self.history.back().copied()
    }
    
    /// 获取最近N次平均
    pub fn recent_average(&self) -> Duration {
        if self.history.is_empty() {
            Duration::ZERO
        } else {
            let total: Duration = self.history.iter().sum();
            total / self.history.len() as u32
        }
    }
    
    /// 获取最小时间
    pub fn min(&self) -> Option<Duration> {
        self.history.iter().min().copied()
    }
    
    /// 获取最大时间
    pub fn max(&self) -> Option<Duration> {
        self.history.iter().max().copied()
    }
    
    /// 重置
    pub fn reset(&mut self) {
        self.start = None;
        self.accumulated = Duration::ZERO;
        self.count = 0;
        self.history.clear();
    }
}

impl Default for GpuTimer {
    fn default() -> Self {
        Self::new(100)
    }
}

/// 性能计数器
#[derive(Debug, Default)]
pub struct GpuCounters {
    /// 梯度计算计时器
    pub gradient: GpuTimer,
    /// 限制器计时器
    pub limiter: GpuTimer,
    /// 重构计时器
    pub reconstruct: GpuTimer,
    /// HLLC计时器
    pub hllc: GpuTimer,
    /// 通量累积计时器
    pub accumulate: GpuTimer,
    /// 源项计时器
    pub source: GpuTimer,
    /// 时间积分计时器
    pub integrate: GpuTimer,
    /// 边界条件计时器
    pub boundary: GpuTimer,
    /// 数据传输计时器
    pub transfer: GpuTimer,
    /// 总计时器
    pub total: GpuTimer,
}

impl GpuCounters {
    /// 创建新计数器
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 重置所有计数器
    pub fn reset_all(&mut self) {
        self.gradient.reset();
        self.limiter.reset();
        self.reconstruct.reset();
        self.hllc.reset();
        self.accumulate.reset();
        self.source.reset();
        self.integrate.reset();
        self.boundary.reset();
        self.transfer.reset();
        self.total.reset();
    }
    
    /// 获取摘要报告
    pub fn summary(&self) -> CounterSummary {
        CounterSummary {
            gradient_ms: self.gradient.average().as_secs_f64() * 1000.0,
            limiter_ms: self.limiter.average().as_secs_f64() * 1000.0,
            reconstruct_ms: self.reconstruct.average().as_secs_f64() * 1000.0,
            hllc_ms: self.hllc.average().as_secs_f64() * 1000.0,
            accumulate_ms: self.accumulate.average().as_secs_f64() * 1000.0,
            source_ms: self.source.average().as_secs_f64() * 1000.0,
            integrate_ms: self.integrate.average().as_secs_f64() * 1000.0,
            boundary_ms: self.boundary.average().as_secs_f64() * 1000.0,
            transfer_ms: self.transfer.average().as_secs_f64() * 1000.0,
            total_ms: self.total.average().as_secs_f64() * 1000.0,
            steps: self.total.count(),
        }
    }
}

/// 计数器摘要
#[derive(Debug, Clone)]
pub struct CounterSummary {
    pub gradient_ms: f64,
    pub limiter_ms: f64,
    pub reconstruct_ms: f64,
    pub hllc_ms: f64,
    pub accumulate_ms: f64,
    pub source_ms: f64,
    pub integrate_ms: f64,
    pub boundary_ms: f64,
    pub transfer_ms: f64,
    pub total_ms: f64,
    pub steps: u64,
}

impl CounterSummary {
    /// 计算时间百分比
    pub fn percentages(&self) -> Vec<(&'static str, f64)> {
        if self.total_ms < 1e-10 {
            return Vec::new();
        }
        
        vec![
            ("梯度计算", self.gradient_ms / self.total_ms * 100.0),
            ("限制器", self.limiter_ms / self.total_ms * 100.0),
            ("重构", self.reconstruct_ms / self.total_ms * 100.0),
            ("HLLC求解", self.hllc_ms / self.total_ms * 100.0),
            ("通量累积", self.accumulate_ms / self.total_ms * 100.0),
            ("源项", self.source_ms / self.total_ms * 100.0),
            ("时间积分", self.integrate_ms / self.total_ms * 100.0),
            ("边界条件", self.boundary_ms / self.total_ms * 100.0),
            ("数据传输", self.transfer_ms / self.total_ms * 100.0),
        ]
    }
    
    /// 生成报告字符串
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== GPU性能统计 ===\n");
        report.push_str(&format!("总步数: {}\n", self.steps));
        report.push_str(&format!("平均步时间: {:.3} ms\n\n", self.total_ms));
        
        report.push_str("各阶段耗时:\n");
        for (name, pct) in self.percentages() {
            report.push_str(&format!("  {:12}: {:5.1}%\n", name, pct));
        }
        
        let compute_ms = self.gradient_ms + self.limiter_ms + self.reconstruct_ms 
            + self.hllc_ms + self.accumulate_ms + self.source_ms + self.integrate_ms;
        report.push_str(&format!("\n计算/传输比: {:.1}:{:.1}\n", 
            compute_ms / self.total_ms.max(1e-10) * 100.0,
            self.transfer_ms / self.total_ms.max(1e-10) * 100.0));
        
        report
    }
}

/// 内存监控
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// 网格数据内存(字节)
    pub mesh_bytes: usize,
    /// 状态数据内存(字节)
    pub state_bytes: usize,
    /// 梯度数据内存(字节)
    pub gradient_bytes: usize,
    /// 通量数据内存(字节)
    pub flux_bytes: usize,
    /// 临时缓冲区内存(字节)
    pub temp_bytes: usize,
    /// 着色数据内存(字节)
    pub coloring_bytes: usize,
}

impl MemoryStats {
    /// 总内存使用
    pub fn total(&self) -> usize {
        self.mesh_bytes + self.state_bytes + self.gradient_bytes 
            + self.flux_bytes + self.temp_bytes + self.coloring_bytes
    }
    
    /// 格式化为MB
    pub fn total_mb(&self) -> f64 {
        self.total() as f64 / (1024.0 * 1024.0)
    }
    
    /// 生成报告
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== GPU内存使用 ===\n");
        report.push_str(&format!("网格数据: {:.2} MB\n", self.mesh_bytes as f64 / 1024.0 / 1024.0));
        report.push_str(&format!("状态数据: {:.2} MB\n", self.state_bytes as f64 / 1024.0 / 1024.0));
        report.push_str(&format!("梯度数据: {:.2} MB\n", self.gradient_bytes as f64 / 1024.0 / 1024.0));
        report.push_str(&format!("通量数据: {:.2} MB\n", self.flux_bytes as f64 / 1024.0 / 1024.0));
        report.push_str(&format!("临时缓冲: {:.2} MB\n", self.temp_bytes as f64 / 1024.0 / 1024.0));
        report.push_str(&format!("着色数据: {:.2} MB\n", self.coloring_bytes as f64 / 1024.0 / 1024.0));
        report.push_str(&format!("─────────────────\n"));
        report.push_str(&format!("总计: {:.2} MB\n", self.total_mb()));
        report
    }
    
    /// 估算给定单元数的内存需求
    pub fn estimate(num_cells: usize, num_faces: usize) -> Self {
        // f32 = 4字节, u32 = 4字节
        let cell_state = num_cells * 4 * 4;  // h, hu, hv, z
        let cell_grad = num_cells * 4 * 8;   // 4变量 * 2分量
        let face_flux = num_faces * 4 * 4;   // h, hu, hv, wave_speed
        let face_recon = num_faces * 4 * 8;  // L/R * 4变量
        
        Self {
            mesh_bytes: num_cells * 4 * 3 + num_faces * 4 * 7, // 几何数据
            state_bytes: cell_state,
            gradient_bytes: cell_grad,
            flux_bytes: face_flux + face_recon,
            temp_bytes: num_cells * 4 * 6 + num_faces * 4 * 2, // RK中间状态等
            coloring_bytes: num_faces * 4 + 1024, // 着色索引
        }
    }
}

/// GPU性能监控器
#[derive(Debug)]
pub struct GpuProfiler {
    /// 计数器
    pub counters: GpuCounters,
    /// 内存统计
    pub memory: MemoryStats,
    /// 是否启用
    enabled: bool,
    /// 采样间隔（每N步采样一次）
    sample_interval: u64,
    /// 当前步数
    current_step: u64,
}

impl GpuProfiler {
    /// 创建新监控器
    pub fn new() -> Self {
        Self {
            counters: GpuCounters::new(),
            memory: MemoryStats::default(),
            enabled: true,
            sample_interval: 1,
            current_step: 0,
        }
    }
    
    /// 设置是否启用
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// 设置采样间隔
    pub fn set_sample_interval(&mut self, interval: u64) {
        self.sample_interval = interval.max(1);
    }
    
    /// 是否应该采样当前步
    pub fn should_sample(&self) -> bool {
        self.enabled && (self.current_step % self.sample_interval == 0)
    }
    
    /// 开始新的一步
    pub fn begin_step(&mut self) {
        if self.should_sample() {
            self.counters.total.start();
        }
    }
    
    /// 结束当前步
    pub fn end_step(&mut self) {
        if self.should_sample() {
            self.counters.total.stop();
        }
        self.current_step += 1;
    }
    
    /// 获取当前步数
    pub fn step_count(&self) -> u64 {
        self.current_step
    }
    
    /// 获取每秒步数（基于最近平均）
    pub fn steps_per_second(&self) -> f64 {
        let avg = self.counters.total.recent_average();
        if avg.as_secs_f64() > 0.0 {
            1.0 / avg.as_secs_f64()
        } else {
            0.0
        }
    }
    
    /// 更新内存统计
    pub fn update_memory(&mut self, stats: MemoryStats) {
        self.memory = stats;
    }
    
    /// 重置统计
    pub fn reset(&mut self) {
        self.counters.reset_all();
        self.current_step = 0;
    }
    
    /// 生成完整报告
    pub fn full_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&self.counters.summary().report());
        report.push_str("\n");
        report.push_str(&self.memory.report());
        report.push_str(&format!("\n吞吐量: {:.1} steps/s\n", self.steps_per_second()));
        report
    }
}

impl Default for GpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// 作用域计时宏
#[macro_export]
macro_rules! gpu_timer_scope {
    ($profiler:expr, $timer:ident) => {
        if $profiler.should_sample() {
            $profiler.counters.$timer.start();
        }
        scopeguard::defer! {
            if $profiler.should_sample() {
                $profiler.counters.$timer.stop();
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_timer() {
        let mut timer = GpuTimer::new(10);
        
        timer.start();
        thread::sleep(Duration::from_millis(10));
        timer.stop();
        
        assert!(timer.total() >= Duration::from_millis(10));
        assert_eq!(timer.count(), 1);
    }
    
    #[test]
    fn test_profiler() {
        let mut profiler = GpuProfiler::new();
        
        profiler.begin_step();
        thread::sleep(Duration::from_millis(5));
        profiler.end_step();
        
        assert_eq!(profiler.step_count(), 1);
        assert!(profiler.counters.total.count() > 0);
    }
    
    #[test]
    fn test_memory_estimate() {
        let stats = MemoryStats::estimate(100_000, 150_000);
        assert!(stats.total() > 0);
        println!("Estimated memory: {:.2} MB", stats.total_mb());
    }
    
    #[test]
    fn test_summary_report() {
        let mut counters = GpuCounters::new();
        counters.gradient.start();
        thread::sleep(Duration::from_millis(2));
        counters.gradient.stop();
        
        counters.total.start();
        thread::sleep(Duration::from_millis(5));
        counters.total.stop();
        
        let summary = counters.summary();
        let report = summary.report();
        assert!(!report.is_empty());
        println!("{}", report);
    }
}
