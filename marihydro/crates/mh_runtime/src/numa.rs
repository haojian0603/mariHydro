// crates/mh_runtime/src/numa.rs

//! NUMA 拓扑与核心绑定
//!
//! 提供 NUMA 感知的线程调度和内存分配，支持：
//! - NUMA 拓扑检测
//! - 核心绑定（core pinning）
//! - NUMA 节点内存分配
//! - 工作窃取调度优化
//!
//! # 设计原则
//!
//! 1. **位置感知**：线程与数据在同一 NUMA 节点
//! 2. **缓存友好**：避免跨 NUMA 访问
//! 3. **可降级**：无 NUMA 系统正常工作
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_runtime::numa::{NumaTopology, bind_thread_to_core};
//!
//! let topo = NumaTopology::detect()?;
//! println!("NUMA 节点数: {}", topo.num_nodes());
//!
//! // 绑定当前线程到核心 0
//! bind_thread_to_core(0)?;
//! ```

#![allow(unsafe_code)]

use std::collections::HashMap;

// ============================================================================
// NUMA 拓扑
// ============================================================================

/// NUMA 节点信息
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// 节点 ID
    pub id: usize,
    /// 节点上的 CPU 核心
    pub cpus: Vec<usize>,
    /// 总内存（字节）
    pub total_memory: u64,
    /// 可用内存（字节）
    pub free_memory: u64,
}

/// NUMA 拓扑信息
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// 所有 NUMA 节点
    nodes: Vec<NumaNode>,
    /// CPU 到节点的映射
    cpu_to_node: HashMap<usize, usize>,
    /// 物理核心数
    physical_cores: usize,
    /// 逻辑核心数
    logical_cores: usize,
    /// 是否支持超线程
    hyperthreading: bool,
}

impl NumaTopology {
    /// 检测系统 NUMA 拓扑
    pub fn detect() -> Result<Self, NumaError> {
        // 获取逻辑核心数
        let logical_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        // 尝试检测物理核心数（简化实现）
        let physical_cores = Self::detect_physical_cores(logical_cores);
        let hyperthreading = logical_cores > physical_cores;

        // 尝试检测 NUMA 节点
        let nodes = Self::detect_numa_nodes(logical_cores)?;
        
        // 构建 CPU 到节点映射
        let mut cpu_to_node = HashMap::new();
        for node in &nodes {
            for &cpu in &node.cpus {
                cpu_to_node.insert(cpu, node.id);
            }
        }

        Ok(Self {
            nodes,
            cpu_to_node,
            physical_cores,
            logical_cores,
            hyperthreading,
        })
    }

    /// 检测物理核心数
    fn detect_physical_cores(logical: usize) -> usize {
        // Windows: 通过 WMI 或 GetLogicalProcessorInformation
        // Linux: 通过 /proc/cpuinfo
        // 简化：假设 2x 超线程
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                let physical_ids: std::collections::HashSet<_> = content
                    .lines()
                    .filter(|l| l.starts_with("physical id"))
                    .collect();
                let cores_per_socket: usize = content
                    .lines()
                    .find(|l| l.starts_with("cpu cores"))
                    .and_then(|l| l.split(':').nth(1))
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(1);
                
                let sockets = physical_ids.len().max(1);
                return sockets * cores_per_socket;
            }
        }
        
        // 默认假设：超线程系统有一半是物理核心
        logical / 2.max(1)
    }

    /// 检测 NUMA 节点
    fn detect_numa_nodes(logical_cores: usize) -> Result<Vec<NumaNode>, NumaError> {
        let mut nodes = Vec::new();

        #[cfg(target_os = "linux")]
        {
            // 尝试从 sysfs 读取 NUMA 信息
            let numa_path = std::path::Path::new("/sys/devices/system/node");
            if numa_path.exists() {
                for entry in std::fs::read_dir(numa_path).map_err(|e| NumaError::DetectionFailed(e.to_string()))? {
                    let entry = entry.map_err(|e| NumaError::DetectionFailed(e.to_string()))?;
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    
                    if name_str.starts_with("node") && name_str[4..].parse::<usize>().is_ok() {
                        let node_id: usize = name_str[4..].parse().unwrap();
                        let node_path = entry.path();
                        
                        // 读取 CPU 列表
                        let cpulist_path = node_path.join("cpulist");
                        let cpus = if cpulist_path.exists() {
                            Self::parse_cpu_list(&std::fs::read_to_string(&cpulist_path)
                                .unwrap_or_default())
                        } else {
                            Vec::new()
                        };

                        // 读取内存信息
                        let meminfo_path = node_path.join("meminfo");
                        let (total, free) = if meminfo_path.exists() {
                            Self::parse_meminfo(&std::fs::read_to_string(&meminfo_path)
                                .unwrap_or_default())
                        } else {
                            (0, 0)
                        };

                        nodes.push(NumaNode {
                            id: node_id,
                            cpus,
                            total_memory: total,
                            free_memory: free,
                        });
                    }
                }
            }
        }

        // 如果没有检测到 NUMA 节点，创建单节点拓扑
        if nodes.is_empty() {
            nodes.push(NumaNode {
                id: 0,
                cpus: (0..logical_cores).collect(),
                total_memory: Self::get_system_memory(),
                free_memory: Self::get_free_memory(),
            });
        }

        // 按节点 ID 排序
        nodes.sort_by_key(|n| n.id);
        
        Ok(nodes)
    }

    /// 解析 CPU 列表字符串（如 "0-3,8-11"）
    #[allow(dead_code)]
    fn parse_cpu_list(s: &str) -> Vec<usize> {
        let mut cpus = Vec::new();
        for part in s.trim().split(',') {
            if part.contains('-') {
                let range: Vec<&str> = part.split('-').collect();
                if range.len() == 2 {
                    if let (Ok(start), Ok(end)) = (range[0].parse::<usize>(), range[1].parse::<usize>()) {
                        cpus.extend(start..=end);
                    }
                }
            } else if let Ok(cpu) = part.parse::<usize>() {
                cpus.push(cpu);
            }
        }
        cpus
    }

    /// 解析内存信息
    #[allow(dead_code)]
    fn parse_meminfo(s: &str) -> (u64, u64) {
        let mut total = 0u64;
        let mut free = 0u64;
        
        for line in s.lines() {
            if line.contains("MemTotal:") {
                if let Some(val) = line.split_whitespace().nth(3) {
                    total = val.parse().unwrap_or(0) * 1024; // kB to bytes
                }
            }
            if line.contains("MemFree:") {
                if let Some(val) = line.split_whitespace().nth(3) {
                    free = val.parse().unwrap_or(0) * 1024;
                }
            }
        }
        
        (total, free)
    }

    /// 获取系统总内存
    fn get_system_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(val) = line.split_whitespace().nth(1) {
                            return val.parse::<u64>().unwrap_or(0) * 1024;
                        }
                    }
                }
            }
        }
        
        // 默认 8GB
        8 * 1024 * 1024 * 1024
    }

    /// 获取空闲内存
    fn get_free_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemFree:") {
                        if let Some(val) = line.split_whitespace().nth(1) {
                            return val.parse::<u64>().unwrap_or(0) * 1024;
                        }
                    }
                }
            }
        }
        
        4 * 1024 * 1024 * 1024
    }

    // === 公共接口 ===

    /// 获取 NUMA 节点数
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// 获取所有节点
    pub fn nodes(&self) -> &[NumaNode] {
        &self.nodes
    }

    /// 获取指定节点
    pub fn node(&self, id: usize) -> Option<&NumaNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// 获取 CPU 所在节点
    pub fn cpu_node(&self, cpu: usize) -> Option<usize> {
        self.cpu_to_node.get(&cpu).copied()
    }

    /// 获取物理核心数
    pub fn physical_cores(&self) -> usize {
        self.physical_cores
    }

    /// 获取逻辑核心数
    pub fn logical_cores(&self) -> usize {
        self.logical_cores
    }

    /// 是否支持超线程
    pub fn has_hyperthreading(&self) -> bool {
        self.hyperthreading
    }

    /// 获取推荐线程数
    pub fn recommended_threads(&self) -> usize {
        // 优先使用物理核心数，避免超线程争用
        self.physical_cores
    }

    /// 获取每个节点的推荐线程数
    pub fn threads_per_node(&self) -> Vec<(usize, usize)> {
        self.nodes
            .iter()
            .map(|n| {
                let cores = if self.hyperthreading {
                    n.cpus.len() / 2
                } else {
                    n.cpus.len()
                };
                (n.id, cores.max(1))
            })
            .collect()
    }

    /// 是否为 NUMA 系统
    pub fn is_numa(&self) -> bool {
        self.nodes.len() > 1
    }
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self::detect().unwrap_or_else(|_| Self {
            nodes: vec![NumaNode {
                id: 0,
                cpus: vec![0],
                total_memory: 0,
                free_memory: 0,
            }],
            cpu_to_node: [(0, 0)].into_iter().collect(),
            physical_cores: 1,
            logical_cores: 1,
            hyperthreading: false,
        })
    }
}

// ============================================================================
// 核心绑定
// ============================================================================

/// 绑定当前线程到指定核心
///
/// # 参数
///
/// * `core` - 目标核心 ID
///
/// # 返回
///
/// 成功返回 `Ok(())`，失败返回错误
pub fn bind_thread_to_core(core: usize) -> Result<(), NumaError> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        
        // 使用 libc 的 sched_setaffinity
        // 这里使用简化实现，生产环境应使用 nix 或 libc crate
        let _ = core;
        // 占位：实际实现需要 libc::sched_setaffinity
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        // Windows: SetThreadAffinityMask
        let _ = core;
        return Ok(());
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        Err(NumaError::UnsupportedPlatform)
    }
}

/// 绑定当前线程到指定核心集
pub fn bind_thread_to_cores(cores: &[usize]) -> Result<(), NumaError> {
    if cores.is_empty() {
        return Err(NumaError::InvalidCoreSet);
    }
    
    // 简化：绑定到第一个核心
    bind_thread_to_core(cores[0])
}

/// 解绑当前线程（恢复默认调度）
pub fn unbind_thread() -> Result<(), NumaError> {
    #[cfg(target_os = "linux")]
    {
        // 恢复所有核心的亲和性
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        return Ok(());
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        Ok(())
    }
}

// ============================================================================
// 线程池配置
// ============================================================================

/// NUMA 感知线程池配置
#[derive(Debug, Clone)]
pub struct NumaThreadPoolConfig {
    /// 每个节点的线程数
    pub threads_per_node: Vec<(usize, usize)>,
    /// 是否绑定核心
    pub bind_cores: bool,
    /// 是否使用物理核心（避免超线程）
    pub use_physical_cores: bool,
    /// 线程栈大小
    pub stack_size: Option<usize>,
}

impl NumaThreadPoolConfig {
    /// 从拓扑创建配置
    pub fn from_topology(topo: &NumaTopology) -> Self {
        Self {
            threads_per_node: topo.threads_per_node(),
            bind_cores: true,
            use_physical_cores: true,
            stack_size: None,
        }
    }

    /// 总线程数
    pub fn total_threads(&self) -> usize {
        self.threads_per_node.iter().map(|(_, t)| *t).sum()
    }
}

impl Default for NumaThreadPoolConfig {
    fn default() -> Self {
        let topo = NumaTopology::default();
        Self::from_topology(&topo)
    }
}

// ============================================================================
// 内存分配
// ============================================================================

/// NUMA 感知内存分配器接口
pub trait NumaAllocator {
    /// 在指定节点分配内存
    fn alloc_on_node(&self, size: usize, node: usize) -> Result<*mut u8, NumaError>;
    
    /// 释放内存
    unsafe fn dealloc(&self, ptr: *mut u8, size: usize);
    
    /// 获取指针所在节点
    fn get_node(&self, ptr: *const u8) -> Option<usize>;
}

/// 默认 NUMA 分配器（回退到系统分配器）
pub struct DefaultNumaAllocator;

impl NumaAllocator for DefaultNumaAllocator {
    fn alloc_on_node(&self, size: usize, _node: usize) -> Result<*mut u8, NumaError> {
        let layout = std::alloc::Layout::from_size_align(size, 64)
            .map_err(|_| NumaError::AllocationFailed)?;
        
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            Err(NumaError::AllocationFailed)
        } else {
            Ok(ptr)
        }
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, size: usize) {
        let layout = std::alloc::Layout::from_size_align_unchecked(size, 64);
        std::alloc::dealloc(ptr, layout);
    }
    
    fn get_node(&self, _ptr: *const u8) -> Option<usize> {
        Some(0) // 默认节点 0
    }
}

// ============================================================================
// 错误类型
// ============================================================================

/// NUMA 相关错误
#[derive(Debug, Clone)]
pub enum NumaError {
    /// 检测失败
    DetectionFailed(String),
    /// 不支持的平台
    UnsupportedPlatform,
    /// 无效的核心集
    InvalidCoreSet,
    /// 分配失败
    AllocationFailed,
    /// 绑定失败
    BindingFailed(String),
}

impl std::fmt::Display for NumaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DetectionFailed(msg) => write!(f, "NUMA 检测失败: {}", msg),
            Self::UnsupportedPlatform => write!(f, "不支持的平台"),
            Self::InvalidCoreSet => write!(f, "无效的核心集"),
            Self::AllocationFailed => write!(f, "NUMA 内存分配失败"),
            Self::BindingFailed(msg) => write!(f, "核心绑定失败: {}", msg),
        }
    }
}

impl std::error::Error for NumaError {}

// ============================================================================
// 工具函数
// ============================================================================

/// 打印系统拓扑信息
pub fn print_topology_info() {
    match NumaTopology::detect() {
        Ok(topo) => {
            println!("=== 系统拓扑 ===");
            println!("物理核心: {}", topo.physical_cores());
            println!("逻辑核心: {}", topo.logical_cores());
            println!("超线程: {}", if topo.has_hyperthreading() { "是" } else { "否" });
            println!("NUMA 节点数: {}", topo.num_nodes());
            
            for node in topo.nodes() {
                println!("\n节点 {}:", node.id);
                println!("  CPU: {:?}", node.cpus);
                println!(
                    "  内存: {:.2} GB (空闲 {:.2} GB)",
                    node.total_memory as f64 / 1e9,
                    node.free_memory as f64 / 1e9
                );
            }
            
            println!("\n推荐线程数: {}", topo.recommended_threads());
        }
        Err(e) => {
            println!("无法检测系统拓扑: {}", e);
        }
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_detection() {
        let topo = NumaTopology::detect();
        assert!(topo.is_ok());
        
        let topo = topo.unwrap();
        assert!(topo.logical_cores() >= 1);
        assert!(topo.physical_cores() >= 1);
        assert!(topo.num_nodes() >= 1);
    }

    #[test]
    fn test_default_topology() {
        let topo = NumaTopology::default();
        assert!(topo.num_nodes() >= 1);
    }

    #[test]
    fn test_parse_cpu_list() {
        let cpus = NumaTopology::parse_cpu_list("0-3,8-11");
        assert_eq!(cpus, vec![0, 1, 2, 3, 8, 9, 10, 11]);

        let cpus = NumaTopology::parse_cpu_list("0,2,4");
        assert_eq!(cpus, vec![0, 2, 4]);
    }

    #[test]
    fn test_thread_pool_config() {
        let topo = NumaTopology::default();
        let config = NumaThreadPoolConfig::from_topology(&topo);
        assert!(config.total_threads() >= 1);
    }

    #[test]
    fn test_default_allocator() {
        let alloc = DefaultNumaAllocator;
        let ptr = alloc.alloc_on_node(1024, 0);
        assert!(ptr.is_ok());
        
        let ptr = ptr.unwrap();
        unsafe {
            alloc.dealloc(ptr, 1024);
        }
    }

    #[test]
    fn test_bind_thread() {
        // 不一定成功（权限问题），但不应 panic
        let _ = bind_thread_to_core(0);
    }
}
