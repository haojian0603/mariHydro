// crates/mh_physics/src/gpu/capabilities.rs

//! 设备能力描述
//!
//! 定义计算设备的能力和特性。

/// 设备类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU
    Cpu,
    /// 独立显卡
    DiscreteGpu,
    /// 集成显卡
    IntegratedGpu,
    /// 虚拟GPU
    VirtualGpu,
    /// 软件渲染
    Software,
    /// 未知类型
    Unknown,
}

impl DeviceType {
    /// 是否为GPU类型
    pub fn is_gpu(&self) -> bool {
        matches!(
            self,
            DeviceType::DiscreteGpu | DeviceType::IntegratedGpu | DeviceType::VirtualGpu
        )
    }

    /// 是否为高性能设备
    pub fn is_high_performance(&self) -> bool {
        matches!(self, DeviceType::DiscreteGpu | DeviceType::Cpu)
    }
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::DiscreteGpu => write!(f, "Discrete GPU"),
            DeviceType::IntegratedGpu => write!(f, "Integrated GPU"),
            DeviceType::VirtualGpu => write!(f, "Virtual GPU"),
            DeviceType::Software => write!(f, "Software"),
            DeviceType::Unknown => write!(f, "Unknown"),
        }
    }
}

/// 设备能力描述
///
/// 描述计算设备的硬件能力和限制。
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// 设备名称
    pub name: String,
    /// 厂商名称
    pub vendor: String,
    /// 设备类型
    pub device_type: DeviceType,
    /// 计算单元数
    pub compute_units: u32,
    /// 工作组最大尺寸
    pub max_workgroup_size: u32,
    /// 各维度最大工作组数
    pub max_workgroups: [u32; 3],
    /// 内存信息
    pub memory: MemoryInfo,
    /// 是否支持 f64
    pub supports_f64: bool,
    /// 是否支持 f16
    pub supports_f16: bool,
    /// 是否支持原子浮点
    pub supports_atomic_float: bool,
    /// 内存带宽（GB/s）
    pub memory_bandwidth_gbps: f64,
}

impl DeviceCapabilities {
    /// 获取计算特性（兼容旧接口）
    pub fn features(&self) -> ComputeFeatures {
        ComputeFeatures {
            supports_f64: self.supports_f64,
            supports_f16: self.supports_f16,
            supports_atomic_float: self.supports_atomic_float,
            supports_subgroups: false,
            max_subgroup_size: 32,
        }
    }
}

/// 内存信息
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// 总内存（字节）
    pub total_bytes: usize,
    /// 可用内存（字节）
    pub available_bytes: usize,
    /// 最大缓冲区尺寸（字节）
    pub max_buffer_size: usize,
}

/// 计算特性
#[derive(Debug, Clone)]
pub struct ComputeFeatures {
    /// 是否支持f64双精度浮点
    pub supports_f64: bool,
    /// 是否支持f16半精度浮点
    pub supports_f16: bool,
    /// 是否支持原子浮点操作
    pub supports_atomic_float: bool,
    /// 是否支持子组操作
    pub supports_subgroups: bool,
    /// 最大子组大小
    pub max_subgroup_size: u32,
}

impl DeviceCapabilities {
    /// 创建CPU设备能力（默认值）
    pub fn cpu_default() -> Self {
        let num_threads = rayon::current_num_threads();
        Self {
            name: format!("CPU ({} threads)", num_threads),
            vendor: "CPU".to_string(),
            device_type: DeviceType::Cpu,
            compute_units: num_threads as u32,
            max_workgroup_size: 1,
            max_workgroups: [u32::MAX, u32::MAX, u32::MAX],
            memory: MemoryInfo {
                total_bytes: usize::MAX,
                available_bytes: usize::MAX,
                max_buffer_size: usize::MAX,
            },
            supports_f64: true,
            supports_f16: false,
            supports_atomic_float: false,
            memory_bandwidth_gbps: 50.0, // 典型DDR4带宽估计
        }
    }

    /// 获取最大并行线程数
    pub fn max_threads(&self) -> usize {
        (self.compute_units as usize) * (self.max_workgroup_size as usize)
    }

    /// 获取最大内存
    pub fn max_memory_bytes(&self) -> usize {
        self.memory.total_bytes
    }

    /// 是否支持f64
    pub fn supports_f64(&self) -> bool {
        self.supports_f64
    }

    /// 是否支持原子操作
    pub fn supports_atomics(&self) -> bool {
        true // 所有现代设备都支持整数原子
    }

    /// 估算可处理的最大单元数
    ///
    /// 基于可用内存估算，假设每单元需要约200字节工作空间
    pub fn max_cells(&self) -> usize {
        const BYTES_PER_CELL: usize = 200;
        self.memory.available_bytes / BYTES_PER_CELL
    }

    /// 检查是否可以处理给定规模的问题
    pub fn can_handle(&self, n_cells: usize, n_faces: usize) -> bool {
        // 估算内存需求
        let state_memory = n_cells * 4 * 8; // h, hu, hv, z
        let workspace_memory = n_cells * 10 * 8; // 各种工作缓冲区
        let mesh_memory = n_faces * 8 * 8; // 面几何数据
        let total = state_memory + workspace_memory + mesh_memory;

        total < self.memory.available_bytes
    }

    /// 建议的最小问题规模（低于此规模GPU可能无优势）
    pub fn min_efficient_size(&self) -> usize {
        match self.device_type {
            DeviceType::DiscreteGpu => 50_000,
            DeviceType::IntegratedGpu | DeviceType::VirtualGpu => 20_000,
            _ => 0,
        }
    }

    /// 是否推荐用于给定问题规模
    pub fn recommended_for_size(&self, problem_size: usize) -> bool {
        match self.device_type {
            DeviceType::Cpu => problem_size < 100_000,
            DeviceType::DiscreteGpu => problem_size >= 50_000,
            DeviceType::IntegratedGpu | DeviceType::VirtualGpu => {
                problem_size >= 20_000 && problem_size < 500_000
            }
            _ => false,
        }
    }

    /// 从 wgpu 适配器创建能力描述
    pub fn from_wgpu(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let limits = adapter.limits();

        let device_type = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => DeviceType::DiscreteGpu,
            wgpu::DeviceType::IntegratedGpu => DeviceType::IntegratedGpu,
            wgpu::DeviceType::VirtualGpu => DeviceType::VirtualGpu,
            wgpu::DeviceType::Cpu => DeviceType::Cpu,
            wgpu::DeviceType::Other => DeviceType::Unknown,
        };

        let vendor = match info.vendor {
            0x1002 => "AMD".to_string(),
            0x10DE => "NVIDIA".to_string(),
            0x8086 => "Intel".to_string(),
            0x13B5 => "ARM".to_string(),
            0x5143 => "Qualcomm".to_string(),
            0x1010 => "ImgTec".to_string(),
            _ => format!("Unknown (0x{:04X})", info.vendor),
        };

        // 估算计算单元数（wgpu 不直接提供）
        let compute_units = match device_type {
            DeviceType::DiscreteGpu => 64,  // 典型值
            DeviceType::IntegratedGpu => 24,
            _ => 4,
        };

        // 估算内存带宽
        let memory_bandwidth_gbps = match device_type {
            DeviceType::DiscreteGpu => 400.0,
            DeviceType::IntegratedGpu => 50.0,
            _ => 20.0,
        };

        Self {
            name: info.name.clone(),
            vendor,
            device_type,
            compute_units,
            max_workgroup_size: limits.max_compute_invocations_per_workgroup,
            max_workgroups: [
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
            ],
            memory: MemoryInfo {
                total_bytes: limits.max_buffer_size as usize,
                available_bytes: limits.max_buffer_size as usize / 2,
                max_buffer_size: limits.max_buffer_size as usize,
            },
            supports_f64: false, // wgpu/WebGPU 不支持 f64
            supports_f16: true,
            supports_atomic_float: false,
            memory_bandwidth_gbps,
        }
    }

    /// 设备是否适合用于计算
    pub fn is_suitable(&self) -> bool {
        self.device_type.is_gpu() && self.max_workgroup_size >= 64
    }

    /// 获取设备名称
    pub fn device_name(&self) -> &str {
        &self.name
    }
}

impl MemoryInfo {
    /// 获取总内存（MB）
    pub fn total_mb(&self) -> usize {
        self.total_bytes / (1024 * 1024)
    }

    /// 获取可用内存（MB）
    pub fn available_mb(&self) -> usize {
        self.available_bytes / (1024 * 1024)
    }
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self::cpu_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_is_gpu() {
        assert!(!DeviceType::Cpu.is_gpu());
        assert!(DeviceType::DiscreteGpu.is_gpu());
        assert!(DeviceType::IntegratedGpu.is_gpu());
        assert!(DeviceType::VirtualGpu.is_gpu());
        assert!(!DeviceType::Software.is_gpu());
        assert!(!DeviceType::Unknown.is_gpu());
    }

    #[test]
    fn test_device_type_is_high_performance() {
        assert!(DeviceType::Cpu.is_high_performance());
        assert!(DeviceType::DiscreteGpu.is_high_performance());
        assert!(!DeviceType::IntegratedGpu.is_high_performance());
    }

    #[test]
    fn test_cpu_capabilities() {
        let cap = DeviceCapabilities::cpu_default();
        assert_eq!(cap.device_type, DeviceType::Cpu);
        assert!(cap.supports_f64());
        assert!(cap.max_threads() > 0);
    }

    #[test]
    fn test_device_type_display() {
        assert_eq!(format!("{}", DeviceType::Cpu), "CPU");
        assert_eq!(format!("{}", DeviceType::DiscreteGpu), "Discrete GPU");
        assert_eq!(format!("{}", DeviceType::IntegratedGpu), "Integrated GPU");
    }

    #[test]
    fn test_can_handle() {
        let cap = DeviceCapabilities::cpu_default();
        assert!(cap.can_handle(100_000, 200_000));
    }

    #[test]
    fn test_min_efficient_size() {
        let cpu = DeviceCapabilities::cpu_default();
        assert_eq!(cpu.min_efficient_size(), 0);
        
        let mut gpu = DeviceCapabilities::cpu_default();
        gpu.device_type = DeviceType::DiscreteGpu;
        assert_eq!(gpu.min_efficient_size(), 50_000);
    }

    #[test]
    fn test_recommended_for_size() {
        let cpu = DeviceCapabilities::cpu_default();
        assert!(cpu.recommended_for_size(50_000));
        assert!(!cpu.recommended_for_size(150_000));
        
        let mut gpu = DeviceCapabilities::cpu_default();
        gpu.device_type = DeviceType::DiscreteGpu;
        assert!(!gpu.recommended_for_size(10_000));
        assert!(gpu.recommended_for_size(100_000));
    }
}
