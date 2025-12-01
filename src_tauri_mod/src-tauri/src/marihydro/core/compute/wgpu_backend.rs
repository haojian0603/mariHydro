// src-tauri/src/marihydro/core/compute/wgpu_backend.rs

//! wgpu GPU计算后端实现
//!
//! 基于wgpu提供跨平台的GPU计算支持（Vulkan/Metal/DX12/WebGPU）

use std::sync::Arc;

use wgpu::{
    Adapter, Device, DeviceDescriptor, Features, Instance, Limits, PowerPreference, Queue,
    RequestAdapterOptions,
};

use super::backend::{ComputeBackend, ComputeOperation, PerformanceEstimate};
use super::capabilities::{ComputeFeatures, DeviceCapabilities, DeviceType, MemoryInfo};
use crate::marihydro::core::error::{MhError, MhResult};

/// wgpu GPU计算后端
pub struct WgpuBackend {
    /// wgpu实例
    instance: Instance,
    /// GPU适配器
    adapter: Arc<Adapter>,
    /// GPU设备
    device: Arc<Device>,
    /// 命令队列
    queue: Arc<Queue>,
    /// 设备能力
    capabilities: DeviceCapabilities,
}

impl WgpuBackend {
    /// 异步创建GPU后端
    ///
    /// 返回 `Ok(None)` 表示没有可用的GPU
    pub async fn new() -> MhResult<Option<Self>> {
        Self::new_with_preference(PowerPreference::HighPerformance).await
    }

    /// 使用指定的电源偏好创建GPU后端
    pub async fn new_with_preference(
        power_preference: PowerPreference,
    ) -> MhResult<Option<Self>> {
        // 创建wgpu实例
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // 请求适配器
        let adapter = match instance
            .request_adapter(&RequestAdapterOptions {
                power_preference,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Some(adapter) => adapter,
            None => return Ok(None),
        };

        // 获取适配器信息
        let adapter_info = adapter.get_info();
        log::info!(
            "Found GPU adapter: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        // 请求设备
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("MariHydro GPU Device"),
                    required_features: Self::required_features(),
                    required_limits: Self::required_limits(&adapter),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| MhError::ComputeError(format!("Failed to request device: {}", e)))?;

        // 构建设备能力描述
        let capabilities = Self::build_capabilities(&adapter, &adapter_info);

        log::info!(
            "GPU backend initialized: {} compute units, {:.1} GB memory",
            capabilities.compute_units,
            capabilities.memory.total_bytes as f64 / 1e9
        );

        Ok(Some(Self {
            instance,
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
            capabilities,
        }))
    }

    /// 同步创建GPU后端（阻塞调用）
    pub fn new_blocking() -> MhResult<Option<Self>> {
        pollster::block_on(Self::new())
    }

    /// 获取wgpu设备引用
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 获取wgpu队列引用
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// 获取Arc包装的设备
    pub fn device_arc(&self) -> Arc<Device> {
        Arc::clone(&self.device)
    }

    /// 获取Arc包装的队列
    pub fn queue_arc(&self) -> Arc<Queue> {
        Arc::clone(&self.queue)
    }

    /// 必需的GPU特性
    fn required_features() -> Features {
        Features::empty()
        // 可以根据需要添加特性，如：
        // | Features::PUSH_CONSTANTS
        // | Features::STORAGE_RESOURCE_BINDING_ARRAY
    }

    /// 必需的GPU限制
    fn required_limits(adapter: &Adapter) -> Limits {
        // 使用适配器支持的限制，但确保满足最低要求
        let supported = adapter.limits();
        Limits {
            max_storage_buffer_binding_size: supported
                .max_storage_buffer_binding_size
                .min(256 * 1024 * 1024), // 256MB
            max_compute_workgroup_size_x: supported.max_compute_workgroup_size_x.min(256),
            max_compute_workgroup_size_y: supported.max_compute_workgroup_size_y.min(256),
            max_compute_workgroup_size_z: supported.max_compute_workgroup_size_z.min(64),
            max_compute_invocations_per_workgroup: supported
                .max_compute_invocations_per_workgroup
                .min(256),
            ..Limits::downlevel_defaults()
        }
    }

    /// 构建设备能力描述
    fn build_capabilities(
        adapter: &Adapter,
        info: &wgpu::AdapterInfo,
    ) -> DeviceCapabilities {
        let limits = adapter.limits();

        // 确定设备类型
        let device_type = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => DeviceType::DiscreteGpu,
            wgpu::DeviceType::IntegratedGpu => DeviceType::IntegratedGpu,
            wgpu::DeviceType::VirtualGpu => DeviceType::VirtualGpu,
            wgpu::DeviceType::Cpu => DeviceType::Cpu,
            wgpu::DeviceType::Other => DeviceType::Unknown,
        };

        // 估算计算单元数（基于设备类型的启发式）
        let compute_units = Self::estimate_compute_units(&info.name, device_type);

        // 估算内存带宽（GB/s）
        let memory_bandwidth = Self::estimate_memory_bandwidth(device_type);

        DeviceCapabilities {
            name: info.name.clone(),
            vendor: Self::vendor_name(info.vendor),
            device_type,
            compute_units,
            max_workgroup_size: limits.max_compute_invocations_per_workgroup,
            max_workgroups: [
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
            ],
            memory: MemoryInfo {
                total_bytes: Self::estimate_total_memory(device_type),
                available_bytes: Self::estimate_total_memory(device_type),
                max_buffer_size: limits.max_storage_buffer_binding_size as usize,
            },
            features: ComputeFeatures {
                supports_f64: false, // wgpu目前不支持f64 compute
                supports_f16: true,
                supports_atomic_float: false,
                supports_subgroups: false,
                max_subgroup_size: 32,
            },
            memory_bandwidth_gbps: memory_bandwidth,
        }
    }

    /// 估算计算单元数
    fn estimate_compute_units(name: &str, device_type: DeviceType) -> u32 {
        let name_lower = name.to_lowercase();

        // NVIDIA GPU
        if name_lower.contains("rtx 40") {
            return 128; // Ada Lovelace
        } else if name_lower.contains("rtx 30") {
            return 84; // Ampere
        } else if name_lower.contains("rtx 20") {
            return 68; // Turing
        } else if name_lower.contains("gtx 10") {
            return 28; // Pascal
        }

        // AMD GPU
        if name_lower.contains("rx 7") {
            return 96; // RDNA 3
        } else if name_lower.contains("rx 6") {
            return 80; // RDNA 2
        }

        // Intel GPU
        if name_lower.contains("arc") {
            return 32; // Intel Arc
        } else if name_lower.contains("iris") || name_lower.contains("uhd") {
            return 8; // 集成显卡
        }

        // 默认值
        match device_type {
            DeviceType::DiscreteGpu => 64,
            DeviceType::IntegratedGpu => 8,
            _ => 4,
        }
    }

    /// 估算内存带宽
    fn estimate_memory_bandwidth(device_type: DeviceType) -> f64 {
        match device_type {
            DeviceType::DiscreteGpu => 400.0,    // 高端独显
            DeviceType::IntegratedGpu => 50.0,   // 集成显卡
            DeviceType::VirtualGpu => 100.0,     // 虚拟GPU
            DeviceType::Cpu => 30.0,             // CPU
            DeviceType::Unknown => 50.0,
        }
    }

    /// 估算总内存
    fn estimate_total_memory(device_type: DeviceType) -> usize {
        match device_type {
            DeviceType::DiscreteGpu => 8 * 1024 * 1024 * 1024,      // 8GB
            DeviceType::IntegratedGpu => 2 * 1024 * 1024 * 1024,    // 2GB
            DeviceType::VirtualGpu => 4 * 1024 * 1024 * 1024,       // 4GB
            DeviceType::Cpu => 16 * 1024 * 1024 * 1024,             // 16GB
            DeviceType::Unknown => 2 * 1024 * 1024 * 1024,          // 2GB
        }
    }

    /// 获取厂商名称
    fn vendor_name(vendor_id: u32) -> String {
        match vendor_id {
            0x10DE => "NVIDIA".to_string(),
            0x1002 => "AMD".to_string(),
            0x8086 => "Intel".to_string(),
            0x13B5 => "ARM".to_string(),
            0x5143 => "Qualcomm".to_string(),
            0x106B => "Apple".to_string(),
            _ => format!("Vendor(0x{:04X})", vendor_id),
        }
    }
}

impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn estimate_performance(
        &self,
        op: ComputeOperation,
        problem_size: usize,
    ) -> PerformanceEstimate {
        let compute_units = self.capabilities.compute_units as f64;
        let bandwidth = self.capabilities.memory_bandwidth_gbps;

        // 基于操作类型和问题规模估算
        let (ops_per_element, bytes_per_element) = match op {
            ComputeOperation::Gradient => (50.0, 64.0),       // 较多计算，中等带宽
            ComputeOperation::Limiter => (20.0, 32.0),        // 少量计算
            ComputeOperation::Riemann => (200.0, 128.0),      // 大量计算
            ComputeOperation::FluxAccumulate => (10.0, 48.0), // 原子操作
            ComputeOperation::SourceTerms => (30.0, 48.0),    // 中等计算
            ComputeOperation::TimeIntegrate => (20.0, 64.0),  // 简单计算
            ComputeOperation::Reduction => (5.0, 8.0),        // 归约
        };

        // 计算时间（毫秒）
        let compute_time = (ops_per_element * problem_size as f64)
            / (compute_units * 1e9 * 0.5); // 假设50% utilization

        let memory_time = (bytes_per_element * problem_size as f64)
            / (bandwidth * 1e9 / 1000.0); // GB/s to B/ms

        let estimated_time_ms = compute_time.max(memory_time);

        // 内存需求
        let memory_required = (bytes_per_element * problem_size as f64) as usize;

        // GPU推荐阈值：问题规模大于10000时推荐使用
        let recommended = problem_size > 10_000;

        PerformanceEstimate {
            estimated_time_ms,
            memory_required,
            recommended,
        }
    }

    fn synchronize(&self) -> MhResult<()> {
        // wgpu的同步通过提交命令并等待完成
        // 这里简单地poll设备
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        // 注意：此测试需要实际GPU
        let result = WgpuBackend::new_blocking();
        assert!(result.is_ok());
        // GPU可能不可用，所以不检查Some
    }

    #[test]
    fn test_vendor_names() {
        assert_eq!(WgpuBackend::vendor_name(0x10DE), "NVIDIA");
        assert_eq!(WgpuBackend::vendor_name(0x1002), "AMD");
        assert_eq!(WgpuBackend::vendor_name(0x8086), "Intel");
    }

    #[test]
    fn test_compute_units_estimation() {
        assert!(WgpuBackend::estimate_compute_units("RTX 4090", DeviceType::DiscreteGpu) > 100);
        assert!(WgpuBackend::estimate_compute_units("Intel UHD", DeviceType::IntegratedGpu) < 20);
    }
}
