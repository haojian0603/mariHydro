// crates/mh_physics/src/gpu/wgpu_backend.rs

//! wgpu GPU 计算后端实现
//!
//! 基于 wgpu 提供跨平台的 GPU 计算支持（Vulkan/Metal/DX12/WebGPU）

use std::sync::Arc;

use log::info;
use wgpu::{
    Adapter, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions,
};

use super::backend::{ComputeBackend, ComputeOperation, PerformanceEstimate};
use super::capabilities::{DeviceCapabilities, DeviceType, MemoryInfo};
use super::solver::GpuError;
use mh_foundation::error::MhResult;

/// wgpu GPU 计算后端
pub struct WgpuBackend {
    /// wgpu 实例
    instance: Instance,
    /// GPU 适配器
    adapter: Arc<Adapter>,
    /// GPU 设备
    device: Arc<Device>,
    /// 命令队列
    queue: Arc<Queue>,
    /// 设备能力
    capabilities: DeviceCapabilities,
}

impl WgpuBackend {
    /// 异步创建 GPU 后端
    ///
    /// 返回 `Ok(None)` 表示没有可用的 GPU
    pub async fn new_async() -> Result<Option<Self>, GpuError> {
        Self::new_with_preference_async(PowerPreference::HighPerformance).await
    }

    /// 使用指定的电源偏好异步创建 GPU 后端
    pub async fn new_with_preference_async(
        power_preference: PowerPreference,
    ) -> Result<Option<Self>, GpuError> {
        // 创建 wgpu 实例
        let instance = Instance::new(InstanceDescriptor {
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
        info!(
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
            .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;

        // 构建设备能力描述
        let capabilities = Self::build_capabilities(&adapter, &adapter_info);

        info!(
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

    /// 同步创建 GPU 后端（阻塞调用）
    pub fn new() -> Result<Option<Self>, GpuError> {
        pollster::block_on(Self::new_async())
    }

    /// 使用指定电源偏好同步创建
    pub fn new_with_preference(power_preference: PowerPreference) -> Result<Option<Self>, GpuError> {
        pollster::block_on(Self::new_with_preference_async(power_preference))
    }

    /// 获取 wgpu 实例引用
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    /// 获取 wgpu 适配器引用
    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    /// 获取 wgpu 设备引用
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 获取 wgpu 队列引用
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// 获取 Arc 包装的设备
    pub fn device_arc(&self) -> Arc<Device> {
        Arc::clone(&self.device)
    }

    /// 获取 Arc 包装的队列
    pub fn queue_arc(&self) -> Arc<Queue> {
        Arc::clone(&self.queue)
    }

    /// 必需的 GPU 特性
    fn required_features() -> Features {
        Features::empty()
        // 可以根据需要添加特性，如：
        // | Features::PUSH_CONSTANTS
        // | Features::STORAGE_RESOURCE_BINDING_ARRAY
    }

    /// 必需的 GPU 限制
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
    fn build_capabilities(adapter: &Adapter, info: &wgpu::AdapterInfo) -> DeviceCapabilities {
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
            supports_f64: false, // wgpu 目前不支持 f64 compute
            supports_f16: true,
            supports_atomic_float: false,
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
        } else if name_lower.contains("rtx") {
            return 64; // 其他 RTX
        } else if name_lower.contains("gtx") {
            return 24; // 其他 GTX
        }

        // AMD GPU
        if name_lower.contains("rx 7") {
            return 96; // RDNA 3
        } else if name_lower.contains("rx 6") {
            return 80; // RDNA 2
        } else if name_lower.contains("rx 5") {
            return 40; // RDNA 1
        } else if name_lower.contains("rx") {
            return 36; // 其他 RX
        }

        // Intel GPU
        if name_lower.contains("arc a7") {
            return 32; // Intel Arc A770/A750
        } else if name_lower.contains("arc a5") {
            return 16; // Intel Arc A580/A380
        } else if name_lower.contains("arc") {
            return 24; // 其他 Arc
        } else if name_lower.contains("iris xe") {
            return 96 / 8; // Iris Xe (EU -> 大致 CU)
        } else if name_lower.contains("iris") || name_lower.contains("uhd") {
            return 8; // 集成显卡
        }

        // Apple GPU
        if name_lower.contains("m3 max") {
            return 40;
        } else if name_lower.contains("m3 pro") {
            return 18;
        } else if name_lower.contains("m3") {
            return 10;
        } else if name_lower.contains("m2 ultra") {
            return 76;
        } else if name_lower.contains("m2 max") {
            return 38;
        } else if name_lower.contains("m2 pro") {
            return 19;
        } else if name_lower.contains("m2") {
            return 10;
        } else if name_lower.contains("m1") {
            return 8;
        }

        // 默认值
        match device_type {
            DeviceType::DiscreteGpu => 64,
            DeviceType::IntegratedGpu => 8,
            DeviceType::VirtualGpu => 32,
            DeviceType::Software => 2,
            DeviceType::Cpu => 4,
            DeviceType::Unknown => 16,
        }
    }

    /// 估算内存带宽 (GB/s)
    fn estimate_memory_bandwidth(device_type: DeviceType) -> f64 {
        match device_type {
            DeviceType::DiscreteGpu => 400.0,   // 高端独显
            DeviceType::IntegratedGpu => 50.0,  // 集成显卡
            DeviceType::VirtualGpu => 100.0,    // 虚拟 GPU
            DeviceType::Software => 10.0,       // 软件渲染
            DeviceType::Cpu => 30.0,            // CPU
            DeviceType::Unknown => 50.0,
        }
    }

    /// 估算总内存
    fn estimate_total_memory(device_type: DeviceType) -> usize {
        match device_type {
            DeviceType::DiscreteGpu => 8 * 1024 * 1024 * 1024,   // 8GB
            DeviceType::IntegratedGpu => 2 * 1024 * 1024 * 1024, // 2GB
            DeviceType::VirtualGpu => 4 * 1024 * 1024 * 1024,    // 4GB
            DeviceType::Software => 1 * 1024 * 1024 * 1024,      // 1GB
            DeviceType::Cpu => 16 * 1024 * 1024 * 1024,          // 16GB
            DeviceType::Unknown => 2 * 1024 * 1024 * 1024,       // 2GB
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

    /// 创建暂存缓冲区用于数据读回
    pub fn create_staging_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// 读取缓冲区数据到 CPU (阻塞)
    pub fn read_buffer<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
        let size = (count * std::mem::size_of::<T>()) as u64;
        let staging = self.create_staging_buffer(size);

        // 复制到暂存缓冲区
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_buffer"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // 映射并读取
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        result
    }

    /// 异步读取缓冲区数据
    pub async fn read_buffer_async<T: bytemuck::Pod>(
        &self,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<T>, GpuError> {
        let size = (count * std::mem::size_of::<T>()) as u64;
        let staging = self.create_staging_buffer(size);

        // 复制到暂存缓冲区
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_buffer_async"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // 映射并读取
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
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
        // 假设 50% GPU 利用率
        let compute_time = (ops_per_element * problem_size as f64) / (compute_units * 1e9 * 0.5);

        // 内存时间
        let memory_time = (bytes_per_element * problem_size as f64) / (bandwidth * 1e9 / 1000.0);

        let estimated_time_ms = compute_time.max(memory_time);

        // 内存需求
        let memory_required = (bytes_per_element * problem_size as f64) as usize;

        // GPU 推荐阈值：问题规模大于 10000 时推荐使用
        let recommended = problem_size > 10_000;

        PerformanceEstimate::new(estimated_time_ms, memory_required, recommended)
    }

    fn synchronize(&self) -> MhResult<()> {
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vendor_names() {
        assert_eq!(WgpuBackend::vendor_name(0x10DE), "NVIDIA");
        assert_eq!(WgpuBackend::vendor_name(0x1002), "AMD");
        assert_eq!(WgpuBackend::vendor_name(0x8086), "Intel");
        assert_eq!(WgpuBackend::vendor_name(0x106B), "Apple");
    }

    #[test]
    fn test_compute_units_estimation() {
        assert!(WgpuBackend::estimate_compute_units("NVIDIA GeForce RTX 4090", DeviceType::DiscreteGpu) > 100);
        assert!(WgpuBackend::estimate_compute_units("RTX 3080", DeviceType::DiscreteGpu) > 60);
        assert!(WgpuBackend::estimate_compute_units("Intel UHD Graphics", DeviceType::IntegratedGpu) < 20);
        assert!(WgpuBackend::estimate_compute_units("AMD RX 6800", DeviceType::DiscreteGpu) > 60);
        assert!(WgpuBackend::estimate_compute_units("Apple M2", DeviceType::IntegratedGpu) >= 10);
    }

    #[test]
    fn test_memory_bandwidth_estimation() {
        assert!(WgpuBackend::estimate_memory_bandwidth(DeviceType::DiscreteGpu) > 200.0);
        assert!(WgpuBackend::estimate_memory_bandwidth(DeviceType::IntegratedGpu) < 100.0);
    }

    #[test]
    #[ignore = "Requires GPU hardware"]
    fn test_backend_creation() {
        let result = WgpuBackend::new();
        assert!(result.is_ok());
    }

    #[test]
    #[ignore = "Requires GPU hardware"]
    fn test_backend_performance_estimate() {
        if let Ok(Some(backend)) = WgpuBackend::new() {
            let estimate = backend.estimate_performance(ComputeOperation::Riemann, 100_000);
            assert!(estimate.estimated_time_ms > 0.0);
            assert!(estimate.memory_required > 0);
            assert!(estimate.recommended);
        }
    }
}
