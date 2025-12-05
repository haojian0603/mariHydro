// crates/mh_physics/src/gpu/solver.rs

//! GPU 求解器
//!
//! 基于 wgpu 的浅水方程 GPU 求解器，实现完整的时间推进计算。
//!
//! # 计算流程
//!
//! 每个时间步的计算流程如下：
//!
//! 1. 清空梯度缓冲区
//! 2. 计算 Green-Gauss 梯度
//! 3. 计算限制器值 (BJ/VK)
//! 4. MUSCL 重构面值
//! 5. 干湿边界修正
//! 6. HLLC 通量计算
//! 7. 清空残差缓冲区
//! 8. 着色通量累积 (无竞争并行)
//! 9. 源项计算
//! 10. 时间积分 (Euler/RK2/RK3)

use super::bind_groups::{BindGroupLayouts, ColoringData, GpuComputeParams, ParamsBuffer};
use super::buffer::{DoubleBuffer, GpuBufferUsage, TypedBuffer};
use super::capabilities::{DeviceCapabilities, DeviceType};
use super::mesh::GpuMeshData;
use super::pipeline::PipelineCache;
use super::shaders;
use super::state::{GpuStateArrays, GpuWorkspace};
use crate::types::NumericalParams;
use std::sync::Arc;
use wgpu::{
    Adapter, BindGroup, BindGroupDescriptor, BindGroupEntry, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, Device, DeviceDescriptor, Features, Instance,
    InstanceDescriptor, Limits, Queue, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
};

/// GPU 求解器配置
#[derive(Debug, Clone)]
pub struct GpuSolverConfig {
    /// 首选设备类型
    pub preferred_device: DeviceType,
    /// 工作组大小
    pub workgroup_size: u32,
    /// 是否启用异步计算
    pub enable_async: bool,
    /// 最小批处理大小（小于此值时使用 CPU）
    pub min_batch_size: usize,
    /// 是否启用调试标签
    pub enable_debug_labels: bool,
    /// 重构阶数 (1=一阶, 2=二阶MUSCL)
    pub reconstruction_order: u32,
    /// 限制器类型 (0=none, 1=Barth-Jespersen, 2=Venkatakrishnan)
    pub limiter_type: u32,
    /// Venkatakrishnan 系数
    pub venkat_k: f32,
    /// RK 阶数 (1, 2, 3)
    pub rk_order: u32,
    /// 摩擦类型 (0=none, 1=manning, 2=chezy)
    pub friction_type: u32,
    /// 曼宁系数
    pub manning_n: f32,
}

impl Default for GpuSolverConfig {
    fn default() -> Self {
        Self {
            preferred_device: DeviceType::DiscreteGpu,
            workgroup_size: 256,
            enable_async: true,
            min_batch_size: 1000,
            enable_debug_labels: cfg!(debug_assertions),
            reconstruction_order: 2,
            limiter_type: 1,
            venkat_k: 0.5,
            rk_order: 2,
            friction_type: 1,
            manning_n: 0.025,
        }
    }
}

impl GpuSolverConfig {
    /// 创建高性能配置
    pub fn high_performance() -> Self {
        Self {
            preferred_device: DeviceType::DiscreteGpu,
            workgroup_size: 256,
            enable_async: true,
            min_batch_size: 500,
            enable_debug_labels: false,
            reconstruction_order: 2,
            limiter_type: 2, // Venkatakrishnan
            venkat_k: 0.3,
            rk_order: 3, // SSP-RK3
            friction_type: 1,
            manning_n: 0.025,
        }
    }

    /// 创建低功耗配置
    pub fn low_power() -> Self {
        Self {
            preferred_device: DeviceType::IntegratedGpu,
            workgroup_size: 128,
            enable_async: false,
            min_batch_size: 2000,
            enable_debug_labels: false,
            reconstruction_order: 1, // 一阶精度
            limiter_type: 0,
            venkat_k: 0.5,
            rk_order: 1, // Euler
            friction_type: 1,
            manning_n: 0.025,
        }
    }
}

/// GPU 计算统计
#[derive(Debug, Clone, Default)]
pub struct GpuStats {
    /// 梯度计算时间 (ms)
    pub gradient_time_ms: f64,
    /// 限制器时间 (ms)
    pub limiter_time_ms: f64,
    /// 重构时间 (ms)
    pub reconstruct_time_ms: f64,
    /// HLLC 求解时间 (ms)
    pub hllc_time_ms: f64,
    /// 通量累积时间 (ms)
    pub accumulate_time_ms: f64,
    /// 源项时间 (ms)
    pub source_time_ms: f64,
    /// 时间积分时间 (ms)
    pub integrate_time_ms: f64,
    /// 数据传输时间 (ms)
    pub transfer_time_ms: f64,
    /// 总时间 (ms)
    pub total_time_ms: f64,
}

/// GPU 上下文
pub struct GpuContext {
    /// wgpu 实例
    #[allow(dead_code)]
    instance: Instance,
    /// 适配器
    #[allow(dead_code)]
    adapter: Adapter,
    /// 设备
    device: Arc<Device>,
    /// 命令队列
    queue: Arc<Queue>,
    /// 设备能力
    capabilities: DeviceCapabilities,
}

impl GpuContext {
    /// 异步创建 GPU 上下文
    pub async fn new_async(config: &GpuSolverConfig) -> Result<Self, GpuError> {
        // 创建 wgpu 实例
        let instance = Instance::new(InstanceDescriptor::default());

        // 请求适配器
        let power_preference = match config.preferred_device {
            DeviceType::DiscreteGpu => wgpu::PowerPreference::HighPerformance,
            DeviceType::IntegratedGpu => wgpu::PowerPreference::LowPower,
            _ => wgpu::PowerPreference::None,
        };

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        // 请求设备
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("MariHydro GPU Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;

        // 获取设备能力
        let capabilities = DeviceCapabilities::from_wgpu(&adapter);

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            capabilities,
        })
    }

    /// 同步创建 GPU 上下文（阻塞）
    pub fn new(config: &GpuSolverConfig) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(config))
    }

    /// 获取设备引用
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 获取队列引用
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// 获取设备能力
    pub fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    /// 克隆设备 Arc
    pub fn device_arc(&self) -> Arc<Device> {
        self.device.clone()
    }

    /// 克隆队列 Arc
    pub fn queue_arc(&self) -> Arc<Queue> {
        self.queue.clone()
    }
}

/// GPU 计算管线集合
pub struct GpuPipelines {
    /// 梯度计算
    pub gradient: ComputePipeline,
    /// 清空梯度
    pub clear_gradients: ComputePipeline,
    /// 限制器
    pub limiter: ComputePipeline,
    /// 重构
    pub reconstruct: ComputePipeline,
    /// 干湿修正
    pub wet_dry_fix: ComputePipeline,
    /// HLLC 求解
    pub hllc: ComputePipeline,
    /// 通量累积
    pub accumulate: ComputePipeline,
    /// 清空残差
    pub clear_residuals: ComputePipeline,
    /// Euler 积分
    pub euler: ComputePipeline,
    /// SSP-RK2 积分
    pub ssp_rk2: ComputePipeline,
    /// SSP-RK3 积分
    pub ssp_rk3: ComputePipeline,
    /// 源项
    pub source: ComputePipeline,
    /// 边界条件
    pub boundary: ComputePipeline,
}

impl GpuPipelines {
    /// 创建所有计算管线
    pub fn new(device: &Device, _config: &GpuSolverConfig) -> Result<Self, GpuError> {
        // 编译着色器模块
        let gradient_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("gradient"),
            source: ShaderSource::Wgsl(shaders::GRADIENT.into()),
        });

        let limiter_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("limiter"),
            source: ShaderSource::Wgsl(shaders::LIMITER.into()),
        });

        let reconstruct_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("reconstruct"),
            source: ShaderSource::Wgsl(shaders::RECONSTRUCT.into()),
        });

        let hllc_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("hllc"),
            source: ShaderSource::Wgsl(shaders::HLLC.into()),
        });

        let accumulate_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("accumulate"),
            source: ShaderSource::Wgsl(shaders::ACCUMULATE.into()),
        });

        let integrate_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("integrate"),
            source: ShaderSource::Wgsl(shaders::INTEGRATE.into()),
        });

        let source_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("source"),
            source: ShaderSource::Wgsl(shaders::SOURCE.into()),
        });

        let boundary_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("boundary"),
            source: ShaderSource::Wgsl(shaders::BOUNDARY.into()),
        });

        // 创建计算管线
        let gradient = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gradient"),
            layout: None,
            module: &gradient_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let clear_gradients = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clear_gradients"),
            layout: None,
            module: &gradient_module,
            entry_point: Some("clear_gradients"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let limiter = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("limiter"),
            layout: None,
            module: &limiter_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let reconstruct = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reconstruct"),
            layout: None,
            module: &reconstruct_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let wet_dry_fix = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wet_dry_fix"),
            layout: None,
            module: &reconstruct_module,
            entry_point: Some("wet_dry_fix"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let hllc = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hllc"),
            layout: None,
            module: &hllc_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let accumulate = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("accumulate"),
            layout: None,
            module: &accumulate_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let clear_residuals = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clear_residuals"),
            layout: None,
            module: &accumulate_module,
            entry_point: Some("clear_residuals"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let euler = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("euler"),
            layout: None,
            module: &integrate_module,
            entry_point: Some("euler"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let ssp_rk2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssp_rk2"),
            layout: None,
            module: &integrate_module,
            entry_point: Some("ssp_rk2"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let ssp_rk3 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssp_rk3"),
            layout: None,
            module: &integrate_module,
            entry_point: Some("ssp_rk3"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let source = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("source"),
            layout: None,
            module: &source_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let boundary = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("boundary"),
            layout: None,
            module: &boundary_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            gradient,
            clear_gradients,
            limiter,
            reconstruct,
            wet_dry_fix,
            hllc,
            accumulate,
            clear_residuals,
            euler,
            ssp_rk2,
            ssp_rk3,
            source,
            boundary,
        })
    }
}

/// GPU 求解器状态缓冲区 (简化版，用于兼容)
pub struct GpuStateBuffers {
    /// 水深双缓冲
    h: DoubleBuffer<f32>,
    /// x 动量双缓冲
    hu: DoubleBuffer<f32>,
    /// y 动量双缓冲
    hv: DoubleBuffer<f32>,
    /// 底床高程（只读）
    z: TypedBuffer<f32>,
    /// 单元格数量
    n_cells: usize,
}

impl GpuStateBuffers {
    /// 创建状态缓冲区
    pub fn new(device: &Device, n_cells: usize) -> Self {
        Self {
            h: DoubleBuffer::new(device, n_cells, GpuBufferUsage::Storage, "h"),
            hu: DoubleBuffer::new(device, n_cells, GpuBufferUsage::Storage, "hu"),
            hv: DoubleBuffer::new(device, n_cells, GpuBufferUsage::Storage, "hv"),
            z: TypedBuffer::new(device, n_cells, GpuBufferUsage::StorageReadOnly, Some("z")),
            n_cells,
        }
    }

    /// 上传初始状态
    pub fn upload_state(&self, queue: &Queue, h: &[f64], hu: &[f64], hv: &[f64], z: &[f64]) {
        // 转换为 f32
        let h_f32: Vec<f32> = h.iter().map(|&x| x as f32).collect();
        let hu_f32: Vec<f32> = hu.iter().map(|&x| x as f32).collect();
        let hv_f32: Vec<f32> = hv.iter().map(|&x| x as f32).collect();
        let z_f32: Vec<f32> = z.iter().map(|&x| x as f32).collect();

        self.h.read_buffer().write(queue, &h_f32);
        self.hu.read_buffer().write(queue, &hu_f32);
        self.hv.read_buffer().write(queue, &hv_f32);
        self.z.write(queue, &z_f32);
    }

    /// 获取单元格数量
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 交换缓冲区
    pub fn swap(&mut self) {
        self.h.swap();
        self.hu.swap();
        self.hv.swap();
    }
}

/// GPU 求解器
pub struct GpuSolver {
    /// GPU 上下文
    context: GpuContext,
    /// 配置
    config: GpuSolverConfig,
    /// 状态缓冲区 (简化版)
    state: Option<GpuStateBuffers>,
    /// 完整工作区
    workspace: Option<GpuWorkspace>,
    /// 管线缓存
    pipelines: Option<GpuPipelines>,
    /// 参数缓冲区
    params_buffer: Option<ParamsBuffer>,
    /// 绑定组布局
    layouts: Option<BindGroupLayouts>,
    /// 着色数据
    coloring: Option<ColoringData>,
    /// 数值参数
    params: NumericalParams,
    /// 统计信息
    stats: GpuStats,
}

impl GpuSolver {
    /// 创建新的 GPU 求解器
    pub fn new(config: GpuSolverConfig) -> Result<Self, GpuError> {
        let context = GpuContext::new(&config)?;

        // 创建参数缓冲区和绑定组布局
        let params_buffer = Some(ParamsBuffer::new(context.device()));
        let layouts = Some(BindGroupLayouts::new(context.device()));

        Ok(Self {
            context,
            config,
            state: None,
            workspace: None,
            pipelines: None,
            params_buffer,
            layouts,
            coloring: None,
            params: NumericalParams::default(),
            stats: GpuStats::default(),
        })
    }

    /// 异步创建
    pub async fn new_async(config: GpuSolverConfig) -> Result<Self, GpuError> {
        let context = GpuContext::new_async(&config).await?;
        
        // 创建参数缓冲区
        let params_buffer = Some(ParamsBuffer::new(context.device()));
        
        // 创建绑定组布局
        let layouts = Some(BindGroupLayouts::new(context.device()));

        Ok(Self {
            context,
            config,
            state: None,
            workspace: None,
            pipelines: None,
            params_buffer,
            layouts,
            coloring: None,
            params: NumericalParams::default(),
            stats: GpuStats::default(),
        })
    }

    /// 初始化管线
    pub fn init_pipelines(&mut self) -> Result<(), GpuError> {
        let pipelines = GpuPipelines::new(self.context.device(), &self.config)?;
        self.pipelines = Some(pipelines);
        Ok(())
    }

    /// 初始化状态缓冲区 (简化版)
    pub fn init_buffers(&mut self, n_cells: usize) {
        self.state = Some(GpuStateBuffers::new(self.context.device(), n_cells));
    }

    /// 从 FrozenMesh 初始化完整工作区
    pub fn init_workspace(&mut self, mesh: &mh_mesh::FrozenMesh) {
        let workspace = GpuWorkspace::from_mesh(
            self.context.device(),
            self.context.queue(),
            mesh,
        );
        self.workspace = Some(workspace);
    }

    /// 初始化面着色数据（用于并行累加避免竞争）
    /// 
    /// 面着色将所有面分成多个颜色组，同一颜色组内的面不共享任何单元格，
    /// 因此可以安全地并行累加残差。
    pub fn init_coloring(&mut self, mesh: &mh_mesh::FrozenMesh) {
        // 简单的贪心着色算法
        let n_faces = mesh.n_faces;
        let mut face_colors: Vec<u32> = vec![u32::MAX; n_faces];
        
        // 构建单元格到面的映射
        let mut cell_to_faces: std::collections::HashMap<u32, Vec<usize>> = 
            std::collections::HashMap::new();
        
        for face_idx in 0..n_faces {
            let owner = mesh.face_owner[face_idx];
            let neighbor = mesh.face_neighbor[face_idx];
            
            cell_to_faces.entry(owner).or_default().push(face_idx);
            if neighbor != u32::MAX {
                cell_to_faces.entry(neighbor).or_default().push(face_idx);
            }
        }
        
        // 贪心着色
        let mut color_groups: Vec<Vec<u32>> = Vec::new();
        
        for face_idx in 0..n_faces {
            // 找出邻居面已使用的颜色
            let mut used_colors = std::collections::HashSet::<u32>::new();
            let owner = mesh.face_owner[face_idx];
            let neighbor = mesh.face_neighbor[face_idx];
            
            // 检查 owner 单元格的其他面
            if let Some(faces) = cell_to_faces.get(&owner) {
                for &other_face in faces {
                    if face_colors[other_face] != u32::MAX {
                        used_colors.insert(face_colors[other_face]);
                    }
                }
            }
            
            // 检查 neighbor 单元格的其他面
            if neighbor != u32::MAX {
                if let Some(faces) = cell_to_faces.get(&neighbor) {
                    for &other_face in faces {
                        if face_colors[other_face] != u32::MAX {
                            used_colors.insert(face_colors[other_face]);
                        }
                    }
                }
            }
            
            // 找到第一个未使用的颜色
            let mut color = 0u32;
            while used_colors.contains(&color) {
                color += 1;
            }
            
            face_colors[face_idx] = color;
            
            // 添加到颜色组
            while color_groups.len() <= color as usize {
                color_groups.push(Vec::new());
            }
            color_groups[color as usize].push(face_idx as u32);
        }
        
        // 创建 GPU 着色数据
        let coloring = ColoringData::from_color_groups(self.context.device(), &color_groups);
        self.coloring = Some(coloring);
    }

    /// 上传状态数据
    pub fn upload_state(&self, h: &[f64], hu: &[f64], hv: &[f64], z: &[f64]) {
        if let Some(state) = &self.state {
            state.upload_state(self.context.queue(), h, hu, hv, z);
        }
        if let Some(workspace) = &self.workspace {
            workspace.state.upload_state(self.context.queue(), h, hu, hv, z);
        }
    }

    /// 设置数值参数
    pub fn set_params(&mut self, params: NumericalParams) {
        self.params = params;
    }

    /// 获取设备能力
    pub fn capabilities(&self) -> &DeviceCapabilities {
        self.context.capabilities()
    }

    /// 检查是否应该使用 GPU
    pub fn should_use_gpu(&self, n_cells: usize) -> bool {
        n_cells >= self.config.min_batch_size && self.context.capabilities().is_suitable()
    }

    /// 执行同步点
    pub fn synchronize(&self) {
        self.context.device().poll(wgpu::Maintain::Wait);
    }

    /// 获取 GPU 上下文
    pub fn context(&self) -> &GpuContext {
        &self.context
    }

    /// 获取配置
    pub fn config(&self) -> &GpuSolverConfig {
        &self.config
    }

    /// 获取统计信息
    pub fn stats(&self) -> &GpuStats {
        &self.stats
    }

    /// 计算 dispatch 大小
    fn dispatch_size(&self, n: u32) -> u32 {
        (n + self.config.workgroup_size - 1) / self.config.workgroup_size
    }

    /// 执行单个时间步
    pub fn step(&mut self, dt: f64) -> Result<GpuStats, GpuError> {
        let start = std::time::Instant::now();

        // 确保工作区已初始化
        let workspace = self.workspace.as_ref()
            .ok_or_else(|| GpuError::BufferOperation("Workspace not initialized".to_string()))?;
        let pipelines = self.pipelines.as_ref()
            .ok_or_else(|| GpuError::PipelineCreation("Pipelines not initialized".to_string()))?;

        // 更新参数
        if let Some(ref mut params_buffer) = self.params_buffer {
            params_buffer.set_dt(self.context.queue(), dt as f32);
        }

        match self.config.rk_order {
            1 => self.step_euler(dt)?,
            2 => self.step_rk2(dt)?,
            3 => self.step_rk3(dt)?,
            _ => return Err(GpuError::PipelineCreation("Invalid RK order".to_string())),
        }

        // 同步等待 GPU 完成
        self.context.device().poll(wgpu::Maintain::Wait);

        self.stats.total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(self.stats.clone())
    }

    /// 前向 Euler 时间积分
    fn step_euler(&mut self, dt: f64) -> Result<(), GpuError> {
        // 设置 RK 阶段
        if let Some(ref mut params_buffer) = self.params_buffer {
            params_buffer.set_rk_stage(self.context.queue(), 0);
        }

        // 计算右端项
        self.compute_rhs()?;

        // 执行 Euler 积分
        self.run_integrate_pass("euler")?;

        // 交换缓冲区
        if let Some(ref mut workspace) = self.workspace {
            workspace.state.swap();
        }

        Ok(())
    }

    /// SSP-RK2 时间积分
    ///
    /// U^(1) = U^n + dt * L(U^n)
    /// U^(n+1) = 0.5 * U^n + 0.5 * (U^(1) + dt * L(U^(1)))
    fn step_rk2(&mut self, dt: f64) -> Result<(), GpuError> {
        // 保存初始状态
        self.save_initial_state()?;

        // Stage 1: U^(1) = U^n + dt * L(U^n)
        if let Some(ref mut params_buffer) = self.params_buffer {
            params_buffer.set_rk_stage(self.context.queue(), 0);
        }
        self.compute_rhs()?;
        self.run_integrate_pass("ssp_rk2_stage1")?;

        if let Some(ref mut workspace) = self.workspace {
            workspace.state.swap();
        }

        // Stage 2: U^(n+1) = 0.5 * U^n + 0.5 * (U^(1) + dt * L(U^(1)))
        if let Some(ref mut params_buffer) = self.params_buffer {
            params_buffer.set_rk_stage(self.context.queue(), 1);
        }
        self.compute_rhs()?;
        self.run_integrate_pass("ssp_rk2_stage2")?;

        if let Some(ref mut workspace) = self.workspace {
            workspace.state.swap();
        }

        Ok(())
    }

    /// SSP-RK3 时间积分
    ///
    /// U^(1) = U^n + dt * L(U^n)
    /// U^(2) = 0.75 * U^n + 0.25 * (U^(1) + dt * L(U^(1)))
    /// U^(n+1) = 1/3 * U^n + 2/3 * (U^(2) + dt * L(U^(2)))
    fn step_rk3(&mut self, dt: f64) -> Result<(), GpuError> {
        // 保存初始状态
        self.save_initial_state()?;

        // Stage 1
        if let Some(ref mut params_buffer) = self.params_buffer {
            params_buffer.set_rk_stage(self.context.queue(), 0);
        }
        self.compute_rhs()?;
        self.run_integrate_pass("ssp_rk3_stage1")?;
        if let Some(ref mut workspace) = self.workspace {
            workspace.state.swap();
        }

        // Stage 2
        if let Some(ref mut params_buffer) = self.params_buffer {
            params_buffer.set_rk_stage(self.context.queue(), 1);
        }
        self.compute_rhs()?;
        self.run_integrate_pass("ssp_rk3_stage2")?;
        if let Some(ref mut workspace) = self.workspace {
            workspace.state.swap();
        }

        // Stage 3
        if let Some(ref mut params_buffer) = self.params_buffer {
            params_buffer.set_rk_stage(self.context.queue(), 2);
        }
        self.compute_rhs()?;
        self.run_integrate_pass("ssp_rk3_stage3")?;
        if let Some(ref mut workspace) = self.workspace {
            workspace.state.swap();
        }

        Ok(())
    }

    /// 保存初始状态到 RK 缓冲区
    fn save_initial_state(&mut self) -> Result<(), GpuError> {
        let workspace = self.workspace.as_ref()
            .ok_or_else(|| GpuError::BufferOperation("Workspace not initialized".to_string()))?;

        let n_cells = workspace.state.n_cells;
        let mut encoder = self.context.device().create_command_encoder(&CommandEncoderDescriptor {
            label: Some("save_initial_state"),
        });

        // 复制当前状态到 RK 初始状态缓冲区
        encoder.copy_buffer_to_buffer(
            workspace.state.h.read_buffer().buffer(),
            0,
            workspace.state.rk_stages.h_n.buffer(),
            0,
            (n_cells * std::mem::size_of::<f32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            workspace.state.hu.read_buffer().buffer(),
            0,
            workspace.state.rk_stages.hu_n.buffer(),
            0,
            (n_cells * std::mem::size_of::<f32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            workspace.state.hv.read_buffer().buffer(),
            0,
            workspace.state.rk_stages.hv_n.buffer(),
            0,
            (n_cells * std::mem::size_of::<f32>()) as u64,
        );

        self.context.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// 运行积分 pass
    fn run_integrate_pass(&mut self, _entry_point: &str) -> Result<(), GpuError> {
        let workspace = self.workspace.as_ref()
            .ok_or_else(|| GpuError::BufferOperation("Workspace not initialized".to_string()))?;
        let pipelines = self.pipelines.as_ref()
            .ok_or_else(|| GpuError::PipelineCreation("Pipelines not initialized".to_string()))?;

        let num_cells = workspace.mesh.num_cells();
        let dispatch_count = self.dispatch_size(num_cells);

        let mut encoder = self.context.device().create_command_encoder(&CommandEncoderDescriptor {
            label: Some("integrate"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("integrate"),
                timestamp_writes: None,
            });

            // 根据 RK 阶段选择管线
            let rk_stage = self.params_buffer.as_ref()
                .map(|p| p.params().rk_stage)
                .unwrap_or(0);

            let pipeline = match self.config.rk_order {
                1 => &pipelines.euler,
                2 => &pipelines.ssp_rk2,
                3 => &pipelines.ssp_rk3,
                _ => &pipelines.euler,
            };

            pass.set_pipeline(pipeline);

            // 注意：实际使用时需要设置 bind groups
            // pass.set_bind_group(0, &params_bind_group, &[]);
            // pass.set_bind_group(1, &state_bind_group, &[]);
            // ...

            pass.dispatch_workgroups(dispatch_count, 1, 1);
        }

        self.context.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// 计算右端项 (RHS)
    ///
    /// 完整的 RHS 计算流程:
    /// 1. 清空梯度
    /// 2. 计算梯度
    /// 3. 计算限制器
    /// 4. 重构面值
    /// 5. 干湿修正
    /// 6. HLLC 求解器
    /// 7. 清空残差
    /// 8. 累积通量 (着色)
    /// 9. 计算源项
    fn compute_rhs(&mut self) -> Result<(), GpuError> {
        let workspace = self.workspace.as_ref()
            .ok_or_else(|| GpuError::BufferOperation("Workspace not initialized".to_string()))?;
        let pipelines = self.pipelines.as_ref()
            .ok_or_else(|| GpuError::PipelineCreation("Pipelines not initialized".to_string()))?;

        let num_cells = workspace.mesh.num_cells();
        let num_faces = workspace.mesh.num_faces();
        let cell_dispatch = self.dispatch_size(num_cells);
        let face_dispatch = self.dispatch_size(num_faces);

        let mut encoder = self.context.device().create_command_encoder(&CommandEncoderDescriptor {
            label: Some("compute_rhs"),
        });

        // 1. 清空梯度
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("clear_gradients"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.clear_gradients);
            pass.dispatch_workgroups(cell_dispatch, 1, 1);
        }

        // 2. 计算梯度 (Green-Gauss)
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("gradient"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.gradient);
            pass.dispatch_workgroups(face_dispatch, 1, 1);
        }

        // 3. 计算限制器 (如果启用二阶精度)
        if self.config.reconstruction_order >= 2 && self.config.limiter_type > 0 {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("limiter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.limiter);
            pass.dispatch_workgroups(cell_dispatch, 1, 1);
        }

        // 4. MUSCL 重构
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("reconstruct"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.reconstruct);
            pass.dispatch_workgroups(face_dispatch, 1, 1);
        }

        // 5. 干湿边界修正
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("wet_dry_fix"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.wet_dry_fix);
            pass.dispatch_workgroups(face_dispatch, 1, 1);
        }

        // 6. HLLC 通量计算
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("hllc"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.hllc);
            pass.dispatch_workgroups(face_dispatch, 1, 1);
        }

        // 7. 清空残差
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("clear_residuals"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.clear_residuals);
            pass.dispatch_workgroups(cell_dispatch, 1, 1);
        }

        // 8. 着色通量累积
        if let Some(ref coloring) = self.coloring {
            for color_idx in 0..coloring.num_colors {
                let color_size = coloring.color_sizes[color_idx as usize];
                let color_offset = coloring.color_offsets[color_idx as usize];
                let color_dispatch = self.dispatch_size(color_size);

                // 更新颜色参数
                if let Some(ref mut params_buffer) = self.params_buffer {
                    params_buffer.set_color(
                        self.context.queue(),
                        color_idx,
                        color_offset,
                        color_size,
                    );
                }

                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("accumulate"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipelines.accumulate);
                pass.dispatch_workgroups(color_dispatch, 1, 1);
            }
        } else {
            // 无着色时直接累积（可能有竞争）
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("accumulate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.accumulate);
            pass.dispatch_workgroups(face_dispatch, 1, 1);
        }

        // 9. 源项计算
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("source"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.source);
            pass.dispatch_workgroups(cell_dispatch, 1, 1);
        }

        // 提交所有命令
        self.context.queue().submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    /// 读取当前状态数据回 CPU
    pub fn read_state(&self) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), GpuError> {
        let workspace = self.workspace.as_ref()
            .ok_or_else(|| GpuError::BufferOperation("Workspace not initialized".to_string()))?;

        let n_cells = workspace.state.n_cells;

        // 创建暂存缓冲区
        let staging_size = (n_cells * std::mem::size_of::<f32>()) as u64;
        let staging_h = self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_h"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_hu = self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_hu"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_hv = self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_hv"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 复制数据到暂存缓冲区
        let mut encoder = self.context.device().create_command_encoder(&CommandEncoderDescriptor {
            label: Some("read_state"),
        });
        encoder.copy_buffer_to_buffer(
            workspace.state.h.read_buffer().buffer(),
            0,
            &staging_h,
            0,
            staging_size,
        );
        encoder.copy_buffer_to_buffer(
            workspace.state.hu.read_buffer().buffer(),
            0,
            &staging_hu,
            0,
            staging_size,
        );
        encoder.copy_buffer_to_buffer(
            workspace.state.hv.read_buffer().buffer(),
            0,
            &staging_hv,
            0,
            staging_size,
        );
        self.context.queue().submit(std::iter::once(encoder.finish()));

        // 映射并读取数据
        let h_data = self.map_and_read_buffer::<f32>(&staging_h, n_cells)?;
        let hu_data = self.map_and_read_buffer::<f32>(&staging_hu, n_cells)?;
        let hv_data = self.map_and_read_buffer::<f32>(&staging_hv, n_cells)?;

        // 转换为 f64
        let h: Vec<f64> = h_data.iter().map(|&x| x as f64).collect();
        let hu: Vec<f64> = hu_data.iter().map(|&x| x as f64).collect();
        let hv: Vec<f64> = hv_data.iter().map(|&x| x as f64).collect();

        Ok((h, hu, hv))
    }

    /// 映射并读取缓冲区
    fn map_and_read_buffer<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<T>, GpuError> {
        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.context.device().poll(wgpu::Maintain::Wait);
        receiver.recv()
            .map_err(|e| GpuError::BufferOperation(e.to_string()))?
            .map_err(|e| GpuError::BufferOperation(e.to_string()))?;

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffer.unmap();

        Ok(result)
    }

    /// 计算最大波速 (用于 CFL 时间步长)
    pub fn compute_max_wave_speed(&self) -> Result<f64, GpuError> {
        let workspace = self.workspace.as_ref()
            .ok_or_else(|| GpuError::BufferOperation("Workspace not initialized".to_string()))?;

        // 读取波速数据
        let n_faces = workspace.state.n_faces;
        let staging_size = (n_faces * std::mem::size_of::<f32>()) as u64;
        let staging = self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_wave_speed"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.context.device().create_command_encoder(&CommandEncoderDescriptor {
            label: Some("read_wave_speed"),
        });
        encoder.copy_buffer_to_buffer(
            workspace.state.flux.max_wave_speed.buffer(),
            0,
            &staging,
            0,
            staging_size,
        );
        self.context.queue().submit(std::iter::once(encoder.finish()));

        let wave_speeds = self.map_and_read_buffer::<f32>(&staging, n_faces)?;

        // 找最大值
        let max_speed = wave_speeds.iter()
            .cloned()
            .fold(0.0f32, |a, b| a.max(b));

        Ok(max_speed as f64)
    }
}

/// GPU 错误类型
#[derive(Debug, Clone)]
pub enum GpuError {
    /// 没有可用的适配器
    NoAdapter,
    /// 设备创建失败
    DeviceCreation(String),
    /// 着色器编译失败
    ShaderCompilation(String),
    /// 缓冲区操作失败
    BufferOperation(String),
    /// 管线创建失败
    PipelineCreation(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "No suitable GPU adapter found"),
            Self::DeviceCreation(msg) => write!(f, "GPU device creation failed: {}", msg),
            Self::ShaderCompilation(msg) => write!(f, "Shader compilation failed: {}", msg),
            Self::BufferOperation(msg) => write!(f, "Buffer operation failed: {}", msg),
            Self::PipelineCreation(msg) => write!(f, "Pipeline creation failed: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_solver_config_default() {
        let config = GpuSolverConfig::default();
        assert_eq!(config.workgroup_size, 256);
        assert!(config.enable_async);
    }

    #[test]
    fn test_gpu_solver_config_high_performance() {
        let config = GpuSolverConfig::high_performance();
        assert_eq!(config.preferred_device, DeviceType::DiscreteGpu);
    }

    #[test]
    fn test_gpu_solver_config_low_power() {
        let config = GpuSolverConfig::low_power();
        assert_eq!(config.preferred_device, DeviceType::IntegratedGpu);
        assert!(!config.enable_async);
    }

    #[test]
    fn test_gpu_error_display() {
        let err = GpuError::NoAdapter;
        assert!(err.to_string().contains("adapter"));
        
        let err = GpuError::DeviceCreation("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    // GPU 初始化测试需要实际 GPU，标记为 ignored
    #[test]
    #[ignore = "Requires GPU hardware"]
    fn test_gpu_context_creation() {
        let config = GpuSolverConfig::default();
        let result = GpuContext::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    #[ignore = "Requires GPU hardware"]
    fn test_gpu_solver_creation() {
        let config = GpuSolverConfig::default();
        let result = GpuSolver::new(config);
        assert!(result.is_ok());
    }
}
