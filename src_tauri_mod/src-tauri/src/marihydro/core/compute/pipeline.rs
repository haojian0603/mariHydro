// src-tauri/src/marihydro/core/compute/pipeline.rs

//! GPU计算管线管理
//!
//! 提供着色器编译、管线缓存和绑定组管理

use std::collections::HashMap;
use std::sync::Arc;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineCompilationOptions, PipelineLayoutDescriptor,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::marihydro::core::error::{MhError, MhResult};

/// 计算管线ID
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PipelineId {
    /// Green-Gauss梯度计算
    Gradient,
    /// Barth-Jespersen限制器
    Limiter,
    /// MUSCL重构
    Reconstruct,
    /// HLLC黎曼求解器
    HllcRiemann,
    /// 通量累加（着色版）
    FluxAccumulate,
    /// 源项计算（摩擦）
    SourceFriction,
    /// 源项计算（风应力）
    SourceWind,
    /// 时间积分（Euler）
    IntegrateEuler,
    /// 时间积分（SSP-RK2）
    IntegrateRk2,
    /// 归约求最大值
    ReduceMax,
    /// 归约求和
    ReduceSum,
}

/// 着色器源码
pub struct ShaderSource2 {
    /// 着色器名称
    pub name: &'static str,
    /// WGSL源码
    pub source: &'static str,
    /// 入口点
    pub entry_point: &'static str,
    /// 工作组大小
    pub workgroup_size: [u32; 3],
}

/// 管线缓存
pub struct PipelineCache {
    device: Arc<Device>,
    /// 已编译的着色器模块
    shader_modules: HashMap<&'static str, ShaderModule>,
    /// 已创建的计算管线
    pipelines: HashMap<PipelineId, ComputePipeline>,
    /// 绑定组布局
    bind_group_layouts: HashMap<PipelineId, BindGroupLayout>,
}

impl PipelineCache {
    /// 创建管线缓存
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            shader_modules: HashMap::new(),
            pipelines: HashMap::new(),
            bind_group_layouts: HashMap::new(),
        }
    }

    /// 编译着色器
    pub fn compile_shader(&mut self, name: &'static str, source: &str) -> MhResult<()> {
        if self.shader_modules.contains_key(name) {
            return Ok(()); // 已编译
        }

        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(name),
            source: ShaderSource::Wgsl(source),
        });

        self.shader_modules.insert(name, module);
        log::debug!("Compiled shader: {}", name);
        Ok(())
    }

    /// 获取或创建计算管线
    pub fn get_or_create_pipeline(
        &mut self,
        id: PipelineId,
        shader: &ShaderSource2,
        bindings: &[BindingEntry],
    ) -> MhResult<&ComputePipeline> {
        if !self.pipelines.contains_key(&id) {
            self.create_pipeline(id, shader, bindings)?;
        }
        
        Ok(self.pipelines.get(&id).unwrap())
    }

    /// 创建计算管线
    fn create_pipeline(
        &mut self,
        id: PipelineId,
        shader: &ShaderSource2,
        bindings: &[BindingEntry],
    ) -> MhResult<()> {
        // 编译着色器（如果尚未编译）
        self.compile_shader(shader.name, shader.source)?;
        let shader_module = self.shader_modules.get(shader.name).unwrap();

        // 创建绑定组布局
        let bind_group_layout = self.create_bind_group_layout(&id, bindings)?;

        // 创建管线布局
        let pipeline_layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{:?} Pipeline Layout", id)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // 创建计算管线
        let pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(&format!("{:?} Pipeline", id)),
            layout: Some(&pipeline_layout),
            module: shader_module,
            entry_point: Some(shader.entry_point),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        self.bind_group_layouts.insert(id, bind_group_layout);
        self.pipelines.insert(id, pipeline);

        log::debug!("Created pipeline: {:?}", id);
        Ok(())
    }

    /// 创建绑定组布局
    fn create_bind_group_layout(
        &self,
        id: &PipelineId,
        bindings: &[BindingEntry],
    ) -> MhResult<BindGroupLayout> {
        let entries: Vec<BindGroupLayoutEntry> = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| BindGroupLayoutEntry {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: match b.buffer_type {
                        BufferType::Storage => BufferBindingType::Storage { read_only: false },
                        BufferType::StorageReadOnly => BufferBindingType::Storage { read_only: true },
                        BufferType::Uniform => BufferBindingType::Uniform,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(&format!("{:?} Bind Group Layout", id)),
            entries: &entries,
        });

        Ok(layout)
    }

    /// 获取绑定组布局
    pub fn get_bind_group_layout(&self, id: PipelineId) -> Option<&BindGroupLayout> {
        self.bind_group_layouts.get(&id)
    }

    /// 创建绑定组
    pub fn create_bind_group(
        &self,
        id: PipelineId,
        resources: &[BindingResource],
    ) -> MhResult<BindGroup> {
        let layout = self.bind_group_layouts.get(&id).ok_or_else(|| {
            MhError::ComputeError(format!("Bind group layout not found for {:?}", id))
        })?;

        let entries: Vec<BindGroupEntry> = resources
            .iter()
            .enumerate()
            .map(|(i, r)| BindGroupEntry {
                binding: i as u32,
                resource: r.clone(),
            })
            .collect();

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some(&format!("{:?} Bind Group", id)),
            layout,
            entries: &entries,
        });

        Ok(bind_group)
    }

    /// 获取已缓存的管线数量
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// 获取已编译的着色器数量
    pub fn shader_count(&self) -> usize {
        self.shader_modules.len()
    }

    /// 清除缓存
    pub fn clear(&mut self) {
        self.pipelines.clear();
        self.bind_group_layouts.clear();
        self.shader_modules.clear();
    }
}

/// 绑定条目描述
#[derive(Debug, Clone, Copy)]
pub struct BindingEntry {
    /// 缓冲区类型
    pub buffer_type: BufferType,
}

/// 缓冲区类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferType {
    /// 存储缓冲区（读写）
    Storage,
    /// 只读存储缓冲区
    StorageReadOnly,
    /// Uniform缓冲区
    Uniform,
}

impl BindingEntry {
    /// 创建存储缓冲区绑定
    pub const fn storage() -> Self {
        Self {
            buffer_type: BufferType::Storage,
        }
    }

    /// 创建只读存储缓冲区绑定
    pub const fn storage_readonly() -> Self {
        Self {
            buffer_type: BufferType::StorageReadOnly,
        }
    }

    /// 创建Uniform缓冲区绑定
    pub const fn uniform() -> Self {
        Self {
            buffer_type: BufferType::Uniform,
        }
    }
}

/// 预定义的着色器源码集合
pub mod shaders {
    use super::ShaderSource2;

    /// 通用常量和结构体
    pub const COMMON: &str = r#"
// 物理常量
const G: f32 = 9.80665;  // 重力加速度
const EPS_H: f32 = 1e-6; // 最小水深
const PI: f32 = 3.14159265359;

// 无效单元标记
const INVALID_CELL: u32 = 0xFFFFFFFFu;

// 状态结构
struct State {
    h: f32,
    hu: f32,
    hv: f32,
    z: f32,
}

// 通量结构
struct Flux {
    f_h: f32,
    f_hu: f32,
    f_hv: f32,
}

// 从守恒变量计算速度
fn get_velocity(state: State) -> vec2<f32> {
    let h_safe = max(state.h, EPS_H);
    return vec2<f32>(state.hu / h_safe, state.hv / h_safe);
}

// 计算波速
fn wave_speed(h: f32) -> f32 {
    return sqrt(G * max(h, 0.0));
}
"#;

    /// 梯度计算着色器
    pub const GRADIENT: ShaderSource2 = ShaderSource2 {
        name: "gradient",
        source: r#"
// 梯度计算 (Green-Gauss)

struct Uniforms {
    n_cells: u32,
    n_faces: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read> cell_area: array<f32>;
@group(0) @binding(3) var<storage, read> face_center_x: array<f32>;
@group(0) @binding(4) var<storage, read> face_center_y: array<f32>;
@group(0) @binding(5) var<storage, read> face_normal_x: array<f32>;
@group(0) @binding(6) var<storage, read> face_normal_y: array<f32>;
@group(0) @binding(7) var<storage, read> face_length: array<f32>;
@group(0) @binding(8) var<storage, read> face_owner: array<u32>;
@group(0) @binding(9) var<storage, read> face_neighbor: array<u32>;
@group(0) @binding(10) var<storage, read_write> grad_x: array<f32>;
@group(0) @binding(11) var<storage, read_write> grad_y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let face_idx = gid.x;
    if (face_idx >= uniforms.n_faces) {
        return;
    }
    
    let owner = face_owner[face_idx];
    let neighbor = face_neighbor[face_idx];
    
    // 计算面值（简单平均或边界处理）
    var face_value: f32;
    if (neighbor == 0xFFFFFFFFu) {
        face_value = values[owner];
    } else {
        face_value = 0.5 * (values[owner] + values[neighbor]);
    }
    
    // 面贡献
    let nx = face_normal_x[face_idx];
    let ny = face_normal_y[face_idx];
    let len = face_length[face_idx];
    let contrib_x = face_value * nx * len;
    let contrib_y = face_value * ny * len;
    
    // 原子累加到owner
    // 注意：wgpu不支持atomicAdd on f32，需要使用i32技巧或着色累加
    // 这里简化为直接写入（需要着色并行）
}
"#,
        entry_point: "main",
        workgroup_size: [256, 1, 1],
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binding_entry() {
        let storage = BindingEntry::storage();
        assert_eq!(storage.buffer_type, BufferType::Storage);
        
        let readonly = BindingEntry::storage_readonly();
        assert_eq!(readonly.buffer_type, BufferType::StorageReadOnly);
    }

    #[test]
    fn test_pipeline_id() {
        let id1 = PipelineId::Gradient;
        let id2 = PipelineId::Gradient;
        assert_eq!(id1, id2);
    }
}
