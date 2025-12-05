// crates/mh_physics/src/gpu/pipeline.rs

//! GPU 计算管线管理
//!
//! 提供计算着色器管线的创建和管理。

use std::collections::HashMap;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderStages,
};

/// 计算管线配置
#[derive(Debug, Clone)]
pub struct ComputePipelineConfig {
    /// 着色器源码
    pub shader_source: String,
    /// 入口点函数名
    pub entry_point: String,
    /// 工作组大小
    pub workgroup_size: (u32, u32, u32),
    /// 绑定布局
    pub bindings: Vec<BindingConfig>,
}

/// 绑定配置
#[derive(Debug, Clone)]
pub struct BindingConfig {
    /// 绑定索引
    pub binding: u32,
    /// 绑定类型
    pub ty: BindingConfigType,
    /// 是否只读
    pub read_only: bool,
}

/// 绑定类型配置
#[derive(Debug, Clone)]
pub enum BindingConfigType {
    /// 存储缓冲区
    StorageBuffer,
    /// 只读存储缓冲区
    ReadOnlyStorageBuffer,
    /// 统一缓冲区
    UniformBuffer,
}

impl BindingConfig {
    /// 创建存储缓冲区绑定
    pub fn storage(binding: u32, read_only: bool) -> Self {
        Self {
            binding,
            ty: if read_only {
                BindingConfigType::ReadOnlyStorageBuffer
            } else {
                BindingConfigType::StorageBuffer
            },
            read_only,
        }
    }

    /// 创建统一缓冲区绑定
    pub fn uniform(binding: u32) -> Self {
        Self {
            binding,
            ty: BindingConfigType::UniformBuffer,
            read_only: true,
        }
    }

    /// 转换为 wgpu 绑定组布局条目
    fn to_layout_entry(&self) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding: self.binding,
            visibility: ShaderStages::COMPUTE,
            ty: match self.ty {
                BindingConfigType::StorageBuffer => BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BindingConfigType::ReadOnlyStorageBuffer => BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BindingConfigType::UniformBuffer => BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
            count: None,
        }
    }
}

/// 已编译的计算管线
pub struct CompiledPipeline {
    /// wgpu 计算管线
    pipeline: ComputePipeline,
    /// 绑定组布局
    bind_group_layout: BindGroupLayout,
    /// 工作组大小
    workgroup_size: (u32, u32, u32),
    /// 入口点名称
    entry_point: String,
}

impl CompiledPipeline {
    /// 创建新的计算管线
    pub fn new(device: &Device, config: &ComputePipelineConfig, label: Option<&str>) -> Self {
        // 创建着色器模块
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: label.map(|l| format!("{}_shader", l)).as_deref(),
            source: wgpu::ShaderSource::Wgsl(config.shader_source.clone().into()),
        });

        // 创建绑定组布局
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: label.map(|l| format!("{}_layout", l)).as_deref(),
            entries: &config
                .bindings
                .iter()
                .map(|b| b.to_layout_entry())
                .collect::<Vec<_>>(),
        });

        // 创建管线布局
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: label.map(|l| format!("{}_pipeline_layout", l)).as_deref(),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // 创建计算管线
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(&config.entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            workgroup_size: config.workgroup_size,
            entry_point: config.entry_point.clone(),
        }
    }

    /// 创建绑定组
    pub fn create_bind_group(&self, device: &Device, buffers: &[&Buffer], label: Option<&str>) -> BindGroup {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();

        device.create_bind_group(&BindGroupDescriptor {
            label,
            layout: &self.bind_group_layout,
            entries: &entries,
        })
    }

    /// 获取管线引用
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }

    /// 获取绑定组布局
    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }

    /// 获取工作组大小
    pub fn workgroup_size(&self) -> (u32, u32, u32) {
        self.workgroup_size
    }

    /// 计算工作组数量
    pub fn compute_dispatch_size(&self, total_work: u32) -> u32 {
        (total_work + self.workgroup_size.0 - 1) / self.workgroup_size.0
    }
}

/// 管线缓存
pub struct PipelineCache {
    /// 已编译的管线
    pipelines: HashMap<String, CompiledPipeline>,
}

impl PipelineCache {
    /// 创建新的管线缓存
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
    }

    /// 获取或创建管线
    pub fn get_or_create(
        &mut self,
        device: &Device,
        name: &str,
        config: &ComputePipelineConfig,
    ) -> &CompiledPipeline {
        if !self.pipelines.contains_key(name) {
            let pipeline = CompiledPipeline::new(device, config, Some(name));
            self.pipelines.insert(name.to_string(), pipeline);
        }
        self.pipelines.get(name).unwrap()
    }

    /// 获取已有管线
    pub fn get(&self, name: &str) -> Option<&CompiledPipeline> {
        self.pipelines.get(name)
    }

    /// 检查管线是否存在
    pub fn contains(&self, name: &str) -> bool {
        self.pipelines.contains_key(name)
    }

    /// 清除所有管线
    pub fn clear(&mut self) {
        self.pipelines.clear();
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

/// 着色器源码常量
pub mod shaders {
    /// 通用工具着色器（占位，实际着色器应从文件加载）
    pub const COMMON: &str = r#"
// 通用定义
struct Params {
    n_cells: u32,
    dt: f32,
    g: f32,
    min_depth: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
"#;

    /// 工作组大小
    pub const WORKGROUP_SIZE: u32 = 256;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binding_config() {
        let storage = BindingConfig::storage(0, false);
        assert_eq!(storage.binding, 0);
        assert!(!storage.read_only);
        
        let uniform = BindingConfig::uniform(1);
        assert_eq!(uniform.binding, 1);
        assert!(uniform.read_only);
    }

    #[test]
    fn test_pipeline_cache_new() {
        let cache = PipelineCache::new();
        assert!(!cache.contains("test"));
    }
}
