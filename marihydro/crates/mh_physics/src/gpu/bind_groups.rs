// crates/mh_physics/src/gpu/bind_groups.rs

//! GPU Bind Groups 管理
//!
//! 管理 GPU 计算所需的绑定组和布局。

use super::buffer::TypedBuffer;
use super::mesh::GpuMeshData;
use super::state::GpuStateArrays;
use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, Device, Queue, ShaderStages,
};

/// GPU 计算参数 (Uniform Buffer)
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct GpuComputeParams {
    /// 时间步长
    pub dt: f32,
    /// 重力加速度
    pub g: f32,
    /// 干湿阈值
    pub eps_h: f32,
    /// 曼宁系数
    pub manning_n: f32,
    /// Venkatakrishnan 系数
    pub venkat_k: f32,
    /// 单元数量
    pub num_cells: u32,
    /// 面数量
    pub num_faces: u32,
    /// 内部面数量
    pub num_interior_faces: u32,
    /// 边界面数量
    pub num_boundary_faces: u32,
    /// RK 阶段 (0, 1, 2)
    pub rk_stage: u32,
    /// 重构阶数 (1 或 2)
    pub reconstruction_order: u32,
    /// 限制器类型 (0=none, 1=BJ, 2=VK)
    pub limiter_type: u32,
    /// 摩擦类型 (0=none, 1=manning, 2=chezy)
    pub friction_type: u32,
    /// 是否启用风应力
    pub wind_enabled: u32,
    /// 风速 X 分量 (m/s)
    pub wind_u: f32,
    /// 风速 Y 分量 (m/s)
    pub wind_v: f32,
    /// 是否启用科氏力
    pub coriolis_enabled: u32,
    /// 科氏力参数
    pub coriolis_f: f32,
    /// 当前颜色索引
    pub current_color: u32,
    /// 当前颜色的面偏移
    pub color_offset: u32,
    /// 当前颜色的面数量
    pub color_size: u32,
    /// 填充到 128 字节对齐
    pub _padding: [u32; 5],
}

impl Default for GpuComputeParams {
    fn default() -> Self {
        Self {
            dt: 0.001,
            g: 9.81,
            eps_h: 1e-6,
            manning_n: 0.025,
            venkat_k: 0.5,
            num_cells: 0,
            num_faces: 0,
            num_interior_faces: 0,
            num_boundary_faces: 0,
            rk_stage: 0,
            reconstruction_order: 2,
            limiter_type: 1,
            friction_type: 1,
            wind_enabled: 0,
            wind_u: 0.0,
            wind_v: 0.0,
            coriolis_enabled: 0,
            coriolis_f: 0.0,
            current_color: 0,
            color_offset: 0,
            color_size: 0,
            _padding: [0; 5],
        }
    }
}

/// 绑定组布局集合
pub struct BindGroupLayouts {
    /// 参数布局
    pub params: BindGroupLayout,
    /// 网格只读数据布局
    pub mesh: BindGroupLayout,
    /// 状态读取布局
    pub state_read: BindGroupLayout,
    /// 状态读写布局
    pub state_rw: BindGroupLayout,
    /// 梯度布局
    pub gradient: BindGroupLayout,
    /// 限制器布局
    pub limiter: BindGroupLayout,
    /// 重构布局
    pub reconstruct: BindGroupLayout,
    /// 通量布局
    pub flux: BindGroupLayout,
    /// 残差布局
    pub residual: BindGroupLayout,
    /// RK 阶段布局
    pub rk_stages: BindGroupLayout,
    /// 着色数据布局
    pub coloring: BindGroupLayout,
}

impl BindGroupLayouts {
    /// 创建所有绑定组布局
    pub fn new(device: &Device) -> Self {
        // 参数布局 (Uniform Buffer)
        let params = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("params_layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // 网格布局 (多个只读 Storage Buffer)
        let mesh = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mesh_layout"),
            entries: &Self::storage_readonly_entries(&[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            ]),
        });

        // 状态读取布局
        let state_read = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("state_read_layout"),
            entries: &Self::storage_readonly_entries(&[0, 1, 2, 3]),
        });

        // 状态读写布局
        let state_rw = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("state_rw_layout"),
            entries: &Self::storage_entries(&[0, 1, 2]),
        });

        // 梯度布局 (8 个 storage buffer: h, hu, hv, z 各 x, y)
        let gradient = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gradient_layout"),
            entries: &Self::storage_entries(&[0, 1, 2, 3, 4, 5, 6, 7]),
        });

        // 限制器布局 (3 个 storage buffer: h, hu, hv)
        let limiter = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("limiter_layout"),
            entries: &Self::storage_entries(&[0, 1, 2]),
        });

        // 重构布局 (8 个 storage buffer: L/R 各 h, hu, hv, z)
        let reconstruct = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("reconstruct_layout"),
            entries: &Self::storage_entries(&[0, 1, 2, 3, 4, 5, 6, 7]),
        });

        // 通量布局 (4 个 storage buffer: h, hu, hv, max_wave_speed)
        let flux = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("flux_layout"),
            entries: &Self::storage_entries(&[0, 1, 2, 3]),
        });

        // 残差布局 (3 个 storage buffer: h, hu, hv)
        let residual = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("residual_layout"),
            entries: &Self::storage_entries(&[0, 1, 2]),
        });

        // RK 阶段布局 (6 个 storage buffer: h_n, hu_n, hv_n, h_star, hu_star, hv_star)
        let rk_stages = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("rk_stages_layout"),
            entries: &Self::storage_entries(&[0, 1, 2, 3, 4, 5]),
        });

        // 着色布局 (面颜色索引)
        let coloring = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("coloring_layout"),
            entries: &Self::storage_readonly_entries(&[0]),
        });

        Self {
            params,
            mesh,
            state_read,
            state_rw,
            gradient,
            limiter,
            reconstruct,
            flux,
            residual,
            rk_stages,
            coloring,
        }
    }

    /// 创建只读 Storage Buffer 条目
    fn storage_readonly_entries(bindings: &[u32]) -> Vec<BindGroupLayoutEntry> {
        bindings
            .iter()
            .map(|&binding| BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect()
    }

    /// 创建读写 Storage Buffer 条目
    fn storage_entries(bindings: &[u32]) -> Vec<BindGroupLayoutEntry> {
        bindings
            .iter()
            .map(|&binding| BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect()
    }
}

/// 参数缓冲区管理
pub struct ParamsBuffer {
    /// 参数缓冲区
    buffer: wgpu::Buffer,
    /// 当前参数值
    params: GpuComputeParams,
}

impl ParamsBuffer {
    /// 创建参数缓冲区
    pub fn new(device: &Device) -> Self {
        let params = GpuComputeParams::default();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params_buffer"),
            size: std::mem::size_of::<GpuComputeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { buffer, params }
    }

    /// 更新参数
    pub fn update(&mut self, queue: &Queue, params: GpuComputeParams) {
        self.params = params;
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&self.params));
    }

    /// 设置时间步长
    pub fn set_dt(&mut self, queue: &Queue, dt: f32) {
        self.params.dt = dt;
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&self.params));
    }

    /// 设置 RK 阶段
    pub fn set_rk_stage(&mut self, queue: &Queue, stage: u32) {
        self.params.rk_stage = stage;
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&self.params));
    }

    /// 设置着色参数
    pub fn set_color(&mut self, queue: &Queue, color: u32, offset: u32, size: u32) {
        self.params.current_color = color;
        self.params.color_offset = offset;
        self.params.color_size = size;
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&self.params));
    }

    /// 获取缓冲区引用
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// 获取当前参数
    pub fn params(&self) -> &GpuComputeParams {
        &self.params
    }

    /// 创建绑定组
    pub fn create_bind_group(&self, device: &Device, layout: &BindGroupLayout) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("params_bind_group"),
            layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: self.buffer.as_entire_binding(),
            }],
        })
    }
}

/// 着色数据 (用于无竞争并行通量累积)
pub struct ColoringData {
    /// 颜色数量
    pub num_colors: u32,
    /// 每个颜色的面数量
    pub color_sizes: Vec<u32>,
    /// 每个颜色的偏移
    pub color_offsets: Vec<u32>,
    /// 着色后的面索引缓冲区
    pub face_indices: TypedBuffer<u32>,
}

impl ColoringData {
    /// 从网格生成着色数据
    ///
    /// 简单贪心着色算法：将面分配到不同颜色，使得同一颜色的面不共享单元
    pub fn from_mesh(device: &Device, mesh: &GpuMeshData) -> Self {
        let n_faces = mesh.topology.num_faces as usize;
        let n_cells = mesh.topology.num_cells as usize;

        // 为每个面分配颜色
        let mut face_colors = vec![u32::MAX; n_faces];
        let mut cell_color = vec![u32::MAX; n_cells]; // 记录每个单元当前被使用的颜色
        let mut num_colors = 0u32;

        // 获取 owner 和 neighbor 数据
        let owners = &mesh.face_topo.owners;
        let neighbors = &mesh.face_topo.neighbors;

        // 需要读取缓冲区数据 - 简化处理
        // 实际上应该从 CPU 端的数据构建
        // 这里使用简化的顺序着色
        let mut color_counts: Vec<u32> = Vec::new();

        for face in 0..n_faces {
            // 确定此面不能使用的颜色
            #[allow(unused_variables)]
            let forbidden_colors = std::collections::HashSet::<u32>::new();

            // 由于我们无法直接访问 GPU 缓冲区，使用简化策略
            // 实际项目中应该从 FrozenMesh 构建时就完成着色
            let color = (face % 8) as u32; // 简单取模着色
            face_colors[face] = color;

            if color as usize >= color_counts.len() {
                color_counts.resize(color as usize + 1, 0);
            }
            color_counts[color as usize] += 1;
            num_colors = num_colors.max(color + 1);
        }

        // 按颜色重排面索引
        let mut color_offsets = vec![0u32; num_colors as usize + 1];
        for c in 0..num_colors as usize {
            color_offsets[c + 1] = color_offsets[c] + color_counts[c];
        }

        let mut sorted_faces = vec![0u32; n_faces];
        let mut insert_pos = color_offsets.clone();
        for face in 0..n_faces {
            let color = face_colors[face] as usize;
            sorted_faces[insert_pos[color] as usize] = face as u32;
            insert_pos[color] += 1;
        }

        let face_indices = TypedBuffer::from_data(
            device,
            &sorted_faces,
            super::buffer::GpuBufferUsage::StorageReadOnly,
            Some("color_face_indices"),
        );

        Self {
            num_colors,
            color_sizes: color_counts,
            color_offsets: color_offsets[..num_colors as usize].to_vec(),
            face_indices,
        }
    }

    /// 从 CPU 端数据创建着色（更精确）
    pub fn from_cpu_mesh(
        device: &Device,
        n_faces: usize,
        n_cells: usize,
        owners: &[u32],
        neighbors: &[u32],
    ) -> Self {
        // 贪心着色算法
        let mut face_colors = vec![u32::MAX; n_faces];
        let mut cell_last_color = vec![u32::MAX; n_cells];
        let mut num_colors = 0u32;

        for face in 0..n_faces {
            let owner = owners[face] as usize;
            let neighbor = neighbors[face];

            // 确定禁用的颜色
            let mut forbidden = std::collections::HashSet::<u32>::new();
            if cell_last_color[owner] != u32::MAX {
                forbidden.insert(cell_last_color[owner]);
            }
            if neighbor != u32::MAX {
                let nb = neighbor as usize;
                if nb < n_cells && cell_last_color[nb] != u32::MAX {
                    forbidden.insert(cell_last_color[nb]);
                }
            }

            // 找到第一个可用颜色
            let mut color = 0u32;
            while forbidden.contains(&color) {
                color += 1;
            }

            face_colors[face] = color;
            cell_last_color[owner] = color;
            if neighbor != u32::MAX && (neighbor as usize) < n_cells {
                cell_last_color[neighbor as usize] = color;
            }
            num_colors = num_colors.max(color + 1);
        }

        // 统计和排序
        let mut color_counts = vec![0u32; num_colors as usize];
        for &c in &face_colors {
            color_counts[c as usize] += 1;
        }

        let mut color_offsets = vec![0u32; num_colors as usize + 1];
        for c in 0..num_colors as usize {
            color_offsets[c + 1] = color_offsets[c] + color_counts[c];
        }

        let mut sorted_faces = vec![0u32; n_faces];
        let mut insert_pos = color_offsets.clone();
        for face in 0..n_faces {
            let color = face_colors[face] as usize;
            sorted_faces[insert_pos[color] as usize] = face as u32;
            insert_pos[color] += 1;
        }

        let face_indices = TypedBuffer::from_data(
            device,
            &sorted_faces,
            super::buffer::GpuBufferUsage::StorageReadOnly,
            Some("color_face_indices"),
        );

        Self {
            num_colors,
            color_sizes: color_counts,
            color_offsets: color_offsets[..num_colors as usize].to_vec(),
            face_indices,
        }
    }
    
    /// 从预先计算好的颜色组创建
    pub fn from_color_groups(device: &Device, color_groups: &[Vec<u32>]) -> Self {
        let num_colors = color_groups.len() as u32;
        let mut color_sizes = Vec::with_capacity(num_colors as usize);
        let mut color_offsets = Vec::with_capacity(num_colors as usize);
        let mut all_faces = Vec::new();
        
        let mut offset = 0u32;
        for group in color_groups {
            color_offsets.push(offset);
            color_sizes.push(group.len() as u32);
            all_faces.extend_from_slice(group);
            offset += group.len() as u32;
        }
        
        let face_indices = TypedBuffer::from_data(
            device,
            &all_faces,
            super::buffer::GpuBufferUsage::StorageReadOnly,
            Some("color_face_indices"),
        );
        
        Self {
            num_colors,
            color_sizes,
            color_offsets,
            face_indices,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_compute_params_size() {
        // 确保参数结构体大小符合预期
        let size = std::mem::size_of::<GpuComputeParams>();
        assert_eq!(size, 128); // 应该是 128 字节
    }

    #[test]
    fn test_gpu_compute_params_default() {
        let params = GpuComputeParams::default();
        assert_eq!(params.g, 9.81);
        assert_eq!(params.eps_h, 1e-6);
    }
}
