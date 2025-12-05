// crates/mh_physics/src/gpu/state.rs

//! GPU 状态缓冲区管理
//!
//! 提供 GPU 计算所需的所有状态缓冲区。

use super::buffer::{DoubleBuffer, GpuBufferUsage, TypedBuffer};
use wgpu::{Device, Queue};

/// GPU 梯度数组
pub struct GpuGradientArrays {
    /// X 方向梯度
    pub x: TypedBuffer<f32>,
    /// Y 方向梯度
    pub y: TypedBuffer<f32>,
}

impl GpuGradientArrays {
    /// 创建新的梯度数组
    pub fn new(device: &Device, n_cells: usize) -> Self {
        Self {
            x: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("grad_x")),
            y: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("grad_y")),
        }
    }
}

/// GPU 重构值数组 (面左右两侧)
pub struct GpuReconstructArrays {
    /// 左侧水深
    pub h_l: TypedBuffer<f32>,
    /// 左侧 X 动量
    pub hu_l: TypedBuffer<f32>,
    /// 左侧 Y 动量
    pub hv_l: TypedBuffer<f32>,
    /// 左侧底床
    pub z_l: TypedBuffer<f32>,
    /// 右侧水深
    pub h_r: TypedBuffer<f32>,
    /// 右侧 X 动量
    pub hu_r: TypedBuffer<f32>,
    /// 右侧 Y 动量
    pub hv_r: TypedBuffer<f32>,
    /// 右侧底床
    pub z_r: TypedBuffer<f32>,
}

impl GpuReconstructArrays {
    /// 创建新的重构数组
    pub fn new(device: &Device, n_faces: usize) -> Self {
        Self {
            h_l: TypedBuffer::new(device, n_faces, GpuBufferUsage::Storage, Some("recon_h_l")),
            hu_l: TypedBuffer::new(device, n_faces, GpuBufferUsage::Storage, Some("recon_hu_l")),
            hv_l: TypedBuffer::new(device, n_faces, GpuBufferUsage::Storage, Some("recon_hv_l")),
            z_l: TypedBuffer::new(device, n_faces, GpuBufferUsage::Storage, Some("recon_z_l")),
            h_r: TypedBuffer::new(device, n_faces, GpuBufferUsage::Storage, Some("recon_h_r")),
            hu_r: TypedBuffer::new(device, n_faces, GpuBufferUsage::Storage, Some("recon_hu_r")),
            hv_r: TypedBuffer::new(device, n_faces, GpuBufferUsage::Storage, Some("recon_hv_r")),
            z_r: TypedBuffer::new(device, n_faces, GpuBufferUsage::Storage, Some("recon_z_r")),
        }
    }
}

/// GPU 通量数组
pub struct GpuFluxArrays {
    /// 水深通量
    pub h: TypedBuffer<f32>,
    /// X 动量通量
    pub hu: TypedBuffer<f32>,
    /// Y 动量通量
    pub hv: TypedBuffer<f32>,
    /// 最大波速 (用于 CFL)
    pub max_wave_speed: TypedBuffer<f32>,
}

impl GpuFluxArrays {
    /// 创建新的通量数组
    pub fn new(device: &Device, n_faces: usize) -> Self {
        Self {
            h: TypedBuffer::new(device, n_faces, GpuBufferUsage::Storage, Some("flux_h")),
            hu: TypedBuffer::new(device, n_faces, GpuBufferUsage::Storage, Some("flux_hu")),
            hv: TypedBuffer::new(device, n_faces, GpuBufferUsage::Storage, Some("flux_hv")),
            max_wave_speed: TypedBuffer::new(device, n_faces, GpuBufferUsage::Storage, Some("max_wave_speed")),
        }
    }
}

/// GPU 残差数组
pub struct GpuResidualArrays {
    /// 水深残差
    pub h: TypedBuffer<f32>,
    /// X 动量残差
    pub hu: TypedBuffer<f32>,
    /// Y 动量残差
    pub hv: TypedBuffer<f32>,
}

impl GpuResidualArrays {
    /// 创建新的残差数组
    pub fn new(device: &Device, n_cells: usize) -> Self {
        Self {
            h: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("residual_h")),
            hu: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("residual_hu")),
            hv: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("residual_hv")),
        }
    }
}

/// GPU 源项数组
pub struct GpuSourceArrays {
    /// 水深源项
    pub h: TypedBuffer<f32>,
    /// X 动量源项
    pub hu: TypedBuffer<f32>,
    /// Y 动量源项
    pub hv: TypedBuffer<f32>,
}

impl GpuSourceArrays {
    /// 创建新的源项数组
    pub fn new(device: &Device, n_cells: usize) -> Self {
        Self {
            h: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("source_h")),
            hu: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("source_hu")),
            hv: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("source_hv")),
        }
    }
}

/// GPU RK 中间状态
pub struct GpuRkStages {
    /// 初始状态水深
    pub h_n: TypedBuffer<f32>,
    /// 初始状态 X 动量
    pub hu_n: TypedBuffer<f32>,
    /// 初始状态 Y 动量
    pub hv_n: TypedBuffer<f32>,
    /// 中间状态水深
    pub h_star: TypedBuffer<f32>,
    /// 中间状态 X 动量
    pub hu_star: TypedBuffer<f32>,
    /// 中间状态 Y 动量
    pub hv_star: TypedBuffer<f32>,
}

impl GpuRkStages {
    /// 创建新的 RK 状态缓冲区
    pub fn new(device: &Device, n_cells: usize) -> Self {
        Self {
            h_n: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("h_n")),
            hu_n: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("hu_n")),
            hv_n: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("hv_n")),
            h_star: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("h_star")),
            hu_star: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("hu_star")),
            hv_star: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("hv_star")),
        }
    }
}

/// GPU 限制器数组
pub struct GpuLimiterArrays {
    /// 水深限制器值
    pub h: TypedBuffer<f32>,
    /// X 动量限制器值
    pub hu: TypedBuffer<f32>,
    /// Y 动量限制器值
    pub hv: TypedBuffer<f32>,
}

impl GpuLimiterArrays {
    /// 创建新的限制器数组
    pub fn new(device: &Device, n_cells: usize) -> Self {
        Self {
            h: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("limiter_h")),
            hu: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("limiter_hu")),
            hv: TypedBuffer::new(device, n_cells, GpuBufferUsage::Storage, Some("limiter_hv")),
        }
    }
}

/// 完整的 GPU 状态数组集合
pub struct GpuStateArrays {
    /// 单元数量
    pub n_cells: usize,
    /// 面数量
    pub n_faces: usize,

    // ===== 主状态变量 (双缓冲) =====
    /// 水深
    pub h: DoubleBuffer<f32>,
    /// X 方向动量
    pub hu: DoubleBuffer<f32>,
    /// Y 方向动量
    pub hv: DoubleBuffer<f32>,
    /// 底床高程 (只读)
    pub z: TypedBuffer<f32>,

    // ===== 梯度 =====
    /// 水深梯度
    pub grad_h: GpuGradientArrays,
    /// X 动量梯度
    pub grad_hu: GpuGradientArrays,
    /// Y 动量梯度
    pub grad_hv: GpuGradientArrays,
    /// 底床梯度
    pub grad_z: GpuGradientArrays,

    // ===== 限制器 =====
    /// 限制器值
    pub limiter: GpuLimiterArrays,

    // ===== 重构值 =====
    /// 面重构值
    pub recon: GpuReconstructArrays,

    // ===== 通量 =====
    /// 面通量
    pub flux: GpuFluxArrays,

    // ===== 残差 =====
    /// 单元残差
    pub residual: GpuResidualArrays,

    // ===== 源项 =====
    /// 源项
    pub source: GpuSourceArrays,

    // ===== RK 状态 =====
    /// RK 中间状态
    pub rk_stages: GpuRkStages,
}

impl GpuStateArrays {
    /// 创建新的 GPU 状态数组
    pub fn new(device: &Device, n_cells: usize, n_faces: usize) -> Self {
        Self {
            n_cells,
            n_faces,

            // 主状态 (双缓冲)
            h: DoubleBuffer::new(device, n_cells, GpuBufferUsage::Storage, "h"),
            hu: DoubleBuffer::new(device, n_cells, GpuBufferUsage::Storage, "hu"),
            hv: DoubleBuffer::new(device, n_cells, GpuBufferUsage::Storage, "hv"),
            z: TypedBuffer::new(device, n_cells, GpuBufferUsage::StorageReadOnly, Some("z")),

            // 梯度
            grad_h: GpuGradientArrays::new(device, n_cells),
            grad_hu: GpuGradientArrays::new(device, n_cells),
            grad_hv: GpuGradientArrays::new(device, n_cells),
            grad_z: GpuGradientArrays::new(device, n_cells),

            // 限制器
            limiter: GpuLimiterArrays::new(device, n_cells),

            // 重构
            recon: GpuReconstructArrays::new(device, n_faces),

            // 通量
            flux: GpuFluxArrays::new(device, n_faces),

            // 残差
            residual: GpuResidualArrays::new(device, n_cells),

            // 源项
            source: GpuSourceArrays::new(device, n_cells),

            // RK
            rk_stages: GpuRkStages::new(device, n_cells),
        }
    }

    /// 上传初始状态
    pub fn upload_state(&self, queue: &Queue, h: &[f64], hu: &[f64], hv: &[f64], z: &[f64]) {
        let h_f32: Vec<f32> = h.iter().map(|&x| x as f32).collect();
        let hu_f32: Vec<f32> = hu.iter().map(|&x| x as f32).collect();
        let hv_f32: Vec<f32> = hv.iter().map(|&x| x as f32).collect();
        let z_f32: Vec<f32> = z.iter().map(|&x| x as f32).collect();

        self.h.read_buffer().write(queue, &h_f32);
        self.hu.read_buffer().write(queue, &hu_f32);
        self.hv.read_buffer().write(queue, &hv_f32);
        self.z.write(queue, &z_f32);
    }

    /// 交换缓冲区
    pub fn swap(&mut self) {
        self.h.swap();
        self.hu.swap();
        self.hv.swap();
    }

    /// 保存当前状态到 RK 初始状态
    pub fn save_initial_state(&self, queue: &Queue) {
        // 实际实现需要使用 GPU copy 命令
        // 这里留作 TODO
    }
}

/// GPU 工作区
///
/// 包含网格和状态的完整 GPU 数据
pub struct GpuWorkspace {
    /// 状态数组
    pub state: GpuStateArrays,
    /// 网格数据
    pub mesh: super::mesh::GpuMeshData,
}

impl GpuWorkspace {
    /// 从 FrozenMesh 和初始状态创建工作区
    pub fn from_mesh(
        device: &Device,
        queue: &Queue,
        mesh: &mh_mesh::FrozenMesh,
    ) -> Self {
        let gpu_mesh = super::mesh::GpuMeshData::from_frozen_mesh(device, queue, mesh);
        let state = GpuStateArrays::new(
            device,
            gpu_mesh.topology.num_cells as usize,
            gpu_mesh.topology.num_faces as usize,
        );

        Self {
            state,
            mesh: gpu_mesh,
        }
    }

    /// 获取单元数量
    pub fn n_cells(&self) -> usize {
        self.state.n_cells
    }

    /// 获取面数量
    pub fn n_faces(&self) -> usize {
        self.state.n_faces
    }
}
