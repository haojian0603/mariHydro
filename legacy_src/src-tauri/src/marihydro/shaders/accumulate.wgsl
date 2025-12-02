// MariHydro GPU计算着色器 - 着色通量累积
// accumulate.wgsl
// 使用面着色策略避免原子操作竞争

// ==================== Uniform缓冲区 ====================
struct AccumulateParams {
    num_cells: u32,
    num_faces_this_color: u32,
    color_offset: u32,        // 当前颜色在索引数组中的偏移
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: AccumulateParams;

// ==================== 存储缓冲区 ====================

// 面拓扑
@group(0) @binding(1) var<storage, read> face_owner: array<u32>;
@group(0) @binding(2) var<storage, read> face_neighbor: array<u32>;

// 当前颜色的面索引列表
@group(0) @binding(3) var<storage, read> color_face_indices: array<u32>;

// 面通量 (由HLLC计算)
@group(0) @binding(4) var<storage, read> flux_h: array<f32>;
@group(0) @binding(5) var<storage, read> flux_hu: array<f32>;
@group(0) @binding(6) var<storage, read> flux_hv: array<f32>;

// 累积的残差 (读写)
@group(0) @binding(7) var<storage, read_write> residual_h: array<f32>;
@group(0) @binding(8) var<storage, read_write> residual_hu: array<f32>;
@group(0) @binding(9) var<storage, read_write> residual_hv: array<f32>;

// ==================== 常量 ====================
const INVALID: u32 = 0xFFFFFFFFu;

// ==================== 主计算核心 ====================
// 每次调用处理一种颜色的所有面

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_face_idx = gid.x;
    
    if local_face_idx >= params.num_faces_this_color {
        return;
    }
    
    // 获取实际面ID
    let face_id = color_face_indices[params.color_offset + local_face_idx];
    
    // 获取面的拓扑
    let owner = face_owner[face_id];
    let neighbor = face_neighbor[face_id];
    
    // 获取面通量
    let f_h = flux_h[face_id];
    let f_hu = flux_hu[face_id];
    let f_hv = flux_hv[face_id];
    
    // 累积到owner单元 (通量从owner流出，取负号)
    // 注意：同一颜色的面不会共享单元，因此无竞争
    residual_h[owner] = residual_h[owner] - f_h;
    residual_hu[owner] = residual_hu[owner] - f_hu;
    residual_hv[owner] = residual_hv[owner] - f_hv;
    
    // 累积到neighbor单元 (通量流入neighbor，取正号)
    if neighbor != INVALID {
        residual_h[neighbor] = residual_h[neighbor] + f_h;
        residual_hu[neighbor] = residual_hu[neighbor] + f_hu;
        residual_hv[neighbor] = residual_hv[neighbor] + f_hv;
    }
}

// ==================== 初始化残差为零 ====================

@compute @workgroup_size(256)
fn clear_residuals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_id = gid.x;
    
    if cell_id >= params.num_cells {
        return;
    }
    
    residual_h[cell_id] = 0.0;
    residual_hu[cell_id] = 0.0;
    residual_hv[cell_id] = 0.0;
}

// ==================== 备选：全局原子累积 ====================
// 使用定点数原子加法（有精度损失，用于验证）

struct AtomicAccumulateParams {
    num_faces: u32,
    scale_factor: f32,  // 定点数缩放因子
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> atomic_params: AtomicAccumulateParams;

// 原子累积缓冲区（i32格式）
@group(0) @binding(10) var<storage, read_write> atomic_residual_h: array<atomic<i32>>;
@group(0) @binding(11) var<storage, read_write> atomic_residual_hu: array<atomic<i32>>;
@group(0) @binding(12) var<storage, read_write> atomic_residual_hv: array<atomic<i32>>;

@compute @workgroup_size(256)
fn atomic_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let face_id = gid.x;
    
    if face_id >= atomic_params.num_faces {
        return;
    }
    
    let owner = face_owner[face_id];
    let neighbor = face_neighbor[face_id];
    
    let f_h = flux_h[face_id];
    let f_hu = flux_hu[face_id];
    let f_hv = flux_hv[face_id];
    
    let scale = atomic_params.scale_factor;
    
    // 转换为定点数
    let i_h = i32(f_h * scale);
    let i_hu = i32(f_hu * scale);
    let i_hv = i32(f_hv * scale);
    
    // 原子减法到owner
    atomicSub(&atomic_residual_h[owner], i_h);
    atomicSub(&atomic_residual_hu[owner], i_hu);
    atomicSub(&atomic_residual_hv[owner], i_hv);
    
    // 原子加法到neighbor
    if neighbor != INVALID {
        atomicAdd(&atomic_residual_h[neighbor], i_h);
        atomicAdd(&atomic_residual_hu[neighbor], i_hu);
        atomicAdd(&atomic_residual_hv[neighbor], i_hv);
    }
}

// 将原子结果转回浮点数
@compute @workgroup_size(256)
fn convert_atomic_to_float(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_id = gid.x;
    
    if cell_id >= params.num_cells {
        return;
    }
    
    let scale = atomic_params.scale_factor;
    let inv_scale = 1.0 / scale;
    
    residual_h[cell_id] = f32(atomicLoad(&atomic_residual_h[cell_id])) * inv_scale;
    residual_hu[cell_id] = f32(atomicLoad(&atomic_residual_hu[cell_id])) * inv_scale;
    residual_hv[cell_id] = f32(atomicLoad(&atomic_residual_hv[cell_id])) * inv_scale;
}
