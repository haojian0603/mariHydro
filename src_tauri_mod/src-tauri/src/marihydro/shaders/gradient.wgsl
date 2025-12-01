// MariHydro GPU计算着色器 - Green-Gauss梯度计算
// gradient.wgsl

// 引入公共模块定义（编译时会被预处理器拼接）

// ==================== Uniform缓冲区 ====================
struct GradientParams {
    num_cells: u32,
    num_faces: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: GradientParams;

// ==================== 存储缓冲区 ====================

// 单元几何数据 (只读)
@group(0) @binding(1) var<storage, read> cell_areas: array<f32>;
@group(0) @binding(2) var<storage, read> cell_cx: array<f32>;
@group(0) @binding(3) var<storage, read> cell_cy: array<f32>;

// 面几何数据 (只读)
@group(0) @binding(4) var<storage, read> face_cx: array<f32>;
@group(0) @binding(5) var<storage, read> face_cy: array<f32>;
@group(0) @binding(6) var<storage, read> face_nx: array<f32>;
@group(0) @binding(7) var<storage, read> face_ny: array<f32>;
@group(0) @binding(8) var<storage, read> face_length: array<f32>;

// 面拓扑 (CSR格式)
@group(0) @binding(9) var<storage, read> face_owner: array<u32>;
@group(0) @binding(10) var<storage, read> face_neighbor: array<u32>;

// 单元-面邻接 (CSR格式)
@group(0) @binding(11) var<storage, read> cell_face_ptr: array<u32>;
@group(0) @binding(12) var<storage, read> cell_face_idx: array<u32>;

// 状态数据 (只读)
@group(0) @binding(13) var<storage, read> cell_h: array<f32>;
@group(0) @binding(14) var<storage, read> cell_hu: array<f32>;
@group(0) @binding(15) var<storage, read> cell_hv: array<f32>;
@group(0) @binding(16) var<storage, read> cell_z: array<f32>;

// 梯度输出 (读写)
@group(0) @binding(17) var<storage, read_write> grad_h_x: array<f32>;
@group(0) @binding(18) var<storage, read_write> grad_h_y: array<f32>;
@group(0) @binding(19) var<storage, read_write> grad_hu_x: array<f32>;
@group(0) @binding(20) var<storage, read_write> grad_hu_y: array<f32>;
@group(0) @binding(21) var<storage, read_write> grad_hv_x: array<f32>;
@group(0) @binding(22) var<storage, read_write> grad_hv_y: array<f32>;
@group(0) @binding(23) var<storage, read_write> grad_z_x: array<f32>;
@group(0) @binding(24) var<storage, read_write> grad_z_y: array<f32>;

// ==================== 常量 ====================
const WORKGROUP_SIZE: u32 = 256u;
const INVALID: u32 = 0xFFFFFFFFu;
const EPS: f32 = 1e-10;

// ==================== 主计算核心 ====================

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_id = gid.x;
    
    if cell_id >= params.num_cells {
        return;
    }
    
    let area = cell_areas[cell_id];
    if area < EPS {
        // 无效单元，梯度为零
        grad_h_x[cell_id] = 0.0;
        grad_h_y[cell_id] = 0.0;
        grad_hu_x[cell_id] = 0.0;
        grad_hu_y[cell_id] = 0.0;
        grad_hv_x[cell_id] = 0.0;
        grad_hv_y[cell_id] = 0.0;
        grad_z_x[cell_id] = 0.0;
        grad_z_y[cell_id] = 0.0;
        return;
    }
    
    let inv_area = 1.0 / area;
    
    // 本单元中心和状态
    let c_cx = cell_cx[cell_id];
    let c_cy = cell_cy[cell_id];
    let c_h = cell_h[cell_id];
    let c_hu = cell_hu[cell_id];
    let c_hv = cell_hv[cell_id];
    let c_z = cell_z[cell_id];
    
    // 累积梯度
    var sum_h_x: f32 = 0.0;
    var sum_h_y: f32 = 0.0;
    var sum_hu_x: f32 = 0.0;
    var sum_hu_y: f32 = 0.0;
    var sum_hv_x: f32 = 0.0;
    var sum_hv_y: f32 = 0.0;
    var sum_z_x: f32 = 0.0;
    var sum_z_y: f32 = 0.0;
    
    // 遍历单元的所有面
    let face_start = cell_face_ptr[cell_id];
    let face_end = cell_face_ptr[cell_id + 1u];
    
    for (var i = face_start; i < face_end; i = i + 1u) {
        let face_id = cell_face_idx[i];
        
        // 获取面几何
        let f_cx = face_cx[face_id];
        let f_cy = face_cy[face_id];
        var f_nx = face_nx[face_id];
        var f_ny = face_ny[face_id];
        let f_len = face_length[face_id];
        
        // 确定法向方向（指向外部）
        let owner = face_owner[face_id];
        let neighbor = face_neighbor[face_id];
        
        if cell_id != owner {
            // 法向需要翻转
            f_nx = -f_nx;
            f_ny = -f_ny;
        }
        
        // 计算面上的值（使用邻居单元的值进行插值）
        var face_h: f32;
        var face_hu: f32;
        var face_hv: f32;
        var face_z: f32;
        
        if neighbor == INVALID {
            // 边界面，使用本单元值
            face_h = c_h;
            face_hu = c_hu;
            face_hv = c_hv;
            face_z = c_z;
        } else {
            // 内部面，使用两侧单元的平均值
            let other_id = select(owner, neighbor, cell_id == owner);
            let o_h = cell_h[other_id];
            let o_hu = cell_hu[other_id];
            let o_hv = cell_hv[other_id];
            let o_z = cell_z[other_id];
            
            face_h = 0.5 * (c_h + o_h);
            face_hu = 0.5 * (c_hu + o_hu);
            face_hv = 0.5 * (c_hv + o_hv);
            face_z = 0.5 * (c_z + o_z);
        }
        
        // Green-Gauss积分：grad(phi) = (1/V) * sum(phi_f * n_f * A_f)
        let weighted_nx = f_nx * f_len;
        let weighted_ny = f_ny * f_len;
        
        sum_h_x = sum_h_x + face_h * weighted_nx;
        sum_h_y = sum_h_y + face_h * weighted_ny;
        sum_hu_x = sum_hu_x + face_hu * weighted_nx;
        sum_hu_y = sum_hu_y + face_hu * weighted_ny;
        sum_hv_x = sum_hv_x + face_hv * weighted_nx;
        sum_hv_y = sum_hv_y + face_hv * weighted_ny;
        sum_z_x = sum_z_x + face_z * weighted_nx;
        sum_z_y = sum_z_y + face_z * weighted_ny;
    }
    
    // 除以单元面积得到最终梯度
    grad_h_x[cell_id] = sum_h_x * inv_area;
    grad_h_y[cell_id] = sum_h_y * inv_area;
    grad_hu_x[cell_id] = sum_hu_x * inv_area;
    grad_hu_y[cell_id] = sum_hu_y * inv_area;
    grad_hv_x[cell_id] = sum_hv_x * inv_area;
    grad_hv_y[cell_id] = sum_hv_y * inv_area;
    grad_z_x[cell_id] = sum_z_x * inv_area;
    grad_z_y[cell_id] = sum_z_y * inv_area;
}

// ==================== 辅助Kernel：初始化梯度为零 ====================
@compute @workgroup_size(256)
fn clear_gradients(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_id = gid.x;
    
    if cell_id >= params.num_cells {
        return;
    }
    
    grad_h_x[cell_id] = 0.0;
    grad_h_y[cell_id] = 0.0;
    grad_hu_x[cell_id] = 0.0;
    grad_hu_y[cell_id] = 0.0;
    grad_hv_x[cell_id] = 0.0;
    grad_hv_y[cell_id] = 0.0;
    grad_z_x[cell_id] = 0.0;
    grad_z_y[cell_id] = 0.0;
}
