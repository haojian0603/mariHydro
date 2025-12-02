// MariHydro GPU计算着色器 - 边界条件处理
// boundary.wgsl

// ==================== 边界类型枚举 ====================
const BC_WALL: u32 = 0u;           // 固壁边界（反射）
const BC_OPEN: u32 = 1u;           // 开边界（自由出流）
const BC_WATER_LEVEL: u32 = 2u;    // 水位边界
const BC_DISCHARGE: u32 = 3u;      // 流量边界
const BC_VELOCITY: u32 = 4u;       // 速度边界
const BC_RADIATION: u32 = 5u;      // 辐射边界
const BC_PERIODIC: u32 = 6u;       // 周期边界
const BC_ABSORBING: u32 = 7u;      // 吸收边界

// ==================== Uniform缓冲区 ====================
struct BoundaryParams {
    num_boundary_faces: u32,
    g: f32,
    eps_h: f32,
    time: f32,              // 当前时间 (用于时变边界)
}

@group(0) @binding(0) var<uniform> params: BoundaryParams;

// ==================== 存储缓冲区 ====================

// 边界面信息
@group(0) @binding(1) var<storage, read> boundary_face_ids: array<u32>;
@group(0) @binding(2) var<storage, read> boundary_types: array<u32>;
@group(0) @binding(3) var<storage, read> boundary_values: array<vec4<f32>>;  // 根据类型不同含义不同

// 面几何
@group(0) @binding(4) var<storage, read> face_nx: array<f32>;
@group(0) @binding(5) var<storage, read> face_ny: array<f32>;
@group(0) @binding(6) var<storage, read> face_length: array<f32>;
@group(0) @binding(7) var<storage, read> face_owner: array<u32>;

// 内部单元状态
@group(0) @binding(8) var<storage, read> cell_h: array<f32>;
@group(0) @binding(9) var<storage, read> cell_hu: array<f32>;
@group(0) @binding(10) var<storage, read> cell_hv: array<f32>;
@group(0) @binding(11) var<storage, read> cell_z: array<f32>;

// 边界面重构值 (读写，修改右侧值)
@group(0) @binding(12) var<storage, read_write> recon_h_R: array<f32>;
@group(0) @binding(13) var<storage, read_write> recon_hu_R: array<f32>;
@group(0) @binding(14) var<storage, read_write> recon_hv_R: array<f32>;
@group(0) @binding(15) var<storage, read_write> recon_z_R: array<f32>;

// 左侧值（只读）
@group(0) @binding(16) var<storage, read> recon_h_L: array<f32>;
@group(0) @binding(17) var<storage, read> recon_hu_L: array<f32>;
@group(0) @binding(18) var<storage, read> recon_hv_L: array<f32>;
@group(0) @binding(19) var<storage, read> recon_z_L: array<f32>;

// ==================== 边界处理函数 ====================

// 固壁边界：动量反射
fn apply_wall_bc(
    h_L: f32, hu_L: f32, hv_L: f32, z_L: f32,
    nx: f32, ny: f32
) -> vec4<f32> {
    // 法向动量反射，切向保持
    let un = hu_L * nx + hv_L * ny;  // 法向分量
    let ut = -hu_L * ny + hv_L * nx; // 切向分量
    
    // 反射法向，保持切向
    let un_R = -un;
    let ut_R = ut;
    
    // 转回全局坐标
    let hu_R = un_R * nx - ut_R * ny;
    let hv_R = un_R * ny + ut_R * nx;
    
    return vec4<f32>(h_L, hu_R, hv_R, z_L);
}

// 开边界：自由出流（零梯度外推）
fn apply_open_bc(
    h_L: f32, hu_L: f32, hv_L: f32, z_L: f32,
    nx: f32, ny: f32
) -> vec4<f32> {
    // 检查是否流出
    let un = hu_L * nx + hv_L * ny;
    
    if un >= 0.0 {
        // 流出：完全外推
        return vec4<f32>(h_L, hu_L, hv_L, z_L);
    } else {
        // 流入：减弱动量
        return vec4<f32>(h_L, 0.0, 0.0, z_L);
    }
}

// 水位边界
fn apply_water_level_bc(
    h_L: f32, hu_L: f32, hv_L: f32, z_L: f32,
    bc_value: vec4<f32>,
    nx: f32, ny: f32, g: f32
) -> vec4<f32> {
    let target_eta = bc_value.x;  // 目标水位
    let h_R = max(target_eta - z_L, 0.0);
    
    // 使用特征方程确定动量
    let un_L = (hu_L * nx + hv_L * ny) / max(h_L, params.eps_h);
    let c_L = sqrt(g * max(h_L, 0.0));
    let c_R = sqrt(g * max(h_R, 0.0));
    
    // 假设亚临界流入
    let un_R = un_L + 2.0 * (c_L - c_R);
    let ut_R = (-hu_L * ny + hv_L * nx) / max(h_L, params.eps_h);
    
    let hu_R = (un_R * nx - ut_R * ny) * h_R;
    let hv_R = (un_R * ny + ut_R * nx) * h_R;
    
    return vec4<f32>(h_R, hu_R, hv_R, z_L);
}

// 流量边界
fn apply_discharge_bc(
    h_L: f32, hu_L: f32, hv_L: f32, z_L: f32,
    bc_value: vec4<f32>,
    face_length_val: f32,
    nx: f32, ny: f32, g: f32
) -> vec4<f32> {
    let target_q = bc_value.x;  // 目标单宽流量 [m²/s]
    
    // 使用内部水深
    let h_R = h_L;
    
    // 设定法向速度
    let un_R = target_q / max(h_R, params.eps_h);
    let ut_R = (-hu_L * ny + hv_L * nx) / max(h_L, params.eps_h);
    
    let hu_R = (un_R * nx - ut_R * ny) * h_R;
    let hv_R = (un_R * ny + ut_R * nx) * h_R;
    
    return vec4<f32>(h_R, hu_R, hv_R, z_L);
}

// 速度边界
fn apply_velocity_bc(
    h_L: f32, hu_L: f32, hv_L: f32, z_L: f32,
    bc_value: vec4<f32>
) -> vec4<f32> {
    let target_u = bc_value.x;
    let target_v = bc_value.y;
    
    // 使用内部水深
    let h_R = h_L;
    let hu_R = h_R * target_u;
    let hv_R = h_R * target_v;
    
    return vec4<f32>(h_R, hu_R, hv_R, z_L);
}

// 辐射边界 (Sommerfeld)
fn apply_radiation_bc(
    h_L: f32, hu_L: f32, hv_L: f32, z_L: f32,
    nx: f32, ny: f32, g: f32
) -> vec4<f32> {
    // c = sqrt(g*h)
    let c = sqrt(g * max(h_L, 0.0));
    
    // 外推，允许波动传出
    let un = (hu_L * nx + hv_L * ny) / max(h_L, params.eps_h);
    
    if un - c > 0.0 {
        // 超临界流出
        return vec4<f32>(h_L, hu_L, hv_L, z_L);
    }
    
    // 亚临界：部分反射
    let h_inf = h_L;  // 远场水深
    let u_inf = 0.0;  // 远场速度
    
    let h_R = h_L;
    let un_R = un;  // 简化处理
    let ut_R = (-hu_L * ny + hv_L * nx) / max(h_L, params.eps_h);
    
    let hu_R = (un_R * nx - ut_R * ny) * h_R;
    let hv_R = (un_R * ny + ut_R * nx) * h_R;
    
    return vec4<f32>(h_R, hu_R, hv_R, z_L);
}

// 吸收边界 (海绵层)
fn apply_absorbing_bc(
    h_L: f32, hu_L: f32, hv_L: f32, z_L: f32,
    bc_value: vec4<f32>
) -> vec4<f32> {
    let target_h = bc_value.x;
    let damping = bc_value.y;  // 阻尼系数 0-1
    
    // 逐渐趋近目标状态
    let h_R = mix(h_L, target_h, damping);
    let hu_R = hu_L * (1.0 - damping);
    let hv_R = hv_L * (1.0 - damping);
    
    return vec4<f32>(h_R, hu_R, hv_R, z_L);
}

// ==================== 主计算核心 ====================

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bc_idx = gid.x;
    
    if bc_idx >= params.num_boundary_faces {
        return;
    }
    
    let face_id = boundary_face_ids[bc_idx];
    let bc_type = boundary_types[bc_idx];
    let bc_value = boundary_values[bc_idx];
    
    let nx = face_nx[face_id];
    let ny = face_ny[face_id];
    let length = face_length[face_id];
    let owner = face_owner[face_id];
    
    // 获取内部状态（左侧）
    let h_L = recon_h_L[face_id];
    let hu_L = recon_hu_L[face_id];
    let hv_L = recon_hv_L[face_id];
    let z_L = recon_z_L[face_id];
    
    var result: vec4<f32>;
    
    switch bc_type {
        case BC_WALL: {
            result = apply_wall_bc(h_L, hu_L, hv_L, z_L, nx, ny);
        }
        case BC_OPEN: {
            result = apply_open_bc(h_L, hu_L, hv_L, z_L, nx, ny);
        }
        case BC_WATER_LEVEL: {
            result = apply_water_level_bc(h_L, hu_L, hv_L, z_L, bc_value, nx, ny, params.g);
        }
        case BC_DISCHARGE: {
            result = apply_discharge_bc(h_L, hu_L, hv_L, z_L, bc_value, length, nx, ny, params.g);
        }
        case BC_VELOCITY: {
            result = apply_velocity_bc(h_L, hu_L, hv_L, z_L, bc_value);
        }
        case BC_RADIATION: {
            result = apply_radiation_bc(h_L, hu_L, hv_L, z_L, nx, ny, params.g);
        }
        case BC_ABSORBING: {
            result = apply_absorbing_bc(h_L, hu_L, hv_L, z_L, bc_value);
        }
        default: {
            // 默认反射
            result = apply_wall_bc(h_L, hu_L, hv_L, z_L, nx, ny);
        }
    }
    
    // 写入右侧重构值
    recon_h_R[face_id] = result.x;
    recon_hu_R[face_id] = result.y;
    recon_hv_R[face_id] = result.z;
    recon_z_R[face_id] = result.w;
}

// ==================== 潮汐边界 ====================

struct TidalParams {
    num_tidal_faces: u32,
    time: f32,
    _pad0: u32,
    _pad1: u32,
}

struct TidalConstituent {
    amplitude: f32,      // 振幅 [m]
    phase: f32,          // 相位 [rad]
    frequency: f32,      // 频率 [rad/s]
    _pad: f32,
}

@group(0) @binding(20) var<uniform> tidal_params: TidalParams;
@group(0) @binding(21) var<storage, read> tidal_face_ids: array<u32>;
@group(0) @binding(22) var<storage, read> tidal_mean_level: array<f32>;
@group(0) @binding(23) var<storage, read> tidal_constituents: array<TidalConstituent>;  // 每个面最多8个分潮
@group(0) @binding(24) var<storage, read> num_constituents_per_face: array<u32>;

@compute @workgroup_size(64)
fn apply_tidal_bc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tidal_idx = gid.x;
    
    if tidal_idx >= tidal_params.num_tidal_faces {
        return;
    }
    
    let face_id = tidal_face_ids[tidal_idx];
    let mean_level = tidal_mean_level[tidal_idx];
    let num_const = num_constituents_per_face[tidal_idx];
    
    // 计算潮位
    var eta: f32 = mean_level;
    let base_idx = tidal_idx * 8u;  // 每个面最多8个分潮
    
    for (var i = 0u; i < min(num_const, 8u); i = i + 1u) {
        let constituent = tidal_constituents[base_idx + i];
        eta = eta + constituent.amplitude * cos(
            constituent.frequency * tidal_params.time + constituent.phase
        );
    }
    
    // 设置边界水位
    let z_L = recon_z_L[face_id];
    let h_R = max(eta - z_L, 0.0);
    
    // 使用辐射条件确定速度
    let h_L = recon_h_L[face_id];
    let hu_L = recon_hu_L[face_id];
    let hv_L = recon_hv_L[face_id];
    let nx = face_nx[face_id];
    let ny = face_ny[face_id];
    
    let c_L = sqrt(params.g * max(h_L, 0.0));
    let c_R = sqrt(params.g * max(h_R, 0.0));
    let un_L = (hu_L * nx + hv_L * ny) / max(h_L, params.eps_h);
    
    // Riemann不变量
    let un_R = un_L + 2.0 * (c_L - c_R);
    let ut_R = (-hu_L * ny + hv_L * nx) / max(h_L, params.eps_h);
    
    recon_h_R[face_id] = h_R;
    recon_hu_R[face_id] = (un_R * nx - ut_R * ny) * h_R;
    recon_hv_R[face_id] = (un_R * ny + ut_R * nx) * h_R;
    recon_z_R[face_id] = z_L;
}
