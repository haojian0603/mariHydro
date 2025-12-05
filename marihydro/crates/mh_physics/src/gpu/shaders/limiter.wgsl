// MariHydro GPU计算着色器 - Barth-Jespersen限制器
// limiter.wgsl

// ==================== Uniform缓冲区 ====================
struct LimiterParams {
    num_cells: u32,
    num_faces: u32,
    limiter_type: u32,  // 0=none, 1=Barth-Jespersen, 2=Venkatakrishnan
    venkat_k: f32,      // Venkatakrishnan系数
}

@group(0) @binding(0) var<uniform> params: LimiterParams;

// ==================== 存储缓冲区 ====================

// 单元几何
@group(0) @binding(1) var<storage, read> cell_cx: array<f32>;
@group(0) @binding(2) var<storage, read> cell_cy: array<f32>;
@group(0) @binding(3) var<storage, read> cell_areas: array<f32>;

// 面几何和拓扑
@group(0) @binding(4) var<storage, read> face_cx: array<f32>;
@group(0) @binding(5) var<storage, read> face_cy: array<f32>;
@group(0) @binding(6) var<storage, read> face_owner: array<u32>;
@group(0) @binding(7) var<storage, read> face_neighbor: array<u32>;

// 单元-面邻接 (CSR)
@group(0) @binding(8) var<storage, read> cell_face_ptr: array<u32>;
@group(0) @binding(9) var<storage, read> cell_face_idx: array<u32>;

// 状态数据
@group(0) @binding(10) var<storage, read> cell_h: array<f32>;
@group(0) @binding(11) var<storage, read> cell_hu: array<f32>;
@group(0) @binding(12) var<storage, read> cell_hv: array<f32>;

// 梯度数据
@group(0) @binding(13) var<storage, read> grad_h_x: array<f32>;
@group(0) @binding(14) var<storage, read> grad_h_y: array<f32>;
@group(0) @binding(15) var<storage, read> grad_hu_x: array<f32>;
@group(0) @binding(16) var<storage, read> grad_hu_y: array<f32>;
@group(0) @binding(17) var<storage, read> grad_hv_x: array<f32>;
@group(0) @binding(18) var<storage, read> grad_hv_y: array<f32>;

// 限制器输出
@group(0) @binding(19) var<storage, read_write> limiter_h: array<f32>;
@group(0) @binding(20) var<storage, read_write> limiter_hu: array<f32>;
@group(0) @binding(21) var<storage, read_write> limiter_hv: array<f32>;

// ==================== 常量 ====================
const INVALID: u32 = 0xFFFFFFFFu;
const EPS: f32 = 1e-10;

// ==================== Barth-Jespersen限制器 ====================

// 计算单变量的限制器
fn compute_limiter_bj(
    phi_c: f32,
    grad_x: f32,
    grad_y: f32,
    cell_id: u32,
    phi_min: f32,
    phi_max: f32
) -> f32 {
    var limiter: f32 = 1.0;
    
    let face_start = cell_face_ptr[cell_id];
    let face_end = cell_face_ptr[cell_id + 1u];
    let c_cx = cell_cx[cell_id];
    let c_cy = cell_cy[cell_id];
    
    for (var i = face_start; i < face_end; i = i + 1u) {
        let face_id = cell_face_idx[i];
        let f_cx = face_cx[face_id];
        let f_cy = face_cy[face_id];
        
        // 从单元中心到面中心的向量
        let dx = f_cx - c_cx;
        let dy = f_cy - c_cy;
        
        // 面中心处的重构值变化量
        let delta = grad_x * dx + grad_y * dy;
        
        if abs(delta) < EPS {
            continue;
        }
        
        var phi_limit: f32;
        if delta > 0.0 {
            // 重构值增加，使用最大值限制
            phi_limit = (phi_max - phi_c) / delta;
        } else {
            // 重构值减少，使用最小值限制
            phi_limit = (phi_min - phi_c) / delta;
        }
        
        limiter = min(limiter, max(0.0, min(1.0, phi_limit)));
    }
    
    return limiter;
}

// Venkatakrishnan限制器（更平滑）
fn compute_limiter_venkat(
    phi_c: f32,
    grad_x: f32,
    grad_y: f32,
    cell_id: u32,
    phi_min: f32,
    phi_max: f32,
    delta_max: f32  // 特征长度
) -> f32 {
    var limiter: f32 = 1.0;
    
    let face_start = cell_face_ptr[cell_id];
    let face_end = cell_face_ptr[cell_id + 1u];
    let c_cx = cell_cx[cell_id];
    let c_cy = cell_cy[cell_id];
    
    // Venkatakrishnan参数
    let k = params.venkat_k;
    let eps2 = (k * delta_max) * (k * delta_max) * (k * delta_max);
    
    for (var i = face_start; i < face_end; i = i + 1u) {
        let face_id = cell_face_idx[i];
        let f_cx = face_cx[face_id];
        let f_cy = face_cy[face_id];
        
        let dx = f_cx - c_cx;
        let dy = f_cy - c_cy;
        let delta = grad_x * dx + grad_y * dy;
        
        if abs(delta) < EPS {
            continue;
        }
        
        var delta_m: f32;
        if delta > 0.0 {
            delta_m = phi_max - phi_c;
        } else {
            delta_m = phi_min - phi_c;
        }
        
        let delta2 = delta * delta;
        let delta_m2 = delta_m * delta_m;
        
        // Venkatakrishnan公式
        let num = (delta_m2 + eps2) * delta + 2.0 * delta2 * delta_m;
        let den = delta_m2 + 2.0 * delta2 + delta_m * delta + eps2;
        
        if abs(den) > EPS {
            let phi = num / (den * abs(delta));
            limiter = min(limiter, max(0.0, min(1.0, phi)));
        }
    }
    
    return limiter;
}

// ==================== 主计算核心 ====================

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_id = gid.x;
    
    if cell_id >= params.num_cells {
        return;
    }
    
    // 无限制器模式
    if params.limiter_type == 0u {
        limiter_h[cell_id] = 1.0;
        limiter_hu[cell_id] = 1.0;
        limiter_hv[cell_id] = 1.0;
        return;
    }
    
    // 获取本单元的值
    let h_c = cell_h[cell_id];
    let hu_c = cell_hu[cell_id];
    let hv_c = cell_hv[cell_id];
    
    // 获取梯度
    let gh_x = grad_h_x[cell_id];
    let gh_y = grad_h_y[cell_id];
    let ghu_x = grad_hu_x[cell_id];
    let ghu_y = grad_hu_y[cell_id];
    let ghv_x = grad_hv_x[cell_id];
    let ghv_y = grad_hv_y[cell_id];
    
    // 计算邻居范围内的最小/最大值
    var h_min = h_c;
    var h_max = h_c;
    var hu_min = hu_c;
    var hu_max = hu_c;
    var hv_min = hv_c;
    var hv_max = hv_c;
    
    let face_start = cell_face_ptr[cell_id];
    let face_end = cell_face_ptr[cell_id + 1u];
    
    for (var i = face_start; i < face_end; i = i + 1u) {
        let face_id = cell_face_idx[i];
        let owner = face_owner[face_id];
        let neighbor = face_neighbor[face_id];
        
        if neighbor == INVALID {
            continue;
        }
        
        let other_id = select(owner, neighbor, cell_id == owner);
        
        let h_n = cell_h[other_id];
        let hu_n = cell_hu[other_id];
        let hv_n = cell_hv[other_id];
        
        h_min = min(h_min, h_n);
        h_max = max(h_max, h_n);
        hu_min = min(hu_min, hu_n);
        hu_max = max(hu_max, hu_n);
        hv_min = min(hv_min, hv_n);
        hv_max = max(hv_max, hv_n);
    }
    
    // 计算特征长度（用于Venkatakrishnan）
    let area = cell_areas[cell_id];
    let delta_max = sqrt(area);
    
    // 根据限制器类型计算
    var lim_h: f32;
    var lim_hu: f32;
    var lim_hv: f32;
    
    if params.limiter_type == 1u {
        // Barth-Jespersen
        lim_h = compute_limiter_bj(h_c, gh_x, gh_y, cell_id, h_min, h_max);
        lim_hu = compute_limiter_bj(hu_c, ghu_x, ghu_y, cell_id, hu_min, hu_max);
        lim_hv = compute_limiter_bj(hv_c, ghv_x, ghv_y, cell_id, hv_min, hv_max);
    } else {
        // Venkatakrishnan
        lim_h = compute_limiter_venkat(h_c, gh_x, gh_y, cell_id, h_min, h_max, delta_max);
        lim_hu = compute_limiter_venkat(hu_c, ghu_x, ghu_y, cell_id, hu_min, hu_max, delta_max);
        lim_hv = compute_limiter_venkat(hv_c, ghv_x, ghv_y, cell_id, hv_min, hv_max, delta_max);
    }
    
    // 输出限制器值
    limiter_h[cell_id] = lim_h;
    limiter_hu[cell_id] = lim_hu;
    limiter_hv[cell_id] = lim_hv;
}

// ==================== 辅助Kernel：初始化限制器为1 ====================
@compute @workgroup_size(256)
fn init_limiters(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_id = gid.x;
    
    if cell_id >= params.num_cells {
        return;
    }
    
    limiter_h[cell_id] = 1.0;
    limiter_hu[cell_id] = 1.0;
    limiter_hv[cell_id] = 1.0;
}
