// MariHydro GPU计算着色器 - MUSCL重构
// reconstruct.wgsl

// ==================== Uniform缓冲区 ====================
struct ReconstructParams {
    num_faces: u32,
    reconstruction_order: u32,  // 1=一阶, 2=二阶MUSCL
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: ReconstructParams;

// ==================== 存储缓冲区 ====================

// 单元几何
@group(0) @binding(1) var<storage, read> cell_cx: array<f32>;
@group(0) @binding(2) var<storage, read> cell_cy: array<f32>;

// 面几何和拓扑
@group(0) @binding(3) var<storage, read> face_cx: array<f32>;
@group(0) @binding(4) var<storage, read> face_cy: array<f32>;
@group(0) @binding(5) var<storage, read> face_owner: array<u32>;
@group(0) @binding(6) var<storage, read> face_neighbor: array<u32>;

// 状态数据
@group(0) @binding(7) var<storage, read> cell_h: array<f32>;
@group(0) @binding(8) var<storage, read> cell_hu: array<f32>;
@group(0) @binding(9) var<storage, read> cell_hv: array<f32>;
@group(0) @binding(10) var<storage, read> cell_z: array<f32>;

// 梯度数据
@group(0) @binding(11) var<storage, read> grad_h_x: array<f32>;
@group(0) @binding(12) var<storage, read> grad_h_y: array<f32>;
@group(0) @binding(13) var<storage, read> grad_hu_x: array<f32>;
@group(0) @binding(14) var<storage, read> grad_hu_y: array<f32>;
@group(0) @binding(15) var<storage, read> grad_hv_x: array<f32>;
@group(0) @binding(16) var<storage, read> grad_hv_y: array<f32>;
@group(0) @binding(17) var<storage, read> grad_z_x: array<f32>;
@group(0) @binding(18) var<storage, read> grad_z_y: array<f32>;

// 限制器
@group(0) @binding(19) var<storage, read> limiter_h: array<f32>;
@group(0) @binding(20) var<storage, read> limiter_hu: array<f32>;
@group(0) @binding(21) var<storage, read> limiter_hv: array<f32>;

// 重构输出 (面左右两侧的值)
// 左侧 (owner侧)
@group(0) @binding(22) var<storage, read_write> recon_h_L: array<f32>;
@group(0) @binding(23) var<storage, read_write> recon_hu_L: array<f32>;
@group(0) @binding(24) var<storage, read_write> recon_hv_L: array<f32>;
@group(0) @binding(25) var<storage, read_write> recon_z_L: array<f32>;

// 右侧 (neighbor侧)
@group(0) @binding(26) var<storage, read_write> recon_h_R: array<f32>;
@group(0) @binding(27) var<storage, read_write> recon_hu_R: array<f32>;
@group(0) @binding(28) var<storage, read_write> recon_hv_R: array<f32>;
@group(0) @binding(29) var<storage, read_write> recon_z_R: array<f32>;

// ==================== 常量 ====================
const INVALID: u32 = 0xFFFFFFFFu;
const EPS_H: f32 = 1e-6;

// ==================== 主计算核心 ====================

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let face_id = gid.x;
    
    if face_id >= params.num_faces {
        return;
    }
    
    let owner = face_owner[face_id];
    let neighbor = face_neighbor[face_id];
    
    // 面中心坐标
    let f_cx = face_cx[face_id];
    let f_cy = face_cy[face_id];
    
    // ===== 左侧 (owner) 重构 =====
    let o_cx = cell_cx[owner];
    let o_cy = cell_cy[owner];
    let o_h = cell_h[owner];
    let o_hu = cell_hu[owner];
    let o_hv = cell_hv[owner];
    let o_z = cell_z[owner];
    
    var h_L: f32;
    var hu_L: f32;
    var hv_L: f32;
    var z_L: f32;
    
    if params.reconstruction_order == 1u {
        // 一阶：使用单元中心值
        h_L = o_h;
        hu_L = o_hu;
        hv_L = o_hv;
        z_L = o_z;
    } else {
        // 二阶MUSCL重构
        let dx_L = f_cx - o_cx;
        let dy_L = f_cy - o_cy;
        
        // 获取梯度和限制器
        let gh_x = grad_h_x[owner];
        let gh_y = grad_h_y[owner];
        let ghu_x = grad_hu_x[owner];
        let ghu_y = grad_hu_y[owner];
        let ghv_x = grad_hv_x[owner];
        let ghv_y = grad_hv_y[owner];
        let gz_x = grad_z_x[owner];
        let gz_y = grad_z_y[owner];
        
        let lim_h = limiter_h[owner];
        let lim_hu = limiter_hu[owner];
        let lim_hv = limiter_hv[owner];
        
        // 带限制器的线性重构
        h_L = o_h + lim_h * (gh_x * dx_L + gh_y * dy_L);
        hu_L = o_hu + lim_hu * (ghu_x * dx_L + ghu_y * dy_L);
        hv_L = o_hv + lim_hv * (ghv_x * dx_L + ghv_y * dy_L);
        z_L = o_z + gz_x * dx_L + gz_y * dy_L;  // 底床不需要限制器
    }
    
    // 确保水深非负
    h_L = max(h_L, 0.0);
    
    // ===== 右侧 (neighbor) 重构 =====
    var h_R: f32;
    var hu_R: f32;
    var hv_R: f32;
    var z_R: f32;
    
    if neighbor == INVALID {
        // 边界面：使用反射或外推边界条件
        // 默认使用反射边界（动量取反）
        h_R = h_L;
        hu_R = -hu_L;  // 反射
        hv_R = -hv_L;  // 反射
        z_R = z_L;
    } else {
        let n_cx = cell_cx[neighbor];
        let n_cy = cell_cy[neighbor];
        let n_h = cell_h[neighbor];
        let n_hu = cell_hu[neighbor];
        let n_hv = cell_hv[neighbor];
        let n_z = cell_z[neighbor];
        
        if params.reconstruction_order == 1u {
            // 一阶
            h_R = n_h;
            hu_R = n_hu;
            hv_R = n_hv;
            z_R = n_z;
        } else {
            // 二阶MUSCL重构
            let dx_R = f_cx - n_cx;
            let dy_R = f_cy - n_cy;
            
            let gh_x = grad_h_x[neighbor];
            let gh_y = grad_h_y[neighbor];
            let ghu_x = grad_hu_x[neighbor];
            let ghu_y = grad_hu_y[neighbor];
            let ghv_x = grad_hv_x[neighbor];
            let ghv_y = grad_hv_y[neighbor];
            let gz_x = grad_z_x[neighbor];
            let gz_y = grad_z_y[neighbor];
            
            let lim_h = limiter_h[neighbor];
            let lim_hu = limiter_hu[neighbor];
            let lim_hv = limiter_hv[neighbor];
            
            h_R = n_h + lim_h * (gh_x * dx_R + gh_y * dy_R);
            hu_R = n_hu + lim_hu * (ghu_x * dx_R + ghu_y * dy_R);
            hv_R = n_hv + lim_hv * (ghv_x * dx_R + ghv_y * dy_R);
            z_R = n_z + gz_x * dx_R + gz_y * dy_R;
        }
        
        h_R = max(h_R, 0.0);
    }
    
    // 输出重构值
    recon_h_L[face_id] = h_L;
    recon_hu_L[face_id] = hu_L;
    recon_hv_L[face_id] = hv_L;
    recon_z_L[face_id] = z_L;
    
    recon_h_R[face_id] = h_R;
    recon_hu_R[face_id] = hu_R;
    recon_hv_R[face_id] = hv_R;
    recon_z_R[face_id] = z_R;
}

// ==================== 干湿边界修正Kernel ====================

@compute @workgroup_size(256)
fn wet_dry_fix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let face_id = gid.x;
    
    if face_id >= params.num_faces {
        return;
    }
    
    var h_L = recon_h_L[face_id];
    var hu_L = recon_hu_L[face_id];
    var hv_L = recon_hv_L[face_id];
    let z_L = recon_z_L[face_id];
    
    var h_R = recon_h_R[face_id];
    var hu_R = recon_hu_R[face_id];
    var hv_R = recon_hv_R[face_id];
    let z_R = recon_z_R[face_id];
    
    let is_L_dry = h_L < EPS_H;
    let is_R_dry = h_R < EPS_H;
    
    if is_L_dry && is_R_dry {
        // 两侧都干，无通量
        recon_h_L[face_id] = 0.0;
        recon_hu_L[face_id] = 0.0;
        recon_hv_L[face_id] = 0.0;
        recon_h_R[face_id] = 0.0;
        recon_hu_R[face_id] = 0.0;
        recon_hv_R[face_id] = 0.0;
    } else if is_L_dry {
        // 左侧干，水从右向左流
        let eta_R = h_R + z_R;
        if eta_R <= z_L {
            // 右侧水面低于左侧底床，无流动
            recon_h_L[face_id] = 0.0;
            recon_hu_L[face_id] = 0.0;
            recon_hv_L[face_id] = 0.0;
            recon_h_R[face_id] = 0.0;
            recon_hu_R[face_id] = 0.0;
            recon_hv_R[face_id] = 0.0;
        } else {
            // 使用局部静水重构
            recon_h_L[face_id] = eta_R - z_L;
            recon_hu_L[face_id] = 0.0;
            recon_hv_L[face_id] = 0.0;
        }
    } else if is_R_dry {
        // 右侧干，水从左向右流
        let eta_L = h_L + z_L;
        if eta_L <= z_R {
            recon_h_L[face_id] = 0.0;
            recon_hu_L[face_id] = 0.0;
            recon_hv_L[face_id] = 0.0;
            recon_h_R[face_id] = 0.0;
            recon_hu_R[face_id] = 0.0;
            recon_hv_R[face_id] = 0.0;
        } else {
            recon_h_R[face_id] = eta_L - z_R;
            recon_hu_R[face_id] = 0.0;
            recon_hv_R[face_id] = 0.0;
        }
    }
}
