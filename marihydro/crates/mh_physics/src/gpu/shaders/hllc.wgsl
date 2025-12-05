// MariHydro GPU计算着色器 - HLLC黎曼求解器
// hllc.wgsl

// ==================== Uniform缓冲区 ====================
struct HLLCParams {
    num_faces: u32,
    g: f32,            // 重力加速度
    eps_h: f32,        // 最小水深
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: HLLCParams;

// ==================== 存储缓冲区 ====================

// 面几何
@group(0) @binding(1) var<storage, read> face_nx: array<f32>;
@group(0) @binding(2) var<storage, read> face_ny: array<f32>;
@group(0) @binding(3) var<storage, read> face_length: array<f32>;
@group(0) @binding(4) var<storage, read> face_owner: array<u32>;
@group(0) @binding(5) var<storage, read> face_neighbor: array<u32>;

// 重构的状态值
@group(0) @binding(6) var<storage, read> recon_h_L: array<f32>;
@group(0) @binding(7) var<storage, read> recon_hu_L: array<f32>;
@group(0) @binding(8) var<storage, read> recon_hv_L: array<f32>;
@group(0) @binding(9) var<storage, read> recon_z_L: array<f32>;

@group(0) @binding(10) var<storage, read> recon_h_R: array<f32>;
@group(0) @binding(11) var<storage, read> recon_hu_R: array<f32>;
@group(0) @binding(12) var<storage, read> recon_hv_R: array<f32>;
@group(0) @binding(13) var<storage, read> recon_z_R: array<f32>;

// 面通量输出
@group(0) @binding(14) var<storage, read_write> flux_h: array<f32>;
@group(0) @binding(15) var<storage, read_write> flux_hu: array<f32>;
@group(0) @binding(16) var<storage, read_write> flux_hv: array<f32>;

// 最大波速 (用于CFL计算)
@group(0) @binding(17) var<storage, read_write> max_wave_speed: array<f32>;

// ==================== 常量 ====================
const INVALID: u32 = 0xFFFFFFFFu;

// ==================== HLLC求解器核心 ====================

// 计算单侧物理通量 (局部坐标系)
fn flux_local(h: f32, un: f32, ut: f32, g: f32) -> vec3<f32> {
    let h2 = h * h;
    return vec3<f32>(
        h * un,
        h * un * un + 0.5 * g * h2,
        h * un * ut
    );
}

// HLLC求解器
fn hllc_solver(
    h_L: f32, un_L: f32, ut_L: f32,
    h_R: f32, un_R: f32, ut_R: f32,
    g: f32, eps_h: f32
) -> vec3<f32> {
    // 处理干湿情况
    if h_L < eps_h && h_R < eps_h {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // 计算声速
    let a_L = sqrt(g * max(h_L, 0.0));
    let a_R = sqrt(g * max(h_R, 0.0));
    
    // Roe平均
    let sqrt_h_L = sqrt(max(h_L, 0.0));
    let sqrt_h_R = sqrt(max(h_R, 0.0));
    let sum_sqrt = sqrt_h_L + sqrt_h_R;
    
    var u_roe: f32;
    var a_roe: f32;
    
    if sum_sqrt > eps_h {
        u_roe = (sqrt_h_L * un_L + sqrt_h_R * un_R) / sum_sqrt;
        a_roe = sqrt(g * 0.5 * (h_L + h_R));
    } else {
        u_roe = 0.0;
        a_roe = 0.0;
    }
    
    // 波速估计
    var s_L: f32;
    var s_R: f32;
    
    if h_L < eps_h {
        // 左侧干燥，dam-break波速
        s_L = un_R - 2.0 * a_R;
        s_R = un_R + a_R;
    } else if h_R < eps_h {
        // 右侧干燥，dam-break波速
        s_L = un_L - a_L;
        s_R = un_L + 2.0 * a_L;
    } else {
        // 正常情况
        s_L = min(un_L - a_L, u_roe - a_roe);
        s_R = max(un_R + a_R, u_roe + a_roe);
    }
    
    // 处理退化情况
    if s_R - s_L < eps_h {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // 中间波速
    let num = s_L * h_R * (un_R - s_R) - s_R * h_L * (un_L - s_L);
    let den = h_R * (un_R - s_R) - h_L * (un_L - s_L);
    var s_star: f32;
    
    if abs(den) < eps_h * eps_h {
        s_star = 0.5 * (s_L + s_R);
    } else {
        s_star = num / den;
    }
    
    // 计算左右通量
    let f_L = flux_local(h_L, un_L, ut_L, g);
    let f_R = flux_local(h_R, un_R, ut_R, g);
    
    // 根据波结构选择通量
    if s_L >= 0.0 {
        // 超音速流向右
        return f_L;
    } else if s_R <= 0.0 {
        // 超音速流向左
        return f_R;
    } else if s_star >= 0.0 {
        // 亚音速，左星区
        let coef = (s_L - un_L) / (s_L - s_star);
        let h_star = h_L * coef;
        let hun_star = h_star * s_star;
        let hut_star = h_star * ut_L;
        
        return vec3<f32>(
            f_L.x + s_L * (h_star - h_L),
            f_L.y + s_L * (hun_star - h_L * un_L),
            f_L.z + s_L * (hut_star - h_L * ut_L)
        );
    } else {
        // 亚音速，右星区
        let coef = (s_R - un_R) / (s_R - s_star);
        let h_star = h_R * coef;
        let hun_star = h_star * s_star;
        let hut_star = h_star * ut_R;
        
        return vec3<f32>(
            f_R.x + s_R * (h_star - h_R),
            f_R.y + s_R * (hun_star - h_R * un_R),
            f_R.z + s_R * (hut_star - h_R * ut_R)
        );
    }
}

// ==================== 主计算核心 ====================

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let face_id = gid.x;
    
    if face_id >= params.num_faces {
        return;
    }
    
    let g = params.g;
    let eps_h = params.eps_h;
    
    // 获取面法向和长度
    let nx = face_nx[face_id];
    let ny = face_ny[face_id];
    let length = face_length[face_id];
    
    // 获取重构状态
    let h_L = recon_h_L[face_id];
    let hu_L = recon_hu_L[face_id];
    let hv_L = recon_hv_L[face_id];
    let z_L = recon_z_L[face_id];
    
    let h_R = recon_h_R[face_id];
    let hu_R = recon_hu_R[face_id];
    let hv_R = recon_hv_R[face_id];
    let z_R = recon_z_R[face_id];
    
    // 转换到局部坐标系
    var un_L: f32;
    var ut_L: f32;
    var un_R: f32;
    var ut_R: f32;
    
    if h_L > eps_h {
        let u_L = hu_L / h_L;
        let v_L = hv_L / h_L;
        un_L = u_L * nx + v_L * ny;
        ut_L = -u_L * ny + v_L * nx;
    } else {
        un_L = 0.0;
        ut_L = 0.0;
    }
    
    if h_R > eps_h {
        let u_R = hu_R / h_R;
        let v_R = hv_R / h_R;
        un_R = u_R * nx + v_R * ny;
        ut_R = -u_R * ny + v_R * nx;
    } else {
        un_R = 0.0;
        ut_R = 0.0;
    }
    
    // 底床处理 - 静水重构
    let z_face = max(z_L, z_R);
    var h_L_eff = max(h_L + z_L - z_face, 0.0);
    var h_R_eff = max(h_R + z_R - z_face, 0.0);
    
    // 调用HLLC求解器
    let flux_local_vec = hllc_solver(
        h_L_eff, un_L, ut_L,
        h_R_eff, un_R, ut_R,
        g, eps_h
    );
    
    // 从局部坐标系转回全局
    let f_h = flux_local_vec.x;
    let f_hun = flux_local_vec.y;
    let f_hut = flux_local_vec.z;
    
    let f_hu = f_hun * nx - f_hut * ny;
    let f_hv = f_hun * ny + f_hut * nx;
    
    // 添加底床坡度源项
    let dz = z_R - z_L;
    let h_avg = 0.5 * (h_L_eff + h_R_eff);
    let bed_flux_hu = -g * h_avg * dz * nx;
    let bed_flux_hv = -g * h_avg * dz * ny;
    
    // 输出通量 (乘以面长度)
    flux_h[face_id] = f_h * length;
    flux_hu[face_id] = (f_hu + bed_flux_hu) * length;
    flux_hv[face_id] = (f_hv + bed_flux_hv) * length;
    
    // 计算最大波速
    let a_L = sqrt(g * h_L_eff);
    let a_R = sqrt(g * h_R_eff);
    let max_speed = max(
        abs(un_L) + a_L,
        abs(un_R) + a_R
    );
    max_wave_speed[face_id] = max_speed;
}

// ==================== HLL求解器备选 ====================
// 更简单但稍不精确的求解器

@compute @workgroup_size(256)
fn hll_solver_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let face_id = gid.x;
    
    if face_id >= params.num_faces {
        return;
    }
    
    let g = params.g;
    let eps_h = params.eps_h;
    
    let nx = face_nx[face_id];
    let ny = face_ny[face_id];
    let length = face_length[face_id];
    
    let h_L = recon_h_L[face_id];
    let hu_L = recon_hu_L[face_id];
    let hv_L = recon_hv_L[face_id];
    
    let h_R = recon_h_R[face_id];
    let hu_R = recon_hu_R[face_id];
    let hv_R = recon_hv_R[face_id];
    
    // 转换到局部坐标
    var un_L = 0.0;
    var ut_L = 0.0;
    var un_R = 0.0;
    var ut_R = 0.0;
    
    if h_L > eps_h {
        let u = hu_L / h_L;
        let v = hv_L / h_L;
        un_L = u * nx + v * ny;
        ut_L = -u * ny + v * nx;
    }
    if h_R > eps_h {
        let u = hu_R / h_R;
        let v = hv_R / h_R;
        un_R = u * nx + v * ny;
        ut_R = -u * ny + v * nx;
    }
    
    // 波速
    let a_L = sqrt(g * max(h_L, 0.0));
    let a_R = sqrt(g * max(h_R, 0.0));
    let s_L = min(un_L - a_L, un_R - a_R);
    let s_R = max(un_L + a_L, un_R + a_R);
    
    // 左右通量
    let f_L = flux_local(h_L, un_L, ut_L, g);
    let f_R = flux_local(h_R, un_R, ut_R, g);
    
    // HLL通量
    var f_hll: vec3<f32>;
    
    if s_L >= 0.0 {
        f_hll = f_L;
    } else if s_R <= 0.0 {
        f_hll = f_R;
    } else {
        let denom = s_R - s_L;
        let u_L = vec3<f32>(h_L, h_L * un_L, h_L * ut_L);
        let u_R = vec3<f32>(h_R, h_R * un_R, h_R * ut_R);
        f_hll = (s_R * f_L - s_L * f_R + s_L * s_R * (u_R - u_L)) / denom;
    }
    
    // 转回全局坐标
    let f_hu = f_hll.y * nx - f_hll.z * ny;
    let f_hv = f_hll.y * ny + f_hll.z * nx;
    
    flux_h[face_id] = f_hll.x * length;
    flux_hu[face_id] = f_hu * length;
    flux_hv[face_id] = f_hv * length;
    
    max_wave_speed[face_id] = max(abs(s_L), abs(s_R));
}
