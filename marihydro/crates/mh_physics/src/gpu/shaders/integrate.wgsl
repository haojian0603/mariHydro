// MariHydro GPU计算着色器 - 时间积分
// integrate.wgsl

// ==================== Uniform缓冲区 ====================
struct IntegrateParams {
    num_cells: u32,
    dt: f32,               // 时间步长
    stage: u32,            // RK阶段 (0, 1, 2)
    rk_order: u32,         // RK阶数 (1, 2, 3)
    eps_h: f32,            // 最小水深
    friction_enabled: u32, // 是否启用摩擦
    manning_n: f32,        // 曼宁系数
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: IntegrateParams;

// ==================== 存储缓冲区 ====================

// 单元面积
@group(0) @binding(1) var<storage, read> cell_areas: array<f32>;

// 当前状态 (stage 0时的初始状态)
@group(0) @binding(2) var<storage, read> state_h_n: array<f32>;
@group(0) @binding(3) var<storage, read> state_hu_n: array<f32>;
@group(0) @binding(4) var<storage, read> state_hv_n: array<f32>;

// 中间状态 (用于多阶段RK)
@group(0) @binding(5) var<storage, read_write> state_h_star: array<f32>;
@group(0) @binding(6) var<storage, read_write> state_hu_star: array<f32>;
@group(0) @binding(7) var<storage, read_write> state_hv_star: array<f32>;

// 残差 (RHS)
@group(0) @binding(8) var<storage, read> residual_h: array<f32>;
@group(0) @binding(9) var<storage, read> residual_hu: array<f32>;
@group(0) @binding(10) var<storage, read> residual_hv: array<f32>;

// 输出状态
@group(0) @binding(11) var<storage, read_write> state_h_out: array<f32>;
@group(0) @binding(12) var<storage, read_write> state_hu_out: array<f32>;
@group(0) @binding(13) var<storage, read_write> state_hv_out: array<f32>;

// 底床高程 (用于摩擦计算)
@group(0) @binding(14) var<storage, read> cell_z: array<f32>;

// ==================== 常量 ====================
const G: f32 = 9.80665;

// ==================== 摩擦源项 ====================

fn compute_friction(h: f32, hu: f32, hv: f32, n: f32, dt: f32, eps_h: f32) -> vec2<f32> {
    if h < eps_h {
        return vec2<f32>(0.0, 0.0);
    }
    
    let u = hu / h;
    let v = hv / h;
    let vel_mag = sqrt(u * u + v * v);
    
    if vel_mag < 1e-10 {
        return vec2<f32>(0.0, 0.0);
    }
    
    // 曼宁摩擦: Sf = n² * |u| * u / h^(4/3)
    let h_pow = pow(h, 4.0 / 3.0);
    let cf = G * n * n / h_pow;
    
    // 隐式处理：1 / (1 + cf * |u| * dt)
    let factor = 1.0 / (1.0 + cf * vel_mag * dt);
    
    return vec2<f32>(hu * factor, hv * factor);
}

// ==================== 主计算核心 ====================

// 前向欧拉 (一阶)
@compute @workgroup_size(256)
fn euler(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_id = gid.x;
    
    if cell_id >= params.num_cells {
        return;
    }
    
    let area = cell_areas[cell_id];
    let inv_area = 1.0 / area;
    let dt = params.dt;
    
    // 当前状态
    let h_n = state_h_n[cell_id];
    let hu_n = state_hu_n[cell_id];
    let hv_n = state_hv_n[cell_id];
    
    // 残差
    let rhs_h = residual_h[cell_id] * inv_area;
    let rhs_hu = residual_hu[cell_id] * inv_area;
    let rhs_hv = residual_hv[cell_id] * inv_area;
    
    // 时间推进
    var h_new = h_n + dt * rhs_h;
    var hu_new = hu_n + dt * rhs_hu;
    var hv_new = hv_n + dt * rhs_hv;
    
    // 干湿处理
    if h_new < params.eps_h {
        h_new = 0.0;
        hu_new = 0.0;
        hv_new = 0.0;
    } else if params.friction_enabled != 0u {
        // 摩擦处理
        let friction = compute_friction(h_new, hu_new, hv_new, params.manning_n, dt, params.eps_h);
        hu_new = friction.x;
        hv_new = friction.y;
    }
    
    // 输出
    state_h_out[cell_id] = h_new;
    state_hu_out[cell_id] = hu_new;
    state_hv_out[cell_id] = hv_new;
}

// SSP-RK2 (两阶段)
@compute @workgroup_size(256)
fn ssp_rk2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_id = gid.x;
    
    if cell_id >= params.num_cells {
        return;
    }
    
    let area = cell_areas[cell_id];
    let inv_area = 1.0 / area;
    let dt = params.dt;
    let stage = params.stage;
    
    // 残差
    let rhs_h = residual_h[cell_id] * inv_area;
    let rhs_hu = residual_hu[cell_id] * inv_area;
    let rhs_hv = residual_hv[cell_id] * inv_area;
    
    var h_new: f32;
    var hu_new: f32;
    var hv_new: f32;
    
    if stage == 0u {
        // Stage 1: u* = u^n + dt * L(u^n)
        let h_n = state_h_n[cell_id];
        let hu_n = state_hu_n[cell_id];
        let hv_n = state_hv_n[cell_id];
        
        h_new = h_n + dt * rhs_h;
        hu_new = hu_n + dt * rhs_hu;
        hv_new = hv_n + dt * rhs_hv;
        
        // 存储中间状态
        state_h_star[cell_id] = h_new;
        state_hu_star[cell_id] = hu_new;
        state_hv_star[cell_id] = hv_new;
    } else {
        // Stage 2: u^{n+1} = 0.5 * (u^n + u* + dt * L(u*))
        let h_n = state_h_n[cell_id];
        let hu_n = state_hu_n[cell_id];
        let hv_n = state_hv_n[cell_id];
        
        let h_star = state_h_star[cell_id];
        let hu_star = state_hu_star[cell_id];
        let hv_star = state_hv_star[cell_id];
        
        h_new = 0.5 * (h_n + h_star + dt * rhs_h);
        hu_new = 0.5 * (hu_n + hu_star + dt * rhs_hu);
        hv_new = 0.5 * (hv_n + hv_star + dt * rhs_hv);
    }
    
    // 干湿处理
    if h_new < params.eps_h {
        h_new = 0.0;
        hu_new = 0.0;
        hv_new = 0.0;
    } else if params.friction_enabled != 0u && stage == 1u {
        let friction = compute_friction(h_new, hu_new, hv_new, params.manning_n, dt, params.eps_h);
        hu_new = friction.x;
        hv_new = friction.y;
    }
    
    // 输出
    state_h_out[cell_id] = h_new;
    state_hu_out[cell_id] = hu_new;
    state_hv_out[cell_id] = hv_new;
}

// SSP-RK3 (三阶段)
@compute @workgroup_size(256)
fn ssp_rk3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_id = gid.x;
    
    if cell_id >= params.num_cells {
        return;
    }
    
    let area = cell_areas[cell_id];
    let inv_area = 1.0 / area;
    let dt = params.dt;
    let stage = params.stage;
    
    let rhs_h = residual_h[cell_id] * inv_area;
    let rhs_hu = residual_hu[cell_id] * inv_area;
    let rhs_hv = residual_hv[cell_id] * inv_area;
    
    let h_n = state_h_n[cell_id];
    let hu_n = state_hu_n[cell_id];
    let hv_n = state_hv_n[cell_id];
    
    var h_new: f32;
    var hu_new: f32;
    var hv_new: f32;
    
    if stage == 0u {
        // Stage 1: u^(1) = u^n + dt * L(u^n)
        h_new = h_n + dt * rhs_h;
        hu_new = hu_n + dt * rhs_hu;
        hv_new = hv_n + dt * rhs_hv;
        
        state_h_star[cell_id] = h_new;
        state_hu_star[cell_id] = hu_new;
        state_hv_star[cell_id] = hv_new;
        
    } else if stage == 1u {
        // Stage 2: u^(2) = 3/4 * u^n + 1/4 * (u^(1) + dt * L(u^(1)))
        let h_1 = state_h_star[cell_id];
        let hu_1 = state_hu_star[cell_id];
        let hv_1 = state_hv_star[cell_id];
        
        h_new = 0.75 * h_n + 0.25 * (h_1 + dt * rhs_h);
        hu_new = 0.75 * hu_n + 0.25 * (hu_1 + dt * rhs_hu);
        hv_new = 0.75 * hv_n + 0.25 * (hv_1 + dt * rhs_hv);
        
        // 更新中间状态为u^(2)
        state_h_star[cell_id] = h_new;
        state_hu_star[cell_id] = hu_new;
        state_hv_star[cell_id] = hv_new;
        
    } else {
        // Stage 3: u^{n+1} = 1/3 * u^n + 2/3 * (u^(2) + dt * L(u^(2)))
        let h_2 = state_h_star[cell_id];
        let hu_2 = state_hu_star[cell_id];
        let hv_2 = state_hv_star[cell_id];
        
        h_new = (1.0 / 3.0) * h_n + (2.0 / 3.0) * (h_2 + dt * rhs_h);
        hu_new = (1.0 / 3.0) * hu_n + (2.0 / 3.0) * (hu_2 + dt * rhs_hu);
        hv_new = (1.0 / 3.0) * hv_n + (2.0 / 3.0) * (hv_2 + dt * rhs_hv);
    }
    
    // 干湿处理
    if h_new < params.eps_h {
        h_new = 0.0;
        hu_new = 0.0;
        hv_new = 0.0;
    } else if params.friction_enabled != 0u && stage == 2u {
        let friction = compute_friction(h_new, hu_new, hv_new, params.manning_n, dt, params.eps_h);
        hu_new = friction.x;
        hv_new = friction.y;
    }
    
    state_h_out[cell_id] = h_new;
    state_hu_out[cell_id] = hu_new;
    state_hv_out[cell_id] = hv_new;
}

// ==================== CFL时间步长计算 ====================

struct CflParams {
    num_cells: u32,
    cfl_number: f32,
    min_dt: f32,
    max_dt: f32,
}

@group(0) @binding(0) var<uniform> cfl_params: CflParams;

// 单元特征长度
@group(0) @binding(15) var<storage, read> cell_char_length: array<f32>;

// 局部最大波速
@group(0) @binding(16) var<storage, read_write> cell_max_speed: array<f32>;

// 全局最小时间步 (使用原子操作)
@group(0) @binding(17) var<storage, read_write> global_min_dt: atomic<u32>;

@compute @workgroup_size(256)
fn compute_local_dt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_id = gid.x;
    
    if cell_id >= cfl_params.num_cells {
        return;
    }
    
    let h = state_h_n[cell_id];
    let hu = state_hu_n[cell_id];
    let hv = state_hv_n[cell_id];
    let dx = cell_char_length[cell_id];
    
    var max_speed: f32;
    
    if h < params.eps_h {
        max_speed = 0.0;
    } else {
        let u = hu / h;
        let v = hv / h;
        let c = sqrt(G * h);
        max_speed = sqrt(u * u + v * v) + c;
    }
    
    cell_max_speed[cell_id] = max_speed;
    
    // 计算局部时间步
    var local_dt: f32;
    if max_speed > 1e-10 {
        local_dt = cfl_params.cfl_number * dx / max_speed;
    } else {
        local_dt = cfl_params.max_dt;
    }
    
    local_dt = clamp(local_dt, cfl_params.min_dt, cfl_params.max_dt);
    
    // 使用原子操作更新全局最小值
    // 将浮点数转换为位表示进行比较
    let dt_bits = bitcast<u32>(local_dt);
    atomicMin(&global_min_dt, dt_bits);
}
