// MariHydro GPU计算着色器 - 源项处理
// source.wgsl

// ==================== Uniform缓冲区 ====================
struct SourceParams {
    num_cells: u32,
    g: f32,               // 重力加速度
    eps_h: f32,           // 最小水深
    
    // 摩擦参数
    friction_type: u32,   // 0=none, 1=manning, 2=chezy
    manning_n: f32,       // 曼宁系数
    chezy_c: f32,         // 谢才系数
    
    // 风应力参数
    wind_enabled: u32,
    wind_ux: f32,
    wind_uy: f32,
    wind_drag_coef: f32,
    air_density: f32,
    water_density: f32,
    
    // 科氏力参数
    coriolis_enabled: u32,
    latitude: f32,        // 纬度 (弧度)
    
    // 源汇项
    source_enabled: u32,
    
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: SourceParams;

// ==================== 存储缓冲区 ====================

// 单元面积
@group(0) @binding(1) var<storage, read> cell_areas: array<f32>;

// 状态数据
@group(0) @binding(2) var<storage, read> cell_h: array<f32>;
@group(0) @binding(3) var<storage, read> cell_hu: array<f32>;
@group(0) @binding(4) var<storage, read> cell_hv: array<f32>;
@group(0) @binding(5) var<storage, read> cell_z: array<f32>;

// 底床梯度 (用于坡度源项)
@group(0) @binding(6) var<storage, read> grad_z_x: array<f32>;
@group(0) @binding(7) var<storage, read> grad_z_y: array<f32>;

// 源项输出 (累加到残差)
@group(0) @binding(8) var<storage, read_write> source_h: array<f32>;
@group(0) @binding(9) var<storage, read_write> source_hu: array<f32>;
@group(0) @binding(10) var<storage, read_write> source_hv: array<f32>;

// 点源汇数据 (可选)
@group(0) @binding(11) var<storage, read> point_source_cells: array<u32>;
@group(0) @binding(12) var<storage, read> point_source_rates: array<f32>;

// ==================== 常量 ====================
const OMEGA: f32 = 7.2921e-5;  // 地球自转角速度 [rad/s]

// ==================== 摩擦源项 ====================

fn manning_friction(h: f32, hu: f32, hv: f32, n: f32, g: f32) -> vec2<f32> {
    if h < params.eps_h {
        return vec2<f32>(0.0, 0.0);
    }
    
    let u = hu / h;
    let v = hv / h;
    let vel_mag = sqrt(u * u + v * v);
    
    if vel_mag < 1e-10 {
        return vec2<f32>(0.0, 0.0);
    }
    
    // Sf = n² * |u| * u / h^(4/3)
    // Source = -g * h * Sf
    let h_pow = pow(h, 4.0 / 3.0);
    let coef = -g * n * n * vel_mag / h_pow;
    
    return vec2<f32>(coef * hu, coef * hv);
}

fn chezy_friction(h: f32, hu: f32, hv: f32, c: f32, g: f32) -> vec2<f32> {
    if h < params.eps_h {
        return vec2<f32>(0.0, 0.0);
    }
    
    let u = hu / h;
    let v = hv / h;
    let vel_mag = sqrt(u * u + v * v);
    
    if vel_mag < 1e-10 {
        return vec2<f32>(0.0, 0.0);
    }
    
    // Sf = g * |u| * u / C²
    let coef = -g * vel_mag / (c * c);
    
    return vec2<f32>(coef * hu, coef * hv);
}

// ==================== 风应力源项 ====================

fn wind_stress(h: f32, hu: f32, hv: f32) -> vec2<f32> {
    if h < params.eps_h {
        return vec2<f32>(0.0, 0.0);
    }
    
    let u = hu / h;
    let v = hv / h;
    
    // 相对风速
    let rel_ux = params.wind_ux - u;
    let rel_uy = params.wind_uy - v;
    let rel_speed = sqrt(rel_ux * rel_ux + rel_uy * rel_uy);
    
    // 风应力: τ = ρ_air * Cd * |W| * W
    // 源项: S = τ / ρ_water
    let coef = params.air_density * params.wind_drag_coef * rel_speed / params.water_density;
    
    return vec2<f32>(coef * rel_ux, coef * rel_uy);
}

// ==================== 科氏力源项 ====================

fn coriolis_force(h: f32, hu: f32, hv: f32) -> vec2<f32> {
    if h < params.eps_h {
        return vec2<f32>(0.0, 0.0);
    }
    
    // f = 2 * Omega * sin(latitude)
    let f = 2.0 * OMEGA * sin(params.latitude);
    
    // 科氏加速度: (f*v, -f*u)
    // 对动量方程: (f*hv, -f*hu)
    return vec2<f32>(f * hv, -f * hu);
}

// ==================== 底床坡度源项 ====================

fn bed_slope_source(h: f32, dz_dx: f32, dz_dy: f32, g: f32) -> vec2<f32> {
    if h < params.eps_h {
        return vec2<f32>(0.0, 0.0);
    }
    
    // S_bx = -g * h * dz/dx
    // S_by = -g * h * dz/dy
    return vec2<f32>(-g * h * dz_dx, -g * h * dz_dy);
}

// ==================== 主计算核心 ====================

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_id = gid.x;
    
    if cell_id >= params.num_cells {
        return;
    }
    
    let h = cell_h[cell_id];
    let hu = cell_hu[cell_id];
    let hv = cell_hv[cell_id];
    let area = cell_areas[cell_id];
    
    var s_h: f32 = 0.0;
    var s_hu: f32 = 0.0;
    var s_hv: f32 = 0.0;
    
    // 底床坡度源项
    let dz_dx = grad_z_x[cell_id];
    let dz_dy = grad_z_y[cell_id];
    let bed_source = bed_slope_source(h, dz_dx, dz_dy, params.g);
    s_hu = s_hu + bed_source.x;
    s_hv = s_hv + bed_source.y;
    
    // 摩擦源项
    if params.friction_type == 1u {
        let friction = manning_friction(h, hu, hv, params.manning_n, params.g);
        s_hu = s_hu + friction.x;
        s_hv = s_hv + friction.y;
    } else if params.friction_type == 2u {
        let friction = chezy_friction(h, hu, hv, params.chezy_c, params.g);
        s_hu = s_hu + friction.x;
        s_hv = s_hv + friction.y;
    }
    
    // 风应力
    if params.wind_enabled != 0u {
        let wind = wind_stress(h, hu, hv);
        s_hu = s_hu + wind.x;
        s_hv = s_hv + wind.y;
    }
    
    // 科氏力
    if params.coriolis_enabled != 0u {
        let coriolis = coriolis_force(h, hu, hv);
        s_hu = s_hu + coriolis.x;
        s_hv = s_hv + coriolis.y;
    }
    
    // 输出 (乘以面积，与残差格式一致)
    source_h[cell_id] = s_h * area;
    source_hu[cell_id] = s_hu * area;
    source_hv[cell_id] = s_hv * area;
}

// ==================== 点源汇处理 ====================

struct PointSourceParams {
    num_sources: u32,
    dt: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(13) var<uniform> ps_params: PointSourceParams;

@compute @workgroup_size(64)
fn apply_point_sources(@builtin(global_invocation_id) gid: vec3<u32>) {
    let source_id = gid.x;
    
    if source_id >= ps_params.num_sources {
        return;
    }
    
    let cell_id = point_source_cells[source_id];
    let rate = point_source_rates[source_id];  // [m³/s]
    let area = cell_areas[cell_id];
    
    // 体积变化率转换为水深变化率
    let dh_dt = rate / area;
    
    // 累加到质量源项
    source_h[cell_id] = source_h[cell_id] + dh_dt * area;
}

// ==================== 降雨蒸发处理 ====================

@group(0) @binding(14) var<storage, read> rainfall_rate: array<f32>;    // [m/s]
@group(0) @binding(15) var<storage, read> evaporation_rate: array<f32>; // [m/s]

@compute @workgroup_size(256)
fn apply_rain_evap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_id = gid.x;
    
    if cell_id >= params.num_cells {
        return;
    }
    
    let h = cell_h[cell_id];
    let area = cell_areas[cell_id];
    
    // 降雨直接加入
    let rain = rainfall_rate[cell_id];
    
    // 蒸发不能超过现有水深
    var evap = evaporation_rate[cell_id];
    evap = min(evap, max(h, 0.0));
    
    let net_rate = rain - evap;
    source_h[cell_id] = source_h[cell_id] + net_rate * area;
}

// ==================== 辅助：清零源项 ====================

@compute @workgroup_size(256)
fn clear_sources(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_id = gid.x;
    
    if cell_id >= params.num_cells {
        return;
    }
    
    source_h[cell_id] = 0.0;
    source_hu[cell_id] = 0.0;
    source_hv[cell_id] = 0.0;
}
