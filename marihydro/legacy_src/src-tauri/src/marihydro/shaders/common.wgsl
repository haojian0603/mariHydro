// MariHydro GPU计算着色器 - 公共模块
// common.wgsl

// ==================== 物理常量 ====================
const G: f32 = 9.80665;           // 重力加速度 [m/s²]
const EPS_H: f32 = 1e-6;          // 最小水深阈值 [m]
const EPS_VEL: f32 = 1e-10;       // 最小速度阈值 [m/s]
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

// 无效单元标记
const INVALID_CELL: u32 = 0xFFFFFFFFu;

// ==================== 基本结构体 ====================

// 守恒变量状态
struct ConservedState {
    h: f32,      // 水深 [m]
    hu: f32,     // x方向动量 [m²/s]
    hv: f32,     // y方向动量 [m²/s]
    z: f32,      // 底床高程 [m]
}

// 原始变量状态
struct PrimitiveState {
    h: f32,      // 水深 [m]
    u: f32,      // x速度 [m/s]
    v: f32,      // y速度 [m/s]
    eta: f32,    // 水面高程 [m]
}

// 通量向量
struct FluxVector {
    f_h: f32,    // 质量通量
    f_hu: f32,   // x动量通量
    f_hv: f32,   // y动量通量
}

// 面几何信息
struct FaceGeometry {
    cx: f32,     // 面中心x
    cy: f32,     // 面中心y
    nx: f32,     // 法向量x
    ny: f32,     // 法向量y
    length: f32, // 面长度
}

// 梯度向量
struct Gradient {
    dx: f32,
    dy: f32,
}

// ==================== 工具函数 ====================

// 从守恒变量计算原始变量
fn to_primitive(state: ConservedState) -> PrimitiveState {
    let h_safe = max(state.h, EPS_H);
    return PrimitiveState(
        state.h,
        state.hu / h_safe,
        state.hv / h_safe,
        state.h + state.z
    );
}

// 从原始变量计算守恒变量
fn to_conserved(prim: PrimitiveState, z: f32) -> ConservedState {
    return ConservedState(
        prim.h,
        prim.h * prim.u,
        prim.h * prim.v,
        z
    );
}

// 计算声速
fn sound_speed(h: f32) -> f32 {
    return sqrt(G * max(h, 0.0));
}

// 计算速度大小
fn velocity_magnitude(u: f32, v: f32) -> f32 {
    return sqrt(u * u + v * v);
}

// 安全除法
fn safe_div(a: f32, b: f32) -> f32 {
    if abs(b) < EPS_VEL {
        return 0.0;
    }
    return a / b;
}

// 限制器函数 - minmod
fn minmod(a: f32, b: f32) -> f32 {
    if a * b <= 0.0 {
        return 0.0;
    }
    if abs(a) < abs(b) {
        return a;
    }
    return b;
}

// 限制器函数 - superbee
fn superbee(a: f32, b: f32) -> f32 {
    if a * b <= 0.0 {
        return 0.0;
    }
    let abs_a = abs(a);
    let abs_b = abs(b);
    let s = sign(a);
    return s * max(min(2.0 * abs_a, abs_b), min(abs_a, 2.0 * abs_b));
}

// 限制器函数 - van Leer
fn van_leer(a: f32, b: f32) -> f32 {
    if a * b <= 0.0 {
        return 0.0;
    }
    return 2.0 * a * b / (a + b);
}

// 计算x方向物理通量
fn flux_x(state: ConservedState) -> FluxVector {
    let h = max(state.h, EPS_H);
    let u = state.hu / h;
    let v = state.hv / h;
    
    return FluxVector(
        state.hu,                              // h*u
        state.hu * u + 0.5 * G * h * h,        // h*u² + 0.5*g*h²
        state.hu * v                           // h*u*v
    );
}

// 计算y方向物理通量
fn flux_y(state: ConservedState) -> FluxVector {
    let h = max(state.h, EPS_H);
    let u = state.hu / h;
    let v = state.hv / h;
    
    return FluxVector(
        state.hv,                              // h*v
        state.hu * v,                          // h*u*v
        state.hv * v + 0.5 * G * h * h         // h*v² + 0.5*g*h²
    );
}

// 计算法向通量
fn flux_normal(state: ConservedState, nx: f32, ny: f32) -> FluxVector {
    let fx = flux_x(state);
    let fy = flux_y(state);
    
    return FluxVector(
        fx.f_h * nx + fy.f_h * ny,
        fx.f_hu * nx + fy.f_hu * ny,
        fx.f_hv * nx + fy.f_hv * ny
    );
}

// 旋转到面局部坐标系
fn rotate_to_local(hu: f32, hv: f32, nx: f32, ny: f32) -> vec2<f32> {
    let hun = hu * nx + hv * ny;   // 法向动量
    let hut = -hu * ny + hv * nx;  // 切向动量
    return vec2<f32>(hun, hut);
}

// 从面局部坐标系旋转回全局
fn rotate_to_global(hun: f32, hut: f32, nx: f32, ny: f32) -> vec2<f32> {
    let hu = hun * nx - hut * ny;
    let hv = hun * ny + hut * nx;
    return vec2<f32>(hu, hv);
}

// ==================== 干湿判断 ====================

fn is_wet(h: f32) -> bool {
    return h > EPS_H;
}

fn is_dry(h: f32) -> bool {
    return h <= EPS_H;
}

// 干湿边界处理
fn handle_dry_cell(h: f32, hu: f32, hv: f32) -> ConservedState {
    if is_dry(h) {
        return ConservedState(0.0, 0.0, 0.0, 0.0);
    }
    return ConservedState(h, hu, hv, 0.0);
}

// ==================== 原子操作辅助 ====================
// 注意：WGSL不支持f32原子操作，需要使用着色并行策略

// 将f32转换为i32进行原子操作（有精度损失）
fn f32_to_i32_fixed(val: f32, scale: f32) -> i32 {
    return i32(val * scale);
}

fn i32_to_f32_fixed(val: i32, scale: f32) -> f32 {
    return f32(val) / scale;
}
