// marihydro\crates\mh_physics\src\engine\strategy\explicit.rs
//! 显式时间积分策略
//!
//! 基于 Godunov 格式的显式有限体积法。
//! 
//! 该模块实现了经典的 Godunov 有限体积方法，使用 HLL 近似黎曼求解器
//! 计算单元间的数值通量。支持静水重构以处理变化的地形。

use super::{ExplicitConfig, StepResult, TimeIntegrationStrategy};
use super::workspace::SolverWorkspaceGeneric;
use crate::core::{Backend, CpuBackend, Scalar};
use crate::mesh::MeshTopology;
use crate::state::ShallowWaterStateGeneric;

/// 显式时间积分策略
/// 
/// 使用 Godunov 格式的显式有限体积法进行时间积分。
/// 通过 HLL 近似黎曼求解器计算单元间的数值通量，
/// 结合静水重构技术处理变化地形。
/// 
/// # 类型参数
/// 
/// - `B`: 计算后端类型，必须实现 `Backend` trait
/// 
/// # 示例
/// 
/// ```ignore
/// let backend = CpuBackend::<f64>::new();
/// let config = ExplicitConfig::new();
/// let strategy = ExplicitStrategy::new_with_backend(backend, config);
/// ```
pub struct ExplicitStrategy<B: Backend> {
    /// 计算后端实例
    backend: B,
    /// 配置
    config: ExplicitConfig,
    /// 重力加速度（缓存的后端标量类型）
    #[allow(dead_code)]
    gravity: B::Scalar,
    /// 干单元阈值（缓存的后端标量类型）
    #[allow(dead_code)]
    h_dry: B::Scalar,
}

impl<B: Backend> ExplicitStrategy<B> {
    /// 使用后端实例创建显式策略
    /// 
    /// # 参数
    /// 
    /// - `backend`: 计算后端实例，用于所有数值计算操作
    /// - `config`: 显式策略配置，包含 CFL 数、重力加速度等参数
    /// 
    /// # 返回
    /// 
    /// 返回初始化完成的显式策略实例
    pub fn new_with_backend(backend: B, config: ExplicitConfig) -> Self {
        Self {
            backend,
            gravity: B::Scalar::from_f64(config.gravity),
            h_dry: B::Scalar::from_f64(config.h_dry),
            config,
        }
    }
    
    /// 获取后端引用
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 获取配置引用
    #[inline]
    pub fn config(&self) -> &ExplicitConfig {
        &self.config
    }
}

impl<B: Backend + Clone> ExplicitStrategy<B> {
    /// 创建显式策略（需要 Clone 后端）
    /// 
    /// 此方法用于兼容默认后端场景。对于大多数情况，
    /// 建议使用 `new_with_backend` 方法显式传入后端实例。
    #[deprecated(note = "请使用 new_with_backend 方法显式传入后端实例")]
    pub fn new(config: ExplicitConfig) -> Self
    where
        B: Default,
    {
        Self::new_with_backend(B::default(), config)
    }
}

/// HLL 通量计算结果
struct HllFlux {
    /// 质量通量
    f_h: f64,
    /// x 方向动量通量
    f_hu: f64,
    /// y 方向动量通量
    f_hv: f64,
    /// 最大波速
    max_speed: f64,
}

/// 计算 HLL 数值通量
/// 
/// 使用 Harten-Lax-van Leer (HLL) 近似黎曼求解器计算单元界面的数值通量。
/// 该方法考虑了左右两侧的状态，使用波速估计来确定通量的方向。
/// 
/// # 参数
/// 
/// - `h_l`, `u_l`, `v_l`: 左侧状态（水深、x速度、y速度）
/// - `h_r`, `u_r`, `v_r`: 右侧状态
/// - `normal`: 界面法向量 [nx, ny]
/// - `gravity`: 重力加速度
/// - `h_dry`: 干单元阈值
/// 
/// # 返回
/// 
/// 返回 HLL 通量结构，包含质量和动量通量以及最大波速
#[inline]
fn compute_hll_flux(
    h_l: f64, u_l: f64, v_l: f64,
    h_r: f64, u_r: f64, v_r: f64,
    normal: [f64; 2],
    gravity: f64,
    h_dry: f64,
) -> HllFlux {
    // 投影到法向的速度分量
    let un_l = u_l * normal[0] + v_l * normal[1];
    let un_r = u_r * normal[0] + v_r * normal[1];
    
    // 波速估计（Einfeldt 估计）
    let c_l = if h_l > h_dry { (gravity * h_l).sqrt() } else { 0.0 };
    let c_r = if h_r > h_dry { (gravity * h_r).sqrt() } else { 0.0 };
    
    // Roe 平均波速
    let h_roe = 0.5 * (h_l + h_r);
    let _c_roe = if h_roe > h_dry { (gravity * h_roe).sqrt() } else { 0.0 };
    
    // HLL 波速边界
    let s_l = (un_l - c_l).min(un_r - c_r).min(0.0);
    let s_r = (un_l + c_l).max(un_r + c_r).max(0.0);
    
    let max_speed = s_l.abs().max(s_r.abs());
    
    // 计算左右通量
    let f_l_h = h_l * un_l;
    let f_l_hu = h_l * u_l * un_l + 0.5 * gravity * h_l * h_l * normal[0];
    let f_l_hv = h_l * v_l * un_l + 0.5 * gravity * h_l * h_l * normal[1];
    
    let f_r_h = h_r * un_r;
    let f_r_hu = h_r * u_r * un_r + 0.5 * gravity * h_r * h_r * normal[0];
    let f_r_hv = h_r * v_r * un_r + 0.5 * gravity * h_r * h_r * normal[1];
    
    // HLL 通量公式
    let (f_h, f_hu, f_hv) = if s_l >= 0.0 {
        // 全部来自左侧
        (f_l_h, f_l_hu, f_l_hv)
    } else if s_r <= 0.0 {
        // 全部来自右侧
        (f_r_h, f_r_hu, f_r_hv)
    } else {
        // 中间状态
        let denom = s_r - s_l;
        if denom.abs() < 1e-14 {
            (0.0, 0.0, 0.0)
        } else {
            let f_h = (s_r * f_l_h - s_l * f_r_h + s_l * s_r * (h_r - h_l)) / denom;
            let f_hu = (s_r * f_l_hu - s_l * f_r_hu + s_l * s_r * (h_r * u_r - h_l * u_l)) / denom;
            let f_hv = (s_r * f_l_hv - s_l * f_r_hv + s_l * s_r * (h_r * v_r - h_l * v_l)) / denom;
            (f_h, f_hu, f_hv)
        }
    };
    
    HllFlux { f_h, f_hu, f_hv, max_speed }
}

impl TimeIntegrationStrategy<CpuBackend<f64>> for ExplicitStrategy<CpuBackend<f64>> {
    fn name(&self) -> &'static str {
        "显式 Godunov (HLL)"
    }
    
    fn step(
        &mut self,
        state: &mut ShallowWaterStateGeneric<CpuBackend<f64>>,
        mesh: &dyn MeshTopology<CpuBackend<f64>>,
        workspace: &mut SolverWorkspaceGeneric<CpuBackend<f64>>,
        dt: f64,
    ) -> StepResult<f64> {
        // ========== 第1步：重置工作区 ==========
        workspace.reset();
        
        let n_cells = mesh.n_cells();
        
        // 获取状态切片（只读）
        let h: &[f64] = &state.h;
        let hu: &[f64] = &state.hu;
        let hv: &[f64] = &state.hv;
        let z: &[f64] = &state.z;
        
        // 获取通量缓冲区（可写）
        let flux_h: &mut [f64] = &mut workspace.flux_h;
        let flux_hu: &mut [f64] = &mut workspace.flux_hu;
        let flux_hv: &mut [f64] = &mut workspace.flux_hv;
        
        let h_dry = self.config.h_dry;
        let gravity = self.config.gravity;
        
        let mut max_wave_speed = 0.0f64;
        let mut dry_cells = 0usize;
        
        // ========== 第2步：计算内部面通量 ==========
        for face in mesh.interior_faces() {
            let owner = mesh.face_owner(*face);
            let neighbor = mesh.face_neighbor(*face).unwrap();
            
            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);
            
            // 获取左右单元状态
            let h_l = h[owner];
            let h_r = h[neighbor];
            let z_l = z[owner];
            let z_r = z[neighbor];
            
            // 计算左右单元的速度分量
            let (u_l, v_l) = if h_l > h_dry {
                (hu[owner] / h_l, hv[owner] / h_l)
            } else {
                (0.0, 0.0)
            };
            
            let (u_r, v_r) = if h_r > h_dry {
                (hu[neighbor] / h_r, hv[neighbor] / h_r)
            } else {
                (0.0, 0.0)
            };
            
            // 静水重构：确保平衡态时通量为零
            // 使用 Audusse et al. (2004) 的静水重构方法
            let eta_l = h_l + z_l;  // 左侧水位
            let eta_r = h_r + z_r;  // 右侧水位
            let z_star = z_l.max(z_r);  // 界面处的最高床底高程
            
            let h_l_star = (eta_l - z_star).max(0.0);  // 重构后的左侧水深
            let h_r_star = (eta_r - z_star).max(0.0);  // 重构后的右侧水深
            
            // 使用重构后的水深计算 HLL 通量
            let hll = compute_hll_flux(
                h_l_star, u_l, v_l,
                h_r_star, u_r, v_r,
                normal, gravity, h_dry,
            );
            
            // 更新最大波速
            max_wave_speed = max_wave_speed.max(hll.max_speed);
            
            // 通量乘以界面长度并累加到单元
            let flux_mag_h = hll.f_h * length;
            let flux_mag_hu = hll.f_hu * length;
            let flux_mag_hv = hll.f_hv * length;
            
            // Owner 单元减去通量，Neighbor 单元加上通量
            flux_h[owner] -= flux_mag_h;
            flux_h[neighbor] += flux_mag_h;
            flux_hu[owner] -= flux_mag_hu;
            flux_hu[neighbor] += flux_mag_hu;
            flux_hv[owner] -= flux_mag_hv;
            flux_hv[neighbor] += flux_mag_hv;
        }
        
        // ========== 第3步：边界面处理（反射边界）==========
        // 对于固壁边界，法向速度为零，仅存在压力作用
        for face in mesh.boundary_faces() {
            let owner = mesh.face_owner(*face);
            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);
            
            let h_l = h[owner];
            
            // 干单元跳过
            if h_l <= h_dry {
                continue;
            }
            
            let u_l = hu[owner] / h_l;
            let v_l = hv[owner] / h_l;
            
            // 计算法向速度（用于波速估计）
            let un_l = u_l * normal[0] + v_l * normal[1];
            
            // 固壁边界：仅静水压力作用于边界
            // F_pressure = (1/2) * g * h^2 * n
            let f_hu = 0.5 * gravity * h_l * h_l * normal[0] * length;
            let f_hv = 0.5 * gravity * h_l * h_l * normal[1] * length;
            
            flux_hu[owner] -= f_hu;
            flux_hv[owner] -= f_hv;
            
            // 更新最大波速
            let c = (gravity * h_l).sqrt();
            max_wave_speed = max_wave_speed.max(un_l.abs() + c);
        }
        
        // ========== 第4步：更新状态 ==========
        // 使用前向欧拉时间积分：U^{n+1} = U^n + dt * (1/A) * Σ F
        let h_mut: &mut [f64] = &mut state.h;
        let hu_mut: &mut [f64] = &mut state.hu;
        let hv_mut: &mut [f64] = &mut state.hv;
        
        for i in 0..n_cells {
            let area = mesh.cell_area(i);
            if area <= 0.0 {
                continue;
            }
            let inv_area = 1.0 / area;
            
            // 前向欧拉更新
            h_mut[i] += dt * flux_h[i] * inv_area;
            hu_mut[i] += dt * flux_hu[i] * inv_area;
            hv_mut[i] += dt * flux_hv[i] * inv_area;
            
            // 干单元处理：水深低于阈值时清零
            if h_mut[i] < h_dry {
                h_mut[i] = 0.0;
                hu_mut[i] = 0.0;
                hv_mut[i] = 0.0;
                dry_cells += 1;
            }
        }
        
        StepResult {
            dt_used: dt,
            max_wave_speed,
            dry_cells,
            limited_cells: 0,
            converged: true,  // 显式方法总是"收敛"
            iterations: 0,
        }
    }
    
    /// 计算稳定时间步长
    /// 
    /// 基于 CFL 条件计算最大允许时间步长：
    /// dt <= CFL * dx / (|u| + c)
    /// 其中 c = sqrt(g*h) 是浅水波速
    fn compute_stable_dt(
        &self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        mesh: &dyn MeshTopology<CpuBackend<f64>>,
        cfl: f64,
    ) -> f64 {
        let h: &[f64] = &state.h;
        let hu: &[f64] = &state.hu;
        let hv: &[f64] = &state.hv;
        
        let h_dry = self.config.h_dry;
        let gravity = self.config.gravity;
        
        let mut dt_min = f64::MAX;
        
        for i in 0..mesh.n_cells() {
            // 跳过干单元
            if h[i] <= h_dry {
                continue;
            }
            
            // 计算速度分量
            let u = hu[i] / h[i];
            let v = hv[i] / h[i];
            
            // 浅水波速
            let c = (gravity * h[i]).sqrt();
            
            // 特征速度 = 流速 + 波速
            let speed = (u * u + v * v).sqrt() + c;
            
            if speed > 1e-10 {
                // 使用单元面积的平方根作为特征长度
                let area = mesh.cell_area(i);
                let dx = area.sqrt();
                let dt_local = cfl * dx / speed;
                dt_min = dt_min.min(dt_local);
            }
        }
        
        // 如果所有单元都是干的，返回一个小的默认值
        if dt_min == f64::MAX {
            dt_min = 1e-6;
        }
        
        dt_min
    }
    
    /// 推荐的 CFL 数
    fn recommended_cfl(&self) -> f64 {
        // 显式方法通常使用 0.5 左右的 CFL 数以确保稳定性
        self.config.cfl.max(0.5)
    }
}
