// marihydro\crates\mh_physics\src\engine\strategy\explicit.rs
//! 显式时间积分策略
//!
//! 基于 Godunov 格式的显式有限体积法。
//! 
//! 该模块实现了经典的 Godunov 有限体积方法，使用 HLL 近似黎曼求解器
//! 计算单元间的数值通量。支持静水重构以处理变化的地形。

use super::{ExplicitConfig, StepResult, TimeIntegrationStrategy};
use super::workspace::SolverWorkspaceGeneric;
use crate::core::Backend;
use mh_runtime::RuntimeScalar as Scalar;
use crate::mesh::MeshTopology;
use crate::state::ShallowWaterState;
use num_traits::Float;

#[inline]
#[track_caller]
fn scalar_from_config_or_panic<S: Scalar>(value: f64, context: &'static str) -> S {
    S::from_config(value).unwrap_or_else(|| {
        panic!(
            "[mh_physics::engine::strategy::explicit] config scalar conversion failed: context={context}, value={value}"
        )
    })
}

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
            gravity: backend.config_scalar(config.gravity, "ExplicitStrategy.new_with_backend.gravity"),
            h_dry: backend.config_scalar(config.h_dry, "ExplicitStrategy.new_with_backend.h_dry"),
            backend,
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

/// HLL 通量计算结果
struct HllFlux<S: Scalar> {
    /// 质量通量
    f_h: S,
    /// x 方向动量通量
    f_hu: S,
    /// y 方向动量通量
    f_hv: S,
    /// 最大波速
    max_speed: S,
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
fn compute_hll_flux<S: Scalar>(
    h_l: S,
    u_l: S,
    v_l: S,
    h_r: S,
    u_r: S,
    v_r: S,
    normal: [S; 2],
    gravity: S,
    h_dry: S,
) -> HllFlux<S> {
    let zero = S::ZERO;
    let half = scalar_from_config_or_panic::<S>(0.5, "compute_hll_flux.half");

    // 投影到法向的速度分量
    let un_l = u_l * normal[0] + v_l * normal[1];
    let un_r = u_r * normal[0] + v_r * normal[1];
    
    // 波速估计（Einfeldt 估计）
    let c_l = if h_l > h_dry { (gravity * h_l).sqrt() } else { zero };
    let c_r = if h_r > h_dry { (gravity * h_r).sqrt() } else { zero };
    
    // Roe 平均波速
    let h_roe = half * (h_l + h_r);
    let _c_roe = if h_roe > h_dry { (gravity * h_roe).sqrt() } else { zero };
    
    // HLL 波速边界
    let s_l = (un_l - c_l).min(un_r - c_r).min(zero);
    let s_r = (un_l + c_l).max(un_r + c_r).max(zero);
    
    let max_speed = s_l.abs().max(s_r.abs());
    
    // 计算左右通量
    let f_l_h = h_l * un_l;
    let f_l_hu = h_l * u_l * un_l + half * gravity * h_l * h_l * normal[0];
    let f_l_hv = h_l * v_l * un_l + half * gravity * h_l * h_l * normal[1];
    
    let f_r_h = h_r * un_r;
    let f_r_hu = h_r * u_r * un_r + half * gravity * h_r * h_r * normal[0];
    let f_r_hv = h_r * v_r * un_r + half * gravity * h_r * h_r * normal[1];
    
    // HLL 通量公式
    let (f_h, f_hu, f_hv) = if s_l >= zero {
        // 全部来自左侧
        (f_l_h, f_l_hu, f_l_hv)
    } else if s_r <= zero {
        // 全部来自右侧
        (f_r_h, f_r_hu, f_r_hv)
    } else {
        // 中间状态
        let denom = s_r - s_l;
        let eps = scalar_from_config_or_panic::<S>(1e-14, "compute_hll_flux.eps");
        if denom.abs() < eps {
            (zero, zero, zero)
        } else {
            let f_h = (s_r * f_l_h - s_l * f_r_h + s_l * s_r * (h_r - h_l)) / denom;
            let f_hu = (s_r * f_l_hu - s_l * f_r_hu + s_l * s_r * (h_r * u_r - h_l * u_l)) / denom;
            let f_hv = (s_r * f_l_hv - s_l * f_r_hv + s_l * s_r * (h_r * v_r - h_l * v_l)) / denom;
            (f_h, f_hu, f_hv)
        }
    };
    
    HllFlux { f_h, f_hu, f_hv, max_speed }
}

impl<B: Backend> TimeIntegrationStrategy<B> for ExplicitStrategy<B> {
    fn name(&self) -> &'static str {
        "显式 Godunov (HLL)"
    }
    
    fn step(
        &mut self,
        state: &mut ShallowWaterState<B>,
        mesh: &dyn MeshTopology<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        dt: B::Scalar,
    ) -> StepResult<B::Scalar> {
        workspace.reset();
        
        let n_cells = mesh.n_cells();
        
        let h: &[B::Scalar] = &state.h;
        let hu: &[B::Scalar] = &state.hu;
        let hv: &[B::Scalar] = &state.hv;
        let z: &[B::Scalar] = &state.z;
        
        let flux_h: &mut [B::Scalar] = &mut workspace.flux_h;
        let flux_hu: &mut [B::Scalar] = &mut workspace.flux_hu;
        let flux_hv: &mut [B::Scalar] = &mut workspace.flux_hv;
        
        let h_dry = self.h_dry;
        let gravity = self.gravity;
        let zero = B::Scalar::ZERO;
        let half = B::Scalar::HALF;
        let one = B::Scalar::ONE;
        
        let mut max_wave_speed = zero;
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
                (zero, zero)
            };
            
            let (u_r, v_r) = if h_r > h_dry {
                (hu[neighbor] / h_r, hv[neighbor] / h_r)
            } else {
                (zero, zero)
            };
            
            // 静水重构：确保平衡态时通量为零
            let eta_l = h_l + z_l;
            let eta_r = h_r + z_r;
            let z_star = z_l.max(z_r);
            
            let h_l_star = (eta_l - z_star).max(zero);
            let h_r_star = (eta_r - z_star).max(zero);
            
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
        for face in mesh.boundary_faces() {
            let owner = mesh.face_owner(*face);
            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);
            
            let h_l = h[owner];
            
            if h_l <= h_dry {
                continue;
            }
            
            let u_l = hu[owner] / h_l;
            let v_l = hv[owner] / h_l;
            
            let un_l = u_l * normal[0] + v_l * normal[1];
            
            // 固壁边界：仅静水压力作用于边界
            let f_hu = half * gravity * h_l * h_l * normal[0] * length;
            let f_hv = half * gravity * h_l * h_l * normal[1] * length;
            
            flux_hu[owner] -= f_hu;
            flux_hv[owner] -= f_hv;
            
            // 更新最大波速
            let c = (gravity * h_l).sqrt();
            max_wave_speed = max_wave_speed.max(un_l.abs() + c);
        }
        
        // ========== 第4步：更新状态 ==========
        // 使用前向欧拉时间积分：U^{n+1} = U^n + dt * (1/A) * Σ F
        let h_mut: &mut [B::Scalar] = &mut state.h;
        let hu_mut: &mut [B::Scalar] = &mut state.hu;
        let hv_mut: &mut [B::Scalar] = &mut state.hv;
        
        for i in 0..n_cells {
            let area = mesh.cell_area(i);
            if !area.is_finite() || area <= zero {
                continue;
            }
            let inv_area = one / area;
            
            // 前向欧拉更新
            h_mut[i] += dt * flux_h[i] * inv_area;
            hu_mut[i] += dt * flux_hu[i] * inv_area;
            hv_mut[i] += dt * flux_hv[i] * inv_area;
            
            // 干单元处理：水深低于阈值时清零
            if h_mut[i] < h_dry {
                h_mut[i] = zero;
                hu_mut[i] = zero;
                hv_mut[i] = zero;
                dry_cells += 1;
            }
        }
        
        StepResult {
            dt_used: dt,
            max_wave_speed,
            dry_cells,
            limited_cells: 0,
            converged: true,
            iterations: 0,
        }
    }
    
    fn compute_stable_dt(
        &self,
        state: &ShallowWaterState<B>,
        mesh: &dyn MeshTopology<B>,
        cfl: B::Scalar,
    ) -> B::Scalar {
        let h: &[B::Scalar] = &state.h;
        let hu: &[B::Scalar] = &state.hu;
        let hv: &[B::Scalar] = &state.hv;
        
        let h_dry = self.h_dry;
        let gravity = self.gravity;
        let zero = B::Scalar::ZERO;
        let tiny = self
            .backend
            .config_scalar(1e-10, "ExplicitStrategy.compute_stable_dt.tiny");
        let default_dt = self
            .backend
            .config_scalar(1e-6, "ExplicitStrategy.compute_stable_dt.default_dt");
        
        let mut dt_min = B::Scalar::MAX;
        
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
            
            if speed > tiny {
                // 使用单元面积的平方根作为特征长度
                let area = mesh.cell_area(i);
                if !area.is_finite() || area <= zero {
                    continue;
                }
                let dx = area.sqrt();
                let dt_local = cfl * dx / speed;
                dt_min = dt_min.min(dt_local);
            }
        }
        
        // 如果所有单元都是干的，返回一个小的默认值
        if dt_min == B::Scalar::MAX {
            default_dt
        } else {
            dt_min
        }
    }
    
    /// 推荐的 CFL 数
    fn recommended_cfl(&self) -> B::Scalar {
        let cfg_cfl = self
            .backend
            .config_scalar(self.config.cfl, "ExplicitStrategy.recommended_cfl.config");
        let min_cfl = self
            .backend
            .config_scalar(0.5, "ExplicitStrategy.recommended_cfl.floor");
        cfg_cfl.max(min_cfl)
    }
}
