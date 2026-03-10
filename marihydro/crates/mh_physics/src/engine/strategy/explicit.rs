// marihydro\crates\mh_physics\src\engine\strategy\explicit.rs
//! 鏄惧紡鏃堕棿绉垎绛栫暐
//!
//! 鍩轰簬 Godunov 鏍煎紡鐨勬樉寮忔湁闄愪綋绉硶銆?
//! 
//! 璇ユā鍧楀疄鐜颁簡缁忓吀鐨?Godunov 鏈夐檺浣撶Н鏂规硶锛屼娇鐢?HLL 杩戜技榛庢浖姹傝В鍣?
//! 璁＄畻鍗曞厓闂寸殑鏁板€奸€氶噺銆傛敮鎸侀潤姘撮噸鏋勪互澶勭悊鍙樺寲鐨勫湴褰€?

use super::{ExplicitConfig, StepResult, TimeIntegrationStrategy};
use super::workspace::SolverWorkspaceGeneric;
use crate::core::{Backend, CpuBackend, Scalar};
use crate::mesh::MeshTopology;
use crate::state::ShallowWaterStateGeneric;

/// 鏄惧紡鏃堕棿绉垎绛栫暐
/// 
/// 浣跨敤 Godunov 鏍煎紡鐨勬樉寮忔湁闄愪綋绉硶杩涜鏃堕棿绉垎銆?
/// 閫氳繃 HLL 杩戜技榛庢浖姹傝В鍣ㄨ绠楀崟鍏冮棿鐨勬暟鍊奸€氶噺锛?
/// 缁撳悎闈欐按閲嶆瀯鎶€鏈鐞嗗彉鍖栧湴褰€?
/// 
/// # 绫诲瀷鍙傛暟
/// 
/// - `B`: 璁＄畻鍚庣绫诲瀷锛屽繀椤诲疄鐜?`Backend` trait
/// 
/// # 绀轰緥
/// 
/// ```ignore
/// let backend = CpuBackend::<f64>::new();
/// let config = ExplicitConfig::new();
/// let strategy = ExplicitStrategy::new_with_backend(backend, config);
/// ```
pub struct ExplicitStrategy<B: Backend> {
    /// 璁＄畻鍚庣瀹炰緥
    backend: B,
    /// 閰嶇疆
    config: ExplicitConfig,
    /// 閲嶅姏鍔犻€熷害锛堢紦瀛樼殑鍚庣鏍囬噺绫诲瀷锛?
    #[allow(dead_code)]
    gravity: B::Scalar,
    /// 骞插崟鍏冮槇鍊硷紙缂撳瓨鐨勫悗绔爣閲忕被鍨嬶級
    #[allow(dead_code)]
    h_dry: B::Scalar,
}

impl<B: Backend> ExplicitStrategy<B> {
    /// 浣跨敤鍚庣瀹炰緥鍒涘缓鏄惧紡绛栫暐
    /// 
    /// # 鍙傛暟
    /// 
    /// - `backend`: 璁＄畻鍚庣瀹炰緥锛岀敤浜庢墍鏈夋暟鍊艰绠楁搷浣?
    /// - `config`: 鏄惧紡绛栫暐閰嶇疆锛屽寘鍚?CFL 鏁般€侀噸鍔涘姞閫熷害绛夊弬鏁?
    /// 
    /// # 杩斿洖
    /// 
    /// 杩斿洖鍒濆鍖栧畬鎴愮殑鏄惧紡绛栫暐瀹炰緥
    pub fn new_with_backend(backend: B, config: ExplicitConfig) -> Self {
        Self {
            backend,
            gravity: <B::Scalar as Scalar>::from_f64(config.gravity),
            h_dry: <B::Scalar as Scalar>::from_f64(config.h_dry),
            config,
        }
    }
    
    /// 鑾峰彇鍚庣寮曠敤
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 鑾峰彇閰嶇疆寮曠敤
    #[inline]
    pub fn config(&self) -> &ExplicitConfig {
        &self.config
    }
}


/// HLL 閫氶噺璁＄畻缁撴灉
struct HllFlux {
    /// 璐ㄩ噺閫氶噺
    f_h: f64,
    /// x 鏂瑰悜鍔ㄩ噺閫氶噺
    f_hu: f64,
    /// y 鏂瑰悜鍔ㄩ噺閫氶噺
    f_hv: f64,
    /// 鏈€澶ф尝閫?
    max_speed: f64,
}

/// 璁＄畻 HLL 鏁板€奸€氶噺
/// 
/// 浣跨敤 Harten-Lax-van Leer (HLL) 杩戜技榛庢浖姹傝В鍣ㄨ绠楀崟鍏冪晫闈㈢殑鏁板€奸€氶噺銆?
/// 璇ユ柟娉曡€冭檻浜嗗乏鍙充袱渚х殑鐘舵€侊紝浣跨敤娉㈤€熶及璁℃潵纭畾閫氶噺鐨勬柟鍚戙€?
/// 
/// # 鍙傛暟
/// 
/// - `h_l`, `u_l`, `v_l`: 宸︿晶鐘舵€侊紙姘存繁銆亁閫熷害銆亂閫熷害锛?
/// - `h_r`, `u_r`, `v_r`: 鍙充晶鐘舵€?
/// - `normal`: 鐣岄潰娉曞悜閲?[nx, ny]
/// - `gravity`: 閲嶅姏鍔犻€熷害
/// - `h_dry`: 骞插崟鍏冮槇鍊?
/// 
/// # 杩斿洖
/// 
/// 杩斿洖 HLL 閫氶噺缁撴瀯锛屽寘鍚川閲忓拰鍔ㄩ噺閫氶噺浠ュ強鏈€澶ф尝閫?
#[inline]
fn compute_hll_flux(
    h_l: f64, u_l: f64, v_l: f64,
    h_r: f64, u_r: f64, v_r: f64,
    normal: [f64; 2],
    gravity: f64,
    h_dry: f64,
) -> HllFlux {
    // 鎶曞奖鍒版硶鍚戠殑閫熷害鍒嗛噺
    let un_l = u_l * normal[0] + v_l * normal[1];
    let un_r = u_r * normal[0] + v_r * normal[1];
    
    // 娉㈤€熶及璁★紙Einfeldt 浼拌锛?
    let c_l = if h_l > h_dry { (gravity * h_l).sqrt() } else { 0.0 };
    let c_r = if h_r > h_dry { (gravity * h_r).sqrt() } else { 0.0 };
    
    // Roe 骞冲潎娉㈤€?
    let h_roe = 0.5 * (h_l + h_r);
    let _c_roe = if h_roe > h_dry { (gravity * h_roe).sqrt() } else { 0.0 };
    
    // HLL 娉㈤€熻竟鐣?
    let s_l = (un_l - c_l).min(un_r - c_r).min(0.0);
    let s_r = (un_l + c_l).max(un_r + c_r).max(0.0);
    
    let max_speed = s_l.abs().max(s_r.abs());
    
    // 璁＄畻宸﹀彸閫氶噺
    let f_l_h = h_l * un_l;
    let f_l_hu = h_l * u_l * un_l + 0.5 * gravity * h_l * h_l * normal[0];
    let f_l_hv = h_l * v_l * un_l + 0.5 * gravity * h_l * h_l * normal[1];
    
    let f_r_h = h_r * un_r;
    let f_r_hu = h_r * u_r * un_r + 0.5 * gravity * h_r * h_r * normal[0];
    let f_r_hv = h_r * v_r * un_r + 0.5 * gravity * h_r * h_r * normal[1];
    
    // HLL 閫氶噺鍏紡
    let (f_h, f_hu, f_hv) = if s_l >= 0.0 {
        // 鍏ㄩ儴鏉ヨ嚜宸︿晶
        (f_l_h, f_l_hu, f_l_hv)
    } else if s_r <= 0.0 {
        // 鍏ㄩ儴鏉ヨ嚜鍙充晶
        (f_r_h, f_r_hu, f_r_hv)
    } else {
        // 涓棿鐘舵€?
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
        "鏄惧紡 Godunov (HLL)"
    }
    
    fn step(
        &mut self,
        state: &mut ShallowWaterStateGeneric<CpuBackend<f64>>,
        mesh: &dyn MeshTopology<CpuBackend<f64>>,
        workspace: &mut SolverWorkspaceGeneric<CpuBackend<f64>>,
        dt: f64,
    ) -> StepResult<f64> {
        // ========== 绗?姝ワ細閲嶇疆宸ヤ綔鍖?==========
        workspace.reset();
        
        let n_cells = mesh.n_cells();
        
        // 鑾峰彇鐘舵€佸垏鐗囷紙鍙锛?
        let h: &[f64] = &state.h;
        let hu: &[f64] = &state.hu;
        let hv: &[f64] = &state.hv;
        let z: &[f64] = &state.z;
        
        // 鑾峰彇閫氶噺缂撳啿鍖猴紙鍙啓锛?
        let flux_h: &mut [f64] = &mut workspace.flux_h;
        let flux_hu: &mut [f64] = &mut workspace.flux_hu;
        let flux_hv: &mut [f64] = &mut workspace.flux_hv;
        
        let h_dry = self.config.h_dry;
        let gravity = self.config.gravity;
        
        let mut max_wave_speed = 0.0f64;
        let mut dry_cells = 0usize;
        
        // ========== 绗?姝ワ細璁＄畻鍐呴儴闈㈤€氶噺 ==========
        for face in mesh.interior_faces() {
            let owner = mesh.face_owner(*face);
            let neighbor = mesh.face_neighbor(*face).unwrap();
            
            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);
            
            // 鑾峰彇宸﹀彸鍗曞厓鐘舵€?
            let h_l = h[owner];
            let h_r = h[neighbor];
            let z_l = z[owner];
            let z_r = z[neighbor];
            
            // 璁＄畻宸﹀彸鍗曞厓鐨勯€熷害鍒嗛噺
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
            
            // 闈欐按閲嶆瀯锛氱‘淇濆钩琛℃€佹椂閫氶噺涓洪浂
            // 浣跨敤 Audusse et al. (2004) 鐨勯潤姘撮噸鏋勬柟娉?
            let eta_l = h_l + z_l;  // 宸︿晶姘翠綅
            let eta_r = h_r + z_r;  // 鍙充晶姘翠綅
            let z_star = z_l.max(z_r);  // 鐣岄潰澶勭殑鏈€楂樺簥搴曢珮绋?
            
            let h_l_star = (eta_l - z_star).max(0.0);  // 閲嶆瀯鍚庣殑宸︿晶姘存繁
            let h_r_star = (eta_r - z_star).max(0.0);  // 閲嶆瀯鍚庣殑鍙充晶姘存繁
            
            // 浣跨敤閲嶆瀯鍚庣殑姘存繁璁＄畻 HLL 閫氶噺
            let hll = compute_hll_flux(
                h_l_star, u_l, v_l,
                h_r_star, u_r, v_r,
                normal, gravity, h_dry,
            );
            
            // 鏇存柊鏈€澶ф尝閫?
            max_wave_speed = max_wave_speed.max(hll.max_speed);
            
            // 閫氶噺涔樹互鐣岄潰闀垮害骞剁疮鍔犲埌鍗曞厓
            let flux_mag_h = hll.f_h * length;
            let flux_mag_hu = hll.f_hu * length;
            let flux_mag_hv = hll.f_hv * length;
            
            // Owner 鍗曞厓鍑忓幓閫氶噺锛孨eighbor 鍗曞厓鍔犱笂閫氶噺
            flux_h[owner] -= flux_mag_h;
            flux_h[neighbor] += flux_mag_h;
            flux_hu[owner] -= flux_mag_hu;
            flux_hu[neighbor] += flux_mag_hu;
            flux_hv[owner] -= flux_mag_hv;
            flux_hv[neighbor] += flux_mag_hv;
        }
        
        // ========== 绗?姝ワ細杈圭晫闈㈠鐞嗭紙鍙嶅皠杈圭晫锛?=========
        // 瀵逛簬鍥哄杈圭晫锛屾硶鍚戦€熷害涓洪浂锛屼粎瀛樺湪鍘嬪姏浣滅敤
        for face in mesh.boundary_faces() {
            let owner = mesh.face_owner(*face);
            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);
            
            let h_l = h[owner];
            
            // 骞插崟鍏冭烦杩?
            if h_l <= h_dry {
                continue;
            }
            
            let u_l = hu[owner] / h_l;
            let v_l = hv[owner] / h_l;
            
            // 璁＄畻娉曞悜閫熷害锛堢敤浜庢尝閫熶及璁★級
            let un_l = u_l * normal[0] + v_l * normal[1];
            
            // 鍥哄杈圭晫锛氫粎闈欐按鍘嬪姏浣滅敤浜庤竟鐣?
            // F_pressure = (1/2) * g * h^2 * n
            let f_hu = 0.5 * gravity * h_l * h_l * normal[0] * length;
            let f_hv = 0.5 * gravity * h_l * h_l * normal[1] * length;
            
            flux_hu[owner] -= f_hu;
            flux_hv[owner] -= f_hv;
            
            // 鏇存柊鏈€澶ф尝閫?
            let c = (gravity * h_l).sqrt();
            max_wave_speed = max_wave_speed.max(un_l.abs() + c);
        }
        
        // ========== 绗?姝ワ細鏇存柊鐘舵€?==========
        // 浣跨敤鍓嶅悜娆ф媺鏃堕棿绉垎锛歎^{n+1} = U^n + dt * (1/A) * 危 F
        let h_mut: &mut [f64] = &mut state.h;
        let hu_mut: &mut [f64] = &mut state.hu;
        let hv_mut: &mut [f64] = &mut state.hv;
        
        for i in 0..n_cells {
            let area = mesh.cell_area(i);
            if area <= 0.0 {
                continue;
            }
            let inv_area = 1.0 / area;
            
            // 鍓嶅悜娆ф媺鏇存柊
            h_mut[i] += dt * flux_h[i] * inv_area;
            hu_mut[i] += dt * flux_hu[i] * inv_area;
            hv_mut[i] += dt * flux_hv[i] * inv_area;
            
            // 骞插崟鍏冨鐞嗭細姘存繁浣庝簬闃堝€兼椂娓呴浂
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
            converged: true,  // 鏄惧紡鏂规硶鎬绘槸"鏀舵暃"
            iterations: 0,
        }
    }
    
    /// 璁＄畻绋冲畾鏃堕棿姝ラ暱
    /// 
    /// 鍩轰簬 CFL 鏉′欢璁＄畻鏈€澶у厑璁告椂闂存闀匡細
    /// dt <= CFL * dx / (|u| + c)
    /// 鍏朵腑 c = sqrt(g*h) 鏄祬姘存尝閫?
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
            // 璺宠繃骞插崟鍏?
            if h[i] <= h_dry {
                continue;
            }
            
            // 璁＄畻閫熷害鍒嗛噺
            let u = hu[i] / h[i];
            let v = hv[i] / h[i];
            
            // 娴呮按娉㈤€?
            let c = (gravity * h[i]).sqrt();
            
            // 鐗瑰緛閫熷害 = 娴侀€?+ 娉㈤€?
            let speed = (u * u + v * v).sqrt() + c;
            
            if speed > 1e-10 {
                // 浣跨敤鍗曞厓闈㈢Н鐨勫钩鏂规牴浣滀负鐗瑰緛闀垮害
                let area = mesh.cell_area(i);
                let dx = area.sqrt();
                let dt_local = cfl * dx / speed;
                dt_min = dt_min.min(dt_local);
            }
        }
        
        // 濡傛灉鎵€鏈夊崟鍏冮兘鏄共鐨勶紝杩斿洖涓€涓皬鐨勯粯璁ゅ€?
        if dt_min == f64::MAX {
            dt_min = 1e-6;
        }
        
        dt_min
    }
    
    /// 鎺ㄨ崘鐨?CFL 鏁?
    fn recommended_cfl(&self) -> f64 {
        // 鏄惧紡鏂规硶閫氬父浣跨敤 0.5 宸﹀彸鐨?CFL 鏁颁互纭繚绋冲畾鎬?
        self.config.cfl.max(0.5)
    }
}

// =============================================================================
// f32 鍚庣瀹炵幇
// =============================================================================

/// 璁＄畻 HLL 鏁板€奸€氶噺锛坒32 鐗堟湰锛?
#[inline]
fn compute_hll_flux_f32(
    h_l: f32, u_l: f32, v_l: f32,
    h_r: f32, u_r: f32, v_r: f32,
    normal: [f32; 2],
    gravity: f32,
    h_dry: f32,
) -> (f32, f32, f32, f32) {
    // 鎶曞奖鍒版硶鍚戠殑閫熷害鍒嗛噺
    let un_l = u_l * normal[0] + v_l * normal[1];
    let un_r = u_r * normal[0] + v_r * normal[1];
    
    // 娉㈤€熶及璁?
    let c_l = if h_l > h_dry { (gravity * h_l).sqrt() } else { 0.0 };
    let c_r = if h_r > h_dry { (gravity * h_r).sqrt() } else { 0.0 };
    
    // HLL 娉㈤€熻竟鐣?
    let s_l = (un_l - c_l).min(un_r - c_r).min(0.0);
    let s_r = (un_l + c_l).max(un_r + c_r).max(0.0);
    
    let max_speed = s_l.abs().max(s_r.abs());
    
    // 璁＄畻宸﹀彸閫氶噺
    let f_l_h = h_l * un_l;
    let f_l_hu = h_l * u_l * un_l + 0.5 * gravity * h_l * h_l * normal[0];
    let f_l_hv = h_l * v_l * un_l + 0.5 * gravity * h_l * h_l * normal[1];
    
    let f_r_h = h_r * un_r;
    let f_r_hu = h_r * u_r * un_r + 0.5 * gravity * h_r * h_r * normal[0];
    let f_r_hv = h_r * v_r * un_r + 0.5 * gravity * h_r * h_r * normal[1];
    
    // HLL 閫氶噺鍏紡
    let (f_h, f_hu, f_hv) = if s_l >= 0.0 {
        (f_l_h, f_l_hu, f_l_hv)
    } else if s_r <= 0.0 {
        (f_r_h, f_r_hu, f_r_hv)
    } else {
        let denom = s_r - s_l;
        if denom.abs() < 1e-7 {
            (0.0, 0.0, 0.0)
        } else {
            let f_h = (s_r * f_l_h - s_l * f_r_h + s_l * s_r * (h_r - h_l)) / denom;
            let f_hu = (s_r * f_l_hu - s_l * f_r_hu + s_l * s_r * (h_r * u_r - h_l * u_l)) / denom;
            let f_hv = (s_r * f_l_hv - s_l * f_r_hv + s_l * s_r * (h_r * v_r - h_l * v_l)) / denom;
            (f_h, f_hu, f_hv)
        }
    };
    
    (f_h, f_hu, f_hv, max_speed)
}

impl TimeIntegrationStrategy<CpuBackend<f32>> for ExplicitStrategy<CpuBackend<f32>> {
    fn name(&self) -> &'static str {
        "鏄惧紡 Godunov (HLL) [f32]"
    }
    
    fn step(
        &mut self,
        state: &mut ShallowWaterStateGeneric<CpuBackend<f32>>,
        mesh: &dyn MeshTopology<CpuBackend<f32>>,
        workspace: &mut SolverWorkspaceGeneric<CpuBackend<f32>>,
        dt: f32,
    ) -> StepResult<f32> {
        workspace.reset();
        
        let n_cells = mesh.n_cells();
        
        let h: &[f32] = &state.h;
        let hu: &[f32] = &state.hu;
        let hv: &[f32] = &state.hv;
        let z: &[f32] = &state.z;
        
        let flux_h: &mut [f32] = &mut workspace.flux_h;
        let flux_hu: &mut [f32] = &mut workspace.flux_hu;
        let flux_hv: &mut [f32] = &mut workspace.flux_hv;
        
        let h_dry = self.config.h_dry as f32;
        let gravity = self.config.gravity as f32;
        
        let mut max_wave_speed = 0.0f32;
        let mut dry_cells = 0usize;
        
        // 璁＄畻鍐呴儴闈㈤€氶噺
        for face in mesh.interior_faces() {
            let owner = mesh.face_owner(*face);
            let neighbor = mesh.face_neighbor(*face).unwrap();
            
            let normal_f64 = mesh.face_normal(*face);
            let normal = [normal_f64[0] as f32, normal_f64[1] as f32];
            let length = mesh.face_length(*face) as f32;
            
            let h_l = h[owner];
            let h_r = h[neighbor];
            let z_l = z[owner];
            let z_r = z[neighbor];
            
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
            
            // 闈欐按閲嶆瀯
            let eta_l = h_l + z_l;
            let eta_r = h_r + z_r;
            let z_star = z_l.max(z_r);
            
            let h_l_star = (eta_l - z_star).max(0.0);
            let h_r_star = (eta_r - z_star).max(0.0);
            
            let (f_h, f_hu, f_hv, wave_speed) = compute_hll_flux_f32(
                h_l_star, u_l, v_l,
                h_r_star, u_r, v_r,
                normal, gravity, h_dry,
            );
            
            max_wave_speed = max_wave_speed.max(wave_speed);
            
            let flux_mag_h = f_h * length;
            let flux_mag_hu = f_hu * length;
            let flux_mag_hv = f_hv * length;
            
            flux_h[owner] -= flux_mag_h;
            flux_h[neighbor] += flux_mag_h;
            flux_hu[owner] -= flux_mag_hu;
            flux_hu[neighbor] += flux_mag_hu;
            flux_hv[owner] -= flux_mag_hv;
            flux_hv[neighbor] += flux_mag_hv;
        }
        
        // 杈圭晫闈㈠鐞?
        for face in mesh.boundary_faces() {
            let owner = mesh.face_owner(*face);
            let normal_f64 = mesh.face_normal(*face);
            let normal = [normal_f64[0] as f32, normal_f64[1] as f32];
            let length = mesh.face_length(*face) as f32;
            
            let h_l = h[owner];
            
            if h_l <= h_dry {
                continue;
            }
            
            let u_l = hu[owner] / h_l;
            let v_l = hv[owner] / h_l;
            let un_l = u_l * normal[0] + v_l * normal[1];
            
            let f_hu = 0.5 * gravity * h_l * h_l * normal[0] * length;
            let f_hv = 0.5 * gravity * h_l * h_l * normal[1] * length;
            
            flux_hu[owner] -= f_hu;
            flux_hv[owner] -= f_hv;
            
            let c = (gravity * h_l).sqrt();
            max_wave_speed = max_wave_speed.max(un_l.abs() + c);
        }
        
        // 鏇存柊鐘舵€?
        let h_mut: &mut [f32] = &mut state.h;
        let hu_mut: &mut [f32] = &mut state.hu;
        let hv_mut: &mut [f32] = &mut state.hv;
        
        for i in 0..n_cells {
            let area = mesh.cell_area(i) as f32;
            if area <= 0.0 {
                continue;
            }
            let inv_area = 1.0 / area;
            
            h_mut[i] += dt * flux_h[i] * inv_area;
            hu_mut[i] += dt * flux_hu[i] * inv_area;
            hv_mut[i] += dt * flux_hv[i] * inv_area;
            
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
            converged: true,
            iterations: 0,
        }
    }
    
    fn compute_stable_dt(
        &self,
        state: &ShallowWaterStateGeneric<CpuBackend<f32>>,
        mesh: &dyn MeshTopology<CpuBackend<f32>>,
        cfl: f32,
    ) -> f32 {
        let h: &[f32] = &state.h;
        let hu: &[f32] = &state.hu;
        let hv: &[f32] = &state.hv;
        
        let h_dry = self.config.h_dry as f32;
        let gravity = self.config.gravity as f32;
        
        let mut dt_min = f32::MAX;
        
        for i in 0..mesh.n_cells() {
            if h[i] <= h_dry {
                continue;
            }
            
            let u = hu[i] / h[i];
            let v = hv[i] / h[i];
            let c = (gravity * h[i]).sqrt();
            let speed = (u * u + v * v).sqrt() + c;
            
            if speed > 1e-6 {
                let area = mesh.cell_area(i) as f32;
                let dx = area.sqrt();
                let dt_local = cfl * dx / speed;
                dt_min = dt_min.min(dt_local);
            }
        }
        
        if dt_min == f32::MAX {
            dt_min = 1e-6;
        }
        
        dt_min
    }
    
    fn recommended_cfl(&self) -> f32 {
        (self.config.cfl as f32).max(0.5)
    }
}
