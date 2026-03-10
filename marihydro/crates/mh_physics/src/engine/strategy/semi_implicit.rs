// marihydro\crates\mh_physics\src\engine\strategy\semi_implicit.rs
//! 鍗婇殣寮忔椂闂寸Н鍒嗙瓥鐣ワ紙娉涘瀷鐗堟湰锛?
//!
//! 鍩轰簬鍘嬪姏鏍℃鐨勫崐闅愬紡鏃堕棿鎺ㄨ繘绠楁硶銆?
//!
//! # 绠楁硶姒傝堪
//!
//! 鍗婇殣寮忔柟娉曞皢娴呮按鏂圭▼鍒嗕负鏄惧紡鍜岄殣寮忎袱閮ㄥ垎锛?
//! 1. **棰勬祴姝?*锛氭樉寮忚绠楀娴佸拰鎵╂暎椤癸紝寰楀埌棰勬祴閫熷害 u*, v*
//! 2. **鍘嬪姏鏍℃姝?*锛氶殣寮忔眰瑙ｅ帇鍔涙硦鏉炬柟绋?
//! 3. **鏍℃姝?*锛氱敤鍘嬪姏姊害鏍℃閫熷害鍜屾按浣?
//!
//! 杩欑鏂规硶鍏佽浣跨敤姣旀樉寮忔柟娉曟洿澶х殑 CFL 鏁帮紙閫氬父 2-10 鍊嶏級锛?
//! 鍥犱负閲嶅姏娉㈢殑浼犳挱鏄殣寮忓鐞嗙殑銆?
//!
//! # 浼樺娍
//!
//! - 瀵逛簬閲嶅姏娉富瀵肩殑娴佸姩锛屽彲浠ヤ娇鐢ㄦ洿澶х殑鏃堕棿姝ラ暱
//! - 鍦ㄤ綆 Froude 鏁版祦鍔ㄤ腑鐗瑰埆鏈夋晥
//! - 閫傚悎闀挎椂闂村昂搴︾殑妯℃嫙

use super::{SemiImplicitConfig, StepResult, TimeIntegrationStrategy};
use super::workspace::SolverWorkspaceGeneric;
use crate::core::{Backend, CpuBackend};
use crate::engine::pcg::{PcgSolver, PcgConfig, DiagonalMatrix, PreconditionerType};
use crate::mesh::MeshTopology;
use crate::state::ShallowWaterStateGeneric;

/// 娉涘瀷鍗婇殣寮忕瓥鐣?
/// 
/// 浣跨敤鍘嬪姏鏍℃娉曠殑鍗婇殣寮忔椂闂寸Н鍒嗙瓥鐣ャ€?
/// 鍐呴儴浣跨敤 PCG 姹傝В鍣ㄦ眰瑙ｅ帇鍔涙硦鏉炬柟绋嬨€?
/// 
/// # 绫诲瀷鍙傛暟
/// 
/// - `B`: 璁＄畻鍚庣绫诲瀷
pub struct SemiImplicitStrategyGeneric<B: Backend> {
    /// 璁＄畻鍚庣瀹炰緥
    backend: B,
    /// 閰嶇疆
    config: SemiImplicitConfig,
    /// PCG 姹傝В鍣?
    pcg_solver: Option<PcgSolver<B>>,
    /// 棰勬祴閫熷害 u*
    u_star: B::Buffer<B::Scalar>,
    /// 棰勬祴閫熷害 v*
    v_star: B::Buffer<B::Scalar>,
    /// 姘翠綅鏍℃閲?畏'
    eta_prime: B::Buffer<B::Scalar>,
    /// 鍙崇椤癸紙鏁ｅ害锛?
    rhs: B::Buffer<B::Scalar>,
    /// 瀵硅鐭╅樀锛堥澶勭悊鍣級
    diag: B::Buffer<B::Scalar>,
    /// 鍘嬪姏姊害 x 鍒嗛噺
    grad_eta_x: B::Buffer<B::Scalar>,
    /// 鍘嬪姏姊害 y 鍒嗛噺
    grad_eta_y: B::Buffer<B::Scalar>,
    /// 姹傝В鍣ㄥ凡鍒嗛厤鐨勫崟鍏冩暟
    n_cells_allocated: usize,
}

impl<B: Backend + Clone> SemiImplicitStrategyGeneric<B> {
    /// 浣跨敤鍚庣瀹炰緥鍒涘缓鍗婇殣寮忕瓥鐣?
    /// 
    /// # 鍙傛暟
    /// 
    /// - `backend`: 璁＄畻鍚庣瀹炰緥
    /// - `n_cells`: 鍗曞厓鏁伴噺
    /// - `config`: 鍗婇殣寮忕瓥鐣ラ厤缃?
    pub fn new_with_backend(backend: B, n_cells: usize, config: SemiImplicitConfig) -> Self {
        // 鍒涘缓 PCG 姹傝В鍣ㄩ厤缃?
        let pcg_config = PcgConfig {
            rtol: config.solver_rtol,
            atol: 1e-14,
            max_iter: config.solver_max_iter,
            preconditioner: PreconditionerType::Jacobi,
            verbose: false,
        };
        
        Self {
            u_star: backend.alloc(n_cells),
            v_star: backend.alloc(n_cells),
            eta_prime: backend.alloc(n_cells),
            rhs: backend.alloc(n_cells),
            diag: backend.alloc(n_cells),
            grad_eta_x: backend.alloc(n_cells),
            grad_eta_y: backend.alloc(n_cells),
            pcg_solver: Some(PcgSolver::new_with_backend(backend.clone(), n_cells, pcg_config)),
            backend,
            config,
            n_cells_allocated: n_cells,
        }
    }
    
    /// 鑾峰彇鍚庣寮曠敤
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 鑾峰彇閰嶇疆寮曠敤
    #[inline]
    pub fn config(&self) -> &SemiImplicitConfig {
        &self.config
    }
    
    /// 纭繚宸ヤ綔鍖哄ぇ灏忚冻澶?
    fn ensure_capacity(&mut self, n_cells: usize) {
        if n_cells > self.n_cells_allocated {
            self.u_star = self.backend.alloc(n_cells);
            self.v_star = self.backend.alloc(n_cells);
            self.eta_prime = self.backend.alloc(n_cells);
            self.rhs = self.backend.alloc(n_cells);
            self.diag = self.backend.alloc(n_cells);
            self.grad_eta_x = self.backend.alloc(n_cells);
            self.grad_eta_y = self.backend.alloc(n_cells);
            
            if let Some(ref mut solver) = self.pcg_solver {
                solver.ensure_capacity(n_cells);
            }
            
            self.n_cells_allocated = n_cells;
        }
    }
}


impl TimeIntegrationStrategy<CpuBackend<f64>> for SemiImplicitStrategyGeneric<CpuBackend<f64>> {
    fn name(&self) -> &'static str {
        "鍗婇殣寮忓帇鍔涙牎姝ｆ硶"
    }
    
    fn step(
        &mut self,
        state: &mut ShallowWaterStateGeneric<CpuBackend<f64>>,
        mesh: &dyn MeshTopology<CpuBackend<f64>>,
        _workspace: &mut SolverWorkspaceGeneric<CpuBackend<f64>>,
        dt: f64,
    ) -> StepResult<f64> {
        let n_cells = mesh.n_cells();
        self.ensure_capacity(n_cells);
        
        let gravity = self.config.gravity;
        let h_min = self.config.h_min;
        let theta = self.config.theta;
        
        // 鑾峰彇鐘舵€佸紩鐢紙鍙锛?
        let h: &[f64] = &state.h;
        let hu: &[f64] = &state.hu;
        let hv: &[f64] = &state.hv;
        let _z: &[f64] = &state.z;
        
        // 鑾峰彇宸ヤ綔缂撳啿鍖猴紙鍙啓锛?
        let u_star: &mut [f64] = &mut self.u_star;
        let v_star: &mut [f64] = &mut self.v_star;
        let eta_prime: &mut [f64] = &mut self.eta_prime;
        let rhs: &mut [f64] = &mut self.rhs;
        let diag: &mut [f64] = &mut self.diag;
        let grad_eta_x: &mut [f64] = &mut self.grad_eta_x;
        let grad_eta_y: &mut [f64] = &mut self.grad_eta_y;
        
        // ========== 绗?姝ワ細棰勬祴姝?==========
        // 璁＄畻棰勬祴閫熷害 u* = u^n + dt * (鏄惧紡椤?
        // 鏄惧紡椤瑰寘鎷細瀵规祦銆佹墿鏁ｃ€佸簥搴曞潯搴︺€佹懇鎿︾瓑
        // 杩欓噷浣跨敤绠€鍖栧疄鐜帮細鐩存帴浠庡綋鍓嶅姩閲忚绠楅€熷害
        for i in 0..n_cells {
            if h[i] > h_min {
                u_star[i] = hu[i] / h[i];
                v_star[i] = hv[i] / h[i];
            } else {
                u_star[i] = 0.0;
                v_star[i] = 0.0;
            }
        }
        
        // ========== 绗?姝ワ細缁勮鍘嬪姏娉婃澗鏂圭▼ ==========
        // 绂绘暎褰㈠紡锛欰 * 畏' = b
        // 鍏朵腑 A 鏄媺鏅媺鏂畻瀛愮殑绂绘暎鍖栵紝b 鏄€熷害鏁ｅ害
        //
        // 瀵逛簬绠€鍖栫殑瀵硅杩戜技锛?
        // A_ii 鈮?危_f (H_f * L_f / d_f)
        // 杩欓噷浣跨敤鏇寸畝鍗曠殑褰㈠紡锛欰_ii = Area_i / (g * 胃 * dt虏 * H_i)
        
        for i in 0..n_cells {
            let area = mesh.cell_area(i);
            let h_eff = h[i].max(h_min);
            
            // 瀵硅椤癸細鏉ヨ嚜鍘嬪姏娉婃澗鏂圭▼鐨勭鏁ｅ寲
            // 绯绘暟涓庢椂闂存闀裤€侀噸鍔涘拰姘存繁鐩稿叧
            diag[i] = area / (gravity * theta * dt * dt * h_eff);
        }
        
        // 璁＄畻鍙崇椤癸細棰勬祴閫熷害鐨勬暎搴?
        // b_i = -鈭埆 鈭嚶?H u*) dA 鈮?-危_f (H_f * u*_f 路 n_f) * L_f
        rhs.fill(0.0);
        for face in mesh.interior_faces() {
            let owner = mesh.face_owner(*face);
            let neighbor = mesh.face_neighbor(*face).unwrap();
            
            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);
            
            // 鐣岄潰澶勭殑姘存繁锛堢畻鏈钩鍧囷級
            let h_face = 0.5 * (h[owner] + h[neighbor]).max(h_min);
            
            // 鐣岄潰澶勭殑棰勬祴閫熷害锛堢畻鏈钩鍧囷級
            let u_face = 0.5 * (u_star[owner] + u_star[neighbor]);
            let v_face = 0.5 * (v_star[owner] + v_star[neighbor]);
            
            // 閫氳繃鐣岄潰鐨勪綋绉€氶噺
            let flux = h_face * (u_face * normal[0] + v_face * normal[1]) * length;
            
            // 绱姞鍒扮浉閭诲崟鍏冿紙瀹堟亽褰㈠紡锛?
            rhs[owner] -= flux;
            rhs[neighbor] += flux;
        }
        
        // ========== 绗?姝ワ細姹傝В鍘嬪姏鏍℃鏂圭▼ ==========
        // 浣跨敤 PCG 姹傝В鍣ㄦ垨绠€鍗曠殑 Jacobi 杩唬
        eta_prime.fill(0.0);
        let mut converged = true;
        let mut iterations = 0;
        
        if let Some(ref mut pcg_solver) = self.pcg_solver {
            // 浣跨敤 PCG 姹傝В鍣?
            let diag_matrix = DiagonalMatrix::new(diag.to_vec(), n_cells);
            let mut eta_vec = eta_prime.to_vec();
            let rhs_vec = rhs.to_vec();
            
            let result = pcg_solver.solve(&diag_matrix, &mut eta_vec, &rhs_vec, Some(&diag_matrix));
            
            converged = result.converged;
            iterations = result.iterations;
            
            // 澶嶅埗缁撴灉鍥炵紦鍐插尯
            for i in 0..n_cells {
                eta_prime[i] = eta_vec[i];
            }
        } else {
            // 鍥為€€鍒扮畝鍗曠殑 Jacobi 杩唬
            for iter in 0..self.config.solver_max_iter {
                let mut max_residual = 0.0f64;
                
                for i in 0..n_cells {
                    if diag[i].abs() > 1e-14 {
                        let new_eta = rhs[i] / diag[i];
                        let residual = (new_eta - eta_prime[i]).abs();
                        max_residual = max_residual.max(residual);
                        eta_prime[i] = new_eta;
                    }
                }
                
                iterations = iter + 1;
                if max_residual < self.config.solver_rtol {
                    break;
                }
                
                if iter == self.config.solver_max_iter - 1 {
                    converged = false;
                }
            }
        }
        
        // ========== 绗?姝ワ細璁＄畻鍘嬪姏姊害 ==========
        // 鈭囄? 閫氳繃 Green-Gauss 鍏紡璁＄畻
        grad_eta_x.fill(0.0);
        grad_eta_y.fill(0.0);
        
        for face in mesh.interior_faces() {
            let owner = mesh.face_owner(*face);
            let neighbor = mesh.face_neighbor(*face).unwrap();
            
            let normal = mesh.face_normal(*face);
            let length = mesh.face_length(*face);
            
            // 鐣岄潰澶勭殑姘翠綅鏍℃锛堢畻鏈钩鍧囷級
            let eta_face = 0.5 * (eta_prime[owner] + eta_prime[neighbor]);
            
            // 姊害璐＄尞锛圙reen-Gauss 瀹氱悊锛?
            let contrib_x = eta_face * normal[0] * length;
            let contrib_y = eta_face * normal[1] * length;
            
            grad_eta_x[owner] += contrib_x;
            grad_eta_x[neighbor] -= contrib_x;
            grad_eta_y[owner] += contrib_y;
            grad_eta_y[neighbor] -= contrib_y;
        }
        
        // 闄や互鍗曞厓闈㈢Н寰楀埌姊害
        for i in 0..n_cells {
            let area = mesh.cell_area(i);
            if area > 1e-14 {
                let inv_area = 1.0 / area;
                grad_eta_x[i] *= inv_area;
                grad_eta_y[i] *= inv_area;
            }
        }
        
        // ========== 绗?姝ワ細鏍℃閫熷害鍜屾按浣?==========
        // u^{n+1} = u* - g * 胃 * dt * 鈭囄?
        // 畏^{n+1} = 畏^n + 畏'
        let h_mut: &mut [f64] = &mut state.h;
        let hu_mut: &mut [f64] = &mut state.hu;
        let hv_mut: &mut [f64] = &mut state.hv;
        
        let mut max_wave_speed = 0.0f64;
        let mut dry_cells = 0usize;
        
        for i in 0..n_cells {
            // 鏇存柊姘翠綅
            h_mut[i] += eta_prime[i];
            
            if h_mut[i] < h_min {
                // 骞插崟鍏冨鐞?
                h_mut[i] = 0.0;
                hu_mut[i] = 0.0;
                hv_mut[i] = 0.0;
                dry_cells += 1;
            } else {
                // 閫熷害鏍℃
                let u_new = u_star[i] - gravity * theta * dt * grad_eta_x[i];
                let v_new = v_star[i] - gravity * theta * dt * grad_eta_y[i];
                
                // 鏇存柊鍔ㄩ噺
                hu_mut[i] = h_mut[i] * u_new;
                hv_mut[i] = h_mut[i] * v_new;
                
                // 璁＄畻鏈€澶ф尝閫?
                let c = (gravity * h_mut[i]).sqrt();
                let speed = (u_new * u_new + v_new * v_new).sqrt() + c;
                max_wave_speed = max_wave_speed.max(speed);
            }
        }
        
        StepResult {
            dt_used: dt,
            max_wave_speed,
            dry_cells,
            limited_cells: 0,
            converged,
            iterations,
        }
    }
    
    /// 璁＄畻绋冲畾鏃堕棿姝ラ暱
    /// 
    /// 鍗婇殣寮忔柟娉曞彲浠ヤ娇鐢ㄦ瘮鏄惧紡鏂规硶鏇村ぇ鐨?CFL 鏁帮紝
    /// 鍥犱负閲嶅姏娉㈡槸闅愬紡澶勭悊鐨勩€?
    fn compute_stable_dt(
        &self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        mesh: &dyn MeshTopology<CpuBackend<f64>>,
        cfl: f64,
    ) -> f64 {
        let h: &[f64] = &state.h;
        let hu: &[f64] = &state.hu;
        let hv: &[f64] = &state.hv;
        
        let h_min = self.config.h_min;
        let gravity = self.config.gravity;
        
        let mut dt_min = f64::MAX;
        
        for i in 0..mesh.n_cells() {
            // 璺宠繃骞插崟鍏?
            if h[i] <= h_min {
                continue;
            }
            
            let u = hu[i] / h[i];
            let v = hv[i] / h[i];
            let c = (gravity * h[i]).sqrt();
            
            // 瀵逛簬鍗婇殣寮忔柟娉曪紝鏃堕棿姝ラ暱涓昏鍙楀娴侀€熷害闄愬埗
            // 閲嶅姏娉㈤€熷害鐨勫奖鍝嶈緝灏?
            let speed = (u * u + v * v).sqrt() + c;
            
            if speed > 1e-10 {
                let area = mesh.cell_area(i);
                let dx = area.sqrt();
                let dt_local = cfl * dx / speed;
                dt_min = dt_min.min(dt_local);
            }
        }
        
        if dt_min == f64::MAX {
            dt_min = 1e-6;
        }
        
        // 鍗婇殣寮忔柟娉曞厑璁告洿澶х殑 CFL 鏁帮紙閫氬父鍙互鏄樉寮忕殑 2-5 鍊嶏級
        dt_min * 2.0
    }
    
    /// 鍗婇殣寮忔柟娉曟敮鎸佸ぇ CFL 鏁?
    fn supports_large_cfl(&self) -> bool {
        true
    }
    
    /// 鎺ㄨ崘鐨?CFL 鏁?
    fn recommended_cfl(&self) -> f64 {
        // 鍗婇殣寮忔柟娉曟帹鑽愪娇鐢?CFL 鈮?2.0
        2.0
    }
}
