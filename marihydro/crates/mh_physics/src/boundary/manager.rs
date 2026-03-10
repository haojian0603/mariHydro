// crates/mh_physics/src/boundary/manager.rs

//! 杈圭晫鏉′欢绠＄悊鍣?
//!
//! 鏈ā鍧楁彁渚涜竟鐣屾潯浠剁殑绠＄悊鍜岃竟鐣岄€氶噺璁＄畻鍔熻兘锛?
//! - BoundaryManager: 杈圭晫鏉′欢绠＄悊鍣?
//! - BoundaryFaceInfo: 杈圭晫闈俊鎭?
//! - BoundaryDataProvider: 杈圭晫鏁版嵁鎻愪緵鑰呮帴鍙?
//!
//! # 璁捐鎬濊矾
//!
//! 1. 杈圭晫鏉′欢涓庤竟鐣岄潰鍒嗙锛氭潯浠舵槸瀹氫箟锛岄潰鏄嚑浣曞疄浣?
//! 2. 閫氳繃鍚嶇О鍏宠仈鏉′欢鍜岄潰
//! 3. 鎸夎竟鐣岀被鍨嬪垎绫诲瓨鍌ㄩ潰淇℃伅锛屼究浜庢壒閲忓鐞?
//! 4. 鏀寔涓庡閮ㄥ己杩暟鎹簮闆嗘垚
//!
//! # 杩佺Щ璇存槑
//!
//! 浠?history_src/domain/boundary/manager.rs 杩佺Щ锛岄€傞厤鏂版灦鏋勶細
//! - 浣跨敤 usize 绱㈠紩锛堜笌 MeshAccess trait 涓€鑷达級
//! - 浣跨敤 glam::DVec2 琛ㄧず娉曞悜閲?
//! - 浣跨敤 thiserror 瀹氫箟閿欒绫诲瀷

use std::collections::HashMap;

use glam::DVec2;
use thiserror::Error;

use super::types::{BoundaryCondition, BoundaryKind, BoundaryParams, ExternalForcing};
use crate::state::ConservedState;

// ============================================================
// 杈圭晫闈俊鎭?
// ============================================================

/// 杈圭晫闈俊鎭?
///
/// 鎻忚堪鍗曚釜杈圭晫闈㈢殑鍑犱綍鍜屾嫇鎵戜俊鎭€?
#[derive(Debug, Clone, Copy)]
pub struct BoundaryFaceInfo {
    /// 闈㈢储寮曪紙鍦ㄧ綉鏍间腑鐨勭储寮曪級
    pub face_id: usize,

    /// 鎵€灞炲崟鍏冪储寮?
    pub cell_id: usize,

    /// 闈㈠娉曞悜閲忥紙鍗曚綅鍚戦噺锛?
    pub normal: DVec2,

    /// 闈㈤暱搴?[m]
    pub length: f64,

    /// 鎵€灞炶竟鐣屾潯浠剁殑绱㈠紩
    pub boundary_idx: usize,
}

impl BoundaryFaceInfo {
    /// 鍒涘缓鏂扮殑杈圭晫闈俊鎭?
    pub fn new(
        face_id: usize,
        cell_id: usize,
        normal: DVec2,
        length: f64,
        boundary_idx: usize,
    ) -> Self {
        Self {
            face_id,
            cell_id,
            normal,
            length,
            boundary_idx,
        }
    }
}

// ============================================================
// 杈圭晫鏁版嵁鎻愪緵鑰呮帴鍙?
// ============================================================

/// 杈圭晫鏁版嵁鎻愪緵鑰呮帴鍙?
///
/// 鐢ㄤ簬浠庡閮ㄦ暟鎹簮鑾峰彇杈圭晫寮鸿揩鏁版嵁锛堟按浣嶃€佹祦閫熺瓑锛夈€?
pub trait BoundaryDataProvider: Send + Sync {
    /// 鑾峰彇鎸囧畾闈㈠湪缁欏畾鏃堕棿鐨勫己杩暟鎹?
    ///
    /// # 鍙傛暟
    /// - `face_id`: 杈圭晫闈㈢储寮?
    /// - `time`: 妯℃嫙鏃堕棿 [s]
    ///
    /// # 杩斿洖
    /// 寮鸿揩鏁版嵁锛岃嫢鏃犳暟鎹繑鍥?None
    fn get_forcing(&self, face_id: usize, time: f64) -> Option<ExternalForcing>;

    /// 鎵归噺鑾峰彇寮鸿揩鏁版嵁
    ///
    /// 榛樿瀹炵幇閫愪釜璋冪敤 `get_forcing`锛屽彲閲嶅啓浠ヤ紭鍖栨€ц兘銆?
    fn get_forcings_batch(
        &self,
        face_ids: &[usize],
        time: f64,
        output: &mut [ExternalForcing],
    ) {
        debug_assert_eq!(face_ids.len(), output.len());
        for (i, &face_id) in face_ids.iter().enumerate() {
            output[i] = self.get_forcing(face_id, time).unwrap_or(ExternalForcing::ZERO);
        }
    }
}

/// 鎭掑畾寮鸿揩鏁版嵁鎻愪緵鑰?
///
/// 鐢ㄤ簬娴嬭瘯鍜岀畝鍗曞満鏅紝杩斿洖鎭掑畾鐨勫己杩暟鎹€?
pub struct ConstantForcingProvider {
    forcing: ExternalForcing,
}

impl ConstantForcingProvider {
    /// 鍒涘缓鎭掑畾寮鸿揩鎻愪緵鑰?
    pub fn new(forcing: ExternalForcing) -> Self {
        Self { forcing }
    }

    /// 鍒涘缓浠呮按浣嶇殑鎭掑畾鎻愪緵鑰?
    pub fn with_eta(eta: f64) -> Self {
        Self::new(ExternalForcing::with_eta(eta))
    }
}

impl BoundaryDataProvider for ConstantForcingProvider {
    fn get_forcing(&self, _face_id: usize, _time: f64) -> Option<ExternalForcing> {
        Some(self.forcing)
    }

    fn get_forcings_batch(
        &self,
        face_ids: &[usize],
        _time: f64,
        output: &mut [ExternalForcing],
    ) {
        output[..face_ids.len()].fill(self.forcing);
    }
}

// ============================================================
// 杈圭晫鏉′欢绠＄悊鍣?
// ============================================================

/// 杈圭晫鏉′欢绠＄悊鍣?
///
/// 璐熻矗锛?
/// 1. 绠＄悊杈圭晫鏉′欢瀹氫箟
/// 2. 娉ㄥ唽杈圭晫闈笌鏉′欢鐨勬槧灏?
/// 3. 鎻愪緵杈圭晫閫氶噺璁＄畻鏂规硶
///
/// # 浣跨敤娴佺▼
///
/// 1. 鍒涘缓绠＄悊鍣?
/// 2. 娣诲姞杈圭晫鏉′欢瀹氫箟 (add_condition)
/// 3. 娉ㄥ唽杈圭晫闈?(register_face)
/// 4. 璁＄畻杈圭晫閫氶噺 (compute_*_flux)
///
/// # 绀轰緥
///
/// ```ignore
/// use mh_physics::boundary::{BoundaryManager, BoundaryCondition, BoundaryParams};
/// use glam::DVec2;
///
/// let mut manager = BoundaryManager::new(BoundaryParams::default());
///
/// // 娣诲姞杈圭晫鏉′欢
/// manager.add_condition(BoundaryCondition::wall("north"));
/// manager.add_condition(BoundaryCondition::open_sea("south"));
///
/// // 娉ㄥ唽杈圭晫闈?
/// manager.register_face(0, 0, DVec2::new(0.0, 1.0), 1.0, "north").unwrap();
/// manager.register_face(1, 1, DVec2::new(0.0, -1.0), 1.0, "south").unwrap();
/// ```
pub struct BoundaryManager {
    /// 杈圭晫鏉′欢瀹氫箟锛堟寜鍚嶇О绱㈠紩锛?
    conditions: HashMap<String, BoundaryCondition>,

    /// 鏉′欢鍚嶇О鍒扮储寮曠殑鏄犲皠
    condition_indices: HashMap<String, usize>,

    /// 鏉′欢鍒楄〃锛堢敤浜庣储寮曡闂級
    condition_list: Vec<BoundaryCondition>,

    /// 鍥哄杈圭晫闈?
    wall_faces: Vec<BoundaryFaceInfo>,

    /// 寮€杈圭晫闈紙Flather锛?
    open_faces: Vec<BoundaryFaceInfo>,

    /// 鍏ユ祦杈圭晫闈?
    inflow_faces: Vec<BoundaryFaceInfo>,

    /// 鍑烘祦杈圭晫闈?
    outflow_faces: Vec<BoundaryFaceInfo>,

    /// 鍛ㄦ湡杈圭晫闈?
    periodic_faces: Vec<BoundaryFaceInfo>,

    /// 璁＄畻鍙傛暟
    params: BoundaryParams,
}

impl BoundaryManager {
    /// 鍒涘缓鏂扮殑杈圭晫绠＄悊鍣?
    pub fn new(params: BoundaryParams) -> Self {
        Self {
            conditions: HashMap::new(),
            condition_indices: HashMap::new(),
            condition_list: Vec::new(),
            wall_faces: Vec::new(),
            open_faces: Vec::new(),
            inflow_faces: Vec::new(),
            outflow_faces: Vec::new(),
            periodic_faces: Vec::new(),
            params,
        }
    }

    /// 浠庢暟鍊煎弬鏁板垱寤?
    pub fn from_numerical_params(params: &crate::types::NumericalParams) -> Self {
        Self::new(BoundaryParams::from_numerical_params(params))
    }

    /// 娣诲姞杈圭晫鏉′欢瀹氫箟
    ///
    /// # 鍙傛暟
    /// - `condition`: 杈圭晫鏉′欢閰嶇疆
    ///
    /// # 杩斿洖
    /// 鏉′欢鍦ㄥ垪琛ㄤ腑鐨勭储寮?
    pub fn add_condition(&mut self, condition: BoundaryCondition) -> usize {
        let idx = self.condition_list.len();
        self.condition_indices.insert(condition.name.clone(), idx);
        self.conditions
            .insert(condition.name.clone(), condition.clone());
        self.condition_list.push(condition);
        idx
    }

    /// 鑾峰彇杈圭晫鏉′欢
    pub fn get_condition(&self, name: &str) -> Option<&BoundaryCondition> {
        self.conditions.get(name)
    }

    /// 鑾峰彇杈圭晫鏉′欢锛堟寜绱㈠紩锛?
    pub fn get_condition_by_index(&self, idx: usize) -> Option<&BoundaryCondition> {
        self.condition_list.get(idx)
    }

    /// 娉ㄥ唽杈圭晫闈?
    ///
    /// # 鍙傛暟
    /// - `face_id`: 闈㈢储寮?
    /// - `cell_id`: 鎵€灞炲崟鍏冪储寮?
    /// - `normal`: 闈㈠娉曞悜閲忥紙搴斾负鍗曚綅鍚戦噺锛?
    /// - `length`: 闈㈤暱搴?[m]
    /// - `boundary_name`: 杈圭晫鏉′欢鍚嶇О
    ///
    /// # 閿欒
    /// - 杈圭晫鏉′欢鏈壘鍒?
    pub fn register_face(
        &mut self,
        face_id: usize,
        cell_id: usize,
        normal: DVec2,
        length: f64,
        boundary_name: &str,
    ) -> Result<(), BoundaryError> {
        let boundary_idx = *self
            .condition_indices
            .get(boundary_name)
            .ok_or_else(|| BoundaryError::ConditionNotFound(boundary_name.to_string()))?;

        let condition = &self.condition_list[boundary_idx];
        let info = BoundaryFaceInfo::new(face_id, cell_id, normal, length, boundary_idx);

        match condition.kind {
            BoundaryKind::Wall | BoundaryKind::Symmetry => self.wall_faces.push(info),
            BoundaryKind::OpenSea => self.open_faces.push(info),
            BoundaryKind::RiverInflow => self.inflow_faces.push(info),
            BoundaryKind::Outflow => self.outflow_faces.push(info),
            BoundaryKind::Periodic => self.periodic_faces.push(info),
        }

        Ok(())
    }

    // ========== 閫氶噺璁＄畻鏂规硶 ==========

    /// 璁＄畻鍥哄杈圭晫閫氶噺
    ///
    /// 鍥哄杈圭晫锛氭棤绌块€忥紝鍙湁鍘嬪姏浣滅敤
    ///
    /// # 鍙傛暟
    /// - `h_interior`: 鍐呴儴鍗曞厓姘存繁 [m]
    /// - `normal`: 闈㈠娉曞悜閲?
    ///
    /// # 杩斿洖
    /// (璐ㄩ噺閫氶噺, 鍔ㄩ噺閫氶噺鍚戦噺)
    pub fn compute_wall_flux(&self, h_interior: f64, normal: DVec2) -> (f64, DVec2) {
        // 璐ㄩ噺閫氶噺涓洪浂锛堟棤绌块€忥級
        let mass_flux = 0.0;

        // 鍔ㄩ噺閫氶噺浠呮湁鍘嬪姏椤?
        let p = self.params.hydrostatic_pressure(h_interior);
        let momentum_flux = normal * p;

        (mass_flux, momentum_flux)
    }

    /// 璁＄畻 Flather 杈愬皠杈圭晫閫氶噺
    ///
    /// Flather 杈圭晫鏉′欢浣跨敤鐗瑰緛鍏崇郴缁撳悎澶栭儴寮鸿揩銆?
    /// 閫傜敤浜庡紑娴疯竟鐣岋紝鍏佽娉㈠姩鑷敱浼犲嚭銆?
    ///
    /// # 鍙傛暟
    /// - `interior`: 鍐呴儴鍗曞厓鐘舵€?
    /// - `z_interior`: 鍐呴儴鍗曞厓搴曢珮绋?[m]
    /// - `external`: 澶栭儴寮鸿揩鏁版嵁
    /// - `normal`: 闈㈠娉曞悜閲?
    ///
    /// # 杩斿洖
    /// (璐ㄩ噺閫氶噺, 鍔ㄩ噺閫氶噺鍚戦噺)
    pub fn compute_flather_flux(
        &self,
        interior: ConservedState,
        z_interior: f64,
        external: &ExternalForcing,
        normal: DVec2,
    ) -> (f64, DVec2) {
        let h = interior.h.max(self.params.h_min);
        let c = self.params.wave_speed(h);

        // 鍐呴儴閫熷害
        let u = interior.hu / h;
        let v = interior.hv / h;
        let velocity = DVec2::new(u, v);

        // 娉曞悜閫熷害
        let un_int = velocity.dot(normal);
        let un_ext = external.velocity.dot(normal);

        // 鍐呴儴姘翠綅
        let eta_int = h + z_interior;

        // Flather 鏉′欢: un* = un_ext + (c/h)(eta_int - eta_ext)
        let un_star = un_ext + (c / h) * (eta_int - external.eta);

        // 閫氶噺璁＄畻
        let mass_flux = h * un_star;
        let p = self.params.hydrostatic_pressure(h);
        let momentum_flux = normal * (mass_flux * un_int + p);

        (mass_flux, momentum_flux)
    }

    /// 璁＄畻鑷敱鍑烘祦閫氶噺
    ///
    /// 闆舵搴﹀鎺紝鐩存帴浣跨敤鍐呴儴鐘舵€佽绠楅€氶噺銆?
    ///
    /// # 鍙傛暟
    /// - `interior`: 鍐呴儴鍗曞厓鐘舵€?
    /// - `normal`: 闈㈠娉曞悜閲?
    ///
    /// # 杩斿洖
    /// (璐ㄩ噺閫氶噺, 鍔ㄩ噺閫氶噺鍚戦噺)
    pub fn compute_outflow_flux(&self, interior: ConservedState, normal: DVec2) -> (f64, DVec2) {
        let h = interior.h.max(self.params.h_min);
        let u = interior.hu / h;
        let v = interior.hv / h;
        let velocity = DVec2::new(u, v);

        let un = velocity.dot(normal);
        let mass_flux = h * un;
        let p = self.params.hydrostatic_pressure(h);
        let momentum_flux = normal * (mass_flux * un + p);

        (mass_flux, momentum_flux)
    }

    /// 璁＄畻鍏ユ祦杈圭晫閫氶噺
    ///
    /// 鏍规嵁缁欏畾鐨勬祦閲忔垨閫熷害璁＄畻閫氶噺銆?
    ///
    /// # 鍙傛暟
    /// - `h_interior`: 鍐呴儴鍗曞厓姘存繁 [m]
    /// - `discharge`: 鍏ユ祦娴侀噺 [m鲁/s]
    /// - `face_length`: 闈㈤暱搴?[m]
    /// - `normal`: 闈㈠娉曞悜閲?
    ///
    /// # 杩斿洖
    /// (璐ㄩ噺閫氶噺, 鍔ㄩ噺閫氶噺鍚戦噺)
    pub fn compute_inflow_flux(
        &self,
        h_interior: f64,
        discharge: f64,
        face_length: f64,
        normal: DVec2,
    ) -> (f64, DVec2) {
        // 鍏ユ祦娴侀噺锛堣礋鍙峰洜涓哄叆娴佹柟鍚戜笌娉曞悜鐩稿弽锛?
        let qn = -discharge / face_length.max(1e-10);

        let p = self.params.hydrostatic_pressure(h_interior);
        let u_in = qn / h_interior.max(self.params.h_min);
        let momentum_flux = normal * (qn * u_in + p);

        (qn, momentum_flux)
    }

    // ========== 璁块棶鏂规硶 ==========

    /// 鑾峰彇鍥哄杈圭晫闈㈠垪琛?
    pub fn wall_faces(&self) -> &[BoundaryFaceInfo] {
        &self.wall_faces
    }

    /// 鑾峰彇寮€杈圭晫闈㈠垪琛?
    pub fn open_faces(&self) -> &[BoundaryFaceInfo] {
        &self.open_faces
    }

    /// 鑾峰彇鍏ユ祦杈圭晫闈㈠垪琛?
    pub fn inflow_faces(&self) -> &[BoundaryFaceInfo] {
        &self.inflow_faces
    }

    /// 鑾峰彇鍑烘祦杈圭晫闈㈠垪琛?
    pub fn outflow_faces(&self) -> &[BoundaryFaceInfo] {
        &self.outflow_faces
    }

    /// 鑾峰彇鍛ㄦ湡杈圭晫闈㈠垪琛?
    pub fn periodic_faces(&self) -> &[BoundaryFaceInfo] {
        &self.periodic_faces
    }

    /// 鑾峰彇鎵€鏈夎竟鐣岄潰鐨勮凯浠ｅ櫒
    pub fn all_faces(&self) -> impl Iterator<Item = &BoundaryFaceInfo> {
        self.wall_faces
            .iter()
            .chain(&self.open_faces)
            .chain(&self.inflow_faces)
            .chain(&self.outflow_faces)
            .chain(&self.periodic_faces)
    }

    /// 鑾峰彇杈圭晫闈㈡€绘暟
    pub fn total_boundary_faces(&self) -> usize {
        self.wall_faces.len()
            + self.open_faces.len()
            + self.inflow_faces.len()
            + self.outflow_faces.len()
            + self.periodic_faces.len()
    }

    /// 鑾峰彇杈圭晫鏉′欢鏁伴噺
    pub fn condition_count(&self) -> usize {
        self.condition_list.len()
    }

    /// 鑾峰彇璁＄畻鍙傛暟
    pub fn params(&self) -> &BoundaryParams {
        &self.params
    }

    /// 娓呯┖鎵€鏈夋敞鍐岀殑杈圭晫闈?
    pub fn clear_faces(&mut self) {
        self.wall_faces.clear();
        self.open_faces.clear();
        self.inflow_faces.clear();
        self.outflow_faces.clear();
        self.periodic_faces.clear();
    }

    /// 楠岃瘉杈圭晫鏉′欢璁剧疆
    ///
    /// 妫€鏌ワ細
    /// 1. 娉曞悜閲忔槸鍚﹀崟浣嶅寲
    /// 2. 鏄惁鏈夐噸澶嶇殑杈圭晫闈?
    pub fn validate(&self) -> Result<(), BoundaryError> {
        use std::collections::HashSet;
        let mut seen_faces = HashSet::new();

        for face in self.all_faces() {
            // 妫€鏌ユ硶鍚戦噺鏄惁鍗曚綅鍖?
            let mag_sq = face.normal.length_squared();
            if (mag_sq - 1.0).abs() > 1e-6 {
                return Err(BoundaryError::InvalidNormal {
                    face_id: face.face_id,
                    magnitude: mag_sq.sqrt(),
                });
            }

            // 妫€鏌ユ槸鍚﹂噸澶?
            if !seen_faces.insert(face.face_id) {
                return Err(BoundaryError::DuplicateFace(face.face_id));
            }
        }

        Ok(())
    }
}

impl Default for BoundaryManager {
    fn default() -> Self {
        Self::new(BoundaryParams::default())
    }
}

// ============================================================
// 閿欒绫诲瀷
// ============================================================

/// 边界模块错误类型
#[derive(Debug, Error)]
pub enum BoundaryError {
    /// 边界条件未找到
    #[error("边界条件 '{0}' 未找到")]
    ConditionNotFound(String),

    /// 边界面法向量未单位化
    #[error("边界面 {face_id} 法向量未单位化，模长为 {magnitude}")]
    InvalidNormal { face_id: usize, magnitude: f64 },

    /// 重复的边界面
    #[error("重复的边界面: {0}")]
    DuplicateFace(usize),

    /// 边界配置错误
    #[error("边界配置错误: {0}")]
    Configuration(String),
}

// ============================================================
// 娴嬭瘯
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_manager_creation() {
        let manager = BoundaryManager::default();
        assert_eq!(manager.total_boundary_faces(), 0);
        assert_eq!(manager.condition_count(), 0);
    }

    #[test]
    fn test_add_condition() {
        let mut manager = BoundaryManager::default();

        let idx1 = manager.add_condition(BoundaryCondition::wall("north"));
        let idx2 = manager.add_condition(BoundaryCondition::open_sea("south"));

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(manager.condition_count(), 2);

        assert!(manager.get_condition("north").is_some());
        assert!(manager.get_condition("south").is_some());
        assert!(manager.get_condition("east").is_none());
    }

    #[test]
    fn test_register_face() {
        let mut manager = BoundaryManager::default();
        manager.add_condition(BoundaryCondition::wall("north"));
        manager.add_condition(BoundaryCondition::open_sea("south"));

        manager
            .register_face(0, 0, DVec2::new(0.0, 1.0), 1.0, "north")
            .unwrap();
        manager
            .register_face(1, 1, DVec2::new(0.0, -1.0), 1.0, "south")
            .unwrap();

        assert_eq!(manager.wall_faces().len(), 1);
        assert_eq!(manager.open_faces().len(), 1);
        assert_eq!(manager.total_boundary_faces(), 2);
    }

    #[test]
    fn test_register_unknown_condition() {
        let mut manager = BoundaryManager::default();
        let result = manager.register_face(0, 0, DVec2::new(0.0, 1.0), 1.0, "unknown");

        assert!(result.is_err());
        if let Err(BoundaryError::ConditionNotFound(name)) = result {
            assert_eq!(name, "unknown");
        } else {
            panic!("Expected ConditionNotFound error");
        }
    }

    #[test]
    fn test_wall_flux() {
        let manager = BoundaryManager::default();
        let (mass, momentum) = manager.compute_wall_flux(1.0, DVec2::new(1.0, 0.0));

        assert_eq!(mass, 0.0);
        assert!(momentum.x > 0.0); // 鍘嬪姏鍚戝
        assert!((momentum.y).abs() < 1e-10);
    }

    #[test]
    fn test_outflow_flux() {
        let manager = BoundaryManager::default();
        let interior = ConservedState::from_primitive(1.0, 1.0, 0.0);
        let (mass, _) = manager.compute_outflow_flux(interior, DVec2::new(1.0, 0.0));

        assert!((mass - 1.0).abs() < 1e-10); // h * u * normal = 1 * 1 * 1
    }

    #[test]
    fn test_validate() {
        let mut manager = BoundaryManager::default();
        manager.add_condition(BoundaryCondition::wall("test"));

        // 鍗曚綅鍚戦噺搴旇閫氳繃
        manager
            .register_face(0, 0, DVec2::new(1.0, 0.0), 1.0, "test")
            .unwrap();
        assert!(manager.validate().is_ok());

        // 娓呯┖骞舵坊鍔犻潪鍗曚綅鍚戦噺
        manager.clear_faces();
        manager
            .register_face(1, 0, DVec2::new(2.0, 0.0), 1.0, "test")
            .unwrap();
        assert!(manager.validate().is_err());
    }

    #[test]
    fn test_constant_forcing_provider() {
        let forcing = ExternalForcing::new(1.5, 0.5, 0.0);
        let provider = ConstantForcingProvider::new(forcing);

        let result = provider.get_forcing(0, 0.0).unwrap();
        assert!((result.eta - 1.5).abs() < 1e-10);

        let mut output = vec![ExternalForcing::ZERO; 3];
        provider.get_forcings_batch(&[0, 1, 2], 0.0, &mut output);
        for f in &output {
            assert!((f.eta - 1.5).abs() < 1e-10);
        }
    }
}
