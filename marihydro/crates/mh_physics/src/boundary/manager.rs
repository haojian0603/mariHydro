// crates/mh_physics/src/boundary/manager.rs

//! 边界条件管理器
//!
//! 本模块提供边界条件的管理和边界通量计算功能：
//! - BoundaryManager: 边界条件管理器
//! - BoundaryFaceInfo: 边界面信息
//! - BoundaryDataProvider: 边界数据提供者接口
//!
//! # 设计思路
//!
//! 1. 边界条件与边界面分离：条件是定义，面是几何实体
//! 2. 通过名称关联条件和面
//! 3. 按边界类型分类存储面信息，便于批量处理
//! 4. 支持与外部强迫数据源集成
//!
//! # 迁移说明
//!
//! 从 legacy_src/domain/boundary/manager.rs 迁移，适配新架构：
//! - 使用 usize 索引（与 MeshAccess trait 一致）
//! - 使用 glam::DVec2 表示法向量
//! - 使用 thiserror 定义错误类型

use std::collections::HashMap;

use glam::DVec2;
use thiserror::Error;

use super::types::{BoundaryCondition, BoundaryKind, BoundaryParams, ExternalForcing};
use crate::state::ConservedState;

// ============================================================
// 边界面信息
// ============================================================

/// 边界面信息
///
/// 描述单个边界面的几何和拓扑信息。
#[derive(Debug, Clone, Copy)]
pub struct BoundaryFaceInfo {
    /// 面索引（在网格中的索引）
    pub face_id: usize,

    /// 所属单元索引
    pub cell_id: usize,

    /// 面外法向量（单位向量）
    pub normal: DVec2,

    /// 面长度 [m]
    pub length: f64, // ALLOW_F64: 来自 PhysicsMesh 的几何数据

    /// 所属边界条件的索引
    pub boundary_idx: usize,
}

impl BoundaryFaceInfo {
    /// 创建新的边界面信息
    pub fn new(
        face_id: usize,
        cell_id: usize,
        normal: DVec2,
        length: f64, // ALLOW_F64: 来自 PhysicsMesh 的几何数据
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
// 边界数据提供者接口
// ============================================================

/// 边界数据提供者接口
///
/// 用于从外部数据源获取边界强迫数据（水位、流速等）。
pub trait BoundaryDataProvider: Send + Sync {
    /// 获取指定面在给定时间的强迫数据
    ///
    /// # 参数
    /// - `face_id`: 边界面索引
    /// - `time`: 模拟时间 [s]
    ///
    /// # 返回
    /// 强迫数据，若无数据返回 None
    // ALLOW_F64: 时间参数与模拟进度配合
    fn get_forcing(&self, face_id: usize, time: f64) -> Option<ExternalForcing>;

    /// 批量获取强迫数据
    ///
    /// 默认实现逐个调用 `get_forcing`，可重写以优化性能。
    fn get_forcings_batch(
        &self,
        face_ids: &[usize],
        time: f64, // ALLOW_F64: 时间参数与模拟进度配合
        output: &mut [ExternalForcing],
    ) {
        debug_assert_eq!(face_ids.len(), output.len());
        for (i, &face_id) in face_ids.iter().enumerate() {
            output[i] = self.get_forcing(face_id, time).unwrap_or(ExternalForcing::ZERO);
        }
    }
}

/// 恒定强迫数据提供者
///
/// 用于测试和简单场景，返回恒定的强迫数据。
pub struct ConstantForcingProvider {
    forcing: ExternalForcing,
}

impl ConstantForcingProvider {
    /// 创建恒定强迫提供者
    pub fn new(forcing: ExternalForcing) -> Self {
        Self { forcing }
    }

    /// 创建仅水位的恒定提供者
    // ALLOW_F64: Layer 4 配置 API
    pub fn with_eta(eta: f64) -> Self {
        Self::new(ExternalForcing::with_eta(eta))
    }
}

impl BoundaryDataProvider for ConstantForcingProvider {
    fn get_forcing(&self, _face_id: usize, _time: f64) -> Option<ExternalForcing> { // ALLOW_F64: 时间参数
        Some(self.forcing)
    }

    fn get_forcings_batch(
        &self,
        face_ids: &[usize],
        _time: f64, // ALLOW_F64: 时间参数与模拟进度配合
        output: &mut [ExternalForcing],
    ) {
        output[..face_ids.len()].fill(self.forcing);
    }
}

// ============================================================
// 边界条件管理器
// ============================================================

/// 边界条件管理器
///
/// 负责：
/// 1. 管理边界条件定义
/// 2. 注册边界面与条件的映射
/// 3. 提供边界通量计算方法
///
/// # 使用流程
///
/// 1. 创建管理器
/// 2. 添加边界条件定义 (add_condition)
/// 3. 注册边界面 (register_face)
/// 4. 计算边界通量 (compute_*_flux)
///
/// # 示例
///
/// ```ignore
/// use mh_physics::boundary::{BoundaryManager, BoundaryCondition, BoundaryParams};
/// use glam::DVec2;
///
/// let mut manager = BoundaryManager::new(BoundaryParams::default());
///
/// // 添加边界条件
/// manager.add_condition(BoundaryCondition::wall("north"));
/// manager.add_condition(BoundaryCondition::open_sea("south"));
///
/// // 注册边界面
/// manager.register_face(0, 0, DVec2::new(0.0, 1.0), 1.0, "north").unwrap();
/// manager.register_face(1, 1, DVec2::new(0.0, -1.0), 1.0, "south").unwrap();
/// ```
pub struct BoundaryManager {
    /// 边界条件定义（按名称索引）
    conditions: HashMap<String, BoundaryCondition>,

    /// 条件名称到索引的映射
    condition_indices: HashMap<String, usize>,

    /// 条件列表（用于索引访问）
    condition_list: Vec<BoundaryCondition>,

    /// 固壁边界面
    wall_faces: Vec<BoundaryFaceInfo>,

    /// 开边界面（Flather）
    open_faces: Vec<BoundaryFaceInfo>,

    /// 入流边界面
    inflow_faces: Vec<BoundaryFaceInfo>,

    /// 出流边界面
    outflow_faces: Vec<BoundaryFaceInfo>,

    /// 周期边界面
    periodic_faces: Vec<BoundaryFaceInfo>,

    /// 计算参数
    params: BoundaryParams,
}

impl BoundaryManager {
    /// 创建新的边界管理器
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

    /// 从数值参数创建
    pub fn from_numerical_params(params: &crate::types::NumericalParams) -> Self {
        Self::new(BoundaryParams::from_numerical_params(params))
    }

    /// 添加边界条件定义
    ///
    /// # 参数
    /// - `condition`: 边界条件配置
    ///
    /// # 返回
    /// 条件在列表中的索引
    pub fn add_condition(&mut self, condition: BoundaryCondition) -> usize {
        let idx = self.condition_list.len();
        self.condition_indices.insert(condition.name.clone(), idx);
        self.conditions
            .insert(condition.name.clone(), condition.clone());
        self.condition_list.push(condition);
        idx
    }

    /// 获取边界条件
    pub fn get_condition(&self, name: &str) -> Option<&BoundaryCondition> {
        self.conditions.get(name)
    }

    /// 获取边界条件（按索引）
    pub fn get_condition_by_index(&self, idx: usize) -> Option<&BoundaryCondition> {
        self.condition_list.get(idx)
    }

    /// 注册边界面
    ///
    /// # 参数
    /// - `face_id`: 面索引
    /// - `cell_id`: 所属单元索引
    /// - `normal`: 面外法向量（应为单位向量）
    /// - `length`: 面长度 [m]
    /// - `boundary_name`: 边界条件名称
    ///
    /// # 错误
    /// - 边界条件未找到
    pub fn register_face(
        &mut self,
        face_id: usize,
        cell_id: usize,
        normal: DVec2,
        length: f64, // ALLOW_F64: 来自 PhysicsMesh 的几何数据
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

    // ========== 通量计算方法 ==========

    /// 计算固壁边界通量
    ///
    /// 固壁边界：无穿透，只有压力作用
    ///
    /// # 参数
    /// - `h_interior`: 内部单元水深 [m]
    /// - `normal`: 面外法向量
    ///
    /// # 返回
    /// (质量通量, 动量通量向量)
    // ALLOW_F64: 与 BoundaryParams 和 DVec2 配合使用
    pub fn compute_wall_flux(&self, h_interior: f64, normal: DVec2) -> (f64, DVec2) {
        // 质量通量为零（无穿透）
        let mass_flux = 0.0;

        // 动量通量仅有压力项
        let p = self.params.hydrostatic_pressure(h_interior);
        let momentum_flux = normal * p;

        (mass_flux, momentum_flux)
    }

    /// 计算 Flather 辐射边界通量
    ///
    /// Flather 边界条件使用特征关系结合外部强迫。
    /// 适用于开海边界，允许波动自由传出。
    ///
    /// # 参数
    /// - `interior`: 内部单元状态
    /// - `z_interior`: 内部单元底高程 [m]
    /// - `external`: 外部强迫数据
    /// - `normal`: 面外法向量
    ///
    /// # 返回
    /// (质量通量, 动量通量向量)
    pub fn compute_flather_flux(
        &self,
        interior: ConservedState,
        z_interior: f64, // ALLOW_F64: 与 BoundaryParams、ConservedState 和 DVec2 配合使用
        external: &ExternalForcing,
        normal: DVec2,
    ) -> (f64, DVec2) { // ALLOW_F64: 与 DVec2 配合
        let h = interior.h.max(self.params.h_min);
        let c = self.params.wave_speed(h);

        // 内部速度
        let u = interior.hu / h;
        let v = interior.hv / h;
        let velocity = DVec2::new(u, v);

        // 法向速度
        let un_int = velocity.dot(normal);
        let un_ext = external.velocity.dot(normal);

        // 内部水位
        let eta_int = h + z_interior;

        // Flather 条件: un* = un_ext + (c/h)(eta_int - eta_ext)
        let un_star = un_ext + (c / h) * (eta_int - external.eta);

        // 通量计算
        let mass_flux = h * un_star;
        let p = self.params.hydrostatic_pressure(h);
        let momentum_flux = normal * (mass_flux * un_int + p);

        (mass_flux, momentum_flux)
    }

    /// 计算自由出流通量
    ///
    /// 零梯度外推，直接使用内部状态计算通量。
    ///
    /// # 参数
    /// - `interior`: 内部单元状态
    /// - `normal`: 面外法向量
    ///
    /// # 返回
    /// (质量通量, 动量通量向量)
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

    /// 计算入流边界通量
    ///
    /// 根据给定的流量或速度计算通量。
    ///
    /// # 参数
    /// - `h_interior`: 内部单元水深 [m]
    /// - `discharge`: 入流流量 [m³/s]
    /// - `face_length`: 面长度 [m]
    /// - `normal`: 面外法向量
    ///
    /// # 返回
    /// (质量通量, 动量通量向量)
    pub fn compute_inflow_flux(
        &self,
        h_interior: f64, // ALLOW_F64: 与 BoundaryParams 和 DVec2 配合使用
        discharge: f64, // ALLOW_F64: 入流流量参数
        face_length: f64, // ALLOW_F64: 来自 PhysicsMesh 的几何数据
        normal: DVec2,
    ) -> (f64, DVec2) { // ALLOW_F64: 与 DVec2 配合
        // 入流流量（负号因为入流方向与法向相反）
        let qn = -discharge / face_length.max(1e-10);

        let p = self.params.hydrostatic_pressure(h_interior);
        let u_in = qn / h_interior.max(self.params.h_min);
        let momentum_flux = normal * (qn * u_in + p);

        (qn, momentum_flux)
    }

    // ========== 访问方法 ==========

    /// 获取固壁边界面列表
    pub fn wall_faces(&self) -> &[BoundaryFaceInfo] {
        &self.wall_faces
    }

    /// 获取开边界面列表
    pub fn open_faces(&self) -> &[BoundaryFaceInfo] {
        &self.open_faces
    }

    /// 获取入流边界面列表
    pub fn inflow_faces(&self) -> &[BoundaryFaceInfo] {
        &self.inflow_faces
    }

    /// 获取出流边界面列表
    pub fn outflow_faces(&self) -> &[BoundaryFaceInfo] {
        &self.outflow_faces
    }

    /// 获取周期边界面列表
    pub fn periodic_faces(&self) -> &[BoundaryFaceInfo] {
        &self.periodic_faces
    }

    /// 获取所有边界面的迭代器
    pub fn all_faces(&self) -> impl Iterator<Item = &BoundaryFaceInfo> {
        self.wall_faces
            .iter()
            .chain(&self.open_faces)
            .chain(&self.inflow_faces)
            .chain(&self.outflow_faces)
            .chain(&self.periodic_faces)
    }

    /// 获取边界面总数
    pub fn total_boundary_faces(&self) -> usize {
        self.wall_faces.len()
            + self.open_faces.len()
            + self.inflow_faces.len()
            + self.outflow_faces.len()
            + self.periodic_faces.len()
    }

    /// 获取边界条件数量
    pub fn condition_count(&self) -> usize {
        self.condition_list.len()
    }

    /// 获取计算参数
    pub fn params(&self) -> &BoundaryParams {
        &self.params
    }

    /// 清空所有注册的边界面
    pub fn clear_faces(&mut self) {
        self.wall_faces.clear();
        self.open_faces.clear();
        self.inflow_faces.clear();
        self.outflow_faces.clear();
        self.periodic_faces.clear();
    }

    /// 验证边界条件设置
    ///
    /// 检查：
    /// 1. 法向量是否单位化
    /// 2. 是否有重复的边界面
    pub fn validate(&self) -> Result<(), BoundaryError> {
        use std::collections::HashSet;
        let mut seen_faces = HashSet::new();

        for face in self.all_faces() {
            // 检查法向量是否单位化
            let mag_sq = face.normal.length_squared();
            if (mag_sq - 1.0).abs() > 1e-6 {
                return Err(BoundaryError::InvalidNormal {
                    face_id: face.face_id,
                    magnitude: mag_sq.sqrt(),
                });
            }

            // 检查是否重复
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
// 错误类型
// ============================================================

/// 边界模块错误类型
#[derive(Debug, Error)]
pub enum BoundaryError {
    /// 边界条件未找到
    #[error("边界条件 '{0}' 未找到")]
    ConditionNotFound(String),

    /// 边界面法向量未单位化
    #[error("边界面 {face_id} 法向量未单位化，模长为 {magnitude}")]
    InvalidNormal {
        face_id: usize,
        magnitude: f64, // ALLOW_F64: 错误信息中的幅度值
    },

    /// 重复的边界面
    #[error("重复的边界面: {0}")]
    DuplicateFace(usize),

    /// 边界配置错误
    #[error("边界配置错误: {0}")]
    Configuration(String),
}

// ============================================================
// 测试
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
        assert!(momentum.x > 0.0); // 压力向外
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

        // 单位向量应该通过
        manager
            .register_face(0, 0, DVec2::new(1.0, 0.0), 1.0, "test")
            .unwrap();
        assert!(manager.validate().is_ok());

        // 清空并添加非单位向量
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
