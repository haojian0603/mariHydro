// crates/mh_physics/src/boundary/manager.rs

//! 边界条件管理器（Backend泛型化版本）
//!
//! 本模块提供边界条件的管理和边界通量计算功能：
//! - BoundaryManager: 边界条件管理器
//! - BoundaryFaceInfo: 边界面信息
//! - BoundaryDataProvider: 边界数据提供者接口
//!
//! # Backend泛型化
//!
//! 所有几何数据使用`B::Vector2D`和`B::Scalar`，支持任意Backend精度切换。

use std::collections::HashMap;
use thiserror::Error;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use super::types::{BoundaryCondition, BoundaryKind, BoundaryParams, ExternalForcing};
use crate::state::ConservedState;
use mh_runtime::{Backend, RuntimeScalar, Vector2D};

// ============================================================
// 边界面信息（Backend泛型化）
// ============================================================

/// 边界面信息（Backend泛型化）
#[derive(Debug, Clone, Copy)]
pub struct BoundaryFaceInfo<B: Backend> {
    /// 面索引（在网格中的索引）
    pub face_id: usize,
    /// 所属单元索引
    pub cell_id: usize,
    /// 面外法向量（单位向量）
    pub normal: B::Vector2D,
    /// 面长度 [m]
    pub length: B::Scalar,
    /// 所属边界条件的索引
    pub boundary_idx: usize,
}

impl<B: Backend> BoundaryFaceInfo<B> {
    /// 创建新的边界面信息
    pub fn new(
        face_id: usize,
        cell_id: usize,
        normal: B::Vector2D,
        length: B::Scalar,
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
pub trait BoundaryDataProvider: Send + Sync {
    /// 获取指定面在给定时间的强迫数据
    fn get_forcing(&self, face_id: usize, time: f64) -> Option<ExternalForcing>;

    /// 批量获取强迫数据
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

/// 恒定强迫数据提供者
pub struct ConstantForcingProvider {
    forcing: ExternalForcing,
}

impl ConstantForcingProvider {
    /// 创建恒定强迫提供者
    pub fn new(forcing: ExternalForcing) -> Self {
        Self { forcing }
    }

    /// 创建仅水位的恒定提供者
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
// 边界条件管理器（Backend泛型化）
// ============================================================

/// 边界条件管理器（Backend泛型化）
pub struct BoundaryManager<B: Backend> {
    /// 边界条件定义（按名称索引）
    conditions: HashMap<String, BoundaryCondition>,

    /// 条件名称到索引的映射
    condition_indices: HashMap<String, usize>,

    /// 条件列表（用于索引访问）
    condition_list: Vec<BoundaryCondition>,

    /// 固壁边界面
    wall_faces: Vec<BoundaryFaceInfo<B>>,

    /// 开边界面（Flather）
    open_faces: Vec<BoundaryFaceInfo<B>>,

    /// 入流边界面
    inflow_faces: Vec<BoundaryFaceInfo<B>>,

    /// 出流边界面
    outflow_faces: Vec<BoundaryFaceInfo<B>>,

    /// 周期边界面
    periodic_faces: Vec<BoundaryFaceInfo<B>>,

    /// 计算参数
    params: BoundaryParams,

    /// 后端实例
    backend: B,
}

impl<B: Backend> BoundaryManager<B> {
    /// 创建新的边界管理器
    pub fn new_with_backend(backend: B, params: BoundaryParams) -> Self {
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
            backend,
        }
    }

    /// 从数值参数创建
    pub fn from_numerical_params(backend: B, params: &crate::types::NumericalParams<B::Scalar>, gravity: B::Scalar) -> Self {
        // 转换参数到 f64（因为 ExternalForcing 使用 f64）
        let gravity_f64 = gravity.to_f64().unwrap_or(9.81);
        let h_min = params.h_min.to_f64().unwrap_or(1e-6);
        Self::new_with_backend(backend, BoundaryParams::new(gravity_f64, h_min))
    }

    /// 添加边界条件定义
    pub fn add_condition(&mut self, condition: BoundaryCondition) -> usize {
        let idx = self.condition_list.len();
        self.condition_indices.insert(condition.name.clone(), idx);
        self.conditions.insert(condition.name.clone(), condition.clone());
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
    pub fn register_face(
        &mut self,
        face_id: usize,
        cell_id: usize,
        normal: B::Vector2D,
        length: B::Scalar,
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

    // ========== 通量计算方法（Backend泛型化） ==========

    /// 计算固壁边界通量（Backend泛型化）
    pub fn compute_wall_flux(
        &self,
        h_interior: B::Scalar,
        normal: B::Vector2D,
    ) -> (B::Scalar, B::Vector2D) {
        // 质量通量为零（无穿透）
        let mass_flux = B::Scalar::ZERO;

        // 动量通量仅有压力项
        let g = B::Scalar::from_f64(self.params.gravity).unwrap_or(B::Scalar::ZERO);
        let pressure = B::Scalar::HALF * g * h_interior * h_interior;
        let momentum_flux = B::vec2_scale(&normal, pressure);

        (mass_flux, momentum_flux)
    }

    /// 计算 Flather 辐射边界通量（Backend泛型化）
    pub fn compute_flather_flux(
        &self,
        interior: ConservedState<B::Scalar>,
        z_interior: B::Scalar,
        external: &ExternalForcing,
        normal: B::Vector2D,
    ) -> (B::Scalar, B::Vector2D) {
        let g = B::Scalar::from_f64(self.params.gravity).unwrap_or(B::Scalar::ZERO);
        let h = interior.h.max(B::Scalar::from_f64(self.params.h_min).unwrap_or(B::Scalar::ZERO));

        // 内部速度
        let u = interior.hu / h;
        let v = interior.hv / h;
        let vel_int = B::vec2_new(u, v);

        // 法向速度 (dot product)
        let un_int = B::vec2_dot(&vel_int, &normal);
        let un_ext = B::Scalar::from_f64(external.velocity.0).unwrap_or(B::Scalar::ZERO) * normal.x()
            + B::Scalar::from_f64(external.velocity.1).unwrap_or(B::Scalar::ZERO) * normal.y();

        // 内部水位
        let eta_int = h + z_interior;
        let eta_ext = B::Scalar::from_f64(external.eta).unwrap_or(B::Scalar::ZERO);

        // Flather 条件: un* = un_ext + (c/h)(eta_int - eta_ext)
        let c = (g * h).sqrt();
        let un_star = un_ext + (c / h) * (eta_int - eta_ext);

        // 通量计算
        let mass_flux = h * un_star;
        let pressure = B::Scalar::HALF * g * h * h;
        let momentum_dot = mass_flux * un_int + pressure;
        let momentum_flux = B::vec2_scale(&normal, momentum_dot);

        (mass_flux, momentum_flux)
    }

    /// 计算自由出流通量（Backend泛型化）
    pub fn compute_outflow_flux(
        &self,
        interior: ConservedState<B::Scalar>,
        normal: B::Vector2D,
    ) -> (B::Scalar, B::Vector2D) {
        let g = B::Scalar::from_f64(self.params.gravity).unwrap_or(B::Scalar::ZERO);
        let h = interior.h.max(B::Scalar::from_f64(self.params.h_min).unwrap_or(B::Scalar::ZERO));
        let u = interior.hu / h;
        let v = interior.hv / h;
        let vel = B::vec2_new(u, v);

        // 法向速度
        let un = B::vec2_dot(&vel, &normal);
        let mass_flux = h * un;
        let pressure = B::Scalar::HALF * g * h * h;
        let momentum_dot = mass_flux * un + pressure;
        let momentum_flux = B::vec2_scale(&normal, momentum_dot);

        (mass_flux, momentum_flux)
    }

    /// 计算入流边界通量（Backend泛型化）
    pub fn compute_inflow_flux(
        &self,
        h_interior: B::Scalar,
        discharge: B::Scalar,
        face_length: B::Scalar,
        normal: B::Vector2D,
    ) -> (B::Scalar, B::Vector2D) {
        let g = B::Scalar::from_f64(self.params.gravity).unwrap_or(B::Scalar::ZERO);
        
        // 入流流量（负号因为入流方向与法向相反）
        let qn = -discharge / face_length.max(B::Scalar::from_f64(1e-10).unwrap_or(B::Scalar::ZERO));

        let pressure = B::Scalar::HALF * g * h_interior * h_interior;
        let u_in = qn / h_interior.max(B::Scalar::from_f64(self.params.h_min).unwrap_or(B::Scalar::ZERO));
        let momentum_dot = qn * u_in + pressure;
        let momentum_flux = B::vec2_scale(&normal, momentum_dot);

        (qn, momentum_flux)
    }

    // ========== 访问方法 ==========

    /// 获取固壁边界面列表
    pub fn wall_faces(&self) -> &[BoundaryFaceInfo<B>] {
        &self.wall_faces
    }

    /// 获取开边界面列表
    pub fn open_faces(&self) -> &[BoundaryFaceInfo<B>] {
        &self.open_faces
    }

    /// 获取入流边界面列表
    pub fn inflow_faces(&self) -> &[BoundaryFaceInfo<B>] {
        &self.inflow_faces
    }

    /// 获取出流边界面列表
    pub fn outflow_faces(&self) -> &[BoundaryFaceInfo<B>] {
        &self.outflow_faces
    }

    /// 获取周期边界面列表
    pub fn periodic_faces(&self) -> &[BoundaryFaceInfo<B>] {
        &self.periodic_faces
    }

    /// 获取所有边界面的迭代器
    pub fn all_faces(&self) -> impl Iterator<Item = &BoundaryFaceInfo<B>> {
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
    pub fn validate(&self) -> Result<(), BoundaryError> {
        use std::collections::HashSet;
        let mut seen_faces = HashSet::new();

        for face in self.all_faces() {
            // 检查法向量是否单位化
            let mag_sq = B::vec2_dot(&face.normal, &face.normal);
            let one = B::Scalar::ONE;
            let eps = B::Scalar::from_f64(1e-6).unwrap_or(B::Scalar::ZERO);
            if (mag_sq - one).abs() > eps {
                return Err(BoundaryError::InvalidNormal {
                    face_id: face.face_id,
                    magnitude: mag_sq.to_f64().unwrap_or(0.0),
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

impl<B: Backend> Default for BoundaryManager<B>
where
    B: Default,
{
    fn default() -> Self {
        Self::new_with_backend(B::default(), BoundaryParams::default())
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
        magnitude: f64,
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
    use mh_runtime::CpuBackend;

    #[test]
    fn test_boundary_manager_creation() {
        let backend = CpuBackend::<f64>::new();
        let manager = BoundaryManager::<CpuBackend<f64>>::new_with_backend(backend, BoundaryParams::default());
        assert_eq!(manager.total_boundary_faces(), 0);
        assert_eq!(manager.condition_count(), 0);
    }

    #[test]
    fn test_add_condition() {
        let backend = CpuBackend::<f64>::new();
        let mut manager = BoundaryManager::<CpuBackend<f64>>::new_with_backend(backend, BoundaryParams::default());

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
    fn test_register_face_f64() {
        let backend = CpuBackend::<f64>::new();
        let mut manager = BoundaryManager::<CpuBackend<f64>>::new_with_backend(backend, BoundaryParams::default());
        manager.add_condition(BoundaryCondition::wall("north"));
        manager.add_condition(BoundaryCondition::open_sea("south"));

        manager
            .register_face(0, 0, CpuBackend::<f64>::vec2_new(0.0, 1.0), 1.0, "north")
            .unwrap();
        manager
            .register_face(1, 1, CpuBackend::<f64>::vec2_new(0.0, -1.0), 1.0, "south")
            .unwrap();

        assert_eq!(manager.wall_faces().len(), 1);
        assert_eq!(manager.open_faces().len(), 1);
        assert_eq!(manager.total_boundary_faces(), 2);
    }

    #[test]
    fn test_register_unknown_condition() {
        let backend = CpuBackend::<f64>::new();
        let mut manager = BoundaryManager::<CpuBackend<f64>>::new_with_backend(backend, BoundaryParams::default());
        let result = manager.register_face(0, 0, CpuBackend::<f64>::vec2_new(0.0, 1.0), 1.0, "unknown");

        assert!(result.is_err());
        if let Err(BoundaryError::ConditionNotFound(name)) = result {
            assert_eq!(name, "unknown");
        } else {
            panic!("Expected ConditionNotFound error");
        }
    }

    #[test]
    fn test_wall_flux_f64() {
        let backend = CpuBackend::<f64>::new();
        let manager = BoundaryManager::<CpuBackend<f64>>::new_with_backend(backend, BoundaryParams::default());
        let (mass, momentum) = manager.compute_wall_flux(1.0, CpuBackend::<f64>::vec2_new(1.0, 0.0));

        assert_eq!(mass, 0.0);
        assert!(momentum.x() > 0.0); // 压力向外
        assert!(momentum.y().abs() < 1e-10);
    }

    #[test]
    fn test_wall_flux_f32() {
        let backend = CpuBackend::<f32>::new();
        let manager = BoundaryManager::<CpuBackend<f32>>::new_with_backend(backend, BoundaryParams::default());
        let (mass, momentum) = manager.compute_wall_flux(1.0f32, CpuBackend::<f32>::vec2_new(1.0, 0.0));

        assert_eq!(mass, 0.0f32);
        assert!(momentum.x() > 0.0f32);
    }

    #[test]
    fn test_outflow_flux_f64() {
        let backend = CpuBackend::<f64>::new();
        let manager = BoundaryManager::<CpuBackend<f64>>::new_with_backend(backend, BoundaryParams::default());
        let interior = ConservedState::<f64>::from_primitive(1.0, 1.0, 0.0);
        let (mass, _) = manager.compute_outflow_flux(interior, CpuBackend::<f64>::vec2_new(1.0, 0.0));

        assert!((mass - 1.0).abs() < 1e-10); // h * u * normal = 1 * 1 * 1
    }

    #[test]
    fn test_validate_normalization() {
        let backend = CpuBackend::<f64>::new();
        let mut manager = BoundaryManager::<CpuBackend<f64>>::new_with_backend(backend, BoundaryParams::default());
        manager.add_condition(BoundaryCondition::wall("test"));

        // 单位向量应该通过
        manager
            .register_face(0, 0, CpuBackend::<f64>::vec2_new(1.0, 0.0), 1.0, "test")
            .unwrap();
        assert!(manager.validate().is_ok());

        // 清空并添加非单位向量
        manager.clear_faces();
        manager
            .register_face(1, 0, CpuBackend::<f64>::vec2_new(2.0, 0.0), 1.0, "test")
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

    #[test]
    fn test_f32_backend_full() {
        let backend = CpuBackend::<f32>::new();
        let mut manager = BoundaryManager::<CpuBackend<f32>>::new_with_backend(backend, BoundaryParams::default());
        
        manager.add_condition(BoundaryCondition::wall("test"));
        manager.register_face(0, 0, CpuBackend::<f32>::vec2_new(1.0, 0.0), 1.0f32, "test").unwrap();
        
        assert_eq!(manager.total_boundary_faces(), 1);
        let (mass, momentum) = manager.compute_wall_flux(1.0f32, CpuBackend::<f32>::vec2_new(1.0, 0.0));
        assert_eq!(std::mem::size_of_val(&mass), 4); // f32验证
        assert_eq!(std::mem::size_of_val(&momentum.x()), 4);
    }
}