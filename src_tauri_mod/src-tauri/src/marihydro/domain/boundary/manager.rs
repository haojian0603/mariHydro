// src-tauri/src/marihydro/domain/boundary/manager.rs

//! 边界条件管理器

use std::collections::HashMap;

use super::types::{BoundaryCondition, BoundaryKind, BoundaryParams, ExternalForcing};
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::types::NumericalParams;
use crate::marihydro::domain::mesh::indices::{CellId, FaceId};

/// 边界面信息
#[derive(Debug, Clone, Copy)]
pub struct BoundaryFaceInfo {
    pub face_id: FaceId,
    pub cell_id: CellId,
    pub normal: (f64, f64),
    pub boundary_idx: usize,
}

/// 边界数据提供者接口
pub trait BoundaryDataProvider: Send + Sync {
    /// 获取指定面的强迫数据
    fn get_forcing(&self, face_id: FaceId, time: f64) -> MhResult<ExternalForcing>;
}

/// 边界条件管理器
///
/// 不依赖 ProjectManifest，通过构造函数或方法设置边界条件
pub struct BoundaryManager {
    /// 边界条件定义（按名称索引）
    conditions: HashMap<String, BoundaryCondition>,

    /// 固壁边界面
    wall_faces: Vec<BoundaryFaceInfo>,
    /// 开边界面
    open_faces: Vec<BoundaryFaceInfo>,
    /// 入流边界面
    inflow_faces: Vec<BoundaryFaceInfo>,
    /// 出流边界面
    outflow_faces: Vec<BoundaryFaceInfo>,

    /// 边界参数
    params: BoundaryParams,
}

impl BoundaryManager {
    /// 创建新的边界管理器
    pub fn new(params: BoundaryParams) -> Self {
        Self {
            conditions: HashMap::new(),
            wall_faces: Vec::new(),
            open_faces: Vec::new(),
            inflow_faces: Vec::new(),
            outflow_faces: Vec::new(),
            params,
        }
    }

    /// 从数值参数创建
    pub fn from_numerical_params(numerical_params: &NumericalParams, gravity: f64) -> Self {
        let params = BoundaryParams::new(gravity, numerical_params.h_min);
        Self::new(params)
    }

    /// 添加边界条件
    pub fn add_condition(&mut self, condition: BoundaryCondition) {
        self.conditions.insert(condition.name.clone(), condition);
    }

    /// 注册边界面
    pub fn register_face(
        &mut self,
        face_id: FaceId,
        cell_id: CellId,
        normal: (f64, f64),
        boundary_name: &str,
    ) -> MhResult<()> {
        let condition =
            self.conditions
                .get(boundary_name)
                .ok_or_else(|| MhError::BoundaryNotFound {
                    boundary_id: boundary_name.to_string(),
                })?;

        let boundary_idx = self
            .conditions
            .keys()
            .position(|k| k == boundary_name)
            .unwrap_or(0);

        let info = BoundaryFaceInfo {
            face_id,
            cell_id,
            normal,
            boundary_idx,
        };

        match condition.kind {
            BoundaryKind::Wall | BoundaryKind::Symmetry => {
                self.wall_faces.push(info);
            }
            BoundaryKind::OpenSea => {
                self.open_faces.push(info);
            }
            BoundaryKind::RiverInflow => {
                self.inflow_faces.push(info);
            }
            BoundaryKind::Outflow => {
                self.outflow_faces.push(info);
            }
            BoundaryKind::Periodic => {
                // 周期边界需要特殊处理
            }
        }

        Ok(())
    }

    /// 从网格注册边界面
    pub fn register_from_mesh(
        &mut self,
        mesh: &crate::marihydro::domain::mesh::UnstructuredMesh,
    ) -> MhResult<()> {
        use crate::marihydro::core::traits::mesh::MeshAccess;
        use crate::marihydro::core::types::FaceIndex;

        for face_idx in mesh.boundary_faces() {
            let face = FaceIndex(face_idx);
            let owner = mesh.face_owner(face);
            let normal = mesh.face_normal(face);

            if let Some(boundary_idx) = mesh.boundary_id(face) {
                if let Some(name) = mesh.boundary_name(boundary_idx) {
                    self.register_face(
                        FaceId(face_idx),
                        CellId(owner.0),
                        (normal.x, normal.y),
                        name,
                    )?;
                }
            }
        }

        log::info!(
            "边界注册完成: {} 固壁, {} 开边界, {} 入流, {} 出流",
            self.wall_faces.len(),
            self.open_faces.len(),
            self.inflow_faces.len(),
            self.outflow_faces.len()
        );

        Ok(())
    }

    /// 获取边界条件
    pub fn get_condition(&self, name: &str) -> Option<&BoundaryCondition> {
        self.conditions.get(name)
    }

    /// 获取固壁边界面
    pub fn wall_faces(&self) -> &[BoundaryFaceInfo] {
        &self.wall_faces
    }

    /// 获取开边界面
    pub fn open_faces(&self) -> &[BoundaryFaceInfo] {
        &self.open_faces
    }

    /// 获取入流边界面
    pub fn inflow_faces(&self) -> &[BoundaryFaceInfo] {
        &self.inflow_faces
    }

    /// 获取出流边界面
    pub fn outflow_faces(&self) -> &[BoundaryFaceInfo] {
        &self.outflow_faces
    }

    /// 获取边界参数
    pub fn params(&self) -> &BoundaryParams {
        &self.params
    }

    /// 计算固壁边界通量
    pub fn compute_wall_flux(
        &self,
        h_interior: f64,
        _u_interior: f64,
        _v_interior: f64,
        normal: (f64, f64),
    ) -> (f64, f64, f64) {
        // 固壁边界：无穿透，只有压力
        let p = 0.5 * self.params.gravity * h_interior * h_interior;
        (0.0, p * normal.0, p * normal.1)
    }

    /// 计算开边界通量（Flather 辐射边界）
    pub fn compute_flather_flux(
        &self,
        h_int: f64,
        u_int: f64,
        v_int: f64,
        z_int: f64,
        external: &ExternalForcing,
        normal: (f64, f64),
    ) -> (f64, f64, f64) {
        let h_safe = h_int.max(self.params.h_min);
        let c = self.params.sqrt_g * h_safe.sqrt();

        let un_int = u_int * normal.0 + v_int * normal.1;
        let eta_int = h_int + z_int;

        // Flather 条件
        let un_star =
            external.u * normal.0 + external.v * normal.1 + (c / h_safe) * (eta_int - external.eta);

        let qn = h_int * un_star;
        let p = 0.5 * self.params.gravity * h_int * h_int;

        (qn, qn * u_int + p * normal.0, qn * v_int + p * normal.1)
    }

    /// 计算自由出流通量
    pub fn compute_outflow_flux(
        &self,
        h: f64,
        u: f64,
        v: f64,
        normal: (f64, f64),
    ) -> (f64, f64, f64) {
        let qn = h * (u * normal.0 + v * normal.1);
        let p = 0.5 * self.params.gravity * h * h;
        (qn, qn * u + p * normal.0, qn * v + p * normal.1)
    }

    /// 计算入流边界通量
    pub fn compute_inflow_flux(
        &self,
        h_int: f64,
        u_int: f64,
        v_int: f64,
        discharge: f64,
        face_length: f64,
        normal: (f64, f64),
    ) -> (f64, f64, f64) {
        // 入流流量
        let qn = -discharge / face_length; // 负号因为入流
        let p = 0.5 * self.params.gravity * h_int * h_int;

        (qn, qn * u_int + p * normal.0, qn * v_int + p * normal.1)
    }

    /// 验证边界条件设置
    pub fn validate(&self) -> MhResult<()> {
        use std::collections::HashSet;

        let mut seen_faces = HashSet::new();

        for face in self.all_faces() {
            // 检查法向量是否单位化
            let (nx, ny) = face.normal;
            let mag_sq = nx * nx + ny * ny;
            if (mag_sq - 1.0).abs() > 1e-6 {
                return Err(MhError::BoundaryCondition {
                    message: format!(
                        "边界面 {} 法向量未单位化: ({}, {}), |n|² = {}",
                        face.face_id, nx, ny, mag_sq
                    ),
                });
            }

            // 检查是否重复
            if !seen_faces.insert(face.face_id) {
                return Err(MhError::BoundaryCondition {
                    message: format!("重复的边界面: {}", face.face_id),
                });
            }
        }

        Ok(())
    }

    /// 所有边界面的迭代器
    fn all_faces(&self) -> impl Iterator<Item = &BoundaryFaceInfo> {
        self.wall_faces
            .iter()
            .chain(&self.open_faces)
            .chain(&self.inflow_faces)
            .chain(&self.outflow_faces)
    }

    /// 边界面总数
    pub fn total_boundary_faces(&self) -> usize {
        self.wall_faces.len()
            + self.open_faces.len()
            + self.inflow_faces.len()
            + self.outflow_faces.len()
    }

    /// 清空所有注册的边界面
    pub fn clear_faces(&mut self) {
        self.wall_faces.clear();
        self.open_faces.clear();
        self.inflow_faces.clear();
        self.outflow_faces.clear();
    }
}

impl Default for BoundaryManager {
    fn default() -> Self {
        Self::new(BoundaryParams::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_manager_creation() {
        let manager = BoundaryManager::default();
        assert_eq!(manager.total_boundary_faces(), 0);
    }

    #[test]
    fn test_add_condition() {
        let mut manager = BoundaryManager::default();
        manager.add_condition(BoundaryCondition::wall("north"));
        manager.add_condition(BoundaryCondition::open_sea("south"));

        assert!(manager.get_condition("north").is_some());
        assert!(manager.get_condition("south").is_some());
        assert!(manager.get_condition("east").is_none());
    }

    #[test]
    fn test_wall_flux() {
        let manager = BoundaryManager::default();
        let (mass, mom_x, mom_y) = manager.compute_wall_flux(1.0, 10.0, 0.0, (1.0, 0.0));

        assert_eq!(mass, 0.0);
        assert!(mom_x > 0.0); // 压力向外
    }

    #[test]
    fn test_outflow_flux() {
        let manager = BoundaryManager::default();
        let (mass, _, _) = manager.compute_outflow_flux(1.0, 1.0, 0.0, (1.0, 0.0));

        assert!((mass - 1.0).abs() < 1e-10);
    }
}
