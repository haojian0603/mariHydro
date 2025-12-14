// crates/mh_mesh/src/frozen.rs

//! 冻结网格（泛型版本）
//!
//! 从半边网格导出的只读 SoA 布局，优化计算性能。
//! 支持 f32/f64 运行时精度切换。

use mh_geo::{Point2D, Point3D};
use mh_runtime::RuntimeScalar;
use serde::{Deserialize, Serialize};

/// 冻结网格（泛型版本）
///
/// 从 HalfEdgeMesh 导出的只读计算用网格，支持 f32/f64 精度运行时切换
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenMesh<S: RuntimeScalar = f64> {
    // ===== 节点数据 =====
    /// 节点数量（几何数据，保持 usize）
    pub n_nodes: usize,
    /// 节点坐标 (3D，几何数据保持 f64)
    pub node_coords: Vec<Point3D>,

    // ===== 单元数据 =====
    /// 单元数量
    pub n_cells: usize,
    /// 单元中心坐标（2D，几何数据保持 f64）
    pub cell_center: Vec<Point2D>,
    /// 单元面积（物理场数据，泛型化）
    pub cell_area: Vec<S>,
    /// 单元底床高程（物理场数据，泛型化）
    pub cell_z_bed: Vec<S>,
    /// 单元节点索引 (压缩格式，索引数据保持 u32)
    pub cell_node_offsets: Vec<usize>,
    /// 单元节点索引列表
    pub cell_node_indices: Vec<u32>,
    /// 单元面索引 (压缩格式)
    pub cell_face_offsets: Vec<usize>,
    /// 单元面索引列表
    pub cell_face_indices: Vec<u32>,
    /// 单元邻居索引 (压缩格式)
    pub cell_neighbor_offsets: Vec<usize>,
    /// 单元邻居索引列表 (u32::MAX 表示无邻居)
    pub cell_neighbor_indices: Vec<u32>,

    // ===== 面数据 =====
    /// 面总数
    pub n_faces: usize,
    /// 内部面数量
    pub n_interior_faces: usize,
    /// 面中心坐标（2D，几何数据）
    pub face_center: Vec<Point2D>,
    /// 面法向量 (3D，几何数据)
    pub face_normal: Vec<Point3D>,
    /// 面长度（物理场数据，泛型化）
    pub face_length: Vec<S>,
    /// 面左侧高程（物理场数据，泛型化）
    pub face_z_left: Vec<S>,
    /// 面右侧高程（物理场数据，泛型化）
    pub face_z_right: Vec<S>,
    /// 面 owner 单元索引（索引数据）
    pub face_owner: Vec<u32>,
    /// 面 neighbor 单元索引 (u32::MAX 表示边界)
    pub face_neighbor: Vec<u32>,
    /// 面到 owner 中心的向量（2D，几何数据）
    pub face_delta_owner: Vec<Point2D>,
    /// 面到 neighbor 中心的向量（2D，几何数据）
    pub face_delta_neighbor: Vec<Point2D>,
    /// owner 到 neighbor 的距离（物理场数据，泛型化）
    pub face_dist_o2n: Vec<S>,

    // ===== 边界数据 =====
    /// 边界面索引列表（索引数据）
    pub boundary_face_indices: Vec<u32>,
    /// 边界名称
    pub boundary_names: Vec<String>,
    /// 面的边界ID (None表示内部面)
    pub face_boundary_id: Vec<Option<u32>>,

    // ===== 统计 =====
    /// 最小单元尺寸（物理场数据，泛型化）
    pub min_cell_size: S,
    /// 最大单元尺寸（物理场数据，泛型化）
    pub max_cell_size: S,

    // ===== AMR 预分配字段 (Phase 2+) =====
    /// 单元细化级别 (0=基础网格, 1,2,3=细化级别)
    pub cell_refinement_level: Vec<u8>,
    /// 父单元索引 (顶层单元指向自身)
    pub cell_parent: Vec<u32>,
    /// Ghost 单元容量 (用于 MPI 边界交换)
    pub ghost_capacity: usize,

    // ===== ID 映射与排列 =====
    /// 原始单元 ID（例如从 MSH 文件中读取的 physical entity id）
    #[serde(default)]
    pub cell_original_id: Vec<u32>,
    /// 原始面 ID（边界标识）
    #[serde(default)]
    pub face_original_id: Vec<u32>,
    /// 单元排列索引（frozen_idx -> 原始索引）
    #[serde(default)]
    pub cell_permutation: Vec<u32>,
    /// 逆排列（原始索引 -> frozen_idx）
    #[serde(default)]
    pub cell_inv_permutation: Vec<u32>,
}


impl<S: RuntimeScalar> Default for FrozenMesh<S> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<S: RuntimeScalar> FrozenMesh<S> {
    /// 创建空的冻结网格
    pub fn empty() -> Self {
        Self {
            n_nodes: 0,
            node_coords: Vec::new(),
            n_cells: 0,
            cell_center: Vec::new(),
            cell_area: Vec::new(),
            cell_z_bed: Vec::new(),
            cell_node_offsets: vec![0],
            cell_node_indices: Vec::new(),
            cell_face_offsets: vec![0],
            cell_face_indices: Vec::new(),
            cell_neighbor_offsets: vec![0],
            cell_neighbor_indices: Vec::new(),
            n_faces: 0,
            n_interior_faces: 0,
            face_center: Vec::new(),
            face_normal: Vec::new(),
            face_length: Vec::new(),
            face_z_left: Vec::new(),
            face_z_right: Vec::new(),
            face_owner: Vec::new(),
            face_neighbor: Vec::new(),
            face_delta_owner: Vec::new(),
            face_delta_neighbor: Vec::new(),
            face_dist_o2n: Vec::new(),
            boundary_face_indices: Vec::new(),
            boundary_names: Vec::new(),
            face_boundary_id: Vec::new(),
            min_cell_size: S::MAX,
            max_cell_size: S::ZERO,
            // AMR 预分配字段
            cell_refinement_level: Vec::new(),
            cell_parent: Vec::new(),
            ghost_capacity: 0,
            // ID 映射与排列
            cell_original_id: Vec::new(),
            face_original_id: Vec::new(),
            cell_permutation: Vec::new(),
            cell_inv_permutation: Vec::new(),
        }
    }

    /// 创建带有指定单元数量的空网格（用于测试）
    ///
    /// 创建的网格有指定数量的单元，但没有实际几何数据。
    pub fn empty_with_cells(n_cells: usize) -> Self {
        Self {
            n_nodes: 0,
            node_coords: Vec::new(),
            n_cells,
            cell_center: vec![Point2D::new(0.0, 0.0); n_cells],
            cell_area: vec![S::ONE; n_cells],
            cell_z_bed: vec![S::ZERO; n_cells],
            cell_node_offsets: vec![0; n_cells + 1],
            cell_node_indices: Vec::new(),
            cell_face_offsets: vec![0; n_cells + 1],
            cell_face_indices: Vec::new(),
            cell_neighbor_offsets: vec![0; n_cells + 1],
            cell_neighbor_indices: Vec::new(),
            n_faces: 0,
            n_interior_faces: 0,
            face_center: Vec::new(),
            face_normal: Vec::new(),
            face_length: Vec::new(),
            face_z_left: Vec::new(),
            face_z_right: Vec::new(),
            face_owner: Vec::new(),
            face_neighbor: Vec::new(),
            face_delta_owner: Vec::new(),
            face_delta_neighbor: Vec::new(),
            face_dist_o2n: Vec::new(),
            boundary_face_indices: Vec::new(),
            boundary_names: Vec::new(),
            face_boundary_id: Vec::new(),
            min_cell_size: S::ONE,
            max_cell_size: S::ONE,
            // AMR 预分配字段
            cell_refinement_level: vec![0; n_cells],
            cell_parent: (0..n_cells as u32).collect(),
            ghost_capacity: 0,
            // ID 映射与排列
            cell_original_id: (0..n_cells as u32).collect(),
            face_original_id: Vec::new(),
            cell_permutation: (0..n_cells as u32).collect(),
            cell_inv_permutation: (0..n_cells as u32).collect(),
        }
    }

    // =========================================================================
    // 基本统计
    // =========================================================================

    /// 节点数量
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// 单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 面数量
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.n_faces
    }

    /// 内部面数量
    #[inline]
    pub fn n_interior_faces(&self) -> usize {
        self.n_interior_faces
    }

    /// 边界面数量
    #[inline]
    pub fn n_boundary_faces(&self) -> usize {
        self.n_faces - self.n_interior_faces
    }

    // =========================================================================
    // 单元访问
    // =========================================================================

    /// 获取单元中心（几何数据，返回 f64）
    #[inline]
    pub fn cell_center(&self, cell: usize) -> Point2D {
        self.cell_center[cell]
    }

    /// 获取单元面积（物理场数据，返回 S）
    #[inline]
    pub fn cell_area(&self, cell: usize) -> S {
        self.cell_area[cell]
    }

    /// 获取单元底床高程（物理场数据，返回 S）
    #[inline]
    pub fn cell_z_bed(&self, cell: usize) -> S {
        self.cell_z_bed[cell]
    }

    /// 获取单元的节点索引
    #[inline]
    pub fn cell_nodes(&self, cell: usize) -> &[u32] {
        let start = self.cell_node_offsets[cell];
        let end = self.cell_node_offsets[cell + 1];
        &self.cell_node_indices[start..end]
    }

    /// 获取单元的面索引
    #[inline]
    pub fn cell_faces(&self, cell: usize) -> &[u32] {
        let start = self.cell_face_offsets[cell];
        let end = self.cell_face_offsets[cell + 1];
        &self.cell_face_indices[start..end]
    }

    /// 获取单元的邻居索引
    #[inline]
    pub fn cell_neighbors(&self, cell: usize) -> &[u32] {
        let start = self.cell_neighbor_offsets[cell];
        let end = self.cell_neighbor_offsets[cell + 1];
        &self.cell_neighbor_indices[start..end]
    }

    // =========================================================================
    // 面访问
    // =========================================================================

    /// 获取面中心（几何数据，返回 f64）
    #[inline]
    pub fn face_center(&self, face: usize) -> Point2D {
        self.face_center[face]
    }

    /// 获取面法向量（3D，几何数据）
    #[inline]
    pub fn face_normal(&self, face: usize) -> Point3D {
        self.face_normal[face]
    }

    /// 获取面长度（物理场数据，返回 S）
    #[inline]
    pub fn face_length(&self, face: usize) -> S {
        self.face_length[face]
    }

    /// 获取面左侧高程（物理场数据，返回 S）
    #[inline]
    pub fn face_z_left(&self, face: usize) -> S {
        self.face_z_left[face]
    }

    /// 获取面右侧高程（物理场数据，返回 S）
    #[inline]
    pub fn face_z_right(&self, face: usize) -> S {
        self.face_z_right[face]
    }

    /// 获取面 owner（索引数据）
    #[inline]
    pub fn face_owner(&self, face: usize) -> u32 {
        self.face_owner[face]
    }

    /// 获取面 neighbor（索引数据）
    #[inline]
    pub fn face_neighbor(&self, face: usize) -> Option<u32> {
        let n = self.face_neighbor[face];
        if n == u32::MAX {
            None
        } else {
            Some(n)
        }
    }

    /// 判断是否为边界面
    #[inline]
    pub fn is_boundary_face(&self, face: usize) -> bool {
        face >= self.n_interior_faces
    }

    // =========================================================================
    // ID 映射与排列访问
    // =========================================================================

    /// 获取单元的原始 ID（从网格文件读取）
    #[inline]
    pub fn cell_original_id(&self, cell: usize) -> u32 {
        self.cell_original_id.get(cell).copied().unwrap_or(cell as u32)
    }

    /// 获取面的原始 ID（边界标识）
    #[inline]
    pub fn face_original_id(&self, face: usize) -> u32 {
        self.face_original_id.get(face).copied().unwrap_or(face as u32)
    }

    /// 从 frozen 索引获取原始索引
    #[inline]
    pub fn to_original_cell(&self, frozen_idx: usize) -> u32 {
        self.cell_permutation.get(frozen_idx).copied().unwrap_or(frozen_idx as u32)
    }

    /// 从原始索引获取 frozen 索引
    #[inline]
    pub fn from_original_cell(&self, original_idx: usize) -> u32 {
        self.cell_inv_permutation.get(original_idx).copied().unwrap_or(original_idx as u32)
    }

    /// 设置单元排列（用于网格重排序后更新映射）
    pub fn set_permutation(&mut self, perm: Vec<u32>) {
        let n = perm.len();
        let mut inv = vec![0u32; n];
        for (frozen, &orig) in perm.iter().enumerate() {
            if (orig as usize) < n {
                inv[orig as usize] = frozen as u32;
            }
        }
        self.cell_permutation = perm;
        self.cell_inv_permutation = inv;
    }

    // =========================================================================
    // 节点访问
    // =========================================================================

    /// 获取节点3D坐标（几何数据，返回 f64）
    #[inline]
    pub fn node_coords(&self, node: usize) -> Point3D {
        self.node_coords[node]
    }

    /// 获取节点2D坐标 (x, y)（几何数据）
    #[inline]
    pub fn node_xy(&self, node: usize) -> Point2D {
        self.node_coords[node].xy()
    }

    /// 获取节点高程 (z)（几何数据）
    #[inline]
    pub fn node_z(&self, node: usize) -> f64 {
        self.node_coords[node].z
    }

    // =========================================================================
    // 范围迭代
    // =========================================================================

    /// 内部面索引范围
    #[inline]
    pub fn interior_faces(&self) -> std::ops::Range<usize> {
        0..self.n_interior_faces
    }

    /// 边界面索引范围
    #[inline]
    pub fn boundary_faces(&self) -> std::ops::Range<usize> {
        self.n_interior_faces..self.n_faces
    }

    /// 单元索引范围
    #[inline]
    pub fn cells(&self) -> std::ops::Range<usize> {
        0..self.n_cells
    }

    /// 节点索引范围
    #[inline]
    pub fn nodes(&self) -> std::ops::Range<usize> {
        0..self.n_nodes
    }

    // =========================================================================
    // 统计信息
    // =========================================================================

    /// 计算统计信息
    pub fn statistics(&self) -> MeshStatistics<S> {
        let mut min_area = S::MAX;
        let mut max_area = S::ZERO;
        let mut total_area = S::ZERO;

        for &area in &self.cell_area {
            min_area = min_area.min(area);
            max_area = max_area.max(area);
            total_area = total_area + area;
        }

        let mut min_length = S::MAX;
        let mut max_length = S::ZERO;

        for &len in &self.face_length {
            min_length = min_length.min(len);
            max_length = max_length.max(len);
        }

        MeshStatistics {
            n_cells: self.n_cells,
            n_faces: self.n_faces,
            n_interior_faces: self.n_interior_faces,
            n_boundary_faces: self.n_faces - self.n_interior_faces,
            n_nodes: self.n_nodes,
            total_area,
            min_cell_area: min_area,
            max_cell_area: max_area,
            min_edge_length: min_length,
            max_edge_length: max_length,
        }
    }

    /// 验证网格完整性
    pub fn validate(&self) -> Result<(), String> {
        // 检查数组长度
        if self.cell_center.len() != self.n_cells {
            return Err(format!(
                "cell_center length {} != n_cells {}",
                self.cell_center.len(),
                self.n_cells
            ));
        }

        if self.node_coords.len() != self.n_nodes {
            return Err(format!(
                "node_coords length {} != n_nodes {}",
                self.node_coords.len(),
                self.n_nodes
            ));
        }

        if self.face_center.len() != self.n_faces {
            return Err(format!(
                "face_center length {} != n_faces {}",
                self.face_center.len(),
                self.n_faces
            ));
        }

        // 检查偏移数组
        if self.cell_node_offsets.len() != self.n_cells + 1 {
            return Err("cell_node_offsets length mismatch".to_string());
        }

        // 检查 owner/neighbor
        for (i, &owner) in self.face_owner.iter().enumerate() {
            if owner as usize >= self.n_cells {
                return Err(format!("face {} owner {} out of range", i, owner));
            }
        }

        for (i, &neighbor) in self.face_neighbor.iter().enumerate() {
            if neighbor != u32::MAX && neighbor as usize >= self.n_cells {
                return Err(format!("face {} neighbor {} out of range", i, neighbor));
            }
        }

        Ok(())
    }

    /// 计算网格中心点（几何中心）
    pub fn centroid(&self) -> Point2D {
        if self.n_cells == 0 {
            return Point2D::new(0.0, 0.0);
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut total_area = 0.0_f64; // 使用 f64 累加器避免精度损失

        for cell in 0..self.n_cells {
            let center = self.cell_center(cell);
            let area = self.cell_area(cell).to_f64(); // 转换为 f64
            sum_x += center.x * area;
            sum_y += center.y * area;
            total_area += area;
        }

        if total_area > 0.0 {
            Point2D::new(sum_x / total_area, sum_y / total_area)
        } else {
            Point2D::new(0.0, 0.0)
        }
    }
}

/// 网格统计信息（泛型版本）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshStatistics<S: RuntimeScalar> {
    pub n_cells: usize,
    pub n_faces: usize,
    pub n_interior_faces: usize,
    pub n_boundary_faces: usize,
    pub n_nodes: usize,
    pub total_area: S,
    pub min_cell_area: S,
    pub max_cell_area: S,
    pub min_edge_length: S,
    pub max_edge_length: S,
}

impl<S: RuntimeScalar> std::fmt::Display for MeshStatistics<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== 网格统计 ===")?;
        writeln!(f, "单元数: {}", self.n_cells)?;
        writeln!(
            f,
            "面数: {} (内部: {}, 边界: {})",
            self.n_faces, self.n_interior_faces, self.n_boundary_faces
        )?;
        writeln!(f, "节点数: {}", self.n_nodes)?;
        writeln!(f, "总面积: {:.2}", self.total_area.to_f64())?;
        writeln!(
            f,
            "单元面积: [{:.2}, {:.2}]",
            self.min_cell_area.to_f64(), self.max_cell_area.to_f64()
        )?;
        writeln!(
            f,
            "边长: [{:.2}, {:.2}]",
            self.min_edge_length.to_f64(), self.max_edge_length.to_f64()
        )
    }
}

// =========================================================================
// MeshAccess trait 实现
// =========================================================================

use crate::traits::{MeshAccess, MeshTopology};

impl<S: RuntimeScalar> MeshAccess for FrozenMesh<S> {
    #[inline]
    fn n_cells(&self) -> usize {
        self.n_cells
    }

    #[inline]
    fn n_faces(&self) -> usize {
        self.n_faces
    }

    #[inline]
    fn n_internal_faces(&self) -> usize {
        self.n_interior_faces
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    fn cell_centroid(&self, cell: usize) -> Point2D {
        self.cell_center[cell]
    }

    #[inline]
    fn cell_area(&self, cell: usize) -> f64 {
        self.cell_area[cell].to_f64()
    }

    #[inline]
    fn face_centroid(&self, face: usize) -> Point2D {
        self.face_center[face]
    }

    #[inline]
    fn face_length(&self, face: usize) -> f64 {
        self.face_length[face].to_f64()
    }

    #[inline]
    fn face_normal(&self, face: usize) -> Point3D {
        self.face_normal[face]
    }

    #[inline]
    fn node_position(&self, node: usize) -> Point3D {
        self.node_coords[node]
    }

    #[inline]
    fn cell_bed_elevation(&self, cell: usize) -> f64 {
        self.cell_z_bed[cell].to_f64()
    }

    #[inline]
    fn face_owner(&self, face: usize) -> usize {
        self.face_owner[face] as usize
    }

    #[inline]
    fn face_neighbor(&self, face: usize) -> Option<usize> {
        let n = self.face_neighbor[face];
        if n == u32::MAX {
            None
        } else {
            Some(n as usize)
        }
    }

    #[inline]
    fn cell_face_indices(&self, cell: usize) -> &[u32] {
        let start = self.cell_face_offsets[cell];
        let end = self.cell_face_offsets[cell + 1];
        &self.cell_face_indices[start..end]
    }

    #[inline]
    fn cell_neighbor_indices(&self, cell: usize) -> &[u32] {
        let start = self.cell_neighbor_offsets[cell];
        let end = self.cell_neighbor_offsets[cell + 1];
        &self.cell_neighbor_indices[start..end]
    }

    #[inline]
    fn cell_node_indices(&self, cell: usize) -> &[u32] {
        let start = self.cell_node_offsets[cell];
        let end = self.cell_node_offsets[cell + 1];
        &self.cell_node_indices[start..end]
    }

    #[inline]
    fn boundary_id(&self, face: usize) -> Option<usize> {
        self.face_boundary_id
            .get(face)
            .and_then(|opt| opt.map(|id| id as usize))
    }

    #[inline]
    fn boundary_name(&self, boundary_id: usize) -> Option<&str> {
        self.boundary_names.get(boundary_id).map(|s| s.as_str())
    }

    #[inline]
    fn all_cell_centroids(&self) -> &[Point2D] {
        &self.cell_center
    }

    #[inline]
    fn all_cell_areas(&self) -> &[f64] {
        &self.cell_area
    }

    #[inline]
    fn all_cell_bed_elevations(&self) -> &[f64] {
        &self.cell_z_bed
    }

    #[inline]
    fn face_z_left(&self, face: usize) -> f64 {
        self.face_z_left[face].to_f64()
    }

    #[inline]
    fn face_z_right(&self, face: usize) -> f64 {
        self.face_z_right[face].to_f64()
    }
}

impl<S: RuntimeScalar> MeshTopology for FrozenMesh<S> {
    #[inline]
    fn face_o2n_distance(&self, face: usize) -> f64 {
        self.face_dist_o2n[face].to_f64()
    }

    #[inline]
    fn face_delta_owner(&self, face: usize) -> Point2D {
        self.face_delta_owner[face]
    }

    #[inline]
    fn face_delta_neighbor(&self, face: usize) -> Point2D {
        self.face_delta_neighbor[face]
    }

    #[inline]
    fn min_cell_size(&self) -> f64 {
        self.min_cell_size.to_f64()
    }

    #[inline]
    fn max_cell_size(&self) -> f64 {
        self.max_cell_size.to_f64()
    }
}

// =========================================================================
// 空间查询扩展方法
// =========================================================================

use crate::locator::MeshLocator;
use crate::spatial_index::MeshSpatialIndex;

impl<S: RuntimeScalar> FrozenMesh<S> {
    // =========================================================================
    // 空间索引创建
    // =========================================================================

    /// 创建网格空间索引
    ///
    /// 基于 R-Tree 的空间索引，支持高效的点定位和范围查询。
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let mesh = FrozenMesh::load("mesh.bin")?;
    /// let index = mesh.create_spatial_index();
    /// let cell = index.locate_point(Point2D::new(100.0, 200.0));
    /// ```
    pub fn create_spatial_index(&self) -> MeshSpatialIndex {
        MeshSpatialIndex::build(self.n_cells, |i| self.get_cell_vertices(i))
    }

    /// 创建网格定位器（支持泛型）
    pub fn create_locator(&self) -> MeshLocator<'_, S> {
        MeshLocator::new(self)
    }

    // =========================================================================
    // 单元几何查询
    // =========================================================================

    /// 获取单元的顶点坐标列表（2D）
    ///
    /// 返回按逆时针顺序排列的顶点坐标。
    ///
    /// # 参数
    ///
    /// * `cell` - 单元索引
    ///
    /// # 返回
    ///
    /// 顶点坐标数组（通常为3-4个顶点）
    pub fn get_cell_vertices(&self, cell: usize) -> Vec<Point2D> {
        self.cell_nodes(cell)
            .iter()
            .map(|&node_idx| self.node_xy(node_idx as usize))
            .collect()
    }

    /// 获取单元的顶点坐标列表（3D）
    pub fn get_cell_vertices_3d(&self, cell: usize) -> Vec<Point3D> {
        self.cell_nodes(cell)
            .iter()
            .map(|&node_idx| self.node_coords(node_idx as usize))
            .collect()
    }

    /// 计算点相对于三角形单元的重心坐标
    ///
    /// 重心坐标 (λ₁, λ₂, λ₃) 满足：
    /// - λ₁ + λ₂ + λ₃ = 1
    /// - 点 P = λ₁·V₁ + λ₂·V₂ + λ₃·V₃
    ///
    /// # 参数
    ///
    /// * `cell` - 单元索引（必须是三角形）
    /// * `point` - 查询点
    ///
    /// # 返回
    ///
    /// - `Some((λ₁, λ₂, λ₃))` - 成功计算重心坐标
    /// - `None` - 单元不是三角形或面积退化
    ///
    /// # 判定规则
    ///
    /// - 所有 λ ∈ [0, 1]: 点在单元内部
    /// - 任一 λ < 0: 点在单元外部
    /// - 任一 λ = 0: 点在边上
    pub fn compute_barycentric(&self, cell: usize, point: Point2D) -> Option<(f64, f64, f64)> {
        let nodes = self.cell_nodes(cell);
        if nodes.len() != 3 {
            return None;
        }

        let v0 = self.node_xy(nodes[0] as usize);
        let v1 = self.node_xy(nodes[1] as usize);
        let v2 = self.node_xy(nodes[2] as usize);

        // 向量
        let v0v1 = Point2D::new(v1.x - v0.x, v1.y - v0.y);
        let v0v2 = Point2D::new(v2.x - v0.x, v2.y - v0.y);
        let v0p = Point2D::new(point.x - v0.x, point.y - v0.y);

        // 计算点积
        let dot00 = v0v1.x * v0v1.x + v0v1.y * v0v1.y;
        let dot01 = v0v1.x * v0v2.x + v0v1.y * v0v2.y;
        let dot02 = v0v1.x * v0p.x + v0v1.y * v0p.y;
        let dot11 = v0v2.x * v0v2.x + v0v2.y * v0v2.y;
        let dot12 = v0v2.x * v0p.x + v0v2.y * v0p.y;

        // 计算重心坐标
        let denom = dot00 * dot11 - dot01 * dot01;
        if denom.abs() < 1e-12 {
            return None; // 退化三角形
        }

        let inv_denom = 1.0 / denom;
        let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
        let w = 1.0 - u - v;

        Some((w, u, v))
    }

    /// 判断点是否在单元内部
    ///
    /// 使用射线法检测点是否在多边形内部。
    ///
    /// # 参数
    ///
    /// * `cell` - 单元索引
    /// * `point` - 查询点
    /// * `tolerance` - 边界容差（默认使用 1e-10）
    pub fn point_in_cell(&self, cell: usize, point: Point2D, tolerance: f64) -> bool {
        let vertices = self.get_cell_vertices(cell);
        let n = vertices.len();
        if n < 3 {
            return false;
        }

        // 射线法
        let mut inside = false;

        for i in 0..n {
            let j = (i + 1) % n;
            let vi = &vertices[i];
            let vj = &vertices[j];

            // 检查是否与水平射线相交
            if ((vi.y > point.y) != (vj.y > point.y))
                && (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x - tolerance)
            {
                inside = !inside;
            }
        }

        inside
    }

    // =========================================================================
    // 边界查询
    // =========================================================================

    /// 查找距离点最近的边界面
    ///
    /// # 参数
    ///
    /// * `point` - 查询点
    ///
    /// # 返回
    ///
    /// `(面索引, 最短距离)` 或 `None`（如果没有边界面）
    pub fn find_nearest_boundary_face(&self, point: Point2D) -> Option<(usize, f64)> {
        if self.n_interior_faces >= self.n_faces {
            return None;
        }

        let mut best_face = None;
        let mut best_dist = f64::INFINITY;

        for face in self.boundary_faces() {
            let center = self.face_center(face);
            let dx = center.x - point.x;
            let dy = center.y - point.y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < best_dist {
                best_dist = dist;
                best_face = Some(face);
            }
        }

        best_face.map(|f| (f, best_dist))
    }

    /// 获取指定边界的所有面索引
    ///
    /// # 参数
    ///
    /// * `boundary_name` - 边界名称
    ///
    /// # 返回
    ///
    /// 属于该边界的面索引列表
    pub fn get_boundary_faces_by_name(&self, boundary_name: &str) -> Vec<usize> {
        // 查找边界ID
        let boundary_id = self
            .boundary_names
            .iter()
            .position(|name| name == boundary_name);

        match boundary_id {
            Some(id) => self
                .boundary_faces()
                .filter(|&face| {
                    self.face_boundary_id
                        .get(face)
                        .and_then(|opt| *opt)
                        .map(|bid| bid as usize == id)
                        .unwrap_or(false)
                })
                .collect(),
            None => Vec::new(),
        }
    }

    // =========================================================================
    // 邻居查询
    // =========================================================================

    /// 获取单元的所有有效邻居（排除边界）
    ///
    /// # 参数
    ///
    /// * `cell` - 单元索引
    ///
    /// # 返回
    ///
    /// 有效邻居单元索引列表
    pub fn get_valid_neighbors(&self, cell: usize) -> Vec<usize> {
        self.cell_neighbors(cell)
            .iter()
            .filter(|&&n| n != u32::MAX)
            .map(|&n| n as usize)
            .collect()
    }

    /// 获取单元的 n 阶邻居（n 跳可达的所有单元）
    ///
    /// # 参数
    ///
    /// * `cell` - 起始单元索引
    /// * `order` - 邻居阶数（1 = 直接邻居，2 = 邻居的邻居，...）
    ///
    /// # 返回
    ///
    /// 所有 n 阶内可达的单元索引集合
    pub fn get_n_order_neighbors(&self, cell: usize, order: usize) -> Vec<usize> {
        use std::collections::HashSet;

        if order == 0 {
            return vec![cell];
        }

        let mut visited = HashSet::new();
        let mut current_layer = vec![cell];
        visited.insert(cell);

        for _ in 0..order {
            let mut next_layer = Vec::new();

            for &c in &current_layer {
                for neighbor in self.get_valid_neighbors(c) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        next_layer.push(neighbor);
                    }
                }
            }

            current_layer = next_layer;
        }

        visited.into_iter().filter(|&c| c != cell).collect()
    }

    // =========================================================================
    // 几何计算
    // =========================================================================

    /// 计算两个单元中心之间的距离
    #[inline]
    pub fn cell_distance(&self, cell1: usize, cell2: usize) -> f64 {
        let c1 = self.cell_center(cell1);
        let c2 = self.cell_center(cell2);
        let dx = c2.x - c1.x;
        let dy = c2.y - c1.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// 计算网格边界框
    pub fn bounding_box(&self) -> (Point2D, Point2D) {
        if self.n_nodes == 0 {
            return (Point2D::new(0.0, 0.0), Point2D::new(0.0, 0.0));
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for &coord in &self.node_coords {
            min_x = min_x.min(coord.x);
            min_y = min_y.min(coord.y);
            max_x = max_x.max(coord.x);
            max_y = max_y.max(coord.y);
        }

        (Point2D::new(min_x, min_y), Point2D::new(max_x, max_y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::RuntimeScalar;

    #[test]
    fn test_empty_frozen_mesh() {
        let mesh: FrozenMesh<f64> = FrozenMesh::empty();
        assert_eq!(mesh.n_cells(), 0);
        assert_eq!(mesh.n_faces(), 0);
        assert_eq!(mesh.n_nodes(), 0);
    }

    #[test]
    fn test_frozen_mesh_f32() {
        let mesh: FrozenMesh<f32> = FrozenMesh::empty_with_cells(5);
        assert_eq!(mesh.n_cells(), 5);
        assert_eq!(mesh.cell_area(0), f32::ONE);
    }

    #[test]
    fn test_validate_empty() {
        let mesh: FrozenMesh<f64> = FrozenMesh::empty();
        assert!(mesh.validate().is_ok());
    }

    #[test]
    fn test_mesh_access_trait() {
        let mesh: FrozenMesh<f64> = FrozenMesh::empty_with_cells(5);
        
        // 通过 trait 访问
        fn check_mesh<M: MeshAccess>(m: &M) -> usize {
            m.n_cells()
        }
        
        assert_eq!(check_mesh(&mesh), 5);
    }

    #[test]
    fn test_statistics_f32() {
        let mesh: FrozenMesh<f32> = FrozenMesh::empty_with_cells(3);
        let stats = mesh.statistics();
        assert_eq!(stats.n_cells, 3);
        assert_eq!(stats.total_area.to_f64(), 3.0);
    }

    #[test]
    fn test_centroid() {
        let mesh: FrozenMesh<f64> = FrozenMesh::empty_with_cells(2);
        // 设置测试数据
        let mesh = FrozenMesh {
            cell_center: vec![Point2D::new(0.0, 0.0), Point2D::new(2.0, 2.0)],
            cell_area: vec![1.0, 1.0],
            ..mesh
        };
        let centroid = mesh.centroid();
        assert_eq!(centroid.x, 1.0);
        assert_eq!(centroid.y, 1.0);
    }
}