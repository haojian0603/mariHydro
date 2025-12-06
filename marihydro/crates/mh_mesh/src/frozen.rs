// marihydro\crates\mh_mesh\src/frozen.rs

//! 冻结网格
//!
//! 从半边网格导出的只读 SoA 布局，优化计算性能。
//!
//! # 设计要点
//!
//! 1. **SoA布局**: 类似旧 UnstructuredMesh 的数组布局
//! 2. **只读**: 冻结后不可修改
//! 3. **空间索引**: 内置 R-tree 用于空间查询
//! 4. **零拷贝序列化**: 支持 mmap 加载

use mh_geo::{Point2D, Point3D};
use serde::{Deserialize, Serialize};

/// 冻结网格
///
/// 从 HalfEdgeMesh 导出的只读计算用网格
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenMesh {
    // ===== 节点数据 =====
    /// 节点数量
    pub n_nodes: usize,
    /// 节点坐标 (3D)
    pub node_coords: Vec<Point3D>,

    // ===== 单元数据 =====
    /// 单元数量
    pub n_cells: usize,
    /// 单元中心坐标（2D，用于空间查询）
    pub cell_center: Vec<Point2D>,
    /// 单元面积
    pub cell_area: Vec<f64>,
    /// 单元底床高程
    pub cell_z_bed: Vec<f64>,
    /// 单元节点索引 (压缩格式: offsets + indices)
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
    /// 面中心坐标（2D）
    pub face_center: Vec<Point2D>,
    /// 面法向量 (3D，用于坡度计算)
    pub face_normal: Vec<Point3D>,
    /// 面长度
    pub face_length: Vec<f64>,
    /// 面左侧高程
    pub face_z_left: Vec<f64>,
    /// 面右侧高程
    pub face_z_right: Vec<f64>,
    /// 面 owner 单元索引
    pub face_owner: Vec<u32>,
    /// 面 neighbor 单元索引 (u32::MAX 表示边界)
    pub face_neighbor: Vec<u32>,
    /// 面到 owner 中心的向量（2D）
    pub face_delta_owner: Vec<Point2D>,
    /// 面到 neighbor 中心的向量（2D）
    pub face_delta_neighbor: Vec<Point2D>,
    /// owner 到 neighbor 的距离
    pub face_dist_o2n: Vec<f64>,

    // ===== 边界数据 =====
    /// 边界面索引列表
    pub boundary_face_indices: Vec<u32>,
    /// 边界名称
    pub boundary_names: Vec<String>,
    /// 面的边界ID (None表示内部面)
    pub face_boundary_id: Vec<Option<u32>>,

    // ===== 统计 =====
    /// 最小单元尺寸
    pub min_cell_size: f64,
    /// 最大单元尺寸
    pub max_cell_size: f64,
}

impl Default for FrozenMesh {
    fn default() -> Self {
        Self::empty()
    }
}

impl FrozenMesh {
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
            min_cell_size: f64::MAX,
            max_cell_size: 0.0,
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
            cell_area: vec![1.0; n_cells],
            cell_z_bed: vec![0.0; n_cells],
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
            min_cell_size: 1.0,
            max_cell_size: 1.0,
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

    /// 获取单元中心
    #[inline]
    pub fn cell_center(&self, cell: usize) -> Point2D {
        self.cell_center[cell]
    }

    /// 获取单元面积
    #[inline]
    pub fn cell_area(&self, cell: usize) -> f64 {
        self.cell_area[cell]
    }

    /// 获取单元底床高程
    #[inline]
    pub fn cell_z_bed(&self, cell: usize) -> f64 {
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

    /// 获取面中心
    #[inline]
    pub fn face_center(&self, face: usize) -> Point2D {
        self.face_center[face]
    }

    /// 获取面法向量（3D）
    #[inline]
    pub fn face_normal(&self, face: usize) -> Point3D {
        self.face_normal[face]
    }

    /// 获取面长度
    #[inline]
    pub fn face_length(&self, face: usize) -> f64 {
        self.face_length[face]
    }

    /// 获取面 owner
    #[inline]
    pub fn face_owner(&self, face: usize) -> u32 {
        self.face_owner[face]
    }

    /// 获取面 neighbor
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
    // 节点访问
    // =========================================================================

    /// 获取节点3D坐标
    #[inline]
    pub fn node_coords(&self, node: usize) -> Point3D {
        self.node_coords[node]
    }

    /// 获取节点2D坐标 (x, y)
    #[inline]
    pub fn node_xy(&self, node: usize) -> Point2D {
        self.node_coords[node].xy()
    }

    /// 获取节点高程 (z)
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
    pub fn statistics(&self) -> MeshStatistics {
        let mut min_area = f64::MAX;
        let mut max_area = f64::MIN;
        let mut total_area = 0.0;

        for &area in &self.cell_area {
            min_area = min_area.min(area);
            max_area = max_area.max(area);
            total_area += area;
        }

        let mut min_length = f64::MAX;
        let mut max_length = f64::MIN;

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
}

/// 网格统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshStatistics {
    pub n_cells: usize,
    pub n_faces: usize,
    pub n_interior_faces: usize,
    pub n_boundary_faces: usize,
    pub n_nodes: usize,
    pub total_area: f64,
    pub min_cell_area: f64,
    pub max_cell_area: f64,
    pub min_edge_length: f64,
    pub max_edge_length: f64,
}

impl std::fmt::Display for MeshStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== 网格统计 ===")?;
        writeln!(f, "单元数: {}", self.n_cells)?;
        writeln!(
            f,
            "面数: {} (内部: {}, 边界: {})",
            self.n_faces, self.n_interior_faces, self.n_boundary_faces
        )?;
        writeln!(f, "节点数: {}", self.n_nodes)?;
        writeln!(f, "总面积: {:.2} m²", self.total_area)?;
        writeln!(
            f,
            "单元面积: [{:.2}, {:.2}] m²",
            self.min_cell_area, self.max_cell_area
        )?;
        writeln!(
            f,
            "边长: [{:.2}, {:.2}] m",
            self.min_edge_length, self.max_edge_length
        )
    }
}

// =========================================================================
// MeshAccess trait 实现
// =========================================================================

use crate::traits::{MeshAccess, MeshTopology};

impl MeshAccess for FrozenMesh {
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
        self.cell_area[cell]
    }

    #[inline]
    fn face_centroid(&self, face: usize) -> Point2D {
        self.face_center[face]
    }

    #[inline]
    fn face_length(&self, face: usize) -> f64 {
        self.face_length[face]
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
        self.cell_z_bed[cell]
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
        self.face_z_left[face]
    }

    #[inline]
    fn face_z_right(&self, face: usize) -> f64 {
        self.face_z_right[face]
    }
}

impl MeshTopology for FrozenMesh {
    #[inline]
    fn face_o2n_distance(&self, face: usize) -> f64 {
        self.face_dist_o2n[face]
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
        self.min_cell_size
    }

    #[inline]
    fn max_cell_size(&self) -> f64 {
        self.max_cell_size
    }
}

// =========================================================================
// 空间查询扩展方法
// =========================================================================

use crate::locator::{LocateResult, MeshLocator};
use crate::spatial_index::MeshSpatialIndex;

impl FrozenMesh {
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

    /// 创建网格定位器
    ///
    /// 高级定位器，支持精确点定位和批量查询。
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let mesh = FrozenMesh::load("mesh.bin")?;
    /// let locator = mesh.create_locator();
    /// match locator.locate(Point2D::new(100.0, 200.0)) {
    ///     LocateResult::InCell { cell, .. } => println!("在单元 {} 内", cell),
    ///     LocateResult::OnBoundary { .. } => println!("在边界上"),
    ///     LocateResult::Outside { .. } => println!("在网格外"),
    /// }
    /// ```
    pub fn create_locator(&self) -> MeshLocator {
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

        // 射线法：从点向右发射水平射线，计算与多边形边的交点数
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

    /// 计算网格中心点
    pub fn centroid(&self) -> Point2D {
        if self.n_cells == 0 {
            return Point2D::new(0.0, 0.0);
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut total_area = 0.0;

        for cell in 0..self.n_cells {
            let center = self.cell_center(cell);
            let area = self.cell_area(cell);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_frozen_mesh() {
        let mesh = FrozenMesh::empty();
        assert_eq!(mesh.n_cells(), 0);
        assert_eq!(mesh.n_faces(), 0);
        assert_eq!(mesh.n_nodes(), 0);
    }

    #[test]
    fn test_validate_empty() {
        let mesh = FrozenMesh::empty();
        assert!(mesh.validate().is_ok());
    }

    #[test]
    fn test_mesh_access_trait() {
        let mesh = FrozenMesh::empty_with_cells(5);
        
        // 通过 trait 访问
        fn check_mesh<M: MeshAccess>(m: &M) -> usize {
            m.n_cells()
        }
        
        assert_eq!(check_mesh(&mesh), 5);
    }

    #[test]
    fn test_mesh_topology_trait() {
        let mesh = FrozenMesh::empty_with_cells(5);
        
        // 通过 trait 访问
        fn check_topology<M: MeshTopology>(m: &M) -> f64 {
            m.min_cell_size()
        }
        
        assert_eq!(check_topology(&mesh), 1.0);
    }
}

