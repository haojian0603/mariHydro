// crates/mh_mesh/src/traits.rs
#![allow(clippy::items_after_test_module)]

//! 网格抽象接口（MeshAccess & MeshTopology）
//!
//! 定义非结构化网格的只读访问接口，支持三角形、四边形及多边形单元。
//! 所有实现必须保证线程安全（Send + Sync），以支持并行计算。
//!
//! # 架构层级
//! 
//! 本模块属于 Layer 2（运行时抽象层），为 Layer 3 引擎层提供统一的网格访问契约。
//! 所有 Layer 3 组件（如 ShallowWaterSolver、FluxCalculator）必须通过此 trait 操作网格，
//! 禁止直接依赖 FrozenMesh 或 HalfEdgeMesh 的具体实现。
//!
//! # 坐标系约定
//!
//! - 笛卡尔坐标系，单位为米
//! - X 轴指向东，Y 轴指向北，Z 轴指向上
//! - 面法向量从 owner 单元指向 neighbor 单元（已归一化）
//!
//! # 索引约束
//!
//! - 单元索引: `0..n_cells()`
//! - 面索引: `0..n_faces()`，其中 `0..n_internal_faces()` 为内部面
//! - 节点索引: `0..n_nodes()`
//!
//! # 线程安全
//!
//! 所有 trait 方法要求实现 `Send + Sync`，确保可在 Rayon 并行迭代器中安全使用。
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_mesh::traits::{MeshAccess, MeshAccessExt};
//! use mh_mesh::FrozenMesh;
//!
//! // 通用算法：计算总面积
//! fn compute_total_area<M: MeshAccess>(mesh: &M) -> f64 {
//!     (0..mesh.n_cells()).map(|i| mesh.cell_area(i)).sum()
//! }
//!
//! // 统计淹没单元
//! fn count_wet_cells<M: MeshAccess>(mesh: &M, depth: &[f64], threshold: f64) -> usize {
//!     (0..mesh.n_cells())
//!         .filter(|&i| depth[i] > threshold)
//!         .count()
//! }
//! ```

use mh_geo::{Point2D, Point3D};
use mh_runtime::prelude::*;

// =========================================================================
// MeshAccess - 网格只读访问接口
// =========================================================================

/// 网格访问接口（只读）
///
/// 提供对网格几何、拓扑和物理场的统一访问。实现类型必须保证 O(1) 时间复杂度的几何查询。
pub trait MeshAccess<B: Backend>: Send + Sync {
    /// 获取后端引用
    fn backend(&self) -> &B;
    // ===== 基本计数 =====

    /// 单元总数
    fn n_cells(&self) -> usize;

    /// 面总数（内部面 + 边界面）
    fn n_faces(&self) -> usize;

    /// 内部面数量（连接两个单元）
    fn n_internal_faces(&self) -> usize;

    /// 边界面数量
    #[inline]
    fn n_boundary_faces(&self) -> usize {
        self.n_faces() - self.n_internal_faces()
    }

    /// 节点总数
    fn n_nodes(&self) -> usize;

    // ===== 几何查询 =====

    /// 单元质心（2D 坐标）
    fn cell_centroid(&self, cell: usize) -> Point2D;

    /// 单元面积
    fn cell_area(&self, cell: usize) -> B::Scalar;

    /// 面中点（2D 坐标）
    fn face_centroid(&self, face: usize) -> Point2D;

    /// 面长度（边长）
    fn face_length(&self, face: usize) -> B::Scalar;

    /// 面外法向量（3D，从 owner 指向 neighbor，已归一化）
    fn face_normal(&self, face: usize) -> Point3D;

    /// 面外法向量（2D，仅 x/y 分量）
    #[inline]
    fn face_normal_2d(&self, face: usize) -> Point2D {
        let n = self.face_normal(face);
        Point2D::new(n.x, n.y)
    }

    /// 节点坐标（3D）
    fn node_position(&self, node: usize) -> Point3D;

    /// 单元底床高程（用于水位-水深转换）
    fn cell_bed_elevation(&self, cell: usize) -> B::Scalar;

    // ===== 拓扑查询 =====

    /// 面的 owner 单元索引（总是有效）
    fn face_owner(&self, face: usize) -> usize;

    /// 面的 neighbor 单元索引（边界面返回 None）
    fn face_neighbor(&self, face: usize) -> Option<usize>;

    /// 面是否为边界面
    #[inline]
    fn is_boundary_face(&self, face: usize) -> bool {
        self.face_neighbor(face).is_none()
    }

    /// 面是否为内部面
    #[inline]
    fn is_internal_face(&self, face: usize) -> bool {
        self.face_neighbor(face).is_some()
    }

    /// 单元的相邻面索引列表
    fn cell_face_indices(&self, cell: usize) -> &[u32];

    /// 单元的相邻单元索引列表（无邻居时为 u32::MAX）
    fn cell_neighbor_indices(&self, cell: usize) -> &[u32];

    /// 单元的顶点索引列表
    fn cell_node_indices(&self, cell: usize) -> &[u32];

    // ===== 边界信息 =====

    /// 边界面的边界标识索引
    fn boundary_id(&self, face: usize) -> Option<usize>;

    /// 边界名称查询
    fn boundary_name(&self, boundary_id: usize) -> Option<&str>;

    // ===== 批量访问 =====

    /// 所有单元质心切片（连续内存，用于 GPU 传输）
    fn all_cell_centroids(&self) -> &[Point2D];

    /// 所有单元面积（运行时转换为 f64，拥有所有权）
    fn all_cell_areas(&self) -> Vec<B::Scalar>;

    /// 所有单元底床高程（运行时转换为 f64，拥有所有权）
    fn all_cell_bed_elevations(&self) -> Vec<B::Scalar>;

    // ===== 面高程 =====

    /// 面左侧（owner）高程
    fn face_z_left(&self, face: usize) -> B::Scalar;

    /// 面右侧（neighbor）高程
    fn face_z_right(&self, face: usize) -> B::Scalar;
}

// =========================================================================
// MeshTopology - 网格拓扑计算接口
// =========================================================================

/// 网格拓扑计算接口
///
/// 基于 MeshAccess 提供高级拓扑计算，如距离、权重、特征长度等。
pub trait MeshTopology<B: Backend>: MeshAccess<B> {
    /// 两单元中心距离
    #[inline]
    fn cell_distance(&self, cell1: usize, cell2: usize) -> B::Scalar {
        let c1 = self.cell_centroid(cell1);
        let c2 = self.cell_centroid(cell2);
        let dx = c2.x - c1.x;
        let dy = c2.y - c1.y;
        let dist = (dx * dx + dy * dy).sqrt();
        self.backend()
            .config_scalar(dist, "MeshMetadata.cell_center_distance")
    }

    /// 单元中心到面中心的距离
    #[inline]
    fn cell_to_face_distance(&self, cell: usize, face: usize) -> B::Scalar {
        let cc = self.cell_centroid(cell);
        let fc = self.face_centroid(face);
        let dx = fc.x - cc.x;
        let dy = fc.y - cc.y;
        let dist = (dx * dx + dy * dy).sqrt();
        self.backend()
            .config_scalar(dist, "MeshMetadata.face_center_distance")
    }

    /// 面的 owner 到 neighbor 中心距离
    fn face_o2n_distance(&self, face: usize) -> B::Scalar;

    /// 面的几何权重（用于梯度插值，owner 侧权重）
    #[inline]
    fn face_weight(&self, face: usize) -> B::Scalar {
        let owner = self.face_owner(face);
        let neighbor = match self.face_neighbor(face) {
            Some(n) => n,
            None => return B::Scalar::ONE,
        };

        let d_owner = self.cell_to_face_distance(owner, face);
        let d_neighbor = self.cell_to_face_distance(neighbor, face);
        let total = d_owner + d_neighbor;

        let eps = self
            .backend()
            .config_scalar(1e-14, "MeshMetadata.interpolation_weight.eps");
        if total < eps { B::Scalar::HALF } else { d_neighbor / total }
    }

    /// 单元特征长度（sqrt(面积)）
    #[inline]
    fn characteristic_length(&self, cell: usize) -> B::Scalar {
        self.cell_area(cell).sqrt()
    }

    /// 面 owner 中心到面中心的向量（2D）
    fn face_delta_owner(&self, face: usize) -> Point2D;

    /// 面 neighbor 中心到面中心的向量（2D）
    fn face_delta_neighbor(&self, face: usize) -> Point2D;

    /// 全局最小单元尺寸
    fn min_cell_size(&self) -> B::Scalar;

    /// 全局最大单元尺寸
    fn max_cell_size(&self) -> B::Scalar;
}

// =========================================================================
// 辅助几何结构体
// =========================================================================

/// 单元几何信息（用于物理场计算）
#[derive(Debug, Clone, Copy)]
pub struct CellGeometry<B: Backend> {
    /// 质心坐标
    pub centroid: Point2D,
    /// 面积
    pub area: B::Scalar,
    /// 特征长度（sqrt(area)）
    pub characteristic_length: B::Scalar,
    /// 底床高程
    pub bed_elevation: B::Scalar,
}

/// 面几何信息（用于通量计算）
#[derive(Debug, Clone, Copy)]
pub struct FaceGeometry<B: Backend> {
    /// 面中心
    pub centroid: Point2D,
    /// 面长度
    pub length: B::Scalar,
    /// 法向量（2D）
    pub normal: Point2D,
    /// Owner 单元索引
    pub owner: usize,
    /// Neighbor 单元索引（边界面为 None）
    pub neighbor: Option<usize>,
}

// =========================================================================
// 网格验证报告
// =========================================================================

/// 网格验证报告（拓扑和质量检查）
#[derive(Debug, Clone)]
pub struct ValidationReport<B: Backend> {
    /// 是否通过验证
    pub is_valid: bool,
    /// 错误列表（导致计算失败）
    pub errors: Vec<String>,
    /// 警告列表（可能影响精度）
    pub warnings: Vec<String>,
    /// 统计信息
    pub stats: ValidationStats<B>,
}

/// 验证统计信息
#[derive(Debug, Clone)]
pub struct ValidationStats<B: Backend> {
    /// 单元数
    pub n_cells: usize,
    /// 面数
    pub n_faces: usize,
    /// 节点数
    pub n_nodes: usize,
    /// 最小面积
    pub min_area: B::Scalar,
    /// 最大面积
    pub max_area: B::Scalar,
    /// 平均面积
    pub avg_area: B::Scalar,
    /// 退化单元数（面积接近零）
    pub degenerate_cells: usize,
}

impl<B: Backend> Default for ValidationReport<B> {
    fn default() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            stats: ValidationStats::default(),
        }
    }
}

impl<B: Backend> Default for ValidationStats<B> {
    fn default() -> Self {
        Self {
            n_cells: 0,
            n_faces: 0,
            n_nodes: 0,
            min_area: B::Scalar::ZERO,
            max_area: B::Scalar::ZERO,
            avg_area: B::Scalar::ZERO,
            degenerate_cells: 0,
        }
    }
}

impl<B: Backend> ValidationReport<B> {
    /// 创建通过验证的报告
    pub fn passed() -> Self {
        Self {
            is_valid: true,
            ..Default::default()
        }
    }

    /// 添加错误（自动标记为无效）
    pub fn add_error(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
        self.is_valid = false;
    }

    /// 添加警告
    pub fn add_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    /// 是否有警告
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

// =========================================================================
// MeshAccess 扩展方法（自动实现）
// =========================================================================

/// MeshAccess 扩展方法（提供便捷工具函数）
///
/// 本 trait 为所有实现 MeshAccess 的类型自动提供扩展功能，无需重复实现。
pub trait MeshAccessExt<B: Backend>: MeshAccess<B> {
    /// 验证网格拓扑一致性
    ///
    /// 检查项目：
    /// - 面的 owner/neighbor 索引有效性
    /// - 单元面积是否为正
    /// - 边界面配置正确性
    fn validate_topology(&self) -> ValidationReport<B> {
        let mut report = ValidationReport::passed();
        report.stats.n_cells = self.n_cells();
        report.stats.n_faces = self.n_faces();
        report.stats.n_nodes = self.n_nodes();
        let backend = self.backend();

        // 检查 owner/neighbor 索引
        for face in 0..self.n_faces() {
            let owner = self.face_owner(face);
            if owner >= self.n_cells() {
                report.add_error(format!("面 {} 的 owner 索引 {} 越界", face, owner));
            }

            if let Some(neighbor) = self.face_neighbor(face) {
                if neighbor >= self.n_cells() {
                    report.add_error(format!("面 {} 的 neighbor 索引 {} 越界", face, neighbor));
                }
            }
        }

        // 检查单元面积
        let areas: Vec<B::Scalar> = (0..self.n_cells()).map(|i| self.cell_area(i)).collect();
        if areas.is_empty() {
            report.stats.min_area = B::Scalar::ZERO;
            report.stats.max_area = B::Scalar::ZERO;
            report.stats.avg_area = B::Scalar::ZERO;
        } else {
            report.stats.min_area = areas.iter().cloned().fold(B::Scalar::MAX, B::Scalar::min);
            report.stats.max_area = areas.iter().cloned().fold(B::Scalar::MIN, B::Scalar::max);
            let count = backend.config_scalar(areas.len() as f64, "MeshValidationReport.area_stats.count");
            report.stats.avg_area = areas.iter().cloned().fold(B::Scalar::ZERO, |acc, v| acc + v) / count;
            let small = backend.config_scalar(1e-12, "MeshValidationReport.area_stats.small_threshold");

            for (i, &area) in areas.iter().enumerate() {
                if area <= B::Scalar::ZERO {
                    report.add_error(format!("单元 {} 面积为负或零: {}", i, area));
                } else if area < small {
                    report.stats.degenerate_cells += 1;
                    report.add_warning(format!("单元 {} 面积过小: {}", i, area));
                }
            }
        }

        report
    }

    /// 获取单元几何信息
    fn cell_geometry(&self, cell: usize) -> CellGeometry<B> {
        CellGeometry {
            centroid: self.cell_centroid(cell),
            area: self.cell_area(cell),
            characteristic_length: self.cell_area(cell).sqrt(),
            bed_elevation: self.cell_bed_elevation(cell),
        }
    }

    /// 获取面几何信息
    fn face_geometry(&self, face: usize) -> FaceGeometry<B> {
        FaceGeometry {
            centroid: self.face_centroid(face),
            length: self.face_length(face),
            normal: self.face_normal_2d(face),
            owner: self.face_owner(face),
            neighbor: self.face_neighbor(face),
        }
    }

    /// 计算网格总面积
    fn total_area(&self) -> B::Scalar {
        (0..self.n_cells()).map(|i| self.cell_area(i)).fold(B::Scalar::ZERO, |acc, v| acc + v)
    }

    /// 计算边界面总长度
    fn total_boundary_length(&self) -> B::Scalar {
        (self.n_internal_faces()..self.n_faces())
            .map(|i| self.face_length(i))
            .fold(B::Scalar::ZERO, |acc, v| acc + v)
    }

    /// 获取边界面索引列表
    fn boundary_face_indices(&self) -> Vec<usize> {
        (self.n_internal_faces()..self.n_faces())
            .filter(|&i| self.is_boundary_face(i))
            .collect()
    }

    /// 按边界 ID 分组的面索引
    fn faces_by_boundary_id(&self) -> std::collections::HashMap<usize, Vec<usize>> {
        let mut map = std::collections::HashMap::new();
        for face in self.n_internal_faces()..self.n_faces() {
            if let Some(bid) = self.boundary_id(face) {
                map.entry(bid).or_insert_with(Vec::new).push(face);
            }
        }
        map
    }

    // ===== 并行迭代器支持（需 feature = "parallel"） =====

    /// 并行遍历所有单元（Rayon）
    #[cfg(feature = "parallel")]
    fn par_cells(&self) -> impl rayon::iter::ParallelIterator<Item = usize>
    where
        Self: Sync,
    {
        use rayon::prelude::*;
        (0..self.n_cells()).into_par_iter()
    }

    /// 并行遍历所有面
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let flux_sum: f64 = mesh.par_faces()
    ///     .map(|face| compute_flux(mesh, face, &state))
    ///     .sum();
    /// ```
    #[cfg(feature = "parallel")]
    fn par_faces(&self) -> impl rayon::iter::ParallelIterator<Item = usize>
    where
        Self: Sync,
    {
        use rayon::prelude::*;
        (0..self.n_faces()).into_par_iter()
    }

    /// 并行遍历所有内部面
    ///
    /// 仅遍历内部面（连接两个单元的面），不包括边界面。
    #[cfg(feature = "parallel")]
    fn par_internal_faces(&self) -> impl rayon::iter::ParallelIterator<Item = usize>
    where
        Self: Sync,
    {
        use rayon::prelude::*;
        (0..self.n_internal_faces()).into_par_iter()
    }

    /// 并行遍历所有边界面
    #[cfg(feature = "parallel")]
    fn par_boundary_faces(&self) -> impl rayon::iter::ParallelIterator<Item = usize>
    where
        Self: Sync,
    {
        use rayon::prelude::*;
        (self.n_internal_faces()..self.n_faces()).into_par_iter()
    }
}

// 自动为所有实现 MeshAccess 的类型提供扩展方法
impl<B: Backend, T: MeshAccess<B> + ?Sized> MeshAccessExt<B> for T {}

// =========================================================================
// 测试模块
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    /// Mock 网格用于单元测试
    struct MockMesh {
        n_cells: usize,
        n_faces: usize,
        n_internal_faces: usize,
        backend: CpuBackend<f64>,
    }

    impl MockMesh {
        fn new(n_cells: usize, n_faces: usize, n_internal_faces: usize) -> Self {
            Self {
                n_cells,
                n_faces,
                n_internal_faces,
                backend: CpuBackend::<f64>::new(),
            }
        }
    }

    impl MeshAccess<CpuBackend<f64>> for MockMesh {
        fn backend(&self) -> &CpuBackend<f64> { &self.backend }
        fn n_cells(&self) -> usize { self.n_cells }
        fn n_faces(&self) -> usize { self.n_faces }
        fn n_internal_faces(&self) -> usize { self.n_internal_faces }
        fn n_nodes(&self) -> usize { 0 }
        fn cell_centroid(&self, _cell: usize) -> Point2D { Point2D::new(0.0, 0.0) }
        fn cell_area(&self, _cell: usize) -> f64 { 1.0 }
        fn face_centroid(&self, _face: usize) -> Point2D { Point2D::new(0.0, 0.0) }
        fn face_length(&self, _face: usize) -> f64 { 1.0 }
        fn face_normal(&self, _face: usize) -> Point3D { Point3D::new(1.0, 0.0, 0.0) }
        fn node_position(&self, _node: usize) -> Point3D { Point3D::new(0.0, 0.0, 0.0) }
        fn cell_bed_elevation(&self, _cell: usize) -> f64 { 0.0 }
        fn face_owner(&self, _face: usize) -> usize { 0 }
        fn face_neighbor(&self, face: usize) -> Option<usize> {
            if face < self.n_internal_faces { Some(1) } else { None }
        }
        fn cell_face_indices(&self, _cell: usize) -> &[u32] { &[] }
        fn cell_neighbor_indices(&self, _cell: usize) -> &[u32] { &[] }
        fn cell_node_indices(&self, _cell: usize) -> &[u32] { &[] }
        fn boundary_id(&self, face: usize) -> Option<usize> {
            if face >= self.n_internal_faces { Some(0) } else { None }
        }
        fn boundary_name(&self, _boundary_id: usize) -> Option<&str> { Some("boundary") }
        fn all_cell_centroids(&self) -> &[Point2D] { &[] }
        fn all_cell_areas(&self) -> Vec<f64> { Vec::new() }
        fn all_cell_bed_elevations(&self) -> Vec<f64> { Vec::new() }
        fn face_z_left(&self, _face: usize) -> f64 { 0.0 }
        fn face_z_right(&self, _face: usize) -> f64 { 0.0 }
    }

    #[test]
    fn test_mesh_access_boundary_count() {
        let mesh = MockMesh::new(10, 20, 15);
        assert_eq!(mesh.n_boundary_faces(), 5);
    }

    #[test]
    fn test_boundary_face_detection() {
        let mesh = MockMesh::new(10, 20, 15);
        assert!(mesh.is_internal_face(0));
        assert!(mesh.is_internal_face(14));
        assert!(mesh.is_boundary_face(15));
        assert!(mesh.is_boundary_face(19));
    }

    #[test]
    fn test_face_normal_2d() {
        let mesh = MockMesh::new(1, 1, 0);
        let n2d = mesh.face_normal_2d(0);
        assert!((n2d.x - 1.0).abs() < 1e-10);
        assert!((n2d.y - 0.0).abs() < 1e-10);
    }
}

// =========================================================================
// MeshAccessTyped - 强类型索引访问接口（层级索引适配）
// =========================================================================

/// 强类型索引访问接口
///
/// 为保持索引分层（基础索引/运行时索引/业务索引），
/// 本 trait 以运行时索引类型为输入输出，
/// 并基于 MeshAccess 的 usize 实现提供类型安全包装。
pub trait MeshAccessTyped<B: Backend>: MeshAccess<B> {
    /// 单元质心（强类型索引）
    #[inline]
    fn cell_centroid_idx(&self, cell: CellIndex) -> Point2D {
        self.cell_centroid(cell.get())
    }

    /// 单元面积（强类型索引）
    #[inline]
    fn cell_area_idx(&self, cell: CellIndex) -> B::Scalar {
        self.cell_area(cell.get())
    }

    /// 面中心（强类型索引）
    #[inline]
    fn face_centroid_idx(&self, face: FaceIndex) -> Point2D {
        self.face_centroid(face.get())
    }

    /// 面长度（强类型索引）
    #[inline]
    fn face_length_idx(&self, face: FaceIndex) -> B::Scalar {
        self.face_length(face.get())
    }

    /// 面法向量（强类型索引）
    #[inline]
    fn face_normal_idx(&self, face: FaceIndex) -> Point3D {
        self.face_normal(face.get())
    }

    /// 节点坐标（强类型索引）
    #[inline]
    fn node_position_idx(&self, node: NodeIndex) -> Point3D {
        self.node_position(node.get())
    }

    /// 单元底床高程（强类型索引）
    #[inline]
    fn cell_bed_elevation_idx(&self, cell: CellIndex) -> B::Scalar {
        self.cell_bed_elevation(cell.get())
    }

    /// 面 owner 单元（强类型索引）
    #[inline]
    fn face_owner_idx(&self, face: FaceIndex) -> CellIndex {
        CellIndex::new(self.face_owner(face.get()))
    }

    /// 面 neighbor 单元（强类型索引）
    #[inline]
    fn face_neighbor_idx(&self, face: FaceIndex) -> Option<CellIndex> {
        self.face_neighbor(face.get()).map(CellIndex::new)
    }

    /// 单元相邻面索引迭代器（强类型索引）
    #[inline]
    fn cell_face_indices_idx(&self, cell: CellIndex) -> impl Iterator<Item = FaceIndex> + '_ {
        self.cell_face_indices(cell.get())
            .iter()
            .map(|&f| FaceIndex::new(f as usize))
    }

    /// 单元相邻单元索引迭代器（强类型索引）
    #[inline]
    fn cell_neighbor_indices_idx(&self, cell: CellIndex) -> impl Iterator<Item = CellIndex> + '_ {
        self.cell_neighbor_indices(cell.get())
            .iter()
            .filter_map(|&c| if c == u32::MAX { None } else { Some(CellIndex::new(c as usize)) })
    }

    /// 单元节点索引迭代器（强类型索引）
    #[inline]
    fn cell_node_indices_idx(&self, cell: CellIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.cell_node_indices(cell.get())
            .iter()
            .map(|&n| NodeIndex::new(n as usize))
    }

    /// 边界 ID（强类型索引）
    #[inline]
    fn boundary_id_idx(&self, face: FaceIndex) -> Option<BoundaryIndex> {
        self.boundary_id(face.get()).map(BoundaryIndex::new)
    }

    /// 边界名称（强类型索引）
    #[inline]
    fn boundary_name_idx(&self, boundary_id: BoundaryIndex) -> Option<&str> {
        self.boundary_name(boundary_id.get())
    }

    /// 面左侧高程（强类型索引）
    #[inline]
    fn face_z_left_idx(&self, face: FaceIndex) -> B::Scalar {
        self.face_z_left(face.get())
    }

    /// 面右侧高程（强类型索引）
    #[inline]
    fn face_z_right_idx(&self, face: FaceIndex) -> B::Scalar {
        self.face_z_right(face.get())
    }

    /// 边界面索引列表（强类型索引）
    #[inline]
    fn boundary_face_indices_idx(&self) -> Vec<FaceIndex> {
        (self.n_internal_faces()..self.n_faces())
            .map(FaceIndex::new)
            .collect()
    }
}

// 自动为所有 MeshAccess 实现类型提供强类型接口
impl<B: Backend, T: MeshAccess<B> + ?Sized> MeshAccessTyped<B> for T {}
