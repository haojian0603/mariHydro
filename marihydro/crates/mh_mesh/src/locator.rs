// crates/mh_mesh/src/locator.rs

//! 网格点定位器（泛型版本）
//!
//! 提供高级的点定位功能，支持 FrozenMesh<S> 泛型。

use crate::frozen::FrozenMesh;
use crate::spatial_index::MeshSpatialIndex;
use mh_geo::Point2D;
use mh_runtime::RuntimeScalar;
use std::cell::Cell;

// ============================================================
// 容差配置
// ============================================================

/// 定位容差配置
///
/// 控制各种几何判断的容差阈值，用于处理浮点精度问题。
#[derive(Debug, Clone, Copy)]
pub struct LocateTolerance {
    /// 边界判断容差
    pub boundary_tol: f64,
    /// 内部点判断容差
    pub inside_tol: f64,
    /// 退化单元判断容差
    pub degenerate_tol: f64,
}

impl Default for LocateTolerance {
    fn default() -> Self {
        Self {
            boundary_tol: 1e-8,
            inside_tol: 1e-10,
            degenerate_tol: 1e-12,
        }
    }
}

impl LocateTolerance {
    /// 高精度容差（适用于科学计算）
    pub const HIGH_PRECISION: Self = Self {
        boundary_tol: 1e-12,
        inside_tol: 1e-14,
        degenerate_tol: 1e-15,
    };

    /// 标准容差（默认设置）
    pub const STANDARD: Self = Self {
        boundary_tol: 1e-8,
        inside_tol: 1e-10,
        degenerate_tol: 1e-12,
    };

    /// 宽松容差（适用于可视化）
    pub const RELAXED: Self = Self {
        boundary_tol: 1e-6,
        inside_tol: 1e-8,
        degenerate_tol: 1e-10,
    };
}

/// 定位结果
#[derive(Debug, Clone)]
pub enum LocateResult {
    /// 点在单元内部
    InCell {
        cell_index: usize,
        barycentric: [f64; 3],
    },
    /// 点在边界边上
    OnBoundary {
        face_index: usize,
        t: f64,
    },
    /// 点在网格外
    Outside {
        nearest_face: usize,
        distance: f64,
    },
}

impl LocateResult {
    /// 是否在单元内
    #[inline]
    pub fn is_inside(&self) -> bool {
        matches!(self, Self::InCell { .. })
    }

    /// 是否在边界上
    #[inline]
    pub fn is_on_boundary(&self) -> bool {
        matches!(self, Self::OnBoundary { .. })
    }

    /// 是否在网格外
    #[inline]
    pub fn is_outside(&self) -> bool {
        matches!(self, Self::Outside { .. })
    }

    pub fn cell_index(&self) -> Option<usize> {
        match self {
            Self::InCell { cell_index, .. } => Some(*cell_index),
            _ => None,
        }
    }
}

/// 网格定位器（泛型版本）
///
/// 提供高效的点定位和空间查询功能。
pub struct MeshLocator<'a, S: RuntimeScalar> {
    /// 空间索引
    index: MeshSpatialIndex,
    /// 网格引用
    mesh: &'a FrozenMesh<S>,
    /// 容差配置
    tolerance: LocateTolerance,
}

impl<'a, S: RuntimeScalar> MeshLocator<'a, S> {
    /// 从冻结网格创建定位器（使用默认容差）
    pub fn new(mesh: &'a FrozenMesh<S>) -> Self {
        let index = MeshSpatialIndex::build(mesh.n_cells(), |i| mesh.get_cell_vertices(i));
        Self {
            index,
            mesh,
            tolerance: LocateTolerance::default(),
        }
    }

    /// 使用自定义容差创建定位器
    pub fn with_tolerance(mesh: &'a FrozenMesh<S>, tolerance: LocateTolerance) -> Self {
        let index = MeshSpatialIndex::build(mesh.n_cells(), |i| mesh.get_cell_vertices(i));
        Self {
            index,
            mesh,
            tolerance,
        }
    }

    /// 从已有的空间索引创建定位器
    pub fn with_index(mesh: &'a FrozenMesh<S>, index: MeshSpatialIndex) -> Self {
        Self {
            index,
            mesh,
            tolerance: LocateTolerance::default(),
        }
    }

    /// 从已有的空间索引和自定义容差创建定位器
    pub fn with_index_and_tolerance(
        mesh: &'a FrozenMesh<S>,
        index: MeshSpatialIndex,
        tolerance: LocateTolerance,
    ) -> Self {
        Self {
            index,
            mesh,
            tolerance,
        }
    }

    /// 获取当前容差配置
    #[inline]
    pub fn tolerance(&self) -> &LocateTolerance {
        &self.tolerance
    }

    /// 定位点
    pub fn locate(&self, x: f64, y: f64) -> LocateResult {
        if let Some(cell_idx) = self.index.locate_point(x, y) {
            let bary = self.compute_barycentric(cell_idx, x, y);
            return LocateResult::InCell {
                cell_index: cell_idx,
                barycentric: bary,
            };
        }

        let (nearest_face, distance) = self.find_nearest_boundary_face(x, y);

        LocateResult::Outside {
            nearest_face,
            distance,
        }
    }

    /// 快速判断点是否在网格内
    #[inline]
    pub fn contains(&self, x: f64, y: f64) -> bool {
        self.index.locate_point(x, y).is_some()
    }

    /// 查找点所在的单元（仅返回单元索引）
    #[inline]
    pub fn find_cell(&self, x: f64, y: f64) -> Option<usize> {
        self.index.locate_point(x, y)
    }

    /// 批量定位点
    pub fn locate_batch(&self, points: &[(f64, f64)]) -> Vec<LocateResult> {
        points.iter().map(|&(x, y)| self.locate(x, y)).collect()
    }

    /// 计算重心坐标
    pub fn compute_barycentric(&self, cell: usize, x: f64, y: f64) -> [f64; 3] {
        let vertices = self.mesh.get_cell_vertices(cell);

        if vertices.len() != 3 {
            let n = vertices.len() as f64;
            let w = 1.0 / n;
            return [w, w, 1.0 - 2.0 * w];
        }

        let (v0, v1, v2) = (&vertices[0], &vertices[1], &vertices[2]);

        let denom = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
        if denom.abs() < 1e-12 {
            return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        }

        let l1 = ((v1.y - v2.y) * (x - v2.x) + (v2.x - v1.x) * (y - v2.y)) / denom;
        let l2 = ((v2.y - v0.y) * (x - v2.x) + (v0.x - v2.x) * (y - v2.y)) / denom;
        let l3 = 1.0 - l1 - l2;

        [l1, l2, l3]
    }

    /// 查找最近的边界面
    pub fn find_nearest_boundary_face(&self, x: f64, y: f64) -> (usize, f64) {
        let mut min_dist = f64::INFINITY;
        let mut nearest_face = 0usize;

        for &face_idx in &self.mesh.boundary_face_indices {
            let face = face_idx as usize;
            let center = self.mesh.face_center[face];
            let dx = center.x - x;
            let dy = center.y - y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < min_dist {
                min_dist = dist;
                nearest_face = face;
            }
        }

        (nearest_face, min_dist)
    }

    /// 在指定单元内插值
    pub fn interpolate_in_cell(&self, cell: usize, x: f64, y: f64, vertex_values: &[f64]) -> f64 {
        let bary = self.compute_barycentric(cell, x, y);
        if vertex_values.len() >= 3 {
            bary[0] * vertex_values[0] + bary[1] * vertex_values[1] + bary[2] * vertex_values[2]
        } else {
            vertex_values.iter().sum::<f64>() / vertex_values.len() as f64
        }
    }

    /// 获取网格引用
    #[inline]
    pub fn mesh(&self) -> &FrozenMesh<S> {
        self.mesh
    }

    /// 获取空间索引引用
    #[inline]
    pub fn spatial_index(&self) -> &MeshSpatialIndex {
        &self.index
    }
}

// =========================================================================
// 缓存定位器
// =========================================================================

/// 缓存定位器（泛型版本）
pub struct CachedLocator<'a, S: RuntimeScalar> {
    locator: MeshLocator<'a, S>,
    last_cell: Cell<Option<usize>>,
    cache_hits: Cell<u64>,
    total_queries: Cell<u64>,
    neighbor_hits: Cell<u64>,
}

/// 缓存定位统计信息
#[derive(Debug, Clone, Copy, Default)]
pub struct LocatorCacheStats {
    pub total_queries: u64,
    pub cache_hits: u64,
    pub neighbor_hits: u64,
    pub global_searches: u64,
}

impl LocatorCacheStats {
    /// 计算总缓存命中率
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            (self.cache_hits + self.neighbor_hits) as f64 / self.total_queries as f64
        }
    }

    /// 计算直接命中率
    #[inline]
    pub fn direct_hit_rate(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_queries as f64
        }
    }
}

impl<'a, S: RuntimeScalar> CachedLocator<'a, S> {
    /// 创建新的缓存定位器
    pub fn new(mesh: &'a FrozenMesh<S>) -> Self {
        Self {
            locator: MeshLocator::new(mesh),
            last_cell: Cell::new(None),
            cache_hits: Cell::new(0),
            total_queries: Cell::new(0),
            neighbor_hits: Cell::new(0),
        }
    }

    /// 使用自定义容差创建缓存定位器
    pub fn with_tolerance(mesh: &'a FrozenMesh<S>, tolerance: LocateTolerance) -> Self {
        Self {
            locator: MeshLocator::with_tolerance(mesh, tolerance),
            last_cell: Cell::new(None),
            cache_hits: Cell::new(0),
            total_queries: Cell::new(0),
            neighbor_hits: Cell::new(0),
        }
    }

    /// 定位点（带缓存优化）
    pub fn locate(&self, x: f64, y: f64) -> LocateResult {
        self.total_queries.set(self.total_queries.get() + 1);

        if let Some(last) = self.last_cell.get() {
            if self.point_in_cell(last, x, y) {
                self.cache_hits.set(self.cache_hits.get() + 1);
                let bary = self.locator.compute_barycentric(last, x, y);
                return LocateResult::InCell {
                    cell_index: last,
                    barycentric: bary,
                };
            }
        }

        let result = self.locator.locate(x, y);

        if let LocateResult::InCell { cell_index, .. } = &result {
            self.last_cell.set(Some(*cell_index));
        }

        result
    }

    /// 快速查找单元（带缓存优化）
    pub fn find_cell(&self, x: f64, y: f64) -> Option<usize> {
        self.total_queries.set(self.total_queries.get() + 1);

        if let Some(last) = self.last_cell.get() {
            if self.point_in_cell(last, x, y) {
                self.cache_hits.set(self.cache_hits.get() + 1);
                return Some(last);
            }
        }

        let result = self.locator.find_cell(x, y);

        if let Some(cell) = result {
            self.last_cell.set(Some(cell));
        }

        result
    }

    /// 检查点是否在指定单元内
    fn point_in_cell(&self, cell: usize, x: f64, y: f64) -> bool {
        let vertices = get_cell_vertices_from_mesh(self.locator.mesh(), cell);
        point_in_polygon(x, y, &vertices)
    }

    /// 获取缓存统计信息
    pub fn stats(&self) -> LocatorCacheStats {
        let total = self.total_queries.get();
        let hits = self.cache_hits.get();
        let neighbor = self.neighbor_hits.get();

        LocatorCacheStats {
            total_queries: total,
            cache_hits: hits,
            neighbor_hits: neighbor,
            global_searches: total.saturating_sub(hits).saturating_sub(neighbor),
        }
    }

    /// 获取缓存命中率
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        self.stats().hit_rate()
    }

    /// 重置缓存和统计信息
    pub fn reset(&self) {
        self.last_cell.set(None);
        self.cache_hits.set(0);
        self.total_queries.set(0);
        self.neighbor_hits.set(0);
    }

    /// 获取内部定位器引用
    #[inline]
    pub fn inner(&self) -> &MeshLocator<'a, S> {
        &self.locator
    }
}

/// 从 FrozenMesh 获取单元顶点坐标
fn get_cell_vertices_from_mesh<S: RuntimeScalar>(mesh: &FrozenMesh<S>, cell: usize) -> Vec<Point2D> {
    let node_indices = mesh.cell_nodes(cell);
    node_indices
        .iter()
        .map(|&node_idx| {
            let coord = mesh.node_coords[node_idx as usize];
            Point2D::new(coord.x, coord.y)
        })
        .collect()
}

/// 射线法判断点是否在多边形内
fn point_in_polygon(x: f64, y: f64, vertices: &[Point2D]) -> bool {
    let n = vertices.len();
    if n < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = n - 1;

    for i in 0..n {
        let vi = &vertices[i];
        let vj = &vertices[j];

        if ((vi.y > y) != (vj.y > y))
            && (x < (vj.x - vi.x) * (y - vi.y) / (vj.y - vi.y) + vi.x)
        {
            inside = !inside;
        }

        j = i;
    }

    inside
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::RuntimeScalar;

    fn create_test_mesh<S: RuntimeScalar>() -> FrozenMesh<S> {
        let mut mesh = FrozenMesh::empty_with_cells(1);

        mesh.n_nodes = 3;
        mesh.node_coords = vec![
            mh_geo::Point3D::new(0.0, 0.0, 0.0),
            mh_geo::Point3D::new(1.0, 0.0, 0.0),
            mh_geo::Point3D::new(0.5, 1.0, 0.0),
        ];

        mesh.cell_node_offsets = vec![0, 3];
        mesh.cell_node_indices = vec![0, 1, 2];

        mesh
    }

    #[test]
    fn test_locate_result() {
        let inside = LocateResult::InCell {
            cell_index: 0,
            barycentric: [0.33, 0.33, 0.34],
        };
        assert!(inside.is_inside());
        assert!(!inside.is_outside());
        assert_eq!(inside.cell_index(), Some(0));

        let outside = LocateResult::Outside {
            nearest_face: 0,
            distance: 1.0,
        };
        assert!(outside.is_outside());
        assert!(!outside.is_inside());
        assert_eq!(outside.cell_index(), None);
    }

    #[test]
    fn test_mesh_locator_creation() {
        let mesh = create_test_mesh::<f64>();
        let locator = MeshLocator::new(&mesh);

        assert_eq!(locator.spatial_index().n_cells(), 1);
    }

    #[test]
    fn test_barycentric_computation() {
        let mesh = create_test_mesh::<f64>();
        let locator = MeshLocator::new(&mesh);

        let bary = locator.compute_barycentric(0, 0.5, 1.0 / 3.0);
        let sum: f64 = bary.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_contains() {
        let mesh = create_test_mesh::<f64>();
        let locator = MeshLocator::new(&mesh);

        assert!(locator.contains(0.5, 0.3));
        assert!(!locator.contains(2.0, 2.0));
    }

    #[test]
    fn test_find_cell() {
        let mesh = create_test_mesh::<f64>();
        let locator = MeshLocator::new(&mesh);

        assert_eq!(locator.find_cell(0.5, 0.3), Some(0));
        assert_eq!(locator.find_cell(2.0, 2.0), None);
    }

    #[test]
    fn test_cached_locator() {
        let mesh = create_test_mesh::<f64>();
        let cached = CachedLocator::new(&mesh);

        assert_eq!(cached.find_cell(0.5, 0.3), Some(0));
        assert!(cached.hit_rate() > 0.0);
    }
}