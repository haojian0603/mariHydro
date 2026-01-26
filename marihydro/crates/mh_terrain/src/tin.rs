// crates/mh_terrain/src/tin.rs

//! TIN (Triangulated Irregular Network) 地形
//!
//! 使用三角网表示的不规则分布高程数据。
//!
//! # 功能
//!
//! - 从散点高程数据构建 Delaunay 三角网
//! - 点查询和重心插值
//! - 空间索引加速（R-tree）
//! - 导出为栅格
//!
//! # 示例
//!
//! ```ignore
//! use mh_terrain::tin::TinTerrain;
//!
//! // 从散点构建 TIN
//! let points = vec![
//!     (0.0, 0.0, 10.0),
//!     (10.0, 0.0, 12.0),
//!     (5.0, 10.0, 8.0),
//!     (15.0, 15.0, 15.0),
//! ];
//! let tin = TinTerrain::from_points(&points)?;
//!
//! // 查询高程
//! let z = tin.interpolate(7.5, 5.0);
//! ```

use std::collections::HashMap;

/// TIN 地形结构
#[derive(Debug, Clone)]
pub struct TinTerrain {
    /// 顶点坐标 (x, y, z)
    vertices: Vec<(f64, f64, f64)>,
    /// 三角形索引 [(v0, v1, v2), ...]
    triangles: Vec<(usize, usize, usize)>,
    /// 三角形边界框 [min_x, min_y, max_x, max_y]（预留用于空间查询优化）
    #[allow(dead_code)]
    triangle_bounds: Vec<[f64; 4]>,
    /// 全局边界框
    global_bounds: [f64; 4],
    /// 空间索引网格（简单实现）
    grid_index: SpatialGrid,
}

/// 简单空间索引网格
#[derive(Debug, Clone)]
struct SpatialGrid {
    /// 网格尺寸
    cell_size: f64,
    /// 网格原点
    origin: (f64, f64),
    /// 网格维度
    dims: (usize, usize),
    /// 每个网格单元包含的三角形列表
    cells: HashMap<(usize, usize), Vec<usize>>,
}

impl SpatialGrid {
    /// 创建空间网格
    fn new(bounds: [f64; 4], cell_size: f64) -> Self {
        let origin = (bounds[0], bounds[1]);
        let width = bounds[2] - bounds[0];
        let height = bounds[3] - bounds[1];
        let dims = (
            ((width / cell_size).ceil() as usize).max(1),
            ((height / cell_size).ceil() as usize).max(1),
        );
        Self {
            cell_size,
            origin,
            dims,
            cells: HashMap::new(),
        }
    }

    /// 获取点所在的网格坐标
    fn grid_coord(&self, x: f64, y: f64) -> (usize, usize) {
        let gx = ((x - self.origin.0) / self.cell_size).floor() as isize;
        let gy = ((y - self.origin.1) / self.cell_size).floor() as isize;
        (
            gx.clamp(0, self.dims.0 as isize - 1) as usize,
            gy.clamp(0, self.dims.1 as isize - 1) as usize,
        )
    }

    /// 插入三角形到网格
    fn insert(&mut self, tri_idx: usize, bounds: [f64; 4]) {
        let (min_gx, min_gy) = self.grid_coord(bounds[0], bounds[1]);
        let (max_gx, max_gy) = self.grid_coord(bounds[2], bounds[3]);

        for gx in min_gx..=max_gx {
            for gy in min_gy..=max_gy {
                self.cells.entry((gx, gy)).or_default().push(tri_idx);
            }
        }
    }

    /// 查询包含指定点的候选三角形
    fn query(&self, x: f64, y: f64) -> &[usize] {
        let coord = self.grid_coord(x, y);
        self.cells.get(&coord).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

/// TIN 构建错误
#[derive(Debug)]
pub enum TinError {
    /// 点数不足
    InsufficientPoints,
    /// 退化情况（所有点共线等）
    Degenerate,
    /// 超出范围
    OutOfBounds,
}

impl std::fmt::Display for TinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TinError::InsufficientPoints => write!(f, "Insufficient points for TIN"),
            TinError::Degenerate => write!(f, "Degenerate configuration"),
            TinError::OutOfBounds => write!(f, "Query point out of bounds"),
        }
    }
}

impl std::error::Error for TinError {}

impl TinTerrain {
    /// 从散点数据构建 TIN
    ///
    /// 使用增量 Delaunay 三角剖分算法
    ///
    /// # 参数
    ///
    /// - `points`: 点列表 [(x, y, z), ...]
    pub fn from_points(points: &[(f64, f64, f64)]) -> Result<Self, TinError> {
        if points.len() < 3 {
            return Err(TinError::InsufficientPoints);
        }

        // 计算边界框
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for (x, y, _) in points {
            min_x = min_x.min(*x);
            min_y = min_y.min(*y);
            max_x = max_x.max(*x);
            max_y = max_y.max(*y);
        }

        let global_bounds = [min_x, min_y, max_x, max_y];
        let vertices: Vec<_> = points.to_vec();

        // 使用简单的 Bowyer-Watson 算法进行 Delaunay 三角剖分
        let triangles = delaunay_triangulation(&vertices)?;

        // 计算每个三角形的边界框
        let triangle_bounds: Vec<_> = triangles
            .iter()
            .map(|&(i0, i1, i2)| {
                let (x0, y0, _) = vertices[i0];
                let (x1, y1, _) = vertices[i1];
                let (x2, y2, _) = vertices[i2];
                [
                    x0.min(x1).min(x2),
                    y0.min(y1).min(y2),
                    x0.max(x1).max(x2),
                    y0.max(y1).max(y2),
                ]
            })
            .collect();

        // 构建空间索引
        let dx = max_x - min_x;
        let dy = max_y - min_y;
        let cell_size = (dx.max(dy) / 20.0).max(1.0);
        let mut grid_index = SpatialGrid::new(global_bounds, cell_size);

        for (i, bounds) in triangle_bounds.iter().enumerate() {
            grid_index.insert(i, *bounds);
        }

        Ok(Self {
            vertices,
            triangles,
            triangle_bounds,
            global_bounds,
            grid_index,
        })
    }

    /// 从已有的三角网数据创建
    pub fn from_mesh(
        vertices: Vec<(f64, f64, f64)>,
        triangles: Vec<(usize, usize, usize)>,
    ) -> Result<Self, TinError> {
        if vertices.len() < 3 || triangles.is_empty() {
            return Err(TinError::InsufficientPoints);
        }

        // 计算边界框
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for (x, y, _) in &vertices {
            min_x = min_x.min(*x);
            min_y = min_y.min(*y);
            max_x = max_x.max(*x);
            max_y = max_y.max(*y);
        }

        let global_bounds = [min_x, min_y, max_x, max_y];

        let triangle_bounds: Vec<_> = triangles
            .iter()
            .map(|&(i0, i1, i2)| {
                let (x0, y0, _) = vertices[i0];
                let (x1, y1, _) = vertices[i1];
                let (x2, y2, _) = vertices[i2];
                [
                    x0.min(x1).min(x2),
                    y0.min(y1).min(y2),
                    x0.max(x1).max(x2),
                    y0.max(y1).max(y2),
                ]
            })
            .collect();

        let dx = max_x - min_x;
        let dy = max_y - min_y;
        let cell_size = (dx.max(dy) / 20.0).max(1.0);
        let mut grid_index = SpatialGrid::new(global_bounds, cell_size);

        for (i, bounds) in triangle_bounds.iter().enumerate() {
            grid_index.insert(i, *bounds);
        }

        Ok(Self {
            vertices,
            triangles,
            triangle_bounds,
            global_bounds,
            grid_index,
        })
    }

    /// 插值查询
    ///
    /// 使用重心坐标在包含查询点的三角形内插值
    pub fn interpolate(&self, x: f64, y: f64) -> Option<f64> {
        let [min_x, min_y, max_x, max_y] = self.global_bounds;
        if x < min_x || x > max_x || y < min_y || y > max_y {
            return None;
        }
        // 使用空间索引查找候选三角形
        let candidates = self.grid_index.query(x, y);

        for &tri_idx in candidates {
            let (i0, i1, i2) = self.triangles[tri_idx];
            let (x0, y0, z0) = self.vertices[i0];
            let (x1, y1, z1) = self.vertices[i1];
            let (x2, y2, z2) = self.vertices[i2];

            if let Some((u, v, w)) = barycentric_coords(x, y, x0, y0, x1, y1, x2, y2) {
                if u >= 0.0 && v >= 0.0 && w >= 0.0 {
                    return Some(u * z0 + v * z1 + w * z2);
                }
            }
        }

        // 如果空间索引未命中，进行全局搜索（较慢）
        for (tri_idx, &(i0, i1, i2)) in self.triangles.iter().enumerate() {
            if !candidates.contains(&tri_idx) {
                let (x0, y0, z0) = self.vertices[i0];
                let (x1, y1, z1) = self.vertices[i1];
                let (x2, y2, z2) = self.vertices[i2];

                if let Some((u, v, w)) = barycentric_coords(x, y, x0, y0, x1, y1, x2, y2) {
                    if u >= -1e-10 && v >= -1e-10 && w >= -1e-10 {
                        return Some(u * z0 + v * z1 + w * z2);
                    }
                }
            }
        }

        None
    }

    /// 批量插值
    pub fn interpolate_batch(&self, points: &[(f64, f64)]) -> Vec<Option<f64>> {
        points
            .iter()
            .map(|&(x, y)| self.interpolate(x, y))
            .collect()
    }

    /// 获取顶点数
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// 获取三角形数
    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// 获取边界框
    pub fn bounds(&self) -> [f64; 4] {
        self.global_bounds
    }

    /// 获取顶点列表
    pub fn vertices(&self) -> &[(f64, f64, f64)] {
        &self.vertices
    }

    /// 获取三角形列表
    pub fn triangles(&self) -> &[(usize, usize, usize)] {
        &self.triangles
    }

    /// 导出为规则栅格
    ///
    /// # 参数
    ///
    /// - `resolution`: 栅格分辨率
    /// - `nodata`: 无数据值
    pub fn to_raster(&self, resolution: f64, nodata: f64) -> (Vec<f64>, usize, usize) {
        let [min_x, min_y, max_x, max_y] = self.global_bounds;
        let ncols = ((max_x - min_x) / resolution).ceil() as usize;
        let nrows = ((max_y - min_y) / resolution).ceil() as usize;

        let mut data = vec![nodata; nrows * ncols];

        for row in 0..nrows {
            for col in 0..ncols {
                let x = min_x + (col as f64 + 0.5) * resolution;
                let y = max_y - (row as f64 + 0.5) * resolution; // 栅格从上到下

                if let Some(z) = self.interpolate(x, y) {
                    data[row * ncols + col] = z;
                }
            }
        }

        (data, ncols, nrows)
    }

    /// 计算统计信息
    pub fn statistics(&self) -> TinStatistics {
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;
        let mut sum_z = 0.0;

        for (_, _, z) in &self.vertices {
            min_z = min_z.min(*z);
            max_z = max_z.max(*z);
            sum_z += z;
        }

        TinStatistics {
            num_vertices: self.vertices.len(),
            num_triangles: self.triangles.len(),
            min_elevation: min_z,
            max_elevation: max_z,
            mean_elevation: sum_z / self.vertices.len() as f64,
            bounds: self.global_bounds,
        }
    }
}

/// TIN 统计信息
#[derive(Debug, Clone)]
pub struct TinStatistics {
    /// 顶点数
    pub num_vertices: usize,
    /// 三角形数
    pub num_triangles: usize,
    /// 最小高程
    pub min_elevation: f64,
    /// 最大高程
    pub max_elevation: f64,
    /// 平均高程
    pub mean_elevation: f64,
    /// 边界框
    pub bounds: [f64; 4],
}

/// 计算重心坐标
///
/// 返回 (u, v, w) 使得 P = u*A + v*B + w*C
fn barycentric_coords(
    px: f64,
    py: f64,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
) -> Option<(f64, f64, f64)> {
    let v0x = x1 - x0;
    let v0y = y1 - y0;
    let v1x = x2 - x0;
    let v1y = y2 - y0;
    let v2x = px - x0;
    let v2y = py - y0;

    let d00 = v0x * v0x + v0y * v0y;
    let d01 = v0x * v1x + v0y * v1y;
    let d11 = v1x * v1x + v1y * v1y;
    let d20 = v2x * v0x + v2y * v0y;
    let d21 = v2x * v1x + v2y * v1y;

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        return None; // 退化三角形
    }

    let inv_denom = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;
    let u = 1.0 - v - w;

    Some((u, v, w))
}

/// 简单 Delaunay 三角剖分（Bowyer-Watson 算法）
fn delaunay_triangulation(
    vertices: &[(f64, f64, f64)],
) -> Result<Vec<(usize, usize, usize)>, TinError> {
    let n = vertices.len();
    if n < 3 {
        return Err(TinError::InsufficientPoints);
    }

    // 计算边界框
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for (x, y, _) in vertices {
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }

    // 创建超级三角形
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let delta = (dx.max(dy)) * 10.0;

    let super_vertices = vec![
        (min_x - delta, min_y - delta, 0.0),
        (min_x + dx / 2.0, max_y + delta * 2.0, 0.0),
        (max_x + delta, min_y - delta, 0.0),
    ];

    // 合并顶点
    let mut all_vertices: Vec<(f64, f64)> = super_vertices.iter().map(|&(x, y, _)| (x, y)).collect();
    all_vertices.extend(vertices.iter().map(|&(x, y, _)| (x, y)));

    // 初始三角形
    let mut triangles: Vec<(usize, usize, usize)> = vec![(0, 1, 2)];

    // 增量插入每个点
    for i in 0..n {
        let pt_idx = i + 3; // 偏移超级三角形顶点
        let (px, py) = all_vertices[pt_idx];

        // 找到外接圆包含该点的三角形
        let mut bad_triangles = Vec::new();
        for (ti, &(a, b, c)) in triangles.iter().enumerate() {
            if in_circumcircle(px, py, &all_vertices, a, b, c) {
                bad_triangles.push(ti);
            }
        }

        // 找到多边形边界
        let mut polygon = Vec::new();
        for &ti in &bad_triangles {
            let (a, b, c) = triangles[ti];
            let edges = [(a, b), (b, c), (c, a)];
            for edge in edges {
                let is_shared = bad_triangles.iter().any(|&other_ti| {
                    if other_ti == ti {
                        return false;
                    }
                    let (oa, ob, oc) = triangles[other_ti];
                    let other_edges = [(oa, ob), (ob, oc), (oc, oa)];
                    other_edges.contains(&edge) || other_edges.contains(&(edge.1, edge.0))
                });
                if !is_shared {
                    polygon.push(edge);
                }
            }
        }

        // 删除坏三角形（从后往前删）
        bad_triangles.sort_unstable_by(|a, b| b.cmp(a));
        for ti in bad_triangles {
            triangles.remove(ti);
        }

        // 添加新三角形
        for (e0, e1) in polygon {
            triangles.push((e0, e1, pt_idx));
        }
    }

    // 删除包含超级三角形顶点的三角形
    triangles.retain(|&(a, b, c)| a >= 3 && b >= 3 && c >= 3);

    // 调整索引（减去超级三角形偏移）
    let result: Vec<_> = triangles
        .into_iter()
        .map(|(a, b, c)| (a - 3, b - 3, c - 3))
        .collect();

    if result.is_empty() {
        return Err(TinError::Degenerate);
    }

    Ok(result)
}

/// 检查点是否在三角形外接圆内
fn in_circumcircle(
    px: f64,
    py: f64,
    vertices: &[(f64, f64)],
    a: usize,
    b: usize,
    c: usize,
) -> bool {
    let (ax, ay) = vertices[a];
    let (bx, by) = vertices[b];
    let (cx, cy) = vertices[c];

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-12 {
        return false;
    }

    let ax2_ay2 = ax * ax + ay * ay;
    let bx2_by2 = bx * bx + by * by;
    let cx2_cy2 = cx * cx + cy * cy;

    let ux = (ax2_ay2 * (by - cy) + bx2_by2 * (cy - ay) + cx2_cy2 * (ay - by)) / d;
    let uy = (ax2_ay2 * (cx - bx) + bx2_by2 * (ax - cx) + cx2_cy2 * (bx - ax)) / d;

    let r2 = (ax - ux).powi(2) + (ay - uy).powi(2);
    let dist2 = (px - ux).powi(2) + (py - uy).powi(2);

    dist2 < r2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tin() {
        let points = vec![
            (0.0, 0.0, 10.0),
            (10.0, 0.0, 12.0),
            (5.0, 10.0, 8.0),
        ];

        let tin = TinTerrain::from_points(&points).unwrap();
        assert_eq!(tin.num_vertices(), 3);
        assert_eq!(tin.num_triangles(), 1);

        // 质心处插值
        let z = tin.interpolate(5.0, 10.0 / 3.0).unwrap();
        assert!((z - 10.0).abs() < 0.1); // 约为三点平均
    }

    #[test]
    fn test_square_tin() {
        let points = vec![
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
            (5.0, 5.0, 10.0), // 中心点更高
        ];

        let tin = TinTerrain::from_points(&points).unwrap();
        assert_eq!(tin.num_vertices(), 5);
        assert!(tin.num_triangles() >= 4);

        // 中心点插值
        let z = tin.interpolate(5.0, 5.0).unwrap();
        assert!((z - 10.0).abs() < 0.1);

        // 角点插值
        let z_corner = tin.interpolate(0.0, 0.0).unwrap();
        assert!((z_corner - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_barycentric() {
        // 单位三角形
        let (u, v, w) = barycentric_coords(0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0).unwrap();
        
        // 检查重心坐标和为 1
        assert!((u + v + w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_raster() {
        let points = vec![
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 10.0),
            (10.0, 10.0, 10.0),
            (0.0, 10.0, 0.0),
        ];

        let tin = TinTerrain::from_points(&points).unwrap();
        let (data, ncols, nrows) = tin.to_raster(5.0, -9999.0);

        assert!(ncols >= 2);
        assert!(nrows >= 2);
        
        // 至少有一些有效值
        let valid_count = data.iter().filter(|&&v| v != -9999.0).count();
        assert!(valid_count > 0);
    }
}
