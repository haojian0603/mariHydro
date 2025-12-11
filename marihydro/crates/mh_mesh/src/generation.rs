// crates/mh_mesh/src/generation.rs

//! 网格生成模块
//!
//! 提供简单的结构化网格生成工具，用于测试和验证：
//!
//! - [`RectMeshGenerator`]: 矩形结构化网格生成器
//! - [`CircularMeshGenerator`]: 圆形网格生成器
//!
//! # 使用示例
//!
//! ```rust
//! use mh_mesh::generation::RectMeshGenerator;
//!
//! // 生成 10x10 的矩形网格
//! let generator = RectMeshGenerator::new(10, 10, 100.0, 100.0);
//! let mesh = generator.build();
//!
//! assert_eq!(mesh.n_faces(), 200); // 10*10*2 triangles
//! ```

use crate::halfedge::HalfEdgeMesh;

/// 矩形结构化网格生成器
///
/// 生成矩形域上的三角形网格，顶点按行主序排列
pub struct RectMeshGenerator {
    /// x 方向单元数
    nx: usize,
    /// y 方向单元数
    ny: usize,
    /// x 方向域长度 [m]
    lx: f64,
    /// y 方向域长度 [m]
    ly: f64,
    /// x 方向起点
    x0: f64,
    /// y 方向起点
    y0: f64,
}

impl RectMeshGenerator {
    /// 创建矩形网格生成器
    ///
    /// # 参数
    ///
    /// - `nx`: x 方向单元数
    /// - `ny`: y 方向单元数
    /// - `lx`: x 方向域长度
    /// - `ly`: y 方向域长度
    pub fn new(nx: usize, ny: usize, lx: f64, ly: f64) -> Self {
        Self {
            nx,
            ny,
            lx,
            ly,
            x0: 0.0,
            y0: 0.0,
        }
    }

    /// 创建方形网格生成器
    pub fn square(n: usize, length: f64) -> Self {
        Self::new(n, n, length, length)
    }

    /// 设置原点偏移
    pub fn with_origin(mut self, x0: f64, y0: f64) -> Self {
        self.x0 = x0;
        self.y0 = y0;
        self
    }

    /// 获取 x 方向网格间距
    pub fn dx(&self) -> f64 {
        self.lx / self.nx as f64
    }

    /// 获取 y 方向网格间距
    pub fn dy(&self) -> f64 {
        self.ly / self.ny as f64
    }

    /// 获取顶点总数
    pub fn n_vertices(&self) -> usize {
        (self.nx + 1) * (self.ny + 1)
    }

    /// 获取单元总数（每个矩形分为 2 个三角形）
    pub fn n_cells(&self) -> usize {
        self.nx * self.ny * 2
    }

    /// 构建网格
    pub fn build(&self) -> HalfEdgeMesh<(), (), ()> {
        let mut mesh = HalfEdgeMesh::new();

        let dx = self.dx();
        let dy = self.dy();

        // 添加顶点
        let mut vertex_ids = Vec::with_capacity(self.n_vertices());
        for j in 0..=self.ny {
            for i in 0..=self.nx {
                let x = self.x0 + i as f64 * dx;
                let y = self.y0 + j as f64 * dy;
                let vid = mesh.add_vertex_xyz(x, y, 0.0);
                vertex_ids.push(vid);
            }
        }

        // 顶点索引辅助函数
        let vertex_idx = |i: usize, j: usize| -> usize { j * (self.nx + 1) + i };

        // 添加三角形
        for j in 0..self.ny {
            for i in 0..self.nx {
                // 四个角的顶点
                let v00 = vertex_ids[vertex_idx(i, j)];
                let v10 = vertex_ids[vertex_idx(i + 1, j)];
                let v01 = vertex_ids[vertex_idx(i, j + 1)];
                let v11 = vertex_ids[vertex_idx(i + 1, j + 1)];

                // 交替对角线方向，避免各向异性
                if (i + j) % 2 == 0 {
                    // 对角线 v00-v11
                    mesh.add_triangle(v00, v10, v11);
                    mesh.add_triangle(v00, v11, v01);
                } else {
                    // 对角线 v01-v10
                    mesh.add_triangle(v00, v10, v01);
                    mesh.add_triangle(v10, v11, v01);
                }
            }
        }

        mesh
    }

    /// 构建带初始水深的网格
    ///
    /// # 参数
    ///
    /// - `depth_fn`: 返回给定 (x, y) 坐标处水深的函数
    pub fn build_with_depth<F>(&self, depth_fn: F) -> (HalfEdgeMesh<(), (), f64>, Vec<f64>)
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut mesh: HalfEdgeMesh<(), (), f64> = HalfEdgeMesh::new();

        let dx = self.dx();
        let dy = self.dy();

        // 添加顶点
        let mut vertex_ids = Vec::with_capacity(self.n_vertices());
        for j in 0..=self.ny {
            for i in 0..=self.nx {
                let x = self.x0 + i as f64 * dx;
                let y = self.y0 + j as f64 * dy;
                let vid = mesh.add_vertex_xyz(x, y, 0.0);
                vertex_ids.push(vid);
            }
        }

        let vertex_idx = |i: usize, j: usize| -> usize { j * (self.nx + 1) + i };

        // 收集单元中心水深
        let mut cell_depths = Vec::with_capacity(self.n_cells());

        for j in 0..self.ny {
            for i in 0..self.nx {
                let v00 = vertex_ids[vertex_idx(i, j)];
                let v10 = vertex_ids[vertex_idx(i + 1, j)];
                let v01 = vertex_ids[vertex_idx(i, j + 1)];
                let v11 = vertex_ids[vertex_idx(i + 1, j + 1)];

                // 两个三角形的中心
                let cx1: f64;
                let cy1: f64;
                let cx2: f64;
                let cy2: f64;

                if (i + j) % 2 == 0 {
                    // 三角形 1: v00, v10, v11
                    cx1 = self.x0 + (i as f64 + 2.0 / 3.0) * dx;
                    cy1 = self.y0 + (j as f64 + 1.0 / 3.0) * dy;
                    // 三角形 2: v00, v11, v01
                    cx2 = self.x0 + (i as f64 + 1.0 / 3.0) * dx;
                    cy2 = self.y0 + (j as f64 + 2.0 / 3.0) * dy;

                    let depth1 = depth_fn(cx1, cy1);
                    let depth2 = depth_fn(cx2, cy2);

                    // 添加三角形并设置面数据
                    if let Some(f1) = mesh.add_triangle(v00, v10, v11) {
                        if let Some(face) = mesh.face_mut(f1) {
                            face.data = depth1;
                        }
                    }
                    if let Some(f2) = mesh.add_triangle(v00, v11, v01) {
                        if let Some(face) = mesh.face_mut(f2) {
                            face.data = depth2;
                        }
                    }

                    cell_depths.push(depth1);
                    cell_depths.push(depth2);
                } else {
                    // 三角形 1: v00, v10, v01
                    cx1 = self.x0 + (i as f64 + 1.0 / 3.0) * dx;
                    cy1 = self.y0 + (j as f64 + 1.0 / 3.0) * dy;
                    // 三角形 2: v10, v11, v01
                    cx2 = self.x0 + (i as f64 + 2.0 / 3.0) * dx;
                    cy2 = self.y0 + (j as f64 + 2.0 / 3.0) * dy;

                    let depth1 = depth_fn(cx1, cy1);
                    let depth2 = depth_fn(cx2, cy2);

                    // 添加三角形并设置面数据
                    if let Some(f1) = mesh.add_triangle(v00, v10, v01) {
                        if let Some(face) = mesh.face_mut(f1) {
                            face.data = depth1;
                        }
                    }
                    if let Some(f2) = mesh.add_triangle(v10, v11, v01) {
                        if let Some(face) = mesh.face_mut(f2) {
                            face.data = depth2;
                        }
                    }

                    cell_depths.push(depth1);
                    cell_depths.push(depth2);
                }
            }
        }

        (mesh, cell_depths)
    }
}

/// 圆形网格生成器
///
/// 生成圆形域上的三角形网格，适用于 Thacker 测试等
pub struct CircularMeshGenerator {
    /// 半径 [m]
    radius: f64,
    /// 径向分割数
    n_radial: usize,
    /// 周向分割数
    n_angular: usize,
    /// 中心 x 坐标
    cx: f64,
    /// 中心 y 坐标
    cy: f64,
}

impl CircularMeshGenerator {
    /// 创建圆形网格生成器
    ///
    /// # 参数
    ///
    /// - `radius`: 圆半径
    /// - `n_radial`: 径向分割数
    /// - `n_angular`: 周向分割数
    pub fn new(radius: f64, n_radial: usize, n_angular: usize) -> Self {
        Self {
            radius,
            n_radial,
            n_angular,
            cx: 0.0,
            cy: 0.0,
        }
    }

    /// 设置圆心
    pub fn with_center(mut self, cx: f64, cy: f64) -> Self {
        self.cx = cx;
        self.cy = cy;
        self
    }

    /// 构建网格
    pub fn build(&self) -> HalfEdgeMesh<(), (), ()> {
        let mut mesh = HalfEdgeMesh::new();

        // 添加中心点
        let center = mesh.add_vertex_xyz(self.cx, self.cy, 0.0);

        // 添加各环的顶点
        let dr = self.radius / self.n_radial as f64;
        let dtheta: f64 = 2.0 * std::f64::consts::PI as f64 / self.n_angular as f64;

        let mut rings: Vec<Vec<_>> = Vec::with_capacity(self.n_radial);

        for ir in 1..=self.n_radial {
            let r = ir as f64 * dr;
            let mut ring = Vec::with_capacity(self.n_angular);

            for ia in 0..self.n_angular {
                let theta = ia as f64 * dtheta;
                let x = self.cx + r * theta.cos();
                let y = self.cy + r * theta.sin();
                let vid = mesh.add_vertex_xyz(x, y, 0.0);
                ring.push(vid);
            }

            rings.push(ring);
        }

        // 中心三角形
        let first_ring = &rings[0];
        for ia in 0..self.n_angular {
            let next_ia = (ia + 1) % self.n_angular;
            mesh.add_triangle(center, first_ring[ia], first_ring[next_ia]);
        }

        // 其余环的四边形分割
        for ir in 1..self.n_radial {
            let inner_ring = &rings[ir - 1];
            let outer_ring = &rings[ir];

            for ia in 0..self.n_angular {
                let next_ia = (ia + 1) % self.n_angular;

                let v_inner = inner_ring[ia];
                let v_inner_next = inner_ring[next_ia];
                let v_outer = outer_ring[ia];
                let v_outer_next = outer_ring[next_ia];

                // 两个三角形
                mesh.add_triangle(v_inner, v_outer, v_outer_next);
                mesh.add_triangle(v_inner, v_outer_next, v_inner_next);
            }
        }

        mesh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_mesh_basic() {
        let gen = RectMeshGenerator::new(2, 2, 10.0, 10.0);
        let mesh = gen.build();

        // 2x2 矩形 -> 8 个三角形
        assert_eq!(mesh.n_faces(), 8);
        // 3x3 顶点
        assert_eq!(mesh.n_vertices(), 9);
    }

    #[test]
    fn test_rect_mesh_square() {
        let gen = RectMeshGenerator::square(5, 100.0);
        let mesh = gen.build();

        assert_eq!(gen.n_cells(), 50);
        assert_eq!(mesh.n_faces(), 50);
    }

    #[test]
    fn test_rect_mesh_with_origin() {
        let gen = RectMeshGenerator::new(1, 1, 10.0, 10.0).with_origin(-5.0, -5.0);
        assert!((gen.dx() - 10.0).abs() < 1e-10);
        assert!((gen.dy() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_circular_mesh_basic() {
        let gen = CircularMeshGenerator::new(10.0, 3, 8);
        let mesh = gen.build();

        // 中心 8 个三角形 + 2 环 * 8 * 2 = 8 + 32 = 40
        assert_eq!(mesh.n_faces(), 8 + 2 * 8 * 2);
    }

    #[test]
    fn test_circular_mesh_with_center() {
        let gen = CircularMeshGenerator::new(5.0, 2, 6).with_center(10.0, 20.0);
        let mesh = gen.build();

        // 验证网格已创建
        assert!(mesh.n_faces() > 0);
        assert!(mesh.n_vertices() > 0);
    }
}
