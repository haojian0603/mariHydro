// crates/mh_mesh/src/structured.rs

//! 结构化网格模块
//!
//! 提供笛卡尔结构化网格支持，适用于：
//! - 规则矩形区域
//! - 高效的邻居查找（O(1)）
//! - 简单的边界条件处理
//! - 易于与栅格地形数据集成
//!
//! # 坐标系统
//!
//! 使用 (i, j) 表示网格索引，(x, y) 表示世界坐标。
//!
//! ```text
//! +-------+-------+-------+
//! |(0,ny-1)       |(nx-1,ny-1)|
//! +-------+-------+-------+
//! |       |       |       |
//! +-------+-------+-------+
//! |(0,0)  |(1,0)  |(nx-1,0)|
//! +-------+-------+-------+
//! ```
//!
//! # 示例
//!
//! ```ignore
//! use mh_mesh::structured::{StructuredMesh, StructuredMeshConfig};
//!
//! let config = StructuredMeshConfig {
//!     nx: 100,
//!     ny: 50,
//!     dx: 10.0,
//!     dy: 10.0,
//!     origin: (0.0, 0.0),
//! };
//!
//! let mesh = StructuredMesh::new(config);
//!
//! // 获取单元中心
//! let (x, y) = mesh.cell_center(50, 25);
//!
//! // 获取邻居
//! let neighbors = mesh.cell_neighbors(50, 25);
//! ```

use mh_geo::{Point2D, Point3D};
use crate::error::{MeshError, MeshResult};
use crate::FrozenMesh;

/// 结构化网格配置
#[derive(Debug, Clone, Copy)]
pub struct StructuredMeshConfig {
    /// x 方向单元数
    pub nx: usize,
    /// y 方向单元数
    pub ny: usize,
    /// x 方向单元尺寸
    pub dx: f64,
    /// y 方向单元尺寸
    pub dy: f64,
    /// 原点坐标
    pub origin: (f64, f64),
}

impl Default for StructuredMeshConfig {
    fn default() -> Self {
        Self {
            nx: 10,
            ny: 10,
            dx: 1.0,
            dy: 1.0,
            origin: (0.0, 0.0),
        }
    }
}

impl StructuredMeshConfig {
    /// 创建正方形网格配置
    pub fn square(n: usize, cell_size: f64) -> Self {
        Self {
            nx: n,
            ny: n,
            dx: cell_size,
            dy: cell_size,
            origin: (0.0, 0.0),
        }
    }

    /// 创建矩形网格配置
    pub fn rectangular(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        Self {
            nx,
            ny,
            dx,
            dy,
            origin: (0.0, 0.0),
        }
    }
}

/// 面方向
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaceDirection {
    /// 西（-x）
    West,
    /// 东（+x）
    East,
    /// 南（-y）
    South,
    /// 北（+y）
    North,
}

impl FaceDirection {
    /// 获取对面方向
    pub fn opposite(self) -> Self {
        match self {
            FaceDirection::West => FaceDirection::East,
            FaceDirection::East => FaceDirection::West,
            FaceDirection::South => FaceDirection::North,
            FaceDirection::North => FaceDirection::South,
        }
    }

    /// 获取法向量
    pub fn normal(self) -> (f64, f64) {
        match self {
            FaceDirection::West => (-1.0, 0.0),
            FaceDirection::East => (1.0, 0.0),
            FaceDirection::South => (0.0, -1.0),
            FaceDirection::North => (0.0, 1.0),
        }
    }
}

/// 结构化网格
#[derive(Debug, Clone)]
pub struct StructuredMesh {
    /// 配置
    config: StructuredMeshConfig,
    /// 单元面积
    cell_area: f64,
    /// 总单元数
    n_cells: usize,
    /// 活动单元掩码（true = 活动）
    active_mask: Option<Vec<bool>>,
    /// 床面高程（可选）
    bed_elevation: Option<Vec<f64>>,
}

impl StructuredMesh {
    /// 创建新的结构化网格
    pub fn new(config: StructuredMeshConfig) -> Self {
        let cell_area = config.dx * config.dy;
        let n_cells = config.nx * config.ny;
        Self {
            config,
            cell_area,
            n_cells,
            active_mask: None,
            bed_elevation: None,
        }
    }

    /// 获取配置
    pub fn config(&self) -> &StructuredMeshConfig {
        &self.config
    }

    /// 获取 x 方向单元数
    pub fn nx(&self) -> usize {
        self.config.nx
    }

    /// 获取 y 方向单元数
    pub fn ny(&self) -> usize {
        self.config.ny
    }

    /// 获取单元尺寸
    pub fn cell_size(&self) -> (f64, f64) {
        (self.config.dx, self.config.dy)
    }

    /// 获取总单元数
    pub fn num_cells(&self) -> usize {
        self.n_cells
    }

    /// 获取单元面积
    pub fn cell_area(&self) -> f64 {
        self.cell_area
    }

    /// 将 (i, j) 转换为线性索引
    #[inline]
    pub fn cell_index(&self, i: usize, j: usize) -> usize {
        j * self.config.nx + i
    }

    /// 将线性索引转换为 (i, j)
    #[inline]
    pub fn cell_ij(&self, idx: usize) -> (usize, usize) {
        let i = idx % self.config.nx;
        let j = idx / self.config.nx;
        (i, j)
    }

    /// 获取单元中心坐标
    #[inline]
    pub fn cell_center(&self, i: usize, j: usize) -> (f64, f64) {
        let x = self.config.origin.0 + (i as f64 + 0.5) * self.config.dx;
        let y = self.config.origin.1 + (j as f64 + 0.5) * self.config.dy;
        (x, y)
    }

    /// 获取单元中心（通过线性索引）
    #[inline]
    pub fn cell_center_idx(&self, idx: usize) -> (f64, f64) {
        let (i, j) = self.cell_ij(idx);
        self.cell_center(i, j)
    }

    /// 获取单元角点坐标
    pub fn cell_corners(&self, i: usize, j: usize) -> [(f64, f64); 4] {
        let x0 = self.config.origin.0 + i as f64 * self.config.dx;
        let y0 = self.config.origin.1 + j as f64 * self.config.dy;
        [
            (x0, y0),                                   // SW
            (x0 + self.config.dx, y0),                  // SE
            (x0 + self.config.dx, y0 + self.config.dy), // NE
            (x0, y0 + self.config.dy),                  // NW
        ]
    }

    /// 获取单元邻居
    ///
    /// 返回 [west, east, south, north]，边界处为 None
    pub fn cell_neighbors(&self, i: usize, j: usize) -> [Option<usize>; 4] {
        let west = if i > 0 {
            Some(self.cell_index(i - 1, j))
        } else {
            None
        };
        let east = if i + 1 < self.config.nx {
            Some(self.cell_index(i + 1, j))
        } else {
            None
        };
        let south = if j > 0 {
            Some(self.cell_index(i, j - 1))
        } else {
            None
        };
        let north = if j + 1 < self.config.ny {
            Some(self.cell_index(i, j + 1))
        } else {
            None
        };
        [west, east, south, north]
    }

    /// 获取单元邻居（通过线性索引）
    pub fn cell_neighbors_idx(&self, idx: usize) -> [Option<usize>; 4] {
        let (i, j) = self.cell_ij(idx);
        self.cell_neighbors(i, j)
    }

    /// 获取面信息
    ///
    /// 返回面的邻居单元、长度和方向
    pub fn face_info(&self, i: usize, j: usize, dir: FaceDirection) -> FaceInfo {
        let cell = self.cell_index(i, j);
        let (neighbor, length) = match dir {
            FaceDirection::West => {
                let n = if i > 0 {
                    Some(self.cell_index(i - 1, j))
                } else {
                    None
                };
                (n, self.config.dy)
            }
            FaceDirection::East => {
                let n = if i + 1 < self.config.nx {
                    Some(self.cell_index(i + 1, j))
                } else {
                    None
                };
                (n, self.config.dy)
            }
            FaceDirection::South => {
                let n = if j > 0 {
                    Some(self.cell_index(i, j - 1))
                } else {
                    None
                };
                (n, self.config.dx)
            }
            FaceDirection::North => {
                let n = if j + 1 < self.config.ny {
                    Some(self.cell_index(i, j + 1))
                } else {
                    None
                };
                (n, self.config.dx)
            }
        };

        FaceInfo {
            owner: cell,
            neighbor,
            length,
            direction: dir,
            normal: dir.normal(),
        }
    }

    /// 从世界坐标查找单元
    pub fn locate_cell(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let i = ((x - self.config.origin.0) / self.config.dx).floor() as isize;
        let j = ((y - self.config.origin.1) / self.config.dy).floor() as isize;

        if i < 0 || j < 0 {
            return None;
        }

        let i = i as usize;
        let j = j as usize;

        if i >= self.config.nx || j >= self.config.ny {
            return None;
        }

        Some((i, j))
    }

    /// 从世界坐标查找单元索引
    pub fn locate_cell_idx(&self, x: f64, y: f64) -> Option<usize> {
        self.locate_cell(x, y).map(|(i, j)| self.cell_index(i, j))
    }

    /// 获取边界单元（按方向）
    pub fn boundary_cells(&self, dir: FaceDirection) -> Vec<usize> {
        match dir {
            FaceDirection::West => (0..self.config.ny)
                .map(|j| self.cell_index(0, j))
                .collect(),
            FaceDirection::East => (0..self.config.ny)
                .map(|j| self.cell_index(self.config.nx - 1, j))
                .collect(),
            FaceDirection::South => (0..self.config.nx)
                .map(|i| self.cell_index(i, 0))
                .collect(),
            FaceDirection::North => (0..self.config.nx)
                .map(|i| self.cell_index(i, self.config.ny - 1))
                .collect(),
        }
    }

    /// 获取所有边界单元
    pub fn all_boundary_cells(&self) -> Vec<usize> {
        let mut result = Vec::new();
        result.extend(self.boundary_cells(FaceDirection::West));
        result.extend(self.boundary_cells(FaceDirection::East));
        result.extend(self.boundary_cells(FaceDirection::South));
        result.extend(self.boundary_cells(FaceDirection::North));
        result.sort_unstable();
        result.dedup();
        result
    }

    /// 设置活动单元掩码
    pub fn set_active_mask(&mut self, mask: Vec<bool>) {
        assert_eq!(mask.len(), self.n_cells);
        self.active_mask = Some(mask);
    }

    /// 检查单元是否活动
    pub fn is_active(&self, idx: usize) -> bool {
        self.active_mask
            .as_ref()
            .map(|m| m.get(idx).copied().unwrap_or(true))
            .unwrap_or(true)
    }

    /// 获取活动单元数
    pub fn num_active_cells(&self) -> usize {
        self.active_mask
            .as_ref()
            .map(|m| m.iter().filter(|&&b| b).count())
            .unwrap_or(self.n_cells)
    }

    /// 设置床面高程
    pub fn set_bed_elevation(&mut self, elevation: Vec<f64>) {
        assert_eq!(elevation.len(), self.n_cells);
        self.bed_elevation = Some(elevation);
    }

    /// 获取床面高程
    pub fn bed_elevation(&self, idx: usize) -> Option<f64> {
        self.bed_elevation.as_ref().and_then(|e| e.get(idx).copied())
    }

    /// 获取覆盖范围
    pub fn bounds(&self) -> [f64; 4] {
        let (ox, oy) = self.config.origin;
        [
            ox,
            oy,
            ox + self.config.nx as f64 * self.config.dx,
            oy + self.config.ny as f64 * self.config.dy,
        ]
    }

    /// 冻结为计算用网格（FrozenMesh）
    ///
    /// 将结构化网格转换为 SoA 布局的只读网格，供物理引擎直接使用。
    pub fn freeze(&self) -> MeshResult<FrozenMesh<f64>> {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let dx = self.config.dx;
        let dy = self.config.dy;
        let (ox, oy) = self.config.origin;

        if nx == 0 || ny == 0 {
            return Err(MeshError::invalid_topology(
                "freeze",
                format!("structured grid has zero size: nx={}, ny={}", nx, ny),
            ));
        }
        if dx <= 0.0 || dy <= 0.0 || !dx.is_finite() || !dy.is_finite() {
            return Err(MeshError::invalid_topology(
                "freeze",
                format!("invalid cell size: dx={}, dy={}", dx, dy),
            ));
        }

        let to_u32 = |value: usize, name: &str| -> MeshResult<u32> {
            u32::try_from(value).map_err(|_| {
                MeshError::invalid_topology(
                    "freeze",
                    format!("{}超过u32上限: {}", name, value),
                )
            })
        };

        let n_cells = nx * ny;
        let n_nodes = (nx + 1) * (ny + 1);
        let n_h_interior = (nx - 1) * ny;
        let n_v_interior = nx * (ny - 1);
        let boundary_start = n_h_interior + n_v_interior;
        let n_faces = boundary_start + 2 * (nx + ny);

        let mut node_coords = Vec::with_capacity(n_nodes);
        for j in 0..=ny {
            for i in 0..=nx {
                node_coords.push(Point3D::new(ox + i as f64 * dx, oy + j as f64 * dy, 0.0));
            }
        }

        let mut cell_center = Vec::with_capacity(n_cells);
        let mut cell_area = Vec::with_capacity(n_cells);
        let mut cell_z_bed = Vec::with_capacity(n_cells);
        for j in 0..ny {
            for i in 0..nx {
                cell_center.push(Point2D::new(
                    ox + (i as f64 + 0.5) * dx,
                    oy + (j as f64 + 0.5) * dy,
                ));
                cell_area.push(dx * dy);
                let bed = self.bed_elevation(i + j * nx).unwrap_or(0.0);
                cell_z_bed.push(bed);
            }
        }

        let mut cell_node_offsets = Vec::with_capacity(n_cells + 1);
        let mut cell_node_indices = Vec::with_capacity(n_cells * 4);
        cell_node_offsets.push(0);
        for j in 0..ny {
            for i in 0..nx {
                let n0 = to_u32(j * (nx + 1) + i, "节点索引")?;
                let n1 = to_u32(j * (nx + 1) + (i + 1), "节点索引")?;
                let n2 = to_u32((j + 1) * (nx + 1) + (i + 1), "节点索引")?;
                let n3 = to_u32((j + 1) * (nx + 1) + i, "节点索引")?;
                cell_node_indices.extend_from_slice(&[n0, n1, n2, n3]);
                cell_node_offsets.push(cell_node_indices.len());
            }
        }

        let mut face_center = Vec::with_capacity(n_faces);
        let mut face_normal = Vec::with_capacity(n_faces);
        let mut face_length = Vec::with_capacity(n_faces);
        let mut face_z_left = Vec::with_capacity(n_faces);
        let mut face_z_right = Vec::with_capacity(n_faces);
        let mut face_owner = Vec::with_capacity(n_faces);
        let mut face_neighbor = Vec::with_capacity(n_faces);
        let mut face_delta_owner = Vec::with_capacity(n_faces);
        let mut face_delta_neighbor = Vec::with_capacity(n_faces);
        let mut face_dist_o2n = Vec::with_capacity(n_faces);
        let mut boundary_face_indices = Vec::with_capacity(2 * (nx + ny));
        let mut face_boundary_id = Vec::with_capacity(n_faces);

        let cell_index = |i: usize, j: usize| -> usize { j * nx + i };

        // 内部垂直面（左右相邻）
        for j in 0..ny {
            for i in 0..nx - 1 {
                let owner = to_u32(cell_index(i, j), "单元索引")?;
                let neighbor = to_u32(cell_index(i + 1, j), "单元索引")?;
                let center = Point2D::new(ox + (i as f64 + 1.0) * dx, oy + (j as f64 + 0.5) * dy);
                let normal = Point3D::new(1.0, 0.0, 0.0);
                let owner_center = cell_center[owner as usize];
                let neighbor_center = cell_center[neighbor as usize];
                face_center.push(center);
                face_normal.push(normal);
                face_length.push(dy);
                face_z_left.push(cell_z_bed[owner as usize]);
                face_z_right.push(cell_z_bed[neighbor as usize]);
                face_owner.push(owner);
                face_neighbor.push(neighbor);
                face_delta_owner.push(Point2D::new(center.x - owner_center.x, center.y - owner_center.y));
                face_delta_neighbor.push(Point2D::new(center.x - neighbor_center.x, center.y - neighbor_center.y));
                face_dist_o2n.push(dx);
                face_boundary_id.push(None);
            }
        }

        // 内部水平面（上下相邻）
        for j in 0..ny - 1 {
            for i in 0..nx {
                let owner = to_u32(cell_index(i, j), "单元索引")?;
                let neighbor = to_u32(cell_index(i, j + 1), "单元索引")?;
                let center = Point2D::new(ox + (i as f64 + 0.5) * dx, oy + (j as f64 + 1.0) * dy);
                let normal = Point3D::new(0.0, 1.0, 0.0);
                let owner_center = cell_center[owner as usize];
                let neighbor_center = cell_center[neighbor as usize];
                face_center.push(center);
                face_normal.push(normal);
                face_length.push(dx);
                face_z_left.push(cell_z_bed[owner as usize]);
                face_z_right.push(cell_z_bed[neighbor as usize]);
                face_owner.push(owner);
                face_neighbor.push(neighbor);
                face_delta_owner.push(Point2D::new(center.x - owner_center.x, center.y - owner_center.y));
                face_delta_neighbor.push(Point2D::new(center.x - neighbor_center.x, center.y - neighbor_center.y));
                face_dist_o2n.push(dy);
                face_boundary_id.push(None);
            }
        }

        // 边界面：南、北、西、东
        let mut push_boundary = |center: Point2D, normal: Point3D, length: f64, owner: u32, boundary_id: u32| {
            let owner_center = cell_center[owner as usize];
            face_center.push(center);
            face_normal.push(normal);
            face_length.push(length);
            face_z_left.push(cell_z_bed[owner as usize]);
            face_z_right.push(cell_z_bed[owner as usize]);
            face_owner.push(owner);
            face_neighbor.push(u32::MAX);
            face_delta_owner.push(Point2D::new(center.x - owner_center.x, center.y - owner_center.y));
            face_delta_neighbor.push(Point2D::new(0.0, 0.0));
            face_dist_o2n.push(0.0);
            face_boundary_id.push(Some(boundary_id));
            boundary_face_indices.push(to_u32(face_center.len() - 1, "边界面索引")?);
        };

        // South
        for i in 0..nx {
            let owner = to_u32(cell_index(i, 0), "单元索引")?;
            let center = Point2D::new(ox + (i as f64 + 0.5) * dx, oy);
            push_boundary(center, Point3D::new(0.0, -1.0, 0.0), dx, owner, 0);
        }
        // North
        for i in 0..nx {
            let owner = to_u32(cell_index(i, ny - 1), "单元索引")?;
            let center = Point2D::new(ox + (i as f64 + 0.5) * dx, oy + ny as f64 * dy);
            push_boundary(center, Point3D::new(0.0, 1.0, 0.0), dx, owner, 1);
        }
        // West
        for j in 0..ny {
            let owner = to_u32(cell_index(0, j), "单元索引")?;
            let center = Point2D::new(ox, oy + (j as f64 + 0.5) * dy);
            push_boundary(center, Point3D::new(-1.0, 0.0, 0.0), dy, owner, 2);
        }
        // East
        for j in 0..ny {
            let owner = to_u32(cell_index(nx - 1, j), "单元索引")?;
            let center = Point2D::new(ox + nx as f64 * dx, oy + (j as f64 + 0.5) * dy);
            push_boundary(center, Point3D::new(1.0, 0.0, 0.0), dy, owner, 3);
        }

        let mut cell_face_offsets = Vec::with_capacity(n_cells + 1);
        let mut cell_face_indices = Vec::with_capacity(n_cells * 4);
        cell_face_offsets.push(0);
        for j in 0..ny {
            for i in 0..nx {
                let left = if i == 0 {
                    boundary_start + 2 * nx + j
                } else {
                    j * (nx - 1) + (i - 1)
                };
                let right = if i == nx - 1 {
                    boundary_start + 2 * nx + ny + j
                } else {
                    j * (nx - 1) + i
                };
                let bottom = if j == 0 {
                    boundary_start + i
                } else {
                    n_h_interior + (j - 1) * nx + i
                };
                let top = if j == ny - 1 {
                    boundary_start + nx + i
                } else {
                    n_h_interior + j * nx + i
                };
                cell_face_indices.extend_from_slice(&[
                    to_u32(left as usize, "单元面索引")?,
                    to_u32(right as usize, "单元面索引")?,
                    to_u32(bottom as usize, "单元面索引")?,
                    to_u32(top as usize, "单元面索引")?,
                ]);
                cell_face_offsets.push(cell_face_indices.len());
            }
        }

        let mut cell_neighbor_offsets = Vec::with_capacity(n_cells + 1);
        let mut cell_neighbor_indices = Vec::with_capacity(n_cells * 4);
        cell_neighbor_offsets.push(0);
        for j in 0..ny {
            for i in 0..nx {
                let west = if i > 0 {
                    to_u32(cell_index(i - 1, j), "邻居单元索引")?
                } else {
                    u32::MAX
                };
                let east = if i + 1 < nx {
                    to_u32(cell_index(i + 1, j), "邻居单元索引")?
                } else {
                    u32::MAX
                };
                let south = if j > 0 {
                    to_u32(cell_index(i, j - 1), "邻居单元索引")?
                } else {
                    u32::MAX
                };
                let north = if j + 1 < ny {
                    to_u32(cell_index(i, j + 1), "邻居单元索引")?
                } else {
                    u32::MAX
                };
                cell_neighbor_indices.extend_from_slice(&[west, east, south, north]);
                cell_neighbor_offsets.push(cell_neighbor_indices.len());
            }
        }

        let mut face_original_id = Vec::with_capacity(n_faces);
        for i in 0..n_faces {
            face_original_id.push(to_u32(i, "面索引")?);
        }
        let mut cell_original_id = Vec::with_capacity(n_cells);
        let mut cell_permutation = Vec::with_capacity(n_cells);
        let mut cell_inv_permutation = Vec::with_capacity(n_cells);
        for i in 0..n_cells {
            cell_original_id.push(to_u32(i, "单元索引")?);
            cell_permutation.push(to_u32(i, "单元索引")?);
            cell_inv_permutation.push(to_u32(i, "单元索引")?);
        }

        let min_cell_size = dx.min(dy);
        let max_cell_size = dx.max(dy);

        let cell_parent = (0..n_cells)
            .map(|i| to_u32(i, "单元索引"))
            .collect::<MeshResult<Vec<u32>>>()?;

        Ok(FrozenMesh {
            n_nodes,
            node_coords,
            n_cells,
            cell_center,
            cell_area,
            cell_z_bed,
            cell_node_offsets,
            cell_node_indices,
            cell_face_offsets,
            cell_face_indices,
            cell_neighbor_offsets,
            cell_neighbor_indices,
            n_faces,
            n_interior_faces: n_h_interior + n_v_interior,
            face_center,
            face_normal,
            face_length,
            face_z_left,
            face_z_right,
            face_owner,
            face_neighbor,
            face_delta_owner,
            face_delta_neighbor,
            face_dist_o2n,
            boundary_face_indices,
            boundary_names: vec![
                "south".to_string(),
                "north".to_string(),
                "west".to_string(),
                "east".to_string(),
            ],
            face_boundary_id,
            min_cell_size,
            max_cell_size,
            cell_refinement_level: vec![0; n_cells],
            cell_parent,
            ghost_capacity: 0,
            cell_original_id,
            face_original_id,
            cell_permutation,
            cell_inv_permutation,
        })
    }

    /// 生成所有单元中心坐标
    pub fn all_cell_centers(&self) -> Vec<(f64, f64)> {
        (0..self.n_cells)
            .map(|idx| self.cell_center_idx(idx))
            .collect()
    }

    /// 迭代所有内部面（不含边界）
    pub fn internal_faces(&self) -> impl Iterator<Item = InternalFace> + '_ {
        let mut faces = Vec::new();

        // 垂直面（x 方向）
        for j in 0..self.config.ny {
            for i in 1..self.config.nx {
                let left = self.cell_index(i - 1, j);
                let right = self.cell_index(i, j);
                faces.push(InternalFace {
                    left_cell: left,
                    right_cell: right,
                    length: self.config.dy,
                    normal: (1.0, 0.0),
                });
            }
        }

        // 水平面（y 方向）
        for j in 1..self.config.ny {
            for i in 0..self.config.nx {
                let bottom = self.cell_index(i, j - 1);
                let top = self.cell_index(i, j);
                faces.push(InternalFace {
                    left_cell: bottom,
                    right_cell: top,
                    length: self.config.dx,
                    normal: (0.0, 1.0),
                });
            }
        }

        faces.into_iter()
    }

    /// 计算网格统计信息
    pub fn statistics(&self) -> MeshStatistics {
        MeshStatistics {
            nx: self.config.nx,
            ny: self.config.ny,
            n_cells: self.n_cells,
            n_vertices: (self.config.nx + 1) * (self.config.ny + 1),
            n_internal_faces: (self.config.nx - 1) * self.config.ny
                + self.config.nx * (self.config.ny - 1),
            n_boundary_faces: 2 * self.config.nx + 2 * self.config.ny,
            dx: self.config.dx,
            dy: self.config.dy,
            bounds: self.bounds(),
            active_cells: self.num_active_cells(),
        }
    }
}

/// 面信息
#[derive(Debug, Clone, Copy)]
pub struct FaceInfo {
    /// 拥有者单元
    pub owner: usize,
    /// 邻居单元（边界时为 None）
    pub neighbor: Option<usize>,
    /// 面长度
    pub length: f64,
    /// 方向
    pub direction: FaceDirection,
    /// 外法向量
    pub normal: (f64, f64),
}

/// 内部面
#[derive(Debug, Clone, Copy)]
pub struct InternalFace {
    /// 左/下单元
    pub left_cell: usize,
    /// 右/上单元
    pub right_cell: usize,
    /// 面长度
    pub length: f64,
    /// 法向量（从 left 指向 right）
    pub normal: (f64, f64),
}

/// 网格统计
#[derive(Debug, Clone)]
pub struct MeshStatistics {
    /// x 方向单元数
    pub nx: usize,
    /// y 方向单元数
    pub ny: usize,
    /// 总单元数
    pub n_cells: usize,
    /// 顶点数
    pub n_vertices: usize,
    /// 内部面数
    pub n_internal_faces: usize,
    /// 边界面数
    pub n_boundary_faces: usize,
    /// x 方向尺寸
    pub dx: f64,
    /// y 方向尺寸
    pub dy: f64,
    /// 边界框
    pub bounds: [f64; 4],
    /// 活动单元数
    pub active_cells: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_mesh() {
        let config = StructuredMeshConfig::square(10, 1.0);
        let mesh = StructuredMesh::new(config);

        assert_eq!(mesh.nx(), 10);
        assert_eq!(mesh.ny(), 10);
        assert_eq!(mesh.num_cells(), 100);
        assert!((mesh.cell_area() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cell_indexing() {
        let mesh = StructuredMesh::new(StructuredMeshConfig::square(5, 1.0));

        // (2, 3) -> 3 * 5 + 2 = 17
        assert_eq!(mesh.cell_index(2, 3), 17);
        assert_eq!(mesh.cell_ij(17), (2, 3));
    }

    #[test]
    fn test_cell_center() {
        let mesh = StructuredMesh::new(StructuredMeshConfig {
            nx: 10,
            ny: 10,
            dx: 10.0,
            dy: 10.0,
            origin: (100.0, 200.0),
        });

        let (x, y) = mesh.cell_center(0, 0);
        assert!((x - 105.0).abs() < 1e-10);
        assert!((y - 205.0).abs() < 1e-10);
    }

    #[test]
    fn test_neighbors() {
        let mesh = StructuredMesh::new(StructuredMeshConfig::square(5, 1.0));

        // 角落单元
        let neighbors = mesh.cell_neighbors(0, 0);
        assert!(neighbors[0].is_none()); // west
        assert!(neighbors[1].is_some()); // east
        assert!(neighbors[2].is_none()); // south
        assert!(neighbors[3].is_some()); // north

        // 中心单元
        let neighbors = mesh.cell_neighbors(2, 2);
        assert!(neighbors.iter().all(|n| n.is_some()));
    }

    #[test]
    fn test_locate_cell() {
        let mesh = StructuredMesh::new(StructuredMeshConfig {
            nx: 10,
            ny: 10,
            dx: 10.0,
            dy: 10.0,
            origin: (0.0, 0.0),
        });

        assert_eq!(mesh.locate_cell(5.0, 5.0), Some((0, 0)));
        assert_eq!(mesh.locate_cell(15.0, 25.0), Some((1, 2)));
        assert_eq!(mesh.locate_cell(-5.0, 5.0), None);
        assert_eq!(mesh.locate_cell(150.0, 5.0), None);
    }

    #[test]
    fn test_boundary_cells() {
        let mesh = StructuredMesh::new(StructuredMeshConfig::square(5, 1.0));

        let west = mesh.boundary_cells(FaceDirection::West);
        assert_eq!(west.len(), 5);
        assert!(west.iter().all(|&idx| mesh.cell_ij(idx).0 == 0));

        let east = mesh.boundary_cells(FaceDirection::East);
        assert_eq!(east.len(), 5);
        assert!(east.iter().all(|&idx| mesh.cell_ij(idx).0 == 4));
    }

    #[test]
    fn test_internal_faces() {
        let mesh = StructuredMesh::new(StructuredMeshConfig::square(3, 1.0));

        let faces: Vec<_> = mesh.internal_faces().collect();
        // 3x3 网格：
        // 垂直面：2 * 3 = 6
        // 水平面：3 * 2 = 6
        // 总计：12
        assert_eq!(faces.len(), 12);
    }

    #[test]
    fn test_statistics() {
        let mesh = StructuredMesh::new(StructuredMeshConfig::square(10, 1.0));
        let stats = mesh.statistics();

        assert_eq!(stats.n_cells, 100);
        assert_eq!(stats.n_vertices, 121);
        assert_eq!(stats.n_boundary_faces, 40);
    }
}
