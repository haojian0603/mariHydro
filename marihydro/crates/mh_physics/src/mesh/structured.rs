//! 结构化矩形网格。

//! 该模块提供基于后端标量的二维结构化网格表示，用于规则网格上的
//! 单元中心、面连接关系和几何量计算。
//!
//! # 设计约束
//!
//! - 网格几何量通过 Backend 标量类型表达，避免主链回退成裸 f64
//! - 结构体保留 PhantomData 语义，确保后端类型信息稳定
//! - 单元、面和邻接关系按照结构化索引规则生成
use super::topology::{MeshKind, MeshTopology, MeshValidationError};
use crate::core::Backend;
use mh_runtime::{DeviceBuffer, RuntimeScalar};

/// 结构化矩形网格。
///
/// 该类型保存规则网格上的尺寸、间距和几何缓存，便于在结构化
/// 网格路径上执行高效的邻接查询与几何计算。
///
/// # 类型参数
///
/// - `B`: 网格几何量使用的后端类型
#[allow(dead_code)]
pub struct StructuredMesh<B: Backend> {
    /// 后端实例
    backend: B,
    /// x 方向单元数量
    nx: usize,
    /// y 方向单元数量
    ny: usize,
    /// x 方向网格间距
    dx: B::Scalar,
    /// y 方向网格间距
    dy: B::Scalar,
    /// 单元面积缓存
    cell_areas: B::Buffer<B::Scalar>,
    /// 面长度缓存
    face_lengths: B::Buffer<B::Scalar>,
    /// 边界面索引缓存
    boundary_faces_cache: Vec<usize>,
    /// 内部面索引缓存
    interior_faces_cache: Vec<usize>,
    /// 单元到面的映射缓存
    cell_faces_cache: Vec<Vec<usize>>,
}
impl<B> StructuredMesh<B>
where
    B: Backend + Clone,
    B::Scalar: RuntimeScalar,
{
    #[inline]
    fn config_scalar(&self, value: f64, context: &'static str) -> B::Scalar {
        self.backend.config_scalar(value, context)
    }

    #[inline]
    fn config_index(&self, value: usize, context: &'static str) -> B::Scalar {
        self.backend.config_scalar(value as f64, context)
    }
    /// 使用 Backend 创建结构化矩形网格
    ///
    /// # 参数
    ///
    /// - `backend`: 后端实例
    /// - `nx`: x 方向单元数量
    /// - `ny`: y 方向单元数量
    /// - `dx`: x 方向网格间距
    /// - `dy`: y 方向网格间距
    ///
    /// # 返回
    ///
    /// ```ignore
    /// use mh_physics::mesh::StructuredMesh;
    /// use mh_runtime::CpuBackend;
    ///
    /// let backend = CpuBackend::<f64>::new();
    /// let mesh = StructuredMesh::new_with_backend(&backend, 100, 100, 1.0, 1.0)?;
    /// ```
    pub fn new_with_backend(
        backend: &B,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
    ) -> Result<Self, MeshValidationError> {
        if nx == 0 || ny == 0 {
            return Err(MeshValidationError::InvalidCounts {
                n_cells: nx.saturating_mul(ny),
                n_faces: 0,
                n_interior_faces: 0,
            });
        }

        if !dx.is_finite() || dx <= 0.0 {
            return Err(MeshValidationError::InvalidSpacing {
                axis: "dx",
                value: dx,
            });
        }

        if !dy.is_finite() || dy <= 0.0 {
            return Err(MeshValidationError::InvalidSpacing {
                axis: "dy",
                value: dy,
            });
        }

        let dx = backend.config_scalar(dx, "structured_mesh.dx");
        let dy = backend.config_scalar(dy, "structured_mesh.dy");

        let n_cells = nx * ny;
        let n_faces = Self::compute_n_faces(nx, ny);

        // 使用 Backend 分配几何缓存
        let cell_area = dx * dy;
        let cell_areas = backend.alloc_init(n_cells, cell_area);
        let mut face_lengths = backend.alloc(n_faces);
        {
            let lengths = face_lengths.as_slice_mut();
            let n_h_interior = (nx - 1) * ny;
            let n_v_interior = nx * (ny - 1);
            let boundary_start = n_h_interior + n_v_interior;

            for f in 0..n_h_interior {
                lengths[f] = dy;
            }
            for f in n_h_interior..boundary_start {
                lengths[f] = dx;
            }

            // 边界面顺序：底边、顶边、左边、右边
            let bottom_start = boundary_start;
            let top_start = bottom_start + nx;
            let left_start = top_start + nx;
            let right_start = left_start + ny;

            for f in bottom_start..top_start {
                lengths[f] = dx;
            }
            for f in top_start..left_start {
                lengths[f] = dx;
            }
            for f in left_start..right_start {
                lengths[f] = dy;
            }
            for f in right_start..n_faces {
                lengths[f] = dy;
            }
        }

        // 构建边界面与内部面缓存
        let (boundary_faces_cache, interior_faces_cache) = Self::build_face_caches(nx, ny);

        // 构建单元到面的映射缓存
        let cell_faces_cache = Self::build_cell_face_map(nx, ny);

        let mesh = Self {
            backend: backend.clone(),
            nx,
            ny,
            dx,
            dy,
            cell_areas,
            face_lengths,
            boundary_faces_cache,
            interior_faces_cache,
            cell_faces_cache,
        };

        mesh.validate()?;

        Ok(mesh)
    }

    /// 计算总面数
    #[inline]
    fn compute_n_faces(nx: usize, ny: usize) -> usize {
        // 水平内部面 + 垂向内部面 + 边界面
        (nx - 1) * ny + nx * (ny - 1) + 2 * (nx + ny)
    }

    /// 构建边界面和内部面缓存
    fn build_face_caches(nx: usize, ny: usize) -> (Vec<usize>, Vec<usize>) {
        let mut boundary = Vec::new();
        let mut interior = Vec::new();

        let n_interior = (nx - 1) * ny + nx * (ny - 1);
        let n_boundary = 2 * (nx + ny);

        for i in 0..n_interior {
            interior.push(i);
        }
        for i in 0..n_boundary {
            boundary.push(n_interior + i);
        }

        (boundary, interior)
    }

    /// 构建单元到面的映射缓存
    fn build_cell_face_map(nx: usize, ny: usize) -> Vec<Vec<usize>> {
        let n_cells = nx * ny;
        let n_h_interior = (nx - 1) * ny;
        let n_v_interior = nx * (ny - 1);
        let boundary_start = n_h_interior + n_v_interior;
        let mut map = vec![Vec::with_capacity(4); n_cells];

        for j in 0..ny {
            for i in 0..nx {
                let cell = j * nx + i;
                let left = if i == 0 {
                    boundary_start + 2 * nx + j
                } else {
                    (j * (nx - 1)) + (i - 1)
                };

                let right = if i == nx - 1 {
                    boundary_start + 2 * nx + ny + j
                } else {
                    (j * (nx - 1)) + i
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

                map[cell] = vec![left, right, bottom, top];
            }
        }

        map
    }

    /// 返回后端实例

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// 获取 x 方向单元数量
    /// 返回 x 方向单元数量
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// 获取 y 方向单元数量
    /// 返回 y 方向单元数量
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// 将 (i, j) 结构化索引
    /// 映射为单元编号
    pub fn cell_index(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// 将单元编号反解为
    /// (i, j) 结构化索引
    pub fn cell_ij(&self, cell: usize) -> (usize, usize) {
        (cell % self.nx, cell / self.nx)
    }
}

impl<B> MeshTopology<B> for StructuredMesh<B>
where
    B: Backend + Clone,
    B::Scalar: RuntimeScalar,
{
    fn n_cells(&self) -> usize {
        self.nx * self.ny
    }

    fn n_faces(&self) -> usize {
        Self::compute_n_faces(self.nx, self.ny)
    }

    fn n_interior_faces(&self) -> usize {
        (self.nx - 1) * self.ny + self.nx * (self.ny - 1)
    }

    fn n_nodes(&self) -> usize {
        (self.nx + 1) * (self.ny + 1)
    }

    fn cell_center(&self, cell: usize) -> [B::Scalar; 2] {
        let (i, j) = self.cell_ij(cell);
        let half = self.config_scalar(0.5, "structured_mesh.cell_center.half");
        let x = self.dx * self.config_index(i, "structured_mesh.cell_center.i") + self.dx * half;
        let y = self.dy * self.config_index(j, "structured_mesh.cell_center.j") + self.dy * half;
        [x, y]
    }

    fn cell_area(&self, cell: usize) -> B::Scalar {
        self.cell_areas
            .get(cell)
            .copied()
            .unwrap_or(self.dx * self.dy)
    }

    fn face_normal(&self, _face: usize) -> [B::Scalar; 2] {
        let n_h_interior = (self.nx - 1) * self.ny;
        let n_v_interior = self.nx * (self.ny - 1);
        let boundary_start = n_h_interior + n_v_interior;

        if _face < n_h_interior {
            [B::Scalar::ONE, B::Scalar::ZERO]
        } else if _face < boundary_start {
            [B::Scalar::ZERO, B::Scalar::ONE]
        } else {
            let boundary_idx = _face - boundary_start;
            if boundary_idx < self.nx {
                [B::Scalar::ZERO, -B::Scalar::ONE]
            } else if boundary_idx < 2 * self.nx {
                [B::Scalar::ZERO, B::Scalar::ONE]
            } else if boundary_idx < 2 * self.nx + self.ny {
                [-B::Scalar::ONE, B::Scalar::ZERO]
            } else {
                [B::Scalar::ONE, B::Scalar::ZERO]
            }
        }
    }

    fn face_length(&self, _face: usize) -> B::Scalar {
        self.face_lengths.get(_face).copied().unwrap_or(self.dx)
    }

    fn face_center(&self, _face: usize) -> [B::Scalar; 2] {
        let n_h_interior = (self.nx - 1) * self.ny;
        let n_v_interior = self.nx * (self.ny - 1);
        let boundary_start = n_h_interior + n_v_interior;
        let half = self.config_scalar(0.5, "structured_mesh.face_center.half");

        if _face < n_h_interior {
            let i = _face % (self.nx - 1);
            let j = _face / (self.nx - 1);
            let x = self.dx
                * self.config_index(i + 1, "structured_mesh.face_center.horizontal.i_plus_1");
            let y =
                self.dy * (self.config_index(j, "structured_mesh.face_center.horizontal.j") + half);
            [x, y]
        } else if _face < boundary_start {
            let local = _face - n_h_interior;
            let i = local % self.nx;
            let j = local / self.nx;
            let x =
                self.dx * (self.config_index(i, "structured_mesh.face_center.vertical.i") + half);
            let y =
                self.dy * self.config_index(j + 1, "structured_mesh.face_center.vertical.j_plus_1");
            [x, y]
        } else {
            let boundary_idx = _face - boundary_start;
            if boundary_idx < self.nx {
                let i = boundary_idx;
                let x =
                    self.dx * (self.config_index(i, "structured_mesh.face_center.bottom.i") + half);
                [x, B::Scalar::ZERO]
            } else if boundary_idx < 2 * self.nx {
                let i = boundary_idx - self.nx;
                let x =
                    self.dx * (self.config_index(i, "structured_mesh.face_center.top.i") + half);
                let y = self.dy * self.config_index(self.ny, "structured_mesh.face_center.top.ny");
                [x, y]
            } else if boundary_idx < 2 * self.nx + self.ny {
                let j = boundary_idx - 2 * self.nx;
                let y =
                    self.dy * (self.config_index(j, "structured_mesh.face_center.left.j") + half);
                [B::Scalar::ZERO, y]
            } else {
                let j = boundary_idx - 2 * self.nx - self.ny;
                let x =
                    self.dx * self.config_index(self.nx, "structured_mesh.face_center.right.nx");
                let y =
                    self.dy * (self.config_index(j, "structured_mesh.face_center.right.j") + half);
                [x, y]
            }
        }
    }

    fn face_owner(&self, face: usize) -> usize {
        let n_h_interior = (self.nx - 1) * self.ny;
        let n_v_interior = self.nx * (self.ny - 1);
        let boundary_start = n_h_interior + n_v_interior;

        if face < n_h_interior {
            let i = face % (self.nx - 1);
            let j = face / (self.nx - 1);
            self.cell_index(i, j)
        } else if face < boundary_start {
            let local = face - n_h_interior;
            let i = local % self.nx;
            let j = local / self.nx;
            self.cell_index(i, j)
        } else {
            let boundary_idx = face - boundary_start;
            if boundary_idx < self.nx {
                let i = boundary_idx;
                self.cell_index(i, 0)
            } else if boundary_idx < 2 * self.nx {
                let i = boundary_idx - self.nx;
                self.cell_index(i, self.ny - 1)
            } else if boundary_idx < 2 * self.nx + self.ny {
                let j = boundary_idx - 2 * self.nx;
                self.cell_index(0, j)
            } else {
                let j = boundary_idx - 2 * self.nx - self.ny;
                self.cell_index(self.nx - 1, j)
            }
        }
    }

    fn face_neighbor(&self, face: usize) -> Option<usize> {
        let n_h_interior = (self.nx - 1) * self.ny;
        let n_v_interior = self.nx * (self.ny - 1);
        let boundary_start = n_h_interior + n_v_interior;

        if face < n_h_interior {
            let i = face % (self.nx - 1);
            let j = face / (self.nx - 1);
            Some(self.cell_index(i + 1, j))
        } else if face < boundary_start {
            let local = face - n_h_interior;
            let i = local % self.nx;
            let j = local / self.nx;
            Some(self.cell_index(i, j + 1))
        } else {
            None
        }
    }

    fn cell_faces(&self, cell: usize) -> &[usize] {
        self.cell_faces_cache
            .get(cell)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    fn cell_neighbors(&self, cell: usize) -> Vec<usize> {
        let (i, j) = self.cell_ij(cell);
        let mut neighbors = Vec::with_capacity(4);

        if i > 0 {
            neighbors.push(self.cell_index(i - 1, j));
        }
        if i < self.nx - 1 {
            neighbors.push(self.cell_index(i + 1, j));
        }
        if j > 0 {
            neighbors.push(self.cell_index(i, j - 1));
        }
        if j < self.ny - 1 {
            neighbors.push(self.cell_index(i, j + 1));
        }

        neighbors
    }

    fn boundary_faces(&self) -> &[usize] {
        &self.boundary_faces_cache
    }

    fn interior_faces(&self) -> &[usize] {
        &self.interior_faces_cache
    }

    fn mesh_kind(&self) -> MeshKind {
        MeshKind::Structured {
            nx: self.nx,
            ny: self.ny,
        }
    }

    fn cell_areas_buffer(&self) -> &B::Buffer<B::Scalar> {
        &self.cell_areas
    }

    fn face_lengths_buffer(&self) -> &B::Buffer<B::Scalar> {
        &self.face_lengths
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    fn test_backend() -> CpuBackend<f64> {
        CpuBackend::<f64>::new()
    }

    #[test]
    fn test_structured_mesh_creation() {
        let backend = test_backend();
        let mesh = StructuredMesh::new_with_backend(&backend, 10, 10, 1.0, 1.0).unwrap();

        assert_eq!(mesh.n_cells(), 100);
        assert_eq!(mesh.n_nodes(), 121);
    }

    #[test]
    fn test_cell_indexing() {
        let backend = test_backend();
        let mesh = StructuredMesh::new_with_backend(&backend, 10, 10, 1.0, 1.0).unwrap();

        assert_eq!(mesh.cell_index(0, 0), 0);
        assert_eq!(mesh.cell_index(9, 9), 99);
        assert_eq!(mesh.cell_ij(55), (5, 5));
    }

    #[test]
    fn test_cell_neighbors() {
        let backend = test_backend();
        let mesh = StructuredMesh::new_with_backend(&backend, 5, 5, 1.0, 1.0).unwrap();

        // 角点单元
        // 仅与两个邻居相连
        let neighbors = mesh.cell_neighbors(0);
        assert_eq!(neighbors.len(), 2);

        // 内部单元
        // 应有四个邻居
        let neighbors = mesh.cell_neighbors(12); // (2, 2)
        assert_eq!(neighbors.len(), 4);
    }

    #[test]
    fn test_cell_center() {
        let backend = test_backend();
        let mesh = StructuredMesh::new_with_backend(&backend, 10, 10, 1.0, 1.0).unwrap();

        let center = mesh.cell_center(0);
        assert!((center[0] - 0.5).abs() < 1e-10);
        assert!((center[1] - 0.5).abs() < 1e-10);
    }
}
