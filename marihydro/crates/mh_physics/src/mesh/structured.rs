//! 缁撴瀯鍖栫綉鏍煎疄鐜?
//!
//! 鎻愪緵瀹屾暣鐨勭粨鏋勫寲缃戞牸鎷撴墤涓庡嚑浣曡绠椼€?
//!
//! # 璁捐璇存槑
//!
//! - 瀛樺偍 Backend 瀹炰緥浠ユ敮鎸佸姩鎬佺紦鍐插尯鍒嗛厤
//! - 绉婚櫎鍐椾綑鐨?PhantomData锛圔ackend 宸查€氳繃瀛楁浣跨敤锛?
//! - 鎵€鏈夋瀯閫犲嚱鏁板繀椤绘帴鏀?Backend 鍙傛暟

use crate::core::Backend;
use mh_runtime::{DeviceBuffer, RuntimeScalar};
use super::topology::{MeshKind, MeshTopology, MeshValidationError};

/// 缁撴瀯鍖栫綉鏍?
///
/// 瑙勫垯鐭╁舰缃戞牸锛屾敮鎸侀珮鏁堢殑绱㈠紩璁＄畻鍜?SIMD 浼樺寲銆?
///
/// # 绫诲瀷鍙傛暟
///
/// - `B`: 璁＄畻鍚庣绫诲瀷锛堟棤榛樿鍊硷級
#[allow(dead_code)]
pub struct StructuredMesh<B: Backend> {
    /// 璁＄畻鍚庣瀹炰緥
    backend: B,
    /// x 鏂瑰悜鍗曞厓鏁?
    nx: usize,
    /// y 鏂瑰悜鍗曞厓鏁?
    ny: usize,
    /// x 鏂瑰悜缃戞牸闂磋窛
    dx: B::Scalar,
    /// y 鏂瑰悜缃戞牸闂磋窛
    dy: B::Scalar,
    /// 鍗曞厓闈㈢Н缂撳啿鍖?
    cell_areas: B::Buffer<B::Scalar>,
    /// 闈㈤暱搴︾紦鍐插尯
    face_lengths: B::Buffer<B::Scalar>,
    /// 杈圭晫闈㈢储寮曠紦瀛?
    boundary_faces_cache: Vec<usize>,
    /// 鍐呴儴闈㈢储寮曠紦瀛?
    interior_faces_cache: Vec<usize>,
    /// 鍗曞厓-闈㈡槧灏勭紦瀛?
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
    /// 浣跨敤 Backend 鍒涘缓缁撴瀯鍖栫綉鏍?
    ///
    /// # 鍙傛暟
    ///
    /// - `backend`: 璁＄畻鍚庣瀹炰緥
    /// - `nx`: x 鏂瑰悜鍗曞厓鏁?
    /// - `ny`: y 鏂瑰悜鍗曞厓鏁?
    /// - `dx`: x 鏂瑰悜缃戞牸闂磋窛
    /// - `dy`: y 鏂瑰悜缃戞牸闂磋窛
    ///
    /// # 绀轰緥
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
        
        // 浣跨敤 Backend 鍒嗛厤缂撳啿鍖?
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

            // 杈圭晫闈細涓嬨€佷笂銆佸乏銆佸彸
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
        
        // 鏋勫缓闈㈢储寮曠紦瀛?
        let (boundary_faces_cache, interior_faces_cache) = 
            Self::build_face_caches(nx, ny);
        
        // 鏋勫缓鍗曞厓-闈㈡槧灏?
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
    
    /// 璁＄畻鎬婚潰鏁?
    #[inline]
    fn compute_n_faces(nx: usize, ny: usize) -> usize {
        // 姘村钩鍐呴儴闈?+ 鍨傜洿鍐呴儴闈?+ 杈圭晫闈?
        (nx - 1) * ny + nx * (ny - 1) + 2 * (nx + ny)
    }
    
    /// 鏋勫缓杈圭晫鍜屽唴閮ㄩ潰缂撳瓨
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
    
    /// 鏋勫缓鍗曞厓-闈㈡槧灏?
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
    
    /// 鑾峰彇鍚庣寮曠敤
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 鑾峰彇 x 鏂瑰悜鍗曞厓鏁?
    #[inline]
    pub fn nx(&self) -> usize {
        self.nx
    }
    
    /// 鑾峰彇 y 鏂瑰悜鍗曞厓鏁?
    #[inline]
    pub fn ny(&self) -> usize {
        self.ny
    }
    
    /// 灏?(i, j) 绱㈠紩杞崲涓哄崟鍏冪储寮?
    #[inline]
    pub fn cell_index(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
    
    /// 灏嗗崟鍏冪储寮曡浆鎹负 (i, j) 绱㈠紩
    #[inline]
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
        self.cell_areas.get(cell).copied().unwrap_or(self.dx * self.dy)
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
            let x = self.dx * self.config_index(i + 1, "structured_mesh.face_center.horizontal.i_plus_1");
            let y = self.dy * (self.config_index(j, "structured_mesh.face_center.horizontal.j") + half);
            [x, y]
        } else if _face < boundary_start {
            let local = _face - n_h_interior;
            let i = local % self.nx;
            let j = local / self.nx;
            let x = self.dx * (self.config_index(i, "structured_mesh.face_center.vertical.i") + half);
            let y = self.dy * self.config_index(j + 1, "structured_mesh.face_center.vertical.j_plus_1");
            [x, y]
        } else {
            let boundary_idx = _face - boundary_start;
            if boundary_idx < self.nx {
                let i = boundary_idx;
                let x = self.dx * (self.config_index(i, "structured_mesh.face_center.bottom.i") + half);
                [x, B::Scalar::ZERO]
            } else if boundary_idx < 2 * self.nx {
                let i = boundary_idx - self.nx;
                let x = self.dx * (self.config_index(i, "structured_mesh.face_center.top.i") + half);
                let y = self.dy * self.config_index(self.ny, "structured_mesh.face_center.top.ny");
                [x, y]
            } else if boundary_idx < 2 * self.nx + self.ny {
                let j = boundary_idx - 2 * self.nx;
                let y = self.dy * (self.config_index(j, "structured_mesh.face_center.left.j") + half);
                [B::Scalar::ZERO, y]
            } else {
                let j = boundary_idx - 2 * self.nx - self.ny;
                let x = self.dx * self.config_index(self.nx, "structured_mesh.face_center.right.nx");
                let y = self.dy * (self.config_index(j, "structured_mesh.face_center.right.j") + half);
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
        self.cell_faces_cache.get(cell).map(|v| v.as_slice()).unwrap_or(&[])
    }
    
    fn cell_neighbors(&self, cell: usize) -> Vec<usize> {
        let (i, j) = self.cell_ij(cell);
        let mut neighbors = Vec::with_capacity(4);
        
        if i > 0 { neighbors.push(self.cell_index(i - 1, j)); }
        if i < self.nx - 1 { neighbors.push(self.cell_index(i + 1, j)); }
        if j > 0 { neighbors.push(self.cell_index(i, j - 1)); }
        if j < self.ny - 1 { neighbors.push(self.cell_index(i, j + 1)); }
        
        neighbors
    }
    
    fn boundary_faces(&self) -> &[usize] {
        &self.boundary_faces_cache
    }
    
    fn interior_faces(&self) -> &[usize] {
        &self.interior_faces_cache
    }
    
    fn mesh_kind(&self) -> MeshKind {
        MeshKind::Structured { nx: self.nx, ny: self.ny }
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
        
        // 瑙掕惤鍗曞厓
        let neighbors = mesh.cell_neighbors(0);
        assert_eq!(neighbors.len(), 2);
        
        // 涓績鍗曞厓
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

