//! 结构化网格（规则矩形网格实现）

use super::topology::{MeshKind, MeshTopology};
use crate::core::{Backend, DeviceBuffer, Scalar};

/// 结构化规则网格
#[allow(dead_code)]
pub struct StructuredMesh<B: Backend> {
    nx: usize,
    ny: usize,
    dx: B::Scalar,
    dy: B::Scalar,
    cell_areas: B::Buffer<B::Scalar>,
    face_lengths: B::Buffer<B::Scalar>,
    boundary_faces_cache: Vec<usize>,
    interior_faces_cache: Vec<usize>,
    cell_faces_cache: Vec<Vec<usize>>,
}

impl<B> StructuredMesh<B>
where
    B: Backend + Default,
{
    /// 创建规则网格（原点固定在 `(0, 0)`）
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        assert!(nx > 0 && ny > 0, "StructuredMesh 要求 nx>0 且 ny>0");
        assert!(dx > 0.0 && dy > 0.0, "StructuredMesh 要求 dx>0 且 dy>0");

        let backend = B::default();
        let n_cells = nx * ny;
        let n_vfaces = (nx + 1) * ny;
        let n_hfaces = nx * (ny + 1);
        let n_faces = n_vfaces + n_hfaces;

        let dx_s = <B::Scalar as Scalar>::from_f64(dx);
        let dy_s = <B::Scalar as Scalar>::from_f64(dy);
        let area = dx_s * dy_s;

        let mut face_lengths_host = vec![<B::Scalar as Scalar>::ZERO; n_faces];
        for j in 0..ny {
            for i in 0..=nx {
                let f = Self::vface_index_static(nx, j, i);
                face_lengths_host[f] = dy_s;
            }
        }

        let hstart = n_vfaces;
        for j in 0..=ny {
            for i in 0..nx {
                let f = hstart + j * nx + i;
                face_lengths_host[f] = dx_s;
            }
        }

        let mut boundary_faces = Vec::new();
        let mut interior_faces = Vec::new();

        for j in 0..ny {
            for i in 0..=nx {
                let f = Self::vface_index_static(nx, j, i);
                if i == 0 || i == nx {
                    boundary_faces.push(f);
                } else {
                    interior_faces.push(f);
                }
            }
        }

        for j in 0..=ny {
            for i in 0..nx {
                let f = hstart + j * nx + i;
                if j == 0 || j == ny {
                    boundary_faces.push(f);
                } else {
                    interior_faces.push(f);
                }
            }
        }

        let mut cell_faces = vec![Vec::with_capacity(4); n_cells];
        for j in 0..ny {
            for i in 0..nx {
                let c = Self::cell_index_static(nx, j, i);
                let west = Self::vface_index_static(nx, j, i);
                let east = Self::vface_index_static(nx, j, i + 1);
                let south = hstart + j * nx + i;
                let north = hstart + (j + 1) * nx + i;
                cell_faces[c].extend_from_slice(&[west, east, south, north]);
            }
        }

        Self {
            nx,
            ny,
            dx: dx_s,
            dy: dy_s,
            cell_areas: backend.alloc_init(n_cells, area),
            face_lengths: {
                let mut b = backend.alloc_init(n_faces, <B::Scalar as Scalar>::ZERO);
                b.copy_from_slice(&face_lengths_host);
                b
            },
            boundary_faces_cache: boundary_faces,
            interior_faces_cache: interior_faces,
            cell_faces_cache: cell_faces,
        }
    }
}

impl<B: Backend> StructuredMesh<B> {
    #[inline]
    fn n_vfaces(&self) -> usize {
        (self.nx + 1) * self.ny
    }

    #[inline]
    fn cell_index_static(nx: usize, j: usize, i: usize) -> usize {
        j * nx + i
    }

    #[inline]
    fn vface_index_static(nx: usize, j: usize, i: usize) -> usize {
        j * (nx + 1) + i
    }
}

impl<B: Backend> MeshTopology<B> for StructuredMesh<B> {
    fn n_cells(&self) -> usize {
        self.nx * self.ny
    }

    fn n_faces(&self) -> usize {
        (self.nx + 1) * self.ny + self.nx * (self.ny + 1)
    }

    fn n_interior_faces(&self) -> usize {
        self.nx.saturating_sub(1) * self.ny + self.nx * self.ny.saturating_sub(1)
    }

    fn n_nodes(&self) -> usize {
        (self.nx + 1) * (self.ny + 1)
    }

    fn cell_center(&self, cell: usize) -> [B::Scalar; 2] {
        let i = cell % self.nx;
        let j = cell / self.nx;
        let half = <B::Scalar as Scalar>::from_f64(0.5);
        let x = (<B::Scalar as Scalar>::from_f64(i as f64) + half) * self.dx;
        let y = (<B::Scalar as Scalar>::from_f64(j as f64) + half) * self.dy;
        [x, y]
    }

    fn cell_area(&self, _cell: usize) -> B::Scalar {
        self.dx * self.dy
    }

    fn face_normal(&self, face: usize) -> [B::Scalar; 2] {
        let n_vfaces = self.n_vfaces();
        if face < n_vfaces {
            let i = face % (self.nx + 1);
            if i == 0 {
                [-<B::Scalar as Scalar>::ONE, <B::Scalar as Scalar>::ZERO]
            } else {
                [<B::Scalar as Scalar>::ONE, <B::Scalar as Scalar>::ZERO]
            }
        } else {
            let local = face - n_vfaces;
            let j = local / self.nx;
            if j == 0 {
                [<B::Scalar as Scalar>::ZERO, -<B::Scalar as Scalar>::ONE]
            } else {
                [<B::Scalar as Scalar>::ZERO, <B::Scalar as Scalar>::ONE]
            }
        }
    }

    fn face_length(&self, face: usize) -> B::Scalar {
        self.face_lengths
            .as_slice()
            .and_then(|s| s.get(face).copied())
            .unwrap_or(<B::Scalar as Scalar>::ZERO)
    }

    fn face_center(&self, face: usize) -> [B::Scalar; 2] {
        let n_vfaces = self.n_vfaces();
        if face < n_vfaces {
            let i = face % (self.nx + 1);
            let j = face / (self.nx + 1);
            let x = <B::Scalar as Scalar>::from_f64(i as f64) * self.dx;
            let y = (<B::Scalar as Scalar>::from_f64(j as f64)
                + <B::Scalar as Scalar>::from_f64(0.5))
                * self.dy;
            [x, y]
        } else {
            let local = face - n_vfaces;
            let i = local % self.nx;
            let j = local / self.nx;
            let x = (<B::Scalar as Scalar>::from_f64(i as f64)
                + <B::Scalar as Scalar>::from_f64(0.5))
                * self.dx;
            let y = <B::Scalar as Scalar>::from_f64(j as f64) * self.dy;
            [x, y]
        }
    }

    fn face_owner(&self, face: usize) -> usize {
        let n_vfaces = self.n_vfaces();
        if face < n_vfaces {
            let i = face % (self.nx + 1);
            let j = face / (self.nx + 1);
            if i == 0 {
                Self::cell_index_static(self.nx, j, 0)
            } else {
                Self::cell_index_static(self.nx, j, i - 1)
            }
        } else {
            let local = face - n_vfaces;
            let i = local % self.nx;
            let j = local / self.nx;
            if j == 0 {
                Self::cell_index_static(self.nx, 0, i)
            } else {
                Self::cell_index_static(self.nx, j - 1, i)
            }
        }
    }

    fn face_neighbor(&self, face: usize) -> Option<usize> {
        let n_vfaces = self.n_vfaces();
        if face < n_vfaces {
            let i = face % (self.nx + 1);
            let j = face / (self.nx + 1);
            if i == 0 || i == self.nx {
                None
            } else {
                Some(Self::cell_index_static(self.nx, j, i))
            }
        } else {
            let local = face - n_vfaces;
            let i = local % self.nx;
            let j = local / self.nx;
            if j == 0 || j == self.ny {
                None
            } else {
                Some(Self::cell_index_static(self.nx, j, i))
            }
        }
    }

    fn cell_faces(&self, cell: usize) -> &[usize] {
        &self.cell_faces_cache[cell]
    }

    fn cell_neighbors(&self, cell: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(4);
        for &f in &self.cell_faces_cache[cell] {
            let owner = self.face_owner(f);
            if owner == cell {
                if let Some(n) = self.face_neighbor(f) {
                    out.push(n);
                }
            } else {
                out.push(owner);
            }
        }
        out
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
