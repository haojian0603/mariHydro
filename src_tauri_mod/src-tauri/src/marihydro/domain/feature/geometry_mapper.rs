// src-tauri/src/marihydro/domain/feature/geometry_mapper.rs

//! 几何映射器

use glam::DVec2;
use rayon::prelude::*;

use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::types::GeoTransform;
use crate::marihydro::domain::mesh::UnstructuredMesh;

/// 几何映射器
///
/// 提供多边形、栅格到网格的映射功能
pub struct GeometryMapper;

impl GeometryMapper {
    /// 将多边形值应用到网格
    pub fn apply_polygon_to_mesh(
        poly_points: &[(f64, f64)],
        mesh: &UnstructuredMesh,
        target: &mut [f64],
        value: f64,
    ) -> MhResult<usize> {
        use crate::marihydro::core::traits::mesh::MeshAccess;

        if poly_points.len() < 3 {
            return Err(MhError::InvalidInput {
                message: "多边形至少需要3个点".into(),
            });
        }

        if target.len() != mesh.n_cells() {
            return Err(MhError::size_mismatch(
                "target",
                mesh.n_cells(),
                target.len(),
            ));
        }

        let bbox = Self::compute_bbox(poly_points);
        let candidates = Self::query_cells_in_bbox(mesh, bbox);

        if candidates.is_empty() {
            return Ok(0);
        }

        let results: Vec<(usize, bool)> = candidates
            .par_iter()
            .map(|&cell_idx| {
                let center = mesh.cell_centroid(crate::marihydro::core::types::CellIndex(cell_idx));
                let inside = Self::point_in_polygon((center.x, center.y), poly_points);
                (cell_idx, inside)
            })
            .collect();

        let mut count = 0;
        for (cell_idx, inside) in results {
            if inside {
                target[cell_idx] = value;
                count += 1;
            }
        }

        Ok(count)
    }

    /// 将多边形值混合应用到网格
    pub fn apply_polygon_blend(
        poly_points: &[(f64, f64)],
        mesh: &UnstructuredMesh,
        target: &mut [f64],
        value: f64,
        blend_factor: f64,
    ) -> MhResult<usize> {
        use crate::marihydro::core::traits::mesh::MeshAccess;

        if poly_points.len() < 3 {
            return Err(MhError::InvalidInput {
                message: "多边形至少需要3个点".into(),
            });
        }

        if target.len() != mesh.n_cells() {
            return Err(MhError::size_mismatch(
                "target",
                mesh.n_cells(),
                target.len(),
            ));
        }

        let bbox = Self::compute_bbox(poly_points);
        let candidates = Self::query_cells_in_bbox(mesh, bbox);

        let results: Vec<(usize, bool)> = candidates
            .par_iter()
            .map(|&cell_idx| {
                let center = mesh.cell_centroid(crate::marihydro::core::types::CellIndex(cell_idx));
                let inside = Self::point_in_polygon((center.x, center.y), poly_points);
                (cell_idx, inside)
            })
            .collect();

        let alpha = blend_factor.clamp(0.0, 1.0);
        let beta = 1.0 - alpha;
        let mut count = 0;

        for (cell_idx, inside) in results {
            if inside {
                target[cell_idx] = target[cell_idx] * beta + value * alpha;
                count += 1;
            }
        }

        Ok(count)
    }

    /// 将栅格映射到网格
    pub fn map_raster_to_mesh(
        raster_data: &[f64],
        raster_dims: (usize, usize),
        geo_transform: &GeoTransform,
        mesh: &UnstructuredMesh,
        target: &mut [f64],
        nodata_value: Option<f64>,
    ) -> MhResult<()> {
        use crate::marihydro::core::traits::mesh::MeshAccess;

        let (raster_width, raster_height) = raster_dims;

        if raster_data.len() != raster_width * raster_height {
            return Err(MhError::size_mismatch(
                "raster_data",
                raster_width * raster_height,
                raster_data.len(),
            ));
        }

        if target.len() != mesh.n_cells() {
            return Err(MhError::size_mismatch(
                "target",
                mesh.n_cells(),
                target.len(),
            ));
        }

        target
            .par_iter_mut()
            .enumerate()
            .for_each(|(cell_idx, out)| {
                let center = mesh.cell_centroid(crate::marihydro::core::types::CellIndex(cell_idx));

                if let Some((col, row)) = geo_transform.geo_to_pixel(center.x, center.y) {
                    let col_i = col.floor() as isize;
                    let row_i = row.floor() as isize;

                    if col_i >= 0
                        && row_i >= 0
                        && (col_i as usize) < raster_width - 1
                        && (row_i as usize) < raster_height - 1
                    {
                        if let Some(value) =
                            Self::bilinear_sample(raster_data, raster_width, col, row, nodata_value)
                        {
                            *out = value;
                        }
                    }
                }
            });

        Ok(())
    }

    /// 双线性采样
    fn bilinear_sample(
        data: &[f64],
        width: usize,
        col: f64,
        row: f64,
        nodata: Option<f64>,
    ) -> Option<f64> {
        let c0 = col.floor() as usize;
        let r0 = row.floor() as usize;
        let c1 = c0 + 1;
        let r1 = r0 + 1;

        let fc = col - c0 as f64;
        let fr = row - r0 as f64;

        let get = |c: usize, r: usize| -> Option<f64> {
            let v = data[r * width + c];
            if let Some(nd) = nodata {
                if (v - nd).abs() < 1e-10 || v.is_nan() {
                    return None;
                }
            }
            if !v.is_finite() {
                return None;
            }
            Some(v)
        };

        let v00 = get(c0, r0)?;
        let v10 = get(c1, r0)?;
        let v01 = get(c0, r1)?;
        let v11 = get(c1, r1)?;

        let v0 = v00 * (1.0 - fc) + v10 * fc;
        let v1 = v01 * (1.0 - fc) + v11 * fc;
        Some(v0 * (1.0 - fr) + v1 * fr)
    }

    /// 查询边界框内的单元
    fn query_cells_in_bbox(mesh: &UnstructuredMesh, bbox: (f64, f64, f64, f64)) -> Vec<usize> {
        use crate::marihydro::core::traits::mesh::MeshAccess;

        let (min_x, min_y, max_x, max_y) = bbox;
        let mut cells = Vec::new();

        for i in 0..mesh.n_cells() {
            let center = mesh.cell_centroid(crate::marihydro::core::types::CellIndex(i));
            if center.x >= min_x && center.x <= max_x && center.y >= min_y && center.y <= max_y {
                cells.push(i);
            }
        }

        cells
    }

    /// 点是否在多边形内（绕数法）
    fn point_in_polygon(point: (f64, f64), poly: &[(f64, f64)]) -> bool {
        let (px, py) = point;
        let n = poly.len();
        let mut winding = 0i32;

        for i in 0..n {
            let (x1, y1) = poly[i];
            let (x2, y2) = poly[(i + 1) % n];

            if y1 <= py {
                if y2 > py {
                    let cross = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1);
                    if cross > 0.0 {
                        winding += 1;
                    }
                }
            } else if y2 <= py {
                let cross = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1);
                if cross < 0.0 {
                    winding -= 1;
                }
            }
        }

        winding != 0
    }

    /// 计算边界框
    fn compute_bbox(points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for &(x, y) in points {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        (min_x, min_y, max_x, max_y)
    }

    /// 计算多边形面积（Shoelace 公式）
    pub fn polygon_area(points: &[(f64, f64)]) -> f64 {
        let n = points.len();
        if n < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        for i in 0..n {
            let (x1, y1) = points[i];
            let (x2, y2) = points[(i + 1) % n];
            area += x1 * y2 - x2 * y1;
        }

        area.abs() / 2.0
    }

    /// 计算多边形质心
    pub fn polygon_centroid(points: &[(f64, f64)]) -> Option<(f64, f64)> {
        let n = points.len();
        if n < 3 {
            return None;
        }

        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut signed_area = 0.0;

        for i in 0..n {
            let (x0, y0) = points[i];
            let (x1, y1) = points[(i + 1) % n];

            let a = x0 * y1 - x1 * y0;
            signed_area += a;
            cx += (x0 + x1) * a;
            cy += (y0 + y1) * a;
        }

        if signed_area.abs() < 1e-14 {
            return None;
        }

        signed_area *= 0.5;
        cx /= 6.0 * signed_area;
        cy /= 6.0 * signed_area;

        Some((cx, cy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_in_polygon_triangle() {
        let triangle = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)];
        assert!(GeometryMapper::point_in_polygon((1.0, 0.5), &triangle));
        assert!(!GeometryMapper::point_in_polygon((3.0, 0.0), &triangle));
    }

    #[test]
    fn test_point_in_polygon_square() {
        let square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(GeometryMapper::point_in_polygon((0.5, 0.5), &square));
        assert!(!GeometryMapper::point_in_polygon((1.5, 0.5), &square));
    }

    #[test]
    fn test_compute_bbox() {
        let points = [(1.0, 2.0), (3.0, 4.0), (0.0, 1.0)];
        let bbox = GeometryMapper::compute_bbox(&points);
        assert_eq!(bbox, (0.0, 1.0, 3.0, 4.0));
    }

    #[test]
    fn test_bilinear_sample() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let result = GeometryMapper::bilinear_sample(&data, 2, 0.5, 0.5, None);
        assert!((result.unwrap() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_area() {
        // 单位正方形
        let square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let area = GeometryMapper::polygon_area(&square);
        assert!((area - 1.0).abs() < 1e-10);

        // 三角形
        let triangle = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)];
        let area = GeometryMapper::polygon_area(&triangle);
        assert!((area - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_centroid() {
        // 正方形质心
        let square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let (cx, cy) = GeometryMapper::polygon_centroid(&square).unwrap();
        assert!((cx - 1.0).abs() < 1e-10);
        assert!((cy - 1.0).abs() < 1e-10);
    }
}
