// src-tauri/src/marihydro/domain/geometry_mapper.rs

use glam::DVec2;
use rayon::prelude::*;

use crate::marihydro::domain::mesh::unstructured::{CellEnvelope, UnstructuredMesh};
use crate::marihydro::io::traits::GeoTransform;

pub struct GeometryMapper;

impl GeometryMapper {
    pub fn apply_polygon_to_mesh(
        poly_points: &[(f64, f64)],
        mesh: &UnstructuredMesh,
        target: &mut [f64],
        value: f64,
    ) -> usize {
        if poly_points.len() < 3 || target.len() != mesh.n_cells {
            return 0;
        }

        let bbox = Self::compute_bbox(poly_points);
        let candidates = Self::query_cells_in_bbox(mesh, bbox);

        if candidates.is_empty() {
            return 0;
        }

        let results: Vec<(usize, bool)> = candidates
            .par_iter()
            .map(|&cell_idx| {
                let center = mesh.cell_center[cell_idx];
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
        count
    }

    pub fn apply_polygon_blend(
        poly_points: &[(f64, f64)],
        mesh: &UnstructuredMesh,
        target: &mut [f64],
        value: f64,
        blend_factor: f64,
    ) -> usize {
        if poly_points.len() < 3 || target.len() != mesh.n_cells {
            return 0;
        }

        let bbox = Self::compute_bbox(poly_points);
        let candidates = Self::query_cells_in_bbox(mesh, bbox);

        let results: Vec<(usize, bool)> = candidates
            .par_iter()
            .map(|&cell_idx| {
                let center = mesh.cell_center[cell_idx];
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
        count
    }

    pub fn map_raster_to_mesh(
        raster_data: &[f64],
        raster_width: usize,
        raster_height: usize,
        geo_transform: &GeoTransform,
        mesh: &UnstructuredMesh,
        target: &mut [f64],
        nodata_value: Option<f64>,
    ) {
        assert_eq!(target.len(), mesh.n_cells);
        assert_eq!(raster_data.len(), raster_width * raster_height);

        target
            .par_iter_mut()
            .enumerate()
            .for_each(|(cell_idx, out)| {
                let center = mesh.cell_center[cell_idx];
                if let Some((col, row)) = geo_transform.inverse(center.x, center.y) {
                    let col_i = col.floor() as isize;
                    let row_i = row.floor() as isize;

                    if col_i >= 0
                        && row_i >= 0
                        && (col_i as usize) < raster_width - 1
                        && (row_i as usize) < raster_height - 1
                    {
                        let value = Self::bilinear_sample(
                            raster_data,
                            raster_width,
                            col,
                            row,
                            nodata_value,
                        );
                        if let Some(v) = value {
                            *out = v;
                        }
                    }
                }
            });
    }

    #[inline]
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

    fn query_cells_in_bbox(mesh: &UnstructuredMesh, bbox: (f64, f64, f64, f64)) -> Vec<usize> {
        let (min_x, min_y, max_x, max_y) = bbox;
        let query_aabb = rstar::AABB::from_corners([min_x, min_y], [max_x, max_y]);

        mesh.spatial_index
            .locate_in_envelope_intersecting(&query_aabb)
            .map(|env: &CellEnvelope| env.cell_id.idx())
            .collect()
    }

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
}
