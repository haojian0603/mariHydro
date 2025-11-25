// src-tauri/src/marihydro/domain/rasterizer.rs

use crate::marihydro::io::types::GeoTransform;
use ndarray::Array2;

pub struct Rasterizer;

impl Rasterizer {
    /// 将多边形区域的值“印”到 2D 数组上
    ///
    /// # 参数
    /// * `poly_points`: 多边形顶点序列 [(x, y), ...] (地理坐标)
    /// * `geo_transform`: 网格坐标映射关系
    /// * `target_array`: 目标数组 (将被修改)
    /// * `value`: 填充值
    pub fn apply_polygon(
        poly_points: &[(f64, f64)],
        geo_transform: &GeoTransform,
        target_array: &mut Array2<f64>,
        value: f64,
    ) {
        if poly_points.len() < 3 {
            return;
        }

        // 1. 计算多边形的包围盒 (Bounding Box) 以减少遍历范围
        let (min_x, min_y, max_x, max_y) = Self::compute_bbox(poly_points);

        // 2. 将地理包围盒转换为网格索引范围
        // 注意：需要处理 GeoTransform 的逆变换
        // 这里简化假设：网格是正交的，直接反算索引范围粗略值
        let (dx, dy) = geo_transform.resolution();
        let (origin_x, origin_y) = (geo_transform.0[0], geo_transform.0[3]);

        // 简单的索引估算 (需处理边界检查)
        let (ny, nx) = target_array.dim();

        // 遍历全图或优化遍历 (这里为稳健性先遍历全图，工程优化可限制 loop 范围)
        // 在生产级代码中，应使用扫描线算法 (Scanline) 或 QuadTree
        for j in 0..ny {
            for i in 0..nx {
                let (px, py) = geo_transform.pixel_to_world(i as f64, j as f64);

                // 快速包围盒剔除
                if px < min_x || px > max_x || py < min_y || py > max_y {
                    continue;
                }

                // 精确几何测试 (射线法)
                if Self::point_in_polygon((px, py), poly_points) {
                    target_array[[j, i]] = value;
                }
            }
        }
    }

    /// 射线法判断点是否在多边形内
    fn point_in_polygon(point: (f64, f64), poly: &[(f64, f64)]) -> bool {
        let (x, y) = point;
        let mut inside = false;
        let n = poly.len();
        let mut j = n - 1;

        for i in 0..n {
            let (xi, yi) = poly[i];
            let (xj, yj) = poly[j];

            let intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);

            if intersect {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    fn compute_bbox(points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for &(x, y) in points {
            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }
        }
        (min_x, min_y, max_x, max_y)
    }
}
