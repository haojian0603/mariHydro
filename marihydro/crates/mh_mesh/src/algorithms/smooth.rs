// marihydro\crates\mh_mesh\src/algorithms/smooth.rs

//! 网格光顺算法
//!
//! 支持:
//! - Laplacian 光顺
//! - Taubin 光顺 (防收缩)
//! - 加权 Laplacian 光顺

use std::collections::HashMap;

/// 光顺配置
#[derive(Debug, Clone)]
pub struct SmoothConfig {
    /// 迭代次数
    pub iterations: usize,
    /// 平滑因子 (0-1)
    pub lambda: f64,
    /// 是否固定边界
    pub fix_boundary: bool,
    /// 平滑方法
    pub method: SmoothMethod,
}

impl Default for SmoothConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            lambda: 0.5,
            fix_boundary: true,
            method: SmoothMethod::Laplacian,
        }
    }
}

impl SmoothConfig {
    /// 创建Laplacian光顺配置
    pub fn laplacian(iterations: usize, lambda: f64) -> Self {
        Self {
            iterations,
            lambda,
            method: SmoothMethod::Laplacian,
            ..Default::default()
        }
    }

    /// 创建Taubin光顺配置
    pub fn taubin(iterations: usize) -> Self {
        Self {
            iterations,
            lambda: 0.5,
            method: SmoothMethod::Taubin { mu: -0.53 },
            ..Default::default()
        }
    }
}

/// 平滑方法
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmoothMethod {
    /// 标准 Laplacian 光顺
    Laplacian,
    /// Taubin 光顺 (防收缩)
    Taubin {
        /// 反向平滑因子
        mu: f64,
    },
    /// 加权 Laplacian (cotangent权重)
    CotangentWeighted,
}

/// 网格光顺器
pub struct Smoother {
    config: SmoothConfig,
}

impl Smoother {
    /// 创建光顺器
    pub fn new(config: SmoothConfig) -> Self {
        Self { config }
    }

    /// 使用默认配置创建
    pub fn default_config() -> Self {
        Self::new(SmoothConfig::default())
    }

    /// 光顺2D点集
    ///
    /// # 参数
    /// - `vertices`: 顶点坐标
    /// - `neighbors`: 每个顶点的邻居索引
    /// - `boundary`: 边界顶点索引集合
    pub fn smooth_2d(
        &self,
        vertices: &mut [[f64; 2]],
        neighbors: &[Vec<usize>],
        boundary: &std::collections::HashSet<usize>,
    ) {
        match self.config.method {
            SmoothMethod::Laplacian => {
                self.laplacian_smooth_2d(vertices, neighbors, boundary);
            }
            SmoothMethod::Taubin { mu } => {
                self.taubin_smooth_2d(vertices, neighbors, boundary, mu);
            }
            SmoothMethod::CotangentWeighted => {
                // 对于2D，使用基于边长的权重
                self.weighted_smooth_2d(vertices, neighbors, boundary);
            }
        }
    }

    /// 光顺3D点集
    ///
    /// # 参数
    /// - `vertices`: 顶点坐标
    /// - `neighbors`: 每个顶点的邻居索引
    /// - `boundary`: 边界顶点索引集合
    pub fn smooth_3d(
        &self,
        vertices: &mut [[f64; 3]],
        neighbors: &[Vec<usize>],
        boundary: &std::collections::HashSet<usize>,
    ) {
        match self.config.method {
            SmoothMethod::Laplacian => {
                self.laplacian_smooth_3d(vertices, neighbors, boundary);
            }
            SmoothMethod::Taubin { mu } => {
                self.taubin_smooth_3d(vertices, neighbors, boundary, mu);
            }
            SmoothMethod::CotangentWeighted => {
                self.weighted_smooth_3d(vertices, neighbors, boundary);
            }
        }
    }

    /// 2D Laplacian 光顺
    fn laplacian_smooth_2d(
        &self,
        vertices: &mut [[f64; 2]],
        neighbors: &[Vec<usize>],
        boundary: &std::collections::HashSet<usize>,
    ) {
        let lambda = self.config.lambda;
        let n = vertices.len();

        for _ in 0..self.config.iterations {
            // 计算位移
            let mut displacements = vec![[0.0, 0.0]; n];

            for i in 0..n {
                // 跳过边界顶点
                if self.config.fix_boundary && boundary.contains(&i) {
                    continue;
                }

                let neighbors_i = &neighbors[i];
                if neighbors_i.is_empty() {
                    continue;
                }

                // 计算邻居中心
                let mut center = [0.0, 0.0];
                for &j in neighbors_i {
                    center[0] += vertices[j][0];
                    center[1] += vertices[j][1];
                }
                center[0] /= neighbors_i.len() as f64;
                center[1] /= neighbors_i.len() as f64;

                // 计算位移
                displacements[i][0] = lambda * (center[0] - vertices[i][0]);
                displacements[i][1] = lambda * (center[1] - vertices[i][1]);
            }

            // 应用位移
            for i in 0..n {
                vertices[i][0] += displacements[i][0];
                vertices[i][1] += displacements[i][1];
            }
        }
    }

    /// 3D Laplacian 光顺
    fn laplacian_smooth_3d(
        &self,
        vertices: &mut [[f64; 3]],
        neighbors: &[Vec<usize>],
        boundary: &std::collections::HashSet<usize>,
    ) {
        let lambda = self.config.lambda;
        let n = vertices.len();

        for _ in 0..self.config.iterations {
            let mut displacements = vec![[0.0, 0.0, 0.0]; n];

            for i in 0..n {
                if self.config.fix_boundary && boundary.contains(&i) {
                    continue;
                }

                let neighbors_i = &neighbors[i];
                if neighbors_i.is_empty() {
                    continue;
                }

                let mut center = [0.0, 0.0, 0.0];
                for &j in neighbors_i {
                    center[0] += vertices[j][0];
                    center[1] += vertices[j][1];
                    center[2] += vertices[j][2];
                }
                center[0] /= neighbors_i.len() as f64;
                center[1] /= neighbors_i.len() as f64;
                center[2] /= neighbors_i.len() as f64;

                displacements[i][0] = lambda * (center[0] - vertices[i][0]);
                displacements[i][1] = lambda * (center[1] - vertices[i][1]);
                displacements[i][2] = lambda * (center[2] - vertices[i][2]);
            }

            for i in 0..n {
                vertices[i][0] += displacements[i][0];
                vertices[i][1] += displacements[i][1];
                vertices[i][2] += displacements[i][2];
            }
        }
    }

    /// 2D Taubin 光顺 (防收缩)
    fn taubin_smooth_2d(
        &self,
        vertices: &mut [[f64; 2]],
        neighbors: &[Vec<usize>],
        boundary: &std::collections::HashSet<usize>,
        mu: f64,
    ) {
        let lambda = self.config.lambda;

        for _ in 0..self.config.iterations {
            // 正向平滑
            self.laplacian_step_2d(vertices, neighbors, boundary, lambda);
            // 反向平滑 (防止收缩)
            self.laplacian_step_2d(vertices, neighbors, boundary, mu);
        }
    }

    /// 3D Taubin 光顺
    fn taubin_smooth_3d(
        &self,
        vertices: &mut [[f64; 3]],
        neighbors: &[Vec<usize>],
        boundary: &std::collections::HashSet<usize>,
        mu: f64,
    ) {
        let lambda = self.config.lambda;

        for _ in 0..self.config.iterations {
            self.laplacian_step_3d(vertices, neighbors, boundary, lambda);
            self.laplacian_step_3d(vertices, neighbors, boundary, mu);
        }
    }

    /// 单步 2D Laplacian
    fn laplacian_step_2d(
        &self,
        vertices: &mut [[f64; 2]],
        neighbors: &[Vec<usize>],
        boundary: &std::collections::HashSet<usize>,
        factor: f64,
    ) {
        let n = vertices.len();
        let mut displacements = vec![[0.0, 0.0]; n];

        for i in 0..n {
            if self.config.fix_boundary && boundary.contains(&i) {
                continue;
            }

            let neighbors_i = &neighbors[i];
            if neighbors_i.is_empty() {
                continue;
            }

            let mut center = [0.0, 0.0];
            for &j in neighbors_i {
                center[0] += vertices[j][0];
                center[1] += vertices[j][1];
            }
            center[0] /= neighbors_i.len() as f64;
            center[1] /= neighbors_i.len() as f64;

            displacements[i][0] = factor * (center[0] - vertices[i][0]);
            displacements[i][1] = factor * (center[1] - vertices[i][1]);
        }

        for i in 0..n {
            vertices[i][0] += displacements[i][0];
            vertices[i][1] += displacements[i][1];
        }
    }

    /// 单步 3D Laplacian
    fn laplacian_step_3d(
        &self,
        vertices: &mut [[f64; 3]],
        neighbors: &[Vec<usize>],
        boundary: &std::collections::HashSet<usize>,
        factor: f64,
    ) {
        let n = vertices.len();
        let mut displacements = vec![[0.0, 0.0, 0.0]; n];

        for i in 0..n {
            if self.config.fix_boundary && boundary.contains(&i) {
                continue;
            }

            let neighbors_i = &neighbors[i];
            if neighbors_i.is_empty() {
                continue;
            }

            let mut center = [0.0, 0.0, 0.0];
            for &j in neighbors_i {
                center[0] += vertices[j][0];
                center[1] += vertices[j][1];
                center[2] += vertices[j][2];
            }
            center[0] /= neighbors_i.len() as f64;
            center[1] /= neighbors_i.len() as f64;
            center[2] /= neighbors_i.len() as f64;

            displacements[i][0] = factor * (center[0] - vertices[i][0]);
            displacements[i][1] = factor * (center[1] - vertices[i][1]);
            displacements[i][2] = factor * (center[2] - vertices[i][2]);
        }

        for i in 0..n {
            vertices[i][0] += displacements[i][0];
            vertices[i][1] += displacements[i][1];
            vertices[i][2] += displacements[i][2];
        }
    }

    /// 加权 2D 光顺 (基于边长)
    fn weighted_smooth_2d(
        &self,
        vertices: &mut [[f64; 2]],
        neighbors: &[Vec<usize>],
        boundary: &std::collections::HashSet<usize>,
    ) {
        let lambda = self.config.lambda;
        let n = vertices.len();

        for _ in 0..self.config.iterations {
            let mut displacements = vec![[0.0, 0.0]; n];

            for i in 0..n {
                if self.config.fix_boundary && boundary.contains(&i) {
                    continue;
                }

                let neighbors_i = &neighbors[i];
                if neighbors_i.is_empty() {
                    continue;
                }

                // 计算边长倒数作为权重
                let mut weights = Vec::with_capacity(neighbors_i.len());
                let mut weight_sum = 0.0;

                for &j in neighbors_i {
                    let dx = vertices[j][0] - vertices[i][0];
                    let dy = vertices[j][1] - vertices[i][1];
                    let dist = (dx * dx + dy * dy).sqrt();
                    let w = if dist > 1e-10 { 1.0 / dist } else { 1e10 };
                    weights.push(w);
                    weight_sum += w;
                }

                if weight_sum <= 0.0 {
                    continue;
                }

                // 加权平均
                let mut center = [0.0, 0.0];
                for (k, &j) in neighbors_i.iter().enumerate() {
                    let w = weights[k] / weight_sum;
                    center[0] += w * vertices[j][0];
                    center[1] += w * vertices[j][1];
                }

                displacements[i][0] = lambda * (center[0] - vertices[i][0]);
                displacements[i][1] = lambda * (center[1] - vertices[i][1]);
            }

            for i in 0..n {
                vertices[i][0] += displacements[i][0];
                vertices[i][1] += displacements[i][1];
            }
        }
    }

    /// 加权 3D 光顺
    fn weighted_smooth_3d(
        &self,
        vertices: &mut [[f64; 3]],
        neighbors: &[Vec<usize>],
        boundary: &std::collections::HashSet<usize>,
    ) {
        let lambda = self.config.lambda;
        let n = vertices.len();

        for _ in 0..self.config.iterations {
            let mut displacements = vec![[0.0, 0.0, 0.0]; n];

            for i in 0..n {
                if self.config.fix_boundary && boundary.contains(&i) {
                    continue;
                }

                let neighbors_i = &neighbors[i];
                if neighbors_i.is_empty() {
                    continue;
                }

                let mut weights = Vec::with_capacity(neighbors_i.len());
                let mut weight_sum = 0.0;

                for &j in neighbors_i {
                    let dx = vertices[j][0] - vertices[i][0];
                    let dy = vertices[j][1] - vertices[i][1];
                    let dz = vertices[j][2] - vertices[i][2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    let w = if dist > 1e-10 { 1.0 / dist } else { 1e10 };
                    weights.push(w);
                    weight_sum += w;
                }

                if weight_sum <= 0.0 {
                    continue;
                }

                let mut center = [0.0, 0.0, 0.0];
                for (k, &j) in neighbors_i.iter().enumerate() {
                    let w = weights[k] / weight_sum;
                    center[0] += w * vertices[j][0];
                    center[1] += w * vertices[j][1];
                    center[2] += w * vertices[j][2];
                }

                displacements[i][0] = lambda * (center[0] - vertices[i][0]);
                displacements[i][1] = lambda * (center[1] - vertices[i][1]);
                displacements[i][2] = lambda * (center[2] - vertices[i][2]);
            }

            for i in 0..n {
                vertices[i][0] += displacements[i][0];
                vertices[i][1] += displacements[i][1];
                vertices[i][2] += displacements[i][2];
            }
        }
    }
}

/// 从三角形列表构建邻接关系
pub fn build_neighbors_from_triangles(
    num_vertices: usize,
    triangles: &[[usize; 3]],
) -> Vec<Vec<usize>> {
    let mut neighbors: Vec<std::collections::HashSet<usize>> = 
        vec![std::collections::HashSet::new(); num_vertices];

    for tri in triangles {
        neighbors[tri[0]].insert(tri[1]);
        neighbors[tri[0]].insert(tri[2]);
        neighbors[tri[1]].insert(tri[0]);
        neighbors[tri[1]].insert(tri[2]);
        neighbors[tri[2]].insert(tri[0]);
        neighbors[tri[2]].insert(tri[1]);
    }

    neighbors.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// 从三角形列表提取边界顶点
pub fn find_boundary_vertices(
    num_vertices: usize,
    triangles: &[[usize; 3]],
) -> std::collections::HashSet<usize> {
    // 边计数，边界边只出现一次
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

    for tri in triangles {
        for i in 0..3 {
            let a = tri[i];
            let b = tri[(i + 1) % 3];
            let edge = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

    let mut boundary = std::collections::HashSet::with_capacity(num_vertices.saturating_div(2));
    for ((a, b), count) in edge_count {
        if count == 1 {
            boundary.insert(a);
            boundary.insert(b);
        }
    }

    boundary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_smooth_2d() {
        // 创建一个简单的三角形网格
        let mut vertices = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.1], // 稍微偏离的内点
            [0.0, 1.0],
            [1.0, 1.0],
        ];

        let triangles = vec![
            [0, 1, 2],
            [0, 2, 3],
            [2, 4, 3],
            [1, 4, 2],
        ];

        let neighbors = build_neighbors_from_triangles(5, &triangles);
        let boundary = find_boundary_vertices(5, &triangles);

        let smoother = Smoother::new(SmoothConfig::laplacian(5, 0.5));
        smoother.smooth_2d(&mut vertices, &neighbors, &boundary);

        // 内点应该被平滑
        // 边界点应该保持不变（fix_boundary=true）
        assert!((vertices[0][0] - 0.0).abs() < 1e-10);
        assert!((vertices[0][1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_taubin_smooth() {
        let mut vertices = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [1.0, 1.0],
        ];

        let triangles = vec![
            [0, 1, 2],
            [0, 2, 3],
            [1, 4, 2],
            [2, 4, 3],
        ];

        let neighbors = build_neighbors_from_triangles(5, &triangles);
        let boundary = find_boundary_vertices(5, &triangles);

        // 记录初始体积（面积）
        let initial_area: f64 = triangles.iter().map(|t| {
            let a = vertices[t[0]];
            let b = vertices[t[1]];
            let c = vertices[t[2]];
            let cross: f64 = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
            cross.abs() / 2.0
        }).sum();

        let smoother = Smoother::new(SmoothConfig::taubin(10));
        smoother.smooth_2d(&mut vertices, &neighbors, &boundary);

        // Taubin 应该保持体积
        let final_area: f64 = triangles.iter().map(|t| {
            let a = vertices[t[0]];
            let b = vertices[t[1]];
            let c = vertices[t[2]];
            let cross: f64 = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
            cross.abs() / 2.0
        }).sum();

        // 面积应该接近（允许一定误差）
        assert!((initial_area - final_area).abs() / initial_area < 0.1);
    }

    #[test]
    fn test_build_neighbors() {
        let triangles = vec![
            [0, 1, 2],
            [1, 3, 2],
        ];

        let neighbors = build_neighbors_from_triangles(4, &triangles);

        assert!(neighbors[0].contains(&1));
        assert!(neighbors[0].contains(&2));
        assert!(neighbors[1].contains(&0));
        assert!(neighbors[1].contains(&2));
        assert!(neighbors[1].contains(&3));
    }

    #[test]
    fn test_find_boundary() {
        let triangles = vec![
            [0, 1, 2],
            [1, 3, 2],
        ];

        let boundary = find_boundary_vertices(4, &triangles);

        // 所有顶点都在边界上（开放网格）
        assert!(boundary.contains(&0));
        assert!(boundary.contains(&1));
        assert!(boundary.contains(&2));
        assert!(boundary.contains(&3));
    }
}
