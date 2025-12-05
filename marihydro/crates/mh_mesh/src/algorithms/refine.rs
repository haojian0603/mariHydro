// marihydro\crates\mh_mesh\src/algorithms/refine.rs

//! 网格细化算法
//!
//! 提供自适应网格加密功能。

use std::collections::{HashMap, HashSet};

/// 细化配置
#[derive(Debug, Clone)]
pub struct RefineConfig {
    /// 最大细化级别
    pub max_level: usize,
    /// 最小边长
    pub min_edge_length: f64,
    /// 最大边长
    pub max_edge_length: f64,
    /// 目标面积
    pub target_area: Option<f64>,
    /// 是否保持特征边
    pub preserve_features: bool,
    /// 细化方法
    pub method: RefineMethod,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self {
            max_level: 5,
            min_edge_length: 0.01,
            max_edge_length: 1.0,
            target_area: None,
            preserve_features: true,
            method: RefineMethod::MidpointSubdivision,
        }
    }
}

impl RefineConfig {
    /// 设置边长约束
    pub fn with_edge_length(mut self, min: f64, max: f64) -> Self {
        self.min_edge_length = min;
        self.max_edge_length = max;
        self
    }

    /// 设置目标面积
    pub fn with_target_area(mut self, area: f64) -> Self {
        self.target_area = Some(area);
        self
    }
}

/// 细化方法
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefineMethod {
    /// 中点细分
    MidpointSubdivision,
    /// 重心细分
    BarycentricSubdivision,
    /// 红绿细分
    RedGreenSubdivision,
}

/// 网格细化器
pub struct Refiner {
    config: RefineConfig,
}

impl Refiner {
    /// 创建细化器
    pub fn new(config: RefineConfig) -> Self {
        Self { config }
    }

    /// 使用默认配置创建
    pub fn default_config() -> Self {
        Self::new(RefineConfig::default())
    }

    /// 全局细化三角形网格
    ///
    /// # 参数
    /// - `vertices`: 顶点坐标
    /// - `triangles`: 三角形顶点索引
    ///
    /// # 返回
    /// (新顶点, 新三角形)
    pub fn refine_uniform(
        &self,
        vertices: &[[f64; 2]],
        triangles: &[[usize; 3]],
    ) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
        match self.config.method {
            RefineMethod::MidpointSubdivision => {
                self.midpoint_subdivide(vertices, triangles)
            }
            RefineMethod::BarycentricSubdivision => {
                self.barycentric_subdivide(vertices, triangles)
            }
            RefineMethod::RedGreenSubdivision => {
                // 红绿细分需要更复杂的处理
                self.midpoint_subdivide(vertices, triangles)
            }
        }
    }

    /// 自适应细化
    ///
    /// # 参数
    /// - `vertices`: 顶点坐标
    /// - `triangles`: 三角形顶点索引
    /// - `criteria`: 每个三角形的细化判据 (true=需要细化)
    pub fn refine_adaptive(
        &self,
        vertices: &[[f64; 2]],
        triangles: &[[usize; 3]],
        criteria: &[bool],
    ) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
        let mut new_vertices = vertices.to_vec();
        let mut new_triangles = Vec::new();

        // 记录需要细分的边和对应的中点
        let mut edge_midpoints: HashMap<(usize, usize), usize> = HashMap::new();

        // 首先处理需要细化的三角形
        for (i, tri) in triangles.iter().enumerate() {
            if i < criteria.len() && criteria[i] {
                // 需要细化：添加边中点
                let mut midpoints = [0usize; 3];
                for j in 0..3 {
                    let a = tri[j];
                    let b = tri[(j + 1) % 3];
                    let edge = if a < b { (a, b) } else { (b, a) };

                    let mid_idx = *edge_midpoints.entry(edge).or_insert_with(|| {
                        let mid = [
                            (vertices[a][0] + vertices[b][0]) / 2.0,
                            (vertices[a][1] + vertices[b][1]) / 2.0,
                        ];
                        let idx = new_vertices.len();
                        new_vertices.push(mid);
                        idx
                    });

                    midpoints[j] = mid_idx;
                }

                // 分成4个小三角形
                new_triangles.push([tri[0], midpoints[0], midpoints[2]]);
                new_triangles.push([midpoints[0], tri[1], midpoints[1]]);
                new_triangles.push([midpoints[2], midpoints[1], tri[2]]);
                new_triangles.push([midpoints[0], midpoints[1], midpoints[2]]);
            } else {
                // 保持原三角形
                new_triangles.push(*tri);
            }
        }

        // 处理需要绿色细分的三角形（与红色三角形相邻但本身不需要细化）
        // 这里简化处理，实际的红绿细分更复杂
        let final_triangles = self.fix_hanging_nodes(&new_vertices, &new_triangles, &edge_midpoints);

        (new_vertices, final_triangles)
    }

    /// 中点细分
    fn midpoint_subdivide(
        &self,
        vertices: &[[f64; 2]],
        triangles: &[[usize; 3]],
    ) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
        let mut new_vertices = vertices.to_vec();
        let mut new_triangles = Vec::with_capacity(triangles.len() * 4);
        let mut edge_midpoints: HashMap<(usize, usize), usize> = HashMap::new();

        for tri in triangles {
            let mut midpoints = [0usize; 3];

            // 计算三条边的中点
            for j in 0..3 {
                let a = tri[j];
                let b = tri[(j + 1) % 3];
                let edge = if a < b { (a, b) } else { (b, a) };

                let mid_idx = *edge_midpoints.entry(edge).or_insert_with(|| {
                    let mid = [
                        (vertices[a][0] + vertices[b][0]) / 2.0,
                        (vertices[a][1] + vertices[b][1]) / 2.0,
                    ];
                    let idx = new_vertices.len();
                    new_vertices.push(mid);
                    idx
                });

                midpoints[j] = mid_idx;
            }

            // 分成4个小三角形
            // T0: v0, m0, m2
            // T1: m0, v1, m1
            // T2: m2, m1, v2
            // T3: m0, m1, m2 (中心)
            new_triangles.push([tri[0], midpoints[0], midpoints[2]]);
            new_triangles.push([midpoints[0], tri[1], midpoints[1]]);
            new_triangles.push([midpoints[2], midpoints[1], tri[2]]);
            new_triangles.push([midpoints[0], midpoints[1], midpoints[2]]);
        }

        (new_vertices, new_triangles)
    }

    /// 重心细分
    fn barycentric_subdivide(
        &self,
        vertices: &[[f64; 2]],
        triangles: &[[usize; 3]],
    ) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
        let mut new_vertices = vertices.to_vec();
        let mut new_triangles = Vec::with_capacity(triangles.len() * 3);

        for tri in triangles {
            // 计算重心
            let centroid = [
                (vertices[tri[0]][0] + vertices[tri[1]][0] + vertices[tri[2]][0]) / 3.0,
                (vertices[tri[0]][1] + vertices[tri[1]][1] + vertices[tri[2]][1]) / 3.0,
            ];

            let centroid_idx = new_vertices.len();
            new_vertices.push(centroid);

            // 分成3个小三角形
            new_triangles.push([tri[0], tri[1], centroid_idx]);
            new_triangles.push([tri[1], tri[2], centroid_idx]);
            new_triangles.push([tri[2], tri[0], centroid_idx]);
        }

        (new_vertices, new_triangles)
    }

    /// 修复悬挂节点（简化版本）
    fn fix_hanging_nodes(
        &self,
        vertices: &[[f64; 2]],
        triangles: &[[usize; 3]],
        edge_midpoints: &HashMap<(usize, usize), usize>,
    ) -> Vec<[usize; 3]> {
        // 简化处理：如果三角形的某条边有中点但三角形本身没有细化，
        // 需要进行绿色细分
        
        let mut result = Vec::new();

        for tri in triangles {
            let mut has_midpoint = [false; 3];
            let mut midpoints = [0usize; 3];

            for j in 0..3 {
                let a = tri[j];
                let b = tri[(j + 1) % 3];
                let edge = if a < b { (a, b) } else { (b, a) };

                if let Some(&mid) = edge_midpoints.get(&edge) {
                    // 检查中点是否在这个三角形内使用
                    if !tri.contains(&mid) {
                        has_midpoint[j] = true;
                        midpoints[j] = mid;
                    }
                }
            }

            let count = has_midpoint.iter().filter(|&&x| x).count();

            match count {
                0 => {
                    result.push(*tri);
                }
                1 => {
                    // 绿色细分：一条边有中点，分成2个三角形
                    let edge_idx = has_midpoint.iter().position(|&x| x).unwrap();
                    let mid = midpoints[edge_idx];
                    let v0 = tri[edge_idx];
                    let v1 = tri[(edge_idx + 1) % 3];
                    let v2 = tri[(edge_idx + 2) % 3];

                    result.push([v0, mid, v2]);
                    result.push([mid, v1, v2]);
                }
                2 => {
                    // 两条边有中点，分成3个三角形
                    let missing = has_midpoint.iter().position(|&x| !x).unwrap();
                    let mid0 = midpoints[(missing + 1) % 3];
                    let mid1 = midpoints[(missing + 2) % 3];
                    let v0 = tri[missing];
                    let v1 = tri[(missing + 1) % 3];
                    let v2 = tri[(missing + 2) % 3];

                    result.push([v0, v1, mid0]);
                    result.push([v0, mid0, mid1]);
                    result.push([mid0, v2, mid1]);
                }
                3 => {
                    // 三条边都有中点，已经是红色细分
                    result.push(*tri);
                }
                _ => unreachable!(),
            }
        }

        result
    }

    /// 基于边长的细化判据
    pub fn edge_length_criteria(
        &self,
        vertices: &[[f64; 2]],
        triangles: &[[usize; 3]],
    ) -> Vec<bool> {
        triangles
            .iter()
            .map(|tri| {
                for j in 0..3 {
                    let a = vertices[tri[j]];
                    let b = vertices[tri[(j + 1) % 3]];
                    let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
                    if len > self.config.max_edge_length {
                        return true;
                    }
                }
                false
            })
            .collect()
    }

    /// 基于面积的细化判据
    pub fn area_criteria(
        &self,
        vertices: &[[f64; 2]],
        triangles: &[[usize; 3]],
    ) -> Vec<bool> {
        let target = self.config.target_area.unwrap_or(f64::MAX);

        triangles
            .iter()
            .map(|tri| {
                let a = vertices[tri[0]];
                let b = vertices[tri[1]];
                let c = vertices[tri[2]];
                let area = ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs() / 2.0;
                area > target
            })
            .collect()
    }

    /// 多级细化
    pub fn refine_levels(
        &self,
        vertices: &[[f64; 2]],
        triangles: &[[usize; 3]],
        levels: usize,
    ) -> (Vec<[f64; 2]>, Vec<[usize; 3]>) {
        let mut current_vertices = vertices.to_vec();
        let mut current_triangles = triangles.to_vec();

        for _ in 0..levels.min(self.config.max_level) {
            let (new_v, new_t) = self.refine_uniform(&current_vertices, &current_triangles);
            current_vertices = new_v;
            current_triangles = new_t;
        }

        (current_vertices, current_triangles)
    }
}

/// 细化统计
#[derive(Debug, Clone)]
pub struct RefineStats {
    /// 原始顶点数
    pub original_vertices: usize,
    /// 原始三角形数
    pub original_triangles: usize,
    /// 新增顶点数
    pub added_vertices: usize,
    /// 新三角形数
    pub new_triangles: usize,
    /// 细化级别
    pub levels: usize,
}

impl RefineStats {
    /// 计算统计信息
    pub fn compute(
        original_vertices: usize,
        original_triangles: usize,
        new_vertices: usize,
        new_triangles: usize,
        levels: usize,
    ) -> Self {
        Self {
            original_vertices,
            original_triangles,
            added_vertices: new_vertices - original_vertices,
            new_triangles,
            levels,
        }
    }

    /// 细化因子
    pub fn refinement_factor(&self) -> f64 {
        self.new_triangles as f64 / self.original_triangles as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_midpoint_subdivide() {
        let vertices = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let triangles = vec![[0, 1, 2]];

        let refiner = Refiner::default_config();
        let (new_v, new_t) = refiner.refine_uniform(&vertices, &triangles);

        // 1个三角形变4个，3个顶点变6个
        assert_eq!(new_t.len(), 4);
        assert_eq!(new_v.len(), 6);
    }

    #[test]
    fn test_barycentric_subdivide() {
        let vertices = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let triangles = vec![[0, 1, 2]];

        let config = RefineConfig {
            method: RefineMethod::BarycentricSubdivision,
            ..Default::default()
        };
        let refiner = Refiner::new(config);
        let (new_v, new_t) = refiner.refine_uniform(&vertices, &triangles);

        // 1个三角形变3个，3个顶点变4个
        assert_eq!(new_t.len(), 3);
        assert_eq!(new_v.len(), 4);
    }

    #[test]
    fn test_multi_level_refine() {
        let vertices = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let triangles = vec![[0, 1, 2]];

        let refiner = Refiner::default_config();
        let (new_v, new_t) = refiner.refine_levels(&vertices, &triangles, 2);

        // 2次中点细分: 1 -> 4 -> 16
        assert_eq!(new_t.len(), 16);
    }

    #[test]
    fn test_adaptive_refine() {
        let vertices = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [1.5, 1.0],
        ];
        let triangles = vec![[0, 1, 2], [1, 3, 2]];
        let criteria = vec![true, false]; // 只细化第一个

        let refiner = Refiner::default_config();
        let (new_v, new_t) = refiner.refine_adaptive(&vertices, &triangles, &criteria);

        // 第一个三角形变4个，第二个保持1个
        // 但可能需要处理悬挂节点
        assert!(new_t.len() >= 5);
        assert!(new_v.len() > vertices.len());
    }

    #[test]
    fn test_edge_length_criteria() {
        let vertices = vec![[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]];
        let triangles = vec![[0, 1, 2]];

        let config = RefineConfig {
            max_edge_length: 1.5,
            ..Default::default()
        };
        let refiner = Refiner::new(config);
        let criteria = refiner.edge_length_criteria(&vertices, &triangles);

        // 底边长度为2，超过阈值
        assert!(criteria[0]);
    }

    #[test]
    fn test_refine_stats() {
        let stats = RefineStats::compute(3, 1, 15, 16, 2);

        assert_eq!(stats.added_vertices, 12);
        assert_eq!(stats.refinement_factor(), 16.0);
    }
}
