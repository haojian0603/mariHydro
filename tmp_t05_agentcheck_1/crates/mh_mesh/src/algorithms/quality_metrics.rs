//! 网格质量指标模块
//!
//! 提供网格质量评估功能，包括：
//! - 单元质量指标（长宽比、偏斜度、正交性等）
//! - 整体网格质量统计
//! - 质量直方图和分布

use std::f64::consts::PI;

/// 质量指标类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityMetric {
    /// 长宽比 (aspect ratio)
    AspectRatio,
    /// 偏斜度 (skewness)
    Skewness,
    /// 正交性 (orthogonality)
    Orthogonality,
    /// 平滑度 (smoothness)
    Smoothness,
    /// 最小角度
    MinAngle,
    /// 最大角度
    MaxAngle,
    /// 面积比
    AreaRatio,
    /// 形状因子
    ShapeFactor,
}

/// 单个单元的质量数据
#[derive(Debug, Clone)]
pub struct CellQuality {
    /// 单元索引
    pub cell_index: usize,
    /// 长宽比
    pub aspect_ratio: f64,
    /// 偏斜度
    pub skewness: f64,
    /// 正交性
    pub orthogonality: f64,
    /// 最小角度（弧度）
    pub min_angle: f64,
    /// 最大角度（弧度）
    pub max_angle: f64,
    /// 面积
    pub area: f64,
}

impl CellQuality {
    /// 计算综合质量得分 (0-1，1为最佳)
    pub fn overall_score(&self) -> f64 {
        let ar_score = 1.0 / self.aspect_ratio.max(1.0);
        let skew_score = 1.0 - self.skewness;
        let ortho_score = self.orthogonality;
        let angle_score = self.min_angle / (PI / 3.0); // 60度为理想

        (ar_score + skew_score + ortho_score + angle_score) / 4.0
    }

    /// 是否为低质量单元
    pub fn is_low_quality(&self, threshold: f64) -> bool {
        self.overall_score() < threshold
    }
}

/// 网格整体质量统计
#[derive(Debug, Clone)]
pub struct MeshQualityStats {
    /// 单元数量
    pub cell_count: usize,
    /// 质量分布
    pub distribution: QualityDistribution,
    /// 按指标的统计
    pub metric_stats: std::collections::HashMap<QualityMetric, MetricStats>,
    /// 低质量单元索引
    pub low_quality_cells: Vec<usize>,
}

/// 质量分布
#[derive(Debug, Clone)]
pub struct QualityDistribution {
    /// 直方图（10个区间）
    pub histogram: [usize; 10],
    /// 各区间边界
    pub bin_edges: [f64; 11],
}

impl QualityDistribution {
    /// 创建新的分布
    pub fn new() -> Self {
        Self {
            histogram: [0; 10],
            bin_edges: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    }

    /// 添加一个质量值
    pub fn add(&mut self, quality: f64) {
        let bin = ((quality * 10.0).floor() as usize).min(9);
        self.histogram[bin] += 1;
    }

    /// 获取中位数所在的区间
    pub fn median_bin(&self) -> usize {
        let total: usize = self.histogram.iter().sum();
        let mut cumulative = 0;
        for (i, count) in self.histogram.iter().enumerate() {
            cumulative += count;
            if cumulative > total / 2 {
                return i;
            }
        }
        9
    }
}

impl Default for QualityDistribution {
    fn default() -> Self {
        Self::new()
    }
}

/// 单个指标的统计
#[derive(Debug, Clone)]
pub struct MetricStats {
    /// 最小值
    pub min: f64,
    /// 最大值
    pub max: f64,
    /// 平均值
    pub mean: f64,
    /// 标准差
    pub std_dev: f64,
}

impl MetricStats {
    /// 从值列表计算统计
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
            };
        }

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() 
            / values.len() as f64;
        let std_dev = variance.sqrt();

        Self { min, max, mean, std_dev }
    }
}

/// 质量计算配置
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// 低质量阈值
    pub low_quality_threshold: f64,
    /// 是否计算直方图
    pub compute_histogram: bool,
    /// 是否收集低质量单元列表
    pub collect_low_quality: bool,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            low_quality_threshold: 0.3,
            compute_histogram: true,
            collect_low_quality: true,
        }
    }
}

/// 网格质量评估器
#[derive(Debug, Clone)]
pub struct QualityEvaluator {
    config: QualityConfig,
}

impl QualityEvaluator {
    /// 创建评估器
    pub fn new(config: QualityConfig) -> Self {
        Self { config }
    }

    /// 使用默认配置创建
    pub fn with_default() -> Self {
        Self::new(QualityConfig::default())
    }

    /// 计算三角形的质量
    pub fn evaluate_triangle(
        &self,
        p0: &[f64; 3],
        p1: &[f64; 3],
        p2: &[f64; 3],
        cell_index: usize,
    ) -> CellQuality {
        // 计算边长
        let e0 = distance(p0, p1);
        let e1 = distance(p1, p2);
        let e2 = distance(p2, p0);

        let edges = [e0, e1, e2];
        let min_edge = edges.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_edge = edges.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // 长宽比
        let aspect_ratio = if min_edge > 0.0 { max_edge / min_edge } else { f64::INFINITY };

        // 计算角度
        let angles = [
            angle_at_vertex(p1, p0, p2),
            angle_at_vertex(p0, p1, p2),
            angle_at_vertex(p0, p2, p1),
        ];
        let min_angle = angles.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_angle = angles.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // 偏斜度：基于等边三角形的偏离
        // 理想三角形所有角度都是 60 度
        let ideal_angle = PI / 3.0;
        let angle_deviation = angles.iter()
            .map(|a| (a - ideal_angle).abs())
            .sum::<f64>() / 3.0;
        let skewness = (angle_deviation / ideal_angle).min(1.0);

        // 正交性：对于三角形，使用角度质量
        let orthogonality = 1.0 - skewness;

        // 面积
        let area = triangle_area(p0, p1, p2);

        CellQuality {
            cell_index,
            aspect_ratio,
            skewness,
            orthogonality,
            min_angle,
            max_angle,
            area,
        }
    }

    /// 计算四边形的质量
    pub fn evaluate_quad(
        &self,
        p0: &[f64; 3],
        p1: &[f64; 3],
        p2: &[f64; 3],
        p3: &[f64; 3],
        cell_index: usize,
    ) -> CellQuality {
        // 计算边长
        let e0 = distance(p0, p1);
        let e1 = distance(p1, p2);
        let e2 = distance(p2, p3);
        let e3 = distance(p3, p0);

        let edges = [e0, e1, e2, e3];
        let min_edge = edges.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_edge = edges.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let aspect_ratio = if min_edge > 0.0 { max_edge / min_edge } else { f64::INFINITY };

        // 计算角度
        let angles = [
            angle_at_vertex(p3, p0, p1),
            angle_at_vertex(p0, p1, p2),
            angle_at_vertex(p1, p2, p3),
            angle_at_vertex(p2, p3, p0),
        ];
        let min_angle = angles.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_angle = angles.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // 偏斜度：基于正方形的偏离
        let ideal_angle = PI / 2.0;
        let angle_deviation = angles.iter()
            .map(|a| (a - ideal_angle).abs())
            .sum::<f64>() / 4.0;
        let skewness = (angle_deviation / ideal_angle).min(1.0);

        // 正交性
        let orthogonality = 1.0 - skewness;

        // 面积（两个三角形之和）
        let area = triangle_area(p0, p1, p2) + triangle_area(p0, p2, p3);

        CellQuality {
            cell_index,
            aspect_ratio,
            skewness,
            orthogonality,
            min_angle,
            max_angle,
            area,
        }
    }

    /// 评估整个网格
    pub fn evaluate_mesh(
        &self,
        positions: &[[f64; 3]],
        faces: &[Vec<usize>],
    ) -> MeshQualityStats {
        let mut cell_qualities = Vec::with_capacity(faces.len());
        let mut distribution = QualityDistribution::new();
        let mut low_quality_cells = Vec::new();

        let mut aspect_ratios = Vec::with_capacity(faces.len());
        let mut skewness_values = Vec::with_capacity(faces.len());
        let mut orthogonality_values = Vec::with_capacity(faces.len());
        let mut min_angles = Vec::with_capacity(faces.len());
        let mut max_angles = Vec::with_capacity(faces.len());

        for (idx, face) in faces.iter().enumerate() {
            let quality = match face.len() {
                3 => self.evaluate_triangle(
                    &positions[face[0]],
                    &positions[face[1]],
                    &positions[face[2]],
                    idx,
                ),
                4 => self.evaluate_quad(
                    &positions[face[0]],
                    &positions[face[1]],
                    &positions[face[2]],
                    &positions[face[3]],
                    idx,
                ),
                _ => continue, // 跳过其他多边形
            };

            aspect_ratios.push(quality.aspect_ratio);
            skewness_values.push(quality.skewness);
            orthogonality_values.push(quality.orthogonality);
            min_angles.push(quality.min_angle);
            max_angles.push(quality.max_angle);

            let score = quality.overall_score();
            
            if self.config.compute_histogram {
                distribution.add(score);
            }

            if self.config.collect_low_quality && 
               quality.is_low_quality(self.config.low_quality_threshold) {
                low_quality_cells.push(idx);
            }

            cell_qualities.push(quality);
        }

        let mut metric_stats = std::collections::HashMap::new();
        metric_stats.insert(QualityMetric::AspectRatio, MetricStats::from_values(&aspect_ratios));
        metric_stats.insert(QualityMetric::Skewness, MetricStats::from_values(&skewness_values));
        metric_stats.insert(QualityMetric::Orthogonality, MetricStats::from_values(&orthogonality_values));
        metric_stats.insert(QualityMetric::MinAngle, MetricStats::from_values(&min_angles));
        metric_stats.insert(QualityMetric::MaxAngle, MetricStats::from_values(&max_angles));

        MeshQualityStats {
            cell_count: faces.len(),
            distribution,
            metric_stats,
            low_quality_cells,
        }
    }

    /// 评估单个面的质量分数
    pub fn evaluate_face_score(&self, positions: &[[f64; 3]], face: &[usize]) -> f64 {
        match face.len() {
            3 => self.evaluate_triangle(
                &positions[face[0]],
                &positions[face[1]],
                &positions[face[2]],
                0,
            ).overall_score(),
            4 => self.evaluate_quad(
                &positions[face[0]],
                &positions[face[1]],
                &positions[face[2]],
                &positions[face[3]],
                0,
            ).overall_score(),
            _ => 0.0,
        }
    }
}

/// 计算两点之间的距离
fn distance(p0: &[f64; 3], p1: &[f64; 3]) -> f64 {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let dz = p1[2] - p0[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// 计算顶点处的角度
fn angle_at_vertex(before: &[f64; 3], vertex: &[f64; 3], after: &[f64; 3]) -> f64 {
    let v1 = [
        before[0] - vertex[0],
        before[1] - vertex[1],
        before[2] - vertex[2],
    ];
    let v2 = [
        after[0] - vertex[0],
        after[1] - vertex[1],
        after[2] - vertex[2],
    ];

    let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    let len1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
    let len2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();

    if len1 == 0.0 || len2 == 0.0 {
        return 0.0;
    }

    let cos_angle = (dot / (len1 * len2)).clamp(-1.0, 1.0);
    cos_angle.acos()
}

/// 计算三角形面积
fn triangle_area(p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let v1 = [
        p1[0] - p0[0],
        p1[1] - p0[1],
        p1[2] - p0[2],
    ];
    let v2 = [
        p2[0] - p0[0],
        p2[1] - p0[1],
        p2[2] - p0[2],
    ];

    // 叉积
    let cross = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];

    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

/// 快速计算三角形质量（只返回综合分数）
pub fn quick_triangle_quality(p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let evaluator = QualityEvaluator::with_default();
    evaluator.evaluate_triangle(p0, p1, p2, 0).overall_score()
}

/// 快速计算四边形质量
pub fn quick_quad_quality(
    p0: &[f64; 3], 
    p1: &[f64; 3], 
    p2: &[f64; 3], 
    p3: &[f64; 3]
) -> f64 {
    let evaluator = QualityEvaluator::with_default();
    evaluator.evaluate_quad(p0, p1, p2, p3, 0).overall_score()
}

/// 计算最小内接圆和最大外接圆的比值（用于三角形质量）
pub fn inscribed_circumscribed_ratio(p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let a = distance(p1, p2);
    let b = distance(p0, p2);
    let c = distance(p0, p1);
    
    let s = (a + b + c) / 2.0;
    let area = triangle_area(p0, p1, p2);
    
    if area == 0.0 || a == 0.0 || b == 0.0 || c == 0.0 {
        return 0.0;
    }
    
    let inradius = area / s;
    let circumradius = (a * b * c) / (4.0 * area);
    
    if circumradius == 0.0 {
        return 0.0;
    }
    
    // 理想等边三角形的比值为 0.5
    inradius / circumradius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance() {
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [1.0, 0.0, 0.0];
        assert!((distance(&p0, &p1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_area() {
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [0.0, 1.0, 0.0];
        assert!((triangle_area(&p0, &p1, &p2) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_equilateral_triangle_quality() {
        // 等边三角形
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [0.5, 0.866, 0.0];

        let evaluator = QualityEvaluator::with_default();
        let quality = evaluator.evaluate_triangle(&p0, &p1, &p2, 0);

        assert!(quality.aspect_ratio < 1.1);
        assert!(quality.skewness < 0.1);
        assert!(quality.orthogonality > 0.9);
    }

    #[test]
    fn test_degenerate_triangle_quality() {
        // 退化三角形（几乎共线）- 非常扁平的三角形
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [10.0, 0.0, 0.0];      // 长边 = 10
        let p2 = [5.0, 0.001, 0.0];     // 很小的高度

        let evaluator = QualityEvaluator::with_default();
        let quality = evaluator.evaluate_triangle(&p0, &p1, &p2, 0);

        // 退化三角形：长宽比 > 1, 偏斜度高，质量分数低
        // 注意：aspect_ratio 是 max_edge/min_edge，这里约为 10/5 = 2
        assert!(quality.aspect_ratio >= 1.0, "aspect_ratio should be >= 1 for any triangle, got {}", quality.aspect_ratio);
        assert!(quality.skewness > 0.5, "skewness should be > 0.5 for degenerate triangle, got {}", quality.skewness);
        assert!(quality.overall_score() < 0.6, "quality score should be low for degenerate triangle, got {}", quality.overall_score());
    }


    #[test]
    fn test_mesh_quality_stats() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [1.5, 0.866, 0.0],
        ];
        let faces = vec![
            vec![0, 1, 2],
            vec![1, 3, 2],
        ];

        let evaluator = QualityEvaluator::with_default();
        let stats = evaluator.evaluate_mesh(&positions, &faces);

        assert_eq!(stats.cell_count, 2);
        assert!(stats.metric_stats.contains_key(&QualityMetric::AspectRatio));
    }

    #[test]
    fn test_inscribed_circumscribed_ratio() {
        // 等边三角形的比值应该接近 0.5
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [0.5, 0.866, 0.0];

        let ratio = inscribed_circumscribed_ratio(&p0, &p1, &p2);
        assert!((ratio - 0.5).abs() < 0.01);
    }
}

