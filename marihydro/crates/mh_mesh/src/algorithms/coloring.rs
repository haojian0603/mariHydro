// marihydro\crates\mh_mesh\src/algorithms/coloring.rs

//! 图着色算法
//!
//! 用于并行计算中的单元分组，确保同一颜色的单元不共享节点。

use std::collections::HashSet;

/// 着色结果
#[derive(Debug, Clone)]
pub struct ColoringResult {
    /// 每个单元的颜色
    pub cell_colors: Vec<usize>,
    /// 颜色数量
    pub num_colors: usize,
    /// 每个颜色包含的单元数
    pub color_sizes: Vec<usize>,
    /// 每个颜色的单元索引
    pub color_cells: Vec<Vec<usize>>,
}

impl ColoringResult {
    /// 获取指定颜色的单元
    pub fn cells_with_color(&self, color: usize) -> &[usize] {
        if color < self.color_cells.len() {
            &self.color_cells[color]
        } else {
            &[]
        }
    }

    /// 获取单元的颜色
    pub fn color_of(&self, cell: usize) -> Option<usize> {
        self.cell_colors.get(cell).copied()
    }

    /// 检查是否为有效着色
    pub fn is_valid(&self, adjacency: &[(usize, usize)]) -> bool {
        for &(i, j) in adjacency {
            if i >= self.cell_colors.len() || j >= self.cell_colors.len() {
                continue;
            }
            if self.cell_colors[i] == self.cell_colors[j] {
                return false;
            }
        }
        true
    }

    /// 计算负载均衡度 (1.0 = 完美均衡)
    pub fn balance_factor(&self) -> f64 {
        if self.color_sizes.is_empty() {
            return 1.0;
        }
        
        let min_size = *self.color_sizes.iter().min().unwrap_or(&0);
        let max_size = *self.color_sizes.iter().max().unwrap_or(&1);
        
        if max_size == 0 {
            1.0
        } else {
            min_size as f64 / max_size as f64
        }
    }
}

/// 贪心着色器
pub struct GreedyColoring;

impl GreedyColoring {
    /// 对单元进行着色
    ///
    /// # 参数
    /// - `num_cells`: 单元数量
    /// - `adjacency`: 邻接关系 (cell_i, cell_j) 表示两个单元相邻
    ///
    /// # 返回
    /// 着色结果
    pub fn color(num_cells: usize, adjacency: &[(usize, usize)]) -> ColoringResult {
        if num_cells == 0 {
            return ColoringResult {
                cell_colors: Vec::new(),
                num_colors: 0,
                color_sizes: Vec::new(),
                color_cells: Vec::new(),
            };
        }

        // 构建邻接表
        let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); num_cells];
        for &(i, j) in adjacency {
            if i < num_cells && j < num_cells && i != j {
                neighbors[i].insert(j);
                neighbors[j].insert(i);
            }
        }

        // 按度数排序 (高度数优先，DSATUR启发式)
        let mut order: Vec<usize> = (0..num_cells).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(neighbors[i].len()));

        // 贪心着色
        let mut colors = vec![usize::MAX; num_cells];
        let mut num_colors = 0;

        for cell in order {
            // 找到邻居使用的颜色
            let used_colors: HashSet<usize> = neighbors[cell]
                .iter()
                .filter_map(|&n| {
                    if colors[n] != usize::MAX {
                        Some(colors[n])
                    } else {
                        None
                    }
                })
                .collect();

            // 找到最小可用颜色
            let mut color = 0;
            while used_colors.contains(&color) {
                color += 1;
            }

            colors[cell] = color;
            num_colors = num_colors.max(color + 1);
        }

        // 统计每个颜色的单元
        let mut color_cells: Vec<Vec<usize>> = vec![Vec::new(); num_colors];
        for (cell, &color) in colors.iter().enumerate() {
            if color != usize::MAX {
                color_cells[color].push(cell);
            }
        }

        let color_sizes: Vec<usize> = color_cells.iter().map(|c| c.len()).collect();

        ColoringResult {
            cell_colors: colors,
            num_colors,
            color_sizes,
            color_cells,
        }
    }

    /// 从网格拓扑生成着色
    ///
    /// # 参数
    /// - `num_cells`: 单元数量
    /// - `cell_neighbors`: 每个单元的邻居单元列表
    pub fn from_mesh_topology(
        num_cells: usize,
        cell_neighbors: &[Vec<usize>],
    ) -> ColoringResult {
        // 构建邻接对
        let mut adjacency = Vec::new();
        for (i, neighbors) in cell_neighbors.iter().enumerate() {
            for &j in neighbors {
                if i < j {
                    adjacency.push((i, j));
                }
            }
        }

        Self::color(num_cells, &adjacency)
    }

    /// 从共享节点生成着色
    ///
    /// 两个单元如果共享节点则视为邻居
    ///
    /// # 参数
    /// - `num_cells`: 单元数量
    /// - `cell_nodes`: 每个单元的节点列表
    pub fn from_shared_nodes(
        num_cells: usize,
        cell_nodes: &[Vec<usize>],
    ) -> ColoringResult {
        use std::collections::HashMap;

        // 构建节点到单元的映射
        let mut node_to_cells: HashMap<usize, Vec<usize>> = HashMap::new();
        for (cell, nodes) in cell_nodes.iter().enumerate() {
            for &node in nodes {
                node_to_cells.entry(node).or_default().push(cell);
            }
        }

        // 找出共享节点的单元对
        let mut adjacency_set: HashSet<(usize, usize)> = HashSet::new();
        for cells in node_to_cells.values() {
            for i in 0..cells.len() {
                for j in (i + 1)..cells.len() {
                    let a = cells[i].min(cells[j]);
                    let b = cells[i].max(cells[j]);
                    adjacency_set.insert((a, b));
                }
            }
        }

        let adjacency: Vec<_> = adjacency_set.into_iter().collect();
        Self::color(num_cells, &adjacency)
    }
}

/// DSATUR 着色器 (饱和度优先)
pub struct DsaturColoring;

impl DsaturColoring {
    /// 使用DSATUR算法着色
    ///
    /// DSATUR: 每次选择饱和度最高的未着色顶点
    /// 饱和度 = 邻居中已使用的不同颜色数量
    pub fn color(num_cells: usize, adjacency: &[(usize, usize)]) -> ColoringResult {
        if num_cells == 0 {
            return ColoringResult {
                cell_colors: Vec::new(),
                num_colors: 0,
                color_sizes: Vec::new(),
                color_cells: Vec::new(),
            };
        }

        // 构建邻接表
        let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); num_cells];
        for &(i, j) in adjacency {
            if i < num_cells && j < num_cells && i != j {
                neighbors[i].insert(j);
                neighbors[j].insert(i);
            }
        }

        let mut colors = vec![usize::MAX; num_cells];
        let mut saturation = vec![HashSet::<usize>::new(); num_cells];
        let mut num_colors = 0;

        for _ in 0..num_cells {
            // 选择饱和度最高的未着色顶点
            // 平局时选择度数最高的
            let next = (0..num_cells)
                .filter(|&i| colors[i] == usize::MAX)
                .max_by_key(|&i| (saturation[i].len(), neighbors[i].len()))
                .unwrap();

            // 找到最小可用颜色
            let mut color = 0;
            while saturation[next].contains(&color) {
                color += 1;
            }

            colors[next] = color;
            num_colors = num_colors.max(color + 1);

            // 更新邻居的饱和度
            for &neighbor in &neighbors[next] {
                saturation[neighbor].insert(color);
            }
        }

        // 统计
        let mut color_cells: Vec<Vec<usize>> = vec![Vec::new(); num_colors];
        for (cell, &color) in colors.iter().enumerate() {
            if color != usize::MAX {
                color_cells[color].push(cell);
            }
        }

        let color_sizes: Vec<usize> = color_cells.iter().map(|c| c.len()).collect();

        ColoringResult {
            cell_colors: colors,
            num_colors,
            color_sizes,
            color_cells,
        }
    }
}

/// 并行着色优化器
pub struct ColoringOptimizer;

impl ColoringOptimizer {
    /// 优化着色以提高负载均衡
    ///
    /// 尝试将单元从大颜色组移动到小颜色组
    pub fn optimize_balance(
        result: &mut ColoringResult,
        adjacency: &[(usize, usize)],
        max_iterations: usize,
    ) {
        let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); result.cell_colors.len()];
        for &(i, j) in adjacency {
            if i < neighbors.len() && j < neighbors.len() {
                neighbors[i].insert(j);
                neighbors[j].insert(i);
            }
        }

        for _ in 0..max_iterations {
            let mut improved = false;

            // 找到最大和最小颜色组
            let (max_color, _) = result.color_sizes
                .iter()
                .enumerate()
                .max_by_key(|&(_, s)| s)
                .unwrap();

            let (min_color, _) = result.color_sizes
                .iter()
                .enumerate()
                .min_by_key(|&(_, s)| s)
                .unwrap();

            // 尝试将最大组的某个单元移到最小组
            if let Some(&cell) = result.color_cells[max_color].last() {
                // 检查是否可以移动
                let neighbor_colors: HashSet<usize> = neighbors[cell]
                    .iter()
                    .filter_map(|&n| {
                        if result.cell_colors[n] != usize::MAX {
                            Some(result.cell_colors[n])
                        } else {
                            None
                        }
                    })
                    .collect();

                if !neighbor_colors.contains(&min_color) {
                    // 可以移动
                    result.cell_colors[cell] = min_color;
                    result.color_cells[max_color].pop();
                    result.color_cells[min_color].push(cell);
                    result.color_sizes[max_color] -= 1;
                    result.color_sizes[min_color] += 1;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_coloring() {
        let result = GreedyColoring::color(0, &[]);
        assert_eq!(result.num_colors, 0);
        assert!(result.cell_colors.is_empty());
    }

    #[test]
    fn test_single_cell() {
        let result = GreedyColoring::color(1, &[]);
        assert_eq!(result.num_colors, 1);
        assert_eq!(result.cell_colors[0], 0);
    }

    #[test]
    fn test_two_adjacent_cells() {
        let adjacency = vec![(0, 1)];
        let result = GreedyColoring::color(2, &adjacency);
        
        assert_eq!(result.num_colors, 2);
        assert_ne!(result.cell_colors[0], result.cell_colors[1]);
        assert!(result.is_valid(&adjacency));
    }

    #[test]
    fn test_triangle() {
        // 三角形需要3种颜色
        let adjacency = vec![(0, 1), (1, 2), (2, 0)];
        let result = GreedyColoring::color(3, &adjacency);
        
        assert_eq!(result.num_colors, 3);
        assert!(result.is_valid(&adjacency));
    }

    #[test]
    fn test_bipartite() {
        // 完全二部图 K2,2 只需要2种颜色
        let adjacency = vec![(0, 2), (0, 3), (1, 2), (1, 3)];
        let result = GreedyColoring::color(4, &adjacency);
        
        assert!(result.num_colors <= 2);
        assert!(result.is_valid(&adjacency));
    }

    #[test]
    fn test_from_shared_nodes() {
        // 两个共享一个节点的三角形
        let cell_nodes = vec![
            vec![0, 1, 2],
            vec![1, 2, 3],
        ];
        
        let result = GreedyColoring::from_shared_nodes(2, &cell_nodes);
        
        // 共享节点1和2，应该有不同颜色
        assert_eq!(result.num_colors, 2);
        assert_ne!(result.cell_colors[0], result.cell_colors[1]);
    }

    #[test]
    fn test_dsatur() {
        let adjacency = vec![(0, 1), (1, 2), (2, 0), (2, 3)];
        let result = DsaturColoring::color(4, &adjacency);
        
        assert!(result.is_valid(&adjacency));
    }

    #[test]
    fn test_balance_factor() {
        let result = ColoringResult {
            cell_colors: vec![0, 0, 1, 1],
            num_colors: 2,
            color_sizes: vec![2, 2],
            color_cells: vec![vec![0, 1], vec![2, 3]],
        };
        
        assert!((result.balance_factor() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mesh_coloring() {
        // 4x4 网格的单元邻接
        let cell_neighbors = vec![
            vec![1, 4],       // 0
            vec![0, 2, 5],    // 1
            vec![1, 3, 6],    // 2
            vec![2, 7],       // 3
            vec![0, 5, 8],    // 4
            vec![1, 4, 6, 9], // 5
            vec![2, 5, 7, 10], // 6
            vec![3, 6, 11],   // 7
            vec![4, 9, 12],   // 8
            vec![5, 8, 10, 13], // 9
            vec![6, 9, 11, 14], // 10
            vec![7, 10, 15],  // 11
            vec![8, 13],      // 12
            vec![9, 12, 14],  // 13
            vec![10, 13, 15], // 14
            vec![11, 14],     // 15
        ];
        
        let result = GreedyColoring::from_mesh_topology(16, &cell_neighbors);
        
        // 棋盘格只需要2种颜色
        assert!(result.num_colors <= 4); // 贪心可能不是最优
        
        // 验证正确性
        for (i, neighbors) in cell_neighbors.iter().enumerate() {
            for &j in neighbors {
                assert_ne!(result.cell_colors[i], result.cell_colors[j]);
            }
        }
    }
}
