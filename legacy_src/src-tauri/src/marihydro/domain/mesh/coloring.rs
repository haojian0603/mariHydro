// File: src-tauri/src/marihydro/domain/mesh/coloring.rs
//! 网格着色算法
//! 
//! 实现图着色算法，使同一颜色组内的面可以安全并行计算通量。
//! 着色保证：同一颜色组内任意两个面不共享单元。

use crate::marihydro::core::traits::mesh::MeshAccess;
use std::collections::HashSet;

/// 网格着色结果
#[derive(Clone, Debug)]
pub struct MeshColoring {
    /// 按颜色分组的面索引
    pub color_groups: Vec<Vec<usize>>,
    /// 每个面的颜色
    pub face_colors: Vec<u8>,
    /// 使用的颜色数
    pub n_colors: u8,
}

impl MeshColoring {
    /// 使用贪心算法构建着色
    /// 
    /// 算法：对每个面，选择最小可用颜色（该颜色未被相邻面使用）
    pub fn build<M: MeshAccess>(mesh: &M) -> Self {
        let n_faces = mesh.n_faces();
        let n_cells = mesh.n_cells();
        
        if n_faces == 0 {
            return Self {
                color_groups: Vec::new(),
                face_colors: Vec::new(),
                n_colors: 0,
            };
        }

        // 构建面-面邻接关系（通过共享单元）
        // cell_faces[cell_idx] = 该单元的所有面
        let mut cell_to_faces: Vec<Vec<usize>> = vec![Vec::new(); n_cells];
        
        for face_idx in 0..n_faces {
            let face = crate::marihydro::core::types::FaceIndex(face_idx);
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            
            cell_to_faces[owner.0].push(face_idx);
            if neighbor.is_valid() {
                cell_to_faces[neighbor.0].push(face_idx);
            }
        }
        
        // 贪心着色
        let mut face_colors: Vec<u8> = vec![255; n_faces]; // 255 表示未着色
        let mut n_colors: u8 = 0;
        
        for face_idx in 0..n_faces {
            let face = crate::marihydro::core::types::FaceIndex(face_idx);
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            
            // 收集相邻面的颜色
            let mut used_colors: HashSet<u8> = HashSet::new();
            
            // owner 单元的其他面
            for &adj_face in &cell_to_faces[owner.0] {
                if adj_face != face_idx && face_colors[adj_face] != 255 {
                    used_colors.insert(face_colors[adj_face]);
                }
            }
            
            // neighbor 单元的其他面（如果存在）
            if neighbor.is_valid() {
                for &adj_face in &cell_to_faces[neighbor.0] {
                    if adj_face != face_idx && face_colors[adj_face] != 255 {
                        used_colors.insert(face_colors[adj_face]);
                    }
                }
            }
            
            // 选择最小可用颜色
            let mut color: u8 = 0;
            while used_colors.contains(&color) {
                color += 1;
            }
            
            face_colors[face_idx] = color;
            if color >= n_colors {
                n_colors = color + 1;
            }
        }
        
        // 构建颜色分组
        let mut color_groups: Vec<Vec<usize>> = vec![Vec::new(); n_colors as usize];
        for (face_idx, &color) in face_colors.iter().enumerate() {
            color_groups[color as usize].push(face_idx);
        }
        
        Self {
            color_groups,
            face_colors,
            n_colors,
        }
    }
    
    /// 验证着色正确性
    /// 
    /// 检查同一颜色组内的面是否共享单元
    pub fn validate<M: MeshAccess>(&self, mesh: &M) -> bool {
        for group in &self.color_groups {
            // 收集该组所有面涉及的单元
            let mut cells_in_group: HashSet<usize> = HashSet::new();
            
            for &face_idx in group {
                let face = crate::marihydro::core::types::FaceIndex(face_idx);
                let owner = mesh.face_owner(face);
                let neighbor = mesh.face_neighbor(face);
                
                // 如果单元已被其他面占用，着色无效
                if cells_in_group.contains(&owner.0) {
                    return false;
                }
                cells_in_group.insert(owner.0);
                
                if neighbor.is_valid() {
                    if cells_in_group.contains(&neighbor.0) {
                        return false;
                    }
                    cells_in_group.insert(neighbor.0);
                }
            }
        }
        true
    }
    
    /// 获取统计信息
    pub fn stats(&self) -> ColoringStats {
        let sizes: Vec<usize> = self.color_groups.iter().map(|g| g.len()).collect();
        let total: usize = sizes.iter().sum();
        
        ColoringStats {
            n_colors: self.n_colors,
            min_group_size: sizes.iter().copied().min().unwrap_or(0),
            max_group_size: sizes.iter().copied().max().unwrap_or(0),
            avg_group_size: if self.n_colors > 0 {
                total as f64 / self.n_colors as f64
            } else {
                0.0
            },
        }
    }
    
    /// 获取指定颜色组的面索引
    #[inline]
    pub fn get_group(&self, color: u8) -> &[usize] {
        &self.color_groups[color as usize]
    }
    
    /// 迭代所有颜色组
    pub fn iter_groups(&self) -> impl Iterator<Item = (u8, &[usize])> {
        self.color_groups
            .iter()
            .enumerate()
            .map(|(i, g)| (i as u8, g.as_slice()))
    }
}

/// 着色统计信息
#[derive(Debug, Clone)]
pub struct ColoringStats {
    /// 使用的颜色数
    pub n_colors: u8,
    /// 最小组大小
    pub min_group_size: usize,
    /// 最大组大小
    pub max_group_size: usize,
    /// 平均组大小
    pub avg_group_size: f64,
}

impl std::fmt::Display for ColoringStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Coloring: {} colors, group sizes: {}-{} (avg: {:.1})",
            self.n_colors, self.min_group_size, self.max_group_size, self.avg_group_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 注：完整测试需要实际网格，这里只测试基本逻辑
    #[test]
    fn test_coloring_stats_display() {
        let stats = ColoringStats {
            n_colors: 4,
            min_group_size: 100,
            max_group_size: 150,
            avg_group_size: 125.5,
        };
        let s = format!("{}", stats);
        assert!(s.contains("4 colors"));
    }
    
    #[test]
    fn test_empty_mesh_coloring() {
        // 空着色
        let coloring = MeshColoring {
            color_groups: Vec::new(),
            face_colors: Vec::new(),
            n_colors: 0,
        };
        let stats = coloring.stats();
        assert_eq!(stats.n_colors, 0);
    }
}
