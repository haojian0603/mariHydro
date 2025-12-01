// src-tauri/src/marihydro/core/compute/gpu_coloring.rs

//! GPU着色数据结构
//!
//! 为避免通量累加时的数据竞争，将面按颜色分组，
//! 同一颜色的面可以安全地并行处理。

/// 无效单元标记（用于边界面的neighbor）
pub const GPU_INVALID_CELL: u32 = u32::MAX;

/// GPU着色信息
///
/// 使用CSR格式存储每个颜色对应的面索引
#[derive(Debug, Clone)]
pub struct GpuColoring {
    /// 颜色数量
    pub n_colors: u32,
    /// CSR偏移数组，长度为n_colors+1
    pub color_offset: Vec<u32>,
    /// CSR索引数组，存储面索引
    pub face_indices: Vec<u32>,
    /// 每个颜色的面数量（用于调度）
    pub color_sizes: Vec<u32>,
    /// 最大颜色大小（用于确定workgroup数）
    pub max_color_size: u32,
}

impl GpuColoring {
    /// 创建空的着色信息
    pub fn empty() -> Self {
        Self {
            n_colors: 0,
            color_offset: vec![0],
            face_indices: Vec::new(),
            color_sizes: Vec::new(),
            max_color_size: 0,
        }
    }

    /// 从面颜色数组构建
    ///
    /// # 参数
    /// - `face_colors`: 每个面的颜色编号
    /// - `n_colors`: 颜色总数
    pub fn from_face_colors(face_colors: &[u32], n_colors: u32) -> Self {
        let n_faces = face_colors.len();
        
        // 计算每个颜色的面数量
        let mut color_sizes = vec![0u32; n_colors as usize];
        for &color in face_colors {
            if (color as usize) < color_sizes.len() {
                color_sizes[color as usize] += 1;
            }
        }

        // 构建CSR偏移
        let mut color_offset = Vec::with_capacity(n_colors as usize + 1);
        color_offset.push(0);
        for &size in &color_sizes {
            let last = *color_offset.last().unwrap();
            color_offset.push(last + size);
        }

        // 填充面索引
        let mut face_indices = vec![0u32; n_faces];
        let mut current_pos = color_offset.clone();
        
        for (face_idx, &color) in face_colors.iter().enumerate() {
            if (color as usize) < current_pos.len() - 1 {
                let pos = current_pos[color as usize] as usize;
                face_indices[pos] = face_idx as u32;
                current_pos[color as usize] += 1;
            }
        }

        let max_color_size = *color_sizes.iter().max().unwrap_or(&0);

        Self {
            n_colors,
            color_offset,
            face_indices,
            color_sizes,
            max_color_size,
        }
    }

    /// 获取指定颜色的面索引范围
    #[inline]
    pub fn color_range(&self, color: u32) -> std::ops::Range<usize> {
        let start = self.color_offset[color as usize] as usize;
        let end = self.color_offset[color as usize + 1] as usize;
        start..end
    }

    /// 获取指定颜色的所有面
    #[inline]
    pub fn color_faces(&self, color: u32) -> &[u32] {
        let range = self.color_range(color);
        &self.face_indices[range]
    }

    /// 获取面总数
    pub fn total_faces(&self) -> usize {
        self.face_indices.len()
    }

    /// GPU内存估计（字节）
    pub fn gpu_memory_estimate(&self) -> usize {
        let u32_size = std::mem::size_of::<u32>();
        (self.color_offset.len() + self.face_indices.len() + self.color_sizes.len()) * u32_size
    }

    /// 验证着色有效性
    pub fn validate(&self) -> Result<(), String> {
        // 检查offset长度
        if self.color_offset.len() != self.n_colors as usize + 1 {
            return Err(format!(
                "color_offset长度错误: {} != {}",
                self.color_offset.len(),
                self.n_colors + 1
            ));
        }

        // 检查单调递增
        for i in 1..self.color_offset.len() {
            if self.color_offset[i] < self.color_offset[i - 1] {
                return Err(format!(
                    "color_offset非单调递增: [{}]={} < [{}]={}",
                    i,
                    self.color_offset[i],
                    i - 1,
                    self.color_offset[i - 1]
                ));
            }
        }

        // 检查总面数
        let expected_faces = *self.color_offset.last().unwrap() as usize;
        if self.face_indices.len() != expected_faces {
            return Err(format!(
                "face_indices长度错误: {} != {}",
                self.face_indices.len(),
                expected_faces
            ));
        }

        Ok(())
    }
}

/// 从网格着色结果转换
pub trait ToGpuColoring {
    /// 转换为GPU友好的着色格式
    fn to_gpu_coloring(&self) -> GpuColoring;
}

/// 简单的贪心着色算法
pub struct GreedyColoring;

impl GreedyColoring {
    /// 对面进行着色
    ///
    /// # 参数
    /// - `n_faces`: 面总数
    /// - `face_owner`: 每个面的owner单元
    /// - `face_neighbor`: 每个面的neighbor单元（边界面为GPU_INVALID_CELL）
    ///
    /// # 返回
    /// 每个面的颜色编号和颜色总数
    pub fn color_faces(
        n_faces: usize,
        face_owner: &[u32],
        face_neighbor: &[u32],
    ) -> (Vec<u32>, u32) {
        let mut face_colors = vec![u32::MAX; n_faces];
        let n_cells = face_owner.iter().chain(face_neighbor.iter())
            .filter(|&&c| c != GPU_INVALID_CELL)
            .max()
            .map(|&c| c as usize + 1)
            .unwrap_or(0);

        // 为每个单元维护已使用的颜色
        let mut cell_used_colors: Vec<Vec<u32>> = vec![Vec::new(); n_cells];
        let mut max_color = 0u32;

        for face_idx in 0..n_faces {
            let owner = face_owner[face_idx] as usize;
            let neighbor = face_neighbor[face_idx];

            // 收集冲突颜色
            let mut forbidden = cell_used_colors[owner].clone();
            if neighbor != GPU_INVALID_CELL {
                forbidden.extend(&cell_used_colors[neighbor as usize]);
            }
            forbidden.sort_unstable();
            forbidden.dedup();

            // 找最小可用颜色
            let mut color = 0u32;
            for &fc in &forbidden {
                if fc == color {
                    color += 1;
                } else if fc > color {
                    break;
                }
            }

            face_colors[face_idx] = color;
            max_color = max_color.max(color);

            // 更新单元使用的颜色
            cell_used_colors[owner].push(color);
            if neighbor != GPU_INVALID_CELL {
                cell_used_colors[neighbor as usize].push(color);
            }
        }

        (face_colors, max_color + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_coloring() {
        let coloring = GpuColoring::empty();
        assert_eq!(coloring.n_colors, 0);
        assert_eq!(coloring.total_faces(), 0);
    }

    #[test]
    fn test_from_face_colors() {
        // 3个面，2种颜色
        let face_colors = vec![0, 1, 0];
        let coloring = GpuColoring::from_face_colors(&face_colors, 2);

        assert_eq!(coloring.n_colors, 2);
        assert_eq!(coloring.color_sizes, vec![2, 1]);
        assert_eq!(coloring.max_color_size, 2);

        // 颜色0有面0和2
        let color0_faces = coloring.color_faces(0);
        assert_eq!(color0_faces.len(), 2);
        assert!(color0_faces.contains(&0));
        assert!(color0_faces.contains(&2));

        // 颜色1有面1
        let color1_faces = coloring.color_faces(1);
        assert_eq!(color1_faces, &[1]);
    }

    #[test]
    fn test_greedy_coloring() {
        // 简单的3个面共享2个单元
        //   Cell 0 -- Face 0 -- Cell 1
        //   Cell 0 -- Face 1 -- Cell 1  
        //   Cell 1 -- Face 2 (boundary)
        let face_owner = vec![0, 0, 1];
        let face_neighbor = vec![1, 1, GPU_INVALID_CELL];

        let (colors, n_colors) = GreedyColoring::color_faces(3, &face_owner, &face_neighbor);

        // 面0和面1共享相同的owner和neighbor，应该有不同颜色
        assert_ne!(colors[0], colors[1]);
        // 颜色数应该>=2
        assert!(n_colors >= 2);
    }

    #[test]
    fn test_validation() {
        let coloring = GpuColoring::from_face_colors(&[0, 1, 0, 1], 2);
        assert!(coloring.validate().is_ok());
    }
}
