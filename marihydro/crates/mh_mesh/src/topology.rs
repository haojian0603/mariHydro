//! 网格拓扑数据结构
//!
//! 提供 CSR (Compressed Sparse Row) 格式的连接性存储。
//!
//! # 设计说明
//!
//! CSR 格式是一种紧凑的稀疏矩阵存储格式，适用于网格拓扑：
//! - `offsets[i]` 和 `offsets[i+1]` 之间的元素是第 i 行的非零元素
//! - 内存紧凑，缓存友好
//! - 适合只读迭代，不适合动态修改
//!
//! # 示例
//!
//! ```
//! use mh_mesh::topology::CsrConnectivity;
//!
//! // 3 个单元，每个单元有不同数量的节点
//! // Cell 0: [0, 1, 2]
//! // Cell 1: [1, 2, 3, 4]
//! // Cell 2: [2, 3]
//! let offsets = vec![0, 3, 7, 9];
//! let indices = vec![0, 1, 2, 1, 2, 3, 4, 2, 3];
//! let csr = CsrConnectivity::<u32>::new(offsets, indices);
//!
//! assert_eq!(csr.row(0), &[0, 1, 2]);
//! assert_eq!(csr.row(1), &[1, 2, 3, 4]);
//! assert_eq!(csr.n_rows(), 3);
//! ```

use serde::{Deserialize, Serialize};

/// CSR (Compressed Sparse Row) 格式连接性
///
/// 通用的 CSR 存储结构，可用于：
/// - 单元-节点连接 (cell_nodes)
/// - 单元-面连接 (cell_faces)
/// - 面-节点连接 (face_nodes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrConnectivity<I: Copy> {
    /// 行偏移数组，长度 = n_rows + 1
    /// offsets[i]..offsets[i+1] 是第 i 行的索引范围
    pub offsets: Vec<u32>,
    /// 列索引数组，长度 = nnz (非零元素数)
    pub indices: Vec<I>,
}

impl<I: Copy + Default> Default for CsrConnectivity<I> {
    fn default() -> Self {
        Self {
            offsets: vec![0],
            indices: Vec::new(),
        }
    }
}

impl<I: Copy> CsrConnectivity<I> {
    /// 创建新的 CSR 连接性
    pub fn new(offsets: Vec<u32>, indices: Vec<I>) -> Self {
        debug_assert!(
            !offsets.is_empty(),
            "offsets must have at least one element"
        );
        debug_assert_eq!(
            offsets.last().copied().unwrap_or(0) as usize,
            indices.len(),
            "last offset must equal indices length"
        );
        Self { offsets, indices }
    }

    /// 创建空的 CSR 结构（0 行）
    pub fn empty() -> Self {
        Self {
            offsets: vec![0],
            indices: Vec::new(),
        }
    }

    /// 从行列表构建 CSR
    pub fn from_rows(rows: &[&[I]]) -> Self {
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        let mut indices = Vec::new();
        
        offsets.push(0);
        for row in rows {
            indices.extend_from_slice(row);
            offsets.push(indices.len() as u32);
        }
        
        Self { offsets, indices }
    }

    /// 获取第 row 行的切片
    #[inline]
    pub fn row(&self, row: usize) -> &[I] {
        let start = self.offsets[row] as usize;
        let end = self.offsets[row + 1] as usize;
        &self.indices[start..end]
    }

    /// 获取第 row 行的可变切片
    #[inline]
    pub fn row_mut(&mut self, row: usize) -> &mut [I] {
        let start = self.offsets[row] as usize;
        let end = self.offsets[row + 1] as usize;
        &mut self.indices[start..end]
    }

    /// 获取行数
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// 获取非零元素总数
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// 检查是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_rows() == 0
    }

    /// 第 row 行的元素个数
    #[inline]
    pub fn row_len(&self, row: usize) -> usize {
        (self.offsets[row + 1] - self.offsets[row]) as usize
    }

    /// 迭代所有行
    pub fn iter_rows(&self) -> impl Iterator<Item = &[I]> {
        (0..self.n_rows()).map(move |i| self.row(i))
    }

    /// 获取偏移数组引用
    #[inline]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// 获取索引数组引用
    #[inline]
    pub fn indices(&self) -> &[I] {
        &self.indices
    }
}

/// 面-单元连接性
///
/// 每个面连接两个单元（内部面）或一个单元（边界面）。
/// 使用哨兵值 `CellIdx::INVALID` 表示边界面的外侧。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FaceCellConnectivity {
    /// 面的所有者单元（法向指向的单元）
    pub owner: Vec<u32>,
    /// 面的邻居单元（内部面）或 INVALID（边界面）
    pub neighbor: Vec<u32>,
}

impl FaceCellConnectivity {
    /// 无效单元索引（边界面的邻居）
    pub const INVALID_CELL: u32 = u32::MAX;

    /// 创建新的面-单元连接性
    pub fn new(owner: Vec<u32>, neighbor: Vec<u32>) -> Self {
        debug_assert_eq!(owner.len(), neighbor.len());
        Self { owner, neighbor }
    }

    /// 创建空连接性
    pub fn empty() -> Self {
        Self {
            owner: Vec::new(),
            neighbor: Vec::new(),
        }
    }

    /// 预分配容量
    pub fn with_capacity(n_faces: usize) -> Self {
        Self {
            owner: Vec::with_capacity(n_faces),
            neighbor: Vec::with_capacity(n_faces),
        }
    }

    /// 添加一个面
    pub fn push(&mut self, owner: u32, neighbor: u32) {
        self.owner.push(owner);
        self.neighbor.push(neighbor);
    }

    /// 获取面数量
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.owner.len()
    }

    /// 判断面是否为边界面
    #[inline]
    pub fn is_boundary(&self, face: usize) -> bool {
        self.neighbor[face] == Self::INVALID_CELL
    }

    /// 获取面的两个相邻单元
    #[inline]
    pub fn cells(&self, face: usize) -> (u32, u32) {
        (self.owner[face], self.neighbor[face])
    }

    /// 获取面的所有者单元
    #[inline]
    pub fn owner(&self, face: usize) -> u32 {
        self.owner[face]
    }

    /// 获取面的邻居单元（可能是 INVALID_CELL）
    #[inline]
    pub fn neighbor(&self, face: usize) -> u32 {
        self.neighbor[face]
    }

    /// 获取面的有效邻居（如果是边界面则返回 None）
    #[inline]
    pub fn neighbor_option(&self, face: usize) -> Option<u32> {
        let n = self.neighbor[face];
        if n == Self::INVALID_CELL { None } else { Some(n) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_basic() {
        let offsets = vec![0, 3, 7, 9];
        let indices = vec![0u32, 1, 2, 1, 2, 3, 4, 2, 3];
        let csr = CsrConnectivity::new(offsets, indices);

        assert_eq!(csr.n_rows(), 3);
        assert_eq!(csr.nnz(), 9);
        assert_eq!(csr.row(0), &[0, 1, 2]);
        assert_eq!(csr.row(1), &[1, 2, 3, 4]);
        assert_eq!(csr.row(2), &[2, 3]);
        assert_eq!(csr.row_len(0), 3);
        assert_eq!(csr.row_len(1), 4);
    }

    #[test]
    fn test_csr_from_rows() {
        let rows: Vec<&[u32]> = vec![&[0, 1, 2], &[1, 2, 3, 4], &[2, 3]];
        let csr = CsrConnectivity::from_rows(&rows);

        assert_eq!(csr.n_rows(), 3);
        assert_eq!(csr.row(0), &[0, 1, 2]);
        assert_eq!(csr.row(1), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_csr_empty() {
        let csr: CsrConnectivity<u32> = CsrConnectivity::empty();
        assert_eq!(csr.n_rows(), 0);
        assert!(csr.is_empty());
    }

    #[test]
    fn test_csr_iter_rows() {
        let rows: Vec<&[u32]> = vec![&[0, 1], &[2, 3, 4]];
        let csr = CsrConnectivity::from_rows(&rows);

        let collected: Vec<&[u32]> = csr.iter_rows().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0], &[0, 1]);
        assert_eq!(collected[1], &[2, 3, 4]);
    }

    #[test]
    fn test_face_cell_connectivity() {
        let mut fcc = FaceCellConnectivity::with_capacity(3);
        fcc.push(0, 1);  // 内部面
        fcc.push(1, 2);  // 内部面
        fcc.push(2, FaceCellConnectivity::INVALID_CELL);  // 边界面

        assert_eq!(fcc.n_faces(), 3);
        assert!(!fcc.is_boundary(0));
        assert!(!fcc.is_boundary(1));
        assert!(fcc.is_boundary(2));

        assert_eq!(fcc.cells(0), (0, 1));
        assert_eq!(fcc.neighbor_option(0), Some(1));
        assert_eq!(fcc.neighbor_option(2), None);
    }
}
