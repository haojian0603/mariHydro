// crates/mh_physics/src/numerics/linear_algebra/csr.rs

//! 压缩稀疏行（CSR）矩阵格式
//!
//! CSR 是最常用的稀疏矩阵存储格式之一，特别适合：
//! - 高效的矩阵-向量乘法 (SpMV)
//! - 行遍历操作
//! - 与有限体积法的自然配合
//!
//! # 格式说明
//!
//! CSR 使用三个数组存储：
//! - `row_ptr`: 行指针，长度 n_rows + 1，row_ptr[i] 是第 i 行第一个非零元的索引
//! - `col_idx`: 列索引，与非零元一一对应
//! - `values`: 非零元值
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::numerics::linear_algebra::csr::{CsrBuilder, CsrMatrix};
//!
//! // 使用构建器创建矩阵
//! let mut builder = CsrBuilder::new(3);
//! builder.set(0, 0, 4.0);
//! builder.set(0, 1, -1.0);
//! builder.set(1, 0, -1.0);
//! builder.set(1, 1, 4.0);
//! builder.set(1, 2, -1.0);
//! builder.set(2, 1, -1.0);
//! builder.set(2, 2, 4.0);
//!
//! let matrix = builder.build();
//!
//! // 矩阵-向量乘法
//! let x = vec![1.0, 2.0, 3.0];
//! let mut y = vec![0.0; 3];
//! matrix.mul_vec(&x, &mut y);
//! ```

use std::collections::BTreeMap;

/// CSR 矩阵的稀疏模式
///
/// 存储矩阵的结构信息（哪些位置有非零元），与值分离
#[derive(Debug, Clone)]
pub struct CsrPattern {
    /// 行数
    n_rows: usize,
    /// 列数
    n_cols: usize,
    /// 行指针
    row_ptr: Vec<usize>,
    /// 列索引
    col_idx: Vec<usize>,
}

impl CsrPattern {
    /// 获取行数
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// 获取列数
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// 获取非零元数量
    #[inline]
    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    /// 获取行指针切片
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// 获取列索引切片
    #[inline]
    pub fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }

    /// 获取第 row 行的非零元列索引
    #[inline]
    pub fn row_indices(&self, row: usize) -> &[usize] {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        &self.col_idx[start..end]
    }

    /// 获取第 row 行的非零元数量
    #[inline]
    pub fn row_nnz(&self, row: usize) -> usize {
        self.row_ptr[row + 1] - self.row_ptr[row]
    }

    /// 查找 (row, col) 对应的值索引
    pub fn find_index(&self, row: usize, col: usize) -> Option<usize> {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        let indices = &self.col_idx[start..end];

        // 由于列索引是有序的，可以使用二分查找
        match indices.binary_search(&col) {
            Ok(local_idx) => Some(start + local_idx),
            Err(_) => None,
        }
    }

    /// 检查 (row, col) 是否有非零元
    pub fn has_entry(&self, row: usize, col: usize) -> bool {
        self.find_index(row, col).is_some()
    }
}

/// CSR 格式稀疏矩阵
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// 稀疏模式
    pattern: CsrPattern,
    /// 非零元值
    values: Vec<f64>,
}

impl CsrMatrix {
    /// 从原始 CSR 数据创建矩阵
    ///
    /// # 参数
    ///
    /// - `n_rows`: 行数
    /// - `n_cols`: 列数
    /// - `row_ptr`: 行指针数组，长度 n_rows + 1
    /// - `col_idx`: 列索引数组
    /// - `values`: 非零元值数组
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let matrix = CsrMatrix::from_raw(
    ///     3, 3,
    ///     vec![0, 2, 4, 6],
    ///     vec![0, 1, 0, 1, 1, 2],
    ///     vec![4.0, -1.0, -1.0, 4.0, -1.0, 4.0],
    /// );
    /// ```
    pub fn from_raw(
        n_rows: usize,
        n_cols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        debug_assert_eq!(row_ptr.len(), n_rows + 1, "row_ptr 长度必须为 n_rows + 1");
        debug_assert_eq!(col_idx.len(), values.len(), "col_idx 和 values 长度必须相等");
        debug_assert_eq!(row_ptr[n_rows], col_idx.len(), "row_ptr 末尾必须等于 nnz");

        Self {
            pattern: CsrPattern {
                n_rows,
                n_cols,
                row_ptr,
                col_idx,
            },
            values,
        }
    }

    /// 创建单位矩阵
    pub fn identity(n: usize) -> Self {
        let mut builder = CsrBuilder::new_square(n);
        for i in 0..n {
            builder.set(i, i, 1.0);
        }
        builder.build()
    }

    /// 创建对角矩阵
    pub fn diagonal(diag: &[f64]) -> Self {
        let n = diag.len();
        let mut builder = CsrBuilder::new_square(n);
        for (i, &v) in diag.iter().enumerate() {
            builder.set(i, i, v);
        }
        builder.build()
    }

    /// 获取行数
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.pattern.n_rows
    }

    /// 获取列数
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.pattern.n_cols
    }

    /// 获取非零元数量
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// 获取稀疏模式引用
    #[inline]
    pub fn pattern(&self) -> &CsrPattern {
        &self.pattern
    }

    /// 获取值切片
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// 获取可变值切片
    #[inline]
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// 获取行指针
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        &self.pattern.row_ptr
    }

    /// 获取列索引
    #[inline]
    pub fn col_idx(&self) -> &[usize] {
        &self.pattern.col_idx
    }

    /// 获取 (row, col) 位置的值
    pub fn get(&self, row: usize, col: usize) -> f64 {
        match self.pattern.find_index(row, col) {
            Some(idx) => self.values[idx],
            None => 0.0,
        }
    }

    /// 设置 (row, col) 位置的值（必须已存在该位置）
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> bool {
        match self.pattern.find_index(row, col) {
            Some(idx) => {
                self.values[idx] = value;
                true
            }
            None => false,
        }
    }

    /// 累加到 (row, col) 位置
    pub fn add(&mut self, row: usize, col: usize, value: f64) -> bool {
        match self.pattern.find_index(row, col) {
            Some(idx) => {
                self.values[idx] += value;
                true
            }
            None => false,
        }
    }

    /// 获取第 row 行的非零元 (列索引, 值) 切片
    pub fn row(&self, row: usize) -> RowView<'_> {
        let start = self.pattern.row_ptr[row];
        let end = self.pattern.row_ptr[row + 1];
        RowView {
            col_idx: &self.pattern.col_idx[start..end],
            values: &self.values[start..end],
        }
    }

    /// 获取对角元素值
    pub fn diagonal_value(&self, row: usize) -> Option<f64> {
        self.pattern.find_index(row, row).map(|idx| self.values[idx])
    }

    /// 获取对角元素向量（提取对角线）
    pub fn extract_diagonal(&self) -> Vec<f64> {
        (0..self.n_rows())
            .map(|i| self.diagonal_value(i).unwrap_or(0.0))
            .collect()
    }

    /// 矩阵-向量乘法 y = A*x
    pub fn mul_vec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n_cols());
        assert_eq!(y.len(), self.n_rows());

        for row in 0..self.n_rows() {
            let start = self.pattern.row_ptr[row];
            let end = self.pattern.row_ptr[row + 1];

            let mut sum = 0.0;
            for idx in start..end {
                let col = self.pattern.col_idx[idx];
                sum += self.values[idx] * x[col];
            }
            y[row] = sum;
        }
    }

    /// 矩阵-向量乘法加法 y = y + alpha * A * x
    pub fn mul_vec_add(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n_cols());
        assert_eq!(y.len(), self.n_rows());

        for row in 0..self.n_rows() {
            let start = self.pattern.row_ptr[row];
            let end = self.pattern.row_ptr[row + 1];

            let mut sum = 0.0;
            for idx in start..end {
                let col = self.pattern.col_idx[idx];
                sum += self.values[idx] * x[col];
            }
            y[row] += alpha * sum;
        }
    }

    /// 将所有值清零（保持模式）
    pub fn clear_values(&mut self) {
        self.values.fill(0.0);
    }

    /// 缩放所有值
    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.values {
            *v *= factor;
        }
    }

    /// 获取 Frobenius 范数
    pub fn frobenius_norm(&self) -> f64 {
        self.values.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    /// 高精度矩阵-向量乘法（Kahan 累加 + 4x 循环展开）
    ///
    /// 使用 Kahan 求和算法减少浮点累加误差，适用于需要高精度的场景。
    /// 4x 循环展开提高 CPU 流水线利用率。
    pub fn mul_vec_kahan(&self, x: &[f64], y: &mut [f64]) {
        use mh_foundation::KahanSum;

        assert_eq!(x.len(), self.n_cols());
        assert_eq!(y.len(), self.n_rows());

        for row in 0..self.n_rows() {
            let start = self.pattern.row_ptr[row];
            let end = self.pattern.row_ptr[row + 1];

            let mut sum = KahanSum::new();
            let mut i = start;

            // 4x 循环展开主循环
            while i + 3 < end {
                sum.add(self.values[i] * x[self.pattern.col_idx[i]]);
                sum.add(self.values[i + 1] * x[self.pattern.col_idx[i + 1]]);
                sum.add(self.values[i + 2] * x[self.pattern.col_idx[i + 2]]);
                sum.add(self.values[i + 3] * x[self.pattern.col_idx[i + 3]]);
                i += 4;
            }

            // 尾部处理
            while i < end {
                sum.add(self.values[i] * x[self.pattern.col_idx[i]]);
                i += 1;
            }

            y[row] = sum.value();
        }
    }

    /// 构建对角元素索引缓存
    ///
    /// 返回一个向量，其中第 i 个元素是第 i 行对角元素在 values 数组中的索引。
    /// 如果该行没有对角元素，则为 None。
    pub fn build_diagonal_cache(&self) -> Vec<Option<usize>> {
        let n = self.n_rows();
        let mut diag_indices = vec![None; n];

        for row in 0..n {
            let start = self.pattern.row_ptr[row];
            let end = self.pattern.row_ptr[row + 1];
            
            // 使用二分查找（列索引是有序的）
            let col_slice = &self.pattern.col_idx[start..end];
            if let Ok(local_idx) = col_slice.binary_search(&row) {
                diag_indices[row] = Some(start + local_idx);
            }
        }

        diag_indices
    }

    /// 使用缓存快速获取对角元素值
    #[inline]
    pub fn diagonal_value_cached(&self, row: usize, cache: &[Option<usize>]) -> Option<f64> {
        cache.get(row)?.map(|idx| self.values[idx])
    }

    /// 检查矩阵是否对称
    ///
    /// 验证所有非零元素 A[i,j] == A[j,i]。
    pub fn is_symmetric(&self, tol: f64) -> bool {
        for i in 0..self.n_rows() {
            let start = self.pattern.row_ptr[i];
            let end = self.pattern.row_ptr[i + 1];

            for idx in start..end {
                let j = self.pattern.col_idx[idx];
                if j > i {
                    let a_ij = self.values[idx];
                    let a_ji = self.get(j, i);
                    if (a_ij - a_ji).abs() > tol {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// 获取矩阵的无穷范数（行最大绝对值和）
    pub fn infinity_norm(&self) -> f64 {
        let mut max_row_sum: f64 = 0.0;
        for row in 0..self.n_rows() {
            let start = self.pattern.row_ptr[row];
            let end = self.pattern.row_ptr[row + 1];
            let row_sum: f64 = self.values[start..end].iter().map(|v| v.abs()).sum();
            max_row_sum = max_row_sum.max(row_sum);
        }
        max_row_sum
    }
}

/// 行视图
pub struct RowView<'a> {
    col_idx: &'a [usize],
    values: &'a [f64],
}

impl<'a> RowView<'a> {
    /// 获取列索引
    pub fn col_indices(&self) -> &'a [usize] {
        self.col_idx
    }

    /// 获取值
    pub fn values(&self) -> &'a [f64] {
        self.values
    }

    /// 获取非零元数量
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// 迭代 (列索引, 值) 对
    pub fn iter(&self) -> impl Iterator<Item = (usize, f64)> + 'a {
        self.col_idx.iter().copied().zip(self.values.iter().copied())
    }
}

/// CSR 矩阵构建器
///
/// 使用 BTreeMap 临时存储，构建时转换为紧凑 CSR 格式
pub struct CsrBuilder {
    n_rows: usize,
    n_cols: usize,
    /// 每行的 (列索引, 值) 映射
    rows: Vec<BTreeMap<usize, f64>>,
}

impl CsrBuilder {
    /// 创建方阵构建器
    pub fn new_square(n: usize) -> Self {
        Self::new(n, n)
    }

    /// 创建构建器
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            n_rows,
            n_cols,
            rows: vec![BTreeMap::new(); n_rows],
        }
    }

    /// 设置 (row, col) 的值
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        assert!(row < self.n_rows, "row index out of bounds");
        assert!(col < self.n_cols, "column index out of bounds");
        self.rows[row].insert(col, value);
    }

    /// 累加到 (row, col)
    pub fn add(&mut self, row: usize, col: usize, value: f64) {
        assert!(row < self.n_rows, "row index out of bounds");
        assert!(col < self.n_cols, "column index out of bounds");
        *self.rows[row].entry(col).or_insert(0.0) += value;
    }

    /// 获取 (row, col) 的值
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.rows[row].get(&col).copied().unwrap_or(0.0)
    }

    /// 清空构建器
    pub fn clear(&mut self) {
        for row in &mut self.rows {
            row.clear();
        }
    }

    /// 获取非零元数量
    pub fn nnz(&self) -> usize {
        self.rows.iter().map(|r| r.len()).sum()
    }

    /// 构建 CSR 矩阵
    pub fn build(self) -> CsrMatrix {
        let nnz = self.nnz();
        let mut row_ptr = Vec::with_capacity(self.n_rows + 1);
        let mut col_idx = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        row_ptr.push(0);

        for row_map in &self.rows {
            for (&col, &val) in row_map {
                col_idx.push(col);
                values.push(val);
            }
            row_ptr.push(col_idx.len());
        }

        CsrMatrix {
            pattern: CsrPattern {
                n_rows: self.n_rows,
                n_cols: self.n_cols,
                row_ptr,
                col_idx,
            },
            values,
        }
    }

    /// 构建并返回稀疏模式（用于模式复用）
    pub fn build_pattern(&self) -> CsrPattern {
        let nnz = self.nnz();
        let mut row_ptr = Vec::with_capacity(self.n_rows + 1);
        let mut col_idx = Vec::with_capacity(nnz);

        row_ptr.push(0);

        for row_map in &self.rows {
            for &col in row_map.keys() {
                col_idx.push(col);
            }
            row_ptr.push(col_idx.len());
        }

        CsrPattern {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            row_ptr,
            col_idx,
        }
    }
}

/// 从模式创建矩阵（值初始化为零）
impl From<CsrPattern> for CsrMatrix {
    fn from(pattern: CsrPattern) -> Self {
        let nnz = pattern.nnz();
        Self {
            pattern,
            values: vec![0.0; nnz],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let mat = CsrMatrix::identity(3);
        assert_eq!(mat.n_rows(), 3);
        assert_eq!(mat.n_cols(), 3);
        assert_eq!(mat.nnz(), 3);

        assert!((mat.get(0, 0) - 1.0).abs() < 1e-14);
        assert!((mat.get(1, 1) - 1.0).abs() < 1e-14);
        assert!((mat.get(2, 2) - 1.0).abs() < 1e-14);
        assert!(mat.get(0, 1).abs() < 1e-14);
    }

    #[test]
    fn test_diagonal() {
        let diag = vec![2.0, 3.0, 4.0];
        let mat = CsrMatrix::diagonal(&diag);

        assert!((mat.get(0, 0) - 2.0).abs() < 1e-14);
        assert!((mat.get(1, 1) - 3.0).abs() < 1e-14);
        assert!((mat.get(2, 2) - 4.0).abs() < 1e-14);

        assert_eq!(mat.extract_diagonal(), diag);
    }

    #[test]
    fn test_builder() {
        let mut builder = CsrBuilder::new_square(3);
        builder.set(0, 0, 4.0);
        builder.set(0, 1, -1.0);
        builder.set(1, 0, -1.0);
        builder.set(1, 1, 4.0);
        builder.set(1, 2, -1.0);
        builder.set(2, 1, -1.0);
        builder.set(2, 2, 4.0);

        let mat = builder.build();

        assert_eq!(mat.nnz(), 7);
        assert!((mat.get(0, 0) - 4.0).abs() < 1e-14);
        assert!((mat.get(0, 1) - (-1.0)).abs() < 1e-14);
        assert!((mat.get(1, 0) - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_mul_vec() {
        // 三对角矩阵
        let mut builder = CsrBuilder::new_square(3);
        builder.set(0, 0, 2.0);
        builder.set(0, 1, -1.0);
        builder.set(1, 0, -1.0);
        builder.set(1, 1, 2.0);
        builder.set(1, 2, -1.0);
        builder.set(2, 1, -1.0);
        builder.set(2, 2, 2.0);

        let mat = builder.build();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];

        mat.mul_vec(&x, &mut y);

        // y[0] = 2*1 - 1*2 = 0
        // y[1] = -1*1 + 2*2 - 1*3 = 0
        // y[2] = -1*2 + 2*3 = 4
        assert!((y[0] - 0.0).abs() < 1e-14);
        assert!((y[1] - 0.0).abs() < 1e-14);
        assert!((y[2] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_add_values() {
        let mut builder = CsrBuilder::new_square(2);
        builder.add(0, 0, 1.0);
        builder.add(0, 0, 2.0);
        builder.add(0, 1, 3.0);
        builder.add(1, 1, 4.0);

        let mat = builder.build();
        assert!((mat.get(0, 0) - 3.0).abs() < 1e-14);
        assert!((mat.get(0, 1) - 3.0).abs() < 1e-14);
        assert!((mat.get(1, 1) - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_row_view() {
        let mut builder = CsrBuilder::new_square(3);
        builder.set(1, 0, 1.0);
        builder.set(1, 1, 2.0);
        builder.set(1, 2, 3.0);

        let mat = builder.build();
        let row = mat.row(1);

        assert_eq!(row.nnz(), 3);
        let entries: Vec<_> = row.iter().collect();
        assert_eq!(entries, vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
    }

    #[test]
    fn test_pattern_reuse() {
        let mut builder = CsrBuilder::new_square(2);
        builder.set(0, 0, 1.0);
        builder.set(0, 1, 2.0);
        builder.set(1, 1, 3.0);

        let pattern = builder.build_pattern();
        let mat: CsrMatrix = pattern.into();

        assert_eq!(mat.nnz(), 3);
        assert!(mat.get(0, 0).abs() < 1e-14); // 值为 0
        assert!(mat.get(0, 1).abs() < 1e-14);
    }

    #[test]
    fn test_frobenius_norm() {
        let mut builder = CsrBuilder::new_square(2);
        builder.set(0, 0, 3.0);
        builder.set(1, 1, 4.0);

        let mat = builder.build();
        // ||A||_F = sqrt(3² + 4²) = 5
        assert!((mat.frobenius_norm() - 5.0).abs() < 1e-14);
    }
}
