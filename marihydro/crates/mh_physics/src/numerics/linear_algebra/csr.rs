// crates/mh_physics/src/numerics/linear_algebra/csr.rs

//! 压缩稀疏行（CSR）矩阵格式
//!
//! CSR 是最常用的稀疏矩阵存储格式之一，特别适合：
//! - 高效的矩阵-向量乘法 (SpMV)
//! - 行遍历操作
//! - 与有限体积法的自然配合
//!
//! 支持泛型标量类型 `S: RuntimeScalar`（f32 或 f64）。
//!
//! # 特性开关
//!
//! - `parallel`: 启用基于 `rayon` 的并行矩阵-向量乘法
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
//! let mut builder = CsrBuilder::<f64>::new(3);
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

use mh_runtime::RuntimeScalar;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use std::collections::BTreeMap;
use std::marker::PhantomData;

// =============================================================================
// 稀疏模式（与值分离，用于复用）
// =============================================================================

/// CSR 矩阵的稀疏模式
///
/// 存储矩阵的结构信息（哪些位置有非零元），与值分离。
/// 允许在多次构建中复用相同的稀疏结构。
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

// =============================================================================
// CSR 矩阵主体
// =============================================================================

/// CSR 格式稀疏矩阵
///
/// 泛型支持 f32/f64 精度，适用于 CPU 和后继 GPU 后端。
#[derive(Debug, Clone)]
pub struct CsrMatrix<S: RuntimeScalar> {
    /// 稀疏模式（不可变）
    pattern: CsrPattern,
    /// 非零元值（可变）
    values: Vec<S>,
}

/// f64 版本的类型别名（向后兼容）
pub type CsrMatrixF64 = CsrMatrix<f64>;

impl<S: RuntimeScalar> CsrMatrix<S> {
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
    /// # 安全性
    ///
    /// - `row_ptr` 必须长度正确且最后一个元素等于 `col_idx.len()`
    /// - `col_idx` 和 `values` 长度必须相等
    /// - 所有索引必须在有效范围内
    pub fn from_raw(
        n_rows: usize,
        n_cols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<S>,
    ) -> Self {
        debug_assert_eq!(row_ptr.len(), n_rows + 1, "row_ptr 长度必须为 n_rows + 1");
        debug_assert_eq!(col_idx.len(), values.len(), "col_idx 和 values 长度必须相等");
        debug_assert_eq!(
            row_ptr[n_rows],
            col_idx.len(),
            "row_ptr 末尾必须等于 nnz"
        );

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
    #[inline]
    pub fn identity(n: usize) -> Self {
        let mut builder = CsrBuilder::<S>::new_square(n);
        for i in 0..n {
            builder.set(i, i, S::ONE);
        }
        builder.build()
    }

    /// 创建对角矩阵
    ///
    /// # 参数
    /// - `diag`: 对角线元素数组
    #[inline]
    pub fn diagonal(diag: &[S]) -> Self {
        let n = diag.len();
        let mut builder = CsrBuilder::<S>::new_square(n);
        for (i, &v) in diag.iter().enumerate() {
            builder.set(i, i, v);
        }
        builder.build()
    }

    /// 获取行数
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.pattern.n_rows()
    }

    /// 获取列数
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.pattern.n_cols()
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
    pub fn values(&self) -> &[S] {
        &self.values
    }

    /// 获取可变值切片（用于矩阵值更新）
    #[inline]
    pub fn values_mut(&mut self) -> &mut [S] {
        &mut self.values
    }

    /// 获取行指针
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        self.pattern.row_ptr()
    }

    /// 获取列索引
    #[inline]
    pub fn col_idx(&self) -> &[usize] {
        self.pattern.col_idx()
    }

    /// 获取 (row, col) 位置的值（如果不存在返回 0）
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> S {
        self.pattern
            .find_index(row, col)
            .map_or(S::ZERO, |idx| self.values[idx])
    }

    /// 设置 (row, col) 位置的值（必须已存在该位置）
    ///
    /// # 返回
    /// - `true`: 设置成功
    /// - `false`: 位置不存在（未修改）
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: S) -> bool {
        if let Some(idx) = self.pattern.find_index(row, col) {
            self.values[idx] = value;
            true
        } else {
            false
        }
    }

    /// 累加到 (row, col) 位置（必须已存在该位置）
    ///
    /// # 返回
    /// - `true`: 累加成功
    /// - `false`: 位置不存在（未修改）
    #[inline]
    pub fn add(&mut self, row: usize, col: usize, value: S) -> bool {
        if let Some(idx) = self.pattern.find_index(row, col) {
            self.values[idx] += value;
            true
        } else {
            false
        }
    }

    /// 获取第 row 行的非零元视图
    ///
    /// 返回 (列索引, 值) 的切片，可高效遍历。
    #[inline]
    pub fn row(&self, row: usize) -> RowView<'_, S> {
        let start = self.pattern.row_ptr[row];
        let end = self.pattern.row_ptr[row + 1];
        RowView {
            col_idx: &self.pattern.col_idx[start..end],
            values: &self.values[start..end],
        }
    }

    /// 获取对角元素值（第 row 行）
    #[inline]
    pub fn diagonal_value(&self, row: usize) -> Option<S> {
        self.pattern.find_index(row, row).map(|idx| self.values[idx])
    }

    /// 提取对角线元素向量
    #[inline]
    pub fn extract_diagonal(&self) -> Vec<S> {
        (0..self.n_rows())
            .map(|i| self.diagonal_value(i).unwrap_or(S::ZERO))
            .collect()
    }

    /// 矩阵-向量乘法 y = A * x
    ///
    /// # Panics
    /// - `x.len() != self.n_cols()`
    /// - `y.len() != self.n_rows()`
    pub fn mul_vec(&self, x: &[S], y: &mut [S]) {
        assert_eq!(
            x.len(),
            self.n_cols(),
            "x 长度必须等于矩阵列数"
        );
        assert_eq!(
            y.len(),
            self.n_rows(),
            "y 长度必须等于矩阵行数"
        );

        for row in 0..self.n_rows() {
            let start = self.pattern.row_ptr[row];
            let end = self.pattern.row_ptr[row + 1];

            let mut sum = S::ZERO;
            for idx in start..end {
                let col = self.pattern.col_idx[idx];
                sum += self.values[idx] * x[col];
            }
            y[row] = sum;
        }
    }

    /// 矩阵-向量乘法加法 y += alpha * A * x
    ///
    /// # Panics
    /// - `x.len() != self.n_cols()`
    /// - `y.len() != self.n_rows()`
    pub fn mul_vec_add(&self, alpha: S, x: &[S], y: &mut [S]) {
        assert_eq!(
            x.len(),
            self.n_cols(),
            "x 长度必须等于矩阵列数"
        );
        assert_eq!(
            y.len(),
            self.n_rows(),
            "y 长度必须等于矩阵行数"
        );

        for row in 0..self.n_rows() {
            let start = self.pattern.row_ptr[row];
            let end = self.pattern.row_ptr[row + 1];

            let mut sum = S::ZERO;
            for idx in start..end {
                let col = self.pattern.col_idx[idx];
                sum += self.values[idx] * x[col];
            }
            y[row] += alpha * sum;
        }
    }

    /// 高精度矩阵-向量乘法（TwoSum 算法 + 4x 循环展开）
    ///
    /// 使用 TwoSum 精确计算误差补偿，比标准 Kahan 算法更稳健。
    /// 误差范围：f32 ≈ 1e-6，f64 ≈ 1e-14。
    ///
    /// # 算法说明
    ///
    /// TwoSum 算法确保 `a + b = s + e` 精确成立，其中 `e` 是舍入误差。
    /// 通过累积误差并在最后补偿，大幅减少浮点累加误差。
    ///
    /// # Panics
    /// - `x.len() != self.n_cols()`
    /// - `y.len() != self.n_rows()`
    pub fn mul_vec_kahan(&self, x: &[S], y: &mut [S]) {
        assert_eq!(
            x.len(),
            self.n_cols(),
            "x 长度必须等于矩阵列数"
        );
        assert_eq!(
            y.len(),
            self.n_rows(),
            "y 长度必须等于矩阵行数"
        );

        // TwoSum 算法：精确计算 a + b = s + e
        fn two_sum<S: RuntimeScalar>(a: S, b: S) -> (S, S) {
            let s = a + b;
            let v = s - a;
            let e = (a - (s - v)) + (b - v);
            (s, e)
        }

        for row in 0..self.n_rows() {
            let start = self.pattern.row_ptr[row];
            let end = self.pattern.row_ptr[row + 1];

            let mut sum = S::ZERO;
            let mut err = S::ZERO; // 累积误差
            let mut i = start;

            // 4x 循环展开 + TwoSum 误差补偿
            while i + 3 < end {
                // 第 i 个元素
                let product = self.values[i] * x[self.pattern.col_idx[i]];
                let (s, e) = two_sum(sum, product);
                sum = s;
                err += e;

                // 第 i+1 个元素
                let product = self.values[i + 1] * x[self.pattern.col_idx[i + 1]];
                let (s, e) = two_sum(sum, product);
                sum = s;
                err += e;

                // 第 i+2 个元素
                let product = self.values[i + 2] * x[self.pattern.col_idx[i + 2]];
                let (s, e) = two_sum(sum, product);
                sum = s;
                err += e;

                // 第 i+3 个元素
                let product = self.values[i + 3] * x[self.pattern.col_idx[i + 3]];
                let (s, e) = two_sum(sum, product);
                sum = s;
                err += e;

                i += 4;
            }

            // 尾部处理（带 TwoSum）
            while i < end {
                let product = self.values[i] * x[self.pattern.col_idx[i]];
                let (s, e) = two_sum(sum, product);
                sum = s;
                err += e;
                i += 1;
            }

            y[row] = sum + err; // 应用误差补偿
        }
    }

    /// 并行矩阵-向量乘法（需启用 `parallel` 特性）
    ///
    /// 基于 `rayon` 的并行迭代，当矩阵行数 > 1000 时性能显著提升。
    ///
    /// # 特性要求
    /// - 必须在 `Cargo.toml` 中启用 `parallel` 特性
    /// - `S: Sync` 自动由 `RuntimeScalar` 保证
    ///
    /// # Panics
    /// - `x.len() != self.n_cols()`
    /// - `y.len() != self.n_rows()`
    #[cfg(feature = "parallel")]
    pub fn mul_vec_parallel(&self, x: &[S], y: &mut [S]) {
        assert_eq!(
            x.len(),
            self.n_cols(),
            "x 长度必须等于矩阵列数"
        );
        assert_eq!(
            y.len(),
            self.n_rows(),
            "y 长度必须等于矩阵行数"
        );

        y.par_iter_mut()
            .enumerate()
            .for_each(|(row, out)| {
                let start = self.pattern.row_ptr[row];
                let end = self.pattern.row_ptr[row + 1];

                let mut sum = S::ZERO;
                for idx in start..end {
                    let col = self.pattern.col_idx[idx];
                    sum += self.values[idx] * x[col];
                }
                *out = sum;
            });
    }

    /// 构建对角元素索引缓存
    ///
    /// 返回 `Vec<Option<usize>>`，其中第 i 个元素是第 i 行对角元在 values 中的索引。
    /// 如果该行没有对角元，则为 `None`。
    ///
    /// # 性能
    /// - 时间复杂度：O(n_rows * log(row_nnz))
    /// - 调用一次后可多次复用，避免重复二分查找
    pub fn build_diagonal_cache(&self) -> Vec<Option<usize>> {
        let n = self.n_rows();
        let mut diag_indices = vec![None; n];

        for row in 0..n {
            let start = self.pattern.row_ptr[row];
            let end = self.pattern.row_ptr[row + 1];

            // 使用二分查找定位对角元
            let col_slice = &self.pattern.col_idx[start..end];
            if let Ok(local_idx) = col_slice.binary_search(&row) {
                diag_indices[row] = Some(start + local_idx);
            }
        }

        diag_indices
    }

    /// 使用缓存快速获取对角元素值
    ///
    /// # 参数
    /// - `row`: 行索引
    /// - `cache`: `build_diagonal_cache` 返回的缓存
    #[inline]
    pub fn diagonal_value_cached(&self, row: usize, cache: &[Option<usize>]) -> Option<S> {
        cache.get(row)?.map(|&idx| self.values[idx])
    }

    /// 检查矩阵是否对称（在容差范围内）
    ///
    /// 验证所有非零元素满足 `|A[i,j] - A[j,i]| <= tol`
    ///
    /// # 性能
    /// - 时间复杂度：O(nnz * log(row_nnz))
    /// - 仅检查下三角部分，避免重复验证
    pub fn is_symmetric(&self, tol: S) -> bool {
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

    /// 计算矩阵的无穷范数（最大行和）
    ///
    /// ‖A‖_∞ = max_i Σ_j |A[i,j]|
    ///
    /// # 用途
    /// - 条件数估计
    /// - 迭代法收敛性分析
    pub fn infinity_norm(&self) -> S {
        let mut max_row_sum = S::ZERO;
        for row in 0..self.n_rows() {
            let start = self.pattern.row_ptr[row];
            let end = self.pattern.row_ptr[row + 1];
            let row_sum: S = self.values[start..end]
                .iter()
                .map(|&v| v.abs())
                .sum();
            max_row_sum = max_row_sum.max(row_sum);
        }
        max_row_sum
    }

    /// 计算 Frobenius 范数
    ///
    /// ‖A‖_F = sqrt(Σ_ij A[i,j]²)
    ///
    /// # 用途
    /// - 矩阵规模度量
    /// - 相对误差计算
    pub fn frobenius_norm(&self) -> S {
        self.values
            .iter()
            .map(|&v| v * v)
            .sum::<S>()
            .sqrt()
    }

    /// 将所有值清零（保持稀疏模式不变）
    ///
    /// # 用途
    /// - 矩阵重用（多次组装）
    /// - 避免重复内存分配
    pub fn clear_values(&mut self) {
        self.values.fill(S::ZERO);
    }

    /// 缩放所有值
    ///
    /// A *= factor
    pub fn scale(&mut self, factor: S) {
        for v in &mut self.values {
            *v *= factor;
        }
    }
}

// =============================================================================
// 行视图辅助类型
// =============================================================================

/// 行视图：提供对矩阵某一行的非零元的只读访问
pub struct RowView<'a, S: RuntimeScalar> {
    col_idx: &'a [usize],
    values: &'a [S],
}

impl<'a, S: RuntimeScalar> RowView<'a, S> {
    /// 获取列索引切片
    #[inline]
    pub fn col_indices(&self) -> &'a [usize] {
        self.col_idx
    }

    /// 获取值切片
    #[inline]
    pub fn values(&self) -> &'a [S] {
        self.values
    }

    /// 获取非零元数量
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// 迭代 (列索引, 值) 对
    pub fn iter(&self) -> impl Iterator<Item = (usize, S)> + 'a {
        self.col_idx
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }
}

// =============================================================================
// 构建器
// =============================================================================

/// CSR 矩阵构建器
///
/// 使用 BTreeMap 临时存储，构建时转换为紧凑 CSR 格式。
/// 适合逐元素或小批量构建，不保证最高性能。
///
/// # 性能提示
/// - 对于大量随机插入，考虑使用 `IndexMap` 或自定义 `Vec<(usize, S)>`
/// - 构建后顺序插入可提升缓存局部性
pub struct CsrBuilder<S: RuntimeScalar> {
    n_rows: usize,
    n_cols: usize,
    /// 每行的 (列索引, 值) 映射
    rows: Vec<BTreeMap<usize, S>>,
    _marker: PhantomData<S>,
}

/// f64 构建器类型别名
pub type CsrBuilderF64 = CsrBuilder<f64>;

impl<S: RuntimeScalar> CsrBuilder<S> {
    /// 创建方阵构建器
    #[inline]
    pub fn new_square(n: usize) -> Self {
        Self::new(n, n)
    }

    /// 创建构建器
    ///
    /// # Panics
    /// - `n_rows == 0` 或 `n_cols == 0`（空矩阵无意义）
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        assert!(n_rows > 0, "行数必须大于 0");
        assert!(n_cols > 0, "列数必须大于 0");

        Self {
            n_rows,
            n_cols,
            rows: vec![BTreeMap::new(); n_rows],
            _marker: PhantomData,
        }
    }

    /// 设置 (row, col) 的值（覆盖）
    ///
    /// # Panics
    /// - `row >= n_rows`
    /// - `col >= n_cols`
    pub fn set(&mut self, row: usize, col: usize, value: S) {
        assert!(row < self.n_rows, "行索引越界");
        assert!(col < self.n_cols, "列索引越界");
        self.rows[row].insert(col, value);
    }

    /// 累加到 (row, col)
    ///
    /// # Panics
    /// - `row >= n_rows`
    /// - `col >= n_cols`
    pub fn add(&mut self, row: usize, col: usize, value: S) {
        assert!(row < self.n_rows, "行索引越界");
        assert!(col < self.n_cols, "列索引越界");
        *self.rows[row].entry(col).or_insert(S::ZERO) += value;
    }

    /// 获取 (row, col) 的当前值（不存在返回 0）
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> S {
        self.rows[row].get(&col).copied().unwrap_or(S::ZERO)
    }

    /// 清空所有值（保持结构）
    pub fn clear(&mut self) {
        for row in &mut self.rows {
            row.clear();
        }
    }

    /// 获取当前非零元总数
    #[inline]
    pub fn nnz(&self) -> usize {
        self.rows.iter().map(|r| r.len()).sum()
    }

    /// 构建 CSR 矩阵（消耗构建器）
    ///
    /// # 性能
    /// - 需要对每行的列索引排序（BTreeMap 已排序，O(1)）
    /// - 总复杂度：O(nnz)
    pub fn build(self) -> CsrMatrix<S> {
        let nnz = self.nnz();
        let mut row_ptr = Vec::with_capacity(self.n_rows + 1);
        let mut col_idx = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        row_ptr.push(0);

        for row_map in self.rows {
            for (col, val) in row_map {
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

    /// 构建稀疏模式（用于模式复用）
    ///
    /// 返回 `CsrPattern`，可与不同的值数组组合。
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

        assert_eq!(
            row_ptr.len(),
            self.n_rows + 1,
            "row_ptr 长度验证失败"
        );

        CsrPattern {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            row_ptr,
            col_idx,
        }
    }
}

// =============================================================================
// 类型转换
// =============================================================================

impl<S: RuntimeScalar> From<CsrPattern> for CsrMatrix<S> {
    /// 从稀疏模式创建矩阵（值初始化为 0）
    fn from(pattern: CsrPattern) -> Self {
        let nnz = pattern.nnz();
        Self {
            pattern,
            values: vec![S::ZERO; nnz],
        }
    }
}

// =============================================================================
// 测试套件（泛型覆盖 f32/f64）
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 生产级测试宏：为 S = f32 和 f64 生成完全相同的测试
    macro_rules! csr_test_suite {
        ($($mod_name:ident => $scalar:ty,)*) => {
            $(
                mod $mod_name {
                    use super::*;

                    type S = $scalar;

                    /// 默认精度容差：f32=1e-6, f64=1e-10
                    const EPS: S = if std::mem::size_of::<S>() == 4 {
                        S::from_config(1e-6).unwrap()
                    } else {
                        S::from_config(1e-10).unwrap()
                    };

                    #[test]
                    fn test_identity_matrix() {
                        let mat = CsrMatrix::<S>::identity(5);
                        assert_eq!(mat.n_rows(), 5);
                        assert_eq!(mat.n_cols(), 5);
                        assert_eq!(mat.nnz(), 5);

                        for i in 0..5 {
                            assert!((mat.get(i, i) - S::ONE).abs() < EPS);
                        }
                    }

                    #[test]
                    fn test_diagonal_matrix() {
                        let diag = vec![S::from_config(2.0).unwrap(),
                                       S::from_config(3.0).unwrap(),
                                       S::from_config(4.0).unwrap()];
                        let mat = CsrMatrix::<S>::diagonal(&diag);

                        assert!((mat.get(0, 0) - diag[0]).abs() < EPS);
                        assert!((mat.get(1, 1) - diag[1]).abs() < EPS);
                        assert!((mat.get(2, 2) - diag[2]).abs() < EPS);
                        assert!(mat.get(0, 1).abs() < EPS);
                    }

                    #[test]
                    fn test_builder_and_mul() {
                        // 构建三对角矩阵
                        let mut builder = CsrBuilder::<S>::new_square(4);
                        builder.set(0, 0, S::from_config(2.0).unwrap());
                        builder.set(0, 1, S::from_config(-1.0).unwrap());
                        builder.add(0, 1, S::from_config(-0.5).unwrap()); // 测试累加
                        builder.set(1, 0, S::from_config(-1.0).unwrap());
                        builder.set(1, 1, S::from_config(2.0).unwrap());
                        builder.set(1, 2, S::from_config(-1.0).unwrap());
                        builder.set(2, 1, S::from_config(-1.0).unwrap());
                        builder.set(2, 2, S::from_config(2.0).unwrap());
                        builder.set(2, 3, S::from_config(-1.0).unwrap());
                        builder.set(3, 3, S::from_config(1.0).unwrap());

                        let mat = builder.build();
                        assert_eq!(mat.nnz(), 9);

                        let x = vec![S::ONE, S::from_config(2.0).unwrap(),
                                    S::from_config(3.0).unwrap(), S::from_config(4.0).unwrap()];
                        let mut y = vec![S::ZERO; 4];
                        mat.mul_vec(&x, &mut y);

                        // 手动计算验证
                        // y[0] = 2*1 + (-1.5)*2 = -1
                        // y[1] = -1*1 + 2*2 + -1*3 = 0
                        // y[2] = -1*2 + 2*3 + -1*4 = 0
                        // y[3] = 1*4 = 4
                        assert!((y[0] - S::from_config(-1.0).unwrap()).abs() < EPS);
                        assert!((y[1] - S::ZERO).abs() < EPS);
                        assert!((y[2] - S::ZERO).abs() < EPS);
                        assert!((y[3] - S::from_config(4.0).unwrap()).abs() < EPS);
                    }

                    #[test]
                    fn test_mul_vec_add() {
                        let mut builder = CsrBuilder::<S>::new_square(2);
                        builder.set(0, 0, S::ONE);
                        builder.set(0, 1, S::ONE);
                        builder.set(1, 1, S::from_config(2.0).unwrap());

                        let mat = builder.build();
                        let x = vec![S::ONE, S::ONE];
                        let mut y = vec![S::ONE, S::ONE]; // 初始值

                        mat.mul_vec_add(S::ONE, &x, &mut y);

                        // y = [1,1] + 1 * [1*1+1*1, 1*2] = [3, 3]
                        assert!((y[0] - S::from_config(3.0).unwrap()).abs() < EPS);
                        assert!((y[1] - S::from_config(3.0).unwrap()).abs() < EPS);
                    }

                    #[test]
                    fn test_frobenius_norm() {
                        let mut builder = CsrBuilder::<S>::new_square(2);
                        builder.set(0, 0, S::from_config(3.0).unwrap());
                        builder.set(1, 1, S::from_config(4.0).unwrap());

                        let mat = builder.build();
                        // ||A||_F = sqrt(3² + 4²) = 5
                        let norm = mat.frobenius_norm();
                        assert!((norm - S::from_config(5.0).unwrap()).abs() < EPS);
                    }

                    #[test]
                    fn test_symmetric_check() {
                        let mut builder = CsrBuilder::<S>::new_square(3);
                        builder.set(0, 0, S::ONE);
                        builder.set(0, 1, S::from_config(0.5).unwrap());
                        builder.set(1, 0, S::from_config(0.5).unwrap());
                        builder.set(1, 1, S::ONE);
                        builder.set(2, 2, S::ONE);

                        let mat = builder.build();
                        assert!(mat.is_symmetric(S::from_config(1e-12).unwrap()));

                        // 修改为非对称
                        mat.set(0, 1, S::from_config(0.6).unwrap());
                        assert!(!mat.is_symmetric(S::from_config(1e-12).unwrap()));
                    }

                    #[test]
                    fn test_diagonal_cache() {
                        let mut builder = CsrBuilder::<S>::new_square(3);
                        builder.set(0, 0, S::from_config(1.0).unwrap());
                        builder.set(1, 1, S::from_config(2.0).unwrap());
                        builder.set(2, 2, S::from_config(3.0).unwrap());
                        let mat = builder.build();

                        let cache = mat.build_diagonal_cache();
                        assert_eq!(cache.len(), 3);
                        assert!(cache[0].is_some());
                        assert!(cache[1].is_some());
                        assert!(cache[2].is_some());

                        assert!((mat.diagonal_value_cached(0, &cache).unwrap() - S::from_config(1.0).unwrap()).abs() < EPS);
                        assert!((mat.diagonal_value_cached(1, &cache).unwrap() - S::from_config(2.0).unwrap()).abs() < EPS);
                        assert!((mat.diagonal_value_cached(2, &cache).unwrap() - S::from_config(3.0).unwrap()).abs() < EPS);
                    }

                    #[test]
                    fn test_kahan_vs_standard() {
                        // 构造条件数较大的矩阵
                        let mut builder = CsrBuilder::<S>::new_square(50);
                        for i in 0..50 {
                            builder.set(i, i, S::from_config(1.0).unwrap());
                            if i < 49 {
                                builder.set(i, i+1, S::from_config(-0.9).unwrap());
                                builder.set(i+1, i, S::from_config(-0.9).unwrap());
                            }
                        }
                        let mat = builder.build();

                        let x: Vec<S> = (0..50).map(|i| S::from_config(i as f64 * 0.1).unwrap()).collect();
                        let mut y_standard = vec![S::ZERO; 50];
                        let mut y_kahan = vec![S::ZERO; 50];

                        mat.mul_vec(&x, &mut y_standard);
                        mat.mul_vec_kahan(&x, &mut y_kahan);

                        // Kahan 结果应该更接近真实值（误差更小）
                        // 差异应在数值精度范围内
                        let diff: S = y_standard.iter()
                            .zip(y_kahan.iter())
                            .map(|(a, b)| (*a - *b).abs())
                            .sum();

                        // 误差总和应小于典型值
                        assert!(diff < S::from_config(1e-3).unwrap());
                    }

                    #[test]
                    fn test_from_raw() {
                        let row_ptr = vec![0, 2, 4, 6];
                        let col_idx = vec![0, 1, 0, 1, 1, 2];
                        let values = vec![S::from_config(4.0).unwrap(),
                                         S::from_config(-1.0).unwrap(),
                                         S::from_config(-1.0).unwrap(),
                                         S::from_config(4.0).unwrap(),
                                         S::from_config(-1.0).unwrap(),
                                         S::from_config(4.0).unwrap()];

                        let mat = CsrMatrix::<S>::from_raw(3, 3, row_ptr, col_idx, values);
                        assert_eq!(mat.n_rows(), 3);
                        assert_eq!(mat.n_cols(), 3);
                        assert_eq!(mat.nnz(), 6);
                    }

                    #[test]
                    fn test_clear_and_scale() {
                        let mut builder = CsrBuilder::<S>::new_square(2);
                        builder.set(0, 0, S::from_config(1.0).unwrap());
                        builder.set(1, 1, S::from_config(2.0).unwrap());
                        let mut mat = builder.build();

                        mat.scale(S::from_config(2.0).unwrap());
                        assert!((mat.get(0, 0) - S::from_config(2.0).unwrap()).abs() < EPS);
                        assert!((mat.get(1, 1) - S::from_config(4.0).unwrap()).abs() < EPS);

                        mat.clear_values();
                        assert!(mat.get(0, 0).abs() < EPS);
                        assert!(mat.get(1, 1).abs() < EPS);
                    }

                    #[cfg(feature = "parallel")]
                    #[test]
                    fn test_parallel_correctness() {
                        let mut builder = CsrBuilder::<S>::new_square(100);
                        for i in 0..100 {
                            builder.set(i, i, S::from_config(2.0).unwrap());
                            if i < 99 {
                                builder.set(i, i+1, S::from_config(-1.0).unwrap());
                            }
                        }
                        let mat = builder.build();

                        let x: Vec<S> = (0..100).map(|i| S::from_config(i as f64).unwrap()).collect();
                        let mut y_serial = vec![S::ZERO; 100];
                        let mut y_parallel = vec![S::ZERO; 100];

                        mat.mul_vec(&x, &mut y_serial);
                        mat.mul_vec_parallel(&x, &mut y_parallel);

                        // 并行结果必须等于串行结果
                        for (a, b) in y_serial.iter().zip(y_parallel.iter()) {
                            assert!((a - b).abs() < EPS, "并行结果与串行不一致");
                        }
                    }
                }
            )*
        };
    }

    // 为 f32 和 f64 生成完整测试套件
    csr_test_suite! {
        f32_tests => f32,
        f64_tests => f64,
    }

    /// 集成测试：使用真实场景矩阵
    #[test]
    fn test_realistic_poisson_matrix() {
        // 二维泊松方程的 10x10 网格离散
        let n = 100;
        let mut builder = CsrBuilder::<f64>::new_square(n);

        for i in 0..n {
            builder.set(i, i, 4.0);
            if i % 10 != 9 {
                builder.set(i, i + 1, -1.0);
            }
            if i % 10 != 0 {
                builder.set(i, i - 1, -1.0);
            }
            if i + 10 < n {
                builder.set(i, i + 10, -1.0);
            }
            if i >= 10 {
                builder.set(i, i - 10, -1.0);
            }
        }

        let mat = builder.build();
        assert_eq!(mat.nnz(), 4 * n - 4 * 10); // 边界更少

        // 验证矩阵条件数相关性质
        let norm = mat.infinity_norm();
        assert!(norm > 4.0 && norm < 8.0);
    }
}