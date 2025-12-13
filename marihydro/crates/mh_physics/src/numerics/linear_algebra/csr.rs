// crates/mh_physics/src/numerics/linear_algebra/csr.rs

//! åŽ‹ç¼©ç¨€ç–è¡Œï¼ˆCSRï¼‰çŸ©é˜µæ ¼å¼
//!
//! CSR æ˜¯æœ€å¸¸ç”¨çš„ç¨€ç–çŸ©é˜µå­˜å‚¨æ ¼å¼ä¹‹ä¸€ï¼Œç‰¹åˆ«é€‚åˆï¼š
//! - é«˜æ•ˆçš„çŸ©é˜µ-å‘é‡ä¹˜æ³• (SpMV)
//! - è¡ŒéåŽ†æ“ä½œ
//! - ä¸Žæœ‰é™ä½“ç§¯æ³•çš„è‡ªç„¶é…åˆ
//!
//! æ”¯æŒæ³›åž‹æ ‡é‡ç±»åž‹ `S: RuntimeScalar`ï¼ˆf32 æˆ– f64ï¼‰ã€‚
//!
//! # æ ¼å¼è¯´æ˜Ž
//!
//! CSR ä½¿ç”¨ä¸‰ä¸ªæ•°ç»„å­˜å‚¨ï¼š
//! - `row_ptr`: è¡ŒæŒ‡é’ˆï¼Œé•¿åº¦ n_rows + 1ï¼Œrow_ptr[i] æ˜¯ç¬¬ i è¡Œç¬¬ä¸€ä¸ªéžé›¶å…ƒçš„ç´¢å¼•
//! - `col_idx`: åˆ—ç´¢å¼•ï¼Œä¸Žéžé›¶å…ƒä¸€ä¸€å¯¹åº”
//! - `values`: éžé›¶å…ƒå€¼
//!
//! # ä½¿ç”¨ç¤ºä¾‹
//!
//! ```ignore
//! use mh_physics::numerics::linear_algebra::csr::{CsrBuilder, CsrMatrix};
//!
//! // ä½¿ç”¨æž„å»ºå™¨åˆ›å»ºçŸ©é˜µ
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
//! // çŸ©é˜µ-å‘é‡ä¹˜æ³•
//! let x = vec![1.0, 2.0, 3.0];
//! let mut y = vec![0.0; 3];
//! matrix.mul_vec(&x, &mut y);
//! ```

use mh_core::RuntimeScalar;
use std::collections::BTreeMap;
use std::marker::PhantomData;

/// CSR çŸ©é˜µçš„ç¨€ç–æ¨¡å¼
///
/// å­˜å‚¨çŸ©é˜µçš„ç»“æž„ä¿¡æ¯ï¼ˆå“ªäº›ä½ç½®æœ‰éžé›¶å…ƒï¼‰ï¼Œä¸Žå€¼åˆ†ç¦»
#[derive(Debug, Clone)]
pub struct CsrPattern {
    /// è¡Œæ•°
    n_rows: usize,
    /// åˆ—æ•°
    n_cols: usize,
    /// è¡ŒæŒ‡é’ˆ
    row_ptr: Vec<usize>,
    /// åˆ—ç´¢å¼•
    col_idx: Vec<usize>,
}

impl CsrPattern {
    /// èŽ·å–è¡Œæ•°
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// èŽ·å–åˆ—æ•°
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// èŽ·å–éžé›¶å…ƒæ•°é‡
    #[inline]
    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    /// èŽ·å–è¡ŒæŒ‡é’ˆåˆ‡ç‰‡
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// èŽ·å–åˆ—ç´¢å¼•åˆ‡ç‰‡
    #[inline]
    pub fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }

    /// èŽ·å–ç¬¬ row è¡Œçš„éžé›¶å…ƒåˆ—ç´¢å¼•
    #[inline]
    pub fn row_indices(&self, row: usize) -> &[usize] {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        &self.col_idx[start..end]
    }

    /// èŽ·å–ç¬¬ row è¡Œçš„éžé›¶å…ƒæ•°é‡
    #[inline]
    pub fn row_nnz(&self, row: usize) -> usize {
        self.row_ptr[row + 1] - self.row_ptr[row]
    }

    /// æŸ¥æ‰¾ (row, col) å¯¹åº”çš„å€¼ç´¢å¼•
    pub fn find_index(&self, row: usize, col: usize) -> Option<usize> {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        let indices = &self.col_idx[start..end];

        // ç”±äºŽåˆ—ç´¢å¼•æ˜¯æœ‰åºçš„ï¼Œå¯ä»¥ä½¿ç”¨äºŒåˆ†æŸ¥æ‰¾
        match indices.binary_search(&col) {
            Ok(local_idx) => Some(start + local_idx),
            Err(_) => None,
        }
    }

    /// æ£€æŸ¥ (row, col) æ˜¯å¦æœ‰éžé›¶å…ƒ
    pub fn has_entry(&self, row: usize, col: usize) -> bool {
        self.find_index(row, col).is_some()
    }
}

/// CSR æ ¼å¼ç¨€ç–çŸ©é˜µ
#[derive(Debug, Clone)]
pub struct CsrMatrix<S: RuntimeScalar> {
    /// ç¨€ç–æ¨¡å¼
    pattern: CsrPattern,
    /// éžé›¶å…ƒå€¼
    values: Vec<S>,
}

/// Legacy ç±»åž‹åˆ«åï¼Œä¿æŒå‘åŽå…¼å®¹
pub type CsrMatrixF64 = CsrMatrix<f64>;

impl<S: RuntimeScalar> CsrMatrix<S> {
    /// ä»ŽåŽŸå§‹ CSR æ•°æ®åˆ›å»ºçŸ©é˜µ
    ///
    /// # å‚æ•°
    ///
    /// - `n_rows`: è¡Œæ•°
    /// - `n_cols`: åˆ—æ•°
    /// - `row_ptr`: è¡ŒæŒ‡é’ˆæ•°ç»„ï¼Œé•¿åº¦ n_rows + 1
    /// - `col_idx`: åˆ—ç´¢å¼•æ•°ç»„
    /// - `values`: éžé›¶å…ƒå€¼æ•°ç»„
    ///
    /// # ç¤ºä¾‹
    ///
    /// ```ignore
    /// let matrix = CsrMatrix::<S>::<S>::<S>::from_raw(
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
        values: Vec<S>,
    ) -> Self {
        debug_assert_eq!(row_ptr.len(), n_rows + 1, "row_ptr é•¿åº¦å¿…é¡»ä¸º n_rows + 1");
        debug_assert_eq!(col_idx.len(), values.len(), "col_idx å’Œ values é•¿åº¦å¿…é¡»ç›¸ç­‰");
        debug_assert_eq!(row_ptr[n_rows], col_idx.len(), "row_ptr æœ«å°¾å¿…é¡»ç­‰äºŽ nnz");

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

    /// åˆ›å»ºå•ä½çŸ©é˜µ
    pub fn identity(n: usize) -> Self {
        let mut builder = CsrBuilder::<S>::new_square(n);
        for i in 0..n {
            builder.set(i, i, S::ONE);
        }
        builder.build()
    }

    /// åˆ›å»ºå¯¹è§’çŸ©é˜µ
    pub fn diagonal(diag: &[S]) -> Self {
        let n = diag.len();
        let mut builder = CsrBuilder::<S>::new_square(n);
        for (i, &v) in diag.iter().enumerate() {
            builder.set(i, i, v);
        }
        builder.build()
    }

    /// èŽ·å–è¡Œæ•°
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.pattern.n_rows
    }

    /// èŽ·å–åˆ—æ•°
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.pattern.n_cols
    }

    /// èŽ·å–éžé›¶å…ƒæ•°é‡
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// èŽ·å–ç¨€ç–æ¨¡å¼å¼•ç”¨
    #[inline]
    pub fn pattern(&self) -> &CsrPattern {
        &self.pattern
    }

    /// èŽ·å–å€¼åˆ‡ç‰‡
    #[inline]
    pub fn values(&self) -> &[S] {
        &self.values
    }

    /// èŽ·å–å¯å˜å€¼åˆ‡ç‰‡
    #[inline]
    pub fn values_mut(&mut self) -> &mut [S] {
        &mut self.values
    }

    /// èŽ·å–è¡ŒæŒ‡é’ˆ
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        &self.pattern.row_ptr
    }

    /// èŽ·å–åˆ—ç´¢å¼•
    #[inline]
    pub fn col_idx(&self) -> &[usize] {
        &self.pattern.col_idx
    }

    /// èŽ·å– (row, col) ä½ç½®çš„å€¼
    pub fn get(&self, row: usize, col: usize) -> S {
        match self.pattern.find_index(row, col) {
            Some(idx) => self.values[idx],
            None => S::ZERO,
        }
    }

    /// è®¾ç½® (row, col) ä½ç½®çš„å€¼ï¼ˆå¿…é¡»å·²å­˜åœ¨è¯¥ä½ç½®ï¼‰
    pub fn set(&mut self, row: usize, col: usize, value: S) -> bool {
        match self.pattern.find_index(row, col) {
            Some(idx) => {
                self.values[idx] = value;
                true
            }
            None => false,
        }
    }

    /// ç´¯åŠ åˆ° (row, col) ä½ç½®
    pub fn add(&mut self, row: usize, col: usize, value: S) -> bool {
        match self.pattern.find_index(row, col) {
            Some(idx) => {
                self.values[idx] += value;
                true
            }
            None => false,
        }
    }

    /// èŽ·å–ç¬¬ row è¡Œçš„éžé›¶å…ƒ (åˆ—ç´¢å¼•, å€¼) åˆ‡ç‰‡
    pub fn row(&self, row: usize) -> RowView<'_, S> {
        let start = self.pattern.row_ptr[row];
        let end = self.pattern.row_ptr[row + 1];
        RowView {
            col_idx: &self.pattern.col_idx[start..end],
            values: &self.values[start..end],
        }
    }

    /// èŽ·å–å¯¹è§’å…ƒç´ å€¼
    pub fn diagonal_value(&self, row: usize) -> Option<S> {
        self.pattern.find_index(row, row).map(|idx| self.values[idx])
    }

    /// èŽ·å–å¯¹è§’å…ƒç´ å‘é‡ï¼ˆæå–å¯¹è§’çº¿ï¼‰
    pub fn extract_diagonal(&self) -> Vec<S> {
        (0..self.n_rows())
            .map(|i| self.diagonal_value(i).unwrap_or(S::ZERO))
            .collect()
    }

    /// çŸ©é˜µ-å‘é‡ä¹˜æ³• y = A*x
    pub fn mul_vec(&self, x: &[S], y: &mut [S]) {
        assert_eq!(x.len(), self.n_cols());
        assert_eq!(y.len(), self.n_rows());

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

    /// çŸ©é˜µ-å‘é‡ä¹˜æ³•åŠ æ³• y = y + alpha * A * x
    pub fn mul_vec_add(&self, alpha: S, x: &[S], y: &mut [S]) {
        assert_eq!(x.len(), self.n_cols());
        assert_eq!(y.len(), self.n_rows());

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

    /// å°†æ‰€æœ‰å€¼æ¸…é›¶ï¼ˆä¿æŒæ¨¡å¼ï¼‰
    pub fn clear_values(&mut self) {
        self.values.fill(S::ZERO);
    }

    /// ç¼©æ”¾æ‰€æœ‰å€¼
    pub fn scale(&mut self, factor: S) {
        for v in &mut self.values {
            *v *= factor;
        }
    }

    /// èŽ·å– Frobenius èŒƒæ•°
    pub fn frobenius_norm(&self) -> S {
        self.values.iter().map(|&v| v * v).sum::<S>().sqrt()
    }

    /// é«˜ç²¾åº¦çŸ©é˜µ-å‘é‡ä¹˜æ³•ï¼ˆKahan ç´¯åŠ  + 4x å¾ªçŽ¯å±•å¼€ï¼‰
    ///
    /// ä½¿ç”¨ Kahan æ±‚å’Œç®—æ³•å‡å°‘æµ®ç‚¹ç´¯åŠ è¯¯å·®ï¼Œé€‚ç”¨äºŽéœ€è¦é«˜ç²¾åº¦çš„åœºæ™¯ã€‚
    /// 4x å¾ªçŽ¯å±•å¼€æé«˜ CPU æµæ°´çº¿åˆ©ç”¨çŽ‡ã€‚
    /// 
    /// æ³¨æ„ï¼šæ³›åž‹ç‰ˆæœ¬ä½¿ç”¨æ™®é€šæ±‚å’Œã€‚å¯¹äºŽ f64 é«˜ç²¾åº¦éœ€æ±‚ï¼Œ
    /// è¯·è€ƒè™‘ä½¿ç”¨ä¸“é—¨çš„ `mul_vec_kahan_f64` æ–¹æ³•ã€‚
    pub fn mul_vec_kahan(&self, x: &[S], y: &mut [S]) {
        assert_eq!(x.len(), self.n_cols());
        assert_eq!(y.len(), self.n_rows());

        for row in 0..self.n_rows() {
            let start = self.pattern.row_ptr[row];
            let end = self.pattern.row_ptr[row + 1];

            let mut sum = S::ZERO;
            let mut i = start;

            // 4x å¾ªçŽ¯å±•å¼€ä¸»å¾ªçŽ¯
            while i + 3 < end {
                sum = sum + self.values[i] * x[self.pattern.col_idx[i]];
                sum = sum + self.values[i + 1] * x[self.pattern.col_idx[i + 1]];
                sum = sum + self.values[i + 2] * x[self.pattern.col_idx[i + 2]];
                sum = sum + self.values[i + 3] * x[self.pattern.col_idx[i + 3]];
                i += 4;
            }

            // å°¾éƒ¨å¤„ç†
            while i < end {
                sum = sum + self.values[i] * x[self.pattern.col_idx[i]];
                i += 1;
            }

            y[row] = sum;
        }
    }

    /// æž„å»ºå¯¹è§’å…ƒç´ ç´¢å¼•ç¼“å­˜
    ///
    /// è¿”å›žä¸€ä¸ªå‘é‡ï¼Œå…¶ä¸­ç¬¬ i ä¸ªå…ƒç´ æ˜¯ç¬¬ i è¡Œå¯¹è§’å…ƒç´ åœ¨ values æ•°ç»„ä¸­çš„ç´¢å¼•ã€‚
    /// å¦‚æžœè¯¥è¡Œæ²¡æœ‰å¯¹è§’å…ƒç´ ï¼Œåˆ™ä¸º Noneã€‚
    pub fn build_diagonal_cache(&self) -> Vec<Option<usize>> {
        let n = self.n_rows();
        let mut diag_indices = vec![None; n];

        for row in 0..n {
            let start = self.pattern.row_ptr[row];
            let end = self.pattern.row_ptr[row + 1];
            
            // ä½¿ç”¨äºŒåˆ†æŸ¥æ‰¾ï¼ˆåˆ—ç´¢å¼•æ˜¯æœ‰åºçš„ï¼‰
            let col_slice = &self.pattern.col_idx[start..end];
            if let Ok(local_idx) = col_slice.binary_search(&row) {
                diag_indices[row] = Some(start + local_idx);
            }
        }

        diag_indices
    }

    /// ä½¿ç”¨ç¼“å­˜å¿«é€ŸèŽ·å–å¯¹è§’å…ƒç´ å€¼
    #[inline]
    pub fn diagonal_value_cached(&self, row: usize, cache: &[Option<usize>]) -> Option<S> {
        cache.get(row)?.map(|idx| self.values[idx])
    }

    /// æ£€æŸ¥çŸ©é˜µæ˜¯å¦å¯¹ç§°
    ///
    /// éªŒè¯æ‰€æœ‰éžé›¶å…ƒç´  A[i,j] == A[j,i]ã€‚
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

    /// èŽ·å–çŸ©é˜µçš„æ— ç©·èŒƒæ•°ï¼ˆè¡Œæœ€å¤§ç»å¯¹å€¼å’Œï¼‰
    pub fn infinity_norm(&self) -> S {
        let mut max_row_sum = S::ZERO;
        for row in 0..self.n_rows() {
            let start = self.pattern.row_ptr[row];
            let end = self.pattern.row_ptr[row + 1];
            let row_sum: S = self.values[start..end].iter().map(|v| v.abs()).sum();
            max_row_sum = max_row_sum.max(row_sum);
        }
        max_row_sum
    }
}

/// è¡Œè§†å›¾
pub struct RowView<'a, S: RuntimeScalar> {
    col_idx: &'a [usize],
    values: &'a [S],
}

impl<'a, S: RuntimeScalar> RowView<'a, S> {
    /// èŽ·å–åˆ—ç´¢å¼•
    pub fn col_indices(&self) -> &'a [usize] {
        self.col_idx
    }

    /// èŽ·å–å€¼
    pub fn values(&self) -> &'a [S] {
        self.values
    }

    /// èŽ·å–éžé›¶å…ƒæ•°é‡
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// è¿­ä»£ (åˆ—ç´¢å¼•, å€¼) å¯¹
    pub fn iter(&self) -> impl Iterator<Item = (usize, S)> + 'a {
        self.col_idx.iter().copied().zip(self.values.iter().copied())
    }
}

/// CSR çŸ©é˜µæž„å»ºå™¨
///
/// ä½¿ç”¨ BTreeMap ä¸´æ—¶å­˜å‚¨ï¼Œæž„å»ºæ—¶è½¬æ¢ä¸ºç´§å‡‘ CSR æ ¼å¼
pub struct CsrBuilder<S: RuntimeScalar> {
    n_rows: usize,
    n_cols: usize,
    /// æ¯è¡Œçš„ (åˆ—ç´¢å¼•, å€¼) æ˜ å°„
    rows: Vec<BTreeMap<usize, S>>,
    _marker: PhantomData<S>,
}

/// Legacy ç±»åž‹åˆ«åï¼Œä¿æŒå‘åŽå…¼å®¹
pub type CsrBuilderF64 = CsrBuilder<f64>;

impl<S: RuntimeScalar> CsrBuilder<S> {
    /// åˆ›å»ºæ–¹é˜µæž„å»ºå™¨
    pub fn new_square(n: usize) -> Self {
        Self::new(n, n)
    }

    /// åˆ›å»ºæž„å»ºå™¨
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            n_rows,
            n_cols,
            rows: vec![BTreeMap::new(); n_rows],
            _marker: PhantomData,
        }
    }

    /// è®¾ç½® (row, col) çš„å€¼
    pub fn set(&mut self, row: usize, col: usize, value: S) {
        assert!(row < self.n_rows, "row index out of bounds");
        assert!(col < self.n_cols, "column index out of bounds");
        self.rows[row].insert(col, value);
    }

    /// ç´¯åŠ åˆ° (row, col)
    pub fn add(&mut self, row: usize, col: usize, value: S) {
        assert!(row < self.n_rows, "row index out of bounds");
        assert!(col < self.n_cols, "column index out of bounds");
        *self.rows[row].entry(col).or_insert(S::ZERO) += value;
    }

    /// èŽ·å– (row, col) çš„å€¼
    pub fn get(&self, row: usize, col: usize) -> S {
        self.rows[row].get(&col).copied().unwrap_or(S::ZERO)
    }

    /// æ¸…ç©ºæž„å»ºå™¨
    pub fn clear(&mut self) {
        for row in &mut self.rows {
            row.clear();
        }
    }

    /// èŽ·å–éžé›¶å…ƒæ•°é‡
    pub fn nnz(&self) -> usize {
        self.rows.iter().map(|r| r.len()).sum()
    }

    /// æž„å»º CSR çŸ©é˜µ
    pub fn build(self) -> CsrMatrix<S> {
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

    /// æž„å»ºå¹¶è¿”å›žç¨€ç–æ¨¡å¼ï¼ˆç”¨äºŽæ¨¡å¼å¤ç”¨ï¼‰
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

/// ä»Žæ¨¡å¼åˆ›å»ºçŸ©é˜µï¼ˆå€¼åˆå§‹åŒ–ä¸ºé›¶ï¼‰
impl<S: RuntimeScalar> From<CsrPattern> for CsrMatrix<S> {
    fn from(pattern: CsrPattern) -> Self {
        let nnz = pattern.nnz();
        Self {
            pattern,
            values: vec![S::ZERO; nnz],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 测试用类型别名
    type S = f64;

    #[test]
    fn test_identity() {
        let mat = CsrMatrix::<S>::identity(3);
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
        let mat = CsrMatrix::<S>::diagonal(&diag);

        assert!((mat.get(0, 0) - 2.0).abs() < 1e-14);
        assert!((mat.get(1, 1) - 3.0).abs() < 1e-14);
        assert!((mat.get(2, 2) - 4.0).abs() < 1e-14);

        assert_eq!(mat.extract_diagonal(), diag);
    }

    #[test]
    fn test_builder() {
        let mut builder = CsrBuilder::<S>::new_square(3);
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
        let mut builder = CsrBuilder::<S>::new_square(3);
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
        let mut builder = CsrBuilder::<S>::new_square(2);
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
        let mut builder = CsrBuilder::<S>::new_square(3);
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
        let mut builder = CsrBuilder::<S>::new_square(2);
        builder.set(0, 0, 1.0);
        builder.set(0, 1, 2.0);
        builder.set(1, 1, 3.0);

        let pattern = builder.build_pattern();
        let mat: CsrMatrix<S> = pattern.into();

        assert_eq!(mat.nnz(), 3);
        assert!(mat.get(0, 0).abs() < 1e-14); // 值为 0
        assert!(mat.get(0, 1).abs() < 1e-14);
    }

    #[test]
    fn test_frobenius_norm() {
        let mut builder = CsrBuilder::<S>::new_square(2);
        builder.set(0, 0, 3.0);
        builder.set(1, 1, 4.0);

        let mat = builder.build();
        // ||A||_F = sqrt(3² + 4²) = 5
        assert!((mat.frobenius_norm() - 5.0).abs() < 1e-14);
    }
}
