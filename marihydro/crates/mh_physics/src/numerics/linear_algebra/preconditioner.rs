// crates/mh_physics/src/numerics/linear_algebra/preconditioner.rs

//! é¢„æ¡ä»¶å™¨æ¨¡å—
//!
//! é¢„æ¡ä»¶å™¨ç”¨äºŽåŠ é€Ÿè¿­ä»£æ±‚è§£å™¨çš„æ”¶æ•›ã€‚æ ¸å¿ƒæ€æƒ³æ˜¯å°†åŽŸé—®é¢˜ Ax = b
//! è½¬æ¢ä¸ºæ¡ä»¶æ•°æ›´å¥½çš„é—®é¢˜ Mâ»Â¹Ax = Mâ»Â¹bã€‚
//! 
//! æ”¯æŒæ³›åž‹æ ‡é‡ç±»åž‹ `S: RuntimeScalar`ï¼ˆf32 æˆ– f64ï¼‰ã€‚
//!
//! # é¢„æ¡ä»¶å™¨ç±»åž‹
//!
//! - [`IdentityPreconditioner`]: æ’ç­‰é¢„æ¡ä»¶å™¨ï¼ˆæ— é¢„æ¡ä»¶ï¼‰
//! - [`JacobiPreconditioner`]: Jacobi é¢„æ¡ä»¶å™¨ï¼ˆå¯¹è§’é¢„æ¡ä»¶ï¼‰
//!
//! # ä½¿ç”¨ç¤ºä¾‹
//!
//! ```ignore
//! use mh_physics::numerics::linear_algebra::{
//!     CsrMatrix, JacobiPreconditioner, Preconditioner,
//! };
//!
//! let matrix: CsrMatrix<f64> = /* ... */;
//! let precond = JacobiPreconditioner::from_matrix(&matrix);
//!
//! let r = vec![1.0, 2.0, 3.0];
//! let mut z = vec![0.0; 3];
//! precond.apply(&r, &mut z);  // z = Mâ»Â¹ * r
//! ```

use super::csr::CsrMatrix;
use mh_runtime::RuntimeScalar;

/// é¢„æ¡ä»¶å™¨ trait
///
/// é¢„æ¡ä»¶å™¨çš„æ ¸å¿ƒæ“ä½œæ˜¯ `apply`: z = Mâ»Â¹ * r
pub trait Preconditioner<S: RuntimeScalar>: Send + Sync {
    /// åº”ç”¨é¢„æ¡ä»¶å™¨: z = Mâ»Â¹ * r
    ///
    /// # å‚æ•°
    ///
    /// - `r`: è¾“å…¥å‘é‡ï¼ˆé€šå¸¸æ˜¯æ®‹å·®ï¼‰
    /// - `z`: è¾“å‡ºå‘é‡ï¼ˆé¢„æ¡ä»¶åŽçš„æ–¹å‘ï¼‰
    fn apply(&self, r: &[S], z: &mut [S]);

    /// èŽ·å–é¢„æ¡ä»¶å™¨åç§°
    fn name(&self) -> &'static str;

    /// æ›´æ–°é¢„æ¡ä»¶å™¨ï¼ˆçŸ©é˜µå€¼å˜åŒ–ä½†ç»“æž„ä¸å˜æ—¶ï¼‰
    ///
    /// # å‚æ•°
    ///
    /// - `matrix`: æ›´æ–°åŽçš„ç³»æ•°çŸ©é˜µ
    fn update(&mut self, matrix: &CsrMatrix<S>);
}

/// æ’ç­‰é¢„æ¡ä»¶å™¨ï¼ˆæ— é¢„æ¡ä»¶ï¼‰
///
/// M = Iï¼Œå³ z = r
#[derive(Debug, Clone, Default)]
pub struct IdentityPreconditioner;

impl IdentityPreconditioner {
    /// åˆ›å»ºæ’ç­‰é¢„æ¡ä»¶å™¨
    pub fn new() -> Self {
        Self
    }
}

impl<S: RuntimeScalar> Preconditioner<S> for IdentityPreconditioner {
    fn apply(&self, r: &[S], z: &mut [S]) {
        z.copy_from_slice(r);
    }

    fn name(&self) -> &'static str {
        "Identity"
    }

    fn update(&mut self, _matrix: &CsrMatrix<S>) {
        // æ’ç­‰é¢„æ¡ä»¶å™¨æ— éœ€æ›´æ–°
    }
}

/// Jacobi é¢„æ¡ä»¶å™¨ï¼ˆå¯¹è§’é¢„æ¡ä»¶ï¼‰
///
/// M = diag(A)ï¼Œå³ z_i = r_i / A_ii
///
/// è¿™æ˜¯æœ€ç®€å•çš„é¢„æ¡ä»¶å™¨ï¼Œè®¡ç®—å¼€é”€æžä½Žï¼Œä½†æ•ˆæžœæœ‰é™ã€‚
/// é€‚ç”¨äºŽå¯¹è§’å ä¼˜çŸ©é˜µã€‚
#[derive(Debug, Clone)]
pub struct JacobiPreconditioner<S: RuntimeScalar> {
    /// å¯¹è§’å…ƒç´ çš„å€’æ•°
    inv_diag: Vec<S>,
}

/// Legacy ç±»åž‹åˆ«åï¼Œä¿æŒå‘åŽå…¼å®¹
pub type JacobiPreconditionerF64 = JacobiPreconditioner<f64>;

impl<S: RuntimeScalar> JacobiPreconditioner<S> {
    /// ä»Ž CSR çŸ©é˜µåˆ›å»º Jacobi é¢„æ¡ä»¶å™¨
    pub fn from_matrix(matrix: &CsrMatrix<S>) -> Self {
        let n = matrix.n_rows();
        let mut inv_diag = vec![S::ONE; n];
        let threshold = S::from_config(1e-14).unwrap_or(S::MIN_POSITIVE);

        for i in 0..n {
            if let Some(diag) = matrix.diagonal_value(i) {
                if diag.abs() > threshold {
                    inv_diag[i] = S::ONE / diag;
                }
            }
        }

        Self { inv_diag }
    }

    /// ä»Ž CSR çŸ©é˜µåˆ›å»º Jacobi é¢„æ¡ä»¶å™¨ï¼ˆå¸¦å¹²å•å…ƒæ£€æµ‹ï¼‰
    ///
    /// å¯¹äºŽå¯¹è§’å…ƒç´ å°äºŽ `h_dry * 1e-6` çš„è¡Œï¼Œä½¿ç”¨å•ä½é¢„æ¡ä»¶ã€‚
    /// è¿™é¿å…äº†å¹²å•å…ƒå¯¼è‡´çš„æ•°å€¼ä¸ç¨³å®šã€‚
    pub fn from_matrix_with_dry_detection(matrix: &CsrMatrix<S>, h_dry: S) -> Self {
        let n = matrix.n_rows();
        let mut inv_diag = vec![S::ONE; n];
        let dry_threshold = h_dry * S::from_config(1e-6).unwrap_or(S::MIN_POSITIVE);
        let zero_threshold = S::from_config(1e-14).unwrap_or(S::MIN_POSITIVE);

        for i in 0..n {
            if let Some(diag) = matrix.diagonal_value(i) {
                if diag.abs() < dry_threshold {
                    // å¹²å•å…ƒï¼šä½¿ç”¨å•ä½é¢„æ¡ä»¶
                    inv_diag[i] = S::ONE;
                } else if diag.abs() > zero_threshold {
                    inv_diag[i] = S::ONE / diag;
                }
            }
        }

        Self { inv_diag }
    }

    /// ä»Žå¯¹è§’å‘é‡åˆ›å»º Jacobi é¢„æ¡ä»¶å™¨
    pub fn from_diagonal(diag: &[S]) -> Self {
        let threshold = S::from_config(1e-14).unwrap_or(S::MIN_POSITIVE);
        let inv_diag: Vec<_> = diag
            .iter()
            .map(|&d| if d.abs() > threshold { S::ONE / d } else { S::ONE })
            .collect();
        Self { inv_diag }
    }

    /// æ›´æ–°é¢„æ¡ä»¶å™¨ï¼ˆçŸ©é˜µå€¼å˜åŒ–ä½†ç»“æž„ä¸å˜æ—¶ï¼‰
    pub fn update_from_matrix(&mut self, matrix: &CsrMatrix<S>) {
        let threshold = S::from_config(1e-14).unwrap_or(S::MIN_POSITIVE);
        for i in 0..self.inv_diag.len().min(matrix.n_rows()) {
            if let Some(diag) = matrix.diagonal_value(i) {
                if diag.abs() > threshold {
                    self.inv_diag[i] = S::ONE / diag;
                }
            }
        }
    }

    /// æ›´æ–°é¢„æ¡ä»¶å™¨ï¼ˆå¸¦å¹²å•å…ƒæ£€æµ‹ï¼‰
    pub fn update_with_dry_detection(&mut self, matrix: &CsrMatrix<S>, h_dry: S) {
        let dry_threshold = h_dry * S::from_config(1e-6).unwrap_or(S::MIN_POSITIVE);
        let zero_threshold = S::from_config(1e-14).unwrap_or(S::MIN_POSITIVE);
        for i in 0..self.inv_diag.len().min(matrix.n_rows()) {
            if let Some(diag) = matrix.diagonal_value(i) {
                if diag.abs() < dry_threshold {
                    self.inv_diag[i] = S::ONE;
                } else if diag.abs() > zero_threshold {
                    self.inv_diag[i] = S::ONE / diag;
                }
            }
        }
    }

    /// èŽ·å–å¯¹è§’å…ƒç´ å€’æ•°å¼•ç”¨
    pub fn inv_diagonal(&self) -> &[S] {
        &self.inv_diag
    }
}

impl<S: RuntimeScalar> Preconditioner<S> for JacobiPreconditioner<S> {
    fn apply(&self, r: &[S], z: &mut [S]) {
        debug_assert_eq!(r.len(), z.len());
        debug_assert_eq!(r.len(), self.inv_diag.len());

        for ((zi, &ri), &inv_d) in z.iter_mut().zip(r.iter()).zip(self.inv_diag.iter()) {
            *zi = ri * inv_d;
        }
    }

    fn name(&self) -> &'static str {
        "Jacobi"
    }

    fn update(&mut self, matrix: &CsrMatrix<S>) {
        let threshold = S::from_config(1e-14).unwrap_or(S::MIN_POSITIVE);
        for i in 0..self.inv_diag.len().min(matrix.n_rows()) {
            if let Some(diag) = matrix.diagonal_value(i) {
                if diag.abs() > threshold {
                    self.inv_diag[i] = S::ONE / diag;
                }
            }
        }
    }
}

/// SSOR é¢„æ¡ä»¶å™¨ï¼ˆå¯¹ç§°é€æ¬¡è¶…æ¾å¼›ï¼‰
///
/// M = (D + Ï‰L) Dâ»Â¹ (D + Ï‰U)
///
/// å…¶ä¸­ Lã€U åˆ†åˆ«æ˜¯ A çš„ä¸¥æ ¼ä¸‹ä¸‰è§’å’Œä¸¥æ ¼ä¸Šä¸‰è§’éƒ¨åˆ†ï¼Œ
/// D æ˜¯å¯¹è§’éƒ¨åˆ†ï¼ŒÏ‰ æ˜¯æ¾å¼›å› å­ã€‚
#[derive(Debug, Clone)]
pub struct SsorPreconditioner<S: RuntimeScalar> {
    /// çŸ©é˜µå¼•ç”¨ï¼ˆç”¨äºŽå‰å‘å’ŒåŽå‘æ‰«æï¼‰
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<S>,
    /// å¯¹è§’å…ƒç´ 
    diag: Vec<S>,
    /// æ¾å¼›å› å­
    omega: S,
    /// ä¸´æ—¶å·¥ä½œå‘é‡
    #[allow(dead_code)]
    work: Vec<S>,
}

/// Legacy ç±»åž‹åˆ«åï¼Œä¿æŒå‘åŽå…¼å®¹
pub type SsorPreconditionerF64 = SsorPreconditioner<f64>;

impl<S: RuntimeScalar> SsorPreconditioner<S> {
    /// ä»Ž CSR çŸ©é˜µåˆ›å»º SSOR é¢„æ¡ä»¶å™¨
    ///
    /// # å‚æ•°
    ///
    /// - `matrix`: CSR çŸ©é˜µ
    /// - `omega`: æ¾å¼›å› å­ï¼ˆé€šå¸¸å– 1.0-1.8ï¼‰
    pub fn from_matrix(matrix: &CsrMatrix<S>, omega: S) -> Self {
        let n = matrix.n_rows();
        let diag: Vec<_> = (0..n)
            .map(|i| matrix.diagonal_value(i).unwrap_or(S::ONE))
            .collect();

        Self {
            row_ptr: matrix.row_ptr().to_vec(),
            col_idx: matrix.col_idx().to_vec(),
            values: matrix.values().to_vec(),
            diag,
            omega,
            work: vec![S::ZERO; n],
        }
    }

    /// æ›´æ–°é¢„æ¡ä»¶å™¨
    pub fn update_from_matrix(&mut self, matrix: &CsrMatrix<S>) {
        self.values.copy_from_slice(matrix.values());
        for i in 0..self.diag.len().min(matrix.n_rows()) {
            self.diag[i] = matrix.diagonal_value(i).unwrap_or(S::ONE);
        }
    }

    /// å‰å‘æ‰«æ (D + Ï‰L) y = r
    #[allow(dead_code)]
    fn forward_sweep(&self, r: &[S], y: &mut [S]) {
        let n = self.diag.len();
        for i in 0..n {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            let mut sum = r[i];
            for idx in start..end {
                let j = self.col_idx[idx];
                if j < i {
                    sum -= self.omega * self.values[idx] * y[j];
                }
            }

            y[i] = sum / self.diag[i];
        }
    }

    /// åŽå‘æ‰«æ (D + Ï‰U) z = D y
    #[allow(dead_code)]
    fn backward_sweep(&self, y: &[S], z: &mut [S]) {
        let n = self.diag.len();
        for i in (0..n).rev() {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            let mut sum = self.diag[i] * y[i];
            for idx in start..end {
                let j = self.col_idx[idx];
                if j > i {
                    sum -= self.omega * self.values[idx] * z[j];
                }
            }

            z[i] = sum / self.diag[i];
        }
    }
}

impl<S: RuntimeScalar> Preconditioner<S> for SsorPreconditioner<S> {
    fn apply(&self, r: &[S], z: &mut [S]) {
        let n = self.diag.len();

        // å‰å‘æ‰«æ: (D + Ï‰L) y = r
        for i in 0..n {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            let mut sum = r[i];
            for idx in start..end {
                let j = self.col_idx[idx];
                if j < i {
                    sum -= self.omega * self.values[idx] * z[j];
                }
            }
            z[i] = sum / self.diag[i];
        }

        // å¯¹è§’ç¼©æ”¾: y = D * (2 - Ï‰) * y
        let scale = S::TWO - self.omega;
        for i in 0..n {
            z[i] *= self.diag[i] * scale;
        }

        // åŽå‘æ‰«æ: (D + Ï‰U) x = scaled_y
        for i in (0..n).rev() {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            let mut sum = z[i];
            for idx in start..end {
                let j = self.col_idx[idx];
                if j > i {
                    sum -= self.omega * self.values[idx] * z[j];
                }
            }
            z[i] = sum / self.diag[i];
        }
    }

    fn name(&self) -> &'static str {
        "SSOR"
    }

    fn update(&mut self, matrix: &CsrMatrix<S>) {
        self.values.copy_from_slice(matrix.values());
        for i in 0..self.diag.len().min(matrix.n_rows()) {
            self.diag[i] = matrix.diagonal_value(i).unwrap_or(S::ONE);
        }
    }
}

/// ILU(0) ä¸å®Œå…¨ LU åˆ†è§£é¢„æ¡ä»¶å™¨
///
/// ä¿æŒåŽŸçŸ©é˜µç¨€ç–æ¨¡å¼çš„ä¸å®Œå…¨ LU åˆ†è§£ã€‚
/// æ¯” Jacobi æ›´å¼ºä½†è®¡ç®—å¼€é”€ä¹Ÿæ›´å¤§ã€‚
#[derive(Debug, Clone)]
pub struct Ilu0Preconditioner<S: RuntimeScalar> {
    /// çŸ©é˜µç»´åº¦
    n: usize,
    /// è¡ŒæŒ‡é’ˆ
    row_ptr: Vec<usize>,
    /// åˆ—ç´¢å¼•
    col_idx: Vec<usize>,
    /// LU åˆ†è§£åŽçš„å€¼ï¼ˆL å’Œ U å…±ç”¨å­˜å‚¨ï¼‰
    lu_values: Vec<S>,
    /// å¯¹è§’å…ƒä½ç½®ç´¢å¼•
    diag_ptr: Vec<usize>,
}

/// Legacy ç±»åž‹åˆ«åï¼Œä¿æŒå‘åŽå…¼å®¹
pub type Ilu0PreconditionerF64 = Ilu0Preconditioner<f64>;

impl<S: RuntimeScalar> Ilu0Preconditioner<S> {
    /// ä»Ž CSR çŸ©é˜µåˆ›å»º ILU(0) é¢„æ¡ä»¶å™¨
    ///
    /// # å‚æ•°
    ///
    /// - `matrix`: ç³»æ•°çŸ©é˜µ
    pub fn new(matrix: &CsrMatrix<S>) -> Self {
        let n = matrix.n_rows();
        let mut lu_values = matrix.values().to_vec();
        let row_ptr = matrix.row_ptr().to_vec();
        let col_idx = matrix.col_idx().to_vec();

        // æŸ¥æ‰¾å¯¹è§’å…ƒä½ç½®
        let mut diag_ptr = vec![0usize; n];
        for i in 0..n {
            for k in row_ptr[i]..row_ptr[i + 1] {
                if col_idx[k] == i {
                    diag_ptr[i] = k;
                    break;
                }
            }
        }

        // æ‰§è¡Œ ILU(0) åˆ†è§£
        Self::factorize(&row_ptr, &col_idx, &mut lu_values, &diag_ptr, n);

        Self {
            n,
            row_ptr,
            col_idx,
            lu_values,
            diag_ptr,
        }
    }

    /// æ‰§è¡Œ ILU(0) åˆ†è§£
    ///
    /// åŽŸåœ°ä¿®æ”¹ lu æ•°ç»„ï¼Œä½¿å¾— L çš„ä¸¥æ ¼ä¸‹ä¸‰è§’éƒ¨åˆ†å’Œ U çš„ä¸Šä¸‰è§’éƒ¨åˆ†ï¼ˆå«å¯¹è§’ï¼‰
    /// å­˜å‚¨åœ¨åŒä¸€æ•°ç»„ä¸­ã€‚
    ///
    /// ä½¿ç”¨ä¸»å…ƒæ­£åˆ™åŒ–å’Œå¢žé•¿å› å­é™åˆ¶æé«˜æ•°å€¼ç¨³å®šæ€§ã€‚
    fn factorize(
        row_ptr: &[usize],
        col_idx: &[usize],
        lu: &mut [S],
        diag_ptr: &[usize],
        n: usize,
    ) {
        // æ•°å€¼ç¨³å®šæ€§å‚æ•°
        let pivot_tol = S::from_config(1e-10).unwrap_or(S::MIN_POSITIVE);
        let growth_limit = S::from_config(1e3).unwrap_or(S::from_config(1000.0).unwrap_or(S::MAX));

        for i in 1..n {
            // éåŽ†ç¬¬ i è¡Œçš„ä¸‹ä¸‰è§’éƒ¨åˆ† (j < i)
            for k_idx in row_ptr[i]..row_ptr[i + 1] {
                let k = col_idx[k_idx];
                if k >= i {
                    break;
                }

                // ä¸»å…ƒæ­£åˆ™åŒ–ï¼šé¿å…é™¤é›¶
                let mut diag_k = lu[diag_ptr[k]];
                if diag_k.abs() < pivot_tol {
                    diag_k = if diag_k >= S::ZERO { pivot_tol } else { -pivot_tol };
                    if diag_k == S::ZERO {
                        diag_k = pivot_tol;
                    }
                    lu[diag_ptr[k]] = diag_k;
                }

                // è®¡ç®—å› å­å¹¶é™åˆ¶å¢žé•¿
                let mut factor = lu[k_idx] / diag_k;
                factor = factor.clamp(-growth_limit, growth_limit);
                lu[k_idx] = factor;

                // æ›´æ–°ç¬¬ i è¡Œçš„å…¶ä½™å…ƒç´ 
                for j_idx in (k_idx + 1)..row_ptr[i + 1] {
                    let j = col_idx[j_idx];
                    // æŸ¥æ‰¾ A[k,j]
                    for m_idx in row_ptr[k]..row_ptr[k + 1] {
                        if col_idx[m_idx] == j {
                            let update = factor * lu[m_idx];
                            // é™åˆ¶æ›´æ–°å¹…åº¦
                            let limited_update = update.clamp(-growth_limit, growth_limit);
                            lu[j_idx] -= limited_update;
                            break;
                        }
                    }
                }
            }
        }
    }

    /// å‰å‘æ›¿æ¢: L * y = r
    fn forward_solve(&self, r: &[S], y: &mut [S]) {
        y.copy_from_slice(r);

        for i in 0..self.n {
            for k_idx in self.row_ptr[i]..self.diag_ptr[i] {
                let j = self.col_idx[k_idx];
                y[i] -= self.lu_values[k_idx] * y[j];
            }
        }
    }

    /// åŽå‘æ›¿æ¢: U * z = y
    fn backward_solve(&self, y: &[S], z: &mut [S]) {
        z.copy_from_slice(y);
        let threshold = S::from_config(1e-14).unwrap_or(S::MIN_POSITIVE);

        for i in (0..self.n).rev() {
            for k_idx in (self.diag_ptr[i] + 1)..self.row_ptr[i + 1] {
                let j = self.col_idx[k_idx];
                z[i] -= self.lu_values[k_idx] * z[j];
            }

            let diag = self.lu_values[self.diag_ptr[i]];
            if diag.abs() > threshold {
                z[i] /= diag;
            }
        }
    }
}

impl<S: RuntimeScalar> Preconditioner<S> for Ilu0Preconditioner<S> {
    fn apply(&self, r: &[S], z: &mut [S]) {
        // è§£ L * U * z = r
        // åˆ†ä¸¤æ­¥: L * y = r, ç„¶åŽ U * z = y
        let mut y = vec![S::ZERO; self.n];
        self.forward_solve(r, &mut y);
        self.backward_solve(&y, z);
    }

    fn name(&self) -> &'static str {
        "ILU(0)"
    }

    fn update(&mut self, matrix: &CsrMatrix<S>) {
        // é‡æ–°å¤åˆ¶çŸ©é˜µå€¼å¹¶é‡æ–°åˆ†è§£
        self.lu_values.copy_from_slice(matrix.values());
        Self::factorize(
            &self.row_ptr,
            &self.col_idx,
            &mut self.lu_values,
            &self.diag_ptr,
            self.n,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::linear_algebra::csr::CsrBuilder;

    // 测试用类型别名
    type S = f64;

    fn create_test_matrix() -> CsrMatrix<S> {
        let mut builder = CsrBuilder::<S>::new_square(3);
        builder.set(0, 0, 4.0);
        builder.set(0, 1, -1.0);
        builder.set(1, 0, -1.0);
        builder.set(1, 1, 4.0);
        builder.set(1, 2, -1.0);
        builder.set(2, 1, -1.0);
        builder.set(2, 2, 4.0);
        builder.build()
    }

    #[test]
    fn test_identity_preconditioner() {
        let precond = IdentityPreconditioner::new();
        let r = vec![1.0, 2.0, 3.0];
        let mut z = vec![0.0; 3];

        Preconditioner::<S>::apply(&precond, &r, &mut z);
        assert_eq!(z, r);
        assert_eq!(Preconditioner::<S>::name(&precond), "Identity");
    }

    #[test]
    fn test_jacobi_preconditioner() {
        let matrix = create_test_matrix();
        let precond = JacobiPreconditioner::<S>::from_matrix(&matrix);

        let r = vec![4.0, 8.0, 12.0];
        let mut z = vec![0.0; 3];

        precond.apply(&r, &mut z);

        // z[i] = r[i] / diag[i] = r[i] / 4.0
        assert!((z[0] - 1.0).abs() < 1e-14);
        assert!((z[1] - 2.0).abs() < 1e-14);
        assert!((z[2] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_jacobi_from_diagonal() {
        let diag = vec![2.0, 4.0, 8.0];
        let precond = JacobiPreconditioner::<S>::from_diagonal(&diag);

        let r = vec![2.0, 4.0, 8.0];
        let mut z = vec![0.0; 3];

        precond.apply(&r, &mut z);

        assert!((z[0] - 1.0).abs() < 1e-14);
        assert!((z[1] - 1.0).abs() < 1e-14);
        assert!((z[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_jacobi_zero_diagonal() {
        let diag = vec![2.0, 0.0, 4.0]; // 中间为零
        let precond = JacobiPreconditioner::<S>::from_diagonal(&diag);

        let r = vec![2.0, 3.0, 4.0];
        let mut z = vec![0.0; 3];

        precond.apply(&r, &mut z);

        // 零对角用 1.0 替代
        assert!((z[0] - 1.0).abs() < 1e-14);
        assert!((z[1] - 3.0).abs() < 1e-14);
        assert!((z[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_ssor_preconditioner() {
        let matrix = create_test_matrix();
        let precond = SsorPreconditioner::<S>::from_matrix(&matrix, 1.0);

        let r = vec![1.0, 1.0, 1.0];
        let mut z = vec![0.0; 3];

        precond.apply(&r, &mut z);

        // SSOR 应该产生比 Jacobi 更好的结果
        // 这里只检查基本正确性
        assert!(!z[0].is_nan());
        assert!(!z[1].is_nan());
        assert!(!z[2].is_nan());
    }
}
