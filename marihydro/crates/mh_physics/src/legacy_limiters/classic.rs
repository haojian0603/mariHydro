// 限制器函数
// ============================================================================

/// 限制器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyLimiterType {
    /// 无限制（一阶迎风）
    None,
    /// Minmod 限制器（最耗散）
    Minmod,
    /// Superbee 限制器（最不耗散）
    Superbee,
    /// Van Leer 限制器（平滑）
    VanLeer,
    /// Van Albada 限制器（可微）
    VanAlbada,
    /// Koren 限制器（三阶）
    Koren,
    /// MC (Monotonized Central) 限制器
    Mc,
    /// Venkatakrishnan 限制器（非结构网格）
    Venkatakrishnan,
    /// Barth-Jespersen 限制器
    BarthJespersen,
}

#[deprecated(note = "Use crate::types::LimiterType for configuration and crate::numerics::limiter for the engine path.")]
pub use LegacyLimiterType as LimiterType;

impl Default for LegacyLimiterType {
    fn default() -> Self {
        Self::VanLeer
    }
}

/// 符号函数
#[inline]
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Minmod 函数
#[inline]
pub fn minmod(a: f64, b: f64) -> f64 {
    if a * b > 0.0 {
        sign(a) * a.abs().min(b.abs())
    } else {
        0.0
    }
}

/// 三参数 Minmod
#[inline]
pub fn minmod3(a: f64, b: f64, c: f64) -> f64 {
    if a * b > 0.0 && b * c > 0.0 {
        sign(a) * a.abs().min(b.abs()).min(c.abs())
    } else {
        0.0
    }
}

/// Maxmod 函数
#[inline]
pub fn maxmod(a: f64, b: f64) -> f64 {
    if a * b > 0.0 {
        sign(a) * a.abs().max(b.abs())
    } else {
        0.0
    }
}

// ============================================================================
// 经典限制器实现
// ============================================================================

/// Minmod 限制器函数
///
/// φ(r) = max(0, min(1, r))
#[inline]
pub fn limiter_minmod(r: f64) -> f64 {
    0.0f64.max(1.0f64.min(r))
}

/// Superbee 限制器函数
///
/// φ(r) = max(0, min(1, 2r), min(2, r))
#[inline]
pub fn limiter_superbee(r: f64) -> f64 {
    0.0f64.max((1.0f64.min(2.0 * r)).max(2.0f64.min(r)))
}

/// Van Leer 限制器函数
///
/// φ(r) = (r + |r|) / (1 + |r|)
#[inline]
pub fn limiter_van_leer(r: f64) -> f64 {
    (r + r.abs()) / (1.0 + r.abs())
}

/// Van Albada 限制器函数
///
/// φ(r) = (r² + r) / (r² + 1)
#[inline]
pub fn limiter_van_albada(r: f64) -> f64 {
    if r <= 0.0 {
        0.0
    } else {
        (r * r + r) / (r * r + 1.0)
    }
}

/// Koren 限制器函数（三阶）
///
/// φ(r) = max(0, min(2r, (2 + r)/3, 2))
#[inline]
pub fn limiter_koren(r: f64) -> f64 {
    0.0f64.max((2.0 * r).min((2.0 + r) / 3.0).min(2.0))
}

/// MC (Monotonized Central) 限制器函数
///
/// φ(r) = max(0, min(2, 2r, (1 + r)/2))
#[inline]
pub fn limiter_mc(r: f64) -> f64 {
    0.0f64.max(2.0f64.min((2.0 * r).min((1.0 + r) / 2.0)))
}

/// 根据类型选择限制器
#[inline]
pub fn apply_limiter(limiter_type: LegacyLimiterType, r: f64) -> f64 {
    match limiter_type {
        LegacyLimiterType::None => 1.0,
        LegacyLimiterType::Minmod => limiter_minmod(r),
        LegacyLimiterType::Superbee => limiter_superbee(r),
        LegacyLimiterType::VanLeer => limiter_van_leer(r),
        LegacyLimiterType::VanAlbada => limiter_van_albada(r),
        LegacyLimiterType::Koren => limiter_koren(r),
        LegacyLimiterType::Mc => limiter_mc(r),
        LegacyLimiterType::Venkatakrishnan | LegacyLimiterType::BarthJespersen => {
            // 这些需要额外参数，使用默认 Van Leer
            limiter_van_leer(r)
        }
    }
}

// ============================================================================
