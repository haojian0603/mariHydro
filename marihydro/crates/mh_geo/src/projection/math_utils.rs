//! High-precision math utilities ported from `GeographicLib`

/// 误差补偿求和 (Kahan summation)
#[inline]
pub fn sum_exact(u: f64, v: f64) -> (f64, f64) {
    let s = u + v;
    let up = s - v;
    let vpp = s - up;
    let up = up - u;
    let vpp = vpp - v;
    let t = -(up + vpp);
    (s, t)
}

/// e * atanh(e * x) 的稳定计算
#[inline]
pub fn eatanhe(x: f64, es: f64) -> f64 {
    if es > 0.0 {
        es * (es * x).atanh()
    } else if es < 0.0 {
        -es * (-es * x).atan()
    } else {
        0.0
    }
}

/// tan(φ) → tan(φ') 共形纬度正向转换 (Karney Eq. 7-9)
#[inline]
pub fn taupf(tau: f64, es: f64) -> f64 {
    let tau1 = (1.0 + tau * tau).sqrt(); // hypot(1, tau)
    let sig = eatanhe(tau / tau1, es).sinh();
    (1.0 + sig * sig).sqrt() * tau - sig * tau1
}

/// tan(φ') → tan(φ) 共形纬度逆向转换 (Karney Eq. 19-21)
/// 使用 Newton 迭代求解
pub fn tauf(taup: f64, es: f64) -> f64 {
    const MAX_ITER: usize = 8;
    // Precompute instead of calling sqrt in a const context.
    const TOL: f64 = 1.4901161193847656e-8; // sqrt(f64::EPSILON)

    let e2m = 1.0 - es * es;
    // 初始估计
    let mut tau = taup / e2m.sqrt();
    let stol = TOL * taup.abs().max(1.0);

    for _ in 0..MAX_ITER {
        let taupa = taupf(tau, es);
        let dtau = (taup - taupa)
            * (1.0 + e2m * tau * tau)
            / (e2m * (1.0 + tau * tau).sqrt() * (1.0 + taupa * taupa).sqrt());
        tau += dtau;
        if dtau.abs() < stol {
            break;
        }
    }
    tau
}

/// 角度归一化到 [-180, 180]
#[inline]
pub fn ang_normalize(x: f64) -> f64 {
    let mut x = x % 360.0;
    if x < -180.0 {
        x += 360.0;
    }
    if x >= 180.0 {
        x -= 360.0;
    }
    x
}

/// 角度差值（精确计算）
pub fn ang_diff(x: f64, y: f64) -> f64 {
    let (d, t) = sum_exact(ang_normalize(-x), ang_normalize(y));
    ang_normalize(d) + t
}

/// sin 和 cos 的度数版本（精确处理特殊角度）
pub fn sincosd(x: f64) -> (f64, f64) {
    let mut r = x % 360.0;
    if r < 0.0 {
        r += 360.0;
    }
    let q = (r / 90.0 + 0.5).floor() as i32;
    r -= 90.0 * f64::from(q);
    let r = r.to_radians();
    let (s, c) = r.sin_cos();
    
    match q & 3 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// 多项式求值 (Horner's method)
#[inline]
pub fn polyval(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * x + c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_taupf_tauf_roundtrip() {
        let es = 0.0818191908426; // WGS84
        for lat in [-85.0_f64, -45.0, 0.0, 45.0, 85.0] {
            let tau = lat.to_radians().tan();
            let taup = taupf(tau, es);
            let tau2 = tauf(taup, es);
            assert!(
                (tau - tau2).abs() < 1e-14,
                "lat={lat}: tau={tau}, tau2={tau2}"
            );
        }
    }

    #[test]
    fn test_sincosd_exact() {
        let (s, c) = sincosd(90.0);
        assert!((s - 1.0).abs() < 1e-15);
        assert!(c.abs() < 1e-15);

        let (s, c) = sincosd(180.0);
        assert!(s.abs() < 1e-15);
        assert!((c + 1.0).abs() < 1e-15);
    }
}