//! 高精度横轴墨卡托投影（Karney 2011 算法）
//!
//! 实现基于 Krüger 级数的横轴墨卡托投影，精度可达亚毫米级。
//!
//! # 参考文献
//!
//! Karney, C. F. F. (2011). "Transverse Mercator with an accuracy of a few nanometers".
//! Journal of Geodesy, 85(8), 475-485.
//!
//! # 算法特点
//!
//! - 使用 6 阶 Krüger 级数展开
//! - 在全球范围内（除极点附近）精度达到纳米级
//! - 支持任意椭球体参数

use super::math_utils::{sincosd, tauf, taupf};
use super::traits::TransverseMercatorParams;
use mh_foundation::error::{MhError, MhResult};
use num_complex::Complex64;
use std::f64::consts::PI;

// ============================================================================
// 系数表 (从 GeographicLib 复制)
// ============================================================================

/// 6阶 alpha 系数 (正向投影)
#[cfg(not(feature = "order-8"))]
const ALPHA_COEFFS: &[&[f64]] = &[
    // alp[1]/n^1
    &[31564.0, -66675.0, 34440.0, 47250.0, -100800.0, 75600.0, 151200.0],
    // alp[2]/n^2
    &[-1983433.0, 863232.0, 748608.0, -1161216.0, 524160.0, 1935360.0],
    // alp[3]/n^3
    &[670412.0, 406647.0, -533952.0, 184464.0, 725760.0],
    // alp[4]/n^4
    &[6601661.0, -7732800.0, 2230245.0, 7257600.0],
    // alp[5]/n^5
    &[-13675556.0, 3438171.0, 7983360.0],
    // alp[6]/n^6
    &[212378941.0, 319334400.0],
];

/// 6阶 beta 系数 (逆向投影)
#[cfg(not(feature = "order-8"))]
const BETA_COEFFS: &[&[f64]] = &[
    // bet[1]/n^1
    &[384796.0, -382725.0, -6720.0, 932400.0, -1612800.0, 1209600.0, 2419200.0],
    // bet[2]/n^2
    &[-1118711.0, 1695744.0, -1174656.0, 258048.0, 80640.0, 3870720.0],
    // bet[3]/n^3
    &[22276.0, -16929.0, -15984.0, 12852.0, 362880.0],
    // bet[4]/n^4
    &[-830251.0, -158400.0, 197865.0, 7257600.0],
    // bet[5]/n^5
    &[-435388.0, 453717.0, 15966720.0],
    // bet[6]/n^6
    &[20648693.0, 638668800.0],
];

/// 8阶 alpha 系数 (正向投影)
#[cfg(feature = "order-8")]
const ALPHA_COEFFS: &[&[f64]] = &[
    &[-75900428.0, 37884525.0, 42422016.0, -89611200.0, 46287360.0, 
      63504000.0, -135475200.0, 101606400.0, 203212800.0],
    &[148003883.0, 83274912.0, -178508970.0, 77690880.0, 67374720.0, 
      -104509440.0, 47174400.0, 174182400.0],
    &[318729724.0, -738126169.0, 294981280.0, 178924680.0, -234938880.0, 
      81164160.0, 319334400.0],
    &[-40176129013.0, 14967552000.0, 6971354016.0, -8165836800.0, 
      2355138720.0, 7664025600.0],
    &[10421654396.0, 3997835751.0, -4266773472.0, 1072709352.0, 2490808320.0],
    &[175214326799.0, -171950693600.0, 38652967262.0, 58118860800.0],
    &[-67039739596.0, 13700311101.0, 12454041600.0],
    &[1424729850961.0, 743921418240.0],
];

/// 8阶 beta 系数 (逆向投影)
#[cfg(feature = "order-8")]
const BETA_COEFFS: &[&[f64]] = &[
    &[31777436.0, -37845269.0, 43097152.0, -42865200.0, -752640.0, 
      104428800.0, -180633600.0, 135475200.0, 270950400.0],
    &[24749483.0, 14930208.0, -100683990.0, 152616960.0, -105719040.0, 
      23224320.0, 7257600.0, 348364800.0],
    &[-232468668.0, 101880889.0, 39205760.0, -29795040.0, -28131840.0, 
      22619520.0, 638668800.0],
    &[324154477.0, 1433121792.0, -876745056.0, -167270400.0, 208945440.0, 
      7664025600.0],
    &[457888660.0, -312227409.0, -67920528.0, 70779852.0, 2490808320.0],
    &[-19841813847.0, -3665348512.0, 3758062126.0, 116237721600.0],
    &[-1989295244.0, 1979471673.0, 49816166400.0],
    &[191773887257.0, 3719607091200.0],
];

/// b1 系数 (用于计算 rectifying radius)
const B1_COEFFS: &[f64] = &[1.0, 4.0, 64.0, 256.0, 256.0];

/// 计算级数阶数
#[cfg(not(feature = "order-8"))]
const MAX_ORDER: usize = 6;

#[cfg(feature = "order-8")]
const MAX_ORDER: usize = 8;

// ============================================================================
// 预计算结构体
// ============================================================================

/// 预计算的投影参数
#[derive(Debug, Clone)]
pub struct TMComputed {
    /// 椭球 e^2
    e2: f64,
    /// signed eccentricity = sign(e^2) * sqrt(|e^2|)
    es: f64,
    /// 1 - e^2
    e2m: f64,
    /// b1 = rectifying radius ratio
    b1: f64,
    /// a1 = a * b1
    a1: f64,
    /// 预计算的 alpha 系数
    alp: [f64; MAX_ORDER],
    /// 预计算的 beta 系数
    bet: [f64; MAX_ORDER],
    /// 比例因子
    k0: f64,
    /// 假东
    false_easting: f64,
    /// 假北
    false_northing: f64,
}

impl TMComputed {
    /// 从参数创建预计算结构
    pub fn new(params: &TransverseMercatorParams) -> Self {
        let a = params.ellipsoid.a;
        let f = params.ellipsoid.f;
        let e2 = f * (2.0 - f);
        let es = if f < 0.0 { -1.0 } else { 1.0 } * e2.abs().sqrt();
        let e2m = 1.0 - e2;
        let n = f / (2.0 - f);

        // 计算 b1
        let n2 = n * n;
        let b1 = super::math_utils::polyval(&B1_COEFFS[..B1_COEFFS.len() - 1], n2)
            / (B1_COEFFS[B1_COEFFS.len() - 1] * (1.0 + n));
        let a1 = b1 * a;

        // 计算 alpha 和 beta 系数
        let mut alp = [0.0; MAX_ORDER];
        let mut bet = [0.0; MAX_ORDER];
        let mut d = n;
        
        for l in 0..MAX_ORDER {
            let coeffs_a = ALPHA_COEFFS[l];
            let coeffs_b = BETA_COEFFS[l];
            let m = coeffs_a.len() - 1;
            
            alp[l] = d * super::math_utils::polyval(&coeffs_a[..m], n) / coeffs_a[m];
            bet[l] = d * super::math_utils::polyval(&coeffs_b[..m], n) / coeffs_b[m];
            d *= n;
        }

        Self {
            e2,
            es,
            e2m,
            b1,
            a1,
            alp,
            bet,
            k0: params.scale_factor,
            false_easting: params.false_easting,
            false_northing: params.false_northing,
        }
    }
}

#[derive(Debug, Clone)]
struct ForwardResult {
    x: f64,
    y: f64,
    gamma: f64,
    k: f64,
}

// ============================================================================
// 核心算法
// ============================================================================

fn forward_internal(
    params: &TransverseMercatorParams,
    lon: f64,
    lat: f64,
) -> MhResult<ForwardResult> {
    // 验证纬度
    if !(-90.0..=90.0).contains(&lat) {
        return Err(MhError::InvalidInput {
            message: format!("Latitude {lat} out of range [-90, 90]"),
        });
    }

    let tm = TMComputed::new(params);

    // 处理经度差
    let lon_diff = super::math_utils::ang_diff(params.central_meridian, lon);

    // 处理符号和象限
    let latsign = if lat.is_sign_negative() { -1.0 } else { 1.0 };
    let lonsign = if lon_diff.is_sign_negative() { -1.0 } else { 1.0 };
    let lat = lat.abs();
    let lon_diff = lon_diff.abs();

    let backside = lon_diff > 90.0;
    let lon_diff = if backside { 180.0 - lon_diff } else { lon_diff };

    // sin/cos 精确计算
    let (sphi, cphi) = sincosd(lat);
    let (slam, clam) = sincosd(lon_diff);

    // 计算 xi', eta'
    let (xip, etap, mut gamma, mut k);

    if lat == 90.0 {
        xip = PI / 2.0;
        etap = 0.0;
        gamma = lon_diff;
        k = f64::midpoint(1.0, tm.e2m.sqrt()) / tm.e2m.sqrt().sqrt();
    } else {
        let tau = sphi / cphi;
        let taup = taupf(tau, tm.es);

        xip = taup.atan2(clam);
        etap = (slam / (taup * taup + clam * clam).sqrt()).asinh();

        // 收敛角和比例因子 (Gauss-Schreiber)
        gamma = (slam * taup).atan2(clam * (1.0 + taup * taup).sqrt()).to_degrees();
        k = (tm.e2m + tm.e2 * cphi * cphi).sqrt() * (1.0 + tau * tau).sqrt()
            / (taup * taup + clam * clam).sqrt();
    }

    // Clenshaw 算法求和
    let c0 = (2.0 * xip).cos();
    let ch0 = (2.0 * etap).cosh();
    let s0 = (2.0 * xip).sin();
    let sh0 = (2.0 * etap).sinh();

    let a = Complex64::new(2.0 * c0 * ch0, -2.0 * s0 * sh0);

    let mut y0 = Complex64::new(0.0, 0.0);
    let mut y1 = Complex64::new(0.0, 0.0);
    let mut z0 = Complex64::new(0.0, 0.0);
    let mut z1 = Complex64::new(0.0, 0.0);

    // 从高阶到低阶
    for j in (0..MAX_ORDER).rev() {
        let tmp_y = y0;
        let tmp_z = z0;
        y0 = a * y0 - y1 + tm.alp[j];
        z0 = a * z0 - z1 + (2 * (j + 1)) as f64 * tm.alp[j];
        y1 = tmp_y;
        z1 = tmp_z;
    }

    // 最终求和
    let sin_zeta = Complex64::new(s0 * ch0, c0 * sh0);
    let y_result = Complex64::new(xip, etap) + sin_zeta * y0;
    let z_result = Complex64::new(1.0, 0.0) - z1 + (a / 2.0) * z0;

    let xi = y_result.re;
    let eta = y_result.im;

    // 应用收敛角和比例因子修正
    gamma -= z_result.im.atan2(z_result.re).to_degrees();
    k = tm.k0 * tm.b1 * k * z_result.norm();

    // 应用符号和假东假北
    let mut y_out = tm.a1 * tm.k0 * xi * latsign;
    let x_out = tm.a1 * tm.k0 * eta * lonsign;

    if backside {
        y_out = tm.a1 * tm.k0 * (PI - xi) * latsign;
    }

    Ok(ForwardResult {
        x: x_out + tm.false_easting,
        y: y_out + tm.false_northing,
        gamma,
        k,
    })
}

/// 正向投影（使用 Clenshaw 算法）
pub fn forward(params: &TransverseMercatorParams, lon: f64, lat: f64) -> MhResult<(f64, f64)> {
    let result = forward_internal(params, lon, lat)?;
    Ok((result.x, result.y))
}

/// 逆向投影（使用 Clenshaw 算法）
pub fn inverse(params: &TransverseMercatorParams, x: f64, y: f64) -> MhResult<(f64, f64)> {
    let tm = TMComputed::new(params);

    // 移除假东假北
    let xi = (y - tm.false_northing) / (tm.a1 * tm.k0);
    let eta = (x - tm.false_easting) / (tm.a1 * tm.k0);

    // 处理符号
    let xisign = if xi.is_sign_negative() { -1.0 } else { 1.0 };
    let etasign = if eta.is_sign_negative() { -1.0 } else { 1.0 };
    let xi = xi.abs();
    let eta = eta.abs();

    let backside = xi > PI / 2.0;
    let xi = if backside { PI - xi } else { xi };

    // Clenshaw 算法求 xi', eta'
    let c0 = (2.0 * xi).cos();
    let ch0 = (2.0 * eta).cosh();
    let s0 = (2.0 * xi).sin();
    let sh0 = (2.0 * eta).sinh();

    let a = Complex64::new(2.0 * c0 * ch0, -2.0 * s0 * sh0);

    let mut y0 = Complex64::new(0.0, 0.0);
    let mut y1 = Complex64::new(0.0, 0.0);

    for j in (0..MAX_ORDER).rev() {
        let tmp = y0;
        y0 = a * y0 - y1 - tm.bet[j];
        y1 = tmp;
    }

    let sin_zeta = Complex64::new(s0 * ch0, c0 * sh0);
    let y_result = Complex64::new(xi, eta) + sin_zeta * y0;

    let xip = y_result.re;
    let etap = y_result.im;

    // 从 xi', eta' 恢复经纬度
    let s = etap.sinh();
    let c = xip.cos().max(0.0);
    let r = (s * s + c * c).sqrt();

    let (lon, lat);
    if r == 0.0 {
        lon = 0.0;
        lat = 90.0;
    } else {
        lon = s.atan2(c).to_degrees();
        let sxip = xip.sin();
        let tau = tauf(sxip / r, tm.es);
        lat = tau.atan().to_degrees();
    }

    // 应用符号
    let lat = lat * xisign;
    let mut lon = lon * etasign;
    
    if backside {
        lon = 180.0 - lon;
    }
    
    // 规范化并添加中央子午线
    lon = super::math_utils::ang_normalize(lon + params.central_meridian);

    Ok((lon, lat))
}

/// 计算比例因子
#[must_use]
pub fn scale_factor_at(params: &TransverseMercatorParams, lon: f64, lat: f64) -> f64 {
    forward_internal(params, lon, lat)
        .map(|res| res.k)
        .unwrap_or(f64::NAN)
}

/// 计算收敛角
#[must_use]
pub fn convergence_angle(params: &TransverseMercatorParams, lon: f64, lat: f64) -> f64 {
    forward_internal(params, lon, lat)
        .map(|res| res.gamma.to_radians())
        .unwrap_or(f64::NAN)
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn utm_51n() -> TransverseMercatorParams {
        TransverseMercatorParams::utm(51, true)
    }

    #[test]
    fn test_forward_central_meridian() {
        let params = utm_51n();
        let (x, _y) = forward(&params, 123.0, 40.0).expect("forward failed");
        assert!((x - 500_000.0).abs() < 1.0, "x = {x}");
    }

    #[test]
    fn test_roundtrip_high_precision() {
        let params = utm_51n();
        
        let test_cases = [
            (121.0, 30.0),
            (123.0, 40.0),
            (125.0, 50.0),
            (120.0, 0.0),
            (126.0, 84.0),
        ];

        for (lon, lat) in test_cases {
            let (x, y) = forward(&params, lon, lat).expect("forward");
            let (lon2, lat2) = inverse(&params, x, y).expect("inverse");

            let err_lon = (lon - lon2).abs();
            let err_lat = (lat - lat2).abs();

            assert!(
                err_lon < 1e-11 && err_lat < 1e-11,
                "({lon}, {lat}): err_lon={err_lon:.2e}, err_lat={err_lat:.2e}"
            );
        }
    }

    /// EPSG 标准验证（目标精度：0.1mm）
    #[test]
    fn test_epsg_validation_submillimeter() {
        let params = utm_51n();

        const TEST_CASES: &[(f64, f64, f64, f64)] = &[
            // Verified against PROJ 9 (pyproj 3.7.2, EPSG:32651)
            (121.880356, 29.887703, 391_888.063_726_413, 3_306_868.456_385_104),
            (121.430427, 28.637151, 346582.4108433011, 3168793.409367069),
            (121.880772, 31.491324, 393700.3650201835, 3484597.440826551),
            (122.625275, 30.246954, 463948.3333072607, 3346209.757229396),
        ];

        const TOLERANCE: f64 = 0.0000001; // 0.0001mm

        println!("\n=== EPSG 亚微米精度验证 ===");
        println!("{:<25} {:>15} {:>15}", "输入(lon,lat)", "误差X(mm)", "误差Y(mm)");
        println!("{}", "-".repeat(60));

        let mut max_err = 0.0_f64;

        for (lon, lat, exp_x, exp_y) in TEST_CASES {
            let (x, y) = forward(&params, *lon, *lat).expect("forward");
            let err_x = (x - exp_x).abs();
            let err_y = (y - exp_y).abs();
            max_err = max_err.max(err_x).max(err_y);

            println!(
                "({:>10.6}, {:>9.6}) {:>15.6} {:>15.6}",
                lon, lat, err_x * 1000.0, err_y * 1000.0
            );
        }

        println!("{}", "-".repeat(60));
        println!("最大误差: {:.6} mm", max_err * 1000.0);
        println!("目标精度: {} mm", TOLERANCE * 1000.0);
        println!("结果: {}", if max_err < TOLERANCE { "✓ 达标" } else { "✗ 未达标" });

        assert!(
            max_err < TOLERANCE,
            "误差 {:.6}mm 超过阈值 {}mm",
            max_err * 1000.0,
            TOLERANCE * 1000.0
        );
    }

    #[test]
    fn test_scale_factor() {
        let params = TransverseMercatorParams::utm(50, true);
        
        // 中央子午线处
        let k = scale_factor_at(&params, 117.0, 40.0);
        assert!((k - 0.9996).abs() < 0.0001, "k = {k}");
    }
}