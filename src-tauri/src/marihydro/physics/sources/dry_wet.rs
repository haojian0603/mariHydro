//! 干湿边界判据

#[inline(always)]
pub fn is_wet(h: f64, h_min: f64, zb: f64, sea_level: f64) -> bool {
    let eta_threshold = sea_level + h_min * 1e-3;
    h > h_min && (h + zb) > eta_threshold
}

#[inline(always)]
pub fn is_wet_simple(h: f64, h_min: f64, zb: f64) -> bool {
    is_wet(h, h_min, zb, 0.0)
}

#[inline(always)]
pub fn enforce_dry_velocity(h: f64, u: f64, v: f64, h_min: f64) -> (f64, f64) {
    if h < h_min {
        (0.0, 0.0)
    } else {
        (u, v)
    }
}

#[inline(always)]
pub fn correct_interface_depth(h: f64, h_min: f64) -> f64 {
    if h < 0.0 {
        0.0
    } else if h < h_min * 1e-12 {
        h_min * 1e-12
    } else {
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_wet_basic() {
        let h_min = 0.05;

        assert!(is_wet_simple(1.0, h_min, -10.0));
        assert!(!is_wet_simple(0.01, h_min, -10.0));
        assert!(!is_wet_simple(0.1, h_min, 5.0));
    }

    #[test]
    fn test_enforce_dry_velocity() {
        let h_min = 0.05;

        let (u, v) = enforce_dry_velocity(0.01, 1.5, 2.3, h_min);
        assert_eq!(u, 0.0);
        assert_eq!(v, 0.0);

        let (u, v) = enforce_dry_velocity(1.0, 1.5, 2.3, h_min);
        assert_eq!(u, 1.5);
        assert_eq!(v, 2.3);
    }

    #[test]
    fn test_correct_interface_depth() {
        let h_min = 0.05;

        assert_eq!(correct_interface_depth(-0.1, h_min), 0.0);

        let h_corrected = correct_interface_depth(1e-15, h_min);
        assert!(h_corrected > 0.0);

        assert_eq!(correct_interface_depth(1.0, h_min), 1.0);
    }
}
