// crates/mh_physics/src/sediment/bed_load.rs

//! 推移质输沙公式
//! 包含 Meyer-Peter-Müller, Van Rijn 等经典公式

use super::properties::SedimentProperties;
use serde::{Deserialize, Serialize};

/// 重力加速度
const G: f64 = 9.81;

/// 推移质输沙公式接口
pub trait BedLoadFormula: Send + Sync {
    /// 公式名称
    fn name(&self) -> &str;
    
    /// 计算单宽推移质输沙率 [m²/s]
    /// 
    /// # Arguments
    /// * `tau_b` - 床面剪切应力 [Pa]
    /// * `props` - 泥沙属性
    /// 
    /// # Returns
    /// 单宽推移质输沙率 q_b [m²/s]
    fn compute_transport_rate(&self, tau_b: f64, props: &SedimentProperties) -> f64;
    
    /// 计算输沙向量（含方向）
    fn compute_transport_vector(
        &self,
        tau_bx: f64,
        tau_by: f64,
        props: &SedimentProperties,
    ) -> (f64, f64) {
        let tau_b = (tau_bx * tau_bx + tau_by * tau_by).sqrt();
        if tau_b < 1e-14 {
            return (0.0, 0.0);
        }
        
        let qb = self.compute_transport_rate(tau_b, props);
        let ratio = qb / tau_b;
        (tau_bx * ratio, tau_by * ratio)
    }
}

/// Meyer-Peter-Müller (1948) 公式
/// 
/// q_b* = 8 × (θ - θ_cr)^1.5
/// q_b = q_b* × √[(s-1)gd³]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MeyerPeterMuller {
    /// 公式系数（默认8）
    pub coefficient: f64,
    /// 指数（默认1.5）
    pub exponent: f64,
}

impl MeyerPeterMuller {
    /// 创建默认参数的MPM公式
    pub fn new() -> Self {
        Self {
            coefficient: 8.0,
            exponent: 1.5,
        }
    }

    /// 设置系数
    pub fn with_coefficient(mut self, c: f64) -> Self {
        self.coefficient = c;
        self
    }
}

impl BedLoadFormula for MeyerPeterMuller {
    fn name(&self) -> &str {
        "Meyer-Peter-Müller"
    }

    fn compute_transport_rate(&self, tau_b: f64, props: &SedimentProperties) -> f64 {
        let excess_theta = props.excess_shields(tau_b);
        if excess_theta <= 0.0 {
            return 0.0;
        }

        // 无量纲输沙率
        let phi = self.coefficient * excess_theta.powf(self.exponent);
        
        // 转换为实际输沙率
        let d = props.d50;
        let s = props.relative_density;
        let scale = ((s - 1.0) * G * d * d * d).sqrt();
        
        phi * scale
    }
}

/// Van Rijn (1984) 公式
/// 
/// q_b = 0.053 × √[(s-1)gd³] × D*^(-0.3) × T^2.1
/// 其中 T = (τ_b - τ_cr) / τ_cr
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct VanRijn {
    /// 公式系数（默认0.053）
    pub coefficient: f64,
}

impl VanRijn {
    /// 创建默认参数的Van Rijn公式
    pub fn new() -> Self {
        Self { coefficient: 0.053 }
    }
}

impl BedLoadFormula for VanRijn {
    fn name(&self) -> &str {
        "Van Rijn"
    }

    fn compute_transport_rate(&self, tau_b: f64, props: &SedimentProperties) -> f64 {
        let tau_cr = props.critical_shear_stress;
        if tau_b <= tau_cr {
            return 0.0;
        }

        // 输沙强度参数 T
        let t_param = (tau_b - tau_cr) / tau_cr;
        
        // D*
        let d_star = props.dimensionless_diameter;
        
        // 输沙率
        let d = props.d50;
        let s = props.relative_density;
        let scale = ((s - 1.0) * G * d * d * d).sqrt();
        
        self.coefficient * scale * d_star.powf(-0.3) * t_param.powf(2.1)
    }
}

/// Einstein (1950) 公式
/// 
/// φ = f(ψ)，其中 ψ = (s-1)gd / (τ_b/ρ)
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Einstein;

impl Einstein {
    /// 创建Einstein公式
    pub fn new() -> Self {
        Self
    }
}

impl BedLoadFormula for Einstein {
    fn name(&self) -> &str {
        "Einstein"
    }

    fn compute_transport_rate(&self, tau_b: f64, props: &SedimentProperties) -> f64 {
        if tau_b < 1e-14 {
            return 0.0;
        }

        let d = props.d50;
        let s = props.relative_density;
        
        // 剪切速度
        let u_star = (tau_b / 1000.0).sqrt();
        
        // Einstein 参数
        let psi = (s - 1.0) * G * d / (u_star * u_star);
        
        if psi > 40.0 {
            return 0.0;  // 无输沙
        }
        
        // 简化的 Einstein 曲线近似
        let phi = if psi < 2.0 {
            40.0 * (-0.39 * psi).exp()
        } else {
            0.465 * psi.powf(-2.5)
        };
        
        let scale = ((s - 1.0) * G * d * d * d).sqrt();
        phi * scale
    }
}

/// 推移质输运计算器
pub struct BedLoadTransport {
    /// 输沙公式
    formula: Box<dyn BedLoadFormula>,
    /// 泥沙属性
    properties: SedimentProperties,
    /// 床面剪切应力 x 分量
    tau_bx: Vec<f64>,
    /// 床面剪切应力 y 分量
    tau_by: Vec<f64>,
    /// 推移质输沙率 x 分量
    qbx: Vec<f64>,
    /// 推移质输沙率 y 分量
    qby: Vec<f64>,
}

impl BedLoadTransport {
    /// 创建新的计算器
    pub fn new(n_cells: usize, properties: SedimentProperties) -> Self {
        Self {
            formula: Box::new(MeyerPeterMuller::new()),
            properties,
            tau_bx: vec![0.0; n_cells],
            tau_by: vec![0.0; n_cells],
            qbx: vec![0.0; n_cells],
            qby: vec![0.0; n_cells],
        }
    }

    /// 设置输沙公式
    pub fn with_formula<F: BedLoadFormula + 'static>(mut self, formula: F) -> Self {
        self.formula = Box::new(formula);
        self
    }

    /// 计算床面剪切应力
    /// 
    /// τ_b = ρ × g × n² × |V|² / h^(1/3)
    pub fn compute_bed_shear_stress(
        &mut self,
        h: &[f64],
        u: &[f64],
        v: &[f64],
        manning_n: f64,
    ) {
        let rho = 1000.0;
        let h_min = 0.01;

        for i in 0..self.tau_bx.len().min(h.len()).min(u.len()).min(v.len()) {
            let hi = h[i].max(h_min);
            let ui = u[i];
            let vi = v[i];
            let speed = (ui * ui + vi * vi).sqrt();
            
            // 床面剪切应力大小
            let tau_mag = rho * G * manning_n * manning_n * speed * speed / hi.powf(1.0 / 3.0);
            
            // 分量
            if speed > 1e-10 {
                self.tau_bx[i] = tau_mag * ui / speed;
                self.tau_by[i] = tau_mag * vi / speed;
            } else {
                self.tau_bx[i] = 0.0;
                self.tau_by[i] = 0.0;
            }
        }
    }

    /// 计算推移质输沙率
    pub fn compute_transport_rates(&mut self) {
        for i in 0..self.qbx.len() {
            let (qx, qy) = self.formula.compute_transport_vector(
                self.tau_bx[i],
                self.tau_by[i],
                &self.properties,
            );
            self.qbx[i] = qx;
            self.qby[i] = qy;
        }
    }

    /// 获取输沙率向量
    pub fn transport_rates(&self) -> (&[f64], &[f64]) {
        (&self.qbx, &self.qby)
    }

    /// 获取床面剪切应力
    pub fn bed_shear_stress(&self) -> (&[f64], &[f64]) {
        (&self.tau_bx, &self.tau_by)
    }

    /// 获取泥沙属性
    pub fn properties(&self) -> &SedimentProperties {
        &self.properties
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpm_formula() {
        let mpm = MeyerPeterMuller::new();
        let props = SedimentProperties::from_d50_mm(0.5);
        
        // 低于临界剪切应力
        let qb = mpm.compute_transport_rate(0.1, &props);
        assert!((qb).abs() < 1e-14);
        
        // 高于临界剪切应力
        let qb = mpm.compute_transport_rate(1.0, &props);
        assert!(qb > 0.0);
    }

    #[test]
    fn test_van_rijn_formula() {
        let vr = VanRijn::new();
        let props = SedimentProperties::from_d50_mm(0.3);
        
        // 高剪切应力
        let qb = vr.compute_transport_rate(2.0, &props);
        assert!(qb > 0.0);
    }

    #[test]
    fn test_einstein_formula() {
        let ein = Einstein::new();
        let props = SedimentProperties::from_d50_mm(0.5);
        
        // 高剪切应力
        let qb = ein.compute_transport_rate(2.0, &props);
        assert!(qb > 0.0);
    }

    #[test]
    fn test_transport_vector() {
        let mpm = MeyerPeterMuller::new();
        let props = SedimentProperties::from_d50_mm(0.5);
        
        // 45度方向
        let (qx, qy) = mpm.compute_transport_vector(1.0, 1.0, &props);
        
        // 两个分量应该相等
        assert!((qx - qy).abs() < 1e-10);
    }

    #[test]
    fn test_bed_load_transport_new() {
        let props = SedimentProperties::from_d50_mm(0.5);
        let transport = BedLoadTransport::new(100, props);
        
        assert_eq!(transport.transport_rates().0.len(), 100);
    }

    #[test]
    fn test_bed_load_transport_compute() {
        let props = SedimentProperties::from_d50_mm(0.5);
        let mut transport = BedLoadTransport::new(10, props);
        
        let h = vec![1.0; 10];
        let u = vec![1.0; 10];
        let v = vec![0.5; 10];
        
        transport.compute_bed_shear_stress(&h, &u, &v, 0.025);
        transport.compute_transport_rates();
        
        let (qx, qy) = transport.transport_rates();
        assert!(qx.iter().all(|&q| q >= 0.0));
        assert!(qy.iter().all(|&q| q >= 0.0));
    }
}
