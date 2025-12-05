// crates/mh_physics/src/waves/radiation_stress.rs

//! 波浪辐射应力计算
//! 
//! 实现波浪辐射应力张量及其梯度计算，用于波流耦合模拟。

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// 重力加速度
const G: f64 = 9.81;
/// 海水密度
const RHO_WATER: f64 = 1025.0;

/// 波浪参数
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WaveParameters {
    /// 有效波高 H [m]
    pub height: f64,
    /// 波浪周期 T [s]
    pub period: f64,
    /// 波向 θ [弧度]，从正北顺时针测量
    pub direction: f64,
}

impl WaveParameters {
    /// 创建新的波浪参数
    pub fn new(height: f64, period: f64, direction: f64) -> Self {
        Self {
            height,
            period,
            direction,
        }
    }

    /// 角频率 ω = 2π/T
    pub fn angular_frequency(&self) -> f64 {
        2.0 * PI / self.period
    }

    /// 深水波长 L0 = gT²/(2π)
    pub fn deep_water_wavelength(&self) -> f64 {
        G * self.period * self.period / (2.0 * PI)
    }

    /// 波浪能量密度 E = ρgH²/8
    pub fn energy(&self) -> f64 {
        RHO_WATER * G * self.height * self.height / 8.0
    }

    /// 波向单位向量 (x, y)
    pub fn direction_vector(&self) -> (f64, f64) {
        (self.direction.sin(), self.direction.cos())
    }
}

impl Default for WaveParameters {
    fn default() -> Self {
        Self {
            height: 1.0,
            period: 8.0,
            direction: 0.0,
        }
    }
}

/// 波场数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveField {
    /// 波高 [m]
    pub height: Vec<f64>,
    /// 周期 [s]
    pub period: Vec<f64>,
    /// 波向 [弧度]
    pub direction: Vec<f64>,
    /// 波长 [m]
    pub wavelength: Vec<f64>,
    /// 波数 k [1/m]
    pub wavenumber: Vec<f64>,
    /// 群速度因子 n = Cg/C
    pub group_factor: Vec<f64>,
    /// 能量密度 [J/m²]
    pub energy: Vec<f64>,
}

impl WaveField {
    /// 创建指定大小的波场
    pub fn new(n_cells: usize) -> Self {
        Self {
            height: vec![0.0; n_cells],
            period: vec![8.0; n_cells],
            direction: vec![0.0; n_cells],
            wavelength: vec![0.0; n_cells],
            wavenumber: vec![0.0; n_cells],
            group_factor: vec![0.5; n_cells],
            energy: vec![0.0; n_cells],
        }
    }

    /// 从均匀参数创建波场
    pub fn from_uniform(n_cells: usize, params: &WaveParameters) -> Self {
        let mut field = Self::new(n_cells);
        field.set_uniform(params);
        field
    }

    /// 设置均匀波浪参数
    pub fn set_uniform(&mut self, params: &WaveParameters) {
        let n = self.height.len();
        self.height = vec![params.height; n];
        self.period = vec![params.period; n];
        self.direction = vec![params.direction; n];
        self.energy = vec![params.energy(); n];
        
        // 波长和波数需要根据水深计算，这里先用深水近似
        let l0 = params.deep_water_wavelength();
        self.wavelength = vec![l0; n];
        self.wavenumber = vec![2.0 * PI / l0; n];
        self.group_factor = vec![0.5; n];  // 深水近似
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        self.height.resize(n_cells, 0.0);
        self.period.resize(n_cells, 8.0);
        self.direction.resize(n_cells, 0.0);
        self.wavelength.resize(n_cells, 0.0);
        self.wavenumber.resize(n_cells, 0.0);
        self.group_factor.resize(n_cells, 0.5);
        self.energy.resize(n_cells, 0.0);
    }

    /// 获取单元格数量
    pub fn len(&self) -> usize {
        self.height.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.height.is_empty()
    }

    /// 根据水深更新波场参数（色散关系）
    pub fn update_dispersion(&mut self, depth: &[f64]) {
        for i in 0..self.len().min(depth.len()) {
            let h = depth[i].max(0.1);
            let omega = 2.0 * PI / self.period[i];
            
            // 求解色散关系
            let (k, n) = compute_wavenumber_and_n(omega, h);
            self.wavenumber[i] = k;
            self.group_factor[i] = n;
            self.wavelength[i] = 2.0 * PI / k;
            
            // 更新能量
            self.energy[i] = RHO_WATER * G * self.height[i] * self.height[i] / 8.0;
        }
    }
}

/// 求解色散关系 ω² = gk·tanh(kh)
/// 
/// 返回 (k, n)，其中 n = Cg/C = 群速度/相速度
pub fn compute_wavenumber_and_n(omega: f64, depth: f64) -> (f64, f64) {
    let h = depth.max(0.01);
    
    // 初始猜测（深水近似）
    let k0 = omega * omega / G;
    
    // Newton-Raphson 迭代
    let mut k = k0;
    for _ in 0..20 {
        let kh = k * h;
        let tanh_kh = kh.tanh();
        let f = omega * omega - G * k * tanh_kh;
        let df = -G * (tanh_kh + k * h * (1.0 - tanh_kh * tanh_kh));
        
        let dk = -f / df;
        k += dk;
        
        if dk.abs() < 1e-10 * k {
            break;
        }
    }
    
    // 群速度因子 n = Cg/C = 0.5(1 + 2kh/sinh(2kh))
    let kh = k * h;
    let sinh_2kh = (2.0 * kh).sinh();
    let n = if sinh_2kh.abs() > 1e-10 {
        0.5 * (1.0 + 2.0 * kh / sinh_2kh)
    } else {
        1.0  // 浅水极限
    };
    
    (k, n)
}

/// 辐射应力张量
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct RadiationStressTensor {
    /// Sxx 分量 [N/m]
    pub sxx: f64,
    /// Syy 分量 [N/m]
    pub syy: f64,
    /// Sxy = Syx 分量 [N/m]
    pub sxy: f64,
}

impl RadiationStressTensor {
    /// 创建零张量
    pub fn zero() -> Self {
        Self::default()
    }

    /// 计算辐射应力张量
    /// 
    /// Sxx = E(n(cos²θ + 1) - 0.5)
    /// Syy = E(n(sin²θ + 1) - 0.5)
    /// Sxy = E·n·sin θ·cos θ
    pub fn compute(energy: f64, n: f64, direction: f64) -> Self {
        let cos_theta = direction.cos();
        let sin_theta = direction.sin();
        let cos2 = cos_theta * cos_theta;
        let sin2 = sin_theta * sin_theta;
        
        Self {
            sxx: energy * (n * (cos2 + 1.0) - 0.5),
            syy: energy * (n * (sin2 + 1.0) - 0.5),
            sxy: energy * n * sin_theta * cos_theta,
        }
    }

    /// 获取主应力
    pub fn principal_stresses(&self) -> (f64, f64) {
        let avg = 0.5 * (self.sxx + self.syy);
        let diff = 0.5 * (self.sxx - self.syy);
        let r = (diff * diff + self.sxy * self.sxy).sqrt();
        (avg + r, avg - r)
    }
}

/// 辐射应力计算器
pub struct RadiationStressCalculator {
    /// 辐射应力 Sxx
    sxx: Vec<f64>,
    /// 辐射应力 Syy
    syy: Vec<f64>,
    /// 辐射应力 Sxy
    sxy: Vec<f64>,
    /// 辐射应力梯度 x 分量（力/面积）
    force_x: Vec<f64>,
    /// 辐射应力梯度 y 分量
    force_y: Vec<f64>,
}

impl RadiationStressCalculator {
    /// 创建新的计算器
    pub fn new(n_cells: usize) -> Self {
        Self {
            sxx: vec![0.0; n_cells],
            syy: vec![0.0; n_cells],
            sxy: vec![0.0; n_cells],
            force_x: vec![0.0; n_cells],
            force_y: vec![0.0; n_cells],
        }
    }

    /// 从波场计算辐射应力
    pub fn compute_stress(&mut self, wave_field: &WaveField) {
        for i in 0..self.sxx.len().min(wave_field.len()) {
            let tensor = RadiationStressTensor::compute(
                wave_field.energy[i],
                wave_field.group_factor[i],
                wave_field.direction[i],
            );
            self.sxx[i] = tensor.sxx;
            self.syy[i] = tensor.syy;
            self.sxy[i] = tensor.sxy;
        }
    }

    /// 计算辐射应力梯度（结构化网格）
    /// 
    /// Fx = -∂Sxx/∂x - ∂Sxy/∂y
    /// Fy = -∂Sxy/∂x - ∂Syy/∂y
    pub fn compute_gradient_structured(
        &mut self,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        depth: &[f64],
    ) {
        let inv_dx = 1.0 / dx;
        let inv_dy = 1.0 / dy;
        let h_min = 0.1;

        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                let h = depth[idx].max(h_min);

                // ∂Sxx/∂x
                let dsxx_dx = if i == 0 {
                    (self.sxx[idx + 1] - self.sxx[idx]) * inv_dx
                } else if i == nx - 1 {
                    (self.sxx[idx] - self.sxx[idx - 1]) * inv_dx
                } else {
                    (self.sxx[idx + 1] - self.sxx[idx - 1]) * 0.5 * inv_dx
                };

                // ∂Sxy/∂y
                let dsxy_dy = if j == 0 {
                    (self.sxy[idx + nx] - self.sxy[idx]) * inv_dy
                } else if j == ny - 1 {
                    (self.sxy[idx] - self.sxy[idx - nx]) * inv_dy
                } else {
                    (self.sxy[idx + nx] - self.sxy[idx - nx]) * 0.5 * inv_dy
                };

                // ∂Sxy/∂x
                let dsxy_dx = if i == 0 {
                    (self.sxy[idx + 1] - self.sxy[idx]) * inv_dx
                } else if i == nx - 1 {
                    (self.sxy[idx] - self.sxy[idx - 1]) * inv_dx
                } else {
                    (self.sxy[idx + 1] - self.sxy[idx - 1]) * 0.5 * inv_dx
                };

                // ∂Syy/∂y
                let dsyy_dy = if j == 0 {
                    (self.syy[idx + nx] - self.syy[idx]) * inv_dy
                } else if j == ny - 1 {
                    (self.syy[idx] - self.syy[idx - nx]) * inv_dy
                } else {
                    (self.syy[idx + nx] - self.syy[idx - nx]) * 0.5 * inv_dy
                };

                // 波浪作用力 = -梯度 / (ρh)
                self.force_x[idx] = -(dsxx_dx + dsxy_dy) / (RHO_WATER * h);
                self.force_y[idx] = -(dsxy_dx + dsyy_dy) / (RHO_WATER * h);
            }
        }
    }

    /// 获取辐射应力分量
    pub fn stress_components(&self) -> (&[f64], &[f64], &[f64]) {
        (&self.sxx, &self.syy, &self.sxy)
    }

    /// 获取波浪力（加速度）
    pub fn wave_forces(&self) -> (&[f64], &[f64]) {
        (&self.force_x, &self.force_y)
    }
}

/// 波浪源项
pub struct WaveSource {
    /// 波浪场
    wave_field: WaveField,
    /// 辐射应力计算器
    stress_calculator: RadiationStressCalculator,
    /// 是否启用
    enabled: bool,
}

impl WaveSource {
    /// 创建新的波浪源项
    pub fn new(n_cells: usize) -> Self {
        Self {
            wave_field: WaveField::new(n_cells),
            stress_calculator: RadiationStressCalculator::new(n_cells),
            enabled: true,
        }
    }

    /// 设置波浪参数
    pub fn set_wave_parameters(&mut self, params: &WaveParameters) {
        self.wave_field.set_uniform(params);
    }

    /// 设置波浪场
    pub fn set_wave_field(&mut self, field: WaveField) {
        self.wave_field = field;
        self.stress_calculator = RadiationStressCalculator::new(self.wave_field.len());
    }

    /// 更新色散关系
    pub fn update_dispersion(&mut self, depth: &[f64]) {
        self.wave_field.update_dispersion(depth);
    }

    /// 计算波浪力
    pub fn compute_forces(&mut self, nx: usize, ny: usize, dx: f64, dy: f64, depth: &[f64]) {
        if !self.enabled {
            return;
        }
        self.stress_calculator.compute_stress(&self.wave_field);
        self.stress_calculator.compute_gradient_structured(nx, ny, dx, dy, depth);
    }

    /// 获取波浪力
    pub fn get_forces(&self) -> (&[f64], &[f64]) {
        self.stress_calculator.wave_forces()
    }

    /// 获取波浪场
    pub fn wave_field(&self) -> &WaveField {
        &self.wave_field
    }

    /// 启用/禁用
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// 是否启用
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wave_parameters() {
        let params = WaveParameters::new(2.0, 10.0, 0.0);
        
        let omega = params.angular_frequency();
        assert!((omega - 2.0 * PI / 10.0).abs() < 1e-10);
        
        let l0 = params.deep_water_wavelength();
        assert!(l0 > 100.0);  // 深水波长约156m
        
        let energy = params.energy();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_wave_parameters_direction_vector() {
        // 北向
        let params = WaveParameters::new(1.0, 8.0, 0.0);
        let (dx, dy) = params.direction_vector();
        assert!(dx.abs() < 1e-10);
        assert!((dy - 1.0).abs() < 1e-10);
        
        // 东向
        let params = WaveParameters::new(1.0, 8.0, PI / 2.0);
        let (dx, dy) = params.direction_vector();
        assert!((dx - 1.0).abs() < 1e-10);
        assert!(dy.abs() < 1e-10);
    }

    #[test]
    fn test_wave_field_from_uniform() {
        let params = WaveParameters::new(2.0, 10.0, PI / 4.0);
        let field = WaveField::from_uniform(100, &params);
        
        assert_eq!(field.len(), 100);
        assert!((field.height[50] - 2.0).abs() < 1e-10);
        assert!((field.period[50] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_wave_field_update_dispersion() {
        let params = WaveParameters::new(1.0, 8.0, 0.0);
        let mut field = WaveField::from_uniform(10, &params);
        
        let depth = vec![10.0; 10];
        field.update_dispersion(&depth);
        
        // 波数应该增加（相比深水）
        assert!(field.wavenumber[0] > 0.0);
        assert!(field.wavelength[0] > 0.0);
    }

    #[test]
    fn test_wavenumber_calculation() {
        let omega = 2.0 * PI / 8.0;  // T = 8s
        
        // 深水
        let (k_deep, n_deep) = compute_wavenumber_and_n(omega, 100.0);
        assert!((n_deep - 0.5).abs() < 0.01);  // 深水 n ≈ 0.5
        
        // 浅水
        let (k_shallow, n_shallow) = compute_wavenumber_and_n(omega, 1.0);
        assert!(n_shallow > 0.9);  // 浅水 n → 1
        assert!(k_shallow > k_deep);  // 浅水波数更大
    }

    #[test]
    fn test_radiation_stress_tensor() {
        let energy = 1000.0;  // J/m²
        let n = 0.5;
        let direction = 0.0;  // 北向
        
        let tensor = RadiationStressTensor::compute(energy, n, direction);
        
        // Sxx = E(n(cos²θ + 1) - 0.5) = 1000(0.5(1+1) - 0.5) = 500
        assert!((tensor.sxx - 500.0).abs() < 1e-10);
        
        // Syy = E(n(sin²θ + 1) - 0.5) = 1000(0.5(0+1) - 0.5) = 0
        assert!((tensor.syy - 0.0).abs() < 1e-10);
        
        // Sxy = 0 (北向)
        assert!(tensor.sxy.abs() < 1e-10);
    }

    #[test]
    fn test_radiation_stress_calculator() {
        let params = WaveParameters::new(2.0, 10.0, PI / 4.0);
        let field = WaveField::from_uniform(25, &params);
        
        let mut calc = RadiationStressCalculator::new(25);
        calc.compute_stress(&field);
        
        let (sxx, syy, sxy) = calc.stress_components();
        assert!(sxx.iter().all(|&s| s >= 0.0));
        assert!(syy.iter().all(|&s| s >= 0.0));
        // 45度方向 sxy 应该非零
        assert!(sxy.iter().any(|&s| s.abs() > 1e-10));
    }

    #[test]
    fn test_radiation_stress_gradient() {
        let params = WaveParameters::new(2.0, 10.0, 0.0);
        let field = WaveField::from_uniform(25, &params);
        
        let mut calc = RadiationStressCalculator::new(25);
        calc.compute_stress(&field);
        
        let depth = vec![10.0; 25];
        calc.compute_gradient_structured(5, 5, 10.0, 10.0, &depth);
        
        let (fx, fy) = calc.wave_forces();
        // 均匀场应该力接近零
        let max_force = fx.iter().chain(fy.iter()).map(|&f| f.abs()).fold(0.0, f64::max);
        assert!(max_force < 1.0);  // 合理范围内
    }

    #[test]
    fn test_wave_source() {
        let mut source = WaveSource::new(25);
        
        let params = WaveParameters::new(1.5, 8.0, 0.0);
        source.set_wave_parameters(&params);
        
        let depth = vec![5.0; 25];
        source.update_dispersion(&depth);
        source.compute_forces(5, 5, 10.0, 10.0, &depth);
        
        let (fx, fy) = source.get_forces();
        assert_eq!(fx.len(), 25);
        assert_eq!(fy.len(), 25);
    }

    #[test]
    fn test_wave_source_disable() {
        let mut source = WaveSource::new(10);
        source.set_enabled(false);
        
        let depth = vec![5.0; 10];
        source.compute_forces(10, 1, 10.0, 10.0, &depth);
        
        // 禁用后力应为零
        let (fx, fy) = source.get_forces();
        assert!(fx.iter().all(|&f| f == 0.0));
        assert!(fy.iter().all(|&f| f == 0.0));
    }
}
