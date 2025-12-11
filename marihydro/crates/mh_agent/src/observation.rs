// crates/mh_agent/src/observation.rs

use crate::PhysicsSnapshot;

/// 观测算子trait
pub trait ObservationOperator: Send + Sync {
    /// 观测类型名称
    fn name(&self) -> &'static str;
    
    /// 模拟状态 → 观测空间
    fn observe(&self, snapshot: &PhysicsSnapshot) -> Vec<f64>;
    
    /// 计算观测-模拟残差
    fn residual(&self, snapshot: &PhysicsSnapshot, observation: &[f64]) -> Vec<f64>;
    
    /// 获取观测误差协方差（对角阵时返回方差）
    fn observation_error_variance(&self) -> Option<Vec<f64>> { None }
    
    /// 线性化观测算子（返回雅可比矩阵）
    fn linearize(&self, _snapshot: &PhysicsSnapshot) -> Option<Vec<Vec<f64>>> { None }
}

/// 遥感反射率观测算子
pub struct ReflectanceOperator {
    /// 波长 [nm]
    wavelength: f64,
    /// 校准参数 [a, b, c, ...] for R = a * ln(C) + b
    calibration: Vec<f64>,
    /// 观测误差标准差
    observation_std: f64,
}

impl ReflectanceOperator {
    pub fn new(wavelength: f64, calibration: Vec<f64>, observation_std: f64) -> Self {
        Self { wavelength, calibration, observation_std }
    }
    
    /// 使用默认的MODIS红波段校准参数
    pub fn modis_red_band() -> Self {
        Self::new(645.0, vec![0.12, 0.01], 0.02)
    }
    
    /// 使用默认的Sentinel-2校准参数
    pub fn sentinel2_b4() -> Self {
        Self::new(665.0, vec![0.09, 0.0], 0.02)
    }
}

impl ObservationOperator for ReflectanceOperator {
    fn name(&self) -> &'static str { "Reflectance" }
    
    fn observe(&self, snapshot: &PhysicsSnapshot) -> Vec<f64> {
        snapshot.sediment.as_ref()
            .map(|c| c.iter().map(|&conc| {
                let c_safe = conc.max(1e-10);
                let a = *self.calibration.get(0).unwrap_or(&1.0);
                let b = *self.calibration.get(1).unwrap_or(&0.0);
                a * c_safe.ln() + b
            }).collect())
            .unwrap_or_else(|| vec![0.0; snapshot.n_cells()])
    }
    
    fn residual(&self, snapshot: &PhysicsSnapshot, observation: &[f64]) -> Vec<f64> {
        let simulated = self.observe(snapshot);
        simulated
            .iter()
            .zip(observation.iter())
            .map(|(s, o)| o - s)
            .collect()
    }
    
    fn observation_error_variance(&self) -> Option<Vec<f64>> {
        Some(vec![self.observation_std.powi(2); 1])
    }
}

/// SAR后向散射观测算子
pub struct SAROperator {
    /// 入射角 [degrees]
    incidence_angle: f64,
    /// 极化方式
    polarization: Polarization,
    /// 风速校正系数
    wind_correction: f64,
    /// 观测误差标准差 [dB]
    observation_std: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum Polarization {
    VV,
    VH,
    HH,
    HV,
}

impl SAROperator {
    pub fn new(incidence_angle: f64, polarization: Polarization) -> Self {
        Self {
            incidence_angle,
            polarization,
            wind_correction: 1.0,
            observation_std: 1.0,
        }
    }
}

impl ObservationOperator for SAROperator {
    fn name(&self) -> &'static str { "SAR-Backscatter" }
    
    fn observe(&self, snapshot: &PhysicsSnapshot) -> Vec<f64> {
        let mut result = Vec::with_capacity(snapshot.n_cells());
        for i in 0..snapshot.n_cells() {
            let speed = snapshot.u.get(i).copied().unwrap_or(0.0).hypot(snapshot.v.get(i).copied().unwrap_or(0.0));
            let depth = snapshot.h.get(i).copied().unwrap_or(0.0).max(1e-6);
            let incidence_factor = self.incidence_angle.to_radians().cos().abs();
            let pol_factor = match self.polarization {
                Polarization::VV | Polarization::HH => 1.0,
                _ => 0.8,
            };
            let backscatter = 10.0 * (speed / depth * incidence_factor * pol_factor * self.wind_correction + 1e-6).ln();
            result.push(backscatter);
        }
        result
    }
    
    fn residual(&self, snapshot: &PhysicsSnapshot, observation: &[f64]) -> Vec<f64> {
        let simulated = self.observe(snapshot);
        simulated
            .iter()
            .zip(observation.iter())
            .map(|(s, o)| o - s)
            .collect()
    }
}

/// 水位观测算子（验潮站）
pub struct WaterLevelOperator {
    /// 观测站位置索引
    station_indices: Vec<usize>,
    /// 观测误差标准差 [m]
    observation_std: f64,
}

impl WaterLevelOperator {
    pub fn new(station_indices: Vec<usize>, observation_std: f64) -> Self {
        Self { station_indices, observation_std }
    }
}

impl ObservationOperator for WaterLevelOperator {
    fn name(&self) -> &'static str { "WaterLevel" }
    
    fn observe(&self, snapshot: &PhysicsSnapshot) -> Vec<f64> {
        self.station_indices.iter()
            .map(|&i| snapshot.h.get(i).copied().unwrap_or(0.0) + snapshot.z.get(i).copied().unwrap_or(0.0))
            .collect()
    }
    
    fn residual(&self, snapshot: &PhysicsSnapshot, observation: &[f64]) -> Vec<f64> {
        let sim = self.observe(snapshot);
        sim.iter()
            .zip(observation.iter())
            .map(|(s, o)| o - s)
            .collect()
    }
    
    fn observation_error_variance(&self) -> Option<Vec<f64>> {
        Some(vec![self.observation_std.powi(2); self.station_indices.len()])
    }
}
