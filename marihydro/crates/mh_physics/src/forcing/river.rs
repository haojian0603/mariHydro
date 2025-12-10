// crates/mh_physics/src/forcing/river.rs

//! 河流入流数据提供者
//!
//! 提供时变河流入流边界条件，包括：
//! - 恒定流量
//! - 时间序列流量
//! - 洪水过程线
//! - 多点河流入流
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::forcing::RiverProvider;
//!
//! // 恒定流量
//! let river = RiverProvider::constant(100.0); // 100 m³/s
//!
//! // 洪水过程线
//! let flood = RiverProvider::flood_wave(
//!     50.0,   // 基流
//!     500.0,  // 峰值
//!     3600.0, // 上升时间
//!     7200.0, // 下降时间
//! );
//! ```

/// 河流入流数据提供者
#[derive(Debug, Clone)]
pub struct RiverProvider {
    /// 数据类型
    data: RiverData,
    /// 入流方向 [弧度]
    direction: f64,
    /// 最后更新时间
    last_update: f64,
    /// 缓存的流量
    cached_discharge: f64,
}

/// 河流数据类型
#[derive(Debug, Clone)]
pub enum RiverData {
    /// 恒定流量
    Constant(f64),
    /// 时间序列
    TimeSeries {
        times: Vec<f64>,
        discharges: Vec<f64>,
    },
    /// 洪水过程线（简化三角形）
    FloodWave {
        /// 基流 [m³/s]
        base_flow: f64,
        /// 峰值流量 [m³/s]
        peak_flow: f64,
        /// 峰值时间 [s]
        peak_time: f64,
        /// 上升时间 [s]
        rise_time: f64,
        /// 下降时间 [s]
        fall_time: f64,
    },
    /// 周期性流量（日变化）
    Periodic {
        /// 平均流量
        mean_discharge: f64,
        /// 振幅
        amplitude: f64,
        /// 周期 [s]
        period: f64,
        /// 相位 [弧度]
        phase: f64,
    },
    /// 阶梯流量
    Step {
        /// 初始流量
        initial: f64,
        /// 变化时间和流量
        steps: Vec<(f64, f64)>,
    },
}

impl RiverProvider {
    /// 创建恒定流量
    pub fn constant(discharge: f64) -> Self {
        Self {
            data: RiverData::Constant(discharge),
            direction: 0.0,
            last_update: 0.0,
            cached_discharge: discharge,
        }
    }

    /// 创建无入流
    pub fn none() -> Self {
        Self::constant(0.0)
    }

    /// 创建时间序列
    pub fn time_series(times: Vec<f64>, discharges: Vec<f64>) -> Self {
        let initial = discharges.first().copied().unwrap_or(0.0);
        Self {
            data: RiverData::TimeSeries { times, discharges },
            direction: 0.0,
            last_update: 0.0,
            cached_discharge: initial,
        }
    }

    /// 创建洪水过程线
    ///
    /// # 参数
    /// - `base_flow`: 基流 [m³/s]
    /// - `peak_flow`: 峰值流量 [m³/s]
    /// - `rise_time`: 上升时间 [s]（从开始到峰值）
    /// - `fall_time`: 下降时间 [s]（从峰值到恢复基流）
    pub fn flood_wave(base_flow: f64, peak_flow: f64, rise_time: f64, fall_time: f64) -> Self {
        Self {
            data: RiverData::FloodWave {
                base_flow,
                peak_flow,
                peak_time: rise_time,
                rise_time,
                fall_time,
            },
            direction: 0.0,
            last_update: 0.0,
            cached_discharge: base_flow,
        }
    }

    /// 创建延迟洪水
    pub fn delayed_flood(
        base_flow: f64,
        peak_flow: f64,
        delay: f64,
        rise_time: f64,
        fall_time: f64,
    ) -> Self {
        Self {
            data: RiverData::FloodWave {
                base_flow,
                peak_flow,
                peak_time: delay + rise_time,
                rise_time,
                fall_time,
            },
            direction: 0.0,
            last_update: 0.0,
            cached_discharge: base_flow,
        }
    }

    /// 创建周期性流量
    pub fn periodic(mean_discharge: f64, amplitude: f64, period_hours: f64) -> Self {
        Self {
            data: RiverData::Periodic {
                mean_discharge,
                amplitude,
                period: period_hours * 3600.0,
                phase: 0.0,
            },
            direction: 0.0,
            last_update: 0.0,
            cached_discharge: mean_discharge,
        }
    }

    /// 创建阶梯流量
    pub fn step_changes(initial: f64, steps: Vec<(f64, f64)>) -> Self {
        Self {
            data: RiverData::Step { initial, steps },
            direction: 0.0,
            last_update: 0.0,
            cached_discharge: initial,
        }
    }

    /// 设置入流方向
    pub fn with_direction(mut self, direction_deg: f64) -> Self {
        self.direction = direction_deg.to_radians();
        self
    }

    /// 获取指定时刻的流量
    pub fn get_discharge_at(&self, time: f64) -> f64 {
        match &self.data {
            RiverData::Constant(q) => *q,

            RiverData::TimeSeries { times, discharges } => {
                if times.is_empty() {
                    return 0.0;
                }

                if time <= times[0] {
                    return discharges[0];
                }
                if time >= *times.last().unwrap() {
                    return *discharges.last().unwrap();
                }

                // 线性插值
                for i in 0..times.len() - 1 {
                    if time >= times[i] && time < times[i + 1] {
                        let t = (time - times[i]) / (times[i + 1] - times[i]);
                        return discharges[i] + t * (discharges[i + 1] - discharges[i]);
                    }
                }

                0.0
            }

            RiverData::FloodWave { base_flow, peak_flow, peak_time, rise_time, fall_time } => {
                let start_time = peak_time - rise_time;
                let end_time = peak_time + fall_time;

                if time < start_time {
                    *base_flow
                } else if time < *peak_time {
                    // 上升段
                    let t = (time - start_time) / rise_time;
                    base_flow + t * (peak_flow - base_flow)
                } else if time < end_time {
                    // 下降段
                    let t = (time - peak_time) / fall_time;
                    peak_flow - t * (peak_flow - base_flow)
                } else {
                    *base_flow
                }
            }

            RiverData::Periodic { mean_discharge, amplitude, period, phase } => {
                let omega = 2.0 * std::f64::consts::PI / period;
                mean_discharge + amplitude * (omega * time + phase).sin()
            }

            RiverData::Step { initial, steps } => {
                let mut q = *initial;
                for (t, new_q) in steps {
                    if time >= *t {
                        q = *new_q;
                    } else {
                        break;
                    }
                }
                q
            }
        }
    }

    /// 获取入流方向
    pub fn get_direction(&self) -> f64 {
        self.direction
    }

    /// 获取入流速度分量
    pub fn get_velocity_components(&self, time: f64, cross_section_area: f64) -> (f64, f64) {
        let q = self.get_discharge_at(time);
        let velocity = if cross_section_area > 1e-6 {
            q / cross_section_area
        } else {
            0.0
        };

        let u = velocity * self.direction.cos();
        let v = velocity * self.direction.sin();
        (u, v)
    }

    /// 更新缓存
    pub fn update(&mut self, time: f64) {
        if (time - self.last_update).abs() > 1e-10 {
            self.cached_discharge = self.get_discharge_at(time);
            self.last_update = time;
        }
    }

    /// 获取缓存的流量
    pub fn cached(&self) -> f64 {
        self.cached_discharge
    }

    /// 获取峰值流量
    pub fn peak_discharge(&self) -> f64 {
        match &self.data {
            RiverData::Constant(q) => *q,
            RiverData::TimeSeries { discharges, .. } => {
                discharges.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            }
            RiverData::FloodWave { peak_flow, .. } => *peak_flow,
            RiverData::Periodic { mean_discharge, amplitude, .. } => mean_discharge + amplitude,
            RiverData::Step { initial, steps } => {
                let max_step = steps.iter().map(|(_, q)| *q).fold(f64::NEG_INFINITY, f64::max);
                f64::max(*initial, max_step)
            }
        }
    }
}

/// 多河流入流管理器
#[derive(Debug, Clone)]
pub struct RiverSystem {
    /// 河流列表
    rivers: Vec<RiverEntry>,
}

/// 单条河流入口
#[derive(Debug, Clone)]
pub struct RiverEntry {
    /// 河流名称
    pub name: String,
    /// 入流单元
    pub cells: Vec<usize>,
    /// 数据提供者
    pub provider: RiverProvider,
    /// 入流分配权重
    pub weights: Vec<f64>,
}

impl RiverSystem {
    /// 创建空系统
    pub fn new() -> Self {
        Self { rivers: Vec::new() }
    }

    /// 添加河流
    pub fn add_river(&mut self, name: &str, cells: Vec<usize>, provider: RiverProvider) {
        let n = cells.len();
        let weight = if n > 0 { 1.0 / n as f64 } else { 0.0 };

        self.rivers.push(RiverEntry {
            name: name.to_string(),
            cells,
            provider,
            weights: vec![weight; n],
        });
    }

    /// 添加带权重的河流
    pub fn add_river_with_weights(
        &mut self,
        name: &str,
        cells: Vec<usize>,
        provider: RiverProvider,
        weights: Vec<f64>,
    ) {
        self.rivers.push(RiverEntry {
            name: name.to_string(),
            cells,
            provider,
            weights,
        });
    }

    /// 获取河流数量
    pub fn count(&self) -> usize {
        self.rivers.len()
    }

    /// 获取所有河流总流量
    pub fn total_discharge(&self, time: f64) -> f64 {
        self.rivers.iter().map(|r| r.provider.get_discharge_at(time)).sum()
    }

    /// 更新所有河流
    pub fn update_all(&mut self, time: f64) {
        for river in &mut self.rivers {
            river.provider.update(time);
        }
    }

    /// 获取单元的总入流量
    pub fn get_cell_inflow(&self, cell: usize, time: f64) -> f64 {
        let mut total = 0.0;
        for river in &self.rivers {
            for (i, &c) in river.cells.iter().enumerate() {
                if c == cell {
                    let q = river.provider.get_discharge_at(time);
                    let w = river.weights.get(i).copied().unwrap_or(0.0);
                    total += q * w;
                }
            }
        }
        total
    }


    /// 应用入流到状态（质量和动量同时更新）
    /// 
    /// 根据质量守恒和动量守恒方程，河流入流应同时贡献：
    /// - 质量源项：dh/dt = Q / A
    /// - 动量源项：d(hu)/dt = Q·u_in / A, d(hv)/dt = Q·v_in / A
    /// 
    /// 其中 (u_in, v_in) 是入流速度，由流量、过水断面和入流方向计算
    /// 
    /// # 参数
    /// - `time`: 当前时间 [s]
    /// - `dt`: 时间步长 [s]
    /// - `h`: 水深数组 [m]（就地更新）
    /// - `hu`: x方向动量 h·u [m²/s]（就地更新）
    /// - `hv`: y方向动量 h·v [m²/s]（就地更新）
    /// - `cell_areas`: 单元面积 [m²]
    /// - `cross_section_areas`: 各入流点的过水断面积 [m²]（用于计算入流速度）
    ///   若为 None，则使用单元面积的平方根乘以水深估算
    pub fn apply_inflow_with_momentum(
        &self,
        time: f64,
        dt: f64,
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
        cell_areas: &[f64],
        cross_section_areas: Option<&[f64]>,
    ) {
        for river in &self.rivers {
            let q = river.provider.get_discharge_at(time);
            let direction = river.provider.get_direction();

            for (i, &cell) in river.cells.iter().enumerate() {
                if cell >= h.len() || cell >= cell_areas.len() {
                    continue;
                }

                let w = river.weights.get(i).copied().unwrap_or(0.0);
                let area = cell_areas[cell].max(1e-6);
                
                // 计算单元的入流量
                let q_cell = q * w;
                
                // 计算水深变化
                let delta_h = q_cell * dt / area;
                
                // 计算入流速度
                // 过水断面面积：优先使用指定值，否则用 sqrt(A) * h_new 估算
                let cross_area = if let Some(cs) = cross_section_areas {
                    cs.get(cell).copied().unwrap_or_else(|| {
                        let h_new = (h[cell] + delta_h).max(1e-3);
                        cell_areas[cell].sqrt() * h_new
                    })
                } else {
                    // 假设过水断面宽度约为 sqrt(A)，深度约为 h
                    let h_new = (h[cell] + delta_h).max(1e-3);
                    cell_areas[cell].sqrt() * h_new
                };
                
                // 入流速度 = Q / A_cross
                let u_inflow = if cross_area > 1e-6 {
                    q_cell / cross_area
                } else {
                    0.0
                };
                
                // 速度分量
                let u_in = u_inflow * direction.cos();
                let v_in = u_inflow * direction.sin();
                
                // 更新质量（水深）
                h[cell] += delta_h;
                
                // 更新动量
                // 动量源项 = Q * u_in / A * dt = q_cell * u_in * dt / area
                // 注意：这里用的是动量通量，不是速度
                hu[cell] += q_cell * u_in * dt / area;
                hv[cell] += q_cell * v_in * dt / area;
            }
        }
    }

    /// 快速版本：使用单元自身水深估算入流速度
    /// 
    /// 这是 apply_inflow_with_momentum 的简化版本：
    /// - 过水断面假设为 sqrt(cell_area) * h
    /// - 入流速度按此断面计算
    pub fn apply_inflow_fast(
        &self,
        time: f64,
        dt: f64,
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
        cell_areas: &[f64],
    ) {
        self.apply_inflow_with_momentum(time, dt, h, hu, hv, cell_areas, None);
    }
}

impl Default for RiverSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_river() {
        let river = RiverProvider::constant(100.0);
        assert!((river.get_discharge_at(0.0) - 100.0).abs() < 1e-10);
        assert!((river.get_discharge_at(3600.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_series_river() {
        let river = RiverProvider::time_series(
            vec![0.0, 3600.0, 7200.0],
            vec![50.0, 100.0, 75.0],
        );

        assert!((river.get_discharge_at(0.0) - 50.0).abs() < 1e-10);
        assert!((river.get_discharge_at(1800.0) - 75.0).abs() < 1e-10); // 插值
        assert!((river.get_discharge_at(3600.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_flood_wave() {
        let river = RiverProvider::flood_wave(10.0, 100.0, 3600.0, 7200.0);

        // 洪水前
        assert!((river.get_discharge_at(0.0) - 10.0).abs() < 1e-10);

        // 峰值
        assert!((river.get_discharge_at(3600.0) - 100.0).abs() < 1e-10);

        // 洪水后
        assert!((river.get_discharge_at(20000.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_flood_wave_rising() {
        let river = RiverProvider::flood_wave(10.0, 100.0, 3600.0, 7200.0);

        // 上升中点
        let q = river.get_discharge_at(1800.0);
        assert!((q - 55.0).abs() < 1e-6);
    }

    #[test]
    fn test_step_changes() {
        let river = RiverProvider::step_changes(50.0, vec![
            (1000.0, 100.0),
            (2000.0, 75.0),
        ]);

        assert!((river.get_discharge_at(0.0) - 50.0).abs() < 1e-10);
        assert!((river.get_discharge_at(500.0) - 50.0).abs() < 1e-10);
        assert!((river.get_discharge_at(1500.0) - 100.0).abs() < 1e-10);
        assert!((river.get_discharge_at(2500.0) - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_river_direction() {
        let river = RiverProvider::constant(100.0).with_direction(90.0);
        assert!((river.get_direction() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_velocity_components() {
        let river = RiverProvider::constant(100.0).with_direction(0.0);
        let (u, v) = river.get_velocity_components(0.0, 10.0);

        // 流速 = 100/10 = 10 m/s，方向 0 度
        assert!((u - 10.0).abs() < 1e-10);
        assert!(v.abs() < 1e-10);
    }

    #[test]
    fn test_peak_discharge() {
        let river = RiverProvider::flood_wave(10.0, 100.0, 3600.0, 7200.0);
        assert!((river.peak_discharge() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_river_system() {
        let mut system = RiverSystem::new();

        system.add_river("River1", vec![0, 1], RiverProvider::constant(100.0));
        system.add_river("River2", vec![5], RiverProvider::constant(50.0));

        assert_eq!(system.count(), 2);
        assert!((system.total_discharge(0.0) - 150.0).abs() < 1e-10);
    }

    #[test]
    fn test_river_system_cell_inflow() {
        let mut system = RiverSystem::new();
        system.add_river("River1", vec![0, 1], RiverProvider::constant(100.0));

        // 流量均分到两个单元
        let q = system.get_cell_inflow(0, 0.0);
        assert!((q - 50.0).abs() < 1e-10);
    }


    #[test]
    fn test_river_system_apply_with_momentum() {
        let mut system = RiverSystem::new();
        // 创建向东(0度)流动的河流
        system.add_river("River1", vec![0], RiverProvider::constant(100.0).with_direction(0.0));

        let mut h = vec![1.0; 10];
        let mut hu = vec![0.0; 10];
        let mut hv = vec![0.0; 10];
        let areas = vec![100.0; 10]; // 100 m² 单元

        // 提供过水断面积 = 10 m²
        let cross_areas = vec![10.0; 10];

        // dt = 1s, Q = 100 m³/s
        // 入流速度 u_in = Q / A_cross = 100 / 10 = 10 m/s（向东）
        // Δh = Q * dt / A = 100 * 1 / 100 = 1.0 m
        // Δ(hu) = Q * u_in * dt / A = 100 * 10 * 1 / 100 = 10.0 m²/s
        system.apply_inflow_with_momentum(
            0.0, 1.0, &mut h, &mut hu, &mut hv, &areas, Some(&cross_areas)
        );

        assert!((h[0] - 2.0).abs() < 1e-10, "h = {}", h[0]);
        assert!((hu[0] - 10.0).abs() < 1e-10, "hu = {}", hu[0]);
        assert!(hv[0].abs() < 1e-10, "hv = {}", hv[0]);
    }

    #[test]
    fn test_river_system_apply_momentum_angled() {
        let mut system = RiverSystem::new();
        // 创建45度角流动的河流
        system.add_river("River1", vec![0], RiverProvider::constant(100.0).with_direction(45.0));

        let mut h = vec![1.0; 10];
        let mut hu = vec![0.0; 10];
        let mut hv = vec![0.0; 10];
        let areas = vec![100.0; 10];
        let cross_areas = vec![10.0; 10];

        system.apply_inflow_with_momentum(
            0.0, 1.0, &mut h, &mut hu, &mut hv, &areas, Some(&cross_areas)
        );

        // u_in = 10 m/s, 方向45度
        // hu 增量 = Q * u_in * cos(45°) * dt / A = 100 * 10 * 0.7071 * 1 / 100 ≈ 7.07
        let sqrt2_2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((hu[0] - 10.0 * sqrt2_2).abs() < 1e-6, "hu = {}", hu[0]);
        assert!((hv[0] - 10.0 * sqrt2_2).abs() < 1e-6, "hv = {}", hv[0]);
    }

    #[test]
    fn test_river_fast_inflow() {
        let mut system = RiverSystem::new();
        system.add_river("River1", vec![0], RiverProvider::constant(10.0).with_direction(0.0));

        let mut h = vec![1.0; 10];
        let mut hu = vec![0.0; 10];
        let mut hv = vec![0.0; 10];
        let areas = vec![100.0; 10];

        // 使用快速版本（自动估算过水断面）
        system.apply_inflow_fast(0.0, 1.0, &mut h, &mut hu, &mut hv, &areas);

        // 水深应该增加
        assert!(h[0] > 1.0);
        // 动量应该增加（向东）
        assert!(hu[0] > 0.0);
        assert!(hv[0].abs() < 1e-10);
    }
}
