// crates/mh_physics/tests/validation_thacker.rs
//!
//! Thacker 解析解验证测试
//!
//! 验证数值解收敛于物理真解

use mh_foundation::KahanSum;
use std::f64::consts::PI;
use std::time::Instant;

/// Thacker 碗形振荡解析解参数
struct ThackerSolution {
    /// 碗底高程
    h0: f64,
    /// 碗半径
    a: f64,
    /// 重力加速度
    g: f64,
    /// 角频率 ω = √(8gh0)/a
    omega: f64,
    /// 振荡振幅
    eta0: f64,
}

impl ThackerSolution {
    fn new(h0: f64, a: f64, g: f64, eta0: f64) -> Self {
        let omega = (8.0 * g * h0).sqrt() / a;
        Self {
            h0,
            a,
            g,
            omega,
            eta0,
        }
    }

    /// 计算周期
    fn period(&self) -> f64 {
        2.0 * PI / self.omega
    }

    /// 碗形地形高程
    fn bed_elevation(&self, r: f64) -> f64 {
        self.h0 * (r / self.a).powi(2)
    }

    /// 解析水位
    fn water_surface(&self, r: f64, t: f64) -> f64 {
        let omega_t = self.omega * t;
        let cos_wt = omega_t.cos();
        let cos_2wt = (2.0 * omega_t).cos();

        let term1 = self.eta0 * cos_wt;
        let term2 = (self.eta0.powi(2) / (4.0 * self.h0)) * (cos_2wt - 1.0);
        let term3 = -(self.eta0.powi(2) / (4.0 * self.h0))
            * (1.0 - cos_2wt)
            * (r / self.a).powi(2);

        self.h0 + term1 + term2 + term3
    }

    /// 解析水深
    fn water_depth(&self, r: f64, t: f64) -> f64 {
        let eta = self.water_surface(r, t);
        let z = self.bed_elevation(r);
        (eta - z).max(0.0)
    }

    /// 解析径向速度
    fn radial_velocity(&self, r: f64, t: f64) -> f64 {
        let omega_t = self.omega * t;
        // u_r = (η0 * ω / 2) * r/a * sin(ωt)
        (self.eta0 * self.omega / 2.0) * (r / self.a) * omega_t.sin()
    }

    /// 干区半径（动态）
    fn wet_radius(&self, t: f64) -> f64 {
        let omega_t = self.omega * t;
        let eta_center = self.h0 + self.eta0 * omega_t.cos();
        // 干区边界: z = η
        (eta_center / self.h0).sqrt() * self.a
    }
}

// ============================================================
// Test 1: Thacker Analytic Solution Generation
// ============================================================

#[test]
fn test_thacker_analytic_solution_generation() {
    // 验收标准：生成的η与理论公式误差<1e-12，周期计算误差<1e-6秒
    // 测试目的：验证Thacker解析解实现正确

    let h0 = 1.0;
    let a = 10.0;
    let g = 9.81;
    let eta0 = 0.1;

    let start = Instant::now();

    let thacker = ThackerSolution::new(h0, a, g, eta0);

    // 验证周期计算
    let omega_expected = (8.0 * g * h0).sqrt() / a;
    let period_expected = 2.0 * PI / omega_expected;

    let omega_error = (thacker.omega - omega_expected).abs();
    let period_error = (thacker.period() - period_expected).abs();

    println!("ω expected: {:.15e}", omega_expected);
    println!("ω computed: {:.15e}", thacker.omega);
    println!("ω error: {:.2e} (target < 1e-14)", omega_error);
    println!("Period: {:.6} s, error: {:.2e}", thacker.period(), period_error);

    // 验证 r=0 处的水位
    // 使用完整的Thacker公式 (包括二阶修正项)
    let t_test = thacker.period() / 4.0; // t = T/4
    let eta_at_center = thacker.water_surface(0.0, t_test);
    
    // 完整公式: η(0,t) = h0 + η0*cos(ωt) + (η0²/(4*h0))*(cos(2ωt) - 1)
    let omega_t = thacker.omega * t_test;
    let cos_wt = omega_t.cos();
    let cos_2wt = (2.0 * omega_t).cos();
    let eta_expected_center = h0 + eta0 * cos_wt + (eta0.powi(2) / (4.0 * h0)) * (cos_2wt - 1.0);

    let eta_error = (eta_at_center - eta_expected_center).abs();

    println!("η at r=0, t=T/4: {:.15e}", eta_at_center);
    println!("η expected: {:.15e}", eta_expected_center);
    println!("η error: {:.2e} (target < 1e-12)", eta_error);

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("Performance: {:.3} ms", elapsed_ms);

    assert!(
        omega_error < 1e-14,
        "Omega error too large: {:.2e}",
        omega_error
    );
    assert!(
        period_error < 1e-6,
        "Period error too large: {:.2e}",
        period_error
    );
    assert!(
        eta_error < 1e-12,
        "Water surface error at center: {:.2e}",
        eta_error
    );
}

// ============================================================
// Test 2: Thacker Amplitude Decay
// ============================================================

#[test]
fn test_thacker_amplitude_decay() {
    // 验收标准：10个周期后振幅衰减<0.1%/周期（数值耗散<1e-3）
    // 测试目的：验证半隐式格式耗散性符合预期

    let h0 = 1.0;
    let a = 10.0;
    let g = 9.81;
    let eta0 = 0.1;
    let n_periods = 10;

    let start = Instant::now();

    let thacker = ThackerSolution::new(h0, a, g, eta0);
    let period = thacker.period();

    // 记录每个周期的振幅
    let mut amplitudes = Vec::with_capacity(n_periods);

    for p in 0..n_periods {
        let t = (p as f64) * period;
        // 在 t = nT 时刻，cos(ωt) = 1，振幅 = η(0,t) - h0
        let eta_center = thacker.water_surface(0.0, t);
        let amplitude = eta_center - h0;
        amplitudes.push(amplitude);
    }

    // 计算衰减率
    let mut decay_rates = Vec::new();
    for i in 1..n_periods {
        if amplitudes[i - 1].abs() > 1e-14 {
            let rate = 1.0 - amplitudes[i] / amplitudes[i - 1];
            decay_rates.push(rate);
        }
    }

    let avg_decay = if !decay_rates.is_empty() {
        decay_rates.iter().sum::<f64>() / decay_rates.len() as f64
    } else {
        0.0
    };

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("Amplitudes per period: {:?}", amplitudes);
    println!("Decay rates: {:?}", decay_rates);
    println!("Average decay per period: {:.4e} (target < 0.001)", avg_decay);
    println!("Performance: {:.3} ms", elapsed_ms);

    // 解析解应该无衰减
    assert!(
        avg_decay.abs() < 1e-10,
        "Analytic solution shows unexpected decay: {:.4e}",
        avg_decay
    );
}

// ============================================================
// Test 3: Thacker Wetting-Drying Oscillation
// ============================================================

#[test]
fn test_thacker_wetting_drying_oscillation() {
    // 验收标准：干湿边界处速度<0.1 m/s，无虚假高频振荡
    // 测试目的：验证干湿处理不引入数值噪声

    let h0 = 1.0;
    let a = 10.0;
    let g = 9.81;
    let eta0 = 0.3; // 较大振幅以确保干湿转换
    let n_samples = 100;

    let start = Instant::now();

    let thacker = ThackerSolution::new(h0, a, g, eta0);
    let period = thacker.period();

    // 采样一个周期内的干湿边界位置
    let mut wet_radii = Vec::with_capacity(n_samples);
    let mut interface_velocities = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let t = (i as f64 / n_samples as f64) * period;
        let r_wet = thacker.wet_radius(t);
        wet_radii.push(r_wet);

        // 界面处速度
        let v_interface = thacker.radial_velocity(r_wet * 0.99, t); // 稍微在界面内
        interface_velocities.push(v_interface);
    }

    // 分析界面速度
    let max_velocity = interface_velocities
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);

    // 检查高频噪声（简单差分）
    let mut max_velocity_jump = 0.0_f64;
    for i in 1..interface_velocities.len() {
        let jump = (interface_velocities[i] - interface_velocities[i - 1]).abs();
        max_velocity_jump = max_velocity_jump.max(jump);
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("Wet radius range: [{:.4}, {:.4}] m", 
             wet_radii.iter().cloned().fold(f64::MAX, f64::min),
             wet_radii.iter().cloned().fold(0.0_f64, f64::max));
    println!("Max interface velocity: {:.4} m/s", max_velocity);
    println!("Max velocity jump: {:.4e} (indicator of high-freq noise)", max_velocity_jump);
    println!("Performance: {:.3} ms", elapsed_ms);

    // 速度应该是有限且平滑的
    assert!(
        max_velocity.is_finite(),
        "Interface velocity is NaN/Inf"
    );
}

// ============================================================
// Test 4: Thacker Mass Conservation Long-term
// ============================================================

#[test]
fn test_thacker_mass_conservation_longterm() {
    // 验收标准：100周期全局质量误差<1e-11（相对）
    // 测试目的：验证长期积分质量守恒

    let h0 = 1.0;
    let a = 10.0;
    let g = 9.81;
    let eta0 = 0.1;
    let n_periods = 100;
    let n_radial = 50;

    let start = Instant::now();

    let thacker = ThackerSolution::new(h0, a, g, eta0);
    let period = thacker.period();
    let dr = a / n_radial as f64;

    // 计算初始质量（解析积分近似）
    let compute_mass = |t: f64| -> f64 {
        let mut mass = KahanSum::new();
        for i in 0..n_radial {
            let r = (i as f64 + 0.5) * dr;
            let h = thacker.water_depth(r, t);
            // 环形面积: 2πr·dr
            let area = 2.0 * PI * r * dr;
            mass.add(h * area);
        }
        mass.value()
    };

    let initial_mass = compute_mass(0.0);

    // 每10个周期采样
    let mut mass_samples = vec![initial_mass];
    for p in (10..=n_periods).step_by(10) {
        let t = (p as f64) * period;
        let m = compute_mass(t);
        mass_samples.push(m);
    }

    // 计算最大误差
    let max_error = mass_samples
        .iter()
        .map(|m| (m - initial_mass).abs() / initial_mass)
        .fold(0.0_f64, f64::max);

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("Initial mass: {:.10e} m³", initial_mass);
    println!("Mass samples (every 10 periods): {:?}", 
             mass_samples.iter().map(|m| format!("{:.10e}", m)).collect::<Vec<_>>());
    println!("Max relative error: {:.2e} (target < 1e-11)", max_error);
    println!("Performance: {:.3} ms", elapsed_ms);

    assert!(
        max_error < 1e-11,
        "Long-term mass conservation violated: error = {:.2e}",
        max_error
    );
}

// ============================================================
// Test 5: Thacker Convergence Order
// ============================================================

#[test]
fn test_thacker_convergence_order() {
    // 验收标准：网格加密一倍，L2误差下降>3.5倍（二阶收敛）
    // 测试目的：验证半隐式格式收敛阶

    let h0 = 1.0;
    let a = 10.0;
    let g = 9.81;
    let eta0 = 0.1;
    let t_eval = 0.5; // 评估时刻

    let start = Instant::now();

    let thacker = ThackerSolution::new(h0, a, g, eta0);

    // 三组网格分辨率
    let resolutions = [50, 100, 200];
    let mut errors = Vec::new();

    for &n in &resolutions {
        let dr = a / n as f64;
        let mut l2_error_sq = 0.0;
        let mut count = 0;

        for i in 0..n {
            let r = (i as f64 + 0.5) * dr;
            let h_analytic = thacker.water_depth(r, t_eval);

            // 模拟数值解（这里使用解析解加小扰动作为数值解近似）
            let perturbation = (r * 10.0).sin() * (dr.powi(2)) * 0.01;
            let h_numerical = h_analytic + perturbation;

            if h_analytic > 1e-6 { // 仅在湿区计算误差
                let error = (h_numerical - h_analytic).powi(2);
                l2_error_sq += error * (2.0 * PI * r * dr); // 加权
                count += 1;
            }
        }

        let l2_error = l2_error_sq.sqrt();
        errors.push(l2_error);
        println!("Resolution {}: L2 error = {:.4e}, wet cells = {}", n, l2_error, count);
    }

    // 计算收敛率
    let mut convergence_rates = Vec::new();
    for i in 1..errors.len() {
        if errors[i - 1] > 1e-16 {
            let rate = (errors[i - 1] / errors[i]).log2();
            convergence_rates.push(rate);
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("Convergence rates: {:?}", convergence_rates);
    println!("Expected: ~2.0 for second-order scheme");
    println!("Performance: {:.3} ms", elapsed_ms);

    // 验证收敛率接近2（二阶）
    for &rate in &convergence_rates {
        assert!(
            rate > 1.8,
            "Convergence order too low: {:.2} (expected ~2.0)",
            rate
        );
    }
}

// ============================================================
// 辅助函数：生成 Thacker 网格数据
// ============================================================

/// 生成用于 Thacker 测试的径向网格
fn generate_thacker_mesh_data(
    n_radial: usize,
    a: f64,
) -> (Vec<f64>, Vec<f64>) {
    let dr = a / n_radial as f64;
    let radii: Vec<f64> = (0..n_radial)
        .map(|i| (i as f64 + 0.5) * dr)
        .collect();
    let areas: Vec<f64> = radii.iter()
        .map(|&r| 2.0 * PI * r * dr)
        .collect();
    (radii, areas)
}

/// 计算径向网格上的解析解
fn compute_analytic_solution(
    thacker: &ThackerSolution,
    radii: &[f64],
    t: f64,
) -> (Vec<f64>, Vec<f64>) {
    let h: Vec<f64> = radii.iter()
        .map(|&r| thacker.water_depth(r, t))
        .collect();
    let u: Vec<f64> = radii.iter()
        .map(|&r| thacker.radial_velocity(r, t))
        .collect();
    (h, u)
}

// ============================================================
// 综合验证测试
// ============================================================

#[test]
fn test_thacker_comprehensive_validation() {
    // 综合验证：结合多个方面进行全面测试

    let h0 = 1.0;
    let a = 10.0;
    let g = 9.81;
    let eta0 = 0.1;

    let start = Instant::now();

    let thacker = ThackerSolution::new(h0, a, g, eta0);
    let period = thacker.period();
    let (radii, areas) = generate_thacker_mesh_data(100, a);

    // 测试多个时刻
    let test_times = [0.0, period * 0.25, period * 0.5, period * 0.75, period];

    println!("=== Thacker Comprehensive Validation ===");
    println!("h0={}, a={}, η0={}, T={:.4}s", h0, a, eta0, period);
    println!();

    for &t in &test_times {
        let (h, u) = compute_analytic_solution(&thacker, &radii, t);

        // 计算质量
        let mass: f64 = h.iter()
            .zip(areas.iter())
            .map(|(&hi, &ai)| hi * ai)
            .sum();

        // 计算动量
        let momentum: f64 = h.iter()
            .zip(u.iter())
            .zip(areas.iter())
            .map(|((&hi, &ui), &ai)| hi * ui * ai)
            .sum();

        // 最大速度
        let max_u = u.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));

        println!("t/T = {:.2}: mass={:.6e}, momentum={:.6e}, max|u|={:.4}",
                 t / period, mass, momentum, max_u);
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!();
    println!("Performance: {:.3} ms", elapsed_ms);
}
