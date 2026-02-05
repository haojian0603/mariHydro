// crates/mh_physics/src/waves/spectral.rs

//! 谱波模型框架（基础版本）
//!
//! 提供波能谱与诊断参数计算，为外部谱波模型耦合与后续扩展提供入口。

use std::f64::consts::PI;
use crate::waves::radiation_stress::{compute_wavenumber_and_n, RadiationStressTensorGeneric};
use crate::prelude::*;

/// 波能谱
#[derive(Debug, Clone)]
pub struct WaveSpectrum {
    /// 频率分档 [Hz]
    pub frequencies: Vec<f64>,
    /// 方向分档 [rad]
    pub directions: Vec<f64>,
    /// 能量密度 E(f, θ)
    pub energy: Vec<Vec<f64>>, // [n_freq][n_dir]
}

impl WaveSpectrum {
    /// 创建空谱
    pub fn new(n_freq: usize, n_dir: usize) -> Self {
        let f_min: f64 = 0.04;
        let f_max: f64 = 1.0;
        let df = (f_max / f_min).ln() / (n_freq.saturating_sub(1).max(1) as f64);

        let frequencies: Vec<f64> = (0..n_freq)
            .map(|i| f_min * (df * i as f64).exp())
            .collect();

        let directions: Vec<f64> = (0..n_dir)
            .map(|i| 2.0 * PI * i as f64 / n_dir.max(1) as f64)
            .collect();

        Self {
            frequencies,
            directions,
            energy: vec![vec![0.0; n_dir]; n_freq],
        }
    }

    /// 从 JONSWAP 谱初始化
    pub fn from_jonswap(
        n_freq: usize,
        n_dir: usize,
        hs: f64,
        tp: f64,
        gamma: f64,
        mean_dir: f64,
        spread: f64,
    ) -> Self {
        let mut spectrum = Self::new(n_freq, n_dir);
        let fp = 1.0 / tp.max(1e-6);
        let g = 9.81;

        let alpha = 0.0624
            / (0.23 + 0.0336 * gamma - 0.185 / (1.9 + gamma))
            * (1.094 - 0.01915 * gamma.ln());

        for i in 0..n_freq {
            let f = spectrum.frequencies[i];
            let sigma = if f < fp { 0.07 } else { 0.09 };
            let r = (-(f - fp).powi(2) / (2.0 * sigma * sigma * fp * fp)).exp();
            let s_j = alpha * g * g / (f.powi(5) * (16.0 * PI.powi(4)))
                * (-1.25 * (fp / f).powi(4)).exp()
                * gamma.powf(r);

            for j in 0..n_dir {
                let theta = spectrum.directions[j];
                let dtheta = (theta - mean_dir).cos().max(0.0);
                let dir_spread = dtheta.powf(spread);
                spectrum.energy[i][j] = (s_j * dir_spread).max(0.0);
            }
        }

        // 归一化到 Hs
        let hs_calc = spectrum.significant_height();
        if hs_calc > 1e-12 {
            let factor = (hs / hs_calc).powi(2);
            for i in 0..n_freq {
                for j in 0..n_dir {
                    spectrum.energy[i][j] *= factor;
                }
            }
        }

        spectrum
    }

    /// 有效波高
    pub fn significant_height(&self) -> f64 {
        4.0 * self.zeroth_moment().sqrt()
    }

    /// 零阶矩
    pub fn zeroth_moment(&self) -> f64 {
        let n_freq = self.frequencies.len();
        let n_dir = self.directions.len();
        let df = self.frequency_spacing();
        let dtheta = 2.0 * PI / n_dir.max(1) as f64;

        let mut m0 = 0.0;
        for i in 0..n_freq {
            for j in 0..n_dir {
                let e = self.energy[i][j];
                if e.is_finite() && e > 0.0 {
                    m0 += e * df[i] * dtheta;
                }
            }
        }
        m0
    }

    /// 频率间距
    fn frequency_spacing(&self) -> Vec<f64> {
        let n = self.frequencies.len();
        let mut df = vec![0.0; n];
        if n == 0 { return df; }
        if n == 1 {
            df[0] = 0.0;
            return df;
        }
        df[0] = self.frequencies[1] - self.frequencies[0];
        for i in 1..n - 1 {
            df[i] = 0.5 * (self.frequencies[i + 1] - self.frequencies[i - 1]);
        }
        df[n - 1] = self.frequencies[n - 1] - self.frequencies[n - 2];
        df
    }

    /// 峰值周期
    pub fn peak_period(&self) -> f64 {
        let mut max_e = 0.0;
        let mut f_peak = self.frequencies.first().copied().unwrap_or(1.0);
        for (i, f) in self.frequencies.iter().copied().enumerate() {
            let e_sum: f64 = self.energy[i].iter().sum();
            if e_sum > max_e {
                max_e = e_sum;
                f_peak = f;
            }
        }
        1.0 / f_peak.max(1e-6)
    }

    /// 主波向
    pub fn mean_direction(&self) -> f64 {
        let mut sin_sum = 0.0;
        let mut cos_sum = 0.0;
        for i in 0..self.frequencies.len() {
            for j in 0..self.directions.len() {
                let e = self.energy[i][j];
                if e.is_finite() && e > 0.0 {
                    sin_sum += e * self.directions[j].sin();
                    cos_sum += e * self.directions[j].cos();
                }
            }
        }
        sin_sum.atan2(cos_sum)
    }
}

/// 谱波配置
#[derive(Debug, Clone)]
pub struct SpectralConfig {
    pub n_freq: usize,
    pub n_dir: usize,
    pub breaking_gamma: f64,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            n_freq: 25,
            n_dir: 36,
            breaking_gamma: 0.73,
        }
    }
}

/// 波场诊断参数
#[derive(Debug, Clone)]
pub struct WaveFieldParams {
    pub hs: Vec<f64>,
    pub tp: Vec<f64>,
    pub dir: Vec<f64>,
    pub sxx: Vec<f64>,
    pub sxy: Vec<f64>,
    pub syy: Vec<f64>,
}

impl WaveFieldParams {
    pub fn new(n_cells: usize) -> Self {
        Self {
            hs: vec![0.0; n_cells],
            tp: vec![8.0; n_cells],
            dir: vec![0.0; n_cells],
            sxx: vec![0.0; n_cells],
            sxy: vec![0.0; n_cells],
            syy: vec![0.0; n_cells],
        }
    }
}

/// 谱波求解器（框架）
#[derive(Debug, Clone)]
pub struct SpectralWaveSolver {
    pub config: SpectralConfig,
    pub spectra: Vec<WaveSpectrum>,
    pub params: WaveFieldParams,
}

impl SpectralWaveSolver {
    pub fn new(n_cells: usize, config: SpectralConfig) -> Self {
        let spectra = (0..n_cells)
            .map(|_| WaveSpectrum::new(config.n_freq, config.n_dir))
            .collect();
        Self {
            config,
            spectra,
            params: WaveFieldParams::new(n_cells),
        }
    }

    /// 更新诊断参数
    pub fn update_params(&mut self) {
        for (i, spectrum) in self.spectra.iter().enumerate() {
            self.params.hs[i] = spectrum.significant_height();
            self.params.tp[i] = spectrum.peak_period();
            self.params.dir[i] = spectrum.mean_direction();
        }
    }

    /// 计算辐射应力（基于深度）
    pub fn compute_radiation_stress<B: Backend>(&mut self, backend: &B, depth: &B::Buffer<B::Scalar>) {
        let rho = 1025.0;
        let g = 9.81;
        let depth = match depth.try_as_slice() {
            Some(s) => s,
            None => {
                for v in &mut self.params.sxx { *v = 0.0; }
                for v in &mut self.params.sxy { *v = 0.0; }
                for v in &mut self.params.syy { *v = 0.0; }
                return;
            }
        };

        for i in 0..self.params.hs.len().min(depth.len()) {
            let h = depth[i].to_f64_lossy().max(0.1);
            let hs_raw = self.params.hs[i];
            let tp = self.params.tp[i];
            if !hs_raw.is_finite() || !tp.is_finite() || tp <= 0.0 {
                self.params.sxx[i] = 0.0;
                self.params.sxy[i] = 0.0;
                self.params.syy[i] = 0.0;
                continue;
            }
            let hs = hs_raw.min(self.config.breaking_gamma * h).max(0.0);
            let dir = self.params.dir[i];
            let omega = 2.0 * PI / tp.max(1e-6);
            let omega_s = backend.scalar_from_f64(omega);
            let h_s = backend.scalar_from_f64(h);
            let (_k, n) = compute_wavenumber_and_n(backend, omega_s, h_s);
            let energy = rho * g * hs * hs / 8.0;
            let stress = RadiationStressTensorGeneric::<f64>::compute(energy, n.to_f64_lossy(), dir);
            self.params.sxx[i] = stress.sxx;
            self.params.sxy[i] = stress.sxy;
            self.params.syy[i] = stress.syy;
        }
    }
}
