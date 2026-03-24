// crates/mh_physics/src/waves/radiation_stress.rs

//! 波浪辐射应力计算
//!
//! 实现波浪辐射应力张量及其梯度计算，用于波流耦合模拟。

use crate::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

fn scalar_const<B: Backend>(backend: &B, v: f64, context: &'static str) -> B::Scalar {
    backend.config_scalar(v, context)
}

fn scalar_pi<B: Backend>(backend: &B) -> B::Scalar {
    scalar_const(backend, std::f64::consts::PI, "wave.pi")
}

fn gravity<B: Backend>(backend: &B) -> B::Scalar {
    scalar_const(backend, 9.81, "wave.gravity")
}

fn rho_water<B: Backend>(backend: &B) -> B::Scalar {
    scalar_const(backend, 1025.0, "wave.rho_water")
}

/// 波场错误
#[derive(Debug, Clone)]
pub enum WaveFieldError {
    /// 后端缓冲区不可直接访问
    BackendAccess(String),
    /// 长度不匹配
    SizeMismatch { expected: usize, actual: usize },
}

/// 波浪参数（泛型）
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WaveParametersGeneric<S: RuntimeScalar> {
    /// 有效波高 H [m]
    pub height: S,
    /// 波浪周期 T [s]
    pub period: S,
    /// 波向 θ [弧度]，从正北顺时针测量
    pub direction: S,
}

impl<S: RuntimeScalar> WaveParametersGeneric<S> {
    /// 创建新的波浪参数
    pub fn new(height: S, period: S, direction: S) -> Self {
        Self {
            height,
            period,
            direction,
        }
    }

    /// 角频率 ω = 2π/T
    pub fn angular_frequency<B: Backend<Scalar = S>>(&self, backend: &B) -> S {
        S::TWO * scalar_pi(backend) / self.period
    }

    /// 深水波长 L0 = gT²/(2π)
    pub fn deep_water_wavelength<B: Backend<Scalar = S>>(&self, backend: &B) -> S {
        gravity(backend) * self.period * self.period / (S::TWO * scalar_pi(backend))
    }

    /// 波浪能量密度 E = ρgH²/8
    pub fn energy<B: Backend<Scalar = S>>(&self, backend: &B) -> S {
        let eight = S::TWO * S::TWO * S::TWO;
        rho_water(backend) * gravity(backend) * self.height * self.height / eight
    }

    /// 波向单位向量 (x, y)
    pub fn direction_vector(&self) -> (S, S) {
        (self.direction.sin(), self.direction.cos())
    }
}

impl<S: RuntimeScalar> WaveParametersGeneric<S> {
    /// 使用后端默认值创建
    pub fn with_backend_defaults<B: Backend<Scalar = S>>(backend: &B) -> Self {
        Self {
            height: scalar_const(backend, 1.0, "WaveParametersGeneric.default_height"),
            period: scalar_const(backend, 8.0, "WaveParametersGeneric.default_period"),
            direction: S::ZERO,
        }
    }
}

/// 波场数据（Backend 感知）
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "B::Buffer<B::Scalar>: Serialize, B::Scalar: Serialize",
    deserialize = "B::Buffer<B::Scalar>: DeserializeOwned, B::Scalar: DeserializeOwned, B: Default"
))]
pub struct WaveFieldGeneric<B: Backend> {
    /// 波高 [m]
    pub height: B::Buffer<B::Scalar>,
    /// 周期 [s]
    pub period: B::Buffer<B::Scalar>,
    /// 波向 [弧度]
    pub direction: B::Buffer<B::Scalar>,
    /// 波长 [m]
    pub wavelength: B::Buffer<B::Scalar>,
    /// 波数 k [1/m]
    pub wavenumber: B::Buffer<B::Scalar>,
    /// 群速度因子 n = Cg/C
    pub group_factor: B::Buffer<B::Scalar>,
    /// 能量密度 [J/m²]
    pub energy: B::Buffer<B::Scalar>,
    /// 后端实例
    #[serde(skip, default)]
    backend: B,
}

/// 波场快照（序列化用）
///
/// 几何数据保持 f64 精度，物理场使用显式标量类型。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "S: Serialize", deserialize = "S: DeserializeOwned"))]
pub struct WaveFieldSnapshot<S: RuntimeScalar> {
    /// 单元数量
    pub n_cells: usize,
    /// 波高 [m]
    pub height: Vec<S>,
    /// 周期 [s]
    pub period: Vec<S>,
    /// 波向 [弧度]
    pub direction: Vec<S>,
    /// 波长 [m]
    pub wavelength: Vec<S>,
    /// 波数 k [1/m]
    pub wavenumber: Vec<S>,
    /// 群速度因子 n = Cg/C
    pub group_factor: Vec<S>,
    /// 能量密度 [J/m²]
    pub energy: Vec<S>,
}

impl<B: Backend> WaveFieldGeneric<B> {
    /// 创建指定大小的波场
    pub fn new(backend: B, n_cells: usize) -> Self {
        let mut height = backend.alloc(n_cells);
        let mut period = backend.alloc(n_cells);
        let mut direction = backend.alloc(n_cells);
        let mut wavelength = backend.alloc(n_cells);
        let mut wavenumber = backend.alloc(n_cells);
        let mut group_factor = backend.alloc(n_cells);
        let mut energy = backend.alloc(n_cells);

        height.fill(B::Scalar::ZERO);
        period.fill(scalar_const(
            &backend,
            8.0,
            "WaveFieldGeneric.default_period",
        ));
        direction.fill(B::Scalar::ZERO);
        wavelength.fill(B::Scalar::ZERO);
        wavenumber.fill(B::Scalar::ZERO);
        group_factor.fill(B::Scalar::HALF);
        energy.fill(B::Scalar::ZERO);

        Self {
            height,
            period,
            direction,
            wavelength,
            wavenumber,
            group_factor,
            energy,
            backend,
        }
    }

    /// 从均匀参数创建波场
    pub fn from_uniform(
        backend: B,
        n_cells: usize,
        params: &WaveParametersGeneric<B::Scalar>,
    ) -> Self {
        let mut field = Self::new(backend, n_cells);
        field.set_uniform(params);
        field
    }

    /// 设置均匀波浪参数
    pub fn set_uniform(&mut self, params: &WaveParametersGeneric<B::Scalar>) {
        self.height.fill(params.height);
        self.period.fill(params.period);
        self.direction.fill(params.direction);
        self.energy.fill(params.energy(&self.backend));

        // 波长和波数需要根据水深计算，这里先用深水近似
        let l0 = params.deep_water_wavelength(&self.backend);
        self.wavelength.fill(l0);
        self.wavenumber
            .fill(B::Scalar::TWO * scalar_pi(&self.backend) / l0);
        self.group_factor.fill(B::Scalar::HALF); // 深水近似
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        self.height.resize(n_cells, B::Scalar::ZERO);
        self.period.resize(
            n_cells,
            scalar_const(&self.backend, 8.0, "WaveFieldGeneric.default_period"),
        );
        self.direction.resize(n_cells, B::Scalar::ZERO);
        self.wavelength.resize(n_cells, B::Scalar::ZERO);
        self.wavenumber.resize(n_cells, B::Scalar::ZERO);
        self.group_factor.resize(n_cells, B::Scalar::HALF);
        self.energy.resize(n_cells, B::Scalar::ZERO);
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
    pub fn update_dispersion(
        &mut self,
        depth: &B::Buffer<B::Scalar>,
    ) -> Result<(), WaveFieldError> {
        let h_min = self
            .backend
            .config_scalar(0.1, "WaveFieldGeneric.update_dispersion.h_min");
        let depth = depth.try_as_slice().ok_or_else(|| {
            WaveFieldError::BackendAccess("depth buffer not accessible".to_string())
        })?;
        let n = self.len().min(depth.len());

        let height = self.height.try_as_slice().ok_or_else(|| {
            WaveFieldError::BackendAccess("height buffer not accessible".to_string())
        })?;
        let period = self.period.try_as_slice().ok_or_else(|| {
            WaveFieldError::BackendAccess("period buffer not accessible".to_string())
        })?;
        let wavenumber = self.wavenumber.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("wavenumber buffer not accessible".to_string())
        })?;
        let group_factor = self.group_factor.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("group_factor buffer not accessible".to_string())
        })?;
        let wavelength = self.wavelength.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("wavelength buffer not accessible".to_string())
        })?;
        let energy = self.energy.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("energy buffer not accessible".to_string())
        })?;

        for i in 0..n {
            let h = if depth[i] > h_min { depth[i] } else { h_min };
            let omega = B::Scalar::TWO * scalar_pi(&self.backend) / period[i];

            let (k, n_factor) = compute_wavenumber_and_n(&self.backend, omega, h);
            wavenumber[i] = k;
            group_factor[i] = n_factor;
            wavelength[i] = B::Scalar::TWO * scalar_pi(&self.backend) / k;

            energy[i] = rho_water(&self.backend) * gravity(&self.backend) * height[i] * height[i]
                / (B::Scalar::TWO * B::Scalar::TWO * B::Scalar::TWO);
        }
        Ok(())
    }

    /// 转换为快照（用于序列化）
    pub fn to_snapshot(&self) -> WaveFieldSnapshot<B::Scalar> {
        WaveFieldSnapshot {
            n_cells: self.len(),
            height: self.height.copy_to_vec(),
            period: self.period.copy_to_vec(),
            direction: self.direction.copy_to_vec(),
            wavelength: self.wavelength.copy_to_vec(),
            wavenumber: self.wavenumber.copy_to_vec(),
            group_factor: self.group_factor.copy_to_vec(),
            energy: self.energy.copy_to_vec(),
        }
    }

    /// 获取后端引用
    pub fn backend(&self) -> &B {
        &self.backend
    }
}

impl<S: RuntimeScalar> WaveFieldSnapshot<S> {
    /// 从快照重建后端波场
    pub fn to_backend<B: Backend<Scalar = S>>(
        self,
        backend: B,
    ) -> Result<WaveFieldGeneric<B>, WaveFieldError> {
        let n_cells = self.n_cells;
        let lengths = [
            self.height.len(),
            self.period.len(),
            self.direction.len(),
            self.wavelength.len(),
            self.wavenumber.len(),
            self.group_factor.len(),
            self.energy.len(),
        ];
        if lengths.iter().any(|&len| len != n_cells) {
            return Err(WaveFieldError::SizeMismatch {
                expected: n_cells,
                actual: lengths.into_iter().min().unwrap_or(0),
            });
        }
        let mut field = WaveFieldGeneric::new(backend, n_cells);
        field.height.copy_from_slice(&self.height);
        field.period.copy_from_slice(&self.period);
        field.direction.copy_from_slice(&self.direction);
        field.wavelength.copy_from_slice(&self.wavelength);
        field.wavenumber.copy_from_slice(&self.wavenumber);
        field.group_factor.copy_from_slice(&self.group_factor);
        field.energy.copy_from_slice(&self.energy);
        Ok(field)
    }

    /// 精度转换（显式）
    pub fn map_scalar<T: RuntimeScalar, B: Backend<Scalar = T>>(
        &self,
        backend: &B,
    ) -> WaveFieldSnapshot<T> {
        self.map_scalar_with(|v| backend.config_scalar(v.to_f64_lossy(), "WaveFieldSnapshot.map_scalar"))
    }

    /// 精度转换（自定义映射）
    pub fn map_scalar_with<T: RuntimeScalar, F: Fn(S) -> T>(&self, map: F) -> WaveFieldSnapshot<T> {
        let map_vec = |src: &[S]| -> Vec<T> { src.iter().copied().map(&map).collect() };
        WaveFieldSnapshot {
            n_cells: self.n_cells,
            height: map_vec(&self.height),
            period: map_vec(&self.period),
            direction: map_vec(&self.direction),
            wavelength: map_vec(&self.wavelength),
            wavenumber: map_vec(&self.wavenumber),
            group_factor: map_vec(&self.group_factor),
            energy: map_vec(&self.energy),
        }
    }
}

/// 求解色散关系 ω² = gk·tanh(kh)
///
/// 返回 (k, n)，其中 n = Cg/C = 群速度/相速度
pub fn compute_wavenumber_and_n<B: Backend>(
    backend: &B,
    omega: B::Scalar,
    depth: B::Scalar,
) -> (B::Scalar, B::Scalar) {
    if !omega.is_finite() || !depth.is_finite() {
        return (B::Scalar::ZERO, B::Scalar::ZERO);
    }

    let h = depth.max(backend.config_scalar(0.01, "wave.dispersion.min_depth"));
    let g = gravity(backend);
    let eps = backend.config_scalar(1e-10, "wave.dispersion.eps");

    // 初始猜测（深水近似）
    let mut k = (omega * omega / g).max(B::Scalar::ZERO);

    // Newton-Raphson 迭代
    for _ in 0..20 {
        let kh = k * h;
        let tanh_kh = kh.tanh();
        let f = omega * omega - g * k * tanh_kh;
        let df = -g * (tanh_kh + k * h * (B::Scalar::ONE - tanh_kh * tanh_kh));
        if !df.is_finite() || df.abs() <= eps {
            break;
        }

        let dk = -f / df;
        if !dk.is_finite() {
            break;
        }
        k = (k + dk).max(B::Scalar::ZERO);

        if dk.abs() < eps * k.max(B::Scalar::ONE) {
            break;
        }
    }

    // 群速度因子 n = Cg/C = 0.5(1 + 2kh/sinh(2kh))
    let kh = k * h;
    let sinh_2kh = (B::Scalar::TWO * kh).sinh();
    let n = if sinh_2kh.abs() > eps {
        B::Scalar::HALF * (B::Scalar::ONE + B::Scalar::TWO * kh / sinh_2kh)
    } else {
        B::Scalar::ONE // 浅水极限
    };

    (k, n)
}

/// 辐射应力张量（泛型）
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct RadiationStressTensorGeneric<S: RuntimeScalar> {
    /// Sxx 分量 [N/m]
    pub sxx: S,
    /// Syy 分量 [N/m]
    pub syy: S,
    /// Sxy = Syx 分量 [N/m]
    pub sxy: S,
}

impl<S: RuntimeScalar> RadiationStressTensorGeneric<S> {
    /// 创建零张量
    pub fn zero() -> Self {
        Self::default()
    }

    /// 计算辐射应力张量
    pub fn compute(energy: S, n: S, direction: S) -> Self {
        let cos_theta = direction.cos();
        let sin_theta = direction.sin();
        let cos2 = cos_theta * cos_theta;
        let sin2 = sin_theta * sin_theta;

        Self {
            sxx: energy * (n * (cos2 + S::ONE) - S::HALF),
            syy: energy * (n * (sin2 + S::ONE) - S::HALF),
            sxy: energy * n * sin_theta * cos_theta,
        }
    }

    /// 获取主应力
    pub fn principal_stresses(&self) -> (S, S) {
        let avg = S::HALF * (self.sxx + self.syy);
        let diff = S::HALF * (self.sxx - self.syy);
        let r = (diff * diff + self.sxy * self.sxy).sqrt();
        (avg + r, avg - r)
    }
}

/// 辐射应力计算器（泛型）
pub struct RadiationStressCalculatorGeneric<B: Backend> {
    /// 辐射应力 Sxx
    sxx: B::Buffer<B::Scalar>,
    /// 辐射应力 Syy
    syy: B::Buffer<B::Scalar>,
    /// 辐射应力 Sxy
    sxy: B::Buffer<B::Scalar>,
    /// 辐射应力梯度 x 分量（力/面积）
    force_x: B::Buffer<B::Scalar>,
    /// 辐射应力梯度 y 分量
    force_y: B::Buffer<B::Scalar>,
    /// 后端实例
    backend: B,
}

impl<B: Backend> RadiationStressCalculatorGeneric<B> {
    /// 创建新的计算器
    pub fn new(backend: B, n_cells: usize) -> Self {
        let mut sxx = backend.alloc(n_cells);
        let mut syy = backend.alloc(n_cells);
        let mut sxy = backend.alloc(n_cells);
        let mut force_x = backend.alloc(n_cells);
        let mut force_y = backend.alloc(n_cells);

        sxx.fill(B::Scalar::ZERO);
        syy.fill(B::Scalar::ZERO);
        sxy.fill(B::Scalar::ZERO);
        force_x.fill(B::Scalar::ZERO);
        force_y.fill(B::Scalar::ZERO);

        Self {
            sxx,
            syy,
            sxy,
            force_x,
            force_y,
            backend,
        }
    }

    /// 从波场计算辐射应力
    pub fn compute_stress(
        &mut self,
        wave_field: &WaveFieldGeneric<B>,
    ) -> Result<(), WaveFieldError> {
        let n = self.sxx.len().min(wave_field.len());
        let energy = wave_field.energy.try_as_slice().ok_or_else(|| {
            WaveFieldError::BackendAccess("energy buffer not accessible".to_string())
        })?;
        let group_factor = wave_field.group_factor.try_as_slice().ok_or_else(|| {
            WaveFieldError::BackendAccess("group_factor buffer not accessible".to_string())
        })?;
        let direction = wave_field.direction.try_as_slice().ok_or_else(|| {
            WaveFieldError::BackendAccess("direction buffer not accessible".to_string())
        })?;
        let sxx = self.sxx.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("sxx buffer not accessible".to_string())
        })?;
        let syy = self.syy.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("syy buffer not accessible".to_string())
        })?;
        let sxy = self.sxy.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("sxy buffer not accessible".to_string())
        })?;

        for i in 0..n {
            let tensor =
                RadiationStressTensorGeneric::compute(energy[i], group_factor[i], direction[i]);
            sxx[i] = tensor.sxx;
            syy[i] = tensor.syy;
            sxy[i] = tensor.sxy;
        }
        Ok(())
    }

    /// 计算辐射应力梯度（结构化网格）
    pub fn compute_gradient_structured(
        &mut self,
        nx: usize,
        ny: usize,
        dx: B::Scalar,
        dy: B::Scalar,
        depth: &B::Buffer<B::Scalar>,
    ) -> Result<(), WaveFieldError> {
        let inv_dx = B::Scalar::ONE / dx;
        let inv_dy = B::Scalar::ONE / dy;
        let h_min = self
            .backend
            .config_scalar(0.1, "RadiationStressCalculatorGeneric.h_min");

        let sxx = self.sxx.try_as_slice().ok_or_else(|| {
            WaveFieldError::BackendAccess("sxx buffer not accessible".to_string())
        })?;
        let syy = self.syy.try_as_slice().ok_or_else(|| {
            WaveFieldError::BackendAccess("syy buffer not accessible".to_string())
        })?;
        let sxy = self.sxy.try_as_slice().ok_or_else(|| {
            WaveFieldError::BackendAccess("sxy buffer not accessible".to_string())
        })?;
        let force_x = self.force_x.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("force_x buffer not accessible".to_string())
        })?;
        let force_y = self.force_y.try_as_slice_mut().ok_or_else(|| {
            WaveFieldError::BackendAccess("force_y buffer not accessible".to_string())
        })?;
        let depth = depth.try_as_slice().ok_or_else(|| {
            WaveFieldError::BackendAccess("depth buffer not accessible".to_string())
        })?;

        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                let h = if depth[idx] > h_min {
                    depth[idx]
                } else {
                    h_min
                };

                let dsxx_dx = if i == 0 {
                    (sxx[idx + 1] - sxx[idx]) * inv_dx
                } else if i == nx - 1 {
                    (sxx[idx] - sxx[idx - 1]) * inv_dx
                } else {
                    (sxx[idx + 1] - sxx[idx - 1]) * B::Scalar::HALF * inv_dx
                };

                let dsxy_dy = if j == 0 {
                    (sxy[idx + nx] - sxy[idx]) * inv_dy
                } else if j == ny - 1 {
                    (sxy[idx] - sxy[idx - nx]) * inv_dy
                } else {
                    (sxy[idx + nx] - sxy[idx - nx]) * B::Scalar::HALF * inv_dy
                };

                let dsxy_dx = if i == 0 {
                    (sxy[idx + 1] - sxy[idx]) * inv_dx
                } else if i == nx - 1 {
                    (sxy[idx] - sxy[idx - 1]) * inv_dx
                } else {
                    (sxy[idx + 1] - sxy[idx - 1]) * B::Scalar::HALF * inv_dx
                };

                let dsyy_dy = if j == 0 {
                    (syy[idx + nx] - syy[idx]) * inv_dy
                } else if j == ny - 1 {
                    (syy[idx] - syy[idx - nx]) * inv_dy
                } else {
                    (syy[idx + nx] - syy[idx - nx]) * B::Scalar::HALF * inv_dy
                };

                force_x[idx] = -(dsxx_dx + dsxy_dy) / (rho_water(&self.backend) * h);
                force_y[idx] = -(dsxy_dx + dsyy_dy) / (rho_water(&self.backend) * h);
            }
        }
        Ok(())
    }

    /// 获取辐射应力分量
    pub fn stress_components(
        &self,
    ) -> (
        &B::Buffer<B::Scalar>,
        &B::Buffer<B::Scalar>,
        &B::Buffer<B::Scalar>,
    ) {
        (&self.sxx, &self.syy, &self.sxy)
    }

    /// 获取波浪力（加速度）
    pub fn wave_forces(&self) -> (&B::Buffer<B::Scalar>, &B::Buffer<B::Scalar>) {
        (&self.force_x, &self.force_y)
    }
}

/// 波浪源项（泛型）
pub struct WaveSourceGeneric<B: Backend> {
    /// 波浪场
    wave_field: WaveFieldGeneric<B>,
    /// 辐射应力计算器
    stress_calculator: RadiationStressCalculatorGeneric<B>,
    /// 是否启用
    enabled: bool,
}

impl<B: Backend> WaveSourceGeneric<B> {
    /// 创建新的波浪源项
    pub fn new(backend: B, n_cells: usize) -> Self {
        Self {
            wave_field: WaveFieldGeneric::new(backend.clone(), n_cells),
            stress_calculator: RadiationStressCalculatorGeneric::new(backend, n_cells),
            enabled: true,
        }
    }

    /// 设置波浪参数
    pub fn set_wave_parameters(&mut self, params: &WaveParametersGeneric<B::Scalar>) {
        self.wave_field.set_uniform(params);
    }

    /// 设置波浪场
    pub fn set_wave_field(&mut self, field: WaveFieldGeneric<B>) {
        let backend = field.backend().clone();
        self.wave_field = field;
        self.stress_calculator =
            RadiationStressCalculatorGeneric::new(backend, self.wave_field.len());
    }

    /// 更新色散关系
    pub fn update_dispersion(
        &mut self,
        depth: &B::Buffer<B::Scalar>,
    ) -> Result<(), WaveFieldError> {
        self.wave_field.update_dispersion(depth)
    }

    /// 计算波浪力
    pub fn compute_forces(
        &mut self,
        nx: usize,
        ny: usize,
        dx: B::Scalar,
        dy: B::Scalar,
        depth: &B::Buffer<B::Scalar>,
    ) -> Result<(), WaveFieldError> {
        if !self.enabled {
            return Ok(());
        }
        self.stress_calculator.compute_stress(&self.wave_field)?;
        self.stress_calculator
            .compute_gradient_structured(nx, ny, dx, dy, depth)?;
        Ok(())
    }

    /// 获取波浪力
    pub fn get_forces(&self) -> (&B::Buffer<B::Scalar>, &B::Buffer<B::Scalar>) {
        self.stress_calculator.wave_forces()
    }

    /// 获取波浪场
    pub fn wave_field(&self) -> &WaveFieldGeneric<B> {
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
    use mh_runtime::CpuBackend;
    use std::f64::consts::PI;

    #[test]
    fn test_wave_parameters() {
        let backend = CpuBackend::<f64>::new();
        let params = WaveParametersGeneric::<f64>::new(2.0, 10.0, 0.0);

        let omega = params.angular_frequency(&backend);
        assert!((omega - 2.0 * PI / 10.0).abs() < 1e-10);

        let l0 = params.deep_water_wavelength(&backend);
        assert!(l0 > 100.0); // 深水波长约156m

        let energy = params.energy(&backend);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_wave_parameters_direction_vector() {
        // 北向
        let params = WaveParametersGeneric::<f64>::new(1.0, 8.0, 0.0);
        let (dx, dy) = params.direction_vector();
        assert!(dx.abs() < 1e-10);
        assert!((dy - 1.0).abs() < 1e-10);

        // 东向
        let params = WaveParametersGeneric::<f64>::new(1.0, 8.0, PI / 2.0);
        let (dx, dy) = params.direction_vector();
        assert!((dx - 1.0).abs() < 1e-10);
        assert!(dy.abs() < 1e-10);
    }

    #[test]
    fn test_wave_field_from_uniform() {
        let backend = CpuBackend::<f64>::new();
        let params = WaveParametersGeneric::<f64>::new(2.0, 10.0, PI / 4.0);
        let field = WaveFieldGeneric::<CpuBackend<f64>>::from_uniform(backend, 100, &params);

        assert_eq!(field.len(), 100);
        let height = field.height.as_slice();
        let period = field.period.as_slice();
        assert!((height[50] - 2.0).abs() < 1e-10);
        assert!((period[50] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_wave_field_update_dispersion() {
        let backend = CpuBackend::<f64>::new();
        let params = WaveParametersGeneric::<f64>::new(1.0, 8.0, 0.0);
        let mut field = WaveFieldGeneric::<CpuBackend<f64>>::from_uniform(backend, 10, &params);

        let mut depth = field.backend().alloc(10);
        depth.copy_from_slice(&[10.0; 10]);
        field.update_dispersion(&depth).unwrap();

        // 波数应该增加（相比深水）
        let wavenumber = field.wavenumber.as_slice();
        let wavelength = field.wavelength.as_slice();
        assert!(wavenumber[0] > 0.0);
        assert!(wavelength[0] > 0.0);
    }

    #[test]
    fn test_wavenumber_calculation() {
        let backend = CpuBackend::<f64>::new();
        let omega = 2.0 * PI / 8.0; // T = 8s

        // 深水
        let (k_deep, n_deep) = compute_wavenumber_and_n(&backend, omega, 100.0);
        assert!((n_deep - 0.5).abs() < 0.01); // 深水 n ≈ 0.5

        // 浅水
        let (k_shallow, n_shallow) = compute_wavenumber_and_n(&backend, omega, 1.0);
        assert!(n_shallow > 0.9); // 浅水 n → 1
        assert!(k_shallow > k_deep); // 浅水波数更大
    }

    #[test]
    fn test_radiation_stress_tensor() {
        let energy = 1000.0; // J/m²
        let n = 0.5;
        let direction = 0.0; // 北向

        let tensor = RadiationStressTensorGeneric::<f64>::compute(energy, n, direction);

        // Sxx = E(n(cos²θ + 1) - 0.5) = 1000(0.5(1+1) - 0.5) = 500
        assert!((tensor.sxx - 500.0).abs() < 1e-10);

        // Syy = E(n(sin²θ + 1) - 0.5) = 1000(0.5(0+1) - 0.5) = 0
        assert!((tensor.syy - 0.0).abs() < 1e-10);

        // Sxy = 0 (北向)
        assert!(tensor.sxy.abs() < 1e-10);
    }

    #[test]
    fn test_radiation_stress_calculator() {
        let backend = CpuBackend::<f64>::new();
        let params = WaveParametersGeneric::<f64>::new(2.0, 10.0, PI / 4.0);
        let field = WaveFieldGeneric::<CpuBackend<f64>>::from_uniform(backend.clone(), 25, &params);

        let mut calc = RadiationStressCalculatorGeneric::<CpuBackend<f64>>::new(backend, 25);
        calc.compute_stress(&field).unwrap();

        let (sxx, syy, sxy) = calc.stress_components();
        let sxx = sxx.as_slice();
        let syy = syy.as_slice();
        let sxy = sxy.as_slice();
        assert!(sxx.iter().all(|&s| s >= 0.0));
        assert!(syy.iter().all(|&s| s >= 0.0));
        // 45度方向 sxy 应该非零
        assert!(sxy.iter().any(|&s| s.abs() > 1e-10));
    }

    #[test]
    fn test_radiation_stress_gradient() {
        let backend = CpuBackend::<f64>::new();
        let params = WaveParametersGeneric::<f64>::new(2.0, 10.0, 0.0);
        let field = WaveFieldGeneric::<CpuBackend<f64>>::from_uniform(backend.clone(), 25, &params);

        let mut calc = RadiationStressCalculatorGeneric::<CpuBackend<f64>>::new(backend, 25);
        calc.compute_stress(&field).unwrap();

        let mut depth = field.backend().alloc(25);
        depth.copy_from_slice(&[10.0; 25]);
        calc.compute_gradient_structured(5, 5, 10.0, 10.0, &depth)
            .unwrap();

        let (fx, fy) = calc.wave_forces();
        // 均匀场应该力接近零
        let fx = fx.as_slice();
        let fy = fy.as_slice();
        let max_force = fx
            .iter()
            .chain(fy.iter())
            .map(|&f| f.abs())
            .fold(0.0, f64::max);
        assert!(max_force < 1.0); // 合理范围内
    }

    #[test]
    fn test_wave_source() {
        let backend = CpuBackend::<f64>::new();
        let mut source = WaveSourceGeneric::<CpuBackend<f64>>::new(backend, 25);

        let params = WaveParametersGeneric::<f64>::new(1.5, 8.0, 0.0);
        source.set_wave_parameters(&params);

        let mut depth = source.wave_field().backend().alloc(25);
        depth.copy_from_slice(&[5.0; 25]);
        source.update_dispersion(&depth).unwrap();
        source.compute_forces(5, 5, 10.0, 10.0, &depth).unwrap();

        let (fx, fy) = source.get_forces();
        assert_eq!(fx.len(), 25);
        assert_eq!(fy.len(), 25);
    }

    #[test]
    fn test_wave_source_disable() {
        let backend = CpuBackend::<f64>::new();
        let mut source = WaveSourceGeneric::<CpuBackend<f64>>::new(backend, 10);
        source.set_enabled(false);

        let mut depth = source.wave_field().backend().alloc(10);
        depth.copy_from_slice(&[5.0; 10]);
        source.compute_forces(10, 1, 10.0, 10.0, &depth).unwrap();

        // 禁用后力应为零
        let (fx, fy) = source.get_forces();
        let fx = fx.as_slice();
        let fy = fy.as_slice();
        assert!(fx.iter().all(|&f| f == 0.0));
        assert!(fy.iter().all(|&f| f == 0.0));
    }
}
