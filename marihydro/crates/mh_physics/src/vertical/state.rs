// marihydro\crates\mh_physics\src\vertical\state.rs
//! 分层状态管理
//!
//! 提供 3D 分层数据结构：
//! - `LayeredScalar`: 单个分层标量场
//! - `LayeredState`: 完整的 3D 状态（速度、标量）

use super::sigma::SigmaCoordinate;
use mh_foundation::AlignedVec;
use serde::{Deserialize, Serialize};

/// 分层标量场
///
/// 存储单个标量量在所有层的值，布局为 `layers[k][cell]`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredScalar {
    /// 层数
    n_layers: usize,
    /// 单元数
    n_cells: usize,
    /// 数据存储 [层索引][单元索引]
    data: Vec<AlignedVec<f64>>,
}

impl LayeredScalar {
    /// 创建零初始化的分层标量
    pub fn zeros(n_cells: usize, n_layers: usize) -> Self {
        Self {
            n_layers,
            n_cells,
            data: (0..n_layers)
                .map(|_| AlignedVec::zeros(n_cells))
                .collect(),
        }
    }

    /// 从 σ 坐标创建
    pub fn from_sigma(n_cells: usize, sigma: &SigmaCoordinate) -> Self {
        Self::zeros(n_cells, sigma.n_layers())
    }

    /// 层数
    #[inline]
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// 单元数
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 获取特定层的切片
    #[inline]
    pub fn layer(&self, k: usize) -> &[f64] {
        &self.data[k]
    }

    /// 获取特定层的可变切片
    #[inline]
    pub fn layer_mut(&mut self, k: usize) -> &mut [f64] {
        &mut self.data[k]
    }

    /// 获取特定单元、层的值
    #[inline]
    pub fn get(&self, cell: usize, k: usize) -> f64 {
        self.data[k][cell]
    }

    /// 设置特定单元、层的值
    #[inline]
    pub fn set(&mut self, cell: usize, k: usize, value: f64) {
        self.data[k][cell] = value;
    }

    /// 设置整个单元柱的值（所有层相同）
    pub fn set_column(&mut self, cell: usize, value: f64) {
        for k in 0..self.n_layers {
            self.data[k][cell] = value;
        }
    }

    /// 设置整层的值
    pub fn fill_layer(&mut self, k: usize, value: f64) {
        self.data[k].fill(value);
    }

    /// 设置所有值
    pub fn fill(&mut self, value: f64) {
        for layer in &mut self.data {
            layer.fill(value);
        }
    }

    /// 深度加权平均（计算 2D 表示）
    pub fn depth_average(&self, layer_weights: &[f64]) -> AlignedVec<f64> {
        let mut avg = AlignedVec::zeros(self.n_cells);
        let mut total_weight = 0.0;

        for (k, weight) in layer_weights.iter().enumerate().take(self.n_layers) {
            total_weight += weight;
            for cell in 0..self.n_cells {
                avg[cell] += self.data[k][cell] * weight;
            }
        }

        if total_weight > 1e-10 {
            for cell in 0..self.n_cells {
                avg[cell] /= total_weight;
            }
        }

        avg
    }

    /// 从 2D 场初始化（所有层相同）
    pub fn from_2d(values: &[f64], n_layers: usize) -> Self {
        let n_cells = values.len();
        let mut layered = Self::zeros(n_cells, n_layers);
        for k in 0..n_layers {
            layered.data[k].as_mut_slice().copy_from_slice(values);
        }
        layered
    }
}

/// 完整的 3D 分层状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredState {
    /// σ 坐标
    sigma: SigmaCoordinate,
    /// 单元数
    n_cells: usize,
    /// x 方向速度 [m/s]
    pub u: LayeredScalar,
    /// y 方向速度 [m/s]
    pub v: LayeredScalar,
    /// 垂向速度 [m/s]（在层界面）
    pub w: LayeredScalar,
    /// 温度 [°C]（可选）
    pub temperature: Option<LayeredScalar>,
    /// 盐度 [PSU]（可选）
    pub salinity: Option<LayeredScalar>,
    /// 悬沙浓度 [kg/m³]（可选）
    pub sediment: Option<LayeredScalar>,
    /// 湍动能 k [m²/s²]（可选，用于 k-ε）
    pub tke: Option<LayeredScalar>,
    /// 湍流耗散率 ε [m²/s³]（可选，用于 k-ε）
    pub dissipation: Option<LayeredScalar>,
}

impl LayeredState {
    /// 创建新的分层状态
    pub fn new(n_cells: usize, sigma: &SigmaCoordinate) -> Self {
        let n_layers = sigma.n_layers();
        Self {
            sigma: sigma.clone(),
            n_cells,
            u: LayeredScalar::zeros(n_cells, n_layers),
            v: LayeredScalar::zeros(n_cells, n_layers),
            w: LayeredScalar::zeros(n_cells, n_layers + 1), // 界面上
            temperature: None,
            salinity: None,
            sediment: None,
            tke: None,
            dissipation: None,
        }
    }

    /// 启用温度场
    pub fn with_temperature(mut self) -> Self {
        self.temperature = Some(LayeredScalar::zeros(self.n_cells, self.sigma.n_layers()));
        self
    }

    /// 启用盐度场
    pub fn with_salinity(mut self) -> Self {
        self.salinity = Some(LayeredScalar::zeros(self.n_cells, self.sigma.n_layers()));
        self
    }

    /// 启用悬沙场
    pub fn with_sediment(mut self) -> Self {
        self.sediment = Some(LayeredScalar::zeros(self.n_cells, self.sigma.n_layers()));
        self
    }

    /// 启用 k-ε 湍流场
    pub fn with_k_epsilon(mut self) -> Self {
        let n = self.sigma.n_layers();
        self.tke = Some(LayeredScalar::zeros(self.n_cells, n));
        self.dissipation = Some(LayeredScalar::zeros(self.n_cells, n));
        self
    }

    /// 层数
    #[inline]
    pub fn n_layers(&self) -> usize {
        self.sigma.n_layers()
    }

    /// 单元数
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// σ 坐标引用
    pub fn sigma(&self) -> &SigmaCoordinate {
        &self.sigma
    }

    /// 从 2D 速度初始化（所有层相同）
    pub fn init_from_2d(&mut self, u_2d: &[f64], v_2d: &[f64]) {
        for k in 0..self.n_layers() {
            let n = self.n_cells.min(u_2d.len()).min(v_2d.len());
            self.u.layer_mut(k)[..n].copy_from_slice(&u_2d[..n]);
            self.v.layer_mut(k)[..n].copy_from_slice(&v_2d[..n]);
        }
    }

    /// 计算深度平均速度
    pub fn depth_average_velocity(&self) -> (AlignedVec<f64>, AlignedVec<f64>) {
        let weights: Vec<f64> = (0..self.n_layers())
            .map(|k| self.sigma.layer_thickness_sigma(k))
            .collect();
        
        (self.u.depth_average(&weights), self.v.depth_average(&weights))
    }

    /// 计算动能
    pub fn kinetic_energy(&self, cell: usize, k: usize) -> f64 {
        let u = self.u.get(cell, k);
        let v = self.v.get(cell, k);
        0.5 * (u * u + v * v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layered_scalar_creation() {
        let f64 = LayeredScalar::zeros(10, 5);
        assert_eq!(f64.n_cells(), 10);
        assert_eq!(f64.n_layers(), 5);
    }

    #[test]
    fn test_layered_scalar_access() {
        let mut f64 = LayeredScalar::zeros(10, 5);
        f64.set(3, 2, 1.5);
        assert!((f64.get(3, 2) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_layered_scalar_from_2d() {
        let values_2d = vec![1.0, 2.0, 3.0];
        let f64 = LayeredScalar::from_2d(&values_2d, 5);

        for k in 0..5 {
            assert!((f64.get(0, k) - 1.0).abs() < 1e-10);
            assert!((f64.get(1, k) - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_depth_average() {
        let mut f64 = LayeredScalar::zeros(2, 5);
        // 设置从表层到底层线性增加的值
        for k in 0..5 {
            f64.set(0, k, k as f64);
        }

        // 均匀权重
        let weights = vec![0.2; 5];
        let avg = f64.depth_average(&weights);
        
        // 平均值应该是 (0+1+2+3+4)/5 = 2
        assert!((avg[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_layered_state_creation() {
        let sigma = SigmaCoordinate::uniform(10);
        let state = LayeredState::new(100, &sigma);

        assert_eq!(state.n_cells(), 100);
        assert_eq!(state.n_layers(), 10);
    }

    #[test]
    fn test_layered_state_optional_fields() {
        let sigma = SigmaCoordinate::uniform(5);
        let state = LayeredState::new(10, &sigma)
            .with_temperature()
            .with_k_epsilon();

        assert!(state.temperature.is_some());
        assert!(state.tke.is_some());
        assert!(state.dissipation.is_some());
        assert!(state.salinity.is_none());
    }

    #[test]
    fn test_init_from_2d() {
        let sigma = SigmaCoordinate::uniform(5);
        let mut state = LayeredState::new(3, &sigma);

        let u_2d = vec![1.0, 2.0, 3.0];
        let v_2d = vec![0.5, 1.0, 1.5];
        state.init_from_2d(&u_2d, &v_2d);

        for k in 0..5 {
            assert!((state.u.get(0, k) - 1.0).abs() < 1e-10);
            assert!((state.v.get(2, k) - 1.5).abs() < 1e-10);
        }
    }
}
