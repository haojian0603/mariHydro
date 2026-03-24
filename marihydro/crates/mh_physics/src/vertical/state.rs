// marihydro\crates\mh_physics\src\vertical\state.rs
//! 分层状态管理
//!
//! 提供 3D 分层数据结构：
//! - `LayeredScalar`: 单个分层标量场
//! - `LayeredState`: 完整的 3D 状态（速度、标量）

use super::sigma::SigmaCoordinate;
use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar};
use thiserror::Error;

/// 分层状态错误
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum LayeredStateError {
    /// 尺寸不匹配
    #[error("size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },
}

/// 分层标量场
///
/// 存储单个标量量在所有层的值，布局为 `layers[k][cell]`
#[derive(Clone)]
pub struct LayeredScalar<B: Backend> {
    /// 层数
    n_layers: usize,
    /// 单元数
    n_cells: usize,
    /// 数据存储 [层索引][单元索引]
    data: Vec<B::Buffer<B::Scalar>>,
    /// 后端实例
    backend: B,
}

impl<B: Backend> std::fmt::Debug for LayeredScalar<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayeredScalar")
            .field("n_layers", &self.n_layers)
            .field("n_cells", &self.n_cells)
            .field("layers", &self.data.len())
            .finish()
    }
}

impl<B: Backend> LayeredScalar<B> {
    /// 创建零初始化的分层标量
    pub fn new_with_backend(backend: B, n_cells: usize, n_layers: usize) -> Self {
        let mut data: Vec<B::Buffer<B::Scalar>> = (0..n_layers)
            .map(|_| backend.alloc(n_cells))
            .collect();
        for layer in &mut data {
            layer.fill(<B::Scalar as RuntimeScalar>::ZERO);
        }

        Self {
            n_layers,
            n_cells,
            data,
            backend,
        }
    }

    /// 从 σ 坐标创建
    pub fn from_sigma(backend: B, n_cells: usize, sigma: &SigmaCoordinate) -> Self {
        Self::new_with_backend(backend, n_cells, sigma.n_layers())
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
    pub fn layer(&self, k: usize) -> &B::Buffer<B::Scalar> {
        &self.data[k]
    }

    /// 获取特定层的可变切片
    #[inline]
    pub fn layer_mut(&mut self, k: usize) -> &mut B::Buffer<B::Scalar> {
        &mut self.data[k]
    }

    /// 获取特定单元、层的值
    #[inline]
    pub fn get(&self, cell: usize, k: usize) -> B::Scalar {
        self.data[k][cell]
    }

    /// 设置特定单元、层的值
    #[inline]
    pub fn set(&mut self, cell: usize, k: usize, value: B::Scalar) {
        self.data[k][cell] = value;
    }

    /// 设置整个单元柱的值（所有层相同）
    pub fn set_column(&mut self, cell: usize, value: B::Scalar) {
        for k in 0..self.n_layers {
            self.data[k][cell] = value;
        }
    }

    /// 设置整层的值
    pub fn fill_layer(&mut self, k: usize, value: B::Scalar) {
        self.data[k].fill(value);
    }

    /// 设置所有值
    pub fn fill(&mut self, value: B::Scalar) {
        for layer in &mut self.data {
            layer.fill(value);
        }
    }

    /// 深度加权平均（计算 2D 表示）
    pub fn depth_average(&self, layer_weights: &[B::Scalar]) -> B::Buffer<B::Scalar> {
        let mut avg = self.backend.alloc(self.n_cells);
        avg.fill(<B::Scalar as RuntimeScalar>::ZERO);
        let mut total_weight = <B::Scalar as RuntimeScalar>::ZERO;
        let n = self.n_layers.min(layer_weights.len());

        for k in 0..n {
            let weight = layer_weights[k];
            total_weight += weight;
            for cell in 0..self.n_cells {
                avg[cell] += self.data[k][cell] * weight;
            }
        }

        let eps = <B::Scalar as RuntimeScalar>::from_config_or_panic(
            1e-10,
            "LayeredField3D::average_over_layers.eps",
        );
        if total_weight > eps {
            for cell in 0..self.n_cells {
                avg[cell] /= total_weight;
            }
        }

        avg
    }

    /// 从 2D 场初始化（所有层相同）
    pub fn from_2d(backend: B, values: &B::Buffer<B::Scalar>, n_layers: usize) -> Self {
        let n_cells = values.len();
        let mut layered = Self::new_with_backend(backend, n_cells, n_layers);
        for k in 0..n_layers {
            layered.backend.copy(values, &mut layered.data[k]);
        }
        layered
    }
}

/// 完整的 3D 分层状态
#[derive(Clone)]
pub struct LayeredState<B: Backend> {
    /// σ 坐标
    sigma: SigmaCoordinate,
    /// 单元数
    n_cells: usize,
    /// x 方向速度 [m/s]
    pub u: LayeredScalar<B>,
    /// y 方向速度 [m/s]
    pub v: LayeredScalar<B>,
    /// 垂向速度 [m/s]（在层界面）
    pub w: LayeredScalar<B>,
    /// 温度 [°C]（可选）
    pub temperature: Option<LayeredScalar<B>>,
    /// 盐度 [PSU]（可选）
    pub salinity: Option<LayeredScalar<B>>,
    /// 悬沙浓度 [kg/m³]（可选）
    pub sediment: Option<LayeredScalar<B>>,
    /// 湍动能 k [m²/s²]（可选，用于 k-ε）
    pub tke: Option<LayeredScalar<B>>,
    /// 湍流耗散率 ε [m²/s³]（可选，用于 k-ε）
    pub dissipation: Option<LayeredScalar<B>>,
    /// 后端实例
    backend: B,
}

impl<B: Backend> std::fmt::Debug for LayeredState<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayeredState")
            .field("n_cells", &self.n_cells)
            .field("n_layers", &self.sigma.n_layers())
            .field("has_temperature", &self.temperature.is_some())
            .field("has_salinity", &self.salinity.is_some())
            .field("has_sediment", &self.sediment.is_some())
            .field("has_tke", &self.tke.is_some())
            .field("has_dissipation", &self.dissipation.is_some())
            .finish()
    }
}

impl<B: Backend> LayeredState<B> {
    /// 创建新的分层状态
    pub fn new_with_backend(backend: B, n_cells: usize, sigma: &SigmaCoordinate) -> Self {
        let n_layers = sigma.n_layers();
        Self {
            sigma: sigma.clone(),
            n_cells,
            u: LayeredScalar::new_with_backend(backend.clone(), n_cells, n_layers),
            v: LayeredScalar::new_with_backend(backend.clone(), n_cells, n_layers),
            w: LayeredScalar::new_with_backend(backend.clone(), n_cells, n_layers + 1), // 界面上
            temperature: None,
            salinity: None,
            sediment: None,
            tke: None,
            dissipation: None,
            backend,
        }
    }

    /// 启用温度场
    pub fn with_temperature(mut self) -> Self {
        self.temperature = Some(LayeredScalar::new_with_backend(
            self.backend.clone(),
            self.n_cells,
            self.sigma.n_layers(),
        ));
        self
    }

    /// 启用盐度场
    pub fn with_salinity(mut self) -> Self {
        self.salinity = Some(LayeredScalar::new_with_backend(
            self.backend.clone(),
            self.n_cells,
            self.sigma.n_layers(),
        ));
        self
    }

    /// 启用悬沙场
    pub fn with_sediment(mut self) -> Self {
        self.sediment = Some(LayeredScalar::new_with_backend(
            self.backend.clone(),
            self.n_cells,
            self.sigma.n_layers(),
        ));
        self
    }

    /// 启用 k-ε 湍流场
    pub fn with_k_epsilon(mut self) -> Self {
        let n = self.sigma.n_layers();
        self.tke = Some(LayeredScalar::new_with_backend(
            self.backend.clone(),
            self.n_cells,
            n,
        ));
        self.dissipation = Some(LayeredScalar::new_with_backend(
            self.backend.clone(),
            self.n_cells,
            n,
        ));
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
    pub fn init_from_2d(
        &mut self,
        u_2d: &B::Buffer<B::Scalar>,
        v_2d: &B::Buffer<B::Scalar>,
    ) -> Result<(), LayeredStateError> {
        if u_2d.len() != self.n_cells {
            return Err(LayeredStateError::SizeMismatch {
                expected: self.n_cells,
                actual: u_2d.len(),
            });
        }
        if v_2d.len() != self.n_cells {
            return Err(LayeredStateError::SizeMismatch {
                expected: self.n_cells,
                actual: v_2d.len(),
            });
        }

        for k in 0..self.n_layers() {
            self.backend.copy(u_2d, self.u.layer_mut(k));
            self.backend.copy(v_2d, self.v.layer_mut(k));
        }
        Ok(())
    }

    /// 计算深度平均速度
    pub fn depth_average_velocity(&self) -> (B::Buffer<B::Scalar>, B::Buffer<B::Scalar>) {
        let weights: Vec<B::Scalar> = (0..self.n_layers())
            .map(|k| {
                self.backend.config_scalar(
                    self.sigma.layer_thickness_sigma(k),
                    "LayeredFlowState.depth_average_velocity.layer_thickness",
                )
            })
            .collect();

        (self.u.depth_average(&weights), self.v.depth_average(&weights))
    }

    /// 计算动能
    pub fn kinetic_energy(&self, cell: usize, k: usize) -> B::Scalar {
        let u = self.u.get(cell, k);
        let v = self.v.get(cell, k);
        <B::Scalar as RuntimeScalar>::HALF * (u * u + v * v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::{Backend, CpuBackend, DeviceBuffer};

    #[test]
    fn test_layered_scalar_creation() {
        let backend = CpuBackend::<f64>::new();
        let scalar = LayeredScalar::new_with_backend(backend, 10, 5);
        assert_eq!(scalar.n_cells(), 10);
        assert_eq!(scalar.n_layers(), 5);
    }

    #[test]
    fn test_layered_scalar_access() {
        let backend = CpuBackend::<f64>::new();
        let mut scalar = LayeredScalar::new_with_backend(backend, 10, 5);
        scalar.set(3, 2, 1.5);
        assert!((scalar.get(3, 2) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_layered_scalar_from_2d() {
        let backend = CpuBackend::<f64>::new();
        let mut values_2d = backend.alloc(3);
        values_2d.copy_from_slice(&[1.0, 2.0, 3.0]);
        let scalar = LayeredScalar::from_2d(backend, &values_2d, 5);

        for k in 0..5 {
            assert!((scalar.get(0, k) - 1.0).abs() < 1e-10);
            assert!((scalar.get(1, k) - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_depth_average() {
        let backend = CpuBackend::<f64>::new();
        let mut scalar = LayeredScalar::new_with_backend(backend, 2, 5);
        // 设置从表层到底层线性增加的值
        for k in 0..5 {
            scalar.set(0, k, k as f64);
        }

        // 均匀权重
        let weights = vec![0.2; 5];
        let avg = scalar.depth_average(&weights);
        
        // 平均值应该是 (0+1+2+3+4)/5 = 2
        assert!((avg[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_layered_state_creation() {
        let sigma = SigmaCoordinate::uniform(10);
        let backend = CpuBackend::<f64>::new();
        let state = LayeredState::new_with_backend(backend, 100, &sigma);

        assert_eq!(state.n_cells(), 100);
        assert_eq!(state.n_layers(), 10);
    }

    #[test]
    fn test_layered_state_optional_fields() {
        let sigma = SigmaCoordinate::uniform(5);
        let backend = CpuBackend::<f64>::new();
        let state = LayeredState::new_with_backend(backend, 10, &sigma)
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
        let backend = CpuBackend::<f64>::new();
        let mut state = LayeredState::new_with_backend(backend.clone(), 3, &sigma);

        let mut u_2d = backend.alloc(3);
        u_2d.copy_from_slice(&[1.0, 2.0, 3.0]);
        let mut v_2d = backend.alloc(3);
        v_2d.copy_from_slice(&[0.5, 1.0, 1.5]);
        state.init_from_2d(&u_2d, &v_2d).unwrap();

        for k in 0..5 {
            assert!((state.u.get(0, k) - 1.0).abs() < 1e-10);
            assert!((state.v.get(2, k) - 1.5).abs() < 1e-10);
        }
    }
}
