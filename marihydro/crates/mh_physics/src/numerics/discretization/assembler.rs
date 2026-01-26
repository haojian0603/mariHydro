// crates/mh_physics/src/numerics/discretization/assembler.rs

//! 系数矩阵组装器
//!
//! 提供隐式求解所需的系数矩阵组装功能：
//!
//! # 主要类型
//!
//! - [`PressureMatrixAssembler`]: 压力泊松方程矩阵组装
//! - [`ImplicitMomentumAssembler`]: 动量方程隐式组装
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::numerics::discretization::{
//!     PressureMatrixAssembler, CellFaceTopology,
//! };
//!
//! let topo = CellFaceTopology::from_mesh(&mesh);
//! let mut assembler = PressureMatrixAssembler::new(&topo);
//!
//! // 组装矩阵
//! assembler.assemble(&mesh, &state, dt, g);
//!
//! // 获取矩阵和右端项
//! let matrix = assembler.matrix();
//! let rhs = assembler.rhs();
//! ```

use super::topology::CellFaceTopology;
use crate::adapter::PhysicsMesh;
use crate::numerics::linear_algebra::{CsrBuilder, CsrMatrix, CsrPattern};
use crate::state::ShallowWaterState;
use mh_runtime::{Backend, CellIndex, DeviceBuffer, RuntimeScalar};
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};

/// 组装器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblerConfig {
    /// 最小水深阈值 [m]
    pub h_min: f64,
    /// 干单元处理因子
    pub dry_factor: f64,
    /// 是否使用面积加权
    pub area_weighted: bool,
}

impl Default for AssemblerConfig {
    fn default() -> Self {
        Self {
            h_min: 1e-4,
            dry_factor: 1e-6,
            area_weighted: true,
        }
    }
}

/// 压力泊松方程矩阵组装器
///
/// 组装压力校正方程的系数矩阵：
/// $$\nabla \cdot (H \nabla \eta') = \text{RHS}$$
///
/// 离散化后得到：
/// $$A_{ii} \eta'_i + \sum_j A_{ij} \eta'_j = b_i$$
pub struct PressureMatrixAssembler<B: Backend> {
    /// 配置
    config: AssemblerConfig,
    /// 矩阵稀疏模式
    pattern: CsrPattern,
    /// 系数矩阵
    matrix: CsrMatrix<B::Scalar>,
    /// 右端项
    rhs: B::Buffer<B::Scalar>,
    /// 对角元素索引缓存
    diag_indices: Vec<usize>,
    /// 后端实例
    backend: B,
}

impl<B> PressureMatrixAssembler<B>
where
    B: Backend,
    B::Scalar: RuntimeScalar + Float + FromPrimitive,
{
    /// 创建压力矩阵组装器
    pub fn new(topo: &CellFaceTopology, backend: B) -> Self {
        let n = topo.n_cells();
        let one = B::Scalar::ONE;
        let zero = B::Scalar::ZERO;

        // 构建稀疏模式
        let mut builder = CsrBuilder::new_square(n);
        for cell_idx in 0..n {
            // 对角元素
            builder.set(cell_idx, cell_idx, one);
            // 非对角元素（邻居）
            for neighbor_info in topo.cell_neighbors(cell_idx) {
                if let Some(neighbor_idx) = neighbor_info.cell_idx {
                    builder.set(cell_idx, neighbor_idx, zero);
                }
            }
        }

        let pattern = builder.build_pattern();
        let matrix: CsrMatrix<B::Scalar> = pattern.clone().into();

        // 缓存对角元素索引
        let diag_indices: Vec<_> = (0..n)
            .map(|i| pattern.find_index(i, i).expect("diagonal must exist"))
            .collect();

        Self {
            config: AssemblerConfig::default(),
            pattern,
            matrix,
            rhs: backend.alloc(n),
            diag_indices,
            backend,
        }
    }

    /// 使用配置创建
    pub fn with_config(topo: &CellFaceTopology, backend: B, config: AssemblerConfig) -> Self {
        let mut assembler = Self::new(topo, backend);
        assembler.config = config;
        assembler
    }

    /// 获取配置引用
    pub fn config(&self) -> &AssemblerConfig {
        &self.config
    }

    /// 获取可变配置引用
    pub fn config_mut(&mut self) -> &mut AssemblerConfig {
        &mut self.config
    }

    /// 组装压力矩阵
    ///
    /// # 参数
    ///
    /// - `mesh`: 物理网格
    /// - `topo`: 拓扑信息
    /// - `state`: 浅水状态
    /// - `dt`: 时间步长
    /// - `g`: 重力加速度
    pub fn assemble(
        &mut self,
        mesh: &PhysicsMesh,
        topo: &CellFaceTopology,
        state: &ShallowWaterState<B>,
        dt: B::Scalar,
        g: B::Scalar,
    ) {
        let n = topo.n_cells();
        let coef = g * dt * dt;
        let h_min = self.backend.scalar_from_f64(self.config.h_min);
        let dry_factor = self.backend.scalar_from_f64(self.config.dry_factor);
        let eps = self.backend.scalar_from_f64(1e-14);
        let zero = B::Scalar::ZERO;

        // 清零
        self.matrix.clear_values();
        self.rhs.fill(zero);

        // 遍历面组装
        for &face_idx in topo.interior_faces() {
            let face = topo.face(face_idx);
            let owner = face.owner;
            let neighbor = face.neighbor.expect("interior face must have neighbor");

            // 对称系数：使用调和平均保证对称性
            let h_o = state.h[owner].max(h_min);
            let h_n = state.h[neighbor].max(h_min);
            
            // 调和平均水深（对称）
            let two = self.backend.scalar_from_f64(2.0);
            let h_f = two * h_o * h_n / (h_o + h_n);
            
            if h_f < h_min {
                continue;
            }

            let dist = self.backend.scalar_from_f64(face.dist_o2n);
            if dist < eps {
                continue;
            }

            // 对称系数 = g * dt² * H_f * L_f / d_{ON}
            let a_coef = coef * h_f * self.backend.scalar_from_f64(face.length) / dist;

            // 如果使用面积加权
            let (a_o, a_n) = if self.config.area_weighted {
                let area_o = self.backend.scalar_from_f64(mesh.cell_area_unchecked(CellIndex(owner)));
                let area_n = self.backend.scalar_from_f64(mesh.cell_area_unchecked(CellIndex(neighbor)));
                (a_coef / area_o, a_coef / area_n)
            } else {
                (a_coef, a_coef)
            };

            // Owner 行
            self.matrix.add(owner, owner, a_o);
            self.matrix.add(owner, neighbor, -a_o);

            // Neighbor 行（对称）
            self.matrix.add(neighbor, neighbor, a_n);
            self.matrix.add(neighbor, owner, -a_n);

            // 床面坡度贡献到 RHS
            let dz = state.z[neighbor] - state.z[owner];
            let bed_slope_flux = g * h_f * dz * self.backend.scalar_from_f64(face.length) / dist;
            let rhs_o = if self.config.area_weighted {
                bed_slope_flux / self.backend.scalar_from_f64(mesh.cell_area_unchecked(mh_runtime::CellIndex(owner)))
            } else {
                bed_slope_flux
            };
            let rhs_n = if self.config.area_weighted {
                bed_slope_flux / self.backend.scalar_from_f64(mesh.cell_area_unchecked(mh_runtime::CellIndex(neighbor)))
            } else {
                bed_slope_flux
            };
            self.rhs[owner] -= rhs_o * dt;
            self.rhs[neighbor] += rhs_n * dt;
        }

        // 处理边界面（假设零梯度）
        for &face_idx in topo.boundary_faces() {
            let face = topo.face(face_idx);
            let owner = face.owner;

            let h_f = state.h[owner];
            if h_f < h_min {
                continue;
            }

            // 边界面不添加贡献（零梯度条件）
            // 如果需要其他边界条件，在此处理
        }

        // 确保对角占优（处理干单元）
        for i in 0..n {
            let diag_idx = self.diag_indices[i];
            let diag = self.matrix.values()[diag_idx];

            // 干单元判断：水深小于阈值或对角元太小
            let is_dry = state.h[i] < h_min || diag.abs() < dry_factor;
            
            if is_dry {
                // 干单元：设置为单位矩阵行，解耦
                let start = self.pattern.row_ptr()[i];
                let end = self.pattern.row_ptr()[i + 1];
                for idx in start..end {
                    self.matrix.values_mut()[idx] = zero;
                }
                self.matrix.values_mut()[diag_idx] = B::Scalar::ONE;
                self.rhs[i] = zero;
            }
        }
    }

    /// 组装压力矩阵（带预测速度散度 RHS）
    pub fn assemble_with_divergence(
        &mut self,
        mesh: &PhysicsMesh,
        topo: &CellFaceTopology,
        state: &ShallowWaterState<B>,
        hu_star: &[B::Scalar],
        hv_star: &[B::Scalar],
        dt: B::Scalar,
        g: B::Scalar,
    ) {
        // 先组装矩阵
        self.assemble(mesh, topo, state, dt, g);

        // 计算预测速度散度加入 RHS
        let half = self.backend.scalar_from_f64(0.5);
        for &face_idx in topo.interior_faces() {
            let face = topo.face(face_idx);
            let owner = face.owner;
            let neighbor = face.neighbor.expect("interior face must have neighbor");

            let h_f = half * (state.h[owner] + state.h[neighbor]);
            if h_f < self.backend.scalar_from_f64(self.config.h_min) {
                continue;
            }

            // 面法向通量
            let hu_f = half * (hu_star[owner] + hu_star[neighbor]);
            let hv_f = half * (hv_star[owner] + hv_star[neighbor]);
            let flux = (self.backend.scalar_from_f64(face.normal.0) * hu_f
                + self.backend.scalar_from_f64(face.normal.1) * hv_f)
                * self.backend.scalar_from_f64(face.length);

            let area_o = self.backend.scalar_from_f64(mesh.cell_area_unchecked(CellIndex(owner)));
            let area_n = self.backend.scalar_from_f64(mesh.cell_area_unchecked(CellIndex(neighbor)));

            self.rhs[owner] -= flux / area_o / dt;
            self.rhs[neighbor] += flux / area_n / dt;
        }
    }

    /// 设置右端项
    pub fn set_rhs(&mut self, rhs: &[B::Scalar]) {
        self.rhs.copy_from_slice(rhs);
    }

    /// 获取矩阵引用
    pub fn matrix(&self) -> &CsrMatrix<B::Scalar> {
        &self.matrix
    }

    /// 获取可变矩阵引用
    pub fn matrix_mut(&mut self) -> &mut CsrMatrix<B::Scalar> {
        &mut self.matrix
    }

    /// 获取右端项引用
    pub fn rhs(&self) -> &[B::Scalar] {
        self.rhs.as_slice()
    }

    /// 获取可变右端项引用
    pub fn rhs_mut(&mut self) -> &mut [B::Scalar] {
        self.rhs.as_slice_mut()
    }

    /// 获取对角元素
    pub fn diagonal(&self) -> Vec<B::Scalar> {
        self.diag_indices
            .iter()
            .map(|&idx| self.matrix.values()[idx])
            .collect()
    }
}

/// 隐式动量组装器
///
/// 组装动量方程的隐式部分
pub struct ImplicitMomentumAssembler<B: Backend> {
    /// 配置
    config: AssemblerConfig,
    /// x 方向矩阵
    matrix_u: CsrMatrix<B::Scalar>,
    /// y 方向矩阵
    matrix_v: CsrMatrix<B::Scalar>,
    /// x 方向右端项
    rhs_u: B::Buffer<B::Scalar>,
    /// y 方向右端项
    rhs_v: B::Buffer<B::Scalar>,
    /// 后端实例
    backend: B,
}

impl<B> ImplicitMomentumAssembler<B>
where
    B: Backend,
    B::Scalar: RuntimeScalar + Float + FromPrimitive,
{
    /// 创建动量组装器
    pub fn new(topo: &CellFaceTopology, backend: B) -> Self {
        let n = topo.n_cells();
        let one = B::Scalar::ONE;
        let zero = B::Scalar::ZERO;

        // 构建稀疏模式（与压力矩阵相同）
        let mut builder = CsrBuilder::new_square(n);
        for cell_idx in 0..n {
            builder.set(cell_idx, cell_idx, one);
            for neighbor_info in topo.cell_neighbors(cell_idx) {
                if let Some(neighbor_idx) = neighbor_info.cell_idx {
                    builder.set(cell_idx, neighbor_idx, zero);
                }
            }
        }

        let pattern = builder.build_pattern();
        let matrix_u: CsrMatrix<B::Scalar> = pattern.clone().into();
        let matrix_v: CsrMatrix<B::Scalar> = pattern.into();

        Self {
            config: AssemblerConfig::default(),
            matrix_u,
            matrix_v,
            rhs_u: backend.alloc(n),
            rhs_v: backend.alloc(n),
            backend,
        }
    }

    /// 组装隐式摩擦项
    ///
    /// 对于曼宁公式，摩擦项可以写为：
    /// $$S_f = \frac{n^2 g \vec{u} |\vec{u}|}{h^{4/3}}$$
    ///
    /// 隐式化后：
    /// $$(1 + dt \cdot C_f) u^{n+1} = u^* + \text{source}$$
    pub fn assemble_friction(
        &mut self,
        topo: &CellFaceTopology,
        state: &ShallowWaterState<B>,
        manning_n: B::Scalar,
        dt: B::Scalar,
        g: B::Scalar,
    ) {
        let n = topo.n_cells();
        let zero = B::Scalar::ZERO;
        let one = B::Scalar::ONE;
        let h_min = self.backend.scalar_from_f64(self.config.h_min);

        // 清零
        self.matrix_u.clear_values();
        self.matrix_v.clear_values();
        self.rhs_u.fill(zero);
        self.rhs_v.fill(zero);

        for i in 0..n {
            let h = state.h[i];
            
            if h < h_min {
                // 干单元：单位矩阵
                self.matrix_u.set(i, i, one);
                self.matrix_v.set(i, i, one);
                self.rhs_u[i] = zero;
                self.rhs_v[i] = zero;
                continue;
            }
            
            // 从动量计算速度
            let u = state.hu[i] / h;
            let v = state.hv[i] / h;

            let speed = (u * u + v * v).sqrt();
            let h43 = h.powf(self.backend.scalar_from_f64(4.0 / 3.0));

            // 摩擦系数
            let cf = manning_n * manning_n * g * speed / h43;
            let diag = one + dt * cf;

            self.matrix_u.set(i, i, diag);
            self.matrix_v.set(i, i, diag);

            // 右端项 = u^* (显式预测值)
            self.rhs_u[i] = u;
            self.rhs_v[i] = v;
        }
    }

    /// 获取 u 方向矩阵
    pub fn matrix_u(&self) -> &CsrMatrix<B::Scalar> {
        &self.matrix_u
    }

    /// 获取 v 方向矩阵
    pub fn matrix_v(&self) -> &CsrMatrix<B::Scalar> {
        &self.matrix_v
    }

    /// 获取 u 方向右端项
    pub fn rhs_u(&self) -> &[B::Scalar] {
        self.rhs_u.as_slice()
    }

    /// 获取 v 方向右端项
    pub fn rhs_v(&self) -> &[B::Scalar] {
        self.rhs_v.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assembler_config_default() {
        let config = AssemblerConfig::default();
        assert!((config.h_min - 1e-4).abs() < 1e-10);
        assert!(config.area_weighted);
    }
}
