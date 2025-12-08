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
use mh_foundation::Scalar;
use serde::{Deserialize, Serialize};

/// 组装器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblerConfig {
    /// 最小水深阈值 [m]
    pub h_min: Scalar,
    /// 干单元处理因子
    pub dry_factor: Scalar,
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
pub struct PressureMatrixAssembler {
    /// 配置
    config: AssemblerConfig,
    /// 矩阵稀疏模式
    pattern: CsrPattern,
    /// 系数矩阵
    matrix: CsrMatrix,
    /// 右端项
    rhs: Vec<Scalar>,
    /// 对角元素索引缓存
    diag_indices: Vec<usize>,
}

impl PressureMatrixAssembler {
    /// 创建压力矩阵组装器
    pub fn new(topo: &CellFaceTopology) -> Self {
        let n = topo.n_cells();

        // 构建稀疏模式
        let mut builder = CsrBuilder::new_square(n);
        for cell_idx in 0..n {
            // 对角元素
            builder.set(cell_idx, cell_idx, 1.0);
            // 非对角元素（邻居）
            for neighbor in topo.cell_neighbor_indices(cell_idx) {
                builder.set(cell_idx, neighbor, 0.0);
            }
        }

        let pattern = builder.build_pattern();
        let matrix: CsrMatrix = pattern.clone().into();

        // 缓存对角元素索引
        let diag_indices: Vec<_> = (0..n)
            .map(|i| pattern.find_index(i, i).expect("diagonal must exist"))
            .collect();

        Self {
            config: AssemblerConfig::default(),
            pattern,
            matrix,
            rhs: vec![0.0; n],
            diag_indices,
        }
    }

    /// 使用配置创建
    pub fn with_config(topo: &CellFaceTopology, config: AssemblerConfig) -> Self {
        let mut assembler = Self::new(topo);
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
        state: &ShallowWaterState,
        dt: Scalar,
        g: Scalar,
    ) {
        let n = topo.n_cells();
        let coef = g * dt * dt;

        // 清零
        self.matrix.clear_values();
        self.rhs.fill(0.0);

        // 遍历面组装
        for &face_idx in topo.interior_faces() {
            let face = topo.face(face_idx);
            let owner = face.owner;
            let neighbor = face.neighbor.expect("interior face must have neighbor");

            // 面水深（算术平均）
            let h_f = 0.5 * (state.h[owner] + state.h[neighbor]);
            if h_f < self.config.h_min {
                continue;
            }

            let dist = face.dist_o2n;
            if dist < 1e-14 {
                continue;
            }

            // 系数 = g * dt² * H_f * L_f / d_{ON}
            let a_coef = coef * h_f * face.length / dist;

            // 如果使用面积加权
            let (a_o, a_n) = if self.config.area_weighted {
                let area_o = mesh.cell_area_unchecked(owner);
                let area_n = mesh.cell_area_unchecked(neighbor);
                (a_coef / area_o, a_coef / area_n)
            } else {
                (a_coef, a_coef)
            };

            // Owner 行
            self.matrix.add(owner, owner, a_o);
            self.matrix.add(owner, neighbor, -a_o);

            // Neighbor 行
            self.matrix.add(neighbor, neighbor, a_n);
            self.matrix.add(neighbor, owner, -a_n);
        }

        // 处理边界面（假设零梯度）
        for &face_idx in topo.boundary_faces() {
            let face = topo.face(face_idx);
            let owner = face.owner;

            let h_f = state.h[owner];
            if h_f < self.config.h_min {
                continue;
            }

            // 边界面不添加贡献（零梯度条件）
            // 如果需要其他边界条件，在此处理
        }

        // 确保对角占优（处理干单元）
        for i in 0..n {
            let diag_idx = self.diag_indices[i];
            let diag = self.matrix.values()[diag_idx];

            if diag.abs() < self.config.dry_factor {
                // 干单元：设置为单位矩阵行
                let start = self.pattern.row_ptr()[i];
                let end = self.pattern.row_ptr()[i + 1];
                for idx in start..end {
                    self.matrix.values_mut()[idx] = 0.0;
                }
                self.matrix.values_mut()[diag_idx] = 1.0;
                self.rhs[i] = 0.0;
            }
        }
    }

    /// 设置右端项
    pub fn set_rhs(&mut self, rhs: &[Scalar]) {
        self.rhs.copy_from_slice(rhs);
    }

    /// 获取矩阵引用
    pub fn matrix(&self) -> &CsrMatrix {
        &self.matrix
    }

    /// 获取可变矩阵引用
    pub fn matrix_mut(&mut self) -> &mut CsrMatrix {
        &mut self.matrix
    }

    /// 获取右端项引用
    pub fn rhs(&self) -> &[Scalar] {
        &self.rhs
    }

    /// 获取可变右端项引用
    pub fn rhs_mut(&mut self) -> &mut [Scalar] {
        &mut self.rhs
    }

    /// 获取对角元素
    pub fn diagonal(&self) -> Vec<Scalar> {
        self.diag_indices
            .iter()
            .map(|&idx| self.matrix.values()[idx])
            .collect()
    }
}

/// 隐式动量组装器
///
/// 组装动量方程的隐式部分
pub struct ImplicitMomentumAssembler {
    /// 配置
    config: AssemblerConfig,
    /// x 方向矩阵
    matrix_u: CsrMatrix,
    /// y 方向矩阵
    matrix_v: CsrMatrix,
    /// x 方向右端项
    rhs_u: Vec<Scalar>,
    /// y 方向右端项
    rhs_v: Vec<Scalar>,
}

impl ImplicitMomentumAssembler {
    /// 创建动量组装器
    pub fn new(topo: &CellFaceTopology) -> Self {
        let n = topo.n_cells();

        // 构建稀疏模式（与压力矩阵相同）
        let mut builder = CsrBuilder::new_square(n);
        for cell_idx in 0..n {
            builder.set(cell_idx, cell_idx, 1.0);
            for neighbor in topo.cell_neighbor_indices(cell_idx) {
                builder.set(cell_idx, neighbor, 0.0);
            }
        }

        let pattern = builder.build_pattern();
        let matrix_u: CsrMatrix = pattern.clone().into();
        let matrix_v: CsrMatrix = pattern.into();

        Self {
            config: AssemblerConfig::default(),
            matrix_u,
            matrix_v,
            rhs_u: vec![0.0; n],
            rhs_v: vec![0.0; n],
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
        state: &ShallowWaterState,
        manning_n: Scalar,
        dt: Scalar,
        g: Scalar,
    ) {
        let n = topo.n_cells();

        // 清零
        self.matrix_u.clear_values();
        self.matrix_v.clear_values();
        self.rhs_u.fill(0.0);
        self.rhs_v.fill(0.0);

        for i in 0..n {
            let h = state.h[i];
            let u = state.u[i];
            let v = state.v[i];

            if h < self.config.h_min {
                // 干单元：单位矩阵
                self.matrix_u.set(i, i, 1.0);
                self.matrix_v.set(i, i, 1.0);
                self.rhs_u[i] = 0.0;
                self.rhs_v[i] = 0.0;
                continue;
            }

            let speed = (u * u + v * v).sqrt();
            let h43 = h.powf(4.0 / 3.0);

            // 摩擦系数
            let cf = manning_n * manning_n * g * speed / h43;
            let diag = 1.0 + dt * cf;

            self.matrix_u.set(i, i, diag);
            self.matrix_v.set(i, i, diag);

            // 右端项 = u^* (显式预测值)
            self.rhs_u[i] = u;
            self.rhs_v[i] = v;
        }
    }

    /// 获取 u 方向矩阵
    pub fn matrix_u(&self) -> &CsrMatrix {
        &self.matrix_u
    }

    /// 获取 v 方向矩阵
    pub fn matrix_v(&self) -> &CsrMatrix {
        &self.matrix_v
    }

    /// 获取 u 方向右端项
    pub fn rhs_u(&self) -> &[Scalar] {
        &self.rhs_u
    }

    /// 获取 v 方向右端项
    pub fn rhs_v(&self) -> &[Scalar] {
        &self.rhs_v
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
