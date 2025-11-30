// src-tauri/src/marihydro/physics/sources/tracer_transport.rs
//! 标量示踪剂输运求解器
//! 实现平流-扩散方程的数值求解

use glam::DVec2;
use rayon::prelude::*;

use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::domain::mesh::unstructured::{BoundaryKind, UnstructuredMesh};
use crate::marihydro::domain::state::tracer::{TracerField, TracerType};

/// 平流格式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvectionScheme {
    /// 一阶迎风（稳定但耗散大）
    FirstOrderUpwind,
    /// 二阶迎风（精度更高）
    SecondOrderUpwind,
    /// 中心差分（精度最高但可能振荡）
    CentralDifference,
    /// TVD 格式 with minmod limiter
    TvdMinmod,
}

impl Default for AdvectionScheme {
    fn default() -> Self {
        Self::FirstOrderUpwind
    }
}

/// 边界条件类型
#[derive(Debug, Clone)]
pub enum TracerBoundaryCondition {
    /// 零梯度（Neumann）
    ZeroGradient,
    /// 固定值（Dirichlet）
    FixedValue(f64),
    /// 开边界（外推）
    Open,
    /// 入流浓度
    Inflow(f64),
}

impl Default for TracerBoundaryCondition {
    fn default() -> Self {
        Self::ZeroGradient
    }
}

/// 示踪剂输运求解器
pub struct TracerTransportSolver {
    /// 平流格式
    pub advection_scheme: AdvectionScheme,
    /// 扩散子步数
    pub diffusion_substeps: usize,
    /// 边界条件映射（按边界标签）
    pub boundary_conditions: Vec<TracerBoundaryCondition>,
    /// 工作数组
    flux_buffer: Vec<f64>,
    /// 浓度梯度存储（用于高阶格式）
    concentration_gradient: Vec<DVec2>,
}

impl TracerTransportSolver {
    /// 创建新的求解器
    pub fn new(n_cells: usize) -> Self {
        Self {
            advection_scheme: AdvectionScheme::FirstOrderUpwind,
            diffusion_substeps: 1,
            boundary_conditions: Vec::new(),
            flux_buffer: vec![0.0; n_cells],
            concentration_gradient: vec![DVec2::ZERO; n_cells],
        }
    }

    /// 设置平流格式
    pub fn with_scheme(mut self, scheme: AdvectionScheme) -> Self {
        self.advection_scheme = scheme;
        self
    }

    /// 设置扩散子步数
    pub fn with_diffusion_substeps(mut self, substeps: usize) -> Self {
        self.diffusion_substeps = substeps.max(1);
        self
    }

    /// 求解一个时间步
    /// 
    /// # Arguments
    /// * `tracer` - 示踪剂场
    /// * `mesh` - 网格
    /// * `h` - 水深场
    /// * `hu` - x方向通量 (h*u)
    /// * `hv` - y方向通量 (h*v)
    /// * `nu_t` - 湍流扩散系数（可选）
    /// * `dt` - 时间步长
    pub fn solve_step(
        &mut self,
        tracer: &mut TracerField,
        mesh: &UnstructuredMesh,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        nu_t: Option<&[f64]>,
        dt: f64,
    ) -> MhResult<()> {
        let n_cells = mesh.n_cells;
        
        // 验证输入
        if tracer.concentration.len() != n_cells {
            return Err(MhError::InvalidInput("Tracer size mismatch".into()));
        }

        // 确保工作数组大小正确
        if self.flux_buffer.len() != n_cells {
            self.flux_buffer.resize(n_cells, 0.0);
        }
        if self.concentration_gradient.len() != n_cells {
            self.concentration_gradient.resize(n_cells, DVec2::ZERO);
        }

        // 0. 对于高阶格式，先计算浓度梯度
        if matches!(
            self.advection_scheme,
            AdvectionScheme::SecondOrderUpwind | AdvectionScheme::TvdMinmod
        ) {
            self.compute_concentration_gradient(&tracer.concentration, mesh);
        }

        // 1. 平流项
        self.apply_advection(tracer, mesh, h, hu, hv, dt)?;

        // 2. 扩散项
        let nu_eff = self.compute_effective_diffusivity(tracer, nu_t, n_cells);
        self.apply_diffusion(tracer, mesh, h, &nu_eff, dt)?;

        // 3. 沉降项（如适用）
        if tracer.properties.tracer_type == TracerType::Settling {
            self.apply_settling(tracer, h, dt);
        }

        // 4. 衰减项
        tracer.apply_decay(dt);

        // 5. 应用限制器
        tracer.apply_limiter();

        Ok(())
    }

    /// Minmod 限制器函数
    /// 
    /// 返回两个数中模最小的，如果符号不同则返回 0
    #[inline]
    fn minmod(a: f64, b: f64) -> f64 {
        if a * b <= 0.0 {
            0.0
        } else if a.abs() < b.abs() {
            a
        } else {
            b
        }
    }

    /// 计算浓度梯度（Green-Gauss 方法）
    /// 
    /// 使用简化的 Green-Gauss 方法计算单元梯度
    fn compute_concentration_gradient(&mut self, c: &[f64], mesh: &UnstructuredMesh) {
        let n_cells = mesh.n_cells;
        
        // 初始化梯度为零
        for grad in self.concentration_gradient.iter_mut() {
            *grad = DVec2::ZERO;
        }
        
        // 累加内部面贡献
        for face_idx in mesh.interior_faces() {
            let owner = mesh.face_owner[face_idx];
            let neighbor = mesh.face_neighbor[face_idx];
            
            // 面平均浓度
            let c_face = 0.5 * (c[owner] + c[neighbor]);
            
            // 面法向和长度
            let nx = mesh.face_normal_x[face_idx];
            let ny = mesh.face_normal_y[face_idx];
            let length = mesh.face_length[face_idx];
            
            // 通量贡献
            let flux = DVec2::new(c_face * nx * length, c_face * ny * length);
            
            // 累加到 owner（出流为正），减去 neighbor（入流为负）
            self.concentration_gradient[owner] += flux;
            self.concentration_gradient[neighbor] -= flux;
        }
        
        // 累加边界面贡献
        for face_idx in mesh.boundary_faces() {
            let owner = mesh.face_owner[face_idx];
            
            // 边界面使用内部单元浓度（零梯度近似）
            let c_face = c[owner];
            
            let nx = mesh.face_normal_x[face_idx];
            let ny = mesh.face_normal_y[face_idx];
            let length = mesh.face_length[face_idx];
            
            let flux = DVec2::new(c_face * nx * length, c_face * ny * length);
            self.concentration_gradient[owner] += flux;
        }
        
        // 除以单元面积得到梯度
        for i in 0..n_cells {
            let area = mesh.cell_area[i];
            if area > 1e-14 {
                self.concentration_gradient[i] /= area;
            }
        }
    }

    /// 计算有效扩散系数
    fn compute_effective_diffusivity(
        &self,
        tracer: &TracerField,
        nu_t: Option<&[f64]>,
        n_cells: usize,
    ) -> Vec<f64> {
        let molecular = tracer.properties.diffusivity;
        
        match nu_t {
            Some(turbulent) => {
                // Schmidt 数取 0.7（温度/盐度约1.0）
                let sc = match tracer.properties.tracer_type {
                    TracerType::Temperature => 0.7,
                    TracerType::Salinity => 1.0,
                    _ => 0.7,
                };
                
                turbulent
                    .iter()
                    .map(|&nut| molecular + nut / sc)
                    .collect()
            }
            None => vec![molecular; n_cells],
        }
    }

    /// 应用平流项
    fn apply_advection(
        &mut self,
        tracer: &mut TracerField,
        mesh: &UnstructuredMesh,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        dt: f64,
    ) -> MhResult<()> {
        let c = &tracer.concentration;
        self.flux_buffer.fill(0.0);

        // 遍历所有内部面
        for face_idx in mesh.interior_faces() {
            let owner = mesh.face_owner[face_idx];
            let neighbor = mesh.face_neighbor[face_idx];

            // 面法向通量
            let nx = mesh.face_normal_x[face_idx];
            let ny = mesh.face_normal_y[face_idx];
            let length = mesh.face_length[face_idx];

            // 计算面速度通量
            let hu_face = 0.5 * (hu[owner] + hu[neighbor]);
            let hv_face = 0.5 * (hv[owner] + hv[neighbor]);
            let flux_velocity = (hu_face * nx + hv_face * ny) * length;

            // 根据格式计算面浓度
            let c_face = match self.advection_scheme {
                AdvectionScheme::FirstOrderUpwind => {
                    if flux_velocity > 0.0 {
                        c[owner]
                    } else {
                        c[neighbor]
                    }
                }
                AdvectionScheme::CentralDifference => {
                    0.5 * (c[owner] + c[neighbor])
                }
                AdvectionScheme::SecondOrderUpwind => {
                    // 二阶迎风格式：使用梯度重构
                    let (upwind_cell, upwind_grad) = if flux_velocity > 0.0 {
                        (owner, self.concentration_gradient[owner])
                    } else {
                        (neighbor, self.concentration_gradient[neighbor])
                    };
                    
                    // 计算面中心到单元中心的向量
                    let cell_center = mesh.cell_centroid(upwind_cell.into());
                    let face_center = mesh.face_centroid(face_idx.into());
                    let r = face_center - cell_center;
                    
                    // 线性重构：c_f = c_c + grad · r
                    c[upwind_cell] + upwind_grad.dot(r)
                }
                AdvectionScheme::TvdMinmod => {
                    // TVD Minmod 格式：带限制器的二阶重构
                    let (upwind_cell, downwind_cell) = if flux_velocity > 0.0 {
                        (owner, neighbor)
                    } else {
                        (neighbor, owner)
                    };
                    
                    let c_u = c[upwind_cell];
                    let c_d = c[downwind_cell];
                    let grad_u = self.concentration_gradient[upwind_cell];
                    
                    // 计算面中心到单元中心的向量
                    let cell_center = mesh.cell_centroid(upwind_cell.into());
                    let face_center = mesh.face_centroid(face_idx.into());
                    let r = face_center - cell_center;
                    
                    // 梯度重构增量
                    let delta_grad = grad_u.dot(r);
                    // 中心差分增量
                    let delta_central = c_d - c_u;
                    
                    // Minmod 限制器
                    let delta = Self::minmod(delta_grad, delta_central);
                    
                    c_u + 0.5 * delta
                }
            };

            // 对流通量
            let advective_flux = flux_velocity * c_face;

            // 累加到通量缓冲
            self.flux_buffer[owner] -= advective_flux;
            self.flux_buffer[neighbor] += advective_flux;
        }

        // 边界面处理
        self.apply_boundary_advection(tracer, mesh, hu, hv);

        // 更新浓度
        let h_min = 1e-6;
        let flux_buffer = &self.flux_buffer;
        
        tracer.concentration
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, c_i)| {
                let h_i = h[i].max(h_min);
                let area = mesh.cell_area[i];
                if area > 1e-14 {
                    // hC 守恒形式
                    let hc = h_i * (*c_i);
                    let dhc = flux_buffer[i] / area * dt;
                    let new_hc = hc + dhc;
                    *c_i = new_hc / h_i;
                }
            });

        Ok(())
    }

    /// 边界平流处理
    fn apply_boundary_advection(
        &mut self,
        tracer: &TracerField,
        mesh: &UnstructuredMesh,
        hu: &[f64],
        hv: &[f64],
    ) {
        for face_idx in mesh.boundary_faces() {
            let owner = mesh.face_owner[face_idx];
            let bc_idx = mesh.boundary_index(face_idx);
            let kind = mesh.bc_kind[bc_idx];

            let nx = mesh.face_normal_x[face_idx];
            let ny = mesh.face_normal_y[face_idx];
            let length = mesh.face_length[face_idx];

            let flux_velocity = (hu[owner] * nx + hv[owner] * ny) * length;

            // 根据边界类型和条件处理
            let bc = self.boundary_conditions
                .get(bc_idx)
                .cloned()
                .unwrap_or_default();

            let boundary_flux = match (kind, bc, flux_velocity > 0.0) {
                // 出流：使用内部值
                (BoundaryKind::OpenSea | BoundaryKind::Outflow, _, true) => {
                    flux_velocity * tracer.concentration[owner]
                }
                // 入流：使用边界值
                (_, TracerBoundaryCondition::Inflow(c_bc), false) |
                (_, TracerBoundaryCondition::FixedValue(c_bc), false) => {
                    flux_velocity * c_bc
                }
                // 河流入流
                (BoundaryKind::RiverInflow, TracerBoundaryCondition::Inflow(c_bc), _) => {
                    flux_velocity * c_bc
                }
                // 墙边界：零通量
                (BoundaryKind::Wall | BoundaryKind::Symmetry, _, _) => 0.0,
                // 默认：零梯度
                _ => 0.0,
            };

            self.flux_buffer[owner] -= boundary_flux;
        }
    }

    /// 应用扩散项
    fn apply_diffusion(
        &self,
        tracer: &mut TracerField,
        mesh: &UnstructuredMesh,
        h: &[f64],
        nu_eff: &[f64],
        dt: f64,
    ) -> MhResult<()> {
        let substep_dt = dt / self.diffusion_substeps as f64;
        let h_min = 1e-6;

        for _ in 0..self.diffusion_substeps {
            let mut diffusion_flux = vec![0.0; mesh.n_cells];

            // 内部面扩散通量
            for face_idx in mesh.interior_faces() {
                let owner = mesh.face_owner[face_idx];
                let neighbor = mesh.face_neighbor[face_idx];

                let d = mesh.face_dist_o2n[face_idx];
                if d < 1e-14 {
                    continue;
                }

                let length = mesh.face_length[face_idx];
                let h_face = 0.5 * (h[owner].max(h_min) + h[neighbor].max(h_min));
                let nu_face = 0.5 * (nu_eff[owner] + nu_eff[neighbor]);

                // F = -h * ν * (C_n - C_o) / d * L
                let dc = tracer.concentration[neighbor] - tracer.concentration[owner];
                let flux = -h_face * nu_face * dc / d * length;

                diffusion_flux[owner] += flux;
                diffusion_flux[neighbor] -= flux;
            }

            // 更新浓度
            tracer.concentration
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, c_i)| {
                    let h_i = h[i].max(h_min);
                    let area = mesh.cell_area[i];
                    if area > 1e-14 {
                        let hc = h_i * (*c_i);
                        let dhc = diffusion_flux[i] / area * substep_dt;
                        *c_i = (hc + dhc) / h_i;
                    }
                });
        }

        Ok(())
    }

    /// 应用沉降（简化的底部沉积模型）
    fn apply_settling(&self, tracer: &mut TracerField, h: &[f64], dt: f64) {
        let ws = tracer.properties.settling_velocity;
        if ws <= 0.0 {
            return;
        }

        let h_min = 0.01; // 最小水深

        tracer.concentration
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, c_i)| {
                let h_i = h[i];
                if h_i > h_min {
                    // 沉降导致的浓度变化：dC/dt = -ws * C / h
                    // 使用隐式格式避免不稳定
                    let factor = 1.0 / (1.0 + ws * dt / h_i);
                    *c_i *= factor;
                }
            });
    }
}

/// 快捷函数：求解单个示踪剂一步
pub fn solve_tracer_step(
    tracer: &mut TracerField,
    mesh: &UnstructuredMesh,
    h: &[f64],
    hu: &[f64],
    hv: &[f64],
    nu_t: Option<&[f64]>,
    dt: f64,
) -> MhResult<()> {
    let mut solver = TracerTransportSolver::new(mesh.n_cells);
    solver.solve_step(tracer, mesh, h, hu, hv, nu_t, dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marihydro::domain::state::tracer::TracerProperties;

    #[test]
    fn test_advection_scheme_default() {
        let solver = TracerTransportSolver::new(100);
        assert_eq!(solver.advection_scheme, AdvectionScheme::FirstOrderUpwind);
    }

    #[test]
    fn test_effective_diffusivity() {
        let tracer = TracerField::new("test", 10, TracerProperties::default());
        let nu_t = vec![0.1; 10];
        let solver = TracerTransportSolver::new(10);
        
        let nu_eff = solver.compute_effective_diffusivity(&tracer, Some(&nu_t), 10);
        
        // ν_eff = ν_mol + ν_t / Sc
        let expected = 1e-6 + 0.1 / 0.7;
        assert!((nu_eff[0] - expected).abs() < 1e-10);
    }
}
