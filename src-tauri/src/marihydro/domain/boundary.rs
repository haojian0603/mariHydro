// src-tauri/src/marihydro/domain/boundary.rs

use ndarray::Array2;
use rayon::prelude::*;
use std::collections::HashSet;

use crate::marihydro::domain::feature::FeatureType;
use crate::marihydro::domain::mesh::Mesh;
use crate::marihydro::domain::state::State;
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::manifest::ProjectManifest;
use crate::marihydro::physics::schemes::{FluxVars, PrimitiveVars};

/// 外部强迫数据
#[derive(Debug, Clone, Default)]
pub struct ExternalForcing {
    pub eta: f64,
    pub u: f64,
    pub v: f64,
}

/// 边界面拓扑信息
#[derive(Debug, Clone, Copy)]
pub struct BoundaryFaceInfo {
    pub face_idx: usize,
    pub cell_idx: usize,
    pub normal: (f64, f64),
}

/// 边界计算参数
pub struct BoundaryParams {
    pub gravity: f64,
    pub h_min: f64,
    pub sqrt_g: f64,
}

/// 边界通量计算策略
pub trait BoundaryFluxCalculator: Send + Sync {
    fn compute_flux(
        &self,
        interior: &PrimitiveVars,
        external: &ExternalForcing,
        normal: (f64, f64),
        params: &BoundaryParams,
    ) -> FluxVars;
}

/// 固壁边界
struct WallBoundary;
impl BoundaryFluxCalculator for WallBoundary {
    fn compute_flux(
        &self,
        interior: &PrimitiveVars,
        _ext: &ExternalForcing,
        normal: (f64, f64),
        params: &BoundaryParams,
    ) -> FluxVars {
        let p = 0.5 * params.gravity * interior.h * interior.h;
        FluxVars {
            mass: 0.0,
            x_mom: p * normal.0,
            y_mom: p * normal.1,
            sed: 0.0,
        }
    }
}

/// 自由出流边界
struct OpenFlowBoundary;
impl BoundaryFluxCalculator for OpenFlowBoundary {
    fn compute_flux(
        &self,
        interior: &PrimitiveVars,
        _ext: &ExternalForcing,
        normal: (f64, f64),
        params: &BoundaryParams,
    ) -> FluxVars {
        let qn = interior.h * (interior.u * normal.0 + interior.v * normal.1);
        let p = 0.5 * params.gravity * interior.h.powi(2);
        FluxVars {
            mass: qn,
            x_mom: qn * interior.u + p * normal.0,
            y_mom: qn * interior.v + p * normal.1,
            sed: qn * interior.c,
        }
    }
}

/// Flather辐射边界 (简化版)
struct FlatherBoundary;
impl FlatherBoundary {
    #[inline(always)]
    fn physical_flux_1d(&self, h: f64, u: f64, g: f64) -> Flux1D {
        let q = h * u;
        Flux1D {
            mass: q,
            momentum: q * u + 0.5 * g * h * h,
        }
    }
}

struct Flux1D {
    mass: f64,
    momentum: f64,
}

impl BoundaryFluxCalculator for FlatherBoundary {
    fn compute_flux(
        &self,
        interior: &PrimitiveVars,
        external: &ExternalForcing,
        normal: (f64, f64),
        params: &BoundaryParams,
    ) -> FluxVars {
        // 步骤1: 法向速度分解 (4分支优化)
        let (u_n_int, u_t_int, u_n_ext) = match normal {
            (1.0, 0.0) => (interior.u, interior.v, external.u),
            (-1.0, 0.0) => (-interior.u, interior.v, -external.u),
            (0.0, 1.0) => (interior.v, interior.u, external.v),
            (0.0, -1.0) => (-interior.v, interior.u, -external.v),
            _ => {
                log::warn!("非轴对齐法向: {:?}", normal);
                return FluxVars::default();
            }
        };

        // 步骤2: Flather修正
        let h_safe = interior.h.max(params.h_min);
        let c = params.sqrt_g * h_safe.sqrt();
        let u_n_star = u_n_ext + (c / h_safe) * (interior.eta - external.eta);

        // 步骤3: 1D HLLC求解
        let h_ext = (external.eta - interior.z).max(params.h_min);
        let (h_l, u_l) = (interior.h, u_n_int);
        let (h_r, u_r) = (h_ext, u_n_star);

        let a_l = (params.gravity * h_l).sqrt();
        let a_r = (params.gravity * h_r).sqrt();
        let sl = (u_l - a_l).min(u_r - a_r);
        let sr = (u_l + a_l).max(u_r + a_r);

        let flux_1d = if sl >= 0.0 {
            self.physical_flux_1d(h_l, u_l, params.gravity)
        } else if sr <= 0.0 {
            self.physical_flux_1d(h_r, u_r, params.gravity)
        } else {
            let s_star = (u_r * h_r * (sr - u_r) - u_l * h_l * (sl - u_l)
                + 0.5 * params.gravity * (h_l * h_l - h_r * h_r))
                / (h_r * (sr - u_r) - h_l * (sl - u_l));

            let (sk, hk, uk) = if s_star >= 0.0 {
                (sl, h_l, u_l)
            } else {
                (sr, h_r, u_r)
            };
            let flux_k = self.physical_flux_1d(hk, uk, params.gravity);
            let h_star = hk * (sk - uk) / (sk - s_star);

            Flux1D {
                mass: flux_k.mass + sk * (h_star - hk),
                momentum: flux_k.momentum + sk * (h_star * s_star - hk * uk),
            }
        };

        // 步骤4: 投影回2D
        match normal {
            (1.0, 0.0) | (-1.0, 0.0) => FluxVars {
                mass: flux_1d.mass,
                x_mom: flux_1d.momentum,
                y_mom: flux_1d.mass * u_t_int,
                sed: flux_1d.mass * interior.c,
            },
            (0.0, 1.0) | (0.0, -1.0) => FluxVars {
                mass: flux_1d.mass,
                x_mom: flux_1d.mass * u_t_int,
                y_mom: flux_1d.momentum,
                sed: flux_1d.mass * interior.c,
            },
            _ => FluxVars::default(),
        }
    }
}

/// 边界数据提供者
pub trait BoundaryDataProvider: Sync + Send {
    fn get_forcing(&self, face_idx: usize, t: f64) -> MhResult<ExternalForcing>;
}

/// 边界管理器
pub struct BoundaryManager {
    walls: Vec<BoundaryFaceInfo>,
    open_flows: Vec<BoundaryFaceInfo>,
    flathers: Vec<BoundaryFaceInfo>,

    params: BoundaryParams,
    impl_wall: WallBoundary,
    impl_open: OpenFlowBoundary,
    impl_flather: FlatherBoundary,
}

impl BoundaryManager {
    pub fn from_manifest(manifest: &ProjectManifest) -> Self {
        Self {
            walls: Vec::new(),
            open_flows: Vec::new(),
            flathers: Vec::new(),
            params: BoundaryParams {
                gravity: manifest.physics.gravity,
                h_min: manifest.physics.h_min,
                sqrt_g: manifest.physics.gravity.sqrt(),
            },
            impl_wall: WallBoundary,
            impl_open: OpenFlowBoundary,
            impl_flather: FlatherBoundary,
        }
    }

    pub fn register_structured_mesh(&mut self, mesh: &Mesh, manifest: &ProjectManifest) {
        let ng = mesh.ng;
        let (ny, nx) = (mesh.ny, mesh.nx);
        let stride = nx + 2 * ng;

        let get_bc_type = |keywords: &[&str]| -> BcKind {
            for feat in &manifest.features {
                if let FeatureType::Boundary { mode, .. } = &feat.feature_type {
                    let name = feat.name.to_lowercase();
                    if keywords.iter().any(|k| name.contains(k)) {
                        return (*mode).into();
                    }
                }
            }
            BcKind::Wall
        };

        let offset_y = ny * (nx + 1);

        let bc = get_bc_type(&["left", "west"]);
        for j in 0..ny {
            self.add_face(bc, j * (nx + 1), (j + ng) * stride + ng, (-1.0, 0.0));
        }

        let bc = get_bc_type(&["right", "east"]);
        for j in 0..ny {
            self.add_face(
                bc,
                j * (nx + 1) + nx,
                (j + ng) * stride + (nx + ng - 1),
                (1.0, 0.0),
            );
        }

        let bc = get_bc_type(&["bottom", "south"]);
        for i in 0..nx {
            self.add_face(bc, offset_y + i, ng * stride + (i + ng), (0.0, -1.0));
        }

        let bc = get_bc_type(&["top", "north"]);
        for i in 0..nx {
            self.add_face(
                bc,
                offset_y + ny * nx + i,
                (ny + ng - 1) * stride + (i + ng),
                (0.0, 1.0),
            );
        }
    }

    fn add_face(&mut self, mode: BcKind, face_idx: usize, cell_idx: usize, normal: (f64, f64)) {
        let info = BoundaryFaceInfo {
            face_idx,
            cell_idx,
            normal,
        };
        match mode {
            BcKind::Wall => self.walls.push(info),
            BcKind::Flow | BcKind::Open => self.open_flows.push(info),
            BcKind::Tide | BcKind::Radiation => self.flathers.push(info),
        }
    }

    /// 填充Ghost Cells
    pub fn fill_ghost_cells(&self, state: &mut State) {
        let ng = state.ng;
        let nx = state.nx;
        let ny = state.ny;
        let stride = nx + 2 * ng;

        for j in 0..ny {
            let inner = (j + ng) * stride + ng;
            for k in 0..ng {
                let ghost = (j + ng) * stride + k;
                state.copy_cell(ghost, inner);
            }
        }

        for j in 0..ny {
            let inner = (j + ng) * stride + (nx + ng - 1);
            for k in 0..ng {
                let ghost = (j + ng) * stride + (nx + ng + k);
                state.copy_cell(ghost, inner);
            }
        }

        for i in 0..nx {
            let inner = ng * stride + (i + ng);
            for k in 0..ng {
                let ghost = k * stride + (i + ng);
                state.copy_cell(ghost, inner);
            }
        }

        for i in 0..nx {
            let inner = (ny + ng - 1) * stride + (i + ng);
            for k in 0..ng {
                let ghost = (ny + ng + k) * stride + (i + ng);
                state.copy_cell(ghost, inner);
            }
        }
    }

    /// 计算并应用边界通量 (安全版)
    pub fn apply_boundary_fluxes(
        &self,
        fluxes: &mut [FluxVars],
        state: &State,
        provider: &impl BoundaryDataProvider,
        time: f64,
    ) -> MhResult<()> {
        // 阶段1: 并行计算 (无副作用)
        let flather_results: Vec<(usize, FluxVars)> = self
            .flathers
            .par_iter()
            .map(|face| -> MhResult<(usize, FluxVars)> {
                let prim = state.get_primitive(face.cell_idx);
                let forcing = provider.get_forcing(face.face_idx, time)?;
                let flux =
                    self.impl_flather
                        .compute_flux(&prim, &forcing, face.normal, &self.params);
                Ok((face.face_idx, flux))
            })
            .collect::<MhResult<Vec<_>>>()?;

        // 阶段2: 串行写入 (无竞争)
        for (idx, flux) in flather_results {
            fluxes[idx] = flux;
        }

        for face in &self.walls {
            let prim = state.get_primitive(face.cell_idx);
            fluxes[face.face_idx] =
                self.impl_wall
                    .compute_flux(&prim, &Default::default(), face.normal, &self.params);
        }

        for face in &self.open_flows {
            let prim = state.get_primitive(face.cell_idx);
            fluxes[face.face_idx] =
                self.impl_open
                    .compute_flux(&prim, &Default::default(), face.normal, &self.params);
        }

        Ok(())
    }

    /// 验证配置
    pub fn validate(&self) -> MhResult<()> {
        let mut seen = HashSet::new();
        for face in self.all_faces() {
            let (nx, ny) = face.normal;
            let mag_sq = nx * nx + ny * ny;
            if (mag_sq - 1.0).abs() > 1e-10 {
                return Err(MhError::InvalidConfig {
                    field: "boundary_normal".into(),
                    message: format!("法向未单位化: ({}, {})", nx, ny),
                });
            }
            if !seen.insert(face.face_idx) {
                return Err(MhError::InvalidConfig {
                    field: "face_idx".into(),
                    message: format!("重复索引: {}", face.face_idx),
                });
            }
        }
        Ok(())
    }

    fn all_faces(&self) -> impl Iterator<Item = &BoundaryFaceInfo> {
        self.walls
            .iter()
            .chain(&self.open_flows)
            .chain(&self.flathers)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BcKind {
    Wall,
    Flow,
    Open,
    Tide,
    Radiation,
}

/// 旧版接口兼容 (待删除)
#[derive(Debug, Clone, Default)]
pub struct BoundaryForcing {
    pub left_level: Option<f64>,
    pub right_level: Option<f64>,
    pub top_level: Option<f64>,
    pub bottom_level: Option<f64>,
}

impl BoundaryManager {
    pub fn update_ghost_cells(
        &self,
        state: &mut State,
        _mesh_zb: &Array2<f64>,
        _forcing: &BoundaryForcing,
    ) {
        self.fill_ghost_cells(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wall_boundary_flux() {
        let wall = WallBoundary;
        let params = BoundaryParams {
            gravity: 9.81,
            h_min: 0.01,
            sqrt_g: 9.81f64.sqrt(),
        };
        let interior = PrimitiveVars {
            h: 1.0,
            u: 10.0, // Strong flow towards the wall
            v: 0.0,
            c: 0.0,
            z: 0.0,
            eta: 1.0,
        };
        let external = ExternalForcing::default();
        let normal = (-1.0, 0.0); // Wall on the right, flow from the left

        let flux = wall.compute_flux(&interior, &external, normal, &params);

        assert_eq!(flux.mass, 0.0, "Wall should block mass flux");

        let pressure_force = 0.5 * params.gravity * interior.h * interior.h * normal.0;
        assert!(
            (flux.x_mom - pressure_force).abs() < 1e-9,
            "Wall x-momentum should only be pressure"
        );
        assert!(
            flux.y_mom.abs() < 1e-9,
            "No y-momentum should be generated for x-normal wall"
        );
    }
}
