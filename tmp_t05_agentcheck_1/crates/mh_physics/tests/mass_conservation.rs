// crates/mh_physics/tests/mass_conservation.rs

//! 干湿质量守恒验证测试
//!
//! 全面检验求解器在各种干湿场景下的质量守恒性能和静水平衡特性

use std::sync::Arc;
use std::sync::LazyLock;
use mh_mesh::FrozenMesh;
use mh_physics::adapter::PhysicsMesh;
use mh_physics::engine::ShallowWaterSolver;
use mh_physics::sources::NoSource;
use mh_physics::NumericalScheme;
use mh_physics::Layer3Config;
use mh_physics::state::ShallowWaterState;
use mh_geo::{Point2D, Point3D};
use mh_runtime::prelude::*;

/// 全局Backend实例，强制单例模式
static BACKEND: LazyLock<CpuBackend<f64>> = LazyLock::new(|| CpuBackend::<f64>::new());

fn create_state(n_cells: usize) -> ShallowWaterState<CpuBackend<f64>> {
    ShallowWaterState::new_with_backend(*BACKEND, n_cells)
}

fn create_solver(
    mesh: Arc<PhysicsMesh>,
    config: Layer3Config<f64>,
) -> ShallowWaterSolver<CpuBackend<f64>, NoSource<CpuBackend<f64>>> {
    ShallowWaterSolver::<CpuBackend<f64>, NoSource<CpuBackend<f64>>>::new(mesh, config, *BACKEND)
}

/// 创建2x2简单网格
fn create_simple_mesh() -> PhysicsMesh {
    create_rectangular_mesh(2, 2, 1.0, 1.0, |_, _| 0.0)
}

/// 创建矩形网格
fn create_rectangular_mesh(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    z_func: impl Fn(f64, f64) -> f64,
) -> PhysicsMesh {
    let n_nodes = (nx + 1) * (ny + 1);
    let n_cells = nx * ny;
    
    let mut node_coords = Vec::with_capacity(n_nodes);
    for j in 0..=ny {
        for i in 0..=nx {
            let x = i as f64 * dx;
            let y = j as f64 * dy;
            let z = z_func(x, y);
            node_coords.push(Point3D::new(x, y, z));
        }
    }
    
    let mut cell_center = Vec::with_capacity(n_cells);
    let mut cell_area = Vec::with_capacity(n_cells);
    let mut cell_z_bed = Vec::with_capacity(n_cells);
    
    for j in 0..ny {
        for i in 0..nx {
            let cx = (i as f64 + 0.5) * dx;
            let cy = (j as f64 + 0.5) * dy;
            cell_center.push(Point2D::new(cx, cy));
            cell_area.push(dx * dy);
            cell_z_bed.push(z_func(cx, cy));
        }
    }
    
    let mut cell_node_offsets = vec![0usize];
    let mut cell_node_indices: Vec<u32> = Vec::new();
    
    for j in 0..ny {
        for i in 0..nx {
            let n0 = (j * (nx + 1) + i) as u32;
            let n1 = n0 + 1;
            let n2 = n1 + (nx + 1) as u32;
            let n3 = n0 + (nx + 1) as u32;
            cell_node_indices.extend_from_slice(&[n0, n1, n2, n3]);
            cell_node_offsets.push(cell_node_indices.len());
        }
    }
    
    let n_interior_h = nx * (ny - 1);
    let n_interior_v = (nx - 1) * ny;
    let n_interior = n_interior_h + n_interior_v;
    let n_boundary = 2 * nx + 2 * ny;
    let n_faces = n_interior + n_boundary;
    
    let mut face_center = Vec::with_capacity(n_faces);
    let mut face_normal = Vec::with_capacity(n_faces);
    let mut face_length = Vec::with_capacity(n_faces);
    let mut face_owner: Vec<u32> = Vec::with_capacity(n_faces);
    let mut face_neighbor: Vec<u32> = Vec::with_capacity(n_faces);
    let mut face_z_left = Vec::with_capacity(n_faces);
    let mut face_z_right = Vec::with_capacity(n_faces);
    
    let cell_idx = |i: usize, j: usize| -> u32 { (j * nx + i) as u32 };
    
    // 垂直内部面
    for j in 0..ny {
        for i in 0..(nx - 1) {
            let x = (i + 1) as f64 * dx;
            let y = (j as f64 + 0.5) * dy;
            face_center.push(Point2D::new(x, y));
            face_normal.push(Point3D::new(1.0, 0.0, 0.0));
            face_length.push(dy);
            face_owner.push(cell_idx(i, j));
            face_neighbor.push(cell_idx(i + 1, j));
            face_z_left.push(z_func(x - 0.5 * dx, y));
            face_z_right.push(z_func(x + 0.5 * dx, y));
        }
    }
    
    // 水平内部面
    for j in 0..(ny - 1) {
        for i in 0..nx {
            let x = (i as f64 + 0.5) * dx;
            let y = (j + 1) as f64 * dy;
            face_center.push(Point2D::new(x, y));
            face_normal.push(Point3D::new(0.0, 1.0, 0.0));
            face_length.push(dx);
            face_owner.push(cell_idx(i, j));
            face_neighbor.push(cell_idx(i, j + 1));
            face_z_left.push(z_func(x, y - 0.5 * dy));
            face_z_right.push(z_func(x, y + 0.5 * dy));
        }
    }
    
    let boundary_start = face_center.len();
    
    // 边界下面
    for i in 0..nx {
        let x = (i as f64 + 0.5) * dx;
        face_center.push(Point2D::new(x, 0.0));
        face_normal.push(Point3D::new(0.0, -1.0, 0.0));
        face_length.push(dx);
        face_owner.push(cell_idx(i, 0));
        face_neighbor.push(u32::MAX);
        let z = z_func(x, 0.5 * dy);
        face_z_left.push(z);
        face_z_right.push(z);
    }
    
    // 边界右面
    for j in 0..ny {
        let y = (j as f64 + 0.5) * dy;
        face_center.push(Point2D::new(nx as f64 * dx, y));
        face_normal.push(Point3D::new(1.0, 0.0, 0.0));
        face_length.push(dy);
        face_owner.push(cell_idx(nx - 1, j));
        face_neighbor.push(u32::MAX);
        let z = z_func((nx as f64 - 0.5) * dx, y);
        face_z_left.push(z);
        face_z_right.push(z);
    }
    
    // 边界上面
    for i in 0..nx {
        let x = (i as f64 + 0.5) * dx;
        face_center.push(Point2D::new(x, ny as f64 * dy));
        face_normal.push(Point3D::new(0.0, 1.0, 0.0));
        face_length.push(dx);
        face_owner.push(cell_idx(i, ny - 1));
        face_neighbor.push(u32::MAX);
        let z = z_func(x, (ny as f64 - 0.5) * dy);
        face_z_left.push(z);
        face_z_right.push(z);
    }
    
    // 边界左面
    for j in 0..ny {
        let y = (j as f64 + 0.5) * dy;
        face_center.push(Point2D::new(0.0, y));
        face_normal.push(Point3D::new(-1.0, 0.0, 0.0));
        face_length.push(dy);
        face_owner.push(cell_idx(0, j));
        face_neighbor.push(u32::MAX);
        let z = z_func(0.5 * dx, y);
        face_z_left.push(z);
        face_z_right.push(z);
    }
    
    let mut cell_face_offsets = vec![0usize];
    let mut cell_face_indices: Vec<u32> = Vec::new();
    let mut cell_neighbor_offsets = vec![0usize];
    let mut cell_neighbor_indices: Vec<u32> = Vec::new();
    
    for j in 0..ny {
        for i in 0..nx {
            let mut faces: Vec<u32> = Vec::new();
            let mut neighbors: Vec<u32> = Vec::new();
            
            if i > 0 {
                let fidx = (j * (nx - 1) + (i - 1)) as u32;
                faces.push(fidx);
                neighbors.push(cell_idx(i - 1, j));
            } else {
                let fidx = (boundary_start + 2 * nx + ny + j) as u32;
                faces.push(fidx);
            }
            
            if i < nx - 1 {
                let fidx = (j * (nx - 1) + i) as u32;
                faces.push(fidx);
                neighbors.push(cell_idx(i + 1, j));
            } else {
                let fidx = (boundary_start + nx + j) as u32;
                faces.push(fidx);
            }
            
            if j > 0 {
                let fidx = (n_interior_v + (j - 1) * nx + i) as u32;
                faces.push(fidx);
                neighbors.push(cell_idx(i, j - 1));
            } else {
                let fidx = (boundary_start + i) as u32;
                faces.push(fidx);
            }
            
            if j < ny - 1 {
                let fidx = (n_interior_v + j * nx + i) as u32;
                faces.push(fidx);
                neighbors.push(cell_idx(i, j + 1));
            } else {
                let fidx = (boundary_start + nx + ny + i) as u32;
                faces.push(fidx);
            }
            
            cell_face_indices.extend_from_slice(&faces);
            cell_face_offsets.push(cell_face_indices.len());
            cell_neighbor_indices.extend_from_slice(&neighbors);
            cell_neighbor_offsets.push(cell_neighbor_indices.len());
        }
    }
    
    let face_delta_owner = vec![Point2D::ZERO; n_faces];
    let face_delta_neighbor = vec![Point2D::ZERO; n_faces];
    let face_dist_o2n = vec![dx.min(dy); n_faces];
    let boundary_face_indices: Vec<u32> = (boundary_start..n_faces).map(|i| i as u32).collect();

    let backend = *BACKEND;
    let mut cell_area_buf = backend.alloc(cell_area.len());
    cell_area_buf.copy_from_slice(&cell_area);
    let mut cell_z_bed_buf = backend.alloc(cell_z_bed.len());
    cell_z_bed_buf.copy_from_slice(&cell_z_bed);
    let mut face_length_buf = backend.alloc(face_length.len());
    face_length_buf.copy_from_slice(&face_length);
    let mut face_z_left_buf = backend.alloc(face_z_left.len());
    face_z_left_buf.copy_from_slice(&face_z_left);
    let mut face_z_right_buf = backend.alloc(face_z_right.len());
    face_z_right_buf.copy_from_slice(&face_z_right);
    let mut face_dist_o2n_buf = backend.alloc(face_dist_o2n.len());
    face_dist_o2n_buf.copy_from_slice(&face_dist_o2n);
    
    let mut frozen = FrozenMesh::empty_with_backend(backend);
    frozen.n_nodes = n_nodes;
    frozen.node_coords = node_coords;
    frozen.n_cells = n_cells;
    frozen.cell_center = cell_center;
    frozen.cell_area = cell_area_buf;
    frozen.cell_z_bed = cell_z_bed_buf;
    frozen.cell_node_offsets = cell_node_offsets;
    frozen.cell_node_indices = cell_node_indices;
    frozen.cell_face_offsets = cell_face_offsets;
    frozen.cell_face_indices = cell_face_indices;
    frozen.cell_neighbor_offsets = cell_neighbor_offsets;
    frozen.cell_neighbor_indices = cell_neighbor_indices;
    frozen.n_faces = n_faces;
    frozen.n_interior_faces = n_interior;
    frozen.face_center = face_center;
    frozen.face_normal = face_normal;
    frozen.face_length = face_length_buf;
    frozen.face_z_left = face_z_left_buf;
    frozen.face_z_right = face_z_right_buf;
    frozen.face_owner = face_owner;
    frozen.face_neighbor = face_neighbor;
    frozen.face_delta_owner = face_delta_owner;
    frozen.face_delta_neighbor = face_delta_neighbor;
    frozen.face_dist_o2n = face_dist_o2n_buf;
    frozen.boundary_face_indices = boundary_face_indices;
    frozen.boundary_names = vec!["boundary".to_string()];
    frozen.face_boundary_id = (0..n_faces)
        .map(|i| if i >= n_interior { Some(0) } else { None })
        .collect();
    frozen.min_cell_size = dx.min(dy);
    frozen.max_cell_size = dx.max(dy);
    frozen.cell_refinement_level = vec![0; n_cells];
    frozen.cell_parent = (0..n_cells as u32).collect();
    frozen.ghost_capacity = 0;
    frozen.cell_original_id = Vec::new();
    frozen.face_original_id = Vec::new();
    frozen.cell_permutation = Vec::new();
    frozen.cell_inv_permutation = Vec::new();
    
    PhysicsMesh::from_frozen(&frozen)
}

/// 计算总质量 (h * area) [m³]
fn compute_total_mass(state: &ShallowWaterState<CpuBackend<f64>>, mesh: &PhysicsMesh) -> f64 {
    let mut total = 0.0;
    for i in 0..state.n_cells() {
        if let Some(area) = mesh.cell_area(CellIndex::new(i)) {
            total += state.h[i] * area;
        }
    }
    total
}

#[allow(dead_code)]
/// 计算总动量 (hu * area, hv * area) [m³/s]
fn compute_total_momentum(state: &ShallowWaterState<CpuBackend<f64>>, mesh: &PhysicsMesh) -> (f64, f64) {
    let mut total_hu = 0.0;
    let mut total_hv = 0.0;
    for i in 0..state.n_cells() {
        if let Some(area) = mesh.cell_area(CellIndex::new(i)) {
            total_hu += state.hu[i] * area;
            total_hv += state.hv[i] * area;
        }
    }
    (total_hu, total_hv)
}

#[allow(dead_code)]
/// 计算总能量 (势能 + 动能) [J/m²]
fn compute_total_energy(state: &ShallowWaterState<CpuBackend<f64>>, mesh: &PhysicsMesh, g: f64) -> f64 {
    let mut total = 0.0;
    for i in 0..state.n_cells() {
        if let Some(area) = mesh.cell_area(CellIndex::new(i)) {
            let h = state.h[i];
            let z = state.z[i];
            let pe = 0.5 * g * h * h + g * h * z;
            let ke = if h > 1e-10 {
                0.5 * (state.hu[i].powi(2) + state.hv[i].powi(2)) / h
            } else {
                0.0
            };
            total += (pe + ke) * area;
        }
    }
    total
}

/// 验证状态有效性
fn validate_state(state: &ShallowWaterState<CpuBackend<f64>>, step: usize) -> Result<(), String> {
    for (i, &h) in state.h.iter().enumerate() {
        if h < 0.0 {
            return Err(format!("步骤 {} 单元 {} 负水深: {}", step, i, h));
        }
        if h.is_nan() {
            return Err(format!("步骤 {} 单元 {} NaN 水深", step, i));
        }
        if !h.is_finite() {
            return Err(format!("步骤 {} 单元 {} 无穷水深", step, i));
        }
    }
    for (i, &hu) in state.hu.iter().enumerate() {
        if hu.is_nan() || !hu.is_finite() {
            return Err(format!("步骤 {} 单元 {} 无效动量 hu={}", step, i, hu));
        }
    }
    for (i, &hv) in state.hv.iter().enumerate() {
        if hv.is_nan() || !hv.is_finite() {
            return Err(format!("步骤 {} 单元 {} 无效动量 hv={}", step, i, hv));
        }
    }
    Ok(())
}

struct SimulationResult {
    initial_mass: f64,
    final_mass: f64,
    mass_error: f64,
    relative_error: f64,
    max_velocity: f64,
    final_state: ShallowWaterState<CpuBackend<f64>>,
}

/// 执行模拟并返回统计结果
fn run_simulation(
    solver: &mut ShallowWaterSolver<CpuBackend<f64>, NoSource<CpuBackend<f64>>>,
    state: &mut ShallowWaterState<CpuBackend<f64>>,
    mesh: &PhysicsMesh,
    dt: f64,
    n_steps: usize,
) -> Result<SimulationResult, String> {
    let initial_mass = compute_total_mass(state, mesh);
    let mut max_velocity = 0.0f64;
    
    for step in 0..n_steps {
        solver.step(state, dt);
        validate_state(state, step)?;
        
        for i in 0..state.n_cells() {
            if state.h[i] > 1e-6 {
                let u = state.hu[i] / state.h[i];
                let v = state.hv[i] / state.h[i];
                let speed = (u * u + v * v).sqrt();
                max_velocity = max_velocity.max(speed);
            }
        }
    }
    
    let final_mass = compute_total_mass(state, mesh);
    let mass_error = (final_mass - initial_mass).abs();
    let relative_error = if initial_mass > 1e-12 {
        mass_error / initial_mass
    } else {
        mass_error
    };
    
    Ok(SimulationResult {
        initial_mass,
        final_mass,
        mass_error,
        relative_error,
        max_velocity,
        final_state: state.clone(),
    })
}

/// 基础干湿测试：验证质量在干湿转换过程中的守恒性
#[test]
fn test_mass_conservation_wetting_drying() {
    let mesh = Arc::new(create_simple_mesh());
    let config = Layer3Config::builder()
        .scheme(NumericalScheme::FirstOrder)
        .build();
    let mut solver = create_solver(mesh.clone(), config);
    
    let mut state = create_state(4);
    state.h = vec![0.1, 0.1, 0.0, 0.0];
    state.z = vec![0.0; 4];
    state.hu = vec![0.0; 4];
    state.hv = vec![0.0; 4];
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.001, 100)
        .expect("模拟失败");
    
    // 打印绝对误差和相对误差，提供完整诊断信息
    println!("基础干湿测试: 初始={:.10} 最终={:.10} 绝对误差={:.2e} 相对误差={:.2e}",
             result.initial_mass, result.final_mass, result.mass_error, result.relative_error);
    
    assert!(result.relative_error < 1e-8,
            "质量守恒失败！相对误差 {:.2e}", result.relative_error);
}

#[test]
fn test_mass_conservation_all_wet() {
    let mesh = Arc::new(create_simple_mesh());
    let config = Layer3Config::default();
    let mut solver = create_solver(mesh.clone(), config);
    
    let mut state = create_state(4);
    state.h = vec![1.0, 1.0, 1.0, 1.0];
    state.z = vec![0.0; 4];
    state.hu = vec![0.0; 4];
    state.hv = vec![0.0; 4];
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.001, 100)
        .expect("模拟失败");
    
    assert!(result.relative_error < 1e-10,
            "全湿质量守恒失败！误差 {:.2e}", result.relative_error);
}

#[test]
fn test_mass_conservation_all_dry() {
    let mesh = Arc::new(create_simple_mesh());
    let config = Layer3Config::default();
    let mut solver = create_solver(mesh.clone(), config);
    
    let mut state = create_state(4);
    state.h = vec![0.0, 0.0, 0.0, 0.0];
    state.z = vec![0.0; 4];
    state.hu = vec![0.0; 4];
    state.hv = vec![0.0; 4];
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.001, 100)
        .expect("模拟失败");
    
    assert!(result.final_mass.abs() < 1e-12,
            "全干情况出现水量！质量 {:.2e}", result.final_mass);
}

#[test]
fn test_mass_conservation_single_wet_cell() {
    let mesh = Arc::new(create_simple_mesh());
    let config = Layer3Config::builder()
        .scheme(NumericalScheme::FirstOrder)
        .build();
    let mut solver = create_solver(mesh.clone(), config);
    
    let mut state = create_state(4);
    state.h = vec![0.5, 0.0, 0.0, 0.0];
    state.z = vec![0.0; 4];
    state.hu = vec![0.0; 4];
    state.hv = vec![0.0; 4];
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.001, 150)
        .expect("模拟失败");
    
    println!("单湿单元: 绝对误差={:.2e} 相对误差={:.2e}", result.mass_error, result.relative_error);
    assert!(result.relative_error < 1e-8,
            "单湿单元守恒失败！误差 {:.2e}", result.relative_error);
}

/// 平底静水测试：验证C-property（静水平衡保持）
#[test]
fn test_lake_at_rest_flat_bed() {
    let mesh = Arc::new(create_simple_mesh());
    let config = Layer3Config::builder()
        .scheme(NumericalScheme::FirstOrder)
        .build();
    let mut solver = create_solver(mesh.clone(), config);
    
    let mut state = create_state(4);
    state.h = vec![1.0; 4];
    state.z = vec![0.0; 4];
    state.hu = vec![0.0; 4];
    state.hv = vec![0.0; 4];
    
    let initial_h = state.h.clone();
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.001, 300)
        .expect("模拟失败");
    
    assert!(result.max_velocity < 1e-8,
            "平底静水产生了速度！max_vel={:.2e}", result.max_velocity);
    
    for (i, (&h_init, &h_final)) in initial_h.iter().zip(result.final_state.h.iter()).enumerate() {
        let diff = (h_final - h_init).abs();
        assert!(diff < 1e-8, "单元 {} 水深变化: {:.2e}", i, diff);
    }
}

/// 倾斜底床静水测试
#[test]
fn test_lake_at_rest_sloped_bed() {
    let mesh = Arc::new(create_rectangular_mesh(4, 4, 1.0, 1.0, |x, _| 0.1 * x));
    let config = Layer3Config::builder()
        .scheme(NumericalScheme::FirstOrder)
        .build();
    let mut solver = create_solver(mesh.clone(), config);
    
    let eta = 1.0;
    let n_cells = 16;
    let mut state = create_state(n_cells);
    
    for i in 0..n_cells {
        let z = mesh.cell_z_bed(CellIndex::new(i));
        state.h[i] = (eta - z).max(0.0);
        state.z[i] = z;
    }
    state.hu = vec![0.0; n_cells];
    state.hv = vec![0.0; n_cells];
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.001, 100)
        .expect("模拟失败");
    
    println!("倾斜底床静水: max_vel={:.2e}, mass_err={:.2e}",
             result.max_velocity, result.relative_error);
    
        assert!(result.relative_error < 1e-8,
            "质量守恒失败！误差 {:.2e}", result.relative_error);
        assert!(result.max_velocity < 1e-6,
            "C-property失败！速度过大 {:.2e}", result.max_velocity);
}

/// 凸起底床静水测试
#[test]
fn test_lake_at_rest_bump() {
    let mesh = Arc::new(create_rectangular_mesh(4, 4, 1.0, 1.0, |x, y| {
        let cx = 2.0;
        let cy = 2.0;
        let r = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        if r < 1.0 {
            0.2 * (1.0 - r)
        } else {
            0.0
        }
    }));
    
    let config = Layer3Config::builder()
        .scheme(NumericalScheme::FirstOrder)
        .build();
    let mut solver = create_solver(mesh.clone(), config);
    
    let eta = 0.5;
    let n_cells = 16;
    let mut state = create_state(n_cells);
    
    for i in 0..n_cells {
        let z = mesh.cell_z_bed(CellIndex::new(i));
        state.h[i] = (eta - z).max(0.0);
        state.z[i] = z;
    }
    state.hu = vec![0.0; n_cells];
    state.hv = vec![0.0; n_cells];
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.001, 120)
        .expect("模拟失败");
    
    println!("凸起底床静水: max_vel={:.2e}, mass_err={:.2e}",
             result.max_velocity, result.relative_error);
    
        assert!(result.relative_error < 1e-8,
            "质量守恒失败！误差 {:.2e}", result.relative_error);
        assert!(result.max_velocity < 1e-6,
            "C-property失败！速度过大 {:.2e}", result.max_velocity);
}

/// 溃坝测试：验证动态流动的质量守恒
#[test]
fn test_dam_break_mass_conservation() {
    let mesh = Arc::new(create_simple_mesh());
    let config = Layer3Config::builder()
        .scheme(NumericalScheme::FirstOrder)
        .build();
    let mut solver = create_solver(mesh.clone(), config);
    
    let mut state = create_state(4);
    state.h = vec![1.0, 0.1, 1.0, 0.1];
    state.z = vec![0.0; 4];
    state.hu = vec![0.0; 4];
    state.hv = vec![0.0; 4];
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.0005, 150)
        .expect("模拟失败");
    
    println!("溃坝测试: mass_err={:.2e}, max_vel={:.2}",
             result.mass_error, result.max_velocity);
    
    assert!(result.relative_error < 1e-8,
            "溃坝质量守恒失败！误差 {:.2e}", result.relative_error);
    assert!(result.max_velocity > 0.1,
            "溃坝未产生足够流速");
}

#[test]
fn test_wetting_drying_cycle() {
    let mesh = Arc::new(create_simple_mesh());
    let config = Layer3Config::builder()
        .scheme(NumericalScheme::FirstOrder)
        .build();
    let mut solver = create_solver(mesh.clone(), config);
    
    let mut state = create_state(4);
    state.h = vec![0.2, 0.2, 0.0, 0.0];
    state.z = vec![0.0; 4];
    state.hu = vec![0.0; 4];
    state.hv = vec![0.0; 4];
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.001, 250)
        .expect("模拟失败");
    
    println!("润湿循环: 绝对误差={:.2e} 相对误差={:.2e}", result.mass_error, result.relative_error);
    assert!(result.relative_error < 1e-8,
            "动态润湿守恒失败！误差 {:.2e}", result.relative_error);
}

#[test]
fn test_uniform_flow_conservation() {
    let mesh = Arc::new(create_rectangular_mesh(4, 4, 1.0, 1.0, |_, _| 0.0));
    let config = Layer3Config::default();
    let mut solver = create_solver(mesh.clone(), config);
    
    let n_cells = 16;
    let mut state = create_state(n_cells);
    let h0 = 1.0;
    let u0 = 0.1;
    
    for i in 0..n_cells {
        state.h[i] = h0;
        state.hu[i] = h0 * u0;
        state.hv[i] = 0.0;
        state.z[i] = 0.0;
    }
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.001, 120)
        .expect("模拟失败");
    
    println!("均匀流: 绝对误差={:.2e} 相对误差={:.2e}", result.mass_error, result.relative_error);
    assert!(result.relative_error < 1e-8,
            "均匀流守恒失败！误差 {:.2e}", result.relative_error);
}

/// 极端水深比测试（100:1）
#[test]
fn test_extreme_depth_ratio() {
    let mesh = Arc::new(create_simple_mesh());
    let config = Layer3Config::builder()
        .scheme(NumericalScheme::FirstOrder)
        .build();
    let mut solver = create_solver(mesh.clone(), config);
    
    let mut state = create_state(4);
    state.h = vec![1.0, 0.01, 1.0, 0.01];
    state.z = vec![0.0; 4];
    state.hu = vec![0.0; 4];
    state.hv = vec![0.0; 4];
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.0001, 120)
        .expect("极端水深比模拟失败");
    
    println!("极端水深比 (100:1): 绝对误差={:.2e} 相对误差={:.2e}", result.mass_error, result.relative_error);
    assert!(result.relative_error < 1e-8,
            "极端水深比守恒失败！误差 {:.2e}", result.relative_error);
}

/// 薄膜稳定性测试（200微米）
#[test]
fn test_thin_film_stability() {
    let mesh = Arc::new(create_simple_mesh());
    let config = Layer3Config::builder()
        .scheme(NumericalScheme::FirstOrder)
        .build();
    let mut solver = create_solver(mesh.clone(), config);
    
    let thin_h = 2e-4;
    let mut state = create_state(4);
    state.h = vec![thin_h; 4];
    state.z = vec![0.0; 4];
    state.hu = vec![0.0; 4];
    state.hv = vec![0.0; 4];
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.001, 100)
        .expect("薄膜模拟失败");
    
    println!("极薄水层: 绝对误差={:.2e} 相对误差={:.2e}", result.mass_error, result.relative_error);
    assert!(result.relative_error < 1e-8,
            "薄膜守恒失败！误差 {:.2e}", result.relative_error);
}

/// 高速流冲击干区测试
#[test]
fn test_high_velocity_wet_dry_interface() {
    let mesh = Arc::new(create_simple_mesh());
    let config = Layer3Config::builder()
        .scheme(NumericalScheme::FirstOrder)
        .build();
    let mut solver = create_solver(mesh.clone(), config);
    
    let mut state = create_state(4);
    state.h = vec![1.0, 0.0, 1.0, 0.0];
    state.z = vec![0.0; 4];
    state.hu = vec![1.0, 0.0, 1.0, 0.0];
    state.hv = vec![0.0; 4];
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.0001, 120)
        .expect("高速干湿界面模拟失败");
    
    println!("高速干湿界面: 绝对误差={:.2e} 相对误差={:.2e}", result.mass_error, result.relative_error);
    assert!(result.relative_error < 1e-8,
            "高速干湿界面守恒失败！误差 {:.2e}", result.relative_error);
}

/// 深水测试（100米）
#[test]
fn test_very_deep_water() {
    let mesh = Arc::new(create_simple_mesh());
    let config = Layer3Config::default();
    let mut solver = create_solver(mesh.clone(), config);
    
    let mut state = create_state(4);
    state.h = vec![100.0; 4];
    state.z = vec![0.0; 4];
    state.hu = vec![0.0; 4];
    state.hv = vec![0.0; 4];
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.0001, 120)
        .expect("深水模拟失败");
    
    println!("深水: 绝对误差={:.2e} 相对误差={:.2e}", result.mass_error, result.relative_error);
    assert!(result.relative_error < 1e-10,
            "深水守恒失败！误差 {:.2e}", result.relative_error);
}

/// 长时间积分测试（10000步）
#[test]
#[ignore = "slow"]
fn test_long_time_integration() {
    let mesh = Arc::new(create_simple_mesh());
    let config = Layer3Config::default();
    let mut solver = create_solver(mesh.clone(), config);
    
    let mut state = create_state(4);
    state.h = vec![1.0; 4];
    state.z = vec![0.0; 4];
    state.hu = vec![0.1; 4];
    state.hv = vec![0.0; 4];
    
    let initial_mass = compute_total_mass(&state, &mesh);
    let dt = 0.001;
    let n_steps = 2000;
    
    for step in 0..n_steps {
        solver.step(&mut state, dt);
        if step % 500 == 0 {
            validate_state(&state, step).expect("状态无效");
        }
    }
    
    let final_mass = compute_total_mass(&state, &mesh);
    let relative_error = (final_mass - initial_mass).abs() / initial_mass;
    
    println!("长时间积分 ({} 步): 绝对误差={:.2e} 相对误差={:.2e}", n_steps, (final_mass - initial_mass).abs(), relative_error);
    assert!(relative_error < 1e-7,
            "长时间积分守恒失败！误差 {:.2e}", relative_error);
}

/// 网格规模测试（100单元）
#[test]
#[ignore = "slow"]
fn test_large_mesh_conservation() {
    let mesh = Arc::new(create_rectangular_mesh(10, 10, 0.5, 0.5, |_, _| 0.0));
    let config = Layer3Config::default();
    let mut solver = create_solver(mesh.clone(), config);
    
    let n_cells = 100;
    let mut state = create_state(n_cells);
    
    for i in 0..n_cells {
        state.h[i] = 0.5 + 0.3 * ((i as f64 * 0.1).sin()).abs();
        state.z[i] = 0.0;
        state.hu[i] = 0.0;
        state.hv[i] = 0.0;
    }
    
    let result = run_simulation(&mut solver, &mut state, &mesh, 0.0005, 120)
        .expect("大网格模拟失败");
    
    println!("大网格 (100单元): 绝对误差={:.2e} 相对误差={:.2e}", result.mass_error, result.relative_error);
    assert!(result.relative_error < 1e-8,
            "大网格守恒失败！误差 {:.2e}", result.relative_error);
}