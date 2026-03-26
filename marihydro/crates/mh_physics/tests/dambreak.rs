// crates/mh_physics/tests/dambreak.rs
//! 溃堤测试
//!
//! 使用结构化网格测试溃堤场景
use std::sync::Arc;
use std::sync::LazyLock;

use mh_mesh::structured::{StructuredMesh, StructuredMeshConfig};
use mh_physics::adapter::PhysicsMesh;
use mh_physics::engine::ShallowWaterSolver;
use mh_physics::sources::NoSource;
use mh_physics::state::ShallowWaterState;
use mh_physics::types::NumericalParams;
use mh_physics::Layer3Config;
use mh_runtime::prelude::*;

/// 全局Backend实例
static BACKEND: LazyLock<CpuBackend<f64>> = LazyLock::new(|| CpuBackend::<f64>::new());

#[cfg(test)]
mod test_harness {
    use super::*;

    pub fn get_backend() -> CpuBackend<f64> {
        *BACKEND
    }

    pub fn create_state(n_cells: usize) -> ShallowWaterState<CpuBackend<f64>> {
        ShallowWaterState::new_with_backend(get_backend(), n_cells)
    }

    pub fn create_solver(
        mesh: Arc<PhysicsMesh>,
        config: Layer3Config<f64>,
    ) -> ShallowWaterSolver<CpuBackend<f64>, NoSource<CpuBackend<f64>>> {
        ShallowWaterSolver::<CpuBackend<f64>, NoSource<CpuBackend<f64>>>::new(
            mesh,
            config,
            get_backend(),
        )
    }
}

use test_harness::{create_solver, create_state};

/// 构建结构化网格并转换为 PhysicsMesh
fn build_structured_mesh(config: StructuredMeshConfig) -> Result<PhysicsMesh, String> {
    if config.nx == 0 || config.ny == 0 || config.dx <= 0.0 || config.dy <= 0.0 {
        return Err("结构化网格参数无效".to_string());
    }
    let mut mesh = StructuredMesh::new(config);
    mesh.set_uniform_bed_elevation(0.0)
        .map_err(|e| format!("结构化网格设置平床失败: {}", e))?;
    let frozen = mesh
        .freeze()
        .map_err(|e| format!("结构化网格冻结失败: {}", e))?;
    Ok(PhysicsMesh::from_frozen(&frozen))
}

/// 溃堤初始条件
fn setup_dambreak_initial_condition(
    mesh: &PhysicsMesh,
    h_left: f64,
    h_right: f64,
    dam_x: f64,
) -> ShallowWaterState<CpuBackend<f64>> {
    let n_cells = mesh.cell_count();
    let mut state = create_state(n_cells);

    for i in 0..n_cells {
        state.z[i] = 0.0; // 平底
    }

    for i in 0..n_cells {
        let center = mesh
            .cell_center_generic::<CpuBackend<f64>>(CellIndex::new(i))
            .expect("cell_center out of range");
        let cx = center.x();
        state.h[i] = if cx < dam_x { h_left } else { h_right };
        state.hu[i] = 0.0; // 初始静止
        state.hv[i] = 0.0;
    }

    state
}

/// 计算总质量
fn compute_total_mass(state: &ShallowWaterState<CpuBackend<f64>>, mesh: &PhysicsMesh) -> f64 {
    let mut total = 0.0;
    for i in 0..state.n_cells() {
        if let Some(area) = mesh.cell_area(CellIndex::new(i)) {
            total += state.h[i] * area;
        }
    }
    total
}

/// 计算最大水深
fn compute_max_depth(state: &ShallowWaterState<CpuBackend<f64>>) -> f64 {
    state.h.iter().cloned().fold(0.0, f64::max)
}

/// 计算最大速度
fn compute_max_velocity(state: &ShallowWaterState<CpuBackend<f64>>) -> f64 {
    let h_min = 1e-6;
    let mut max_vel: f64 = 0.0;
    for i in 0..state.n_cells() {
        if state.h[i] > h_min {
            let u = state.hu[i] / state.h[i];
            let v = state.hv[i] / state.h[i];
            let vel = (u * u + v * v).sqrt();
            max_vel = max_vel.max(vel);
        }
    }
    max_vel
}

/// 验证状态有效性
fn validate_state(state: &ShallowWaterState<CpuBackend<f64>>) -> Result<(), String> {
    for (i, &h) in state.h.iter().enumerate() {
        if h.is_nan() {
            return Err(format!("单元 {} 水深为 NaN", i));
        }
        if h < -1e-10 {
            return Err(format!("单元 {} 水深为负: {:.6e}", i, h));
        }
        if !h.is_finite() {
            return Err(format!("单元 {} 水深不有限: {}", i, h));
        }
    }
    for (i, &hu) in state.hu.iter().enumerate() {
        if !hu.is_finite() {
            return Err(format!("单元 {} x动量不有限: {}", i, hu));
        }
    }
    for (i, &hv) in state.hv.iter().enumerate() {
        if !hv.is_finite() {
            return Err(format!("单元 {} y动量不有限: {}", i, hv));
        }
    }
    Ok(())
}

/// 运行溃堤模拟
fn run_dambreak_simulation(
    mesh_label: &str,
    mesh: PhysicsMesh,
    h_left: f64,
    h_right: f64,
    dam_x: f64,
    end_time: f64,
    max_steps: usize,
) -> Result<(), String> {
    println!("\n========================================");
    println!("溃堤测试: {}", mesh_label);
    println!("========================================");

    println!("网格加载完成:");
    println!("  - 单元数: {}", mesh.cell_count());
    println!("  - 面数: {}", mesh.face_count());
    println!("  - 节点数: {}", mesh.node_count());

    let mut state = setup_dambreak_initial_condition(&mesh, h_left, h_right, dam_x);
    let initial_mass = compute_total_mass(&state, &mesh);

    println!("\n初始条件:");
    println!("  - 左侧水深: {:.2} m", h_left);
    println!("  - 右侧水深: {:.2} m", h_right);
    println!("  - 坝位置: x = {:.1} m", dam_x);
    println!("  - 初始总质量: {:.6e} m³", initial_mass);

    // 创建求解器
    let params = NumericalParams {
        cfl: 0.5,
        ..Default::default()
    };

    let config = Layer3Config::builder()
        .gravity(9.81)
        .params(params)
        .use_hydrostatic_reconstruction(true)
        .build();

    let mut solver = create_solver(Arc::new(mesh.clone()), config);

    let mut time = 0.0;
    let mut step = 0;
    let output_interval = max_steps / 10;

    println!("\n开始模拟...");

    while time < end_time && step < max_steps {
        let dt = solver.compute_dt(&state);
        if dt < 1e-12 {
            return Err(format!("步骤 {} 时间步太小: {:.6e}", step, dt));
        }

        solver.step(&mut state, dt);
        time += dt;
        step += 1;

        validate_state(&state)?;

        if step % output_interval == 0 || step == 1 {
            let mass = compute_total_mass(&state, &mesh);
            let mass_error = (mass - initial_mass).abs() / initial_mass;
            let max_h = compute_max_depth(&state);
            let max_v = compute_max_velocity(&state);

            println!(
                "  步骤 {:5}: t={:.4}s, dt={:.6e}s, 质量误差={:.2e}, h_max={:.3}m, v_max={:.3}m/s",
                step, time, dt, mass_error, max_h, max_v
            );
        }
    }

    // 最终结果
    let final_mass = compute_total_mass(&state, &mesh);
    let mass_error = (final_mass - initial_mass).abs() / initial_mass;

    println!("\n模拟完成:");
    println!("  - 总步数: {}", step);
    println!("  - 模拟时间: {:.4} s", time);
    println!("  - 最终总质量: {:.6e} m³", final_mass);
    println!("  - 质量相对误差: {:.6e}", mass_error);
    println!("  - 最大水深: {:.3} m", compute_max_depth(&state));
    println!("  - 最大速度: {:.3} m/s", compute_max_velocity(&state));

    if mass_error > 1e-6 {
        println!("  [警告] 质量误差较大: {:.2e}", mass_error);
    } else {
        println!("  [通过] 质量守恒良好");
    }

    Ok(())
}

#[test]
fn test_dambreak_coarse() {
    let config = StructuredMeshConfig::rectangular(80, 20, 0.25, 0.25);
    let mesh = build_structured_mesh(config).expect("结构化网格创建失败");

    let result = run_dambreak_simulation(
        "structured_80x20",
        mesh,
        2.0,  // 左侧水深 2m
        0.5,  // 右侧水深 0.5m
        10.0, // 坝在 x=10m 处
        0.5,  // 模拟 0.5 秒
        200,  // 最多 200 步
    );

    assert!(result.is_ok(), "溃堤测试失败: {:?}", result);
}

#[test]
fn test_dambreak_medium() {
    let config = StructuredMeshConfig::rectangular(120, 30, 0.2, 0.2);
    let mesh = build_structured_mesh(config).expect("结构化网格创建失败");

    let result = run_dambreak_simulation("structured_120x30", mesh, 2.0, 0.5, 10.0, 0.3, 150);

    assert!(result.is_ok(), "中等分辨率溃堤测试失败: {:?}", result);
}

#[test]
fn test_dambreak_dry_bed() {
    let config = StructuredMeshConfig::rectangular(60, 20, 0.3, 0.3);
    let mesh = build_structured_mesh(config).expect("结构化网格创建失败");

    let result = run_dambreak_simulation("structured_dry_bed", mesh, 1.5, 1e-4, 8.0, 0.2, 120);

    assert!(result.is_ok(), "干床溃堤测试失败: {:?}", result);
}

#[test]
fn test_dambreak_slope() {
    let config = StructuredMeshConfig::rectangular(50, 20, 0.4, 0.4);
    let mesh = build_structured_mesh(config).expect("结构化网格创建失败");

    let result = run_dambreak_simulation("structured_slope", mesh, 1.8, 0.6, 6.0, 0.2, 120);

    assert!(result.is_ok(), "坡面溃堤测试失败: {:?}", result);
}

#[test]
fn test_mesh_loading() {
    let config = StructuredMeshConfig::rectangular(10, 5, 1.0, 1.0);
    let mesh = build_structured_mesh(config).expect("结构化网格创建失败");
    assert!(mesh.cell_count() > 0, "结构化网格单元数为零");
    assert!(mesh.node_count() > 0, "结构化网格节点数为零");
}
