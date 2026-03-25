// marihydro\apps\mh_cli\src\commands\run.rs

//! 运行模拟命令
//!
//! 执行真实的浅水方程求解链。命令内部会按精度分发到具体的
//! `ShallowWaterSolver<CpuBackend<f32/f64>, NoSource<_>>`，并输出真实快照文件。

use anyhow::{Context, Result, bail};
use clap::Args;
use mh_config::{Precision, SolverConfig as Layer4Config};
use mh_geo::{Point2D, Point3D};
use mh_mesh::{FrozenMesh, io::load_mhb};
use mh_physics::{
    Backend, CpuBackend, DeviceBuffer, Layer3Config, NoSource, PhysicsMesh, RuntimeScalar,
    ShallowWaterSolver, ShallowWaterState,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

/// 运行模拟参数
#[derive(Args)]
pub struct RunArgs {
    /// 配置文件路径
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// 网格文件路径（当前仅支持 .mhb）
    #[arg(short, long)]
    pub mesh: Option<PathBuf>,

    /// 输出目录
    #[arg(short, long, default_value = "output")]
    pub output: PathBuf,

    /// 模拟结束时间 [秒]
    #[arg(short = 't', long, default_value = "100.0")]
    pub end_time: f64,

    /// 时间步长上限 [秒]
    #[arg(long, default_value = "0.01")]
    pub dt: f64,

    /// 输出间隔 [秒]
    #[arg(long, default_value = "1.0")]
    pub output_interval: f64,

    /// 使用 f32 精度
    #[arg(long)]
    pub f32: bool,

    /// 内置条带网格单元数量（未提供真实网格时使用）
    #[arg(long, default_value = "1000")]
    pub cells: usize,

    /// 初始水深 [m]
    #[arg(long, default_value = "1.0")]
    pub initial_depth: f64,

    /// 溃坝模拟模式
    #[arg(long)]
    pub dambreak: bool,
}

#[derive(Debug, Clone, Copy)]
struct StateSummary {
    h_max: f64,
    h_min_wet: f64,
    v_max: f64,
}

/// 执行运行命令
pub fn execute(args: RunArgs) -> Result<()> {
    info!("=== MariHydro 模拟启动 ===");

    let config = load_run_config(&args)?;
    validate_supported_run_configuration(&config)?;
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("创建输出目录失败: {}", args.output.display()))?;

    match config.precision {
        Precision::F32 => execute_with_precision::<f32>(&args, &config),
        Precision::F64 => execute_with_precision::<f64>(&args, &config),
    }
}

fn load_run_config(args: &RunArgs) -> Result<Layer4Config> {
    let mut config = if let Some(path) = &args.config {
        Layer4Config::from_file(path)
            .with_context(|| format!("读取配置文件失败: {}", path.display()))?
    } else {
        Layer4Config::default()
    };

    if args.f32 {
        config.precision = Precision::F32;
    }

    if let Some(mesh_path) = &args.mesh {
        config.mesh.file = mesh_path.clone();
    }

    config.max_time = args.end_time;
    config.output.directory = args.output.clone();
    config.output.interval = args.output_interval;
    config.time.initial_dt = args.dt;
    config.time.max_dt = config.time.max_dt.min(args.dt);

    if config.time.max_dt < config.time.min_dt {
        config.time.min_dt = config.time.max_dt;
    }

    config.validate().context("运行配置未通过校验")?;
    Ok(config)
}

fn validate_supported_run_configuration(config: &Layer4Config) -> Result<()> {
    let mut unsupported = Vec::new();

    if config.numerical.friction {
        unsupported.push("底摩擦源项");
    }
    if config.numerical.coriolis {
        unsupported.push("科里奥利源项");
    }
    if config.numerical.wind_forcing {
        unsupported.push("风应力源项");
    }

    if unsupported.is_empty() {
        return Ok(());
    }

    bail!(
        "mh_cli run 当前仅接入无源项基线工况，以下能力尚未接入真实运行链：{}。请改用已接入对应源项的 workflow 路径。",
        unsupported.join("、")
    )
}

fn execute_with_precision<S>(args: &RunArgs, config: &Layer4Config) -> Result<()>
where
    S: RuntimeScalar,
    CpuBackend<S>: Backend<Scalar = S>,
{
    let backend = CpuBackend::<S>::new();
    let mesh = create_run_mesh(args, config)?;
    let layer3_config: Layer3Config<S> = Layer3Config::from_layer4(config)
        .map_err(|e| anyhow::anyhow!("Layer3 配置转换失败: {}", e))?;
    let mut solver = ShallowWaterSolver::<CpuBackend<S>, NoSource<CpuBackend<S>>>::new(
        mesh.clone(),
        layer3_config,
        backend.clone(),
    );
    let mut state = create_initial_state(backend, mesh.cell_count(), args.initial_depth, args.dambreak)?;

    info!(
        "使用精度: {:?}, 网格: {} 单元, {} 面",
        config.precision,
        mesh.cell_count(),
        mesh.face_count()
    );
    info!(
        "开始模拟: 结束时间={} s, 时间步长上限={} s, 输出间隔={} s",
        args.end_time,
        args.dt,
        args.output_interval
    );

    let start = Instant::now();
    let mut sim_time = 0.0_f64;
    let mut last_output_time = -args.output_interval;
    let mut output_count = 0usize;
    let mut step_count = 0usize;
    let min_cell_size = mesh.inner().min_cell_size;

    write_snapshot(&args.output, output_count, sim_time, &state, config.physics.h_min)?;
    output_count += 1;

    while sim_time < args.end_time {
        let suggested_dt = solver.compute_dt(&state).to_f64_lossy();
        if !suggested_dt.is_finite() || suggested_dt <= 0.0 {
            bail!("求解器给出了无效时间步长: {}", suggested_dt);
        }

        let remaining = args.end_time - sim_time;
        let dt_step = suggested_dt.min(args.dt).min(remaining);
        let dt_scalar = S::from_config_or_panic(dt_step, "mh_cli.run.dt_step");
        let dt_used = solver.step(&mut state, dt_scalar).to_f64_lossy();
        if !dt_used.is_finite() || dt_used <= 0.0 {
            bail!("时间推进返回了无效时间步长: {}", dt_used);
        }

        sim_time += dt_used;
        step_count += 1;

        if sim_time - last_output_time >= args.output_interval || sim_time >= args.end_time {
            let summary = summarize_state(&state, config.physics.h_min);
            let max_wave_speed = solver.max_wave_speed();
            let cfl = if min_cell_size > 0.0 {
                dt_used * max_wave_speed / min_cell_size
            } else {
                0.0
            };
            let solver_stats = solver.stats().to_f64();

            write_snapshot(&args.output, output_count, sim_time, &state, config.physics.h_min)?;
            output_count += 1;
            last_output_time = sim_time;

            info!(
                "t={:.3} s: h_max={:.4} m, h_min={:.4} m, v_max={:.4} m/s, CFL={:.3}, 波速={:.4} m/s, 干单元={}, 限制面={}",
                sim_time,
                summary.h_max,
                summary.h_min_wet,
                summary.v_max,
                cfl,
                max_wave_speed,
                solver_stats.dry_cells,
                solver_stats.limited_faces,
            );
        }
    }

    let elapsed = start.elapsed();
    let final_stats = solver.stats().to_f64();

    info!("=== 模拟完成 ===");
    info!("总步数: {}", step_count);
    info!("计算时间: {:.2} s", elapsed.as_secs_f64());
    info!(
        "平均步耗时: {:.3} ms",
        if step_count > 0 {
            elapsed.as_secs_f64() * 1000.0 / step_count as f64
        } else {
            0.0
        }
    );
    info!("求解器统计: {}", final_stats.summary());
    info!("输出文件数: {}", output_count);

    Ok(())
}

fn create_run_mesh(args: &RunArgs, config: &Layer4Config) -> Result<Arc<PhysicsMesh>> {
    if let Some(mesh_path) = resolve_mesh_path(args, config)? {
        return load_real_mesh(&mesh_path);
    }

    build_strip_mesh(args.cells)
}

fn resolve_mesh_path(args: &RunArgs, config: &Layer4Config) -> Result<Option<PathBuf>> {
    if let Some(path) = &args.mesh {
        return Ok(Some(path.clone()));
    }

    if let Some(config_path) = &args.config {
        if config.mesh.file.as_os_str().is_empty() {
            return Ok(None);
        }

        let raw = &config.mesh.file;
        let default_placeholder = Path::new("mesh.msh");
        if raw == default_placeholder && !raw.exists() {
            return Ok(None);
        }

        let candidate = if raw.is_absolute() {
            raw.clone()
        } else {
            config_path
                .parent()
                .map(|dir| dir.join(raw))
                .unwrap_or_else(|| raw.clone())
        };

        if candidate.exists() {
            return Ok(Some(candidate));
        }

        if raw == default_placeholder {
            return Ok(None);
        }

        bail!("配置引用的网格文件不存在: {}", candidate.display());
    }

    Ok(None)
}

fn load_real_mesh(path: &Path) -> Result<Arc<PhysicsMesh>> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_lowercase();

    if extension != "mhb" {
        bail!(
            "mh_cli run 当前仅接入 .mhb 真实网格运行路径；收到 {}",
            path.display()
        );
    }

    let frozen = load_mhb(path)
        .with_context(|| format!("读取 MHB 网格失败: {}", path.display()))?;
    frozen
        .validate()
        .map_err(|e| anyhow::anyhow!("MHB 网格校验失败: {}", e))?;
    Ok(Arc::new(PhysicsMesh::from_frozen(&frozen)))
}

fn build_strip_mesh(n_cells: usize) -> Result<Arc<PhysicsMesh>> {
    if n_cells == 0 {
        bail!("cells 必须大于 0");
    }

    let backend = CpuBackend::<f64>::new();
    let mut mesh = FrozenMesh::empty_with_cells_backend(backend.clone(), n_cells);
    let top_offset = n_cells + 1;
    let interior_faces = n_cells.saturating_sub(1);
    let bottom_start = interior_faces;
    let left_face = bottom_start + n_cells;
    let top_start = left_face + 1;
    let right_face = top_start + n_cells;
    let total_faces = right_face + 1;

    mesh.n_nodes = 2 * (n_cells + 1);
    mesh.node_coords = (0..=n_cells)
        .map(|i| Point3D::new(i as f64, 0.0, 0.0))
        .chain((0..=n_cells).map(|i| Point3D::new(i as f64, 1.0, 0.0)))
        .collect();
    mesh.n_cells = n_cells;
    mesh.cell_center = (0..n_cells)
        .map(|i| Point2D::new(i as f64 + 0.5, 0.5))
        .collect();
    mesh.cell_area.copy_from_slice(&vec![1.0; n_cells]);
    mesh.cell_z_bed.copy_from_slice(&vec![0.0; n_cells]);

    mesh.cell_node_offsets = Vec::with_capacity(n_cells + 1);
    mesh.cell_face_offsets = Vec::with_capacity(n_cells + 1);
    mesh.cell_neighbor_offsets = Vec::with_capacity(n_cells + 1);
    mesh.cell_node_offsets.push(0);
    mesh.cell_face_offsets.push(0);
    mesh.cell_neighbor_offsets.push(0);
    mesh.cell_node_indices.clear();
    mesh.cell_face_indices.clear();
    mesh.cell_neighbor_indices.clear();

    for i in 0..n_cells {
        mesh.cell_node_indices.extend_from_slice(&[
            i as u32,
            (i + 1) as u32,
            (top_offset + i + 1) as u32,
            (top_offset + i) as u32,
        ]);
        mesh.cell_node_offsets.push(mesh.cell_node_indices.len());

        let left = if i == 0 { left_face } else { i - 1 };
        let right = if i + 1 == n_cells { right_face } else { i };
        let bottom = bottom_start + i;
        let top = top_start + i;
        mesh.cell_face_indices
            .extend_from_slice(&[left as u32, bottom as u32, top as u32, right as u32]);
        mesh.cell_face_offsets.push(mesh.cell_face_indices.len());

        if i > 0 {
            mesh.cell_neighbor_indices.push((i - 1) as u32);
        }
        if i + 1 < n_cells {
            mesh.cell_neighbor_indices.push((i + 1) as u32);
        }
        mesh.cell_neighbor_offsets.push(mesh.cell_neighbor_indices.len());
    }

    mesh.n_faces = total_faces;
    mesh.n_interior_faces = interior_faces;
    mesh.face_center = Vec::with_capacity(total_faces);
    mesh.face_normal = Vec::with_capacity(total_faces);
    mesh.face_owner = Vec::with_capacity(total_faces);
    mesh.face_neighbor = Vec::with_capacity(total_faces);
    mesh.face_delta_owner = Vec::with_capacity(total_faces);
    mesh.face_delta_neighbor = Vec::with_capacity(total_faces);
    mesh.face_boundary_id = Vec::with_capacity(total_faces);

    for i in 0..interior_faces {
        mesh.face_center.push(Point2D::new(i as f64 + 1.0, 0.5));
        mesh.face_normal.push(Point3D::new(1.0, 0.0, 0.0));
        mesh.face_owner.push(i as u32);
        mesh.face_neighbor.push((i + 1) as u32);
        mesh.face_delta_owner.push(Point2D::new(0.0, 0.0));
        mesh.face_delta_neighbor.push(Point2D::new(0.0, 0.0));
        mesh.face_boundary_id.push(None);
    }

    for i in 0..n_cells {
        mesh.face_center.push(Point2D::new(i as f64 + 0.5, 0.0));
        mesh.face_normal.push(Point3D::new(0.0, -1.0, 0.0));
        mesh.face_owner.push(i as u32);
        mesh.face_neighbor.push(u32::MAX);
        mesh.face_delta_owner.push(Point2D::new(0.0, 0.0));
        mesh.face_delta_neighbor.push(Point2D::new(0.0, 0.0));
        mesh.face_boundary_id.push(Some(0));
    }

    mesh.face_center.push(Point2D::new(0.0, 0.5));
    mesh.face_normal.push(Point3D::new(-1.0, 0.0, 0.0));
    mesh.face_owner.push(0);
    mesh.face_neighbor.push(u32::MAX);
    mesh.face_delta_owner.push(Point2D::new(0.0, 0.0));
    mesh.face_delta_neighbor.push(Point2D::new(0.0, 0.0));
    mesh.face_boundary_id.push(Some(0));

    for i in 0..n_cells {
        mesh.face_center.push(Point2D::new(i as f64 + 0.5, 1.0));
        mesh.face_normal.push(Point3D::new(0.0, 1.0, 0.0));
        mesh.face_owner.push(i as u32);
        mesh.face_neighbor.push(u32::MAX);
        mesh.face_delta_owner.push(Point2D::new(0.0, 0.0));
        mesh.face_delta_neighbor.push(Point2D::new(0.0, 0.0));
        mesh.face_boundary_id.push(Some(0));
    }

    mesh.face_center.push(Point2D::new(n_cells as f64, 0.5));
    mesh.face_normal.push(Point3D::new(1.0, 0.0, 0.0));
    mesh.face_owner.push((n_cells - 1) as u32);
    mesh.face_neighbor.push(u32::MAX);
    mesh.face_delta_owner.push(Point2D::new(0.0, 0.0));
    mesh.face_delta_neighbor.push(Point2D::new(0.0, 0.0));
    mesh.face_boundary_id.push(Some(0));

    mesh.face_length.resize(total_faces, 1.0);
    mesh.face_length.copy_from_slice(&vec![1.0; total_faces]);
    mesh.face_z_left.resize(total_faces, 0.0);
    mesh.face_z_left.copy_from_slice(&vec![0.0; total_faces]);
    mesh.face_z_right.resize(total_faces, 0.0);
    mesh.face_z_right.copy_from_slice(&vec![0.0; total_faces]);
    mesh.face_dist_o2n.resize(total_faces, 1.0);
    mesh.face_dist_o2n.copy_from_slice(&vec![1.0; total_faces]);
    mesh.boundary_face_indices = (bottom_start..total_faces).map(|idx| idx as u32).collect();
    mesh.boundary_names = vec!["outer".to_string()];
    mesh.min_cell_size = 1.0;
    mesh.max_cell_size = 1.0;
    mesh.cell_refinement_level = vec![0; n_cells];
    mesh.cell_parent = (0..n_cells as u32).collect();
    mesh.ghost_capacity = 0;
    mesh.cell_original_id = Vec::new();
    mesh.face_original_id = Vec::new();
    mesh.cell_permutation = Vec::new();
    mesh.cell_inv_permutation = Vec::new();

    mesh.validate()
        .map_err(|e| anyhow::anyhow!("内置条带网格构造失败: {}", e))?;

    Ok(Arc::new(PhysicsMesh::from_frozen(&mesh)))
}

fn create_initial_state<S>(
    backend: CpuBackend<S>,
    n_cells: usize,
    initial_depth: f64,
    dambreak: bool,
) -> Result<ShallowWaterState<CpuBackend<S>>>
where
    S: RuntimeScalar,
    CpuBackend<S>: Backend<Scalar = S>,
{
    let mut state = ShallowWaterState::new_with_backend(backend, n_cells);
    let wet_depth = S::from_config_or_panic(initial_depth, "mh_cli.run.initial_depth");
    let tail_depth = S::from_config_or_panic(0.1, "mh_cli.run.dambreak_tail_depth");

    {
        let h = state.h_slice_mut();
        for (idx, value) in h.iter_mut().enumerate() {
            *value = if dambreak {
                if idx < n_cells / 2 {
                    wet_depth
                } else {
                    tail_depth
                }
            } else {
                wet_depth
            };
        }
    }

    state.hu_slice_mut().fill(S::ZERO);
    state.hv_slice_mut().fill(S::ZERO);
    state.z_slice_mut().fill(S::ZERO);
    Ok(state)
}

fn summarize_state<S>(state: &ShallowWaterState<CpuBackend<S>>, h_min: f64) -> StateSummary
where
    S: RuntimeScalar,
    CpuBackend<S>: Backend<Scalar = S>,
{
    let h_min_scalar = S::from_config_or_panic(h_min, "mh_cli.run.summary.h_min");
    let mut h_max = 0.0_f64;
    let mut h_min_wet = f64::INFINITY;
    let mut v_max = 0.0_f64;

    for ((&h, &hu), &hv) in state
        .h_slice()
        .iter()
        .zip(state.hu_slice().iter())
        .zip(state.hv_slice().iter())
    {
        let h64 = h.to_f64_lossy();
        h_max = h_max.max(h64);
        if h > h_min_scalar {
            h_min_wet = h_min_wet.min(h64);
            let u = (hu / h).to_f64_lossy();
            let v = (hv / h).to_f64_lossy();
            v_max = v_max.max((u * u + v * v).sqrt());
        }
    }

    StateSummary {
        h_max,
        h_min_wet: if h_min_wet.is_finite() { h_min_wet } else { 0.0 },
        v_max,
    }
}

fn write_snapshot<S>(
    output_dir: &Path,
    index: usize,
    time: f64,
    state: &ShallowWaterState<CpuBackend<S>>,
    h_min: f64,
) -> Result<()>
where
    S: RuntimeScalar,
    CpuBackend<S>: Backend<Scalar = S>,
{
    let path = output_dir.join(format!("snapshot_{index:05}.csv"));
    let h_min_scalar = S::from_config_or_panic(h_min, "mh_cli.run.snapshot.h_min");
    let mut csv = String::from("time,cell,h,z,eta,hu,hv,u,v\n");

    for (cell, (((&h, &hu), &hv), &z)) in state
        .h_slice()
        .iter()
        .zip(state.hu_slice().iter())
        .zip(state.hv_slice().iter())
        .zip(state.z_slice().iter())
        .enumerate()
    {
        let (u, v) = if h > h_min_scalar {
            ((hu / h).to_f64_lossy(), (hv / h).to_f64_lossy())
        } else {
            (0.0, 0.0)
        };
        let h64 = h.to_f64_lossy();
        let z64 = z.to_f64_lossy();
        csv.push_str(&format!(
            "{time:.6},{cell},{h64:.10},{z64:.10},{eta:.10},{hu:.10},{hv:.10},{u:.10},{v:.10}\n",
            eta = h64 + z64,
            hu = hu.to_f64_lossy(),
            hv = hv.to_f64_lossy(),
        ));
    }

    std::fs::write(&path, csv)
        .with_context(|| format!("写出快照失败: {}", path.display()))?;
    Ok(())
}
