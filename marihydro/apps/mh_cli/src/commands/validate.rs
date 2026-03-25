// marihydro\apps\mh_cli\src\commands\validate.rs

//! 配置验证命令
//!
//! 验证真实配置结构和已接入的网格格式。

use anyhow::{Context, Result, bail};
use clap::Args;
use mh_config::SolverConfig;
use mh_mesh::io::{GmshLoader, load_mhb};
use mh_mesh::io::geojson::read_geojson_polygons;
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};

/// 验证参数
#[derive(Args)]
pub struct ValidateArgs {
    /// 配置文件路径
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// 网格文件路径
    #[arg(short, long)]
    pub mesh: Option<PathBuf>,

    /// 严格模式（警告也视为错误）
    #[arg(long)]
    pub strict: bool,
}

/// 验证结果
#[derive(Default)]
struct ValidationResult {
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl ValidationResult {
    fn add_error(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
    }

    fn add_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    fn is_ok_strict(&self) -> bool {
        self.errors.is_empty() && self.warnings.is_empty()
    }
}

/// 执行验证命令
pub fn execute(args: ValidateArgs) -> Result<()> {
    info!("=== MariHydro 配置验证 ===");

    let mut result = ValidationResult::default();

    if let Some(config_path) = &args.config {
        validate_config(config_path, &mut result)?;
    }

    if let Some(mesh_path) = &args.mesh {
        validate_mesh(mesh_path, &mut result)?;
    }

    if args.config.is_none() && args.mesh.is_none() {
        println!("用法: mh_cli validate --config <配置文件> [--mesh <网格文件>]");
        println!("      mh_cli validate --mesh <网格文件>");
        return Ok(());
    }

    print_validation_result(&result, args.strict)
}

fn validate_config(path: &Path, result: &mut ValidationResult) -> Result<()> {
    println!("\n检查配置文件: {}", path.display());

    if !path.exists() {
        result.add_error(format!("配置文件不存在: {}", path.display()));
        return Ok(());
    }

    let config = match SolverConfig::from_file(path)
        .with_context(|| format!("解析配置文件失败: {}", path.display()))
    {
        Ok(config) => config,
        Err(err) => {
            result.add_error(err.to_string());
            return Ok(());
        }
    };

    println!("  ✓ 配置结构与数值约束有效");
    println!("  - 精度: {:?}", config.precision);
    println!("  - CFL: {}", config.physics.cfl);
    println!("  - 时间积分: {:?}", config.numerical.time_integration);
    println!("  - 黎曼求解器: {:?}", config.numerical.riemann_solver);

    if let Some(mesh_path) = resolve_config_mesh_path(path, &config) {
        validate_mesh(&mesh_path, result)?;
    } else {
        result.add_warning("配置未提供可解析的网格文件路径；仅完成配置结构验证".to_string());
    }

    Ok(())
}

fn resolve_config_mesh_path(config_path: &Path, config: &SolverConfig) -> Option<PathBuf> {
    if config.mesh.file.as_os_str().is_empty() {
        return None;
    }

    let raw = &config.mesh.file;
    let default_placeholder = Path::new("mesh.msh");
    if raw == default_placeholder && !raw.exists() {
        return None;
    }

    if raw.is_absolute() {
        Some(raw.clone())
    } else {
        config_path
            .parent()
            .map(|dir| dir.join(raw))
            .or_else(|| Some(raw.clone()))
    }
}

fn validate_mesh(path: &Path, result: &mut ValidationResult) -> Result<()> {
    println!("\n检查网格文件: {}", path.display());

    if !path.exists() {
        result.add_error(format!("网格文件不存在: {}", path.display()));
        return Ok(());
    }

    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "mhb" => validate_mhb_mesh(path, result)?,
        "msh" => validate_gmsh_mesh(path, result)?,
        "geojson" => validate_geojson_mesh(path, result)?,
        "qmd" => result.add_warning("QMD 校验尚未接入真实解析器；当前不会伪装成已验证通过".to_string()),
        _ => result.add_warning(format!("未接入的网格格式: .{}", extension)),
    }

    Ok(())
}

fn validate_mhb_mesh(path: &Path, result: &mut ValidationResult) -> Result<()> {
    match load_mhb(path) {
        Ok(mesh) => {
            mesh.validate()
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            println!("  ✓ MHB 网格有效: 单元={}, 面={}", mesh.n_cells, mesh.n_faces);
        }
        Err(err) => result.add_error(format!("MHB 读取失败: {}", err)),
    }
    Ok(())
}

fn validate_gmsh_mesh(path: &Path, result: &mut ValidationResult) -> Result<()> {
    match GmshLoader::load(path) {
        Ok(mesh) => {
            if mesh.n_nodes() == 0 || mesh.n_cells() == 0 {
                result.add_error("Gmsh 网格缺少节点或单元".to_string());
            } else {
                println!(
                    "  ✓ Gmsh 网格有效: 节点={}, 单元={}, 边界边={}",
                    mesh.n_nodes(),
                    mesh.n_cells(),
                    mesh.n_boundary_edges()
                );
            }
        }
        Err(err) => result.add_error(format!("Gmsh 读取失败: {}", err)),
    }
    Ok(())
}

fn validate_geojson_mesh(path: &Path, result: &mut ValidationResult) -> Result<()> {
    match read_geojson_polygons(path) {
        Ok(polygons) => {
            if polygons.is_empty() {
                result.add_error("GeoJSON 未解析出任何多边形".to_string());
            } else {
                println!("  ✓ GeoJSON 多边形读取有效: {} 个外环", polygons.len());
            }
        }
        Err(err) => result.add_error(format!("GeoJSON 读取失败: {}", err)),
    }
    Ok(())
}

fn print_validation_result(result: &ValidationResult, strict: bool) -> Result<()> {
    println!("\n=== 验证结果 ===");

    if !result.errors.is_empty() {
        println!("\n错误 ({}):", result.errors.len());
        for err in &result.errors {
            error!("  ✗ {}", err);
            println!("  ✗ {}", err);
        }
    }

    if !result.warnings.is_empty() {
        println!("\n警告 ({}):", result.warnings.len());
        for warning in &result.warnings {
            warn!("  ⚠ {}", warning);
            println!("  ⚠ {}", warning);
        }
    }

    let success = if strict {
        result.is_ok_strict()
    } else {
        result.is_ok()
    };

    if success {
        println!("\n✓ 验证通过");
        Ok(())
    } else {
        println!("\n✗ 验证失败");
        bail!(
            "验证失败：发现 {} 个错误，{} 个警告",
            result.errors.len(),
            result.warnings.len()
        )
    }
}
