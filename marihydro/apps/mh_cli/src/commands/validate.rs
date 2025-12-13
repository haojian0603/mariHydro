// marihydro\apps\mh_cli\src\commands\validate.rs

//! 配置验证命令
//!
//! 验证配置文件和网格文件的正确性。

use anyhow::{bail, Context, Result};
use clap::Args;
use std::path::PathBuf;
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

    // 验证配置文件
    if let Some(config_path) = &args.config {
        validate_config(config_path, &mut result)?;
    }

    // 验证网格文件
    if let Some(mesh_path) = &args.mesh {
        validate_mesh(mesh_path, &mut result)?;
    }

    // 如果没有指定任何文件
    if args.config.is_none() && args.mesh.is_none() {
        println!("用法: mh_cli validate --config <配置文件> [--mesh <网格文件>]");
        println!("      mh_cli validate --mesh <网格文件>");
        return Ok(());
    }

    // 输出结果
    print_validation_result(&result, args.strict)
}

fn validate_config(path: &PathBuf, result: &mut ValidationResult) -> Result<()> {
    println!("\n检查配置文件: {}", path.display());

    // 检查文件是否存在
    if !path.exists() {
        result.add_error(format!("配置文件不存在: {}", path.display()));
        return Ok(());
    }

    // 读取文件
    let content = std::fs::read_to_string(path)
        .context("无法读取配置文件")?;

    // 尝试解析 JSON
    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            result.add_error(format!("JSON 解析错误: {}", e));
            return Ok(());
        }
    };

    // 验证必需字段
    validate_config_fields(&json, result);

    println!("  ✓ 配置文件格式有效");

    Ok(())
}

fn validate_config_fields(json: &serde_json::Value, result: &mut ValidationResult) {
    // 检查精度设置
    if let Some(precision) = json.get("precision") {
        match precision.as_str() {
            Some("f32") | Some("f64") | Some("F32") | Some("F64") => {}
            Some(other) => result.add_error(format!("无效的精度值: {}", other)),
            None => result.add_error("precision 字段应为字符串"),
        }
    }

    // 检查 CFL 数
    if let Some(cfl) = json.get("cfl") {
        if let Some(v) = cfl.as_f64() {
            if v <= 0.0 {
                result.add_error("CFL 数必须为正数");
            } else if v > 1.0 {
                result.add_warning("CFL 数大于 1.0 可能导致不稳定");
            }
        }
    }

    // 检查重力加速度
    if let Some(gravity) = json.get("gravity") {
        if let Some(v) = gravity.as_f64() {
            if v <= 0.0 {
                result.add_error("重力加速度必须为正数");
            } else if (v - 9.81).abs() > 1.0 {
                result.add_warning(format!("重力加速度 {} 偏离地球标准值较大", v));
            }
        }
    }

    // 检查干单元阈值
    if let Some(h_dry) = json.get("h_dry") {
        if let Some(v) = h_dry.as_f64() {
            if v < 0.0 {
                result.add_error("h_dry 不能为负数");
            } else if v > 0.1 {
                result.add_warning(format!("h_dry = {} 较大，可能影响精度", v));
            }
        }
    }

    // 检查最小水深
    if let Some(h_min) = json.get("h_min") {
        if let Some(v) = h_min.as_f64() {
            if v < 0.0 {
                result.add_error("h_min 不能为负数");
            }
        }
    }

    // 检查网格路径
    if let Some(mesh_path) = json.get("mesh_path") {
        if let Some(p) = mesh_path.as_str() {
            if !std::path::Path::new(p).exists() {
                result.add_warning(format!("网格文件不存在: {}", p));
            }
        }
    }
}

fn validate_mesh(path: &PathBuf, result: &mut ValidationResult) -> Result<()> {
    println!("\n检查网格文件: {}", path.display());

    // 检查文件是否存在
    if !path.exists() {
        result.add_error(format!("网格文件不存在: {}", path.display()));
        return Ok(());
    }

    // 检查文件扩展名
    let extension = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match extension.to_lowercase().as_str() {
        "msh" => validate_gmsh_mesh(path, result)?,
        "geojson" => validate_geojson_mesh(path, result)?,
        "qmd" => validate_qmd_mesh(path, result)?,
        _ => {
            result.add_warning(format!("未知的网格文件格式: .{}", extension));
        }
    }

    Ok(())
}

fn validate_gmsh_mesh(path: &PathBuf, result: &mut ValidationResult) -> Result<()> {
    // 读取文件头
    let content = std::fs::read_to_string(path)
        .context("无法读取网格文件")?;

    let lines: Vec<&str> = content.lines().take(10).collect();

    // 检查 Gmsh 格式标记
    if lines.is_empty() || !lines[0].starts_with("$MeshFormat") {
        result.add_error("无效的 Gmsh 格式：缺少 $MeshFormat 标记");
        return Ok(());
    }

    // 检查版本
    if lines.len() > 1 {
        let version_parts: Vec<&str> = lines[1].split_whitespace().collect();
        if !version_parts.is_empty() {
            let version: f64 = version_parts[0].parse().unwrap_or(0.0);
            if version < 2.0 {
                result.add_warning(format!("Gmsh 版本 {} 较旧，建议使用 2.0 或更高版本", version));
            }
        }
    }

    println!("  ✓ Gmsh 格式有效");

    Ok(())
}

fn validate_geojson_mesh(path: &PathBuf, result: &mut ValidationResult) -> Result<()> {
    let content = std::fs::read_to_string(path)
        .context("无法读取 GeoJSON 文件")?;

    // 尝试解析 JSON
    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            result.add_error(format!("GeoJSON 解析错误: {}", e));
            return Ok(());
        }
    };

    // 检查类型
    if let Some(type_field) = json.get("type") {
        match type_field.as_str() {
            Some("FeatureCollection") | Some("Feature") | Some("Polygon") | Some("MultiPolygon") => {}
            Some(other) => result.add_warning(format!("非标准 GeoJSON 类型: {}", other)),
            None => result.add_error("type 字段应为字符串"),
        }
    } else {
        result.add_error("GeoJSON 缺少 type 字段");
    }

    println!("  ✓ GeoJSON 格式有效");

    Ok(())
}

fn validate_qmd_mesh(path: &PathBuf, result: &mut ValidationResult) -> Result<()> {
    // QMD 是内部四叉树网格格式
    let content = std::fs::read_to_string(path)
        .context("无法读取 QMD 文件")?;

    // 简单检查文件头
    if !content.starts_with("QMD") && !content.starts_with("{") {
        result.add_warning("QMD 文件格式可能不正确");
    }

    println!("  ✓ QMD 格式检查完成");

    Ok(())
}

fn print_validation_result(result: &ValidationResult, strict: bool) -> Result<()> {
    println!("\n=== 验证结果 ===");

    // 输出错误
    if !result.errors.is_empty() {
        println!("\n错误 ({}):", result.errors.len());
        for err in &result.errors {
            error!("  ✗ {}", err);
            println!("  ✗ {}", err);
        }
    }

    // 输出警告
    if !result.warnings.is_empty() {
        println!("\n警告 ({}):", result.warnings.len());
        for warning in &result.warnings {
            warn!("  ⚠ {}", warning);
            println!("  ⚠ {}", warning);
        }
    }

    // 最终判定
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
        bail!("验证失败：发现 {} 个错误，{} 个警告",
              result.errors.len(), result.warnings.len())
    }
}
