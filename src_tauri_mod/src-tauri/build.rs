// build.rs
// 编译期检查脚本：禁止违规代码模式

use regex::Regex;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/");

    // ========== 1. 编译期禁令 ==========

    #[cfg(feature = "mock_db")]
    compile_error!("❌ MockDb已永久移除！请使用SqliteDatabase。");

    #[cfg(feature = "structured_grid")]
    compile_error!("❌ 结构化网格支持已永久移除！");

    // ========== 2. 源码违规检查 ==========

    let violations = check_source_violations();
    if !violations.is_empty() {
        for v in &violations {
            println!("cargo:warning=⚠️ {}", v);
        }

        // 在release模式下，违规直接报错
        #[cfg(not(debug_assertions))]
        {
            panic!("❌ 发现 {} 处代码违规，请修复后重新编译", violations.len());
        }
    }

    // ========== 3. 生成编译信息 ==========

    let git_hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".into());

    println!("cargo:rustc-env=GIT_HASH={}", git_hash.trim());
    println!(
        "cargo:rustc-env=BUILD_TIME={}",
        chrono::Utc::now().to_rfc3339()
    );
}

fn check_source_violations() -> Vec<String> {
    let mut violations = Vec::new();
    let src_dir = Path::new("src/marihydro");

    if !src_dir.exists() {
        return violations;
    }

    // 定义违规模式
    let patterns: Vec<(&str, &str, Severity)> = vec![
        // 严重：必须修复
        (
            r"\.unwrap\(\)",
            "使用unwrap() - 应改为?或ok_or()",
            Severity::Error,
        ),
        (
            r"\.expect\(",
            "使用expect() - 应改为?或map_err()",
            Severity::Error,
        ),
        (r"panic!\(", "使用panic!() - 应返回MhError", Severity::Error),
        // 结构化网格残留
        (r"\bnx:\s*usize", "结构化网格残留: nx字段", Severity::Error),
        (r"\bny:\s*usize", "结构化网格残留: ny字段", Severity::Error),
        (r"Array2<", "结构化网格残留: Array2类型", Severity::Error),
        (r"ndarray::", "结构化网格残留: ndarray导入", Severity::Error),
        // 警告：应该修复
        (r"TODO:", "未完成的TODO", Severity::Warning),
        (r"FIXME:", "未修复的FIXME", Severity::Warning),
        (r"HACK:", "临时HACK代码", Severity::Warning),
    ];

    scan_directory(src_dir, &patterns, &mut violations);
    violations
}

#[derive(Clone, Copy)]
enum Severity {
    Error,
    Warning,
}

fn scan_directory(dir: &Path, patterns: &[(&str, &str, Severity)], violations: &mut Vec<String>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();

            // 跳过测试文件
            if path.to_string_lossy().contains("/tests/")
                || path.to_string_lossy().contains("\\tests\\")
            {
                continue;
            }

            if path.is_dir() {
                scan_directory(&path, patterns, violations);
            } else if path.extension().map_or(false, |e| e == "rs") {
                if let Ok(content) = fs::read_to_string(&path) {
                    for (line_num, line) in content.lines().enumerate() {
                        // 跳过注释中的模式
                        let trimmed = line.trim();
                        if trimmed.starts_with("//") || trimmed.starts_with("///") {
                            continue;
                        }

                        for (pattern, msg, severity) in patterns {
                            if Regex::new(pattern).unwrap().is_match(line) {
                                let prefix = match severity {
                                    Severity::Error => "ERROR",
                                    Severity::Warning => "WARN",
                                };
                                violations.push(format!(
                                    "[{}] {}:{}: {}",
                                    prefix,
                                    path.display(),
                                    line_num + 1,
                                    msg
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
}
