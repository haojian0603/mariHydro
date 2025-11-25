// src-tauri/src/marihydro/infra/config.rs

use crate::marihydro::geo::crs::CrsStrategy;
use crate::marihydro::infra::time::TimezoneConfig;
use serde::{Deserialize, Serialize};

/// 项目配置 (持久化层)
/// 对应前端的 "Project Settings" 页面
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    // --- 元数据 ---
    pub project_name: String,
    pub version: String, // 配置文件版本

    // --- 核心策略 (Core Strategies) ---
    /// 坐标系策略：手动 / 自动
    pub crs_strategy: CrsStrategy,
    /// 时间显示策略：本地 / UTC / 指定
    pub timezone: TimezoneConfig,

    // --- 空间范围 (Domain Scope) ---
    /// 期望的网格尺寸
    pub nx: usize,
    pub ny: usize,
    /// 期望的网格分辨率 (米)
    pub dx: f64,
    pub dy: f64,

    // --- 时间控制 (Temporal Control) ---
    /// ISO8601 格式起始时间
    pub start_time_iso: String,
    /// 模拟总秒数
    pub duration_seconds: f64,
    /// 结果输出间隔 (秒)
    pub output_interval: f64,

    // --- 物理默认值 (Physics Defaults) ---
    /// 默认曼宁系数 (当没有糙率文件时使用)
    pub default_roughness: f64,
    /// 最小水深 (干湿边界)
    pub min_depth: f64,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            project_name: "Untitled_Project".into(),
            version: "1.0".into(),
            // 默认不做假设，要求用户明确或自动推导
            crs_strategy: CrsStrategy::FromFirstFile,
            // 默认跟随用户系统时区
            timezone: TimezoneConfig::Local,

            nx: 200,
            ny: 200,
            dx: 50.0,
            dy: 50.0,

            start_time_iso: "2024-01-01T00:00:00Z".into(),
            duration_seconds: 86400.0,
            output_interval: 3600.0,

            default_roughness: 0.025,
            min_depth: 0.05,
        }
    }
}
