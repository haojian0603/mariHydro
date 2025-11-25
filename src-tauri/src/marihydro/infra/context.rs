// src-tauri/src/marihydro/infra/context.rs
use anyhow::Context;

use crate::marihydro::geo::crs::{CrsStrategy, ResolvedCrs};
use crate::marihydro::infra::config::ProjectConfig;
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::time::TimeManager;


/// 模拟运行时上下文
/// 包含所有已经初始化、已经验证过的对象
pub struct SimContext {
    /// 时间管理器 (已解析时区和起始时间)
    pub timer: TimeManager,

    /// 确定的坐标系 (已解析 EPSG)
    pub crs: ResolvedCrs,

    /// 物理常数缓存
    pub dx: f64,
    pub dy: f64,
    pub h_min: f64,
}

impl SimContext {
    /// 从用户配置构建上下文
    /// 注意：如果策略是 FromFirstFile，需要传入 optional 的 file_crs_def
    pub fn from_config(cfg: &ProjectConfig, detected_file_crs: Option<&str>) -> anyhow::Result<Self> {
        // 1. 初始化时间
        let timer = TimeManager::new(&cfg.start_time_iso, cfg.timezone.clone())
            .map_err(|e| anyhow::anyhow!(e))
            .context("初始化时间管理器失败")?;

        // 2. 解析坐标系 (核心逻辑)
        let crs_def = match &cfg.crs_strategy {
            CrsStrategy::Manual(def) => def.clone(),
            CrsStrategy::FromFirstFile => {
                // 必须由外部 (IO Layer) 先读取文件头，获取 WKT，再传给这里
                match detected_file_crs {
                    Some(def) => def.to_string(),
                    None => {
                        return Err(MhError::Config(
                            "CRS策略为'FromFirstFile'，但未提供参考文件的坐标定义".into(),
                        )
                        .into())
                    }
                }
            }
            CrsStrategy::ForceWGS84 => "EPSG:4326".to_string(),
        };

        let crs = ResolvedCrs::new(&crs_def)?;

        // 警告：水动力模拟通常要求投影坐标 (米)
        if !crs.is_metric() {
            // 这里仅记录日志或返回警告，暂时不阻断 (Return Err/Warn)
            log::warn!("检测到非米制坐标系 ({}). 模拟结果可能不正确。", crs.wkt);
        }

        Ok(Self {
            timer,
            crs,
            dx: cfg.dx,
            dy: cfg.dy,
            h_min: cfg.min_depth,
        })
    }
}
