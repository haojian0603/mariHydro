// src-tauri/src/marihydro/infra/context.rs

use anyhow::Context;

use crate::marihydro::geo::crs::{CrsStrategy, ResolvedCrs};
use crate::marihydro::infra::config::ProjectConfig;
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::time::TimeManager;

pub struct SimContext {
    pub timer: TimeManager,
    pub crs: ResolvedCrs,
}

impl SimContext {
    pub fn from_config(
        cfg: &ProjectConfig,
        detected_file_crs: Option<&str>,
    ) -> anyhow::Result<Self> {
        let timer = TimeManager::new(&cfg.start_time_iso, cfg.timezone.clone())
            .map_err(|e| anyhow::anyhow!(e))
            .context("初始化时间管理器失败")?;

        let crs_def = match &cfg.crs_strategy {
            CrsStrategy::Manual(def) => def.clone(),
            CrsStrategy::FromFirstFile => match detected_file_crs {
                Some(def) => def.to_string(),
                None => {
                    return Err(MhError::Config(
                        "CRS策略为'FromFirstFile'，但未提供参考文件的坐标定义".into(),
                    )
                    .into())
                }
            },
            CrsStrategy::ForceWGS84 => "EPSG:4326".to_string(),
        };

        let crs = ResolvedCrs::new(&crs_def)?;

        if !crs.is_metric() {
            log::warn!("检测到非米制坐标系 ({}). 模拟结果可能不正确。", crs.wkt);
        }

        Ok(Self { timer, crs })
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.timer.elapsed_seconds()
    }

    pub fn is_metric_crs(&self) -> bool {
        self.crs.is_metric()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_requires_crs_for_from_first_file() {
        let config = ProjectConfig {
            crs_strategy: CrsStrategy::FromFirstFile,
            ..Default::default()
        };
        let result = SimContext::from_config(&config, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_context_with_manual_crs() {
        let config = ProjectConfig {
            crs_strategy: CrsStrategy::Manual("EPSG:32651".to_string()),
            ..Default::default()
        };
        let result = SimContext::from_config(&config, None);
        assert!(result.is_ok());
    }
}
