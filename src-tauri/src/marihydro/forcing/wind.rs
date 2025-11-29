// src-tauri/src/marihydro/forcing/wind.rs

use chrono::{DateTime, Utc};
use log::{debug, info, trace, warn};
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::Serialize;
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::marihydro::domain::interpolator::{NoDataStrategy, SpatialInterpolator};
use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::manifest::{DataSourceConfig, ProjectManifest};
use crate::marihydro::io::drivers::nc_adapter::{
    core::NcCore,
    time::{calculate_utc_time, parse_cf_time_units},
};
use crate::marihydro::io::drivers::nc_loader::NetCdfLoader;
use crate::marihydro::io::traits::RasterLoader;

const MIN_TIME_STEPS: usize = 2;
const TIME_EPSILON: f64 = 1e-6;
const FRAME_LOAD_WARNING_MS: u128 = 500;
const LOCK_WAIT_WARNING_MS: u128 = 100;
const MAX_REASONABLE_WIND_SPEED: f64 = 100.0;
const MAX_SCALE_FACTOR: f64 = 1e6;
const MIN_SCALE_FACTOR: f64 = 1e-6;
const INVALID_VALUE_RATIO_THRESHOLD: f64 = 0.01;

#[derive(Debug, Default)]
struct LockMetrics {
    read_wait_ms: AtomicU64,
    metadata_wait_ms: AtomicU64,
}

impl LockMetrics {
    fn record_read_wait(&self, duration_ms: u64) {
        self.read_wait_ms.fetch_add(duration_ms, Ordering::Relaxed);
    }
    fn record_metadata_wait(&self, duration_ms: u64) {
        self.metadata_wait_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }
    fn total(&self) -> u64 {
        self.read_wait_ms.load(Ordering::Acquire) + self.metadata_wait_ms.load(Ordering::Acquire)
    }
    fn read_wait(&self) -> u64 {
        self.read_wait_ms.load(Ordering::Acquire)
    }
    fn metadata_wait(&self) -> u64 {
        self.metadata_wait_ms.load(Ordering::Acquire)
    }
}

#[derive(Debug, Default)]
struct ProviderStats {
    frame_loads: AtomicUsize,
    time_queries: AtomicUsize,
    total_load_time_ms: AtomicU64,
    lock_metrics: LockMetrics,
    time_rewinds: AtomicUsize,
    exhausted_calls: AtomicUsize,
    data_exhausted_warned: AtomicBool,
}

impl ProviderStats {
    fn record_load(&self, duration_ms: u64) {
        self.frame_loads.fetch_add(1, Ordering::Relaxed);
        self.total_load_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }
    fn record_query(&self) {
        self.time_queries.fetch_add(1, Ordering::Relaxed);
    }
    fn record_rewind(&self) {
        self.time_rewinds.fetch_add(1, Ordering::Relaxed);
    }
    fn should_warn_exhausted(&self) -> bool {
        !self.data_exhausted_warned.swap(true, Ordering::Relaxed)
    }
    fn to_metrics(&self) -> Metrics {
        let loads = self.frame_loads.load(Ordering::Acquire);
        let queries = self.time_queries.load(Ordering::Acquire);
        let total_load_ms = self.total_load_time_ms.load(Ordering::Acquire);
        Metrics {
            frame_loads: loads,
            time_queries: queries,
            time_rewinds: self.time_rewinds.load(Ordering::Acquire),
            exhausted_calls: self.exhausted_calls.load(Ordering::Acquire),
            avg_load_ms: if loads > 0 {
                total_load_ms as f64 / loads as f64
            } else {
                0.0
            },
            total_load_time_ms: total_load_ms,
            total_lock_wait_ms: self.lock_metrics.total(),
            read_lock_wait_ms: self.lock_metrics.read_wait(),
            metadata_lock_wait_ms: self.lock_metrics.metadata_wait(),
            avg_lock_wait_ms: if queries > 0 {
                self.lock_metrics.total() as f64 / queries as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    pub frame_loads: usize,
    pub time_queries: usize,
    pub time_rewinds: usize,
    pub exhausted_calls: usize,
    pub avg_load_ms: f64,
    pub total_load_time_ms: u64,
    pub total_lock_wait_ms: u64,
    pub read_lock_wait_ms: u64,
    pub metadata_lock_wait_ms: u64,
    pub avg_lock_wait_ms: f64,
}

struct CellFrame {
    time: DateTime<Utc>,
    u: Vec<f64>,
    v: Vec<f64>,
}

impl CellFrame {
    fn new(time: DateTime<Utc>, n_cells: usize) -> Self {
        Self {
            time,
            u: vec![0.0; n_cells],
            v: vec![0.0; n_cells],
        }
    }
    fn reset(&mut self) {
        self.u.iter_mut().for_each(|x| *x = 0.0);
        self.v.iter_mut().for_each(|x| *x = 0.0);
    }
}

pub struct WindProvider {
    config: DataSourceConfig,
    interpolator: SpatialInterpolator,
    n_cells: usize,
    nodata_strategy: NoDataStrategy,
    src_dims: (usize, usize),
    time_axis: Vec<DateTime<Utc>>,
    current_idx: usize,
    frame_curr: CellFrame,
    frame_next: CellFrame,
    nc_handle: Arc<Mutex<NcCore>>,
    stats: Arc<ProviderStats>,
    is_exhausted: bool,
}

impl WindProvider {
    pub fn init(
        source: &DataSourceConfig,
        mesh: &UnstructuredMesh,
        manifest: &ProjectManifest,
    ) -> MhResult<Self> {
        info!("[WindProvider] 初始化风场源: {}", source.name);
        if mesh.n_cells == 0 {
            return Err(MhError::InvalidMesh {
                message: "网格单元数为零".into(),
            });
        }
        Self::validate_mappings(source)?;

        let nc_handle = Arc::new(Mutex::new(NcCore::open(&source.file_path)?));
        let stats = Arc::new(ProviderStats::default());

        let time_axis = Self::load_time_axis(&nc_handle, &stats, source.time_dim_name.as_deref())?;
        Self::validate_time_axis(&time_axis, &source.file_path)?;

        info!(
            "[WindProvider] 时间范围: {} -> {} ({} 步)",
            time_axis.first().unwrap(),
            time_axis.last().unwrap(),
            time_axis.len()
        );

        let loader = NetCdfLoader;
        let src_meta = loader.read_metadata(&source.file_path)?;
        let src_dims = (src_meta.width, src_meta.height);

        let target_points: Vec<_> = mesh.cell_center.iter().map(|c| *c).collect();
        let interpolator =
            SpatialInterpolator::new_from_points(&target_points, &manifest.crs_wkt, &src_meta)?;

        info!(
            "[WindProvider] 插值器就绪 ({} 单元, 源 {}x{})",
            mesh.n_cells, src_dims.0, src_dims.1
        );

        let nodata_strategy = Self::resolve_nodata_strategy(source);
        let start_idx = Self::find_start_index(&time_axis, manifest)?;

        if start_idx + 1 >= time_axis.len() {
            return Err(MhError::DataLoad {
                file: source.file_path.clone(),
                message: format!(
                    "起始索引 {} 超出范围 (总帧数 {})",
                    start_idx,
                    time_axis.len()
                ),
            });
        }

        let mut frame_curr = CellFrame::new(time_axis[start_idx], mesh.n_cells);
        let mut frame_next = CellFrame::new(time_axis[start_idx + 1], mesh.n_cells);

        info!("[WindProvider] 预加载初始帧...");
        let start_load = Instant::now();

        let context = LoadContext {
            config: source,
            interpolator: &interpolator,
            nodata_strategy: &nodata_strategy,
            nc_handle: Arc::clone(&nc_handle),
            stats: Arc::clone(&stats),
            src_dims,
        };

        rayon::join(
            || Self::load_into_buffer(&context, start_idx, &mut frame_curr),
            || Self::load_into_buffer(&context, start_idx + 1, &mut frame_next),
        );

        info!(
            "[WindProvider] 就绪 (预加载耗时 {} ms)",
            start_load.elapsed().as_millis()
        );

        Ok(Self {
            config: source.clone(),
            interpolator,
            n_cells: mesh.n_cells,
            nodata_strategy,
            src_dims,
            time_axis,
            current_idx: start_idx,
            frame_curr,
            frame_next,
            nc_handle,
            stats,
            is_exhausted: false,
        })
    }

    pub fn get_wind_at(
        &mut self,
        time: DateTime<Utc>,
        out_u: &mut [f64],
        out_v: &mut [f64],
    ) -> MhResult<()> {
        trace!("[WindProvider] get_wind_at: {}", time);
        self.stats.record_query();

        if out_u.len() != self.n_cells || out_v.len() != self.n_cells {
            return Err(MhError::InvalidMesh {
                message: format!(
                    "输出缓冲区长度不匹配: 期望 {}, 实际 u={}, v={}",
                    self.n_cells,
                    out_u.len(),
                    out_v.len()
                ),
            });
        }

        if self.is_exhausted {
            out_u.copy_from_slice(&self.frame_next.u);
            out_v.copy_from_slice(&self.frame_next.v);
            return Ok(());
        }

        if time < self.frame_curr.time {
            info!("[WindProvider] 检测到时间回退，重新定位...");
            self.seek_to_time(time)?;
        }

        while time > self.frame_next.time {
            let next_idx = self
                .current_idx
                .checked_add(2)
                .ok_or_else(|| MhError::Config("时间索引溢出".into()))?;

            if next_idx >= self.time_axis.len() {
                if self.stats.should_warn_exhausted() {
                    warn!("[WindProvider] 数据耗尽，保持最后一帧");
                }
                self.is_exhausted = true;
                out_u.copy_from_slice(&self.frame_next.u);
                out_v.copy_from_slice(&self.frame_next.v);
                return Ok(());
            }
            self.advance_frame()?;
        }

        let dt_total = (self.frame_next.time - self.frame_curr.time).num_seconds() as f64;
        let dt_curr = (time - self.frame_curr.time).num_seconds() as f64;

        let alpha = if dt_total.abs() < TIME_EPSILON {
            0.0
        } else {
            (dt_curr / dt_total).clamp(0.0, 1.0)
        };
        let beta = 1.0 - alpha;

        out_u
            .par_iter_mut()
            .zip(out_v.par_iter_mut())
            .zip(self.frame_curr.u.par_iter())
            .zip(self.frame_next.u.par_iter())
            .zip(self.frame_curr.v.par_iter())
            .zip(self.frame_next.v.par_iter())
            .for_each(|(((((ou, ov), &u0), &u1), &v0), &v1)| {
                *ou = u0.mul_add(beta, u1 * alpha);
                *ov = v0.mul_add(beta, v1 * alpha);
            });

        Ok(())
    }

    pub fn get_metrics(&self) -> Metrics {
        self.stats.to_metrics()
    }

    pub fn time_range(&self) -> (DateTime<Utc>, DateTime<Utc>) {
        (
            *self.time_axis.first().unwrap(),
            *self.time_axis.last().unwrap(),
        )
    }

    pub fn is_data_exhausted(&self) -> bool {
        self.is_exhausted
    }

    pub fn memory_usage(&self) -> usize {
        let frames = (self.frame_curr.u.len()
            + self.frame_curr.v.len()
            + self.frame_next.u.len()
            + self.frame_next.v.len())
            * std::mem::size_of::<f64>();
        let time_axis = self.time_axis.len() * std::mem::size_of::<DateTime<Utc>>();
        frames + time_axis + std::mem::size_of::<Self>()
    }

    fn advance_frame(&mut self) -> MhResult<()> {
        self.current_idx += 1;
        mem::swap(&mut self.frame_curr, &mut self.frame_next);

        let next_idx = self.current_idx + 1;
        self.frame_next.time = self.time_axis[next_idx];

        let context = LoadContext {
            config: &self.config,
            interpolator: &self.interpolator,
            nodata_strategy: &self.nodata_strategy,
            nc_handle: Arc::clone(&self.nc_handle),
            stats: Arc::clone(&self.stats),
            src_dims: self.src_dims,
        };

        let start = Instant::now();
        Self::load_into_buffer(&context, next_idx, &mut self.frame_next)?;
        let elapsed = start.elapsed().as_millis();

        if elapsed > FRAME_LOAD_WARNING_MS {
            warn!(
                "[WindProvider] 帧加载缓慢: idx={}, 耗时={}ms",
                next_idx, elapsed
            );
        }
        Ok(())
    }

    fn seek_to_time(&mut self, time: DateTime<Utc>) -> MhResult<()> {
        self.stats.record_rewind();
        self.is_exhausted = false;

        let target_idx = Self::binary_search_time(&self.time_axis, time)?;
        let next_idx = target_idx + 1;

        if next_idx >= self.time_axis.len() {
            return Err(MhError::DataLoad {
                file: self.config.file_path.clone(),
                message: "Seek 目标越界".into(),
            });
        }

        self.current_idx = target_idx;
        self.frame_curr.time = self.time_axis[target_idx];
        self.frame_next.time = self.time_axis[next_idx];

        let context = LoadContext {
            config: &self.config,
            interpolator: &self.interpolator,
            nodata_strategy: &self.nodata_strategy,
            nc_handle: Arc::clone(&self.nc_handle),
            stats: Arc::clone(&self.stats),
            src_dims: self.src_dims,
        };

        let (r1, r2) = rayon::join(
            || Self::load_into_buffer(&context, target_idx, &mut self.frame_curr),
            || Self::load_into_buffer(&context, next_idx, &mut self.frame_next),
        );
        r1?;
        r2?;
        Ok(())
    }

    fn load_into_buffer(context: &LoadContext, idx: usize, target: &mut CellFrame) -> MhResult<()> {
        let load_start = Instant::now();

        let u_map = context
            .config
            .mappings
            .iter()
            .find(|m| m.target_var == "wind_u")
            .ok_or_else(|| MhError::Config("缺少 wind_u 映射".into()))?;
        let v_map = context
            .config
            .mappings
            .iter()
            .find(|m| m.target_var == "wind_v")
            .ok_or_else(|| MhError::Config("缺少 wind_v 映射".into()))?;

        let (raw_u, raw_v) = {
            let lock_start = Instant::now();
            let nc = context.nc_handle.lock();
            let lock_wait = lock_start.elapsed().as_millis();
            if lock_wait > LOCK_WAIT_WARNING_MS {
                warn!("[WindProvider] 锁等待过长: {} ms", lock_wait);
            }
            context
                .stats
                .lock_metrics
                .record_read_wait(lock_wait as u64);
            (
                nc.read_2d_slice(&u_map.source_var, Some(idx)),
                nc.read_2d_slice(&v_map.source_var, Some(idx)),
            )
        };

        let (mut raw_u, mut raw_v) = (raw_u?, raw_v?);
        let total_cells = raw_u.len();

        let u_invalid = Self::apply_transform(&mut raw_u, u_map.scale_factor, u_map.offset);
        let v_invalid = Self::apply_transform(&mut raw_v, v_map.scale_factor, v_map.offset);

        let invalid_ratio = (u_invalid + v_invalid) as f64 / (total_cells * 2) as f64;
        if invalid_ratio > INVALID_VALUE_RATIO_THRESHOLD {
            warn!(
                "[WindProvider] 帧 {} 异常值占比: {:.2}%",
                idx,
                invalid_ratio * 100.0
            );
        }

        target.reset();

        let raw_u_flat: Vec<f64> = raw_u.into_iter().collect();
        let raw_v_flat: Vec<f64> = raw_v.into_iter().collect();

        context.interpolator.interpolate_vector_field(
            &raw_u_flat,
            &raw_v_flat,
            &mut target.u,
            &mut target.v,
            *context.nodata_strategy,
        )?;

        context
            .stats
            .record_load(load_start.elapsed().as_millis() as u64);
        Ok(())
    }

    fn apply_transform(data: &mut ndarray::Array2<f64>, scale: f64, offset: f64) -> usize {
        if !scale.is_finite() || !offset.is_finite() {
            data.fill(f64::NAN);
            return data.len();
        }
        let mut invalid_count = 0;
        for val in data.iter_mut() {
            if !val.is_nan() {
                *val = val.mul_add(scale, offset);
                if !val.is_finite() || val.abs() > MAX_REASONABLE_WIND_SPEED {
                    invalid_count += 1;
                }
            }
        }
        invalid_count
    }

    fn validate_mappings(source: &DataSourceConfig) -> MhResult<()> {
        for mapping in &source.mappings {
            let scale = mapping.scale_factor;
            let offset = mapping.offset;
            if !scale.is_finite() || !offset.is_finite() {
                return Err(MhError::Config(format!(
                    "变量 '{}' 的 scale_factor 或 offset 无效",
                    mapping.target_var
                )));
            }
            if scale.abs() > MAX_SCALE_FACTOR || scale.abs() < MIN_SCALE_FACTOR {
                warn!(
                    "[WindProvider] 变量 '{}' 的 scale_factor={} 异常",
                    mapping.target_var, scale
                );
            }
        }
        Ok(())
    }

    fn load_time_axis(
        nc_handle: &Arc<Mutex<NcCore>>,
        stats: &Arc<ProviderStats>,
        time_var: Option<&str>,
    ) -> MhResult<Vec<DateTime<Utc>>> {
        let lock_start = Instant::now();
        let nc = nc_handle.lock();
        stats
            .lock_metrics
            .record_metadata_wait(lock_start.elapsed().as_millis() as u64);

        let var_name = time_var.unwrap_or("time");
        let var = nc.get_variable(var_name)?;
        let units = var
            .attribute("units")
            .and_then(|a| a.value().as_str().map(|s| s.to_string()))
            .ok_or_else(|| MhError::DataLoad {
                file: "".into(),
                message: format!("时间变量 '{}' 缺失 units 属性", var_name),
            })?;

        let (base, mult) = parse_cf_time_units(&units)?;
        let values = var.values::<f64>(None, None).map_err(MhError::NetCdf)?;
        Ok(values
            .iter()
            .map(|&v| calculate_utc_time(v, base, mult))
            .collect())
    }

    fn validate_time_axis(axis: &[DateTime<Utc>], file: &str) -> MhResult<()> {
        if axis.is_empty() {
            return Err(MhError::DataLoad {
                file: file.into(),
                message: "时间轴为空".into(),
            });
        }
        if axis.len() < MIN_TIME_STEPS {
            return Err(MhError::DataLoad {
                file: file.into(),
                message: format!("时间步不足 (需要至少 {} 帧)", MIN_TIME_STEPS),
            });
        }
        if axis.windows(2).any(|w| w[0] >= w[1]) {
            return Err(MhError::DataLoad {
                file: file.into(),
                message: "时间轴非单调递增".into(),
            });
        }
        Ok(())
    }

    fn resolve_nodata_strategy(source: &DataSourceConfig) -> NoDataStrategy {
        source
            .metadata
            .get("nodata_strategy")
            .and_then(|s| match s.to_lowercase().as_str() {
                "zero" => Some(NoDataStrategy::for_wind()),
                "nearest" => Some(NoDataStrategy::KeepOriginal),
                "error" | "fail" => Some(NoDataStrategy::SetNaN),
                _ => None,
            })
            .unwrap_or_else(NoDataStrategy::for_wind)
    }

    fn find_start_index(
        time_axis: &[DateTime<Utc>],
        manifest: &ProjectManifest,
    ) -> MhResult<usize> {
        let start_time = if let Some(time_str) = manifest.meta.get("start_time") {
            DateTime::parse_from_rfc3339(time_str)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|_| MhError::Config("无法解析起始时间，请使用 RFC3339 格式".into()))?
        } else {
            debug!("[WindProvider] 未指定起始时间，使用数据首帧");
            time_axis[0]
        };
        Self::binary_search_time(time_axis, start_time)
    }

    fn binary_search_time(time_axis: &[DateTime<Utc>], target: DateTime<Utc>) -> MhResult<usize> {
        if time_axis.is_empty() {
            return Err(MhError::Config("时间轴为空".into()));
        }
        match time_axis.binary_search(&target) {
            Ok(idx) => Ok(idx),
            Err(insert_pos) => {
                if insert_pos == 0 {
                    Ok(0)
                } else if insert_pos >= time_axis.len() {
                    Ok(time_axis.len().saturating_sub(1))
                } else {
                    Ok(insert_pos - 1)
                }
            }
        }
    }
}

struct LoadContext<'a> {
    config: &'a DataSourceConfig,
    interpolator: &'a SpatialInterpolator,
    nodata_strategy: &'a NoDataStrategy,
    nc_handle: Arc<Mutex<NcCore>>,
    stats: Arc<ProviderStats>,
    src_dims: (usize, usize),
}

impl Drop for WindProvider {
    fn drop(&mut self) {
        #[cfg(not(test))]
        {
            let queries = self.stats.time_queries.load(Ordering::Relaxed);
            let loads = self.stats.frame_loads.load(Ordering::Relaxed);
            if queries > 10 || loads > 2 {
                info!(
                    "[WindProvider] 释放资源: 查询 {} 次, 加载 {} 帧",
                    queries, loads
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_search_boundaries() {
        let axis = vec![
            "2024-01-01T00:00:00Z".parse().unwrap(),
            "2024-01-01T01:00:00Z".parse().unwrap(),
            "2024-01-01T02:00:00Z".parse().unwrap(),
        ];
        assert_eq!(
            WindProvider::binary_search_time(&axis, "2023-12-31T00:00:00Z".parse().unwrap())
                .unwrap(),
            0
        );
        assert_eq!(
            WindProvider::binary_search_time(&axis, "2025-01-01T00:00:00Z".parse().unwrap())
                .unwrap(),
            2
        );
        assert_eq!(
            WindProvider::binary_search_time(&axis, "2024-01-01T01:30:00Z".parse().unwrap())
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_metrics_serialization() {
        let stats = ProviderStats::default();
        stats.record_load(100);
        stats.record_query();
        let metrics = stats.to_metrics();
        let json = serde_json::to_string(&metrics).unwrap();
        assert!(json.contains("\"frame_loads\":1"));
    }
}
