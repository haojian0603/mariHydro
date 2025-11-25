use chrono::{DateTime, Utc};
use log::{debug, info, trace, warn};
use ndarray::{Array2, ArrayViewMut2, Zip};
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::Serialize;
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::marihydro::domain::interpolator::{NoDataStrategy, SpatialInterpolator};
use crate::marihydro::domain::mesh::Mesh;
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::manifest::{DataSourceConfig, ProjectManifest};
use crate::marihydro::io::drivers::nc_adapter::{
    core::NcCore,
    time::{calculate_utc_time, parse_cf_time_units},
};
use crate::marihydro::io::drivers::nc_loader::NetCdfLoader;
use crate::marihydro::io::traits::RasterDriver;

// ============================================================================
// 常量配置
// ============================================================================

const MIN_TIME_STEPS: usize = 2;
const TIME_EPSILON: f64 = 1e-6;
const FRAME_LOAD_WARNING_MS: u128 = 500;
const LOCK_WAIT_WARNING_MS: u128 = 100;

/// 物理合理的风速上限 (m/s)，参考：地球最大观测风速 ~113 m/s
const MAX_REASONABLE_WIND_SPEED: f64 = 100.0;

const MAX_SCALE_FACTOR: f64 = 1e6;
const MIN_SCALE_FACTOR: f64 = 1e-6;

/// 异常值报警阈值（占比 1%）
const INVALID_VALUE_RATIO_THRESHOLD: f64 = 0.01;

// ============================================================================
// 内部数据结构
// ============================================================================

#[derive(Debug, Clone)]
struct BufferedFrame {
    time: DateTime<Utc>,
    u: Array2<f64>,
    v: Array2<f64>,
}

impl BufferedFrame {
    fn new(time: DateTime<Utc>, ny: usize, nx: usize) -> MhResult<Self> {
        let total_size = ny
            .checked_mul(nx)
            .ok_or_else(|| MhError::InvalidMesh(format!("网格维度溢出: {} × {}", ny, nx)))?;

        let u = Array2::zeros((ny, nx));
        let v = Array2::zeros((ny, nx));

        if u.len() != total_size {
            return Err(MhError::Config(format!(
                "内存分配失败: 请求 {} 单元 ({:.2} MB)",
                total_size,
                total_size as f64 * 16.0 / 1_000_000.0
            )));
        }

        Ok(Self { time, u, v })
    }

    #[inline]
    fn update_time(&mut self, time: DateTime<Utc>) {
        self.time = time;
    }

    #[inline]
    fn reset(&mut self) {
        rayon::join(|| self.u.fill(0.0), || self.v.fill(0.0));
    }
}

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

    #[cfg(feature = "detailed_stats")]
    fn record_exhausted_call(&self) {
        self.exhausted_calls.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(not(feature = "detailed_stats"))]
    #[inline]
    fn record_exhausted_call(&self) {}

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

/// 性能指标（可导出为 JSON）
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

// ============================================================================
// 核心 Provider
// ============================================================================

pub struct WindProvider {
    config: DataSourceConfig,
    interpolator: SpatialInterpolator,
    mesh_indices: Vec<(usize, usize)>,
    nodata_strategy: NoDataStrategy,

    ny: usize,
    nx: usize,

    time_axis: Vec<DateTime<Utc>>,
    current_idx: usize,

    frame_curr: BufferedFrame,
    frame_next: BufferedFrame,

    nc_handle: Arc<Mutex<NcCore>>,
    stats: Arc<ProviderStats>,

    is_exhausted: bool,
}

impl WindProvider {
    pub fn init(
        source: &DataSourceConfig,
        mesh: &Mesh,
        manifest: &ProjectManifest,
    ) -> MhResult<Self> {
        info!("[WindProvider] 初始化风场源: {}", source.name);

        if mesh.active_indices.is_empty() {
            return Err(MhError::InvalidMesh("活动单元格为空".into()));
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
        let interpolator = SpatialInterpolator::new(mesh, &manifest.crs_wkt, &src_meta)?;
        let mesh_indices = mesh.active_indices.clone();

        info!(
            "[WindProvider] 插值器就绪 ({} 活动单元, 覆盖率 {:.1}%)",
            mesh_indices.len(),
            100.0 * mesh_indices.len() as f64 / (mesh.ny * mesh.nx) as f64
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

        let mut frame_curr = BufferedFrame::new(time_axis[start_idx], mesh.ny, mesh.nx)?;
        let mut frame_next = BufferedFrame::new(time_axis[start_idx + 1], mesh.ny, mesh.nx)?;

        let context = LoadContext {
            config: source,
            interpolator: &interpolator,
            mesh_indices: &mesh_indices,
            nodata_strategy: &nodata_strategy,
            nc_handle: Arc::clone(&nc_handle),
            stats: Arc::clone(&stats),
        };

        info!("[WindProvider] 预加载初始帧...");
        let start_load = Instant::now();

        let (res_curr, res_next) = rayon::join(
            || Self::load_into_buffer(&context, start_idx, &mut frame_curr),
            || Self::load_into_buffer(&context, start_idx + 1, &mut frame_next),
        );

        res_curr?;
        res_next?;

        info!(
            "[WindProvider] 就绪 (预加载耗时 {} ms)",
            start_load.elapsed().as_millis()
        );

        Ok(Self {
            config: source.clone(),
            interpolator,
            mesh_indices,
            nodata_strategy,
            ny: mesh.ny,
            nx: mesh.nx,
            time_axis,
            current_idx: start_idx,
            frame_curr,
            frame_next,
            nc_handle,
            stats,
            is_exhausted: false,
        })
    }

    /// 更新风场到指定时刻（零分配热路径）
    ///
    /// 警告：禁止在 rayon 线程池内调用，会导致死锁！
    pub fn update_wind_at(
        &mut self,
        time: DateTime<Utc>,
        out_u: &mut ArrayViewMut2<f64>,
        out_v: &mut ArrayViewMut2<f64>,
    ) -> MhResult<()> {
        trace!("[WindProvider] update_wind_at: {}", time);
        self.stats.record_query();

        if out_u.dim() != (self.ny, self.nx) || out_v.dim() != (self.ny, self.nx) {
            return Err(MhError::InvalidMesh(format!(
                "输出缓冲区维度不匹配: 期望 ({}, {}), 实际 u={:?}, v={:?}",
                self.ny,
                self.nx,
                out_u.dim(),
                out_v.dim()
            )));
        }

        // 快速路径：数据耗尽
        if self.is_exhausted {
            self.stats.record_exhausted_call();
            Self::copy_into_optimized(&self.frame_next.u, &self.frame_next.v, out_u, out_v);
            return Ok(());
        }

        // 时间回退检测
        if time < self.frame_curr.time {
            info!("[WindProvider] 检测到时间回退，重新定位...");
            self.seek_to_time(time)?;
        }

        // 滚动缓冲
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
                Self::copy_into_optimized(&self.frame_next.u, &self.frame_next.v, out_u, out_v);
                return Ok(());
            }

            self.advance_frame()?;
        }

        // 时间线性插值
        let dt_total = (self.frame_next.time - self.frame_curr.time).num_seconds() as f64;
        let dt_curr = (time - self.frame_curr.time).num_seconds() as f64;

        let alpha = if dt_total.abs() < TIME_EPSILON {
            0.0
        } else {
            (dt_curr / dt_total).clamp(0.0, 1.0)
        };
        let beta = 1.0 - alpha;

        // 合并 U/V 插值循环，提高 Cache 命中率
        Zip::from(out_u)
            .and(out_v)
            .and(&self.frame_curr.u)
            .and(&self.frame_next.u)
            .and(&self.frame_curr.v)
            .and(&self.frame_next.v)
            .par_for_each(|res_u, res_v, &u0, &u1, &v0, &v1| {
                *res_u = u0.mul_add(beta, u1 * alpha);
                *res_v = v0.mul_add(beta, v1 * alpha);
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
        let indices = self.mesh_indices.len() * std::mem::size_of::<(usize, usize)>();
        let time_axis = self.time_axis.len() * std::mem::size_of::<DateTime<Utc>>();

        frames + indices + time_axis + std::mem::size_of::<Self>()
    }

    // ========================================================================
    // 内部实现
    // ========================================================================

    fn advance_frame(&mut self) -> MhResult<()> {
        self.current_idx = self
            .current_idx
            .checked_add(1)
            .ok_or_else(|| MhError::Config("当前索引溢出".into()))?;

        mem::swap(&mut self.frame_curr, &mut self.frame_next);

        let next_idx = self
            .current_idx
            .checked_add(1)
            .ok_or_else(|| MhError::Config("下一帧索引溢出".into()))?;

        self.frame_next.update_time(self.time_axis[next_idx]);

        let context = LoadContext {
            config: &self.config,
            interpolator: &self.interpolator,
            mesh_indices: &self.mesh_indices,
            nodata_strategy: &self.nodata_strategy,
            nc_handle: Arc::clone(&self.nc_handle),
            stats: Arc::clone(&self.stats),
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
        let next_idx = target_idx
            .checked_add(1)
            .ok_or_else(|| MhError::Config("目标索引溢出".into()))?;

        if next_idx >= self.time_axis.len() {
            return Err(MhError::DataLoad {
                file: self.config.file_path.clone(),
                message: "Seek 目标越界".into(),
            });
        }

        self.current_idx = target_idx;
        self.frame_curr.update_time(self.time_axis[target_idx]);
        self.frame_next.update_time(self.time_axis[next_idx]);

        let context = LoadContext {
            config: &self.config,
            interpolator: &self.interpolator,
            mesh_indices: &self.mesh_indices,
            nodata_strategy: &self.nodata_strategy,
            nc_handle: Arc::clone(&self.nc_handle),
            stats: Arc::clone(&self.stats),
        };

        rayon::join(
            || Self::load_into_buffer(&context, target_idx, &mut self.frame_curr),
            || Self::load_into_buffer(&context, next_idx, &mut self.frame_next),
        );

        Ok(())
    }

    #[inline]
    fn copy_into_optimized(
        src_u: &Array2<f64>,
        src_v: &Array2<f64>,
        dst_u: &mut ArrayViewMut2<f64>,
        dst_v: &mut ArrayViewMut2<f64>,
    ) {
        rayon::join(|| dst_u.assign(src_u), || dst_v.assign(src_v));
    }

    fn load_into_buffer(
        context: &LoadContext,
        idx: usize,
        target: &mut BufferedFrame,
    ) -> MhResult<()> {
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

        // 串行 I/O（避免磁头跳跃）
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

        // 并行应用线性变换 + 原子计数异常值
        let (u_invalid, v_invalid) = rayon::join(
            || Self::apply_transform(&mut raw_u, u_map.scale_factor, u_map.offset),
            || Self::apply_transform(&mut raw_v, v_map.scale_factor, v_map.offset),
        );

        let invalid_ratio = (u_invalid + v_invalid) as f64 / (total_cells * 2) as f64;
        if invalid_ratio > INVALID_VALUE_RATIO_THRESHOLD {
            warn!(
                "[WindProvider] 帧 {} 异常值占比: {:.2}%",
                idx,
                invalid_ratio * 100.0
            );
        }

        target.reset();

        context.interpolator.interpolate_vector_field(
            &raw_u,
            &raw_v,
            &mut target.u,
            &mut target.v,
            context.mesh_indices,
            *context.nodata_strategy,
        )?;

        context
            .stats
            .record_load(load_start.elapsed().as_millis() as u64);

        Ok(())
    }

    /// 应用线性变换并统计异常值（并行安全）
    fn apply_transform(data: &mut Array2<f64>, scale: f64, offset: f64) -> usize {
        if !scale.is_finite() || !offset.is_finite() {
            data.fill(f64::NAN);
            return data.len();
        }

        let invalid_count = AtomicUsize::new(0);

        Zip::from(data).par_for_each(|val| {
            if !val.is_nan() {
                *val = val.mul_add(scale, offset);
                if !val.is_finite() || val.abs() > MAX_REASONABLE_WIND_SPEED {
                    invalid_count.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        invalid_count.load(Ordering::Acquire)
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
                "nearest" => Some(NoDataStrategy::NearestNeighbor),
                "error" | "fail" => Some(NoDataStrategy::Error),
                _ => None,
            })
            .unwrap_or_else(NoDataStrategy::for_wind)
    }

    fn find_start_index(
        time_axis: &[DateTime<Utc>],
        manifest: &ProjectManifest,
    ) -> MhResult<usize> {
        let start_time = if let Some(time_str) = manifest.metadata.get("start_time") {
            DateTime::parse_from_rfc3339(time_str)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|_| {
                    MhError::Config(
                        "无法解析起始时间，请使用 RFC3339 格式 (如 '2024-01-01T00:00:00Z')".into(),
                    )
                })?
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
    mesh_indices: &'a [(usize, usize)],
    nodata_strategy: &'a NoDataStrategy,
    nc_handle: Arc<Mutex<NcCore>>,
    stats: Arc<ProviderStats>,
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
    fn test_buffered_frame_allocation() {
        let frame = BufferedFrame::new(Utc::now(), 10, 10);
        assert!(frame.is_ok());
        assert_eq!(frame.unwrap().u.dim(), (10, 10));
    }

    #[test]
    fn test_apply_transform_parallel_counting() {
        let mut data = Array2::from_elem((10, 10), 200.0);
        let invalid = WindProvider::apply_transform(&mut data, 1.0, 0.0);
        assert_eq!(invalid, 100); // 全部超过 MAX_REASONABLE_WIND_SPEED
    }

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
