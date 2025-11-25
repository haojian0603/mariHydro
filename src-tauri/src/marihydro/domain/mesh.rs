// src-tauri/src/marihydro/domain/mesh.rs

use log::{info, warn};
use ndarray::{s, Array2, ArrayView2, Axis, Zip};
use rayon::prelude::*;
use std::sync::Arc;

use crate::marihydro::geo::crs::Crs;
use crate::marihydro::geo::transform::GeoTransformer;
use crate::marihydro::infra::constants::{defaults, physics, tolerances, validation};
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::manifest::ProjectManifest;
use crate::marihydro::io::loaders::raster::StandardRasterLoader;
use crate::marihydro::io::traits::{RasterLoader, RasterRequest};
use crate::marihydro::io::types::GeoTransform;

const PREALLOCATE_FACTOR: f64 = 0.9;
const MAX_CELLS_WARNING: usize = 100_000_000;
const MIN_ACTIVE_RATIO: f64 = 0.01;
const SUSPICIOUS_ELEVATION_THRESHOLD: f64 = 100.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellType {
    Active = 1,
    Inactive = 0,
}

impl From<u8> for CellType {
    fn from(val: u8) -> Self {
        match val {
            1 => CellType::Active,
            _ => CellType::Inactive,
        }
    }
}

impl From<CellType> for u8 {
    fn from(ct: CellType) -> Self {
        ct as u8
    }
}

#[derive(Debug, Clone)]
pub enum IndexStorage {
    TwoDim(Vec<(usize, usize)>),
    OneDim { indices: Vec<usize>, stride: usize },
}

impl IndexStorage {
    pub fn len(&self) -> usize {
        match self {
            Self::TwoDim(v) => v.len(),
            Self::OneDim { indices, .. } => indices.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get_2d(&self, k: usize) -> (usize, usize) {
        match self {
            Self::TwoDim(v) => v[k],
            Self::OneDim { indices, stride } => {
                let idx = indices[k];
                (idx / stride, idx % stride)
            }
        }
    }

    fn optimal_chunk_size(&self) -> usize {
        let len = self.len();
        match len {
            0..=1000 => 100,
            1001..=100_000 => 1024,
            _ => (len / 64).max(1024),
        }
    }

    pub fn for_each_active<F>(&self, op: F)
    where
        F: Fn(usize, usize) + Sync + Send,
    {
        let chunk = self.optimal_chunk_size();
        match self {
            Self::TwoDim(v) => {
                v.par_iter()
                    .with_min_len(chunk)
                    .for_each(|&(j, i)| op(j, i));
            }
            Self::OneDim { indices, stride } => {
                let s = *stride;
                indices.par_iter().with_min_len(chunk).for_each(|&idx| {
                    op(idx / s, idx % s);
                });
            }
        }
    }

    pub fn try_for_each_active<F, E>(&self, op: F) -> Result<(), E>
    where
        F: Fn(usize, usize) -> Result<(), E> + Sync + Send,
        E: Send,
    {
        let chunk = self.optimal_chunk_size();
        match self {
            Self::TwoDim(v) => v
                .par_iter()
                .with_min_len(chunk)
                .try_for_each(|&(j, i)| op(j, i)),
            Self::OneDim { indices, stride } => {
                let s = *stride;
                indices
                    .par_iter()
                    .with_min_len(chunk)
                    .try_for_each(|&idx| op(idx / s, idx % s))
            }
        }
    }

    pub fn iter(&self) -> IndexIterator {
        IndexIterator {
            storage: self,
            index: 0,
        }
    }

    pub fn par_iter(&self) -> impl ParallelIterator<Item = (usize, usize)> + '_ {
        match self {
            Self::TwoDim(v) => rayon::iter::Either::Left(v.par_iter().copied()),
            Self::OneDim { indices, stride } => {
                let s = *stride;
                rayon::iter::Either::Right(indices.par_iter().map(move |&idx| (idx / s, idx % s)))
            }
        }
    }

    fn memory_bytes(&self) -> usize {
        match self {
            Self::TwoDim(v) => {
                v.capacity() * std::mem::size_of::<(usize, usize)>()
                    + std::mem::size_of::<Vec<(usize, usize)>>()
            }
            Self::OneDim { indices, .. } => {
                indices.capacity() * std::mem::size_of::<usize>()
                    + std::mem::size_of::<Vec<usize>>()
                    + std::mem::size_of::<usize>()
            }
        }
    }
}

pub struct IndexIterator<'a> {
    storage: &'a IndexStorage,
    index: usize,
}

impl<'a> Iterator for IndexIterator<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.storage.len() {
            let result = self.storage.get_2d(self.index);
            self.index += 1;
            Some(result)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.storage.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for IndexIterator<'a> {}

pub struct Mesh {
    pub nx: usize,
    pub ny: usize,
    pub ng: usize,
    pub transform: GeoTransform,

    pub zb: Arc<Array2<f64>>, // ✅ 修改为Arc
    pub roughness: Array2<f64>,
    pub mask: Array2<u8>,
    pub coriolis_f: Array2<f64>,

    pub active_indices: IndexStorage,
    pub h_min: f64,
}

impl Mesh {
    pub fn init(manifest: &ProjectManifest) -> MhResult<Self> {
        Self::init_with_ghost_width(manifest, defaults::GHOST_WIDTH)
    }

    pub fn init_with_ghost_width(manifest: &ProjectManifest, ng: usize) -> MhResult<Self> {
        Self::validate_grid_parameters(manifest, ng)?;

        let nx = manifest.grid_nx;
        let ny = manifest.grid_ny;
        let total_nx = nx + 2 * ng;
        let total_ny = ny + 2 * ng;

        let transform = Self::build_geotransform(manifest)?;

        let mut zb = Array2::from_elem((total_ny, total_nx), f64::NAN);
        let mut roughness = Array2::from_elem((total_ny, total_nx), f64::NAN);
        let mut mask = Array2::zeros((total_ny, total_nx));

        let (raw_zb, raw_roughness) = rayon::join(
            || Self::load_elevation(manifest, nx, ny),
            || Self::load_roughness(manifest, nx, ny),
        );
        let raw_zb = raw_zb?;
        let raw_roughness = raw_roughness?;

        Self::fill_physical_domain(&mut zb, &raw_zb, nx, ny, ng);
        Self::fill_physical_domain(&mut roughness, &raw_roughness, nx, ny, ng);

        rayon::join(
            || Self::extrapolate_ghost_vectorized(&mut zb, nx, ny, ng),
            || Self::extrapolate_ghost_vectorized(&mut roughness, nx, ny, ng),
        );

        let coriolis_f = Self::compute_coriolis_field(
            total_nx,
            total_ny,
            ng,
            &transform,
            &manifest.crs_wkt,
            manifest.physics.latitude_ref,
            manifest.physics.enable_coriolis,
        );

        let (active_indices, stats) =
            Self::build_mask_and_indices(&zb, &mut mask, nx, ny, ng, total_nx);

        Self::validate_mesh(&active_indices, &stats, &zb, nx, ny, ng)?;

        info!(
            "Mesh就绪: {}×{} (Ghost {}), Active: {}, Ratio: {:.1}%, 索引内存: {:.2}KB",
            nx,
            ny,
            ng,
            stats.active_count,
            stats.active_ratio() * 100.0,
            active_indices.memory_bytes() as f64 / 1024.0
        );

        Ok(Self {
            nx,
            ny,
            ng,
            transform,
            zb: Arc::new(zb), // ✅ 包装为Arc
            roughness,
            mask,
            coriolis_f,
            active_indices,
            h_min: manifest.physics.h_min,
        })
    }

    /// 获取Arc引用 (供State使用)
    pub fn zb_arc(&self) -> Arc<Array2<f64>> {
        Arc::clone(&self.zb)
    }

    fn validate_grid_parameters(manifest: &ProjectManifest, ng: usize) -> MhResult<()> {
        if manifest.grid_nx == 0 || manifest.grid_ny == 0 {
            return Err(MhError::InvalidMesh("网格尺寸无效".into()));
        }
        if manifest.grid_dx <= 0.0 || manifest.grid_dy <= 0.0 {
            return Err(MhError::InvalidMesh("网格分辨率无效".into()));
        }
        if ng == 0 {
            return Err(MhError::InvalidMesh("幽灵层宽度必须>0".into()));
        }

        let total_cells = (manifest.grid_nx + 2 * ng) * (manifest.grid_ny + 2 * ng);
        if total_cells > MAX_CELLS_WARNING {
            warn!("网格规模较大 ({} 单元)", total_cells);
        }
        Ok(())
    }

    fn build_geotransform(manifest: &ProjectManifest) -> MhResult<GeoTransform> {
        let origin_x = manifest.origin_x.unwrap_or(0.0);
        let origin_y = manifest.origin_y.unwrap_or(0.0);
        Ok(GeoTransform::new(
            origin_x,
            origin_y,
            manifest.grid_dx,
            -manifest.grid_dy,
        ))
    }

    fn compute_coriolis_field(
        total_nx: usize,
        total_ny: usize,
        ng: usize,
        transform: &GeoTransform,
        crs_wkt: &str,
        lat_ref: f64,
        enabled: bool,
    ) -> Array2<f64> {
        let mut field = Array2::zeros((total_ny, total_nx));
        if !enabled {
            return field;
        }

        let transformer_res = GeoTransformer::new(&Crs::from_string(crs_wkt), &Crs::wgs84());
        match transformer_res {
            Ok(transformer) => {
                Zip::from(&mut field).indexed_par_for_each(|(j, i), val| {
                    let px = i as f64 - ng as f64;
                    let py = j as f64 - ng as f64;
                    let (mx, my) = transform.pixel_to_world(px, py);
                    let lat = transformer
                        .transform_point(mx, my)
                        .map(|(_, lat)| lat)
                        .unwrap_or(lat_ref);
                    *val = 2.0 * physics::EARTH_ROTATION_RATE_RAD * lat.to_radians().sin();
                });
            }
            Err(_) => {
                let f0 = 2.0 * physics::EARTH_ROTATION_RATE_RAD * lat_ref.to_radians().sin();
                field.fill(f0);
            }
        }
        field
    }

    fn load_elevation(manifest: &ProjectManifest, nx: usize, ny: usize) -> MhResult<Array2<f64>> {
        Self::load_layer(
            manifest,
            &["zb", "elevation"],
            nx,
            ny,
            defaults::ELEVATION,
            "地形高程",
        )
    }

    fn load_roughness(manifest: &ProjectManifest, nx: usize, ny: usize) -> MhResult<Array2<f64>> {
        Self::load_layer(
            manifest,
            &["roughness", "manning"],
            nx,
            ny,
            manifest.physics.bottom_friction_coeff,
            "曼宁糙率",
        )
    }

    fn load_layer(
        manifest: &ProjectManifest,
        target_vars: &[&str],
        nx: usize,
        ny: usize,
        fallback_val: f64,
        layer_name: &str,
    ) -> MhResult<Array2<f64>> {
        let source_opt = manifest.sources.iter().find(|s| {
            s.mappings
                .iter()
                .any(|m| target_vars.contains(&m.target_var.as_str()))
        });

        match source_opt {
            Some(src) => Self::load_from_source(src, target_vars, nx, ny, layer_name),
            None => {
                info!("未配置{}数据源，使用默认值: {}", layer_name, fallback_val);
                Ok(Array2::from_elem((ny, nx), fallback_val))
            }
        }
    }

    fn load_from_source(
        source: &crate::marihydro::infra::manifest::DataSource,
        target_vars: &[&str],
        nx: usize,
        ny: usize,
        layer_name: &str,
    ) -> MhResult<Array2<f64>> {
        let mapping = source
            .mappings
            .iter()
            .find(|m| target_vars.contains(&m.target_var.as_str()))
            .ok_or_else(|| MhError::Config(format!("未找到{}的映射", layer_name)))?;

        let loader = StandardRasterLoader;
        let mut data = loader
            .load_array(&source.file_path, (nx, ny), None)
            .map_err(|e| MhError::DataLoad {
                file: source.file_path.clone(),
                message: format!("{}加载失败: {}", layer_name, e),
            })?;

        Self::apply_transformation(&mut data, mapping.scale_factor, mapping.offset);
        if let Some(fill) = mapping.fallback_value {
            Self::fill_nodata(&mut data, fill);
        }
        Ok(data)
    }

    #[inline]
    fn apply_transformation(data: &mut Array2<f64>, scale: f64, offset: f64) {
        if (scale - 1.0).abs() < tolerances::EPSILON_TRANSFORM
            && offset.abs() < tolerances::EPSILON_TRANSFORM
        {
            return;
        }
        Zip::from(data).par_for_each(|val| {
            if !val.is_nan() {
                *val = *val * scale + offset;
            }
        });
    }

    #[inline]
    fn fill_nodata(data: &mut Array2<f64>, fill_value: f64) {
        Zip::from(data).par_for_each(|val| {
            if val.is_nan() {
                *val = fill_value;
            }
        });
    }

    fn fill_physical_domain(
        target: &mut Array2<f64>,
        source: &Array2<f64>,
        nx: usize,
        ny: usize,
        ng: usize,
    ) {
        target
            .slice_mut(s![ng..ng + ny, ng..ng + nx])
            .assign(source);
    }

    fn extrapolate_ghost_vectorized(arr: &mut Array2<f64>, nx: usize, ny: usize, ng: usize) {
        let (total_ny, total_nx) = arr.dim();

        arr.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(j, mut row)| {
                if j >= ng && j < ny + ng {
                    let left_val = row[ng];
                    let right_val = row[nx + ng - 1];
                    row.slice_mut(s![0..ng]).fill(left_val);
                    row.slice_mut(s![nx + ng..total_nx]).fill(right_val);
                }
            });

        let bottom_row = arr.row(ng).to_owned();
        let top_row = arr.row(ny + ng - 1).to_owned();

        arr.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(j, mut row)| {
                if j < ng {
                    row.assign(&bottom_row);
                } else if j >= ny + ng {
                    row.assign(&top_row);
                }
            });
    }

    fn build_mask_and_indices(
        zb: &Array2<f64>,
        mask: &mut Array2<u8>,
        nx: usize,
        ny: usize,
        ng: usize,
        total_nx: usize,
    ) -> (IndexStorage, MeshStats) {
        let estimated_capacity = (nx * ny as f64 * PREALLOCATE_FACTOR) as usize;
        let mut indices = Vec::with_capacity(estimated_capacity);
        let mut active_count = 0;
        let mut inactive_count = 0;

        for j in ng..ng + ny {
            for i in ng..ng + nx {
                let z_val = zb[[j, i]];
                if !z_val.is_nan() {
                    mask[[j, i]] = CellType::Active.into();
                    indices.push(j * total_nx + i);
                    active_count += 1;
                } else {
                    mask[[j, i]] = CellType::Inactive.into();
                    inactive_count += 1;
                }
            }
        }

        let stats = MeshStats {
            active_count,
            inactive_count,
            total_count: nx * ny,
        };
        let storage = IndexStorage::OneDim {
            indices,
            stride: total_nx,
        };
        (storage, stats)
    }

    fn validate_mesh(
        active_indices: &IndexStorage,
        stats: &MeshStats,
        zb: &Array2<f64>,
        nx: usize,
        ny: usize,
        ng: usize,
    ) -> MhResult<()> {
        if active_indices.is_empty() {
            return Err(MhError::InvalidMesh("有效网格数量为0".into()));
        }
        if stats.active_ratio() < MIN_ACTIVE_RATIO {
            warn!("网格激活率过低 ({:.2}%)", stats.active_ratio() * 100.0);
        }

        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;
        let mut suspicious_high_count = 0;

        for j in ng..ng + ny {
            for i in ng..ng + nx {
                let z = zb[[j, i]];
                if !z.is_nan() {
                    min_z = min_z.min(z);
                    max_z = max_z.max(z);
                    if z > SUSPICIOUS_ELEVATION_THRESHOLD {
                        suspicious_high_count += 1;
                    }
                }
            }
        }

        info!("地形高程范围: [{:.2}, {:.2}]m", min_z, max_z);
        if min_z > 0.0 {
            warn!("所有网格高程>0，请确认基准面设置");
        }
        if suspicious_high_count > stats.active_count / 2 {
            warn!(
                "超过50%网格高程>{}m，请检查数据单位",
                SUSPICIOUS_ELEVATION_THRESHOLD
            );
        }
        Ok(())
    }

    pub fn physical_domain(&self) -> ArrayView2<f64> {
        self.zb
            .slice(s![self.ng..self.ng + self.ny, self.ng..self.ng + self.nx])
    }

    pub fn total_size(&self) -> (usize, usize) {
        (self.ny + 2 * self.ng, self.nx + 2 * self.ng)
    }

    #[inline]
    pub fn is_physical(&self, j: usize, i: usize) -> bool {
        j >= self.ng && j < self.ng + self.ny && i >= self.ng && i < self.ng + self.nx
    }

    #[inline]
    pub fn is_active(&self, j: usize, i: usize) -> bool {
        self.mask[[j, i]] == CellType::Active as u8
    }

    pub fn memory_usage(&self) -> usize {
        let arrays_size = self.zb.len() * std::mem::size_of::<f64>() * 3;
        let mask_size = self.mask.len() * std::mem::size_of::<u8>();
        let indices_size = self.active_indices.memory_bytes();
        arrays_size + mask_size + indices_size + std::mem::size_of::<Self>()
    }
}

#[derive(Debug, Clone)]
struct MeshStats {
    active_count: usize,
    inactive_count: usize,
    total_count: usize,
}

impl MeshStats {
    fn active_ratio(&self) -> f64 {
        self.active_count as f64 / self.total_count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_type_conversion() {
        assert_eq!(u8::from(CellType::Active), 1);
        assert_eq!(CellType::from(1u8), CellType::Active);
    }

    #[test]
    fn test_index_storage_onedim() {
        let storage = IndexStorage::OneDim {
            indices: vec![25, 35, 45],
            stride: 10,
        };
        assert_eq!(storage.len(), 3);
        assert_eq!(storage.get_2d(0), (2, 5));
    }
}
