// crates/mh_terrain/src/tiled.rs

//! 分块地形管理
//!
//! 支持大规模地形数据的分块加载和管理，适用于：
//! - 大范围地形数据（超出内存限制）
//! - 按需加载（只加载视野内或计算区域内的瓦片）
//! - 多级细节（LOD）支持
//! - 瓦片缓存和预取
//!
//! # 坐标系统
//!
//! 使用 (tile_x, tile_y) 表示瓦片坐标，(col, row) 表示瓦片内像素坐标。
//! 全局坐标 (x, y) 可以通过地理变换转换。
//!
//! # 示例
//!
//! ```ignore
//! use mh_terrain::tiled::{TiledTerrain, TileSource, TileConfig};
//!
//! let config = TileConfig {
//!     tile_size: 256,
//!     resolution: 10.0,  // 10m/pixel
//!     origin: (0.0, 0.0),
//!     num_tiles: (10, 10),
//!     nodata: -9999.0,
//! };
//!
//! let source = FileSystemTileSource::new("tiles/", ".tif");
//! let mut terrain = TiledTerrain::new(config, Box::new(source));
//!
//! // 查询单点
//! let z = terrain.get_elevation(500.0, 300.0)?;
//!
//! // 批量查询
//! let points = vec![(100.0, 100.0), (200.0, 200.0)];
//! let elevations = terrain.get_elevations(&points);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

/// 瓦片配置
#[derive(Debug, Clone)]
pub struct TileConfig {
    /// 瓦片尺寸（像素）
    pub tile_size: usize,
    /// 分辨率（米/像素）
    pub resolution: f64,
    /// 原点坐标 (x, y)
    pub origin: (f64, f64),
    /// 瓦片数量 (nx, ny)
    pub num_tiles: (usize, usize),
    /// 无数据值
    pub nodata: f64,
    /// 最大缓存瓦片数
    pub max_cache_tiles: usize,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_size: 256,
            resolution: 1.0,
            origin: (0.0, 0.0),
            num_tiles: (1, 1),
            nodata: -9999.0,
            max_cache_tiles: 100,
        }
    }
}

impl TileConfig {
    /// 计算覆盖范围
    pub fn bounds(&self) -> [f64; 4] {
        let width = self.tile_size as f64 * self.resolution * self.num_tiles.0 as f64;
        let height = self.tile_size as f64 * self.resolution * self.num_tiles.1 as f64;
        [
            self.origin.0,
            self.origin.1,
            self.origin.0 + width,
            self.origin.1 + height,
        ]
    }

    /// 获取瓦片坐标
    pub fn tile_coords(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let dx = x - self.origin.0;
        let dy = y - self.origin.1;

        if dx < 0.0 || dy < 0.0 {
            return None;
        }

        let tile_world_size = self.tile_size as f64 * self.resolution;
        let tx = (dx / tile_world_size) as usize;
        let ty = (dy / tile_world_size) as usize;

        if tx >= self.num_tiles.0 || ty >= self.num_tiles.1 {
            return None;
        }

        Some((tx, ty))
    }

    /// 获取瓦片内局部坐标
    pub fn local_coords(&self, x: f64, y: f64) -> (f64, f64) {
        let tile_world_size = self.tile_size as f64 * self.resolution;
        let dx = x - self.origin.0;
        let dy = y - self.origin.1;

        let local_x = (dx % tile_world_size) / self.resolution;
        let local_y = (dy % tile_world_size) / self.resolution;

        (local_x, local_y)
    }
}

/// 瓦片数据
#[derive(Debug, Clone)]
pub struct Tile {
    /// 瓦片坐标
    pub coords: (usize, usize),
    /// 数据（按行存储）
    pub data: Vec<f64>,
    /// 瓦片尺寸
    pub size: usize,
    /// 无数据值
    pub nodata: f64,
}

impl Tile {
    /// 创建空瓦片
    pub fn empty(coords: (usize, usize), size: usize, nodata: f64) -> Self {
        Self {
            coords,
            data: vec![nodata; size * size],
            size,
            nodata,
        }
    }

    /// 获取像素值
    pub fn get(&self, col: usize, row: usize) -> Option<f64> {
        if col >= self.size || row >= self.size {
            return None;
        }
        let val = self.data[row * self.size + col];
        if (val - self.nodata).abs() < 1e-6 {
            None
        } else {
            Some(val)
        }
    }

    /// 设置像素值
    pub fn set(&mut self, col: usize, row: usize, value: f64) {
        if col < self.size && row < self.size {
            self.data[row * self.size + col] = value;
        }
    }

    /// 双线性插值
    pub fn interpolate(&self, x: f64, y: f64) -> Option<f64> {
        let col = x.floor() as isize;
        let row = y.floor() as isize;

        if col < 0 || row < 0 {
            return None;
        }

        let col = col as usize;
        let row = row as usize;

        if col >= self.size - 1 || row >= self.size - 1 {
            return self.get(col.min(self.size - 1), row.min(self.size - 1));
        }

        let fx = x - col as f64;
        let fy = y - row as f64;

        let z00 = self.get(col, row)?;
        let z10 = self.get(col + 1, row)?;
        let z01 = self.get(col, row + 1)?;
        let z11 = self.get(col + 1, row + 1)?;

        let z0 = z00 * (1.0 - fx) + z10 * fx;
        let z1 = z01 * (1.0 - fx) + z11 * fx;

        Some(z0 * (1.0 - fy) + z1 * fy)
    }
}

/// 瓦片数据源 trait
pub trait TileSource: Send + Sync {
    /// 加载瓦片
    fn load_tile(&self, tx: usize, ty: usize) -> Result<Tile, TileError>;

    /// 检查瓦片是否存在
    fn tile_exists(&self, tx: usize, ty: usize) -> bool;
}

/// 瓦片错误
#[derive(Debug)]
pub enum TileError {
    /// 瓦片不存在
    NotFound(usize, usize),
    /// IO 错误
    Io(std::io::Error),
    /// 格式错误
    Format(String),
}

impl std::fmt::Display for TileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TileError::NotFound(tx, ty) => write!(f, "Tile ({}, {}) not found", tx, ty),
            TileError::Io(e) => write!(f, "IO error: {}", e),
            TileError::Format(msg) => write!(f, "Format error: {}", msg),
        }
    }
}

impl std::error::Error for TileError {}

/// 内存瓦片源（用于测试）
pub struct MemoryTileSource {
    tiles: HashMap<(usize, usize), Tile>,
}

impl MemoryTileSource {
    pub fn new() -> Self {
        Self {
            tiles: HashMap::new(),
        }
    }

    pub fn add_tile(&mut self, tile: Tile) {
        self.tiles.insert(tile.coords, tile);
    }
}

impl Default for MemoryTileSource {
    fn default() -> Self {
        Self::new()
    }
}

impl TileSource for MemoryTileSource {
    fn load_tile(&self, tx: usize, ty: usize) -> Result<Tile, TileError> {
        self.tiles
            .get(&(tx, ty))
            .cloned()
            .ok_or(TileError::NotFound(tx, ty))
    }

    fn tile_exists(&self, tx: usize, ty: usize) -> bool {
        self.tiles.contains_key(&(tx, ty))
    }
}

/// 分块地形管理器
pub struct TiledTerrain {
    /// 配置
    config: TileConfig,
    /// 数据源
    source: Arc<dyn TileSource>,
    /// 瓦片缓存
    cache: HashMap<(usize, usize), Tile>,
    /// 缓存访问顺序（用于 LRU）
    cache_order: Vec<(usize, usize)>,
}

impl TiledTerrain {
    /// 创建新的分块地形
    pub fn new(config: TileConfig, source: Arc<dyn TileSource>) -> Self {
        Self {
            config,
            source,
            cache: HashMap::new(),
            cache_order: Vec::new(),
        }
    }

    /// 获取配置
    pub fn config(&self) -> &TileConfig {
        &self.config
    }

    /// 获取单点高程
    pub fn get_elevation(&mut self, x: f64, y: f64) -> Option<f64> {
        let (tx, ty) = self.config.tile_coords(x, y)?;
        let (lx, ly) = self.config.local_coords(x, y);

        let tile = self.get_tile(tx, ty)?;
        tile.interpolate(lx, ly)
    }

    /// 批量获取高程
    pub fn get_elevations(&mut self, points: &[(f64, f64)]) -> Vec<Option<f64>> {
        points
            .iter()
            .map(|&(x, y)| self.get_elevation(x, y))
            .collect()
    }

    /// 获取瓦片（带缓存）
    fn get_tile(&mut self, tx: usize, ty: usize) -> Option<&Tile> {
        // 检查缓存
        if !self.cache.contains_key(&(tx, ty)) {
            // 加载瓦片
            match self.source.load_tile(tx, ty) {
                Ok(tile) => {
                    self.insert_cache(tx, ty, tile);
                }
                Err(_) => return None,
            }
        } else {
            // 更新 LRU 顺序
            self.update_cache_order(tx, ty);
        }

        self.cache.get(&(tx, ty))
    }

    /// 插入缓存
    fn insert_cache(&mut self, tx: usize, ty: usize, tile: Tile) {
        // 检查缓存大小
        while self.cache.len() >= self.config.max_cache_tiles {
            self.evict_oldest();
        }

        self.cache.insert((tx, ty), tile);
        self.cache_order.push((tx, ty));
    }

    /// 更新 LRU 顺序
    fn update_cache_order(&mut self, tx: usize, ty: usize) {
        if let Some(pos) = self.cache_order.iter().position(|&c| c == (tx, ty)) {
            self.cache_order.remove(pos);
            self.cache_order.push((tx, ty));
        }
    }

    /// 淘汰最旧的缓存
    fn evict_oldest(&mut self) {
        if let Some(oldest) = self.cache_order.first().copied() {
            self.cache.remove(&oldest);
            self.cache_order.remove(0);
        }
    }

    /// 清除缓存
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_order.clear();
    }

    /// 预加载指定范围的瓦片
    pub fn preload_region(&mut self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) {
        if let (Some((tx0, ty0)), Some((tx1, ty1))) = (
            self.config.tile_coords(min_x, min_y),
            self.config.tile_coords(max_x, max_y),
        ) {
            for tx in tx0..=tx1 {
                for ty in ty0..=ty1 {
                    let _ = self.get_tile(tx, ty);
                }
            }
        }
    }

    /// 获取缓存状态
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            cached_tiles: self.cache.len(),
            max_tiles: self.config.max_cache_tiles,
            cache_memory_bytes: self.cache.values().map(|t| t.data.len() * 8).sum(),
        }
    }

    /// 获取覆盖范围
    pub fn bounds(&self) -> [f64; 4] {
        self.config.bounds()
    }
}

/// 缓存统计
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// 已缓存瓦片数
    pub cached_tiles: usize,
    /// 最大瓦片数
    pub max_tiles: usize,
    /// 缓存内存占用（字节）
    pub cache_memory_bytes: usize,
}

/// LOD 级别配置
#[derive(Debug, Clone)]
pub struct LodLevel {
    /// 级别（0 = 最详细）
    pub level: usize,
    /// 分辨率
    pub resolution: f64,
    /// 瓦片尺寸
    pub tile_size: usize,
    /// 瓦片数量
    pub num_tiles: (usize, usize),
}

/// 多级细节地形管理器
pub struct MultiLodTerrain {
    /// 各级别地形
    levels: Vec<(LodLevel, TiledTerrain)>,
}

impl MultiLodTerrain {
    /// 创建多级细节地形
    pub fn new() -> Self {
        Self { levels: Vec::new() }
    }

    /// 添加 LOD 级别
    pub fn add_level(
        &mut self,
        level: LodLevel,
        source: Arc<dyn TileSource>,
    ) {
        let config = TileConfig {
            tile_size: level.tile_size,
            resolution: level.resolution,
            origin: (0.0, 0.0),
            num_tiles: level.num_tiles,
            nodata: -9999.0,
            max_cache_tiles: 50,
        };
        let terrain = TiledTerrain::new(config, source);
        self.levels.push((level, terrain));
        self.levels.sort_by_key(|(l, _)| l.level);
    }

    /// 根据请求分辨率选择合适的 LOD 级别
    pub fn select_level(&self, target_resolution: f64) -> Option<usize> {
        for (level, _) in &self.levels {
            if level.resolution <= target_resolution {
                return Some(level.level);
            }
        }
        self.levels.last().map(|(l, _)| l.level)
    }

    /// 获取高程（自动选择 LOD）
    pub fn get_elevation(&mut self, x: f64, y: f64, target_resolution: f64) -> Option<f64> {
        let level_idx = self.select_level(target_resolution)?;
        let (_, terrain) = self.levels.get_mut(level_idx)?;
        terrain.get_elevation(x, y)
    }
}

impl Default for MultiLodTerrain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_coords() {
        let config = TileConfig {
            tile_size: 256,
            resolution: 10.0,
            origin: (0.0, 0.0),
            num_tiles: (10, 10),
            ..Default::default()
        };

        // 第一个瓦片
        assert_eq!(config.tile_coords(100.0, 100.0), Some((0, 0)));

        // 第二个瓦片
        assert_eq!(config.tile_coords(2600.0, 100.0), Some((1, 0)));

        // 超出范围
        assert_eq!(config.tile_coords(-100.0, 100.0), None);
    }

    #[test]
    fn test_tile_interpolation() {
        let mut tile = Tile::empty((0, 0), 3, -9999.0);
        tile.set(0, 0, 0.0);
        tile.set(1, 0, 10.0);
        tile.set(0, 1, 10.0);
        tile.set(1, 1, 20.0);

        // 中心插值
        let z = tile.interpolate(0.5, 0.5).unwrap();
        assert!((z - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_tiled_terrain() {
        let mut source = MemoryTileSource::new();

        let mut tile = Tile::empty((0, 0), 10, -9999.0);
        for row in 0..10 {
            for col in 0..10 {
                tile.set(col, row, (col + row) as f64);
            }
        }
        source.add_tile(tile);

        let config = TileConfig {
            tile_size: 10,
            resolution: 1.0,
            origin: (0.0, 0.0),
            num_tiles: (1, 1),
            max_cache_tiles: 10,
            ..Default::default()
        };

        let mut terrain = TiledTerrain::new(config, Arc::new(source));

        let z = terrain.get_elevation(0.5, 0.5);
        assert!(z.is_some());
    }

    #[test]
    fn test_cache_eviction() {
        let mut source = MemoryTileSource::new();
        for tx in 0..5 {
            for ty in 0..5 {
                source.add_tile(Tile::empty((tx, ty), 10, -9999.0));
            }
        }

        let config = TileConfig {
            tile_size: 10,
            resolution: 1.0,
            origin: (0.0, 0.0),
            num_tiles: (5, 5),
            max_cache_tiles: 3,
            ..Default::default()
        };

        let mut terrain = TiledTerrain::new(config, Arc::new(source));

        // 加载超过缓存限制的瓦片
        for tx in 0..5 {
            let _ = terrain.get_elevation((tx as f64 * 10.0) + 5.0, 5.0);
        }

        assert!(terrain.cache.len() <= 3);
    }
}
