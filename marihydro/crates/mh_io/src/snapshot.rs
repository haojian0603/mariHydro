// crates/mh_io/src/snapshot.rs

//! 网格和状态快照
//!
//! 用于异步 IO 传输的轻量级数据拷贝。
//!
//! # 设计说明
//!
//! 快照是网格和状态的只读副本，用于：
//! - 异步文件输出（避免阻塞计算线程）
//! - 检查点保存/恢复
//! - 跨模块数据传输
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use mh_io::snapshot::{MeshSnapshot, StateSnapshot};
//!
//! // 从网格创建快照
//! let mesh_snap = MeshSnapshot::from_mesh_data(
//!     n_nodes, n_cells, positions, cell_nodes, areas, elevations
//! );
//!
//! // 从状态创建快照
//! let state_snap = StateSnapshot::from_state_data(h, hu, hv);
//! ```

use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use mh_runtime::RuntimeScalar;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

// ============================================================
// 网格快照
// ============================================================

/// 网格快照（用于异步传输）
///
/// 包含网格几何和拓扑的只读副本，适用于：
/// - VTU 文件导出
/// - 检查点保存
/// - 可视化预览
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshSnapshot {
    /// 节点数
    pub n_nodes: usize,
    /// 单元数
    pub n_cells: usize,
    /// 节点坐标 (x, y)
    pub node_positions: Vec<(f64, f64)>,
    /// 单元节点索引
    pub cell_nodes: Vec<Vec<usize>>,
    /// 单元面积
    pub cell_areas: Vec<f64>,
    /// 床面高程
    pub bed_elevations: Vec<f64>,
    /// 边界面索引（可选）
    pub boundary_faces: Option<Vec<u32>>,
    /// 边界标识（可选，与边界面对应）
    pub boundary_ids: Option<Vec<u32>>,
    /// 边界名称列表（可选）
    pub boundary_names: Option<Vec<String>>,
    /// 元数据（可选）
    pub meta: Option<SnapshotMeta>,
}

/// 快照元数据
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SnapshotMeta {
    /// 创建时间戳（Unix 秒）
    pub created_at: u64,
    /// 坐标参考系 EPSG 代码
    pub crs_epsg: Option<u32>,
    /// 数据哈希（用于校验）
    pub hash: Option<u64>,
    /// 描述信息
    pub description: Option<String>,
}

impl MeshSnapshot {
    /// 创建空快照
    pub fn empty() -> Self {
        Self {
            n_nodes: 0,
            n_cells: 0,
            node_positions: Vec::new(),
            cell_nodes: Vec::new(),
            cell_areas: Vec::new(),
            bed_elevations: Vec::new(),
            boundary_faces: None,
            boundary_ids: None,
            boundary_names: None,
            meta: None,
        }
    }

    /// 从网格数据创建快照
    ///
    /// # 参数
    ///
    /// - `n_nodes`: 节点数量
    /// - `n_cells`: 单元数量
    /// - `node_positions`: 节点坐标列表
    /// - `cell_nodes`: 每个单元的节点索引
    /// - `cell_areas`: 单元面积
    /// - `bed_elevations`: 床面高程
    pub fn from_mesh_data(
        n_nodes: usize,
        n_cells: usize,
        node_positions: Vec<(f64, f64)>,
        cell_nodes: Vec<Vec<usize>>,
        cell_areas: Vec<f64>,
        bed_elevations: Vec<f64>,
    ) -> Self {
        Self {
            n_nodes,
            n_cells,
            node_positions,
            cell_nodes,
            cell_areas,
            bed_elevations,
            boundary_faces: None,
            boundary_ids: None,
            boundary_names: None,
            meta: None,
        }
    }

    /// 添加边界数据
    pub fn with_boundaries(
        mut self,
        faces: Vec<u32>,
        ids: Vec<u32>,
        names: Vec<String>,
    ) -> Self {
        self.boundary_faces = Some(faces);
        self.boundary_ids = Some(ids);
        self.boundary_names = Some(names);
        self
    }

    /// 设置底床高程（Builder 方式）
    ///
    /// 用于在空快照上单独设置底床高程，而非在 `from_mesh_data` 中传入。
    /// 确保 API 的对称性和灵活性。
    ///
    /// # 参数
    /// - `elevations`: 每个单元的底床高程数组
    ///
    /// # 示例
    /// ```ignore
    /// let snapshot = MeshSnapshot::empty()
    ///     .with_bed_elevations(vec![0.0; n_cells]);
    /// ```
    pub fn with_bed_elevations(mut self, elevations: Vec<f64>) -> Self {
        self.bed_elevations = elevations;
        self
    }

    /// 添加元数据
    pub fn with_meta(mut self, meta: SnapshotMeta) -> Self {
        self.meta = Some(meta);
        self
    }

    /// 设置 CRS
    pub fn with_crs(mut self, epsg: u32) -> Self {
        let meta = self.meta.get_or_insert_with(SnapshotMeta::default);
        meta.crs_epsg = Some(epsg);
        self
    }

    /// 计算网格哈希（用于一致性校验）
    pub fn compute_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.n_nodes.hash(&mut hasher);
        self.n_cells.hash(&mut hasher);
        for &(x, y) in &self.node_positions {
            x.to_bits().hash(&mut hasher);
            y.to_bits().hash(&mut hasher);
        }
        for nodes in &self.cell_nodes {
            nodes.len().hash(&mut hasher);
            for &n in nodes {
                n.hash(&mut hasher);
            }
        }
        for &a in &self.cell_areas {
            a.to_bits().hash(&mut hasher);
        }
        for &z in &self.bed_elevations {
            z.to_bits().hash(&mut hasher);
        }
        if let Some(faces) = &self.boundary_faces {
            for &f in faces {
                f.hash(&mut hasher);
            }
        }
        if let Some(ids) = &self.boundary_ids {
            for &id in ids {
                id.hash(&mut hasher);
            }
        }
        if let Some(names) = &self.boundary_names {
            for name in names {
                name.hash(&mut hasher);
            }
        }
        if let Some(meta) = &self.meta {
            if let Some(epsg) = meta.crs_epsg {
                epsg.hash(&mut hasher);
            }
            if let Some(desc) = &meta.description {
                desc.hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// 写入哈希到元数据
    pub fn with_hash(mut self) -> Self {
        let hash = self.compute_hash();
        let meta = self.meta.get_or_insert_with(SnapshotMeta::default);
        meta.hash = Some(hash);
        self
    }

    /// 内存占用估计（字节）
    pub fn memory_usage(&self) -> usize {
        // 节点坐标: 2 * f64 = 16 bytes
        let nodes_mem = self.node_positions.len() * 16;
        // 单元节点索引: 每个 Vec 的元素 * 8 bytes
        let cell_nodes_mem: usize = self.cell_nodes.iter().map(|v| v.len() * 8).sum();
        // 面积和高程: 各 8 bytes
        let areas_mem = self.cell_areas.len() * 8;
        let elev_mem = self.bed_elevations.len() * 8;
        // 边界数据
        let boundary_mem = self.boundary_faces.as_ref().map_or(0, |v| v.len() * 4)
            + self.boundary_ids.as_ref().map_or(0, |v| v.len() * 4);

        nodes_mem + cell_nodes_mem + areas_mem + elev_mem + boundary_mem
    }

    /// 验证数据一致性
    pub fn validate(&self) -> Result<(), String> {
        if self.node_positions.len() != self.n_nodes {
            return Err(format!(
                "节点数不匹配: 期望 {}, 实际 {}",
                self.n_nodes,
                self.node_positions.len()
            ));
        }
        for (i, (x, y)) in self.node_positions.iter().enumerate() {
            if !x.is_finite() || !y.is_finite() {
                return Err(format!("节点 {} 坐标无效: ({}, {})", i, x, y));
            }
        }
        if self.cell_nodes.len() != self.n_cells {
            return Err(format!(
                "单元数不匹配: 期望 {}, 实际 {}",
                self.n_cells,
                self.cell_nodes.len()
            ));
        }
        if self.cell_areas.len() != self.n_cells {
            return Err(format!(
                "面积数组长度不匹配: 期望 {}, 实际 {}",
                self.n_cells,
                self.cell_areas.len()
            ));
        }
        for (i, &area) in self.cell_areas.iter().enumerate() {
            if !area.is_finite() || area < 0.0 {
                return Err(format!("单元 {} 面积无效: {}", i, area));
            }
        }
        if self.bed_elevations.len() != self.n_cells {
            return Err(format!(
                "高程数组长度不匹配: 期望 {}, 实际 {}",
                self.n_cells,
                self.bed_elevations.len()
            ));
        }
        for (i, &z) in self.bed_elevations.iter().enumerate() {
            if !z.is_finite() {
                return Err(format!("单元 {} 床面高程无效: {}", i, z));
            }
        }
        match (&self.boundary_faces, &self.boundary_ids) {
            (Some(faces), Some(ids)) => {
                if faces.len() != ids.len() {
                    return Err("boundary length mismatch".into());
                }
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err("boundary ids missing".into());
            }
            (None, None) => {}
        }
        if let (Some(names), Some(ids)) = (&self.boundary_names, &self.boundary_ids) {
            let unique_ids: std::collections::HashSet<_> = ids.iter().copied().collect();
            if names.len() != unique_ids.len() {
                return Err("boundary name mismatch".into());
            }
        }
        if let Some(meta) = &self.meta {
            if let Some(expected) = meta.hash {
                let actual = self.compute_hash();
                if actual != expected {
                    return Err("mesh hash mismatch".into());
                }
            }
        }
        // 检查节点索引是否越界
        for (i, nodes) in self.cell_nodes.iter().enumerate() {
            if nodes.len() < 3 {
                return Err(format!("单元 {} 节点数不足: {}", i, nodes.len()));
            }
            {
                let mut set = std::collections::HashSet::new();
                for &idx in nodes {
                    if !set.insert(idx) {
                        return Err(format!("单元 {} 存在重复节点索引 {}", i, idx));
                    }
                }
            }
            for &idx in nodes {
                if idx >= self.n_nodes {
                    return Err(format!(
                        "单元 {} 节点索引 {} 越界 (最大 {})",
                        i,
                        idx,
                        self.n_nodes - 1
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Default for MeshSnapshot {
    fn default() -> Self {
        Self::empty()
    }
}

// ============================================================
// 状态快照
// ============================================================

/// 状态快照（用于异步传输）
///
/// 包含浅水方程守恒变量的只读副本。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "S: Serialize", deserialize = "S: DeserializeOwned"))]
pub struct StateSnapshot<S: RuntimeScalar = f64> {
    /// 水深 [m]
    pub h: Vec<S>,
    /// x 动量 [m²/s]
    pub hu: Vec<S>,
    /// y 动量 [m²/s]
    pub hv: Vec<S>,
    /// 底床高程（可选，用于完整状态恢复）
    pub z: Option<Vec<S>>,
    /// 标量场（可选，如示踪剂浓度）
    pub scalars: Option<Vec<Vec<S>>>,
    /// 标量场名称（可选）
    pub scalar_names: Option<Vec<String>>,
    /// 元数据
    pub meta: Option<StateSnapshotMeta>,
}

/// 状态快照元数据
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateSnapshotMeta {
    /// 模拟时间 [s]
    pub time: f64,
    /// 时间步数
    pub step: usize,
    /// 创建时间戳
    pub created_at: u64,
    /// 数据哈希（用于校验）
    pub hash: Option<u64>,
}

impl<S: RuntimeScalar> StateSnapshot<S> {
    /// 创建空快照
    pub fn empty() -> Self {
        Self {
            h: Vec::new(),
            hu: Vec::new(),
            hv: Vec::new(),
            z: None,
            scalars: None,
            scalar_names: None,
            meta: None,
        }
    }

    /// 从状态数据创建快照
    pub fn from_state_data(h: Vec<S>, hu: Vec<S>, hv: Vec<S>) -> Self {
        Self {
            h,
            hu,
            hv,
            z: None,
            scalars: None,
            scalar_names: None,
            meta: None,
        }
    }

    /// 包含底床高程
    pub fn with_bed(mut self, z: Vec<S>) -> Self {
        self.z = Some(z);
        self
    }

    /// 添加标量场
    pub fn with_scalar(mut self, name: &str, values: Vec<S>) -> Result<Self, String> {
        if values.len() != self.n_cells() {
            return Err(format!(
                "标量场长度不匹配: name={}, 期望 {}, 实际 {}",
                name,
                self.n_cells(),
                values.len()
            ));
        }
        if self.scalars.is_none() {
            self.scalars = Some(Vec::new());
            self.scalar_names = Some(Vec::new());
        }
        self.scalars.as_mut().unwrap().push(values);
        self.scalar_names.as_mut().unwrap().push(name.to_string());
        Ok(self)
    }

    /// 添加元数据
    pub fn with_meta(mut self, time: f64, step: usize) -> Self {
        self.meta = Some(StateSnapshotMeta {
            time,
            step,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            hash: None,
        });
        self
    }

    /// 计算状态哈希（用于一致性校验）
    pub fn compute_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.h.len().hash(&mut hasher);
        for &v in &self.h {
            v.to_f64().unwrap_or(0.0).to_bits().hash(&mut hasher);
        }
        for &v in &self.hu {
            v.to_f64().unwrap_or(0.0).to_bits().hash(&mut hasher);
        }
        for &v in &self.hv {
            v.to_f64().unwrap_or(0.0).to_bits().hash(&mut hasher);
        }
        if let Some(z) = &self.z {
            for &v in z {
                v.to_f64().unwrap_or(0.0).to_bits().hash(&mut hasher);
            }
        }
        if let (Some(vals), Some(names)) = (&self.scalars, &self.scalar_names) {
            for name in names {
                name.hash(&mut hasher);
            }
            for scalar in vals {
                for &v in scalar {
                    v.to_f64().unwrap_or(0.0).to_bits().hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    /// 写入哈希到元数据
    pub fn with_hash(mut self) -> Self {
        let hash = self.compute_hash();
        let meta = self.meta.get_or_insert_with(StateSnapshotMeta::default);
        meta.hash = Some(hash);
        self
    }

    /// 单元数
    pub fn n_cells(&self) -> usize {
        self.h.len()
    }

    /// 内存占用估计（字节）
    pub fn memory_usage(&self) -> usize {
        let elem_size = std::mem::size_of::<S>();
        let base = (self.h.len() + self.hu.len() + self.hv.len()) * elem_size;
        let z_mem = self.z.as_ref().map_or(0, |v| v.len() * elem_size);
        let scalars_mem = self
            .scalars
            .as_ref()
            .map_or(0, |vecs| vecs.iter().map(|v| v.len() * elem_size).sum());
        base + z_mem + scalars_mem
    }

    /// 验证数据一致性
    pub fn validate(&self) -> Result<(), String> {
        let n = self.h.len();
        if self.hu.len() != n {
            return Err(format!("hu 长度不匹配: 期望 {}, 实际 {}", n, self.hu.len()));
        }
        if self.hv.len() != n {
            return Err(format!("hv 长度不匹配: 期望 {}, 实际 {}", n, self.hv.len()));
        }
        if let Some(z) = &self.z {
            if z.len() != n {
                return Err(format!("z 长度不匹配: 期望 {}, 实际 {}", n, z.len()));
            }
        }
        match (&self.scalars, &self.scalar_names) {
            (Some(vals), Some(names)) => {
                if vals.len() != names.len() {
                    return Err("scalar name mismatch".into());
                }
                {
                    let mut set = std::collections::HashSet::new();
                    for name in names {
                        if !set.insert(name) {
                            return Err("scalar name duplicate".into());
                        }
                    }
                }
                for v in vals {
                    if v.len() != n {
                        return Err("scalar length mismatch".into());
                    }
                }
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err("scalar names missing".into());
            }
            (None, None) => {}
        }
        if let Some(meta) = &self.meta {
            if let Some(expected) = meta.hash {
                let actual = self.compute_hash();
                if actual != expected {
                    return Err("state hash mismatch".into());
                }
            }
        }
        // 检查 NaN/Inf
        for (i, &val) in self.h.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!("h[{}] = {} 非有限值", i, val));
            }
            if val < S::ZERO {
                return Err(format!("h[{}] = {} 为负值", i, val));
            }
        }
        for (i, &val) in self.hu.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!("hu[{}] = {} 非有限值", i, val));
            }
        }
        for (i, &val) in self.hv.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!("hv[{}] = {} 非有限值", i, val));
            }
        }
        if let Some(z) = &self.z {
            for (i, &val) in z.iter().enumerate() {
                if !val.is_finite() {
                    return Err(format!("z[{}] = {} 非有限值", i, val));
                }
            }
        }
        if let Some(scalars) = &self.scalars {
            for (field_idx, values) in scalars.iter().enumerate() {
                for (i, &val) in values.iter().enumerate() {
                    if !val.is_finite() {
                        return Err(format!(
                            "scalar[{}][{}] = {} 非有限值",
                            field_idx, i, val
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// 计算统计信息
    pub fn statistics(&self) -> StateStatistics {
        let n = self.h.len();
        if n == 0 {
            return StateStatistics::default();
        }

        let h_sum: f64 = self.h.iter().map(|v| v.to_f64().unwrap_or(0.0)).sum();
        let h_min = self
            .h
            .iter()
            .map(|v| v.to_f64().unwrap_or(f64::INFINITY))
            .fold(f64::INFINITY, f64::min);
        let h_max = self
            .h
            .iter()
            .map(|v| v.to_f64().unwrap_or(f64::NEG_INFINITY))
            .fold(f64::NEG_INFINITY, f64::max);

        StateStatistics {
            n_cells: n,
            h_min,
            h_max,
            h_mean: h_sum / n as f64,
        }
    }
}

impl<S: RuntimeScalar> Default for StateSnapshot<S> {
    fn default() -> Self {
        Self::empty()
    }
}

/// 状态统计信息
#[derive(Debug, Clone, Default)]
pub struct StateStatistics {
    /// 单元数
    pub n_cells: usize,
    /// 最小水深
    pub h_min: f64,
    /// 最大水深
    pub h_max: f64,
    /// 平均水深
    pub h_mean: f64,
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_snapshot_creation() {
        let snapshot = MeshSnapshot::from_mesh_data(
            4,
            1,
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            vec![vec![0, 1, 2, 3]],
            vec![1.0],
            vec![0.0],
        );

        assert_eq!(snapshot.n_nodes, 4);
        assert_eq!(snapshot.n_cells, 1);
        assert!(snapshot.validate().is_ok());
    }

    #[test]
    fn test_mesh_snapshot_validation_error() {
        let mut snapshot = MeshSnapshot::from_mesh_data(
            4,
            1,
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            vec![vec![0, 1, 2, 10]], // 越界索引
            vec![1.0],
            vec![0.0],
        );

        assert!(snapshot.validate().is_err());

        // 修正错误
        snapshot.cell_nodes[0][3] = 3;
        assert!(snapshot.validate().is_ok());
    }

    #[test]
    fn test_state_snapshot_creation() {
        let snapshot = StateSnapshot::from_state_data(
            vec![1.0, 2.0, 3.0],
            vec![0.1, 0.2, 0.3],
            vec![0.0, 0.0, 0.0],
        );

        assert_eq!(snapshot.n_cells(), 3);
        assert!(snapshot.validate().is_ok());
    }

    #[test]
    fn test_state_snapshot_with_scalar() {
        let snapshot = StateSnapshot::from_state_data(
            vec![1.0, 2.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        )
        .with_scalar("temperature", vec![20.0, 21.0])
        .and_then(|s| s.with_scalar("salinity", vec![35.0, 34.5]))
        .expect("添加标量场失败");

        assert_eq!(snapshot.scalars.as_ref().unwrap().len(), 2);
        assert_eq!(
            snapshot.scalar_names.as_ref().unwrap(),
            &["temperature", "salinity"]
        );
    }

    #[test]
    fn test_state_statistics() {
        let snapshot = StateSnapshot::from_state_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0.0; 4],
            vec![0.0; 4],
        );

        let stats = snapshot.statistics();
        assert_eq!(stats.n_cells, 4);
        assert!((stats.h_min - 1.0).abs() < 1e-10);
        assert!((stats.h_max - 4.0).abs() < 1e-10);
        assert!((stats.h_mean - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_memory_usage() {
        let mesh_snap = MeshSnapshot::from_mesh_data(
            100,
            50,
            vec![(0.0, 0.0); 100],
            vec![vec![0, 1, 2]; 50],
            vec![1.0; 50],
            vec![0.0; 50],
        );

        // 内存估计应该大于 0
        assert!(mesh_snap.memory_usage() > 0);

        let state_snap = StateSnapshot::from_state_data(
            vec![1.0; 50],
            vec![0.0; 50],
            vec![0.0; 50],
        );

        assert!(state_snap.memory_usage() > 0);
    }
}
