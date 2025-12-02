// marihydro\crates\mh_mesh\src/attributes.rs

//! 网格属性系统
//!
//! 提供向网格元素附加物理场数据的能力。
//! 支持单元/节点/边上的标量场和向量场存储。

use mh_foundation::error::{MhError, MhResult};
use mh_geo::geometry::Point3D;
use std::collections::HashMap;

// ============================================================================
// 预定义物理场名称常量
// ============================================================================

/// 曼宁糙率系数 (n)
pub const ATTR_MANNING_N: &str = "manning_n";
/// 河床高程 (z_bed)
pub const ATTR_BED_ELEVATION: &str = "z_bed";
/// 水深 (h)
pub const ATTR_WATER_DEPTH: &str = "h";
/// x方向流速 (u)
pub const ATTR_VELOCITY_X: &str = "u";
/// y方向流速 (v)
pub const ATTR_VELOCITY_Y: &str = "v";
/// 水面高程 (eta = z_bed + h)
pub const ATTR_WATER_SURFACE: &str = "eta";
/// x方向单位宽度流量 (qx = h * u)
pub const ATTR_DISCHARGE_X: &str = "qx";
/// y方向单位宽度流量 (qy = h * v)
pub const ATTR_DISCHARGE_Y: &str = "qy";

// ============================================================================
// 属性存储
// ============================================================================

/// 属性存储 - 支持任意类型的物理场
/// 
/// 为网格元素(单元、节点、面)提供标量场和向量场存储。
/// 
/// # 设计原则
/// - 类型安全：标量用 `f64`，向量用 `Point3D`
/// - 维度检查：写入时验证数组长度
/// - 物理场命名：使用预定义常量避免拼写错误
/// 
/// # 示例
/// ```ignore
/// let mut store = AttributeStore::new(100, 50, 150);
/// 
/// // 设置曼宁系数
/// store.set_cell_scalar(ATTR_MANNING_N, vec![0.03; 100])?;
/// 
/// // 获取曼宁系数
/// if let Some(manning) = store.get_cell_scalar(ATTR_MANNING_N) {
///     println!("Cell 0 Manning: {}", manning[0]);
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct AttributeStore {
    /// 期望的单元数量
    expected_cell_count: usize,
    /// 期望的节点数量
    expected_node_count: usize,
    /// 期望的面(边)数量
    expected_face_count: usize,

    /// 单元标量属性 (曼宁系数、底高程等)
    cell_scalars: HashMap<String, Vec<f64>>,
    /// 单元向量属性
    cell_vectors: HashMap<String, Vec<Point3D>>,

    /// 节点标量属性 (边界条件等)
    node_scalars: HashMap<String, Vec<f64>>,
    /// 节点向量属性
    node_vectors: HashMap<String, Vec<Point3D>>,

    /// 面(边)标量属性 (通量等)
    face_scalars: HashMap<String, Vec<f64>>,
    /// 面(边)向量属性
    face_vectors: HashMap<String, Vec<Point3D>>,
}

impl AttributeStore {
    /// 创建新的属性存储
    /// 
    /// # 参数
    /// - `cell_count`: 网格单元数量
    /// - `node_count`: 网格节点数量
    /// - `face_count`: 网格面(边)数量
    pub fn new(cell_count: usize, node_count: usize, face_count: usize) -> Self {
        Self {
            expected_cell_count: cell_count,
            expected_node_count: node_count,
            expected_face_count: face_count,
            cell_scalars: HashMap::new(),
            cell_vectors: HashMap::new(),
            node_scalars: HashMap::new(),
            node_vectors: HashMap::new(),
            face_scalars: HashMap::new(),
            face_vectors: HashMap::new(),
        }
    }

    /// 获取单元数量
    pub fn cell_count(&self) -> usize {
        self.expected_cell_count
    }

    /// 获取节点数量
    pub fn node_count(&self) -> usize {
        self.expected_node_count
    }

    /// 获取面数量
    pub fn face_count(&self) -> usize {
        self.expected_face_count
    }

    // ========================================================================
    // 单元属性
    // ========================================================================

    /// 设置单元标量场
    /// 
    /// # 错误
    /// 如果 `values.len() != cell_count` 返回维度不匹配错误
    pub fn set_cell_scalar(&mut self, name: &str, values: Vec<f64>) -> MhResult<()> {
        if values.len() != self.expected_cell_count {
            return Err(MhError::SizeMismatch {
                name: "cell_scalar",
                expected: self.expected_cell_count,
                actual: values.len(),
            });
        }
        self.cell_scalars.insert(name.to_string(), values);
        Ok(())
    }

    /// 获取单元标量场（不可变）
    pub fn get_cell_scalar(&self, name: &str) -> Option<&[f64]> {
        self.cell_scalars.get(name).map(|v| v.as_slice())
    }

    /// 获取单元标量场（可变）
    pub fn get_cell_scalar_mut(&mut self, name: &str) -> Option<&mut [f64]> {
        self.cell_scalars.get_mut(name).map(|v| v.as_mut_slice())
    }

    /// 设置单元向量场
    pub fn set_cell_vector(&mut self, name: &str, values: Vec<Point3D>) -> MhResult<()> {
        if values.len() != self.expected_cell_count {
            return Err(MhError::SizeMismatch {
                name: "cell_vector",
                expected: self.expected_cell_count,
                actual: values.len(),
            });
        }
        self.cell_vectors.insert(name.to_string(), values);
        Ok(())
    }

    /// 获取单元向量场（不可变）
    pub fn get_cell_vector(&self, name: &str) -> Option<&[Point3D]> {
        self.cell_vectors.get(name).map(|v| v.as_slice())
    }

    /// 获取单元向量场（可变）
    pub fn get_cell_vector_mut(&mut self, name: &str) -> Option<&mut [Point3D]> {
        self.cell_vectors.get_mut(name).map(|v| v.as_mut_slice())
    }

    /// 检查单元标量场是否存在
    pub fn has_cell_scalar(&self, name: &str) -> bool {
        self.cell_scalars.contains_key(name)
    }

    /// 检查单元向量场是否存在
    pub fn has_cell_vector(&self, name: &str) -> bool {
        self.cell_vectors.contains_key(name)
    }

    /// 删除单元标量场
    pub fn remove_cell_scalar(&mut self, name: &str) -> Option<Vec<f64>> {
        self.cell_scalars.remove(name)
    }

    /// 删除单元向量场
    pub fn remove_cell_vector(&mut self, name: &str) -> Option<Vec<Point3D>> {
        self.cell_vectors.remove(name)
    }

    /// 列出所有单元标量场名称
    pub fn cell_scalar_names(&self) -> impl Iterator<Item = &str> {
        self.cell_scalars.keys().map(|s| s.as_str())
    }

    /// 列出所有单元向量场名称
    pub fn cell_vector_names(&self) -> impl Iterator<Item = &str> {
        self.cell_vectors.keys().map(|s| s.as_str())
    }

    // ========================================================================
    // 节点属性
    // ========================================================================

    /// 设置节点标量场
    pub fn set_node_scalar(&mut self, name: &str, values: Vec<f64>) -> MhResult<()> {
        if values.len() != self.expected_node_count {
            return Err(MhError::SizeMismatch {
                name: "node_scalar",
                expected: self.expected_node_count,
                actual: values.len(),
            });
        }
        self.node_scalars.insert(name.to_string(), values);
        Ok(())
    }

    /// 获取节点标量场（不可变）
    pub fn get_node_scalar(&self, name: &str) -> Option<&[f64]> {
        self.node_scalars.get(name).map(|v| v.as_slice())
    }

    /// 获取节点标量场（可变）
    pub fn get_node_scalar_mut(&mut self, name: &str) -> Option<&mut [f64]> {
        self.node_scalars.get_mut(name).map(|v| v.as_mut_slice())
    }

    /// 设置节点向量场
    pub fn set_node_vector(&mut self, name: &str, values: Vec<Point3D>) -> MhResult<()> {
        if values.len() != self.expected_node_count {
            return Err(MhError::SizeMismatch {
                name: "node_vector",
                expected: self.expected_node_count,
                actual: values.len(),
            });
        }
        self.node_vectors.insert(name.to_string(), values);
        Ok(())
    }

    /// 获取节点向量场（不可变）
    pub fn get_node_vector(&self, name: &str) -> Option<&[Point3D]> {
        self.node_vectors.get(name).map(|v| v.as_slice())
    }

    /// 获取节点向量场（可变）
    pub fn get_node_vector_mut(&mut self, name: &str) -> Option<&mut [Point3D]> {
        self.node_vectors.get_mut(name).map(|v| v.as_mut_slice())
    }

    /// 检查节点标量场是否存在
    pub fn has_node_scalar(&self, name: &str) -> bool {
        self.node_scalars.contains_key(name)
    }

    /// 检查节点向量场是否存在
    pub fn has_node_vector(&self, name: &str) -> bool {
        self.node_vectors.contains_key(name)
    }

    /// 列出所有节点标量场名称
    pub fn node_scalar_names(&self) -> impl Iterator<Item = &str> {
        self.node_scalars.keys().map(|s| s.as_str())
    }

    // ========================================================================
    // 面(边)属性
    // ========================================================================

    /// 设置面标量场
    pub fn set_face_scalar(&mut self, name: &str, values: Vec<f64>) -> MhResult<()> {
        if values.len() != self.expected_face_count {
            return Err(MhError::SizeMismatch {
                name: "face_scalar",
                expected: self.expected_face_count,
                actual: values.len(),
            });
        }
        self.face_scalars.insert(name.to_string(), values);
        Ok(())
    }

    /// 获取面标量场（不可变）
    pub fn get_face_scalar(&self, name: &str) -> Option<&[f64]> {
        self.face_scalars.get(name).map(|v| v.as_slice())
    }

    /// 获取面标量场（可变）
    pub fn get_face_scalar_mut(&mut self, name: &str) -> Option<&mut [f64]> {
        self.face_scalars.get_mut(name).map(|v| v.as_mut_slice())
    }

    /// 设置面向量场
    pub fn set_face_vector(&mut self, name: &str, values: Vec<Point3D>) -> MhResult<()> {
        if values.len() != self.expected_face_count {
            return Err(MhError::SizeMismatch {
                name: "face_vector",
                expected: self.expected_face_count,
                actual: values.len(),
            });
        }
        self.face_vectors.insert(name.to_string(), values);
        Ok(())
    }

    /// 获取面向量场（不可变）
    pub fn get_face_vector(&self, name: &str) -> Option<&[Point3D]> {
        self.face_vectors.get(name).map(|v| v.as_slice())
    }

    /// 获取面向量场（可变）
    pub fn get_face_vector_mut(&mut self, name: &str) -> Option<&mut [Point3D]> {
        self.face_vectors.get_mut(name).map(|v| v.as_mut_slice())
    }

    /// 检查面标量场是否存在
    pub fn has_face_scalar(&self, name: &str) -> bool {
        self.face_scalars.contains_key(name)
    }

    /// 检查面向量场是否存在
    pub fn has_face_vector(&self, name: &str) -> bool {
        self.face_vectors.contains_key(name)
    }

    /// 列出所有面标量场名称
    pub fn face_scalar_names(&self) -> impl Iterator<Item = &str> {
        self.face_scalars.keys().map(|s| s.as_str())
    }

    // ========================================================================
    // 批量操作
    // ========================================================================

    /// 清空所有属性
    pub fn clear(&mut self) {
        self.cell_scalars.clear();
        self.cell_vectors.clear();
        self.node_scalars.clear();
        self.node_vectors.clear();
        self.face_scalars.clear();
        self.face_vectors.clear();
    }

    /// 更新维度（当网格拓扑改变时调用）
    /// 
    /// 注意：这会清空所有现有属性
    pub fn resize(&mut self, cell_count: usize, node_count: usize, face_count: usize) {
        self.expected_cell_count = cell_count;
        self.expected_node_count = node_count;
        self.expected_face_count = face_count;
        self.clear();
    }

    /// 获取属性统计信息
    pub fn stats(&self) -> AttributeStats {
        AttributeStats {
            cell_count: self.expected_cell_count,
            node_count: self.expected_node_count,
            face_count: self.expected_face_count,
            cell_scalar_count: self.cell_scalars.len(),
            cell_vector_count: self.cell_vectors.len(),
            node_scalar_count: self.node_scalars.len(),
            node_vector_count: self.node_vectors.len(),
            face_scalar_count: self.face_scalars.len(),
            face_vector_count: self.face_vectors.len(),
        }
    }
}

/// 属性统计信息
#[derive(Debug, Clone, Copy)]
pub struct AttributeStats {
    /// 单元数量
    pub cell_count: usize,
    /// 节点数量
    pub node_count: usize,
    /// 面数量
    pub face_count: usize,
    /// 单元标量场数量
    pub cell_scalar_count: usize,
    /// 单元向量场数量
    pub cell_vector_count: usize,
    /// 节点标量场数量
    pub node_scalar_count: usize,
    /// 节点向量场数量
    pub node_vector_count: usize,
    /// 面标量场数量
    pub face_scalar_count: usize,
    /// 面向量场数量
    pub face_vector_count: usize,
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_store_basic() {
        let mut store = AttributeStore::new(10, 5, 15);
        
        assert_eq!(store.cell_count(), 10);
        assert_eq!(store.node_count(), 5);
        assert_eq!(store.face_count(), 15);
    }

    #[test]
    fn test_cell_scalar() {
        let mut store = AttributeStore::new(3, 2, 4);
        
        // 设置曼宁系数
        store.set_cell_scalar(ATTR_MANNING_N, vec![0.03, 0.04, 0.05]).unwrap();
        
        // 验证存在
        assert!(store.has_cell_scalar(ATTR_MANNING_N));
        
        // 读取
        let manning = store.get_cell_scalar(ATTR_MANNING_N).unwrap();
        assert_eq!(manning.len(), 3);
        assert!((manning[0] - 0.03).abs() < 1e-10);
        
        // 修改
        let manning_mut = store.get_cell_scalar_mut(ATTR_MANNING_N).unwrap();
        manning_mut[0] = 0.025;
        
        let manning = store.get_cell_scalar(ATTR_MANNING_N).unwrap();
        assert!((manning[0] - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut store = AttributeStore::new(5, 3, 8);
        
        // 错误维度
        let result = store.set_cell_scalar("test", vec![1.0, 2.0]); // 期望5个，提供2个
        assert!(result.is_err());
        
        // 正确维度
        let result = store.set_cell_scalar("test", vec![1.0; 5]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cell_vector() {
        let mut store = AttributeStore::new(2, 1, 3);
        
        let velocities = vec![
            Point3D::new(1.0, 0.5, 0.0),
            Point3D::new(2.0, 1.0, 0.0),
        ];
        store.set_cell_vector("velocity", velocities).unwrap();
        
        let vel = store.get_cell_vector("velocity").unwrap();
        assert_eq!(vel.len(), 2);
        assert!((vel[0].x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_node_scalar() {
        let mut store = AttributeStore::new(10, 4, 12);
        
        store.set_node_scalar("boundary_h", vec![1.0, 1.5, 2.0, 2.5]).unwrap();
        
        let h = store.get_node_scalar("boundary_h").unwrap();
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn test_face_scalar() {
        let mut store = AttributeStore::new(6, 4, 8);
        
        store.set_face_scalar("flux", vec![0.0; 8]).unwrap();
        
        assert!(store.has_face_scalar("flux"));
        assert!(!store.has_face_scalar("nonexistent"));
    }

    #[test]
    fn test_clear_and_resize() {
        let mut store = AttributeStore::new(5, 3, 7);
        store.set_cell_scalar("test", vec![1.0; 5]).unwrap();
        
        assert!(store.has_cell_scalar("test"));
        
        store.resize(10, 6, 14);
        
        assert!(!store.has_cell_scalar("test")); // 清空了
        assert_eq!(store.cell_count(), 10);
    }

    #[test]
    fn test_list_names() {
        let mut store = AttributeStore::new(5, 3, 7);
        store.set_cell_scalar("a", vec![0.0; 5]).unwrap();
        store.set_cell_scalar("b", vec![0.0; 5]).unwrap();
        
        let names: Vec<_> = store.cell_scalar_names().collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
    }

    #[test]
    fn test_remove() {
        let mut store = AttributeStore::new(3, 2, 4);
        store.set_cell_scalar("temp", vec![1.0, 2.0, 3.0]).unwrap();
        
        let removed = store.remove_cell_scalar("temp");
        assert!(removed.is_some());
        assert!(!store.has_cell_scalar("temp"));
        
        let removed_again = store.remove_cell_scalar("temp");
        assert!(removed_again.is_none());
    }

    #[test]
    fn test_stats() {
        let mut store = AttributeStore::new(10, 5, 15);
        store.set_cell_scalar("a", vec![0.0; 10]).unwrap();
        store.set_cell_scalar("b", vec![0.0; 10]).unwrap();
        store.set_node_scalar("c", vec![0.0; 5]).unwrap();
        
        let stats = store.stats();
        assert_eq!(stats.cell_count, 10);
        assert_eq!(stats.cell_scalar_count, 2);
        assert_eq!(stats.node_scalar_count, 1);
        assert_eq!(stats.face_scalar_count, 0);
    }
}