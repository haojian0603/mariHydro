//! GPU边界条件处理
//!
//! 提供边界条件数据准备和GPU缓冲区管理

use std::collections::HashMap;

#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};

use crate::marihydro::core::error::MhResult;

/// 边界条件类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum BoundaryType {
    /// 固壁边界（反射）
    Wall = 0,
    /// 开边界（自由出流）
    Open = 1,
    /// 水位边界
    WaterLevel = 2,
    /// 流量边界
    Discharge = 3,
    /// 速度边界
    Velocity = 4,
    /// 辐射边界
    Radiation = 5,
    /// 周期边界
    Periodic = 6,
    /// 吸收边界
    Absorbing = 7,
}

impl From<u32> for BoundaryType {
    fn from(value: u32) -> Self {
        match value {
            0 => BoundaryType::Wall,
            1 => BoundaryType::Open,
            2 => BoundaryType::WaterLevel,
            3 => BoundaryType::Discharge,
            4 => BoundaryType::Velocity,
            5 => BoundaryType::Radiation,
            6 => BoundaryType::Periodic,
            7 => BoundaryType::Absorbing,
            _ => BoundaryType::Wall,
        }
    }
}

/// 边界条件值（GPU格式）
#[cfg_attr(feature = "gpu", derive(Pod, Zeroable))]
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct BoundaryValue {
    /// 值1（根据类型不同含义不同）
    pub v0: f32,
    /// 值2
    pub v1: f32,
    /// 值3
    pub v2: f32,
    /// 值4
    pub v3: f32,
}

impl BoundaryValue {
    /// 创建水位边界值
    pub fn water_level(eta: f64) -> Self {
        Self {
            v0: eta as f32,
            v1: 0.0,
            v2: 0.0,
            v3: 0.0,
        }
    }
    
    /// 创建流量边界值
    pub fn discharge(q: f64) -> Self {
        Self {
            v0: q as f32,
            v1: 0.0,
            v2: 0.0,
            v3: 0.0,
        }
    }
    
    /// 创建速度边界值
    pub fn velocity(u: f64, v: f64) -> Self {
        Self {
            v0: u as f32,
            v1: v as f32,
            v2: 0.0,
            v3: 0.0,
        }
    }
    
    /// 创建吸收边界值
    pub fn absorbing(target_h: f64, damping: f64) -> Self {
        Self {
            v0: target_h as f32,
            v1: damping as f32,
            v2: 0.0,
            v3: 0.0,
        }
    }
}

/// 潮汐分潮
#[cfg_attr(feature = "gpu", derive(Pod, Zeroable))]
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct TidalConstituent {
    /// 振幅 [m]
    pub amplitude: f32,
    /// 相位 [rad]
    pub phase: f32,
    /// 频率 [rad/s]
    pub frequency: f32,
    /// 填充
    pub _pad: f32,
}

impl TidalConstituent {
    /// M2分潮（主太阴半日潮）
    pub fn m2(amplitude: f64, phase_deg: f64) -> Self {
        Self {
            amplitude: amplitude as f32,
            phase: (phase_deg * std::f64::consts::PI / 180.0) as f32,
            frequency: 1.405189e-4_f32, // rad/s
            _pad: 0.0,
        }
    }
    
    /// S2分潮（主太阳半日潮）
    pub fn s2(amplitude: f64, phase_deg: f64) -> Self {
        Self {
            amplitude: amplitude as f32,
            phase: (phase_deg * std::f64::consts::PI / 180.0) as f32,
            frequency: 1.454441e-4_f32,
            _pad: 0.0,
        }
    }
    
    /// K1分潮（日月赤纬日潮）
    pub fn k1(amplitude: f64, phase_deg: f64) -> Self {
        Self {
            amplitude: amplitude as f32,
            phase: (phase_deg * std::f64::consts::PI / 180.0) as f32,
            frequency: 7.292117e-5_f32,
            _pad: 0.0,
        }
    }
    
    /// O1分潮（主太阴日潮）
    pub fn o1(amplitude: f64, phase_deg: f64) -> Self {
        Self {
            amplitude: amplitude as f32,
            phase: (phase_deg * std::f64::consts::PI / 180.0) as f32,
            frequency: 6.759775e-5_f32,
            _pad: 0.0,
        }
    }
}

/// 边界面数据
#[derive(Debug, Clone)]
pub struct BoundaryFace {
    /// 面ID
    pub face_id: u32,
    /// 边界类型
    pub bc_type: BoundaryType,
    /// 边界值
    pub value: BoundaryValue,
}

/// 潮汐边界数据
#[derive(Debug, Clone)]
pub struct TidalBoundary {
    /// 面ID
    pub face_id: u32,
    /// 平均水位
    pub mean_level: f64,
    /// 分潮列表
    pub constituents: Vec<TidalConstituent>,
}

/// GPU边界条件管理器
#[derive(Debug, Default)]
pub struct GpuBoundaryManager {
    /// 边界面列表
    boundary_faces: Vec<BoundaryFace>,
    /// 潮汐边界列表
    tidal_boundaries: Vec<TidalBoundary>,
    /// 边界标签映射
    label_map: HashMap<String, Vec<u32>>,
}

impl GpuBoundaryManager {
    /// 创建新的边界管理器
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 添加边界面
    pub fn add_boundary(&mut self, face: BoundaryFace) {
        self.boundary_faces.push(face);
    }
    
    /// 添加固壁边界
    pub fn add_wall(&mut self, face_ids: &[u32]) {
        for &id in face_ids {
            self.boundary_faces.push(BoundaryFace {
                face_id: id,
                bc_type: BoundaryType::Wall,
                value: BoundaryValue::default(),
            });
        }
    }
    
    /// 添加开边界
    pub fn add_open(&mut self, face_ids: &[u32]) {
        for &id in face_ids {
            self.boundary_faces.push(BoundaryFace {
                face_id: id,
                bc_type: BoundaryType::Open,
                value: BoundaryValue::default(),
            });
        }
    }
    
    /// 添加水位边界
    pub fn add_water_level(&mut self, face_ids: &[u32], eta: f64) {
        for &id in face_ids {
            self.boundary_faces.push(BoundaryFace {
                face_id: id,
                bc_type: BoundaryType::WaterLevel,
                value: BoundaryValue::water_level(eta),
            });
        }
    }
    
    /// 添加流量边界
    pub fn add_discharge(&mut self, face_ids: &[u32], q: f64) {
        for &id in face_ids {
            self.boundary_faces.push(BoundaryFace {
                face_id: id,
                bc_type: BoundaryType::Discharge,
                value: BoundaryValue::discharge(q),
            });
        }
    }
    
    /// 添加潮汐边界
    pub fn add_tidal(&mut self, face_id: u32, mean_level: f64, constituents: Vec<TidalConstituent>) {
        self.tidal_boundaries.push(TidalBoundary {
            face_id,
            mean_level,
            constituents,
        });
    }
    
    /// 设置边界标签
    pub fn set_label(&mut self, label: &str, face_ids: Vec<u32>) {
        self.label_map.insert(label.to_string(), face_ids);
    }
    
    /// 根据标签获取面ID
    pub fn get_faces_by_label(&self, label: &str) -> Option<&[u32]> {
        self.label_map.get(label).map(|v| v.as_slice())
    }
    
    /// 更新水位边界值
    pub fn update_water_level(&mut self, label: &str, eta: f64) {
        if let Some(face_ids) = self.label_map.get(label) {
            let face_set: std::collections::HashSet<_> = face_ids.iter().collect();
            for face in &mut self.boundary_faces {
                if face_set.contains(&face.face_id) && face.bc_type == BoundaryType::WaterLevel {
                    face.value = BoundaryValue::water_level(eta);
                }
            }
        }
    }
    
    /// 更新流量边界值
    pub fn update_discharge(&mut self, label: &str, q: f64) {
        if let Some(face_ids) = self.label_map.get(label) {
            let face_set: std::collections::HashSet<_> = face_ids.iter().collect();
            for face in &mut self.boundary_faces {
                if face_set.contains(&face.face_id) && face.bc_type == BoundaryType::Discharge {
                    face.value = BoundaryValue::discharge(q);
                }
            }
        }
    }
    
    /// 获取边界面数量
    pub fn num_boundary_faces(&self) -> usize {
        self.boundary_faces.len()
    }
    
    /// 获取潮汐边界数量
    pub fn num_tidal_boundaries(&self) -> usize {
        self.tidal_boundaries.len()
    }
    
    /// 导出GPU格式的面ID数组
    pub fn export_face_ids(&self) -> Vec<u32> {
        self.boundary_faces.iter().map(|f| f.face_id).collect()
    }
    
    /// 导出GPU格式的边界类型数组
    pub fn export_types(&self) -> Vec<u32> {
        self.boundary_faces.iter().map(|f| f.bc_type as u32).collect()
    }
    
    /// 导出GPU格式的边界值数组
    pub fn export_values(&self) -> Vec<BoundaryValue> {
        self.boundary_faces.iter().map(|f| f.value).collect()
    }
    
    /// 导出潮汐边界的面ID
    pub fn export_tidal_face_ids(&self) -> Vec<u32> {
        self.tidal_boundaries.iter().map(|t| t.face_id).collect()
    }
    
    /// 导出潮汐边界的平均水位
    pub fn export_tidal_mean_levels(&self) -> Vec<f32> {
        self.tidal_boundaries.iter().map(|t| t.mean_level as f32).collect()
    }
    
    /// 导出潮汐分潮（每个边界最多8个分潮）
    pub fn export_tidal_constituents(&self) -> (Vec<TidalConstituent>, Vec<u32>) {
        let mut constituents = Vec::new();
        let mut counts = Vec::new();
        
        for boundary in &self.tidal_boundaries {
            let n = boundary.constituents.len().min(8);
            counts.push(n as u32);
            
            // 填充8个分潮位置
            for i in 0..8 {
                if i < n {
                    constituents.push(boundary.constituents[i]);
                } else {
                    constituents.push(TidalConstituent::default());
                }
            }
        }
        
        (constituents, counts)
    }
    
    /// 清空所有边界
    pub fn clear(&mut self) {
        self.boundary_faces.clear();
        self.tidal_boundaries.clear();
        self.label_map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_boundary_manager() {
        let mut manager = GpuBoundaryManager::new();
        
        manager.add_wall(&[0, 1, 2]);
        manager.add_water_level(&[3, 4], 1.5);
        
        assert_eq!(manager.num_boundary_faces(), 5);
        
        let types = manager.export_types();
        assert_eq!(types[0], BoundaryType::Wall as u32);
        assert_eq!(types[3], BoundaryType::WaterLevel as u32);
    }
    
    #[test]
    fn test_tidal_constituents() {
        let m2 = TidalConstituent::m2(0.5, 30.0);
        let s2 = TidalConstituent::s2(0.2, 45.0);
        
        assert!((m2.amplitude - 0.5).abs() < 1e-6);
        assert!(m2.frequency > 0.0);
        assert!(s2.frequency > m2.frequency);
    }
    
    #[test]
    fn test_label_mapping() {
        let mut manager = GpuBoundaryManager::new();
        
        manager.add_water_level(&[10, 11, 12], 2.0);
        manager.set_label("inlet", vec![10, 11, 12]);
        
        assert!(manager.get_faces_by_label("inlet").is_some());
        assert_eq!(manager.get_faces_by_label("inlet").unwrap().len(), 3);
        
        manager.update_water_level("inlet", 3.0);
        let values = manager.export_values();
        assert!((values[0].v0 - 3.0).abs() < 1e-6);
    }
}
