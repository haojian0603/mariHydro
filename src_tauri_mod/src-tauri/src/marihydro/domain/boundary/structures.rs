// src-tauri/src/marihydro/domain/boundary/structures.rs
//! 水工结构物边界条件
//! 支持堰、闸门、泵站、涵洞等

use crate::marihydro::core::error::MhResult;
use serde::{Deserialize, Serialize};

/// 重力加速度
const G: f64 = 9.81;

/// 水工结构物特征接口
pub trait HydraulicStructure: Send + Sync {
    /// 结构物名称
    fn name(&self) -> &str;
    
    /// 计算过流流量
    /// 
    /// # Arguments
    /// * `h_up` - 上游水位 [m]
    /// * `h_down` - 下游水位 [m]
    /// * `time` - 当前时间 [s]
    /// 
    /// # Returns
    /// 流量 [m³/s]，正值为上游→下游
    fn compute_discharge(&mut self, h_up: f64, h_down: f64, time: f64) -> f64;
    
    /// 堰顶/底槛高程
    fn crest_level(&self) -> f64;
    
    /// 是否允许逆流
    fn allows_reverse_flow(&self) -> bool {
        true
    }
    
    /// 获取当前开度（用于闸门）
    fn current_opening(&self) -> Option<f64> {
        None
    }
    
    /// 设置开度（用于闸门控制）
    fn set_opening(&mut self, _opening: f64) -> MhResult<()> {
        Ok(())
    }
    
    /// 是否处于运行状态（用于泵站）
    fn is_operating(&self) -> bool {
        true
    }
}

/// 宽顶堰
/// 
/// 流量公式：Q = Cd × b × √(2g) × H^(3/2)
/// 其中 H 为堰上水头
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BroadCrestedWeir {
    /// 名称
    pub name: String,
    /// 堰宽 [m]
    pub width: f64,
    /// 堰顶高程 [m]
    pub crest_elevation: f64,
    /// 流量系数（典型值 0.32-0.38）
    pub discharge_coefficient: f64,
    /// 淹没系数修正（可选）
    pub submergence_limit: f64,
}

impl BroadCrestedWeir {
    /// 创建宽顶堰
    pub fn new(name: &str, width: f64, crest_elevation: f64) -> Self {
        Self {
            name: name.to_string(),
            width,
            crest_elevation,
            discharge_coefficient: 0.35,
            submergence_limit: 0.8,
        }
    }

    /// 设置流量系数
    pub fn with_coefficient(mut self, cd: f64) -> Self {
        self.discharge_coefficient = cd;
        self
    }

    /// 计算淹没系数
    fn submergence_factor(&self, h_up: f64, h_down: f64) -> f64 {
        let head_up = h_up - self.crest_elevation;
        let head_down = h_down - self.crest_elevation;
        
        if head_down <= 0.0 || head_up <= 0.0 {
            return 1.0;
        }
        
        let ratio = head_down / head_up;
        if ratio < self.submergence_limit {
            1.0
        } else {
            // Villemonte 公式
            (1.0 - ratio.powf(1.5)).powf(0.385)
        }
    }
}

impl HydraulicStructure for BroadCrestedWeir {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_discharge(&mut self, h_up: f64, h_down: f64, _time: f64) -> f64 {
        // 确定流向
        let (upstream, downstream, sign) = if h_up >= h_down {
            (h_up, h_down, 1.0)
        } else {
            (h_down, h_up, -1.0)
        };

        let head = upstream - self.crest_elevation;
        if head <= 0.0 {
            return 0.0;
        }

        let submergence = self.submergence_factor(upstream, downstream);
        let q = self.discharge_coefficient
            * self.width
            * (2.0 * G).sqrt()
            * head.powf(1.5)
            * submergence;

        sign * q
    }

    fn crest_level(&self) -> f64 {
        self.crest_elevation
    }
}

/// 闸门类型
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GateType {
    /// 平面闸门（底孔出流）
    Sluice,
    /// 弧形闸门
    Radial,
    /// 翻板闸门
    Flap,
}

/// 闸门
/// 
/// 自由出流：Q = Cd × b × a × √(2gH)
/// 淹没出流：Q = Cd × b × a × √(2g(H1-H2))
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gate {
    /// 名称
    pub name: String,
    /// 闸门类型
    pub gate_type: GateType,
    /// 闸门宽度 [m]
    pub width: f64,
    /// 底槛高程 [m]
    pub sill_elevation: f64,
    /// 当前开度 [m]
    pub opening: f64,
    /// 最大开度 [m]
    pub max_opening: f64,
    /// 流量系数（自由出流，典型 0.6）
    pub discharge_coefficient: f64,
    /// 收缩系数
    pub contraction_coefficient: f64,
}

impl Gate {
    /// 创建平面闸门
    pub fn sluice(name: &str, width: f64, sill_elevation: f64, max_opening: f64) -> Self {
        Self {
            name: name.to_string(),
            gate_type: GateType::Sluice,
            width,
            sill_elevation,
            opening: 0.0,
            max_opening,
            discharge_coefficient: 0.6,
            contraction_coefficient: 0.61,
        }
    }

    /// 设置流量系数
    pub fn with_coefficient(mut self, cd: f64) -> Self {
        self.discharge_coefficient = cd;
        self
    }
}

impl HydraulicStructure for Gate {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_discharge(&mut self, h_up: f64, h_down: f64, _time: f64) -> f64 {
        if self.opening <= 0.0 {
            return 0.0;
        }

        // 确定流向
        let (upstream, downstream, sign) = if h_up >= h_down {
            (h_up, h_down, 1.0)
        } else {
            (h_down, h_up, -1.0)
        };

        let h1 = upstream - self.sill_elevation;
        let h2 = downstream - self.sill_elevation;
        
        if h1 <= 0.0 {
            return 0.0;
        }

        let a = self.opening.min(h1);
        let cc = self.contraction_coefficient;
        
        // 判断出流状态
        let contracted_depth = cc * a;
        let is_submerged = h2 > contracted_depth;

        let q = if is_submerged {
            // 淹没出流
            let delta_h = (h1 - h2).max(0.0);
            self.discharge_coefficient * self.width * a * (2.0 * G * delta_h).sqrt()
        } else {
            // 自由出流
            self.discharge_coefficient * self.width * a * (2.0 * G * h1).sqrt()
        };

        sign * q
    }

    fn crest_level(&self) -> f64 {
        self.sill_elevation
    }

    fn current_opening(&self) -> Option<f64> {
        Some(self.opening)
    }

    fn set_opening(&mut self, opening: f64) -> MhResult<()> {
        self.opening = opening.clamp(0.0, self.max_opening);
        Ok(())
    }
}

/// 泵站运行模式
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PumpMode {
    /// 固定流量
    ConstantFlow,
    /// 按特性曲线
    Characteristic,
    /// 水位控制
    LevelControlled,
}

/// 泵站
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PumpStation {
    /// 名称
    pub name: String,
    /// 运行模式
    pub mode: PumpMode,
    /// 设计流量 [m³/s]
    pub design_discharge: f64,
    /// 进水侧（true=上游，false=下游）
    pub intake_upstream: bool,
    /// 启动水位 [m]
    pub start_level: f64,
    /// 停止水位 [m]
    pub stop_level: f64,
    /// 当前运行状态
    operating: bool,
    /// 当前流量
    current_discharge: f64,
}

impl PumpStation {
    /// 创建定流量泵站
    pub fn constant_flow(
        name: &str,
        discharge: f64,
        intake_upstream: bool,
        start_level: f64,
        stop_level: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            mode: PumpMode::ConstantFlow,
            design_discharge: discharge,
            intake_upstream,
            start_level,
            stop_level,
            operating: false,
            current_discharge: 0.0,
        }
    }

    /// 更新运行状态
    fn update_state(&mut self, intake_level: f64) {
        if self.intake_upstream {
            // 上游取水（排涝泵站）
            if intake_level >= self.start_level {
                self.operating = true;
            } else if intake_level <= self.stop_level {
                self.operating = false;
            }
        } else {
            // 下游取水（引水泵站）
            if intake_level <= self.start_level {
                self.operating = true;
            } else if intake_level >= self.stop_level {
                self.operating = false;
            }
        }
    }
}

impl HydraulicStructure for PumpStation {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_discharge(&mut self, h_up: f64, h_down: f64, _time: f64) -> f64 {
        let intake_level = if self.intake_upstream { h_up } else { h_down };
        self.update_state(intake_level);

        if !self.operating {
            self.current_discharge = 0.0;
            return 0.0;
        }

        self.current_discharge = match self.mode {
            PumpMode::ConstantFlow => self.design_discharge,
            PumpMode::Characteristic => {
                // TODO: 实现扬程-流量曲线
                self.design_discharge
            }
            PumpMode::LevelControlled => {
                // TODO: 实现PID控制
                self.design_discharge
            }
        };

        // 返回带符号的流量（上游取水为负，下游取水为正）
        if self.intake_upstream {
            -self.current_discharge
        } else {
            self.current_discharge
        }
    }

    fn crest_level(&self) -> f64 {
        self.stop_level.min(self.start_level)
    }

    fn allows_reverse_flow(&self) -> bool {
        false
    }

    fn is_operating(&self) -> bool {
        self.operating
    }
}

/// 涵洞流态
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CulvertFlowRegime {
    /// 无水流
    Dry,
    /// 自由入口控制
    InletControl,
    /// 出口控制（满管流）
    OutletControl,
    /// 部分满管流
    PartiallyFull,
}

/// 涵洞
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Culvert {
    /// 名称
    pub name: String,
    /// 直径或等效直径 [m]
    pub diameter: f64,
    /// 长度 [m]
    pub length: f64,
    /// 入口底高程 [m]
    pub inlet_invert: f64,
    /// 出口底高程 [m]
    pub outlet_invert: f64,
    /// 曼宁糙率系数
    pub manning_n: f64,
    /// 入口损失系数（典型 0.5）
    pub inlet_loss_coefficient: f64,
    /// 出口损失系数（典型 1.0）
    pub outlet_loss_coefficient: f64,
}

impl Culvert {
    /// 创建圆形涵洞
    pub fn circular(
        name: &str,
        diameter: f64,
        length: f64,
        inlet_invert: f64,
        outlet_invert: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            diameter,
            length,
            inlet_invert,
            outlet_invert,
            manning_n: 0.013,
            inlet_loss_coefficient: 0.5,
            outlet_loss_coefficient: 1.0,
        }
    }

    /// 设置曼宁系数
    pub fn with_manning(mut self, n: f64) -> Self {
        self.manning_n = n;
        self
    }

    /// 计算流态
    fn determine_flow_regime(&self, h_up: f64, h_down: f64) -> CulvertFlowRegime {
        let inlet_crown = self.inlet_invert + self.diameter;
        let outlet_crown = self.outlet_invert + self.diameter;
        
        let hw_inlet = h_up - self.inlet_invert;
        let tw_outlet = h_down - self.outlet_invert;

        if hw_inlet <= 0.0 {
            return CulvertFlowRegime::Dry;
        }

        // 简化判断
        if h_up > inlet_crown && h_down > outlet_crown {
            CulvertFlowRegime::OutletControl
        } else if h_up > inlet_crown * 1.2 {
            CulvertFlowRegime::InletControl
        } else {
            CulvertFlowRegime::PartiallyFull
        }
    }
}

impl HydraulicStructure for Culvert {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_discharge(&mut self, h_up: f64, h_down: f64, _time: f64) -> f64 {
        // 确定流向
        let (upstream, downstream, sign) = if h_up >= h_down {
            (h_up, h_down, 1.0)
        } else {
            (h_down, h_up, -1.0)
        };

        let regime = self.determine_flow_regime(upstream, downstream);
        
        match regime {
            CulvertFlowRegime::Dry => 0.0,
            CulvertFlowRegime::InletControl => {
                // 入口控制：堰流公式
                let hw = upstream - self.inlet_invert;
                let area = std::f64::consts::PI * self.diameter * self.diameter / 4.0;
                let cd = 0.6;
                sign * cd * area * (2.0 * G * hw).sqrt()
            }
            CulvertFlowRegime::OutletControl => {
                // 出口控制：管流公式 + 水头损失
                let delta_h = (upstream - downstream).max(0.0);
                let area = std::f64::consts::PI * self.diameter * self.diameter / 4.0;
                let perimeter = std::f64::consts::PI * self.diameter;
                let hydraulic_radius = area / perimeter;
                
                // 沿程损失系数
                let sf = self.manning_n * self.manning_n / hydraulic_radius.powf(4.0 / 3.0);
                let friction_loss = sf * self.length;
                
                // 总水头损失
                let k_total = self.inlet_loss_coefficient
                    + friction_loss
                    + self.outlet_loss_coefficient;
                
                if k_total > 0.0 {
                    let v = (2.0 * G * delta_h / k_total).sqrt();
                    sign * area * v
                } else {
                    0.0
                }
            }
            CulvertFlowRegime::PartiallyFull => {
                // 部分满管：简化为入口控制
                let hw = (upstream - self.inlet_invert).max(0.0);
                let area = std::f64::consts::PI * self.diameter * self.diameter / 4.0;
                let fill_ratio = (hw / self.diameter).clamp(0.0, 1.0);
                sign * 0.5 * area * fill_ratio * (2.0 * G * hw).sqrt()
            }
        }
    }

    fn crest_level(&self) -> f64 {
        self.inlet_invert.min(self.outlet_invert)
    }
}

/// 结构物管理器
#[derive(Default)]
pub struct StructureManager {
    /// 结构物列表
    structures: Vec<Box<dyn HydraulicStructure>>,
    /// 名称到索引的映射
    name_index: std::collections::HashMap<String, usize>,
}

impl StructureManager {
    /// 创建空管理器
    pub fn new() -> Self {
        Self {
            structures: Vec::new(),
            name_index: std::collections::HashMap::new(),
        }
    }

    /// 添加结构物
    pub fn add_structure(&mut self, structure: Box<dyn HydraulicStructure>) -> usize {
        let idx = self.structures.len();
        self.name_index.insert(structure.name().to_string(), idx);
        self.structures.push(structure);
        idx
    }

    /// 按名称获取结构物
    pub fn get_by_name(&self, name: &str) -> Option<&dyn HydraulicStructure> {
        self.name_index
            .get(name)
            .and_then(|&idx| self.structures.get(idx))
            .map(|s| s.as_ref())
    }

    /// 按名称获取可变引用
    pub fn get_by_name_mut(&mut self, name: &str) -> Option<&mut dyn HydraulicStructure> {
        self.name_index
            .get(name)
            .copied()
            .and_then(move |idx| self.structures.get_mut(idx))
            .map(|s| s.as_mut())
    }

    /// 获取结构物数量
    pub fn count(&self) -> usize {
        self.structures.len()
    }

    /// 计算所有结构物流量
    pub fn compute_all_discharges(
        &mut self,
        h_up: &[f64],
        h_down: &[f64],
        time: f64,
    ) -> Vec<f64> {
        self.structures
            .iter_mut()
            .enumerate()
            .map(|(i, s)| {
                let hu = h_up.get(i).copied().unwrap_or(0.0);
                let hd = h_down.get(i).copied().unwrap_or(0.0);
                s.compute_discharge(hu, hd, time)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broad_crested_weir() {
        let mut weir = BroadCrestedWeir::new("test_weir", 10.0, 5.0);
        
        // 自由出流
        let q = weir.compute_discharge(6.0, 4.0, 0.0);
        assert!(q > 0.0);
        
        // 无水流
        let q = weir.compute_discharge(4.0, 4.0, 0.0);
        assert!((q).abs() < 1e-10);
    }

    #[test]
    fn test_gate() {
        let mut gate = Gate::sluice("test_gate", 5.0, 0.0, 2.0);
        
        // 闸门关闭
        let q = gate.compute_discharge(3.0, 1.0, 0.0);
        assert!((q).abs() < 1e-10);
        
        // 开启闸门
        gate.set_opening(1.0).ok();
        let q = gate.compute_discharge(3.0, 1.0, 0.0);
        assert!(q > 0.0);
    }

    #[test]
    fn test_pump_station() {
        let mut pump = PumpStation::constant_flow("test_pump", 10.0, true, 5.0, 3.0);
        
        // 低水位不启动
        let q = pump.compute_discharge(2.0, 0.0, 0.0);
        assert!((q).abs() < 1e-10);
        
        // 高水位启动
        let q = pump.compute_discharge(6.0, 0.0, 0.0);
        assert!((q + 10.0).abs() < 1e-10); // 负值表示从上游抽水
    }

    #[test]
    fn test_culvert() {
        let mut culvert = Culvert::circular("test_culvert", 1.0, 50.0, 0.0, -0.1);
        
        // 有水头差时有流量
        let q = culvert.compute_discharge(2.0, 0.5, 0.0);
        assert!(q > 0.0);
    }
}
