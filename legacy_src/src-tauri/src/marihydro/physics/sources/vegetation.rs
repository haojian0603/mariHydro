// src-tauri/src/marihydro/physics/sources/vegetation.rs
//! 植被阻力模型
//!
//! 实现基于柱体阻力理论的植被模型，适用于湿地、红树林、滨海植被等。
//! 参考：Baptist et al. (2007), Nepf (1999)

use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, NumericalParams};
use glam::DVec2;

/// 植被类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VegetationType {
    /// 刚性植被（如芦苇、红树林）
    Rigid,
    /// 柔性植被（如海草、水草）
    Flexible,
    /// 浮叶植被（如睡莲）
    FloatingLeaves,
}

impl Default for VegetationType {
    fn default() -> Self {
        Self::Rigid
    }
}

/// 植被属性
#[derive(Debug, Clone, Copy)]
pub struct VegetationProperties {
    /// 阻力系数 CD [-]
    pub drag_coefficient: f64,
    /// 植被密度 [stems/m²]
    pub stem_density: f64,
    /// 茎杆直径 [m]
    pub stem_diameter: f64,
    /// 植被高度 [m]
    pub height: f64,
    /// 植被类型
    pub veg_type: VegetationType,
    /// 柔性系数（用于柔性植被弯曲）
    pub flexibility: f64,
}

impl Default for VegetationProperties {
    fn default() -> Self {
        Self {
            drag_coefficient: 1.0,
            stem_density: 100.0,      // 100 stems/m²
            stem_diameter: 0.01,      // 1 cm
            height: 0.5,              // 0.5 m
            veg_type: VegetationType::Rigid,
            flexibility: 0.0,
        }
    }
}

impl VegetationProperties {
    /// 创建刚性植被
    pub fn rigid(cd: f64, density: f64, diameter: f64, height: f64) -> Self {
        Self {
            drag_coefficient: cd,
            stem_density: density,
            stem_diameter: diameter,
            height,
            veg_type: VegetationType::Rigid,
            flexibility: 0.0,
        }
    }

    /// 创建柔性植被
    pub fn flexible(cd: f64, density: f64, diameter: f64, height: f64, flexibility: f64) -> Self {
        Self {
            drag_coefficient: cd,
            stem_density: density,
            stem_diameter: diameter,
            height,
            veg_type: VegetationType::Flexible,
            flexibility,
        }
    }

    /// 创建典型芦苇
    pub fn reeds() -> Self {
        Self::rigid(1.0, 200.0, 0.008, 2.0)
    }

    /// 创建典型红树林
    pub fn mangrove() -> Self {
        Self::rigid(1.5, 50.0, 0.03, 1.5)
    }

    /// 创建典型海草
    pub fn seagrass() -> Self {
        Self::flexible(0.8, 500.0, 0.005, 0.3, 0.5)
    }

    /// 计算有效阻力高度（考虑淹没）
    pub fn effective_height(&self, water_depth: f64) -> f64 {
        self.height.min(water_depth).max(0.0)
    }

    /// 计算植被阻力面积密度 a = n * d [1/m]
    pub fn frontal_area_density(&self) -> f64 {
        self.stem_density * self.stem_diameter
    }

    /// 计算柔性植被弯曲后的有效高度
    /// 
    /// 使用 Luhar & Nepf (2011) 简化公式
    pub fn bent_height(&self, water_depth: f64, velocity: f64) -> f64 {
        if self.veg_type != VegetationType::Flexible || self.flexibility < 1e-6 {
            return self.effective_height(water_depth);
        }

        // Cauchy 数 Ca = ρ * CD * U² * L / (E * I)
        // 简化：有效高度 = h * (1 - 柔性系数 * (U/U_ref))
        let u_ref = 0.3; // 参考速度
        let reduction = (self.flexibility * velocity / u_ref).min(0.8);
        let bent_h = self.height * (1.0 - reduction);

        bent_h.min(water_depth).max(0.0)
    }
}

/// 植被场
#[derive(Debug, Clone)]
pub struct VegetationField {
    /// 各单元植被属性
    properties: Vec<Option<VegetationProperties>>,
    /// 植被覆盖率 [0-1]
    coverage: Vec<f64>,
}

impl VegetationField {
    /// 创建空的植被场
    pub fn new(n_cells: usize) -> Self {
        Self {
            properties: vec![None; n_cells],
            coverage: vec![0.0; n_cells],
        }
    }

    /// 创建均匀植被场
    pub fn uniform(n_cells: usize, props: VegetationProperties) -> Self {
        Self {
            properties: vec![Some(props); n_cells],
            coverage: vec![1.0; n_cells],
        }
    }

    /// 设置单元植被
    pub fn set_cell(&mut self, cell: usize, props: VegetationProperties, coverage: f64) {
        if cell < self.properties.len() {
            self.properties[cell] = Some(props);
            self.coverage[cell] = coverage.clamp(0.0, 1.0);
        }
    }

    /// 清除单元植被
    pub fn clear_cell(&mut self, cell: usize) {
        if cell < self.properties.len() {
            self.properties[cell] = None;
            self.coverage[cell] = 0.0;
        }
    }

    /// 设置区域植被（通过谓词函数）
    pub fn set_region<F>(&mut self, props: VegetationProperties, coverage: f64, predicate: F)
    where
        F: Fn(usize) -> bool,
    {
        for i in 0..self.properties.len() {
            if predicate(i) {
                self.properties[i] = Some(props);
                self.coverage[i] = coverage.clamp(0.0, 1.0);
            }
        }
    }

    /// 获取单元植被属性
    pub fn get(&self, cell: usize) -> Option<&VegetationProperties> {
        self.properties.get(cell).and_then(|p| p.as_ref())
    }

    /// 获取覆盖率
    pub fn coverage(&self, cell: usize) -> f64 {
        self.coverage.get(cell).copied().unwrap_or(0.0)
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        self.properties.resize(n_cells, None);
        self.coverage.resize(n_cells, 0.0);
    }

    /// 获取有植被的单元数量
    pub fn vegetated_cell_count(&self) -> usize {
        self.properties.iter().filter(|p| p.is_some()).count()
    }
}

/// 植被阻力计算器
pub struct VegetationDrag {
    /// 水密度 [kg/m³]
    rho: f64,
    /// 植被场
    vegetation: VegetationField,
    /// 阻力加速度
    drag_x: Vec<f64>,
    drag_y: Vec<f64>,
}

impl VegetationDrag {
    /// 创建新的计算器
    pub fn new(n_cells: usize, rho: f64) -> Self {
        Self {
            rho,
            vegetation: VegetationField::new(n_cells),
            drag_x: vec![0.0; n_cells],
            drag_y: vec![0.0; n_cells],
        }
    }

    /// 从植被场创建
    pub fn with_vegetation(vegetation: VegetationField, rho: f64) -> Self {
        let n = vegetation.properties.len();
        Self {
            rho,
            vegetation,
            drag_x: vec![0.0; n],
            drag_y: vec![0.0; n],
        }
    }

    /// 设置植被场
    pub fn set_vegetation(&mut self, vegetation: VegetationField) {
        let n = vegetation.properties.len();
        self.vegetation = vegetation;
        self.drag_x.resize(n, 0.0);
        self.drag_y.resize(n, 0.0);
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        self.vegetation.resize(n_cells);
        self.drag_x.resize(n_cells, 0.0);
        self.drag_y.resize(n_cells, 0.0);
    }

    /// 计算植被阻力
    /// 
    /// F_v = 0.5 * CD * n * d * h_eff * |u| * u
    /// 
    /// 其中阻力表示为加速度（除以 ρh）
    pub fn compute(
        &mut self,
        depths: &[f64],
        velocities_u: &[f64],
        velocities_v: &[f64],
        params: &NumericalParams,
    ) {
        self.drag_x.fill(0.0);
        self.drag_y.fill(0.0);

        let n = self.drag_x.len()
            .min(depths.len())
            .min(velocities_u.len())
            .min(velocities_v.len());

        for i in 0..n {
            let h = depths[i];
            if params.is_dry(h) {
                continue;
            }

            let props = match self.vegetation.get(i) {
                Some(p) => p,
                None => continue,
            };

            let coverage = self.vegetation.coverage(i);
            if coverage < 1e-6 {
                continue;
            }

            let u = velocities_u[i];
            let v = velocities_v[i];
            let vel_mag = (u * u + v * v).sqrt();

            if vel_mag < 1e-10 {
                continue;
            }

            // 有效高度
            let h_eff = match props.veg_type {
                VegetationType::Flexible => props.bent_height(h, vel_mag),
                _ => props.effective_height(h),
            };

            if h_eff < 1e-6 {
                continue;
            }

            // 阻力公式：F = 0.5 * CD * n * d * h_eff * |u| * u
            // 加速度 = F / (ρ * h) = 0.5 * CD * n * d * h_eff / h * |u| * u
            let cd = props.drag_coefficient;
            let a = props.frontal_area_density(); // n * d

            let coeff = 0.5 * cd * a * h_eff / h * vel_mag * coverage;

            self.drag_x[i] = -coeff * u;
            self.drag_y[i] = -coeff * v;
        }
    }

    /// 计算并直接应用到动量方程（隐式处理）
    /// 
    /// 使用隐式格式避免数值不稳定：
    /// u^(n+1) = u^n / (1 + α*dt)
    pub fn apply_implicit(
        &mut self,
        depths: &[f64],
        hu: &mut [f64],
        hv: &mut [f64],
        params: &NumericalParams,
        dt: f64,
    ) {
        let n = hu.len().min(hv.len()).min(depths.len());

        for i in 0..n {
            let h = depths[i];
            if params.is_dry(h) {
                continue;
            }

            let props = match self.vegetation.get(i) {
                Some(p) => p,
                None => continue,
            };

            let coverage = self.vegetation.coverage(i);
            if coverage < 1e-6 {
                continue;
            }

            let u = hu[i] / h;
            let v = hv[i] / h;
            let vel_mag = (u * u + v * v).sqrt();

            if vel_mag < 1e-10 {
                continue;
            }

            // 有效高度
            let h_eff = match props.veg_type {
                VegetationType::Flexible => props.bent_height(h, vel_mag),
                _ => props.effective_height(h),
            };

            if h_eff < 1e-6 {
                continue;
            }

            // 阻力系数
            let cd = props.drag_coefficient;
            let a = props.frontal_area_density();
            let alpha = 0.5 * cd * a * h_eff / h * vel_mag * coverage;

            // 隐式更新
            let factor = 1.0 / (1.0 + alpha * dt);
            hu[i] *= factor;
            hv[i] *= factor;
        }
    }

    /// 获取阻力加速度
    pub fn drag_acceleration(&self) -> (&[f64], &[f64]) {
        (&self.drag_x, &self.drag_y)
    }

    /// 获取植被场引用
    pub fn vegetation(&self) -> &VegetationField {
        &self.vegetation
    }

    /// 获取可变植被场引用
    pub fn vegetation_mut(&mut self) -> &mut VegetationField {
        &mut self.vegetation
    }
}

/// 计算 Baptist 等效糙率
/// 
/// 将植被阻力转换为等效 Manning 糙率
/// n_eq = sqrt(n_b² + CD * n_v * d * h / (2g))
pub fn equivalent_manning(
    base_manning: f64,
    props: &VegetationProperties,
    water_depth: f64,
    g: f64,
) -> f64 {
    let h_eff = props.effective_height(water_depth);
    if h_eff < 1e-6 {
        return base_manning;
    }

    let cd = props.drag_coefficient;
    let a = props.frontal_area_density();

    // Baptist (2007) 公式
    let n_veg_sq = cd * a * h_eff / (2.0 * g);
    let n_eq = (base_manning * base_manning + n_veg_sq).sqrt();

    n_eq
}

/// 计算湍流增强系数（用于 k-ε 模型）
/// 
/// 植被产生的湍流 P_veg = CD * a * |u|³
pub fn vegetation_turbulence_production(
    props: &VegetationProperties,
    velocity_mag: f64,
    water_depth: f64,
) -> f64 {
    let h_eff = props.effective_height(water_depth);
    if h_eff < 1e-6 {
        return 0.0;
    }

    let cd = props.drag_coefficient;
    let a = props.frontal_area_density();

    cd * a * h_eff / water_depth * velocity_mag.powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vegetation_properties() {
        let props = VegetationProperties::default();
        assert!((props.drag_coefficient - 1.0).abs() < 1e-10);
        assert!((props.frontal_area_density() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_effective_height() {
        let props = VegetationProperties::rigid(1.0, 100.0, 0.01, 0.5);

        // 完全淹没
        assert!((props.effective_height(1.0) - 0.5).abs() < 1e-10);

        // 部分淹没
        assert!((props.effective_height(0.3) - 0.3).abs() < 1e-10);

        // 干涸
        assert!((props.effective_height(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_flexible_bending() {
        let props = VegetationProperties::flexible(1.0, 100.0, 0.01, 0.5, 0.5);

        let h_low_v = props.bent_height(1.0, 0.1);
        let h_high_v = props.bent_height(1.0, 0.5);

        // 高流速时弯曲更多
        assert!(h_high_v < h_low_v);
    }

    #[test]
    fn test_vegetation_field() {
        let mut field = VegetationField::new(100);
        assert_eq!(field.vegetated_cell_count(), 0);

        field.set_cell(10, VegetationProperties::reeds(), 1.0);
        assert_eq!(field.vegetated_cell_count(), 1);
        assert!(field.get(10).is_some());
        assert!((field.coverage(10) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equivalent_manning() {
        let props = VegetationProperties::rigid(1.0, 100.0, 0.01, 0.5);
        let n_base = 0.03;
        let g = 9.81;

        let n_eq = equivalent_manning(n_base, &props, 1.0, g);
        assert!(n_eq > n_base);
    }

    #[test]
    fn test_preset_vegetation() {
        let reeds = VegetationProperties::reeds();
        assert!(reeds.height > 1.0);

        let mangrove = VegetationProperties::mangrove();
        assert!(mangrove.stem_diameter > reeds.stem_diameter);

        let seagrass = VegetationProperties::seagrass();
        assert!(seagrass.flexibility > 0.0);
    }
}
