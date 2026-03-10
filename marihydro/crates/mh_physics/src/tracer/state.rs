// crates/mh_physics/src/tracer/state.rs

//! 绀鸿釜鍓傜姸鎬佹ā鍧?
//!
//! 鏈ā鍧楀畾涔夌ず韪墏鐩稿叧鐨勭姸鎬佺被鍨嬶細
//! - TracerType: 绀鸿釜鍓傜被鍨嬫灇涓?
//! - TracerProperties: 绀鸿釜鍓傜墿鐞嗗睘鎬?
//! - TracerField: 鍗曚釜绀鸿釜鍓傜殑鍦烘暟鎹?
//! - TracerState: 澶氱ず韪墏闆嗗悎鐘舵€?
//!
//! # 姒傚康璇存槑
//!
//! 绀鸿釜鍓傦紙Tracer锛夋槸鎸囬殢姘存祦杩愮Щ鐨勭墿璐紝鍖呮嫭锛?
//! - 琚姩绀鸿釜鍓傦細鐩愬害銆佹俯搴︾瓑锛堜笉褰卞搷姘村姩鍔涳級
//! - 涓诲姩绀鸿釜鍓傦細娉ユ矙绛夛紙鍙兘褰卞搷姘村瘑搴﹀拰娴佸姩锛?
//!
//! # 杩佺Щ璇存槑
//!
//! 浠?history_src/tracer/tracer.rs 杩佺Щ锛屾敼杩涳細
//! - 浣跨敤鏋氫妇绫诲瀷鏇夸唬瀛楃涓叉爣璇?
//! - 鏀寔 serde 搴忓垪鍖?
//! - 涓庢柊鏋舵瀯鐨?StateAccess trait 闆嗘垚

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ============================================================
// 绀鸿釜鍓傜被鍨?
// ============================================================

/// 绀鸿釜鍓傜被鍨?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum TracerType {
    /// 鐩愬害 [PSU 鎴?kg/m鲁]
    #[default]
    Salinity,

    /// 娓╁害 [掳C 鎴?K]
    Temperature,

    /// 鎮诞娉ユ矙 [kg/m鲁]
    Sediment,

    /// 姹℃煋鐗?[浠绘剰娴撳害鍗曚綅]
    Pollutant,

    /// 婧惰В姘?[mg/L]
    DissolvedOxygen,

    /// 鍙剁豢绱?[渭g/L]
    Chlorophyll,

    /// 鑷畾涔夌ず韪墏
    Custom(u16),
}

impl TracerType {
    /// 鑾峰彇绫诲瀷鐨勫瓧绗︿覆鏍囪瘑
    pub fn name(&self) -> &'static str {
        match self {
            Self::Salinity => "salinity",
            Self::Temperature => "temperature",
            Self::Sediment => "sediment",
            Self::Pollutant => "pollutant",
            Self::DissolvedOxygen => "dissolved_oxygen",
            Self::Chlorophyll => "chlorophyll",
            Self::Custom(_) => "custom",
        }
    }

    /// 鏄惁涓鸿鍔ㄧず韪墏
    ///
    /// 琚姩绀鸿釜鍓備笉褰卞搷姘村姩鍔涙柟绋嬨€?
    pub fn is_passive(&self) -> bool {
        match self {
            Self::Sediment => false, // 娉ユ矙鍙兘褰卞搷瀵嗗害
            _ => true,
        }
    }

    /// 鏄惁闇€瑕侀澶栫殑婧愭眹椤?
    pub fn has_source_terms(&self) -> bool {
        match self {
            Self::DissolvedOxygen | Self::Chlorophyll => true, // 鐢熷寲鍙嶅簲
            Self::Sediment => true, // 娌夐檷/鍐嶆偓娴?
            _ => false,
        }
    }
}


// ============================================================
// 绀鸿釜鍓傚睘鎬?
// ============================================================

/// 绀鸿釜鍓傜墿鐞嗗睘鎬?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerProperties {
    /// 绀鸿釜鍓傜被鍨?
    pub tracer_type: TracerType,

    /// 绀鸿釜鍓傚悕绉帮紙鐢ㄤ簬鏄剧ず锛?
    pub name: String,

    /// 鍗曚綅
    pub unit: String,

    /// 鍒嗗瓙鎵╂暎绯绘暟 [m虏/s]
    pub molecular_diffusivity: f64,

    /// 鑳屾櫙娴撳害锛堢敤浜庤竟鐣屽拰鍒濆鍖栵級
    pub background_value: f64,

    /// 琛板噺绯绘暟 [1/s]
    ///
    /// 鐢ㄤ簬绠€鍗曠殑涓€闃惰“鍑忔ā鍨嬶細dC/dt = -k * C
    pub decay_rate: f64,

    /// 娌夐檷閫熷害 [m/s]
    ///
    /// 浠呴€傜敤浜庢偿娌欑瓑鍙矇闄嶇墿璐紝姝ｅ€艰〃绀哄悜涓嬫矇闄嶃€?
    pub settling_velocity: f64,

    /// 鏄惁鍚敤
    pub enabled: bool,
}

impl TracerProperties {
    /// 鍒涘缓榛樿鐩愬害绀鸿釜鍓?
    pub fn salinity() -> Self {
        Self {
            tracer_type: TracerType::Salinity,
            name: "Salinity".to_string(),
            unit: "PSU".to_string(),
            molecular_diffusivity: 1.5e-9,
            background_value: 35.0,
            decay_rate: 0.0,
            settling_velocity: 0.0,
            enabled: true,
        }
    }

    /// 鍒涘缓榛樿娓╁害绀鸿釜鍓?
    pub fn temperature() -> Self {
        Self {
            tracer_type: TracerType::Temperature,
            name: "Temperature".to_string(),
            unit: "掳C".to_string(),
            molecular_diffusivity: 1.4e-7,
            background_value: 20.0,
            decay_rate: 0.0,
            settling_velocity: 0.0,
            enabled: true,
        }
    }

    /// 鍒涘缓榛樿娉ユ矙绀鸿釜鍓?
    pub fn sediment() -> Self {
        Self {
            tracer_type: TracerType::Sediment,
            name: "Suspended Sediment".to_string(),
            unit: "kg/m鲁".to_string(),
            molecular_diffusivity: 0.0, // 涓昏闈犳箥娴佹墿鏁?
            background_value: 0.0,
            decay_rate: 0.0,
            settling_velocity: 1e-4, // 0.1 mm/s
            enabled: true,
        }
    }

    /// 鍒涘缓鑷畾涔夌ず韪墏
    pub fn custom(id: u16, name: &str, unit: &str) -> Self {
        Self {
            tracer_type: TracerType::Custom(id),
            name: name.to_string(),
            unit: unit.to_string(),
            molecular_diffusivity: 1e-9,
            background_value: 0.0,
            decay_rate: 0.0,
            settling_velocity: 0.0,
            enabled: true,
        }
    }

    /// 浣跨敤 Builder 妯″紡璁剧疆鍒嗗瓙鎵╂暎绯绘暟
    pub fn with_diffusivity(mut self, diffusivity: f64) -> Self {
        self.molecular_diffusivity = diffusivity;
        self
    }

    /// 浣跨敤 Builder 妯″紡璁剧疆鑳屾櫙鍊?
    pub fn with_background(mut self, value: f64) -> Self {
        self.background_value = value;
        self
    }

    /// 浣跨敤 Builder 妯″紡璁剧疆琛板噺鐜?
    pub fn with_decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = rate;
        self
    }

    /// 浣跨敤 Builder 妯″紡璁剧疆娌夐檷閫熷害
    pub fn with_settling_velocity(mut self, velocity: f64) -> Self {
        self.settling_velocity = velocity;
        self
    }
}

impl Default for TracerProperties {
    fn default() -> Self {
        Self::salinity()
    }
}

// ============================================================
// 绀鸿釜鍓傚満
// ============================================================

/// 鍗曚釜绀鸿釜鍓傜殑鍦烘暟鎹?
///
/// 瀛樺偍绀鸿釜鍓傚湪鎵€鏈夎绠楀崟鍏冧笂鐨勬祿搴﹀€笺€?
#[derive(Debug, Clone)]
pub struct TracerField {
    /// 绀鸿釜鍓傚睘鎬?
    properties: TracerProperties,

    /// 娴撳害鍦?[鍗曚綅鍙栧喅浜庣ず韪墏绫诲瀷]
    ///
    /// 绱㈠紩涓庤绠楀崟鍏冨搴斻€?
    concentration: Vec<f64>,

    /// 瀹堟亽閲忓満 (h * C)
    ///
    /// 鐢ㄤ簬鏈夐檺浣撶Н娉曡绠椼€?
    conserved: Vec<f64>,

    /// 鍙虫墜椤圭疮鍔犲櫒 (dC/dt)
    rhs: Vec<f64>,
}

impl TracerField {
    /// 鍒涘缓鏂扮殑绀鸿釜鍓傚満
    ///
    /// # 鍙傛暟
    /// - `properties`: 绀鸿釜鍓傚睘鎬?
    /// - `n_cells`: 璁＄畻鍗曞厓鏁伴噺
    pub fn new(properties: TracerProperties, n_cells: usize) -> Self {
        let background = properties.background_value;
        Self {
            properties,
            concentration: vec![background; n_cells],
            conserved: vec![0.0; n_cells], // 闇€瑕佷笌姘存繁閰嶅悎鍒濆鍖?
            rhs: vec![0.0; n_cells],
        }
    }

    /// 浠庢祿搴︽暟缁勫垱寤?
    pub fn from_concentration(properties: TracerProperties, concentration: Vec<f64>) -> Self {
        let n = concentration.len();
        Self {
            properties,
            concentration,
            conserved: vec![0.0; n],
            rhs: vec![0.0; n],
        }
    }

    /// 鑾峰彇绀鸿釜鍓傚睘鎬?
    pub fn properties(&self) -> &TracerProperties {
        &self.properties
    }

    /// 鑾峰彇绀鸿釜鍓傜被鍨?
    pub fn tracer_type(&self) -> TracerType {
        self.properties.tracer_type
    }

    /// 鑾峰彇鍗曞厓鏁伴噺
    pub fn len(&self) -> usize {
        self.concentration.len()
    }

    /// 妫€鏌ユ槸鍚︿负绌?
    pub fn is_empty(&self) -> bool {
        self.concentration.is_empty()
    }

    /// 鑾峰彇鍗曞厓娴撳害锛堝彧璇伙級
    pub fn concentration(&self, cell_idx: usize) -> f64 {
        self.concentration[cell_idx]
    }

    /// 鑾峰彇娴撳害鍦哄垏鐗?
    pub fn concentration_slice(&self) -> &[f64] {
        &self.concentration
    }

    /// 鑾峰彇娴撳害鍦哄彲鍙樺垏鐗?
    pub fn concentration_slice_mut(&mut self) -> &mut [f64] {
        &mut self.concentration
    }

    /// 鑾峰彇瀹堟亽閲忥紙h * C锛?
    pub fn conserved(&self, cell_idx: usize) -> f64 {
        self.conserved[cell_idx]
    }

    /// 鑾峰彇瀹堟亽閲忓満鍒囩墖
    pub fn conserved_slice(&self) -> &[f64] {
        &self.conserved
    }

    /// 鑾峰彇瀹堟亽閲忓満鍙彉鍒囩墖
    pub fn conserved_slice_mut(&mut self) -> &mut [f64] {
        &mut self.conserved
    }

    /// 璁剧疆鍗曞厓娴撳害
    pub fn set_concentration(&mut self, cell_idx: usize, value: f64) {
        self.concentration[cell_idx] = value;
    }

    /// 璁剧疆瀹堟亽閲?
    pub fn set_conserved(&mut self, cell_idx: usize, value: f64) {
        self.conserved[cell_idx] = value;
    }

    /// 浠庢按娣辨洿鏂板畧鎭掗噺
    ///
    /// 鐢ㄤ簬鍒濆鍖栨垨閲嶇疆锛歝onserved = h * concentration
    pub fn update_conserved_from_depth(&mut self, water_depths: &[f64]) {
        debug_assert_eq!(water_depths.len(), self.concentration.len());
        for i in 0..self.concentration.len() {
            self.conserved[i] = water_depths[i] * self.concentration[i];
        }
    }

    /// 浠庡畧鎭掗噺鏇存柊娴撳害
    ///
    /// 鐢ㄤ簬鏃堕棿姝ヨ繘鍚庯細concentration = conserved / h
    pub fn update_concentration_from_conserved(&mut self, water_depths: &[f64], h_min: f64) {
        debug_assert_eq!(water_depths.len(), self.concentration.len());
        for i in 0..self.concentration.len() {
            let h = water_depths[i].max(h_min);
            self.concentration[i] = self.conserved[i] / h;
        }
    }

    /// 鑾峰彇 RHS 鍒囩墖锛堢敤浜庢椂闂寸Н鍒嗭級
    pub fn rhs_slice(&self) -> &[f64] {
        &self.rhs
    }

    /// 鑾峰彇 RHS 鍙彉鍒囩墖
    pub fn rhs_slice_mut(&mut self) -> &mut [f64] {
        &mut self.rhs
    }

    /// 娓呴浂 RHS
    pub fn clear_rhs(&mut self) {
        self.rhs.fill(0.0);
    }

    /// 绱姞 RHS
    pub fn add_rhs(&mut self, cell_idx: usize, value: f64) {
        self.rhs[cell_idx] += value;
    }

    /// 浣跨敤鏄惧紡娆ф媺鏍煎紡鏇存柊瀹堟亽閲?
    ///
    /// conserved += dt * rhs
    ///
    /// # 鍙傛暟
    /// - `dt`: 鏃堕棿姝ラ暱 [s]
    pub fn apply_euler_update(&mut self, dt: f64) {
        for i in 0..self.conserved.len() {
            self.conserved[i] += dt * self.rhs[i];
        }
    }

    /// 搴旂敤琛板噺锛堜竴闃惰“鍑忔ā鍨嬶級
    ///
    /// # 鍙傛暟
    /// - `dt`: 鏃堕棿姝ラ暱 [s]
    pub fn apply_decay(&mut self, dt: f64) {
        let k = self.properties.decay_rate;
        if k > 0.0 {
            let factor = (-k * dt).exp();
            for c in &mut self.concentration {
                *c *= factor;
            }
            for hc in &mut self.conserved {
                *hc *= factor;
            }
        }
    }

    /// 璁＄畻鍦虹粺璁￠噺
    pub fn statistics(&self) -> TracerFieldStats {
        if self.concentration.is_empty() {
            return TracerFieldStats::default();
        }

        let mut min = f64::MAX;
        let mut max = f64::MIN;
        let mut sum = 0.0;

        for &c in &self.concentration {
            min = min.min(c);
            max = max.max(c);
            sum += c;
        }

        TracerFieldStats {
            min,
            max,
            mean: sum / self.concentration.len() as f64,
        }
    }

    /// 闄愬埗娴撳害鍦ㄧ墿鐞嗚寖鍥村唴
    ///
    /// # 鍙傛暟
    /// - `c_min`: 鏈€灏忔祿搴︼紙閫氬父涓?0锛?
    /// - `c_max`: 鏈€澶ф祿搴︼紙鍙€夛級
    pub fn clamp_concentration(&mut self, c_min: f64, c_max: Option<f64>) {
        for c in &mut self.concentration {
            *c = c.max(c_min);
            if let Some(max_val) = c_max {
                *c = c.min(max_val);
            }
        }
    }
}

/// 绀鸿釜鍓傚満缁熻閲?
#[derive(Debug, Clone, Copy, Default)]
pub struct TracerFieldStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

// ============================================================
// 澶氱ず韪墏鐘舵€?
// ============================================================

/// 澶氱ず韪墏闆嗗悎鐘舵€?
///
/// 绠＄悊澶氫釜绀鸿釜鍓傜殑鍦烘暟鎹€?
#[derive(Debug, Clone)]
pub struct TracerState {
    /// 绀鸿釜鍓傚満闆嗗悎锛堟寜绫诲瀷绱㈠紩锛?
    fields: HashMap<TracerType, TracerField>,

    /// 绫诲瀷鍒楄〃锛堜繚鎸佹坊鍔犻『搴忥級
    types: Vec<TracerType>,

    /// 璁＄畻鍗曞厓鏁伴噺
    n_cells: usize,
}

impl TracerState {
    /// 鍒涘缓鏂扮殑澶氱ず韪墏鐘舵€?
    pub fn new(n_cells: usize) -> Self {
        Self {
            fields: HashMap::new(),
            types: Vec::new(),
            n_cells,
        }
    }

    /// 娣诲姞绀鸿釜鍓?
    ///
    /// # 鍙傛暟
    /// - `properties`: 绀鸿釜鍓傚睘鎬?
    ///
    /// # 杩斿洖
    /// 濡傛灉绫诲瀷宸插瓨鍦ㄥ垯杩斿洖閿欒
    pub fn add_tracer(&mut self, properties: TracerProperties) -> Result<(), TracerError> {
        let tracer_type = properties.tracer_type;
        if self.fields.contains_key(&tracer_type) {
            return Err(TracerError::DuplicateType(tracer_type));
        }

        let field = TracerField::new(properties, self.n_cells);
        self.fields.insert(tracer_type, field);
        self.types.push(tracer_type);
        Ok(())
    }

    /// 鑾峰彇绀鸿釜鍓傚満
    pub fn get(&self, tracer_type: TracerType) -> Option<&TracerField> {
        self.fields.get(&tracer_type)
    }

    /// 鑾峰彇绀鸿釜鍓傚満锛堝彲鍙橈級
    pub fn get_mut(&mut self, tracer_type: TracerType) -> Option<&mut TracerField> {
        self.fields.get_mut(&tracer_type)
    }

    /// 妫€鏌ユ槸鍚﹀寘鍚寚瀹氱被鍨?
    pub fn contains(&self, tracer_type: TracerType) -> bool {
        self.fields.contains_key(&tracer_type)
    }

    /// 鑾峰彇绀鸿釜鍓傛暟閲?
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// 妫€鏌ユ槸鍚︿负绌?
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// 鑾峰彇鎵€鏈夌ず韪墏绫诲瀷
    pub fn types(&self) -> &[TracerType] {
        &self.types
    }

    /// 閬嶅巻鎵€鏈夊満
    pub fn iter(&self) -> impl Iterator<Item = (&TracerType, &TracerField)> {
        self.fields.iter()
    }

    /// 閬嶅巻鎵€鏈夊満锛堝彲鍙橈級
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&TracerType, &mut TracerField)> {
        self.fields.iter_mut()
    }

    /// 浠庢按娣辨洿鏂版墍鏈夊畧鎭掗噺
    pub fn update_conserved_from_depth(&mut self, water_depths: &[f64]) {
        for field in self.fields.values_mut() {
            field.update_conserved_from_depth(water_depths);
        }
    }

    /// 浠庡畧鎭掗噺鏇存柊鎵€鏈夋祿搴?
    pub fn update_concentration_from_conserved(&mut self, water_depths: &[f64], h_min: f64) {
        for field in self.fields.values_mut() {
            field.update_concentration_from_conserved(water_depths, h_min);
        }
    }

    /// 娓呴浂鎵€鏈?RHS
    pub fn clear_all_rhs(&mut self) {
        for field in self.fields.values_mut() {
            field.clear_rhs();
        }
    }

    /// 搴旂敤琛板噺鍒版墍鏈夌ず韪墏
    pub fn apply_all_decay(&mut self, dt: f64) {
        for field in self.fields.values_mut() {
            field.apply_decay(dt);
        }
    }
}

// ============================================================
// 閿欒绫诲瀷
// ============================================================

/// 绀鸿釜鍓傛ā鍧楅敊璇?
#[derive(Debug, Error)]
pub enum TracerError {
    /// 閲嶅鐨勭ず韪墏绫诲瀷
    #[error("示踪剂类型 {0:?} 已存在")]
    DuplicateType(TracerType),

    /// 绀鸿釜鍓傛湭鎵惧埌
    #[error("示踪剂类型 {0:?} 未找到")]
    NotFound(TracerType),

    /// 鏁扮粍澶у皬涓嶅尮閰?
    #[error("鏁扮粍澶у皬涓嶅尮閰? 鏈熸湜 {expected}, 瀹為檯 {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    /// 鏃犳晥鐨勬祿搴﹀€?
    #[error("鏃犳晥鐨勬祿搴﹀€? {0}")]
    InvalidValue(f64),
}

// ============================================================
// 娴嬭瘯
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_tracer_type() {
        let t = TracerType::Salinity;
        assert_eq!(t.name(), "salinity");
        assert!(t.is_passive());
        assert!(!t.has_source_terms());

        let t = TracerType::Sediment;
        assert!(!t.is_passive());
        assert!(t.has_source_terms());
    }

    #[test]
    fn test_tracer_properties() {
        let props = TracerProperties::salinity()
            .with_background(30.0)
            .with_diffusivity(2e-9);

        assert_eq!(props.tracer_type, TracerType::Salinity);
        assert!(approx_eq(props.background_value, 30.0));
        assert!(approx_eq(props.molecular_diffusivity, 2e-9));
    }

    #[test]
    fn test_tracer_field_creation() {
        let props = TracerProperties::salinity();
        let field = TracerField::new(props, 100);

        assert_eq!(field.len(), 100);
        assert_eq!(field.tracer_type(), TracerType::Salinity);
        assert!(approx_eq(field.concentration(0), 35.0)); // 鑳屾櫙鍊?
    }

    #[test]
    fn test_tracer_field_conserved() {
        let props = TracerProperties::salinity().with_background(10.0);
        let mut field = TracerField::new(props, 3);

        // 鍋囪姘存繁
        let depths = vec![1.0, 2.0, 3.0];
        field.update_conserved_from_depth(&depths);

        assert!(approx_eq(field.conserved(0), 10.0)); // 1.0 * 10
        assert!(approx_eq(field.conserved(1), 20.0)); // 2.0 * 10
        assert!(approx_eq(field.conserved(2), 30.0)); // 3.0 * 10
    }

    #[test]
    fn test_tracer_field_decay() {
        let props = TracerProperties::salinity()
            .with_background(100.0)
            .with_decay_rate(0.1);
        let mut field = TracerField::new(props, 1);

        field.apply_decay(1.0);
        // 绮剧‘鎸囨暟琛板噺瑙ｏ細C = C0 * exp(-k * dt)
        let expected = 100.0 * (-0.1_f64).exp();
        assert!((field.concentration(0) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_tracer_field_statistics() {
        let props = TracerProperties::salinity();
        let mut field = TracerField::from_concentration(props, vec![10.0, 20.0, 30.0]);

        let stats = field.statistics();
        assert!(approx_eq(stats.min, 10.0));
        assert!(approx_eq(stats.max, 30.0));
        assert!(approx_eq(stats.mean, 20.0));

        // 娴嬭瘯闄愬埗
        field.clamp_concentration(15.0, Some(25.0));
        assert!(approx_eq(field.concentration(0), 15.0));
        assert!(approx_eq(field.concentration(1), 20.0));
        assert!(approx_eq(field.concentration(2), 25.0));
    }

    #[test]
    fn test_tracer_state() {
        let mut state = TracerState::new(100);

        state.add_tracer(TracerProperties::salinity()).unwrap();
        state.add_tracer(TracerProperties::temperature()).unwrap();

        assert_eq!(state.len(), 2);
        assert!(state.contains(TracerType::Salinity));
        assert!(state.contains(TracerType::Temperature));
        assert!(!state.contains(TracerType::Sediment));
    }

    #[test]
    fn test_duplicate_tracer_error() {
        let mut state = TracerState::new(100);

        state.add_tracer(TracerProperties::salinity()).unwrap();
        let result = state.add_tracer(TracerProperties::salinity());

        assert!(result.is_err());
        if let Err(TracerError::DuplicateType(t)) = result {
            assert_eq!(t, TracerType::Salinity);
        } else {
            panic!("Expected DuplicateType error");
        }
    }

    #[test]
    fn test_tracer_state_update() {
        let mut state = TracerState::new(3);
        state.add_tracer(TracerProperties::salinity().with_background(10.0)).unwrap();

        let depths = vec![1.0, 2.0, 3.0];
        state.update_conserved_from_depth(&depths);

        let field = state.get(TracerType::Salinity).unwrap();
        assert!(approx_eq(field.conserved(1), 20.0));
    }
}


// ============================================================
// 娉涘瀷绀鸿釜鍓傚満锛圔ackend 鎶借薄锛?
// ============================================================

use crate::core::{Backend, CpuBackend, DeviceBuffer, Scalar};

/// 娉涘瀷绀鸿釜鍓傚満
///
/// 浣跨敤 Backend trait 鎶借薄瀛樺偍锛屾敮鎸?CPU/GPU 鍚庣銆?
///
/// # 绫诲瀷鍙傛暟
///
/// - `B`: 璁＄畻鍚庣绫诲瀷锛屽繀椤诲疄鐜?`Backend` trait
#[derive(Debug, Clone)]
pub struct TracerFieldGeneric<B: Backend> {
    /// 绀鸿釜鍓傚睘鎬?
    properties: TracerProperties,
    /// 娴撳害鍦?[鍗曚綅鍙栧喅浜庣ず韪墏绫诲瀷]
    concentration: B::Buffer<B::Scalar>,
    /// 瀹堟亽閲忓満 (h * C)
    conserved: B::Buffer<B::Scalar>,
    /// 鍙虫墜椤圭疮鍔犲櫒 (dC/dt)
    rhs: B::Buffer<B::Scalar>,
    /// 鍗曞厓鏁伴噺
    n_cells: usize,
    /// 鍚庣瀹炰緥
    backend: B,
}

impl<B: Backend> TracerFieldGeneric<B> {
    /// 浣跨敤鍚庣瀹炰緥鍒涘缓鏂扮殑绀鸿釜鍓傚満
    pub fn new_with_backend(backend: B, properties: TracerProperties, n_cells: usize) -> Self {
        let background = <B::Scalar as Scalar>::from_f64(properties.background_value);
        let mut concentration = backend.alloc(n_cells);
        concentration.fill(background);
        
        Self {
            properties,
            concentration,
            conserved: backend.alloc(n_cells),
            rhs: backend.alloc(n_cells),
            n_cells,
            backend,
        }
    }
    
    /// 鑾峰彇绀鸿釜鍓傚睘鎬?
    pub fn properties(&self) -> &TracerProperties {
        &self.properties
    }
    
    /// 鑾峰彇绀鸿釜鍓傜被鍨?
    pub fn tracer_type(&self) -> TracerType {
        self.properties.tracer_type
    }
    
    /// 鑾峰彇鍗曞厓鏁伴噺
    pub fn len(&self) -> usize {
        self.n_cells
    }
    
    /// 妫€鏌ユ槸鍚︿负绌?
    pub fn is_empty(&self) -> bool {
        self.n_cells == 0
    }
    
    /// 鑾峰彇鍚庣寮曠敤
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 鑾峰彇娴撳害鍦哄紩鐢?
    pub fn concentration(&self) -> &B::Buffer<B::Scalar> {
        &self.concentration
    }
    
    /// 鑾峰彇娴撳害鍦哄彲鍙樺紩鐢?
    pub fn concentration_mut(&mut self) -> &mut B::Buffer<B::Scalar> {
        &mut self.concentration
    }
    
    /// 鑾峰彇瀹堟亽閲忓満寮曠敤
    pub fn conserved(&self) -> &B::Buffer<B::Scalar> {
        &self.conserved
    }
    
    /// 鑾峰彇瀹堟亽閲忓満鍙彉寮曠敤
    pub fn conserved_mut(&mut self) -> &mut B::Buffer<B::Scalar> {
        &mut self.conserved
    }
    
    /// 鑾峰彇 RHS 寮曠敤
    pub fn rhs(&self) -> &B::Buffer<B::Scalar> {
        &self.rhs
    }
    
    /// 鑾峰彇 RHS 鍙彉寮曠敤
    pub fn rhs_mut(&mut self) -> &mut B::Buffer<B::Scalar> {
        &mut self.rhs
    }
    
    /// 娓呴浂 RHS
    pub fn clear_rhs(&mut self) {
        self.rhs.fill(B::Scalar::ZERO);
    }
    
    /// 閲嶇疆涓鸿儗鏅€?
    pub fn reset(&mut self) {
        let background = <B::Scalar as Scalar>::from_f64(self.properties.background_value);
        self.concentration.fill(background);
        self.conserved.fill(B::Scalar::ZERO);
        self.rhs.fill(B::Scalar::ZERO);
    }
}

/// CPU f64 鍚庣鐨勪究鎹锋柟娉?
impl TracerFieldGeneric<CpuBackend<f64>> {
    /// 浣跨敤榛樿 CPU f64 鍚庣鍒涘缓
    pub fn new(properties: TracerProperties, n_cells: usize) -> Self {
        Self::new_with_backend(CpuBackend::<f64>::new(), properties, n_cells)
    }
    
    
    /// 鑾峰彇娴撳害鍒囩墖锛堜粎 CPU 鍚庣锛?
    pub fn concentration_slice(&self) -> &[f64] {
        &self.concentration
    }
    
    /// 鑾峰彇娴撳害鍙彉鍒囩墖锛堜粎 CPU 鍚庣锛?
    pub fn concentration_slice_mut(&mut self) -> &mut [f64] {
        &mut self.concentration
    }
    
    /// 鑾峰彇瀹堟亽閲忓垏鐗囷紙浠?CPU 鍚庣锛?
    pub fn conserved_slice(&self) -> &[f64] {
        &self.conserved
    }
    
    /// 鑾峰彇瀹堟亽閲忓彲鍙樺垏鐗囷紙浠?CPU 鍚庣锛?
    pub fn conserved_slice_mut(&mut self) -> &mut [f64] {
        &mut self.conserved
    }
    
    /// 鑾峰彇 RHS 鍒囩墖锛堜粎 CPU 鍚庣锛?
    pub fn rhs_slice(&self) -> &[f64] {
        &self.rhs
    }
    
    /// 鑾峰彇 RHS 鍙彉鍒囩墖锛堜粎 CPU 鍚庣锛?
    pub fn rhs_slice_mut(&mut self) -> &mut [f64] {
        &mut self.rhs
    }
    
    /// 浠庢按娣辨洿鏂板畧鎭掗噺
    pub fn update_conserved_from_depth(&mut self, water_depths: &[f64]) {
        debug_assert_eq!(water_depths.len(), self.n_cells);
        for i in 0..self.n_cells {
            self.conserved[i] = water_depths[i] * self.concentration[i];
        }
    }
    
    /// 浠庡畧鎭掗噺鏇存柊娴撳害
    pub fn update_concentration_from_conserved(&mut self, water_depths: &[f64], h_min: f64) {
        debug_assert_eq!(water_depths.len(), self.n_cells);
        for i in 0..self.n_cells {
            let h = water_depths[i].max(h_min);
            self.concentration[i] = self.conserved[i] / h;
        }
    }
    
    /// 浣跨敤鏄惧紡娆ф媺鏍煎紡鏇存柊瀹堟亽閲?
    pub fn apply_euler_update(&mut self, dt: f64) {
        for i in 0..self.n_cells {
            self.conserved[i] += dt * self.rhs[i];
        }
    }
    
    /// 搴旂敤琛板噺
    pub fn apply_decay(&mut self, dt: f64) {
        let k = self.properties.decay_rate;
        if k > 0.0 {
            let factor = (-k * dt).exp();
            for c in self.concentration.iter_mut() {
                *c *= factor;
            }
            for hc in self.conserved.iter_mut() {
                *hc *= factor;
            }
        }
    }
    
    /// 璁＄畻鍦虹粺璁￠噺
    pub fn statistics(&self) -> TracerFieldStats {
        if self.n_cells == 0 {
            return TracerFieldStats::default();
        }
        
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        let mut sum = 0.0;
        
        for &c in self.concentration.iter() {
            min = min.min(c);
            max = max.max(c);
            sum += c;
        }
        
        TracerFieldStats {
            min,
            max,
            mean: sum / self.n_cells as f64,
        }
    }
    
    /// 闄愬埗娴撳害鍦ㄧ墿鐞嗚寖鍥村唴
    pub fn clamp_concentration(&mut self, c_min: f64, c_max: Option<f64>) {
        for c in self.concentration.iter_mut() {
            *c = c.max(c_min);
            if let Some(max_val) = c_max {
                *c = c.min(max_val);
            }
        }
    }
}

/// CPU f32 鍚庣鐨勪究鎹锋柟娉?
impl TracerFieldGeneric<CpuBackend<f32>> {
    /// 浣跨敤 CPU f32 鍚庣鍒涘缓
    pub fn new_f32(properties: TracerProperties, n_cells: usize) -> Self {
        Self::new_with_backend(CpuBackend::<f32>::new(), properties, n_cells)
    }
}

/// 绫诲瀷鍒悕锛氶粯璁ゅ悗绔殑绀鸿釜鍓傚満
pub type TracerFieldDefault = TracerFieldGeneric<CpuBackend<f64>>;

// ============================================================
// 娉涘瀷绀鸿釜鍓傚満娴嬭瘯
// ============================================================

#[cfg(test)]
mod generic_tests {
    use super::*;
    
    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }
    
    #[test]
    fn test_generic_tracer_field_creation() {
        let props = TracerProperties::salinity();
        let field = TracerFieldGeneric::<CpuBackend<f64>>::new(props, 100);
        
        assert_eq!(field.len(), 100);
        assert_eq!(field.tracer_type(), TracerType::Salinity);
        assert!(approx_eq(field.concentration_slice()[0], 35.0));
    }
    
    #[test]
    fn test_generic_tracer_field_conserved() {
        let props = TracerProperties::salinity().with_background(10.0);
        let mut field = TracerFieldGeneric::<CpuBackend<f64>>::new(props, 3);
        
        let depths = vec![1.0, 2.0, 3.0];
        field.update_conserved_from_depth(&depths);
        
        assert!(approx_eq(field.conserved_slice()[0], 10.0));
        assert!(approx_eq(field.conserved_slice()[1], 20.0));
        assert!(approx_eq(field.conserved_slice()[2], 30.0));
    }
    
    #[test]
    fn test_generic_tracer_field_decay() {
        let props = TracerProperties::salinity()
            .with_background(100.0)
            .with_decay_rate(0.1);
        let mut field = TracerFieldGeneric::<CpuBackend<f64>>::new(props, 1);
        
        field.apply_decay(1.0);
        let expected = 100.0 * (-0.1_f64).exp();
        assert!((field.concentration_slice()[0] - expected).abs() < 1e-12);
    }
    
    #[test]
    fn test_generic_tracer_field_statistics() {
        let props = TracerProperties::salinity();
        let mut field = TracerFieldGeneric::<CpuBackend<f64>>::new(props, 3);
        
        field.concentration_slice_mut()[0] = 10.0;
        field.concentration_slice_mut()[1] = 20.0;
        field.concentration_slice_mut()[2] = 30.0;
        
        let stats = field.statistics();
        assert!(approx_eq(stats.min, 10.0));
        assert!(approx_eq(stats.max, 30.0));
        assert!(approx_eq(stats.mean, 20.0));
    }
    
}
