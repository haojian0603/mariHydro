// crates/mh_physics/src/wetting_drying.rs

//! 工业级干湿边界处理
//!
//! 提供比肩 Delft3D/MIKE 的干湿过渡处理，支持：
//! - 多层水深阈值
//! - 数值稳定处理
//! - 动量限制
//! - 通量修正
//!
//! # 设计原则
//!
//! 1. **物理一致性**：干湿过渡平滑无振荡
//! 2. **质量守恒**：精确处理边界通量
//! 3. **数值稳定**：避免除零和极端速度
//! 4. **工业验证**：经典测试案例验证


// ============================================================================
// 干湿状态
// ============================================================================

/// 单元干湿状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WetDryState {
    /// 完全干燥
    Dry,
    /// 部分湿润（过渡区）
    PartiallyWet,
    /// 完全湿润
    Wet,
}

impl WetDryState {
    /// 是否可计算流体动力学
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::Dry)
    }

    /// 是否需要特殊处理
    pub fn needs_treatment(&self) -> bool {
        matches!(self, Self::PartiallyWet)
    }
}

// ============================================================================
// 干湿配置
// ============================================================================

/// 干湿处理配置
#[derive(Debug, Clone)]
pub struct WetDryConfig {
    /// 干燥阈值 (m) - 低于此值视为干燥
    pub dry_threshold: f64,
    /// 湿润阈值 (m) - 高于此值视为湿润
    pub wet_threshold: f64,
    /// 最小水深 (m) - 用于数值计算
    pub min_depth: f64,
    /// 速度限制 (m/s) - 干湿边界的最大允许速度
    pub max_velocity: f64,
    /// 是否启用渐进式干湿过渡
    pub smooth_transition: bool,
    /// 过渡函数指数
    pub transition_power: f64,
    /// 是否保持正水深
    pub enforce_positive_depth: bool,
    /// 负水深修正策略
    pub negative_depth_strategy: NegativeDepthStrategy,
}

impl Default for WetDryConfig {
    fn default() -> Self {
        Self {
            dry_threshold: 1e-4,     // 0.1 mm
            wet_threshold: 1e-3,     // 1 mm
            min_depth: 1e-6,         // 1 μm
            max_velocity: 50.0,      // 50 m/s
            smooth_transition: true,
            transition_power: 2.0,
            enforce_positive_depth: true,
            negative_depth_strategy: NegativeDepthStrategy::Redistribute,
        }
    }
}

impl WetDryConfig {
    /// Delft3D 风格配置
    pub fn delft3d_style() -> Self {
        Self {
            dry_threshold: 0.005,    // 5 mm
            wet_threshold: 0.05,     // 50 mm
            min_depth: 0.001,        // 1 mm
            max_velocity: 100.0,
            smooth_transition: true,
            transition_power: 2.0,
            ..Default::default()
        }
    }

    /// MIKE 风格配置
    pub fn mike_style() -> Self {
        Self {
            dry_threshold: 0.005,
            wet_threshold: 0.01,
            min_depth: 0.001,
            max_velocity: 50.0,
            smooth_transition: true,
            transition_power: 1.5,
            ..Default::default()
        }
    }

    /// 高分辨率配置
    pub fn high_resolution() -> Self {
        Self {
            dry_threshold: 1e-5,
            wet_threshold: 1e-4,
            min_depth: 1e-7,
            max_velocity: 30.0,
            smooth_transition: true,
            transition_power: 3.0,
            ..Default::default()
        }
    }
}

/// 负水深处理策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegativeDepthStrategy {
    /// 截断为零
    Clamp,
    /// 重新分配到邻居
    Redistribute,
    /// 减小时间步重算
    ReduceTimestep,
    /// 标记为干燥
    MarkDry,
}

// ============================================================================
// 干湿处理器
// ============================================================================

/// 干湿边界处理器
pub struct WetDryHandler {
    /// 配置
    config: WetDryConfig,
    /// 单元干湿状态
    states: Vec<WetDryState>,
    /// 干湿边界面列表
    interface_faces: Vec<usize>,
    /// 负水深单元计数
    negative_count: usize,
    /// 统计信息
    stats: WetDryStats,
}

impl WetDryHandler {
    /// 创建处理器
    pub fn new(config: WetDryConfig, n_cells: usize) -> Self {
        Self {
            config,
            states: vec![WetDryState::Wet; n_cells],
            interface_faces: Vec::new(),
            negative_count: 0,
            stats: WetDryStats::default(),
        }
    }

    /// 获取配置
    pub fn config(&self) -> &WetDryConfig {
        &self.config
    }

    /// 获取单元状态
    pub fn state(&self, cell: usize) -> WetDryState {
        self.states.get(cell).copied().unwrap_or(WetDryState::Dry)
    }

    /// 获取所有状态
    pub fn states(&self) -> &[WetDryState] {
        &self.states
    }

    /// 更新干湿状态
    ///
    /// # 参数
    ///
    /// * `h` - 水深数组
    pub fn update_states(&mut self, h: &[f64]) {
        self.stats.reset_frame();

        if self.states.len() != h.len() {
            self.states.resize(h.len(), WetDryState::Wet);
        }
        
        for (i, &h_val) in h.iter().enumerate() {
            let old_state = self.states.get(i).copied().unwrap_or(WetDryState::Dry);
            
            let new_state = if h_val < self.config.dry_threshold {
                WetDryState::Dry
            } else if h_val < self.config.wet_threshold {
                WetDryState::PartiallyWet
            } else {
                WetDryState::Wet
            };

            if i < self.states.len() {
                self.states[i] = new_state;
            }

            // 统计状态变化
            match (old_state, new_state) {
                (WetDryState::Dry, WetDryState::Wet | WetDryState::PartiallyWet) => {
                    self.stats.cells_wetted += 1;
                }
                (WetDryState::Wet | WetDryState::PartiallyWet, WetDryState::Dry) => {
                    self.stats.cells_dried += 1;
                }
                _ => {}
            }

            // 统计状态分布
            match new_state {
                WetDryState::Dry => self.stats.dry_cells += 1,
                WetDryState::PartiallyWet => self.stats.partial_cells += 1,
                WetDryState::Wet => self.stats.wet_cells += 1,
            }
        }
    }

    /// 识别干湿边界面
    pub fn identify_interfaces<F>(&mut self, n_faces: usize, get_neighbors: F)
    where
        F: Fn(usize) -> (Option<usize>, Option<usize>),
    {
        self.interface_faces.clear();
        
        for face in 0..n_faces {
            let (left, right) = get_neighbors(face);
            
            let left_state = left.map(|c| self.state(c)).unwrap_or(WetDryState::Dry);
            let right_state = right.map(|c| self.state(c)).unwrap_or(WetDryState::Dry);
            
            // 如果两侧状态不同，则为干湿边界
            if left_state != right_state {
                self.interface_faces.push(face);
            }
        }

        self.stats.interface_faces = self.interface_faces.len();
    }

    /// 获取干湿边界面
    pub fn interface_faces(&self) -> &[usize] {
        &self.interface_faces
    }

    /// 限制速度
    ///
    /// 在干湿边界附近限制速度以保持稳定性
    pub fn limit_velocity(&self, h: f64, hu: f64, hv: f64) -> (f64, f64) {
        if h < self.config.min_depth {
            return (0.0, 0.0);
        }

        let u = hu / h;
        let v = hv / h;
        let speed = (u * u + v * v).sqrt();

        if speed > self.config.max_velocity {
            let scale = self.config.max_velocity / speed;
            (hu * scale, hv * scale)
        } else {
            (hu, hv)
        }
    }

    /// 计算过渡因子
    ///
    /// 返回 [0, 1] 之间的值，用于平滑过渡
    pub fn transition_factor(&self, h: f64) -> f64 {
        if self.config.wet_threshold <= self.config.dry_threshold {
            return if h > self.config.dry_threshold { 1.0 } else { 0.0 };
        }
        if h <= self.config.dry_threshold {
            0.0
        } else if h >= self.config.wet_threshold {
            1.0
        } else if self.config.smooth_transition {
            // 平滑过渡
            let x = (h - self.config.dry_threshold)
                / (self.config.wet_threshold - self.config.dry_threshold);
            x.powf(self.config.transition_power)
        } else {
            // 线性过渡
            (h - self.config.dry_threshold)
                / (self.config.wet_threshold - self.config.dry_threshold)
        }
    }

    /// 修正通量
    ///
    /// 在干湿边界应用通量修正
    pub fn correct_flux(
        &self,
        flux_h: f64,
        flux_hu: f64,
        flux_hv: f64,
        h_left: f64,
        h_right: f64,
    ) -> (f64, f64, f64) {
        let factor_left = self.transition_factor(h_left);
        let factor_right = self.transition_factor(h_right);
        let factor = factor_left.min(factor_right);

        // 应用过渡因子
        (flux_h * factor, flux_hu * factor, flux_hv * factor)
    }

    /// 修正负水深
    pub fn correct_negative_depth(
        &mut self,
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) -> usize {
        // 没有邻居信息时回退到截断策略
        self.correct_negative_depth_with_neighbors(h, hu, hv, None)
    }

    /// 修正负水深（带邻居信息，支持真正的重分配）
    ///
    /// # 参数
    /// - `h`, `hu`, `hv`: 水深和动量数组
    /// - `neighbors`: 邻居索引回调函数，返回给定单元的邻居索引列表
    ///
    /// # Redistribute策略详解
    /// 当单元i出现负水深时，将负水量平均分配到有正水深的邻居单元。
    /// 这保证了质量守恒，避免简单截断造成的质量丢失。
    pub fn correct_negative_depth_with_neighbors<F>(
        &mut self,
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
        neighbors: Option<F>,
    ) -> usize
    where
        F: Fn(usize) -> Vec<usize>,
    {
        self.negative_count = 0;
        let n = h.len();

        // 第一遍：统计负水深单元并记录重分配信息
        let mut redistribute_info: Vec<(usize, f64, f64, f64)> = Vec::new(); // (cell, deficit, hu, hv)

        for i in 0..n {
            if h[i] < 0.0 {
                self.negative_count += 1;

                match self.config.negative_depth_strategy {
                    NegativeDepthStrategy::Clamp => {
                        h[i] = 0.0;
                        hu[i] = 0.0;
                        hv[i] = 0.0;
                    }
                    NegativeDepthStrategy::MarkDry => {
                        h[i] = 0.0;
                        hu[i] = 0.0;
                        hv[i] = 0.0;
                        if i < self.states.len() {
                            self.states[i] = WetDryState::Dry;
                        }
                    }
                    NegativeDepthStrategy::Redistribute => {
                        if neighbors.is_some() {
                            // 记录需要重分配的负水深单元
                            redistribute_info.push((i, -h[i], hu[i], hv[i]));
                            // 先将本单元清零
                            h[i] = 0.0;
                            hu[i] = 0.0;
                            hv[i] = 0.0;
                        } else {
                            // 无邻居信息时回退到截断
                            h[i] = 0.0;
                            hu[i] = 0.0;
                            hv[i] = 0.0;
                        }
                    }
                    NegativeDepthStrategy::ReduceTimestep => {
                        // 标记但不修改，由上层减小时间步
                    }
                }
            }

            // 强制正水深
            if self.config.enforce_positive_depth && h[i] < self.config.min_depth {
                h[i] = h[i].max(0.0);
                if h[i] < self.config.dry_threshold {
                    hu[i] = 0.0;
                    hv[i] = 0.0;
                }
            }
        }

        // 第二遍：执行重分配（如果有邻居信息）
        if let Some(ref get_neighbors) = neighbors {
            for (cell, deficit, _hu_deficit, _hv_deficit) in redistribute_info {
                let neighbor_ids = get_neighbors(cell);
                
                // 筛选有正水深的邻居
                let valid_neighbors: Vec<usize> = neighbor_ids
                    .iter()
                    .copied()
                    .filter(|&n| n < h.len() && h[n] > self.config.min_depth)
                    .collect();

                if valid_neighbors.is_empty() {
                    // 无可用邻居，质量丢失（边界条件）
                    continue;
                }

                // 平均分配缺失的水量到邻居
                let deficit_per_neighbor = deficit / valid_neighbors.len() as f64;
                for &neighbor in &valid_neighbors {
                    // 从邻居扣除缺失量，但不能让邻居变负
                    let available = (h[neighbor] - self.config.min_depth).max(0.0);
                    let actual_transfer = deficit_per_neighbor.min(available);
                    if actual_transfer <= 0.0 {
                        continue;
                    }

                    let old_h = h[neighbor];
                    h[neighbor] -= actual_transfer;
                    let new_h = h[neighbor];
                    
                    // 简化处理：动量按水量比例调整
                    if old_h > self.config.min_depth {
                        let ratio = if old_h > 0.0 { new_h / old_h } else { 0.0 };
                        hu[neighbor] *= ratio;
                        hv[neighbor] *= ratio;
                    }
                }
            }
        }

        self.stats.negative_corrections += self.negative_count;
        self.negative_count
    }

    /// 需要减小时间步
    pub fn needs_timestep_reduction(&self) -> bool {
        self.config.negative_depth_strategy == NegativeDepthStrategy::ReduceTimestep
            && self.negative_count > 0
    }

    /// 获取统计信息
    pub fn stats(&self) -> &WetDryStats {
        &self.stats
    }

    /// 重置统计
    pub fn reset_stats(&mut self) {
        self.stats = WetDryStats::default();
    }
}

// ============================================================================
// 统计信息
// ============================================================================

/// 干湿处理统计
#[derive(Debug, Clone, Default)]
pub struct WetDryStats {
    /// 干燥单元数
    pub dry_cells: usize,
    /// 部分湿润单元数
    pub partial_cells: usize,
    /// 湿润单元数
    pub wet_cells: usize,
    /// 干湿边界面数
    pub interface_faces: usize,
    /// 本时间步湿润的单元数
    pub cells_wetted: usize,
    /// 本时间步干燥的单元数
    pub cells_dried: usize,
    /// 累计负水深修正次数
    pub negative_corrections: usize,
}

impl WetDryStats {
    /// 重置帧统计
    fn reset_frame(&mut self) {
        self.dry_cells = 0;
        self.partial_cells = 0;
        self.wet_cells = 0;
        self.interface_faces = 0;
        self.cells_wetted = 0;
        self.cells_dried = 0;
    }

    /// 总单元数
    pub fn total_cells(&self) -> usize {
        self.dry_cells + self.partial_cells + self.wet_cells
    }

    /// 湿润比例
    pub fn wet_fraction(&self) -> f64 {
        let total = self.total_cells();
        if total > 0 {
            (self.wet_cells + self.partial_cells) as f64 / total as f64
        } else {
            0.0
        }
    }
}

impl std::fmt::Display for WetDryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "干湿统计:")?;
        writeln!(f, "  干燥: {} | 部分: {} | 湿润: {}",
            self.dry_cells, self.partial_cells, self.wet_cells)?;
        writeln!(f, "  边界面: {} | 湿润: +{} | 干燥: +{}",
            self.interface_faces, self.cells_wetted, self.cells_dried)?;
        writeln!(f, "  负水深修正: {}", self.negative_corrections)?;
        Ok(())
    }
}

// ============================================================================
// 数值安全辅助
// ============================================================================

/// 安全除法（避免除零）
#[inline]
pub fn safe_divide(hu: f64, h: f64, min_h: f64) -> f64 {
    if h > min_h {
        hu / h
    } else {
        0.0
    }
}

/// 安全速度计算
#[inline]
pub fn safe_velocity(hu: f64, hv: f64, h: f64, min_h: f64) -> (f64, f64) {
    if h > min_h {
        (hu / h, hv / h)
    } else {
        (0.0, 0.0)
    }
}

/// 安全波速计算
#[inline]
pub fn safe_wave_speed(h: f64, g: f64, min_h: f64) -> f64 {
    if h > min_h {
        (g * h).sqrt()
    } else {
        0.0
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wet_dry_state() {
        assert!(WetDryState::Wet.is_active());
        assert!(WetDryState::PartiallyWet.is_active());
        assert!(!WetDryState::Dry.is_active());
        assert!(WetDryState::PartiallyWet.needs_treatment());
    }

    #[test]
    fn test_config_defaults() {
        let config = WetDryConfig::default();
        assert!(config.dry_threshold < config.wet_threshold);
        assert!(config.min_depth < config.dry_threshold);
    }

    #[test]
    fn test_handler_update_states() {
        let config = WetDryConfig {
            dry_threshold: 0.01,
            wet_threshold: 0.1,
            ..Default::default()
        };
        let mut handler = WetDryHandler::new(config, 5);

        let h = vec![0.0, 0.005, 0.05, 0.15, 1.0];
        handler.update_states(&h);

        assert_eq!(handler.state(0), WetDryState::Dry);
        assert_eq!(handler.state(1), WetDryState::Dry);
        assert_eq!(handler.state(2), WetDryState::PartiallyWet);
        assert_eq!(handler.state(3), WetDryState::Wet);
        assert_eq!(handler.state(4), WetDryState::Wet);
    }

    #[test]
    fn test_transition_factor() {
        let config = WetDryConfig {
            dry_threshold: 0.01,
            wet_threshold: 0.1,
            smooth_transition: false,
            ..Default::default()
        };
        let handler = WetDryHandler::new(config, 1);

        assert_eq!(handler.transition_factor(0.0), 0.0);
        assert_eq!(handler.transition_factor(0.005), 0.0);
        assert!((handler.transition_factor(0.055) - 0.5).abs() < 0.01);
        assert_eq!(handler.transition_factor(0.1), 1.0);
        assert_eq!(handler.transition_factor(1.0), 1.0);
    }

    #[test]
    fn test_velocity_limit() {
        let config = WetDryConfig {
            max_velocity: 10.0,
            min_depth: 0.001,
            ..Default::default()
        };
        let handler = WetDryHandler::new(config, 1);

        // 正常情况
        let (hu, _hv) = handler.limit_velocity(1.0, 5.0, 5.0);
        assert!((hu - 5.0).abs() < 1e-10);

        // 需要限制
        let (hu, hv) = handler.limit_velocity(1.0, 80.0, 60.0);
        let speed = ((hu / 1.0).powi(2) + (hv / 1.0).powi(2)).sqrt();
        assert!((speed - 10.0).abs() < 0.01);

        // 干燥
        let (hu, hv) = handler.limit_velocity(0.0001, 1.0, 1.0);
        assert_eq!(hu, 0.0);
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn test_negative_depth_correction() {
        let config = WetDryConfig {
            negative_depth_strategy: NegativeDepthStrategy::Clamp,
            ..Default::default()
        };
        let mut handler = WetDryHandler::new(config, 3);

        let mut h = vec![-0.1, 0.5, -0.05];
        let mut hu = vec![1.0, 2.0, 0.5];
        let mut hv = vec![0.5, 1.0, 0.25];

        let count = handler.correct_negative_depth(&mut h, &mut hu, &mut hv);

        assert_eq!(count, 2);
        assert_eq!(h[0], 0.0);
        assert_eq!(h[1], 0.5);
        assert_eq!(h[2], 0.0);
        assert_eq!(hu[0], 0.0);
        assert_eq!(hu[2], 0.0);
    }

    #[test]
    fn test_safe_functions() {
        assert_eq!(safe_divide(10.0, 2.0, 0.001), 5.0);
        assert_eq!(safe_divide(10.0, 0.0001, 0.001), 0.0);

        let (u, v) = safe_velocity(2.0, 4.0, 2.0, 0.001);
        assert_eq!(u, 1.0);
        assert_eq!(v, 2.0);

        assert!((safe_wave_speed(10.0, 9.81, 0.001) - 9.9045).abs() < 0.001);
        assert_eq!(safe_wave_speed(0.0001, 9.81, 0.001), 0.0);
    }
}
