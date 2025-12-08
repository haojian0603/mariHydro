//! 算子抽象层
//!
//! 定义物理算子的通用接口，支持不同的通量格式和时间积分方法。
//!
//! # 设计说明
//!
//! - FluxOperator: 通量计算接口（Riemann求解器等）
//! - SourceOperator: 源项计算接口（摩擦、风等）
//! - TimeIntegrator: 时间积分接口（显式/隐式）



/// 通量算子 trait
///
/// 计算面通量的抽象接口。
pub trait FluxOperator: Send + Sync {
    /// 算子名称
    fn name(&self) -> &str;
    
    /// 计算面通量
    /// 
    /// # 参数
    /// - mesh: 网格
    /// - h_l, h_r: 左右单元水深
    /// - qx_l, qx_r: 左右单元 x 流量
    /// - qy_l, qy_r: 左右单元 y 流量
    /// - z_l, z_r: 左右单元底床高程
    /// - nx, ny: 面法向量
    /// 
    /// # 返回
    /// (flux_h, flux_qx, flux_qy) 三个守恒量的通量
    fn compute_flux(
        &self,
        h_l: f64, h_r: f64,
        qx_l: f64, qx_r: f64,
        qy_l: f64, qy_r: f64,
        z_l: f64, z_r: f64,
        nx: f64, ny: f64,
    ) -> (f64, f64, f64);
    
    /// 最大波速（用于 CFL 条件）
    fn max_wave_speed(
        &self,
        h_l: f64, h_r: f64,
        qx_l: f64, qx_r: f64,
        qy_l: f64, qy_r: f64,
    ) -> f64;
}

/// 源项算子 trait
///
/// 计算源项（摩擦、风力、科氏力等）。
pub trait SourceOperator: Send + Sync {
    /// 算子名称
    fn name(&self) -> &str;
    
    /// 计算源项
    ///
    /// # 参数
    /// - cell: 单元索引
    /// - h: 水深
    /// - qx, qy: 流量
    /// - params: 额外参数（如 Manning n）
    ///
    /// # 返回
    /// (source_h, source_qx, source_qy)
    fn compute_source(
        &self,
        cell: usize,
        h: f64,
        qx: f64,
        qy: f64,
        params: &SourceParams,
    ) -> (f64, f64, f64);
}

/// 源项参数
#[derive(Debug, Clone, Default)]
pub struct SourceParams {
    /// 曼宁糙率
    pub manning_n: f64,
    /// 底床坡度 x
    pub dz_dx: f64,
    /// 底床坡度 y
    pub dz_dy: f64,
    /// 风速 x
    pub wind_x: f64,
    /// 风速 y
    pub wind_y: f64,
    /// 单元面积
    pub area: f64,
}

/// 时间积分器 trait
///
/// 执行时间步进的抽象接口。
pub trait TimeIntegrator: Send + Sync {
    /// 积分器名称
    fn name(&self) -> &str;
    
    /// 积分阶数
    fn order(&self) -> usize;
    
    /// 是否为隐式方法
    fn is_implicit(&self) -> bool { false }
}

/// 边界条件算子 trait
pub trait BoundaryOperator: Send + Sync {
    /// 边界类型名称
    fn name(&self) -> &str;
    
    /// 应用边界条件
    ///
    /// # 参数
    /// - face: 边界面索引
    /// - h_in: 内部水深
    /// - qx_in, qy_in: 内部流量
    /// - time: 当前时间
    ///
    /// # 返回
    /// (h_ghost, qx_ghost, qy_ghost) 虚拟单元的状态
    fn apply(
        &self,
        face: usize,
        h_in: f64,
        qx_in: f64,
        qy_in: f64,
        time: f64,
    ) -> (f64, f64, f64);
}

/// 算子工厂
///
/// 注册和创建算子实例。
pub struct OperatorFactory {
    /// 已注册的通量算子名称
    flux_ops: Vec<String>,
    /// 已注册的源项算子名称
    source_ops: Vec<String>,
}

impl Default for OperatorFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorFactory {
    /// 创建新工厂
    pub fn new() -> Self {
        Self {
            flux_ops: vec!["HLL".into(), "Roe".into(), "Rusanov".into()],
            source_ops: vec!["Manning".into(), "Coriolis".into(), "Wind".into()],
        }
    }

    /// 获取可用的通量算子名称
    pub fn available_flux_operators(&self) -> &[String] {
        &self.flux_ops
    }

    /// 获取可用的源项算子名称
    pub fn available_source_operators(&self) -> &[String] {
        &self.source_ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_factory() {
        let factory = OperatorFactory::new();
        assert!(factory.available_flux_operators().contains(&"HLL".to_string()));
        assert!(factory.available_source_operators().contains(&"Manning".to_string()));
    }

    #[test]
    fn test_source_params() {
        let params = SourceParams {
            manning_n: 0.03,
            dz_dx: 0.001,
            dz_dy: 0.0,
            wind_x: 5.0,
            wind_y: 0.0,
            area: 100.0,
        };
        assert!((params.manning_n - 0.03).abs() < 1e-10);
    }
}
