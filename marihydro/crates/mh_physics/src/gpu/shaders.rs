// crates/mh_physics/src/gpu/shaders.rs

//! WGSL 着色器源码模块
//!
//! 使用 `include_str!` 在编译时嵌入着色器代码。
//!
//! # 着色器组织
//!
//! - `common.wgsl`: 公共类型和工具函数
//! - `gradient.wgsl`: Green-Gauss 梯度计算
//! - `limiter.wgsl`: BJ/VK 限制器
//! - `reconstruct.wgsl`: MUSCL 重构
//! - `hllc.wgsl`: HLLC 黎曼求解器
//! - `accumulate.wgsl`: 通量累积 (着色策略)
//! - `integrate.wgsl`: SSP-RK 时间积分
//! - `source.wgsl`: 源项计算 (摩擦、风、科氏力)
//! - `boundary.wgsl`: 边界条件处理
//!
//! # 着色器编译
//!
//! 着色器在运行时由 wgpu 编译。使用 `ShaderValidator` 可在开发时验证着色器语法。

/// 公共类型和工具函数
///
/// 包含:
/// - 物理常量 (G, EPS_H 等)
/// - 基本结构体 (ConservedState, PrimitiveState, FluxVector)
/// - 工具函数 (to_primitive, sound_speed, minmod 等)
pub const COMMON: &str = include_str!("shaders/common.wgsl");

/// Green-Gauss 梯度计算着色器
///
/// 入口点:
/// - `main`: 计算单元梯度
/// - `clear_gradients`: 清零梯度缓冲区
pub const GRADIENT: &str = include_str!("shaders/gradient.wgsl");

/// 限制器着色器
///
/// 支持:
/// - Barth-Jespersen 限制器
/// - Venkatakrishnan 限制器
pub const LIMITER: &str = include_str!("shaders/limiter.wgsl");

/// MUSCL 重构着色器
///
/// 入口点:
/// - `main`: 计算面重构值
/// - `wet_dry_fix`: 干湿边界修正
pub const RECONSTRUCT: &str = include_str!("shaders/reconstruct.wgsl");

/// HLLC 黎曼求解器着色器
///
/// 计算面通量和最大波速
pub const HLLC: &str = include_str!("shaders/hllc.wgsl");

/// 通量累积着色器
///
/// 入口点:
/// - `main`: 着色累积 (无竞争)
/// - `clear_residuals`: 清零残差
/// - `atomic_accumulate`: 原子累积 (备选)
pub const ACCUMULATE: &str = include_str!("shaders/accumulate.wgsl");

/// 时间积分着色器
///
/// 入口点:
/// - `euler`: 前向欧拉
/// - `ssp_rk2`: SSP-RK2 (TVD)
/// - `ssp_rk3`: SSP-RK3 (TVD)
pub const INTEGRATE: &str = include_str!("shaders/integrate.wgsl");

/// 源项着色器
///
/// 包含:
/// - 曼宁/谢才摩擦
/// - 风应力
/// - 科氏力
/// - 底床坡度源项
pub const SOURCE: &str = include_str!("shaders/source.wgsl");

/// 边界条件着色器
///
/// 支持边界类型:
/// - 固壁 (反射)
/// - 开边界 (自由出流)
/// - 水位边界
/// - 流量边界
/// - 速度边界
/// - 辐射边界
/// - 周期边界
/// - 吸收边界
pub const BOUNDARY: &str = include_str!("shaders/boundary.wgsl");

/// 着色器验证器
///
/// 在开发阶段验证着色器语法正确性
#[cfg(feature = "shader-validation")]
pub struct ShaderValidator;

#[cfg(feature = "shader-validation")]
impl ShaderValidator {
    /// 验证所有着色器
    pub fn validate_all() -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // 使用 naga 进行静态验证
        let shaders = [
            ("common", COMMON),
            ("gradient", GRADIENT),
            ("limiter", LIMITER),
            ("reconstruct", RECONSTRUCT),
            ("hllc", HLLC),
            ("accumulate", ACCUMULATE),
            ("integrate", INTEGRATE),
            ("source", SOURCE),
            ("boundary", BOUNDARY),
        ];

        for (name, source) in shaders {
            if let Err(e) = Self::validate_shader(source) {
                errors.push(format!("{}: {}", name, e));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn validate_shader(source: &str) -> Result<(), String> {
        use naga::front::wgsl;
        use naga::valid::{Capabilities, ValidationFlags, Validator};

        let module = wgsl::parse_str(source).map_err(|e| format!("Parse error: {:?}", e))?;

        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        validator
            .validate(&module)
            .map_err(|e| format!("Validation error: {:?}", e))?;

        Ok(())
    }
}

/// 获取组合的着色器源码
///
/// 将 common.wgsl 与指定着色器组合，用于需要公共定义的场景
pub fn combined_shader(shader: &str) -> String {
    format!("{}\n\n{}", COMMON, shader)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shaders_not_empty() {
        assert!(!COMMON.is_empty());
        assert!(!GRADIENT.is_empty());
        assert!(!LIMITER.is_empty());
        assert!(!RECONSTRUCT.is_empty());
        assert!(!HLLC.is_empty());
        assert!(!ACCUMULATE.is_empty());
        assert!(!INTEGRATE.is_empty());
        assert!(!SOURCE.is_empty());
        assert!(!BOUNDARY.is_empty());
    }

    #[test]
    fn test_common_contains_constants() {
        assert!(COMMON.contains("const G:"));
        assert!(COMMON.contains("const EPS_H:"));
        assert!(COMMON.contains("struct ConservedState"));
    }

    #[test]
    fn test_gradient_has_entry_points() {
        assert!(GRADIENT.contains("fn main"));
    }

    #[test]
    fn test_integrate_has_rk_methods() {
        assert!(INTEGRATE.contains("fn euler"));
        assert!(INTEGRATE.contains("fn ssp_rk2") || INTEGRATE.contains("ssp_rk2"));
    }

    #[test]
    fn test_combined_shader() {
        let combined = combined_shader(GRADIENT);
        assert!(combined.contains("const G:"));
        assert!(combined.contains("@compute"));
    }
}
