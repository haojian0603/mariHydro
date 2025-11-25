// src-tauri/src/marihydro/geo/transform.rs

use crate::marihydro::geo::crs::Crs;
use crate::marihydro::infra::error::{MhError, MhResult};
use log::{debug, warn};
use proj::Proj;
use rayon::prelude::*;

// 常量定义
const ANGLE_THRESHOLD: f64 = 1e-6; // 小于此角度视为无需旋转
const GEOGRAPHIC_EPSILON: f64 = 1e-5; // 地理坐标系的数值微分步长 (~1m)
const PROJECTED_EPSILON: f64 = 1.0; // 投影坐标系的数值微分步长 (1m)
const PARALLEL_THRESHOLD: usize = 1000; // 批量操作的并行阈值

/// 地理空间变换器
///
/// # 职责
/// 1. 点坐标转换（正向/反向）
/// 2. 计算子午线收敛角
/// 3. 矢量场旋转修正
///
/// # 坐标系约定
/// - Source: 通常是气象数据的坐标系（WGS84 经纬度）
/// - Target: 通常是模型网格的坐标系（UTM 或其他投影）
pub struct GeoTransformer {
    source_crs: Crs,
    target_crs: Crs,

    /// 预编译的 Proj 实例
    forward_proj: Proj,
    inverse_proj: Proj,

    /// 源坐标系是否为地理坐标
    source_is_geographic: bool,

    /// 目标坐标系是否为地理坐标
    target_is_geographic: bool,
}

impl GeoTransformer {
    /// 创建坐标变换器
    pub fn new(source: &Crs, target: &Crs) -> MhResult<Self> {
        debug!(
            "创建坐标变换器: {} -> {}",
            source.definition, target.definition
        );

        if source.definition == target.definition {
            debug!("源和目标坐标系相同，将使用恒等变换");
        }

        let forward =
            Proj::new_known_crs(&source.definition, &target.definition, None).map_err(|e| {
                MhError::Projection(format!(
                    "正向投影初始化失败 ({} -> {}): {}",
                    source.definition, target.definition, e
                ))
            })?;

        let inverse =
            Proj::new_known_crs(&target.definition, &source.definition, None).map_err(|e| {
                MhError::Projection(format!(
                    "反向投影初始化失败 ({} -> {}): {}",
                    target.definition, source.definition, e
                ))
            })?;

        let source_is_geographic = source.is_geographic();
        let target_is_geographic = target.is_geographic();

        Ok(Self {
            source_crs: source.clone(),
            target_crs: target.clone(),
            forward_proj: forward,
            inverse_proj: inverse,
            source_is_geographic,
            target_is_geographic,
        })
    }

    // --- 单点坐标转换 ---

    /// 正向变换: Source -> Target
    #[inline]
    pub fn transform_point(&self, x: f64, y: f64) -> MhResult<(f64, f64)> {
        self.forward_proj
            .convert((x, y))
            .map_err(|e| MhError::Projection(format!("坐标转换失败 ({:.6}, {:.6}): {}", x, y, e)))
    }

    /// 反向变换: Target -> Source
    #[inline]
    pub fn inverse_transform_point(&self, x: f64, y: f64) -> MhResult<(f64, f64)> {
        self.inverse_proj.convert((x, y)).map_err(|e| {
            MhError::Projection(format!("反向坐标转换失败 ({:.6}, {:.6}): {}", x, y, e))
        })
    }

    // --- 批量坐标转换 ---

    /// 批量正向转换
    pub fn transform_points(&self, points: &[(f64, f64)]) -> MhResult<Vec<(f64, f64)>> {
        if points.len() >= PARALLEL_THRESHOLD {
            points
                .par_iter()
                .map(|&(x, y)| self.transform_point(x, y))
                .collect()
        } else {
            points
                .iter()
                .map(|&(x, y)| self.transform_point(x, y))
                .collect()
        }
    }

    /// 批量反向转换
    pub fn inverse_transform_points(&self, points: &[(f64, f64)]) -> MhResult<Vec<(f64, f64)>> {
        if points.len() >= PARALLEL_THRESHOLD {
            points
                .par_iter()
                .map(|&(x, y)| self.inverse_transform_point(x, y))
                .collect()
        } else {
            points
                .iter()
                .map(|&(x, y)| self.inverse_transform_point(x, y))
                .collect()
        }
    }

    // --- 收敛角计算 ---

    /// 计算子午线收敛角 (Meridian Convergence Angle)
    ///
    /// # 定义
    /// 返回真北方向相对于网格北（Y轴）的夹角（弧度）。
    /// - 正值：真北在网格北的东侧（顺时针偏转）
    /// - 负值：真北在网格北的西侧（逆时针偏转）
    ///
    /// # 用途
    /// 修正气象矢量：气象数据通常基于真北定义（地理坐标系），
    /// 而模型网格使用投影坐标系，两者的"北"不一致。
    ///
    /// # 算法
    /// 1. 从目标坐标反算回源坐标（通常是经纬度）
    /// 2. 在源坐标系中向北移动一小步
    /// 3. 正向投影回目标坐标系
    /// 4. 计算位移向量相对于Y轴的偏角
    pub fn calculate_convergence_angle(&self, x_target: f64, y_target: f64) -> MhResult<f64> {
        // 1. 反算到源坐标系
        let (x_src, y_src) = self.inverse_transform_point(x_target, y_target)?;

        // 2. 确定数值微分步长
        let epsilon = if self.source_is_geographic {
            GEOGRAPHIC_EPSILON
        } else {
            PROJECTED_EPSILON
        };

        // 3. 在源坐标系中向"正北"（Y轴正方向）移动
        let (x_north, y_north) = (x_src, y_src + epsilon);

        // 4. 投影回目标坐标系
        let (x_target_north, y_target_north) = self.transform_point(x_north, y_north)?;

        // 5. 计算位移向量
        let dx = x_target_north - x_target;
        let dy = y_target_north - y_target;

        // 6. 计算收敛角
        // atan2(dx, dy) 计算向量 (dx, dy) 相对于 Y 轴的角度
        let angle = dx.atan2(dy);

        // 检查异常值
        if !angle.is_finite() {
            warn!(
                "收敛角计算异常: 位置=({:.6}, {:.6}), dx={:.6}, dy={:.6}",
                x_target, y_target, dx, dy
            );
            return Ok(0.0);
        }

        Ok(angle)
    }

    /// 批量计算收敛角
    pub fn calculate_convergence_angles(&self, points: &[(f64, f64)]) -> MhResult<Vec<f32>> {
        if points.len() >= PARALLEL_THRESHOLD {
            points
                .par_iter()
                .map(|&(x, y)| self.calculate_convergence_angle(x, y).map(|a| a as f32))
                .collect()
        } else {
            points
                .iter()
                .map(|&(x, y)| self.calculate_convergence_angle(x, y).map(|a| a as f32))
                .collect()
        }
    }

    // --- 矢量旋转 ---

    /// 原位旋转矢量场
    ///
    /// # 旋转公式
    /// 标准旋转矩阵（逆时针旋转角度 θ）：
    /// ```text
    /// [u']   [cos θ  -sin θ] [u]
    /// [v'] = [sin θ   cos θ] [v]
    /// ```
    ///
    /// 即：
    /// - u' = u * cos(θ) - v * sin(θ)
    /// - v' = u * sin(θ) + v * cos(θ)
    pub fn rotate_vectors_in_place(
        &self,
        u: &mut [f64],
        v: &mut [f64],
        angles: &[f32],
    ) -> MhResult<()> {
        // 维度检查
        if u.len() != v.len() || u.len() != angles.len() {
            return Err(MhError::InvalidMesh(format!(
                "矢量旋转维度不匹配: u={}, v={}, angles={}",
                u.len(),
                v.len(),
                angles.len()
            )));
        }

        if u.len() >= PARALLEL_THRESHOLD {
            self.rotate_vectors_parallel(u, v, angles);
        } else {
            self.rotate_vectors_serial(u, v, angles);
        }

        Ok(())
    }

    fn rotate_vectors_serial(&self, u: &mut [f64], v: &mut [f64], angles: &[f32]) {
        for i in 0..u.len() {
            let angle = angles[i] as f64;

            // 跳过小角度
            if angle.abs() < ANGLE_THRESHOLD {
                continue;
            }

            // 跳过 NaN 值
            if u[i].is_nan() || v[i].is_nan() {
                continue;
            }

            let (sin_a, cos_a) = angle.sin_cos();
            let u_old = u[i];
            let v_old = v[i];

            // 应用标准旋转矩阵
            u[i] = u_old * cos_a - v_old * sin_a;
            v[i] = u_old * sin_a + v_old * cos_a;
        }
    }

    fn rotate_vectors_parallel(&self, u: &mut [f64], v: &mut [f64], angles: &[f32]) {
        u.par_iter_mut()
            .zip(v.par_iter_mut())
            .zip(angles.par_iter())
            .for_each(|((u_val, v_val), &angle)| {
                let angle = angle as f64;

                if angle.abs() < ANGLE_THRESHOLD || u_val.is_nan() || v_val.is_nan() {
                    return;
                }

                let (sin_a, cos_a) = angle.sin_cos();
                let u_old = *u_val;
                let v_old = *v_val;

                *u_val = u_old * cos_a - v_old * sin_a;
                *v_val = u_old * sin_a + v_old * cos_a;
            });
    }

    /// 创建旋转后的新矢量（非原位）
    pub fn rotate_vectors(
        &self,
        u: &[f64],
        v: &[f64],
        angles: &[f32],
    ) -> MhResult<(Vec<f64>, Vec<f64>)> {
        let mut u_new = u.to_vec();
        let mut v_new = v.to_vec();
        self.rotate_vectors_in_place(&mut u_new, &mut v_new, angles)?;
        Ok((u_new, v_new))
    }

    // --- 辅助方法 ---

    /// 检查是否为恒等变换
    pub fn is_identity(&self) -> bool {
        self.source_crs.definition == self.target_crs.definition
    }

    pub fn source_crs(&self) -> &Crs {
        &self.source_crs
    }

    pub fn target_crs(&self) -> &Crs {
        &self.target_crs
    }

    pub fn is_source_geographic(&self) -> bool {
        self.source_is_geographic
    }

    pub fn is_target_geographic(&self) -> bool {
        self.target_is_geographic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let crs = Crs::wgs84();
        let transformer = GeoTransformer::new(&crs, &crs).unwrap();

        assert!(transformer.is_identity());

        let (x, y) = transformer.transform_point(120.0, 30.0).unwrap();
        assert!((x - 120.0).abs() < 1e-10);
        assert!((y - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_rotation() {
        let crs = Crs::wgs84();
        let transformer = GeoTransformer::new(&crs, &crs).unwrap();

        let mut u = vec![1.0, 0.0];
        let mut v = vec![0.0, 1.0];
        let angles = vec![std::f32::consts::FRAC_PI_2, 0.0]; // 90度, 0度

        transformer
            .rotate_vectors_in_place(&mut u, &mut v, &angles)
            .unwrap();

        // 第一个向量旋转90度: (1,0) -> (0,1)
        assert!((u[0] - 0.0).abs() < 1e-6);
        assert!((v[0] - 1.0).abs() < 1e-6);

        // 第二个向量不旋转: (0,1) -> (0,1)
        assert!((u[1] - 0.0).abs() < 1e-6);
        assert!((v[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotation_with_nan() {
        let crs = Crs::wgs84();
        let transformer = GeoTransformer::new(&crs, &crs).unwrap();

        let mut u = vec![1.0, f64::NAN];
        let mut v = vec![0.0, 1.0];
        let angles = vec![std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2];

        transformer
            .rotate_vectors_in_place(&mut u, &mut v, &angles)
            .unwrap();

        // NaN 值不应被旋转
        assert!(u[1].is_nan());
    }

    #[test]
    fn test_dimension_mismatch() {
        let crs = Crs::wgs84();
        let transformer = GeoTransformer::new(&crs, &crs).unwrap();

        let mut u = vec![1.0, 2.0];
        let mut v = vec![3.0];
        let angles = vec![0.0, 0.0];

        let result = transformer.rotate_vectors_in_place(&mut u, &mut v, &angles);
        assert!(result.is_err());
    }

    #[test]
    fn test_small_angle_optimization() {
        let crs = Crs::wgs84();
        let transformer = GeoTransformer::new(&crs, &crs).unwrap();

        let mut u = vec![1.0];
        let mut v = vec![0.0];
        let angles = vec![1e-7]; // 非常小的角度

        let u_before = u[0];
        let v_before = v[0];

        transformer
            .rotate_vectors_in_place(&mut u, &mut v, &angles)
            .unwrap();

        // 小角度应该被跳过
        assert_eq!(u[0], u_before);
        assert_eq!(v[0], v_before);
    }

    #[test]
    fn test_batch_transform() {
        let crs = Crs::wgs84();
        let transformer = GeoTransformer::new(&crs, &crs).unwrap();

        let points = vec![(120.0, 30.0), (121.0, 31.0)];
        let transformed = transformer.transform_points(&points).unwrap();

        assert_eq!(transformed.len(), 2);
        assert!((transformed[0].0 - 120.0).abs() < 1e-10);
        assert!((transformed[1].1 - 31.0).abs() < 1e-10);
    }
}
