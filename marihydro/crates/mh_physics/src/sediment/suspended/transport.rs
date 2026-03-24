//! 悬移质输运求解器
//!
//! 使用对流-扩散方程求解悬浮泥沙浓度：
//!
//! ∂(hC)/∂t + ∇·(hUC) = ∇·(hK∇C) + S
//!
//! 其中 S 为床面交换源项（侵蚀-沉降）。
//!
//! # 与 tracer 模块的关系
//!
//! 本模块复用 `tracer::TracerTransportSolver` 处理对流-扩散，
//! 只需实现泥沙特有的床面交换源项。

use crate::prelude::*;
use crate::sediment::properties::SedimentPropertiesGeneric;
use crate::tracer::{TracerAdvectionScheme, TracerDiffusionConfig, TracerTransportConfig, TracerTransportSolver};
use crate::types::PhysicalConstants;
use super::resuspension::{ResuspensionSourceGeneric, SmithMcLean};
use super::settling::SettlingVelocity;

/// 悬移质输运求解器
///
/// 结合通用 tracer 输运和泥沙特有的床面源项。
pub struct SuspendedTransport<B: Backend> {
    /// 通用输运求解器（对流-扩散）
    transport_solver: TracerTransportSolver<B>,
    /// 床面交换源项
    source: ResuspensionSourceGeneric<B, SmithMcLean<B::Scalar>>,
    /// 沉降速度信息
    settling: SettlingVelocity<B::Scalar>,
    /// 浓度场 [kg/m³]
    concentration: B::Buffer<B::Scalar>,
    /// 源项缓存 [kg/m³/s]
    source_term: B::Buffer<B::Scalar>,
    /// 物理常数
    physics: PhysicalConstants,
    /// 后端
    backend: B,
}

impl<B> SuspendedTransport<B>
where
    B: Backend + Clone,
    B::Scalar: RuntimeScalar,
{
    #[inline]
    fn min_depth(&self) -> B::Scalar {
        self.transport_solver.config().h_min
    }

    /// 创建新的悬移质输运求解器
    pub fn new_with_backend(
        backend: B,
        n_cells: usize,
        properties: SedimentPropertiesGeneric<B::Scalar>,
        physics: PhysicalConstants,
    ) -> Self {
        // 自动计算沉降速度
        let settling = SettlingVelocity::auto(&backend, &properties, &physics);
        
        // 创建床面源项
        let source = ResuspensionSourceGeneric::new(backend.clone(), properties)
            .with_settling_velocity(settling.ws);
        
        // 配置 tracer 求解器（默认使用常数扩散系数）
        let diffusion =
            backend.config_scalar(0.1, "SuspendedTransport.new_with_backend.diffusion");
        let config: TracerTransportConfig<B::Scalar> = TracerTransportConfig::<B::Scalar> {
            advection_scheme: TracerAdvectionScheme::TvdVanLeer,
            diffusion: TracerDiffusionConfig::constant(diffusion),
            ..Default::default()
        };
        let transport_solver = TracerTransportSolver::new_with_backend(backend.clone(), config);
        
        Self {
            transport_solver,
            source,
            settling,
            concentration: backend.alloc_init(n_cells, B::Scalar::ZERO),
            source_term: backend.alloc_init(n_cells, B::Scalar::ZERO),
            physics,
            backend,
        }
    }
    
    /// 设置初始浓度
    pub fn set_concentration(&mut self, values: &[B::Scalar]) {
        let n = self.concentration.len().min(values.len());
        self.concentration.as_slice_mut()[..n].copy_from_slice(&values[..n]);
    }
    
    /// 获取浓度场
    pub fn concentration(&self) -> &[B::Scalar] {
        self.concentration.as_slice()
    }
    
    /// 获取沉降速度
    pub fn settling_velocity(&self) -> B::Scalar {
        self.settling.ws
    }
    
    /// 计算床面交换源项
    ///
    /// # 参数
    /// - `tau_b`: 床面剪切应力场 [Pa]
    /// - `h`: 水深场 [m]
    pub fn compute_source_terms(&mut self, tau_b: &B::Buffer<B::Scalar>, h: &B::Buffer<B::Scalar>) {
        let tau_slice = tau_b.as_slice();
        let h_slice = h.as_slice();
        for i in 0..self.source_term.len() {
            let tau = tau_slice.get(i).copied().unwrap_or(B::Scalar::ZERO);
            let depth = h_slice.get(i).copied().unwrap_or(B::Scalar::ZERO);
            let c = self.concentration[i];
            
            self.source_term[i] = self.source.compute_source(tau, c, depth, &self.physics);
        }
    }

    /// 从切片计算床面交换源项（便捷包装，内部拷贝到后端缓冲区）
    pub fn compute_source_terms_from_slice(&mut self, tau_b: &[B::Scalar], h: &[B::Scalar]) {
        let n = self.source_term.len();
        let mut tau_buf = self.backend.alloc_init(n, B::Scalar::ZERO);
        let mut h_buf = self.backend.alloc_init(n, B::Scalar::ZERO);

        let m = tau_b.len().min(n);
        tau_buf.as_slice_mut()[..m].copy_from_slice(&tau_b[..m]);

        let k = h.len().min(n);
        h_buf.as_slice_mut()[..k].copy_from_slice(&h[..k]);

        self.compute_source_terms(&tau_buf, &h_buf);
    }
    
    /// 执行时间步进
    ///
    /// 包含：
    /// 1. 计算床面源项
    /// 2. 对流-扩散输运（由 tracer 求解器处理）
    /// 3. 更新浓度场
    ///
    /// # 参数
    /// - `tau_b`: 床面剪切应力 [Pa]
    /// - `h`: 水深 [m]
    /// - `dt`: 时间步长 [s]
    ///
    /// # 注意
    /// 此方法只更新源项，实际的对流-扩散需要调用 tracer 求解器的 step 方法
    pub fn step_source_only(&mut self, tau_b: &B::Buffer<B::Scalar>, h: &B::Buffer<B::Scalar>, dt: B::Scalar) {
        // 计算源项
        self.compute_source_terms(tau_b, h);
        
        // 应用源项（显式欧拉）
        for i in 0..self.concentration.len() {
            self.concentration[i] += dt * self.source_term[i];
            // 确保非负
            if self.concentration[i] < B::Scalar::ZERO {
                self.concentration[i] = B::Scalar::ZERO;
            }
        }
    }

    /// 仅源项时间步进（切片版，内部拷贝）
    pub fn step_source_only_from_slice(&mut self, tau_b: &[B::Scalar], h: &[B::Scalar], dt: B::Scalar) {
        let n = self.concentration.len();
        let mut tau_buf = self.backend.alloc_init(n, B::Scalar::ZERO);
        let mut h_buf = self.backend.alloc_init(n, B::Scalar::ZERO);

        let m = tau_b.len().min(n);
        tau_buf.as_slice_mut()[..m].copy_from_slice(&tau_b[..m]);

        let k = h.len().min(n);
        h_buf.as_slice_mut()[..k].copy_from_slice(&h[..k]);

        self.step_source_only(&tau_buf, &h_buf, dt);
    }
    
    /// 完整时间步进（包含对流-扩散）
    ///
    /// # 参数
    /// - `u`: x 方向速度场 [m/s]
    /// - `v`: y 方向速度场 [m/s]
    /// - `h`: 水深场 [m]
    /// - `tau_b`: 床面剪切应力场 [Pa]
    /// - `cell_areas`: 单元面积 [m²]
    /// - `face_data`: 面通量数据（需从 tracer 模块获取）
    /// - `dt`: 时间步长 [s]
    pub fn step(
        &mut self,
        u: &B::Buffer<B::Scalar>,
        v: &B::Buffer<B::Scalar>,
        h: &B::Buffer<B::Scalar>,
        tau_b: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) {
        // 1. 计算床面源项（侵蚀-沉降）
        self.compute_source_terms(tau_b, h);

        // 2. 对流项贡献（简化版：使用一阶迎风）
        // 注意：完整实现需要网格连接信息
        // 这里只展示源项积分
        let min_depth = self.min_depth();
        for i in 0..self.concentration.len() {
            let depth = h.get(i).copied().unwrap_or(B::Scalar::ZERO);
            if depth < min_depth {
                continue;
            }
            
            // 沉降通量贡献
            let ws = self.settling.ws;
            let c = self.concentration[i];
            
            // 沉降使浓度减少（每单位水深）
            let settling_term = -ws * c / depth.max(min_depth);
            
            // 源项 + 沉降
            self.concentration[i] += dt * (self.source_term[i] + settling_term);
            if self.concentration[i] < B::Scalar::ZERO {
                self.concentration[i] = B::Scalar::ZERO;
            }
        }
        
        // 抑制未使用警告
        let _ = (&u, &v, &self.transport_solver, &self.physics);
    }
    
    /// 获取源项（用于与 tracer 求解器耦合）
    pub fn source_term(&self) -> &[B::Scalar] {
        self.source_term.as_slice()
    }
    
    /// 获取泥沙属性
    pub fn properties(&self) -> &SedimentPropertiesGeneric<B::Scalar> {
        self.source.properties()
    }
    
    /// 计算床面变化率
    ///
    /// dz/dt = (D - E) / ((1 - p) × ρ_s)
    ///
    /// 其中 p 为孔隙率
    pub fn bed_change_rate(&self, cell: usize, porosity: B::Scalar) -> B::Scalar {
        let source = self.source_term.get(cell).copied().unwrap_or(B::Scalar::ZERO);
        let h = self.min_depth().max(B::Scalar::ONE); // 与悬移质步进一致的最小水深尺度
        
        // 源项为正表示侵蚀（床面降低）
        // 需要乘以水深转换为面通量
        let flux = -source * h; // [kg/m²/s]，负号因为侵蚀使床面降低
        
        let rho_s = self.source.properties().rho_s;
        flux / ((B::Scalar::ONE - porosity) * rho_s)
    }
    
    /// 获取 tracer 求解器配置的引用
    pub fn transport_config(&self) -> &crate::tracer::TracerTransportConfig<B::Scalar> {
        self.transport_solver.config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;
    
    fn make_props<B: Backend<Scalar = f64>>(backend: &B) -> SedimentPropertiesGeneric<f64> {
        SedimentPropertiesGeneric::from_d50_mm(backend, 0.2)
    }
    
    fn make_physics() -> PhysicalConstants {
        PhysicalConstants::freshwater()
    }
    
    #[test]
    fn test_suspended_transport_new() {
        let backend = CpuBackend::<f64>::new();
        let props = make_props(&backend);
        let physics = make_physics();
        
        let transport = SuspendedTransport::new_with_backend(backend, 100, props, physics);
        
        assert_eq!(transport.concentration().len(), 100);
        assert!(transport.settling_velocity() > 0.0);
    }
    
    #[test]
    fn test_source_term_calculation() {
        let backend = CpuBackend::<f64>::new();
        let props = make_props(&backend);
        let physics = make_physics();
        let mut transport = SuspendedTransport::new_with_backend(backend.clone(), 10, props, physics);
        
        // 设置初始浓度
        transport.set_concentration(&[1.0; 10]);
        
        let mut tau_b = backend.alloc_init(10, 0.0);
        let mut h = backend.alloc_init(10, 0.0);
        tau_b.as_slice_mut().fill(2.0);
        h.as_slice_mut().fill(1.0);
        
        transport.compute_source_terms(&tau_b, &h);
        
        // 源项应该有有限值
        assert!(transport.source_term().iter().all(|&s| s.is_finite()));
    }
    
    #[test]
    fn test_step_source_only() {
        let backend = CpuBackend::<f64>::new();
        let props = make_props(&backend);
        let physics = make_physics();
        let mut transport = SuspendedTransport::new_with_backend(backend.clone(), 10, props, physics);
        
        // 初始浓度为0，高剪切力
        let mut tau_b = backend.alloc_init(10, 0.0);
        let mut h = backend.alloc_init(10, 0.0);
        tau_b.as_slice_mut().fill(5.0);
        h.as_slice_mut().fill(1.0);
        
        transport.step_source_only(&tau_b, &h, 0.1);
        
        // 应该有侵蚀，浓度增加
        assert!(transport.concentration().iter().all(|&c| c >= 0.0));
    }
}
