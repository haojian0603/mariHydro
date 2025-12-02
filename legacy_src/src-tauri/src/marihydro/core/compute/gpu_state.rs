// src-tauri/src/marihydro/core/compute/gpu_state.rs

//! GPU友好的状态数据结构
//!
//! 将浅水方程状态变量转换为SoA（Structure of Arrays）格式，
//! 以满足GPU的内存对齐和访问模式要求。

#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};

/// GPU状态数据（SoA格式）
///
/// 存储浅水方程的守恒变量，使用独立的f32数组而非DVec2
#[derive(Debug, Clone)]
pub struct GpuStateArrays {
    /// 水深 h [m]
    pub h: Vec<f32>,
    /// x方向动量 hu [m²/s]
    pub hu: Vec<f32>,
    /// y方向动量 hv [m²/s]
    pub hv: Vec<f32>,
    /// 底床高程 z_bed [m]
    pub z_bed: Vec<f32>,
}

impl GpuStateArrays {
    /// 创建指定大小的状态数组
    pub fn new(n_cells: usize) -> Self {
        Self {
            h: vec![0.0; n_cells],
            hu: vec![0.0; n_cells],
            hv: vec![0.0; n_cells],
            z_bed: vec![0.0; n_cells],
        }
    }

    /// 从f64数组转换
    pub fn from_f64(h: &[f64], hu: &[f64], hv: &[f64], z_bed: &[f64]) -> Self {
        let n = h.len();
        debug_assert_eq!(hu.len(), n);
        debug_assert_eq!(hv.len(), n);
        debug_assert_eq!(z_bed.len(), n);

        Self {
            h: h.iter().map(|&x| x as f32).collect(),
            hu: hu.iter().map(|&x| x as f32).collect(),
            hv: hv.iter().map(|&x| x as f32).collect(),
            z_bed: z_bed.iter().map(|&x| x as f32).collect(),
        }
    }

    /// 单元数量
    #[inline]
    pub fn len(&self) -> usize {
        self.h.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.h.is_empty()
    }

    /// 重置所有值为零
    pub fn reset(&mut self) {
        self.h.iter_mut().for_each(|x| *x = 0.0);
        self.hu.iter_mut().for_each(|x| *x = 0.0);
        self.hv.iter_mut().for_each(|x| *x = 0.0);
    }

    /// 复制到f64数组
    pub fn to_f64(&self, h: &mut [f64], hu: &mut [f64], hv: &mut [f64]) {
        let n = self.h.len();
        debug_assert_eq!(h.len(), n);
        debug_assert_eq!(hu.len(), n);
        debug_assert_eq!(hv.len(), n);

        for i in 0..n {
            h[i] = self.h[i] as f64;
            hu[i] = self.hu[i] as f64;
            hv[i] = self.hv[i] as f64;
        }
    }

    /// GPU内存估计（字节）
    pub fn gpu_memory_estimate(&self) -> usize {
        self.h.len() * 4 * std::mem::size_of::<f32>()
    }
}

/// GPU通量累加器（SoA格式）
#[derive(Debug, Clone)]
pub struct GpuFluxArrays {
    /// 质量通量 [m³/s]
    pub flux_h: Vec<f32>,
    /// x动量通量 [m⁴/s²]
    pub flux_hu: Vec<f32>,
    /// y动量通量 [m⁴/s²]
    pub flux_hv: Vec<f32>,
}

impl GpuFluxArrays {
    /// 创建指定大小的通量数组
    pub fn new(n_cells: usize) -> Self {
        Self {
            flux_h: vec![0.0; n_cells],
            flux_hu: vec![0.0; n_cells],
            flux_hv: vec![0.0; n_cells],
        }
    }

    /// 重置所有值为零
    pub fn reset(&mut self) {
        self.flux_h.iter_mut().for_each(|x| *x = 0.0);
        self.flux_hu.iter_mut().for_each(|x| *x = 0.0);
        self.flux_hv.iter_mut().for_each(|x| *x = 0.0);
    }

    /// 单元数量
    #[inline]
    pub fn len(&self) -> usize {
        self.flux_h.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.flux_h.is_empty()
    }
}

/// GPU源项数组（SoA格式）
#[derive(Debug, Clone)]
pub struct GpuSourceArrays {
    /// 质量源项
    pub source_h: Vec<f32>,
    /// x动量源项
    pub source_hu: Vec<f32>,
    /// y动量源项
    pub source_hv: Vec<f32>,
}

impl GpuSourceArrays {
    /// 创建指定大小的源项数组
    pub fn new(n_cells: usize) -> Self {
        Self {
            source_h: vec![0.0; n_cells],
            source_hu: vec![0.0; n_cells],
            source_hv: vec![0.0; n_cells],
        }
    }

    /// 重置所有值为零
    pub fn reset(&mut self) {
        self.source_h.iter_mut().for_each(|x| *x = 0.0);
        self.source_hu.iter_mut().for_each(|x| *x = 0.0);
        self.source_hv.iter_mut().for_each(|x| *x = 0.0);
    }

    /// 单元数量
    #[inline]
    pub fn len(&self) -> usize {
        self.source_h.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.source_h.is_empty()
    }
}

/// GPU梯度数组（标量场）
#[derive(Debug, Clone)]
pub struct GpuGradientArrays {
    /// x方向梯度
    pub grad_x: Vec<f32>,
    /// y方向梯度
    pub grad_y: Vec<f32>,
}

impl GpuGradientArrays {
    /// 创建指定大小的梯度数组
    pub fn new(n_cells: usize) -> Self {
        Self {
            grad_x: vec![0.0; n_cells],
            grad_y: vec![0.0; n_cells],
        }
    }

    /// 重置所有值为零
    pub fn reset(&mut self) {
        self.grad_x.iter_mut().for_each(|x| *x = 0.0);
        self.grad_y.iter_mut().for_each(|x| *x = 0.0);
    }

    /// 单元数量
    #[inline]
    pub fn len(&self) -> usize {
        self.grad_x.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.grad_x.is_empty()
    }
}

/// GPU速度梯度张量（2x2）
#[derive(Debug, Clone)]
pub struct GpuVelocityGradients {
    /// du/dx
    pub du_dx: Vec<f32>,
    /// du/dy
    pub du_dy: Vec<f32>,
    /// dv/dx
    pub dv_dx: Vec<f32>,
    /// dv/dy
    pub dv_dy: Vec<f32>,
}

impl GpuVelocityGradients {
    /// 创建指定大小的速度梯度数组
    pub fn new(n_cells: usize) -> Self {
        Self {
            du_dx: vec![0.0; n_cells],
            du_dy: vec![0.0; n_cells],
            dv_dx: vec![0.0; n_cells],
            dv_dy: vec![0.0; n_cells],
        }
    }

    /// 重置所有值为零
    pub fn reset(&mut self) {
        self.du_dx.iter_mut().for_each(|x| *x = 0.0);
        self.du_dy.iter_mut().for_each(|x| *x = 0.0);
        self.dv_dx.iter_mut().for_each(|x| *x = 0.0);
        self.dv_dy.iter_mut().for_each(|x| *x = 0.0);
    }

    /// 单元数量
    #[inline]
    pub fn len(&self) -> usize {
        self.du_dx.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.du_dx.is_empty()
    }
}

/// GPU面重构值（左右状态）
#[derive(Debug, Clone)]
pub struct GpuFaceReconstruction {
    /// 左侧水深
    pub h_left: Vec<f32>,
    /// 右侧水深
    pub h_right: Vec<f32>,
    /// 左侧x速度
    pub u_left: Vec<f32>,
    /// 右侧x速度
    pub u_right: Vec<f32>,
    /// 左侧y速度
    pub v_left: Vec<f32>,
    /// 右侧y速度
    pub v_right: Vec<f32>,
}

impl GpuFaceReconstruction {
    /// 创建指定大小的重构数组
    pub fn new(n_faces: usize) -> Self {
        Self {
            h_left: vec![0.0; n_faces],
            h_right: vec![0.0; n_faces],
            u_left: vec![0.0; n_faces],
            u_right: vec![0.0; n_faces],
            v_left: vec![0.0; n_faces],
            v_right: vec![0.0; n_faces],
        }
    }

    /// 面数量
    #[inline]
    pub fn len(&self) -> usize {
        self.h_left.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.h_left.is_empty()
    }
}

/// 完整的GPU计算工作空间
#[derive(Debug, Clone)]
pub struct GpuWorkspace {
    /// 当前状态
    pub state: GpuStateArrays,
    /// RK阶段状态（用于SSP-RK积分）
    pub state_stage: GpuStateArrays,
    /// 通量累加
    pub flux: GpuFluxArrays,
    /// 源项
    pub source: GpuSourceArrays,
    /// 水深梯度
    pub grad_h: GpuGradientArrays,
    /// 速度梯度张量
    pub grad_vel: GpuVelocityGradients,
    /// 面重构值
    pub face_recon: GpuFaceReconstruction,
    /// 限制器值
    pub limiter: Vec<f32>,
    /// 涡粘系数
    pub nu_t: Vec<f32>,
    /// 单元数
    n_cells: usize,
    /// 面数
    n_faces: usize,
}

impl GpuWorkspace {
    /// 创建GPU工作空间
    pub fn new(n_cells: usize, n_faces: usize) -> Self {
        Self {
            state: GpuStateArrays::new(n_cells),
            state_stage: GpuStateArrays::new(n_cells),
            flux: GpuFluxArrays::new(n_cells),
            source: GpuSourceArrays::new(n_cells),
            grad_h: GpuGradientArrays::new(n_cells),
            grad_vel: GpuVelocityGradients::new(n_cells),
            face_recon: GpuFaceReconstruction::new(n_faces),
            limiter: vec![1.0; n_cells],
            nu_t: vec![0.0; n_cells],
            n_cells,
            n_faces,
        }
    }

    /// 准备下一个时间步
    pub fn prepare_for_step(&mut self) {
        self.flux.reset();
        self.source.reset();
    }

    /// 重置梯度缓冲区
    pub fn reset_gradients(&mut self) {
        self.grad_h.reset();
        self.grad_vel.reset();
    }

    /// 重置限制器为1.0
    pub fn reset_limiter(&mut self) {
        self.limiter.iter_mut().for_each(|x| *x = 1.0);
    }

    /// 获取单元数
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 获取面数
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.n_faces
    }

    /// GPU内存估计（字节）
    pub fn gpu_memory_estimate(&self) -> usize {
        let f32_size = std::mem::size_of::<f32>();

        // 状态: 4 fields * 2 (current + stage) * n_cells
        let state_mem = 8 * self.n_cells * f32_size;

        // 通量和源项: 3 fields each * n_cells
        let flux_source_mem = 6 * self.n_cells * f32_size;

        // 梯度: 2 (h) + 4 (vel) * n_cells
        let grad_mem = 6 * self.n_cells * f32_size;

        // 面重构: 6 fields * n_faces
        let recon_mem = 6 * self.n_faces * f32_size;

        // 限制器和涡粘: 2 * n_cells
        let misc_mem = 2 * self.n_cells * f32_size;

        state_mem + flux_source_mem + grad_mem + recon_mem + misc_mem
    }
}

// ============= POD类型 =============

/// 紧凑的梯度数据（用于GPU buffer）
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuGradientPod {
    pub grad_x: f32,
    pub grad_y: f32,
}

/// 紧凑的速度梯度张量
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuVelGradPod {
    pub du_dx: f32,
    pub du_dy: f32,
    pub dv_dx: f32,
    pub dv_dy: f32,
}

/// 紧凑的面重构数据
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuFaceReconPod {
    pub h_left: f32,
    pub h_right: f32,
    pub u_left: f32,
    pub u_right: f32,
    pub v_left: f32,
    pub v_right: f32,
    pub _padding: [f32; 2],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_state_arrays() {
        let state = GpuStateArrays::new(100);
        assert_eq!(state.len(), 100);
        assert_eq!(state.h.len(), 100);
    }

    #[test]
    fn test_from_f64() {
        let h = vec![1.0f64, 2.0, 3.0];
        let hu = vec![0.1, 0.2, 0.3];
        let hv = vec![0.01, 0.02, 0.03];
        let z = vec![0.0, 0.0, 0.0];

        let state = GpuStateArrays::from_f64(&h, &hu, &hv, &z);
        assert_eq!(state.h[0], 1.0f32);
        assert_eq!(state.hu[1], 0.2f32);
    }

    #[test]
    fn test_gpu_workspace() {
        let ws = GpuWorkspace::new(1000, 2000);
        assert_eq!(ws.n_cells(), 1000);
        assert_eq!(ws.n_faces(), 2000);
        assert!(ws.gpu_memory_estimate() > 0);
    }

    #[test]
    fn test_workspace_reset() {
        let mut ws = GpuWorkspace::new(10, 20);
        ws.flux.flux_h[0] = 1.0;
        ws.source.source_hu[0] = 2.0;

        ws.prepare_for_step();

        assert_eq!(ws.flux.flux_h[0], 0.0);
        assert_eq!(ws.source.source_hu[0], 0.0);
    }

    #[test]
    fn test_pod_sizes() {
        assert_eq!(std::mem::size_of::<GpuGradientPod>(), 8);
        assert_eq!(std::mem::size_of::<GpuVelGradPod>(), 16);
        assert_eq!(std::mem::size_of::<GpuFaceReconPod>(), 32);
    }
}
