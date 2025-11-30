//! 预分配工作空间
//!
//! 提供模拟过程中使用的预分配缓冲区。

use super::pool::{BufferPool, PooledBuffer};
use glam::DVec2;

/// 预分配的工作区缓冲区
///
/// 用于存储模拟过程中的临时数据，避免在热路径上分配内存。
///
/// # 使用模式
///
/// ```rust
/// let mut workspace = Workspace::new(mesh.n_cells(), mesh.n_faces());
///
/// // 每个时间步开始时准备工作空间
/// workspace.prepare_for_step();
///
/// // 使用固定缓冲区
/// for i in 0..n_cells {
///     workspace.flux_h[i] += contribution;
/// }
///
/// // 使用池化临时缓冲区
/// let mut temp = workspace.temp_scalar();
/// // temp 在作用域结束时自动归还
/// ```
pub struct Workspace {
    n_cells: usize,
    n_faces: usize,

    // ===== 固定缓冲区（整个模拟期间复用）=====
    /// 速度场 [m/s]
    pub velocities: Vec<DVec2>,

    /// 标量场梯度 x 分量
    pub grad_x: Vec<f64>,
    /// 标量场梯度 y 分量
    pub grad_y: Vec<f64>,

    /// 速度 u 对 x 的梯度
    pub du_dx: Vec<f64>,
    /// 速度 u 对 y 的梯度
    pub du_dy: Vec<f64>,
    /// 速度 v 对 x 的梯度
    pub dv_dx: Vec<f64>,
    /// 速度 v 对 y 的梯度
    pub dv_dy: Vec<f64>,

    /// 水深通量累加缓冲区 [m³/s]
    pub flux_h: Vec<f64>,
    /// x动量通量累加缓冲区 [m⁴/s²]
    pub flux_hu: Vec<f64>,
    /// y动量通量累加缓冲区 [m⁴/s²]
    pub flux_hv: Vec<f64>,

    /// 涡粘系数 [m²/s]
    pub nu_t: Vec<f64>,

    /// 水深源项累加缓冲区
    pub source_h: Vec<f64>,
    /// x动量源项累加缓冲区
    pub source_hu: Vec<f64>,
    /// y动量源项累加缓冲区
    pub source_hv: Vec<f64>,

    /// 左侧重构水深
    pub h_left: Vec<f64>,
    /// 右侧重构水深
    pub h_right: Vec<f64>,
    /// 左侧重构速度
    pub vel_left: Vec<DVec2>,
    /// 右侧重构速度
    pub vel_right: Vec<DVec2>,

    /// 梯度限制器
    pub limiter: Vec<f64>,

    // ===== 临时缓冲区池 =====
    scalar_pool: BufferPool<f64>,
    vector_pool: BufferPool<DVec2>,
}

impl Workspace {
    /// 创建工作区
    pub fn new(n_cells: usize, n_faces: usize) -> Self {
        Self {
            n_cells,
            n_faces,

            // 固定缓冲区
            velocities: vec![DVec2::ZERO; n_cells],
            grad_x: vec![0.0; n_cells],
            grad_y: vec![0.0; n_cells],
            du_dx: vec![0.0; n_cells],
            du_dy: vec![0.0; n_cells],
            dv_dx: vec![0.0; n_cells],
            dv_dy: vec![0.0; n_cells],
            flux_h: vec![0.0; n_cells],
            flux_hu: vec![0.0; n_cells],
            flux_hv: vec![0.0; n_cells],
            nu_t: vec![0.0; n_cells],
            source_h: vec![0.0; n_cells],
            source_hu: vec![0.0; n_cells],
            source_hv: vec![0.0; n_cells],
            h_left: vec![0.0; n_faces],
            h_right: vec![0.0; n_faces],
            vel_left: vec![DVec2::ZERO; n_faces],
            vel_right: vec![DVec2::ZERO; n_faces],
            limiter: vec![1.0; n_cells],

            // 缓冲区池
            scalar_pool: BufferPool::new(n_cells),
            vector_pool: BufferPool::new(n_cells),
        }
    }

    /// 使用构建器创建
    pub fn builder(n_cells: usize, n_faces: usize) -> WorkspaceBuilder {
        WorkspaceBuilder::new(n_cells, n_faces)
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

    /// 准备下一个时间步
    ///
    /// 重置所有累加缓冲区为零。
    pub fn prepare_for_step(&mut self) {
        self.reset_fluxes();
        self.reset_sources();
    }

    /// 重置通量缓冲区
    pub fn reset_fluxes(&mut self) {
        self.flux_h.iter_mut().for_each(|x| *x = 0.0);
        self.flux_hu.iter_mut().for_each(|x| *x = 0.0);
        self.flux_hv.iter_mut().for_each(|x| *x = 0.0);
    }

    /// 重置源项缓冲区
    pub fn reset_sources(&mut self) {
        self.source_h.iter_mut().for_each(|x| *x = 0.0);
        self.source_hu.iter_mut().for_each(|x| *x = 0.0);
        self.source_hv.iter_mut().for_each(|x| *x = 0.0);
    }

    /// 重置梯度缓冲区
    pub fn reset_gradients(&mut self) {
        self.grad_x.iter_mut().for_each(|x| *x = 0.0);
        self.grad_y.iter_mut().for_each(|x| *x = 0.0);
        self.du_dx.iter_mut().for_each(|x| *x = 0.0);
        self.du_dy.iter_mut().for_each(|x| *x = 0.0);
        self.dv_dx.iter_mut().for_each(|x| *x = 0.0);
        self.dv_dy.iter_mut().for_each(|x| *x = 0.0);
    }

    /// 重置限制器为 1.0
    pub fn reset_limiter(&mut self) {
        self.limiter.iter_mut().for_each(|x| *x = 1.0);
    }

    /// 获取临时标量缓冲区
    pub fn temp_scalar(&self) -> PooledBuffer<'_, f64> {
        self.scalar_pool.acquire()
    }

    /// 获取临时向量缓冲区
    pub fn temp_vector(&self) -> PooledBuffer<'_, DVec2> {
        self.vector_pool.acquire()
    }

    /// 获取已清零的临时标量缓冲区
    pub fn temp_scalar_zeroed(&self) -> PooledBuffer<'_, f64> {
        self.scalar_pool.acquire_zeroed()
    }

    /// 预热缓冲池
    pub fn warm_up_pools(&self, count: usize) {
        self.scalar_pool.warm_up(count);
        self.vector_pool.warm_up(count);
    }

    /// 获取内存使用估计（字节）
    pub fn memory_usage(&self) -> usize {
        let f64_size = std::mem::size_of::<f64>();
        let dvec2_size = std::mem::size_of::<DVec2>();

        // 固定缓冲区
        let fixed = (self.n_cells * 14 + self.n_faces * 2) * f64_size
            + (self.n_cells + self.n_faces * 2) * dvec2_size;

        // 池中缓冲区（估计）
        let pool_estimate = (self.scalar_pool.available() * self.n_cells * f64_size)
            + (self.vector_pool.available() * self.n_cells * dvec2_size);

        fixed + pool_estimate
    }
}

/// 工作空间构建器
pub struct WorkspaceBuilder {
    n_cells: usize,
    n_faces: usize,
    pool_size: usize,
    warm_up: bool,
}

impl WorkspaceBuilder {
    /// 创建新的构建器
    pub fn new(n_cells: usize, n_faces: usize) -> Self {
        Self {
            n_cells,
            n_faces,
            pool_size: 16,
            warm_up: false,
        }
    }

    /// 设置池大小
    pub fn with_pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }

    /// 启用预热
    pub fn with_warm_up(mut self) -> Self {
        self.warm_up = true;
        self
    }

    /// 构建工作空间
    pub fn build(self) -> Workspace {
        let ws = Workspace::new(self.n_cells, self.n_faces);
        if self.warm_up {
            ws.warm_up_pools(4);
        }
        ws
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_creation() {
        let ws = Workspace::new(100, 200);
        assert_eq!(ws.n_cells(), 100);
        assert_eq!(ws.n_faces(), 200);
        assert_eq!(ws.flux_h.len(), 100);
        assert_eq!(ws.h_left.len(), 200);
    }

    #[test]
    fn test_prepare_for_step() {
        let mut ws = Workspace::new(10, 20);

        // 修改一些值
        ws.flux_h[0] = 1.0;
        ws.source_hu[0] = 2.0;

        // 准备下一步
        ws.prepare_for_step();

        assert_eq!(ws.flux_h[0], 0.0);
        assert_eq!(ws.source_hu[0], 0.0);
    }

    #[test]
    fn test_temp_buffers() {
        let ws = Workspace::new(100, 200);

        {
            let mut temp = ws.temp_scalar();
            temp[0] = 42.0;
        }

        // 再次获取应该已清零
        let temp = ws.temp_scalar();
        assert_eq!(temp[0], 0.0);
    }

    #[test]
    fn test_builder() {
        let ws = Workspace::builder(100, 200)
            .with_pool_size(8)
            .with_warm_up()
            .build();

        assert_eq!(ws.n_cells(), 100);
    }

    #[test]
    fn test_memory_usage() {
        let ws = Workspace::new(1000, 2000);
        let usage = ws.memory_usage();
        // 应该大于 0
        assert!(usage > 0);
    }
}
