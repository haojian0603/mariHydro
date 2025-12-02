// src-tauri/src/marihydro/core/compute/transfer.rs

//! GPU数据传输管理
//!
//! 管理CPU和GPU之间的数据同步

use std::sync::Arc;

use crate::marihydro::core::error::{MhError, MhResult};

#[cfg(feature = "gpu")]
use super::gpu_mesh::GpuMeshData;
#[cfg(feature = "gpu")]
use super::gpu_state::{GpuFluxArrays, GpuSourceArrays, GpuStateArrays, GpuWorkspace};
#[cfg(feature = "gpu")]
use super::wgpu_buffer::WgpuBuffer;
#[cfg(feature = "gpu")]
use super::buffer::BufferUsage;
#[cfg(feature = "gpu")]
use wgpu::{Device, Queue};
#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};

/// 传输方向
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// CPU到GPU
    HostToDevice,
    /// GPU到CPU
    DeviceToHost,
    /// 双向同步
    Bidirectional,
}

/// 传输状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferState {
    /// 同步
    Synchronized,
    /// CPU数据已修改
    CpuDirty,
    /// GPU数据已修改
    GpuDirty,
    /// 两边都修改（需要冲突解决）
    Conflicted,
}

/// 同步策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// 立即同步
    Immediate,
    /// 延迟同步（批量）
    Deferred,
    /// 手动同步
    Manual,
}

/// 传输统计
#[derive(Debug, Clone, Default)]
pub struct TransferStats {
    /// 上传次数
    pub upload_count: usize,
    /// 下载次数
    pub download_count: usize,
    /// 上传字节数
    pub uploaded_bytes: usize,
    /// 下载字节数
    pub downloaded_bytes: usize,
    /// 总传输时间（毫秒）
    pub total_time_ms: f64,
}

impl TransferStats {
    /// 重置统计
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// 记录上传
    pub fn record_upload(&mut self, bytes: usize, time_ms: f64) {
        self.upload_count += 1;
        self.uploaded_bytes += bytes;
        self.total_time_ms += time_ms;
    }

    /// 记录下载
    pub fn record_download(&mut self, bytes: usize, time_ms: f64) {
        self.download_count += 1;
        self.downloaded_bytes += bytes;
        self.total_time_ms += time_ms;
    }

    /// 计算平均带宽（MB/s）
    pub fn average_bandwidth_mbps(&self) -> f64 {
        if self.total_time_ms < 1e-6 {
            return 0.0;
        }
        let total_bytes = (self.uploaded_bytes + self.downloaded_bytes) as f64;
        total_bytes / (self.total_time_ms * 1000.0) // bytes/ms -> MB/s
    }
}

/// CPU端状态缓存
pub struct CpuStateCache {
    /// 水深
    pub h: Vec<f64>,
    /// x动量
    pub hu: Vec<f64>,
    /// y动量
    pub hv: Vec<f64>,
    /// 底床高程
    pub z_bed: Vec<f64>,
    /// 脏标记
    pub dirty: bool,
}

impl CpuStateCache {
    /// 创建新的状态缓存
    pub fn new(n_cells: usize) -> Self {
        Self {
            h: vec![0.0; n_cells],
            hu: vec![0.0; n_cells],
            hv: vec![0.0; n_cells],
            z_bed: vec![0.0; n_cells],
            dirty: false,
        }
    }

    /// 标记为脏
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// 清除脏标记
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// 转换为GPU格式
    #[cfg(feature = "gpu")]
    pub fn to_gpu_arrays(&self) -> GpuStateArrays {
        GpuStateArrays::from_f64(&self.h, &self.hu, &self.hv, &self.z_bed)
    }

    /// 从GPU格式更新
    #[cfg(feature = "gpu")]
    pub fn from_gpu_arrays(&mut self, gpu: &GpuStateArrays) {
        gpu.to_f64(&mut self.h, &mut self.hu, &mut self.hv);
    }
}

/// 传输管理器（CPU版本，无实际GPU传输）
#[cfg(not(feature = "gpu"))]
pub struct TransferManager {
    stats: TransferStats,
}

#[cfg(not(feature = "gpu"))]
impl TransferManager {
    /// 创建传输管理器
    pub fn new() -> Self {
        Self {
            stats: TransferStats::default(),
        }
    }

    /// 获取统计信息
    pub fn stats(&self) -> &TransferStats {
        &self.stats
    }
}

/// 传输管理器（GPU版本）
#[cfg(feature = "gpu")]
pub struct TransferManager {
    device: Arc<Device>,
    queue: Arc<Queue>,
    stats: TransferStats,
    policy: SyncPolicy,
}

#[cfg(feature = "gpu")]
impl TransferManager {
    /// 创建传输管理器
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device,
            queue,
            stats: TransferStats::default(),
            policy: SyncPolicy::Deferred,
        }
    }

    /// 设置同步策略
    pub fn set_policy(&mut self, policy: SyncPolicy) {
        self.policy = policy;
    }

    /// 上传f32数组到GPU
    pub fn upload_f32(&mut self, data: &[f32], buffer: &mut WgpuBuffer<f32>) -> MhResult<()> {
        let start = std::time::Instant::now();
        buffer.upload(data)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        self.stats.record_upload(data.len() * 4, elapsed);
        Ok(())
    }

    /// 上传u32数组到GPU
    pub fn upload_u32(&mut self, data: &[u32], buffer: &mut WgpuBuffer<u32>) -> MhResult<()> {
        let start = std::time::Instant::now();
        buffer.upload(data)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        self.stats.record_upload(data.len() * 4, elapsed);
        Ok(())
    }

    /// 从GPU下载f32数组
    pub fn download_f32(&mut self, buffer: &WgpuBuffer<f32>, data: &mut [f32]) -> MhResult<()> {
        let start = std::time::Instant::now();
        buffer.download(data)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        self.stats.record_download(data.len() * 4, elapsed);
        Ok(())
    }

    /// 上传状态到GPU
    pub fn upload_state(
        &mut self,
        cpu_state: &CpuStateCache,
        gpu_h: &mut WgpuBuffer<f32>,
        gpu_hu: &mut WgpuBuffer<f32>,
        gpu_hv: &mut WgpuBuffer<f32>,
    ) -> MhResult<()> {
        let gpu_arrays = cpu_state.to_gpu_arrays();
        self.upload_f32(&gpu_arrays.h, gpu_h)?;
        self.upload_f32(&gpu_arrays.hu, gpu_hu)?;
        self.upload_f32(&gpu_arrays.hv, gpu_hv)?;
        Ok(())
    }

    /// 从GPU下载状态
    pub fn download_state(
        &mut self,
        gpu_h: &WgpuBuffer<f32>,
        gpu_hu: &WgpuBuffer<f32>,
        gpu_hv: &WgpuBuffer<f32>,
        cpu_state: &mut CpuStateCache,
    ) -> MhResult<()> {
        let n = cpu_state.h.len();
        let mut gpu_arrays = GpuStateArrays::new(n);
        
        self.download_f32(gpu_h, &mut gpu_arrays.h)?;
        self.download_f32(gpu_hu, &mut gpu_arrays.hu)?;
        self.download_f32(gpu_hv, &mut gpu_arrays.hv)?;
        
        cpu_state.from_gpu_arrays(&gpu_arrays);
        cpu_state.clear_dirty();
        Ok(())
    }

    /// 获取统计信息
    pub fn stats(&self) -> &TransferStats {
        &self.stats
    }

    /// 重置统计
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// 同步设备
    pub fn synchronize(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_stats() {
        let mut stats = TransferStats::default();
        stats.record_upload(1000, 0.1);
        stats.record_download(2000, 0.2);

        assert_eq!(stats.upload_count, 1);
        assert_eq!(stats.download_count, 1);
        assert_eq!(stats.uploaded_bytes, 1000);
        assert_eq!(stats.downloaded_bytes, 2000);
    }

    #[test]
    fn test_cpu_state_cache() {
        let mut cache = CpuStateCache::new(100);
        assert!(!cache.dirty);
        
        cache.mark_dirty();
        assert!(cache.dirty);
        
        cache.clear_dirty();
        assert!(!cache.dirty);
    }

    #[test]
    fn test_transfer_direction() {
        let dir = TransferDirection::HostToDevice;
        assert_eq!(dir, TransferDirection::HostToDevice);
    }
}
