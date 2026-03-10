// marihydro\crates\mh_physics\src\core\gpu.rs
//! GPU 鍚庣锛堥鏋跺疄鐜帮級
//!
//! 棰勭暀 CUDA 鍚庣鏀寔锛屽綋鍓嶄粎鎻愪緵鎺ュ彛瀹氫箟銆?
//! 瀹為檯 GPU 瀹炵幇灏嗗湪鏈潵闃舵瀹屾垚銆?

use super::buffer::DeviceBuffer;
use super::scalar::Scalar;
use bytemuck::Pod;
use std::marker::PhantomData;

/// CUDA 鍚庣鍗犱綅绗?
/// 
/// 杩欐槸涓€涓崰浣嶇粨鏋勶紝鐢ㄤ簬瀹氫箟 GPU 鍚庣鎺ュ彛銆?
/// 瀹為檯瀹炵幇闇€瑕佸湪鍚敤 `cuda` feature 鏃跺畬鎴愩€?
#[derive(Debug, Clone)]
pub struct CudaBackendPlaceholder<S: Scalar> {
    _marker: PhantomData<S>,
}

impl<S: Scalar> CudaBackendPlaceholder<S> {
    /// 鍒涘缓 CUDA 鍚庣锛堝崰浣嶏級
    pub fn new(_device_id: usize) -> Result<Self, CudaError> {
        Err(CudaError("CUDA backend not implemented yet".into()))
    }
}

/// GPU 缂撳啿鍖哄崰浣嶇
#[derive(Debug, Clone)]
pub struct GpuBuffer<T: Pod> {
    data: Vec<T>,
}

// 鎵嬪姩瀹炵幇 Send 鍜?Sync锛圙PU 缂撳啿鍖烘槸瀹夊叏鐨勶級
unsafe impl<T: Pod> Send for GpuBuffer<T> {}
unsafe impl<T: Pod> Sync for GpuBuffer<T> {}

impl<T: Pod + Clone + Default + Send + Sync> DeviceBuffer<T> for GpuBuffer<T> {
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn copy_from_slice(&mut self, src: &[T]) {
        self.data.clear();
        self.data.extend_from_slice(src);
    }
    
    fn copy_to_vec(&self) -> Vec<T> {
        self.data.clone()
    }
    
    fn as_slice(&self) -> Option<&[T]> {
        None // GPU 缂撳啿鍖烘棤娉曠洿鎺ヨ闂?
    }
    
    fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        None // GPU 缂撳啿鍖烘棤娉曠洿鎺ヨ闂?
    }
    
    fn fill(&mut self, value: T) {
        self.data.iter_mut().for_each(|x| *x = value.clone());
    }
}

/// CUDA 閿欒绫诲瀷
#[derive(Debug)]
pub struct CudaError(pub String);

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDA error: {}", self.0)
    }
}

impl std::error::Error for CudaError {}

/// GPU 璁惧淇℃伅
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// 璁惧 ID
    pub id: usize,
    /// 璁惧鍚嶇О
    pub name: String,
    /// 鏄惧瓨澶у皬锛堝瓧鑺傦級
    pub memory_bytes: usize,
    /// 璁＄畻鑳藉姏
    pub compute_capability: (u32, u32),
}

/// 鏌ヨ鍙敤 GPU 璁惧
pub fn available_gpus() -> Vec<GpuDeviceInfo> {
    // 鍗犱綅瀹炵幇
    Vec::new()
}

/// 妫€鏌ユ槸鍚︽湁鍙敤 GPU
pub fn has_cuda() -> bool {
    false
}
