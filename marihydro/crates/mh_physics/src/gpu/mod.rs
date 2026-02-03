//! GPU 加速模块
//!
//! 提供 GPU 后端的统一入口，包括：
//! - GPU 可用性检测
//! - 设备信息访问
//! - 能力标记
//!
//! 说明：当前 GPU 为模拟实现，运行时可安全回退到 CPU。

// 重新导出核心 GPU 类型
pub use crate::core::gpu::{
	CudaBackendPlaceholder,
	GpuBuffer,
	GpuDeviceInfo,
	CudaError,
	available_gpus,
	has_cuda,
};

/// GPU 状态标记
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuStatus {
	/// GPU 可用
	Available {
		/// 设备数量
		device_count: usize,
		/// 主设备显存 (MB)
		primary_memory_mb: usize,
	},
	/// GPU 未编译（feature 未启用）
	NotCompiled,
	/// GPU 未检测到
	NotDetected,
	/// GPU 初始化失败
	InitFailed,
}

impl GpuStatus {
	/// 检测当前 GPU 状态
	pub fn detect() -> Self {
		#[cfg(not(feature = "cuda"))]
		{
			return Self::NotCompiled;
		}

		#[cfg(feature = "cuda")]
		{
			let devices = available_gpus();
			if devices.is_empty() {
				return Self::NotDetected;
			}

			let primary = &devices[0];
			Self::Available {
				device_count: devices.len(),
				primary_memory_mb: primary.memory_bytes / (1024 * 1024),
			}
		}
	}

	/// GPU 是否可用
	pub fn is_available(&self) -> bool {
		matches!(self, Self::Available { .. })
	}

	/// 获取设备数量
	pub fn device_count(&self) -> usize {
		match self {
			Self::Available { device_count, .. } => *device_count,
			_ => 0,
		}
	}
}

impl std::fmt::Display for GpuStatus {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Available { device_count, primary_memory_mb } => {
				write!(f, "GPU 可用: {} 设备, 主设备 {} MB 显存", device_count, primary_memory_mb)
			}
			Self::NotCompiled => write!(f, "GPU 未编译 (需要 --features cuda)"),
			Self::NotDetected => write!(f, "GPU 未检测到"),
			Self::InitFailed => write!(f, "GPU 初始化失败"),
		}
	}
}

/// GPU 能力标志
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuCapabilities(u32);

impl GpuCapabilities {
	pub const BASIC_COMPUTE: Self = Self(0b0001);
	pub const DOUBLE_PRECISION: Self = Self(0b0010);
	pub const UNIFIED_MEMORY: Self = Self(0b0100);
	pub const MULTI_GPU: Self = Self(0b1000);
	pub const TENSOR_CORES: Self = Self(0b0001_0000);

	pub const fn empty() -> Self {
		Self(0)
	}

	pub const fn bits(self) -> u32 {
		self.0
	}

	pub fn contains(self, other: Self) -> bool {
		(self.0 & other.0) == other.0
	}

	pub fn insert(&mut self, other: Self) {
		self.0 |= other.0;
	}

	/// 从设备信息推断能力
	pub fn from_device(info: &GpuDeviceInfo) -> Self {
		let mut caps = Self::BASIC_COMPUTE;

		if info.compute_capability.0 > 1 ||
			(info.compute_capability.0 == 1 && info.compute_capability.1 >= 3) {
			caps.insert(Self::DOUBLE_PRECISION);
		}

		if info.compute_capability.0 >= 6 {
			caps.insert(Self::UNIFIED_MEMORY);
		}

		if info.compute_capability.0 >= 7 {
			caps.insert(Self::TENSOR_CORES);
		}

		caps
	}
}
