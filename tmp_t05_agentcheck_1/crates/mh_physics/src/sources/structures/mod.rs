//! 亚网格结构物源项
//!
//! 实现水工结构物的亚网格处理：
//! - 桥墩拖曳力 (`bridge_pier`)
//! - 堰流 (`weir`)
//! - 闸门控制 (`sluice_gate`)

pub mod bridge_pier;
pub mod weir;

pub use bridge_pier::BridgePierDrag;
pub use weir::{WeirFlow, WeirType};
