//! 网格拓扑抽象
//!
//! 提供结构化和非结构化网格的统一接口。

use crate::core::{Backend, Scalar};

/// 网格类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshKind {
    /// 非结构化网格
    Unstructured,
    /// 结构化网格
    Structured { nx: usize, ny: usize },
}

/// 面信息
#[derive(Debug, Clone, Copy)]
pub struct FaceInfo<S: Scalar> {
    /// 面法向量 (指向 neighbor)
    pub normal: [S; 2],
    /// 面长度
    pub length: S,
    /// 面中心坐标
    pub center: [S; 2],
    /// 所属单元 (owner)
    pub owner: usize,
    /// 相邻单元 (neighbor)，边界面为 None
    pub neighbor: Option<usize>,
}

/// 网格拓扑 trait
pub trait MeshTopology<B: Backend>: Send + Sync {
    // ========== 基本信息 ==========
    
    /// 单元数量
    fn n_cells(&self) -> usize;
    
    /// 面数量
    fn n_faces(&self) -> usize;
    
    /// 内部面数量
    fn n_interior_faces(&self) -> usize;
    
    /// 边界面数量
    fn n_boundary_faces(&self) -> usize {
        self.n_faces() - self.n_interior_faces()
    }
    
    /// 节点数量
    fn n_nodes(&self) -> usize;
    
    // ========== 几何数据 ==========
    
    /// 单元中心坐标
    fn cell_center(&self, cell: usize) -> [B::Scalar; 2];
    
    /// 单元面积
    fn cell_area(&self, cell: usize) -> B::Scalar;
    
    /// 面法向量
    fn face_normal(&self, face: usize) -> [B::Scalar; 2];
    
    /// 面长度
    fn face_length(&self, face: usize) -> B::Scalar;
    
    /// 面中心坐标
    fn face_center(&self, face: usize) -> [B::Scalar; 2];
    
    // ========== 拓扑数据 ==========
    
    /// 面的 owner 单元
    fn face_owner(&self, face: usize) -> usize;
    
    /// 面的 neighbor 单元（边界面返回 None）
    fn face_neighbor(&self, face: usize) -> Option<usize>;
    
    /// 单元的所有面索引
    fn cell_faces(&self, cell: usize) -> &[usize];
    
    /// 单元的相邻单元索引
    fn cell_neighbors(&self, cell: usize) -> Vec<usize>;
    
    // ========== 边界信息 ==========
    
    /// 是否为边界面
    fn is_boundary_face(&self, face: usize) -> bool {
        self.face_neighbor(face).is_none()
    }
    
    /// 边界面索引列表
    fn boundary_faces(&self) -> &[usize];
    
    /// 内部面索引列表
    fn interior_faces(&self) -> &[usize];
    
    // ========== 网格类型 ==========
    
    /// 网格类型
    fn mesh_kind(&self) -> MeshKind;
    
    // ========== 批量访问（GPU 优化入口）==========
    
    /// 获取所有单元面积（设备缓冲区）
    fn cell_areas_buffer(&self) -> &B::Buffer<B::Scalar>;
    
    /// 获取所有面长度（设备缓冲区）
    fn face_lengths_buffer(&self) -> &B::Buffer<B::Scalar>;
}

/// 网格几何计算辅助函数
pub struct MeshGeometry;

impl MeshGeometry {
    /// 计算两点间距离
    #[inline]
    pub fn distance<S: Scalar>(p1: [S; 2], p2: [S; 2]) -> S {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        (dx * dx + dy * dy).sqrt()
    }
    
    /// 计算单位法向量
    #[inline]
    pub fn unit_normal<S: Scalar>(p1: [S; 2], p2: [S; 2]) -> [S; 2] {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let length = (dx * dx + dy * dy).sqrt();
        if length > S::min_positive_value() {
            [-dy / length, dx / length]
        } else {
            [<S as Scalar>::from_f64(0.0), <S as Scalar>::from_f64(0.0)]
        }
    }
}
