// src-tauri/src/marihydro/io/exporters/vtk.rs

use crate::marihydro::infra::error::{MhError, MhResult};
use ndarray::Array2;
use std::path::Path;
use vkt::{Cells, Data, UnstructuredGridPiece, Vtk};

/// VTK 导出工具类
pub struct VtkExporter;

impl VtkExporter {
    /// 导出当前帧
    /// mesh_coords: [(x, y, z)] 节点坐标列表
    /// cells: [i0, i1, i2, i3] 单元连接关系
    /// data_fields: [("Name", Array)] 数据场列表
    pub fn export(
        file_path: &Path,
        mesh_coords: &[(f64, f64, f64)],
        connectivity: &[u32], // Quad connectivity (4 nodes per cell)
        data_fields: &[(&str, &Array2<f64>)],
    ) -> MhResult<()> {
        // 1. 构建节点 (Points)
        // Vtk crate 需要平铺的 Vec<f64> [x1, y1, z1, x2, y2, z2...]
        let mut points = Vec::with_capacity(mesh_coords.len() * 3);
        for (x, y, z) in mesh_coords {
            points.push(*x);
            points.push(*y);
            points.push(*z);
        }

        // 2. 构建单元 (Cells)
        // connectivity 已经是平铺的索引
        let cells = Cells::quads(connectivity.to_vec());

        // 3. 构建数据 (Cell Data)
        let mut vtk_data = Vec::new();
        for (name, array) in data_fields {
            // ndarray 转 Vec (Row-major)
            // 注意：需要确保 array 的迭代顺序与 cells 的顺序一致
            let vec_data: Vec<f64> = array.iter().cloned().collect();
            vtk_data.push(Data::scalar(*name, vec_data));
        }

        // 4. 写入
        let piece = UnstructuredGridPiece::new(points, cells, vec![]).with_cell_data(vtk_data);

        Vtk::new(piece).write(file_path).map_err(|e| {
            MhError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;

        Ok(())
    }
}
