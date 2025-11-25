// src-tauri/src/marihydro/io/drivers/vtk_driver.rs

use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::io::types::GeoGridData;
use ndarray::Array2;
use std::path::Path;
use vkt::{Cells, Data, UnstructuredGridPiece, Vtk};

pub struct VtkDriver;

impl VtkDriver {
    /// 导出带有无效值剔除的 VTK 文件
    /// mask: 可选的布尔掩膜 (true=有效/水, false=无效/陆地)
    /// 如果 mask 为 None，则根据 GeoGridData 的 no_data_value 自动判断
    pub fn export(
        path: &str,
        h_data: &GeoGridData, // 水深 (主拓扑参考)
        u_data: &GeoGridData,
        v_data: &GeoGridData,
        mask: Option<&Array2<u8>>,
    ) -> MhResult<()> {
        let (ny, nx) = h_data.data.dim();
        let gt = h_data.meta.transform;
        let no_data = h_data.meta.no_data_value;

        // 1. 拓扑映射构建 (Topology Mapping)
        // 我们需要把 2D 网格 (i,j) 映射到 VTK 的线性 PointID
        // 只有有效的单元格顶点才会被添加到 VTK Points 中吗？
        // 简单策略：添加所有点，但只构建有效单元 (Cells)。这样未使用的点会被 ParaView 忽略。

        let mut points = Vec::with_capacity(nx * ny * 3);

        // 生成点坐标
        for j in 0..ny {
            for i in 0..nx {
                // 像素中心坐标
                let (x, y) = gt.pixel_to_world(i as f64, j as f64);
                points.push(x);
                points.push(y);
                points.push(0.0); // Z=0 (或者可以把 h 放这里做地形起伏)
            }
        }

        // 2. 构建单元 (Cells) 并过滤无效区域
        let mut cell_indices = Vec::with_capacity(nx * ny * 4);
        let mut valid_cell_count = 0;

        // 数据缓冲区 (只存有效单元的数据)
        let mut out_h = Vec::with_capacity(nx * ny);
        let mut out_u = Vec::with_capacity(nx * ny);
        let mut out_v = Vec::with_capacity(nx * ny);

        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                // 检查是否为有效单元
                // 判据1: 显式掩膜
                let is_masked = if let Some(m) = mask {
                    m[[j, i]] == 0
                } else {
                    false
                };
                if is_masked {
                    continue;
                }

                // 判据2: NoData 值检查
                let h_val = h_data.data[[j, i]];
                if let Some(nd) = no_data {
                    if (h_val - nd).abs() < 1e-6 {
                        continue;
                    }
                }

                // 添加单元连接 (Quad: Counter-clockwise)
                // i,j -> i+1,j -> i+1,j+1 -> i,j+1
                let idx0 = (j * nx + i) as u32;
                let idx1 = idx0 + 1;
                let idx2 = ((j + 1) * nx + i + 1) as u32;
                let idx3 = ((j + 1) * nx + i) as u32;

                cell_indices.push(idx0);
                cell_indices.push(idx1);
                cell_indices.push(idx2);
                cell_indices.push(idx3);
                valid_cell_count += 1;

                // 添加 Cell Data
                out_h.push(h_val);
                out_u.push(u_data.data[[j, i]]);
                out_v.push(v_data.data[[j, i]]);
            }
        }

        // 3. 写入文件
        let piece = UnstructuredGridPiece::new(points, Cells::quads(cell_indices), vec![])
            .with_cell_data(vec![
                Data::scalar("Water_Depth", out_h),
                Data::scalar("Velocity_U", out_u),
                Data::scalar("Velocity_V", out_v),
            ]);

        Vtk::new(piece)
            .write(Path::new(path))
            .map_err(|e| MhError::Io {
                context: "VTK写入失败".into(),
                source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
            })?;

        Ok(())
    }
}
