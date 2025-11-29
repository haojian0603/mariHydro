//! VTU (VTK Unstructured Grid) 输出器

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;
use crate::marihydro::domain::state::ConservedState;
use crate::marihydro::infra::error::{MhError, MhResult};

pub struct VtuExporter;

impl VtuExporter {
    pub fn export(
        path: &str,
        mesh: &UnstructuredMesh,
        state: &ConservedState,
        time: f64,
    ) -> MhResult<()> {
        let file =
            File::create(path).map_err(|e| MhError::Io(format!("无法创建文件 {}: {}", path, e)))?;

        let mut w = BufWriter::new(file);

        Self::write_header(&mut w, time)?;
        Self::write_piece(&mut w, mesh, state)?;
        Self::write_footer(&mut w)?;

        w.flush()
            .map_err(|e| MhError::Io(format!("写入失败: {}", e)))?;

        log::info!("VTU 文件已导出: {}", path);

        Ok(())
    }

    pub fn export_series(
        base_path: &str,
        mesh: &UnstructuredMesh,
        states: &[(f64, &ConservedState)],
    ) -> MhResult<()> {
        let base = Path::new(base_path);
        let dir = base.parent().unwrap_or_else(|| Path::new("."));
        let stem = base.file_stem().unwrap().to_str().unwrap();

        std::fs::create_dir_all(dir).map_err(|e| MhError::Io(format!("无法创建目录: {}", e)))?;

        let mut vtu_files = Vec::new();

        for (i, (time, state)) in states.iter().enumerate() {
            let vtu_name = format!("{}_{:06}.vtu", stem, i);
            let vtu_path = dir.join(&vtu_name);
            Self::export(vtu_path.to_str().unwrap(), mesh, state, *time)?;
            vtu_files.push((vtu_name, *time));
        }

        let pvd_path = dir.join(format!("{}.pvd", stem));
        Self::write_pvd(pvd_path.to_str().unwrap(), &vtu_files)?;

        log::info!("时间序列已导出: {} 个时间步", states.len());

        Ok(())
    }

    fn write_pvd(path: &str, files: &[(String, f64)]) -> MhResult<()> {
        let file =
            File::create(path).map_err(|e| MhError::Io(format!("无法创建 PVD 文件: {}", e)))?;

        let mut w = BufWriter::new(file);

        writeln!(w, r#"<?xml version="1.0"?>"#).map_err(|e| MhError::Io(e.to_string()))?;
        writeln!(
            w,
            r#"<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">"#
        )
        .map_err(|e| MhError::Io(e.to_string()))?;
        writeln!(w, r#"  <Collection>"#).map_err(|e| MhError::Io(e.to_string()))?;

        for (file, time) in files {
            writeln!(w, r#"    <DataSet timestep="{}" file="{}"/>"#, time, file)
                .map_err(|e| MhError::Io(e.to_string()))?;
        }

        writeln!(w, r#"  </Collection>"#).map_err(|e| MhError::Io(e.to_string()))?;
        writeln!(w, r#"</VTKFile>"#).map_err(|e| MhError::Io(e.to_string()))?;

        w.flush()
            .map_err(|e| MhError::Io(format!("写入失败: {}", e)))?;

        log::info!("PVD 文件已创建: {}", path);

        Ok(())
    }

    fn write_header(w: &mut BufWriter<File>, time: f64) -> MhResult<()> {
        writeln!(w, r#"<?xml version="1.0"?>"#).map_err(|e| MhError::Io(e.to_string()))?;
        writeln!(
            w,
            r#"<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">"#
        )
        .map_err(|e| MhError::Io(e.to_string()))?;
        writeln!(w, r#"  <UnstructuredGrid>"#).map_err(|e| MhError::Io(e.to_string()))?;

        writeln!(w, r#"    <FieldData>"#).map_err(|e| MhError::Io(e.to_string()))?;
        writeln!(
            w,
            r#"      <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1">{}</DataArray>"#,
            time
        )
        .map_err(|e| MhError::Io(e.to_string()))?;
        writeln!(w, r#"    </FieldData>"#).map_err(|e| MhError::Io(e.to_string()))?;

        Ok(())
    }

    fn write_piece(
        w: &mut BufWriter<File>,
        mesh: &UnstructuredMesh,
        state: &ConservedState,
    ) -> MhResult<()> {
        writeln!(
            w,
            r#"    <Piece NumberOfPoints="{}" NumberOfCells="{}">"#,
            mesh.n_nodes, mesh.n_cells
        )
        .map_err(|e| MhError::Io(e.to_string()))?;

        Self::write_points(w, mesh)?;
        Self::write_cells(w, mesh)?;
        Self::write_cell_data(w, mesh, state)?;

        writeln!(w, r#"    </Piece>"#).map_err(|e| MhError::Io(e.to_string()))?;

        Ok(())
    }

    fn write_points(w: &mut BufWriter<File>, mesh: &UnstructuredMesh) -> MhResult<()> {
        writeln!(w, r#"      <Points>"#).map_err(|e| MhError::Io(e.to_string()))?;
        writeln!(
            w,
            r#"        <DataArray type="Float64" NumberOfComponents="3" format="ascii">"#
        )
        .map_err(|e| MhError::Io(e.to_string()))?;

        for i in 0..mesh.n_nodes {
            let pos = mesh.node_xy[i];
            let z = mesh.node_z[i];
            writeln!(w, "          {:.6} {:.6} {:.6}", pos.x, pos.y, z)
                .map_err(|e| MhError::Io(e.to_string()))?;
        }

        writeln!(w, r#"        </DataArray>"#).map_err(|e| MhError::Io(e.to_string()))?;
        writeln!(w, r#"      </Points>"#).map_err(|e| MhError::Io(e.to_string()))?;

        Ok(())
    }

    fn write_cells(w: &mut BufWriter<File>, mesh: &UnstructuredMesh) -> MhResult<()> {
        writeln!(w, r#"      <Cells>"#).map_err(|e| MhError::Io(e.to_string()))?;

        writeln!(
            w,
            r#"        <DataArray type="Int32" Name="connectivity" format="ascii">"#
        )
        .map_err(|e| MhError::Io(e.to_string()))?;

        for nodes in &mesh.cell_node_ids {
            let indices: Vec<String> = nodes.iter().map(|n| n.0.to_string()).collect();
            writeln!(w, "          {}", indices.join(" "))
                .map_err(|e| MhError::Io(e.to_string()))?;
        }

        writeln!(w, r#"        </DataArray>"#).map_err(|e| MhError::Io(e.to_string()))?;

        writeln!(
            w,
            r#"        <DataArray type="Int32" Name="offsets" format="ascii">"#
        )
        .map_err(|e| MhError::Io(e.to_string()))?;

        let mut offset = 0;
        for nodes in &mesh.cell_node_ids {
            offset += nodes.len();
            writeln!(w, "          {}", offset).map_err(|e| MhError::Io(e.to_string()))?;
        }

        writeln!(w, r#"        </DataArray>"#).map_err(|e| MhError::Io(e.to_string()))?;

        writeln!(
            w,
            r#"        <DataArray type="UInt8" Name="types" format="ascii">"#
        )
        .map_err(|e| MhError::Io(e.to_string()))?;

        for nodes in &mesh.cell_node_ids {
            let cell_type = match nodes.len() {
                3 => 5,
                4 => 9,
                _ => 7,
            };
            writeln!(w, "          {}", cell_type).map_err(|e| MhError::Io(e.to_string()))?;
        }

        writeln!(w, r#"        </DataArray>"#).map_err(|e| MhError::Io(e.to_string()))?;
        writeln!(w, r#"      </Cells>"#).map_err(|e| MhError::Io(e.to_string()))?;

        Ok(())
    }

    fn write_cell_data(
        w: &mut BufWriter<File>,
        mesh: &UnstructuredMesh,
        state: &ConservedState,
    ) -> MhResult<()> {
        writeln!(
            w,
            r#"      <CellData Scalars="WaterDepth" Vectors="Velocity">"#
        )
        .map_err(|e| MhError::Io(e.to_string()))?;

        Self::write_scalar_field(w, "WaterDepth", &state.h)?;

        let water_level: Vec<f64> = state
            .h
            .iter()
            .zip(&mesh.cell_z_bed)
            .map(|(h, z)| h + z)
            .collect();
        Self::write_scalar_field(w, "WaterLevel", &water_level)?;

        Self::write_velocity_field(w, state, 1e-6)?;

        let vel_mag: Vec<f64> = state
            .h
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                if h > 1e-6 {
                    let u = state.hu[i] / h;
                    let v = state.hv[i] / h;
                    (u * u + v * v).sqrt()
                } else {
                    0.0
                }
            })
            .collect();
        Self::write_scalar_field(w, "VelocityMagnitude", &vel_mag)?;

        Self::write_scalar_field(w, "BedElevation", &mesh.cell_z_bed)?;

        let froude: Vec<f64> = state
            .h
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                if h > 1e-3 {
                    let u = state.hu[i] / h;
                    let v = state.hv[i] / h;
                    let vel = (u * u + v * v).sqrt();
                    vel / (9.81 * h).sqrt()
                } else {
                    0.0
                }
            })
            .collect();
        Self::write_scalar_field(w, "FroudeNumber", &froude)?;

        writeln!(w, r#"      </CellData>"#).map_err(|e| MhError::Io(e.to_string()))?;

        Ok(())
    }

    fn write_scalar_field(w: &mut BufWriter<File>, name: &str, data: &[f64]) -> MhResult<()> {
        writeln!(
            w,
            r#"        <DataArray type="Float64" Name="{}" format="ascii">"#,
            name
        )
        .map_err(|e| MhError::Io(e.to_string()))?;

        for val in data {
            writeln!(w, "          {:.6}", val).map_err(|e| MhError::Io(e.to_string()))?;
        }

        writeln!(w, r#"        </DataArray>"#).map_err(|e| MhError::Io(e.to_string()))?;

        Ok(())
    }

    fn write_velocity_field(
        w: &mut BufWriter<File>,
        state: &ConservedState,
        eps: f64,
    ) -> MhResult<()> {
        writeln!(
            w,
            r#"        <DataArray type="Float64" Name="Velocity" NumberOfComponents="3" format="ascii">"#
        )
        .map_err(|e| MhError::Io(e.to_string()))?;

        for i in 0..state.n_cells {
            let (u, v) = if state.h[i] > eps {
                (state.hu[i] / state.h[i], state.hv[i] / state.h[i])
            } else {
                (0.0, 0.0)
            };
            writeln!(w, "          {:.6} {:.6} 0.0", u, v)
                .map_err(|e| MhError::Io(e.to_string()))?;
        }

        writeln!(w, r#"        </DataArray>"#).map_err(|e| MhError::Io(e.to_string()))?;

        Ok(())
    }

    fn write_footer(w: &mut BufWriter<File>) -> MhResult<()> {
        writeln!(w, r#"  </UnstructuredGrid>"#).map_err(|e| MhError::Io(e.to_string()))?;
        writeln!(w, r#"</VTKFile>"#).map_err(|e| MhError::Io(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;
    use glam::DVec2;
    use smallvec::smallvec;

    #[test]
    fn test_vtu_write() {
        let mut mesh = UnstructuredMesh::new();
        mesh.n_nodes = 3;
        mesh.node_xy = vec![DVec2::ZERO, DVec2::X, DVec2::Y];
        mesh.node_z = vec![0.0, 0.0, 0.0];

        mesh.n_cells = 1;
        mesh.cell_center = vec![DVec2::new(0.33, 0.33)];
        mesh.cell_area = vec![0.5];
        mesh.cell_z_bed = vec![0.0];
        mesh.cell_node_ids = vec![smallvec![
            crate::marihydro::domain::mesh::indices::NodeId(0),
            crate::marihydro::domain::mesh::indices::NodeId(1),
            crate::marihydro::domain::mesh::indices::NodeId(2)
        ]];

        let state = ConservedState::new(1);

        let result = VtuExporter::export("/tmp/marihydro_test.vtu", &mesh, &state, 0.0);
        assert!(result.is_ok());
    }
}
