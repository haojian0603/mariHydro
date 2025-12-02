// src-tauri/src/marihydro/io/exporters/vtu.rs
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::state::StateAccess;
use crate::marihydro::core::types::{CellIndex, NodeIndex};

/// VTU导出器的默认干湿阈值（用于速度计算）
const DEFAULT_VTU_H_DRY: f64 = 1e-6;

pub struct VtuExporter;

impl VtuExporter {
    /// 使用默认阈值导出
    pub fn export<M: MeshAccess, S: StateAccess>(path: &str, mesh: &M, state: &S, time: f64) -> MhResult<()> {
        Self::export_with_params(path, mesh, state, time, DEFAULT_VTU_H_DRY)
    }

    /// 使用自定义阈值导出
    pub fn export_with_params<M: MeshAccess, S: StateAccess>(
        path: &str, mesh: &M, state: &S, time: f64, h_dry: f64
    ) -> MhResult<()> {
        let file = File::create(path).map_err(|e| MhError::io(format!("Cannot create {}: {}", path, e)))?;
        let mut w = BufWriter::new(file);
        Self::write_header(&mut w, time)?;
        Self::write_piece(&mut w, mesh, state, h_dry)?;
        Self::write_footer(&mut w)?;
        w.flush().map_err(|e| MhError::io(e.to_string()))?;
        Ok(())
    }

    pub fn export_series<M: MeshAccess, S: StateAccess>(base_path: &str, mesh: &M, states: &[(f64, &S)]) -> MhResult<()> {
        let base = Path::new(base_path);
        let dir = base.parent().unwrap_or_else(|| Path::new("."));
        let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        std::fs::create_dir_all(dir).map_err(|e| MhError::io(format!("Cannot create dir: {}", e)))?;
        let mut vtu_files = Vec::new();
        for (i, (time, state)) in states.iter().enumerate() {
            let name = format!("{}_{:06}.vtu", stem, i);
            let path = dir.join(&name);
            Self::export(path.to_str().unwrap(), mesh, *state, *time)?;
            vtu_files.push((name, *time));
        }
        Self::write_pvd(&dir.join(format!("{}.pvd", stem)).to_string_lossy(), &vtu_files)?;
        Ok(())
    }

    fn write_pvd(path: &str, files: &[(String, f64)]) -> MhResult<()> {
        let file = File::create(path).map_err(|e| MhError::io(e.to_string()))?;
        let mut w = BufWriter::new(file);
        writeln!(w, r#"<?xml version="1.0"?>"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"  <Collection>"#).map_err(|e| MhError::io(e.to_string()))?;
        for (file, time) in files { writeln!(w, r#"    <DataSet timestep="{}" file="{}"/>"#, time, file).map_err(|e| MhError::io(e.to_string()))?; }
        writeln!(w, r#"  </Collection>"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"</VTKFile>"#).map_err(|e| MhError::io(e.to_string()))?;
        w.flush().map_err(|e| MhError::io(e.to_string()))?;
        Ok(())
    }

    fn write_header(w: &mut BufWriter<File>, time: f64) -> MhResult<()> {
        writeln!(w, r#"<?xml version="1.0"?>"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"  <UnstructuredGrid>"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"    <FieldData>"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"      <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1">{}</DataArray>"#, time).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"    </FieldData>"#).map_err(|e| MhError::io(e.to_string()))?;
        Ok(())
    }

    fn write_piece<M: MeshAccess, S: StateAccess>(w: &mut BufWriter<File>, mesh: &M, state: &S, h_dry: f64) -> MhResult<()> {
        let n_nodes = mesh.n_nodes();
        let n_cells = mesh.n_cells();
        writeln!(w, r#"    <Piece NumberOfPoints="{}" NumberOfCells="{}">"#, n_nodes, n_cells).map_err(|e| MhError::io(e.to_string()))?;
        Self::write_points(w, mesh)?;
        Self::write_cells(w, mesh)?;
        Self::write_cell_data(w, mesh, state, h_dry)?;
        writeln!(w, r#"    </Piece>"#).map_err(|e| MhError::io(e.to_string()))?;
        Ok(())
    }

    fn write_points<M: MeshAccess>(w: &mut BufWriter<File>, mesh: &M) -> MhResult<()> {
        writeln!(w, r#"      <Points>"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"        <DataArray type="Float64" NumberOfComponents="3" format="ascii">"#).map_err(|e| MhError::io(e.to_string()))?;
        for i in 0..mesh.n_nodes() {
            let pos = mesh.node_position(NodeIndex(i));
            writeln!(w, "          {:.6} {:.6} 0.0", pos.x, pos.y).map_err(|e| MhError::io(e.to_string()))?;
        }
        writeln!(w, r#"        </DataArray>"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"      </Points>"#).map_err(|e| MhError::io(e.to_string()))?;
        Ok(())
    }

    fn write_cells<M: MeshAccess>(w: &mut BufWriter<File>, mesh: &M) -> MhResult<()> {
        writeln!(w, r#"      <Cells>"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"        <DataArray type="Int32" Name="connectivity" format="ascii">"#).map_err(|e| MhError::io(e.to_string()))?;
        for i in 0..mesh.n_cells() {
            let nodes = mesh.cell_nodes(CellIndex(i));
            let s: Vec<String> = nodes.iter().map(|n| n.0.to_string()).collect();
            writeln!(w, "          {}", s.join(" ")).map_err(|e| MhError::io(e.to_string()))?;
        }
        writeln!(w, r#"        </DataArray>"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"        <DataArray type="Int32" Name="offsets" format="ascii">"#).map_err(|e| MhError::io(e.to_string()))?;
        let mut off = 0;
        for i in 0..mesh.n_cells() { off += mesh.cell_nodes(CellIndex(i)).len(); writeln!(w, "          {}", off).map_err(|e| MhError::io(e.to_string()))?; }
        writeln!(w, r#"        </DataArray>"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"        <DataArray type="UInt8" Name="types" format="ascii">"#).map_err(|e| MhError::io(e.to_string()))?;
        for i in 0..mesh.n_cells() { let t = match mesh.cell_nodes(CellIndex(i)).len() { 3 => 5, 4 => 9, _ => 7 }; writeln!(w, "          {}", t).map_err(|e| MhError::io(e.to_string()))?; }
        writeln!(w, r#"        </DataArray>"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"      </Cells>"#).map_err(|e| MhError::io(e.to_string()))?;
        Ok(())
    }

    fn write_cell_data<M: MeshAccess, S: StateAccess>(w: &mut BufWriter<File>, mesh: &M, state: &S, h_dry: f64) -> MhResult<()> {
        writeln!(w, r#"      <CellData>"#).map_err(|e| MhError::io(e.to_string()))?;
        Self::write_scalar_field(w, "h", mesh.n_cells(), |i| state.h(CellIndex(i)))?;
        Self::write_scalar_field(w, "eta", mesh.n_cells(), |i| state.h(CellIndex(i)) + mesh.cell_bed_elevation(CellIndex(i)))?;
        Self::write_scalar_field(w, "u", mesh.n_cells(), |i| { let h = state.h(CellIndex(i)); if h > h_dry { state.hu(CellIndex(i)) / h } else { 0.0 } })?;
        Self::write_scalar_field(w, "v", mesh.n_cells(), |i| { let h = state.h(CellIndex(i)); if h > h_dry { state.hv(CellIndex(i)) / h } else { 0.0 } })?;
        Self::write_scalar_field(w, "velocity_mag", mesh.n_cells(), |i| {
            let h = state.h(CellIndex(i)); if h > h_dry { let u = state.hu(CellIndex(i))/h; let v = state.hv(CellIndex(i))/h; (u*u + v*v).sqrt() } else { 0.0 }
        })?;
        writeln!(w, r#"      </CellData>"#).map_err(|e| MhError::io(e.to_string()))?;
        Ok(())
    }

    fn write_scalar_field<F: Fn(usize) -> f64>(w: &mut BufWriter<File>, name: &str, n: usize, f: F) -> MhResult<()> {
        writeln!(w, r#"        <DataArray type="Float64" Name="{}" format="ascii">"#, name).map_err(|e| MhError::io(e.to_string()))?;
        for i in 0..n { writeln!(w, "          {:.6}", f(i)).map_err(|e| MhError::io(e.to_string()))?; }
        writeln!(w, r#"        </DataArray>"#).map_err(|e| MhError::io(e.to_string()))?;
        Ok(())
    }

    fn write_footer(w: &mut BufWriter<File>) -> MhResult<()> {
        writeln!(w, r#"  </UnstructuredGrid>"#).map_err(|e| MhError::io(e.to_string()))?;
        writeln!(w, r#"</VTKFile>"#).map_err(|e| MhError::io(e.to_string()))?;
        Ok(())
    }
}
