//! 二进制 VTU 编码器 (Base64 + AppendedData)
//!
//! VTK 二进制格式规范：
//! - 数据块：Header (u32, 小端) + Data (原始二进制)
//! - AppendedData：base64 编码，用于 <DataArray format="appended">
//! - 支持 f32/f64 混合精度

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use std::io::{self, Write};
use mh_runtime::Backend;
use num_traits::ToPrimitive;
use serde_json;

/// 二进制编码器
pub struct BinaryEncoder {
    buffer: Vec<u8>,
}

impl BinaryEncoder {
    /// 创建新编码器，预分配 1MB 缓冲区
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(1 * 1024 * 1024),
        }
    }

    /// 编码 Header (4 bytes)
    #[inline]
    fn encode_header(&mut self, byte_len: usize) -> io::Result<usize> {
        if byte_len > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "VTU appended data too large",
            ));
        }
        let offset = self.buffer.len();
        let len = byte_len as u32;
        self.buffer.extend_from_slice(&len.to_le_bytes());
        Ok(offset)
    }

    /// 编码 f64 数组（小端字节序）
    pub fn encode_f64(&mut self, data: &[f64]) -> io::Result<usize> {
        let byte_len = data.len().checked_mul(8).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "VTU data size overflow")
        })?;
        let offset = self.encode_header(byte_len)?;
        for &val in data {
            self.buffer.extend_from_slice(&val.to_le_bytes());
        }
        Ok(offset)
    }

    /// 编码 f32 数组（小端字节序）
    ///
    /// 用于大规模模拟场景，可再减少50%存储空间
    #[allow(dead_code)]
    pub fn encode_f32(&mut self, data: &[f32]) -> io::Result<usize> {
        let byte_len = data.len().checked_mul(4).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "VTU data size overflow")
        })?;
        let offset = self.encode_header(byte_len)?;
        for &val in data {
            self.buffer.extend_from_slice(&val.to_le_bytes());
        }
        Ok(offset)
    }

    /// 编码 i32 数组（小端字节序）
    pub fn encode_i32(&mut self, data: &[i32]) -> io::Result<usize> {
        let byte_len = data.len().checked_mul(4).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "VTU data size overflow")
        })?;
        let offset = self.encode_header(byte_len)?;
        for &val in data {
            self.buffer.extend_from_slice(&val.to_le_bytes());
        }
        Ok(offset)
    }

    /// 编码 u8 数组（小端字节序）
    pub fn encode_u8(&mut self, data: &[u8]) -> io::Result<usize> {
        let offset = self.encode_header(data.len())?;
        self.buffer.extend_from_slice(data);
        Ok(offset)
    }

    /// 将 AppendedData 节写入 XML
    pub fn write_appended<W: Write>(&mut self, writer: &mut W) -> io::Result<()> {
        let encoded = BASE64.encode(&self.buffer);

        writeln!(writer, "  <AppendedData encoding=\"base64\">")?;
        writeln!(writer, "   _{}", encoded)?;
        writeln!(writer, "  </AppendedData>")?;

        self.buffer.clear();
        Ok(())
    }
}

impl Default for BinaryEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// 二进制 VTU 完整写入器
pub fn write_vtu_binary<W: Write, B: Backend>(
    writer: &mut W,
    mesh: &crate::snapshot::MeshSnapshot<B>,
    state: &crate::snapshot::StateSnapshot<B>,
    time: f64,
) -> io::Result<()> {
    writeln!(writer, r#"<?xml version="1.0"?>"#)?;
    writeln!(
        writer,
        r#"<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">"#
    )?;
    writeln!(writer, r#"  <UnstructuredGrid>"#)?;

    writeln!(writer, r#"    <FieldData>"#)?;
    writeln!(
        writer,
        r#"      <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1" format="ascii">{}</DataArray>"#,
        time
    )?;
    if let (Some(faces), Some(ids)) = (&mesh.boundary_faces, &mesh.boundary_ids) {
        writeln!(
            writer,
            r#"      <DataArray type="Int32" Name="boundary_faces" format="ascii">{} </DataArray>"#,
            faces.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")
        )?;
        writeln!(
            writer,
            r#"      <DataArray type="Int32" Name="boundary_ids" format="ascii">{} </DataArray>"#,
            ids.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")
        )?;
    }
    if let Some(names) = &mesh.boundary_names {
        let serialized = serde_json::to_string(names).unwrap_or_else(|_| "[]".into());
        writeln!(
            writer,
            r#"      <DataArray type="String" Name="boundary_names" NumberOfTuples="1" format="ascii">{}</DataArray>"#,
            serialized
        )?;
    }
    writeln!(writer, r#"    </FieldData>"#)?;

    writeln!(
        writer,
        r#"    <Piece NumberOfPoints="{}" NumberOfCells="{}">"#,
        mesh.n_nodes, mesh.n_cells
    )?;

    let mut encoder = BinaryEncoder::new();

    writeln!(writer, r#"      <Points>"#)?;
    let mut coords_3d = Vec::with_capacity(mesh.n_nodes * 3);
    for &(x, y) in &mesh.node_positions {
        coords_3d.push(x);
        coords_3d.push(y);
        coords_3d.push(0.0);
    }
    let points_offset = encoder.encode_f64(&coords_3d)?;
    writeln!(
        writer,
        r#"        <DataArray type="Float64" NumberOfComponents="3" format="appended" offset="{}"/>"#,
        points_offset
    )?;
    writeln!(writer, r#"      </Points>"#)?;

    writeln!(writer, r#"      <Cells>"#)?;
    let connectivity: Vec<i32> = mesh
        .cell_nodes
        .iter()
        .flatten()
        .map(|&n| n as i32)
        .collect();
    let connectivity_offset = encoder.encode_i32(&connectivity)?;
    writeln!(
        writer,
        r#"        <DataArray type="Int32" Name="connectivity" format="appended" offset="{}"/>"#,
        connectivity_offset
    )?;

    let mut offset = 0i32;
    let offsets: Vec<i32> = mesh
        .cell_nodes
        .iter()
        .map(|nodes| {
            offset += nodes.len() as i32;
            offset
        })
        .collect();
    let offsets_offset = encoder.encode_i32(&offsets)?;
    writeln!(
        writer,
        r#"        <DataArray type="Int32" Name="offsets" format="appended" offset="{}"/>"#,
        offsets_offset
    )?;

    let types: Vec<u8> = mesh
        .cell_nodes
        .iter()
        .map(|nodes| match nodes.len() {
            3 => 5,
            4 => 9,
            _ => 7,
        })
        .collect();
    let types_offset = encoder.encode_u8(&types)?;
    writeln!(
        writer,
        r#"        <DataArray type="UInt8" Name="types" format="appended" offset="{}"/>"#,
        types_offset
    )?;
    writeln!(writer, r#"      </Cells>"#)?;

    writeln!(writer, r#"      <CellData Scalars="h">"#)?;
    let h: Vec<f64> = state.h.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
    let hu: Vec<f64> = state.hu.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
    let hv: Vec<f64> = state.hv.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
    let h_offset = encoder.encode_f64(&h)?;
    writeln!(
        writer,
        r#"        <DataArray type="Float64" Name="h" format="appended" offset="{}"/>"#,
        h_offset
    )?;

    let bed: Vec<f64> = mesh
        .bed_elevations
        .iter()
        .map(|v| v.to_f64().unwrap_or(0.0))
        .collect();
    let eta: Vec<f64> = h.iter().zip(&bed).map(|(h, z)| h + z).collect();
    let eta_offset = encoder.encode_f64(&eta)?;
    writeln!(
        writer,
        r#"        <DataArray type="Float64" Name="eta" format="appended" offset="{}"/>"#,
        eta_offset
    )?;

    let mut velocity = Vec::with_capacity(h.len() * 2);
    for i in 0..h.len() {
        let (u, v) = if h[i] > 1e-6 {
            (hu[i] / h[i], hv[i] / h[i])
        } else {
            (0.0, 0.0)
        };
        velocity.push(u);
        velocity.push(v);
    }
    let velocity_offset = encoder.encode_f64(&velocity)?;
    writeln!(
        writer,
        r#"        <DataArray type="Float64" Name="velocity" NumberOfComponents="2" format="appended" offset="{}"/>"#,
        velocity_offset
    )?;

    let bed_offset = encoder.encode_f64(&bed)?;
    writeln!(
        writer,
        r#"        <DataArray type="Float64" Name="bed_elevation" format="appended" offset="{}"/>"#,
        bed_offset
    )?;

    if let (Some(scalars), Some(names)) = (&state.scalars, &state.scalar_names) {
        for (scalar, name) in scalars.iter().zip(names.iter()) {
            let vals: Vec<f64> = scalar.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
            let offset = encoder.encode_f64(&vals)?;
            writeln!(
                writer,
                r#"        <DataArray type="Float64" Name="{}" format="appended" offset="{}"/>"#,
                name,
                offset
            )?;
        }
    }
    writeln!(writer, r#"      </CellData>"#)?;
    writeln!(writer, r#"    </Piece>"#)?;

    encoder.write_appended(writer)?;

    writeln!(writer, r#"  </UnstructuredGrid>"#)?;
    writeln!(writer, r#"</VTKFile>"#)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{MeshSnapshot, StateSnapshot};
    use mh_runtime::CpuBackend;

    #[test]
    fn test_binary_encoder_f64() {
        let mut encoder = BinaryEncoder::new();
        let data = vec![1.0f64, 2.0, 3.0];
        encoder.encode_f64(&data).unwrap();

        assert_eq!(encoder.buffer.len(), 28);
        let expected_len = 24u32.to_le_bytes();
        assert_eq!(&encoder.buffer[0..4], &expected_len);
    }

    #[test]
    fn test_binary_encoder_f32() {
        let mut encoder = BinaryEncoder::new();
        let data = vec![1.0f32, 2.0, 3.0];
        encoder.encode_f32(&data).unwrap();

        assert_eq!(encoder.buffer.len(), 16);
        let expected_len = 12u32.to_le_bytes();
        assert_eq!(&encoder.buffer[0..4], &expected_len);
    }

    #[test]
    fn test_binary_vtu_output() {
        let mesh = MeshSnapshot::<CpuBackend<f64>>::from_mesh_data(
            4, 1,
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            vec![vec![0, 1, 2, 3]],
            vec![1.0],
            vec![0.0],
        );

        let state = StateSnapshot::<CpuBackend<f64>>::from_state_data(
            vec![1.0],
            vec![0.1],
            vec![0.0],
        );

        let mut output = Vec::new();
        write_vtu_binary(&mut output, &mesh, &state, 0.0).unwrap();

        let output_str = String::from_utf8_lossy(&output);
        assert!(output_str.contains("<VTKFile"));
        assert!(output_str.contains("format=\"appended\""));
        assert!(output_str.contains("<AppendedData"));
    }
}