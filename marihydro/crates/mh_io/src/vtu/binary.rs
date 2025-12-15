//! 二进制 VTU 编码器 (Base64 + AppendedData)
//!
//! VTK 二进制格式规范：
//! - 数据块：Header (u32, 小端) + Data (原始二进制)
//! - AppendedData：base64 编码，用于 <DataArray format="appended">
//! - 支持 f32/f64 混合精度

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use std::io::{self, Write};

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

    /// 编码 Header (4 bytes) + Data
    #[inline]
    fn encode_with_header(&mut self, data: &[u8]) {
        let len = data.len() as u32;
        self.buffer.extend_from_slice(&len.to_le_bytes());
        self.buffer.extend_from_slice(data);
    }

    /// 编码 f64 数组（小端字节序）
    pub fn encode_f64(&mut self, data: &[f64]) {
        for &val in data {
            self.encode_with_header(&val.to_le_bytes());
        }
    }

    /// 编码 f32 数组（小端字节序）
    /// 
    /// 用于大规模模拟场景，可再减少50%存储空间
    #[allow(dead_code)]
    pub fn encode_f32(&mut self, data: &[f32]) {
        for &val in data {
            self.encode_with_header(&val.to_le_bytes());
        }
    }

    /// 编码 i32 数组（小端字节序）
    pub fn encode_i32(&mut self, data: &[i32]) {
        for &val in data {
            self.encode_with_header(&val.to_le_bytes());
        }
    }

    /// 将 AppendedData 节写入 XML
    pub fn write_appended<W: Write>(&mut self, writer: &mut W) -> io::Result<()> {
        let encoded = BASE64.encode(&self.buffer);
        
        writeln!(writer, "  <AppendedData encoding=\"base64\">")?;
        writeln!(writer, "   {}", encoded)?;
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
pub fn write_vtu_binary<W: Write>(
    writer: &mut W,
    mesh: &crate::snapshot::MeshSnapshot,
    state: &crate::snapshot::StateSnapshot,
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
    writeln!(writer, r#"    </FieldData>"#)?;
    
    writeln!(
        writer,
        r#"    <Piece NumberOfPoints="{}" NumberOfCells="{}">"#,
        mesh.n_nodes, mesh.n_cells
    )?;

    let mut encoder = BinaryEncoder::new();

    writeln!(writer, r#"      <Points>"#)?;
    writeln!(
        writer,
        r#"        <DataArray type="Float64" NumberOfComponents="3" format="appended">"#
    )?;
    
    let mut coords_3d = Vec::with_capacity(mesh.n_nodes * 3);
    for &(x, y) in &mesh.node_positions {
        coords_3d.push(x);
        coords_3d.push(y);
        coords_3d.push(0.0);
    }
    encoder.encode_f64(&coords_3d);
    writeln!(writer, "        </DataArray>")?;
    writeln!(writer, r#"      </Points>"#)?;

    writeln!(writer, r#"      <Cells>"#)?;
    
    writeln!(
        writer,
        r#"        <DataArray type="Int32" Name="connectivity" format="appended">"#
    )?;
    let connectivity: Vec<i32> = mesh.cell_nodes.iter()
        .flatten()
        .map(|&n| n as i32)
        .collect();
    encoder.encode_i32(&connectivity);
    writeln!(writer, "        </DataArray>")?;

    writeln!(
        writer,
        r#"        <DataArray type="Int32" Name="offsets" format="appended">"#
    )?;
    let mut offset = 0i32;
    let offsets: Vec<i32> = mesh.cell_nodes.iter()
        .map(|nodes| {
            offset += nodes.len() as i32;
            offset
        })
        .collect();
    encoder.encode_i32(&offsets);
    writeln!(writer, "        </DataArray>")?;

    writeln!(
        writer,
        r#"        <DataArray type="UInt8" Name="types" format="appended">"#
    )?;
    let types: Vec<u8> = mesh.cell_nodes.iter()
        .map(|nodes| match nodes.len() {
            3 => 5,
            4 => 9,
            _ => 7,
        })
        .collect();
    encoder.buffer.extend_from_slice(&types);
    writeln!(writer, "        </DataArray>")?;

    writeln!(writer, r#"      </Cells>"#)?;

    writeln!(writer, r#"      <CellData Scalars="h">"#)?;
    
    writeln!(
        writer,
        r#"        <DataArray type="Float64" Name="h" format="appended">"#
    )?;
    encoder.encode_f64(&state.h);
    writeln!(writer, "        </DataArray>")?;

    writeln!(
        writer,
        r#"        <DataArray type="Float64" Name="eta" format="appended">"#
    )?;
    let eta: Vec<f64> = state.h.iter()
        .zip(&mesh.bed_elevations)
        .map(|(h, z)| h + z)
        .collect();
    encoder.encode_f64(&eta);
    writeln!(writer, "        </DataArray>")?;

    writeln!(
        writer,
        r#"        <DataArray type="Float64" Name="velocity" NumberOfComponents="2" format="appended">"#
    )?;
    let mut velocity = Vec::with_capacity(state.h.len() * 2);
    for i in 0..state.h.len() {
        let (u, v) = if state.h[i] > 1e-6 {
            (state.hu[i] / state.h[i], state.hv[i] / state.h[i])
        } else {
            (0.0, 0.0)
        };
        velocity.push(u);
        velocity.push(v);
    }
    encoder.encode_f64(&velocity);
    writeln!(writer, "        </DataArray>")?;

    writeln!(
        writer,
        r#"        <DataArray type="Float64" Name="bed_elevation" format="appended">"#
    )?;
    encoder.encode_f64(&mesh.bed_elevations);
    writeln!(writer, "        </DataArray>")?;

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

    #[test]
    fn test_binary_encoder_f64() {
        let mut encoder = BinaryEncoder::new();
        let data = vec![1.0f64, 2.0, 3.0];
        encoder.encode_f64(&data);
        
        assert_eq!(encoder.buffer.len(), 36);
        let expected_len = 8u32.to_le_bytes();
        assert_eq!(&encoder.buffer[0..4], &expected_len);
    }

    #[test]
    fn test_binary_encoder_f32() {
        let mut encoder = BinaryEncoder::new();
        let data = vec![1.0f32, 2.0, 3.0];
        encoder.encode_f32(&data);
        
        assert_eq!(encoder.buffer.len(), 24);
        let expected_len = 4u32.to_le_bytes();
        assert_eq!(&encoder.buffer[0..4], &expected_len);
    }

    #[test]
    fn test_binary_vtu_output() {
        let mesh = MeshSnapshot::from_mesh_data(
            4, 1,
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            vec![vec![0, 1, 2, 3]],
            vec![1.0],
            vec![0.0],
        );
        
        let state = StateSnapshot::from_state_data(
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