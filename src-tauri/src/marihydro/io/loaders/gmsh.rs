//! src-tauri\src\marihydro\io\loaders\gmsh.rs

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use glam::DVec2;
use smallvec::SmallVec;

use crate::marihydro::domain::mesh::indices::*;
use crate::marihydro::domain::mesh::unstructured::*;
use crate::marihydro::infra::error::{MhError, MhResult};

const BC_MAPPING: &[(&str, BoundaryKind)] = &[
    ("wall", BoundaryKind::Wall),
    ("solid", BoundaryKind::Wall),
    ("land", BoundaryKind::Wall),
    ("coast", BoundaryKind::Wall),
    ("inlet", BoundaryKind::RiverInflow),
    ("inflow", BoundaryKind::RiverInflow),
    ("river", BoundaryKind::RiverInflow),
    ("upstream", BoundaryKind::RiverInflow),
    ("outlet", BoundaryKind::Outflow),
    ("outflow", BoundaryKind::Outflow),
    ("downstream", BoundaryKind::Outflow),
    ("open", BoundaryKind::OpenSea),
    ("sea", BoundaryKind::OpenSea),
    ("tide", BoundaryKind::OpenSea),
    ("ocean", BoundaryKind::OpenSea),
    ("symmetry", BoundaryKind::Symmetry),
    ("sym", BoundaryKind::Symmetry),
];

pub struct GmshLoader;

impl GmshLoader {
    pub fn load(path: &str) -> MhResult<UnstructuredMesh> {
        let file = File::open(path)
            .map_err(|e| MhError::Io(format!("无法打开网格文件 {}: {}", path, e)))?;

        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let mut nodes_xy: Vec<DVec2> = Vec::new();
        let mut nodes_z: Vec<f64> = Vec::new();
        let mut node_tag_map: HashMap<usize, usize> = HashMap::new();
        let mut elements_2d: Vec<Vec<usize>> = Vec::new();
        let mut elements_1d: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut physical_names: HashMap<usize, String> = HashMap::new();
        let mut format_version = 2;

        while let Some(Ok(line)) = lines.next() {
            match line.trim() {
                "$MeshFormat" => {
                    if let Some(Ok(fmt_line)) = lines.next() {
                        let version: f64 = fmt_line
                            .split_whitespace()
                            .next()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(2.0);
                        format_version = version as i32;
                    }
                    Self::skip_to_end(&mut lines, "$EndMeshFormat");
                }
                "$PhysicalNames" => {
                    physical_names = Self::parse_physical_names(&mut lines)?;
                }
                "$Nodes" => {
                    let (xy, z, tag_map) = if format_version >= 4 {
                        Self::parse_nodes_v4(&mut lines)?
                    } else {
                        Self::parse_nodes_v2(&mut lines)?
                    };
                    nodes_xy = xy;
                    nodes_z = z;
                    node_tag_map = tag_map;
                }
                "$Elements" => {
                    let (e2d, e1d) = if format_version >= 4 {
                        Self::parse_elements_v4(&mut lines, &node_tag_map)?
                    } else {
                        Self::parse_elements_v2(&mut lines, &node_tag_map)?
                    };
                    elements_2d = e2d;
                    elements_1d = e1d;
                }
                _ => {}
            }
        }

        if nodes_xy.is_empty() {
            return Err(MhError::InvalidMesh {
                message: "网格文件不包含节点".into(),
            });
        }

        if elements_2d.is_empty() {
            return Err(MhError::InvalidMesh {
                message: "网格文件不包含 2D 单元".into(),
            });
        }

        log::info!(
            "Gmsh 加载完成: {} 节点, {} 单元, {} 边界边",
            nodes_xy.len(),
            elements_2d.len(),
            elements_1d.len()
        );

        Self::build_mesh(nodes_xy, nodes_z, elements_2d, elements_1d, physical_names)
    }

    fn skip_to_end(lines: &mut impl Iterator<Item = std::io::Result<String>>, end_marker: &str) {
        while let Some(Ok(line)) = lines.next() {
            if line.trim() == end_marker {
                break;
            }
        }
    }

    fn parse_physical_names(
        lines: &mut impl Iterator<Item = std::io::Result<String>>,
    ) -> MhResult<HashMap<usize, String>> {
        let mut groups = HashMap::new();

        if let Some(Ok(_count_line)) = lines.next() {}

        while let Some(Ok(line)) = lines.next() {
            let trimmed = line.trim();
            if trimmed == "$EndPhysicalNames" {
                break;
            }

            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(tag) = parts[1].parse::<usize>() {
                    let name = parts[2..].join(" ");
                    let name = name.trim_matches('"').to_lowercase();
                    groups.insert(tag, name);
                }
            }
        }

        Ok(groups)
    }

    fn parse_nodes_v2(
        lines: &mut impl Iterator<Item = std::io::Result<String>>,
    ) -> MhResult<(Vec<DVec2>, Vec<f64>, HashMap<usize, usize>)> {
        let mut xy = Vec::new();
        let mut z = Vec::new();
        let mut tag_map = HashMap::new();

        if let Some(Ok(count_line)) = lines.next() {
            if let Ok(count) = count_line.trim().parse::<usize>() {
                xy.reserve(count);
                z.reserve(count);
                tag_map.reserve(count);
            }
        }

        while let Some(Ok(line)) = lines.next() {
            let trimmed = line.trim();
            if trimmed == "$EndNodes" {
                break;
            }

            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 4 {
                if let (Ok(tag), Ok(x), Ok(y), Ok(zv)) = (
                    parts[0].parse::<usize>(),
                    parts[1].parse::<f64>(),
                    parts[2].parse::<f64>(),
                    parts[3].parse::<f64>(),
                ) {
                    tag_map.insert(tag, xy.len());
                    xy.push(DVec2::new(x, y));
                    z.push(zv);
                }
            }
        }

        Ok((xy, z, tag_map))
    }

    fn parse_nodes_v4(
        lines: &mut impl Iterator<Item = std::io::Result<String>>,
    ) -> MhResult<(Vec<DVec2>, Vec<f64>, HashMap<usize, usize>)> {
        let mut xy = Vec::new();
        let mut z = Vec::new();
        let mut tag_map = HashMap::new();

        let header = lines.next().ok_or_else(|| MhError::InvalidMesh {
            message: "节点块头部缺失".into(),
        })??;
        let parts: Vec<usize> = header
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        if parts.len() < 4 {
            return Err(MhError::InvalidMesh {
                message: "节点块头部格式错误".into(),
            });
        }

        let num_blocks = parts[0];
        let total_nodes = parts[1];

        xy.reserve(total_nodes);
        z.reserve(total_nodes);
        tag_map.reserve(total_nodes);

        for _ in 0..num_blocks {
            let block_header = lines.next().ok_or_else(|| MhError::InvalidMesh {
                message: "节点实体块头部缺失".into(),
            })??;
            let bh: Vec<usize> = block_header
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            if bh.len() < 4 {
                continue;
            }

            let num_nodes_in_block = bh[3];

            let mut tags = Vec::with_capacity(num_nodes_in_block);
            for _ in 0..num_nodes_in_block {
                let tag_line = lines.next().ok_or_else(|| MhError::InvalidMesh {
                    message: "节点标签缺失".into(),
                })??;
                if let Ok(tag) = tag_line.trim().parse::<usize>() {
                    tags.push(tag);
                }
            }

            for tag in tags {
                let coord_line = lines.next().ok_or_else(|| MhError::InvalidMesh {
                    message: "节点坐标缺失".into(),
                })??;
                let coords: Vec<f64> = coord_line
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();

                if coords.len() >= 3 {
                    tag_map.insert(tag, xy.len());
                    xy.push(DVec2::new(coords[0], coords[1]));
                    z.push(coords[2]);
                }
            }
        }

        Self::skip_to_end(lines, "$EndNodes");

        Ok((xy, z, tag_map))
    }

    fn parse_elements_v2(
        lines: &mut impl Iterator<Item = std::io::Result<String>>,
        node_tag_map: &HashMap<usize, usize>,
    ) -> MhResult<(Vec<Vec<usize>>, Vec<(usize, Vec<usize>)>)> {
        let mut elements_2d = Vec::new();
        let mut elements_1d = Vec::new();

        lines.next();

        while let Some(Ok(line)) = lines.next() {
            let trimmed = line.trim();
            if trimmed == "$EndElements" {
                break;
            }

            let parts: Vec<&str> = trimmed.split_whitespace().collect();

            if parts.len() < 4 {
                continue;
            }

            let elem_type = parts[1].parse::<usize>().unwrap_or(0);
            let num_tags = parts[2].parse::<usize>().unwrap_or(0);
            let tag = if num_tags > 0 {
                parts[3].parse::<usize>().unwrap_or(0)
            } else {
                0
            };
            let nodes_start = 3 + num_tags;

            match elem_type {
                1 => {
                    if parts.len() >= nodes_start + 2 {
                        let nodes: Option<Vec<usize>> = parts[nodes_start..]
                            .iter()
                            .take(2)
                            .map(|s| {
                                s.parse::<usize>()
                                    .ok()
                                    .and_then(|tag| node_tag_map.get(&tag).copied())
                            })
                            .collect();

                        if let Some(nodes) = nodes {
                            elements_1d.push((tag, nodes));
                        }
                    }
                }
                2 => {
                    if parts.len() >= nodes_start + 3 {
                        let nodes: Option<Vec<usize>> = parts[nodes_start..]
                            .iter()
                            .take(3)
                            .map(|s| {
                                s.parse::<usize>()
                                    .ok()
                                    .and_then(|tag| node_tag_map.get(&tag).copied())
                            })
                            .collect();

                        if let Some(nodes) = nodes {
                            elements_2d.push(nodes);
                        }
                    }
                }
                3 => {
                    if parts.len() >= nodes_start + 4 {
                        let nodes: Option<Vec<usize>> = parts[nodes_start..]
                            .iter()
                            .take(4)
                            .map(|s| {
                                s.parse::<usize>()
                                    .ok()
                                    .and_then(|tag| node_tag_map.get(&tag).copied())
                            })
                            .collect();

                        if let Some(nodes) = nodes {
                            elements_2d.push(nodes);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok((elements_2d, elements_1d))
    }

    fn parse_elements_v4(
        lines: &mut impl Iterator<Item = std::io::Result<String>>,
        node_tag_map: &HashMap<usize, usize>,
    ) -> MhResult<(Vec<Vec<usize>>, Vec<(usize, Vec<usize>)>)> {
        let mut elements_2d = Vec::new();
        let mut elements_1d = Vec::new();

        let header = lines.next().ok_or_else(|| MhError::InvalidMesh {
            message: "单元块头部缺失".into(),
        })??;
        let parts: Vec<usize> = header
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        if parts.len() < 4 {
            return Err(MhError::InvalidMesh {
                message: "单元块头部格式错误".into(),
            });
        }

        let num_blocks = parts[0];

        for _ in 0..num_blocks {
            let block_header = lines.next().ok_or_else(|| MhError::InvalidMesh {
                message: "单元实体块头部缺失".into(),
            })??;
            let bh: Vec<usize> = block_header
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            if bh.len() < 4 {
                continue;
            }

            let entity_tag = bh[1];
            let elem_type = bh[2];
            let num_elems = bh[3];

            for _ in 0..num_elems {
                let elem_line = lines.next().ok_or_else(|| MhError::InvalidMesh {
                    message: "单元数据缺失".into(),
                })??;
                let parts: Vec<usize> = elem_line
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();

                if parts.is_empty() {
                    continue;
                }

                let node_tags = &parts[1..];

                match elem_type {
                    1 => {
                        if node_tags.len() >= 2 {
                            let nodes: Option<Vec<usize>> = node_tags
                                .iter()
                                .take(2)
                                .map(|&tag| node_tag_map.get(&tag).copied())
                                .collect();

                            if let Some(nodes) = nodes {
                                elements_1d.push((entity_tag, nodes));
                            }
                        }
                    }
                    2 => {
                        if node_tags.len() >= 3 {
                            let nodes: Option<Vec<usize>> = node_tags
                                .iter()
                                .take(3)
                                .map(|&tag| node_tag_map.get(&tag).copied())
                                .collect();

                            if let Some(nodes) = nodes {
                                elements_2d.push(nodes);
                            }
                        }
                    }
                    3 => {
                        if node_tags.len() >= 4 {
                            let nodes: Option<Vec<usize>> = node_tags
                                .iter()
                                .take(4)
                                .map(|&tag| node_tag_map.get(&tag).copied())
                                .collect();

                            if let Some(nodes) = nodes {
                                elements_2d.push(nodes);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        Self::skip_to_end(lines, "$EndElements");

        Ok((elements_2d, elements_1d))
    }

    fn map_boundary_kind(name: &str) -> BoundaryKind {
        let lower = name.to_lowercase();

        for (pattern, kind) in BC_MAPPING {
            if lower.contains(pattern) {
                return *kind;
            }
        }

        BoundaryKind::Wall
    }

    fn build_mesh(
        nodes_xy: Vec<DVec2>,
        nodes_z: Vec<f64>,
        elements_2d: Vec<Vec<usize>>,
        elements_1d: Vec<(usize, Vec<usize>)>,
        physical_names: HashMap<usize, String>,
    ) -> MhResult<UnstructuredMesh> {
        let n_nodes = nodes_xy.len();
        let n_cells = elements_2d.len();

        let mut cell_center = Vec::with_capacity(n_cells);
        let mut cell_area = Vec::with_capacity(n_cells);
        let mut cell_z_bed = Vec::with_capacity(n_cells);
        let mut cell_node_ids = Vec::with_capacity(n_cells);

        for nodes in &elements_2d {
            let n = nodes.len();

            let mut cx = 0.0;
            let mut cy = 0.0;
            let mut zb = 0.0;
            for &nid in nodes {
                cx += nodes_xy[nid].x;
                cy += nodes_xy[nid].y;
                zb += nodes_z[nid];
            }
            cell_center.push(DVec2::new(cx / n as f64, cy / n as f64));
            cell_z_bed.push(zb / n as f64);

            let mut area = 0.0;
            for i in 0..n {
                let j = (i + 1) % n;
                let pi = nodes_xy[nodes[i]];
                let pj = nodes_xy[nodes[j]];
                area += pi.x * pj.y - pj.x * pi.y;
            }
            cell_area.push(area.abs() * 0.5);

            cell_node_ids.push(nodes.iter().map(|&n| NodeId(n)).collect());
        }

        let mut edge_to_cells: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

        for (cell_idx, nodes) in elements_2d.iter().enumerate() {
            let n = nodes.len();
            for i in 0..n {
                let n1 = nodes[i];
                let n2 = nodes[(i + 1) % n];
                let edge = if n1 < n2 { (n1, n2) } else { (n2, n1) };
                edge_to_cells.entry(edge).or_default().push(cell_idx);
            }
        }

        let mut boundary_edges: HashMap<(usize, usize), usize> = HashMap::new();
        for (tag, nodes) in &elements_1d {
            if nodes.len() >= 2 {
                let n1 = nodes[0];
                let n2 = nodes[1];
                let edge = if n1 < n2 { (n1, n2) } else { (n2, n1) };
                boundary_edges.insert(edge, *tag);
            }
        }

        let mut interior_faces_data: Vec<((usize, usize), usize, usize)> = Vec::new();
        let mut boundary_faces_data: Vec<((usize, usize), usize, usize)> = Vec::new();

        let mut sorted_edges: Vec<_> = edge_to_cells.iter().collect();
        sorted_edges.sort_by_key(|(edge, _)| *edge);

        for (edge, cells) in sorted_edges {
            if cells.len() == 2 {
                interior_faces_data.push((*edge, cells[0], cells[1]));
            } else if cells.len() == 1 {
                let bc_tag = boundary_edges.get(edge).copied().unwrap_or(0);
                boundary_faces_data.push((*edge, cells[0], bc_tag));
            }
        }

        let n_interior = interior_faces_data.len();
        let n_boundary = boundary_faces_data.len();
        let n_faces = n_interior + n_boundary;

        let mut face_center = Vec::with_capacity(n_faces);
        let mut face_normal = Vec::with_capacity(n_faces);
        let mut face_length = Vec::with_capacity(n_faces);
        let mut face_z_left = Vec::with_capacity(n_faces);
        let mut face_z_right = Vec::with_capacity(n_faces);
        let mut face_owner = Vec::with_capacity(n_faces);
        let mut face_neighbor = Vec::with_capacity(n_faces);
        let mut face_delta_owner = Vec::with_capacity(n_faces);
        let mut face_delta_neighbor = Vec::with_capacity(n_faces);
        let mut face_dist_o2n = Vec::with_capacity(n_faces);

        let mut cell_faces: Vec<CellFaces> = (0..n_cells).map(|_| CellFaces::default()).collect();

        let compute_face_geo = |edge: (usize, usize),
                                owner: usize,
                                is_boundary: bool|
         -> (DVec2, DVec2, f64, f64, f64) {
            let p1 = nodes_xy[edge.0];
            let p2 = nodes_xy[edge.1];
            let z1 = nodes_z[edge.0];
            let z2 = nodes_z[edge.1];

            let center = (p1 + p2) * 0.5;
            let diff = p2 - p1;
            let length = diff.length();

            let outward_normal = DVec2::new(-diff.y, diff.x) / length;

            let owner_center = cell_center[owner];
            let to_owner = owner_center - center;

            let normal = if is_boundary {
                if outward_normal.dot(to_owner) > 0.0 {
                    -outward_normal
                } else {
                    outward_normal
                }
            } else {
                if outward_normal.dot(to_owner) > 0.0 {
                    -outward_normal
                } else {
                    outward_normal
                }
            };

            let z_face = (z1 + z2) * 0.5;

            (center, normal, length, z_face, z_face)
        };

        for (edge, owner, neighbor) in &interior_faces_data {
            let (center, normal, length, zl, zr) = compute_face_geo(*edge, *owner, false);

            let face_idx = face_center.len();

            face_center.push(center);
            face_normal.push(normal);
            face_length.push(length);
            face_z_left.push(zl);
            face_z_right.push(zr);
            face_owner.push(*owner);
            face_neighbor.push(*neighbor);

            let delta_o = center - cell_center[*owner];
            let delta_n = center - cell_center[*neighbor];
            face_delta_owner.push(delta_o);
            face_delta_neighbor.push(delta_n);
            face_dist_o2n.push((cell_center[*neighbor] - cell_center[*owner]).length());

            cell_faces[*owner].push(FaceId(face_idx), true);
            cell_faces[*neighbor].push(FaceId(face_idx), false);
        }

        let mut bc_kind = Vec::with_capacity(n_boundary);
        let mut bc_value_h = Vec::with_capacity(n_boundary);
        let mut bc_value_q = Vec::with_capacity(n_boundary);
        let mut bc_forcing_id = Vec::with_capacity(n_boundary);

        for (edge, owner, tag) in &boundary_faces_data {
            let (center, normal, length, zl, _) = compute_face_geo(*edge, *owner, true);

            let face_idx = face_center.len();

            face_center.push(center);
            face_normal.push(normal);
            face_length.push(length);
            face_z_left.push(zl);
            face_z_right.push(zl);
            face_owner.push(*owner);
            face_neighbor.push(INVALID_CELL);

            let delta_o = center - cell_center[*owner];
            face_delta_owner.push(delta_o);
            face_delta_neighbor.push(-delta_o);
            face_dist_o2n.push(delta_o.length() * 2.0);

            cell_faces[*owner].push(FaceId(face_idx), true);

            let name = physical_names
                .get(tag)
                .map(|s| s.as_str())
                .unwrap_or("wall");
            let kind = Self::map_boundary_kind(name);

            log::debug!(
                "边界边 {}: tag={}, name='{}', kind={:?}",
                face_idx,
                tag,
                name,
                kind
            );

            bc_kind.push(kind);
            bc_value_h.push(0.0);
            bc_value_q.push(0.0);
            bc_forcing_id.push(None);
        }

        let mut mesh = UnstructuredMesh {
            n_nodes,
            node_xy: nodes_xy,
            node_z: nodes_z,
            n_cells,
            cell_center,
            cell_area,
            cell_z_bed,
            cell_node_ids,
            cell_faces,
            n_faces,
            n_interior_faces: n_interior,
            face_center,
            face_normal,
            face_length,
            face_z_left,
            face_z_right,
            face_owner,
            face_neighbor,
            face_delta_owner,
            face_delta_neighbor,
            face_dist_o2n,
            bc_kind,
            bc_value_h,
            bc_value_q,
            bc_forcing_id,
            spatial_index: rstar::RTree::new(),
        };

        mesh.build_spatial_index();

        log::info!("{}", mesh.statistics());

        Ok(mesh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_mapping() {
        assert_eq!(GmshLoader::map_boundary_kind("wall"), BoundaryKind::Wall);
        assert_eq!(
            GmshLoader::map_boundary_kind("inlet_1"),
            BoundaryKind::RiverInflow
        );
        assert_eq!(
            GmshLoader::map_boundary_kind("open_sea"),
            BoundaryKind::OpenSea
        );
        assert_eq!(GmshLoader::map_boundary_kind("unknown"), BoundaryKind::Wall);
    }
}
