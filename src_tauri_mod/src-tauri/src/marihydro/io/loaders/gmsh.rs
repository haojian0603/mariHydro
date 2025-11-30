// src-tauri/src/marihydro/io/loaders/gmsh.rs
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use glam::DVec2;
use crate::marihydro::core::error::{MhError, MhResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryKind { Wall, RiverInflow, Outflow, OpenSea, Symmetry }

const BC_MAPPING: &[(&str, BoundaryKind)] = &[
    ("wall", BoundaryKind::Wall), ("solid", BoundaryKind::Wall), ("land", BoundaryKind::Wall),
    ("inlet", BoundaryKind::RiverInflow), ("inflow", BoundaryKind::RiverInflow), ("river", BoundaryKind::RiverInflow),
    ("outlet", BoundaryKind::Outflow), ("outflow", BoundaryKind::Outflow),
    ("open", BoundaryKind::OpenSea), ("sea", BoundaryKind::OpenSea), ("tide", BoundaryKind::OpenSea),
    ("symmetry", BoundaryKind::Symmetry), ("sym", BoundaryKind::Symmetry),
];

#[derive(Debug, Clone)]
pub struct GmshMeshData {
    pub nodes: Vec<DVec2>,
    pub nodes_z: Vec<f64>,
    pub cells: Vec<Vec<usize>>,
    pub boundary_edges: Vec<(usize, Vec<usize>)>,
    pub physical_names: HashMap<usize, String>,
}

pub struct GmshLoader;

impl GmshLoader {
    pub fn load(path: &str) -> MhResult<GmshMeshData> {
        let file = File::open(path).map_err(|e| MhError::Io(format!("Cannot open {}: {}", path, e)))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut nodes = Vec::new();
        let mut nodes_z = Vec::new();
        let mut node_map: HashMap<usize, usize> = HashMap::new();
        let mut cells = Vec::new();
        let mut boundary_edges = Vec::new();
        let mut physical_names = HashMap::new();
        let mut version = 2;

        while let Some(Ok(line)) = lines.next() {
            match line.trim() {
                "$MeshFormat" => {
                    if let Some(Ok(fmt)) = lines.next() {
                        version = fmt.split_whitespace().next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(2.0) as i32;
                    }
                    Self::skip_to(&mut lines, "$EndMeshFormat");
                }
                "$PhysicalNames" => { physical_names = Self::parse_physical_names(&mut lines)?; }
                "$Nodes" => {
                    let (xy, z, map) = if version >= 4 { Self::parse_nodes_v4(&mut lines)? } else { Self::parse_nodes_v2(&mut lines)? };
                    nodes = xy; nodes_z = z; node_map = map;
                }
                "$Elements" => {
                    let (c, b) = if version >= 4 { Self::parse_elements_v4(&mut lines, &node_map)? } else { Self::parse_elements_v2(&mut lines, &node_map)? };
                    cells = c; boundary_edges = b;
                }
                _ => {}
            }
        }
        if nodes.is_empty() { return Err(MhError::InvalidMesh { message: "No nodes".into() }); }
        if cells.is_empty() { return Err(MhError::InvalidMesh { message: "No cells".into() }); }
        Ok(GmshMeshData { nodes, nodes_z, cells, boundary_edges, physical_names })
    }

    fn skip_to(lines: &mut impl Iterator<Item = std::io::Result<String>>, end: &str) {
        while let Some(Ok(l)) = lines.next() { if l.trim() == end { break; } }
    }

    fn parse_physical_names(lines: &mut impl Iterator<Item = std::io::Result<String>>) -> MhResult<HashMap<usize, String>> {
        let mut m = HashMap::new();
        lines.next();
        while let Some(Ok(l)) = lines.next() {
            let t = l.trim();
            if t == "$EndPhysicalNames" { break; }
            let p: Vec<&str> = t.split_whitespace().collect();
            if p.len() >= 3 { if let Ok(tag) = p[1].parse::<usize>() { m.insert(tag, p[2..].join(" ").trim_matches('"').to_lowercase()); } }
        }
        Ok(m)
    }

    fn parse_nodes_v2(lines: &mut impl Iterator<Item = std::io::Result<String>>) -> MhResult<(Vec<DVec2>, Vec<f64>, HashMap<usize, usize>)> {
        let mut xy = Vec::new(); let mut z = Vec::new(); let mut m = HashMap::new();
        if let Some(Ok(c)) = lines.next() { if let Ok(n) = c.trim().parse::<usize>() { xy.reserve(n); z.reserve(n); m.reserve(n); } }
        while let Some(Ok(l)) = lines.next() {
            let t = l.trim(); if t == "$EndNodes" { break; }
            let p: Vec<&str> = t.split_whitespace().collect();
            if p.len() >= 4 {
                if let (Ok(tag), Ok(x), Ok(y), Ok(zv)) = (p[0].parse(), p[1].parse(), p[2].parse(), p[3].parse()) {
                    m.insert(tag, xy.len()); xy.push(DVec2::new(x, y)); z.push(zv);
                }
            }
        }
        Ok((xy, z, m))
    }

    fn parse_nodes_v4(lines: &mut impl Iterator<Item = std::io::Result<String>>) -> MhResult<(Vec<DVec2>, Vec<f64>, HashMap<usize, usize>)> {
        let mut xy = Vec::new(); let mut z = Vec::new(); let mut m = HashMap::new();
        let header = lines.next().ok_or_else(|| MhError::InvalidMesh { message: "Missing node header".into() })??;
        let parts: Vec<usize> = header.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if parts.len() < 4 { return Err(MhError::InvalidMesh { message: "Bad node header".into() }); }
        let (num_blocks, total) = (parts[0], parts[1]);
        xy.reserve(total); z.reserve(total); m.reserve(total);
        for _ in 0..num_blocks {
            let bh = lines.next().ok_or_else(|| MhError::InvalidMesh { message: "Missing block header".into() })??;
            let bh: Vec<usize> = bh.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if bh.len() < 4 { continue; }
            let n = bh[3];
            let mut tags = Vec::with_capacity(n);
            for _ in 0..n { if let Some(Ok(tl)) = lines.next() { if let Ok(t) = tl.trim().parse::<usize>() { tags.push(t); } } }
            for tag in tags {
                if let Some(Ok(cl)) = lines.next() {
                    let c: Vec<f64> = cl.split_whitespace().filter_map(|s| s.parse().ok()).collect();
                    if c.len() >= 3 { m.insert(tag, xy.len()); xy.push(DVec2::new(c[0], c[1])); z.push(c[2]); }
                }
            }
        }
        Self::skip_to(lines, "$EndNodes");
        Ok((xy, z, m))
    }

    fn parse_elements_v2(lines: &mut impl Iterator<Item = std::io::Result<String>>, nm: &HashMap<usize, usize>) -> MhResult<(Vec<Vec<usize>>, Vec<(usize, Vec<usize>)>)> {
        let mut cells = Vec::new(); let mut edges = Vec::new();
        lines.next();
        while let Some(Ok(l)) = lines.next() {
            let t = l.trim(); if t == "$EndElements" { break; }
            let p: Vec<&str> = t.split_whitespace().collect();
            if p.len() < 4 { continue; }
            let et = p[1].parse::<usize>().unwrap_or(0);
            let ntags = p[2].parse::<usize>().unwrap_or(0);
            let tag = if ntags > 0 { p[3].parse().unwrap_or(0) } else { 0 };
            let start = 3 + ntags;
            match et {
                1 => { if p.len() >= start + 2 { let ns: Option<Vec<usize>> = p[start..].iter().take(2).map(|s| s.parse().ok().and_then(|t| nm.get(&t).copied())).collect(); if let Some(ns) = ns { edges.push((tag, ns)); } } }
                2 => { if p.len() >= start + 3 { let ns: Option<Vec<usize>> = p[start..].iter().take(3).map(|s| s.parse().ok().and_then(|t| nm.get(&t).copied())).collect(); if let Some(ns) = ns { cells.push(ns); } } }
                3 => { if p.len() >= start + 4 { let ns: Option<Vec<usize>> = p[start..].iter().take(4).map(|s| s.parse().ok().and_then(|t| nm.get(&t).copied())).collect(); if let Some(ns) = ns { cells.push(ns); } } }
                _ => {}
            }
        }
        Ok((cells, edges))
    }

    fn parse_elements_v4(lines: &mut impl Iterator<Item = std::io::Result<String>>, nm: &HashMap<usize, usize>) -> MhResult<(Vec<Vec<usize>>, Vec<(usize, Vec<usize>)>)> {
        let mut cells = Vec::new(); let mut edges = Vec::new();
        let header = lines.next().ok_or_else(|| MhError::InvalidMesh { message: "Missing elem header".into() })??;
        let parts: Vec<usize> = header.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if parts.len() < 4 { return Err(MhError::InvalidMesh { message: "Bad elem header".into() }); }
        let num_blocks = parts[0];
        for _ in 0..num_blocks {
            let bh = lines.next().ok_or_else(|| MhError::InvalidMesh { message: "Missing block".into() })??;
            let bh: Vec<usize> = bh.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if bh.len() < 4 { continue; }
            let (etag, et, ne) = (bh[1], bh[2], bh[3]);
            for _ in 0..ne {
                if let Some(Ok(el)) = lines.next() {
                    let p: Vec<usize> = el.split_whitespace().filter_map(|s| s.parse().ok()).collect();
                    if p.is_empty() { continue; }
                    let nt = &p[1..];
                    match et {
                        1 => { if nt.len() >= 2 { let ns: Option<Vec<usize>> = nt.iter().take(2).map(|&t| nm.get(&t).copied()).collect(); if let Some(ns) = ns { edges.push((etag, ns)); } } }
                        2 => { if nt.len() >= 3 { let ns: Option<Vec<usize>> = nt.iter().take(3).map(|&t| nm.get(&t).copied()).collect(); if let Some(ns) = ns { cells.push(ns); } } }
                        3 => { if nt.len() >= 4 { let ns: Option<Vec<usize>> = nt.iter().take(4).map(|&t| nm.get(&t).copied()).collect(); if let Some(ns) = ns { cells.push(ns); } } }
                        _ => {}
                    }
                }
            }
        }
        Self::skip_to(lines, "$EndElements");
        Ok((cells, edges))
    }

    pub fn boundary_kind_from_name(name: &str) -> BoundaryKind {
        let lower = name.to_lowercase();
        for (pat, kind) in BC_MAPPING { if lower.contains(pat) { return *kind; } }
        BoundaryKind::Wall
    }
}
