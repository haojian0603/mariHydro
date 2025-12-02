// marihydro\crates\mh_mesh\src/io/gmsh.rs

//! GMSH 格式读写
//!
//! 支持 GMSH 2.x 和 4.x 格式。
//!
//! # 示例
//!
//! ```ignore
//! use mh_mesh::io::gmsh::GmshLoader;
//!
//! let mesh_data = GmshLoader::load("mesh.msh")?;
//! println!("Loaded {} nodes and {} cells", mesh_data.nodes.len(), mesh_data.cells.len());
//! ```

use mh_foundation::error::{MhError, MhResult};
use mh_geo::Point2D;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// 边界类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundaryKind {
    /// 固壁边界
    #[default]
    Wall,
    /// 河流入流
    RiverInflow,
    /// 出流边界
    Outflow,
    /// 开放海边界
    OpenSea,
    /// 对称边界
    Symmetry,
}

impl BoundaryKind {
    /// 从名称推断边界类型
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();

        const PATTERNS: &[(&[&str], BoundaryKind)] = &[
            (&["wall", "solid", "land"], BoundaryKind::Wall),
            (&["inlet", "inflow", "river"], BoundaryKind::RiverInflow),
            (&["outlet", "outflow"], BoundaryKind::Outflow),
            (&["open", "sea", "tide"], BoundaryKind::OpenSea),
            (&["symmetry", "sym"], BoundaryKind::Symmetry),
        ];

        for (patterns, kind) in PATTERNS {
            for pat in *patterns {
                if lower.contains(pat) {
                    return *kind;
                }
            }
        }

        BoundaryKind::Wall
    }

    /// 获取名称
    pub fn name(&self) -> &'static str {
        match self {
            Self::Wall => "wall",
            Self::RiverInflow => "river_inflow",
            Self::Outflow => "outflow",
            Self::OpenSea => "open_sea",
            Self::Symmetry => "symmetry",
        }
    }
}

/// GMSH 加载的网格数据
#[derive(Debug, Clone)]
pub struct GmshMeshData {
    /// 节点坐标
    pub nodes: Vec<Point2D>,
    /// 节点高程
    pub nodes_z: Vec<f64>,
    /// 单元节点索引列表
    pub cells: Vec<Vec<usize>>,
    /// 边界边 (物理标签, 节点索引)
    pub boundary_edges: Vec<(usize, Vec<usize>)>,
    /// 物理名称映射
    pub physical_names: HashMap<usize, String>,
}

impl GmshMeshData {
    /// 创建空数据
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            nodes_z: Vec::new(),
            cells: Vec::new(),
            boundary_edges: Vec::new(),
            physical_names: HashMap::new(),
        }
    }

    /// 节点数量
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// 单元数量
    pub fn n_cells(&self) -> usize {
        self.cells.len()
    }

    /// 边界边数量
    pub fn n_boundary_edges(&self) -> usize {
        self.boundary_edges.len()
    }
}

/// GMSH 文件加载器
pub struct GmshLoader;

impl GmshLoader {
    /// 加载 GMSH 文件
    pub fn load<P: AsRef<Path>>(path: P) -> MhResult<GmshMeshData> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| {
            MhError::io(format!("Cannot open {}: {}", path.display(), e))
        })?;
        let reader = BufReader::new(file);
        Self::load_from_reader(reader)
    }

    /// 从 reader 加载
    pub fn load_from_reader<R: BufRead>(reader: R) -> MhResult<GmshMeshData> {
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
                        version = fmt
                            .split_whitespace()
                            .next()
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(2.0) as i32;
                    }
                    Self::skip_to(&mut lines, "$EndMeshFormat");
                }
                "$PhysicalNames" => {
                    physical_names = Self::parse_physical_names(&mut lines)?;
                }
                "$Nodes" => {
                    let (xy, z, map) = if version >= 4 {
                        Self::parse_nodes_v4(&mut lines)?
                    } else {
                        Self::parse_nodes_v2(&mut lines)?
                    };
                    nodes = xy;
                    nodes_z = z;
                    node_map = map;
                }
                "$Elements" => {
                    let (c, b) = if version >= 4 {
                        Self::parse_elements_v4(&mut lines, &node_map)?
                    } else {
                        Self::parse_elements_v2(&mut lines, &node_map)?
                    };
                    cells = c;
                    boundary_edges = b;
                }
                _ => {}
            }
        }

        if nodes.is_empty() {
            return Err(MhError::InvalidMesh {
                message: "No nodes in GMSH file".into(),
            });
        }
        if cells.is_empty() {
            return Err(MhError::InvalidMesh {
                message: "No cells in GMSH file".into(),
            });
        }

        Ok(GmshMeshData {
            nodes,
            nodes_z,
            cells,
            boundary_edges,
            physical_names,
        })
    }

    /// 跳过到指定结束标记
    fn skip_to<I: Iterator<Item = std::io::Result<String>>>(lines: &mut I, end: &str) {
        while let Some(Ok(l)) = lines.next() {
            if l.trim() == end {
                break;
            }
        }
    }

    /// 解析物理名称
    fn parse_physical_names<I: Iterator<Item = std::io::Result<String>>>(
        lines: &mut I,
    ) -> MhResult<HashMap<usize, String>> {
        let mut m = HashMap::new();
        lines.next(); // 跳过数量行

        while let Some(Ok(l)) = lines.next() {
            let t = l.trim();
            if t == "$EndPhysicalNames" {
                break;
            }

            let parts: Vec<&str> = t.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(tag) = parts[1].parse::<usize>() {
                    let name = parts[2..]
                        .join(" ")
                        .trim_matches('"')
                        .to_lowercase();
                    m.insert(tag, name);
                }
            }
        }
        Ok(m)
    }

    /// 解析节点 (v2 格式)
    fn parse_nodes_v2<I: Iterator<Item = std::io::Result<String>>>(
        lines: &mut I,
    ) -> MhResult<(Vec<Point2D>, Vec<f64>, HashMap<usize, usize>)> {
        let mut xy = Vec::new();
        let mut z = Vec::new();
        let mut m = HashMap::new();

        // 读取节点数量
        if let Some(Ok(c)) = lines.next() {
            if let Ok(n) = c.trim().parse::<usize>() {
                xy.reserve(n);
                z.reserve(n);
                m.reserve(n);
            }
        }

        while let Some(Ok(l)) = lines.next() {
            let t = l.trim();
            if t == "$EndNodes" {
                break;
            }

            let parts: Vec<&str> = t.split_whitespace().collect();
            if parts.len() >= 4 {
                if let (Ok(tag), Ok(x), Ok(y), Ok(zv)) = (
                    parts[0].parse(),
                    parts[1].parse(),
                    parts[2].parse(),
                    parts[3].parse(),
                ) {
                    m.insert(tag, xy.len());
                    xy.push(Point2D::new(x, y));
                    z.push(zv);
                }
            }
        }
        Ok((xy, z, m))
    }

    /// 解析节点 (v4 格式)
    fn parse_nodes_v4<I: Iterator<Item = std::io::Result<String>>>(
        lines: &mut I,
    ) -> MhResult<(Vec<Point2D>, Vec<f64>, HashMap<usize, usize>)> {
        let mut xy = Vec::new();
        let mut z = Vec::new();
        let mut m = HashMap::new();

        let header = lines
            .next()
            .ok_or_else(|| MhError::InvalidMesh {
                message: "Missing node header".into(),
            })??;

        let parts: Vec<usize> = header
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        if parts.len() < 4 {
            return Err(MhError::InvalidMesh {
                message: "Bad node header".into(),
            });
        }

        let (num_blocks, total) = (parts[0], parts[1]);
        xy.reserve(total);
        z.reserve(total);
        m.reserve(total);

        for _ in 0..num_blocks {
            let bh = lines
                .next()
                .ok_or_else(|| MhError::InvalidMesh {
                    message: "Missing block header".into(),
                })??;

            let bh: Vec<usize> = bh
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            if bh.len() < 4 {
                continue;
            }
            let n = bh[3];

            // 读取标签
            let mut tags = Vec::with_capacity(n);
            for _ in 0..n {
                if let Some(Ok(tl)) = lines.next() {
                    if let Ok(t) = tl.trim().parse::<usize>() {
                        tags.push(t);
                    }
                }
            }

            // 读取坐标
            for tag in tags {
                if let Some(Ok(cl)) = lines.next() {
                    let c: Vec<f64> = cl
                        .split_whitespace()
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    if c.len() >= 3 {
                        m.insert(tag, xy.len());
                        xy.push(Point2D::new(c[0], c[1]));
                        z.push(c[2]);
                    }
                }
            }
        }

        Self::skip_to(lines, "$EndNodes");
        Ok((xy, z, m))
    }

    /// 解析单元 (v2 格式)
    fn parse_elements_v2<I: Iterator<Item = std::io::Result<String>>>(
        lines: &mut I,
        nm: &HashMap<usize, usize>,
    ) -> MhResult<(Vec<Vec<usize>>, Vec<(usize, Vec<usize>)>)> {
        let mut cells = Vec::new();
        let mut edges = Vec::new();

        lines.next(); // 跳过数量行

        while let Some(Ok(l)) = lines.next() {
            let t = l.trim();
            if t == "$EndElements" {
                break;
            }

            let parts: Vec<&str> = t.split_whitespace().collect();
            if parts.len() < 4 {
                continue;
            }

            let elem_type = parts[1].parse::<usize>().unwrap_or(0);
            let n_tags = parts[2].parse::<usize>().unwrap_or(0);
            let tag = if n_tags > 0 {
                parts[3].parse().unwrap_or(0)
            } else {
                0
            };
            let start = 3 + n_tags;

            match elem_type {
                1 => {
                    // 2-node line
                    if parts.len() >= start + 2 {
                        let ns: Option<Vec<usize>> = parts[start..]
                            .iter()
                            .take(2)
                            .map(|s| s.parse().ok().and_then(|t| nm.get(&t).copied()))
                            .collect();
                        if let Some(ns) = ns {
                            edges.push((tag, ns));
                        }
                    }
                }
                2 => {
                    // 3-node triangle
                    if parts.len() >= start + 3 {
                        let ns: Option<Vec<usize>> = parts[start..]
                            .iter()
                            .take(3)
                            .map(|s| s.parse().ok().and_then(|t| nm.get(&t).copied()))
                            .collect();
                        if let Some(ns) = ns {
                            cells.push(ns);
                        }
                    }
                }
                3 => {
                    // 4-node quadrilateral
                    if parts.len() >= start + 4 {
                        let ns: Option<Vec<usize>> = parts[start..]
                            .iter()
                            .take(4)
                            .map(|s| s.parse().ok().and_then(|t| nm.get(&t).copied()))
                            .collect();
                        if let Some(ns) = ns {
                            cells.push(ns);
                        }
                    }
                }
                _ => {}
            }
        }
        Ok((cells, edges))
    }

    /// 解析单元 (v4 格式)
    fn parse_elements_v4<I: Iterator<Item = std::io::Result<String>>>(
        lines: &mut I,
        nm: &HashMap<usize, usize>,
    ) -> MhResult<(Vec<Vec<usize>>, Vec<(usize, Vec<usize>)>)> {
        let mut cells = Vec::new();
        let mut edges = Vec::new();

        let header = lines
            .next()
            .ok_or_else(|| MhError::InvalidMesh {
                message: "Missing element header".into(),
            })??;

        let parts: Vec<usize> = header
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        if parts.len() < 4 {
            return Err(MhError::InvalidMesh {
                message: "Bad element header".into(),
            });
        }

        let num_blocks = parts[0];

        for _ in 0..num_blocks {
            let bh = lines
                .next()
                .ok_or_else(|| MhError::InvalidMesh {
                    message: "Missing block".into(),
                })??;

            let bh: Vec<usize> = bh
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            if bh.len() < 4 {
                continue;
            }

            let (etag, elem_type, n_elems) = (bh[1], bh[2], bh[3]);

            for _ in 0..n_elems {
                if let Some(Ok(el)) = lines.next() {
                    let p: Vec<usize> = el
                        .split_whitespace()
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    if p.is_empty() {
                        continue;
                    }

                    let node_tags = &p[1..];

                    match elem_type {
                        1 => {
                            // 2-node line
                            if node_tags.len() >= 2 {
                                let ns: Option<Vec<usize>> = node_tags
                                    .iter()
                                    .take(2)
                                    .map(|&t| nm.get(&t).copied())
                                    .collect();
                                if let Some(ns) = ns {
                                    edges.push((etag, ns));
                                }
                            }
                        }
                        2 => {
                            // 3-node triangle
                            if node_tags.len() >= 3 {
                                let ns: Option<Vec<usize>> = node_tags
                                    .iter()
                                    .take(3)
                                    .map(|&t| nm.get(&t).copied())
                                    .collect();
                                if let Some(ns) = ns {
                                    cells.push(ns);
                                }
                            }
                        }
                        3 => {
                            // 4-node quadrilateral
                            if node_tags.len() >= 4 {
                                let ns: Option<Vec<usize>> = node_tags
                                    .iter()
                                    .take(4)
                                    .map(|&t| nm.get(&t).copied())
                                    .collect();
                                if let Some(ns) = ns {
                                    cells.push(ns);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        Self::skip_to(lines, "$EndElements");
        Ok((cells, edges))
    }
}

/// GMSH 文件写入器
pub struct GmshWriter;

impl GmshWriter {
    /// 将网格数据写入 GMSH 文件
    pub fn write<P: AsRef<Path>>(path: P, data: &GmshMeshData) -> MhResult<()> {
        let file = File::create(path.as_ref()).map_err(|e| {
            MhError::io(format!("Cannot create file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);
        Self::write_to(&mut writer, data)
    }

    /// 写入到 writer
    pub fn write_to<W: Write>(writer: &mut W, data: &GmshMeshData) -> MhResult<()> {
        // 写入文件头
        writeln!(writer, "$MeshFormat").map_err(|e| MhError::io(e.to_string()))?;
        writeln!(writer, "2.2 0 8").map_err(|e| MhError::io(e.to_string()))?;
        writeln!(writer, "$EndMeshFormat").map_err(|e| MhError::io(e.to_string()))?;

        // 写入节点
        writeln!(writer, "$Nodes").map_err(|e| MhError::io(e.to_string()))?;
        writeln!(writer, "{}", data.nodes.len()).map_err(|e| MhError::io(e.to_string()))?;
        for (i, (node, z)) in data.nodes.iter().zip(data.nodes_z.iter()).enumerate() {
            writeln!(writer, "{} {} {} {}", i + 1, node.x, node.y, z)
                .map_err(|e| MhError::io(e.to_string()))?;
        }
        writeln!(writer, "$EndNodes").map_err(|e| MhError::io(e.to_string()))?;

        // 写入单元
        let total_elems = data.cells.len() + data.boundary_edges.len();
        writeln!(writer, "$Elements").map_err(|e| MhError::io(e.to_string()))?;
        writeln!(writer, "{}", total_elems).map_err(|e| MhError::io(e.to_string()))?;

        let mut elem_id = 1;

        // 写入边界边
        for (tag, nodes) in &data.boundary_edges {
            write!(writer, "{} 1 2 {} 0", elem_id, tag)
                .map_err(|e| MhError::io(e.to_string()))?;
            for n in nodes {
                write!(writer, " {}", n + 1).map_err(|e| MhError::io(e.to_string()))?;
            }
            writeln!(writer).map_err(|e| MhError::io(e.to_string()))?;
            elem_id += 1;
        }

        // 写入单元
        for nodes in &data.cells {
            let elem_type = if nodes.len() == 3 { 2 } else { 3 };
            write!(writer, "{} {} 2 0 0", elem_id, elem_type)
                .map_err(|e| MhError::io(e.to_string()))?;
            for n in nodes {
                write!(writer, " {}", n + 1).map_err(|e| MhError::io(e.to_string()))?;
            }
            writeln!(writer).map_err(|e| MhError::io(e.to_string()))?;
            elem_id += 1;
        }

        writeln!(writer, "$EndElements").map_err(|e| MhError::io(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SIMPLE_MSH_V2: &str = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.5 1.0 0.0
$EndNodes
$Elements
1
1 2 2 0 0 1 2 3
$EndElements
"#;

    #[test]
    fn test_load_v2() {
        let cursor = Cursor::new(SIMPLE_MSH_V2);
        let data = GmshLoader::load_from_reader(cursor).unwrap();

        assert_eq!(data.n_nodes(), 3);
        assert_eq!(data.n_cells(), 1);
        assert_eq!(data.cells[0].len(), 3);
    }

    #[test]
    fn test_boundary_kind() {
        assert_eq!(BoundaryKind::from_name("wall_left"), BoundaryKind::Wall);
        assert_eq!(BoundaryKind::from_name("river_inlet"), BoundaryKind::RiverInflow);
        assert_eq!(BoundaryKind::from_name("open_sea"), BoundaryKind::OpenSea);
        assert_eq!(BoundaryKind::from_name("outlet_right"), BoundaryKind::Outflow);
    }

    #[test]
    fn test_roundtrip() {
        let original = GmshMeshData {
            nodes: vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(1.0, 0.0),
                Point2D::new(0.5, 1.0),
            ],
            nodes_z: vec![0.0, 0.0, 0.0],
            cells: vec![vec![0, 1, 2]],
            boundary_edges: vec![],
            physical_names: HashMap::new(),
        };

        let mut buffer = Vec::new();
        GmshWriter::write_to(&mut buffer, &original).unwrap();

        let cursor = Cursor::new(buffer);
        let loaded = GmshLoader::load_from_reader(cursor).unwrap();

        assert_eq!(loaded.n_nodes(), original.n_nodes());
        assert_eq!(loaded.n_cells(), original.n_cells());
    }
}
