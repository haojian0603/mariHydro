//! GMSH 格式读写
//!
//! 支持 GMSH 2.x 和 4.x 格式。
//!
//! GMSH_SOURCE: Gmsh MSH 2.2 / 4.1 ASCII section layout (`$MeshFormat`, `$PhysicalNames`, `$Nodes`, `$Elements`)。
//! GMSH_SCOPE: 主链只接受结构完整、数值 token 可完整解释、且支持单元节点映射可显式建立的 GMSH 输入；
//!             版本行、块头、标签数、物理组编号或节点引用任一处损坏时立即报错，不把坏字段折成 `0` 或静默跳过。
//!
//! # 示例
//!
//! ```ignore
//! use mh_mesh::io::gmsh::GmshLoader;
//!
//! let mesh_data = GmshLoader::load("mesh.msh")?;
//! println!("Loaded {} nodes and {} cells", mesh_data.nodes.len(), mesh_data.cells.len());
//! ```

use crate::error::MeshError;
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
    fn gmsh_format_error(filename: &str, message: impl Into<String>) -> MhError {
        MeshError::mesh_format_error("gmsh", filename.to_string(), 0, message.into()).into()
    }

    fn next_required_line<I: Iterator<Item = std::io::Result<String>>>(
        lines: &mut I,
        filename: &str,
        context: &str,
    ) -> MhResult<String> {
        match lines.next() {
            Some(Ok(line)) => Ok(line),
            Some(Err(err)) => Err(Self::gmsh_format_error(
                filename,
                format!("{context} read failed: {err}"),
            )),
            None => Err(Self::gmsh_format_error(
                filename,
                format!("missing {context}"),
            )),
        }
    }

    fn parse_usize_token(filename: &str, context: &str, token: &str) -> MhResult<usize> {
        token.parse::<usize>().map_err(|_| {
            Self::gmsh_format_error(filename, format!("{context} is not a valid usize: {token}"))
        })
    }

    fn parse_f64_token(filename: &str, context: &str, token: &str) -> MhResult<f64> {
        token.parse::<f64>().map_err(|_| {
            Self::gmsh_format_error(filename, format!("{context} is not a valid f64: {token}"))
        })
    }

    fn parse_node_index(
        filename: &str,
        node_map: &HashMap<usize, usize>,
        token: &str,
        context: &str,
    ) -> MhResult<usize> {
        let tag = Self::parse_usize_token(filename, context, token)?;
        node_map.get(&tag).copied().ok_or_else(|| {
            Self::gmsh_format_error(
                filename,
                format!("{context} references unknown node tag {tag}"),
            )
        })
    }

    fn expect_section_end<I: Iterator<Item = std::io::Result<String>>>(
        lines: &mut I,
        filename: &str,
        end: &str,
        context: &str,
    ) -> MhResult<()> {
        let line = Self::next_required_line(lines, filename, context)?;
        if line.trim() == end {
            return Ok(());
        }
        Err(Self::gmsh_format_error(
            filename,
            format!("{context} must end with {end}"),
        ))
    }

    /// 加载 GMSH 文件
    pub fn load<P: AsRef<Path>>(path: P) -> MhResult<GmshMeshData> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| MhError::io(format!("Cannot open {}: {}", path.display(), e)))?;
        let reader = BufReader::new(file);
        Self::load_from_reader(reader, path.to_string_lossy().to_string())
    }

    /// 从 reader 加载
    pub fn load_from_reader<R: BufRead>(reader: R, filename: String) -> MhResult<GmshMeshData> {
        let mut lines = reader.lines();
        let mut nodes = Vec::new();
        let mut nodes_z = Vec::new();
        let mut node_map: HashMap<usize, usize> = HashMap::new();
        let mut cells = Vec::new();
        let mut boundary_edges = Vec::new();
        let mut physical_names = HashMap::new();
        let mut version = 2;

        use crate::error::MeshError as ME;

        while let Some(Ok(line)) = lines.next() {
            match line.trim() {
                "$MeshFormat" => {
                    let fmt = Self::next_required_line(&mut lines, &filename, "mesh format line")?;
                    let version_token = fmt.split_whitespace().next().ok_or_else(|| {
                        Self::gmsh_format_error(&filename, "missing mesh format version token")
                    })?;
                    let parsed_version = version_token.parse::<f64>().map_err(|_| {
                        Self::gmsh_format_error(
                            &filename,
                            format!("mesh format version is invalid: {version_token}"),
                        )
                    })?;
                    version = parsed_version as i32;
                    Self::skip_to(&mut lines, "$EndMeshFormat");
                }
                "$PhysicalNames" => {
                    physical_names = Self::parse_physical_names(&mut lines, &filename)?;
                }
                "$Nodes" => {
                    let (xy, z, map) = if version >= 4 {
                        Self::parse_nodes_v4(&mut lines, &filename)?
                    } else {
                        Self::parse_nodes_v2(&mut lines, &filename)?
                    };
                    nodes = xy;
                    nodes_z = z;
                    node_map = map;
                }
                "$Elements" => {
                    let (c, b) = if version >= 4 {
                        Self::parse_elements_v4(&mut lines, &node_map, &filename)?
                    } else {
                        Self::parse_elements_v2(&mut lines, &node_map, &filename)?
                    };
                    cells = c;
                    boundary_edges = b;
                }
                _ => {}
            }
        }

        if nodes.is_empty() {
            return Err(ME::MeshFormatError {
                format: "gmsh",
                file: filename.clone(),
                line: 0,
                message: "No nodes in GMSH file".to_string(),
            }
            .into());
        }
        if cells.is_empty() {
            return Err(ME::MeshFormatError {
                format: "gmsh",
                file: filename.clone(),
                line: 0,
                message: "No cells in GMSH file".to_string(),
            }
            .into());
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
        filename: &str,
    ) -> MhResult<HashMap<usize, String>> {
        let mut m = HashMap::new();
        let count_line = Self::next_required_line(lines, filename, "physical name count")?;
        let n = Self::parse_usize_token(filename, "physical name count", count_line.trim())?;

        for index in 0..n {
            let line =
                Self::next_required_line(lines, filename, &format!("physical name entry {index}"))?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                return Err(Self::gmsh_format_error(
                    filename,
                    format!("physical name entry {index} is incomplete"),
                ));
            }
            let tag =
                Self::parse_usize_token(filename, &format!("physical name tag {index}"), parts[1])?;
            let name = parts[2..].join(" ").trim_matches('"').to_lowercase();
            if name.is_empty() {
                return Err(Self::gmsh_format_error(
                    filename,
                    format!("physical name entry {index} has empty semantic name"),
                ));
            }
            m.insert(tag, name);
        }
        Self::expect_section_end(
            lines,
            filename,
            "$EndPhysicalNames",
            "physical name section",
        )?;
        Ok(m)
    }

    /// 解析节点 (v2 格式)
    fn parse_nodes_v2<I: Iterator<Item = std::io::Result<String>>>(
        lines: &mut I,
        filename: &str,
    ) -> MhResult<(Vec<Point2D>, Vec<f64>, HashMap<usize, usize>)> {
        let mut xy = Vec::new();
        let mut z = Vec::new();
        let mut m = HashMap::new();

        let count_line = Self::next_required_line(lines, filename, "node count")?;
        let n = Self::parse_usize_token(filename, "node count", count_line.trim())?;
        xy.reserve(n);
        z.reserve(n);
        m.reserve(n);

        for index in 0..n {
            let line = Self::next_required_line(lines, filename, &format!("node entry {index}"))?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(Self::gmsh_format_error(
                    filename,
                    format!("node entry {index} is incomplete"),
                ));
            }

            let tag = Self::parse_usize_token(filename, &format!("node tag {index}"), parts[0])?;
            let xv = Self::parse_f64_token(filename, &format!("node x {index}"), parts[1])?;
            let yv = Self::parse_f64_token(filename, &format!("node y {index}"), parts[2])?;
            let zv = Self::parse_f64_token(filename, &format!("node z {index}"), parts[3])?;

            m.insert(tag, xy.len());
            xy.push(Point2D::new(xv, yv));
            z.push(zv);
        }
        Self::expect_section_end(lines, filename, "$EndNodes", "node section")?;
        Ok((xy, z, m))
    }

    /// 解析节点 (v4 格式)
    fn parse_nodes_v4<I: Iterator<Item = std::io::Result<String>>>(
        lines: &mut I,
        filename: &str,
    ) -> MhResult<(Vec<Point2D>, Vec<f64>, HashMap<usize, usize>)> {
        let mut xy = Vec::new();
        let mut z = Vec::new();
        let mut m = HashMap::new();

        let header = match lines.next() {
            Some(Ok(h)) => h,
            _ => return Err(Self::gmsh_format_error(filename, "missing node header")),
        };

        let header_parts: Vec<&str> = header.split_whitespace().collect();
        if header_parts.len() < 4 {
            return Err(Self::gmsh_format_error(filename, "bad node header"));
        }

        let num_blocks =
            Self::parse_usize_token(filename, "node header block count", header_parts[0])?;
        let total = Self::parse_usize_token(filename, "node header total count", header_parts[1])?;
        xy.reserve(total);
        z.reserve(total);
        m.reserve(total);

        for block_index in 0..num_blocks {
            let block_header = Self::next_required_line(
                lines,
                filename,
                &format!("node block header {block_index}"),
            )?;
            let block_parts: Vec<&str> = block_header.split_whitespace().collect();
            if block_parts.len() < 4 {
                return Err(Self::gmsh_format_error(
                    filename,
                    format!("node block header {block_index} is incomplete"),
                ));
            }
            let _entity_dim = Self::parse_usize_token(
                filename,
                &format!("node block entity dimension {block_index}"),
                block_parts[0],
            )?;
            let _entity_tag = Self::parse_usize_token(
                filename,
                &format!("node block entity tag {block_index}"),
                block_parts[1],
            )?;
            let parametric = Self::parse_usize_token(
                filename,
                &format!("node block parametric flag {block_index}"),
                block_parts[2],
            )?;
            if parametric > 1 {
                return Err(Self::gmsh_format_error(
                    filename,
                    format!("node block parametric flag {block_index} must be 0 or 1"),
                ));
            }
            let n = Self::parse_usize_token(
                filename,
                &format!("node block size {block_index}"),
                block_parts[3],
            )?;

            let mut tags = Vec::with_capacity(n);
            for tag_index in 0..n {
                let line = Self::next_required_line(
                    lines,
                    filename,
                    &format!("node block {block_index} tag {tag_index}"),
                )?;
                tags.push(Self::parse_usize_token(
                    filename,
                    &format!("node block {block_index} tag {tag_index}"),
                    line.trim(),
                )?);
            }

            for (coord_index, tag) in tags.into_iter().enumerate() {
                let line = Self::next_required_line(
                    lines,
                    filename,
                    &format!("node block {block_index} coordinate {coord_index}"),
                )?;
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 3 {
                    return Err(Self::gmsh_format_error(
                        filename,
                        format!("node block {block_index} coordinate {coord_index} is incomplete"),
                    ));
                }
                let x = Self::parse_f64_token(
                    filename,
                    &format!("node block {block_index} x {coord_index}"),
                    parts[0],
                )?;
                let y = Self::parse_f64_token(
                    filename,
                    &format!("node block {block_index} y {coord_index}"),
                    parts[1],
                )?;
                let z_value = Self::parse_f64_token(
                    filename,
                    &format!("node block {block_index} z {coord_index}"),
                    parts[2],
                )?;
                m.insert(tag, xy.len());
                xy.push(Point2D::new(x, y));
                z.push(z_value);
            }
        }

        Self::expect_section_end(lines, filename, "$EndNodes", "node section")?;
        Ok((xy, z, m))
    }

    /// 解析单元 (v2 格式)
    fn parse_elements_v2<I: Iterator<Item = std::io::Result<String>>>(
        lines: &mut I,
        nm: &HashMap<usize, usize>,
        filename: &str,
    ) -> MhResult<(Vec<Vec<usize>>, Vec<(usize, Vec<usize>)>)> {
        let mut cells = Vec::new();
        let mut edges = Vec::new();

        let count_line = Self::next_required_line(lines, filename, "element count")?;
        let n_elements = Self::parse_usize_token(filename, "element count", count_line.trim())?;

        for element_index in 0..n_elements {
            let line = Self::next_required_line(
                lines,
                filename,
                &format!("element entry {element_index}"),
            )?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(Self::gmsh_format_error(
                    filename,
                    format!("element entry {element_index} is incomplete"),
                ));
            }

            let elem_type = Self::parse_usize_token(
                filename,
                &format!("element type {element_index}"),
                parts[1],
            )?;
            let n_tags = Self::parse_usize_token(
                filename,
                &format!("element tag count {element_index}"),
                parts[2],
            )?;
            if parts.len() < 3 + n_tags {
                return Err(Self::gmsh_format_error(
                    filename,
                    format!("element entry {element_index} has fewer tag fields than declared"),
                ));
            }
            let tag = if n_tags > 0 {
                Self::parse_usize_token(
                    filename,
                    &format!("element physical tag {element_index}"),
                    parts[3],
                )?
            } else {
                0
            };
            let start = 3 + n_tags;

            match elem_type {
                1 => {
                    if parts.len() == start + 2 {
                        let ns = parts[start..]
                            .iter()
                            .take(2)
                            .enumerate()
                            .map(|(offset, token)| {
                                Self::parse_node_index(
                                    filename,
                                    nm,
                                    token,
                                    &format!("edge node {element_index}:{offset}"),
                                )
                            })
                            .collect::<MhResult<Vec<_>>>()?;
                        edges.push((tag, ns));
                    } else {
                        return Err(Self::gmsh_format_error(
                            filename,
                            format!(
                                "edge element {element_index} must contain exactly 2 node tags"
                            ),
                        ));
                    }
                }
                2 => {
                    if parts.len() == start + 3 {
                        let ns = parts[start..]
                            .iter()
                            .take(3)
                            .enumerate()
                            .map(|(offset, token)| {
                                Self::parse_node_index(
                                    filename,
                                    nm,
                                    token,
                                    &format!("triangle node {element_index}:{offset}"),
                                )
                            })
                            .collect::<MhResult<Vec<_>>>()?;
                        cells.push(ns);
                    } else {
                        return Err(Self::gmsh_format_error(
                            filename,
                            format!(
                                "triangle element {element_index} must contain exactly 3 node tags"
                            ),
                        ));
                    }
                }
                3 => {
                    if parts.len() == start + 4 {
                        let ns = parts[start..]
                            .iter()
                            .take(4)
                            .enumerate()
                            .map(|(offset, token)| {
                                Self::parse_node_index(
                                    filename,
                                    nm,
                                    token,
                                    &format!("quad node {element_index}:{offset}"),
                                )
                            })
                            .collect::<MhResult<Vec<_>>>()?;
                        cells.push(ns);
                    } else {
                        return Err(Self::gmsh_format_error(
                            filename,
                            format!(
                                "quad element {element_index} must contain exactly 4 node tags"
                            ),
                        ));
                    }
                }
                _ => {}
            }
        }
        Self::expect_section_end(lines, filename, "$EndElements", "element section")?;
        Ok((cells, edges))
    }

    /// 解析单元 (v4 格式)
    fn parse_elements_v4<I: Iterator<Item = std::io::Result<String>>>(
        lines: &mut I,
        nm: &HashMap<usize, usize>,
        filename: &str,
    ) -> MhResult<(Vec<Vec<usize>>, Vec<(usize, Vec<usize>)>)> {
        let mut cells = Vec::new();
        let mut edges = Vec::new();

        let header = match lines.next() {
            Some(Ok(h)) => h,
            _ => return Err(Self::gmsh_format_error(filename, "missing element header")),
        };

        let header_parts: Vec<&str> = header.split_whitespace().collect();
        if header_parts.len() < 4 {
            return Err(Self::gmsh_format_error(filename, "bad element header"));
        }

        let num_blocks =
            Self::parse_usize_token(filename, "element header block count", header_parts[0])?;

        for block_index in 0..num_blocks {
            let block_header = Self::next_required_line(
                lines,
                filename,
                &format!("element block header {block_index}"),
            )?;
            let block_parts: Vec<&str> = block_header.split_whitespace().collect();
            if block_parts.len() < 4 {
                return Err(Self::gmsh_format_error(
                    filename,
                    format!("element block header {block_index} is incomplete"),
                ));
            }
            let _entity_dim = Self::parse_usize_token(
                filename,
                &format!("element block entity dimension {block_index}"),
                block_parts[0],
            )?;

            let etag = Self::parse_usize_token(
                filename,
                &format!("element block tag {block_index}"),
                block_parts[1],
            )?;
            let elem_type = Self::parse_usize_token(
                filename,
                &format!("element block type {block_index}"),
                block_parts[2],
            )?;
            let n_elems = Self::parse_usize_token(
                filename,
                &format!("element block count {block_index}"),
                block_parts[3],
            )?;

            for element_index in 0..n_elems {
                let line = Self::next_required_line(
                    lines,
                    filename,
                    &format!("element block {block_index} item {element_index}"),
                )?;
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.is_empty() {
                    return Err(Self::gmsh_format_error(
                        filename,
                        format!("element block {block_index} item {element_index} is empty"),
                    ));
                }

                let node_tags = &parts[1..];

                match elem_type {
                    1 => {
                        if node_tags.len() != 2 {
                            return Err(Self::gmsh_format_error(
                                filename,
                                format!(
                                    "edge block {block_index} item {element_index} must contain exactly 2 node tags"
                                ),
                            ));
                        }
                        let ns = node_tags
                            .iter()
                            .take(2)
                            .enumerate()
                            .map(|(offset, token)| {
                                Self::parse_node_index(
                                    filename,
                                    nm,
                                    token,
                                    &format!("edge block {block_index} item {element_index} node {offset}"),
                                )
                            })
                            .collect::<MhResult<Vec<_>>>()?;
                        edges.push((etag, ns));
                    }
                    2 => {
                        if node_tags.len() != 3 {
                            return Err(Self::gmsh_format_error(
                                filename,
                                format!(
                                    "triangle block {block_index} item {element_index} must contain exactly 3 node tags"
                                ),
                            ));
                        }
                        let ns = node_tags
                            .iter()
                            .take(3)
                            .enumerate()
                            .map(|(offset, token)| {
                                Self::parse_node_index(
                                    filename,
                                    nm,
                                    token,
                                    &format!("triangle block {block_index} item {element_index} node {offset}"),
                                )
                            })
                            .collect::<MhResult<Vec<_>>>()?;
                        cells.push(ns);
                    }
                    3 => {
                        if node_tags.len() != 4 {
                            return Err(Self::gmsh_format_error(
                                filename,
                                format!(
                                    "quad block {block_index} item {element_index} must contain exactly 4 node tags"
                                ),
                            ));
                        }
                        let ns = node_tags
                            .iter()
                            .take(4)
                            .enumerate()
                            .map(|(offset, token)| {
                                Self::parse_node_index(
                                    filename,
                                    nm,
                                    token,
                                    &format!("quad block {block_index} item {element_index} node {offset}"),
                                )
                            })
                            .collect::<MhResult<Vec<_>>>()?;
                        cells.push(ns);
                    }
                    _ => {}
                }
            }
        }

        Self::expect_section_end(lines, filename, "$EndElements", "element section")?;
        Ok((cells, edges))
    }
}

/// GMSH 文件写入器
pub struct GmshWriter;

impl GmshWriter {
    /// 将网格数据写入 GMSH 文件
    pub fn write<P: AsRef<Path>>(path: P, data: &GmshMeshData) -> MhResult<()> {
        let file = File::create(path.as_ref())
            .map_err(|e| MhError::io(format!("Cannot create file: {}", e)))?;
        let mut writer = BufWriter::new(file);
        Self::write_to(&mut writer, data)
    }

    /// 写入到 writer
    pub fn write_to<W: Write>(writer: &mut W, data: &GmshMeshData) -> MhResult<()> {
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

        let total_elems = data.cells.len() + data.boundary_edges.len();
        writeln!(writer, "$Elements").map_err(|e| MhError::io(e.to_string()))?;
        writeln!(writer, "{}", total_elems).map_err(|e| MhError::io(e.to_string()))?;

        let mut elem_id = 1;

        for (tag, nodes) in &data.boundary_edges {
            write!(writer, "{} 1 2 {} 0", elem_id, tag).map_err(|e| MhError::io(e.to_string()))?;
            for n in nodes {
                write!(writer, " {}", n + 1).map_err(|e| MhError::io(e.to_string()))?;
            }
            writeln!(writer).map_err(|e| MhError::io(e.to_string()))?;
            elem_id += 1;
        }

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

    const BAD_MSH_V2_ELEMENT_TAG_COUNT: &str = r#"$MeshFormat
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
1 2 bad 0 0 1 2 3
$EndElements
"#;

    const BAD_MSH_V4_NODE_BLOCK: &str = r#"$MeshFormat
4.1 0 8
$EndMeshFormat
$Nodes
1 3 1 3
2 1 bad 3
1
2
3
0.0 0.0 0.0
1.0 0.0 0.0
0.5 1.0 0.0
$EndNodes
$Elements
1 1 1 1
2 2 1 1
1 1 2 3
$EndElements
"#;

    const BAD_MSH_V4_UNKNOWN_NODE_REF: &str = r#"$MeshFormat
4.1 0 8
$EndMeshFormat
$Nodes
1 3 1 3
2 1 0 3
1
2
3
0.0 0.0 0.0
1.0 0.0 0.0
0.5 1.0 0.0
$EndNodes
$Elements
1 1 1 1
2 2 2 1
1 1 2 99
$EndElements
"#;

    const BAD_MSH_V4_EXTRA_EDGE_NODE: &str = r#"$MeshFormat
4.1 0 8
$EndMeshFormat
$Nodes
1 3 1 3
2 1 0 3
1
2
3
0.0 0.0 0.0
1.0 0.0 0.0
0.5 1.0 0.0
$EndNodes
$Elements
1 1 1 1
2 2 1 1
1 1 2 3
$EndElements
"#;

    #[test]
    fn test_load_v2() {
        let cursor = Cursor::new(SIMPLE_MSH_V2);
        let data = GmshLoader::load_from_reader(cursor, "test.msh".to_string()).unwrap();

        assert_eq!(data.n_nodes(), 3);
        assert_eq!(data.n_cells(), 1);
        assert_eq!(data.cells[0].len(), 3);
    }

    #[test]
    fn test_boundary_kind() {
        assert_eq!(BoundaryKind::from_name("wall_left"), BoundaryKind::Wall);
        assert_eq!(
            BoundaryKind::from_name("river_inlet"),
            BoundaryKind::RiverInflow
        );
        assert_eq!(BoundaryKind::from_name("open_sea"), BoundaryKind::OpenSea);
        assert_eq!(
            BoundaryKind::from_name("outlet_right"),
            BoundaryKind::Outflow
        );
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
        let loaded = GmshLoader::load_from_reader(cursor, "test.msh".to_string()).unwrap();

        assert_eq!(loaded.n_nodes(), original.n_nodes());
        assert_eq!(loaded.n_cells(), original.n_cells());
    }

    #[test]
    fn test_load_v2_rejects_invalid_tag_count_token() {
        let cursor = Cursor::new(BAD_MSH_V2_ELEMENT_TAG_COUNT);
        let err = GmshLoader::load_from_reader(cursor, "bad_v2.msh".to_string()).unwrap_err();
        assert!(err
            .to_string()
            .contains("element tag count 0 is not a valid usize"));
    }

    #[test]
    fn test_load_v4_rejects_invalid_node_block_header() {
        let cursor = Cursor::new(BAD_MSH_V4_NODE_BLOCK);
        let err = GmshLoader::load_from_reader(cursor, "bad_v4_nodes.msh".to_string()).unwrap_err();
        assert!(err
            .to_string()
            .contains("node block parametric flag 0 is not a valid usize"));
    }

    #[test]
    fn test_load_v4_rejects_unknown_node_reference() {
        let cursor = Cursor::new(BAD_MSH_V4_UNKNOWN_NODE_REF);
        let err =
            GmshLoader::load_from_reader(cursor, "bad_v4_elements.msh".to_string()).unwrap_err();
        assert!(err.to_string().contains("references unknown node tag 99"));
    }

    #[test]
    fn test_load_v4_rejects_extra_edge_node_tags() {
        let cursor = Cursor::new(BAD_MSH_V4_EXTRA_EDGE_NODE);
        let err =
            GmshLoader::load_from_reader(cursor, "bad_v4_extra_edge.msh".to_string()).unwrap_err();
        assert!(err.to_string().contains("must contain exactly 2 node tags"));
    }
}
