// marihydro\crates\mh_mesh\src/io/geojson.rs

//! GeoJSON 格式读写

use crate::error::{MeshError, MeshResult};
use crate::FrozenMesh;
use mh_geo::Point2D;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeoJsonFeatureCollection {
	#[serde(rename = "type")]
	kind: String,
	#[serde(default)]
	name: Option<String>,
	#[serde(default)]
	features: Vec<GeoJsonFeature>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeoJsonFeature {
	#[serde(rename = "type")]
	kind: String,
	#[serde(default)]
	properties: serde_json::Map<String, Value>,
	geometry: GeoJsonGeometry,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeoJsonGeometry {
	#[serde(rename = "type")]
	kind: String,
	coordinates: Value,
}

/// 从 GeoJSON 文件读取多边形外环
pub fn read_geojson_polygons(path: impl AsRef<Path>) -> MeshResult<Vec<Vec<Point2D>>> {
	let content = fs::read_to_string(&path)
		.map_err(|e| MeshError::mesh_format_error("GeoJSON", path.as_ref().display().to_string(), 0, e.to_string()))?;
	read_geojson_polygons_from_str(&content)
}

/// 从 GeoJSON 字符串读取多边形外环
pub fn read_geojson_polygons_from_str(json: &str) -> MeshResult<Vec<Vec<Point2D>>> {
	let collection: GeoJsonFeatureCollection = serde_json::from_str(json)
		.map_err(|e| MeshError::mesh_format_error("GeoJSON", "<string>", 0, e.to_string()))?;

	if collection.kind != "FeatureCollection" {
		return Err(MeshError::mesh_format_error(
			"GeoJSON",
			"<string>",
			0,
			"Root type must be FeatureCollection",
		));
	}

	let mut polygons = Vec::new();
	for feature in collection.features {
		if feature.kind != "Feature" {
			continue;
		}

		match feature.geometry.kind.as_str() {
			"Polygon" => {
				let poly = parse_polygon(&feature.geometry.coordinates)?;
				polygons.extend(poly);
			}
			"MultiPolygon" => {
				let polys = parse_multipolygon(&feature.geometry.coordinates)?;
				polygons.extend(polys);
			}
			_ => continue,
		}
	}

	Ok(polygons)
}

/// 将多边形写入 GeoJSON 文件
pub fn write_geojson_polygons(
	path: impl AsRef<Path>,
	polygons: &[Vec<Point2D>],
	name: Option<&str>,
) -> MeshResult<()> {
	let mut features = Vec::new();

	for (idx, polygon) in polygons.iter().enumerate() {
		let ring = ensure_closed_ring(polygon);
		let coords: Vec<Vec<Vec<f64>>> = vec![ring
			.iter()
			.map(|p| vec![p.x, p.y])
			.collect()];

		let geometry = GeoJsonGeometry {
			kind: "Polygon".to_string(),
			coordinates: serde_json::to_value(coords)
				.map_err(|e| MeshError::mesh_format_error("GeoJSON", "<serialize>", 0, e.to_string()))?,
		};

		let mut properties = serde_json::Map::new();
		properties.insert("id".to_string(), Value::from(idx as i64));

		features.push(GeoJsonFeature {
			kind: "Feature".to_string(),
			properties,
			geometry,
		});
	}

	let collection = GeoJsonFeatureCollection {
		kind: "FeatureCollection".to_string(),
		name: name.map(|s| s.to_string()),
		features,
	};

	let json = serde_json::to_string_pretty(&collection)
		.map_err(|e| MeshError::mesh_format_error("GeoJSON", "<serialize>", 0, e.to_string()))?;
	fs::write(&path, json)
		.map_err(|e| MeshError::mesh_format_error("GeoJSON", path.as_ref().display().to_string(), 0, e.to_string()))?;

	Ok(())
}

/// 将冻结网格的单元多边形导出为 GeoJSON
pub fn write_geojson_from_mesh(
	path: impl AsRef<Path>,
	mesh: &FrozenMesh,
	name: Option<&str>,
) -> MeshResult<()> {
	let mut polygons = Vec::with_capacity(mesh.n_cells());
	for cell in 0..mesh.n_cells() {
		let ring: Vec<Point2D> = mesh
			.cell_nodes(cell)
			.iter()
			.map(|&node| mesh.node_xy(node as usize))
			.collect();
		polygons.push(ring);
	}
	write_geojson_polygons(path, &polygons, name)
}

fn parse_polygon(value: &Value) -> MeshResult<Vec<Vec<Point2D>>> {
	let rings = value.as_array().ok_or_else(|| {
		MeshError::mesh_format_error("GeoJSON", "<parse>", 0, "Invalid Polygon coordinates")
	})?;

	if rings.is_empty() {
		return Ok(Vec::new());
	}

	let mut polygons = Vec::new();
	for ring in rings.iter().take(1) {
		polygons.push(parse_ring(ring)?);
	}
	Ok(polygons)
}

fn parse_multipolygon(value: &Value) -> MeshResult<Vec<Vec<Point2D>>> {
	let polys = value.as_array().ok_or_else(|| {
		MeshError::mesh_format_error("GeoJSON", "<parse>", 0, "Invalid MultiPolygon coordinates")
	})?;

	let mut result = Vec::new();
	for poly in polys {
		let rings = poly.as_array().ok_or_else(|| {
			MeshError::mesh_format_error("GeoJSON", "<parse>", 0, "Invalid MultiPolygon ring")
		})?;
		if let Some(first_ring) = rings.first() {
			result.push(parse_ring(first_ring)?);
		}
	}
	Ok(result)
}

fn parse_ring(value: &Value) -> MeshResult<Vec<Point2D>> {
	let arr = value.as_array().ok_or_else(|| {
		MeshError::mesh_format_error("GeoJSON", "<parse>", 0, "Invalid ring coordinates")
	})?;

	if arr.len() < 4 {
		return Err(MeshError::mesh_format_error(
			"GeoJSON",
			"<parse>",
			0,
			"Ring must have at least 4 points",
		));
	}

	let mut ring = Vec::with_capacity(arr.len());
	for p in arr {
		let coords = p.as_array().ok_or_else(|| {
			MeshError::mesh_format_error("GeoJSON", "<parse>", 0, "Invalid point coordinates")
		})?;
		if coords.len() < 2 {
			return Err(MeshError::mesh_format_error(
				"GeoJSON",
				"<parse>",
				0,
				"Point must have at least two coordinates",
			));
		}
		let x = coords[0].as_f64().ok_or_else(|| {
			MeshError::mesh_format_error("GeoJSON", "<parse>", 0, "Invalid x coordinate")
		})?;
		let y = coords[1].as_f64().ok_or_else(|| {
			MeshError::mesh_format_error("GeoJSON", "<parse>", 0, "Invalid y coordinate")
		})?;
		ring.push(Point2D::new(x, y));
	}

	Ok(ensure_closed_ring(&ring))
}

fn ensure_closed_ring(points: &[Point2D]) -> Vec<Point2D> {
	if points.is_empty() {
		return Vec::new();
	}

	let mut ring = points.to_vec();
	let first = ring.first().copied().unwrap();
	let last = ring.last().copied().unwrap();
	if (first.x - last.x).abs() > 1e-12 || (first.y - last.y).abs() > 1e-12 {
		ring.push(first);
	}
	ring
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_read_polygon_from_str() {
		let json = r#"{
		  "type": "FeatureCollection",
		  "features": [
			{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[0,0],[1,0],[1,1],[0,0]]]}}
		  ]
		}"#;

		let polys = read_geojson_polygons_from_str(json).unwrap();
		assert_eq!(polys.len(), 1);
		assert!(polys[0].len() >= 4);
	}

	#[test]
	fn test_write_polygons() {
		let json = serde_json::to_string_pretty(&GeoJsonFeatureCollection {
			kind: "FeatureCollection".to_string(),
			name: None,
			features: vec![GeoJsonFeature {
				kind: "Feature".to_string(),
				properties: serde_json::Map::new(),
				geometry: GeoJsonGeometry {
					kind: "Polygon".to_string(),
					coordinates: serde_json::to_value(vec![vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]]]).unwrap(),
				},
			}],
		}).unwrap();
		assert!(json.contains("FeatureCollection"));
	}
}
