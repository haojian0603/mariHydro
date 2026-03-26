// crates/mh_io/src/import/mod.rs
// IO_SOURCE: Module-level import surface for GeoJSON, MIKE, and time-series readers; each submodule must document its actual upstream format contracts.
// IO_SCOPE: The public import entrypoint only re-exports readers whose failure semantics stay explicit. Unsupported or structurally invalid inputs must fail instead of synthesizing geometry, metadata, default payloads, or semantic names for boundary and zone features. Multi-part semantic features must preserve their declared name and expose explicit part indices instead of fabricating suffixed names, and GeoJSON Feature.id values must be preserved explicitly instead of being dropped or collapsed into empty strings.

//! 数据导入模块

pub mod geojson;
pub mod mike;
pub mod timeseries_csv;

pub use timeseries_csv::{
    load_multi_column_timeseries, load_timeseries, parse_csv_string, CsvConfig,
};

pub use geojson::{
    BcLocation, BoundaryConditionLocation, Feature, GeoJsonError, GeoJsonReader, GeometryData,
    PropertyValue, ZoneProperties,
};
