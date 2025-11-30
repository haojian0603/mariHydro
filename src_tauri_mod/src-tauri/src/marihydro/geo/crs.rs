// src-tauri/src/marihydro/geo/crs.rs
use crate::marihydro::core::error::{MhError, MhResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "mode", content = "value")]
pub enum CrsStrategy { Manual(String), FromFirstFile, ForceWGS84 }

impl Default for CrsStrategy { fn default() -> Self { Self::FromFirstFile } }

#[derive(Debug, Clone)]
pub struct ResolvedCrs {
    pub wkt: String,
    pub epsg: Option<u32>,
}

impl ResolvedCrs {
    pub fn new(definition: &str) -> MhResult<Self> {
        let epsg = Self::parse_epsg(definition);
        Ok(Self { wkt: definition.into(), epsg })
    }

    pub fn wgs84() -> Self { Self { wkt: "EPSG:4326".into(), epsg: Some(4326) } }

    fn parse_epsg(s: &str) -> Option<u32> {
        if s.starts_with("EPSG:") { s[5..].parse().ok() } else { None }
    }

    pub fn is_geographic(&self) -> bool {
        self.epsg == Some(4326) || self.wkt.contains("GEOGCS") || self.wkt.to_lowercase().contains("longlat")
    }

    pub fn is_metric(&self) -> bool { !self.is_geographic() }
}

#[derive(Debug, Clone)]
pub struct Crs {
    pub definition: String,
}

impl Crs {
    pub fn new(def: &str) -> MhResult<Self> { Ok(Self { definition: def.into() }) }
    pub fn wgs84() -> Self { Self { definition: "EPSG:4326".into() } }
    pub fn is_geographic(&self) -> bool {
        self.definition.contains("4326") || self.definition.to_lowercase().contains("longlat")
    }
}
