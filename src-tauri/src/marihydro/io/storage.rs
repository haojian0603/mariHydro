// src-tauri/src/marihydro/io/storage.rs

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::time::TimeManager;

#[derive(Serialize, Deserialize, Debug)]
pub struct Snapshot {
    pub version: String,
    pub time_state: TimeManager,
    pub n_cells: usize,
    pub h: Vec<f64>,
    pub hu: Vec<f64>,
    pub hv: Vec<f64>,
    pub hc: Option<Vec<f64>>,
}

impl Snapshot {
    pub fn new(
        time_state: TimeManager,
        n_cells: usize,
        h: Vec<f64>,
        hu: Vec<f64>,
        hv: Vec<f64>,
        hc: Option<Vec<f64>>,
    ) -> Self {
        Self {
            version: "2.0.0".to_string(),
            time_state,
            n_cells,
            h,
            hu,
            hv,
            hc,
        }
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> MhResult<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| MhError::Serialization(format!("快照保存失败: {}", e)))?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> MhResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let snapshot: Self = bincode::deserialize_from(reader)
            .map_err(|e| MhError::Serialization(format!("快照读取失败: {}", e)))?;
        Ok(snapshot)
    }

    pub fn validate(&self) -> MhResult<()> {
        if self.h.len() != self.n_cells {
            return Err(MhError::Serialization(format!(
                "h 长度 {} 与 n_cells {} 不匹配",
                self.h.len(),
                self.n_cells
            )));
        }
        if self.hu.len() != self.n_cells {
            return Err(MhError::Serialization(format!(
                "hu 长度 {} 与 n_cells {} 不匹配",
                self.hu.len(),
                self.n_cells
            )));
        }
        if self.hv.len() != self.n_cells {
            return Err(MhError::Serialization(format!(
                "hv 长度 {} 与 n_cells {} 不匹配",
                self.hv.len(),
                self.n_cells
            )));
        }
        if let Some(ref hc) = self.hc {
            if hc.len() != self.n_cells {
                return Err(MhError::Serialization(format!(
                    "hc 长度 {} 与 n_cells {} 不匹配",
                    hc.len(),
                    self.n_cells
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_snapshot_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.mhs");

        let time_state = TimeManager::default();
        let snapshot = Snapshot::new(
            time_state,
            3,
            vec![1.0, 2.0, 3.0],
            vec![0.1, 0.2, 0.3],
            vec![0.0, 0.0, 0.0],
            None,
        );

        snapshot.save_to_file(&path).unwrap();
        let loaded = Snapshot::load_from_file(&path).unwrap();

        assert_eq!(loaded.n_cells, 3);
        assert_eq!(loaded.h, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_validate() {
        let time_state = TimeManager::default();
        let bad_snapshot = Snapshot {
            version: "2.0.0".to_string(),
            time_state,
            n_cells: 5,
            h: vec![1.0, 2.0],
            hu: vec![0.0; 5],
            hv: vec![0.0; 5],
            hc: None,
        };
        assert!(bad_snapshot.validate().is_err());
    }
}
