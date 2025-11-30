// src-tauri/src/marihydro/physics/numerics/limiter/mod.rs

//! 梯度限制器模块

pub mod barth_jespersen;
pub mod venkatakrishnan;

pub use barth_jespersen::BarthJespersenLimiter;
pub use venkatakrishnan::VenkatakrishnanLimiter;

use super::gradient::ScalarGradientStorage;
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LimiterType {
    None,
    BarthJespersen,
    #[default]
    Venkatakrishnan,
}

pub trait GradientLimiter: Send + Sync {
    fn limit<M: MeshAccess>(
        &self,
        field: &[f64],
        gradient: &mut ScalarGradientStorage,
        mesh: &M,
    ) -> MhResult<()>;

    fn compute_limiters<M: MeshAccess>(
        &self,
        field: &[f64],
        gradient: &ScalarGradientStorage,
        mesh: &M,
        output: &mut [f64],
    ) -> MhResult<()>;

    fn name(&self) -> &'static str;
}

pub struct NoLimiter;

impl GradientLimiter for NoLimiter {
    fn limit<M: MeshAccess>(
        &self,
        _: &[f64],
        _: &mut ScalarGradientStorage,
        _: &M,
    ) -> MhResult<()> {
        Ok(())
    }
    fn compute_limiters<M: MeshAccess>(
        &self,
        _: &[f64],
        _: &ScalarGradientStorage,
        mesh: &M,
        output: &mut [f64],
    ) -> MhResult<()> {
        output[..mesh.n_cells()].fill(1.0);
        Ok(())
    }
    fn name(&self) -> &'static str {
        "None"
    }
}
