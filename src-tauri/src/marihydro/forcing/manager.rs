// src\marihydro\forcing\manager.rs
use super::context::ForcingContext;
use super::tide::{ConstantTide, TideProvider};
use super::wind::WindProvider;
use crate::marihydro::domain::boundary::BoundaryForcing;
use crate::marihydro::domain::mesh::Mesh;
use crate::marihydro::infra::error::MhResult;
use crate::marihydro::infra::manifest::ProjectManifest;
use chrono::{DateTime, Utc}; // 实际项目中应根据配置加载 HarmonicTide 或 TPXO

pub struct ForcingManager {
    context: ForcingContext,
    boundary_forcing: BoundaryForcing,

    wind_provider: Option<WindProvider>,
    tide_provider: Box<dyn TideProvider>,
}

impl ForcingManager {
    pub fn init(manifest: &ProjectManifest, mesh: &Mesh) -> MhResult<Self> {
        // 初始化空上下文
        let context = ForcingContext::new(mesh.nx, mesh.ny, mesh.ng, 0.0, 101325.0);

        // 初始化风场
        let wind_src = manifest
            .sources
            .iter()
            .find(|s| s.mappings.iter().any(|m| m.target_var == "wind_u"));

        let wind_provider = if let Some(src) = wind_src {
            Some(WindProvider::init(src, mesh, manifest)?)
        } else {
            None
        };

        // 初始化潮汐
        // 根据 Manifest 配置选择 Tide Provider
        // 此处暂时硬编码为 ConstantTide(0.0) 作为基础实现
        // 生产环境应解析 manifest.features 中的 Boundary 定义
        let tide_provider = Box::new(ConstantTide { level: 0.0 });

        Ok(Self {
            context,
            boundary_forcing: BoundaryForcing::default(),
            wind_provider,
            tide_provider,
        })
    }

    /// 更新所有环境场到指定时刻
    pub fn update(&mut self, time: DateTime<Utc>, _mesh: &Mesh) -> MhResult<()> {
        // 1. 更新风场
        if let Some(wp) = &mut self.wind_provider {
            wp.get_wind_at(
                time,
                &mut self.context.wind_u.view_mut(),
                &mut self.context.wind_v.view_mut(),
            )?;
        }

        // 2. 更新潮位边界
        self.boundary_forcing = self.tide_provider.get_forcing(time)?;

        // 3. 重置源项累加器 (例如河流流量是每步刷新的还是持续的？)
        // 对于 ForcingContext 中的 source_mass_flux，通常在 Engine 内部根据 RiverProvider 计算
        // 这里仅作重置，防止上一时刻的残留
        self.context.reset_sources();

        Ok(())
    }

    pub fn get_context(&self) -> &ForcingContext {
        &self.context
    }

    pub fn get_boundary_forcing(&self) -> &BoundaryForcing {
        &self.boundary_forcing
    }
}
