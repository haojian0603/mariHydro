//! 求解器模块
//! 变更：step_strang_split 增加 AI 代理模型的逻辑分支

use crate::simulation::grid::Grid;

pub struct Solver {
    // ...
}

impl Solver {
    // ... new, compute_dt ...

    /// 执行单步时间积分
    pub fn step_strang_split(&mut self, grid: &mut Grid, dt: f64, should_assimilate: bool) {
        // [核心特性] AI 代理模型融合 (阶段 5 预留)
        // 如果 Config 开启了 use_ai_proxy，且满足某些条件，则调用 AI 推理
        /*
        if grid.config.use_ai_proxy {
            // 调用 crate::simulation::ai_proxy::predict_next_step(grid, dt);
            // return;
        }
        */

        // 正常的 FVM 物理求解流程
        self.apply_all_sources(grid, dt / 2.0, should_assimilate);
        self.apply_advection(grid, dt);
        self.apply_all_sources(grid, dt / 2.0, should_assimilate);
    }

    // ...
}
