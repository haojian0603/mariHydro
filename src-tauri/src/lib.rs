// src-tauri/src/lib.rs

// 声明核心引擎模块
// 具体的代码将位于 src/marihydro/ 目录下
pub mod marihydro {
    // 声明子模块
    pub mod domain; // 计算域 (网格, 掩膜)
    pub mod forcing; // 外部强迫 (风, 潮)
    pub mod geo; // 地理空间 (投影, 变换)
    pub mod infra; // 基础设施 (配置, 日志, 错误)
    pub mod io;
    pub mod physics; // 物理核心 // 输入输出
}

// 声明 Tauri 命令模块
pub mod commands {
    // 这里将来会包含具体的 API 文件
    // pub mod project_cmd;
    // pub mod sim_cmd;
}

// 初始化函数 (供 main.rs 调用)
pub fn init_logging() {
    // 初始化日志系统，设置默认级别为 Info
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();
}
