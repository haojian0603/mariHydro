//! 存储 trait 定义
//!
//! 定义工作流存储的抽象接口。

use crate::marihydro::core::error::MhResult;
use crate::marihydro::workflow::job::{JobStatus, SimulationJob};

/// 工作流存储 trait
///
/// 提供项目和作业的 CRUD 操作接口。
/// 所有实现都必须是线程安全的 (`Send + Sync`)。
///
/// # 实现者注意
///
/// - 所有方法都应该是幂等的（多次调用产生相同结果）
/// - 更新操作应该返回是否成功修改
/// - 列表操作应该返回空列表而非错误
pub trait WorkflowStorage: Send + Sync {
    // ========== 项目操作 ==========

    /// 保存项目配置
    ///
    /// 如果项目已存在，将覆盖旧数据。
    fn save_project(&self, id: &str, data: &str) -> MhResult<()>;

    /// 加载项目配置
    ///
    /// 如果项目不存在，返回 `Ok(None)`。
    fn load_project(&self, id: &str) -> MhResult<Option<String>>;

    /// 删除项目
    ///
    /// 返回是否成功删除（项目不存在时返回 `false`）。
    fn delete_project(&self, id: &str) -> MhResult<bool>;

    /// 列出所有项目 ID
    fn list_project_ids(&self) -> MhResult<Vec<String>>;

    /// 检查项目是否存在
    fn project_exists(&self, id: &str) -> MhResult<bool> {
        Ok(self.load_project(id)?.is_some())
    }

    // ========== 作业操作 ==========

    /// 保存作业
    ///
    /// 如果作业已存在，将覆盖旧数据。
    fn save_job(&self, job: &SimulationJob) -> MhResult<()>;

    /// 获取作业
    ///
    /// 如果作业不存在，返回 `Ok(None)`。
    fn get_job(&self, id: &str) -> MhResult<Option<SimulationJob>>;

    /// 更新作业状态
    ///
    /// 返回是否成功更新（作业不存在时返回 `false`）。
    fn update_job_status(
        &self,
        id: &str,
        status: JobStatus,
        message: Option<&str>,
    ) -> MhResult<bool>;

    /// 更新作业进度
    ///
    /// 进度值会被限制在 [0.0, 100.0] 范围内。
    /// 返回是否成功更新（作业不存在时返回 `false`）。
    fn update_job_progress(
        &self,
        id: &str,
        progress: f64,
        message: Option<&str>,
    ) -> MhResult<bool>;

    /// 列出项目的所有作业
    fn list_jobs(&self, project_id: &str) -> MhResult<Vec<SimulationJob>>;

    /// 列出指定状态的作业
    fn list_jobs_by_status(&self, status: JobStatus) -> MhResult<Vec<SimulationJob>>;

    /// 删除作业
    ///
    /// 返回是否成功删除（作业不存在时返回 `false`）。
    fn delete_job(&self, id: &str) -> MhResult<bool>;

    /// 删除项目的所有作业
    fn delete_jobs_by_project(&self, project_id: &str) -> MhResult<usize>;
}

/// 带事务支持的存储 trait
///
/// 扩展 `WorkflowStorage`，提供事务功能。
pub trait TransactionalStorage: WorkflowStorage {
    /// 事务类型
    type Transaction<'a>: TransactionContext
    where
        Self: 'a;

    /// 开启事务
    fn begin_transaction(&self) -> MhResult<Self::Transaction<'_>>;
}

/// 事务上下文
///
/// 在事务中执行的操作，支持提交或回滚。
pub trait TransactionContext {
    /// 在事务中保存作业
    fn save_job(&mut self, job: &SimulationJob) -> MhResult<()>;

    /// 在事务中更新作业状态
    fn update_job_status(&mut self, id: &str, status: JobStatus) -> MhResult<()>;

    /// 提交事务
    fn commit(self) -> MhResult<()>;

    /// 回滚事务
    fn rollback(self) -> MhResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // 验证 trait object 可以创建
    fn _assert_object_safe(_: &dyn WorkflowStorage) {}
}
