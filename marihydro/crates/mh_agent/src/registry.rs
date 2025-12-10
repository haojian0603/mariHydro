//! AI 代理注册中心
//!
//! 管理多个 AI 代理的注册、启用/禁用和执行。

use crate::{AIAgent, AiError, PhysicsSnapshot, Assimilable};
use std::collections::HashMap;

/// AI 代理注册中心
///
/// 负责管理所有 AI 代理的生命周期，提供统一的更新和应用接口。
///
/// # 示例
///
/// ```ignore
/// let mut registry = AgentRegistry::new();
///
/// // 注册代理
/// registry.register(Box::new(my_agent));
///
/// // 更新所有代理
/// registry.update_all(&snapshot)?;
///
/// // 应用所有代理的修正
/// registry.apply_all(&mut state)?;
/// ```
pub struct AgentRegistry {
    /// 已注册的代理
    agents: HashMap<String, Box<dyn AIAgent>>,
    /// 代理启用状态
    enabled: HashMap<String, bool>,
    /// 执行顺序（按注册顺序）
    order: Vec<String>,
    /// 是否启用守恒校验
    conservation_check_enabled: bool,
    /// 守恒误差容限
    conservation_tolerance: f64,
}

impl AgentRegistry {
    /// 创建新的注册中心
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            enabled: HashMap::new(),
            order: Vec::new(),
            conservation_check_enabled: true,
            conservation_tolerance: 1e-10,
        }
    }
    
    /// 注册代理
    ///
    /// 代理按注册顺序执行。
    pub fn register(&mut self, agent: Box<dyn AIAgent>) {
        let name = agent.name().to_string();
        if !self.agents.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.enabled.insert(name.clone(), true);
        self.agents.insert(name, agent);
    }
    
    /// 移除代理
    pub fn unregister(&mut self, name: &str) -> Option<Box<dyn AIAgent>> {
        self.enabled.remove(name);
        self.order.retain(|n| n != name);
        self.agents.remove(name)
    }
    
    /// 启用/禁用代理
    pub fn set_enabled(&mut self, name: &str, enabled: bool) {
        if let Some(e) = self.enabled.get_mut(name) {
            *e = enabled;
        }
    }
    
    /// 检查代理是否启用
    pub fn is_enabled(&self, name: &str) -> bool {
        *self.enabled.get(name).unwrap_or(&false)
    }
    
    /// 获取已注册的代理数量
    pub fn len(&self) -> usize {
        self.agents.len()
    }
    
    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }
    
    /// 获取所有代理名称
    pub fn names(&self) -> Vec<&str> {
        self.order.iter().map(|s| s.as_str()).collect()
    }
    
    /// 设置守恒校验开关
    pub fn set_conservation_check(&mut self, enabled: bool) {
        self.conservation_check_enabled = enabled;
    }
    
    /// 设置守恒误差容限
    pub fn set_conservation_tolerance(&mut self, tolerance: f64) {
        self.conservation_tolerance = tolerance;
    }
    
    /// 更新所有启用的代理
    ///
    /// 按注册顺序调用每个启用代理的 `update()` 方法。
    ///
    /// # 参数
    ///
    /// - `snapshot`: 当前物理状态快照
    ///
    /// # 返回
    ///
    /// 如果任何代理更新失败，返回第一个错误
    pub fn update_all(&mut self, snapshot: &PhysicsSnapshot) -> Result<(), AiError> {
        for name in &self.order {
            if *self.enabled.get(name).unwrap_or(&false) {
                if let Some(agent) = self.agents.get_mut(name) {
                    agent.update(snapshot)?;
                }
            }
        }
        Ok(())
    }
    
    /// 应用所有启用的代理修正
    ///
    /// 按注册顺序调用每个启用代理的 `apply()` 方法。
    /// 如果代理要求守恒校验且校验失败，返回错误。
    ///
    /// # 参数
    ///
    /// - `state`: 可同化的物理状态
    ///
    /// # 返回
    ///
    /// 如果任何代理应用失败或守恒校验失败，返回错误
    pub fn apply_all(&self, state: &mut dyn Assimilable) -> Result<(), AiError> {
        for name in &self.order {
            if *self.enabled.get(name).unwrap_or(&false) {
                if let Some(agent) = self.agents.get(name) {
                    // 记录应用前的总量（如果需要守恒校验）
                    let volume_before = if self.conservation_check_enabled 
                        && agent.requires_conservation_check() 
                    {
                        Some(state.total_water_volume())
                    } else {
                        None
                    };
                    
                    // 应用修正
                    agent.apply(state)?;
                    
                    // 守恒校验
                    if let Some(before) = volume_before {
                        let after = state.total_water_volume();
                        let error = (after - before).abs();
                        let relative_error = if before.abs() > 1e-14 {
                            error / before
                        } else {
                            error
                        };
                        
                        if relative_error > self.conservation_tolerance {
                            return Err(AiError::ConservationViolated {
                                expected: before,
                                actual: after,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
    
    /// 更新并应用（组合调用）
    pub fn update_and_apply(
        &mut self,
        snapshot: &PhysicsSnapshot,
        state: &mut dyn Assimilable,
    ) -> Result<(), AiError> {
        self.update_all(snapshot)?;
        self.apply_all(state)?;
        Ok(())
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// 测试用的简单代理
    struct TestAgent {
        name: &'static str,
        updated: bool,
    }
    
    impl TestAgent {
        fn new(name: &'static str) -> Self {
            Self { name, updated: false }
        }
    }
    
    impl AIAgent for TestAgent {
        fn name(&self) -> &'static str {
            self.name
        }
        
        fn update(&mut self, _snapshot: &PhysicsSnapshot) -> Result<(), AiError> {
            self.updated = true;
            Ok(())
        }
        
        fn apply(&self, _state: &mut dyn Assimilable) -> Result<(), AiError> {
            Ok(())
        }
        
        fn requires_conservation_check(&self) -> bool {
            false
        }
    }
    
    #[test]
    fn test_registry_basic() {
        let mut registry = AgentRegistry::new();
        
        registry.register(Box::new(TestAgent::new("test1")));
        registry.register(Box::new(TestAgent::new("test2")));
        
        assert_eq!(registry.len(), 2);
        assert!(registry.is_enabled("test1"));
        assert!(registry.is_enabled("test2"));
        
        registry.set_enabled("test1", false);
        assert!(!registry.is_enabled("test1"));
    }
}
