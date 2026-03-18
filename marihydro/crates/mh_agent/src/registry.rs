//! AI 代理注册中心。

use crate::{AIAgent, AiError, Assimilable, DefaultBackend, PhysicsSnapshot};
use bytemuck::Pod;
use mh_runtime::{Backend, RuntimeScalar};
use mh_runtime::prelude::{Float, FromPrimitive};
use std::collections::HashMap;

/// AI 代理注册中心。
pub struct AgentRegistry<B: Backend = DefaultBackend>
where
    B::Vector2D: Pod,
{
    /// 已注册的代理。
    agents: HashMap<String, Box<dyn AIAgent<B>>>,
    /// 启用状态。
    enabled: HashMap<String, bool>,
    /// 执行顺序。
    order: Vec<String>,
    /// 是否启用守恒检查。
    conservation_check_enabled: bool,
    /// 守恒误差容限。
    conservation_tolerance: B::Scalar,
}

impl<B: Backend> AgentRegistry<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: Pod,
{
    /// 创建注册中心。
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            enabled: HashMap::new(),
            order: Vec::new(),
            conservation_check_enabled: true,
            conservation_tolerance: B::Scalar::from_f64(1e-10).unwrap_or(B::Scalar::EPSILON),
        }
    }

    /// 注册代理。
    pub fn register(&mut self, agent: Box<dyn AIAgent<B>>) {
        let name = agent.name().to_string();
        if !self.agents.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.enabled.insert(name.clone(), true);
        self.agents.insert(name, agent);
    }

    /// 移除代理。
    pub fn unregister(&mut self, name: &str) -> Option<Box<dyn AIAgent<B>>> {
        self.enabled.remove(name);
        self.order.retain(|n| n != name);
        self.agents.remove(name)
    }

    /// 设置代理启用状态。
    pub fn set_enabled(&mut self, name: &str, enabled: bool) {
        if let Some(value) = self.enabled.get_mut(name) {
            *value = enabled;
        }
    }

    /// 查询代理是否启用。
    pub fn is_enabled(&self, name: &str) -> bool {
        *self.enabled.get(name).unwrap_or(&false)
    }

    /// 代理数量。
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// 获取代理名称列表。
    pub fn names(&self) -> Vec<&str> {
        self.order.iter().map(String::as_str).collect()
    }

    /// 设置守恒检查开关。
    pub fn set_conservation_check(&mut self, enabled: bool) {
        self.conservation_check_enabled = enabled;
    }

    /// 设置守恒误差容限。
    pub fn set_conservation_tolerance(&mut self, tolerance: B::Scalar) {
        self.conservation_tolerance = tolerance;
    }

    /// 更新全部启用代理。
    pub fn update_all(&mut self, snapshot: &PhysicsSnapshot<B>) -> Result<(), AiError> {
        for name in &self.order {
            if *self.enabled.get(name).unwrap_or(&false) {
                if let Some(agent) = self.agents.get_mut(name) {
                    agent.update(snapshot)?;
                }
            }
        }
        Ok(())
    }

    /// 应用全部启用代理。
    pub fn apply_all(&self, state: &mut dyn Assimilable<B>) -> Result<(), AiError> {
        for name in &self.order {
            if !*self.enabled.get(name).unwrap_or(&false) {
                continue;
            }

            if let Some(agent) = self.agents.get(name) {
                let volume_before = if self.conservation_check_enabled
                    && agent.requires_conservation_check()
                {
                    Some(state.total_water_volume())
                } else {
                    None
                };

                agent.apply(state)?;

                if let Some(before) = volume_before {
                    let after = state.total_water_volume();
                    let diff = (after - before).abs();
                    let relative_error = if before.abs()
                        > B::Scalar::from_f64(1e-14).unwrap_or(B::Scalar::EPSILON)
                    {
                        diff / before.abs()
                    } else {
                        diff
                    };

                    if relative_error > self.conservation_tolerance {
                        return Err(AiError::ConservationViolated {
                            expected: before.to_f64_lossy(),
                            actual: after.to_f64_lossy(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// 更新并应用全部启用代理。
    pub fn update_and_apply(
        &mut self,
        snapshot: &PhysicsSnapshot<B>,
        state: &mut dyn Assimilable<B>,
    ) -> Result<(), AiError> {
        self.update_all(snapshot)?;
        self.apply_all(state)?;
        Ok(())
    }
}

impl<B: Backend> Default for AgentRegistry<B>
where
    B::Scalar: RuntimeScalar,
    B::Vector2D: Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultBackend;
    use mh_runtime::{DeviceBuffer, RuntimeScalar};

    struct TestAgent {
        name: &'static str,
        updated: bool,
    }

    impl TestAgent {
        fn new(name: &'static str) -> Self {
            Self { name, updated: false }
        }
    }

    impl AIAgent<DefaultBackend> for TestAgent {
        fn name(&self) -> &'static str {
            self.name
        }

        fn update(
            &mut self,
            _snapshot: &PhysicsSnapshot<DefaultBackend>,
        ) -> Result<(), AiError> {
            self.updated = true;
            Ok(())
        }

        fn apply(&self, _state: &mut dyn Assimilable<DefaultBackend>) -> Result<(), AiError> {
            Ok(())
        }

        fn requires_conservation_check(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_registry_basic() {
        let mut registry: AgentRegistry<DefaultBackend> = AgentRegistry::new();

        registry.register(Box::new(TestAgent::new("test1")));
        registry.register(Box::new(TestAgent::new("test2")));

        assert_eq!(registry.len(), 2);
        assert!(registry.is_enabled("test1"));
        assert!(registry.is_enabled("test2"));

        registry.set_enabled("test1", false);
        assert!(!registry.is_enabled("test1"));
    }
}
