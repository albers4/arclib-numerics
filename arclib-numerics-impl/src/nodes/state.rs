// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use arclib_graph_impl::fnv1a_hash;
use arclib_graph_spec::{GraphContext, Node, NodeId, Shape};
use uuid::Uuid;

use crate::NumericsContextValue;

#[derive(Clone)]
pub struct StateNode {
    pub id: NodeId,
    pub name: String,
}

impl StateNode {
    pub fn new(name: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
        }
    }
}

impl Node<NumericsContextValue> for StateNode {
    fn type_id_static() -> u64
    where
        Self: Sized,
    {
        fnv1a_hash("StateNode")
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn compute(&mut self, ctx: &mut GraphContext<'_, NumericsContextValue>) {
        if let Some(val) = ctx.state.get(&self.id) {
            ctx.temp.insert(self.id, val.clone());
        } else {
            panic!(
                "StateNode '{}': Missing initial state in ctx.state!",
                self.name
            );
        }
    }

    fn dependencies(&self) -> Vec<NodeId> {
        Vec::new()
    }

    fn infer_shape(&self, _inputs: &[Shape]) -> Result<Shape, String> {
        Err("StateNode shape must be derived from ctx.state during compilation".to_string())
    }

    fn as_node(&self) -> &dyn Node<NumericsContextValue> {
        self
    }

    fn as_node_mut(&mut self) -> &mut dyn Node<NumericsContextValue> {
        self
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Node<NumericsContextValue>> {
        Box::new(self.clone())
    }
}
