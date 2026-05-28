// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use arclib_graph_impl::fnv1a_hash;
use arclib_graph_spec::{Node, NodeId, Shape};
use uuid::Uuid;

use crate::context::NumericsContextValue;

#[derive(Clone)]
pub struct EqVariableNode {
    pub id: NodeId,
    pub name: String,
}

impl EqVariableNode {
    pub fn new(name: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
        }
    }
}

impl Node<NumericsContextValue> for EqVariableNode {
    fn type_id_static() -> u64
    where
        Self: Sized,
    {
        fnv1a_hash("EqVariableNode")
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn compute(&mut self, ctx: &mut arclib_graph_spec::GraphContext<'_, NumericsContextValue>) {
        if !ctx.temp.contains_key(&self.id) {
            panic!(
                "EqVariableNode '{}': Value was not injected into the context by the outer graph!",
                self.name
            );
        }
    }

    fn dependencies(&self) -> Vec<NodeId> {
        Vec::new()
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

    fn infer_shape(&self, _inputs: &[Shape]) -> Result<Shape, String> {
        Err(format!(
            "EqVariableNode '{}': Shape must be provided externally via trace_output_shape.",
            self.name
        ))
    }
}
