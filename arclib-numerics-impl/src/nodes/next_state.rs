// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use arclib_graph_impl::fnv1a_hash;
use arclib_graph_spec::{GraphContext, Node, NodeId, Shape};
use uuid::Uuid;

use crate::NumericsContextValue;

#[derive(Clone)]
pub struct NextStateNode {
    pub id: NodeId,
    pub state_id: NodeId,
    pub source_id: NodeId,
}

impl NextStateNode {
    pub fn new(state_id: NodeId, source_id: NodeId) -> Self {
        Self {
            id: Uuid::new_v4(),
            state_id,
            source_id,
        }
    }
}

impl Node<NumericsContextValue> for NextStateNode {
    fn type_id_static() -> u64
    where
        Self: Sized,
    {
        fnv1a_hash("NextStateNode")
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn compute(&mut self, ctx: &mut GraphContext<'_, NumericsContextValue>) {
        let new_val = ctx.temp.remove(&self.source_id).unwrap_or_else(|| {
            panic!(
                "NextStateNode: Source {:?} missing from temp",
                self.source_id
            )
        });

        ctx.next_state.insert(self.state_id, new_val);

        ctx.temp.insert(self.id, NumericsContextValue::Empty);
    }

    fn dependencies(&self) -> Vec<NodeId> {
        vec![self.source_id]
    }

    fn infer_shape(&self, _inputs: &[Shape]) -> Result<Shape, String> {
        Ok(Shape(vec![]))
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
