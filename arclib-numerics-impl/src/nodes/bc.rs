// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::sync::Arc;

use arclib_graph_impl::fnv1a_hash;
use arclib_graph_spec::{GraphContext, Node, NodeId, Shape};
use arclib_numerics_spec::bc::BcEvaluator;
use uuid::Uuid;

use crate::{NumericsContextValue, bc::BoundaryCondition};

#[derive(Clone)]
pub struct BoundaryConditionNode {
    pub id: NodeId,
    pub state_id: NodeId,
    pub mask_id: NodeId,
    pub values_id: NodeId,
    pub bc: BoundaryCondition,
    pub evaluator: Arc<dyn BcEvaluator>,
}

impl BoundaryConditionNode {
    pub fn new(
        state_id: NodeId,
        mask_id: NodeId,
        values_id: NodeId,
        bc: BoundaryCondition,
        evaluator: Arc<dyn BcEvaluator>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            state_id,
            mask_id,
            values_id,
            bc,
            evaluator,
        }
    }
}

impl Node<NumericsContextValue> for BoundaryConditionNode {
    fn type_id_static() -> u64
    where
        Self: Sized,
    {
        fnv1a_hash("BoundaryConditionNode")
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn compute(&mut self, ctx: &mut GraphContext<'_, NumericsContextValue>) {
        let mut state_arc = match ctx.temp.remove(&self.state_id) {
            Some(NumericsContextValue::Tensor(t)) => t,
            _ => panic!("BoundaryConditionNode: State missing"),
        };

        let mask_arc = match ctx.temp.remove(&self.mask_id) {
            Some(NumericsContextValue::Tensor(t)) => t,
            _ => panic!("BoundaryConditionNode: Mask missing"),
        };
        let values_arc = match ctx.temp.remove(&self.values_id) {
            Some(NumericsContextValue::Tensor(t)) => t,
            _ => panic!("BoundaryConditionNode: Values missing"),
        };

        let state_mut = Arc::make_mut(&mut state_arc);
        self.evaluator.apply(state_mut, &mask_arc, &values_arc);

        ctx.temp
            .insert(self.mask_id, NumericsContextValue::Tensor(mask_arc));
        ctx.temp
            .insert(self.values_id, NumericsContextValue::Tensor(values_arc));

        ctx.temp
            .insert(self.id, NumericsContextValue::Tensor(state_arc));
    }

    fn dependencies(&self) -> Vec<NodeId> {
        vec![self.state_id, self.mask_id, self.values_id]
    }

    fn infer_shape(&self, inputs: &[Shape]) -> Result<Shape, String> {
        if inputs.is_empty() {
            return Err("BoundaryConditionNode requires the state input".to_string());
        }
        Ok(inputs[0].clone())
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
