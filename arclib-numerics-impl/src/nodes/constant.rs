// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::sync::Arc;

use arclib_graph_impl::fnv1a_hash;
use arclib_graph_spec::{GraphContext, Node, NodeId, Shape};
use arclib_numerics_spec::Tensor;
use ndarray::{ArrayD, IxDyn};
use uuid::Uuid;

use crate::NumericsContextValue;

#[derive(Clone)]
pub struct ConstantNode {
    pub id: NodeId,
    pub tensor: Arc<Tensor>,
}

impl ConstantNode {
    pub fn new(tensor: Tensor) -> Self {
        Self {
            id: Uuid::new_v4(),
            tensor: Arc::new(tensor),
        }
    }

    pub fn zeros(shape: Shape) -> Self {
        Self::new(ArrayD::zeros(IxDyn(&shape.0)))
    }

    pub fn from_arc(tensor: Arc<Tensor>) -> Self {
        Self {
            id: Uuid::new_v4(),
            tensor,
        }
    }
}

impl Node<NumericsContextValue> for ConstantNode {
    fn type_id_static() -> u64
    where
        Self: Sized,
    {
        fnv1a_hash("ConstantNode")
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn compute(&mut self, ctx: &mut GraphContext<'_, NumericsContextValue>) {
        ctx.temp
            .insert(self.id, NumericsContextValue::Tensor(self.tensor.clone()));
    }

    fn dependencies(&self) -> Vec<NodeId> {
        Vec::new()
    }

    fn infer_shape(&self, _inputs: &[Shape]) -> Result<Shape, String> {
        Ok(Shape(self.tensor.shape().to_vec()))
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
