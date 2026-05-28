// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::sync::Arc;

use arclib_graph_impl::fnv1a_hash;
use arclib_graph_spec::{Node, NodeId, Shape};
use ndarray::ArrayD;
use uuid::Uuid;

use crate::context::NumericsContextValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EqUnaryOp {
    Neg,
    Sqrt,
    Exp,
    Abs,
}

#[derive(Clone)]
pub struct EqUnaryNode {
    pub id: NodeId,
    pub op: EqUnaryOp,
    pub input: NodeId,
}

impl EqUnaryNode {
    pub fn new(op: EqUnaryOp, input: NodeId) -> Self {
        Self {
            id: Uuid::new_v4(),
            op,
            input,
        }
    }
}

impl Node<NumericsContextValue> for EqUnaryNode {
    fn type_id_static() -> u64
    where
        Self: Sized,
    {
        fnv1a_hash("EqUnaryNode")
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn compute(&mut self, ctx: &mut arclib_graph_spec::GraphContext<'_, NumericsContextValue>) {
        let in_arc = match ctx.temp.get(&self.input) {
            Some(NumericsContextValue::Tensor(t)) => t,
            _ => panic!("EqUnaryNode: Input missing or not a tensor"),
        };

        let in_tensor = in_arc.as_ref();

        let result: ArrayD<f32> = match self.op {
            EqUnaryOp::Neg => -in_tensor,
            EqUnaryOp::Sqrt => in_tensor.mapv(|x| x.sqrt()),
            EqUnaryOp::Exp => in_tensor.mapv(|x| x.exp()),
            EqUnaryOp::Abs => in_tensor.mapv(|x| x.abs()),
        };

        ctx.temp
            .insert(self.id, NumericsContextValue::Tensor(Arc::new(result)));
    }

    fn dependencies(&self) -> Vec<NodeId> {
        vec![self.input]
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

    fn infer_shape(&self, inputs: &[Shape]) -> Result<Shape, String> {
        Ok(inputs[0].clone())
    }
}
