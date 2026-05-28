// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::sync::Arc;

use arclib_graph_impl::fnv1a_hash;
use arclib_graph_spec::{Node, NodeId, Shape};
use arclib_numerics_spec::tensor::Tensor;
use ndarray::ArrayD;
use uuid::Uuid;

use crate::context::NumericsContextValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EqBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone)]
pub struct EqBinaryNode {
    pub id: NodeId,
    pub op: EqBinaryOp,
    pub lhs: NodeId,
    pub rhs: NodeId,
}

impl EqBinaryNode {
    pub fn new(op: EqBinaryOp, lhs: NodeId, rhs: NodeId) -> Self {
        Self {
            id: Uuid::new_v4(),
            op,
            lhs,
            rhs,
        }
    }
}

impl Node<NumericsContextValue> for EqBinaryNode {
    fn type_id_static() -> u64
    where
        Self: Sized,
    {
        fnv1a_hash("EqBinaryNode")
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn compute(&mut self, ctx: &mut arclib_graph_spec::GraphContext<'_, NumericsContextValue>) {
        let lhs: &Tensor = match ctx.temp.get(&self.lhs) {
            Some(NumericsContextValue::Tensor(t)) => t,
            _ => panic!("EqBinaryNode: LHS missing"),
        };
        let rhs: &Tensor = match ctx.temp.get(&self.rhs) {
            Some(NumericsContextValue::Tensor(t)) => t,
            _ => panic!("EqBinaryNode: RHS missing"),
        };

        let lhs_arr = lhs.as_cpu();
        let rhs_arr = rhs.as_cpu();

        let result: ArrayD<f32> = match self.op {
            EqBinaryOp::Add => lhs_arr + rhs_arr,
            EqBinaryOp::Sub => lhs_arr - rhs_arr,
            EqBinaryOp::Mul => lhs_arr * rhs_arr,
            EqBinaryOp::Div => lhs_arr / rhs_arr,
        };

        let result_tensor = Tensor::from_cpu_array(result);

        ctx.temp.insert(
            self.id,
            NumericsContextValue::Tensor(Arc::new(result_tensor)),
        );
    }

    fn dependencies(&self) -> Vec<NodeId> {
        vec![self.lhs, self.rhs]
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
        let a = &inputs[0].0;
        let b = &inputs[1].0;

        if a == b {
            return Ok(inputs[0].clone());
        }

        // Scalar broadcasting
        if a.is_empty() {
            return Ok(inputs[1].clone());
        }
        if b.is_empty() {
            return Ok(inputs[0].clone());
        }

        // Standard NumPy trailing-dimension broadcasting
        let mut out_shape = Vec::new();
        let max_len = a.len().max(b.len());

        for i in 0..max_len {
            let dim_a = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
            let dim_b = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

            if dim_a == dim_b {
                out_shape.push(dim_a);
            } else if dim_a == 1 {
                out_shape.push(dim_b);
            } else if dim_b == 1 {
                out_shape.push(dim_a);
            } else {
                return Err(format!(
                    "EqBinaryNode ({:?}): Cannot broadcast shapes {:?} and {:?}",
                    self.op, inputs[0], inputs[1]
                ));
            }
        }

        out_shape.reverse();
        Ok(Shape(out_shape))
    }
}
