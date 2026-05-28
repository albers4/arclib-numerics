// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::{collections::HashMap, fmt::Debug, sync::Arc};

use arclib_graph_impl::{Graph, topological_sort};
use arclib_graph_spec::{GraphLike, Node, NodeId, Shape};
use arclib_numerics_spec::{Tensor, domain::DomainCompiler};

use crate::{
    context::NumericsContextValue,
    nodes::{EqBinaryNode, EqBinaryOp, EqConstantNode, EqUnaryNode, EqUnaryOp, EqVariableNode},
};

pub struct EquationGraph {
    pub inner: Graph<NumericsContextValue>,
    pub compiler: Arc<dyn DomainCompiler<NumericsContextValue>>,

    pub input_vars: HashMap<String, NodeId>,
    pub output_id: Option<NodeId>,
}

impl EquationGraph {
    pub fn new(compiler: Arc<dyn DomainCompiler<NumericsContextValue>>) -> Self {
        let mut inner = Graph::new();

        inner.register_pool::<EqVariableNode>();
        inner.register_pool::<EqConstantNode>();
        inner.register_pool::<EqUnaryNode>();
        inner.register_pool::<EqBinaryNode>();

        Self {
            inner,
            compiler,
            input_vars: HashMap::new(),
            output_id: None,
        }
    }

    pub fn var(&mut self, name: &str) -> NodeId {
        let id = self.inner.add_node(EqVariableNode::new(name));
        self.input_vars.insert(name.to_string(), id);
        id
    }

    // --- Binary Ops ---
    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.inner
            .add_node(EqBinaryNode::new(EqBinaryOp::Add, lhs, rhs))
    }

    pub fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.inner
            .add_node(EqBinaryNode::new(EqBinaryOp::Sub, lhs, rhs))
    }

    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.inner
            .add_node(EqBinaryNode::new(EqBinaryOp::Mul, lhs, rhs))
    }

    pub fn div(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.inner
            .add_node(EqBinaryNode::new(EqBinaryOp::Div, lhs, rhs))
    }

    // --- Unary Ops ---
    pub fn neg(&mut self, input: NodeId) -> NodeId {
        self.inner.add_node(EqUnaryNode::new(EqUnaryOp::Neg, input))
    }

    pub fn sqrt(&mut self, input: NodeId) -> NodeId {
        self.inner
            .add_node(EqUnaryNode::new(EqUnaryOp::Sqrt, input))
    }

    pub fn exp(&mut self, input: NodeId) -> NodeId {
        self.inner.add_node(EqUnaryNode::new(EqUnaryOp::Exp, input))
    }

    pub fn abs(&mut self, input: NodeId) -> NodeId {
        self.inner.add_node(EqUnaryNode::new(EqUnaryOp::Abs, input))
    }

    // --- Constant ---
    pub fn const_scalar(&mut self, val: f32) -> NodeId {
        self.inner.add_node(EqConstantNode::scalar(val))
    }

    pub fn const_vector(&mut self, vals: &[f32]) -> NodeId {
        self.inner.add_node(EqConstantNode::vector(vals))
    }

    pub fn const_tensor(&mut self, tensor: Tensor) -> NodeId {
        self.inner.add_node(EqConstantNode::tensor(tensor))
    }

    pub fn add_input(&mut self, name: &str, node: impl Node<NumericsContextValue>) -> NodeId {
        let id = self.inner.add_node(node);
        self.input_vars.insert(name.to_string(), id);
        id
    }

    pub fn set_output(&mut self, id: NodeId) {
        self.output_id = Some(id);
    }

    pub fn trace_output_shape(
        &mut self,
        input_shapes: &HashMap<String, Shape>,
    ) -> Result<arclib_graph_spec::Shape, String> {
        let output_id = self.output_id.ok_or_else(|| {
            "EquationGraph output_id not set! Call set_output() before compiling.".to_string()
        })?;

        let mut node_shapes: HashMap<NodeId, Shape> = HashMap::new();

        for (name, &id) in &self.input_vars {
            let shape = input_shapes
                .get(name)
                .ok_or_else(|| format!("Missing shape for input variable '{}'", name))?;
            node_shapes.insert(id, shape.clone());
        }

        let schedule = match self.inner.get_execution_order() {
            Ok(s) => s,
            Err(_) => {
                self.inner.storage.build_dependency_edges();
                topological_sort::<NumericsContextValue>(&self.inner.storage)?
            }
        };

        for &node_id in &schedule {
            if node_shapes.contains_key(&node_id) {
                continue;
            }

            let node = self.inner.get_node_dyn(node_id)?;
            let deps = node.dependencies();

            let in_shapes: Vec<Shape> = deps
                .iter()
                .map(|dep_id| node_shapes.get(dep_id).cloned().unwrap())
                .collect();

            let out_shape = node.infer_shape(&in_shapes)?;
            node_shapes.insert(node_id, out_shape);
        }

        node_shapes.get(&output_id).cloned().ok_or_else(|| {
            format!(
                "Output node {:?} shape not found in traced graph",
                self.output_id
            )
        })
    }
}

impl GraphLike<NumericsContextValue> for EquationGraph {
    fn get_node<T: Node<NumericsContextValue>>(&self, id: &NodeId) -> Option<&T> {
        self.inner.get_node(id)
    }

    fn get_node_mut<T: Node<NumericsContextValue>>(&mut self, id: &NodeId) -> Option<&mut T> {
        self.inner.get_node_mut(id)
    }

    fn iter<T: Node<NumericsContextValue>>(&self) -> impl Iterator<Item = &T> + '_ {
        self.inner.iter()
    }

    fn iter_mut<T: Node<NumericsContextValue>>(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        self.inner.iter_mut()
    }

    fn register_pool<T: Node<NumericsContextValue>>(&mut self) {
        self.inner.register_pool::<T>();
    }

    fn add_node<T: Node<NumericsContextValue>>(&mut self, node: T) -> arclib_graph_spec::NodeId {
        self.inner.add_node(node)
    }

    fn connect(&mut self, source: NodeId, target: NodeId) -> Result<(), String> {
        self.inner.connect(source, target)
    }

    fn compile(&mut self) -> Result<(), String> {
        self.inner.compile()
    }

    fn validate_inputs(&self) -> Result<(), String> {
        self.inner.validate_inputs()
    }

    fn step(&mut self) -> Result<(), String> {
        self.inner.step()
    }

    fn get_execution_order(&self) -> Result<Vec<NodeId>, String> {
        self.inner.get_execution_order()
    }

    fn get_node_dyn(&self, id: NodeId) -> Result<&dyn Node<NumericsContextValue>, String> {
        self.inner.get_node_dyn(id)
    }

    fn rebuild_schedule(&mut self) -> Result<(), String> {
        self.inner.rebuild_schedule()
    }

    fn get_node_dyn_mut(
        &mut self,
        id: NodeId,
    ) -> Result<&mut dyn Node<NumericsContextValue>, String> {
        self.inner.get_node_dyn_mut(id)
    }
}

impl Debug for EquationGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}
