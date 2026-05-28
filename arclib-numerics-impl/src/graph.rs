// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, RwLock},
};

use arclib_graph_impl::{Graph, topological_sort};
use arclib_graph_spec::{GraphLike, Node, NodeId, Shape};

use crate::{
    BoundaryConditionNode, ConstantNode, NextStateNode, ProbeNode, StateNode,
    context::NumericsContextValue,
    nodes::{EquationGraph, EquationNode, ExportNode, MigrateNode},
};

#[derive(Default)]
pub struct NumericsGraph {
    pub inner: Graph<NumericsContextValue>,
}

impl NumericsGraph {
    pub fn new() -> Self {
        let mut inner = Graph::new();

        inner.register_pool::<EquationNode>();
        inner.register_pool::<ConstantNode>();
        inner.register_pool::<StateNode>();
        inner.register_pool::<NextStateNode>();
        inner.register_pool::<BoundaryConditionNode>();
        inner.register_pool::<ExportNode>();
        inner.register_pool::<ProbeNode>();
        inner.register_pool::<MigrateNode>();

        Self { inner }
    }

    fn get_outer_shape(&self, id: NodeId) -> Result<Shape, String> {
        match self.inner.state_map.get(&id) {
            Some(NumericsContextValue::Tensor(t)) => Ok(t.shape.clone()),
            _ => Err(format!(
                "Cannot determine shape for outer node {:?}. Is it initialized in the context?",
                id
            )),
        }
    }

    pub fn mutate_equation(
        &mut self,
        eq_node_id: NodeId,
        new_graph: EquationGraph,
        new_mapping: HashMap<String, NodeId>,
    ) -> Result<(), String> {
        let mut outer_input_shapes = HashMap::new();
        for (name, &outer_id) in &new_mapping {
            let shape = self.get_outer_shape(outer_id)?;
            outer_input_shapes.insert(name.clone(), shape);
        }

        let node = self
            .inner
            .get_node_mut::<EquationNode>(&eq_node_id)
            .ok_or("Node not found or not an Equationnode")?;

        node.graph = Arc::new(RwLock::new(new_graph));
        node.input_mapping = new_mapping;
        node.invalidate();

        node.lower(&outer_input_shapes)?;

        self.inner.rebuild_schedule()?;

        Ok(())
    }
}

impl GraphLike<NumericsContextValue> for NumericsGraph {
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
        self.inner.storage.build_dependency_edges();

        let schedule = topological_sort::<NumericsContextValue>(&self.inner.storage)?;

        let mut outer_shape = HashMap::new();
        for &node_id in &schedule {
            // Prefer actual runtime shapes if user seeded the values_map
            if let Some(NumericsContextValue::Tensor(t)) = self.inner.state_map.get(&node_id) {
                outer_shape.insert(node_id, t.shape.clone());
                continue;
            }

            // Fallback to symbolic shape inference
            let node = self.inner.get_node_dyn(node_id)?;
            let deps = node.dependencies();
            let in_shapes: Vec<Shape> = deps.iter()
                .map(|dep_id| outer_shape.get(dep_id)
                    .cloned()
                    .unwrap_or_else(|| panic!("Dependency {:?} shape missing during inference. Topological order is broken.", dep_id)))
                .collect();

            let shape = node.infer_shape(&in_shapes)?;
            outer_shape.insert(node_id, shape);
        }

        for &node_id in &schedule {
            let node = self.inner.get_node_dyn_mut(node_id)?;
            node.try_lower(&outer_shape)?;
        }

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

impl Debug for NumericsGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}
