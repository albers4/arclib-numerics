// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::ffi::c_void;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use arclib_graph_impl::fnv1a_hash;
use arclib_graph_spec::{GraphContext, GraphLike, Node, NodeId, Shape};
use arclib_numerics_spec::kernel::CompiledKernel;
use arclib_numerics_spec::tensor::Tensor;
use ndarray::ArrayD;
use uuid::Uuid;

use crate::{context::NumericsContextValue, nodes::equation::graph::EquationGraph};

#[derive(Clone)]
pub struct EquationNode {
    pub id: NodeId,
    pub graph: Arc<RwLock<EquationGraph>>,
    pub input_mapping: HashMap<String, NodeId>,
    pub compiled_kernel: Option<CompiledKernel>,
    pub output_shape: Option<Shape>,
}

impl EquationNode {
    pub fn new(graph: EquationGraph, input_mapping: HashMap<String, NodeId>) -> Self {
        Self {
            id: Uuid::new_v4(),
            graph: Arc::new(RwLock::new(graph)),
            input_mapping,
            compiled_kernel: None,
            output_shape: None,
        }
    }

    pub fn lower(&mut self, outer_input_shapes: &HashMap<String, Shape>) -> Result<(), String> {
        let mut graph = self
            .graph
            .write()
            .map_err(|e| format!("Lock poisoned: {}", e))?;

        graph.inner.compile()?;

        let out_shape = graph.trace_output_shape(outer_input_shapes)?;
        self.output_shape = Some(out_shape);

        let kernel = graph.compiler.compile(&graph.inner);
        self.compiled_kernel = Some(kernel);

        Ok(())
    }

    pub fn invalidate(&mut self) {
        self.compiled_kernel = None;
    }

    /// Safely extracts an Arc<Tensor> from the context.
    /// Panics if missing or wrong type.
    fn take_tensor(ctx: &mut GraphContext<'_, NumericsContextValue>, id: &NodeId) -> Arc<Tensor> {
        match ctx.temp.remove(id) {
            Some(NumericsContextValue::Tensor(arc_t)) => arc_t,
            Some(other) => {
                ctx.temp.insert(*id, other);
                panic!("Node {} is not a Tensor", id);
            }
            None => panic!("Missing node {} in context", id),
        }
    }

    /// Helper to get or allocate the output tensor
    fn take_or_alloc_output(
        ctx: &mut GraphContext<'_, NumericsContextValue>,
        id: &NodeId,
        shape: Shape,
    ) -> Arc<Tensor> {
        let ix_shape = ndarray::IxDyn(&shape.0);
        match ctx.temp.remove(id) {
            Some(NumericsContextValue::Tensor(v)) if v.shape == shape => v,
            Some(other) => {
                ctx.temp.insert(*id, other);
                Arc::new(Tensor::from_cpu_array(ArrayD::zeros(ix_shape)))
            }
            None => Arc::new(Tensor::from_cpu_array(ArrayD::zeros(ix_shape))),
        }
    }
}

impl Node<NumericsContextValue> for EquationNode {
    fn type_id_static() -> u64
    where
        Self: Sized,
    {
        fnv1a_hash("EquationNode")
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn compute(&mut self, ctx: &mut GraphContext<'_, NumericsContextValue>) {
        let mut input_arcs: Vec<Arc<Tensor>> = Vec::new();
        let mut input_ids: Vec<NodeId> = Vec::new();

        let mut sorted_keys: Vec<_> = self.input_mapping.keys().collect();
        sorted_keys.sort();

        for key in sorted_keys {
            let id = self.input_mapping[key];
            input_arcs.push(Self::take_tensor(ctx, &id));
            input_ids.push(id);
        }

        let out_shape = self.output_shape.clone().expect("Node not lowered");
        let mut output_arc = Self::take_or_alloc_output(ctx, &self.id, out_shape.clone());

        // CRITICAL: Ensure contiguous C-style memory layout for the C-kernel
        let input_views: Vec<_> = input_arcs
            .iter()
            .map(|arc_t| arc_t.as_cpu().as_standard_layout())
            .collect();

        let input_ptrs: Vec<*const c_void> = input_views
            .iter()
            .map(|v| v.as_ptr() as *const c_void)
            .collect();

        let output_tensor_mut = Arc::make_mut(&mut output_arc);
        let output_ptr = output_tensor_mut.as_mut_ptr();
        let mut output_ptrs = [output_ptr as *mut c_void];

        if let Some(kernel) = &self.compiled_kernel {
            let params_ptr = kernel.params_bytes.as_ptr() as *const c_void;

            unsafe {
                (kernel.forward)(
                    input_ptrs.as_ptr(),
                    output_ptrs.as_mut_ptr(),
                    kernel.workspace.as_ptr() as *mut c_void,
                    params_ptr,
                )
            }
        } else {
            todo!("Fallback interpret");
        }

        for (id, arc_t) in input_ids.into_iter().zip(input_arcs) {
            ctx.temp.insert(id, NumericsContextValue::Tensor(arc_t));
        }
        ctx.temp
            .insert(self.id, NumericsContextValue::Tensor(output_arc));
    }

    fn dependencies(&self) -> Vec<NodeId> {
        let mut deps: Vec<NodeId> = self.input_mapping.values().cloned().collect();
        deps.sort();
        deps
    }

    fn try_lower(&mut self, outer_shape: &HashMap<NodeId, Shape>) -> Result<(), String> {
        let mut named_shapes = HashMap::new();

        for (name, &outer_id) in &self.input_mapping {
            if let Some(shape) = outer_shape.get(&outer_id) {
                named_shapes.insert(name.clone(), shape.clone());
            } else {
                return Err(format!("Missing outer shape for input '{}'", name));
            }
        }

        self.lower(&named_shapes)
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

    fn as_node(&self) -> &dyn Node<NumericsContextValue> {
        self
    }

    fn infer_shape(&self, inputs: &[Shape]) -> Result<Shape, String> {
        let deps = self.dependencies();
        let mut named_shapes = HashMap::new();

        for (dep_id, shape) in deps.into_iter().zip(inputs.iter()) {
            if let Some((name, _)) = self.input_mapping.iter().find(|&(_, &id)| id == dep_id) {
                named_shapes.insert(name.clone(), shape.clone());
            }
        }

        let mut graph_write = self
            .graph
            .write()
            .map_err(|e| format!("RwLock poisoned: {}", e))?;
        graph_write.trace_output_shape(&named_shapes)
    }
}
