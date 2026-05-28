// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use arclib_graph_impl::Graph;
use arclib_graph_spec::{GraphContext, GraphLike, Node, Shape};
use arclib_numerics_impl::{
    EquationGraph, EquationNode, NumericsContextValue, NumericsGraph, StateNode,
};
use arclib_numerics_spec::{
    checkpoint::CheckpointStrategy, domain::DomainCompiler, kernel::CompiledKernel, tensor::Tensor,
};
use ndarray::{ArrayD, arr1};
use std::{collections::HashMap, ffi::c_void, sync::Arc};
use uuid::Uuid;

pub struct MockCompiler;
impl DomainCompiler<NumericsContextValue> for MockCompiler {
    fn compile(&self, _graph: &Graph<NumericsContextValue>) -> CompiledKernel {
        CompiledKernel {
            name: "mock_kernel".into(),
            forward: mock_ffi_add_one,
            backward: None,
            workspace: vec![],
            strategy: CheckpointStrategy::Stateless,
            params_bytes: vec![],
        }
    }
}

unsafe extern "C" fn mock_ffi_add_one(
    inputs: *const *const c_void,
    outputs: *const *mut c_void,
    _workspace: *mut c_void,
    _params: *const c_void,
) {
    unsafe {
        let in_ptr = *(inputs.add(0)) as *const f32;
        let out_ptr = *(outputs.add(0)) as *mut f32;
        // Hardcoded size 4 for this specific test
        for i in 0..4 {
            *out_ptr.add(i) = *in_ptr.add(i) + 1.0;
        }
    }
}

#[test]
fn test_shape_inference_and_broadcasting() {
    // vel + (dt * gravity)
    // vel: [10, 10, 3], dt: [], gravity: [3]
    let mut eq = EquationGraph::new(Arc::new(MockCompiler));

    let vel_id = eq.var("vel");
    let dt_id = eq.const_scalar(0.01);
    let grav_id = eq.const_vector(&[0.0, -9.81, 0.0]);

    let dt_grav = eq.mul(dt_id, grav_id); // [] * [3] -> [3]
    let result = eq.add(vel_id, dt_grav); // [10, 10, 3] + [3] -> [10, 10, 3]

    eq.set_output(result);
    eq.compile().unwrap();

    let mut shapes = HashMap::new();
    shapes.insert("vel".to_string(), Shape(vec![10, 10, 3]));

    let out_shape = eq.trace_output_shape(&shapes).unwrap();
    assert_eq!(out_shape, Shape(vec![10, 10, 3]), "Broadcasting failed!");
}

#[test]
fn test_equation_node_ffi_execution() {
    let eq_graph = EquationGraph::new(Arc::new(MockCompiler));
    let mut eq_node = EquationNode::new(eq_graph, HashMap::new());

    let id_a = Uuid::new_v4();
    eq_node.input_mapping.insert("a".to_string(), id_a);
    eq_node.output_shape = Some(Shape(vec![4]));

    // Manually inject the mock FFI kernel
    eq_node.compiled_kernel = Some(MockCompiler.compile(&Graph::new()));

    let mut temp_map = HashMap::new();
    let state_map = HashMap::new();
    let mut next_state_map = HashMap::new();

    let mut ctx: GraphContext<'_, NumericsContextValue> = GraphContext {
        temp: &mut temp_map,
        state: &state_map,
        next_state: &mut next_state_map,
    };

    let input_tensor: ArrayD<f32> = arr1(&[1.0, 2.0, 3.0, 4.0]).into_dyn();

    ctx.temp.insert(
        id_a,
        NumericsContextValue::Tensor(Arc::new(Tensor::from_cpu_array(input_tensor))),
    );

    eq_node.compute(&mut ctx);

    if let Some(NumericsContextValue::Tensor(out)) = ctx.temp.get(&eq_node.id) {
        let expected: ArrayD<f32> = arr1(&[2.0, 3.0, 4.0, 5.0]).into_dyn();
        assert_eq!(out.as_cpu(), &expected, "C-Kernel FFI addition failed");
    } else {
        panic!("Output tensor missing from context");
    }

    assert!(ctx.temp.contains_key(&id_a), "Input tensor was lost!");
}

#[test]
fn test_dynamic_mutation_and_rewiring() {
    let mut num_graph = NumericsGraph::new();

    let id_u = num_graph.add_node(StateNode::new("u"));
    let id_v = num_graph.add_node(StateNode::new("v"));

    num_graph.inner.state_map.insert(
        id_u,
        NumericsContextValue::Tensor(Arc::new(Tensor::from_cpu_array(
            arr1(&[1.0, 2.0]).into_dyn(),
        ))),
    );
    num_graph.inner.state_map.insert(
        id_v,
        NumericsContextValue::Tensor(Arc::new(Tensor::from_cpu_array(
            arr1(&[10.0, 20.0]).into_dyn(),
        ))),
    );

    // Initial equation u + v
    let mut initial_eq = EquationGraph::new(Arc::new(MockCompiler));
    let var_u = initial_eq.var("u");
    let var_v = initial_eq.var("v");
    let result = initial_eq.add(var_u, var_v);
    initial_eq.set_output(result);

    let mut mapping = HashMap::new();
    mapping.insert("u".to_string(), id_u);
    mapping.insert("v".to_string(), id_v);

    let eq_node_id = num_graph.add_node(EquationNode::new(initial_eq, mapping));

    num_graph.compile().unwrap();
    num_graph.step().unwrap();

    let deps = num_graph.get_node_dyn(eq_node_id).unwrap().dependencies();
    assert_eq!(deps.len(), 2);

    let mut new_eq = EquationGraph::new(Arc::new(MockCompiler));
    let new_var_u = new_eq.var("u");
    let c_2 = new_eq.const_scalar(2.0);
    let new_result = new_eq.mul(new_var_u, c_2);
    new_eq.set_output(new_result);

    let mut new_mapping = HashMap::new();
    new_mapping.insert("u".to_string(), id_u); // no 'v' in the new mapping

    num_graph
        .mutate_equation(eq_node_id, new_eq, new_mapping)
        .unwrap();

    // Verify the graph was rewired dynamically
    let new_deps = num_graph.get_node_dyn(eq_node_id).unwrap().dependencies();
    assert_eq!(
        new_deps.len(),
        1,
        "Dependency on 'v' should have been dropped!"
    );
    assert_eq!(new_deps[0], id_u);

    // Verify the state was NOT wuped (Context still holds u and v)
    assert!(num_graph.inner.state_map.contains_key(&id_u));
    assert!(num_graph.inner.state_map.contains_key(&id_v));

    // Verify the shape was successfully re-inferred for the new equation
    let node = num_graph.get_node::<EquationNode>(&eq_node_id).unwrap();
    assert_eq!(
        node.output_shape,
        Some(Shape(vec![2])),
        "Shape re-inference failed!"
    );
}
