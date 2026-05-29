// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use arclib_graph_spec::GraphLike;
use arclib_numerics_impl::{ConstantNode, MigrateNode, NumericsContextValue, NumericsGraph};
use arclib_numerics_spec::tensor::{Device, Tensor};
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_migrate_node_cpu_gpu_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let cpu_array = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), data.clone()).unwrap();
    let cpu_tensor = Tensor::from_cpu_array(cpu_array);

    let mut graph = NumericsGraph::new();

    let const_id = graph.add_node(ConstantNode::new(cpu_tensor));
    let upload_id = graph.add_node(MigrateNode::new(const_id, Device::Gpu(0)));
    let download_id = graph.add_node(MigrateNode::new(upload_id, Device::Cpu));

    graph.compile().expect("Graph compilation failed");
    graph.step().expect("Graph step failed");

    let result_tensor = match graph.inner.state_map.get(&download_id) {
        Some(NumericsContextValue::Tensor(t)) => t,
        _ => panic!("Expected Tensor in state_map"),
    };

    assert_eq!(
        result_tensor.device,
        Device::Cpu,
        "Tensor should be back on CPU"
    );

    let result_array = result_tensor.as_cpu();
    let result_data: Vec<f32> = result_array.iter().copied().collect();

    assert_eq!(result_data, data, "GPU roundtrip corrupted data!");

    println!("MigrateNode roundtrip successful! Data: {:?}", result_data);
}
