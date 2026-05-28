// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::{collections::HashMap, env, path::PathBuf, sync::Arc};

use arclib_graph_spec::GraphLike;
use arclib_numerics_impl::{
    BoundaryCondition, BoundaryConditionNode, ConstantNode, EquationGraph, EquationNode,
    ExportNode, NextStateNode, NumericsContextValue, NumericsGraph, ProbeNode, StateNode,
    domains::{
        grid::{LatticeCompiler, LbmProbeExtractor, LbmVtkExporter, LbmZouHeVelocity},
        probe::ScalarProbeExtractor,
    },
};
use ndarray::{ArrayD, IxDyn};

fn test_output_dir() -> PathBuf {
    let dir = env::var("ARCLIB_TEST_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::temp_dir().join("arclib_test"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_lbm_cavity() {
    let nx = 100;
    let ny = 100;
    let omega = 1.7; // relaxation parameter
    let u_lid = 0.1;

    let mut graph = NumericsGraph::new();

    // --- SOLID MASK ---
    let mut solid_data = vec![0.0f32; nx * ny];
    for x in 0..nx {
        for y in 0..ny {
            let idx = x * ny + y;

            if y == 0 {
                solid_data[idx] = 1.0;
            }
            if x == 0 {
                solid_data[idx] = 1.0;
            }
            if x == nx - 1 {
                solid_data[idx] = 1.0;
            }
        }
    }
    let solid_tensor = ArrayD::from_shape_vec(IxDyn(&[nx, ny]), solid_data).unwrap();
    let solid_id = graph.add_node(ConstantNode::new(solid_tensor));

    // --- FLUID STATE ----
    let f_id = graph.add_node(StateNode::new("f"));
    let mut f_data = vec![0.0f32; nx * ny * 9];
    let w = [
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ];
    for x in 0..nx {
        for y in 0..ny {
            let idx = (x * ny + y) * 9;
            for q in 0..9 {
                f_data[idx + q] = w[q];
            }
        }
    }
    let f_tensor = ArrayD::from_shape_vec(IxDyn(&[nx, ny, 9]), f_data).unwrap();
    graph
        .inner
        .state_map
        .insert(f_id, NumericsContextValue::Tensor(Arc::new(f_tensor)));

    // --- LBM EQUATION (Stream + Collide) ---
    let compiler = Arc::new(LatticeCompiler::new(nx, ny, omega));
    let mut eq = EquationGraph::new(compiler.clone());
    let f_var = eq.var("f");
    let _solid_var = eq.var("solid");
    eq.set_output(f_var);

    let mut mapping = HashMap::new();
    mapping.insert("f".to_string(), f_id);
    mapping.insert("solid".to_string(), solid_id);
    let eq_node_id = graph.add_node(EquationNode::new(eq, mapping));

    // --- MOVING LID BC (Zou/He)
    let mut lid_mask_data = vec![0.0f32; nx * ny];
    let mut u_lid_data = vec![0.0f32; nx * ny * 2];

    for x in 1..(nx - 1) {
        let y = ny - 1;

        let mask_idx = x * ny + y;
        lid_mask_data[mask_idx] = 1.0;

        let u_idx = (x * ny + y) * 2;
        u_lid_data[u_idx + 0] = u_lid;
        u_lid_data[u_idx + 1] = 0.0;
    }
    let lid_mask_tensor = ArrayD::from_shape_vec(IxDyn(&[nx, ny]), lid_mask_data).unwrap();
    let lid_mask_id = graph.add_node(ConstantNode::new(lid_mask_tensor));

    let u_lid_tensor = ArrayD::from_shape_vec(IxDyn(&[nx, ny, 2]), u_lid_data).unwrap();
    let u_lid_id = graph.add_node(ConstantNode::new(u_lid_tensor));

    let bc_meta = BoundaryCondition::dirichlet();
    let bc_node_id = graph.add_node(BoundaryConditionNode::new(
        eq_node_id,
        lid_mask_id,
        u_lid_id,
        bc_meta,
        Arc::new(LbmZouHeVelocity),
    ));

    // --- PROBE 1 ---
    let mask_probe_id = graph.add_node(ProbeNode::new(
        "Solid Mask",
        solid_id,
        vec![vec![0, 0], vec![nx - 1, 0], vec![nx / 2, ny / 2]],
        1000,
        Arc::new(ScalarProbeExtractor),
    ));

    // --- PROBE 2 ---
    let fluid_probe_id = graph.add_node(ProbeNode::new(
        "Fluid State",
        bc_node_id,
        vec![vec![50, ny - 1], vec![50, 50], vec![50, 1]],
        1000,
        Arc::new(LbmProbeExtractor),
    ));

    // --- EXPORT ---
    let path = test_output_dir().join("lid-driven-cavity");
    let exporter = Arc::new(LbmVtkExporter::new(nx, ny));
    let export_node_id = graph.add_node(ExportNode::new(
        bc_node_id,
        1000,
        path.to_str().unwrap(),
        exporter,
    ));

    let _next_state_id = graph.add_node(NextStateNode::new(f_id, fluid_probe_id));

    graph.compile().unwrap();
    for t in 0..3_000 {
        graph.step().unwrap();

        if t % 1000 == 0 {
            println!("Step {}", t);
        }
    }

    assert!(graph.inner.state_map.contains_key(&f_id));
}
