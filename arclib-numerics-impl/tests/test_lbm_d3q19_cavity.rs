// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::{collections::HashMap, env, path::PathBuf, sync::Arc};

use arclib_graph_spec::GraphLike;
use arclib_numerics_impl::{
    BoundaryCondition, BoundaryConditionNode, ConstantNode, EquationGraph, EquationNode,
    ExportNode, NextStateNode, NumericsContextValue, NumericsGraph, ProbeNode, StateNode,
    domains::{
        grid::{D3Q19, LatticeCompiler, LbmProbeExtractor, LbmVelocityBC, LbmVtkExporter},
        probe::ScalarProbeExtractor,
    },
};
use arclib_numerics_spec::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

fn test_output_dir() -> PathBuf {
    let dir = env::var("ARCLIB_TEST_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::temp_dir().join("arclib_test"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_lbm_d3q19_cavity() {
    let nx = 40;
    let ny = 40;
    let nz = 40;
    let omega = 1.7; // relaxation parameter
    let u_lid = 0.1;

    let mut graph = NumericsGraph::new();

    // --- SOLID MASK ---
    let mut solid_data = vec![0.0f32; nx * ny * nz];
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let idx = (x * ny + y) * nz + z;

                if y == 0 {
                    solid_data[idx] = 1.0;
                }
                if x == 0 {
                    solid_data[idx] = 1.0;
                }
                if x == nx - 1 {
                    solid_data[idx] = 1.0;
                }
                if z == 0 {
                    solid_data[idx] = 1.0;
                }
                if z == nz - 1 {
                    solid_data[idx] = 1.0;
                }
            }
        }
    }
    let solid_tensor = ArrayD::from_shape_vec(IxDyn(&[nx, ny, nz]), solid_data).unwrap();
    let solid_id = graph.add_node(ConstantNode::new(Tensor::from_cpu_array(solid_tensor)));

    // --- FLUID STATE ----
    let f_id = graph.add_node(StateNode::new("f"));
    let mut f_data = vec![0.0f32; nx * ny * nz * 19];
    let w = [
        1.0 / 3.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ];
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let idx = ((x * ny + y) * nz + z) * 19;
                for q in 0..19 {
                    f_data[idx + q] = w[q];
                }
            }
        }
    }
    let f_tensor = ArrayD::from_shape_vec(IxDyn(&[nx, ny, nz, 19]), f_data).unwrap();
    graph.inner.state_map.insert(
        f_id,
        NumericsContextValue::Tensor(Arc::new(Tensor::from_cpu_array(f_tensor))),
    );

    // --- LBM EQUATION (Stream + Collide) ---
    let compiler = Arc::new(LatticeCompiler::<D3Q19>::new(nx, ny, nz, omega));
    let mut eq = EquationGraph::new(compiler.clone());
    let f_var = eq.var("f");
    let _solid_var = eq.var("solid");
    eq.set_output(f_var);

    let mut mapping = HashMap::new();
    mapping.insert("f".to_string(), f_id);
    mapping.insert("solid".to_string(), solid_id);
    let eq_node_id = graph.add_node(EquationNode::new(eq, mapping));

    // --- MOVING LID BC ---
    let mut lid_mask_data = vec![0.0f32; nx * ny * nz];
    let mut u_lid_data = vec![0.0f32; nx * ny * nz * 3];

    for x in 1..(nx - 1) {
        for z in 1..(nz - 1) {
            let y = ny - 1;

            let mask_idx = (x * ny + y) * nz + z;
            lid_mask_data[mask_idx] = 1.0;

            let u_idx = ((x * ny + y) * nz + z) * 3;
            u_lid_data[u_idx + 0] = u_lid;
            u_lid_data[u_idx + 1] = 0.0;
            u_lid_data[u_idx + 2] = 0.0;
        }
    }
    let lid_mask_tensor = ArrayD::from_shape_vec(IxDyn(&[nx, ny, nz]), lid_mask_data).unwrap();
    let lid_mask_id = graph.add_node(ConstantNode::new(Tensor::from_cpu_array(lid_mask_tensor)));

    let u_lid_tensor = ArrayD::from_shape_vec(IxDyn(&[nx, ny, nz, 3]), u_lid_data).unwrap();
    let u_lid_id = graph.add_node(ConstantNode::new(Tensor::from_cpu_array(u_lid_tensor)));

    let bc_meta = BoundaryCondition::dirichlet();
    let bc_node_id = graph.add_node(BoundaryConditionNode::new(
        eq_node_id,
        lid_mask_id,
        u_lid_id,
        bc_meta,
        Arc::new(LbmVelocityBC::<D3Q19>::new()),
    ));

    // --- PROBE 1 ---
    let _mask_probe_id = graph.add_node(ProbeNode::new(
        "Solid Mask",
        solid_id,
        vec![
            vec![0, 0, 0],
            vec![nx - 1, 0, nz - 1],
            vec![nx / 2, ny / 2, nz / 2],
        ],
        1000,
        Arc::new(ScalarProbeExtractor),
    ));

    // --- PROBE 2 ---
    let fluid_probe_id = graph.add_node(ProbeNode::new(
        "Fluid State",
        bc_node_id,
        vec![
            vec![nx / 2, ny - 1, nz / 2],
            vec![nx / 2, ny / 2, nz / 2],
            vec![nx / 2, 1, nz / 2],
        ],
        1000,
        Arc::new(LbmProbeExtractor::<D3Q19>::new()),
    ));

    // --- EXPORT ---
    let path = test_output_dir().join("lid-driven-cavity-d3q19");
    let exporter = Arc::new(LbmVtkExporter::<D3Q19>::new(nx, ny, nz));
    let _export_node_id = graph.add_node(ExportNode::new(
        bc_node_id,
        1000,
        path.to_str().unwrap(),
        exporter,
    ));

    let _next_state_id = graph.add_node(NextStateNode::new(f_id, fluid_probe_id));

    graph.compile().unwrap();
    for t in 0..10_000 {
        graph.step().unwrap();

        if t % 1000 == 0 {
            println!("Step {}", t);
        }
    }

    assert!(graph.inner.state_map.contains_key(&f_id));
}
