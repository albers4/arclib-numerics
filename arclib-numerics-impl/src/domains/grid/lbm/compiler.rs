// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use arclib_graph_impl::Graph;
use arclib_numerics_spec::{
    checkpoint::CheckpointStrategy,
    domain::DomainCompiler,
    kernel::{CompiledKernel, struct_to_bytes},
};

use crate::{
    NumericsContextValue,
    kernels::grid_kernels::{
        LbmParams, lbm_d2q9_fused_backward_wrapper, lbm_d2q9_fused_forward_wrapper,
    },
};

pub struct LatticeCompiler {
    pub nx: usize,
    pub ny: usize,
    pub omega: f32,
}

impl LatticeCompiler {
    pub fn new(nx: usize, ny: usize, omega: f32) -> Self {
        Self { nx, ny, omega }
    }
}

impl DomainCompiler<NumericsContextValue> for LatticeCompiler {
    fn compile(&self, _graph: &Graph<NumericsContextValue>) -> CompiledKernel {
        let params = LbmParams {
            omega: self.omega,
            nx: self.nx as i32,
            ny: self.ny as i32,
        };

        let params_bytes = struct_to_bytes(&params);

        CompiledKernel {
            name: "lbm_d2q9_fused".to_string(),
            strategy: CheckpointStrategy::BoundaryOnly,
            forward: lbm_d2q9_fused_forward_wrapper,
            backward: Some(lbm_d2q9_fused_backward_wrapper),
            workspace: vec![],
            params_bytes,
        }
    }
}

impl Drop for LatticeCompiler {
    fn drop(&mut self) {
        // Clean up params if needed
    }
}
