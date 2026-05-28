// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::{ffi::c_void, marker::PhantomData};

use arclib_graph_impl::Graph;
use arclib_numerics_spec::{
    checkpoint::CheckpointStrategy,
    domain::DomainCompiler,
    kernel::{CompiledKernel, struct_to_bytes},
};

use crate::{
    NumericsContextValue,
    domains::grid::lbm::topology::LatticeTopology,
    kernels::grid_kernels::{
        LbmParams, lbm_d2q9_fused_backward_wrapper, lbm_d2q9_fused_forward_wrapper,
        lbm_d3q19_fused_backward_wrapper, lbm_d3q19_fused_forward_wrapper,
    },
};

pub struct LatticeCompiler<T: LatticeTopology> {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub omega: f32,
    _marker: PhantomData<T>,
}

impl<T: LatticeTopology> LatticeCompiler<T> {
    pub fn new(nx: usize, ny: usize, nz: usize, omega: f32) -> Self {
        Self {
            nx,
            ny,
            nz,
            omega,
            _marker: PhantomData,
        }
    }

    fn get_ffi_functions(
        &self,
    ) -> (
        unsafe extern "C" fn(*const *const c_void, *const *mut c_void, *mut c_void, *const c_void),
        unsafe extern "C" fn(
            *const *const c_void,
            *const *const c_void,
            *const *mut c_void,
            *const c_void,
            *const c_void,
        ),
    ) {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<super::topology::D2Q9>() {
            (
                lbm_d2q9_fused_forward_wrapper,
                lbm_d2q9_fused_backward_wrapper,
            )
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<super::topology::D3Q19>() {
            (
                lbm_d3q19_fused_forward_wrapper,
                lbm_d3q19_fused_backward_wrapper,
            )
        } else {
            panic!("Unsupported Lattice Topology for FII");
        }
    }
}

impl<T: LatticeTopology> DomainCompiler<NumericsContextValue> for LatticeCompiler<T> {
    fn compile(&self, _graph: &Graph<NumericsContextValue>) -> CompiledKernel {
        let (forward_fn, backward_fn) = self.get_ffi_functions();
        let params = LbmParams {
            omega: self.omega,
            nx: self.nx as i32,
            ny: self.ny as i32,
            nz: self.nz as i32,
            q: T::Q as i32,
        };

        let params_bytes = struct_to_bytes(&params);

        CompiledKernel {
            name: format!("lbm_d{}q{}_fused", T::DIM, T::Q),
            strategy: CheckpointStrategy::BoundaryOnly,
            forward: forward_fn,
            backward: Some(backward_fn),
            workspace: vec![],
            params_bytes,
        }
    }
}

impl<T: LatticeTopology> Drop for LatticeCompiler<T> {
    fn drop(&mut self) {
        // Clean up params if needed
    }
}
