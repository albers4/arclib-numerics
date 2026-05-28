// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::os::raw::c_void;

use crate::checkpoint::CheckpointStrategy;

#[derive(Clone)]
pub struct CompiledKernel {
    pub name: String,
    pub strategy: CheckpointStrategy,

    pub forward: unsafe extern "C" fn(
        inputs: *const *const c_void,
        outputs: *const *mut c_void,
        workspace: *mut c_void,
        params: *const c_void,
    ),
    pub backward: Option<
        unsafe extern "C" fn(
            inputs: *const *const c_void,
            grad_outputs: *const *const c_void,
            grad_inputs: *const *mut c_void,
            workspace: *const c_void,
            params: *const c_void,
        ),
    >,
    /// Pre-allocated scratchpad memory for the C-kernel
    pub workspace: Vec<u8>,
    /// Raw bytes of the parameters struct
    pub params_bytes: Vec<u8>,
}

unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

pub fn struct_to_bytes<T: Sized>(s: &T) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts((s as *const T) as *const u8, std::mem::size_of::<T>()).to_vec()
    }
}
