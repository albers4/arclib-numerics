// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::ffi::c_void;

#[repr(C)]
pub struct LbmParams {
    pub omega: f32,
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub q: i32,
}

unsafe extern "C" {
    pub unsafe fn lbm_d2q9_fused_forward(
        f_in: *const f32,
        f_out: *mut f32,
        solid: *const f32,
        omega: f32,
        nx: i32,
        ny: i32,
    );

    pub unsafe fn lbm_d2q9_fused_backward(
        f_in: *const f32,
        grad_f_out: *const f32,
        grad_f_in: *mut f32,
        solid: *const f32,
        omega: f32,
        nx: i32,
        ny: i32,
    );

    pub unsafe fn lbm_d3q19_fused_forward(
        f_in: *const f32,
        f_out: *mut f32,
        solid: *const f32,
        omega: f32,
        nx: i32,
        ny: i32,
        nz: i32,
    );

    pub unsafe fn lbm_d3q19_fused_backward(
        f_in: *const f32,
        grad_f_out: *const f32,
        grad_f_in: *mut f32,
        solid: *const f32,
        omega: f32,
        nx: i32,
        ny: i32,
        nz: i32,
    );
}

/// # Safety
pub unsafe extern "C" fn lbm_d2q9_fused_forward_wrapper(
    inputs: *const *const c_void,
    outputs: *const *mut c_void,
    _workspace: *mut c_void,
    params: *const c_void,
) {
    unsafe {
        let f_in = *(inputs.add(0)) as *const f32;
        let solid = *(inputs.add(1)) as *const f32;
        let f_out = *(outputs.add(0)) as *mut f32;

        let params_ptr = params as *const LbmParams;
        let params = &*params_ptr;

        lbm_d2q9_fused_forward(f_in, f_out, solid, params.omega, params.nx, params.ny);
    }
}

/// # Safety
pub unsafe extern "C" fn lbm_d2q9_fused_backward_wrapper(
    inputs: *const *const c_void,
    grad_outputs: *const *const c_void,
    grad_inputs: *const *mut c_void,
    _workspace: *const c_void,
    params: *const c_void,
) {
    unsafe {
        let f_in = *(inputs.add(0)) as *const f32;
        let solid = *(inputs.add(1)) as *const f32;
        let grad_f_out = *(grad_outputs.add(0)) as *const f32;
        let grad_f_in = *(grad_inputs.add(0)) as *mut f32;

        let params_ptr = params as *const LbmParams;
        let params = &*params_ptr;

        lbm_d2q9_fused_backward(
            f_in,
            grad_f_out,
            grad_f_in,
            solid,
            params.omega,
            params.nx,
            params.ny,
        );
    }
}

/// # Safety
pub unsafe extern "C" fn lbm_d3q19_fused_forward_wrapper(
    inputs: *const *const c_void,
    outputs: *const *mut c_void,
    _workspace: *mut c_void,
    params: *const c_void,
) {
    unsafe {
        let f_in = *(inputs.add(0)) as *const f32;
        let solid = *(inputs.add(1)) as *const f32;
        let f_out = *(outputs.add(0)) as *mut f32;

        let params_ptr = params as *const LbmParams;
        let params = &*params_ptr;

        lbm_d3q19_fused_forward(
            f_in,
            f_out,
            solid,
            params.omega,
            params.nx,
            params.ny,
            params.nz,
        );
    }
}

/// # Safety
pub unsafe extern "C" fn lbm_d3q19_fused_backward_wrapper(
    inputs: *const *const c_void,
    grad_outputs: *const *const c_void,
    grad_inputs: *const *mut c_void,
    _workspace: *const c_void,
    params: *const c_void,
) {
    unsafe {
        let f_in = *(inputs.add(0)) as *const f32;
        let solid = *(inputs.add(1)) as *const f32;
        let grad_f_out = *(grad_outputs.add(0)) as *const f32;
        let grad_f_in = *(grad_inputs.add(0)) as *mut f32;

        let params_ptr = params as *const LbmParams;
        let params = &*params_ptr;

        lbm_d3q19_fused_backward(
            f_in,
            grad_f_out,
            grad_f_in,
            solid,
            params.omega,
            params.nx,
            params.ny,
            params.nz,
        );
    }
}
