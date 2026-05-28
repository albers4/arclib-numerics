// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::env;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .opt_level(3)
        .compiler("clang++-16")
        .flag("-std=c++17")
        .flag("-fno-exceptions")
        .flag_if_supported("-march=native")
        .flag_if_supported("-ffast-math")
        .flag_if_supported("-fopenmp")
        .include("cpp")
        .file("cpp/grid/lbm_d2q9_fused.cpp");

    build.out_dir(&out_dir).compile("grid_kernels");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=grid_kernels");
    println!("cargo:rerun-if-changed=cpp/");
}
