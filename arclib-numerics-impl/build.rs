// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::env;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .include("cpp")
        .file("cpp/grid/lbm_d2q9_fused.cpp")
        .opt_level(3)
        .flag_if_supported("-march=native")
        .flag_if_supported("-ffast-math")
        .flag_if_supported("-fno-exceptions")
        .flag_if_supported("-std=c++17");

    let compiler = build.get_compiler();

    if compiler.is_like_msvc() {
        build.flag("/openmp");
    } else {
        build.flag("-fopenmp");
    }

    build.out_dir(&out_dir).compile("lbm_kernels");

    if compiler.is_like_msvc() {
        // MSVC handles linking automatically when /openmp is used
    } else if compiler.is_like_gnu() {
        println!("cargo:rustc-link-lib=dylib=gomp");
    } else {
        println!("cargo:rustc-link-lib=dylib=omp");
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=lbm_kernels");
    println!("cargo:rerun-if-changed=cpp/");
}
