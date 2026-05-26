// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::hint::black_box;

use arclib_numerics_impl::DenseArray;
use criterion::{Criterion, criterion_group, criterion_main};
use pprof::criterion::{Output, PProfProfiler};

fn dense_array_capacity_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_array_capacity");

    group.bench_function("scalar", |bench| {
        let arr: DenseArray<f64> = DenseArray::new(black_box(10_000));

        bench.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..arr.capacity() {
                sum += black_box(i as f64);
            }
            black_box(sum)
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = dense_array_capacity_benchmark
}
criterion_main!(benches);
