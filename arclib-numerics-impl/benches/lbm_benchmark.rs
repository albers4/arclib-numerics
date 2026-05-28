// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use pprof::criterion::{Output, PProfProfiler};

fn lbm_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbm_benchmark");

    group.bench_function("scalar", |bench| {
        bench.iter(|| {
            let mut sum = 0.0f64;
            for i in 0..10_000 {
                sum += black_box(i as f64);
            }
            black_box(sum)
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = lbm_benchmark
}
criterion_main!(benches);
