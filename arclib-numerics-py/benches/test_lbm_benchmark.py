# Copyright (c) 2026 ARC (Applied Research & Computation)
# SPDX-License-Identifier: LGPL-2.1-or-later


def test_lbm_benchmark(benchmark):
    def create():
        return 10_000

    result = benchmark(create)
    assert result == 10_000
