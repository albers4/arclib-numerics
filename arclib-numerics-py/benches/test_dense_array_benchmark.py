# Copyright (c) 2026 ARC (Applied Research & Computation)
# SPDX-License-Identifier: LGPL-2.1-or-later

from arclib_numerics import DenseArray


def test_dense_array_capacity_benchmark(benchmark):
    def create_and_capacity():
        arr = DenseArray(10_000)
        return arr.capacity()

    result = benchmark(create_and_capacity)
    assert result == 10_000
