# Copyright (c) 2026 ARC (Applied Research & Computation)
# SPDX-License-Identifier: LGPL-2.1-or-later

from arclib_numerics import DenseArray


def test_dense_array_len():
    arr = DenseArray(10)
    assert arr.capacity() == 10
