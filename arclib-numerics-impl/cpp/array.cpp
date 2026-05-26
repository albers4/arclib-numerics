// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#include "array.h"
#include <vector>
#include <cstring>
#include <cstdio>

struct ArrayHandle {
    std::vector<double> data;
    size_t len;

    ArrayHandle(const double* d, size_t length)
        : data(d, d + length), len(length) {}
};

extern "C" ArrayHandle* array_create(const double* data, size_t len) {
    auto *arr = new ArrayHandle(data, len);
    arr->len = len;
    return arr;
}

extern "C" void array_add(ArrayHandle *res, const ArrayHandle *a, const ArrayHandle *b) {
    res->len = a->len;

    for (size_t i = 0; i < res->len; ++i) {
        res->data[i] = a->data[i] + b->data[i];
    }
}

extern "C" void array_copy(const ArrayHandle *arr, double *buffer, size_t len) {
    memcpy(buffer, arr->data.data(), len * sizeof(double));
}

extern "C" void array_destroy(ArrayHandle *arr) {
    delete arr;
}