// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#pragma once
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ArrayHandle ArrayHandle;

// Lifecylce
ArrayHandle* array_create(const double* data, size_t len);
void array_destroy(ArrayHandle *arr);

// Operation
void array_add(ArrayHandle* result, const ArrayHandle *a, const ArrayHandle *b);

// Helper
void array_copy(const ArrayHandle *arr, double* buffer, size_t len);

#ifdef __cplusplus
}
#endif