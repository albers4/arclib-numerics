// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

mod ffi;

use arclib_numerics_spec::ArrayLike;

pub struct DenseArray<T> {
    data: Vec<T>,
}

impl<T> DenseArray<T> {
    pub fn new(size: usize) -> DenseArray<T> {
        Self {
            data: Vec::<T>::with_capacity(size),
        }
    }

    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }
}

impl<T> ArrayLike for DenseArray<T> {
    type Item = T;
    type Error = String;

    fn get(&self, index: usize) -> Result<&T, String> {
        self.data
            .get(index)
            .ok_or_else(|| format!("index {index} out of bounds"))
    }
}
