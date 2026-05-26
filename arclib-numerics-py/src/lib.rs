// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use arclib_numerics_impl::DenseArray;
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyclass]
struct PyDenseArray {
    inner: DenseArray<f64>,
}

#[pymethods]
impl PyDenseArray {
    #[new]
    fn new(size: usize) -> Self {
        Self {
            inner: DenseArray::new(size),
        }
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
}

#[pymodule]
fn arclib_numerics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDenseArray>()?;
    Ok(())
}
