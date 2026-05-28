// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::sync::Arc;

use arclib_graph_spec::ContextValueLike;
use arclib_numerics_spec::Tensor;

#[derive(Clone)]
pub enum NumericsContextValue {
    Tensor(Arc<Tensor>),
    Empty,
}

impl ContextValueLike for NumericsContextValue {}

impl std::fmt::Display for NumericsContextValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Geometry Context Value (todo)")
    }
}
