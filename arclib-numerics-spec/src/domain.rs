// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use arclib_graph_impl::Graph;
use arclib_graph_spec::ContextValueLike;

use crate::kernel::CompiledKernel;

pub trait Domain<V: ContextValueLike>: 'static + Send + Sync {
    type Compiler: DomainCompiler<V>;
}

pub trait DomainCompiler<V: ContextValueLike>: Send + Sync {
    fn compile(&self, graph: &Graph<V>) -> CompiledKernel;
}
