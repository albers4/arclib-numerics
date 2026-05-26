// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

pub trait DomainCompiler {}

pub trait Domain: 'static + Send + Sync {
    type TopologyState;
    type Compiler: DomainCompiler;
}