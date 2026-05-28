// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

mod bc;
mod compiler;
mod export;
mod probe;

use arclib_numerics_spec::domain::Domain;
pub use bc::LbmZouHeVelocity;
pub use compiler::LatticeCompiler;
pub use export::LbmVtkExporter;
pub use probe::LbmProbeExtractor;

use crate::NumericsContextValue;

pub struct Lattice;

impl Domain<NumericsContextValue> for Lattice {
    type Compiler = LatticeCompiler;
}
