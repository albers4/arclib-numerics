// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use crate::Tensor;

pub trait DataExporter: Send + Sync {
    fn export(&self, state: &Tensor, step: usize, base_path: &str);
}

pub trait ProbeExtractor: Send + Sync {
    fn extract(&self, tensor: &Tensor, coords: &[Vec<usize>]) -> String;
}
