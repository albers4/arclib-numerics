// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use arclib_numerics_spec::{Tensor, utils::ProbeExtractor};
use ndarray::IxDyn;

pub struct ScalarProbeExtractor;

impl ProbeExtractor for ScalarProbeExtractor {
    fn extract(&self, tensor: &Tensor, coords: &[Vec<usize>]) -> String {
        let mut out = String::new();
        for coord in coords {
            let idx = IxDyn(coord);
            if let Some(val) = tensor.get(idx) {
                out.push_str(&format!("  @ {:?} | val: {:.1}\n", coord, val));
            } else {
                out.push_str(&format!("  @ {:?} | Out of bounds\n", coord));
            }
        }
        out
    }
}
