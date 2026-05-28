// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use arclib_numerics_spec::{Tensor, utils::ProbeExtractor};
use ndarray::s;

pub struct LbmProbeExtractor;

impl ProbeExtractor for LbmProbeExtractor {
    fn extract(&self, tensor: &Tensor, coords: &[Vec<usize>]) -> String {
        let mut out = String::new();
        let cx = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0];
        let cy = [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0];

        for coord in coords {
            if coord.len() >= 2 {
                let x = coord[0];
                let y = coord[1];

                if x >= tensor.shape()[0] || y >= tensor.shape()[1] {
                    out.push_str(&format!("  @ {:?} | Out of bounds\n", coord));
                    continue;
                }

                let f = tensor.slice(s![x, y, ..]);
                let mut rho = 0.0;
                let mut mux = 0.0;
                let mut muy = 0.0;
                for q in 0..9 {
                    rho += f[q];
                    mux += f[q] * cx[q];
                    muy += f[q] * cy[q];
                }

                let ux = if rho > 1e-6 { mux / rho } else { 0.0 };
                let uy = if rho > 1e-6 { muy / rho } else { 0.0 };

                out.push_str(&format!(
                    "  @ {:?} | rho: {:.4} | u: [{:.4}, {:.4}]\n",
                    coord, rho, ux, uy
                ));
            }
        }

        out
    }
}
