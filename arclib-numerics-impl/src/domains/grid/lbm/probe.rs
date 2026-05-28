// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::marker::PhantomData;

use arclib_numerics_spec::{tensor::Tensor, utils::ProbeExtractor};
use ndarray::s;

use crate::domains::grid::lbm::topology::LatticeTopology;

#[derive(Default)]
pub struct LbmProbeExtractor<T: LatticeTopology> {
    _marker: PhantomData<T>,
}

impl<T: LatticeTopology> LbmProbeExtractor<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: LatticeTopology> ProbeExtractor for LbmProbeExtractor<T> {
    fn extract(&self, tensor: &Tensor, coords: &[Vec<usize>]) -> String {
        let mut out = String::new();

        for coord in coords {
            if coord.len() != T::DIM {
                out.push_str(&format!(
                    "  @ {:?} | Dimension mismatch (expected {})\n",
                    coord,
                    T::DIM
                ));
                continue;
            }

            let mut in_bounds = true;
            for (d, _) in coord.iter().enumerate().take(T::DIM) {
                if coord[d] >= tensor.shape.0[d] {
                    in_bounds = false;
                    break;
                }
            }
            if !in_bounds {
                out.push_str(&format!("  @ {:?} | Out of bounds)\n", coord));
                continue;
            }

            let f_slice = match T::DIM {
                2 => tensor.slice(s![coord[0], coord[1], ..]),
                3 => tensor.slice(s![coord[0], coord[1], coord[2], ..]),
                _ => panic!("Unsupported DIM"),
            };

            let mut rho = 0.0;
            let mut u = vec![0.0; T::DIM];

            for q in 0..T::Q {
                let fq = f_slice[q];
                rho += fq;
                u[0] += fq * T::CX[q] as f32;
                if T::DIM > 1 {
                    u[1] += fq * T::CY[q] as f32;
                }
                if T::DIM > 2 {
                    u[2] += fq * T::CZ[q] as f32;
                }
            }

            if rho > 1e-6 {
                for ud in u.iter_mut().take(T::DIM) {
                    *ud /= rho;
                }
            }

            out.push_str(&format!("  @ {:?} | rho: {:.4} | u: [", coord, rho));
            for (d, _) in u.iter().enumerate().take(T::DIM) {
                out.push_str(&format!(
                    "{:.4}{}",
                    u[d],
                    if d == T::DIM - 1 { "" } else { ", " }
                ));
            }
            out.push_str("]\n");
        }
        out
    }
}
