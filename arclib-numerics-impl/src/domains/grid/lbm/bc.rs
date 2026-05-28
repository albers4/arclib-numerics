// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use std::marker::PhantomData;

use arclib_numerics_spec::{bc::BcEvaluator, tensor::Tensor};
use ndarray::s;

use crate::domains::grid::lbm::topology::LatticeTopology;

#[derive(Default)]
pub struct LbmVelocityBC<T: LatticeTopology> {
    _marker: PhantomData<T>,
}

impl<T: LatticeTopology> LbmVelocityBC<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: LatticeTopology> BcEvaluator for LbmVelocityBC<T> {
    fn apply(&self, state: &mut Tensor, mask: &Tensor, values: &Tensor) {
        let mask_cpu = mask.as_cpu();
        let values_cpu = values.as_cpu();
        let state_cpu = state.as_cpu();

        let shape = state_cpu.shape();
        let nx = shape[0];
        let ny = if T::DIM > 1 { shape[1] } else { 1 };
        let nz = if T::DIM > 2 { shape[2] } else { 1 };

        // Helper to check if a neighbor is fluid (mask < 0.5)
        let is_fluid = |x: i32, y: i32, z: i32| -> bool {
            if x < 0 || x >= nx as i32 || y < 0 || y >= ny as i32 || z < 0 || z >= nz as i32 {
                return false;
            }
            let val = match T::DIM {
                2 => mask_cpu[[x as usize, y as usize]],
                3 => mask_cpu[[x as usize, y as usize, z as usize]],
                _ => 0.0,
            };
            val < 0.5
        };

        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let m = match T::DIM {
                        2 => mask_cpu[[x, y]],
                        3 => mask_cpu[[x, y, z]],
                        _ => 0.0,
                    };

                    if m > 0.5 {
                        let mut f_slice = match T::DIM {
                            2 => state.slice_mut(s![x, y, ..]),
                            3 => state.slice_mut(s![x, y, z, ..]),
                            _ => panic!("Unsupported DIM"),
                        };

                        let u_wall = match T::DIM {
                            2 => [values_cpu[[x, y, 0]], values_cpu[[x, y, 1]], 0.0],
                            3 => [
                                values_cpu[[x, y, z, 0]],
                                values_cpu[[x, y, z, 1]],
                                values_cpu[[x, y, z, 2]],
                            ],
                            _ => [0.0; 3],
                        };

                        // Compute rho from population
                        let mut rho = 0.0;
                        let mut known_weight = 0.0;
                        for q in 0..T::Q {
                            let x_src = x as i32 - T::CX[q];
                            let y_src = y as i32 - T::CY[q];
                            let z_src = z as i32 - T::CZ[q];

                            if is_fluid(x_src, y_src, z_src) {
                                rho += f_slice[q];
                                known_weight += T::W[q];
                            }
                        }
                        if known_weight > 1e-6 {
                            rho /= known_weight;
                        } else {
                            rho = 1.0;
                        }

                        // Reconstruct unknown populations using NEBB
                        for q in 0..T::Q {
                            let x_src = x as i32 - T::CX[q];
                            let y_src = y as i32 - T::CY[q];
                            let z_src = z as i32 - T::CZ[q];

                            // If the neighbor in direction q is NOT fluid,
                            // then population q is unknown (it should have come from the wall).
                            if !is_fluid(x_src, y_src, z_src) {
                                let opp_q = T::OPP[q];
                                let f_opp = f_slice[opp_q]; // The known opposite population

                                let cu = (T::CX[q] as f32) * u_wall[0]
                                    + (T::CY[q] as f32) * u_wall[1]
                                    + (T::CZ[q] as f32) * u_wall[2];

                                // NEBB Formula
                                f_slice[q] = f_opp - 6.0 * T::W[q] * rho * cu;
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
pub struct LbmZouHeVelocity;

impl BcEvaluator for LbmZouHeVelocity {
    fn apply(&self, state: &mut Tensor, mask: &Tensor, values: &Tensor) {
        let shape = state.shape();
        let nx = shape[0];
        let ny = shape[1];

        // Helper to check if a neighbor is fluid (mask < 0.5)
        let is_fluid = |cx: i32, cy: i32, x: usize, y: usize| -> bool {
            let nx_idx = x as i32 + cx;
            let ny_idx = y as i32 + cy;
            if nx_idx >= 0 && nx_idx < nx as i32 && ny_idx >= 0 && ny_idx < ny as i32 {
                mask[[nx_idx as usize, ny_idx as usize]] < 0.5
            } else {
                false // Out of bounds is treated as solid
            }
        };

        for x in 0..nx {
            for y in 0..ny {
                if mask[[x, y]] > 0.5 {
                    let ux = values[[x, y, 0]];
                    let uy = values[[x, y, 1]];
                    let f = &mut state.slice_mut(s![x, y, ..]);

                    let fluid_below = is_fluid(0, -1, x, y);
                    let fluid_above = is_fluid(0, 1, x, y);
                    let fluid_left = is_fluid(-1, 0, x, y);
                    let fluid_right = is_fluid(1, 0, x, y);

                    if fluid_below && !fluid_above {
                        // TOP WALL (y = ny - 1) (Normal points UP, missing f4, f7, f8)
                        let rho = f[0] + f[1] + f[3] + 2.0 * (f[2] + f[5] + f[6]);
                        f[4] = f[2];
                        f[8] = f[6] + 0.5 * (rho * ux - (f[1] - f[3]));
                        f[7] = f[5] - 0.5 * (rho * ux - (f[1] - f[3]));
                    } else if fluid_above && !fluid_below {
                        // BOTTOM WALL (y = 0) (Normal points DOWN, missing f2, f5, f6)
                        let rho = f[0] + f[1] + f[3] + 2.0 * (f[4] + f[7] + f[8]);
                        f[2] = f[4];
                        f[5] = f[7] + 0.5 * (rho * ux - (f[1] - f[3]));
                        f[6] = f[8] - 0.5 * (rho * ux - (f[1] - f[3]));
                    } else if fluid_right && !fluid_left {
                        // LEFT WALL (Normal points LEFT, missing f1, f5, f8)
                        let rho = f[0] + f[2] + f[4] + 2.0 * (f[3] + f[6] + f[7]);
                        f[1] = f[3];
                        f[5] = f[7] + 0.5 * (rho * uy - (f[2] - f[4]));
                        f[8] = f[6] - 0.5 * (rho * uy - (f[2] - f[4]));
                    } else if fluid_left && !fluid_right {
                        // RIGHT WALL (Normal points RIGHT, missing f3, f6, f7)
                        let rho = f[0] + f[2] + f[4] + 2.0 * (f[1] + f[5] + f[8]);
                        f[3] = f[1];
                        f[6] = f[8] + 0.5 * (rho * uy - (f[2] - f[4]));
                        f[7] = f[5] - 0.5 * (rho * uy - (f[2] - f[4]));
                    }
                }
            }
        }
    }
}

*/
