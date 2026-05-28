// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

use arclib_numerics_spec::{Tensor, bc::BcEvaluator};
use ndarray::s;

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
