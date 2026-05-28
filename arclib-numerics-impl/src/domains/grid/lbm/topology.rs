// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

pub trait LatticeTopology: Send + Sync + Clone + 'static {
    const DIM: usize;
    const Q: usize;

    const CX: &'static [i32];
    const CY: &'static [i32];
    const CZ: &'static [i32];
    const OPP: &'static [usize];
    const W: &'static [f32];
}

#[derive(Clone, Copy, Debug)]
pub struct D2Q9;

impl LatticeTopology for D2Q9 {
    const DIM: usize = 2;
    const Q: usize = 9;

    const CX: &'static [i32] = &[0, 1, 0, -1, 0, 1, -1, -1, 1];
    const CY: &'static [i32] = &[0, 0, 1, 0, -1, 1, 1, -1, -1];
    const CZ: &'static [i32] = &[0, 0, 0, 0, 0, 0, 0, 0, 0];

    const OPP: &'static [usize] = &[0, 3, 4, 1, 2, 7, 8, 5, 6];

    const W: &'static [f32] = &[
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ];
}

#[derive(Clone, Copy, Debug)]
pub struct D3Q19;

impl LatticeTopology for D3Q19 {
    const DIM: usize = 3;
    const Q: usize = 19;

    const CX: &'static [i32] = &[0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0];
    const CY: &'static [i32] = &[0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1];
    const CZ: &'static [i32] = &[0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1];

    const OPP: &'static [usize] = &[
        0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15,
    ];

    const W: &'static [f32] = &[
        1.0 / 3.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ];
}
