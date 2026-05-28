// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BCType {
    Dirichlet,
    Neumann,
    Robin,
    Periodic,
}

#[derive(Clone)]
pub struct BoundaryCondition {
    pub bc_type: BCType,
    // Optional: algorithmic parameters (e.g., penalty factor for Robin BCs)
    pub penalty_factor: Option<f32>,
}

impl BoundaryCondition {
    pub fn dirichlet() -> Self {
        Self {
            bc_type: BCType::Dirichlet,
            penalty_factor: None,
        }
    }
}
